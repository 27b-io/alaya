//! Two-tier caching decorator for EmbeddingProvider.
//!
//! L1: mini-moka unsync in-process cache (hot path, ~0ns).
//! L2: cachekit-rs Redis (persists across restarts, shared across pods).
//!
//! Embeddings are immutable and content-addressed — same text always produces
//! the same vector. Cache keys are SHA-256(prompt_name + ":" + text).

use std::cell::{Cell, RefCell};

use async_trait::async_trait;
use mini_moka::unsync::Cache;
use sha2::{Digest, Sha256};

use alaya_backends::traits::EmbeddingProvider;
use alaya_types::Result;
use alaya_types::search::PromptName;

/// Cache key: SHA-256 of (prompt_name, text). Returns (raw bytes for L1, hex string for L2).
fn cache_key(prompt: PromptName, text: &str) -> ([u8; 32], String) {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_str().as_bytes());
    hasher.update(b":");
    hasher.update(text.as_bytes());
    let bytes: [u8; 32] = hasher.finalize().into();
    let hex = bytes.iter().fold(String::with_capacity(64), |mut s, b| {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
        s
    });
    (bytes, hex)
}

pub struct CachedEmbedding {
    inner: Box<dyn EmbeddingProvider>,
    /// L1: in-process, single-threaded, zero-cost hit.
    l1: RefCell<Cache<[u8; 32], Vec<f32>>>,
    /// L2: Redis via cachekit-rs. Optional — degrades to L1-only if not configured.
    l2: Option<cachekit::CacheKit>,
    hits_l1: Cell<u64>,
    hits_l2: Cell<u64>,
    misses: Cell<u64>,
}

impl CachedEmbedding {
    pub fn new(
        inner: Box<dyn EmbeddingProvider>,
        l1_capacity: u64,
        l2: Option<cachekit::CacheKit>,
    ) -> Self {
        Self {
            inner,
            l1: RefCell::new(Cache::builder().max_capacity(l1_capacity).build()),
            l2,
            hits_l1: Cell::new(0),
            hits_l2: Cell::new(0),
            misses: Cell::new(0),
        }
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits_l1.get(), self.hits_l2.get(), self.misses.get())
    }

    /// Try L2 get, backfill L1 on hit.
    async fn l2_get(&self, key_bytes: [u8; 32], key_hex: &str) -> Option<Vec<f32>> {
        let l2 = self.l2.as_ref()?;
        match l2.get::<Vec<f32>>(key_hex).await {
            Ok(Some(embedding)) => {
                // Backfill L1
                self.l1.borrow_mut().insert(key_bytes, embedding.clone());
                self.hits_l2.set(self.hits_l2.get() + 1);
                Some(embedding)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::debug!("L2 cache get failed (non-fatal): {e}");
                None
            }
        }
    }

    /// Write to L2 (fire-and-forget, errors logged).
    async fn l2_set(&self, key_hex: &str, embedding: &Vec<f32>) {
        if let Some(l2) = self.l2.as_ref()
            && let Err(e) = l2.set(key_hex, embedding).await
        {
            tracing::debug!("L2 cache set failed (non-fatal): {e}");
        }
    }
}

#[async_trait(?Send)]
impl EmbeddingProvider for CachedEmbedding {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
        // Compute all keys upfront
        let keys: Vec<([u8; 32], String)> =
            texts.iter().map(|t| cache_key(prompt_name, t)).collect();

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_indices: Vec<usize> = Vec::new();

        // L1 check
        for (i, (kb, _)) in keys.iter().enumerate() {
            if let Some(cached) = self.l1.borrow_mut().get(kb).cloned() {
                results[i] = Some(cached);
                self.hits_l1.set(self.hits_l1.get() + 1);
            } else {
                miss_indices.push(i);
            }
        }

        // L2 check for L1 misses
        if self.l2.is_some() {
            let mut still_missing = Vec::new();
            for &idx in &miss_indices {
                let (kb, ref kh) = keys[idx];
                if let Some(embedding) = self.l2_get(kb, kh).await {
                    results[idx] = Some(embedding);
                } else {
                    still_missing.push(idx);
                }
            }
            miss_indices = still_missing;
        }

        if miss_indices.is_empty() {
            return Ok(results.into_iter().map(|o| o.unwrap()).collect());
        }

        self.misses
            .set(self.misses.get() + miss_indices.len() as u64);

        let miss_texts: Vec<&str> = miss_indices.iter().map(|&i| texts[i]).collect();
        let fresh = self.inner.embed_batch(&miss_texts, prompt_name).await?;

        // Populate both tiers
        for (&idx, embedding) in miss_indices.iter().zip(fresh) {
            let (kb, ref kh) = keys[idx];
            self.l1.borrow_mut().insert(kb, embedding.clone());
            self.l2_set(kh, &embedding).await;
            results[idx] = Some(embedding);
        }

        Ok(results.into_iter().map(|o| o.unwrap()).collect())
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }
}
