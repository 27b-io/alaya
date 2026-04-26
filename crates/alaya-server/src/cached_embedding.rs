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

    /// Concurrent L2 reads for a batch of keys. Returns results in same order.
    /// Handles backfill into L1 and hit counting after all reads complete.
    async fn l2_get_batch(&self, keys: &[([u8; 32], &str)]) -> Vec<Option<Vec<f32>>> {
        let l2 = match self.l2.as_ref() {
            Some(l2) => l2,
            None => return vec![None; keys.len()],
        };

        // Fan out all Redis reads concurrently (single thread, interleaved I/O)
        let futs: Vec<_> = keys.iter().map(|(_, kh)| l2.get::<Vec<f32>>(kh)).collect();
        let raw_results = futures::future::join_all(futs).await;

        // Process results: backfill L1, count hits
        raw_results
            .into_iter()
            .zip(keys)
            .map(|(result, (kb, _))| match result {
                Ok(Some(embedding)) => {
                    self.l1.borrow_mut().insert(*kb, embedding.clone());
                    self.hits_l2.set(self.hits_l2.get() + 1);
                    Some(embedding)
                }
                Ok(None) => None,
                Err(e) => {
                    tracing::debug!("L2 cache get failed (non-fatal): {e}");
                    None
                }
            })
            .collect()
    }

    /// Batch L2 writes — concurrent, errors logged.
    async fn l2_set_batch(&self, entries: &[(&str, &Vec<f32>)]) {
        let l2 = match self.l2.as_ref() {
            Some(l2) => l2,
            None => return,
        };
        let futs: Vec<_> = entries.iter().map(|(k, v)| l2.set(k, *v)).collect();
        for result in futures::future::join_all(futs).await {
            if let Err(e) = result {
                tracing::debug!("L2 cache set failed (non-fatal): {e}");
            }
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

        // L2 check — fan out all misses concurrently
        if self.l2.is_some() && !miss_indices.is_empty() {
            let l2_keys: Vec<([u8; 32], &str)> = miss_indices
                .iter()
                .map(|&i| (keys[i].0, keys[i].1.as_str()))
                .collect();
            let l2_results = self.l2_get_batch(&l2_keys).await;

            let mut still_missing = Vec::new();
            for (&idx, l2_result) in miss_indices.iter().zip(l2_results) {
                if let Some(embedding) = l2_result {
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

        // Populate L1 + results, collect keys for L2 batch write
        let mut l2_write_keys: Vec<usize> = Vec::new();
        for (&idx, embedding) in miss_indices.iter().zip(fresh) {
            let (kb, _) = keys[idx];
            self.l1.borrow_mut().insert(kb, embedding.clone());
            l2_write_keys.push(idx);
            results[idx] = Some(embedding);
        }

        // Batch L2 writes concurrently (non-blocking: results already assigned)
        if self.l2.is_some() {
            let entries: Vec<(&str, &Vec<f32>)> = l2_write_keys
                .iter()
                .filter_map(|&idx| {
                    let kh = keys[idx].1.as_str();
                    let emb = results[idx].as_ref()?;
                    Some((kh, emb))
                })
                .collect();
            self.l2_set_batch(&entries).await;
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
