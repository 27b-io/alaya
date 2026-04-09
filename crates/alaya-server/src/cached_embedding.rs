//! Caching decorator for EmbeddingProvider.
//!
//! Embeddings are content-addressed (same text → same vector), so cache hits
//! skip the TEI HTTP round-trip entirely. Uses mini-moka's unsync::Cache
//! which matches the ?Send / LocalSet execution model.

use std::cell::{Cell, RefCell};

use async_trait::async_trait;
use mini_moka::unsync::Cache;
use sha2::{Digest, Sha256};

use alaya_backends::traits::EmbeddingProvider;
use alaya_types::Result;
use alaya_types::search::PromptName;

/// Cache key: SHA-256 of (prompt_name, text). 32 bytes, no allocations beyond the hash.
fn cache_key(prompt: PromptName, text: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_str().as_bytes());
    hasher.update(b":");
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

pub struct CachedEmbedding {
    inner: Box<dyn EmbeddingProvider>,
    cache: RefCell<Cache<[u8; 32], Vec<f32>>>,
    hits: Cell<u64>,
    misses: Cell<u64>,
}

impl CachedEmbedding {
    pub fn new(inner: Box<dyn EmbeddingProvider>, max_capacity: u64) -> Self {
        Self {
            inner,
            cache: RefCell::new(Cache::builder().max_capacity(max_capacity).build()),
            hits: Cell::new(0),
            misses: Cell::new(0),
        }
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> (u64, u64) {
        (self.hits.get(), self.misses.get())
    }
}

#[async_trait(?Send)]
impl EmbeddingProvider for CachedEmbedding {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
        let mut results: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<&str> = Vec::new();

        for (i, text) in texts.iter().enumerate() {
            let key = cache_key(prompt_name, text);
            if let Some(cached) = self.cache.borrow_mut().get(&key).cloned() {
                results.push(Some(cached));
                self.hits.set(self.hits.get() + 1);
            } else {
                results.push(None);
                miss_indices.push(i);
                miss_texts.push(text);
            }
        }

        // All cached — skip HTTP entirely
        if miss_texts.is_empty() {
            return Ok(results.into_iter().map(|o| o.unwrap()).collect());
        }

        self.misses.set(self.misses.get() + miss_texts.len() as u64);

        // Embed only the misses
        let fresh = self.inner.embed_batch(&miss_texts, prompt_name).await?;

        // Merge fresh results and populate cache
        {
            let mut cache = self.cache.borrow_mut();
            for (idx, embedding) in miss_indices.into_iter().zip(fresh) {
                let key = cache_key(prompt_name, texts[idx]);
                cache.insert(key, embedding.clone());
                results[idx] = Some(embedding);
            }
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
