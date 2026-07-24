//! Two-tier caching decorator for EmbeddingProvider.
//!
//! L1: mini-moka unsync in-process cache (hot path, ~0ns).
//! L2: cachekit-rs Redis (persists across restarts, shared across pods).
//!
//! Embeddings are immutable and content-addressed — same text always produces
//! the same vector. Cache keys are SHA-256(prompt_name + ":" + text).
//!
//! L2 has a circuit breaker: after 3 consecutive failures, L2 is bypassed
//! for 30 seconds before retrying. Prevents hammering a dead Redis.

use std::cell::{Cell, RefCell};

use async_trait::async_trait;
use mini_moka::unsync::Cache;
use sha2::{Digest, Sha256};

use alaya_backends::traits::EmbeddingProvider;
use alaya_types::Result;
use alaya_types::search::PromptName;

/// Consecutive L2 failures before the circuit opens.
const BREAKER_THRESHOLD: u32 = 3;
/// Seconds to wait before retrying L2 after circuit opens.
const BREAKER_COOLDOWN_SECS: f64 = 30.0;
/// Deadline for one L2 batch (Redis round-trips normally take <10ms).
///
/// cachekit-rs 0.3 sets no fred command timeout (default = wait forever) and
/// no reconnect policy, so a blackholed connection — e.g. the Redis pod IP
/// vanishing without an RST — hangs `get`/`set` awaits indefinitely. That
/// silence never trips the circuit breaker (it only counts errors) and wedged
/// the whole service for 25h (#63). This timeout converts the hang into a
/// failure the breaker can act on, degrading to L1-only.
const L2_OP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

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
    /// Circuit breaker: consecutive L2 failure count.
    l2_failures: Cell<u32>,
    /// Timestamp when circuit opened (0.0 = closed).
    l2_open_since: Cell<f64>,
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
            l2_failures: Cell::new(0),
            l2_open_since: Cell::new(0.0),
        }
    }

    #[allow(dead_code)]
    pub fn stats(&self) -> (u64, u64, u64) {
        (self.hits_l1.get(), self.hits_l2.get(), self.misses.get())
    }

    /// Check if L2 circuit breaker allows a request.
    fn l2_available(&self) -> bool {
        if self.l2.is_none() {
            return false;
        }
        let open_since = self.l2_open_since.get();
        if open_since == 0.0 {
            return true; // closed
        }
        // Half-open: try again after cooldown
        if now_secs() - open_since >= BREAKER_COOLDOWN_SECS {
            return true;
        }
        false // open
    }

    /// Record L2 success — reset breaker.
    fn l2_success(&self) {
        if self.l2_failures.get() > 0 {
            self.l2_failures.set(0);
            if self.l2_open_since.get() != 0.0 {
                tracing::info!("L2 cache circuit breaker closed (recovered)");
                self.l2_open_since.set(0.0);
            }
        }
    }

    /// Record L2 failure — trip breaker after threshold, refresh on half-open failure.
    fn l2_failure(&self) {
        let n = self.l2_failures.get() + 1;
        self.l2_failures.set(n);
        if self.l2_open_since.get() != 0.0 {
            // Half-open probe failed — re-enter open state with fresh cooldown
            self.l2_open_since.set(now_secs());
            return;
        }
        if n >= BREAKER_THRESHOLD {
            tracing::warn!(
                failures = n,
                cooldown_secs = BREAKER_COOLDOWN_SECS,
                "L2 cache circuit breaker OPEN — bypassing Redis"
            );
            self.l2_open_since.set(now_secs());
        }
    }

    /// Concurrent L2 reads for a batch of keys.
    async fn l2_get_batch(&self, keys: &[([u8; 32], &str)]) -> Vec<Option<Vec<f32>>> {
        let l2 = match self.l2.as_ref() {
            Some(l2) => l2,
            None => return vec![None; keys.len()],
        };

        let futs: Vec<_> = keys.iter().map(|(_, kh)| l2.get::<Vec<f32>>(kh)).collect();
        let raw_results =
            match tokio::time::timeout(L2_OP_TIMEOUT, futures::future::join_all(futs)).await {
                Ok(r) => r,
                Err(_) => {
                    tracing::warn!(
                        timeout_s = L2_OP_TIMEOUT.as_secs(),
                        "L2 cache get batch timed out — treating as failure (non-fatal)"
                    );
                    self.l2_failure();
                    return vec![None; keys.len()];
                }
            };

        let mut any_success = false;
        let mut any_failure = false;

        let results: Vec<Option<Vec<f32>>> = raw_results
            .into_iter()
            .zip(keys)
            .map(|(result, (kb, _))| match result {
                Ok(Some(embedding)) => {
                    self.l1.borrow_mut().insert(*kb, embedding.clone());
                    self.hits_l2.set(self.hits_l2.get() + 1);
                    any_success = true;
                    Some(embedding)
                }
                Ok(None) => {
                    any_success = true;
                    None
                }
                Err(e) => {
                    tracing::debug!("L2 cache get failed (non-fatal): {e}");
                    any_failure = true;
                    None
                }
            })
            .collect();

        if any_success {
            self.l2_success();
        } else if any_failure {
            self.l2_failure();
        }

        results
    }

    /// Batch L2 writes — concurrent, errors tracked by circuit breaker.
    async fn l2_set_batch(&self, entries: &[(&str, &Vec<f32>)]) {
        let l2 = match self.l2.as_ref() {
            Some(l2) => l2,
            None => return,
        };
        let futs: Vec<_> = entries.iter().map(|(k, v)| l2.set(k, *v)).collect();
        let results =
            match tokio::time::timeout(L2_OP_TIMEOUT, futures::future::join_all(futs)).await {
                Ok(r) => r,
                Err(_) => {
                    tracing::warn!(
                        timeout_s = L2_OP_TIMEOUT.as_secs(),
                        "L2 cache set batch timed out — treating as failure (non-fatal)"
                    );
                    self.l2_failure();
                    return;
                }
            };
        let mut any_failure = false;
        for result in results {
            if let Err(e) = result {
                tracing::debug!("L2 cache set failed (non-fatal): {e}");
                any_failure = true;
            }
        }
        if any_failure {
            self.l2_failure();
        } else {
            self.l2_success();
        }
    }
}

#[async_trait(?Send)]
impl EmbeddingProvider for CachedEmbedding {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
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

        // L2 check — fan out concurrently (if circuit breaker allows)
        if self.l2_available() && !miss_indices.is_empty() {
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

        // Batch L2 writes concurrently (if circuit breaker allows)
        if self.l2_available() {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Backend whose every op blackholes — models a Redis connection to a
    /// vanished pod IP (no RST, no error, just silence). This is the
    /// suspected trigger of the 25h wedge (#63).
    struct HangingBackend;

    #[async_trait(?Send)]
    impl cachekit::backend::Backend for HangingBackend {
        async fn get(&self, _key: &str) -> BackendResult<Option<Vec<u8>>> {
            std::future::pending().await
        }
        async fn set(
            &self,
            _key: &str,
            _value: Vec<u8>,
            _ttl: Option<std::time::Duration>,
        ) -> BackendResult<()> {
            std::future::pending().await
        }
        async fn delete(&self, _key: &str) -> BackendResult<bool> {
            std::future::pending().await
        }
        async fn exists(&self, _key: &str) -> BackendResult<bool> {
            std::future::pending().await
        }
        async fn health(&self) -> BackendResult<cachekit::backend::HealthStatus> {
            std::future::pending().await
        }
    }

    type BackendResult<T> = std::result::Result<T, cachekit::BackendError>;

    struct StubEmbeddings;

    #[async_trait(?Send)]
    impl EmbeddingProvider for StubEmbeddings {
        async fn embed_batch(
            &self,
            texts: &[&str],
            _prompt_name: PromptName,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.5_f32; 4]).collect())
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn model_name(&self) -> &str {
            "stub"
        }
    }

    /// A blackholed L2 must not hang embed_batch: both batch phases time out
    /// (each counts as one breaker failure) and the inner provider answers.
    /// Paused clock — the 3s timeouts elapse instantly.
    #[tokio::test(start_paused = true)]
    async fn hanging_l2_times_out_and_degrades_to_inner_provider() {
        let l2 = cachekit::CacheKit::builder()
            .backend(std::rc::Rc::new(HangingBackend))
            .namespace("test")
            .no_l1()
            .build()
            .unwrap();
        let cache = CachedEmbedding::new(Box::new(StubEmbeddings), 10, Some(l2));

        let out = cache
            .embed_batch(&["hello"], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 4);
        // get-batch and set-batch each timed out → two breaker failures.
        assert_eq!(cache.l2_failures.get(), 2);
    }
}
