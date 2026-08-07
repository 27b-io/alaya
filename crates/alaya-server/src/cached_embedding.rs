//! Two-tier caching decorator for EmbeddingProvider.
//!
//! L1: mini-moka unsync in-process cache (hot path, ~0ns).
//! L2: cachekit-rs (Redis by default, cachekit.io SaaS via `CACHE_BACKEND=saas`
//! — persists across restarts, shared across pods).
//!
//! Embeddings are immutable and content-addressed — same text always produces
//! the same vector. Cache keys are cross-SDK interop/v1
//! (`alaya:embed:{blake2b256-hex}` over the canonical argument array
//! `[model, dims, prompt_name, text]`), so any cachekit SDK computing the same
//! arguments derives the same key and the L2 cache is shareable across SDKs.
//!
//! L2 has a circuit breaker: after 3 consecutive failures, L2 is bypassed
//! for 30 seconds before retrying. Prevents hammering a dead Redis.

use std::cell::{Cell, RefCell};

use async_trait::async_trait;
use mini_moka::unsync::Cache;

use alaya_backends::traits::EmbeddingProvider;
use alaya_types::Result;
use alaya_types::search::PromptName;

/// Consecutive L2 failures before the circuit opens.
const BREAKER_THRESHOLD: u32 = 3;
/// Seconds to wait before retrying L2 after circuit opens.
const BREAKER_COOLDOWN_SECS: f64 = 30.0;
/// Per-op deadline for L2 reads/writes (Redis round-trips normally take
/// <10ms; SaaS WAN round-trips hundreds of ms — ops in a batch run
/// concurrently, so the batch wall clock stays ~this bound either way).
///
/// cachekit-rs sets no fred command timeout (default = wait forever) and
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

/// Cache key: interop/v1 (`alaya:embed:` + Blake2b-256 hex over the canonical
/// argument array `[model, dims, prompt_name, text]`). Cross-SDK shareable —
/// any cachekit SDK hashing the same arguments derives the same key (spec:
/// cachekit-protocol/spec/interop-mode.md). Model and dims live in the args,
/// not a client namespace — see [`build_l2_client`] for why that matters.
///
/// Replaced the legacy `SHA-256(prompt:text)` key under a
/// `alaya:embed:{model}:{dims}` client namespace (LAB-372, 2026-08-08).
fn cache_key(model: &str, dims: usize, prompt: PromptName, text: &str) -> Result<String> {
    use cachekit::interop::{InteropValue, interop_key};
    interop_key(
        "alaya",
        "embed",
        &[
            InteropValue::from(model),
            InteropValue::Int(dims as i128),
            InteropValue::from(prompt.as_str()),
            InteropValue::from(text),
        ],
    )
    // Infallible by construction: both segments satisfy the key grammar and
    // Str / in-range Int args always encode. An Err means cachekit changed
    // its contract — surface it as a request error rather than panicking,
    // which would kill the LocalSet worker thread the service runs on.
    .map_err(|e| {
        alaya_types::AlayaError::Embedding(format!("interop cache key construction failed: {e}"))
    })
}

/// Build the L2 CacheKit client used for embeddings.
///
/// MUST stay un-namespaced: interop keys carry their own `alaya:embed:`
/// segment, and a client namespace silently prefixes `ns:` onto every plain
/// `get`/`set` key (unlike `interop_get`, which fails closed), producing keys
/// no other SDK can compute. Verified at the wire level by
/// `l2_keys_reach_backend_verbatim` below.
pub fn build_l2_client(
    backend: cachekit::SharedBackend,
) -> std::result::Result<cachekit::CacheKit, cachekit::CachekitError> {
    cachekit::CacheKit::builder()
        .backend(backend)
        .default_ttl(std::time::Duration::from_secs(86400 * 30))
        .no_l1()
        .build()
}

pub struct CachedEmbedding {
    inner: Box<dyn EmbeddingProvider>,
    /// L1: in-process, single-threaded, zero-cost hit. Keyed by the same
    /// interop key as L2 — one key derivation per text, two tiers.
    l1: RefCell<Cache<String, Vec<f32>>>,
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

    /// Feed the shared circuit breaker with one batch's per-op outcomes.
    ///
    /// Per-op deadlines make partial failure the normal degraded mode, so
    /// batch-level flags would let a single fast key reset the breaker while
    /// the rest of the batch times out. Failure wins ties: a half-dead L2
    /// (e.g. 1 hit + 1 timeout per 2-key batch) must still trip the breaker
    /// it exists for (#63), and a half-open probe that ties must not re-close
    /// it.
    fn record_l2_batch(
        &self,
        op: &'static str,
        successes: usize,
        failures: usize,
        timeouts: usize,
    ) {
        if timeouts > 0 {
            tracing::warn!(
                op,
                timeouts,
                timeout_s = L2_OP_TIMEOUT.as_secs(),
                "L2 cache ops timed out — treating as failures (non-fatal)"
            );
        }
        if failures > 0 && failures >= successes {
            self.l2_failure();
        } else if successes > 0 {
            self.l2_success();
        }
    }

    /// Concurrent L2 reads for a batch of keys. Reads use `interop_get`
    /// (strict interop/v1 decode): an off-format entry written by a foreign
    /// SDK becomes a diagnosable per-key miss instead of a breaker-tripping
    /// decode error — the exact mixed-SDK deployment this key format enables.
    async fn l2_get_batch(&self, keys: &[&str]) -> Vec<Option<Vec<f32>>> {
        let l2 = match self.l2.as_ref() {
            Some(l2) => l2,
            None => return vec![None; keys.len()],
        };

        // Per-op deadline (not whole-batch): ops run concurrently so the
        // wall clock stays ~L2_OP_TIMEOUT, a single slow key degrades to a
        // per-key miss instead of failing the whole batch, and the budget
        // holds for both in-cluster Redis and WAN SaaS round-trips.
        let futs: Vec<_> = keys
            .iter()
            .map(|k| tokio::time::timeout(L2_OP_TIMEOUT, l2.interop_get::<Vec<f32>>(k)))
            .collect();
        let raw_results = futures::future::join_all(futs).await;

        let mut successes = 0usize;
        let mut failures = 0usize;
        let mut timeouts = 0usize;

        let results: Vec<Option<Vec<f32>>> = raw_results
            .into_iter()
            .zip(keys)
            .map(|(result, k)| match result {
                Ok(Ok(Some(embedding))) => {
                    self.l1
                        .borrow_mut()
                        .insert(k.to_string(), embedding.clone());
                    self.hits_l2.set(self.hits_l2.get() + 1);
                    successes += 1;
                    Some(embedding)
                }
                Ok(Ok(None)) => {
                    successes += 1;
                    None
                }
                Ok(Err(e)) => {
                    tracing::debug!("L2 cache get failed (non-fatal): {e}");
                    failures += 1;
                    None
                }
                Err(_) => {
                    timeouts += 1;
                    failures += 1;
                    None
                }
            })
            .collect();

        self.record_l2_batch("get", successes, failures, timeouts);

        results
    }

    /// Batch L2 writes — concurrent, per-op deadline, errors tracked by
    /// circuit breaker.
    ///
    /// Plain `set` is the interop write path: cachekit-rs has no
    /// `interop_set` because `set` already stores plain MessagePack with no
    /// ByteStorage envelope — exactly the interop/v1 value format that
    /// `interop_get` strict-decodes (documented on `CacheKit::interop_get`).
    /// The write/read symmetry is proven cross-client by
    /// `l2_keys_reach_backend_verbatim` below.
    async fn l2_set_batch(&self, entries: &[(&str, &Vec<f32>)]) {
        let l2 = match self.l2.as_ref() {
            Some(l2) => l2,
            None => return,
        };
        let futs: Vec<_> = entries
            .iter()
            .map(|(k, v)| tokio::time::timeout(L2_OP_TIMEOUT, l2.set(k, *v)))
            .collect();
        let results = futures::future::join_all(futs).await;

        let mut successes = 0usize;
        let mut failures = 0usize;
        let mut timeouts = 0usize;
        for result in results {
            match result {
                Ok(Ok(())) => successes += 1,
                Ok(Err(e)) => {
                    tracing::debug!("L2 cache set failed (non-fatal): {e}");
                    failures += 1;
                }
                Err(_) => {
                    timeouts += 1;
                    failures += 1;
                }
            }
        }
        self.record_l2_batch("set", successes, failures, timeouts);
    }
}

#[async_trait(?Send)]
impl EmbeddingProvider for CachedEmbedding {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
        let model = self.inner.model_name();
        let dims = self.inner.dimensions();
        let keys: Vec<String> = texts
            .iter()
            .map(|t| cache_key(model, dims, prompt_name, t))
            .collect::<Result<_>>()?;

        let mut results: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_indices: Vec<usize> = Vec::new();

        // L1 check
        for (i, k) in keys.iter().enumerate() {
            if let Some(cached) = self.l1.borrow_mut().get(k).cloned() {
                results[i] = Some(cached);
                self.hits_l1.set(self.hits_l1.get() + 1);
            } else {
                miss_indices.push(i);
            }
        }

        // L2 check — fan out concurrently (if circuit breaker allows)
        if self.l2_available() && !miss_indices.is_empty() {
            let l2_keys: Vec<&str> = miss_indices.iter().map(|&i| keys[i].as_str()).collect();
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

        // Populate L1 + results
        for (&idx, embedding) in miss_indices.iter().zip(fresh) {
            self.l1
                .borrow_mut()
                .insert(keys[idx].clone(), embedding.clone());
            results[idx] = Some(embedding);
        }

        // Batch L2 writes concurrently (if circuit breaker allows)
        if self.l2_available() {
            let entries: Vec<(&str, &Vec<f32>)> = miss_indices
                .iter()
                .filter_map(|&idx| Some((keys[idx].as_str(), results[idx].as_ref()?)))
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
        let l2 = build_l2_client(std::rc::Rc::new(HangingBackend)).unwrap();
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

    /// In-memory backend that records every key it is handed, verbatim.
    #[derive(Default)]
    struct RecordingBackend {
        store: RefCell<std::collections::HashMap<String, Vec<u8>>>,
        keys_seen: RefCell<Vec<String>>,
    }

    #[async_trait(?Send)]
    impl cachekit::backend::Backend for RecordingBackend {
        async fn get(&self, key: &str) -> BackendResult<Option<Vec<u8>>> {
            self.keys_seen.borrow_mut().push(key.to_string());
            Ok(self.store.borrow().get(key).cloned())
        }
        async fn set(
            &self,
            key: &str,
            value: Vec<u8>,
            _ttl: Option<std::time::Duration>,
        ) -> BackendResult<()> {
            self.keys_seen.borrow_mut().push(key.to_string());
            self.store.borrow_mut().insert(key.to_string(), value);
            Ok(())
        }
        async fn delete(&self, key: &str) -> BackendResult<bool> {
            Ok(self.store.borrow_mut().remove(key).is_some())
        }
        async fn exists(&self, key: &str) -> BackendResult<bool> {
            Ok(self.store.borrow().contains_key(key))
        }
        async fn health(&self) -> BackendResult<cachekit::backend::HealthStatus> {
            Ok(cachekit::backend::HealthStatus {
                is_healthy: true,
                latency_ms: 0.0,
                backend_type: "recording".to_string(),
                details: Default::default(),
            })
        }
    }

    fn is_interop_embed_key(k: &str) -> bool {
        k.strip_prefix("alaya:embed:").is_some_and(|h| {
            h.len() == 64
                && h.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        })
    }

    /// LAB-372 Phase 2 namespace trap (see [`build_l2_client`]): keys must
    /// reach the backend as verbatim interop/v1 keys — `alaya:embed:` + 64
    /// lowercase hex, no client `ns:` prefix. Exercises the same
    /// `build_l2_client` production uses, and proves a second process
    /// (cold L1, shared L2) hits the cache.
    #[tokio::test]
    async fn l2_keys_reach_backend_verbatim() {
        let backend = std::rc::Rc::new(RecordingBackend::default());
        let l2_writer = build_l2_client(backend.clone()).unwrap();
        let writer = CachedEmbedding::new(Box::new(StubEmbeddings), 10, Some(l2_writer));
        writer
            .embed_batch(&["hello"], PromptName::Passage)
            .await
            .unwrap();

        // Fresh client + fresh L1 over the same backend — models another pod
        // (or another SDK that derived the same interop key).
        let l2_reader = build_l2_client(backend.clone()).unwrap();
        let reader = CachedEmbedding::new(Box::new(StubEmbeddings), 10, Some(l2_reader));
        reader
            .embed_batch(&["hello"], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(reader.hits_l2.get(), 1, "second process must hit shared L2");

        let keys = backend.keys_seen.borrow();
        assert!(!keys.is_empty());
        for k in keys.iter() {
            assert!(
                is_interop_embed_key(k),
                "key on the wire is not a verbatim interop/v1 key (namespaced client?): {k}"
            );
        }
    }

    /// Key derivation is deterministic and every argument is key-affecting.
    #[test]
    fn interop_key_deterministic_and_arg_sensitive() {
        let key = |m, d, p, t| cache_key(m, d, p, t).unwrap();
        let k = key("model-a", 1024, PromptName::Passage, "hello");
        assert_eq!(k, key("model-a", 1024, PromptName::Passage, "hello"));
        assert!(is_interop_embed_key(&k));

        assert_ne!(k, key("model-b", 1024, PromptName::Passage, "hello"));
        assert_ne!(k, key("model-a", 512, PromptName::Passage, "hello"));
        assert_ne!(k, key("model-a", 1024, PromptName::Query, "hello"));
        assert_ne!(k, key("model-a", 1024, PromptName::Passage, "world"));
    }
}
