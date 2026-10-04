//! Failure counters and the self-check (LAB-4026), behind the `pod` section
//! of `GET /stats` and `/health/detail`.
//!
//! An embedder can answer its readiness check and still fail every embed,
//! and a reranker that is down only costs ranking quality, so neither shows
//! on a probe. These counters and a periodic read-only search make both
//! visible. Counts run from process start and are per pod: a restart zeroes
//! them and each replica keeps its own, which is why the section carries
//! the pod name and start time.
//!
//! Atomics and one mutex, because the worker thread writes them and the
//! health checker on the axum runtime reads them.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::{Value, json};

use alaya_backends::EmbeddingProvider;
use alaya_types::memory::HealthStatus;
use alaya_types::search::{PromptName, SearchMode};
use alaya_types::{AlayaError, Result};

use crate::service::{MemoryService, OutputMode, SearchParams, with_budget};

/// Results the expected memory must appear in.
pub const SELFCHECK_TOP_K: usize = 10;

/// What one hybrid search's rerank pass did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rerank {
    /// No reranker configured.
    NotConfigured,
    /// Nothing to rerank: the search found no candidates.
    NoCandidates,
    Ran,
    /// Error, timeout or score-count mismatch (the cause): RRF order was
    /// served.
    FellBack(&'static str),
}

impl Rerank {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NotConfigured => "not_configured",
            Self::NoCandidates => "no_candidates",
            Self::Ran => "ran",
            Self::FellBack(_) => "fell_back",
        }
    }
}

/// The self-check step that failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// The embedder could not embed the query.
    Embed,
    /// The hybrid search itself failed.
    Search,
    /// The search ran, but the reranker fell back to RRF order.
    Rerank,
    /// The search ran, but the expected memory was not in the top 10.
    Expect,
    /// The server got no answer from the worker (overloaded, wedged or gone).
    Worker,
}

impl Step {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::Search => "search",
            Self::Rerank => "rerank",
            Self::Expect => "expect",
            Self::Worker => "worker",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelfCheckOutcome {
    /// `None` is a pass.
    pub failing_step: Option<Step>,
    pub error: Option<String>,
    /// `None` when the check failed before the search finished.
    pub rerank: Option<Rerank>,
}

impl SelfCheckOutcome {
    pub fn fail(step: Step, error: impl Into<String>, rerank: Option<Rerank>) -> Self {
        Self {
            failing_step: Some(step),
            error: Some(error.into()),
            rerank,
        }
    }

    pub fn ok(&self) -> bool {
        self.failing_step.is_none()
    }
}

#[derive(Default)]
struct SelfCheckState {
    /// Unix seconds, elapsed ms and outcome of the latest run.
    last: Option<(u64, u64, SelfCheckOutcome)>,
    consecutive_failures: u64,
}

#[derive(Default)]
pub struct Vitals {
    pod: String,
    started_at: u64,
    selfcheck_enabled: bool,
    embed_failures: AtomicU64,
    rerank_failures: AtomicU64,
    store_failures: AtomicU64,
    selfcheck: Mutex<SelfCheckState>,
}

impl Vitals {
    pub fn new(pod: String, started_at: u64, selfcheck_enabled: bool) -> Self {
        Self {
            pod,
            started_at,
            selfcheck_enabled,
            ..Default::default()
        }
    }

    pub fn embed_failed(&self) {
        self.embed_failures.fetch_add(1, Relaxed);
    }

    pub fn rerank_failed(&self) {
        self.rerank_failures.fetch_add(1, Relaxed);
    }

    pub fn store_failed(&self) {
        self.store_failures.fetch_add(1, Relaxed);
    }

    /// Record a self-check run; returns the consecutive failures it leaves.
    pub fn record_selfcheck(&self, at: u64, elapsed_ms: u64, outcome: SelfCheckOutcome) -> u64 {
        let mut s = self.lock();
        s.consecutive_failures = if outcome.ok() {
            0
        } else {
            s.consecutive_failures + 1
        };
        s.last = Some((at, elapsed_ms, outcome));
        s.consecutive_failures
    }

    /// Whether the latest self-check failed. False before the first run.
    pub fn selfcheck_failing(&self) -> bool {
        self.lock().last.as_ref().is_some_and(|(_, _, o)| !o.ok())
    }

    /// The `pod` section. See `docs/rest-api.md`.
    pub fn snapshot(&self) -> Value {
        let s = self.lock();
        let last = s.last.as_ref().map(|(at, elapsed_ms, o)| {
            json!({
                "at": at,
                "ok": o.ok(),
                "failing_step": o.failing_step.map(Step::as_str),
                "error": o.error,
                "rerank": o.rerank.map(Rerank::as_str),
                "elapsed_ms": elapsed_ms,
            })
        });
        json!({
            "name": self.pod,
            "started_at": self.started_at,
            "failures": {
                "embedding": self.embed_failures.load(Relaxed),
                "rerank": self.rerank_failures.load(Relaxed),
                "store": self.store_failures.load(Relaxed),
            },
            "selfcheck": {
                "enabled": self.selfcheck_enabled,
                "consecutive_failures": s.consecutive_failures,
                "last": last,
            },
        })
    }

    /// A poisoned lock still holds plain counters: keep serving them.
    fn lock(&self) -> std::sync::MutexGuard<'_, SelfCheckState> {
        self.selfcheck
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// Counts every failed embed the wrapped provider returns. Wrap the raw
/// client, under any cache: a cache hit is not a call that could fail.
pub struct CountingEmbedding {
    inner: Box<dyn EmbeddingProvider>,
    vitals: Arc<Vitals>,
}

impl CountingEmbedding {
    pub fn new(inner: Box<dyn EmbeddingProvider>, vitals: Arc<Vitals>) -> Self {
        Self { inner, vitals }
    }

    fn count<T>(&self, r: Result<T>) -> Result<T> {
        if r.is_err() {
            self.vitals.embed_failed();
        }
        r
    }
}

#[async_trait(?Send)]
impl EmbeddingProvider for CountingEmbedding {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
        let r = self.inner.embed_batch(texts, prompt_name).await;
        self.count(r)
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    /// Not counted: a readiness check, not an embed.
    async fn health(&self) -> Result<HealthStatus> {
        self.inner.health().await
    }

    async fn probe(&self, text: &str) -> Result<()> {
        let r = self.inner.probe(text).await;
        self.count(r)
    }
}

impl MemoryService {
    /// One self-check: embed `query` past any cache, then run it as a
    /// read-only hybrid search (no access bump, no Hebbian write) and look
    /// for `expect_hash` in the top 10, all within `budget`. A rerank
    /// fallback fails the check even when the memory is found: a reranker
    /// that is down every time costs ranking quality and nothing else
    /// would say so.
    pub async fn self_check(
        &self,
        query: &str,
        expect_hash: &str,
        budget: std::time::Duration,
    ) -> SelfCheckOutcome {
        let step = std::cell::Cell::new(Step::Embed);
        let run = async {
            self.embeddings
                .probe(query)
                .await
                .map_err(|e| SelfCheckOutcome::fail(Step::Embed, e.to_string(), None))?;
            step.set(Step::Search);
            let params = SearchParams {
                query: query.into(),
                mode: SearchMode::Hybrid,
                page: 1,
                page_size: SELFCHECK_TOP_K,
                tags: None,
                match_all: false,
                k: SELFCHECK_TOP_K,
                min_similarity: None,
                memory_type: None,
                encoding_context: None,
                include_superseded: false,
                min_trust_score: None,
                output: OutputMode::Summary,
                cursor: None,
            };
            let (result, rerank) = self.search_hybrid(&params, true).await.map_err(|e| {
                // The search embeds again; if that call is the one that
                // failed, the embedder is the broken step.
                let step = match e {
                    AlayaError::Embedding(_) => Step::Embed,
                    _ => Step::Search,
                };
                SelfCheckOutcome::fail(step, e.to_string(), None)
            })?;
            if let Rerank::FellBack(cause) = rerank {
                return Err(SelfCheckOutcome::fail(
                    Step::Rerank,
                    format!("rerank fell back to RRF order: {cause}"),
                    Some(rerank),
                ));
            }
            let found = result["results"]
                .as_array()
                .is_some_and(|r| r.iter().any(|m| m["content_hash"] == expect_hash));
            if !found {
                return Err(SelfCheckOutcome::fail(
                    Step::Expect,
                    format!("expected memory not in the top {SELFCHECK_TOP_K}"),
                    Some(rerank),
                ));
            }
            Ok(SelfCheckOutcome {
                failing_step: None,
                error: None,
                rerank: Some(rerank),
            })
        };
        match with_budget(budget, run).await {
            Some(Ok(o)) | Some(Err(o)) => o,
            None => SelfCheckOutcome::fail(
                step.get(),
                format!("timed out after {}s", budget.as_secs()),
                None,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Flaky(bool);

    #[async_trait(?Send)]
    impl EmbeddingProvider for Flaky {
        async fn embed_batch(&self, texts: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            if self.0 {
                return Err(AlayaError::Embedding("down".into()));
            }
            Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
        }
        fn dimensions(&self) -> usize {
            4
        }
        fn model_name(&self) -> &str {
            "flaky"
        }
        async fn health(&self) -> Result<HealthStatus> {
            Err(AlayaError::Embedding("down".into()))
        }
    }

    /// Failed embeds and probes count; successes and readiness checks don't.
    #[tokio::test(flavor = "current_thread")]
    async fn counting_embedding_counts_failed_embeds_only() {
        let vitals = Arc::new(Vitals::default());
        let up = CountingEmbedding::new(Box::new(Flaky(false)), vitals.clone());
        let down = CountingEmbedding::new(Box::new(Flaky(true)), vitals.clone());

        up.embed_batch(&["x"], PromptName::Query).await.unwrap();
        up.probe("x").await.unwrap();
        assert!(down.embed_batch(&["x"], PromptName::Query).await.is_err());
        assert!(down.probe("x").await.is_err());
        assert!(down.health().await.is_err());

        assert_eq!(vitals.snapshot()["failures"]["embedding"], 2);
    }

    #[test]
    fn record_selfcheck_tracks_consecutive_failures_and_resets_on_pass() {
        let v = Vitals::new("alaya-server-0".into(), 1_700_000_000, true);
        assert!(!v.selfcheck_failing(), "no run yet is not a failure");
        let fail = || SelfCheckOutcome::fail(Step::Embed, "down", None);

        assert_eq!(v.record_selfcheck(1, 5, fail()), 1);
        assert_eq!(v.record_selfcheck(2, 5, fail()), 2);
        assert!(v.selfcheck_failing());
        let s = v.snapshot();
        assert_eq!(s["name"], "alaya-server-0");
        assert_eq!(s["started_at"], 1_700_000_000);
        assert_eq!(s["selfcheck"]["enabled"], true);
        assert_eq!(s["selfcheck"]["consecutive_failures"], 2);
        assert_eq!(s["selfcheck"]["last"]["at"], 2);
        assert_eq!(s["selfcheck"]["last"]["ok"], false);
        assert_eq!(s["selfcheck"]["last"]["failing_step"], "embed");
        assert_eq!(s["selfcheck"]["last"]["error"], "down");

        let pass = SelfCheckOutcome {
            failing_step: None,
            error: None,
            rerank: Some(Rerank::Ran),
        };
        assert_eq!(v.record_selfcheck(3, 7, pass), 0);
        assert!(!v.selfcheck_failing());
        let last = &v.snapshot()["selfcheck"]["last"];
        assert_eq!(last["ok"], true);
        assert_eq!(last["failing_step"], Value::Null);
        assert_eq!(last["rerank"], "ran");
        assert_eq!(last["elapsed_ms"], 7);
    }
}
