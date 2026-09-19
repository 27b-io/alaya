//! MemoryService — orchestrates all 9 MCP tools across backends.
//!
//! This is the core business logic layer. Each public method corresponds to
//! one MCP tool. All backend calls go through trait abstractions.

use std::cell::RefCell;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use alaya_backends::judge::sanitize_reason;
use alaya_backends::{ContradictionJudge, Judgement, Survivor};
use alaya_types::graph::{ContradictionQuery, EdgeVerdict, Resolution, Verdict};

// ─── Tag deserialization ───────────────────────────────────────────────────────

/// Accept `["a","b"]`, `"a, b"`, or `null` — always yields `Option<Vec<String>>`.
fn deserialize_tags<'de, D>(deserializer: D) -> std::result::Result<Option<Vec<String>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use std::collections::HashSet;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrVec {
        Vec(Vec<String>),
        Str(String),
    }

    fn dedup(iter: impl Iterator<Item = String>) -> Option<Vec<String>> {
        let mut seen = HashSet::new();
        let v: Vec<String> = iter
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty() && seen.insert(s.clone()))
            .collect();
        if v.is_empty() { None } else { Some(v) }
    }

    let opt: Option<StringOrVec> = Option::deserialize(deserializer)?;
    Ok(match opt {
        None => None,
        Some(StringOrVec::Vec(v)) => dedup(v.into_iter()),
        Some(StringOrVec::Str(s)) => {
            let t = s.trim();
            // Handle stringified JSON arrays: "[\"a\",\"b\"]" → vec!["a","b"]
            if t.starts_with('[')
                && let Ok(v) = serde_json::from_str::<Vec<String>>(t)
            {
                return Ok(dedup(v.into_iter()));
            }
            dedup(t.split(',').map(String::from))
        }
    })
}
use tracing;

use alaya_backends::{
    ConsolidationService, EmbeddingProvider, GraphService, HebbianService, RerankingService,
    SummaryProvider, VectorStorage,
};
use alaya_types::{
    AlayaError, Result,
    graph::{CoAccessPair, Direction, EdgeMeta, SystemRelationType, UserRelationType},
    memory::{Memory, MetadataUpdate, PatchMemoryRequest, ScoredMemory},
    search::{PayloadFilter, PromptName, SearchMode},
};

use crate::{
    deduplication::{self, CanonicalStrategy},
    encoding_context,
    hashing::generate_content_hash,
    hybrid_search::{self, RRF_K},
    interference, provenance, salience, spaced_repetition,
};

// ─── Search types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchParams {
    #[serde(default)]
    pub query: String,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default = "default_page")]
    pub page: usize,
    #[serde(default = "default_page_size")]
    pub page_size: usize,
    #[serde(default, deserialize_with = "deserialize_tags")]
    pub tags: Option<Vec<String>>,
    #[serde(default)]
    pub match_all: bool,
    #[serde(default = "default_k")]
    pub k: usize,
    pub min_similarity: Option<f64>,
    pub memory_type: Option<String>,
    pub encoding_context: Option<HashMap<String, Value>>,
    #[serde(default)]
    pub include_superseded: bool,
    pub min_trust_score: Option<f64>,
    #[serde(default)]
    pub output: OutputMode,
    /// Cursor for recent mode: `created_at` timestamp of the last result
    /// from the previous page. Memories with `created_at < cursor` are returned.
    pub cursor: Option<f64>,
}

fn default_page() -> usize {
    1
}
fn default_page_size() -> usize {
    10
}
fn default_k() -> usize {
    10
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputMode {
    #[default]
    Full,
    Summary,
    Both,
}

/// Store memory input parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoreParams {
    pub content: String,
    #[serde(default, deserialize_with = "deserialize_tags")]
    pub tags: Option<Vec<String>>,
    pub memory_type: Option<String>,
    pub metadata: Option<HashMap<String, Value>>,
    pub client_hostname: Option<String>,
    pub summary: Option<String>,
    /// If set, skip storage when an existing memory has cosine similarity >= threshold.
    /// Enables Prajna's `store_if_novel()` pattern in a single call.
    #[serde(default)]
    pub dedup_threshold: Option<f64>,
}

/// Relation action parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationParams {
    pub action: String,
    pub content_hash: String,
    pub target_hash: Option<String>,
    pub relation_type: Option<String>,
}

// ─── MemoryService ──────────────────────────────────────────────────────────

/// TTL for the tag cache in seconds. Tags change infrequently, so 60s is
/// acceptable — new tags may not appear in hybrid keyword extraction for
/// up to one minute after being stored.
const TAG_CACHE_TTL: f64 = 60.0;

/// Maximum number of memories to scan in `find_duplicates`. Embedding large
/// batches blocks the LocalSet for tens of seconds. 200 memories = ~4 embedding
/// batches of 64, keeping the total under ~10-20s instead of ~40s at 500.
const MAX_DEDUP_SCAN: usize = 200;

// ─── Search ranking weights ────────────────────────────────────────────────
//
// Multiplicative boosts applied to each result after RRF fusion.
// Each weight controls the maximum influence of that signal:
//   score *= 1.0 + WEIGHT * signal_value
// where signal_value is in [0.0, 1.0].

/// Salience: emotional weight + access frequency + explicit importance.
const BOOST_SALIENCE: f64 = 0.15;

/// Spaced repetition: rewards memories accessed at healthy intervals.
const BOOST_SPACING: f64 = 0.10;

/// Encoding context: similarity between storage context and query context.
const BOOST_CONTEXT: f64 = 0.10;

/// Summary embedding: cosine similarity between query and distilled summary.
/// Higher than content-level signals because summaries are query-shaped.
const BOOST_SUMMARY: f64 = 0.15;

/// Recency decay lambda: exponential half-life ~70 days.
const RECENCY_DECAY_LAMBDA: f64 = 0.01;

/// Graph spreading activation: multi-hop associative relevance.
const BOOST_GRAPH_ACTIVATION: f64 = 0.10;

/// Hebbian co-access: memories frequently retrieved together.
const BOOST_HEBBIAN: f64 = 0.10;

/// Trust: provenance-based quality signal (mcp=0.9, api=0.8, cli=0.7, unknown=0.5).
const BOOST_TRUST: f64 = 0.15;

/// Score cap — keeps scores bounded after multiplicative boosts.
const SCORE_CAP: f64 = 1.5;

/// RRF blend weight: how much the fused rank signal contributes to the
/// final score vs raw cosine similarity.  0.0 = pure cosine (pre-fix
/// behavior), 1.0 = pure RRF rank.  GEPA-optimized on LongMemEval:
/// 0.4 ties with 0.9/1.0 at R@5=0.938; 0.4 chosen to preserve cosine
/// signal for production (where boosts amplify it).
const RRF_BLEND_WEIGHT: f64 = 0.4;

/// `tracing` target of the contradiction-judge shadow log (LAB-3283 AC-4b).
/// Exactly one INFO event per *judged* pair, always the same field set
/// (`memory_a`, `memory_b`, `verdict`, `survivor`, `confidence`, `model`,
/// `would_supersede`, `persisted`, `input_tokens`, `output_tokens`,
/// `reason`); unjudged outcomes warn on the default target and never appear
/// here. Phase 2 promotion is decided on these events — the field set is a
/// contract.
pub const SHADOW_LOG_TARGET: &str = "alaya::judge";

/// Upper bound on `resolved_via` (LAB-3885): a short principal tag, not a
/// free-text reason — the reason for a resolution is the verdict's.
const MAX_RESOLVED_VIA_LEN: usize = 128;

/// Verdict filter applied by `memory_contradictions` when the caller passes
/// none: genuine conflicts plus pairs the judge has not seen yet. Callers
/// that want `coexist`/`unrelated` name them explicitly.
const DEFAULT_VERDICT_FILTER: [&str; 3] = ["contradiction", "supersession", Verdict::UNJUDGED];

/// Result of judging one CONTRADICTS pair.
#[derive(Debug)]
pub enum JudgeOutcome {
    /// Verdict produced. `persisted` is whether the edge write landed; a
    /// graph blip (non-fatal by design) leaves the pair judged-but-unwritten,
    /// so the next backfill pass re-judges it.
    Judged {
        judgement: Judgement,
        persisted: bool,
    },
    /// No verdict. `marked` = a *deterministic* failure (schema / parse /
    /// empty answer / request fault) was persisted as `verdict = unjudged`
    /// with the error class as reason, so the backfill's NULL filter skips
    /// the pair instead of re-billing it forever. `marked = false` =
    /// transient (judge disabled, endpoint missing, fetch failed, upstream
    /// unavailable): nothing written, the pair is retried later.
    Unjudged { marked: bool },
    /// Upstream 429. Nothing was written; the caller owns any backoff.
    RateLimited { retry_after_secs: Option<u64> },
}

pub struct MemoryService {
    pub vectors: Box<dyn VectorStorage>,
    pub embeddings: Box<dyn EmbeddingProvider>,
    pub graph: Box<dyn GraphService>,
    pub hebbian: Box<dyn HebbianService>,
    pub consolidation: Box<dyn ConsolidationService>,
    /// Optional summary generator. When set, summaries are auto-generated
    /// fire-and-forget after store when the caller omits one.
    pub summary: Option<Box<dyn SummaryProvider>>,
    /// Optional contradiction judge (LAB-3283). When set, new CONTRADICTS
    /// signals are judged fire-and-forget after store and the operator
    /// backfill can annotate the existing queue. Advisory only in Phase 1.
    pub judge: Option<Box<dyn ContradictionJudge>>,
    /// Optional cross-encoder reranker. When set, hybrid search re-scores
    /// the top-N RRF candidates as (query, doc) pairs and reorders them.
    pub reranker: Option<Box<dyn RerankingService>>,
    /// Cached (timestamp, tags) from `get_all_tags()`. RefCell is fine:
    /// MemoryService runs single-threaded on a LocalSet (`!Send`).
    tag_cache: RefCell<Option<(f64, Vec<String>)>>,
    /// Clock function for timestamps. Defaults to wall clock; injectable for tests.
    clock: fn() -> f64,
}

impl MemoryService {
    pub fn new(
        vectors: Box<dyn VectorStorage>,
        embeddings: Box<dyn EmbeddingProvider>,
        graph: Box<dyn GraphService>,
        hebbian: Box<dyn HebbianService>,
        consolidation: Box<dyn ConsolidationService>,
        summary: Option<Box<dyn SummaryProvider>>,
    ) -> Self {
        Self {
            vectors,
            embeddings,
            graph,
            hebbian,
            consolidation,
            summary,
            judge: None,
            reranker: None,
            tag_cache: RefCell::new(None),
            clock: current_timestamp,
        }
    }

    /// Builder: attach a cross-encoder reranker. When set, `search_hybrid`
    /// re-scores the top-N RRF candidates and reorders them.
    pub fn with_reranker(mut self, reranker: Box<dyn RerankingService>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    /// Builder: attach a contradiction judge (LAB-3283).
    pub fn with_judge(mut self, judge: Box<dyn ContradictionJudge>) -> Self {
        self.judge = Some(judge);
        self
    }

    /// Create a `MemoryService` with a custom clock (for testing).
    #[cfg(test)]
    pub fn with_clock(
        vectors: Box<dyn VectorStorage>,
        embeddings: Box<dyn EmbeddingProvider>,
        graph: Box<dyn GraphService>,
        hebbian: Box<dyn HebbianService>,
        consolidation: Box<dyn ConsolidationService>,
        clock: fn() -> f64,
    ) -> Self {
        Self {
            vectors,
            embeddings,
            graph,
            hebbian,
            consolidation,
            summary: None,
            judge: None,
            reranker: None,
            tag_cache: RefCell::new(None),
            clock,
        }
    }

    // ─── Tool 1: store_memory ───────────────────────────────────────────

    #[tracing::instrument(skip(self, params), fields(content_len = params.content.len()))]
    /// Default-policy wrapper — full side effects. Use [`store_memory_with`] to
    /// request read-only behaviour (no shared-state writes) from authorized
    /// call sites; existing callers (tests, internal) keep the original
    /// semantics by going through this entry point.
    pub async fn store_memory(&self, params: StoreParams) -> Result<HashMap<String, Value>> {
        self.store_memory_with(params, false).await
    }

    /// Store a memory. When `read_only` is true, the side-effect writes that
    /// touch shared owner state — tag-index upsert and interference graph-edge
    /// creation — are skipped (vector + own node only). The caller's
    /// fire-and-forget summary patch must also be suppressed (see worker).
    pub async fn store_memory_with(
        &self,
        params: StoreParams,
        read_only: bool,
    ) -> Result<HashMap<String, Value>> {
        if params.content.is_empty() {
            return Err(AlayaError::Validation("content cannot be empty".into()));
        }

        let now = (self.clock)();
        let content_hash = generate_content_hash(&params.content);
        let tags = params.tags.unwrap_or_default();
        let memory_type = params.memory_type.unwrap_or_else(|| "note".into());

        // Build provenance
        let prov = provenance::build_provenance(
            params.client_hostname.as_deref().map(|_| "api"),
            Some("direct"),
            params.client_hostname.as_deref(),
            now,
        );

        // Compute salience (emotional = 0.0 in v1)
        let importance = params
            .metadata
            .as_ref()
            .and_then(|m| m.get("importance"))
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let salience_score = salience::compute_salience(0.0, 0, importance);

        // Capture encoding context
        let enc_ctx = encoding_context::capture_encoding_context(&tags, None, now);

        // Generate embedding
        let embedding = {
            let _span = tracing::info_span!("embed").entered();
            let embeddings = self
                .embeddings
                .embed_batch(&[params.content.as_str()], PromptName::Passage)
                .await?;
            embeddings.into_iter().next().ok_or_else(|| {
                AlayaError::Embedding("embedding service returned empty result".into())
            })?
        };

        // Dedup-on-write: skip storage if a near-duplicate exists
        if let Some(threshold) = params.dedup_threshold {
            // Over-fetch a few and skip superseded at the app layer (see
            // is_superseded): a superseded nearest-neighbor must neither
            // falsely reject new content as a duplicate of a dead memory nor
            // mask a live duplicate ranked just behind it.
            let similar = self
                .vectors
                .search_by_vector(&embedding, 5, None)
                .await
                .unwrap_or_default();

            if let Some(top) = similar.iter().find(|sm| !is_superseded(&sm.memory))
                && top.score >= threshold
            {
                let mut result = HashMap::new();
                result.insert("success".into(), serde_json::json!(true));
                result.insert("duplicate".into(), serde_json::json!(true));
                result.insert(
                    "existing_hash".into(),
                    serde_json::json!(top.memory.content_hash),
                );
                result.insert("similarity".into(), serde_json::json!(top.score));
                result.insert("content_hash".into(), serde_json::json!(content_hash));
                result.insert(
                    "message".into(),
                    serde_json::json!("Duplicate detected, storage skipped"),
                );
                return Ok(result);
            }
        }

        // Build memory struct
        let memory = Memory {
            content: params.content.clone(),
            content_hash: content_hash.clone(),
            tags: tags.clone(),
            memory_type,
            metadata: params.metadata,
            created_at: now,
            updated_at: now,
            embedding: Some(embedding.clone()),
            summary: params.summary,
            salience_score,
            access_count: 0,
            access_timestamps: Vec::new(),
            emotional_valence: None,
            encoding_context: Some(enc_ctx),
            provenance: Some(prov),
            summary_embedding: None,
        };

        // A read-only principal's store is additive only. Re-storing content
        // that already exists would replace the record's caller-owned fields
        // (tags, metadata, summary, memory_type) — the effect of patch /
        // supersede, which read-only is denied. `exists` judges presence
        // exactly as `store` does (raw point, not parseability), so guard and
        // write cannot disagree; it runs immediately before the write to keep
        // the window against concurrent writers minimal. Fails closed on a
        // lookup error.
        if read_only && self.vectors.exists(&content_hash).await? {
            return Err(AlayaError::Validation(format!(
                "memory {content_hash} already exists; read-only principals may only add new memories"
            )));
        }

        // Store in vector DB. `created == false` means a point with this
        // content_hash already existed: the backend carried its created_at,
        // access history and supersession marker over (see
        // VectorStorage::store), while the caller-supplied fields above — and
        // provenance — replaced the stored ones. Who owns provenance when two
        // callers store the same content is LAB-1084's decision; today's
        // replace semantics stand until then.
        let (created, _) = self.vectors.store(&memory).await?;
        if !created {
            tracing::info!(
                hash = %content_hash,
                "re-store of existing memory: payload replaced, access history preserved"
            );
        }

        // The memory itself was stored unconditionally above, so its tags are
        // now in Qdrant. Always invalidate the in-process tag cache so the
        // next keyword extraction reflects them — cache is local-process
        // ranking metadata, not shared owner state.
        if !tags.is_empty() {
            *self.tag_cache.borrow_mut() = None;

            // Tag-embedding collection upsert IS shared owner state — gated
            // by read_only so a browser-issued store can't seed the keyword
            // index with attacker-chosen embeddings.
            if !read_only {
                let tag_strs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                match self
                    .embeddings
                    .embed_batch(&tag_strs, PromptName::Passage)
                    .await
                {
                    Ok(tag_embeddings) if tag_embeddings.len() == tags.len() => {
                        let pairs: Vec<(&str, Vec<f32>)> = tag_strs
                            .iter()
                            .zip(tag_embeddings)
                            .map(|(t, e)| (*t, e))
                            .collect();
                        if let Err(e) = self.vectors.upsert_tags(&pairs).await {
                            tracing::warn!("tag index upsert failed (non-fatal): {e}");
                        }
                    }
                    Ok(_) => {
                        tracing::warn!("tag embedding batch size mismatch (non-fatal)");
                    }
                    Err(e) => {
                        tracing::warn!("tag embedding failed (non-fatal): {e}");
                    }
                }
            }
        }

        // Create graph node (non-fatal). Suppressed under read_only — even
        // an "own" node is a write to the shared owner graph and would let
        // downstream graph ops (consolidation, neighbor traversal) see a
        // browser-injected node.
        if !read_only && let Err(e) = self.graph.ensure_node(&content_hash, now).await {
            tracing::warn!("graph ensure_node failed (non-fatal): {e}");
        }

        // Interference detection: search for similar content + create graph edges.
        // Suppressed under read_only: edges write to the shared owner graph
        // (would be a side-channel to the gated `relation` tool).
        let mut contradiction_signals = Vec::new();
        if !read_only && let Ok(similar) = self.vectors.search_by_vector(&embedding, 10, None).await
        {
            let mut edges_to_create: Vec<(String, String, UserRelationType, EdgeMeta)> = Vec::new();

            for scored in &similar {
                if scored.memory.content_hash == content_hash {
                    continue;
                }
                // Never relate/contradict against a superseded memory (see
                // is_superseded — the Qdrant-side filter is a no-op).
                if is_superseded(&scored.memory) {
                    continue;
                }
                if scored.score < 0.7 {
                    continue;
                }

                let signals = interference::detect_contradiction_signals(
                    &params.content,
                    &scored.memory.content,
                    &scored.memory.content_hash,
                    scored.score,
                );

                for signal in &signals {
                    edges_to_create.push((
                        content_hash.clone(),
                        signal.existing_hash.clone(),
                        UserRelationType::Contradicts,
                        EdgeMeta {
                            created_at: Some(now),
                            confidence: Some(signal.confidence),
                        },
                    ));
                }

                contradiction_signals.extend(signals);
            }

            // Cross-reference detection (lower threshold)
            for scored in &similar {
                if scored.memory.content_hash == content_hash {
                    continue;
                }
                if scored.score < 0.4 || scored.score >= 0.7 {
                    continue; // Only create RELATES_TO for moderate similarity
                }

                edges_to_create.push((
                    content_hash.clone(),
                    scored.memory.content_hash.clone(),
                    UserRelationType::RelatesTo,
                    EdgeMeta {
                        created_at: Some(now),
                        confidence: None,
                    },
                ));
            }

            // Batch-create all interference edges in a single round-trip
            if !edges_to_create.is_empty()
                && let Err(e) = self.graph.create_typed_edges_batch(&edges_to_create).await
            {
                tracing::warn!("failed to batch-create interference edges: {e}");
            }
        }

        let mut result = HashMap::new();
        result.insert("success".into(), serde_json::json!(true));
        result.insert("content_hash".into(), serde_json::json!(content_hash));
        result.insert("memory_type".into(), serde_json::json!(memory.memory_type));
        result.insert("created".into(), serde_json::json!(created));
        result.insert(
            "message".into(),
            serde_json::json!(if created {
                "Memory stored successfully"
            } else {
                "Memory updated"
            }),
        );
        if !tags.is_empty() {
            result.insert("tags".into(), serde_json::json!(tags));
        }

        if !contradiction_signals.is_empty() {
            let interference_data: Vec<Value> = contradiction_signals
                .iter()
                .map(|s| {
                    serde_json::json!({
                        "existing_hash": s.existing_hash,
                        "signal_type": format!("{:?}", s.signal_type),
                        "confidence": s.confidence,
                        "detail": s.detail,
                    })
                })
                .collect();
            result.insert(
                "interference".into(),
                serde_json::json!({ "contradictions": interference_data }),
            );
        }

        Ok(result)
    }

    // ─── Background enrichment ──────────────────────────────────────────

    /// Generate and commit a summary (plus its embedding) for a memory stored
    /// without one. This runs detached from the request, so by the time the
    /// providers return a caller may have supplied a summary of their own.
    /// The commit is therefore conditional and decided by the backend under
    /// its write lock (`VectorStorage::set_generated_summary`): a caller's
    /// summary always wins. Returns whether the generated summary was
    /// applied. Failures are logged and swallowed; enrichment is best-effort
    /// by design.
    pub async fn enrich_summary(&self, hash: &str, content: &str) -> bool {
        // Guard before anything else: the hash may come from a stored payload
        // (backfill), and both the log prefix below and the backend's
        // point-id derivation byte-slice it. A malformed value must be a
        // logged skip, never a panic that kills the detached task.
        if !alaya_types::memory::validate_content_hash(hash) {
            tracing::warn!(
                hash_len = hash.len(),
                "enrich_summary: invalid content_hash, skipping"
            );
            return false;
        }
        let Some(ref summarizer) = self.summary else {
            return false;
        };
        let h = &hash[..8.min(hash.len())];
        let summary = match summarizer.summarize(content).await {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(hash = h, "summary generation failed (non-fatal): {e}");
                return false;
            }
        };
        // Embed the summary for the search boost (non-fatal if it fails).
        let embedding = match self
            .embeddings
            .embed_batch(&[summary.as_str()], PromptName::Passage)
            .await
        {
            Ok(mut embs) if !embs.is_empty() => Some(embs.remove(0)),
            Ok(_) => {
                tracing::warn!(hash = h, "summary embedding returned empty (non-fatal)");
                None
            }
            Err(e) => {
                tracing::warn!(hash = h, "summary embedding failed (non-fatal): {e}");
                None
            }
        };
        match self
            .vectors
            .set_generated_summary(hash, &summary, embedding)
            .await
        {
            Ok(true) => {
                tracing::debug!(hash = h, "auto-summary + embedding applied");
                true
            }
            Ok(false) => {
                tracing::debug!(
                    hash = h,
                    "auto-summary skipped: a caller summary landed first"
                );
                false
            }
            Err(e) => {
                tracing::warn!(hash = h, "summary commit failed (non-fatal): {e}");
                false
            }
        }
    }

    // ─── Tool 2: search ─────────────────────────────────────────────────

    /// Default-policy wrapper — runs Hybrid with full side effects.
    #[tracing::instrument(skip(self, params), fields(mode = ?params.mode))]
    pub async fn search(&self, params: SearchParams) -> Result<Value> {
        self.search_with(params, false).await
    }

    /// Search. When `read_only` is true, Hybrid skips its side-effect writes
    /// (access-count increment, Hebbian co-access enqueue) and returns the
    /// un-incremented `access_count`. Non-Hybrid modes are pure reads already.
    #[tracing::instrument(skip(self, params), fields(mode = ?params.mode, read_only))]
    pub async fn search_with(&self, params: SearchParams, read_only: bool) -> Result<Value> {
        if params.page_size == 0 {
            return Err(AlayaError::Validation("page_size must be > 0".into()));
        }
        let mode_str = format!("{:?}", params.mode).to_lowercase();
        let mut result = match params.mode {
            SearchMode::Hybrid => self.search_hybrid(&params, read_only).await?,
            SearchMode::Scan => self.search_scan(&params).await?,
            SearchMode::Similar => self.search_similar(&params).await?,
            SearchMode::Tag => self.search_tag(&params).await?,
            SearchMode::Recent => self.search_recent(&params).await?,
        };
        // Inject mode into every search response for caller context
        if let Some(obj) = result.as_object_mut() {
            obj.insert("mode".into(), serde_json::json!(mode_str));
        }
        Ok(result)
    }

    #[tracing::instrument(skip(self, params))]
    async fn search_hybrid(&self, params: &SearchParams, read_only: bool) -> Result<Value> {
        if params.query.trim().is_empty() {
            return Err(AlayaError::Validation(
                "query is required for hybrid mode".into(),
            ));
        }
        let now = (self.clock)();

        let fetch_size = std::cmp::min(
            std::cmp::max(params.page_size * 3, params.page * params.page_size),
            100,
        );

        // Stage 1: Fan-out — embed starts immediately (no tag dependency),
        // tags→keywords→tag_search chains as a concurrent branch.
        let (embed_result, mut tag_results, corpus_size, n_keywords) = {
            let _span = tracing::info_span!("fan_out").entered();
            let query_texts = [params.query.as_str()];
            let embed_fut = self.embeddings.embed_batch(&query_texts, PromptName::Query);

            let tag_search_fut = async {
                let _span = tracing::info_span!("get_all_tags").entered();
                let all_tags = {
                    let cached = self.tag_cache.borrow().clone();
                    if let Some((ts, tags)) = cached {
                        if now - ts < TAG_CACHE_TTL {
                            tags
                        } else {
                            let fresh = self.vectors.get_all_tags().await.unwrap_or_default();
                            *self.tag_cache.borrow_mut() = Some((now, fresh.clone()));
                            fresh
                        }
                    } else {
                        let fresh = self.vectors.get_all_tags().await.unwrap_or_default();
                        *self.tag_cache.borrow_mut() = Some((now, fresh.clone()));
                        fresh
                    }
                };
                drop(_span);

                let tag_set: std::collections::HashSet<String> = all_tags.into_iter().collect();
                let keywords = hybrid_search::extract_query_keywords(&params.query, Some(&tag_set));
                let n_keywords = keywords.len();

                let results = if keywords.is_empty() {
                    Vec::new()
                } else {
                    let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
                    self.vectors
                        .search_by_tags(&keyword_refs, false, fetch_size)
                        .await
                        .unwrap_or_default()
                };
                (results, n_keywords)
            };

            let count_fut = self.vectors.count();

            let (embed, (tags, n_kw), count) = futures::join!(embed_fut, tag_search_fut, count_fut);
            (embed, tags, count, n_kw)
        };

        let query_embedding = embed_result?
            .into_iter()
            .next()
            .ok_or_else(|| AlayaError::Embedding("empty embedding result".into()))?;
        let alpha = hybrid_search::get_adaptive_alpha(corpus_size.unwrap_or(0), n_keywords);

        let filter = PayloadFilter {
            memory_type: params.memory_type.clone(),
            min_trust_score: params.min_trust_score,
            ..Default::default()
        };

        // Stage 2: Vector search + semantic tag pipeline run concurrently.
        // search_similar_tags→search_by_tags chains inside one branch so the
        // 44-64ms semantic search overlaps with search_by_vector.
        let (mut vector_results, semantic_tag_results) = {
            let _span = tracing::info_span!("vector_search").entered();
            let vector_fut =
                self.vectors
                    .search_by_vector(&query_embedding, fetch_size, Some(filter));

            let semantic_pipeline_fut = async {
                let tags = self
                    .vectors
                    .search_similar_tags(&query_embedding, 10)
                    .await
                    .unwrap_or_default();
                if tags.is_empty() {
                    return Vec::new();
                }
                let refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
                self.vectors
                    .search_by_tags(&refs, false, fetch_size)
                    .await
                    .unwrap_or_default()
            };

            let (vector_result, semantic) = futures::join!(vector_fut, semantic_pipeline_fut);
            (vector_result?, semantic)
        };

        // Merge semantic tag matches into the keyword tag pool (deduplicated)
        if !semantic_tag_results.is_empty() {
            let existing: std::collections::HashSet<String> = tag_results
                .iter()
                .map(|s| s.memory.content_hash.clone())
                .collect();
            for sr in semantic_tag_results {
                if !existing.contains(&sr.memory.content_hash) {
                    tag_results.push(sr);
                }
            }
        }

        // Drop superseded memories from every candidate pool before fusion —
        // neither search_by_vector nor search_by_tags filters them (see
        // is_superseded), and superseded entries must not consume RRF ranks,
        // rerank slots, or spreading-activation seeds.
        if !params.include_superseded {
            vector_results.retain(|sm| !is_superseded(&sm.memory));
            tag_results.retain(|sm| !is_superseded(&sm.memory));
        }

        // Stage 3: Fuse (RRF) — pure computation
        let mut fused = {
            let _span = tracing::info_span!(
                "rrf_fuse",
                vectors = vector_results.len(),
                tags = tag_results.len()
            )
            .entered();
            let v_tuples: Vec<(String, f64)> = vector_results
                .iter()
                .map(|s| (s.memory.content_hash.clone(), s.score))
                .collect();
            let t_tuples: Vec<(String, f64)> = tag_results
                .iter()
                .map(|s| (s.memory.content_hash.clone(), s.score))
                .collect();

            hybrid_search::combine_results_rrf(&v_tuples, &t_tuples, alpha, RRF_K)
        };

        // Build hash→memory lookup
        let mut memory_map: HashMap<String, &ScoredMemory> = HashMap::new();
        for sm in &vector_results {
            memory_map.insert(sm.memory.content_hash.clone(), sm);
        }
        for sm in &tag_results {
            memory_map
                .entry(sm.memory.content_hash.clone())
                .or_insert(sm);
        }

        // Stage 4: Boost — graph queries run concurrently
        let (spreading, hebbian_boosts) = {
            let _span = tracing::info_span!("graph_boost").entered();
            let result_hashes: Vec<&str> =
                fused.iter().take(20).map(|(h, _, _)| h.as_str()).collect();

            let spreading_fut = self.graph.spreading_activation(
                &result_hashes[..std::cmp::min(5, result_hashes.len())],
                2,
                0.5,
                0.05,
                50,
            );
            let hebbian_fut = self.graph.hebbian_boosts_within(&result_hashes);

            let (s, h) = futures::join!(spreading_fut, hebbian_fut);
            (s.unwrap_or_default(), h.unwrap_or_default())
        };

        // Stage 4b: Graph injection — activated neighbors that are NOT already
        // in the fused results get fetched from Qdrant and spliced into the
        // candidate pool. This lets Hebbian-connected memories surface even
        // when their cosine similarity to the query is low.
        let injected_neighbors: Vec<ScoredMemory> = if !spreading.is_empty() {
            let _span = tracing::info_span!("graph_inject").entered();
            let fused_hashes: std::collections::HashSet<&str> =
                fused.iter().map(|(h, _, _)| h.as_str()).collect();

            // Top-10 activated neighbors not already in results
            let mut inject_candidates: Vec<(&str, f64)> = spreading
                .iter()
                .filter(|(h, _)| !fused_hashes.contains(h.as_str()))
                .map(|(h, s)| (h.as_str(), *s))
                .collect();
            inject_candidates
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            inject_candidates.truncate(10);

            if inject_candidates.is_empty() {
                Vec::new()
            } else {
                let neighbor_hashes: Vec<&str> =
                    inject_candidates.iter().map(|(h, _)| *h).collect();
                let activation_map: HashMap<&str, f64> = inject_candidates.into_iter().collect();

                // Use the minimum existing display score as a floor so
                // injected memories don't get filtered by min_similarity.
                let min_existing = fused
                    .iter()
                    .map(|(_, _, s)| *s)
                    .fold(f64::INFINITY, f64::min)
                    .max(0.1);

                match self.vectors.get_batch(&neighbor_hashes).await {
                    Ok(memories) => {
                        let mut result = Vec::with_capacity(memories.len());
                        for mem in memories {
                            if !params.include_superseded && is_superseded(&mem) {
                                continue;
                            }
                            let activation = activation_map
                                .get(mem.content_hash.as_str())
                                .copied()
                                .unwrap_or(0.0);
                            let display_score = min_existing.max(activation);
                            result.push(ScoredMemory {
                                memory: mem,
                                score: display_score,
                            });
                        }
                        tracing::debug!(
                            injected = result.len(),
                            "graph injection: added neighbors from spreading activation"
                        );
                        result
                    }
                    Err(e) => {
                        tracing::warn!("graph injection fetch failed (non-fatal): {e}");
                        Vec::new()
                    }
                }
            }
        } else {
            Vec::new()
        };

        // Extend memory_map and fused with injected neighbors
        for sm in &injected_neighbors {
            memory_map
                .entry(sm.memory.content_hash.clone())
                .or_insert(sm);
            fused.push((
                sm.memory.content_hash.clone(),
                0.0, // no RRF rank
                sm.score,
            ));
        }

        // Stage 4c: Cross-encoder rerank — re-score top-N candidates as
        // (query, doc) pairs and reorder. When successful, the rerank score
        // replaces the RRF+cosine blend in the scoring loop for those entries.
        // Validated on LongMemEval (2026-05-23): R@5 0.936 → 0.990 with
        // BAAI/bge-reranker-v2-m3 and top_n=20.
        let rerank_score_map: HashMap<String, f64> = 'rerank: {
            let Some(reranker) = self.reranker.as_ref() else {
                break 'rerank HashMap::new();
            };
            let _span = tracing::info_span!("rerank", top_n = reranker.top_n()).entered();
            let top_n = reranker.top_n().min(fused.len());
            if top_n == 0 {
                break 'rerank HashMap::new();
            }

            let candidate_contents: Vec<&str> = fused
                .iter()
                .take(top_n)
                .map(|(hash, _, _)| {
                    memory_map
                        .get(hash)
                        .map(|sm| sm.memory.content.as_str())
                        .unwrap_or("")
                })
                .collect();

            let budget = reranker.timeout();
            let started = std::time::Instant::now();
            let outcome =
                rerank_with_budget(budget, reranker.rerank(&params.query, &candidate_contents))
                    .await;
            let elapsed = started.elapsed();
            let scores = match outcome {
                Some(Ok(s)) if s.len() == top_n => s,
                Some(Ok(s)) => {
                    tracing::warn!(
                        got = s.len(),
                        expected = top_n,
                        "rerank score count mismatch; skipping rerank"
                    );
                    break 'rerank HashMap::new();
                }
                Some(Err(e)) => {
                    tracing::warn!(
                        error = %e,
                        budget_ms = budget.as_millis() as u64,
                        elapsed_ms = elapsed.as_millis() as u64,
                        "rerank failed (non-fatal); using RRF order"
                    );
                    break 'rerank HashMap::new();
                }
                None => {
                    tracing::warn!(
                        budget_ms = budget.as_millis() as u64,
                        elapsed_ms = elapsed.as_millis() as u64,
                        "rerank timed out (non-fatal); using RRF order"
                    );
                    break 'rerank HashMap::new();
                }
            };

            // Reorder the top-N slice of fused by rerank score desc, then
            // splice it back to the front of `fused`.
            let mut top_with_scores: Vec<((String, f64, f64), f64)> = fused
                .drain(..top_n)
                .zip(scores.iter().copied())
                .map(|(entry, s)| (entry, s as f64))
                .collect();
            top_with_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let map: HashMap<String, f64> = top_with_scores
                .iter()
                .map(|(entry, s)| (entry.0.clone(), *s))
                .collect();
            let reordered: Vec<(String, f64, f64)> =
                top_with_scores.into_iter().map(|(e, _)| e).collect();
            let mut new_fused = reordered;
            new_fused.append(&mut fused);
            fused = new_fused;
            tracing::debug!(
                reranked = map.len(),
                "cross-encoder rerank reordered top-N candidates"
            );
            map
        };

        // Normalize RRF scores to [0, 1] for blending with display_score (cosine).
        // Max RRF score is 1/(k+1) for rank 1; scale so rank-1 maps to ~1.0.
        let max_rrf = fused
            .iter()
            .map(|(_, rrf, _)| *rrf)
            .fold(0.0_f64, f64::max)
            .max(1e-9);

        let mut scored_results: Vec<(String, f64)> = fused
            .iter()
            .map(|(hash, rrf_combined, display_score)| {
                // When the entry was reranked, the cross-encoder score replaces
                // the RRF+cosine blend entirely — it's a much stronger relevance
                // signal. Otherwise fall back to blended RRF + cosine.
                let mut score = if let Some(&rerank) = rerank_score_map.get(hash) {
                    rerank
                } else {
                    let rrf_norm = rrf_combined / max_rrf;
                    RRF_BLEND_WEIGHT * rrf_norm + (1.0 - RRF_BLEND_WEIGHT) * display_score
                };

                // Salience boost — recompute from live access_count (stored
                // salience_score was baked at write time with access_count=0)
                if let Some(sm) = memory_map.get(hash) {
                    let importance = sm
                        .memory
                        .metadata
                        .as_ref()
                        .and_then(|m| m.get("importance"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let emotional = sm
                        .memory
                        .emotional_valence
                        .as_ref()
                        .and_then(|ev| ev.get("sentiment"))
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    let live_salience =
                        salience::compute_salience(emotional, sm.memory.access_count, importance);
                    score = salience::apply_salience_boost(score, live_salience, BOOST_SALIENCE);

                    // Trust boost — provenance-based quality signal
                    let trust = sm
                        .memory
                        .provenance
                        .as_ref()
                        .map(provenance::resolve_trust_score)
                        .unwrap_or(provenance::DEFAULT_TRUST_SCORE);
                    score *= 1.0 + BOOST_TRUST * trust;

                    // Spacing boost
                    let sq =
                        spaced_repetition::compute_spacing_quality(&sm.memory.access_timestamps);
                    score = spaced_repetition::apply_spacing_boost(score, sq, BOOST_SPACING);

                    // Encoding context boost
                    if let (Some(stored_ctx), Some(query_ctx)) =
                        (&sm.memory.encoding_context, &params.encoding_context)
                    {
                        let ctx_sim =
                            encoding_context::compute_context_similarity(stored_ctx, query_ctx);
                        score =
                            encoding_context::apply_context_boost(score, ctx_sim, BOOST_CONTEXT);
                    }

                    // Summary embedding boost — rewards memories whose distilled
                    // meaning aligns with the query, independent of content noise.
                    if let Some(ref summary_emb) = sm.memory.summary_embedding {
                        let sim = hybrid_search::cosine_similarity(&query_embedding, summary_emb);
                        if sim > 0.0 {
                            score *= 1.0 + BOOST_SUMMARY * sim as f64;
                        }
                    }

                    // Recency decay
                    score = hybrid_search::apply_recency_decay(
                        score,
                        sm.memory.created_at,
                        now,
                        RECENCY_DECAY_LAMBDA,
                    );
                }

                // Graph boosts — spreading activation now correctly applies to
                // both injected neighbors AND fused results at positions 6+
                // (which may be HEBBIAN neighbors of the top-5 seeds).
                if let Some(&activation) = spreading.get(hash) {
                    score *= 1.0 + BOOST_GRAPH_ACTIVATION * activation;
                }
                if let Some(&boost) = hebbian_boosts.get(hash) {
                    score *= 1.0 + BOOST_HEBBIAN * boost;
                }

                (hash.clone(), score.min(SCORE_CAP))
            })
            .collect();

        // Reranked entries always dominate the tail. Cross-encoder scores live
        // in roughly [0, 1] and can be smaller than blended (RRF+cosine) scores
        // for non-reranked tail entries, so a plain score-desc sort would let
        // tail entries leapfrog reranked top-N. Compound key (is_reranked, score)
        // guarantees the rerank order is preserved as the prefix of results.
        scored_results.sort_by(|a, b| {
            let a_rer = rerank_score_map.contains_key(&a.0);
            let b_rer = rerank_score_map.contains_key(&b.0);
            b_rer
                .cmp(&a_rer)
                .then_with(|| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Stage 5: Filter
        if let Some(min_sim) = params.min_similarity {
            scored_results.retain(|(_, s)| *s >= min_sim);
        }

        // Pagination
        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let total = scored_results.len();
        let page_results: Vec<(String, f64)> = scored_results
            .into_iter()
            .skip(offset)
            .take(params.page_size)
            .collect();

        // Stage 6: Enrich — side-effect writes run concurrently.
        // Suppressed under read_only: access_count + Hebbian touch shared
        // owner ranking state. Skipping leaves rankings unchanged.
        let page_hashes: Vec<&str> = page_results.iter().map(|(h, _)| h.as_str()).collect();

        if !read_only {
            let _span = tracing::info_span!("enrich", results = page_hashes.len()).entered();
            let access_fut = self.vectors.increment_access_count_batch(&page_hashes);

            let hebbian_enqueue_fut = async {
                if page_hashes.len() >= 2 {
                    let pairs: Vec<CoAccessPair> = page_hashes
                        .windows(2)
                        .map(|w| CoAccessPair {
                            src: w[0].to_string(),
                            dst: w[1].to_string(),
                            spacing_quality: 0.5,
                            timestamp: now,
                        })
                        .collect();
                    let _ = self.hebbian.enqueue_strengthen(&pairs).await;
                }
            };

            let _ = futures::join!(access_fut, hebbian_enqueue_fut);
        }

        // Stage 7: Format response
        let total_pages = total.div_ceil(params.page_size);
        let has_more = params.page < total_pages;

        let results: Vec<Value> = page_results
            .iter()
            .filter_map(|(hash, score)| {
                let sm = memory_map.get(hash)?;
                let mut item = format_memory_result(&sm.memory, *score, params.output);
                // Reflect the post-increment access_count (batch already wrote N+1)
                // — except under read_only, where the batch was skipped and the
                // stored value is what's still on disk.
                if let Some(obj) = item.as_object_mut() {
                    let reported = if read_only {
                        sm.memory.access_count
                    } else {
                        sm.memory.access_count.saturating_add(1)
                    };
                    obj.insert("access_count".into(), serde_json::json!(reported));
                }
                Some(item)
            })
            .collect();

        Ok(serde_json::json!({
            "page": params.page,
            "total": total,
            "page_size": params.page_size,
            "has_more": has_more,
            "total_pages": total_pages,
            "results": results,
        }))
    }

    #[tracing::instrument(skip(self, params))]
    async fn search_scan(&self, params: &SearchParams) -> Result<Value> {
        const MAX_RAW_SCANNED: usize = 5000;

        let target = (params.page.saturating_sub(1)) * params.page_size + params.page_size + 1;
        let scroll_page: usize = 100;
        let mut filtered: Vec<Memory> = Vec::new();
        let mut cursor: Option<String> = None;
        let mut raw_scanned: usize = 0;

        // Scroll until we have enough filtered results, hit EOF, or safety cap
        loop {
            let batch_size = scroll_page.min(target.saturating_sub(filtered.len()));
            let scroll = self.vectors.get_all(batch_size, cursor.as_deref()).await?;
            let batch_empty = scroll.memories.is_empty();
            raw_scanned += scroll.memories.len();

            for m in scroll.memories {
                if !params.include_superseded && is_superseded(&m) {
                    continue;
                }
                if params
                    .memory_type
                    .as_ref()
                    .is_some_and(|mt| m.memory_type != *mt)
                {
                    continue;
                }
                filtered.push(m);
            }

            cursor = scroll.next_offset;
            if batch_empty || cursor.is_none() || filtered.len() >= target {
                break;
            }
            if raw_scanned >= MAX_RAW_SCANNED {
                tracing::warn!(
                    raw_scanned,
                    filtered = filtered.len(),
                    target,
                    "search_scan hit safety cap"
                );
                break;
            }
        }

        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let page: Vec<Value> = filtered
            .iter()
            .skip(offset)
            .take(params.page_size)
            .map(|m| format_memory_result(m, 1.0, params.output))
            .collect();

        let has_more = filtered.len() > offset + params.page_size || cursor.is_some();

        Ok(serde_json::json!({
            "page": params.page,
            "page_size": params.page_size,
            "has_more": has_more,
            "count": page.len(),
            "results": page,
        }))
    }

    #[tracing::instrument(skip(self, params))]
    async fn search_similar(&self, params: &SearchParams) -> Result<Value> {
        if params.query.trim().is_empty() {
            return Err(AlayaError::Validation(
                "query is required for similar mode".into(),
            ));
        }
        let query_embedding = self
            .embeddings
            .embed_batch(&[params.query.as_str()], PromptName::Query)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| AlayaError::Embedding("empty embedding result".into()))?;

        const MAX_SIMILAR_FETCH: usize = 5000;

        // memory_type (exact match) and min_trust_score (range) are reliable
        // Qdrant-side filters; only superseded filtering must stay app-side.
        let filter = PayloadFilter {
            memory_type: params.memory_type.clone(),
            min_trust_score: params.min_trust_score,
            ..Default::default()
        };

        let initial_fetch = if params.include_superseded {
            params.k
        } else {
            params
                .k
                .saturating_mul(2)
                .min(MAX_SIMILAR_FETCH)
                .max(params.k)
        };
        let (mut results, _) = fetch_live_with_retry(
            initial_fetch,
            params.k,
            MAX_SIMILAR_FETCH,
            params.include_superseded,
            |n| {
                self.vectors
                    .search_by_vector(&query_embedding, n, Some(filter.clone()))
            },
        )
        .await?;
        results.truncate(params.k);

        let items: Vec<Value> = results
            .iter()
            .map(|sm| format_memory_result(&sm.memory, sm.score, params.output))
            .collect();

        Ok(serde_json::json!({
            "results": items,
            "total": results.len(),
        }))
    }

    #[tracing::instrument(skip(self, params))]
    async fn search_tag(&self, params: &SearchParams) -> Result<Value> {
        const MAX_TAG_FETCH: usize = 5000;

        let tags = params.tags.as_deref().unwrap_or_default();
        if tags.is_empty() {
            return Err(AlayaError::Validation("tags required for tag mode".into()));
        }

        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let target = offset + params.page_size + 1; // +1 to detect has_more
        let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();

        let (filtered, exhausted) = fetch_live_with_retry(
            target * 2,
            target,
            MAX_TAG_FETCH,
            params.include_superseded,
            |n| self.vectors.search_by_tags(&tag_refs, params.match_all, n),
        )
        .await?;

        let has_more =
            filtered.len() > offset + params.page_size || (!exhausted && filtered.len() < target);
        let page: Vec<Value> = filtered
            .iter()
            .skip(offset)
            .take(params.page_size)
            .map(|sm| format_memory_result(&sm.memory, sm.score, params.output))
            .collect();

        Ok(serde_json::json!({
            "page": params.page,
            "page_size": params.page_size,
            "count": page.len(),
            "has_more": has_more,
            "results": page,
        }))
    }

    #[tracing::instrument(skip(self, params))]
    async fn search_recent(&self, params: &SearchParams) -> Result<Value> {
        const MAX_RECENT_SCANNED: usize = 5000;

        let target = params.page_size + 1; // +1 detects has_more
        // Over-fetch and filter superseded at the application layer, advancing
        // the created_at cursor until the page fills (same strategy as scan/tag;
        // see is_superseded for why Qdrant can't filter this server-side).
        let batch_size = if params.include_superseded {
            target
        } else {
            target.saturating_mul(2).min(MAX_RECENT_SCANNED)
        };

        let mut results: Vec<Memory> = Vec::new();
        let mut scan_cursor = params.cursor;
        let mut raw_scanned: usize = 0;
        loop {
            let batch = self
                .vectors
                .get_recent(batch_size, scan_cursor, params.memory_type.as_deref())
                .await?;
            let exhausted = batch.len() < batch_size;
            raw_scanned += batch.len();

            for m in batch {
                scan_cursor = Some(m.created_at);
                if !params.include_superseded && is_superseded(&m) {
                    continue;
                }
                results.push(m);
            }

            if results.len() >= target || exhausted {
                break;
            }
            if raw_scanned >= MAX_RECENT_SCANNED {
                tracing::warn!(
                    raw_scanned,
                    filtered = results.len(),
                    target,
                    "search_recent hit safety cap"
                );
                break;
            }
        }

        let has_more = results.len() > params.page_size;
        let page_results: Vec<&Memory> = results.iter().take(params.page_size).collect();

        // Cursor for next page: created_at of the last result on this page
        let next_cursor = page_results.last().map(|m| m.created_at);

        let items: Vec<Value> = page_results
            .iter()
            .map(|m| format_memory_result(m, 1.0, params.output))
            .collect();

        let mut resp = serde_json::json!({
            "page_size": params.page_size,
            "has_more": has_more,
            "results": items,
        });
        if let Some(c) = next_cursor {
            resp["next_cursor"] = serde_json::json!(c);
        }
        Ok(resp)
    }

    // ─── Tool 3: delete_memory ──────────────────────────────────────────

    #[tracing::instrument(skip(self))]
    pub async fn delete_memory(&self, content_hash: &str) -> Result<HashMap<String, Value>> {
        if !alaya_types::memory::validate_content_hash(content_hash) {
            return Err(AlayaError::Validation("invalid content_hash format".into()));
        }

        let deleted = self.vectors.delete(content_hash).await?;

        // Delete graph node (non-fatal)
        if let Err(e) = self.graph.delete_node(content_hash).await {
            tracing::warn!("graph delete_node failed (non-fatal): {e}");
        }

        let mut result = HashMap::new();
        result.insert("success".into(), serde_json::json!(deleted));
        result.insert(
            "message".into(),
            serde_json::json!(if deleted {
                "Memory deleted"
            } else {
                "Memory not found"
            }),
        );
        Ok(result)
    }

    // ─── patch_memory (REST only, not an MCP tool) ─────────────────────

    /// Patch mutable fields on an existing memory.
    ///
    /// Delegates to VectorStorage and invalidates tag cache when tags change.
    /// Returns the full updated Memory on success.
    #[tracing::instrument(skip(self, patch))]
    pub async fn patch_memory(
        &self,
        content_hash: &str,
        patch: &PatchMemoryRequest,
    ) -> Result<Memory> {
        if !alaya_types::memory::validate_content_hash(content_hash) {
            return Err(AlayaError::Validation("invalid content_hash format".into()));
        }

        let mem = self.vectors.patch_memory(content_hash, patch).await?;

        // Invalidate tag cache so hybrid search picks up new tags immediately
        if patch.tags.is_some() {
            *self.tag_cache.borrow_mut() = None;
        }

        Ok(mem)
    }

    // ─── Tool 4: check_database_health ──────────────────────────────────

    #[tracing::instrument(skip(self))]
    pub async fn check_database_health(&self) -> Result<HashMap<String, Value>> {
        let vector_health = self.vectors.health().await?;

        let graph_health = match self.graph.get_stats().await {
            Ok(stats) => serde_json::json!({
                "status": "healthy",
                "node_count": stats.node_count,
                "edge_count": stats.edge_count,
            }),
            Err(e) => serde_json::json!({
                "status": "unhealthy",
                "error": e.safe_message(),
            }),
        };

        let mut result = HashMap::new();
        result.insert(
            "status".into(),
            serde_json::json!(
                if vector_health.status == "green" || vector_health.status == "ok" {
                    "healthy"
                } else {
                    "degraded"
                }
            ),
        );
        result.insert("backend".into(), serde_json::json!("qdrant"));
        result.insert(
            "vector_health".into(),
            serde_json::to_value(&vector_health).unwrap_or_default(),
        );
        result.insert("graph_health".into(), graph_health);
        result.insert(
            "total_memories".into(),
            serde_json::json!(self.vectors.count().await.unwrap_or(0)),
        );
        Ok(result)
    }

    // ─── Tool 5: relation ───────────────────────────────────────────────

    #[tracing::instrument(skip(self, params), fields(action = %params.action))]
    pub async fn relation(&self, params: RelationParams) -> Result<Value> {
        if !alaya_types::memory::validate_content_hash(&params.content_hash) {
            return Err(AlayaError::Validation("invalid content_hash".into()));
        }

        match params.action.as_str() {
            "create" => {
                let target = params
                    .target_hash
                    .as_deref()
                    .ok_or_else(|| AlayaError::Validation("target_hash required".into()))?;
                let rel_str = params
                    .relation_type
                    .as_deref()
                    .ok_or_else(|| AlayaError::Validation("relation_type required".into()))?;
                let rel = parse_user_relation(rel_str)?;

                let now = (self.clock)();
                let created = self
                    .graph
                    .create_typed_edge(
                        &params.content_hash,
                        target,
                        rel,
                        EdgeMeta {
                            created_at: Some(now),
                            confidence: None,
                        },
                    )
                    .await?;

                Ok(serde_json::json!({
                    "success": true,
                    "source": params.content_hash,
                    "target": target,
                    "relation_type": rel_str,
                    "created": created,
                }))
            }
            "get" => {
                let rel = params
                    .relation_type
                    .as_deref()
                    .map(parse_user_relation)
                    .transpose()?;

                let edges = self
                    .graph
                    .get_typed_edges(&params.content_hash, rel, Direction::Both, 50)
                    .await?;

                Ok(serde_json::json!({
                    "relations": edges,
                    "content_hash": params.content_hash,
                    "count": edges.len(),
                }))
            }
            "delete" => {
                let target = params
                    .target_hash
                    .as_deref()
                    .ok_or_else(|| AlayaError::Validation("target_hash required".into()))?;
                let rel_str = params
                    .relation_type
                    .as_deref()
                    .ok_or_else(|| AlayaError::Validation("relation_type required".into()))?;
                let rel = parse_user_relation(rel_str)?;

                let deleted = self
                    .graph
                    .delete_typed_edge(&params.content_hash, target, rel)
                    .await?;

                Ok(serde_json::json!({
                    "success": true,
                    "source": params.content_hash,
                    "target": target,
                    "relation_type": rel_str,
                    "deleted": deleted,
                }))
            }
            _ => Err(AlayaError::Validation(format!(
                "unknown action: {}",
                params.action
            ))),
        }
    }

    // ─── Tool 6: memory_supersede ───────────────────────────────────────

    #[tracing::instrument(skip(self))]
    pub async fn memory_supersede(
        &self,
        old_hash: &str,
        new_hash: &str,
        reason: &str,
    ) -> Result<Value> {
        if old_hash == new_hash {
            return Err(AlayaError::Validation(
                "old_hash and new_hash must differ".into(),
            ));
        }

        // Verify both exist (single batch GET)
        let batch = self.vectors.get_batch(&[old_hash, new_hash]).await?;
        if !batch.iter().any(|m| m.content_hash == old_hash) {
            return Err(AlayaError::Validation(format!(
                "old memory not found: {old_hash}"
            )));
        }
        if !batch.iter().any(|m| m.content_hash == new_hash) {
            return Err(AlayaError::Validation(format!(
                "new memory not found: {new_hash}"
            )));
        }

        self.mark_superseded(&[old_hash], new_hash, reason).await?;

        Ok(serde_json::json!({
            "success": true,
            "superseded": old_hash,
            "superseded_by": new_hash,
            "reason": reason,
        }))
    }

    /// Mark every memory in `old_hashes` as superseded by `new_hash`.
    ///
    /// One batched metadata update writes the audit trail (`superseded_by` +
    /// `supersession_reason`) for all memories, then one batched edge write
    /// creates the SUPERSEDES edges. Edge failures only warn — graph
    /// operations are non-fatal by design.
    async fn mark_superseded(
        &self,
        old_hashes: &[&str],
        new_hash: &str,
        reason: &str,
    ) -> Result<()> {
        let mut extra = HashMap::new();
        extra.insert("supersession_reason".into(), serde_json::json!(reason));
        self.vectors
            .update_metadata_batch(
                old_hashes,
                MetadataUpdate {
                    superseded_by: Some(new_hash.to_string()),
                    extra: Some(extra),
                    ..Default::default()
                },
            )
            .await?;

        let now = (self.clock)();
        let edges: Vec<(String, String, SystemRelationType, f64)> = old_hashes
            .iter()
            .map(|old| {
                (
                    new_hash.to_string(),
                    (*old).to_string(),
                    SystemRelationType::Supersedes,
                    now,
                )
            })
            .collect();
        if let Err(e) = self.graph.create_system_edges_batch(&edges).await {
            tracing::warn!("failed to create SUPERSEDES edge(s): {e}");
        }

        Ok(())
    }

    // ─── Contradiction judge (LAB-3283, Phase 1: advisory) ──────────────

    /// Judge one `src -> dst` CONTRADICTS pair, off the request path.
    ///
    /// Phase 1 is advisory: the verdict is written onto the edge (through
    /// the `GraphService` trait, so scoping applies uniformly) and one
    /// structured shadow-log event records the supersession the engine
    /// *would* apply. No memory payload is touched and no edge is created
    /// or deleted. Every failure is a logged `Unjudged`, never a panic —
    /// this runs detached from any request.
    pub async fn judge_contradiction(&self, src: &str, dst: &str) -> JudgeOutcome {
        let Some(ref judge) = self.judge else {
            return JudgeOutcome::Unjudged { marked: false };
        };
        if !alaya_types::memory::validate_content_hash(src)
            || !alaya_types::memory::validate_content_hash(dst)
            || src == dst
        {
            tracing::warn!(
                src_len = src.len(),
                dst_len = dst.len(),
                "judge_contradiction: invalid pair, skipping"
            );
            return JudgeOutcome::Unjudged { marked: false };
        }
        let (sa, sd) = (&src[..8], &dst[..8]);

        let batch = match self.vectors.get_batch(&[src, dst]).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(a = sa, b = sd, "judge_contradiction: fetch failed: {e}");
                return JudgeOutcome::Unjudged { marked: false };
            }
        };
        let a = batch.iter().find(|m| m.content_hash == src);
        let b = batch.iter().find(|m| m.content_hash == dst);
        let (Some(a), Some(b)) = (a, b) else {
            // Deterministic until the memory reappears: mark it, or the
            // backfill's NULL selection re-spends a slot on it every pass.
            tracing::warn!(
                a = sa,
                b = sd,
                "judge_contradiction: endpoint missing; marking edge"
            );
            let marked = self
                .mark_unjudged(
                    src,
                    dst,
                    judge.model_name(),
                    "endpoint missing from vector store",
                )
                .await;
            return JudgeOutcome::Unjudged { marked };
        };

        let j = match judge.judge(a, b).await {
            Ok(j) => j,
            Err(AlayaError::RateLimited { retry_after_secs }) => {
                tracing::warn!(a = sa, b = sd, retry_after_secs, "judge rate limited");
                return JudgeOutcome::RateLimited { retry_after_secs };
            }
            Err(AlayaError::Unavailable(e)) => {
                tracing::warn!(
                    memory_a = src,
                    memory_b = dst,
                    verdict = Verdict::UNJUDGED,
                    error = ?e,
                    "contradiction unjudged (transient; will retry)"
                );
                return JudgeOutcome::Unjudged { marked: false };
            }
            Err(e) => {
                // Deterministic: this pair fails the same way every time, so
                // mark the edge `unjudged` with the error class — otherwise
                // the backfill's NULL filter re-matches it on every pass
                // (poison pill). Default log target on purpose: the shadow
                // log carries judged pairs only. `?e` keeps a body escaped.
                tracing::warn!(
                    memory_a = src,
                    memory_b = dst,
                    verdict = Verdict::UNJUDGED,
                    error = ?e,
                    "contradiction unjudged (deterministic; marking edge)"
                );
                let marked = self
                    .mark_unjudged(src, dst, judge.model_name(), &e.to_string())
                    .await;
                return JudgeOutcome::Unjudged { marked };
            }
        };

        let survivor = j.survivor.map(|s| match s {
            Survivor::A => src.to_string(),
            Survivor::B => dst.to_string(),
        });
        let edge = EdgeVerdict {
            verdict: j.verdict,
            verdict_survivor: survivor.clone(),
            verdict_reason: j.reason.clone(),
            verdict_confidence: j.confidence,
            verdict_model: j.model.clone(),
            judged_at: (self.clock)(),
        };
        let persisted = self.persist_verdict(src, dst, &edge).await;

        // Shadow log (AC-4b): what Phase 2 would write. Only a supersession
        // with a named survivor is actionable; everything else is "none".
        let would_supersede = match (j.verdict, survivor.as_deref()) {
            (Verdict::Supersession, Some(s)) => {
                let loser = if s == src { dst } else { src };
                format!("{loser} -> {s}")
            }
            _ => "none".to_string(),
        };
        tracing::info!(
            target: SHADOW_LOG_TARGET,
            memory_a = src,
            memory_b = dst,
            verdict = j.verdict.as_str(),
            survivor = survivor.as_deref().unwrap_or("none"),
            confidence = j.confidence,
            model = %j.model,
            would_supersede = %would_supersede,
            persisted,
            input_tokens = j.input_tokens,
            output_tokens = j.output_tokens,
            // Debug-quoted: model-authored text in a line-oriented log.
            reason = ?j.reason,
            "contradiction judged"
        );
        JudgeOutcome::Judged {
            judgement: j,
            persisted,
        }
    }

    /// Persist a deterministic-failure marker: `verdict = unjudged`, the
    /// error class as reason, the judge model. The bridge only lets a marker
    /// land on an edge with no real verdict.
    async fn mark_unjudged(&self, src: &str, dst: &str, model: &str, why: &str) -> bool {
        let marker = EdgeVerdict {
            verdict: Verdict::Unjudged,
            verdict_survivor: None,
            verdict_reason: sanitize_reason(&format!("unjudged: {why}")),
            verdict_confidence: 0.0,
            verdict_model: model.to_string(),
            judged_at: (self.clock)(),
        };
        self.persist_verdict(src, dst, &marker).await
    }

    /// Write a verdict (or failure marker) onto the `src -> dst` edge through
    /// the graph trait. Graph blips are non-fatal by design: `false` means the
    /// edge is still unannotated and the next backfill pass sees it again.
    async fn persist_verdict(&self, src: &str, dst: &str, edge: &EdgeVerdict) -> bool {
        let (sa, sd) = (&src[..8.min(src.len())], &dst[..8.min(dst.len())]);
        match self.graph.set_contradiction_verdict(src, dst, edge).await {
            Ok(true) => true,
            Ok(false) => {
                tracing::warn!(
                    a = sa,
                    b = sd,
                    "no CONTRADICTS edge matched; verdict not persisted"
                );
                false
            }
            Err(e) => {
                tracing::warn!(a = sa, b = sd, "verdict persist failed (non-fatal): {e}");
                false
            }
        }
    }

    // ─── Tool 7: memory_contradictions ──────────────────────────────────

    /// List CONTRADICTS pairs with their judge verdicts, newest first.
    ///
    /// Every filter runs graph-side (`ContradictionQuery`), so `offset` and
    /// `limit` page over *matching* pairs and a run of resolved pairs at the
    /// top of the queue can never hide the rest (LAB-3283 review).
    /// `include_resolved = false` excludes pairs whose endpoint carries an
    /// incoming `SUPERSEDES` edge — the graph-side twin of Qdrant's
    /// `superseded_by`, measured complete on 2026-09-10 — and pairs stamped
    /// `keep_both` by `resolve_contradiction` (LAB-3885). The Qdrant flag is
    /// still applied per pair as a guard against a failed edge write (graph
    /// writes are non-fatal); that can only shorten a page, never hide the
    /// next one. `verdicts = None` applies `DEFAULT_VERDICT_FILTER`.
    /// `next_offset` is set while the graph page was full.
    #[tracing::instrument(skip(self))]
    pub async fn memory_contradictions(
        &self,
        limit: usize,
        offset: usize,
        include_resolved: bool,
        verdicts: Option<&[String]>,
    ) -> Result<Value> {
        let default: Vec<String> = DEFAULT_VERDICT_FILTER
            .iter()
            .map(|s| s.to_string())
            .collect();
        let verdicts = verdicts.unwrap_or(&default);
        if verdicts.is_empty() || verdicts.iter().any(|v| Verdict::parse(v).is_none()) {
            return Err(AlayaError::Validation(format!(
                "verdicts must be a non-empty subset of {:?}",
                Verdict::ALL.map(|v| v.as_str())
            )));
        }
        let limit = limit.clamp(1, ContradictionQuery::MAX_LIMIT);

        let query = ContradictionQuery {
            limit,
            skip: offset,
            verdicts: Some(verdicts.to_vec()),
            exclude_resolved: !include_resolved,
            ..Default::default()
        };
        let pairs = self.graph.get_all_contradictions(&query).await?;
        let next_offset = (pairs.len() == limit).then_some(offset + limit);

        // Batch fetch all referenced memories (was: N+1 sequential queries)
        let all_hashes: Vec<&str> = pairs
            .iter()
            .flat_map(|p| [p.memory_a_hash.as_str(), p.memory_b_hash.as_str()])
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        let memories = self
            .vectors
            .get_batch(&all_hashes)
            .await
            .unwrap_or_default();
        let lookup: std::collections::HashMap<&str, &Memory> = memories
            .iter()
            .map(|m| (m.content_hash.as_str(), m))
            .collect();

        let mut enriched: Vec<Value> = Vec::with_capacity(pairs.len());
        for pair in &pairs {
            let a = lookup.get(pair.memory_a_hash.as_str());
            let b = lookup.get(pair.memory_b_hash.as_str());
            let a_superseded = a.is_some_and(|m| is_superseded(m));
            let b_superseded = b.is_some_and(|m| is_superseded(m));
            if !include_resolved && (a_superseded || b_superseded) {
                continue;
            }
            let v = pair.verdict.as_ref();

            enriched.push(serde_json::json!({
                "memory_a_hash": pair.memory_a_hash,
                "memory_b_hash": pair.memory_b_hash,
                "confidence": pair.confidence,
                "created_at": pair.created_at,
                "memory_a_content": a.map(|m| {
                    m.summary.clone().unwrap_or_else(|| truncate(&m.content, 200))
                }),
                "memory_b_content": b.map(|m| {
                    m.summary.clone().unwrap_or_else(|| truncate(&m.content, 200))
                }),
                "memory_a_superseded": a_superseded,
                "memory_b_superseded": b_superseded,
                "verdict": v.map(|v| v.verdict.as_str()).unwrap_or(Verdict::UNJUDGED),
                "verdict_reason": v.map(|v| v.verdict_reason.as_str()),
                "survivor": v.and_then(|v| v.verdict_survivor.as_deref()),
                "verdict_confidence": v.map(|v| v.verdict_confidence),
                "verdict_model": v.map(|v| v.verdict_model.as_str()),
                "judged_at": v.map(|v| v.judged_at),
                "resolution": pair.resolution.map(|r| r.as_str()),
                "resolved_at": pair.resolved_at,
                "resolved_via": pair.resolved_via,
            }));
        }

        Ok(serde_json::json!({
            "success": true,
            "pairs": enriched,
            "total": enriched.len(),
            "next_offset": next_offset,
        }))
    }

    // ─── Tool: resolve_contradiction (LAB-3885) ──────────────────────────

    /// Resolve a `memory_a -> memory_b` CONTRADICTS pair as "keep both"
    /// (`Some(KeepBoth)`) — or reverse that (`None`) — without touching
    /// either memory. The stamp lives on the edge; the default queue skips
    /// stamped pairs and `include_resolved` still shows them. This is the
    /// only write path to `e.resolution*`: `relation` cannot set it, and the
    /// judge writes the verdict namespace only. Unlike `persist_verdict`
    /// this is an operator verb, so a missing edge or a graph failure is an
    /// error, not a warning. `resolved_via` is recorded verbatim
    /// (`operator:console`, `operator:mcp`, later `engine:<run-id>`);
    /// `resolved_at` is server-set.
    #[tracing::instrument(skip(self))]
    pub async fn resolve_contradiction(
        &self,
        memory_a_hash: &str,
        memory_b_hash: &str,
        resolution: Option<Resolution>,
        resolved_via: &str,
    ) -> Result<Value> {
        for h in [memory_a_hash, memory_b_hash] {
            if !alaya_types::memory::validate_content_hash(h) {
                return Err(AlayaError::Validation(
                    "invalid content_hash: expected 64-char lowercase SHA-256 hex".into(),
                ));
            }
        }
        if memory_a_hash == memory_b_hash {
            return Err(AlayaError::Validation(
                "memory_a_hash and memory_b_hash must differ".into(),
            ));
        }
        let via = resolved_via.trim();
        if via.is_empty() || via.len() > MAX_RESOLVED_VIA_LEN {
            return Err(AlayaError::Validation(format!(
                "resolved_via is required (1..={MAX_RESOLVED_VIA_LEN} chars, e.g. operator:console)"
            )));
        }

        let now = (self.clock)();
        let matched = self
            .graph
            .set_contradiction_resolution(memory_a_hash, memory_b_hash, resolution, via, now)
            .await?;
        if !matched {
            return Err(AlayaError::NotFound(format!(
                "no CONTRADICTS edge {memory_a_hash} -> {memory_b_hash}"
            )));
        }

        Ok(serde_json::json!({
            "success": true,
            "memory_a_hash": memory_a_hash,
            "memory_b_hash": memory_b_hash,
            "resolution": resolution.map(|r| r.as_str()),
            "resolved_at": resolution.map(|_| now),
            "resolved_via": resolution.map(|_| via),
        }))
    }

    // ─── Tool 8: find_duplicates ────────────────────────────────────────

    #[tracing::instrument(skip(self))]
    pub async fn find_duplicates(
        &self,
        similarity_threshold: f64,
        limit: usize,
        strategy: CanonicalStrategy,
    ) -> Result<Value> {
        let group_limit = limit.min(500);
        let scan_limit: usize = limit.min(MAX_DEDUP_SCAN);
        if limit > MAX_DEDUP_SCAN {
            tracing::warn!(
                requested = limit,
                capped_to = MAX_DEDUP_SCAN,
                "find_duplicates limit capped"
            );
        }
        let scroll_page: usize = 100;

        // Paginated scroll to collect up to scan_limit live (non-superseded) memories
        let mut memories: Vec<Memory> = Vec::new();
        let mut offset: Option<String> = None;
        let mut raw_scanned: usize = 0;
        while memories.len() < scan_limit {
            let batch_size = scroll_page.min(scan_limit - memories.len());
            let scroll = self.vectors.get_all(batch_size, offset.as_deref()).await?;
            let batch_empty = scroll.memories.is_empty();
            raw_scanned += scroll.memories.len();
            for m in scroll.memories {
                if is_superseded(&m) {
                    continue;
                }
                memories.push(m);
            }
            offset = scroll.next_offset;
            if batch_empty || offset.is_none() {
                break;
            }
        }

        if memories.len() < 2 {
            return Ok(serde_json::json!({
                "success": true,
                "groups": [],
                "total_memories_scanned": raw_scanned,
                "total_duplicates_found": 0,
            }));
        }

        // Batch embed all content
        let contents: Vec<&str> = memories.iter().map(|m| m.content.as_str()).collect();
        let embeddings = self
            .embeddings
            .embed_batch(&contents, PromptName::Passage)
            .await?;

        // Build duplicate groups
        let hashes: Vec<&str> = memories.iter().map(|m| m.content_hash.as_str()).collect();
        let created_ats: Vec<f64> = memories.iter().map(|m| m.created_at).collect();
        let access_counts: Vec<u64> = memories.iter().map(|m| m.access_count).collect();

        let mut groups = deduplication::build_duplicate_groups(
            &hashes,
            &embeddings,
            &created_ats,
            &access_counts,
            similarity_threshold,
            strategy,
        );

        // Limit output groups
        groups.truncate(group_limit);

        let total_dups: usize = groups.iter().map(|g| g.size - 1).sum();

        Ok(serde_json::json!({
            "success": true,
            "groups": groups,
            "total_memories_scanned": raw_scanned,
            "total_duplicates_found": total_dups,
        }))
    }

    // ─── Tool 9: merge_duplicates ───────────────────────────────────────

    #[tracing::instrument(skip(self, duplicate_hashes))]
    pub async fn merge_duplicates(
        &self,
        canonical_hash: &str,
        duplicate_hashes: &[&str],
        reason: &str,
        dry_run: bool,
    ) -> Result<Value> {
        if !alaya_types::memory::validate_content_hash(canonical_hash) {
            return Err(AlayaError::Validation("invalid canonical_hash".into()));
        }

        // Verify canonical exists
        if self.vectors.get_by_hash(canonical_hash).await?.is_none() {
            return Err(AlayaError::Validation(format!(
                "canonical memory not found: {canonical_hash}"
            )));
        }

        if dry_run {
            return Ok(serde_json::json!({
                "success": true,
                "canonical_hash": canonical_hash,
                "superseded": duplicate_hashes,
                "errors": [],
                "dry_run": true,
            }));
        }

        // Per-item validation first: a malformed or self-referential hash gets
        // its own error entry and must not poison the batch existence check
        // (get_batch rejects the whole call on any invalid hash).
        let mut errors: Vec<Value> = Vec::new();
        let mut candidates: Vec<&str> = Vec::new();
        for &dup_hash in duplicate_hashes {
            if dup_hash == canonical_hash {
                errors.push(serde_json::json!({
                    "hash": dup_hash,
                    "error": AlayaError::Validation(
                        "old_hash and new_hash must differ".into()
                    )
                    .safe_message(),
                }));
            } else if !alaya_types::memory::validate_content_hash(dup_hash) {
                errors.push(serde_json::json!({
                    "hash": dup_hash,
                    "error": AlayaError::Validation("invalid content_hash".into())
                        .safe_message(),
                }));
            } else {
                candidates.push(dup_hash);
            }
        }

        // One batch GET replaces N per-duplicate existence checks.
        let mut to_supersede: Vec<&str> = Vec::new();
        if !candidates.is_empty() {
            let existing: std::collections::HashSet<String> = self
                .vectors
                .get_batch(&candidates)
                .await?
                .into_iter()
                .map(|m| m.content_hash)
                .collect();
            for &dup_hash in &candidates {
                if existing.contains(dup_hash) {
                    to_supersede.push(dup_hash);
                } else {
                    errors.push(serde_json::json!({
                        "hash": dup_hash,
                        "error": AlayaError::Validation(format!(
                            "old memory not found: {dup_hash}"
                        ))
                        .safe_message(),
                    }));
                }
            }
        }

        // One batched metadata update + one batched edge write for the whole
        // set — same audit trail per memory as a per-duplicate supersede.
        let mut superseded: Vec<String> = Vec::new();
        if !to_supersede.is_empty() {
            match self
                .mark_superseded(&to_supersede, canonical_hash, reason)
                .await
            {
                Ok(()) => superseded.extend(to_supersede.iter().map(|s| s.to_string())),
                Err(e) => {
                    let msg = e.safe_message();
                    for &dup_hash in &to_supersede {
                        errors.push(serde_json::json!({
                            "hash": dup_hash,
                            "error": msg,
                        }));
                    }
                }
            }
        }

        Ok(serde_json::json!({
            "success": errors.is_empty(),
            "canonical_hash": canonical_hash,
            "superseded": superseded,
            "errors": errors,
            "dry_run": false,
        }))
    }

    // ─── Tool 10: get_memory ────────────────────────────────────────────

    /// Exact retrieval by `content_hash`. Unlike `search`, this is a
    /// deterministic single-item lookup with explicit not-found semantics:
    /// a missing memory returns `found: false`, never an empty result set
    /// that masquerades as low recall.
    ///
    /// Superseded memories are always returned — the caller asked for a
    /// specific hash, typically to inspect before a supersede/delete, so
    /// hiding it would defeat the purpose. `metadata.superseded_by` signals
    /// superseded status. Pure read: no access-count mutation.
    #[tracing::instrument(skip(self))]
    pub async fn get_memory(&self, content_hash: &str, output: OutputMode) -> Result<Value> {
        if !alaya_types::memory::validate_content_hash(content_hash) {
            return Err(AlayaError::Validation(
                "invalid content_hash: expected 64-char lowercase SHA-256 hex".into(),
            ));
        }

        match self.vectors.get_by_hash(content_hash).await? {
            Some(memory) => Ok(serde_json::json!({
                "found": true,
                "memory": format_memory_result(&memory, 1.0, output),
            })),
            None => Ok(serde_json::json!({
                "found": false,
                "content_hash": content_hash,
            })),
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Runs `fut` under [`RerankingService::timeout`]; `None` means the budget
/// expired. Native builds (alaya-server, production) bound it with
/// `tokio::time::timeout` and nothing else: it polls `fut` before its own
/// deadline, so a response that has already arrived is used even on a late
/// poll, and dropping `fut` on elapse aborts the request. That makes the two
/// outcomes exact — `None` is "nothing arrived within the budget", `Some(Err)`
/// is a real transport or HTTP error with its detail intact. wasm32
/// (`alaya-worker`, deferred) has no tokio timer and awaits directly — there
/// the per-request reqwest timeout in `RerankClient::rerank` is the bound and
/// surfaces as `Some(Err)`.
#[cfg(not(target_arch = "wasm32"))]
async fn rerank_with_budget(
    budget: std::time::Duration,
    fut: impl std::future::Future<Output = Result<Vec<f32>>>,
) -> Option<Result<Vec<f32>>> {
    tokio::time::timeout(budget, fut).await.ok()
}

#[cfg(target_arch = "wasm32")]
async fn rerank_with_budget(
    _budget: std::time::Duration,
    fut: impl std::future::Future<Output = Result<Vec<f32>>>,
) -> Option<Result<Vec<f32>>> {
    Some(fut.await)
}

fn current_timestamp() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

fn parse_user_relation(s: &str) -> Result<UserRelationType> {
    match s {
        "RELATES_TO" => Ok(UserRelationType::RelatesTo),
        "PRECEDES" => Ok(UserRelationType::Precedes),
        "CONTRADICTS" => Ok(UserRelationType::Contradicts),
        "SUPERSEDES" => Err(AlayaError::Validation(
            "SUPERSEDES is system-only; use memory_supersede".into(),
        )),
        _ => Err(AlayaError::Validation(format!(
            "unknown relation type: {s}"
        ))),
    }
}

/// True when the memory has been superseded (`metadata.superseded_by` set).
///
/// Superseded filtering MUST happen here at the application layer: the
/// `PayloadFilter.exclude_superseded` flag is a documented no-op in the
/// Qdrant backend because `is_null` on nested payload fields is unreliable
/// without an explicit payload index (issue #30, repo CLAUDE.md).
fn is_superseded(m: &Memory) -> bool {
    m.metadata
        .as_ref()
        .and_then(|md| md.get("superseded_by"))
        .is_some()
}

/// Over-fetch and filter superseded at the application layer (the
/// PayloadFilter route is a no-op — see is_superseded). Calls `fetch` with a
/// growing fetch size, doubling until `target` live results are collected,
/// the backend is exhausted, or `max_fetch` is reached. Returns the filtered
/// results and whether the backend was exhausted. Each caller supplies its
/// own initial size, target, and cap — the loop shape is the shared part.
async fn fetch_live_with_retry<F, Fut>(
    mut fetch_size: usize,
    target: usize,
    max_fetch: usize,
    include_superseded: bool,
    mut fetch: F,
) -> Result<(Vec<ScoredMemory>, bool)>
where
    F: FnMut(usize) -> Fut,
    Fut: std::future::Future<Output = Result<Vec<ScoredMemory>>>,
{
    loop {
        let raw = fetch(fetch_size).await?;
        let exhausted = raw.len() < fetch_size;

        let filtered: Vec<ScoredMemory> = raw
            .into_iter()
            .filter(|sm| include_superseded || !is_superseded(&sm.memory))
            .collect();

        if filtered.len() >= target || exhausted || fetch_size >= max_fetch {
            return Ok((filtered, exhausted));
        }
        fetch_size = (fetch_size * 2).min(max_fetch);
    }
}

fn format_memory_result(memory: &Memory, score: f64, output: OutputMode) -> Value {
    let mut v = serde_json::json!({
        "content_hash": memory.content_hash,
        "tags": memory.tags,
        "memory_type": memory.memory_type,
        "metadata": memory.metadata,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "salience_score": memory.salience_score,
        "score": score,
        "provenance": memory.provenance,
        "access_count": memory.access_count,
    });
    let obj = v.as_object_mut().unwrap();
    match output {
        OutputMode::Full => {
            obj.insert("content".into(), serde_json::json!(memory.content));
            if memory.summary.is_some() {
                obj.insert("summary".into(), serde_json::json!(memory.summary));
            }
        }
        OutputMode::Summary => {
            // Fallback to truncated content when summary is not yet available
            let summary_val = match &memory.summary {
                Some(s) => serde_json::json!(s),
                None => serde_json::json!(truncate(&memory.content, 200)),
            };
            obj.insert("summary".into(), summary_val);
        }
        OutputMode::Both => {
            obj.insert("content".into(), serde_json::json!(memory.content));
            obj.insert("summary".into(), serde_json::json!(memory.summary));
        }
    }
    v
}

fn truncate(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn truncate_handles_multibyte_utf8() {
        let chinese = "你好世界测试内容";
        let result = truncate(chinese, 4);
        assert!(result.ends_with("..."));
        assert_eq!(result, "你好世界...");
    }

    #[test]
    fn truncate_handles_emoji() {
        let emoji = "🎉🎊🎈🎁🎂";
        let result = truncate(emoji, 3);
        assert!(result.ends_with("..."));
        assert_eq!(result, "🎉🎊🎈...");
    }

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    // ─── Mock backends for tag cache tests ─────────────────────────────

    use alaya_backends::{
        ConsolidationService, EmbeddingProvider, GraphService, HebbianService, SummaryProvider,
        VectorStorage,
    };
    use alaya_types::{
        AlayaError,
        graph::{
            CoAccessPair, Contradiction, ContradictionRef, Direction, Edge, EdgeMeta, GraphStats,
            Neighbor, SystemRelationType, UserRelationType,
        },
        memory::{
            HealthStatus, Memory, MetadataUpdate, PatchMemoryRequest, ScoredMemory, ScrollResult,
        },
        search::{PayloadFilter, PromptName},
    };
    use async_trait::async_trait;

    /// Mock VectorStorage that counts `get_all_tags` calls.
    struct MockVectors {
        get_all_tags_calls: Rc<Cell<usize>>,
        tags: Vec<String>,
    }

    impl MockVectors {
        fn new(tags: Vec<String>, counter: Rc<Cell<usize>>) -> Self {
            Self {
                get_all_tags_calls: counter,
                tags,
            }
        }
    }

    fn dummy_memory() -> Memory {
        Memory {
            content: "mock".into(),
            content_hash: "a".repeat(64),
            tags: vec![],
            memory_type: "note".into(),
            metadata: None,
            created_at: 0.0,
            updated_at: 0.0,
            embedding: None,
            summary: None,
            salience_score: 0.0,
            access_count: 0,
            access_timestamps: vec![],
            emotional_valence: None,
            encoding_context: None,
            provenance: None,
            summary_embedding: None,
        }
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectors {
        async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
        }
        async fn exists(&self, _h: &str) -> Result<bool> {
            Ok(false)
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn get_batch(&self, _h: &[&str]) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            Ok(dummy_memory())
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
            Ok(ScrollResult {
                memories: vec![],
                next_offset: None,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            Ok(100)
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            self.get_all_tags_calls
                .set(self.get_all_tags_calls.get() + 1);
            Ok(self.tags.clone())
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    /// Mock embedding provider returning a fixed vector.
    struct MockEmbeddings;

    #[async_trait(?Send)]
    impl EmbeddingProvider for MockEmbeddings {
        async fn embed_batch(&self, texts: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![0.0; 1024]).collect())
        }
        fn dimensions(&self) -> usize {
            1024
        }
        fn model_name(&self) -> &str {
            "mock"
        }
    }

    /// No-op graph service.
    struct MockGraph;

    #[async_trait(?Send)]
    impl GraphService for MockGraph {
        async fn ensure_node(&self, _h: &str, _t: f64) -> Result<()> {
            Ok(())
        }
        async fn delete_node(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn create_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
            _m: EdgeMeta,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_typed_edges(
            &self,
            _h: &str,
            _r: Option<UserRelationType>,
            _d: Direction,
            _l: usize,
        ) -> Result<Vec<Edge>> {
            Ok(vec![])
        }
        async fn delete_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn create_system_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: SystemRelationType,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_all_contradictions(
            &self,
            _q: &alaya_types::graph::ContradictionQuery,
        ) -> Result<Vec<Contradiction>> {
            Ok(vec![])
        }
        async fn set_contradiction_verdict(
            &self,
            _s: &str,
            _d: &str,
            _v: &alaya_types::graph::EdgeVerdict,
        ) -> Result<bool> {
            Ok(true)
        }

        async fn set_contradiction_resolution(
            &self,
            _s: &str,
            _d: &str,
            _r: Option<Resolution>,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_contradictions_for_hashes(
            &self,
            _h: &[&str],
        ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
            Ok(HashMap::new())
        }
        async fn get_neighbors(
            &self,
            _h: &str,
            _m: u8,
            _w: f64,
            _l: usize,
        ) -> Result<Vec<Neighbor>> {
            Ok(vec![])
        }
        async fn spreading_activation(
            &self,
            _s: &[&str],
            _m: u8,
            _d: f64,
            _a: f64,
            _l: usize,
        ) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn hebbian_boosts_within(&self, _h: &[&str]) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn get_stats(&self) -> Result<GraphStats> {
            Ok(GraphStats {
                graph_name: "mock".into(),
                node_count: 0,
                edge_count: 0,
                hebbian_edge_count: 0,
                typed_edge_counts: HashMap::new(),
                status: "ok".into(),
            })
        }
    }

    struct MockHebbian;
    #[async_trait(?Send)]
    impl HebbianService for MockHebbian {
        async fn enqueue_strengthen(&self, _p: &[CoAccessPair]) -> Result<()> {
            Ok(())
        }
    }

    struct MockConsolidation;
    #[async_trait(?Send)]
    impl ConsolidationService for MockConsolidation {
        async fn decay_all_edges(&self, _d: f64, _l: usize) -> Result<usize> {
            Ok(0)
        }
        async fn decay_stale_edges(&self, _s: f64, _d: f64, _l: usize) -> Result<usize> {
            Ok(0)
        }
        async fn prune_weak_edges(&self, _t: f64, _l: usize) -> Result<usize> {
            Ok(0)
        }
        async fn get_orphan_nodes(&self, _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
    }

    fn build_mock_service(tags: Vec<String>) -> (MemoryService, Rc<Cell<usize>>) {
        let counter = Rc::new(Cell::new(0));
        let svc = MemoryService::new(
            Box::new(MockVectors::new(tags, counter.clone())),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );
        (svc, counter)
    }

    fn build_mock_service_with_clock(
        tags: Vec<String>,
        clock: fn() -> f64,
    ) -> (MemoryService, Rc<Cell<usize>>) {
        let counter = Rc::new(Cell::new(0));
        let svc = MemoryService::with_clock(
            Box::new(MockVectors::new(tags, counter.clone())),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            clock,
        );
        (svc, counter)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tag_cache_avoids_repeat_fetches() {
        let (svc, counter) = build_mock_service(vec!["rust".into(), "alaya".into()]);

        let params = SearchParams {
            query: "test query about rust".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // First search — must call get_all_tags
        let _ = svc.search(params.clone()).await;
        assert_eq!(counter.get(), 1, "first search should fetch tags");

        // Second search within TTL — must NOT call get_all_tags again
        let _ = svc.search(params.clone()).await;
        assert_eq!(
            counter.get(),
            1,
            "second search within TTL should use cached tags"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tag_cache_invalidated_on_store_with_tags() {
        let (svc, counter) = build_mock_service(vec!["rust".into()]);

        let search_params = SearchParams {
            query: "test query".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // Populate cache
        let _ = svc.search(search_params.clone()).await;
        assert_eq!(counter.get(), 1);

        // Store a memory WITH tags — should invalidate cache
        let store_params = StoreParams {
            content: "test content".into(),
            tags: Some(vec!["new-tag".into()]),
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };
        let _ = svc.store_memory(store_params).await;

        // Next search should re-fetch tags
        let _ = svc.search(search_params).await;
        assert_eq!(
            counter.get(),
            2,
            "search after store-with-tags should re-fetch"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tag_cache_not_invalidated_on_store_without_tags() {
        let (svc, counter) = build_mock_service(vec!["rust".into()]);

        let search_params = SearchParams {
            query: "test query".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // Populate cache
        let _ = svc.search(search_params.clone()).await;
        assert_eq!(counter.get(), 1);

        // Store a memory WITHOUT tags — should NOT invalidate cache
        let store_params = StoreParams {
            content: "tagless content".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };
        let _ = svc.store_memory(store_params).await;

        // Next search should still use cached tags
        let _ = svc.search(search_params).await;
        assert_eq!(
            counter.get(),
            1,
            "store without tags should not invalidate cache"
        );
    }

    // ─── TTL expiry tests ─────────────────────────────────────────────────

    thread_local! {
        static MOCK_TIME: Cell<f64> = const { Cell::new(1_000_000.0) };
    }

    fn mock_clock() -> f64 {
        MOCK_TIME.with(|t| t.get())
    }

    fn advance_clock(seconds: f64) {
        MOCK_TIME.with(|t| t.set(t.get() + seconds));
    }

    fn reset_clock() {
        MOCK_TIME.with(|t| t.set(1_000_000.0));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tag_cache_expires_after_ttl() {
        reset_clock();
        let (svc, counter) =
            build_mock_service_with_clock(vec!["rust".into(), "alaya".into()], mock_clock);

        let params = SearchParams {
            query: "test query about rust".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // First search — populates cache
        let _ = svc.search(params.clone()).await;
        assert_eq!(counter.get(), 1, "first search fetches tags");

        // 30s later — within TTL, should use cache
        advance_clock(30.0);
        let _ = svc.search(params.clone()).await;
        assert_eq!(counter.get(), 1, "30s later: still cached");

        // 61s total — past TTL, must re-fetch
        advance_clock(31.0);
        let _ = svc.search(params.clone()).await;
        assert_eq!(counter.get(), 2, "61s later: cache expired, re-fetched");

        // Immediately after — fresh cache, should not fetch again
        let _ = svc.search(params).await;
        assert_eq!(counter.get(), 2, "immediately after refresh: still cached");
    }

    // ─── find_duplicates scan cap tests ──────────────────────────────────

    /// Mock VectorStorage that returns `total` synthetic memories from `get_all`,
    /// paginated in chunks. Tracks how many memories were actually fetched.
    struct MockVectorsWithMemories {
        total: usize,
        fetched: Rc<Cell<usize>>,
        /// Shared counter for get_all_tags calls (satisfies MockVectors API).
        get_all_tags_calls: Rc<Cell<usize>>,
    }

    impl MockVectorsWithMemories {
        fn new(total: usize, fetched: Rc<Cell<usize>>) -> Self {
            Self {
                total,
                fetched,
                get_all_tags_calls: Rc::new(Cell::new(0)),
            }
        }

        fn make_memory(i: usize) -> Memory {
            Memory {
                content: format!("memory content {i}"),
                content_hash: format!("{i:064x}"),
                tags: vec![],
                memory_type: "note".into(),
                metadata: None,
                created_at: 1_000_000.0 + i as f64,
                updated_at: 1_000_000.0 + i as f64,
                embedding: None,
                summary: None,
                salience_score: 0.5,
                access_count: 1,
                access_timestamps: vec![],
                emotional_valence: None,
                encoding_context: None,
                provenance: None,
                summary_embedding: None,
            }
        }
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsWithMemories {
        async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
        }
        async fn exists(&self, _h: &str) -> Result<bool> {
            Ok(false)
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn get_batch(&self, _h: &[&str]) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            Err(AlayaError::NotFound("mock".into()))
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult> {
            let start = offset.and_then(|o| o.parse::<usize>().ok()).unwrap_or(0);
            let end = (start + limit).min(self.total);
            let memories: Vec<Memory> = (start..end).map(Self::make_memory).collect();
            self.fetched.set(self.fetched.get() + memories.len());
            let next_offset = if end < self.total {
                Some(end.to_string())
            } else {
                None
            };
            Ok(ScrollResult {
                memories,
                next_offset,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            Ok(self.total)
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            self.get_all_tags_calls
                .set(self.get_all_tags_calls.get() + 1);
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    /// Mock embedding provider that tracks how many texts were embedded.
    struct MockEmbeddingsTracked {
        embedded_count: Rc<Cell<usize>>,
    }

    #[async_trait(?Send)]
    impl EmbeddingProvider for MockEmbeddingsTracked {
        async fn embed_batch(&self, texts: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            self.embedded_count
                .set(self.embedded_count.get() + texts.len());
            // Return distinct vectors so deduplication doesn't merge them all
            Ok(texts
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    let mut v = vec![0.0_f32; 1024];
                    v[i % 1024] = 1.0;
                    v
                })
                .collect())
        }
        fn dimensions(&self) -> usize {
            1024
        }
        fn model_name(&self) -> &str {
            "mock-tracked"
        }
    }

    fn build_dedup_service(
        total_memories: usize,
    ) -> (MemoryService, Rc<Cell<usize>>, Rc<Cell<usize>>) {
        let fetched = Rc::new(Cell::new(0));
        let embedded = Rc::new(Cell::new(0));
        let svc = MemoryService::new(
            Box::new(MockVectorsWithMemories::new(
                total_memories,
                fetched.clone(),
            )),
            Box::new(MockEmbeddingsTracked {
                embedded_count: embedded.clone(),
            }),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );
        (svc, fetched, embedded)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn find_duplicates_caps_scan_at_max_dedup_scan() {
        // 300 memories available — more than MAX_DEDUP_SCAN (200)
        let (svc, _fetched, embedded) = build_dedup_service(300);

        let result = svc
            .find_duplicates(0.95, 500, CanonicalStrategy::KeepNewest)
            .await
            .unwrap();

        // The service should have capped scan to MAX_DEDUP_SCAN, not scanned all 300
        let scanned = result["total_memories_scanned"].as_u64().unwrap() as usize;
        assert!(
            scanned <= MAX_DEDUP_SCAN,
            "scan should be capped at {MAX_DEDUP_SCAN}, but scanned {scanned}"
        );

        // Embeddings should match the capped count, not the full 300
        let embed_count = embedded.get();
        assert!(
            embed_count <= MAX_DEDUP_SCAN,
            "should embed at most {MAX_DEDUP_SCAN} memories, but embedded {embed_count}"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn find_duplicates_under_cap_scans_all() {
        // 50 memories — well under the cap
        let (svc, _fetched, embedded) = build_dedup_service(50);

        let result = svc
            .find_duplicates(0.95, 500, CanonicalStrategy::KeepNewest)
            .await
            .unwrap();

        // Should scan all 50 since that's under the cap
        let scanned = result["total_memories_scanned"].as_u64().unwrap() as usize;
        assert_eq!(scanned, 50, "should scan all 50 when under cap");
        assert_eq!(embedded.get(), 50, "should embed all 50 when under cap");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tag_cache_ttl_boundary_exactly_at_expiry() {
        reset_clock();
        let (svc, counter) = build_mock_service_with_clock(vec!["test".into()], mock_clock);

        let params = SearchParams {
            query: "boundary test".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // Populate cache
        let _ = svc.search(params.clone()).await;
        assert_eq!(counter.get(), 1);

        // Exactly at TTL boundary (60.0s) — cache expires (< is strict, not <=)
        advance_clock(60.0);
        let _ = svc.search(params.clone()).await;
        assert_eq!(
            counter.get(),
            2,
            "exactly at TTL: expires (strict < comparison)"
        );

        // Immediately after refresh — should be cached again
        advance_clock(0.001);
        let _ = svc.search(params).await;
        assert_eq!(counter.get(), 2, "just after refresh: still cached");
    }

    // ─── Mock backends for batch edge tests ──────────────────────────────

    /// Mock graph that tracks individual vs batch create_typed_edge calls.
    struct MockGraphBatchTracker {
        individual_calls: Rc<Cell<usize>>,
        batch_calls: Rc<Cell<usize>>,
        batch_edge_count: Rc<Cell<usize>>,
    }

    impl MockGraphBatchTracker {
        fn new(
            individual: Rc<Cell<usize>>,
            batch: Rc<Cell<usize>>,
            batch_edges: Rc<Cell<usize>>,
        ) -> Self {
            Self {
                individual_calls: individual,
                batch_calls: batch,
                batch_edge_count: batch_edges,
            }
        }
    }

    #[async_trait(?Send)]
    impl GraphService for MockGraphBatchTracker {
        async fn ensure_node(&self, _h: &str, _t: f64) -> Result<()> {
            Ok(())
        }
        async fn delete_node(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn create_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
            _m: EdgeMeta,
        ) -> Result<bool> {
            self.individual_calls.set(self.individual_calls.get() + 1);
            Ok(true)
        }
        async fn create_typed_edges_batch(
            &self,
            edges: &[(String, String, UserRelationType, EdgeMeta)],
        ) -> Result<usize> {
            self.batch_calls.set(self.batch_calls.get() + 1);
            self.batch_edge_count
                .set(self.batch_edge_count.get() + edges.len());
            Ok(edges.len())
        }
        async fn get_typed_edges(
            &self,
            _h: &str,
            _r: Option<UserRelationType>,
            _d: Direction,
            _l: usize,
        ) -> Result<Vec<Edge>> {
            Ok(vec![])
        }
        async fn delete_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn create_system_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: SystemRelationType,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_all_contradictions(
            &self,
            _q: &alaya_types::graph::ContradictionQuery,
        ) -> Result<Vec<Contradiction>> {
            Ok(vec![])
        }
        async fn set_contradiction_verdict(
            &self,
            _s: &str,
            _d: &str,
            _v: &alaya_types::graph::EdgeVerdict,
        ) -> Result<bool> {
            Ok(true)
        }

        async fn set_contradiction_resolution(
            &self,
            _s: &str,
            _d: &str,
            _r: Option<Resolution>,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_contradictions_for_hashes(
            &self,
            _h: &[&str],
        ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
            Ok(HashMap::new())
        }
        async fn get_neighbors(
            &self,
            _h: &str,
            _m: u8,
            _w: f64,
            _l: usize,
        ) -> Result<Vec<Neighbor>> {
            Ok(vec![])
        }
        async fn spreading_activation(
            &self,
            _s: &[&str],
            _m: u8,
            _d: f64,
            _a: f64,
            _l: usize,
        ) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn hebbian_boosts_within(&self, _h: &[&str]) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn get_stats(&self) -> Result<GraphStats> {
            Ok(GraphStats {
                graph_name: "mock".into(),
                node_count: 0,
                edge_count: 0,
                hebbian_edge_count: 0,
                typed_edge_counts: HashMap::new(),
                status: "ok".into(),
            })
        }
    }

    /// Mock VectorStorage that returns similar memories for interference detection.
    struct MockVectorsWithSimilar {
        similar_memories: Vec<ScoredMemory>,
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsWithSimilar {
        async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
        }
        async fn exists(&self, _h: &str) -> Result<bool> {
            Ok(false)
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn get_batch(&self, _h: &[&str]) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            Err(AlayaError::NotFound("mock".into()))
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(self.similar_memories.clone())
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
            Ok(ScrollResult {
                memories: vec![],
                next_offset: None,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            Ok(100)
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    fn make_contradicting_memory(hash: &str, content: &str, score: f64) -> ScoredMemory {
        ScoredMemory {
            memory: Memory {
                content: content.into(),
                content_hash: hash.into(),
                tags: vec![],
                memory_type: "note".into(),
                metadata: None,
                created_at: 1_000_000.0,
                updated_at: 1_000_000.0,
                embedding: None,
                summary: None,
                salience_score: 0.5,
                access_count: 0,
                access_timestamps: vec![],
                emotional_valence: None,
                encoding_context: None,
                provenance: None,
                summary_embedding: None,
            },
            score,
        }
    }

    #[allow(clippy::type_complexity)]
    fn build_batch_test_service(
        similar: Vec<ScoredMemory>,
    ) -> (
        MemoryService,
        Rc<Cell<usize>>,
        Rc<Cell<usize>>,
        Rc<Cell<usize>>,
    ) {
        let individual = Rc::new(Cell::new(0));
        let batch = Rc::new(Cell::new(0));
        let batch_edges = Rc::new(Cell::new(0));
        let svc = MemoryService::new(
            Box::new(MockVectorsWithSimilar {
                similar_memories: similar,
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraphBatchTracker::new(
                individual.clone(),
                batch.clone(),
                batch_edges.clone(),
            )),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );
        (svc, individual, batch, batch_edges)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn store_memory_batches_interference_edges() {
        // Two existing memories with high similarity — trigger CONTRADICTS edges
        // (negation asymmetry: new content has "not", "failed", "cannot", "won't")
        let similar = vec![
            make_contradicting_memory(
                "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000",
                "Authentication is required for all API endpoints",
                0.92,
            ),
            make_contradicting_memory(
                "bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000",
                "The cache should be enabled for performance",
                0.85,
            ),
            // Moderate similarity — should become RELATES_TO
            make_contradicting_memory(
                "cccc0000cccc0000cccc0000cccc0000cccc0000cccc0000cccc0000cccc0000",
                "API security configuration notes",
                0.55,
            ),
        ];

        let (svc, individual_calls, batch_calls, batch_edge_count) =
            build_batch_test_service(similar);

        // Content with negation asymmetry vs the existing memories
        let params = StoreParams {
            content: "Authentication is not required, it failed and cannot be used and won't work"
                .into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };

        let result = svc.store_memory(params).await;
        assert!(result.is_ok(), "store_memory should succeed");

        // Key assertion: edges should be batched, not created individually
        assert_eq!(
            individual_calls.get(),
            0,
            "should NOT call create_typed_edge individually"
        );
        assert_eq!(
            batch_calls.get(),
            1,
            "should call create_typed_edges_batch exactly once"
        );
        // At minimum: CONTRADICTS edges for the high-similarity memories + RELATES_TO
        assert!(
            batch_edge_count.get() >= 1,
            "batch should contain at least 1 edge, got {}",
            batch_edge_count.get()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn store_memory_no_batch_when_no_interference() {
        // No similar memories — no edges to create
        let (svc, individual_calls, _batch_calls, batch_edge_count) =
            build_batch_test_service(vec![]);

        let params = StoreParams {
            content: "A completely standalone memory with no similar content".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };

        let result = svc.store_memory(params).await;
        assert!(result.is_ok());

        // No edges created at all
        assert_eq!(individual_calls.get(), 0);
        assert_eq!(batch_edge_count.get(), 0);
    }

    fn superseded_scored(hash: &str, content: &str, score: f64) -> ScoredMemory {
        let mut sm = make_contradicting_memory(hash, content, score);
        let mut md = HashMap::new();
        md.insert(
            "superseded_by".to_string(),
            serde_json::json!("f".repeat(64)),
        );
        sm.memory.metadata = Some(md);
        sm
    }

    /// Superseded memories must not create interference (CONTRADICTS) edges.
    #[tokio::test(flavor = "current_thread")]
    async fn interference_skips_superseded_memories() {
        let similar = vec![superseded_scored(
            "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000",
            "Authentication is required for all API endpoints",
            0.92,
        )];

        let (svc, individual_calls, _batch_calls, batch_edge_count) =
            build_batch_test_service(similar);

        let params = StoreParams {
            content: "Authentication is not required, it failed and cannot be used".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };

        svc.store_memory(params).await.expect("store succeeds");

        assert_eq!(individual_calls.get(), 0);
        assert_eq!(
            batch_edge_count.get(),
            0,
            "no edges may be created against a superseded memory"
        );
    }

    /// A superseded near-duplicate must not block storing new content.
    #[tokio::test(flavor = "current_thread")]
    async fn dedup_ignores_superseded_near_duplicate() {
        let similar = vec![
            superseded_scored(
                "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000",
                "near-identical but superseded",
                0.99,
            ),
            make_contradicting_memory(
                "bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000",
                "vaguely related live memory",
                0.5,
            ),
        ];

        let (svc, _, _, _) = build_batch_test_service(similar);

        let params = StoreParams {
            content: "brand new content".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: Some(0.95),
        };

        let result = svc.store_memory(params).await.expect("store succeeds");
        assert!(
            !result.contains_key("duplicate"),
            "superseded neighbor at 0.99 must not trigger dedup rejection: {result:?}"
        );
    }

    /// A live duplicate ranked behind a superseded neighbor is still detected.
    #[tokio::test(flavor = "current_thread")]
    async fn dedup_detects_live_duplicate_behind_superseded() {
        let live_hash = "bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000bbbb0000";
        let similar = vec![
            superseded_scored(
                "aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000aaaa0000",
                "near-identical but superseded",
                0.99,
            ),
            make_contradicting_memory(live_hash, "near-identical and live", 0.97),
        ];

        let (svc, _, _, _) = build_batch_test_service(similar);

        let params = StoreParams {
            content: "brand new content".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: Some(0.95),
        };

        let result = svc.store_memory(params).await.expect("store succeeds");
        assert_eq!(result.get("duplicate"), Some(&serde_json::json!(true)));
        assert_eq!(
            result.get("existing_hash"),
            Some(&serde_json::json!(live_hash)),
            "duplicate must be reported against the LIVE memory, not the superseded one"
        );
    }

    /// In-memory VectorStorage honouring the `store` contract: an existing
    /// point keeps its server-maintained history and reports `created=false`.
    struct MockVectorsPersisting {
        stored: Rc<RefCell<HashMap<String, Memory>>>,
        /// Hashes present as raw points whose payload no longer parses as a
        /// `Memory` (e.g. left by the legacy writer): `exists` sees them,
        /// `get_by_hash` does not.
        raw_only: std::collections::HashSet<String>,
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsPersisting {
        async fn store(&self, m: &Memory) -> Result<(bool, String)> {
            let mut stored = self.stored.borrow_mut();
            let mut next = m.clone();
            let created = match stored.get(&m.content_hash) {
                Some(prev) => {
                    next.created_at = prev.created_at;
                    next.access_count = prev.access_count;
                    next.access_timestamps = prev.access_timestamps.clone();
                    false
                }
                None => !self.raw_only.contains(&m.content_hash),
            };
            stored.insert(m.content_hash.clone(), next);
            Ok((created, m.content_hash.clone()))
        }
        async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
            Ok(self.stored.borrow().get(h).cloned())
        }
        async fn exists(&self, h: &str) -> Result<bool> {
            Ok(self.raw_only.contains(h) || self.stored.borrow().contains_key(h))
        }
        async fn set_generated_summary(
            &self,
            h: &str,
            summary: &str,
            embedding: Option<Vec<f32>>,
        ) -> Result<bool> {
            let mut stored = self.stored.borrow_mut();
            let Some(m) = stored.get_mut(h) else {
                return Err(AlayaError::NotFound(h.into()));
            };
            if m.summary.is_some() {
                return Ok(false);
            }
            m.summary = Some(summary.into());
            m.summary_embedding = embedding;
            Ok(true)
        }
        async fn get_batch(&self, _h: &[&str]) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            Ok(())
        }
        async fn patch_memory(&self, h: &str, p: &PatchMemoryRequest) -> Result<Memory> {
            let mut stored = self.stored.borrow_mut();
            let Some(m) = stored.get_mut(h) else {
                return Err(AlayaError::NotFound(h.into()));
            };
            if let Some(ref s) = p.summary {
                if p.summary_embedding.is_none() && m.summary.as_deref() != Some(s.as_str()) {
                    m.summary_embedding = None;
                }
                m.summary = Some(s.clone());
            }
            if let Some(ref e) = p.summary_embedding {
                m.summary_embedding = Some(e.clone());
            }
            if let Some(ref t) = p.tags {
                m.tags = t.clone();
            }
            Ok(m.clone())
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
            Ok(ScrollResult {
                memories: vec![],
                next_offset: None,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            Ok(self.stored.borrow().len())
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    /// alaya#86: a second `store_memory` of identical content, with no
    /// `dedup_threshold`, must report `created: false` / "Memory updated" and
    /// leave the record's server-maintained history intact.
    #[tokio::test(flavor = "current_thread")]
    async fn restore_of_existing_content_preserves_history_and_reports_updated() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let svc = MemoryService::with_clock(
            Box::new(MockVectorsPersisting {
                stored: stored.clone(),
                raw_only: Default::default(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            mock_clock,
        );
        let params = || StoreParams {
            content: "the same fact, stored twice".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };

        let first_now = mock_clock();
        let first = svc.store_memory(params()).await.expect("first store");
        assert_eq!(first.get("created"), Some(&serde_json::json!(true)));
        assert_eq!(
            first.get("message"),
            Some(&serde_json::json!("Memory stored successfully"))
        );
        let hash = first["content_hash"].as_str().unwrap().to_string();

        // Searches accrue access history on the record between the two stores.
        {
            let mut s = stored.borrow_mut();
            let m = s.get_mut(&hash).unwrap();
            m.access_count = 7;
            m.access_timestamps = vec![first_now + 1.0, first_now + 2.0];
        }
        advance_clock(3600.0);

        let second = svc.store_memory(params()).await.expect("second store");
        assert_eq!(second.get("created"), Some(&serde_json::json!(false)));
        assert_eq!(
            second.get("message"),
            Some(&serde_json::json!("Memory updated"))
        );

        let m = stored.borrow().get(&hash).cloned().unwrap();
        assert_eq!(m.created_at, first_now, "created_at must survive re-store");
        assert_eq!(m.access_count, 7, "access_count must survive re-store");
        assert_eq!(m.access_timestamps, vec![first_now + 1.0, first_now + 2.0]);
        assert_eq!(m.updated_at, first_now + 3600.0, "updated_at moves to now");
    }

    /// A read-only principal may add memories but never reshape an existing
    /// record by re-storing its content (panel finding on alaya#86).
    #[tokio::test(flavor = "current_thread")]
    async fn read_only_restore_of_existing_content_is_refused() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let svc = MemoryService::with_clock(
            Box::new(MockVectorsPersisting {
                stored: stored.clone(),
                raw_only: Default::default(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            mock_clock,
        );
        let params = |tag: &str| StoreParams {
            content: "shared fact".into(),
            tags: Some(vec![tag.into()]),
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };

        let first = svc
            .store_memory_with(params("original"), true)
            .await
            .expect("read-only may add a new memory");
        assert_eq!(first.get("created"), Some(&serde_json::json!(true)));
        let hash = first["content_hash"].as_str().unwrap().to_string();

        let err = svc
            .store_memory_with(params("hijack"), true)
            .await
            .expect_err("read-only re-store of existing content must be refused");
        assert!(matches!(err, AlayaError::Validation(_)), "got {err:?}");
        assert_eq!(
            stored.borrow()[&hash].tags,
            vec!["original".to_string()],
            "nothing on the existing record may be rewritten"
        );
    }

    /// The read-only guard must judge existence exactly as `store` does: a
    /// point whose payload no longer parses as a `Memory` still exists, so a
    /// read-only re-store of its content is refused and nothing is written.
    #[tokio::test(flavor = "current_thread")]
    async fn read_only_guard_uses_raw_existence_not_parseability() {
        let content = "shared fact";
        let hash = generate_content_hash(content);
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let svc = MemoryService::with_clock(
            Box::new(MockVectorsPersisting {
                stored: stored.clone(),
                raw_only: std::iter::once(hash.clone()).collect(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            mock_clock,
        );
        let params = StoreParams {
            content: content.into(),
            tags: Some(vec!["hijack".into()]),
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: None,
            dedup_threshold: None,
        };

        let err = svc
            .store_memory_with(params, true)
            .await
            .expect_err("present-but-unparseable point must still refuse a read-only re-store");
        assert!(matches!(err, AlayaError::Validation(_)), "got {err:?}");
        assert!(
            !stored.borrow().contains_key(&hash),
            "the write must not have been reached"
        );
    }

    /// Summary provider that parks until released, so a caller's re-store can
    /// land between enrichment starting and its commit.
    struct BlockingSummary(RefCell<Option<tokio::sync::oneshot::Receiver<()>>>);

    #[async_trait(?Send)]
    impl SummaryProvider for BlockingSummary {
        async fn summarize(&self, _content: &str) -> Result<String> {
            let gate = self.0.borrow_mut().take();
            if let Some(gate) = gate {
                let _ = gate.await;
            }
            Ok("generated A".into())
        }
    }

    fn enrichment_service(
        stored: Rc<RefCell<HashMap<String, Memory>>>,
        gate: Option<tokio::sync::oneshot::Receiver<()>>,
    ) -> MemoryService {
        MemoryService::new(
            Box::new(MockVectorsPersisting {
                stored,
                raw_only: Default::default(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            Some(Box::new(BlockingSummary(RefCell::new(gate)))),
        )
    }

    fn fact_h(summary: Option<&str>) -> StoreParams {
        StoreParams {
            content: "fact H".into(),
            tags: None,
            memory_type: None,
            metadata: None,
            client_hostname: None,
            summary: summary.map(Into::into),
            dedup_threshold: None,
        }
    }

    /// Helly R's sixth finding: enrichment sampled "no summary" at request
    /// time and committed unconditionally after provider latency, so a
    /// caller's summary written in between was overwritten. The commit is now
    /// decided by the backend under its write lock: the caller's summary wins.
    #[tokio::test(flavor = "current_thread")]
    async fn stale_enrichment_cannot_overwrite_a_newer_caller_summary() {
        let (release, gate) = tokio::sync::oneshot::channel::<()>();
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let svc = enrichment_service(stored.clone(), Some(gate));

        let first = svc.store_memory(fact_h(None)).await.expect("first store");
        let hash = first["content_hash"].as_str().unwrap().to_string();

        // Enrichment starts and parks in the provider; the caller re-stores
        // with an explicit summary; then the stale job is released.
        let enrichment = svc.enrich_summary(&hash, "fact H");
        let caller = async {
            svc.store_memory(fact_h(Some("caller B")))
                .await
                .expect("re-store with a caller summary");
            assert_eq!(stored.borrow()[&hash].summary.as_deref(), Some("caller B"));
            release.send(()).expect("enrichment is parked on the gate");
        };
        let (applied, ()) = tokio::join!(enrichment, caller);

        assert!(
            !applied,
            "stale enrichment must not commit over a caller summary"
        );
        let m = stored.borrow()[&hash].clone();
        assert_eq!(m.summary.as_deref(), Some("caller B"));
        assert!(
            m.summary_embedding.is_none(),
            "no generated vector may accompany the caller's summary"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn enrichment_applies_when_no_caller_summary_exists() {
        let stored = Rc::new(RefCell::new(HashMap::new()));
        let svc = enrichment_service(stored.clone(), None);

        let first = svc.store_memory(fact_h(None)).await.expect("first store");
        let hash = first["content_hash"].as_str().unwrap().to_string();

        assert!(svc.enrich_summary(&hash, "fact H").await);
        let m = stored.borrow()[&hash].clone();
        assert_eq!(m.summary.as_deref(), Some("generated A"));
        assert!(
            m.summary_embedding.is_some(),
            "generated summary carries its vector"
        );
    }

    /// Summary provider that must never be reached.
    struct UnreachableSummary;

    #[async_trait(?Send)]
    impl SummaryProvider for UnreachableSummary {
        async fn summarize(&self, _content: &str) -> Result<String> {
            unreachable!("provider must not be called for a malformed hash")
        }
    }

    /// Helly R's seventh finding: backfill forwards stored `content_hash`
    /// values, and a 64-byte value with a multibyte character panicked the
    /// byte slices in the log prefix and in the point-id derivation, killing
    /// the detached backfill task. Malformed hashes are now a logged skip
    /// before any provider or storage call.
    #[tokio::test(flavor = "current_thread")]
    async fn enrich_summary_skips_malformed_hash_without_panicking() {
        let svc = MemoryService::new(
            Box::new(MockVectorsPersisting {
                stored: Rc::new(RefCell::new(HashMap::new())),
                raw_only: Default::default(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            Some(Box::new(UnreachableSummary)),
        );
        // 'é' is two bytes: across byte 8 (the log prefix slice) and across
        // byte 32 (the point-id slice). Both values are exactly 64 bytes.
        let across_8 = format!("abcdefgé{}", "a".repeat(55));
        let across_32 = format!("{}é{}", "a".repeat(31), "a".repeat(31));
        for hash in [across_8, across_32] {
            assert_eq!(hash.len(), 64);
            assert!(
                !svc.enrich_summary(&hash, "content").await,
                "a malformed hash must be a skip, not a panic"
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn patch_memory_invalidates_tag_cache_when_tags_present() {
        let (svc, counter) = build_mock_service(vec!["rust".into()]);

        let search_params = SearchParams {
            query: "test query".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // Populate cache
        let _ = svc.search(search_params.clone()).await;
        assert_eq!(counter.get(), 1);

        // Patch with tags — should invalidate cache
        let patch = PatchMemoryRequest {
            tags: Some(vec!["new-tag".into()]),
            ..Default::default()
        };
        let _ = svc.patch_memory(&"a".repeat(64), &patch).await;

        // Next search should re-fetch tags
        let _ = svc.search(search_params).await;
        assert_eq!(
            counter.get(),
            2,
            "search after patch-with-tags should re-fetch"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn patch_memory_does_not_invalidate_tag_cache_without_tags() {
        let (svc, counter) = build_mock_service(vec!["rust".into()]);

        let search_params = SearchParams {
            query: "test query".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        // Populate cache
        let _ = svc.search(search_params.clone()).await;
        assert_eq!(counter.get(), 1);

        // Patch without tags — should NOT invalidate cache
        let patch = PatchMemoryRequest {
            summary: Some("updated".into()),
            ..Default::default()
        };
        let _ = svc.patch_memory(&"a".repeat(64), &patch).await;

        // Next search should use cached tags
        let _ = svc.search(search_params).await;
        assert_eq!(
            counter.get(),
            1,
            "search after patch-without-tags should use cache"
        );
    }

    // ─── Tag deserialization ───────────────────────────────────────────

    #[test]
    fn tags_from_json_array() {
        let json = r#"{"content":"x","tags":["a","b"]}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn tags_from_csv_string() {
        let json = r#"{"content":"x","tags":"a, b, c"}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["a".into(), "b".into(), "c".into()]));
    }

    #[test]
    fn tags_from_null() {
        let json = r#"{"content":"x","tags":null}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, None);
    }

    #[test]
    fn tags_omitted() {
        let json = r#"{"content":"x"}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, None);
    }

    #[test]
    fn tags_empty_string_becomes_none() {
        let json = r#"{"content":"x","tags":""}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, None);
    }

    #[test]
    fn tags_from_stringified_json_array() {
        // Bug #17: Claude Code sometimes sends tags as a stringified JSON array
        let json = r#"{"content":"x","tags":"[\"a\",\"b\",\"c\"]"}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["a".into(), "b".into(), "c".into()]));
    }

    #[test]
    fn tags_stringified_array_with_spaces() {
        let json = r#"{"content":"x","tags":"[\"lab\", \"hooks\", \"ntfy\"]"}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(
            p.tags,
            Some(vec!["lab".into(), "hooks".into(), "ntfy".into()])
        );
    }

    #[test]
    fn tags_stringified_array_with_leading_whitespace() {
        let json = r#"{"content":"x","tags":"  [\"a\",\"b\"]  "}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn tags_whitespace_trimmed() {
        let json = r#"{"content":"x","tags":"  alpha , beta  "}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["alpha".into(), "beta".into()]));
    }

    #[test]
    fn search_tags_from_csv_string() {
        let json = r#"{"tags":"lab,infra"}"#;
        let p: SearchParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["lab".into(), "infra".into()]));
    }

    #[test]
    fn tags_deduped_from_array() {
        let json = r#"{"content":"x","tags":["a","b","a"]}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["a".into(), "b".into()]));
    }

    #[test]
    fn tags_deduped_from_csv() {
        let json = r#"{"content":"x","tags":"x, y, x"}"#;
        let p: StoreParams = serde_json::from_str(json).unwrap();
        assert_eq!(p.tags, Some(vec!["x".into(), "y".into()]));
    }

    // ─── Spreading activation injection tests ────────────────────────────

    /// Mock VectorStorage that returns pre-configured vector search results
    /// and can serve specific memories from get_batch (for graph injection).
    struct MockVectorsWithInjection {
        search_results: Vec<ScoredMemory>,
        injectable_memories: HashMap<String, Memory>,
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsWithInjection {
        async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
            Ok(self.injectable_memories.get(h).cloned())
        }
        async fn exists(&self, h: &str) -> Result<bool> {
            Ok(self.injectable_memories.contains_key(h))
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>> {
            Ok(hashes
                .iter()
                .filter_map(|h| self.injectable_memories.get(*h).cloned())
                .collect())
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            Ok(dummy_memory())
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(self.search_results.clone())
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
            Ok(ScrollResult {
                memories: vec![],
                next_offset: None,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            Ok(100)
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec!["stability".into()])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    /// Mock graph that returns pre-configured spreading activation results.
    struct MockGraphWithActivation {
        activation: HashMap<String, f64>,
    }

    #[async_trait(?Send)]
    impl GraphService for MockGraphWithActivation {
        async fn ensure_node(&self, _h: &str, _t: f64) -> Result<()> {
            Ok(())
        }
        async fn delete_node(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn create_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
            _m: EdgeMeta,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_typed_edges(
            &self,
            _h: &str,
            _r: Option<UserRelationType>,
            _d: Direction,
            _l: usize,
        ) -> Result<Vec<Edge>> {
            Ok(vec![])
        }
        async fn delete_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn create_system_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: SystemRelationType,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_all_contradictions(
            &self,
            _q: &alaya_types::graph::ContradictionQuery,
        ) -> Result<Vec<Contradiction>> {
            Ok(vec![])
        }
        async fn set_contradiction_verdict(
            &self,
            _s: &str,
            _d: &str,
            _v: &alaya_types::graph::EdgeVerdict,
        ) -> Result<bool> {
            Ok(true)
        }

        async fn set_contradiction_resolution(
            &self,
            _s: &str,
            _d: &str,
            _r: Option<Resolution>,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_contradictions_for_hashes(
            &self,
            _h: &[&str],
        ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
            Ok(HashMap::new())
        }
        async fn get_neighbors(
            &self,
            _h: &str,
            _m: u8,
            _w: f64,
            _l: usize,
        ) -> Result<Vec<Neighbor>> {
            Ok(vec![])
        }
        async fn spreading_activation(
            &self,
            _s: &[&str],
            _m: u8,
            _d: f64,
            _a: f64,
            _l: usize,
        ) -> Result<HashMap<String, f64>> {
            Ok(self.activation.clone())
        }
        async fn hebbian_boosts_within(&self, _h: &[&str]) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn get_stats(&self) -> Result<GraphStats> {
            Ok(GraphStats {
                graph_name: "mock".into(),
                node_count: 0,
                edge_count: 0,
                hebbian_edge_count: 0,
                typed_edge_counts: HashMap::new(),
                status: "ok".into(),
            })
        }
    }

    fn make_scored_memory(hash: &str, content: &str, score: f64) -> ScoredMemory {
        ScoredMemory {
            memory: Memory {
                content: content.into(),
                content_hash: hash.into(),
                tags: vec!["stability".into()],
                memory_type: "decision".into(),
                metadata: None,
                created_at: 1_000_000.0,
                updated_at: 1_000_000.0,
                embedding: None,
                summary: None,
                salience_score: 0.3,
                access_count: 5,
                access_timestamps: vec![],
                emotional_valence: None,
                encoding_context: None,
                provenance: None,
                summary_embedding: None,
            },
            score,
        }
    }

    /// Spreading activation returns neighbor hashes that should be injected
    /// into search results. This test verifies that graph-activated neighbors
    /// appear in the final results even though they weren't in the initial
    /// vector search.
    ///
    /// Bug: spreading_activation() returns {neighbor_hash: activation} but
    /// the scoring loop looks up seed hashes (which are excluded). The
    /// neighbor is never injected into the result set.
    #[tokio::test(flavor = "current_thread")]
    async fn spreading_activation_injects_neighbor_into_results() {
        let seed_hash = "a".repeat(64);
        let neighbor_hash = "b".repeat(64);

        // Seed memory returned by vector search
        let seed = make_scored_memory(&seed_hash, "VictoriaMetrics OOM incident", 0.6);

        // Neighbor memory exists in Qdrant but was not returned by vector search
        let neighbor = make_scored_memory(&neighbor_hash, "dm-cache writeback fix", 0.0);

        let mut injectable = HashMap::new();
        injectable.insert(neighbor_hash.clone(), neighbor.memory.clone());

        // Spreading activation returns the neighbor with high activation
        let mut activation = HashMap::new();
        activation.insert(neighbor_hash.clone(), 0.8);

        let svc = MemoryService::new(
            Box::new(MockVectorsWithInjection {
                search_results: vec![seed],
                injectable_memories: injectable,
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraphWithActivation { activation }),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );

        let params = SearchParams {
            query: "stability enhancements lab".into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };

        let result = svc.search(params).await.expect("search should succeed");
        let results = result["results"]
            .as_array()
            .expect("results should be array");

        // The neighbor should appear in results because spreading activation
        // identified it as strongly connected to the seed.
        let result_hashes: Vec<&str> = results
            .iter()
            .filter_map(|r| r["content_hash"].as_str())
            .collect();

        assert!(
            result_hashes.contains(&neighbor_hash.as_str()),
            "Graph-activated neighbor should be injected into search results.\n\
             Expected neighbor hash {} to be in results, but got: {:?}",
            neighbor_hash,
            result_hashes,
        );
    }

    // ─── get_memory (Tool 10) tests ──────────────────────────────────────

    /// Build a MemoryService whose vector store serves `mem` (if any) via
    /// get_by_hash. Other backends are inert mocks.
    fn service_with_memory(mem: Option<Memory>) -> MemoryService {
        let mut injectable = HashMap::new();
        if let Some(m) = mem {
            injectable.insert(m.content_hash.clone(), m);
        }
        MemoryService::new(
            Box::new(MockVectorsWithInjection {
                search_results: vec![],
                injectable_memories: injectable,
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraphWithActivation {
                activation: HashMap::new(),
            }),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        )
    }

    #[tokio::test(flavor = "current_thread")]
    async fn get_memory_found_returns_envelope() {
        let hash = "a".repeat(64);
        let mem = make_scored_memory(&hash, "lab incident postmortem", 0.0).memory;
        let svc = service_with_memory(Some(mem));

        let v = svc
            .get_memory(&hash, OutputMode::Full)
            .await
            .expect("get_memory should succeed");

        assert_eq!(v["found"], true);
        assert_eq!(v["memory"]["content_hash"], hash);
        assert_eq!(v["memory"]["content"], "lab incident postmortem");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn get_memory_missing_returns_found_false() {
        let svc = service_with_memory(None);

        let v = svc
            .get_memory(&"b".repeat(64), OutputMode::Full)
            .await
            .expect("missing hash is not an error");

        assert_eq!(v["found"], false);
        assert!(v.get("memory").is_none(), "no memory body on a miss");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn get_memory_invalid_hash_is_validation_error() {
        let svc = service_with_memory(None);

        // Truncated 8-char prefix — the exact failure class this PR targets
        let err = svc
            .get_memory("ffa51984", OutputMode::Full)
            .await
            .expect_err("short hash must be rejected");
        assert!(matches!(err, AlayaError::Validation(_)));

        // Uppercase hex is also rejected by validate_content_hash
        assert!(
            svc.get_memory(&"A".repeat(64), OutputMode::Full)
                .await
                .is_err()
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn get_memory_summary_mode_omits_content() {
        let hash = "c".repeat(64);
        let mem = make_scored_memory(&hash, "full body text", 0.0).memory;
        let svc = service_with_memory(Some(mem));

        let v = svc
            .get_memory(&hash, OutputMode::Summary)
            .await
            .expect("get_memory should succeed");

        assert_eq!(v["found"], true);
        assert!(
            v["memory"].get("content").is_none(),
            "summary mode must not include raw content"
        );
        assert!(v["memory"].get("summary").is_some());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn get_memory_returns_superseded_memory_with_marker() {
        let hash = "d".repeat(64);
        let mut mem = make_scored_memory(&hash, "outdated decision", 0.0).memory;
        let mut md = HashMap::new();
        md.insert(
            "superseded_by".to_string(),
            serde_json::json!("e".repeat(64)),
        );
        mem.metadata = Some(md);
        let svc = service_with_memory(Some(mem));

        let v = svc
            .get_memory(&hash, OutputMode::Full)
            .await
            .expect("get_memory should succeed");

        // Exact lookup returns superseded memories — caller asked for this hash.
        assert_eq!(v["found"], true);
        assert_eq!(v["memory"]["metadata"]["superseded_by"], "e".repeat(64));
    }

    // ─── Cross-encoder rerank tests ──────────────────────────────────────

    /// Mock reranker that returns a configured score per (query, doc) pair,
    /// keyed by the first 8 chars of the doc content. Unknown docs get 0.0.
    /// `sleep_for` stalls before scoring and `budget` is what `timeout()`
    /// reports, so the `RERANK_TIMEOUT_MS` budget (LAB-3507) can be exercised.
    struct MockReranker {
        top_n: usize,
        scores_by_doc_prefix: HashMap<String, f32>,
        sleep_for: std::time::Duration,
        budget: std::time::Duration,
    }

    #[async_trait(?Send)]
    impl alaya_backends::RerankingService for MockReranker {
        async fn rerank(&self, _query: &str, texts: &[&str]) -> Result<Vec<f32>> {
            tokio::time::sleep(self.sleep_for).await;
            Ok(texts
                .iter()
                .map(|t| {
                    let key: String = t.chars().take(8).collect();
                    self.scores_by_doc_prefix.get(&key).copied().unwrap_or(0.0)
                })
                .collect())
        }
        fn top_n(&self) -> usize {
            self.top_n
        }
        fn timeout(&self) -> std::time::Duration {
            self.budget
        }
    }

    /// Reranker that always returns Err — exercises graceful-degradation path.
    struct FailingReranker;

    #[async_trait(?Send)]
    impl alaya_backends::RerankingService for FailingReranker {
        async fn rerank(&self, _query: &str, _texts: &[&str]) -> Result<Vec<f32>> {
            Err(AlayaError::Rerank("simulated upstream failure".into()))
        }
        fn top_n(&self) -> usize {
            10
        }
        fn timeout(&self) -> std::time::Duration {
            std::time::Duration::from_secs(5)
        }
    }

    fn build_rerank_test_service(
        search_results: Vec<ScoredMemory>,
        reranker: Option<Box<dyn alaya_backends::RerankingService>>,
    ) -> MemoryService {
        let mut svc = MemoryService::new(
            Box::new(MockVectorsWithInjection {
                search_results,
                injectable_memories: HashMap::new(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraphWithActivation {
                activation: HashMap::new(),
            }),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );
        if let Some(r) = reranker {
            svc = svc.with_reranker(r);
        }
        svc
    }

    fn search_params(query: &str) -> SearchParams {
        SearchParams {
            query: query.into(),
            mode: SearchMode::Hybrid,
            page: 1,
            page_size: 10,
            tags: None,
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded: false,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        }
    }

    /// When a reranker is attached, top-N candidates are reordered by rerank
    /// score, overriding the original RRF order driven by vector cosine. The
    /// stub answers inside its budget, so this is also the happy path of the
    /// `RERANK_TIMEOUT_MS` wrapper (LAB-3507): a reranker that finishes in
    /// time still reorders. Paused tokio time keeps the stall zero wall-clock.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn rerank_reorders_top_n_by_cross_encoder_score() {
        // Three memories. Vector cosine order = [doc-aaaa (0.9), doc-bbbb (0.7), doc-cccc (0.5)].
        // Reranker disagrees: doc-cccc is the most relevant.
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let hash_c = "c".repeat(64);

        let results = vec![
            make_scored_memory(&hash_a, "doc-aaaa first by cosine", 0.9),
            make_scored_memory(&hash_b, "doc-bbbb second by cosine", 0.7),
            make_scored_memory(&hash_c, "doc-cccc third by cosine but best by rerank", 0.5),
        ];

        let mut scores = HashMap::new();
        scores.insert("doc-aaaa".to_string(), 0.2_f32);
        scores.insert("doc-bbbb".to_string(), 0.5_f32);
        scores.insert("doc-cccc".to_string(), 0.95_f32);

        let svc = build_rerank_test_service(
            results,
            Some(Box::new(MockReranker {
                top_n: 20,
                scores_by_doc_prefix: scores,
                sleep_for: std::time::Duration::from_millis(10),
                budget: std::time::Duration::from_secs(5),
            })),
        );

        let r = svc
            .search(search_params("which doc is most relevant"))
            .await
            .expect("search succeeds");
        let result_hashes: Vec<&str> = r["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x["content_hash"].as_str())
            .collect();

        assert_eq!(
            result_hashes,
            vec![hash_c.as_str(), hash_b.as_str(), hash_a.as_str()],
            "rerank should put the cross-encoder's best candidate (cccc) at position 1"
        );
    }

    /// When the reranker call fails, search still returns results in the
    /// pre-rerank (RRF/cosine) order — graceful degradation.
    #[tokio::test(flavor = "current_thread")]
    async fn rerank_failure_falls_back_to_rrf_order() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let results = vec![
            make_scored_memory(&hash_a, "doc-aaaa cosine wins", 0.9),
            make_scored_memory(&hash_b, "doc-bbbb cosine second", 0.5),
        ];

        let svc = build_rerank_test_service(results, Some(Box::new(FailingReranker)));

        let r = svc
            .search(search_params("test query"))
            .await
            .expect("search should still succeed when rerank errors");
        let result_hashes: Vec<&str> = r["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x["content_hash"].as_str())
            .collect();

        assert_eq!(
            result_hashes.first(),
            Some(&hash_a.as_str()),
            "RRF order (cosine-driven) preserved when rerank fails"
        );
    }

    /// Regression: a strong tail entry (high cosine, large RRF) must NOT
    /// outrank a weak-but-positive rerank entry. The Python validation slices
    /// top-N and reorders within it, with tail strictly below; the Rust
    /// implementation must preserve that invariant even when rerank scores are
    /// small in absolute terms (BGE sigmoid outputs are often in [0, 0.05]
    /// for unrelated pairs and [0.5, 0.95] for matches).
    #[tokio::test(flavor = "current_thread")]
    async fn reranked_entries_dominate_high_score_tail() {
        // top_n=2 — the third memory acts as a high-scoring tail entry.
        let hash_a = "a".repeat(64); // reranked, weak rerank score
        let hash_b = "b".repeat(64); // reranked, strong rerank score
        let hash_c = "c".repeat(64); // NOT reranked, very high cosine

        let results = vec![
            make_scored_memory(&hash_a, "doc-aaaa first by cosine", 0.6),
            make_scored_memory(&hash_b, "doc-bbbb second by cosine", 0.55),
            make_scored_memory(&hash_c, "doc-cccc third — strong tail (high cosine)", 0.95),
        ];

        let mut scores = HashMap::new();
        // Even though doc-cccc would beat reranked entries in raw cosine,
        // the rerank should still come first.
        scores.insert("doc-aaaa".to_string(), 0.05_f32);
        scores.insert("doc-bbbb".to_string(), 0.20_f32);
        // doc-cccc is NOT in the reranker's input — top_n=2 means only the
        // first two RRF entries are reranked.

        let svc = build_rerank_test_service(
            results,
            Some(Box::new(MockReranker {
                top_n: 2,
                scores_by_doc_prefix: scores,
                sleep_for: std::time::Duration::ZERO,
                budget: std::time::Duration::from_secs(5),
            })),
        );

        let r = svc
            .search(search_params("test query"))
            .await
            .expect("search succeeds");
        let result_hashes: Vec<&str> = r["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x["content_hash"].as_str())
            .collect();

        // Both reranked entries (bbbb stronger, aaaa weaker) must appear
        // before doc-cccc, even though doc-cccc has a much higher cosine.
        let pos_b = result_hashes.iter().position(|h| *h == hash_b.as_str());
        let pos_a = result_hashes.iter().position(|h| *h == hash_a.as_str());
        let pos_c = result_hashes.iter().position(|h| *h == hash_c.as_str());

        assert_eq!(pos_b, Some(0), "strong rerank entry must be first");
        assert_eq!(pos_a, Some(1), "weak rerank entry must still beat tail");
        assert_eq!(
            pos_c,
            Some(2),
            "high-cosine tail must follow reranked top-N"
        );
    }

    /// A reranker that stalls past its budget is cut off at exactly the
    /// budget and RRF order is used (LAB-3507). Its scores would *invert*
    /// RRF order, so `hash_a` first proves the rerank result was discarded,
    /// not merely a tie. Paused tokio time makes the cut-off exact — a
    /// wrapper that ignored `timeout()` for any hardcoded value would fail
    /// the equality — and the test zero wall-clock.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn rerank_timeout_falls_back_to_rrf_order() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let results = vec![
            make_scored_memory(&hash_a, "doc-aaaa cosine wins", 0.9),
            make_scored_memory(&hash_b, "doc-bbbb cosine second", 0.5),
        ];

        let mut scores = HashMap::new();
        scores.insert("doc-aaaa".to_string(), 0.1_f32);
        scores.insert("doc-bbbb".to_string(), 0.9_f32);

        let budget = std::time::Duration::from_millis(50);
        let svc = build_rerank_test_service(
            results,
            Some(Box::new(MockReranker {
                top_n: 20,
                scores_by_doc_prefix: scores,
                sleep_for: budget * 10,
                budget,
            })),
        );

        let started = tokio::time::Instant::now();
        let r = svc
            .search(search_params("test query"))
            .await
            .expect("search should still succeed when rerank times out");

        assert_eq!(
            started.elapsed(),
            budget,
            "search must be cut off at exactly the reranker's budget"
        );

        let result_hashes: Vec<&str> = r["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x["content_hash"].as_str())
            .collect();
        assert_eq!(
            result_hashes.first(),
            Some(&hash_a.as_str()),
            "RRF order (cosine-driven) preserved when rerank times out — a \
             completed rerank would have put doc-bbbb first"
        );
    }

    /// Without a reranker, search behavior is unchanged — RRF/cosine wins.
    #[tokio::test(flavor = "current_thread")]
    async fn no_reranker_preserves_rrf_order() {
        let hash_a = "a".repeat(64);
        let hash_b = "b".repeat(64);
        let results = vec![
            make_scored_memory(&hash_a, "doc-aaaa cosine wins", 0.9),
            make_scored_memory(&hash_b, "doc-bbbb cosine second", 0.5),
        ];

        let svc = build_rerank_test_service(results, None);

        let r = svc
            .search(search_params("test query"))
            .await
            .expect("search succeeds");
        let result_hashes: Vec<&str> = r["results"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|x| x["content_hash"].as_str())
            .collect();

        assert_eq!(result_hashes.first(), Some(&hash_a.as_str()));
    }

    // ─── Mock backends for merge_duplicates batch tests ──────────────────

    /// Shared recording state: seeded memories in, backend calls out.
    #[derive(Default)]
    struct MergeRecorder {
        memories: RefCell<HashMap<String, Memory>>,
        /// Each update_metadata_batch call: (hashes, update).
        updates: RefCell<Vec<(Vec<String>, MetadataUpdate)>>,
        /// Each system edge from create_system_edges_batch: (src, dst).
        system_edges: RefCell<Vec<(String, String)>>,
        get_by_hash_calls: Cell<usize>,
        get_batch_calls: Cell<usize>,
        update_batch_calls: Cell<usize>,
        single_update_calls: Cell<usize>,
        single_edge_calls: Cell<usize>,
        edge_batch_calls: Cell<usize>,
        fail_update: Cell<bool>,
    }

    impl MergeRecorder {
        fn seed(&self, hashes: &[&str]) {
            let mut mems = self.memories.borrow_mut();
            for h in hashes {
                let mut m = dummy_memory();
                m.content_hash = (*h).to_string();
                mems.insert((*h).to_string(), m);
            }
        }
    }

    struct MergeVectors(Rc<MergeRecorder>);

    #[async_trait(?Send)]
    impl VectorStorage for MergeVectors {
        async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
            self.0
                .get_by_hash_calls
                .set(self.0.get_by_hash_calls.get() + 1);
            Ok(self.0.memories.borrow().get(h).cloned())
        }
        async fn exists(&self, h: &str) -> Result<bool> {
            Ok(self.0.memories.borrow().contains_key(h))
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>> {
            self.0.get_batch_calls.set(self.0.get_batch_calls.get() + 1);
            let mems = self.0.memories.borrow();
            Ok(hashes
                .iter()
                .filter_map(|h| mems.get(*h).cloned())
                .collect())
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            self.0
                .single_update_calls
                .set(self.0.single_update_calls.get() + 1);
            Ok(())
        }
        async fn update_metadata_batch(
            &self,
            hashes: &[&str],
            updates: MetadataUpdate,
        ) -> Result<()> {
            self.0
                .update_batch_calls
                .set(self.0.update_batch_calls.get() + 1);
            if self.0.fail_update.get() {
                return Err(AlayaError::Storage("mock update failure".into()));
            }
            self.0
                .updates
                .borrow_mut()
                .push((hashes.iter().map(|h| h.to_string()).collect(), updates));
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            Err(AlayaError::NotFound("mock".into()))
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            _l: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            _l: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(vec![])
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
            Ok(ScrollResult {
                memories: vec![],
                next_offset: None,
            })
        }
        async fn get_recent(
            &self,
            _l: usize,
            _s: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn count(&self) -> Result<usize> {
            Ok(0)
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    struct MergeGraph(Rc<MergeRecorder>);

    #[async_trait(?Send)]
    impl GraphService for MergeGraph {
        async fn ensure_node(&self, _h: &str, _t: f64) -> Result<()> {
            Ok(())
        }
        async fn delete_node(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn create_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
            _m: EdgeMeta,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_typed_edges(
            &self,
            _h: &str,
            _r: Option<UserRelationType>,
            _d: Direction,
            _l: usize,
        ) -> Result<Vec<Edge>> {
            Ok(vec![])
        }
        async fn delete_typed_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: UserRelationType,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn create_system_edge(
            &self,
            _s: &str,
            _d: &str,
            _r: SystemRelationType,
            _t: f64,
        ) -> Result<bool> {
            self.0
                .single_edge_calls
                .set(self.0.single_edge_calls.get() + 1);
            Ok(true)
        }
        async fn create_system_edges_batch(
            &self,
            edges: &[(String, String, SystemRelationType, f64)],
        ) -> Result<usize> {
            self.0
                .edge_batch_calls
                .set(self.0.edge_batch_calls.get() + 1);
            let mut recorded = self.0.system_edges.borrow_mut();
            for (src, dst, rel, _ts) in edges {
                assert_eq!(*rel, SystemRelationType::Supersedes);
                recorded.push((src.clone(), dst.clone()));
            }
            Ok(edges.len())
        }
        async fn get_all_contradictions(
            &self,
            _q: &alaya_types::graph::ContradictionQuery,
        ) -> Result<Vec<Contradiction>> {
            Ok(vec![])
        }
        async fn set_contradiction_verdict(
            &self,
            _s: &str,
            _d: &str,
            _v: &alaya_types::graph::EdgeVerdict,
        ) -> Result<bool> {
            Ok(true)
        }

        async fn set_contradiction_resolution(
            &self,
            _s: &str,
            _d: &str,
            _r: Option<Resolution>,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            Ok(true)
        }
        async fn get_contradictions_for_hashes(
            &self,
            _h: &[&str],
        ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
            Ok(HashMap::new())
        }
        async fn get_neighbors(
            &self,
            _h: &str,
            _m: u8,
            _w: f64,
            _l: usize,
        ) -> Result<Vec<Neighbor>> {
            Ok(vec![])
        }
        async fn spreading_activation(
            &self,
            _s: &[&str],
            _m: u8,
            _d: f64,
            _a: f64,
            _l: usize,
        ) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn hebbian_boosts_within(&self, _h: &[&str]) -> Result<HashMap<String, f64>> {
            Ok(HashMap::new())
        }
        async fn get_stats(&self) -> Result<GraphStats> {
            Ok(GraphStats {
                graph_name: "mock".into(),
                node_count: 0,
                edge_count: 0,
                hebbian_edge_count: 0,
                typed_edge_counts: HashMap::new(),
                status: "ok".into(),
            })
        }
    }

    fn build_merge_service() -> (MemoryService, Rc<MergeRecorder>) {
        let rec = Rc::new(MergeRecorder::default());
        let svc = MemoryService::new(
            Box::new(MergeVectors(rec.clone())),
            Box::new(MockEmbeddings),
            Box::new(MergeGraph(rec.clone())),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );
        (svc, rec)
    }

    /// Result parity + batching (alaya#6): merging N duplicates must produce
    /// the exact per-memory audit trail the per-duplicate implementation
    /// produced — superseded_by = canonical, supersession_reason = reason,
    /// SUPERSEDES edge canonical→duplicate — while issuing a constant number
    /// of backend calls instead of 4 per duplicate.
    #[tokio::test(flavor = "current_thread")]
    async fn merge_duplicates_batches_backend_calls_with_result_parity() {
        let canonical = "c".repeat(64);
        let d1 = "1".repeat(64);
        let d2 = "2".repeat(64);
        let d3 = "3".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&canonical, &d1, &d2, &d3]);

        let result = svc
            .merge_duplicates(&canonical, &[&d1, &d2, &d3], "dedup test", false)
            .await
            .expect("merge succeeds");

        // Result JSON parity with the per-duplicate implementation
        assert_eq!(result["success"], serde_json::json!(true));
        assert_eq!(result["canonical_hash"], serde_json::json!(canonical));
        assert_eq!(result["superseded"], serde_json::json!([d1, d2, d3]));
        assert_eq!(result["errors"], serde_json::json!([]));

        // Call counts: 1 canonical check + 1 batch GET + 1 batch update +
        // 1 batch edge write. No per-duplicate fallbacks.
        assert_eq!(rec.get_by_hash_calls.get(), 1, "canonical checked once");
        assert_eq!(rec.get_batch_calls.get(), 1, "one batch existence check");
        assert_eq!(rec.update_batch_calls.get(), 1, "one batch metadata update");
        assert_eq!(rec.single_update_calls.get(), 0, "no per-item updates");
        assert_eq!(rec.edge_batch_calls.get(), 1, "one batch edge write");
        assert_eq!(rec.single_edge_calls.get(), 0, "no per-item edge writes");

        // Audit trail parity: same fields a single supersede writes
        let updates = rec.updates.borrow();
        let (hashes, update) = &updates[0];
        assert_eq!(hashes, &[d1.clone(), d2.clone(), d3.clone()]);
        assert_eq!(update.superseded_by.as_deref(), Some(canonical.as_str()));
        let extra = update.extra.as_ref().expect("supersession_reason present");
        assert_eq!(
            extra["supersession_reason"],
            serde_json::json!("dedup test")
        );

        // SUPERSEDES edges: canonical → each duplicate
        let edges = rec.system_edges.borrow();
        assert_eq!(
            *edges,
            vec![
                (canonical.clone(), d1.clone()),
                (canonical.clone(), d2.clone()),
                (canonical.clone(), d3.clone()),
            ]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn merge_duplicates_reports_per_item_errors() {
        let canonical = "c".repeat(64);
        let d1 = "1".repeat(64);
        let missing = "e".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&canonical, &d1]);

        let result = svc
            .merge_duplicates(
                &canonical,
                &[&d1, &missing, &canonical, "not-a-hash"],
                "dedup",
                false,
            )
            .await
            .expect("merge returns per-item errors, not a top-level failure");

        assert_eq!(result["success"], serde_json::json!(false));
        assert_eq!(result["superseded"], serde_json::json!([d1]));

        let errors = result["errors"].as_array().expect("errors array");
        let error_hashes: Vec<&str> = errors.iter().filter_map(|e| e["hash"].as_str()).collect();
        assert!(
            error_hashes.contains(&missing.as_str()),
            "missing dup errored"
        );
        assert!(
            error_hashes.contains(&canonical.as_str()),
            "self-referential dup errored"
        );
        assert!(
            error_hashes.contains(&"not-a-hash"),
            "malformed hash errored"
        );

        // The valid duplicate still got the full audit trail
        assert_eq!(rec.updates.borrow()[0].0, vec![d1.clone()]);
        assert_eq!(*rec.system_edges.borrow(), vec![(canonical, d1)]);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn merge_duplicates_update_failure_marks_all_pending_as_errors() {
        let canonical = "c".repeat(64);
        let d1 = "1".repeat(64);
        let d2 = "2".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&canonical, &d1, &d2]);
        rec.fail_update.set(true);

        let result = svc
            .merge_duplicates(&canonical, &[&d1, &d2], "dedup", false)
            .await
            .expect("backend failure becomes per-item errors");

        assert_eq!(result["success"], serde_json::json!(false));
        assert_eq!(result["superseded"], serde_json::json!([] as [&str; 0]));
        assert_eq!(result["errors"].as_array().unwrap().len(), 2);
        // No edges when the metadata commit failed
        assert!(rec.system_edges.borrow().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn merge_duplicates_dry_run_writes_nothing() {
        let canonical = "c".repeat(64);
        let d1 = "1".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&canonical, &d1]);

        let result = svc
            .merge_duplicates(&canonical, &[&d1], "dedup", true)
            .await
            .expect("dry run succeeds");

        assert_eq!(result["dry_run"], serde_json::json!(true));
        assert_eq!(rec.update_batch_calls.get(), 0);
        assert_eq!(rec.edge_batch_calls.get(), 0);
        assert_eq!(rec.get_batch_calls.get(), 0, "dry run skips batch GET");
    }

    /// memory_supersede routes through the same batched write path.
    #[tokio::test(flavor = "current_thread")]
    async fn memory_supersede_writes_audit_trail_via_batch_path() {
        let old = "0".repeat(64);
        let new = "f".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&old, &new]);

        let result = svc
            .memory_supersede(&old, &new, "corrected")
            .await
            .expect("supersede succeeds");

        assert_eq!(result["success"], serde_json::json!(true));
        assert_eq!(result["superseded"], serde_json::json!(old));
        assert_eq!(result["superseded_by"], serde_json::json!(new));

        let updates = rec.updates.borrow();
        assert_eq!(updates[0].0, vec![old.clone()]);
        assert_eq!(updates[0].1.superseded_by.as_deref(), Some(new.as_str()));
        assert_eq!(*rec.system_edges.borrow(), vec![(new, old)]);
    }

    // ─── Superseded filtering across search modes (issue #30) ───────────
    //
    // Regression for the evidence case: after superseding 5 duplicate
    // memories, all 5 still appeared in recent/similar (and hybrid/tag
    // shared the same defect — the Qdrant PayloadFilter route is a no-op).

    /// Mock VectorStorage over a fixed corpus. Returns the corpus from every
    /// search entry point with NO superseded filtering — mirroring the real
    /// Qdrant backend, where that responsibility lives at the app layer.
    struct MockVectorsCorpus {
        memories: Vec<Memory>,
    }

    impl MockVectorsCorpus {
        fn scored(&self, limit: usize) -> Vec<ScoredMemory> {
            self.memories
                .iter()
                .take(limit)
                .enumerate()
                .map(|(i, m)| ScoredMemory {
                    memory: m.clone(),
                    score: 0.9 - i as f64 * 0.01,
                })
                .collect()
        }
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsCorpus {
        async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
        }
        async fn exists(&self, _h: &str) -> Result<bool> {
            Ok(false)
        }
        async fn set_generated_summary(
            &self,
            _h: &str,
            _s: &str,
            _e: Option<Vec<f32>>,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn get_batch(&self, _h: &[&str]) -> Result<Vec<Memory>> {
            Ok(vec![])
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            Ok(true)
        }
        async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            Ok(dummy_memory())
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            limit: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(self.scored(limit))
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            limit: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(self.scored(limit))
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult> {
            let start: usize = offset.and_then(|o| o.parse().ok()).unwrap_or(0);
            let end = (start + limit).min(self.memories.len());
            Ok(ScrollResult {
                memories: self.memories[start..end].to_vec(),
                next_offset: (end < self.memories.len()).then(|| end.to_string()),
            })
        }
        async fn get_recent(
            &self,
            limit: usize,
            start_from: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            let mut sorted = self.memories.clone();
            sorted.sort_by(|a, b| b.created_at.partial_cmp(&a.created_at).unwrap());
            Ok(sorted
                .into_iter()
                .filter(|m| start_from.is_none_or(|ts| m.created_at < ts))
                .take(limit)
                .collect())
        }
        async fn count(&self) -> Result<usize> {
            Ok(self.memories.len())
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "ok".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    /// 10 memories: even indices live, odd indices superseded (5 of each).
    fn superseded_corpus() -> Vec<Memory> {
        (0..10)
            .map(|i| {
                let metadata = (i % 2 == 1).then(|| {
                    let mut md = HashMap::new();
                    md.insert(
                        "superseded_by".to_string(),
                        serde_json::json!("f".repeat(64)),
                    );
                    md
                });
                Memory {
                    content: format!("watchdog sidecar memory {i}"),
                    content_hash: format!("{i:064x}"),
                    tags: vec!["watchdog".into()],
                    memory_type: "note".into(),
                    metadata,
                    created_at: 1_000_000.0 + i as f64,
                    updated_at: 1_000_000.0 + i as f64,
                    embedding: None,
                    summary: None,
                    salience_score: 0.5,
                    access_count: 1,
                    access_timestamps: vec![],
                    emotional_valence: None,
                    encoding_context: None,
                    provenance: None,
                    summary_embedding: None,
                }
            })
            .collect()
    }

    async fn corpus_search_hashes(
        mode: SearchMode,
        include_superseded: bool,
    ) -> std::collections::HashSet<String> {
        let svc = MemoryService::new(
            Box::new(MockVectorsCorpus {
                memories: superseded_corpus(),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );
        let params = SearchParams {
            query: "watchdog sidecar".into(),
            mode,
            page: 1,
            page_size: 10,
            tags: Some(vec!["watchdog".into()]),
            match_all: false,
            k: 10,
            min_similarity: None,
            memory_type: None,
            encoding_context: None,
            include_superseded,
            min_trust_score: None,
            output: OutputMode::Full,
            cursor: None,
        };
        svc.search(params).await.expect("search succeeds")["results"]
            .as_array()
            .expect("results array")
            .iter()
            .filter_map(|x| x["content_hash"].as_str().map(String::from))
            .collect()
    }

    const ALL_MODES: [SearchMode; 5] = [
        SearchMode::Recent,
        SearchMode::Similar,
        SearchMode::Hybrid,
        SearchMode::Tag,
        SearchMode::Scan,
    ];

    #[tokio::test(flavor = "current_thread")]
    async fn superseded_memories_hidden_in_every_search_mode() {
        let live: std::collections::HashSet<String> =
            (0..10).step_by(2).map(|i| format!("{i:064x}")).collect();

        for mode in ALL_MODES {
            let hashes = corpus_search_hashes(mode, false).await;
            assert_eq!(
                hashes, live,
                "{mode:?}: expected exactly the 5 live memories, superseded must not leak"
            );
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn include_superseded_returns_them_in_every_search_mode() {
        let all: std::collections::HashSet<String> = (0..10).map(|i| format!("{i:064x}")).collect();

        for mode in ALL_MODES {
            let hashes = corpus_search_hashes(mode, true).await;
            assert_eq!(
                hashes, all,
                "{mode:?}: include_superseded=true must return all 10 memories"
            );
        }
    }

    // ─── Contradiction judge (LAB-3283: AC-2 failure paths, AC-4, AC-4b) ──

    mod judge_tests {
        use super::*;
        use crate::service::{JudgeOutcome, SHADOW_LOG_TARGET};
        use alaya_backends::{ContradictionJudge, Judgement, Survivor};
        use alaya_types::graph::{EdgeVerdict, Resolution, Verdict};
        use std::rc::Rc;
        use std::sync::{Arc, Mutex};
        use tracing::instrument::WithSubscriber;
        use tracing_subscriber::layer::SubscriberExt;

        fn src() -> String {
            "a".repeat(64)
        }
        fn dst() -> String {
            "b".repeat(64)
        }

        fn mem(hash: &str, created_at: f64) -> Memory {
            Memory {
                content: format!("content of {}", &hash[..4]),
                content_hash: hash.to_string(),
                tags: vec![],
                memory_type: "note".into(),
                metadata: None,
                created_at,
                updated_at: created_at,
                embedding: None,
                summary: None,
                salience_score: 0.0,
                access_count: 0,
                access_timestamps: vec![],
                emotional_valence: None,
                encoding_context: None,
                provenance: None,
                summary_embedding: None,
            }
        }

        enum Script {
            Ok(Judgement),
            Err(fn() -> AlayaError),
            Hang,
        }

        struct ScriptedJudge(Script);

        #[async_trait(?Send)]
        impl ContradictionJudge for ScriptedJudge {
            async fn judge(&self, _a: &Memory, _b: &Memory) -> Result<Judgement> {
                match &self.0 {
                    Script::Ok(j) => Ok(j.clone()),
                    Script::Err(mk) => Err(mk()),
                    Script::Hang => std::future::pending().await,
                }
            }

            fn model_name(&self) -> &str {
                "test-model"
            }
        }

        fn judgement(verdict: Verdict, survivor: Option<Survivor>) -> Judgement {
            Judgement {
                verdict,
                survivor,
                reason: "because".into(),
                confidence: 0.9,
                model: "test-model".into(),
                input_tokens: 10,
                output_tokens: 2,
            }
        }

        /// Serves exactly the memories it was built with. Every write method
        /// panics: the judge must never touch the vector store (AC-2, AC-4b).
        struct PairVectors(Vec<Memory>);

        #[async_trait(?Send)]
        impl VectorStorage for PairVectors {
            async fn store(&self, _m: &Memory) -> Result<(bool, String)> {
                unreachable!("judge must never write to the vector store")
            }
            async fn exists(&self, h: &str) -> Result<bool> {
                Ok(self.0.iter().any(|m| m.content_hash == h))
            }
            async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
                Ok(self.0.iter().find(|m| m.content_hash == h).cloned())
            }
            async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>> {
                Ok(self
                    .0
                    .iter()
                    .filter(|m| hashes.contains(&m.content_hash.as_str()))
                    .cloned()
                    .collect())
            }
            async fn delete(&self, _h: &str) -> Result<bool> {
                unreachable!("judge must never write to the vector store")
            }
            async fn update_metadata(&self, _h: &str, _u: MetadataUpdate) -> Result<()> {
                unreachable!("judge must never write to the vector store")
            }
            async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
                unreachable!("judge must never write to the vector store")
            }
            async fn set_generated_summary(
                &self,
                _h: &str,
                _s: &str,
                _e: Option<Vec<f32>>,
            ) -> Result<bool> {
                unreachable!("judge must never write to the vector store")
            }
            async fn search_by_vector(
                &self,
                _e: &[f32],
                _l: usize,
                _f: Option<PayloadFilter>,
            ) -> Result<Vec<ScoredMemory>> {
                unimplemented!()
            }
            async fn search_by_tags(
                &self,
                _t: &[&str],
                _a: bool,
                _l: usize,
            ) -> Result<Vec<ScoredMemory>> {
                unimplemented!()
            }
            async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
                unimplemented!()
            }
            async fn upsert_tags(&self, _t: &[(&str, Vec<f32>)]) -> Result<()> {
                unreachable!("judge must never write to the vector store")
            }
            async fn get_all(&self, _l: usize, _o: Option<&str>) -> Result<ScrollResult> {
                unimplemented!()
            }
            async fn get_recent(
                &self,
                _l: usize,
                _s: Option<f64>,
                _t: Option<&str>,
            ) -> Result<Vec<Memory>> {
                unimplemented!()
            }
            async fn count(&self) -> Result<usize> {
                Ok(self.0.len())
            }
            async fn get_all_tags(&self) -> Result<Vec<String>> {
                Ok(vec![])
            }
            async fn increment_access_count(&self, _h: &str) -> Result<()> {
                unreachable!("judge must never write to the vector store")
            }
            async fn health(&self) -> Result<HealthStatus> {
                unimplemented!()
            }
        }

        type Recorded = Rc<RefCell<Vec<(String, String, EdgeVerdict)>>>;

        /// Records verdict writes and serves `get_all_contradictions` with the
        /// real selection semantics (newest first with pair tiebreak,
        /// SKIP/LIMIT, `exclude_resolved`) over `edges`; every other graph
        /// call is out of scope.
        struct RecordingGraph {
            verdicts: Recorded,
            /// What `set_contradiction_verdict` reports: did an edge match?
            matched: bool,
            /// (edge, resolved-in-graph)
            edges: Vec<(Contradiction, bool)>,
        }

        #[async_trait(?Send)]
        impl GraphService for RecordingGraph {
            async fn ensure_node(&self, _h: &str, _t: f64) -> Result<()> {
                unimplemented!()
            }
            async fn delete_node(&self, _h: &str) -> Result<()> {
                unimplemented!()
            }
            async fn create_typed_edge(
                &self,
                _s: &str,
                _d: &str,
                _r: UserRelationType,
                _m: EdgeMeta,
            ) -> Result<bool> {
                unimplemented!()
            }
            async fn get_typed_edges(
                &self,
                _h: &str,
                _r: Option<UserRelationType>,
                _d: Direction,
                _l: usize,
            ) -> Result<Vec<Edge>> {
                unimplemented!()
            }
            async fn delete_typed_edge(
                &self,
                _s: &str,
                _d: &str,
                _r: UserRelationType,
            ) -> Result<bool> {
                unimplemented!()
            }
            async fn create_system_edge(
                &self,
                _s: &str,
                _d: &str,
                _r: SystemRelationType,
                _t: f64,
            ) -> Result<bool> {
                unimplemented!()
            }
            async fn get_all_contradictions(
                &self,
                q: &ContradictionQuery,
            ) -> Result<Vec<Contradiction>> {
                let mut edges: Vec<&(Contradiction, bool)> = self
                    .edges
                    .iter()
                    .filter(|(_, resolved)| !(q.exclude_resolved && *resolved))
                    .collect();
                edges.sort_by(|x, y| {
                    y.0.created_at
                        .partial_cmp(&x.0.created_at)
                        .unwrap()
                        .then_with(|| x.0.memory_a_hash.cmp(&y.0.memory_a_hash))
                });
                Ok(edges
                    .into_iter()
                    .skip(q.skip)
                    .take(q.limit.clamp(1, ContradictionQuery::MAX_LIMIT))
                    .map(|(c, _)| c.clone())
                    .collect())
            }
            async fn set_contradiction_verdict(
                &self,
                s: &str,
                d: &str,
                v: &EdgeVerdict,
            ) -> Result<bool> {
                self.verdicts
                    .borrow_mut()
                    .push((s.to_string(), d.to_string(), v.clone()));
                Ok(self.matched)
            }

            async fn set_contradiction_resolution(
                &self,
                _s: &str,
                _d: &str,
                _r: Option<Resolution>,
                _v: &str,
                _t: f64,
            ) -> Result<bool> {
                Ok(self.matched)
            }
            async fn get_contradictions_for_hashes(
                &self,
                _h: &[&str],
            ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
                unimplemented!()
            }
            async fn get_neighbors(
                &self,
                _h: &str,
                _hops: u8,
                _w: f64,
                _l: usize,
            ) -> Result<Vec<Neighbor>> {
                unimplemented!()
            }
            async fn spreading_activation(
                &self,
                _s: &[&str],
                _hops: u8,
                _d: f64,
                _min: f64,
                _l: usize,
            ) -> Result<HashMap<String, f64>> {
                unimplemented!()
            }
            async fn hebbian_boosts_within(&self, _h: &[&str]) -> Result<HashMap<String, f64>> {
                unimplemented!()
            }
            async fn get_stats(&self) -> Result<GraphStats> {
                unimplemented!()
            }
        }

        /// Collects every event on the shadow-log target as a field map.
        struct Capture(Arc<Mutex<Vec<HashMap<String, String>>>>);

        impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Capture {
            fn on_event(
                &self,
                event: &tracing::Event<'_>,
                _ctx: tracing_subscriber::layer::Context<'_, S>,
            ) {
                if event.metadata().target() != SHADOW_LOG_TARGET {
                    return;
                }
                let mut fields = HashMap::new();
                fields.insert("level".to_string(), event.metadata().level().to_string());
                event.record(&mut Fields(&mut fields));
                self.0.lock().unwrap().push(fields);
            }
        }

        struct Fields<'a>(&'a mut HashMap<String, String>);

        impl tracing::field::Visit for Fields<'_> {
            fn record_debug(&mut self, f: &tracing::field::Field, v: &dyn std::fmt::Debug) {
                self.0.insert(f.name().to_string(), format!("{v:?}"));
            }
            fn record_str(&mut self, f: &tracing::field::Field, v: &str) {
                self.0.insert(f.name().to_string(), v.to_string());
            }
            fn record_f64(&mut self, f: &tracing::field::Field, v: f64) {
                self.0.insert(f.name().to_string(), v.to_string());
            }
            fn record_u64(&mut self, f: &tracing::field::Field, v: u64) {
                self.0.insert(f.name().to_string(), v.to_string());
            }
            fn record_i64(&mut self, f: &tracing::field::Field, v: i64) {
                self.0.insert(f.name().to_string(), v.to_string());
            }
        }

        async fn captured<T>(
            fut: impl std::future::Future<Output = T>,
        ) -> (T, Vec<HashMap<String, String>>) {
            let events = Arc::new(Mutex::new(Vec::new()));
            let subscriber = tracing_subscriber::registry().with(Capture(events.clone()));
            let out = fut.with_subscriber(subscriber).await;
            let collected = events.lock().unwrap().clone();
            (out, collected)
        }

        fn service(
            judge: Option<Script>,
            vectors: Vec<Memory>,
            matched: bool,
        ) -> (MemoryService, Recorded) {
            let verdicts: Recorded = Rc::new(RefCell::new(Vec::new()));
            let mut svc = MemoryService::new(
                Box::new(PairVectors(vectors)),
                Box::new(MockEmbeddings),
                Box::new(RecordingGraph {
                    verdicts: verdicts.clone(),
                    matched,
                    edges: vec![],
                }),
                Box::new(MockHebbian),
                Box::new(MockConsolidation),
                None,
            );
            if let Some(s) = judge {
                svc = svc.with_judge(Box::new(ScriptedJudge(s)));
            }
            (svc, verdicts)
        }

        fn pair() -> Vec<Memory> {
            vec![mem(&src(), 1_000.0), mem(&dst(), 2_000.0)]
        }

        #[tokio::test(flavor = "current_thread")]
        async fn every_verdict_class_persists_on_the_edge_and_emits_one_shadow_event() {
            for (verdict, survivor) in [
                (Verdict::Supersession, Some(Survivor::B)),
                (Verdict::Contradiction, Some(Survivor::A)),
                (Verdict::Coexist, None),
                (Verdict::Unrelated, None),
            ] {
                let (svc, verdicts) =
                    service(Some(Script::Ok(judgement(verdict, survivor))), pair(), true);
                let (outcome, events) = captured(svc.judge_contradiction(&src(), &dst())).await;
                assert!(
                    matches!(
                        outcome,
                        JudgeOutcome::Judged { ref judgement, persisted: true }
                            if judgement.verdict == verdict
                    ),
                    "{verdict:?}: {outcome:?}"
                );

                // Persisted through the graph trait; survivor resolved to a hash.
                let recorded = verdicts.borrow();
                assert_eq!(recorded.len(), 1, "{verdict:?}");
                let (s, d, ev) = &recorded[0];
                assert_eq!((s.as_str(), d.as_str()), (src().as_str(), dst().as_str()));
                assert_eq!(ev.verdict, verdict);
                let expected_survivor = match survivor {
                    Some(Survivor::A) => Some(src()),
                    Some(Survivor::B) => Some(dst()),
                    None => None,
                };
                assert_eq!(ev.verdict_survivor, expected_survivor, "{verdict:?}");
                assert_eq!(ev.verdict_model, "test-model");
                assert_eq!(ev.verdict_reason, "because");

                // Exactly one shadow-log event carrying the contract fields.
                assert_eq!(events.len(), 1, "{verdict:?}: {events:?}");
                let e = &events[0];
                assert_eq!(e["level"], "INFO");
                assert_eq!(e["verdict"], verdict.as_str());
                assert_eq!(e["memory_a"], src());
                assert_eq!(e["memory_b"], dst());
                assert_eq!(e["model"], "test-model");
                assert_eq!(e["persisted"], "true");
                assert_eq!(e["confidence"], "0.9");
                let expected_would = if verdict == Verdict::Supersession {
                    format!("{} -> {}", src(), dst())
                } else {
                    "none".to_string()
                };
                assert_eq!(e["would_supersede"], expected_would, "{verdict:?}");
                assert_eq!(
                    e["survivor"],
                    expected_survivor.unwrap_or_else(|| "none".into())
                );
            }
        }

        /// AC-5 (amended): a deterministic failure is persisted as an
        /// `unjudged` marker carrying the error class, so the backfill's
        /// NULL filter never re-matches the pair; no shadow event.
        #[tokio::test(flavor = "current_thread")]
        async fn deterministic_failure_marks_the_edge_unjudged_and_emits_no_shadow_event() {
            let (svc, verdicts) = service(
                Some(Script::Err(|| {
                    AlayaError::Judge(
                        "verdict is not valid JSON\n(stop_reason=\"max_tokens\")".into(),
                    )
                })),
                pair(),
                true,
            );
            let (outcome, events) = captured(svc.judge_contradiction(&src(), &dst())).await;
            assert!(
                matches!(outcome, JudgeOutcome::Unjudged { marked: true }),
                "{outcome:?}"
            );
            let recorded = verdicts.borrow();
            assert_eq!(recorded.len(), 1, "the marker is the only graph write");
            let (s, d, marker) = &recorded[0];
            assert_eq!((s.as_str(), d.as_str()), (src().as_str(), dst().as_str()));
            assert_eq!(marker.verdict, Verdict::Unjudged);
            assert_eq!(marker.verdict_survivor, None);
            assert_eq!(marker.verdict_model, "test-model");
            assert!(
                marker
                    .verdict_reason
                    .starts_with("unjudged: contradiction judge error: verdict is not valid JSON"),
                "{}",
                marker.verdict_reason
            );
            assert!(
                !marker.verdict_reason.contains('\n'),
                "control chars stripped"
            );
            assert!(
                events.is_empty(),
                "the shadow-log target carries judged pairs only: {events:?}"
            );
        }

        /// A transient failure (upstream down, 5xx, timeout) writes nothing so
        /// the pair is retried by the next pass.
        #[tokio::test(flavor = "current_thread")]
        async fn transient_failure_is_unjudged_and_writes_nothing() {
            let (svc, verdicts) = service(
                Some(Script::Err(|| {
                    AlayaError::Unavailable("502 from the LB".into())
                })),
                pair(),
                true,
            );
            let (outcome, events) = captured(svc.judge_contradiction(&src(), &dst())).await;
            assert!(
                matches!(outcome, JudgeOutcome::Unjudged { marked: false }),
                "{outcome:?}"
            );
            assert!(
                verdicts.borrow().is_empty(),
                "no graph write on a transient failure"
            );
            assert!(events.is_empty(), "{events:?}");
        }

        fn hash(i: usize) -> String {
            format!("{i:064x}")
        }

        /// AC-6 (amended): with more resolved pairs at the top of the queue
        /// than one page holds, `limit = N` still returns N unresolved pairs
        /// and `next_offset` walks the rest. A pair Qdrant alone knows is
        /// resolved is dropped from its page without disturbing the cursor.
        #[tokio::test(flavor = "current_thread")]
        async fn resolved_run_at_the_top_cannot_starve_the_queue() {
            // 60 pairs, newest first by created_at; the newest 50 are resolved
            // graph-side (SUPERSEDES). Pair #57 is superseded in Qdrant only.
            let mut edges = Vec::new();
            let mut memories = Vec::new();
            for i in 0..60usize {
                let (a, b) = (hash(1000 + i), hash(2000 + i));
                let ts = 100_000.0 - i as f64; // i = 0 is the newest
                // Pair #0 is resolved by a keep_both stamp (LAB-3885); the
                // rest of the resolved run by SUPERSEDES. Both leave the
                // default queue; both show under include_resolved.
                let kept = i == 0;
                edges.push((
                    Contradiction {
                        memory_a_hash: a.clone(),
                        memory_b_hash: b.clone(),
                        confidence: Some(0.7),
                        created_at: Some(ts),
                        verdict: None,
                        resolution: kept.then_some(Resolution::KeepBoth),
                        resolved_at: kept.then_some(77.0),
                        resolved_via: kept.then(|| "operator:console".to_string()),
                    },
                    i < 50,
                ));
                let mut ma = mem(&a, ts);
                if i == 57 {
                    ma.metadata = Some(HashMap::from([(
                        "superseded_by".to_string(),
                        serde_json::json!(hash(9)),
                    )]));
                }
                memories.push(ma);
                memories.push(mem(&b, ts));
            }
            let verdicts: Recorded = Rc::new(RefCell::new(Vec::new()));
            let svc = MemoryService::new(
                Box::new(PairVectors(memories)),
                Box::new(MockEmbeddings),
                Box::new(RecordingGraph {
                    verdicts: verdicts.clone(),
                    matched: true,
                    edges,
                }),
                Box::new(MockHebbian),
                Box::new(MockConsolidation),
                None,
            );
            let a_hashes = |page: &Value| -> Vec<String> {
                page["pairs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["memory_a_hash"].as_str().unwrap().to_string())
                    .collect()
            };

            // Page 1: five unresolved pairs despite 50 resolved ones on top.
            let page = svc.memory_contradictions(5, 0, false, None).await.unwrap();
            assert_eq!(
                a_hashes(&page),
                [hash(1050), hash(1051), hash(1052), hash(1053), hash(1054)]
            );
            assert!(
                page["pairs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|p| p["verdict"] == "unjudged")
            );
            assert_eq!(page["next_offset"], 5);

            // Page 2: #57 is dropped by the Qdrant guard; the cursor still
            // advances by the full graph page.
            let page2 = svc.memory_contradictions(5, 5, false, None).await.unwrap();
            assert_eq!(
                a_hashes(&page2),
                [hash(1055), hash(1056), hash(1058), hash(1059)]
            );
            assert_eq!(page2["next_offset"], 10);

            // Page 3: nothing left, cursor ends.
            let page3 = svc.memory_contradictions(5, 10, false, None).await.unwrap();
            assert_eq!(page3["total"], 0);
            assert_eq!(page3["next_offset"], Value::Null);

            // include_resolved shows the resolved run again, each row
            // carrying its resolution stamp (or null).
            let all = svc.memory_contradictions(5, 0, true, None).await.unwrap();
            assert_eq!(all["pairs"][0]["memory_a_hash"], hash(1000));
            assert_eq!(all["pairs"][0]["memory_a_superseded"], false);
            assert_eq!(all["pairs"][0]["resolution"], "keep_both");
            assert_eq!(all["pairs"][0]["resolved_at"], 77.0);
            assert_eq!(all["pairs"][0]["resolved_via"], "operator:console");
            assert_eq!(all["pairs"][1]["resolution"], Value::Null);
            assert_eq!(all["pairs"][1]["resolved_via"], Value::Null);

            assert!(
                verdicts.borrow().is_empty(),
                "the read surface never writes"
            );
        }

        // ─── resolve_contradiction (LAB-3885) ────────────────────────────

        fn resolver(matched: bool) -> MemoryService {
            MemoryService::with_clock(
                Box::new(PairVectors(vec![])),
                Box::new(MockEmbeddings),
                Box::new(RecordingGraph {
                    verdicts: Rc::new(RefCell::new(Vec::new())),
                    matched,
                    edges: vec![],
                }),
                Box::new(MockHebbian),
                Box::new(MockConsolidation),
                || 4_242.0,
            )
        }

        #[tokio::test(flavor = "current_thread")]
        async fn resolve_contradiction_stamps_with_server_clock_and_verbatim_via() {
            let svc = resolver(true);
            let r = svc
                .resolve_contradiction(
                    &src(),
                    &dst(),
                    Some(Resolution::KeepBoth),
                    "  operator:console ",
                )
                .await
                .unwrap();
            assert_eq!(r["success"], true);
            assert_eq!(r["memory_a_hash"], src());
            assert_eq!(r["memory_b_hash"], dst());
            assert_eq!(r["resolution"], "keep_both");
            assert_eq!(r["resolved_at"], 4_242.0, "resolved_at is server-set");
            assert_eq!(
                r["resolved_via"], "operator:console",
                "trimmed, otherwise verbatim"
            );

            // Clear reports the cleared state, not a stale stamp.
            let r = svc
                .resolve_contradiction(&src(), &dst(), None, "operator:mcp")
                .await
                .unwrap();
            assert_eq!(r["success"], true);
            assert_eq!(r["resolution"], Value::Null);
            assert_eq!(r["resolved_at"], Value::Null);
            assert_eq!(r["resolved_via"], Value::Null);
        }

        #[tokio::test(flavor = "current_thread")]
        async fn resolve_contradiction_is_an_error_when_no_edge_matches() {
            let e = resolver(false)
                .resolve_contradiction(&src(), &dst(), Some(Resolution::KeepBoth), "operator:mcp")
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::NotFound(_)), "{e}");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn resolve_contradiction_validates_hashes_and_via() {
            let svc = resolver(true);
            for (a, b, via) in [
                ("short", dst().as_str(), "operator:mcp"),
                (src().as_str(), "SHORT", "operator:mcp"),
                (src().as_str(), src().as_str(), "operator:mcp"),
                (src().as_str(), dst().as_str(), ""),
                (src().as_str(), dst().as_str(), "   "),
            ] {
                let e = svc
                    .resolve_contradiction(a, b, Some(Resolution::KeepBoth), via)
                    .await
                    .unwrap_err();
                assert!(
                    matches!(e, AlayaError::Validation(_)),
                    "{a} {b} {via:?}: {e}"
                );
            }
            let long = "x".repeat(MAX_RESOLVED_VIA_LEN + 1);
            let e = svc
                .resolve_contradiction(&src(), &dst(), Some(Resolution::KeepBoth), &long)
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::Validation(_)), "{e}");
        }

        #[tokio::test(flavor = "current_thread")]
        async fn unknown_verdict_filter_is_a_validation_error() {
            let (svc, _) = service(None, vec![], true);
            for bad in [
                vec![],
                vec!["Supersession".to_string()],
                vec!["foo".to_string()],
            ] {
                let e = svc
                    .memory_contradictions(5, 0, false, Some(&bad))
                    .await
                    .unwrap_err();
                assert!(matches!(e, AlayaError::Validation(_)), "{bad:?}: {e:?}");
            }
        }

        #[tokio::test(flavor = "current_thread")]
        async fn rate_limit_is_surfaced_for_the_caller_to_back_off() {
            let (svc, verdicts) = service(
                Some(Script::Err(|| AlayaError::RateLimited {
                    retry_after_secs: Some(3),
                })),
                pair(),
                true,
            );
            let outcome = svc.judge_contradiction(&src(), &dst()).await;
            assert!(
                matches!(
                    outcome,
                    JudgeOutcome::RateLimited {
                        retry_after_secs: Some(3)
                    }
                ),
                "{outcome:?}"
            );
            assert!(verdicts.borrow().is_empty());
        }

        #[tokio::test(flavor = "current_thread")]
        async fn missing_endpoint_is_marked_bad_hash_or_no_judge_writes_nothing() {
            // Endpoint missing from the vector store: deterministic until the
            // memory reappears, so it is marked rather than re-selected forever.
            let (svc, verdicts) = service(
                Some(Script::Ok(judgement(Verdict::Coexist, None))),
                vec![mem(&src(), 1.0)],
                true,
            );
            assert!(matches!(
                svc.judge_contradiction(&src(), &dst()).await,
                JudgeOutcome::Unjudged { marked: true }
            ));
            {
                let recorded = verdicts.borrow();
                assert_eq!(recorded.len(), 1);
                assert_eq!(recorded[0].2.verdict, Verdict::Unjudged);
                assert_eq!(
                    recorded[0].2.verdict_reason,
                    "unjudged: endpoint missing from vector store"
                );
            }
            // Malformed hash and self-pair never reach the judge or the graph.
            assert!(matches!(
                svc.judge_contradiction("nope", &dst()).await,
                JudgeOutcome::Unjudged { marked: false }
            ));
            assert!(matches!(
                svc.judge_contradiction(&src(), &src()).await,
                JudgeOutcome::Unjudged { marked: false }
            ));
            assert_eq!(verdicts.borrow().len(), 1);
            // Judge not configured.
            let (svc, verdicts) = service(None, pair(), true);
            assert!(matches!(
                svc.judge_contradiction(&src(), &dst()).await,
                JudgeOutcome::Unjudged { marked: false }
            ));
            assert!(verdicts.borrow().is_empty());
        }

        #[tokio::test(flavor = "current_thread")]
        async fn unmatched_edge_still_emits_the_shadow_event() {
            let (svc, verdicts) = service(
                Some(Script::Ok(judgement(
                    Verdict::Supersession,
                    Some(Survivor::B),
                ))),
                pair(),
                false,
            );
            let (outcome, events) = captured(svc.judge_contradiction(&src(), &dst())).await;
            assert!(matches!(
                outcome,
                JudgeOutcome::Judged {
                    persisted: false,
                    ..
                }
            ));
            assert_eq!(verdicts.borrow().len(), 1, "the write was attempted");
            assert_eq!(
                events.len(),
                1,
                "shadow log does not depend on the graph write landing"
            );
            assert_eq!(events[0]["persisted"], "false");
        }

        /// AC-4: the store path never awaits the judge. With a judge that
        /// hangs forever, store completes and returns the same shape as with
        /// no judge configured.
        #[tokio::test(flavor = "current_thread")]
        async fn store_path_never_awaits_the_judge_and_response_shape_is_unchanged() {
            let mk = |judge: Option<Script>| {
                let mut svc = MemoryService::new(
                    Box::new(MockVectorsPersisting {
                        stored: Rc::new(RefCell::new(HashMap::new())),
                        raw_only: Default::default(),
                    }),
                    Box::new(MockEmbeddings),
                    Box::new(MockGraph),
                    Box::new(MockHebbian),
                    Box::new(MockConsolidation),
                    None,
                );
                if let Some(s) = judge {
                    svc = svc.with_judge(Box::new(ScriptedJudge(s)));
                }
                svc
            };
            let params = || StoreParams {
                content: "judge me".into(),
                tags: None,
                memory_type: None,
                metadata: None,
                client_hostname: None,
                summary: None,
                dedup_threshold: None,
            };

            let with_judge = tokio::time::timeout(
                std::time::Duration::from_secs(5),
                mk(Some(Script::Hang)).store_memory(params()),
            )
            .await
            .expect("store must not wait on the judge")
            .expect("store ok");
            let without = mk(None).store_memory(params()).await.expect("store ok");

            let keys = |m: &HashMap<String, Value>| {
                let mut k: Vec<_> = m.keys().cloned().collect();
                k.sort();
                k
            };
            assert_eq!(keys(&with_judge), keys(&without));
            assert_eq!(with_judge["content_hash"], without["content_hash"]);
        }
    }
}
