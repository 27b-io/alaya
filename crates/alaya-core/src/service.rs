//! MemoryService — orchestrates all 9 MCP tools across backends.
//!
//! This is the core business logic layer. Each public method corresponds to
//! one MCP tool. All backend calls go through trait abstractions.

use std::cell::{Cell, RefCell};
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
    ReversalOutcome, ReversalRecord, StoreMode, SummaryProvider, VectorStorage,
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

/// Upper bound on `resolved_via` (LAB-3885) and `unsuperseded_via`: a short
/// principal tag, not a free-text reason.
const MAX_VIA_LEN: usize = 128;

/// Cap on a `memory_unsupersede` reason: it is stored on the memory for good.
pub const MAX_UNSUPERSEDE_REASON_LEN: usize = 2000;

/// `resolved_via` stamped on the CONTRADICTS pair of a reversed supersession
/// (LAB-6876), so an automatic apply of the judge's verdict can tell the pair
/// is settled and never re-applies it.
pub const UNSUPERSEDE_RESOLVED_VIA: &str = "unsupersede";

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
    /// No verdict. `marked` = the failure was persisted as `verdict =
    /// unjudged` with its cause as reason, so the backfill's NULL filter
    /// skips the pair instead of re-selecting it forever: a deterministic
    /// judge error (schema / parse / empty answer / request fault) or an
    /// endpoint missing from the vector store. `marked = false` = nothing
    /// written, the pair is retried later: transient (judge disabled, fetch
    /// failed, upstream unavailable) or a marker write that did not land.
    /// `spent` = a request reached the judge, so tokens may have been
    /// billed. Independent of `marked`: a missing endpoint is marked but
    /// free; a malformed verdict is marked and paid for. Conservative on
    /// purpose — a request the API rejected (400), or one that timed out
    /// after leaving the box, still counts: a spend ceiling must err high.
    Unjudged { marked: bool, spent: bool },
    /// Upstream 429. Nothing was written; the caller owns any backoff.
    RateLimited { retry_after_secs: Option<u64> },
}

impl JudgeOutcome {
    /// Whether tokens may have been billed: a spend budget charges exactly these.
    pub fn spent(&self) -> bool {
        match self {
            Self::Judged { .. } => true,
            Self::Unjudged { spent, .. } => *spent,
            Self::RateLimited { .. } => false,
        }
    }
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
        // supersede, which read-only is denied. Read-only stores are
        // insert-only: an existing record, judged on the raw point rather than
        // parseability, is reported as created=false by the write itself.
        let mode = if read_only {
            StoreMode::InsertOnly
        } else {
            StoreMode::Upsert
        };

        // Store in vector DB. `created == false` means a point with this
        // content_hash already existed: the backend carried its created_at,
        // access history and supersession marker over (see
        // VectorStorage::store), while the caller-supplied fields above — and
        // provenance — replaced the stored ones. Who owns provenance when two
        // callers store the same content is LAB-1084's decision; today's
        // replace semantics stand until then.
        let (created, _) = self.vectors.store(&memory, mode).await?;
        if read_only && !created {
            return Err(AlayaError::Validation(format!(
                "memory {content_hash} already exists; read-only principals may only add new memories"
            )));
        }
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
        let stages = StageClock::start();
        let result = self.search_hybrid_timed(params, read_only, &stages).await;
        stages.finish(result.is_ok());
        result
    }

    /// The body of [`Self::search_hybrid`], which owns `stages` so that a
    /// failed or dropped search still logs its stage timings.
    async fn search_hybrid_timed(
        &self,
        params: &SearchParams,
        read_only: bool,
        stages: &StageClock,
    ) -> Result<Value> {
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
            let _stage = stages.stage(Stage::FanOut);
            let query_texts = [params.query.as_str()];
            let embed_fut = stages.time(
                Stage::Embed,
                self.embeddings.embed_batch(&query_texts, PromptName::Query),
            );

            let tag_search_fut = async {
                let _span = tracing::info_span!("get_all_tags").entered();
                let all_tags = stages
                    .time(Stage::Tags, async {
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
                    })
                    .await;
                drop(_span);

                let tag_set: std::collections::HashSet<String> = all_tags.into_iter().collect();
                let keywords = hybrid_search::extract_query_keywords(&params.query, Some(&tag_set));
                let n_keywords = keywords.len();

                let results = if keywords.is_empty() {
                    Vec::new()
                } else {
                    let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
                    let search = self
                        .vectors
                        .search_by_tags(&keyword_refs, false, fetch_size);
                    stages
                        .time(Stage::TagSearch, search)
                        .await
                        .unwrap_or_default()
                };
                (results, n_keywords)
            };

            let count_fut = stages.time(Stage::Count, self.vectors.count());

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
            let _stage = stages.stage(Stage::VectorSearch);
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
            let _stage = stages.stage(Stage::RrfFuse);
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
            let _stage = stages.stage(Stage::GraphBoost);
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
            let _stage = stages.stage(Stage::GraphInject);
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
            let _stage = stages.stage(Stage::Rerank);
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
                with_budget(budget, reranker.rerank(&params.query, &candidate_contents)).await;
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
            let _stage = stages.stage(Stage::Enrich);
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
        // The supersession marker changes only through supersede / merge and
        // unsupersede, which keep its edge, reason and audit entry. A patch
        // that sets or deletes it (null deletes) would keep none of them.
        if patch
            .metadata
            .as_ref()
            .is_some_and(|m| m.contains_key("superseded_by"))
        {
            return Err(AlayaError::Validation(
                "metadata.superseded_by is server-maintained: use memory_supersede / \
                 memory_unsupersede"
                    .into(),
            ));
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

        // The embedding probe gets its own 5s budget: the client's 60s embed
        // timeout would let an endpoint that accepts TCP and never answers
        // park this single-threaded worker — and every read queued behind
        // it — for a minute per health call.
        let (graph, embedding) = futures::join!(
            self.graph.get_stats(),
            with_budget(std::time::Duration::from_secs(5), self.embeddings.health())
        );
        let embedding = embedding.unwrap_or_else(|| {
            Err(alaya_types::AlayaError::Embedding(
                "health probe timed out".into(),
            ))
        });

        let graph_health = match graph {
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

        // LAB-4025: a dead embedding endpoint fails every store and every
        // semantic search, so it degrades `status` exactly as the vector
        // store does. Reads and tag-mode search still work — `degraded`,
        // not `unhealthy`. Graph stays informational: its calls are
        // non-fatal by design, so a graph blip degrades nothing.
        let embedding_ok = embedding.is_ok();
        let embedding_health = match embedding {
            Ok(h) => serde_json::to_value(h).unwrap_or_default(),
            Err(e) => {
                // The response is sanitised; the pod log keeps the reason
                // (refused vs 503 vs timed out) an operator actually needs.
                tracing::warn!(error = %e, "embedding health probe failed");
                serde_json::json!({
                    "status": "unhealthy",
                    "error": e.safe_message(),
                })
            }
        };

        let vector_ok = vector_health.status == "green" || vector_health.status == "ok";
        let mut result = HashMap::new();
        result.insert(
            "status".into(),
            serde_json::json!(if vector_ok && embedding_ok {
                "healthy"
            } else {
                "degraded"
            }),
        );
        result.insert("backend".into(), serde_json::json!("qdrant"));
        result.insert(
            "vector_health".into(),
            serde_json::to_value(&vector_health).unwrap_or_default(),
        );
        result.insert("graph_health".into(), graph_health);
        result.insert("embedding_health".into(), embedding_health);
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

        self.mark_superseded(&[old_hash], new_hash, reason)
            .await
            .map_err(|f| f.error)?;

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
    ///
    /// The update is atomic per memory, not per batch, so a failure can leave
    /// some memories marked (alaya#130). Those still get their edge — the
    /// marked set is re-read, so a write that landed without being confirmed
    /// counts too — and are returned with the error, so the caller reports
    /// them as superseded.
    ///
    /// When that re-read fails as well, which memories carry the marker is
    /// unknown. They are returned as in doubt — never as untouched — and get
    /// no edge: an edge on a memory whose marker did not land would claim a
    /// supersession search does not apply. Nothing is lost either way. The
    /// marker is the durable record and the edge is derived from it, so
    /// retrying the same call converges (re-marking writes the same marker,
    /// edge creation is a MERGE), and `scripts/backfill_graph.py` rebuilds
    /// every SUPERSEDES edge from the markers.
    async fn mark_superseded(
        &self,
        old_hashes: &[&str],
        new_hash: &str,
        reason: &str,
    ) -> std::result::Result<(), SupersedeFailure> {
        let mut extra = HashMap::new();
        extra.insert("supersession_reason".into(), serde_json::json!(reason));
        let updated = self
            .vectors
            .update_metadata_batch(
                old_hashes,
                MetadataUpdate {
                    superseded_by: Some(new_hash.to_string()),
                    extra: Some(extra),
                    ..Default::default()
                },
            )
            .await;
        if let Err(error) = updated {
            let memories = match self.vectors.get_batch(old_hashes).await {
                Ok(memories) => memories,
                Err(read) => {
                    let in_doubt: Vec<String> =
                        old_hashes.iter().map(|h| (*h).to_string()).collect();
                    tracing::error!(
                        new_hash,
                        in_doubt = ?in_doubt,
                        error = %error,
                        read_error = %read,
                        "supersession outcome unknown: markers may have landed without \
                         SUPERSEDES edges; retry the call (it is idempotent) or run \
                         scripts/backfill_graph.py"
                    );
                    return Err(SupersedeFailure {
                        marked: Vec::new(),
                        in_doubt,
                        error,
                    });
                }
            };
            let marked: Vec<String> = memories
                .into_iter()
                .filter(|m| superseded_by(m) == Some(new_hash))
                .map(|m| m.content_hash)
                .collect();
            let refs: Vec<&str> = marked.iter().map(String::as_str).collect();
            self.write_supersedes_edges(&refs, new_hash).await;
            return Err(SupersedeFailure {
                marked,
                in_doubt: Vec::new(),
                error,
            });
        }

        self.write_supersedes_edges(old_hashes, new_hash).await;
        Ok(())
    }

    /// One batched write of `new_hash -> old` SUPERSEDES edges; non-fatal.
    async fn write_supersedes_edges(&self, old_hashes: &[&str], new_hash: &str) {
        if old_hashes.is_empty() {
            return;
        }
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
    }

    // ─── Tool: memory_unsupersede (LAB-6876) ────────────────────────────

    /// Reverse a supersession: `content_hash` returns to every search path,
    /// and the reversal is recorded on the memory (`supersession_log`).
    ///
    /// Three writes, ordered so a retry converges — the marker is what every
    /// read path consults, so it goes last:
    ///
    /// 1. the CONTRADICTS pair between the memory and the survivor its marker
    ///    names is stamped `keep_both` (either direction), `resolved_via =
    ///    unsupersede`, so a later judge run cannot re-apply the reversed
    ///    supersession;
    /// 2. every SUPERSEDES edge into the memory is deleted. Normally that is
    ///    the one from the survivor; an older one exists only when the memory
    ///    was superseded again without a reversal, and was already stale.
    ///    Edges out of the memory stay, so a chain is reversed one link at a
    ///    time: in A→B→C, unsuperseding B leaves A superseded by B, and
    ///    unsuperseding A leaves B superseded by C;
    /// 3. the marker and `supersession_reason` are removed and the audit entry
    ///    appended, in one write that applies only while the marker is still
    ///    the one read before step 1 (`VectorStorage::reverse_supersession`).
    ///
    /// A graph failure in step 1 or 2 is an error before the marker is
    /// touched: the memory stays superseded and the same call can be
    /// retried. A marker that moved to another survivor meanwhile is left
    /// whole, its edge restored, and reported as `superseded_by_changed`. A
    /// memory that is not superseded is `not_superseded`: a typed no-op,
    /// never a success. Nothing is deleted but SUPERSEDES edges.
    #[tracing::instrument(skip(self))]
    pub async fn memory_unsupersede(
        &self,
        content_hash: &str,
        reason: &str,
        via: &str,
    ) -> Result<Value> {
        if !alaya_types::memory::validate_content_hash(content_hash) {
            return Err(AlayaError::Validation(
                "invalid content_hash: expected 64-char lowercase SHA-256 hex".into(),
            ));
        }
        let reason = reason.trim();
        // Characters, not bytes: the MCP schema's maxLength counts characters.
        if reason.is_empty() || reason.chars().count() > MAX_UNSUPERSEDE_REASON_LEN {
            return Err(AlayaError::Validation(format!(
                "reason is required (1..={MAX_UNSUPERSEDE_REASON_LEN} chars): it is the audit record"
            )));
        }
        let via = require_via("unsuperseded_via", via)?;

        let memory = self
            .vectors
            .get_by_hash(content_hash)
            .await?
            .ok_or_else(|| AlayaError::NotFound(format!("memory {content_hash} not found")))?;
        let Some(marker) = supersession_marker(&memory).cloned() else {
            return Ok(not_superseded(content_hash));
        };

        let now = (self.clock)();
        let mut stamped: Vec<[String; 2]> = Vec::new();
        if let Some(survivor) = survivor_of(&marker, content_hash) {
            for (a, b) in [(content_hash, survivor), (survivor, content_hash)] {
                if self
                    .graph
                    .settle_contradiction(a, b, UNSUPERSEDE_RESOLVED_VIA, now)
                    .await?
                {
                    stamped.push([a.to_string(), b.to_string()]);
                }
            }
        }
        let mut edges_removed = self
            .graph
            .delete_incoming_system_edges(content_hash, SystemRelationType::Supersedes)
            .await?;

        let reversal = ReversalRecord {
            at: now,
            via: via.to_string(),
            reason: reason.to_string(),
        };
        match self
            .vectors
            .reverse_supersession(content_hash, &marker, &reversal)
            .await?
        {
            ReversalOutcome::Cleared {
                supersession_reason,
            } => {
                // With the marker gone every incoming SUPERSEDES edge is
                // stale, including one a concurrent supersede to the same
                // survivor wrote after step 2. A stale edge keeps a live
                // memory's pairs out of the queue and nothing repairs it; a
                // missing one is covered by the marker. So sweep again, and
                // only warn: the reversal itself has landed.
                match self
                    .graph
                    .delete_incoming_system_edges(content_hash, SystemRelationType::Supersedes)
                    .await
                {
                    Ok(late) => edges_removed.extend(late),
                    Err(e) => tracing::warn!(
                        hash = content_hash,
                        error = %e,
                        "unsupersede: second SUPERSEDES sweep failed; a stale edge may remain"
                    ),
                }
                // That sweep can also take the edge of a supersede that landed
                // after the reversal. Its marker is written before its edge,
                // so any such edge the sweep could reach has a marker this
                // read sees: put its edge back.
                let now_superseded_by = match self.vectors.get_by_hash(content_hash).await {
                    Ok(m) => m.as_ref().and_then(supersession_marker).cloned(),
                    Err(e) => {
                        tracing::warn!(
                            hash = content_hash,
                            error = %e,
                            "unsupersede: re-read after the sweep failed; a new supersession \
                             may have lost its edge (scripts/backfill_graph.py restores it)"
                        );
                        None
                    }
                };
                if let Some(s) = now_superseded_by
                    .as_ref()
                    .and_then(|m| survivor_of(m, content_hash))
                {
                    self.write_supersedes_edges(&[content_hash], s).await;
                }
                tracing::info!(
                    target: "alaya::supersession",
                    hash = content_hash,
                    superseded_by = %marker,
                    via,
                    reason,
                    edges_removed = ?edges_removed,
                    stamped = ?stamped,
                    "supersession reversed"
                );
                Ok(serde_json::json!({
                    "success": true,
                    "status": "unsuperseded",
                    "content_hash": content_hash,
                    "superseded_by": marker,
                    "supersession_reason": supersession_reason,
                    "reason": reason,
                    "unsuperseded_via": via,
                    "unsuperseded_at": now,
                    "supersedes_edges_removed": edges_removed,
                    "contradictions_stamped": stamped,
                    "now_superseded_by": now_superseded_by,
                }))
            }
            ReversalOutcome::NotSuperseded => Ok(not_superseded(content_hash)),
            ReversalOutcome::SupersededByOther(current) => {
                // Nothing was reversed, so undo this call's graph writes: each
                // stamp it made, only while that stamp is still its own (an
                // operator may have resolved the pair since), and the current
                // survivor's edge.
                for [a, b] in &stamped {
                    if let Err(e) = self
                        .graph
                        .unsettle_contradiction(a, b, UNSUPERSEDE_RESOLVED_VIA, now)
                        .await
                    {
                        tracing::warn!(a, b, error = %e, "unsupersede: could not clear its stamp");
                    }
                }
                if let Some(s) = survivor_of(&current, content_hash) {
                    self.write_supersedes_edges(&[content_hash], s).await;
                }
                Ok(serde_json::json!({
                    "success": false,
                    "status": "superseded_by_changed",
                    "content_hash": content_hash,
                    "superseded_by": current,
                    "error": "Superseded again while being restored; nothing reversed. \
                              Inspect the memory and retry if still intended",
                }))
            }
        }
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
            return JudgeOutcome::Unjudged {
                marked: false,
                spent: false,
            };
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
            return JudgeOutcome::Unjudged {
                marked: false,
                spent: false,
            };
        }
        let (sa, sd) = (&src[..8], &dst[..8]);

        let batch = match self.vectors.get_batch(&[src, dst]).await {
            Ok(b) => b,
            Err(e) => {
                tracing::warn!(a = sa, b = sd, "judge_contradiction: fetch failed: {e}");
                return JudgeOutcome::Unjudged {
                    marked: false,
                    spent: false,
                };
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
            return JudgeOutcome::Unjudged {
                marked,
                spent: false,
            };
        };

        let j = match judge.judge(a, b).await {
            Ok(j) => j,
            Err(AlayaError::RateLimited { retry_after_secs }) => {
                tracing::warn!(a = sa, b = sd, retry_after_secs, "judge rate limited");
                return JudgeOutcome::RateLimited { retry_after_secs };
            }
            Err(AlayaError::Unavailable { message, spent }) => {
                tracing::warn!(
                    memory_a = src,
                    memory_b = dst,
                    verdict = Verdict::UNJUDGED,
                    error = ?message,
                    spent,
                    "contradiction unjudged (transient; will retry)"
                );
                return JudgeOutcome::Unjudged {
                    marked: false,
                    spent,
                };
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
                return JudgeOutcome::Unjudged {
                    marked,
                    spent: true,
                };
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
    /// stamped pairs and `include_resolved` still shows them. This and
    /// `memory_unsupersede` (which only stamps an unresolved pair, as
    /// `resolved_via = unsupersede`) are the only writers of
    /// `e.resolution*`: `relation` cannot set it, and the judge writes the
    /// verdict namespace only. Unlike `persist_verdict`
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
        let via = require_via("resolved_via", resolved_via)?;

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
                Err(SupersedeFailure {
                    marked,
                    in_doubt,
                    error,
                }) => {
                    for &dup_hash in &to_supersede {
                        if marked.iter().any(|m| m == dup_hash) {
                            superseded.push(dup_hash.to_string());
                            continue;
                        }
                        let msg = if in_doubt.iter().any(|m| m == dup_hash) {
                            SUPERSEDE_OUTCOME_UNKNOWN
                        } else {
                            error.safe_message()
                        };
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

/// Runs `fut` under `budget`; `None` means the budget expired. Callers: the
/// rerank pass ([`RerankingService::timeout`]) and the embedding health probe.
///
/// Native builds (alaya-server, production) bound it with
/// `tokio::time::timeout` and nothing else: it polls `fut` before its own
/// deadline, so a response that has already arrived is used even on a late
/// poll, and dropping `fut` on elapse aborts the request. That makes the two
/// outcomes exact — `None` is "nothing arrived within the budget", `Some(Err)`
/// is a real transport or HTTP error with its detail intact. wasm32
/// (`alaya-worker`, deferred) has no tokio timer and awaits directly — there
/// the per-request reqwest timeout in the backend client is the bound and
/// surfaces as `Some(Err)`.
#[cfg(not(target_arch = "wasm32"))]
async fn with_budget<T>(
    budget: std::time::Duration,
    fut: impl std::future::Future<Output = T>,
) -> Option<T> {
    tokio::time::timeout(budget, fut).await.ok()
}

#[cfg(target_arch = "wasm32")]
async fn with_budget<T>(
    _budget: std::time::Duration,
    fut: impl std::future::Future<Output = T>,
) -> Option<T> {
    Some(fut.await)
}

/// A hybrid-search stage timed by [`StageClock`]. `Embed`, `Tags`,
/// `TagSearch` and `Count` are the concurrent branches of `FanOut`.
#[derive(Clone, Copy, PartialEq)]
enum Stage {
    FanOut,
    Embed,
    Tags,
    TagSearch,
    Count,
    VectorSearch,
    RrfFuse,
    GraphBoost,
    GraphInject,
    Rerank,
    Enrich,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::FanOut => "fan_out",
            Stage::Embed => "embed",
            Stage::Tags => "tags",
            Stage::TagSearch => "tag_search",
            Stage::Count => "count",
            Stage::VectorSearch => "vector_search",
            Stage::RrfFuse => "rrf_fuse",
            Stage::GraphBoost => "graph_boost",
            Stage::GraphInject => "graph_inject",
            Stage::Rerank => "rerank",
            Stage::Enrich => "enrich",
        }
    }
}

/// Logs one `hybrid stages` line per hybrid search with each stage's wall
/// time, so a slow stage can be found from logs alone — including a search
/// dropped at the worker's deadline, which is why the line comes from `Drop`.
///
/// `outcome` is `ok` or `error` (set by [`Self::finish`] from the result), or
/// `dropped` when the future was dropped (or unwound) before finishing. A
/// dropped search logs `stage` as the stage still running, whose `_ms` is its
/// elapsed-so-far. An error logs `stage` as the last stage entered, the one
/// whose result failed, whose `_ms` is complete; a validation error before any
/// stage has no `stage`. A stage that did not run has no `_ms`. A fan-out
/// branch only gets one when it completes, so in a dropped `fan_out` a missing
/// `embed_ms`, `tags_ms` or `count_ms` is a branch that never answered, while
/// `tag_search_ms` is also absent whenever no query word matched a tag.
struct StageClock {
    started: std::time::Instant,
    ms: RefCell<Vec<(Stage, u64)>>,
    last: Cell<Option<Stage>>,
    outcome: Cell<&'static str>,
}

impl StageClock {
    fn start() -> Self {
        Self {
            started: std::time::Instant::now(),
            ms: RefCell::default(),
            last: Cell::new(None),
            outcome: Cell::new("dropped"),
        }
    }

    /// Enters a sequential stage; it is timed until the returned guard drops.
    fn stage(&self, stage: Stage) -> StageTimer<'_> {
        self.last.set(Some(stage));
        StageTimer {
            clock: self,
            stage,
            started: std::time::Instant::now(),
        }
    }

    /// Times a concurrent branch from its first poll to its completion.
    async fn time<T>(&self, stage: Stage, fut: impl std::future::Future<Output = T>) -> T {
        let started = std::time::Instant::now();
        let out = fut.await;
        self.record(stage, started);
        out
    }

    fn record(&self, stage: Stage, started: std::time::Instant) {
        let ms = started.elapsed().as_millis() as u64;
        self.ms.borrow_mut().push((stage, ms));
    }

    fn finish(&self, ok: bool) {
        self.outcome.set(if ok { "ok" } else { "error" });
    }
}

impl Drop for StageClock {
    fn drop(&mut self) {
        let outcome = self.outcome.get();
        let stage = self.last.get().filter(|_| outcome != "ok").map(Stage::name);
        let recorded = self.ms.borrow();
        let ms = |s: Stage| recorded.iter().find(|(r, _)| *r == s).map(|(_, ms)| *ms);
        tracing::info!(
            outcome,
            stage,
            total_ms = self.started.elapsed().as_millis() as u64,
            fan_out_ms = ms(Stage::FanOut),
            embed_ms = ms(Stage::Embed),
            tags_ms = ms(Stage::Tags),
            tag_search_ms = ms(Stage::TagSearch),
            count_ms = ms(Stage::Count),
            vector_search_ms = ms(Stage::VectorSearch),
            rrf_fuse_ms = ms(Stage::RrfFuse),
            graph_boost_ms = ms(Stage::GraphBoost),
            graph_inject_ms = ms(Stage::GraphInject),
            rerank_ms = ms(Stage::Rerank),
            enrich_ms = ms(Stage::Enrich),
            "hybrid stages"
        );
    }
}

/// Records its stage's wall time when dropped: at the end of the stage's
/// block, or mid-stage when the search future is dropped.
struct StageTimer<'a> {
    clock: &'a StageClock,
    stage: Stage,
    started: std::time::Instant,
}

impl Drop for StageTimer<'_> {
    fn drop(&mut self) {
        self.clock.record(self.stage, self.started);
    }
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

/// The supersession marker (`metadata.superseded_by`), in whatever shape it
/// was stored. Its presence is what hides a memory.
fn supersession_marker(m: &Memory) -> Option<&Value> {
    m.metadata.as_ref()?.get("superseded_by")
}

/// True when the memory has been superseded (`metadata.superseded_by` set).
///
/// Superseded filtering MUST happen here at the application layer: the
/// `PayloadFilter.exclude_superseded` flag is a documented no-op in the
/// Qdrant backend because `is_null` on nested payload fields is unreliable
/// without an explicit payload index (issue #30, repo CLAUDE.md).
fn is_superseded(m: &Memory) -> bool {
    supersession_marker(m).is_some()
}

/// The memory a marker on `hash` names, when it names another valid one;
/// a legacy marker hides `hash` just the same but has no survivor.
fn survivor_of<'a>(marker: &'a Value, hash: &str) -> Option<&'a str> {
    marker
        .as_str()
        .filter(|s| alaya_types::memory::validate_content_hash(s) && *s != hash)
}

/// A trimmed `*_via` principal tag, 1..=`MAX_VIA_LEN` chars.
fn require_via<'a>(field: &str, raw: &'a str) -> Result<&'a str> {
    let via = raw.trim();
    if via.is_empty() || via.len() > MAX_VIA_LEN {
        return Err(AlayaError::Validation(format!(
            "{field} is required (1..={MAX_VIA_LEN} chars, e.g. operator:console)"
        )));
    }
    Ok(via)
}

/// `memory_unsupersede` on a memory with no marker: nothing to reverse.
fn not_superseded(content_hash: &str) -> Value {
    serde_json::json!({
        "success": false,
        "status": "not_superseded",
        "content_hash": content_hash,
        "error": "Memory is not superseded; nothing to reverse",
    })
}

/// A supersession that failed part-way (see `mark_superseded`).
struct SupersedeFailure {
    /// Memories that carry the marker anyway; their edges are written.
    marked: Vec<String>,
    /// Memories whose outcome could not be read back: the marker may have
    /// landed, without its edge. A retry of the same call settles them.
    in_doubt: Vec<String>,
    error: AlayaError,
}

/// Per-item error for a memory in `SupersedeFailure::in_doubt`. Distinct from
/// a plain failure so an operator can tell which memories need a retry.
const SUPERSEDE_OUTCOME_UNKNOWN: &str = "Outcome unknown: this memory may already be superseded \
     without its audit edge. Retry the same call; it is idempotent and converges.";

/// The hash `m` is superseded by, when set.
fn superseded_by(m: &Memory) -> Option<&str> {
    supersession_marker(m).and_then(Value::as_str)
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
    use std::sync::{Arc, Mutex};
    use tracing::instrument::WithSubscriber;
    use tracing_subscriber::layer::SubscriberExt;

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
        ConsolidationService, EmbeddingProvider, GraphService, HebbianService, StoreMode,
        SummaryProvider, VectorStorage,
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
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
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
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "healthy".into(),
                backend: "mock".into(),
                details: None,
            })
        }
    }

    /// No-op graph service.
    struct MockGraph;

    #[async_trait(?Send)]
    impl GraphService for MockGraph {
        async fn unsettle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn settle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            Ok(false)
        }
        async fn delete_incoming_system_edges(
            &self,
            _d: &str,
            _r: alaya_types::graph::SystemRelationType,
        ) -> Result<Vec<String>> {
            Ok(vec![])
        }
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

    // ─── check_database_health (LAB-4025) ──────────────────────────────

    /// Embedding endpoint that refuses everything — models TEI down.
    struct FailingEmbeddings;

    #[async_trait(?Send)]
    impl EmbeddingProvider for FailingEmbeddings {
        async fn embed_batch(&self, _t: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            Err(AlayaError::Embedding("connection refused".into()))
        }
        fn dimensions(&self) -> usize {
            1024
        }
        fn model_name(&self) -> &str {
            "failing"
        }
        async fn health(&self) -> Result<HealthStatus> {
            Err(AlayaError::Embedding("connection refused".into()))
        }
    }

    /// Before LAB-4025 the health document never consulted the embedding
    /// client: TEI down reported `healthy` while every store and semantic
    /// search failed. Now it degrades, names the probe, and — safe_message —
    /// leaks no endpoint detail.
    #[tokio::test(flavor = "current_thread")]
    async fn health_degrades_when_embedding_endpoint_is_down() {
        let svc = MemoryService::new(
            Box::new(MockVectors::new(vec![], Rc::new(Cell::new(0)))),
            Box::new(FailingEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );

        let h = svc.check_database_health().await.unwrap();

        assert_eq!(h["status"], "degraded");
        assert_eq!(h["embedding_health"]["status"], "unhealthy");
        assert_eq!(
            h["embedding_health"]["error"],
            "Embedding generation failed"
        );
        // The new probe must not disturb the vector verdict.
        assert_eq!(h["vector_health"]["status"], "ok");
        assert_eq!(h["graph_health"]["status"], "healthy");
    }

    /// Embedding endpoint that accepts the call and never answers — a wedged
    /// model server behind a live TCP listener.
    struct HangingEmbeddings;

    #[async_trait(?Send)]
    impl EmbeddingProvider for HangingEmbeddings {
        async fn embed_batch(&self, _t: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            std::future::pending().await
        }
        fn dimensions(&self) -> usize {
            1024
        }
        fn model_name(&self) -> &str {
            "hanging"
        }
        async fn health(&self) -> Result<HealthStatus> {
            std::future::pending().await
        }
    }

    /// The probe runs under its own budget so a hung endpoint cannot park the
    /// worker for the embed client's full 60s. Paused clock: the budget
    /// elapses instantly and exactly, so `elapsed` pins it at 5s; lose the
    /// bound and this test hangs instead.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn health_bounds_a_hung_embedding_probe() {
        let svc = MemoryService::new(
            Box::new(MockVectors::new(vec![], Rc::new(Cell::new(0)))),
            Box::new(HangingEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );

        let started = tokio::time::Instant::now();
        let h = svc.check_database_health().await.unwrap();
        assert_eq!(started.elapsed(), std::time::Duration::from_secs(5));

        assert_eq!(h["status"], "degraded");
        assert_eq!(h["embedding_health"]["status"], "unhealthy");
        assert_eq!(
            h["embedding_health"]["error"],
            "Embedding generation failed"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn health_reports_embedding_endpoint_when_up() {
        let (svc, _) = build_mock_service(vec![]);

        let h = svc.check_database_health().await.unwrap();

        assert_eq!(h["status"], "healthy");
        assert_eq!(h["embedding_health"]["status"], "healthy");
        assert_eq!(h["embedding_health"]["backend"], "mock");
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
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
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
        async fn health(&self) -> Result<HealthStatus> {
            Ok(HealthStatus {
                status: "healthy".into(),
                backend: "mock-tracked".into(),
                details: None,
            })
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
        async fn unsettle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn settle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn delete_incoming_system_edges(
            &self,
            _d: &str,
            _r: alaya_types::graph::SystemRelationType,
        ) -> Result<Vec<String>> {
            unimplemented!()
        }
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
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
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
        /// `Memory` (e.g. left by the legacy writer): `store` sees them,
        /// `get_by_hash` does not.
        raw_only: std::collections::HashSet<String>,
    }

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsPersisting {
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, m: &Memory, mode: StoreMode) -> Result<(bool, String)> {
            let mut stored = self.stored.borrow_mut();
            let exists =
                self.raw_only.contains(&m.content_hash) || stored.contains_key(&m.content_hash);
            if exists && mode == StoreMode::InsertOnly {
                return Ok((false, m.content_hash.clone()));
            }
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
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
            Ok(self.injectable_memories.get(h).cloned())
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
        async fn unsettle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn settle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn delete_incoming_system_edges(
            &self,
            _d: &str,
            _r: alaya_types::graph::SystemRelationType,
        ) -> Result<Vec<String>> {
            unimplemented!()
        }
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
        /// When set, update_metadata_batch marks only these memories, then
        /// fails: a batch that committed part-way.
        partial_update: RefCell<Option<Vec<String>>>,
        /// When set to `n`, the n-th get_batch call and every later one fail:
        /// Qdrant stops answering reads mid-operation.
        fail_get_batch_from: Cell<Option<usize>>,
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
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
            self.0
                .get_by_hash_calls
                .set(self.0.get_by_hash_calls.get() + 1);
            Ok(self.0.memories.borrow().get(h).cloned())
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
            if self
                .0
                .fail_get_batch_from
                .get()
                .is_some_and(|n| self.0.get_batch_calls.get() >= n)
            {
                return Err(AlayaError::Storage("mock read failure".into()));
            }
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
            if let Some(marked) = self.0.partial_update.borrow().as_ref() {
                let mut mems = self.0.memories.borrow_mut();
                for h in marked {
                    let m = mems
                        .get_mut(h)
                        .expect("partial_update names a seeded memory");
                    m.metadata.get_or_insert_with(HashMap::new).insert(
                        "superseded_by".into(),
                        serde_json::json!(updates.superseded_by),
                    );
                }
                return Err(AlayaError::Storage("mock partial failure".into()));
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
        async fn unsettle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn settle_contradiction(
            &self,
            _s: &str,
            _d: &str,
            _v: &str,
            _t: f64,
        ) -> Result<bool> {
            unimplemented!()
        }
        async fn delete_incoming_system_edges(
            &self,
            _d: &str,
            _r: alaya_types::graph::SystemRelationType,
        ) -> Result<Vec<String>> {
            unimplemented!()
        }
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

    /// A batch that commits part-way (alaya#130): the memory that was marked
    /// gets its SUPERSEDES edge and is reported as superseded; only the other
    /// one is an error. Nothing is hidden from search without its audit edge.
    #[tokio::test(flavor = "current_thread")]
    async fn merge_duplicates_partial_commit_writes_edges_for_what_landed() {
        let canonical = "c".repeat(64);
        let d1 = "1".repeat(64);
        let d2 = "2".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&canonical, &d1, &d2]);
        *rec.partial_update.borrow_mut() = Some(vec![d1.clone()]);

        let result = svc
            .merge_duplicates(&canonical, &[&d1, &d2], "dedup", false)
            .await
            .expect("a partial failure is per-item, not a call failure");

        assert_eq!(result["success"], serde_json::json!(false));
        assert_eq!(result["superseded"], serde_json::json!([d1]));
        let errors = result["errors"].as_array().unwrap();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert_eq!(errors[0]["hash"], serde_json::json!(d2));
        assert_eq!(*rec.system_edges.borrow(), vec![(canonical, d1)]);
    }

    /// Both markers land, the batch still errors, and the
    /// recovery read fails too. Which memories were marked is unknown, so both
    /// are reported as outcome-unknown — not as untouched — and no edge is
    /// guessed. Once Qdrant answers again, retrying the same merge converges:
    /// both superseded, both edges written.
    #[tokio::test(flavor = "current_thread")]
    async fn merge_duplicates_unreadable_outcome_is_unknown_and_a_retry_converges() {
        let canonical = "c".repeat(64);
        let d1 = "1".repeat(64);
        let d2 = "2".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&canonical, &d1, &d2]);
        *rec.partial_update.borrow_mut() = Some(vec![d1.clone(), d2.clone()]);
        // Call 1 is the existence pre-check; call 2, the recovery read, fails.
        rec.fail_get_batch_from.set(Some(2));

        let result = svc
            .merge_duplicates(&canonical, &[&d1, &d2], "dedup", false)
            .await
            .expect("an unknown outcome is per-item, not a call failure");
        assert_eq!(result["success"], serde_json::json!(false));
        assert_eq!(result["superseded"], serde_json::json!([] as [&str; 0]));
        let errors = result["errors"].as_array().unwrap();
        assert_eq!(errors.len(), 2, "{errors:?}");
        for e in errors {
            assert_eq!(
                e["error"],
                serde_json::json!(SUPERSEDE_OUTCOME_UNKNOWN),
                "{e}"
            );
        }
        assert!(
            rec.system_edges.borrow().is_empty(),
            "no edge is guessed for an unknown outcome"
        );

        // Qdrant is back: the same call again.
        *rec.partial_update.borrow_mut() = None;
        rec.fail_get_batch_from.set(None);
        let retry = svc
            .merge_duplicates(&canonical, &[&d1, &d2], "dedup", false)
            .await
            .expect("retry succeeds");
        assert_eq!(retry["success"], serde_json::json!(true));
        assert_eq!(retry["superseded"], serde_json::json!([d1, d2]));
        assert_eq!(
            *rec.system_edges.borrow(),
            vec![(canonical.clone(), d1), (canonical, d2)]
        );
    }

    /// The single-memory form of the same schedule: the error surfaces, no
    /// edge is guessed, and a retry writes the edge.
    #[tokio::test(flavor = "current_thread")]
    async fn memory_supersede_unreadable_outcome_errors_and_a_retry_converges() {
        let old = "0".repeat(64);
        let new = "f".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&old, &new]);
        *rec.partial_update.borrow_mut() = Some(vec![old.clone()]);
        rec.fail_get_batch_from.set(Some(2));

        let err = svc
            .memory_supersede(&old, &new, "corrected")
            .await
            .expect_err("the failure is reported");
        assert!(matches!(err, AlayaError::Storage(_)), "{err:?}");
        assert!(rec.system_edges.borrow().is_empty());

        *rec.partial_update.borrow_mut() = None;
        rec.fail_get_batch_from.set(None);
        svc.memory_supersede(&old, &new, "corrected")
            .await
            .expect("retry succeeds");
        assert_eq!(*rec.system_edges.borrow(), vec![(new, old)]);
    }

    /// A single supersede whose write landed but failed to confirm still gets
    /// its edge, and the caller still sees the error.
    #[tokio::test(flavor = "current_thread")]
    async fn memory_supersede_writes_the_edge_when_the_marker_landed_despite_an_error() {
        let old = "0".repeat(64);
        let new = "f".repeat(64);

        let (svc, rec) = build_merge_service();
        rec.seed(&[&old, &new]);
        *rec.partial_update.borrow_mut() = Some(vec![old.clone()]);

        let err = svc
            .memory_supersede(&old, &new, "corrected")
            .await
            .expect_err("the failure is reported");
        assert!(matches!(err, AlayaError::Storage(_)), "{err:?}");
        assert_eq!(*rec.system_edges.borrow(), vec![(new, old)]);
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
            scored_from(&self.memories, limit)
        }
    }

    // Search entry points over a memory list, returned unfiltered — superseded
    // filtering is the app layer's job, as with the real backend.

    fn scored_from(memories: &[Memory], limit: usize) -> Vec<ScoredMemory> {
        memories
            .iter()
            .take(limit)
            .enumerate()
            .map(|(i, m)| ScoredMemory {
                memory: m.clone(),
                score: 0.9 - i as f64 * 0.01,
            })
            .collect()
    }

    fn page_from(memories: &[Memory], limit: usize, offset: Option<&str>) -> ScrollResult {
        let start: usize = offset.and_then(|o| o.parse().ok()).unwrap_or(0);
        let end = (start + limit).min(memories.len());
        ScrollResult {
            memories: memories[start..end].to_vec(),
            next_offset: (end < memories.len()).then(|| end.to_string()),
        }
    }

    fn recent_from(memories: &[Memory], limit: usize, start_from: Option<f64>) -> Vec<Memory> {
        let mut sorted = memories.to_vec();
        sorted.sort_by(|a, b| b.created_at.total_cmp(&a.created_at));
        sorted
            .into_iter()
            .filter(|m| start_from.is_none_or(|ts| m.created_at < ts))
            .take(limit)
            .collect()
    }

    /// The hashes one search in `mode` returns.
    async fn result_hashes(
        svc: &MemoryService,
        mode: SearchMode,
        include_superseded: bool,
    ) -> std::collections::HashSet<String> {
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

    #[async_trait(?Send)]
    impl VectorStorage for MockVectorsCorpus {
        async fn reverse_supersession(
            &self,
            _h: &str,
            _e: &serde_json::Value,
            _r: &alaya_backends::ReversalRecord,
        ) -> Result<alaya_backends::ReversalOutcome> {
            unimplemented!()
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            Ok((true, "mock".into()))
        }
        async fn get_by_hash(&self, _h: &str) -> Result<Option<Memory>> {
            Ok(None)
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
            Ok(page_from(&self.memories, limit, offset))
        }
        async fn get_recent(
            &self,
            limit: usize,
            start_from: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(recent_from(&self.memories, limit, start_from))
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
        result_hashes(&svc, mode, include_superseded).await
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

    // ─── Log capture ────────────────────────────────────────────────────

    /// Collects every event on target `.0` as a field map, plus its `level`.
    struct Capture(&'static str, Arc<Mutex<Vec<HashMap<String, String>>>>);

    impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for Capture {
        fn on_event(
            &self,
            event: &tracing::Event<'_>,
            _ctx: tracing_subscriber::layer::Context<'_, S>,
        ) {
            if event.metadata().target() != self.0 {
                return;
            }
            let mut fields = HashMap::new();
            fields.insert("level".to_string(), event.metadata().level().to_string());
            event.record(&mut Fields(&mut fields));
            self.1.lock().unwrap().push(fields);
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

    /// Keeps a second dispatcher registered for the whole test process. A
    /// callsite caches its interest when first hit; while only one dispatcher
    /// is registered, tracing-core asks just the hitting thread's default —
    /// none, when a parallel test hits it first — and caches `never`. With two
    /// registered it asks every live one, the capturing dispatcher included.
    static KEEPALIVE: std::sync::LazyLock<tracing::Dispatch> =
        std::sync::LazyLock::new(|| tracing::Dispatch::new(tracing_subscriber::registry()));

    /// Runs `fut` and returns what it logged on `target`. Only the polling
    /// thread is captured, and only while `fut` is being polled.
    async fn captured<T>(
        target: &'static str,
        fut: impl std::future::Future<Output = T>,
    ) -> (T, Vec<HashMap<String, String>>) {
        std::sync::LazyLock::force(&KEEPALIVE);
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture = tracing::Dispatch::new(
            tracing_subscriber::registry().with(Capture(target, events.clone())),
        );
        let out = fut.with_subscriber(capture).await;
        let collected = events.lock().unwrap().clone();
        (out, collected)
    }

    // ─── Hybrid stage timings ───────────────────────────────────────────

    /// The `hybrid stages` lines `fut` logged.
    async fn stage_lines<T>(
        fut: impl std::future::Future<Output = T>,
    ) -> (T, Vec<HashMap<String, String>>) {
        let (out, events) = captured("alaya_core::service", fut).await;
        let lines = events
            .into_iter()
            .filter(|e| e.get("message").map(String::as_str) == Some("hybrid stages"))
            .collect();
        (out, lines)
    }

    fn keys(line: &HashMap<String, String>) -> Vec<&str> {
        let mut k: Vec<&str> = line.keys().map(String::as_str).collect();
        k.sort_unstable();
        k
    }

    /// One line per search, durations only: the exact key set also proves no
    /// query text, hash or tag is logged. Stages that did not run (no graph
    /// neighbours to inject, no reranker) are absent, and so is `stage`.
    #[tokio::test(flavor = "current_thread")]
    async fn hybrid_search_logs_one_stage_timing_line() {
        // A query word that is a known tag makes the keyword tag search run.
        let (svc, _) = build_mock_service(vec!["rust".into()]);

        let (out, lines) = stage_lines(svc.search(search_params("notes about rust"))).await;

        out.expect("search succeeds");
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["level"], "INFO");
        assert_eq!(line["outcome"], "ok");
        assert_eq!(
            keys(line),
            [
                "count_ms",
                "embed_ms",
                "enrich_ms",
                "fan_out_ms",
                "graph_boost_ms",
                "level",
                "message",
                "outcome",
                "rrf_fuse_ms",
                "tag_search_ms",
                "tags_ms",
                "total_ms",
                "vector_search_ms",
            ]
        );
        for (k, v) in line {
            if k.ends_with("_ms") {
                v.parse::<u64>()
                    .unwrap_or_else(|_| panic!("{k}={v} is not a u64"));
            }
        }
    }

    /// Graph injection and rerank run only with graph neighbours to inject and
    /// a reranker configured; with both, their durations are on the ok line.
    #[tokio::test(flavor = "current_thread")]
    async fn hybrid_search_logs_graph_inject_and_rerank_timings() {
        let seed = make_scored_memory(&"a".repeat(64), "seed", 0.6);
        let neighbor = make_scored_memory(&"b".repeat(64), "neighbor", 0.0).memory;
        let svc = MemoryService::new(
            Box::new(MockVectorsWithInjection {
                search_results: vec![seed],
                injectable_memories: HashMap::from([(neighbor.content_hash.clone(), neighbor)]),
            }),
            Box::new(MockEmbeddings),
            Box::new(MockGraphWithActivation {
                activation: HashMap::from([("b".repeat(64), 0.8)]),
            }),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        )
        .with_reranker(Box::new(MockReranker {
            top_n: 20,
            scores_by_doc_prefix: HashMap::new(),
            sleep_for: std::time::Duration::ZERO,
            budget: std::time::Duration::from_secs(5),
        }));

        let (out, lines) = stage_lines(svc.search(search_params("anything"))).await;

        out.expect("search succeeds");
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["outcome"], "ok");
        let keys = keys(line);
        assert!(
            keys.contains(&"graph_inject_ms") && keys.contains(&"rerank_ms"),
            "{keys:?}"
        );
    }

    /// The worker drops a search at its deadline; the line must still come
    /// out, naming the stage it was stuck in. Inside fan-out a branch that
    /// never answered has no duration (here `embed_ms`); `tag_search_ms` is
    /// absent too, because no query word matched a tag.
    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn dropped_hybrid_search_logs_the_stage_it_was_stuck_in() {
        let svc = MemoryService::new(
            Box::new(MockVectors::new(vec![], Rc::new(Cell::new(0)))),
            Box::new(HangingEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );

        // The timeout — and so the drop — runs inside the captured future.
        let (timed_out, lines) = stage_lines(async {
            tokio::time::timeout(
                std::time::Duration::from_secs(120),
                svc.search(search_params("anything")),
            )
            .await
            .is_err()
        })
        .await;

        assert!(
            timed_out,
            "the hung embed must hold the search to the deadline"
        );
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["outcome"], "dropped");
        assert_eq!(line["stage"], "fan_out");
        assert_eq!(
            keys(line),
            [
                "count_ms",
                "fan_out_ms",
                "level",
                "message",
                "outcome",
                "stage",
                "tags_ms",
                "total_ms",
            ]
        );
    }

    /// An error return is `error`, not `dropped`, and names the stage whose
    /// result failed; the search's own error is unchanged.
    #[tokio::test(flavor = "current_thread")]
    async fn failed_hybrid_search_logs_error_and_its_stage() {
        let svc = MemoryService::new(
            Box::new(MockVectors::new(vec![], Rc::new(Cell::new(0)))),
            Box::new(FailingEmbeddings),
            Box::new(MockGraph),
            Box::new(MockHebbian),
            Box::new(MockConsolidation),
            None,
        );

        let (out, lines) = stage_lines(svc.search(search_params("anything"))).await;

        assert!(matches!(out, Err(AlayaError::Embedding(_))), "{out:?}");
        assert_eq!(lines.len(), 1, "{lines:?}");
        let line = &lines[0];
        assert_eq!(line["outcome"], "error");
        assert_eq!(line["stage"], "fan_out");
        assert_eq!(
            keys(line),
            [
                "count_ms",
                "embed_ms",
                "fan_out_ms",
                "level",
                "message",
                "outcome",
                "stage",
                "tags_ms",
                "total_ms",
            ]
        );
    }

    // ─── Contradiction judge (LAB-3283: AC-2 failure paths, AC-4, AC-4b) ──

    mod judge_tests {
        use super::*;
        use crate::service::{JudgeOutcome, SHADOW_LOG_TARGET};
        use alaya_backends::{ContradictionJudge, Judgement, Survivor};
        use alaya_types::graph::{EdgeVerdict, Resolution, Verdict};
        use std::rc::Rc;

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
            async fn reverse_supersession(
                &self,
                _h: &str,
                _e: &serde_json::Value,
                _r: &alaya_backends::ReversalRecord,
            ) -> Result<alaya_backends::ReversalOutcome> {
                unimplemented!()
            }
            async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
                unreachable!("judge must never write to the vector store")
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
            async fn unsettle_contradiction(
                &self,
                _s: &str,
                _d: &str,
                _v: &str,
                _t: f64,
            ) -> Result<bool> {
                unimplemented!()
            }
            async fn settle_contradiction(
                &self,
                _s: &str,
                _d: &str,
                _v: &str,
                _t: f64,
            ) -> Result<bool> {
                unimplemented!()
            }
            async fn delete_incoming_system_edges(
                &self,
                _d: &str,
                _r: alaya_types::graph::SystemRelationType,
            ) -> Result<Vec<String>> {
                unimplemented!()
            }
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
                let (outcome, events) =
                    captured(SHADOW_LOG_TARGET, svc.judge_contradiction(&src(), &dst())).await;
                assert!(
                    matches!(
                        outcome,
                        JudgeOutcome::Judged { ref judgement, persisted: true }
                            if judgement.verdict == verdict
                    ),
                    "{verdict:?}: {outcome:?}"
                );
                assert!(outcome.spent(), "a verdict was paid for");

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
            let (outcome, events) =
                captured(SHADOW_LOG_TARGET, svc.judge_contradiction(&src(), &dst())).await;
            assert!(
                matches!(
                    outcome,
                    JudgeOutcome::Unjudged {
                        marked: true,
                        spent: true
                    }
                ),
                "the request went out, so it stays billed: {outcome:?}"
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

        /// A deterministic failure whose marker write did not land is still
        /// billed: the marker is about what the backfill re-selects, the bill
        /// about what left the box. A marker-keyed refund would credit this.
        #[tokio::test(flavor = "current_thread")]
        async fn deterministic_failure_stays_billed_when_the_marker_does_not_land() {
            let (svc, _) = service(
                Some(Script::Err(|| {
                    AlayaError::Judge("verdict is not valid JSON".into())
                })),
                pair(),
                false,
            );
            let outcome = svc.judge_contradiction(&src(), &dst()).await;
            assert!(
                matches!(
                    outcome,
                    JudgeOutcome::Unjudged {
                        marked: false,
                        spent: true
                    }
                ),
                "{outcome:?}"
            );
        }

        /// A transient failure (upstream down, 5xx, timeout) writes nothing so
        /// the pair is retried by the next pass. Whether it was billed is the
        /// transport's call — a 504 after forwarding may have been, a refused
        /// connect was not — and is carried through unchanged.
        #[tokio::test(flavor = "current_thread")]
        async fn transient_failure_is_unjudged_and_writes_nothing() {
            fn billed() -> AlayaError {
                AlayaError::Unavailable {
                    message: "504 from the LB".into(),
                    spent: true,
                }
            }
            fn free() -> AlayaError {
                AlayaError::Unavailable {
                    message: "connection refused".into(),
                    spent: false,
                }
            }
            for (err, spent) in [(billed as fn() -> AlayaError, true), (free, false)] {
                let (svc, verdicts) = service(Some(Script::Err(err)), pair(), true);
                let (outcome, events) =
                    captured(SHADOW_LOG_TARGET, svc.judge_contradiction(&src(), &dst())).await;
                assert!(
                    matches!(
                        outcome,
                        JudgeOutcome::Unjudged {
                            marked: false,
                            spent: s
                        } if s == spent
                    ),
                    "spent={spent}: {outcome:?}"
                );
                assert!(
                    verdicts.borrow().is_empty(),
                    "no graph write on a transient failure"
                );
                assert!(events.is_empty(), "{events:?}");
            }
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
            let long = "x".repeat(MAX_VIA_LEN + 1);
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
            assert!(!outcome.spent(), "a 429 is refunded");
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
            let outcome = svc.judge_contradiction(&src(), &dst()).await;
            assert!(
                matches!(
                    outcome,
                    JudgeOutcome::Unjudged {
                        marked: true,
                        spent: false
                    }
                ),
                "marked so the backfill skips it, but no request went out: {outcome:?}"
            );
            assert!(!outcome.spent(), "a marked pair is not a paid pair");
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
                JudgeOutcome::Unjudged {
                    marked: false,
                    spent: false
                }
            ));
            assert!(matches!(
                svc.judge_contradiction(&src(), &src()).await,
                JudgeOutcome::Unjudged {
                    marked: false,
                    spent: false
                }
            ));
            assert_eq!(verdicts.borrow().len(), 1);
            // Judge not configured.
            let (svc, verdicts) = service(None, pair(), true);
            assert!(matches!(
                svc.judge_contradiction(&src(), &dst()).await,
                JudgeOutcome::Unjudged {
                    marked: false,
                    spent: false
                }
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
            let (outcome, events) =
                captured(SHADOW_LOG_TARGET, svc.judge_contradiction(&src(), &dst())).await;
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

    // ─── memory_unsupersede (LAB-6876) ──────────────────────────────────
    //
    // One shared ledger plays both backends, so a test can drive supersede
    // and unsupersede end to end and then read what every layer holds.

    /// Qdrant and FalkorDB as a supersession round trip sees them. Memories
    /// come back from every search entry point UNFILTERED — superseded
    /// filtering is the app layer's job, as with the real backend.
    #[derive(Default)]
    struct Ledger {
        memories: RefCell<Vec<Memory>>,
        /// Payload fields `Memory` does not carry, per hash.
        reasons: RefCell<HashMap<String, Value>>,
        logs: RefCell<HashMap<String, Vec<Value>>>,
        /// SUPERSEDES edges, (src, dst).
        supersedes: RefCell<Vec<(String, String)>>,
        /// CONTRADICTS edges, (src, dst) -> (`resolved_via`, `resolved_at`)
        /// of a keep_both stamp.
        contradicts: RefCell<HashMap<(String, String), Option<(String, f64)>>>,
        /// Every graph write fails, as when FalkorDB is down.
        graph_down: Cell<bool>,
        /// Another writer re-supersedes the memory to this hash inside the
        /// next `reverse_supersession`'s read→write window.
        resupersede: RefCell<Option<String>>,
        /// A concurrent supersede to the SAME survivor writes this SUPERSEDES
        /// edge inside that window (after unsupersede's edge sweep).
        late_edge: RefCell<Option<(String, String)>>,
        /// An operator stamps this pair inside that window.
        operator_stamp: RefCell<Option<(String, String)>>,
        /// Another writer supersedes the memory to this hash right AFTER a
        /// reversal lands: marker, then edge.
        resupersede_after: RefCell<Option<String>>,
    }

    impl Ledger {
        fn with(hashes: &[&str]) -> Rc<Self> {
            let ledger = Self::default();
            for (i, h) in hashes.iter().enumerate() {
                let mut m = dummy_memory();
                m.content = format!("ledger memory {i}");
                m.content_hash = (*h).to_string();
                m.created_at = 1_000_000.0 + i as f64;
                m.updated_at = m.created_at;
                ledger.memories.borrow_mut().push(m);
            }
            Rc::new(ledger)
        }

        fn service(self: &Rc<Self>) -> MemoryService {
            MemoryService::new(
                Box::new(LedgerVectors(self.clone())),
                Box::new(MockEmbeddings),
                Box::new(LedgerGraph(self.clone())),
                Box::new(MockHebbian),
                Box::new(MockConsolidation),
                None,
            )
        }

        fn marker(&self, h: &str) -> Option<Value> {
            self.memories
                .borrow()
                .iter()
                .find(|m| m.content_hash == h)?
                .metadata
                .as_ref()?
                .get("superseded_by")
                .cloned()
        }

        fn set_marker(&self, h: &str, marker: Option<Value>) {
            let mut mems = self.memories.borrow_mut();
            let m = mems.iter_mut().find(|m| m.content_hash == h).unwrap();
            let md = m.metadata.get_or_insert_with(HashMap::new);
            match marker {
                Some(v) => md.insert("superseded_by".into(), v),
                None => md.remove("superseded_by"),
            };
        }

        /// Who stamped the `a -> b` pair, if anyone.
        fn stamped_by(&self, a: &str, b: &str) -> Option<String> {
            self.contradicts.borrow()[&(a.to_string(), b.to_string())]
                .as_ref()
                .map(|(via, _)| via.clone())
        }

        fn edges(&self) -> Vec<(String, String)> {
            let mut e = self.supersedes.borrow().clone();
            e.sort();
            e
        }

        fn graph_write(&self) -> Result<()> {
            if self.graph_down.get() {
                return Err(AlayaError::Graph("ledger graph down".into()));
            }
            Ok(())
        }
    }

    struct LedgerVectors(Rc<Ledger>);

    #[async_trait(?Send)]
    impl VectorStorage for LedgerVectors {
        async fn reverse_supersession(
            &self,
            h: &str,
            expected: &Value,
            r: &ReversalRecord,
        ) -> Result<ReversalOutcome> {
            let l = &self.0;
            if let Some(other) = l.resupersede.borrow_mut().take() {
                l.set_marker(h, Some(serde_json::json!(other)));
            }
            if let Some(edge) = l.late_edge.borrow_mut().take() {
                l.supersedes.borrow_mut().push(edge);
            }
            if let Some(pair) = l.operator_stamp.borrow_mut().take() {
                l.contradicts
                    .borrow_mut()
                    .insert(pair, Some(("operator:console".into(), 1.0)));
            }
            if !l.memories.borrow().iter().any(|m| m.content_hash == h) {
                return Err(AlayaError::NotFound(h.into()));
            }
            let Some(marker) = l.marker(h) else {
                return Ok(ReversalOutcome::NotSuperseded);
            };
            if &marker != expected {
                return Ok(ReversalOutcome::SupersededByOther(marker));
            }
            l.set_marker(h, None);
            let reason = l.reasons.borrow_mut().remove(h);
            l.logs
                .borrow_mut()
                .entry(h.into())
                .or_default()
                .push(serde_json::json!({
                    "superseded_by": marker,
                    "supersession_reason": reason,
                    "unsuperseded_at": r.at,
                    "unsuperseded_via": r.via,
                    "reason": r.reason,
                }));
            if let Some(by) = l.resupersede_after.borrow_mut().take() {
                l.set_marker(h, Some(serde_json::json!(by)));
                l.supersedes.borrow_mut().push((by, h.to_string()));
            }
            Ok(ReversalOutcome::Cleared {
                supersession_reason: reason,
            })
        }
        async fn store(&self, _m: &Memory, _mode: StoreMode) -> Result<(bool, String)> {
            unimplemented!()
        }
        async fn get_by_hash(&self, h: &str) -> Result<Option<Memory>> {
            Ok(self
                .0
                .memories
                .borrow()
                .iter()
                .find(|m| m.content_hash == h)
                .cloned())
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
            Ok(self
                .0
                .memories
                .borrow()
                .iter()
                .filter(|m| hashes.contains(&m.content_hash.as_str()))
                .cloned()
                .collect())
        }
        async fn delete(&self, _h: &str) -> Result<bool> {
            unimplemented!()
        }
        async fn update_metadata(&self, h: &str, u: MetadataUpdate) -> Result<()> {
            self.update_metadata_batch(&[h], u).await
        }
        async fn update_metadata_batch(&self, hashes: &[&str], u: MetadataUpdate) -> Result<()> {
            for h in hashes {
                if self.get_by_hash(h).await?.is_none() {
                    return Err(AlayaError::NotFound((*h).into()));
                }
            }
            for h in hashes {
                if let Some(ref by) = u.superseded_by {
                    self.0.set_marker(h, Some(serde_json::json!(by)));
                }
                if let Some(reason) = u.extra.as_ref().and_then(|e| e.get("supersession_reason")) {
                    self.0
                        .reasons
                        .borrow_mut()
                        .insert((*h).into(), reason.clone());
                }
            }
            Ok(())
        }
        async fn patch_memory(&self, _h: &str, _p: &PatchMemoryRequest) -> Result<Memory> {
            unimplemented!()
        }
        async fn search_by_vector(
            &self,
            _e: &[f32],
            limit: usize,
            _f: Option<PayloadFilter>,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(scored_from(&self.0.memories.borrow(), limit))
        }
        async fn search_by_tags(
            &self,
            _t: &[&str],
            _m: bool,
            limit: usize,
        ) -> Result<Vec<ScoredMemory>> {
            Ok(scored_from(&self.0.memories.borrow(), limit))
        }
        async fn search_similar_tags(&self, _e: &[f32], _l: usize) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn upsert_tags(&self, _tags: &[(&str, Vec<f32>)]) -> Result<()> {
            Ok(())
        }
        async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult> {
            Ok(page_from(&self.0.memories.borrow(), limit, offset))
        }
        async fn get_recent(
            &self,
            limit: usize,
            start_from: Option<f64>,
            _t: Option<&str>,
        ) -> Result<Vec<Memory>> {
            Ok(recent_from(&self.0.memories.borrow(), limit, start_from))
        }
        async fn count(&self) -> Result<usize> {
            Ok(self.0.memories.borrow().len())
        }
        async fn get_all_tags(&self) -> Result<Vec<String>> {
            Ok(vec![])
        }
        async fn increment_access_count(&self, _h: &str) -> Result<()> {
            Ok(())
        }
        async fn health(&self) -> Result<HealthStatus> {
            unimplemented!()
        }
    }

    struct LedgerGraph(Rc<Ledger>);

    #[async_trait(?Send)]
    impl GraphService for LedgerGraph {
        async fn delete_incoming_system_edges(
            &self,
            dst: &str,
            _r: SystemRelationType,
        ) -> Result<Vec<String>> {
            self.0.graph_write()?;
            let mut sources = Vec::new();
            self.0.supersedes.borrow_mut().retain(|(s, d)| {
                let into = d == dst;
                if into {
                    sources.push(s.clone());
                }
                !into
            });
            Ok(sources)
        }
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
            unimplemented!()
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
            unimplemented!()
        }
        async fn create_system_edge(
            &self,
            s: &str,
            d: &str,
            _r: SystemRelationType,
            _t: f64,
        ) -> Result<bool> {
            self.0.graph_write()?;
            let edge = (s.to_string(), d.to_string());
            let mut edges = self.0.supersedes.borrow_mut();
            if edges.contains(&edge) {
                return Ok(false);
            }
            edges.push(edge);
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
            unimplemented!()
        }
        async fn set_contradiction_resolution(
            &self,
            s: &str,
            d: &str,
            r: Option<Resolution>,
            via: &str,
            t: f64,
        ) -> Result<bool> {
            self.0.graph_write()?;
            let mut pairs = self.0.contradicts.borrow_mut();
            let Some(stamp) = pairs.get_mut(&(s.to_string(), d.to_string())) else {
                return Ok(false);
            };
            *stamp = r.map(|_| (via.to_string(), t));
            Ok(true)
        }
        async fn settle_contradiction(&self, s: &str, d: &str, via: &str, t: f64) -> Result<bool> {
            self.0.graph_write()?;
            let mut pairs = self.0.contradicts.borrow_mut();
            match pairs.get_mut(&(s.to_string(), d.to_string())) {
                Some(stamp @ None) => {
                    *stamp = Some((via.to_string(), t));
                    Ok(true)
                }
                _ => Ok(false),
            }
        }
        async fn unsettle_contradiction(
            &self,
            s: &str,
            d: &str,
            via: &str,
            t: f64,
        ) -> Result<bool> {
            self.0.graph_write()?;
            let mut pairs = self.0.contradicts.borrow_mut();
            match pairs.get_mut(&(s.to_string(), d.to_string())) {
                Some(stamp) if *stamp == Some((via.to_string(), t)) => {
                    *stamp = None;
                    Ok(true)
                }
                _ => Ok(false),
            }
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
            unimplemented!()
        }
    }

    fn h(c: char) -> String {
        c.to_string().repeat(64)
    }

    /// Hashes `svc` returns for a default (superseded-hidden) search per mode.
    async fn visible(svc: &MemoryService) -> Vec<std::collections::HashSet<String>> {
        let mut per_mode = Vec::new();
        for mode in ALL_MODES {
            per_mode.push(result_hashes(svc, mode, false).await);
        }
        per_mode
    }

    fn set_of(hashes: &[&str]) -> std::collections::HashSet<String> {
        hashes.iter().map(|s| s.to_string()).collect()
    }

    async fn unsupersede(svc: &MemoryService, hash: &str) -> Value {
        svc.memory_unsupersede(hash, "wrong merge", "operator:test")
            .await
            .expect("unsupersede returns an outcome")
    }

    /// Supersede → unsupersede → the memory is back in every search mode, the
    /// marker, reason and SUPERSEDES edge are gone, and the audit entry
    /// records who, when, why and what was reversed.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_round_trips_search_visibility_in_every_mode() {
        let (a, b) = (h('a'), h('b'));
        let ledger = Ledger::with(&[&a, &b]);
        let svc = ledger.service();

        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();
        for (mode, seen) in ALL_MODES.iter().zip(visible(&svc).await) {
            assert_eq!(seen, set_of(&[&b]), "{mode:?}: superseded memory hidden");
        }
        assert_eq!(ledger.edges(), [(b.clone(), a.clone())]);

        let (out, events) = captured("alaya::supersession", unsupersede(&svc, &a)).await;
        assert_eq!(out["success"], true, "{out}");
        assert_eq!(out["status"], "unsuperseded");
        assert_eq!(out["superseded_by"], serde_json::json!(b));
        assert_eq!(out["supersession_reason"], "duplicate");
        assert_eq!(out["supersedes_edges_removed"], serde_json::json!([b]));
        for (mode, seen) in ALL_MODES.iter().zip(visible(&svc).await) {
            assert_eq!(seen, set_of(&[&a, &b]), "{mode:?}: restored memory visible");
        }

        assert_eq!(ledger.marker(&a), None);
        assert!(!ledger.reasons.borrow().contains_key(&a));
        assert!(ledger.edges().is_empty());
        assert_eq!(
            ledger.logs.borrow()[&a],
            [serde_json::json!({
                "superseded_by": b,
                "supersession_reason": "duplicate",
                "unsuperseded_at": out["unsuperseded_at"],
                "unsuperseded_via": "operator:test",
                "reason": "wrong merge",
            })]
        );
        assert_eq!(events.len(), 1, "one audit event: {events:?}");
        assert_eq!(events[0]["hash"], a);
        assert_eq!(events[0]["via"], "operator:test");
        assert_eq!(events[0]["reason"], "wrong merge");
    }

    /// Not superseded is a typed no-op: `success: false`, a status saying
    /// why, and no write to either backend.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_of_a_live_memory_is_a_typed_no_op_that_writes_nothing() {
        let (a, b) = (h('a'), h('b'));
        let ledger = Ledger::with(&[&a, &b]);
        ledger
            .contradicts
            .borrow_mut()
            .insert((a.clone(), b.clone()), None);
        ledger.supersedes.borrow_mut().push((a.clone(), b.clone()));
        let svc = ledger.service();

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["success"], false, "{out}");
        assert_eq!(out["status"], "not_superseded");
        assert_eq!(ledger.edges(), [(a.clone(), b.clone())], "untouched");
        assert_eq!(ledger.stamped_by(&a, &b), None);
        assert!(ledger.logs.borrow().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_refuses_bad_input_and_absent_memories() {
        let a = h('a');
        let svc = Ledger::with(&[&a]).service();
        let non_hex = "g".repeat(64);
        for (hash, reason, via) in [
            ("abc", "why", "operator:test"),
            (non_hex.as_str(), "why", "operator:test"),
            (a.as_str(), "  ", "operator:test"),
            (a.as_str(), "why", ""),
            (a.as_str(), "why", "   "),
        ] {
            let err = svc.memory_unsupersede(hash, reason, via).await.unwrap_err();
            assert!(matches!(err, AlayaError::Validation(_)), "{err:?}");
        }
        let long = "x".repeat(MAX_UNSUPERSEDE_REASON_LEN + 1);
        let err = svc
            .memory_unsupersede(&a, &long, "operator:test")
            .await
            .unwrap_err();
        assert!(matches!(err, AlayaError::Validation(_)), "{err:?}");
        // The cap counts characters, as the MCP schema's maxLength does: a
        // full-length multi-byte reason passes validation.
        let cjk = "界".repeat(MAX_UNSUPERSEDE_REASON_LEN);
        let out = svc
            .memory_unsupersede(&a, &cjk, "operator:test")
            .await
            .expect("a reason at the character cap is valid");
        assert_eq!(out["status"], "not_superseded", "{out}");
        let err = svc
            .memory_unsupersede(&h('f'), "why", "operator:test")
            .await
            .unwrap_err();
        assert!(matches!(err, AlayaError::NotFound(_)), "{err:?}");
    }

    /// AC-5: the CONTRADICTS pair behind the reversed supersession is stamped
    /// keep_both by `unsupersede`, in whichever direction it was detected, so
    /// the judge's apply path skips it. Other pairs of the memory are not.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_stamps_the_reversed_pair_so_the_judge_cannot_reapply_it() {
        let (a, b, c) = (h('a'), h('b'), h('c'));
        let ledger = Ledger::with(&[&a, &b, &c]);
        for pair in [(b.clone(), a.clone()), (a.clone(), c.clone())] {
            ledger.contradicts.borrow_mut().insert(pair, None);
        }
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "judge said so").await.unwrap();

        let out = unsupersede(&svc, &a).await;
        assert_eq!(
            out["contradictions_stamped"],
            serde_json::json!([[b, a]]),
            "{out}"
        );
        assert_eq!(
            ledger.stamped_by(&b, &a).as_deref(),
            Some(UNSUPERSEDE_RESOLVED_VIA)
        );
        assert_eq!(
            ledger.stamped_by(&a, &c),
            None,
            "an unrelated pair stays in the queue"
        );
    }

    /// A pair an operator already settled keeps that stamp, who and when
    /// included: the reversal only stamps an unresolved pair.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_keeps_an_operators_earlier_stamp_on_the_pair() {
        let (a, b) = (h('a'), h('b'));
        let ledger = Ledger::with(&[&a, &b]);
        ledger.contradicts.borrow_mut().insert(
            (a.clone(), b.clone()),
            Some(("operator:console".into(), 1.0)),
        );
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "overrode keep-both")
            .await
            .unwrap();

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert_eq!(out["contradictions_stamped"], serde_json::json!([]));
        assert_eq!(
            ledger.stamped_by(&a, &b).as_deref(),
            Some("operator:console")
        );
    }

    /// A concurrent supersede to the SAME survivor can write its edge after
    /// the first sweep; the marker check cannot tell it apart, so the
    /// reversal sweeps again once the marker is gone. No stale edge is left
    /// to keep the live memory's pairs out of the queue.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_sweeps_an_edge_written_inside_its_window() {
        let (a, b) = (h('a'), h('b'));
        let ledger = Ledger::with(&[&a, &b]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();
        *ledger.late_edge.borrow_mut() = Some((b.clone(), a.clone()));

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert!(ledger.edges().is_empty(), "{:?}", ledger.edges());
        assert_eq!(out["supersedes_edges_removed"], serde_json::json!([b, b]));
    }

    /// AC-3, middle link: in A→B→C, unsuperseding B restores B alone. A stays
    /// superseded by B (its marker and the B→A edge are untouched) and B's
    /// own edge from C goes.
    #[tokio::test(flavor = "current_thread")]
    async fn unsuperseding_a_middle_link_leaves_the_links_around_it() {
        let (a, b, c) = (h('a'), h('b'), h('c'));
        let ledger = Ledger::with(&[&a, &b, &c]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "a→b").await.unwrap();
        svc.memory_supersede(&b, &c, "b→c").await.unwrap();

        let out = unsupersede(&svc, &b).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert_eq!(ledger.marker(&a), Some(serde_json::json!(b)));
        assert_eq!(ledger.marker(&b), None);
        assert_eq!(ledger.edges(), [(b.clone(), a.clone())]);
        for (mode, seen) in ALL_MODES.iter().zip(visible(&svc).await) {
            assert_eq!(seen, set_of(&[&b, &c]), "{mode:?}");
        }
    }

    /// AC-3, survivor superseded since: A→B, then B→C. Unsuperseding A
    /// restores A without following the chain — B stays superseded by C.
    #[tokio::test(flavor = "current_thread")]
    async fn unsuperseding_a_memory_whose_survivor_was_superseded_since_leaves_the_survivor_alone()
    {
        let (a, b, c) = (h('a'), h('b'), h('c'));
        let ledger = Ledger::with(&[&a, &b, &c]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "a→b").await.unwrap();
        svc.memory_supersede(&b, &c, "b→c").await.unwrap();

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert_eq!(ledger.marker(&a), None);
        assert_eq!(ledger.marker(&b), Some(serde_json::json!(c)));
        assert_eq!(ledger.edges(), [(c.clone(), b.clone())]);
        for (mode, seen) in ALL_MODES.iter().zip(visible(&svc).await) {
            assert_eq!(seen, set_of(&[&a, &c]), "{mode:?}");
        }
    }

    /// A memory superseded again to another survivor holds two incoming
    /// SUPERSEDES edges. Both go — one left behind would keep its pairs out of
    /// the contradiction queue (`indegree(SUPERSEDES)`) though it is live —
    /// while an edge OUT of it (a memory it supersedes) stays.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_removes_every_incoming_supersedes_edge_and_no_outgoing_one() {
        let (a, b, c, d) = (h('a'), h('b'), h('c'), h('d'));
        let ledger = Ledger::with(&[&a, &b, &c, &d]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "first").await.unwrap();
        svc.memory_supersede(&a, &c, "second").await.unwrap();
        svc.memory_supersede(&d, &a, "a supersedes d")
            .await
            .unwrap();

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["superseded_by"], serde_json::json!(c), "{out}");
        let mut removed: Vec<String> =
            serde_json::from_value(out["supersedes_edges_removed"].clone()).unwrap();
        removed.sort();
        assert_eq!(removed, [b, c]);
        assert_eq!(ledger.edges(), [(a.clone(), d.clone())]);
        assert_eq!(ledger.marker(&d), Some(serde_json::json!(a)));
    }

    /// A graph failure aborts BEFORE the marker is touched: the memory stays
    /// superseded with its reason, nothing is logged, and the same call
    /// converges once the graph is back.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_graph_failure_leaves_it_superseded_and_a_retry_converges() {
        let (a, b) = (h('a'), h('b'));
        let ledger = Ledger::with(&[&a, &b]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();

        ledger.graph_down.set(true);
        let err = svc
            .memory_unsupersede(&a, "wrong merge", "operator:test")
            .await
            .unwrap_err();
        assert!(matches!(err, AlayaError::Graph(_)), "{err:?}");
        assert_eq!(ledger.marker(&a), Some(serde_json::json!(b)));
        assert_eq!(ledger.reasons.borrow()[&a], "duplicate");
        assert!(ledger.logs.borrow().is_empty());

        ledger.graph_down.set(false);
        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert_eq!(ledger.marker(&a), None);
        assert!(ledger.edges().is_empty());
    }

    /// Another writer re-supersedes the memory to C between the read and the
    /// conditional write: C's supersession stands, its edge (removed with the
    /// rest) is put back, and the caller is told rather than handed a success.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_racing_a_re_supersede_leaves_it_whole_and_says_so() {
        let (a, b, c) = (h('a'), h('b'), h('c'));
        let ledger = Ledger::with(&[&a, &b, &c]);
        ledger
            .contradicts
            .borrow_mut()
            .insert((a.clone(), b.clone()), None);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();
        *ledger.resupersede.borrow_mut() = Some(c.clone());

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["success"], false, "{out}");
        assert_eq!(out["status"], "superseded_by_changed");
        assert_eq!(out["superseded_by"], serde_json::json!(c));
        assert_eq!(ledger.marker(&a), Some(serde_json::json!(c)));
        assert_eq!(ledger.edges(), [(c, a.clone())]);
        assert_eq!(
            ledger.stamped_by(&a, &b),
            None,
            "its stamp is undone: no reversal stands behind it"
        );
        assert!(ledger.logs.borrow().is_empty());
    }

    /// An operator resolves the pair after this call stamped it, then the
    /// reversal finds the memory superseded to someone else. Undoing the
    /// call's stamp must not take the operator's with it.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_compensation_keeps_a_stamp_written_after_its_own() {
        let (a, b, c) = (h('a'), h('b'), h('c'));
        let ledger = Ledger::with(&[&a, &b, &c]);
        ledger
            .contradicts
            .borrow_mut()
            .insert((a.clone(), b.clone()), None);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();
        *ledger.resupersede.borrow_mut() = Some(c.clone());
        *ledger.operator_stamp.borrow_mut() = Some((a.clone(), b.clone()));

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "superseded_by_changed", "{out}");
        assert_eq!(
            ledger.stamped_by(&a, &b).as_deref(),
            Some("operator:console")
        );
    }

    /// Another supersede lands right after the reversal: marker, then edge.
    /// The second sweep takes that edge, so the re-read puts it back, and the
    /// caller is told the memory is superseded again.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_restores_the_edge_of_a_supersede_that_lands_right_after() {
        let (a, b, c) = (h('a'), h('b'), h('c'));
        let ledger = Ledger::with(&[&a, &b, &c]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();
        *ledger.resupersede_after.borrow_mut() = Some(c.clone());

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert_eq!(out["now_superseded_by"], serde_json::json!(c));
        assert_eq!(ledger.marker(&a), Some(serde_json::json!(c)));
        assert_eq!(ledger.edges(), [(c, a.clone())], "the new edge survives");
        assert_eq!(
            ledger.logs.borrow()[&a].len(),
            1,
            "the reversal itself landed"
        );
    }

    /// The marker is server-owned below the surfaces too: a patch that sets
    /// it or deletes it (null) is refused before storage is touched, so a
    /// reversal cannot be undone, nor a memory hidden, without the audit trail.
    #[tokio::test(flavor = "current_thread")]
    async fn patch_refuses_the_supersession_marker_set_or_deleted() {
        let (a, b) = (h('a'), h('b'));
        let ledger = Ledger::with(&[&a, &b]);
        let svc = ledger.service();
        svc.memory_supersede(&a, &b, "duplicate").await.unwrap();
        for v in [Value::Null, serde_json::json!(h('c'))] {
            let patch = PatchMemoryRequest {
                metadata: Some(HashMap::from([("superseded_by".to_string(), v)])),
                ..Default::default()
            };
            // LedgerVectors::patch_memory panics: reaching it fails the test.
            let err = svc.patch_memory(&a, &patch).await.unwrap_err();
            assert!(matches!(err, AlayaError::Validation(_)), "{err:?}");
        }
        assert_eq!(ledger.marker(&a), Some(serde_json::json!(b)));
    }

    /// A marker that names no memory (a legacy shape) still hides the memory
    /// — `is_superseded` tests presence — so it can still be reversed; there
    /// is just no pair to stamp.
    #[tokio::test(flavor = "current_thread")]
    async fn unsupersede_reverses_a_marker_that_names_no_memory() {
        let a = h('a');
        let ledger = Ledger::with(&[&a]);
        ledger.set_marker(&a, Some(Value::Null));
        let svc = ledger.service();
        assert!(visible(&svc).await.iter().all(|seen| seen.is_empty()));

        let out = unsupersede(&svc, &a).await;
        assert_eq!(out["status"], "unsuperseded", "{out}");
        assert_eq!(out["contradictions_stamped"], serde_json::json!([]));
        assert_eq!(ledger.marker(&a), None);
        for seen in visible(&svc).await {
            assert_eq!(seen, set_of(&[&a]));
        }
    }
}
