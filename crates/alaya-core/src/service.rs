//! MemoryService — orchestrates all 9 MCP tools across backends.
//!
//! This is the core business logic layer. Each public method corresponds to
//! one MCP tool. All backend calls go through trait abstractions.

use std::cell::RefCell;
use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

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
        Some(StringOrVec::Str(s)) => dedup(s.split(',').map(String::from)),
    })
}
use tracing;

use alaya_backends::{
    ConsolidationService, EmbeddingProvider, GraphService, HebbianService, SummaryProvider,
    VectorStorage,
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

pub struct MemoryService {
    pub vectors: Box<dyn VectorStorage>,
    pub embeddings: Box<dyn EmbeddingProvider>,
    pub graph: Box<dyn GraphService>,
    pub hebbian: Box<dyn HebbianService>,
    pub consolidation: Box<dyn ConsolidationService>,
    /// Optional summary generator. When set, summaries are auto-generated
    /// fire-and-forget after store when the caller omits one.
    pub summary: Option<Box<dyn SummaryProvider>>,
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
            tag_cache: RefCell::new(None),
            clock: current_timestamp,
        }
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
            tag_cache: RefCell::new(None),
            clock,
        }
    }

    // ─── Tool 1: store_memory ───────────────────────────────────────────

    #[tracing::instrument(skip(self, params), fields(content_len = params.content.len()))]
    pub async fn store_memory(&self, params: StoreParams) -> Result<HashMap<String, Value>> {
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
            let dedup_filter = PayloadFilter {
                exclude_superseded: true,
                ..Default::default()
            };
            let similar = self
                .vectors
                .search_by_vector(&embedding, 1, Some(dedup_filter))
                .await
                .unwrap_or_default();

            if let Some(top) = similar.first()
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

        // Store in vector DB
        let (created, _) = self.vectors.store(&memory).await?;

        // Invalidate tag cache — new tags should appear in keyword extraction immediately
        if !tags.is_empty() {
            *self.tag_cache.borrow_mut() = None;

            // Index new tags into the tag embedding collection (non-fatal).
            // Uses the same embedding we'd generate for any passage — tags are
            // short text, but the embedding model handles them fine.
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

        // Create graph node (non-fatal)
        if let Err(e) = self.graph.ensure_node(&content_hash, now).await {
            tracing::warn!("graph ensure_node failed (non-fatal): {e}");
        }

        // Interference detection: search for similar content
        let mut contradiction_signals = Vec::new();
        let interference_filter = PayloadFilter {
            exclude_superseded: true,
            ..Default::default()
        };
        if let Ok(similar) = self
            .vectors
            .search_by_vector(&embedding, 10, Some(interference_filter))
            .await
        {
            let mut edges_to_create: Vec<(String, String, UserRelationType, EdgeMeta)> = Vec::new();

            for scored in &similar {
                if scored.memory.content_hash == content_hash {
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

    // ─── Tool 2: search ─────────────────────────────────────────────────

    #[tracing::instrument(skip(self, params), fields(mode = ?params.mode))]
    pub async fn search(&self, params: SearchParams) -> Result<Value> {
        if params.page_size == 0 {
            return Err(AlayaError::Validation("page_size must be > 0".into()));
        }
        let mode_str = format!("{:?}", params.mode).to_lowercase();
        let mut result = match params.mode {
            SearchMode::Hybrid => self.search_hybrid(&params).await?,
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
    async fn search_hybrid(&self, params: &SearchParams) -> Result<Value> {
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
            exclude_superseded: !params.include_superseded,
            min_trust_score: params.min_trust_score,
            ..Default::default()
        };

        // Stage 2: Vector search + semantic tag pipeline run concurrently.
        // search_similar_tags→search_by_tags chains inside one branch so the
        // 44-64ms semantic search overlaps with search_by_vector.
        let (vector_results, semantic_tag_results) = {
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

        let mut scored_results: Vec<(String, f64)> = fused
            .iter()
            .map(|(hash, _rrf, display_score)| {
                let mut score = *display_score;

                // Salience boost
                if let Some(sm) = memory_map.get(hash) {
                    score = salience::apply_salience_boost(
                        score,
                        sm.memory.salience_score,
                        BOOST_SALIENCE,
                    );

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

                // Cap at 1.0
                (hash.clone(), score.min(1.0))
            })
            .collect();

        scored_results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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

        // Stage 6: Enrich — side-effect writes run concurrently
        let page_hashes: Vec<&str> = page_results.iter().map(|(h, _)| h.as_str()).collect();

        {
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
                Some(format_memory_result(&sm.memory, *score, params.output))
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
                if !params.include_superseded
                    && m.metadata
                        .as_ref()
                        .and_then(|md| md.get("superseded_by"))
                        .is_some()
                {
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

        let filter = PayloadFilter {
            exclude_superseded: !params.include_superseded,
            ..Default::default()
        };

        let results = self
            .vectors
            .search_by_vector(&query_embedding, params.k, Some(filter))
            .await?;

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
        let tags = params.tags.as_deref().unwrap_or_default();
        if tags.is_empty() {
            return Err(AlayaError::Validation("tags required for tag mode".into()));
        }

        let tag_refs: Vec<&str> = tags.iter().map(|s| s.as_str()).collect();
        let results = self
            .vectors
            .search_by_tags(&tag_refs, params.match_all, params.page_size * params.page)
            .await?;

        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let total = results.len();
        let page: Vec<Value> = results
            .iter()
            .skip(offset)
            .take(params.page_size)
            .map(|sm| format_memory_result(&sm.memory, sm.score, params.output))
            .collect();

        let total_pages = total.div_ceil(params.page_size);

        Ok(serde_json::json!({
            "page": params.page,
            "total": total,
            "page_size": params.page_size,
            "has_more": params.page < total_pages,
            "total_pages": total_pages,
            "results": page,
        }))
    }

    #[tracing::instrument(skip(self, params))]
    async fn search_recent(&self, params: &SearchParams) -> Result<Value> {
        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let results = self
            .vectors
            .get_recent(params.page_size + 1, offset, params.memory_type.as_deref())
            .await?;

        let has_more = results.len() > params.page_size;
        let items: Vec<Value> = results
            .iter()
            .take(params.page_size)
            .map(|m| format_memory_result(m, 1.0, params.output))
            .collect();

        Ok(serde_json::json!({
            "page": params.page,
            "page_size": params.page_size,
            "has_more": has_more,
            "results": items,
        }))
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

        // Verify both exist (single batch GET instead of 2 sequential calls)
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

        // Update metadata on old memory
        let mut extra = HashMap::new();
        extra.insert("supersession_reason".into(), serde_json::json!(reason));
        self.vectors
            .update_metadata(
                old_hash,
                MetadataUpdate {
                    superseded_by: Some(new_hash.to_string()),
                    extra: Some(extra),
                    ..Default::default()
                },
            )
            .await?;

        // Create SUPERSEDES graph edge
        let now = (self.clock)();
        if let Err(e) = self
            .graph
            .create_system_edge(new_hash, old_hash, SystemRelationType::Supersedes, now)
            .await
        {
            tracing::warn!("failed to create SUPERSEDES edge: {e}");
        }

        Ok(serde_json::json!({
            "success": true,
            "superseded": old_hash,
            "superseded_by": new_hash,
            "reason": reason,
        }))
    }

    // ─── Tool 7: memory_contradictions ──────────────────────────────────

    #[tracing::instrument(skip(self))]
    pub async fn memory_contradictions(&self, limit: usize) -> Result<Value> {
        let pairs = self.graph.get_all_contradictions(limit).await?;

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

        let mut enriched: Vec<Value> = Vec::new();
        for pair in &pairs {
            let a = lookup.get(pair.memory_a_hash.as_str());
            let b = lookup.get(pair.memory_b_hash.as_str());

            enriched.push(serde_json::json!({
                "memory_a_hash": pair.memory_a_hash,
                "memory_b_hash": pair.memory_b_hash,
                "confidence": pair.confidence,
                "memory_a_content": a.map(|m| {
                    m.summary.clone().unwrap_or_else(|| truncate(&m.content, 200))
                }),
                "memory_b_content": b.map(|m| {
                    m.summary.clone().unwrap_or_else(|| truncate(&m.content, 200))
                }),
                "memory_a_superseded": a.and_then(|m| {
                    m.metadata.as_ref()?.get("superseded_by")
                }).is_some(),
                "memory_b_superseded": b.and_then(|m| {
                    m.metadata.as_ref()?.get("superseded_by")
                }).is_some(),
            }));
        }

        Ok(serde_json::json!({
            "success": true,
            "pairs": enriched,
            "total": enriched.len(),
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
                if m.metadata
                    .as_ref()
                    .and_then(|md| md.get("superseded_by"))
                    .is_some()
                {
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

        let mut superseded = Vec::new();
        let mut errors = Vec::new();

        for &dup_hash in duplicate_hashes {
            match self
                .memory_supersede(dup_hash, canonical_hash, reason)
                .await
            {
                Ok(_) => superseded.push(dup_hash.to_string()),
                Err(e) => errors.push(serde_json::json!({
                    "hash": dup_hash,
                    "error": e.safe_message(),
                })),
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
}

// ─── Helpers ────────────────────────────────────────────────────────────────

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
        ConsolidationService, EmbeddingProvider, GraphService, HebbianService, VectorStorage,
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
        async fn get_recent(&self, _l: usize, _o: usize, _t: Option<&str>) -> Result<Vec<Memory>> {
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
        async fn get_all_contradictions(&self, _l: usize) -> Result<Vec<Contradiction>> {
            Ok(vec![])
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
        async fn get_recent(&self, _l: usize, _o: usize, _t: Option<&str>) -> Result<Vec<Memory>> {
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
        async fn get_all_contradictions(&self, _l: usize) -> Result<Vec<Contradiction>> {
            Ok(vec![])
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
        async fn get_recent(&self, _l: usize, _o: usize, _t: Option<&str>) -> Result<Vec<Memory>> {
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
        async fn get_recent(&self, _l: usize, _o: usize, _t: Option<&str>) -> Result<Vec<Memory>> {
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
        async fn get_all_contradictions(&self, _l: usize) -> Result<Vec<Contradiction>> {
            Ok(vec![])
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
            &neighbor_hash,
            result_hashes,
        );
    }
}
