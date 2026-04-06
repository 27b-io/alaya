//! MemoryService — orchestrates all 9 MCP tools across backends.
//!
//! This is the core business logic layer. Each public method corresponds to
//! one MCP tool. All backend calls go through trait abstractions.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing;

use alaya_backends::{
    ConsolidationService, EmbeddingProvider, GraphService, HebbianService, VectorStorage,
};
use alaya_types::{
    AlayaError, Result,
    graph::{CoAccessPair, Direction, EdgeMeta, SystemRelationType, UserRelationType},
    memory::{Memory, MetadataUpdate, ScoredMemory},
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

pub struct MemoryService {
    pub vectors: Box<dyn VectorStorage>,
    pub embeddings: Box<dyn EmbeddingProvider>,
    pub graph: Box<dyn GraphService>,
    pub hebbian: Box<dyn HebbianService>,
    pub consolidation: Box<dyn ConsolidationService>,
}

impl MemoryService {
    pub fn new(
        vectors: Box<dyn VectorStorage>,
        embeddings: Box<dyn EmbeddingProvider>,
        graph: Box<dyn GraphService>,
        hebbian: Box<dyn HebbianService>,
        consolidation: Box<dyn ConsolidationService>,
    ) -> Self {
        Self {
            vectors,
            embeddings,
            graph,
            hebbian,
            consolidation,
        }
    }

    // ─── Tool 1: store_memory ───────────────────────────────────────────

    pub async fn store_memory(&self, params: StoreParams) -> Result<HashMap<String, Value>> {
        if params.content.is_empty() {
            return Err(AlayaError::Validation("content cannot be empty".into()));
        }

        let now = current_timestamp();
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
        let embeddings = self
            .embeddings
            .embed_batch(&[params.content.as_str()], PromptName::Passage)
            .await?;
        let embedding = embeddings.into_iter().next().ok_or_else(|| {
            AlayaError::Embedding("embedding service returned empty result".into())
        })?;

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
        };

        // Store in vector DB
        let (created, _) = self.vectors.store(&memory).await?;

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
                    // Create CONTRADICTS edge
                    if let Err(e) = self
                        .graph
                        .create_typed_edge(
                            &content_hash,
                            &signal.existing_hash,
                            UserRelationType::Contradicts,
                            EdgeMeta {
                                created_at: Some(now),
                                confidence: Some(signal.confidence),
                            },
                        )
                        .await
                    {
                        tracing::warn!("failed to create CONTRADICTS edge: {e}");
                    }
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

                if let Err(e) = self
                    .graph
                    .create_typed_edge(
                        &content_hash,
                        &scored.memory.content_hash,
                        UserRelationType::RelatesTo,
                        EdgeMeta {
                            created_at: Some(now),
                            confidence: None,
                        },
                    )
                    .await
                {
                    tracing::warn!("failed to create RELATES_TO edge: {e}");
                }
            }
        }

        let mut result = HashMap::new();
        result.insert("success".into(), serde_json::json!(true));
        result.insert("content_hash".into(), serde_json::json!(content_hash));
        result.insert(
            "message".into(),
            serde_json::json!(if created {
                "Memory stored successfully"
            } else {
                "Memory updated"
            }),
        );

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

    pub async fn search(&self, params: SearchParams) -> Result<Value> {
        if params.page_size == 0 {
            return Err(AlayaError::Validation("page_size must be > 0".into()));
        }
        match params.mode {
            SearchMode::Hybrid => self.search_hybrid(&params).await,
            SearchMode::Scan => self.search_scan(&params).await,
            SearchMode::Similar => self.search_similar(&params).await,
            SearchMode::Tag => self.search_tag(&params).await,
            SearchMode::Recent => self.search_recent(&params).await,
        }
    }

    async fn search_hybrid(&self, params: &SearchParams) -> Result<Value> {
        if params.query.trim().is_empty() {
            return Err(AlayaError::Validation(
                "query is required for hybrid mode".into(),
            ));
        }
        let now = current_timestamp();

        // Stage 1: Prepare
        let all_tags = self.vectors.get_all_tags().await.unwrap_or_default();
        let tag_set: std::collections::HashSet<String> = all_tags.into_iter().collect();
        let keywords = hybrid_search::extract_query_keywords(&params.query, Some(&tag_set));
        let corpus_size = self.vectors.count().await.unwrap_or(0);
        let alpha = hybrid_search::get_adaptive_alpha(corpus_size, keywords.len());

        // Stage 2: Fan-out — embed + vector search + tag search
        let query_embedding = self
            .embeddings
            .embed_batch(&[params.query.as_str()], PromptName::Query)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| AlayaError::Embedding("empty embedding result".into()))?;

        let fetch_size = std::cmp::min(
            std::cmp::max(params.page_size * 3, params.page * params.page_size),
            100,
        );

        let filter = PayloadFilter {
            memory_type: params.memory_type.clone(),
            exclude_superseded: !params.include_superseded,
            min_trust_score: params.min_trust_score,
            ..Default::default()
        };

        let vector_results = self
            .vectors
            .search_by_vector(&query_embedding, fetch_size, Some(filter))
            .await?;

        let tag_results = if !keywords.is_empty() {
            let keyword_refs: Vec<&str> = keywords.iter().map(|s| s.as_str()).collect();
            self.vectors
                .search_by_tags(&keyword_refs, false, fetch_size)
                .await
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        // Stage 3: Fuse (RRF)
        let v_tuples: Vec<(String, f64)> = vector_results
            .iter()
            .map(|s| (s.memory.content_hash.clone(), s.score))
            .collect();
        let t_tuples: Vec<(String, f64)> = tag_results
            .iter()
            .map(|s| (s.memory.content_hash.clone(), s.score))
            .collect();

        let fused = hybrid_search::combine_results_rrf(&v_tuples, &t_tuples, alpha, RRF_K);

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

        // Stage 4: Boost
        let result_hashes: Vec<&str> = fused.iter().take(20).map(|(h, _, _)| h.as_str()).collect();

        // Graph boosts (non-fatal)
        let spreading = self
            .graph
            .spreading_activation(
                &result_hashes[..std::cmp::min(5, result_hashes.len())],
                2,
                0.5,
                0.05,
                50,
            )
            .await
            .unwrap_or_default();

        let hebbian_boosts = self
            .graph
            .hebbian_boosts_within(&result_hashes)
            .await
            .unwrap_or_default();

        let mut scored_results: Vec<(String, f64)> = fused
            .iter()
            .map(|(hash, _rrf, display_score)| {
                let mut score = *display_score;

                // Salience boost
                if let Some(sm) = memory_map.get(hash) {
                    score = salience::apply_salience_boost(score, sm.memory.salience_score, 0.15);

                    // Spacing boost
                    let sq =
                        spaced_repetition::compute_spacing_quality(&sm.memory.access_timestamps);
                    score = spaced_repetition::apply_spacing_boost(score, sq, 0.1);

                    // Encoding context boost
                    if let (Some(stored_ctx), Some(query_ctx)) =
                        (&sm.memory.encoding_context, &params.encoding_context)
                    {
                        let ctx_sim =
                            encoding_context::compute_context_similarity(stored_ctx, query_ctx);
                        score = encoding_context::apply_context_boost(score, ctx_sim, 0.1);
                    }

                    // Recency decay
                    score =
                        hybrid_search::apply_recency_decay(score, sm.memory.created_at, now, 0.01);
                }

                // Graph boosts
                if let Some(&activation) = spreading.get(hash) {
                    score *= 1.0 + 0.1 * activation;
                }
                if let Some(&boost) = hebbian_boosts.get(hash) {
                    score *= 1.0 + 0.1 * boost;
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

        // Stage 6: Enrich — fire-and-forget side effects
        let page_hashes: Vec<&str> = page_results.iter().map(|(h, _)| h.as_str()).collect();

        // Increment access counts (non-fatal)
        for hash in &page_hashes {
            let _ = self.vectors.increment_access_count(hash).await;
        }

        // Hebbian co-access (non-fatal)
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

    async fn search_scan(&self, params: &SearchParams) -> Result<Value> {
        let fetch_size = params.page * params.page_size;
        let scroll = self.vectors.get_all(fetch_size, None).await?;
        let has_more_in_db = scroll.next_offset.is_some();
        let mut memories = scroll.memories;

        // Filter superseded at application layer
        if !params.include_superseded {
            memories.retain(|m| {
                m.metadata
                    .as_ref()
                    .and_then(|md| md.get("superseded_by"))
                    .is_none()
            });
        }

        // Apply memory_type filter
        if let Some(ref mt) = params.memory_type {
            memories.retain(|m| m.memory_type == *mt);
        }

        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let page: Vec<Value> = memories
            .iter()
            .skip(offset)
            .take(params.page_size)
            .map(|m| format_memory_result(m, 1.0, params.output))
            .collect();

        // has_more: either we have unfetched data in Qdrant, or there are
        // more items in our filtered set beyond this page
        let has_more = has_more_in_db || (offset + params.page_size) < memories.len();

        Ok(serde_json::json!({
            "page": params.page,
            "page_size": params.page_size,
            "has_more": has_more,
            "count": page.len(),
            "results": page,
        }))
    }

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

    async fn search_recent(&self, params: &SearchParams) -> Result<Value> {
        let offset = (params.page.saturating_sub(1)) * params.page_size;
        let results = self
            .vectors
            .get_recent(params.page_size, offset, params.memory_type.as_deref())
            .await?;

        let items: Vec<Value> = results
            .iter()
            .map(|m| format_memory_result(m, 1.0, params.output))
            .collect();

        // has_more: if we got a full page, there are likely more
        let has_more = items.len() == params.page_size;

        Ok(serde_json::json!({
            "page": params.page,
            "page_size": params.page_size,
            "has_more": has_more,
            "results": items,
        }))
    }

    // ─── Tool 3: delete_memory ──────────────────────────────────────────

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

    // ─── Tool 4: check_database_health ──────────────────────────────────

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

                let now = current_timestamp();
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

        // Verify both exist
        let old = self.vectors.get_by_hash(old_hash).await?;
        if old.is_none() {
            return Err(AlayaError::Validation(format!(
                "old memory not found: {old_hash}"
            )));
        }
        let new = self.vectors.get_by_hash(new_hash).await?;
        if new.is_none() {
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
        let now = current_timestamp();
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
                "memory_a_content": a.map(|m| truncate(&m.content, 200)),
                "memory_b_content": b.map(|m| truncate(&m.content, 200)),
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

    pub async fn find_duplicates(
        &self,
        similarity_threshold: f64,
        limit: usize,
        strategy: CanonicalStrategy,
    ) -> Result<Value> {
        let group_limit = limit.min(500);
        let scan_limit: usize = 500;
        let scroll_page: usize = 100;

        // Paginated scroll to collect up to scan_limit memories
        let mut memories: Vec<Memory> = Vec::new();
        let mut offset: Option<String> = None;
        while memories.len() < scan_limit {
            let batch_size = scroll_page.min(scan_limit - memories.len());
            let scroll = self.vectors.get_all(batch_size, offset.as_deref()).await?;
            let batch_empty = scroll.memories.is_empty();
            memories.extend(scroll.memories);
            offset = scroll.next_offset;
            if batch_empty || offset.is_none() {
                break;
            }
        }

        // Filter superseded
        memories.retain(|m| {
            m.metadata
                .as_ref()
                .and_then(|md| md.get("superseded_by"))
                .is_none()
        });

        let scanned = memories.len();

        if memories.len() < 2 {
            return Ok(serde_json::json!({
                "success": true,
                "groups": [],
                "total_memories_scanned": scanned,
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
            "total_memories_scanned": scanned,
            "total_duplicates_found": total_dups,
        }))
    }

    // ─── Tool 9: merge_duplicates ───────────────────────────────────────

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
        }
        OutputMode::Summary => {
            obj.insert("summary".into(), serde_json::json!(memory.summary));
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
}
