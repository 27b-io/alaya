//! QdrantClient — VectorStorage implementation over the Qdrant REST API.
//!
//! All calls use raw `reqwest` HTTP to stay WASM-compatible (no qdrant-client crate).

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use alaya_types::{
    AlayaError, Result,
    memory::{
        HealthStatus, Memory, MetadataUpdate, PatchMemoryRequest, ScoredMemory, ScrollResult,
    },
    search::PayloadFilter,
};

use crate::VectorStorage;

// ─── Client ──────────────────────────────────────────────────────────────────

pub struct QdrantClient {
    client: reqwest::Client,
    base_url: String,
    collection: String,
    tag_collection: String,
}

impl QdrantClient {
    pub fn new(base_url: String, collection: String, api_key: Option<String>) -> Self {
        let mut headers = HeaderMap::new();
        if let Some(key) = api_key {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {key}"))
                    .expect("invalid API key characters"),
            );
        }

        let builder = reqwest::Client::builder().default_headers(headers);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30));

        let client = builder.build().expect("failed to build reqwest client");

        let tag_collection = format!("{collection}_tags");
        Self {
            client,
            base_url,
            collection,
            tag_collection,
        }
    }
}

// ─── Access timestamp capping ────────────────────────────────────────────────

/// Maximum number of access timestamps to retain per memory.
/// The spaced_repetition module only needs recent inter-access intervals,
/// so 100 entries is more than sufficient.
const MAX_ACCESS_TIMESTAMPS: usize = 100;

/// Trim `timestamps` to keep only the most recent `max` entries.
fn cap_timestamps(timestamps: &mut Vec<f64>, max: usize) {
    if timestamps.len() > max {
        let drain_count = timestamps.len() - max;
        timestamps.drain(..drain_count);
    }
}

// ─── UUID generation (must match Python _hash_to_uuid) ──────────────────────

/// Convert a content_hash (64-char hex SHA-256) to a UUID string.
///
/// Python takes the first 32 hex chars and formats as UUID-4 style.
/// Must match exactly for data compatibility.
fn hash_to_uuid(content_hash: &str) -> Result<String> {
    if content_hash.len() < 32 {
        return Err(AlayaError::Validation(format!(
            "content_hash too short: {} chars (need 32)",
            content_hash.len()
        )));
    }
    let hex = &content_hash[..32];
    uuid::Uuid::parse_str(hex)
        .map(|u| u.to_string())
        .map_err(|e| AlayaError::Validation(format!("invalid content_hash hex: {e}")))
}

// ─── Filter construction ────────────────────────────────────────────────────

fn build_filter(filter: &PayloadFilter) -> Value {
    let mut must = Vec::new();
    let mut should = Vec::new();

    if let Some(ref mt) = filter.memory_type {
        must.push(json!({
            "key": "memory_type",
            "match": { "value": mt }
        }));
    }

    if let Some(ref tags) = filter.tags {
        if filter.tags_match_all {
            // AND — each tag must be present
            for tag in tags {
                must.push(json!({
                    "key": "tags",
                    "match": { "value": tag }
                }));
            }
        } else {
            // OR — any tag matches
            for tag in tags {
                should.push(json!({
                    "key": "tags",
                    "match": { "value": tag }
                }));
            }
        }
    }

    // Note: exclude_superseded is handled at the application layer (MemoryService)
    // after retrieval, not at the Qdrant filter level. Qdrant's nested payload
    // filtering for "field does not exist" is unreliable without explicit indexes.

    if let Some(min_trust) = filter.min_trust_score {
        must.push(json!({
            "key": "metadata.provenance.trust_score",
            "range": { "gte": min_trust }
        }));
    }

    let mut f = json!({});
    if !must.is_empty() {
        f["must"] = json!(must);
    }
    if !should.is_empty() {
        f["should"] = json!(should);
    }
    f
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse a Qdrant point payload into a Memory struct.
fn point_to_memory(point: &Value) -> Option<Memory> {
    let payload = point.get("payload")?;

    Some(Memory {
        content: payload.get("content")?.as_str()?.to_string(),
        content_hash: payload.get("content_hash")?.as_str()?.to_string(),
        tags: payload
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        memory_type: payload
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("note")
            .to_string(),
        metadata: payload
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        created_at: payload
            .get("created_at")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        updated_at: payload
            .get("updated_at")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        embedding: None, // Never return embeddings from Qdrant queries
        summary: payload
            .get("summary")
            .and_then(|v| v.as_str())
            .map(String::from),
        salience_score: payload
            .get("salience_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        access_count: payload
            .get("access_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        access_timestamps: payload
            .get("access_timestamps")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default(),
        emotional_valence: payload
            .get("emotional_valence")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        encoding_context: payload
            .get("encoding_context")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        provenance: payload
            .get("provenance")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        summary_embedding: payload
            .get("summary_embedding")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            }),
    })
}

fn point_to_scored(point: &Value) -> Option<ScoredMemory> {
    let score = point.get("score")?.as_f64()?;
    let memory = point_to_memory(point)?;
    Some(ScoredMemory { memory, score })
}

/// Build the payload JSON for upsert from a Memory struct.
fn memory_to_payload(memory: &Memory) -> Value {
    let mut payload = json!({
        "content": memory.content,
        "content_hash": memory.content_hash,
        "tags": memory.tags,
        "memory_type": memory.memory_type,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "salience_score": memory.salience_score,
        "access_count": memory.access_count,
        "access_timestamps": memory.access_timestamps,
    });

    if let Some(ref m) = memory.metadata {
        payload["metadata"] = json!(m);
    }
    if let Some(ref s) = memory.summary {
        payload["summary"] = json!(s);
    }
    if let Some(ref ev) = memory.emotional_valence {
        payload["emotional_valence"] = json!(ev);
    }
    if let Some(ref ec) = memory.encoding_context {
        payload["encoding_context"] = json!(ec);
    }
    if let Some(ref p) = memory.provenance {
        payload["provenance"] = json!(p);
    }
    if let Some(ref se) = memory.summary_embedding {
        payload["summary_embedding"] = json!(se);
    }

    payload
}

async fn qdrant_error(resp: reqwest::Response) -> AlayaError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    AlayaError::Storage(format!("Qdrant {status}: {body}"))
}

// ─── VectorStorage ──────────────────────────────────────────────────────────

#[async_trait(?Send)]
impl VectorStorage for QdrantClient {
    #[tracing::instrument(skip(self, memory), fields(hash = %memory.content_hash))]
    async fn store(&self, memory: &Memory) -> Result<(bool, String)> {
        let point_id = hash_to_uuid(&memory.content_hash)?;
        let embedding = memory
            .embedding
            .as_ref()
            .ok_or_else(|| AlayaError::Validation("memory has no embedding".into()))?;

        let body = json!({
            "points": [{
                "id": point_id,
                "vector": embedding,
                "payload": memory_to_payload(memory),
            }]
        });

        let resp = self
            .client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok((true, memory.content_hash.clone()))
    }

    async fn get_by_hash(&self, content_hash: &str) -> Result<Option<Memory>> {
        let point_id = hash_to_uuid(content_hash)?;

        let body = json!({
            "ids": [point_id],
            "with_payload": true,
            "with_vector": false,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        let points = data.result.unwrap_or_default();
        if points.is_empty() {
            return Ok(None);
        }

        Ok(point_to_memory(&points[0]))
    }

    async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>> {
        if hashes.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<String> = hashes
            .iter()
            .map(|h| hash_to_uuid(h))
            .collect::<Result<Vec<String>>>()?;

        let body = json!({
            "ids": ids,
            "with_payload": true,
            "with_vector": false,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        Ok(data
            .result
            .unwrap_or_default()
            .iter()
            .filter_map(point_to_memory)
            .collect())
    }

    async fn delete(&self, content_hash: &str) -> Result<bool> {
        let point_id = hash_to_uuid(content_hash)?;

        let body = json!({
            "points": [point_id]
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/delete?wait=true",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok(true)
    }

    async fn update_metadata(&self, content_hash: &str, updates: MetadataUpdate) -> Result<()> {
        let point_id = hash_to_uuid(content_hash)?;

        let mut payload = serde_json::Map::new();
        if let Some(ref sb) = updates.superseded_by {
            // Dot-path sets nested field without replacing the entire metadata object
            payload.insert("metadata.superseded_by".into(), json!(sb));
        }
        if let Some(ac) = updates.access_count {
            payload.insert("access_count".into(), json!(ac));
        }
        if let Some(ref extra) = updates.extra {
            for (k, v) in extra {
                payload.insert(k.clone(), v.clone());
            }
        }

        let body = json!({
            "payload": payload,
            "points": [point_id],
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/payload?wait=true",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok(())
    }

    #[tracing::instrument(skip(self, patch), fields(hash = %content_hash))]
    async fn patch_memory(&self, content_hash: &str, patch: &PatchMemoryRequest) -> Result<Memory> {
        if patch.is_empty() {
            return Err(AlayaError::Validation("patch is empty".into()));
        }

        let point_id = hash_to_uuid(content_hash)?;

        // Verify the memory exists before any writes. Also needed for metadata merge.
        let existing = self
            .get_by_hash(content_hash)
            .await?
            .ok_or_else(|| AlayaError::NotFound(format!("memory {content_hash} not found")))?;

        // Build set_payload with only the provided fields
        let mut payload = serde_json::Map::new();

        if let Some(ref tags) = patch.tags {
            payload.insert("tags".into(), json!(tags));
        }
        if let Some(ref summary) = patch.summary {
            payload.insert("summary".into(), json!(summary));
        }
        if let Some(ref memory_type) = patch.memory_type {
            payload.insert("memory_type".into(), json!(memory_type));
        }
        if let Some(ref se) = patch.summary_embedding {
            payload.insert("summary_embedding".into(), json!(se));
        }

        // Metadata merge: apply incoming keys, delete null keys
        if let Some(ref incoming) = patch.metadata {
            let mut merged = existing.metadata.clone().unwrap_or_default();
            for (k, v) in incoming {
                if v.is_null() {
                    merged.remove(k);
                } else {
                    merged.insert(k.clone(), v.clone());
                }
            }
            payload.insert("metadata".into(), json!(merged));
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        payload.insert("updated_at".into(), json!(now));

        let body = json!({
            "payload": payload,
            "points": [point_id],
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/payload?wait=true",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        // Return the updated memory
        self.get_by_hash(content_hash)
            .await?
            .ok_or_else(|| AlayaError::NotFound(format!("memory {content_hash} not found")))
    }

    #[tracing::instrument(skip(self, embedding, filters), fields(limit))]
    async fn search_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<PayloadFilter>,
    ) -> Result<Vec<ScoredMemory>> {
        let mut body = json!({
            "vector": embedding,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
        });

        if let Some(ref f) = filters {
            let filter = build_filter(f);
            if filter != json!({}) {
                body["filter"] = filter;
            }
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        Ok(data
            .result
            .unwrap_or_default()
            .iter()
            .filter_map(point_to_scored)
            .collect())
    }

    #[tracing::instrument(skip(self), fields(n_tags = tags.len(), match_all, limit))]
    async fn search_by_tags(
        &self,
        tags: &[&str],
        match_all: bool,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let filter = if match_all {
            let must: Vec<Value> = tags
                .iter()
                .map(|t| json!({"key": "tags", "match": {"value": t}}))
                .collect();
            json!({"must": must})
        } else {
            let should: Vec<Value> = tags
                .iter()
                .map(|t| json!({"key": "tags", "match": {"value": t}}))
                .collect();
            json!({"should": should})
        };

        let body = json!({
            "filter": filter,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/scroll",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<ScrollResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        let points = data.result.map(|r| r.points).unwrap_or_default();

        // Scroll doesn't have scores — assign 1.0 (tag-matched)
        Ok(points
            .iter()
            .filter_map(|p| {
                let memory = point_to_memory(p)?;
                Some(ScoredMemory { memory, score: 1.0 })
            })
            .collect())
    }

    #[tracing::instrument(skip(self, tag_embedding), fields(limit))]
    async fn search_similar_tags(
        &self,
        tag_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>> {
        let body = json!({
            "vector": tag_embedding,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "score_threshold": 0.5,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, self.tag_collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        Ok(data
            .result
            .unwrap_or_default()
            .iter()
            .filter_map(|p| {
                p.get("payload")
                    .and_then(|pl| pl.get("tag"))
                    .and_then(|t| t.as_str())
                    .map(String::from)
            })
            .collect())
    }

    async fn upsert_tags(&self, tags: &[(&str, Vec<f32>)]) -> Result<()> {
        if tags.is_empty() {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let points: Vec<Value> = tags
            .iter()
            .map(|(tag, embedding)| {
                let hex = format!("{:x}", Sha256::digest(tag.as_bytes()));
                let uuid = uuid::Uuid::parse_str(&hex[..32])
                    .expect("first 32 hex chars are always valid UUID input");
                json!({
                    "id": uuid.to_string(),
                    "vector": embedding,
                    "payload": { "tag": *tag, "created_at": now }
                })
            })
            .collect();

        let body = json!({ "points": points });
        let resp = self
            .client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.base_url, self.tag_collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok(())
    }

    async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult> {
        let mut body = json!({
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "order_by": {
                "key": "created_at",
                "direction": "desc"
            },
        });

        if let Some(off) = offset {
            body["offset"] = json!(off);
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/scroll",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<ScrollResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        let result = data.result.unwrap_or_default();
        let memories = result.points.iter().filter_map(point_to_memory).collect();

        let next_offset = result.next_page_offset.map(|v| v.to_string());

        Ok(ScrollResult {
            memories,
            next_offset,
        })
    }

    async fn get_recent(
        &self,
        limit: usize,
        offset: usize,
        memory_type: Option<&str>,
    ) -> Result<Vec<Memory>> {
        // Use Query API for native numeric offset (avoids fetching and
        // discarding `offset` records that the scroll API requires).
        let mut body = json!({
            "limit": limit,
            "offset": offset,
            "with_payload": true,
            "with_vector": false,
            "order_by": {
                "key": "created_at",
                "direction": "desc"
            },
        });

        if let Some(mt) = memory_type {
            body["filter"] = json!({
                "must": [{
                    "key": "memory_type",
                    "match": {"value": mt}
                }]
            });
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/query",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<QueryResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        let points = data.result.map(|r| r.points).unwrap_or_default();

        Ok(points.iter().filter_map(point_to_memory).collect())
    }

    #[tracing::instrument(skip(self))]
    async fn count(&self) -> Result<usize> {
        let body = json!({"exact": true});

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/count",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<CountResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        Ok(data.result.map(|r| r.count).unwrap_or(0))
    }

    #[tracing::instrument(skip(self))]
    async fn get_all_tags(&self) -> Result<Vec<String>> {
        // Scroll the tag collection to get all tags
        let mut all_tags = Vec::new();
        let mut offset: Option<Value> = None;

        loop {
            let mut body = json!({
                "limit": 100,
                "with_payload": true,
                "with_vector": false,
            });

            if let Some(ref off) = offset {
                body["offset"] = off.clone();
            }

            let resp = self
                .client
                .post(format!(
                    "{}/collections/{}/points/scroll",
                    self.base_url, self.tag_collection
                ))
                .json(&body)
                .send()
                .await
                .map_err(|e| AlayaError::Storage(e.to_string()))?;

            if !resp.status().is_success() {
                return Err(qdrant_error(resp).await);
            }

            let data: QdrantResponse<ScrollResponse> = resp
                .json()
                .await
                .map_err(|e| AlayaError::Storage(e.to_string()))?;

            let result = data.result.unwrap_or_default();
            for point in &result.points {
                if let Some(tag) = point
                    .get("payload")
                    .and_then(|p| p.get("tag"))
                    .and_then(|t| t.as_str())
                {
                    all_tags.push(tag.to_string());
                }
            }

            match result.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }

        Ok(all_tags)
    }

    async fn increment_access_count(&self, content_hash: &str) -> Result<()> {
        // Read current value, increment, write back
        let memory = self.get_by_hash(content_hash).await?;
        let Some(memory) = memory else {
            return Ok(());
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let new_count = memory.access_count + 1;
        let mut timestamps = memory.access_timestamps;
        timestamps.push(now);
        cap_timestamps(&mut timestamps, MAX_ACCESS_TIMESTAMPS);

        let point_id = hash_to_uuid(content_hash)?;
        let body = json!({
            "payload": {
                "access_count": new_count,
                "access_timestamps": timestamps,
            },
            "points": [point_id],
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/payload",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok(())
    }

    #[tracing::instrument(skip(self), fields(n = content_hashes.len()))]
    async fn increment_access_count_batch(&self, content_hashes: &[&str]) -> Result<()> {
        if content_hashes.is_empty() {
            return Ok(());
        }

        // Single batch GET instead of N individual GETs
        let memories = match self.get_batch(content_hashes).await {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("batch increment_access_count get_batch failed: {e}");
                return Ok(());
            }
        };
        if memories.is_empty() {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        // Fire individual set-payload calls (each point has different values)
        for memory in &memories {
            let new_count = memory.access_count + 1;
            let mut timestamps = memory.access_timestamps.clone();
            timestamps.push(now);
            cap_timestamps(&mut timestamps, MAX_ACCESS_TIMESTAMPS);

            let point_id = match hash_to_uuid(&memory.content_hash) {
                Ok(id) => id,
                Err(_) => continue,
            };

            let body = json!({
                "payload": {
                    "access_count": new_count,
                    "access_timestamps": timestamps,
                },
                "points": [point_id],
            });

            let resp = self
                .client
                .post(format!(
                    "{}/collections/{}/points/payload",
                    self.base_url, self.collection
                ))
                .json(&body)
                .send()
                .await;

            // Non-fatal: log and continue on individual failures
            match resp {
                Ok(r) if !r.status().is_success() => {
                    tracing::warn!(
                        hash = %memory.content_hash,
                        "batch increment_access_count set-payload failed"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        hash = %memory.content_hash,
                        error = %e,
                        "batch increment_access_count set-payload error"
                    );
                }
                _ => {}
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn health(&self) -> Result<HealthStatus> {
        let resp = self
            .client
            .get(format!("{}/collections/{}", self.base_url, self.collection))
            .send()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        if !resp.status().is_success() {
            return Ok(HealthStatus {
                status: "unhealthy".into(),
                backend: "qdrant".into(),
                details: None,
            });
        }

        let data: Value = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(e.to_string()))?;

        let status = data
            .pointer("/result/status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let points_count = data
            .pointer("/result/points_count")
            .and_then(|v| v.as_u64());

        let mut details = HashMap::new();
        if let Some(pc) = points_count {
            details.insert("points_count".into(), json!(pc));
        }

        Ok(HealthStatus {
            status: status.into(),
            backend: "qdrant".into(),
            details: if details.is_empty() {
                None
            } else {
                Some(details)
            },
        })
    }
}

// ─── Qdrant response wrapper types ──────────────────────────────────────────

#[derive(Deserialize)]
struct QdrantResponse<T> {
    #[allow(dead_code)]
    status: Option<String>,
    result: Option<T>,
}

#[derive(Deserialize, Default)]
struct ScrollResponse {
    #[serde(default)]
    points: Vec<Value>,
    next_page_offset: Option<Value>,
}

/// Response from POST /collections/{name}/points/query
#[derive(Deserialize, Default)]
struct QueryResponse {
    #[serde(default)]
    points: Vec<Value>,
}

#[derive(Deserialize)]
struct CountResponse {
    count: usize,
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_to_uuid_matches_python() {
        // Python: uuid.UUID("a" * 32) = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        let hash = "a".repeat(64);
        let uuid = hash_to_uuid(&hash).unwrap();
        assert_eq!(uuid, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
    }

    #[test]
    fn hash_to_uuid_real_hash() {
        // SHA-256 of "test" starts with "9f86d081..."
        let hash = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        let uuid = hash_to_uuid(hash).unwrap();
        assert_eq!(uuid, "9f86d081-884c-7d65-9a2f-eaa0c55ad015");
    }

    #[test]
    fn hash_to_uuid_short_input_returns_error() {
        let result = hash_to_uuid("abc");
        assert!(result.is_err());
    }

    #[test]
    fn hash_to_uuid_non_hex_returns_error() {
        let result = hash_to_uuid(&"zz".repeat(32));
        assert!(result.is_err());
    }

    #[test]
    fn build_filter_empty() {
        let f = PayloadFilter::default();
        let filter = build_filter(&f);
        assert_eq!(filter, json!({}));
    }

    #[test]
    fn build_filter_memory_type() {
        let f = PayloadFilter {
            memory_type: Some("note".into()),
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert!(filter["must"].is_array());
        assert_eq!(filter["must"][0]["key"], "memory_type");
    }

    #[test]
    fn build_filter_tags_or() {
        let f = PayloadFilter {
            tags: Some(vec!["rust".into(), "wasm".into()]),
            tags_match_all: false,
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert!(filter["should"].is_array());
        assert_eq!(filter["should"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn build_filter_tags_and() {
        let f = PayloadFilter {
            tags: Some(vec!["rust".into(), "wasm".into()]),
            tags_match_all: true,
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert!(filter["must"].is_array());
        assert_eq!(filter["must"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn build_filter_superseded_is_noop() {
        // Superseded filtering happens at application layer, not Qdrant
        let f = PayloadFilter {
            exclude_superseded: true,
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert_eq!(filter, json!({}));
    }

    #[test]
    fn build_filter_trust_score() {
        let f = PayloadFilter {
            min_trust_score: Some(0.5),
            ..Default::default()
        };
        let filter = build_filter(&f);
        let must = filter["must"].as_array().unwrap();
        assert!(must.iter().any(|c| c.get("range").is_some()));
    }

    #[test]
    fn point_to_memory_parses_full() {
        let point = json!({
            "id": "some-uuid",
            "payload": {
                "content": "hello world",
                "content_hash": "a".repeat(64),
                "tags": ["tag1", "tag2"],
                "memory_type": "note",
                "created_at": 1710432000.0,
                "updated_at": 1710432001.0,
                "salience_score": 0.5,
                "access_count": 3,
                "access_timestamps": [1.0, 2.0, 3.0],
                "summary": "a summary",
            }
        });
        let mem = point_to_memory(&point).unwrap();
        assert_eq!(mem.content, "hello world");
        assert_eq!(mem.tags.len(), 2);
        assert_eq!(mem.salience_score, 0.5);
        assert_eq!(mem.access_count, 3);
    }

    #[test]
    fn point_to_scored_includes_score() {
        let point = json!({
            "id": "uuid",
            "score": 0.95,
            "payload": {
                "content": "test",
                "content_hash": "b".repeat(64),
                "tags": [],
                "memory_type": "note",
                "created_at": 0.0,
                "updated_at": 0.0,
            }
        });
        let scored = point_to_scored(&point).unwrap();
        assert_eq!(scored.score, 0.95);
    }

    #[test]
    fn memory_to_payload_roundtrip() {
        let mem = Memory {
            content: "test content".into(),
            content_hash: "c".repeat(64),
            tags: vec!["t1".into()],
            memory_type: "note".into(),
            metadata: None,
            created_at: 100.0,
            updated_at: 200.0,
            embedding: None,
            summary: Some("summary".into()),
            salience_score: 0.7,
            access_count: 5,
            access_timestamps: vec![1.0, 2.0],
            emotional_valence: None,
            encoding_context: None,
            provenance: None,
            summary_embedding: None,
        };
        let payload = memory_to_payload(&mem);
        assert_eq!(payload["content"], "test content");
        assert_eq!(payload["salience_score"], 0.7);
        assert_eq!(payload["access_count"], 5);
        assert!(payload.get("metadata").is_none());
        assert_eq!(payload["summary"], "summary");
    }

    #[test]
    fn cap_timestamps_noop_below_limit() {
        let mut ts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0.0);
    }

    #[test]
    fn cap_timestamps_noop_at_limit() {
        let mut ts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 100);
        assert_eq!(ts[0], 0.0);
        assert_eq!(ts[99], 99.0);
    }

    #[test]
    fn cap_timestamps_trims_oldest_when_over_limit() {
        let mut ts: Vec<f64> = (0..150).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 100);
        // Oldest 50 removed, first remaining is 50.0
        assert_eq!(ts[0], 50.0);
        assert_eq!(ts[99], 149.0);
    }

    #[test]
    fn cap_timestamps_trims_one_over() {
        let mut ts: Vec<f64> = (0..101).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 100);
        assert_eq!(ts[0], 1.0);
    }

    #[test]
    fn cap_timestamps_empty_vec() {
        let mut ts: Vec<f64> = vec![];
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert!(ts.is_empty());
    }
}
