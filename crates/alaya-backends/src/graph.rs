//! GraphHttpClient — thin HTTP wrapper around alaya-bridge endpoints.
//!
//! Implements `GraphService`, `HebbianService`, and `ConsolidationService` traits
//! by mapping each method 1:1 to a bridge HTTP endpoint.

use std::collections::HashMap;

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use alaya_types::{
    AlayaError, Result,
    graph::{
        CoAccessPair, Contradiction, ContradictionRef, Direction, Edge, EdgeMeta, GraphStats,
        Neighbor, SystemRelationType, UserRelationType,
    },
};

use crate::{ConsolidationService, GraphService, HebbianService};

// ─── Client ──────────────────────────────────────────────────────────────────

pub struct GraphHttpClient {
    client: reqwest::Client,
    base_url: String,
}

impl GraphHttpClient {
    pub fn new(base_url: String, api_key: &str) -> Self {
        let mut headers = HeaderMap::new();
        if !api_key.is_empty() {
            let val = HeaderValue::from_str(&format!("Bearer {api_key}"))
                .expect("invalid api_key for HTTP header");
            headers.insert(AUTHORIZATION, val);
        }

        let builder = reqwest::Client::builder().default_headers(headers);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30));

        let client = builder.build().expect("failed to build reqwest client");

        Self { client, base_url }
    }
}

// ─── Response helper ─────────────────────────────────────────────────────────

async fn handle_response<T: serde::de::DeserializeOwned>(resp: reqwest::Response) -> Result<T> {
    if resp.status().is_success() {
        resp.json::<T>()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status.is_client_error() {
            Err(AlayaError::Validation(format!("Bridge: {status} {body}")))
        } else {
            Err(AlayaError::Graph(format!("Bridge: {status} {body}")))
        }
    }
}

/// Check that response status is 2xx, ignoring the body.
async fn check_success(resp: reqwest::Response) -> Result<()> {
    if resp.status().is_success() {
        Ok(())
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        if status.is_client_error() {
            Err(AlayaError::Validation(format!("Bridge: {status} {body}")))
        } else {
            Err(AlayaError::Graph(format!("Bridge: {status} {body}")))
        }
    }
}

/// Check that response status is 202 Accepted.
async fn check_accepted(resp: reqwest::Response) -> Result<()> {
    let status = resp.status();
    if status == reqwest::StatusCode::ACCEPTED {
        Ok(())
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(AlayaError::Graph(format!("Bridge: {status} {body}")))
    }
}

// ─── Request body types ──────────────────────────────────────────────────────

#[derive(Serialize)]
struct EnsureNodeReq<'a> {
    content_hash: &'a str,
    created_at: f64,
}

#[derive(Serialize)]
struct DeleteNodeReq<'a> {
    content_hash: &'a str,
}

#[derive(Serialize)]
struct CreateEdgeReq<'a> {
    source: &'a str,
    target: &'a str,
    relation_type: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    created_at: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    confidence: Option<f64>,
}

#[derive(Serialize)]
struct BatchCreateEdgeReq<'a> {
    edges: Vec<CreateEdgeReq<'a>>,
}

#[derive(Serialize)]
struct GetEdgesReq<'a> {
    content_hash: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    relation_type: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    direction: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    limit: Option<usize>,
}

#[derive(Serialize)]
struct DeleteEdgeReq<'a> {
    source: &'a str,
    target: &'a str,
    relation_type: &'a str,
}

#[derive(Serialize)]
struct CreateSystemEdgeReq<'a> {
    source: &'a str,
    target: &'a str,
    relation_type: &'a str,
    created_at: f64,
}

#[derive(Serialize)]
struct LimitReq {
    limit: usize,
}

#[derive(Serialize)]
struct HashesReq<'a> {
    hashes: &'a [&'a str],
}

#[derive(Serialize)]
struct NeighborsReq<'a> {
    content_hash: &'a str,
    max_hops: u8,
    min_weight: f64,
    limit: usize,
}

#[derive(Serialize)]
struct SpreadingReq<'a> {
    seed_hashes: &'a [&'a str],
    max_hops: u8,
    decay_factor: f64,
    min_activation: f64,
    limit: usize,
}

#[derive(Serialize)]
struct StrengthenReq<'a> {
    pairs: &'a [CoAccessPair],
}

#[derive(Serialize)]
struct DecayAllReq {
    decay_factor: f64,
    limit: usize,
}

#[derive(Serialize)]
struct DecayStaleReq {
    stale_before: f64,
    decay_factor: f64,
    limit: usize,
}

#[derive(Serialize)]
struct PruneReq {
    threshold: f64,
    limit: usize,
}

// ─── Response wrapper types ──────────────────────────────────────────────────

#[derive(Deserialize)]
struct CreatedResp {
    #[allow(dead_code)]
    created: bool,
}

#[derive(Deserialize)]
struct BatchCreatedResp {
    created: usize,
}

#[derive(Deserialize)]
struct DeletedResp {
    #[allow(dead_code)]
    deleted: bool,
}

#[derive(Deserialize)]
struct EdgesResp {
    edges: Vec<Edge>,
}

#[derive(Deserialize)]
struct ContradictionsResp {
    contradictions: Vec<Contradiction>,
}

#[derive(Deserialize)]
struct ContradictionsForResp {
    contradictions: HashMap<String, Vec<ContradictionRefRaw>>,
}

/// Raw contradiction ref from the bridge (may include extra fields).
#[derive(Deserialize)]
struct ContradictionRefRaw {
    memory_a_hash: String,
    memory_b_hash: String,
    confidence: Option<f64>,
}

#[derive(Deserialize)]
struct NeighborsResp {
    neighbors: Vec<Neighbor>,
}

#[derive(Deserialize)]
struct ActivationsResp {
    activations: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct BoostsResp {
    boosts: HashMap<String, f64>,
}

#[derive(Deserialize)]
struct OrphansResp {
    orphans: Vec<String>,
}

// ─── GraphService ────────────────────────────────────────────────────────────

#[async_trait(?Send)]
impl GraphService for GraphHttpClient {
    #[tracing::instrument(skip(self))]
    async fn ensure_node(&self, content_hash: &str, created_at: f64) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/nodes/ensure", self.base_url))
            .json(&EnsureNodeReq {
                content_hash,
                created_at,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        check_success(resp).await
    }

    async fn delete_node(&self, content_hash: &str) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/nodes/delete", self.base_url))
            .json(&DeleteNodeReq { content_hash })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        check_success(resp).await
    }

    async fn create_typed_edge(
        &self,
        src: &str,
        dst: &str,
        rel: UserRelationType,
        meta: alaya_types::graph::EdgeMeta,
    ) -> Result<bool> {
        let resp = self
            .client
            .post(format!("{}/edges/create", self.base_url))
            .json(&CreateEdgeReq {
                source: src,
                target: dst,
                relation_type: rel.cypher_label(),
                created_at: meta.created_at,
                confidence: meta.confidence,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: CreatedResp = handle_response(resp).await?;
        Ok(body.created)
    }

    #[tracing::instrument(skip(self, edges), fields(n = edges.len()))]
    async fn create_typed_edges_batch(
        &self,
        edges: &[(String, String, UserRelationType, EdgeMeta)],
    ) -> Result<usize> {
        if edges.is_empty() {
            return Ok(0);
        }

        let edge_reqs: Vec<CreateEdgeReq<'_>> = edges
            .iter()
            .map(|(src, dst, rel, meta)| CreateEdgeReq {
                source: src,
                target: dst,
                relation_type: rel.cypher_label(),
                created_at: meta.created_at,
                confidence: meta.confidence,
            })
            .collect();

        let resp = self
            .client
            .post(format!("{}/edges/create-batch", self.base_url))
            .json(&BatchCreateEdgeReq { edges: edge_reqs })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: BatchCreatedResp = handle_response(resp).await?;
        Ok(body.created)
    }

    async fn get_typed_edges(
        &self,
        hash: &str,
        rel: Option<UserRelationType>,
        dir: Direction,
        limit: usize,
    ) -> Result<Vec<Edge>> {
        let dir_str = match dir {
            Direction::Outgoing => "outgoing",
            Direction::Incoming => "incoming",
            Direction::Both => "both",
        };

        let resp = self
            .client
            .post(format!("{}/edges/get", self.base_url))
            .json(&GetEdgesReq {
                content_hash: hash,
                relation_type: rel.map(|r| r.cypher_label()),
                direction: Some(dir_str),
                limit: Some(limit),
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: EdgesResp = handle_response(resp).await?;
        Ok(body.edges)
    }

    async fn delete_typed_edge(&self, src: &str, dst: &str, rel: UserRelationType) -> Result<bool> {
        let resp = self
            .client
            .post(format!("{}/edges/delete", self.base_url))
            .json(&DeleteEdgeReq {
                source: src,
                target: dst,
                relation_type: rel.cypher_label(),
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: DeletedResp = handle_response(resp).await?;
        Ok(body.deleted)
    }

    async fn create_system_edge(
        &self,
        src: &str,
        dst: &str,
        rel: SystemRelationType,
        created_at: f64,
    ) -> Result<bool> {
        let resp = self
            .client
            .post(format!("{}/edges/create-system", self.base_url))
            .json(&CreateSystemEdgeReq {
                source: src,
                target: dst,
                relation_type: rel.cypher_label(),
                created_at,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: CreatedResp = handle_response(resp).await?;
        Ok(body.created)
    }

    async fn get_all_contradictions(&self, limit: usize) -> Result<Vec<Contradiction>> {
        let resp = self
            .client
            .post(format!("{}/contradictions/all", self.base_url))
            .json(&LimitReq { limit })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: ContradictionsResp = handle_response(resp).await?;
        Ok(body.contradictions)
    }

    async fn get_contradictions_for_hashes(
        &self,
        hashes: &[&str],
    ) -> Result<HashMap<String, Vec<ContradictionRef>>> {
        let resp = self
            .client
            .post(format!("{}/contradictions/for", self.base_url))
            .json(&HashesReq { hashes })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: ContradictionsForResp = handle_response(resp).await?;

        // Convert raw bridge response to ContradictionRef
        let result = body
            .contradictions
            .into_iter()
            .map(|(hash, refs)| {
                let converted = refs
                    .into_iter()
                    .map(|r| {
                        // The "other" hash is whichever one isn't the key
                        let contradicts_hash = if r.memory_a_hash == hash {
                            r.memory_b_hash
                        } else {
                            r.memory_a_hash
                        };
                        ContradictionRef {
                            contradicts_hash,
                            confidence: r.confidence,
                        }
                    })
                    .collect();
                (hash, converted)
            })
            .collect();

        Ok(result)
    }

    async fn get_neighbors(
        &self,
        hash: &str,
        max_hops: u8,
        min_weight: f64,
        limit: usize,
    ) -> Result<Vec<Neighbor>> {
        let resp = self
            .client
            .post(format!("{}/hebbian/neighbors", self.base_url))
            .json(&NeighborsReq {
                content_hash: hash,
                max_hops,
                min_weight,
                limit,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: NeighborsResp = handle_response(resp).await?;
        Ok(body.neighbors)
    }

    #[tracing::instrument(skip(self, seeds), fields(n_seeds = seeds.len(), max_hops))]
    async fn spreading_activation(
        &self,
        seeds: &[&str],
        max_hops: u8,
        decay: f64,
        min_activation: f64,
        limit: usize,
    ) -> Result<HashMap<String, f64>> {
        let resp = self
            .client
            .post(format!("{}/hebbian/spreading", self.base_url))
            .json(&SpreadingReq {
                seed_hashes: seeds,
                max_hops,
                decay_factor: decay,
                min_activation,
                limit,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: ActivationsResp = handle_response(resp).await?;
        Ok(body.activations)
    }

    #[tracing::instrument(skip(self, hashes), fields(n = hashes.len()))]
    async fn hebbian_boosts_within(&self, hashes: &[&str]) -> Result<HashMap<String, f64>> {
        let resp = self
            .client
            .post(format!("{}/hebbian/boosts-within", self.base_url))
            .json(&HashesReq { hashes })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: BoostsResp = handle_response(resp).await?;
        Ok(body.boosts)
    }

    #[tracing::instrument(skip(self))]
    async fn get_stats(&self) -> Result<GraphStats> {
        let resp = self
            .client
            .get(format!("{}/stats", self.base_url))
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        handle_response(resp).await
    }
}

// ─── HebbianService ──────────────────────────────────────────────────────────

#[async_trait(?Send)]
impl HebbianService for GraphHttpClient {
    async fn enqueue_strengthen(&self, pairs: &[CoAccessPair]) -> Result<()> {
        let resp = self
            .client
            .post(format!("{}/hebbian/strengthen", self.base_url))
            .json(&StrengthenReq { pairs })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        check_accepted(resp).await
    }
}

// ─── ConsolidationService ────────────────────────────────────────────────────

#[async_trait(?Send)]
impl ConsolidationService for GraphHttpClient {
    async fn decay_all_edges(&self, decay_factor: f64, limit: usize) -> Result<usize> {
        let resp = self
            .client
            .post(format!("{}/consolidation/decay-all", self.base_url))
            .json(&DecayAllReq {
                decay_factor,
                limit,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        // Bridge returns {"decayed": N}
        #[derive(Deserialize)]
        struct R {
            decayed: usize,
        }
        let body: R = handle_response(resp).await?;
        Ok(body.decayed)
    }

    async fn decay_stale_edges(
        &self,
        stale_before: f64,
        decay_factor: f64,
        limit: usize,
    ) -> Result<usize> {
        let resp = self
            .client
            .post(format!("{}/consolidation/decay-stale", self.base_url))
            .json(&DecayStaleReq {
                stale_before,
                decay_factor,
                limit,
            })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        #[derive(Deserialize)]
        struct R {
            decayed: usize,
        }
        let body: R = handle_response(resp).await?;
        Ok(body.decayed)
    }

    async fn prune_weak_edges(&self, threshold: f64, limit: usize) -> Result<usize> {
        let resp = self
            .client
            .post(format!("{}/consolidation/prune", self.base_url))
            .json(&PruneReq { threshold, limit })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        #[derive(Deserialize)]
        struct R {
            pruned: usize,
        }
        let body: R = handle_response(resp).await?;
        Ok(body.pruned)
    }

    async fn get_orphan_nodes(&self, limit: usize) -> Result<Vec<String>> {
        let resp = self
            .client
            .post(format!("{}/consolidation/orphans", self.base_url))
            .json(&LimitReq { limit })
            .send()
            .await
            .map_err(|e| AlayaError::Graph(e.to_string()))?;

        let body: OrphansResp = handle_response(resp).await?;
        Ok(body.orphans)
    }
}
