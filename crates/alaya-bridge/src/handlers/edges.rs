//! Edge handlers — POST /edges/create, POST /edges/create-batch, POST /edges/get, POST /edges/delete

use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::{
    graph::{Direction, Edge, SystemRelationType, UserRelationType},
    memory::validate_content_hash,
};

use crate::{AppState, cypher, handlers::exec_query};

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct CreateEdgeRequest {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub created_at: Option<f64>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct GetEdgesRequest {
    pub content_hash: String,
    pub relation_type: Option<String>,
    /// "outgoing", "incoming", or "both" (default "both")
    pub direction: Option<String>,
    pub limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct DeleteEdgeRequest {
    pub source: String,
    pub target: String,
    pub relation_type: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateSystemEdgeRequest {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub created_at: f64,
}

#[derive(Debug, Deserialize)]
pub struct BatchCreateEdgeRequest {
    pub edges: Vec<CreateEdgeRequest>,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Parse a relation_type string into a `UserRelationType`.
/// Rejects system relation types (e.g. SUPERSEDES) and unknown values.
fn parse_user_relation(s: &str) -> Result<UserRelationType, StatusCode> {
    match s {
        "RELATES_TO" => Ok(UserRelationType::RelatesTo),
        "PRECEDES" => Ok(UserRelationType::Precedes),
        "CONTRADICTS" => Ok(UserRelationType::Contradicts),
        // Explicitly reject system relations
        "SUPERSEDES" | "HEBBIAN" => Err(StatusCode::UNPROCESSABLE_ENTITY),
        _ => Err(StatusCode::UNPROCESSABLE_ENTITY),
    }
}

fn parse_direction(s: &str) -> Result<Direction, StatusCode> {
    match s {
        "outgoing" => Ok(Direction::Outgoing),
        "incoming" => Ok(Direction::Incoming),
        "both" => Ok(Direction::Both),
        _ => Err(StatusCode::UNPROCESSABLE_ENTITY),
    }
}

/// Parse a system relation_type string into a `SystemRelationType`.
fn parse_system_relation(s: &str) -> Result<SystemRelationType, StatusCode> {
    match s {
        "SUPERSEDES" => Ok(SystemRelationType::Supersedes),
        _ => Err(StatusCode::UNPROCESSABLE_ENTITY),
    }
}

/// Collect all `UserRelationType` variants.
fn all_user_relation_types() -> &'static [UserRelationType] {
    &[
        UserRelationType::RelatesTo,
        UserRelationType::Precedes,
        UserRelationType::Contradicts,
    ]
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /edges/create
pub async fn create(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateEdgeRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.source) || !validate_content_hash(&req.target) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let rel = parse_user_relation(&req.relation_type)?;
    let ts = req.created_at.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    });

    let (cypher, params, readonly) =
        cypher::create_typed_edge(&req.source, &req.target, rel, ts, req.confidence);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // RETURN count(e) — 1 means created (ON CREATE fired), 0 means already existed
    let count = result.count().unwrap_or(0);
    Ok(Json(json!({ "created": count > 0 })))
}

/// POST /edges/create-batch
///
/// Validate all edges up front, then execute each Cypher query sequentially.
/// Single HTTP call from the client replaces N individual calls.
pub async fn create_batch(
    State(state): State<Arc<AppState>>,
    Json(req): Json<BatchCreateEdgeRequest>,
) -> Result<Json<Value>, StatusCode> {
    if req.edges.is_empty() {
        return Ok(Json(json!({ "created": 0 })));
    }

    // Validate all edges before executing any
    let mut queries = Vec::with_capacity(req.edges.len());
    for edge in &req.edges {
        if !validate_content_hash(&edge.source) || !validate_content_hash(&edge.target) {
            return Err(StatusCode::UNPROCESSABLE_ENTITY);
        }
        let rel = parse_user_relation(&edge.relation_type)?;
        let ts = edge.created_at.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
        });
        queries.push(cypher::create_typed_edge(
            &edge.source,
            &edge.target,
            rel,
            ts,
            edge.confidence,
        ));
    }

    // Execute all queries (each is a Redis round-trip, but within the bridge process)
    let mut created = 0usize;
    for (cypher, params, readonly) in queries {
        let result = exec_query(&state, &cypher, params, readonly).await?;
        if result.count().unwrap_or(0) > 0 {
            created += 1;
        }
    }

    Ok(Json(json!({ "created": created })))
}
pub async fn get(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GetEdgesRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.content_hash) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let direction = req
        .direction
        .as_deref()
        .map(parse_direction)
        .transpose()?
        .unwrap_or(Direction::Both);

    let limit = req.limit.unwrap_or(100).clamp(1, 500) as u32;

    let mut edges: Vec<Edge> = Vec::new();

    match req.relation_type.as_deref() {
        Some(rt) => {
            // Single type — one query
            let rel = parse_user_relation(rt)?;
            let (cypher, params, readonly) =
                cypher::get_typed_edges(&req.content_hash, rel, direction, limit);
            let result = exec_query(&state, &cypher, params, readonly).await?;

            for row in &result.result_set {
                if row.len() < 2 {
                    continue;
                }
                edges.push(Edge {
                    source: row[0].as_str().unwrap_or("").to_string(),
                    target: row[1].as_str().unwrap_or("").to_string(),
                    relation_type: rel.cypher_label().to_string(),
                    direction,
                    created_at: row.get(2).and_then(Value::as_f64),
                    confidence: None,
                });
            }
        }
        None => {
            // All types — single UNION ALL query instead of 3 round-trips
            let types = all_user_relation_types();
            let (cypher, params, readonly) =
                cypher::get_all_typed_edges(&req.content_hash, types, direction, limit);
            let result = exec_query(&state, &cypher, params, readonly).await?;

            for row in &result.result_set {
                if row.len() < 4 {
                    continue;
                }
                edges.push(Edge {
                    source: row[0].as_str().unwrap_or("").to_string(),
                    target: row[1].as_str().unwrap_or("").to_string(),
                    relation_type: row[3].as_str().unwrap_or("").to_string(),
                    direction,
                    created_at: row.get(2).and_then(Value::as_f64),
                    confidence: None,
                });
            }
        }
    }

    Ok(Json(json!({ "edges": edges })))
}

/// POST /edges/delete
pub async fn delete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteEdgeRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.source) || !validate_content_hash(&req.target) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let rel = parse_user_relation(&req.relation_type)?;
    let (cypher, params, readonly) = cypher::delete_typed_edge(&req.source, &req.target, rel);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let count = result.count().unwrap_or(0);
    Ok(Json(json!({ "deleted": count > 0 })))
}

/// POST /edges/create-system
pub async fn create_system(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateSystemEdgeRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.source) || !validate_content_hash(&req.target) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let rel = parse_system_relation(&req.relation_type)?;
    let (cypher, params, readonly) =
        cypher::create_system_edge(&req.source, &req.target, rel, req.created_at);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    let count = result.count().unwrap_or(0);
    Ok(Json(json!({ "created": count > 0 })))
}
