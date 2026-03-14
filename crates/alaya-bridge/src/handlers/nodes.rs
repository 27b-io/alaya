//! Node handlers — POST /nodes/ensure, POST /nodes/delete

use std::sync::Arc;

use axum::{Json, extract::State, http::StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::memory::validate_content_hash;

use crate::{AppState, cypher, handlers::exec_query};

// ─── Request types ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct EnsureNodeRequest {
    pub content_hash: String,
    pub created_at: f64,
}

#[derive(Debug, Deserialize)]
pub struct DeleteNodeRequest {
    pub content_hash: String,
}

// ─── Handlers ─────────────────────────────────────────────────────────────────

/// POST /nodes/ensure
///
/// MERGE a Memory node, setting `created_at` only on creation.
pub async fn ensure(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EnsureNodeRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.content_hash) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let (cypher, params, readonly) = cypher::ensure_node(&req.content_hash, req.created_at);
    let result = exec_query(&state, &cypher, params, readonly).await?;

    // "Nodes created: 1" means it was a new node; 0 means it already existed
    let nodes_created: u64 = result
        .stats
        .get("Nodes created")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    Ok(Json(json!({ "created": nodes_created > 0 })))
}

/// POST /nodes/delete
///
/// DETACH DELETE a Memory node by content_hash.
pub async fn delete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DeleteNodeRequest>,
) -> Result<Json<Value>, StatusCode> {
    if !validate_content_hash(&req.content_hash) {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let (cypher, params, readonly) = cypher::delete_node(&req.content_hash);
    exec_query(&state, &cypher, params, readonly).await?;

    Ok(Json(json!({ "deleted": true })))
}
