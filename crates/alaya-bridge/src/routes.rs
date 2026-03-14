use axum::{routing::get, Json, Router};
use serde_json::{json, Value};

pub fn router() -> Router {
    Router::new().route("/health", get(health))
}

async fn health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "falkordb_connected": false,
        "write_queue_depth": 0
    }))
}
