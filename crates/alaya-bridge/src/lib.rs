//! Library interface for alaya-bridge.
//!
//! Exposes internal modules so that integration tests can exercise the
//! Cypher → Redis → RESP pipeline directly without HTTP overhead.

pub mod cypher;
pub mod handlers;
pub mod resp;
pub mod routes;

mod auth;
pub mod queue;

pub use handlers::exec_query;
pub use handlers::value_to_cypher_literal;
pub use resp::FalkorResult;

/// Shared application state injected via axum's `State` extractor.
pub struct AppState {
    pub redis: redis::aio::ConnectionManager,
    pub graph_name: String,
}
