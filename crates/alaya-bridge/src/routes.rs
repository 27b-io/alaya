use std::sync::Arc;

use axum::{
    Router, middleware,
    routing::{get, post},
};

use crate::{AppState, auth, handlers};

pub fn router(state: Arc<AppState>) -> Router {
    // Routes that require bearer auth
    let api = Router::new()
        .route("/nodes/ensure", post(handlers::nodes::ensure))
        .route("/nodes/delete", post(handlers::nodes::delete))
        .route("/edges/create", post(handlers::edges::create))
        .route("/edges/create-batch", post(handlers::edges::create_batch))
        .route("/edges/create-system", post(handlers::edges::create_system))
        .route(
            "/edges/create-system-batch",
            post(handlers::edges::create_system_batch),
        )
        .route("/edges/get", post(handlers::edges::get))
        .route("/edges/delete", post(handlers::edges::delete))
        .route("/stats", get(handlers::health::stats))
        .route("/hebbian/neighbors", post(handlers::hebbian::neighbors))
        .route("/hebbian/spreading", post(handlers::hebbian::spreading))
        .route(
            "/hebbian/boosts-within",
            post(handlers::hebbian::boosts_within),
        )
        .route("/hebbian/strengthen", post(handlers::hebbian::strengthen))
        .route("/contradictions/all", post(handlers::contradictions::all))
        .route(
            "/contradictions/for",
            post(handlers::contradictions::for_hashes),
        )
        .route(
            "/consolidation/decay-all",
            post(handlers::consolidation::decay_all),
        )
        .route(
            "/consolidation/decay-stale",
            post(handlers::consolidation::decay_stale),
        )
        .route("/consolidation/prune", post(handlers::consolidation::prune))
        .route(
            "/consolidation/orphans",
            post(handlers::consolidation::orphans),
        )
        .layer(middleware::from_fn(auth::require_bearer));

    // /health is unauthenticated — merge after auth layer
    Router::new()
        .route("/health", get(handlers::health::health))
        .merge(api)
        .with_state(state)
}
