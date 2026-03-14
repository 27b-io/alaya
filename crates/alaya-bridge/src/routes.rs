use std::sync::Arc;

use axum::{middleware, routing::{get, post}, Router};

use crate::{auth, handlers, AppState};

pub fn router(state: Arc<AppState>) -> Router {
    // Routes that require bearer auth
    let api = Router::new()
        .route("/nodes/ensure", post(handlers::nodes::ensure))
        .route("/nodes/delete", post(handlers::nodes::delete))
        .route("/edges/create", post(handlers::edges::create))
        .route("/edges/get", post(handlers::edges::get))
        .route("/edges/delete", post(handlers::edges::delete))
        .route("/stats", get(handlers::health::stats))
        .layer(middleware::from_fn(auth::require_bearer));

    // /health is unauthenticated — merge after auth layer
    Router::new()
        .route("/health", get(handlers::health::health))
        .merge(api)
        .with_state(state)
}
