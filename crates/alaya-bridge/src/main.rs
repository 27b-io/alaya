use std::net::SocketAddr;
use std::sync::Arc;

use alaya_bridge::AppState;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let redis_url = std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".into());
    let graph_name = std::env::var("GRAPH_NAME").unwrap_or_else(|_| "memory".into());

    let client = redis::Client::open(redis_url.as_str()).expect("Invalid REDIS_URL");
    let redis = redis::aio::ConnectionManager::new(client)
        .await
        .expect("Failed to connect to Redis");

    let state = Arc::new(AppState { redis, graph_name });

    // Start background Hebbian write-queue consumer (LPUSH/BRPOP pattern).
    alaya_bridge::queue::spawn_consumer(state.clone());

    let app = alaya_bridge::routes::router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("alaya-bridge listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
