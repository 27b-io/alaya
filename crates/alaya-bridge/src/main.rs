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
    let graph_name = std::env::var("GRAPH_NAME").unwrap_or_else(|_| "memory_graph".into());

    let client = redis::Client::open(redis_url.as_str()).expect("Invalid REDIS_URL");
    let redis = redis::aio::ConnectionManager::new(client.clone())
        .await
        .expect("Failed to connect to Redis");

    let state = Arc::new(AppState { redis, graph_name });

    // BRPOP is a blocking command that starves a multiplexed ConnectionManager.
    // Give the consumer its own dedicated connection so it doesn't block query handlers.
    let queue_conn = redis::aio::ConnectionManager::new(client)
        .await
        .expect("Failed to create queue connection");

    alaya_bridge::queue::spawn_consumer(state.clone(), queue_conn);

    let app = alaya_bridge::routes::router(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("alaya-bridge listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    // axum returned — in-flight requests done. The Hebbian queue consumer
    // (spawned task) will be cancelled when the tokio runtime drops on
    // return from main. Hebbian strengthening is idempotent, so a
    // partially-processed item is safe to lose.
    tracing::info!("shutdown complete");
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    let mut sigterm = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .expect("failed to register SIGTERM handler");

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT, shutting down…"),
        _ = sigterm.recv() => tracing::info!("received SIGTERM, shutting down…"),
    }
}
