use std::net::SocketAddr;
use tracing_subscriber::EnvFilter;

mod auth;
pub mod cypher;
mod routes;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let app = routes::router();
    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    tracing::info!("alaya-bridge listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
