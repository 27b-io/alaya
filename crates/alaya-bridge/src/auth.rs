use axum::{
    extract::Request,
    http::{header, StatusCode},
    middleware::Next,
    response::Response,
};

#[allow(dead_code)]
pub async fn require_bearer(req: Request, next: Next) -> Result<Response, StatusCode> {
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    let expected = std::env::var("GRAPH_API_KEY").unwrap_or_default();

    match token {
        Some(t) if !expected.is_empty() && t == expected => Ok(next.run(req).await),
        _ if expected.is_empty() => Ok(next.run(req).await),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}
