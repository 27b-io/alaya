//! Bearer auth for the bridge API.
//!
//! Fail-closed, mirroring alaya-server: the key is resolved ONCE at startup
//! (`BridgeAuth::from_env`) and compared constant-time on every request. Open
//! mode exists only as an explicit, logged opt-in
//! (`DANGEROUSLY_ALLOW_UNAUTHENTICATED=true`); an empty `GRAPH_API_KEY` with no
//! opt-in is a startup error, never a silently open service.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

/// Auth policy for every route in the API group (`routes::router`).
#[derive(Clone)]
pub enum BridgeAuth {
    /// `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` with no key: every request passes.
    Open,
    /// `GRAPH_API_KEY` set: the bearer must match, compared constant-time.
    Bearer(Arc<str>),
}

/// Never prints the key: `AppState` or the policy may end up in a `{:?}` log.
impl std::fmt::Debug for BridgeAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open => f.write_str("Open"),
            Self::Bearer(_) => f.write_str("Bearer(<redacted>)"),
        }
    }
}

impl BridgeAuth {
    /// Resolve the policy from `GRAPH_API_KEY` + `DANGEROUSLY_ALLOW_UNAUTHENTICATED`.
    pub fn from_env() -> Result<Self, &'static str> {
        let key = std::env::var("GRAPH_API_KEY").unwrap_or_default();
        let allow_unauthenticated = std::env::var("DANGEROUSLY_ALLOW_UNAUTHENTICATED")
            .unwrap_or_default()
            .eq_ignore_ascii_case("true");
        Self::resolve(&key, allow_unauthenticated)
    }

    /// Fail-closed startup invariant: no key and no opt-in is an error.
    fn resolve(key: &str, allow_unauthenticated: bool) -> Result<Self, &'static str> {
        if !key.is_empty() {
            if allow_unauthenticated {
                tracing::warn!("DANGEROUSLY_ALLOW_UNAUTHENTICATED ignored — auth is configured");
            }
            return Ok(Self::Bearer(key.into()));
        }
        if !allow_unauthenticated {
            return Err("no auth configured: set GRAPH_API_KEY, or \
                 DANGEROUSLY_ALLOW_UNAUTHENTICATED=true for dev");
        }
        tracing::warn!(
            "DANGEROUSLY_ALLOW_UNAUTHENTICATED — all bridge endpoints are UNAUTHENTICATED"
        );
        Ok(Self::Open)
    }

    fn permits(&self, token: Option<&str>) -> bool {
        match self {
            Self::Open => true,
            Self::Bearer(key) => {
                token.is_some_and(|t| bool::from(t.as_bytes().ct_eq(key.as_bytes())))
            }
        }
    }
}

pub async fn require_bearer(
    State(auth): State<BridgeAuth>,
    req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));

    if auth.permits(token) {
        return Ok(next.run(req).await);
    }
    tracing::warn!(path = %req.uri().path(), "bearer rejected");
    Err(StatusCode::UNAUTHORIZED)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::Body, http::Request, middleware, routing::get};
    use tower::ServiceExt;

    /// One protected route behind the real middleware — exercises the layering,
    /// not a replica of it.
    async fn status(auth: BridgeAuth, bearer: Option<&str>) -> StatusCode {
        let app = Router::new()
            .route("/", get(|| async { "ok" }))
            .layer(middleware::from_fn_with_state(auth, require_bearer));
        let mut req = Request::builder().uri("/");
        if let Some(b) = bearer {
            req = req.header(header::AUTHORIZATION, format!("Bearer {b}"));
        }
        app.oneshot(req.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    // Empty key, no opt-in: startup error naming both remedies.
    #[test]
    fn empty_key_without_opt_in_is_startup_error() {
        let err = BridgeAuth::resolve("", false).unwrap_err();
        assert!(err.contains("GRAPH_API_KEY"), "{err}");
        assert!(err.contains("DANGEROUSLY_ALLOW_UNAUTHENTICATED"), "{err}");
    }

    #[test]
    fn configured_key_ignores_opt_in() {
        assert!(matches!(
            BridgeAuth::resolve("k", true),
            Ok(BridgeAuth::Bearer(_))
        ));
    }

    // Empty key, opt-in: requests pass with no bearer at all.
    #[tokio::test]
    async fn open_mode_passes_without_bearer() {
        let auth = BridgeAuth::resolve("", true).unwrap();
        assert_eq!(status(auth, None).await, StatusCode::OK);
    }

    // Key set: missing or wrong bearer is 401, exact bearer is 200.
    #[tokio::test]
    async fn bearer_mode_rejects_missing_and_wrong_accepts_exact() {
        let auth = BridgeAuth::resolve("s3cret", false).unwrap();
        assert_eq!(status(auth.clone(), None).await, StatusCode::UNAUTHORIZED);
        assert_eq!(
            status(auth.clone(), Some("wrong")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(auth.clone(), Some("s3cre")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status(auth.clone(), Some("s3cretx")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(status(auth, Some("s3cret")).await, StatusCode::OK);
    }
}
