//! RFC 9728 Protected Resource Metadata.
//!
//! Served unauthenticated at both `/.well-known/oauth-protected-resource` and
//! the resource-suffixed `/.well-known/oauth-protected-resource/mcp` (claude.ai
//! probes the suffixed form). Returns `404` when OIDC is disabled.

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::auth::AuthState;

pub async fn protected_resource_metadata(State(auth): State<AuthState>) -> Response {
    let Some(ref oidc) = auth.oidc else {
        // Nothing to advertise when OAuth is disabled.
        return StatusCode::NOT_FOUND.into_response();
    };

    Json(json!({
        "resource": format!("{}/mcp", auth.public_base_url),
        "authorization_servers": [oidc.issuer()],
        "bearer_methods_supported": ["header"],
    }))
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;

    fn auth(oidc_on: bool) -> crate::auth::AuthState {
        crate::testkit::auth_state(Some("k"), oidc_on)
    }

    #[tokio::test]
    async fn returns_404_when_oidc_disabled() {
        let resp = protected_resource_metadata(State(auth(false))).await;
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn advertises_resource_and_issuer_when_enabled() {
        let resp = protected_resource_metadata(State(auth(true))).await;
        assert_eq!(resp.status(), StatusCode::OK);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["resource"], "https://rs.test/mcp");
        assert_eq!(v["authorization_servers"][0], "https://issuer.test");
        assert_eq!(v["bearer_methods_supported"][0], "header");
    }
}
