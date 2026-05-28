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
