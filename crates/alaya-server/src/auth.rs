//! Dual-mode authentication + default-deny authorization.
//!
//! Two auth modes on the protected router: a static bearer key (service/CLI)
//! and a provider-agnostic OIDC JWT (browser). Authorization is a default-deny
//! allowlist: an `Oidc` principal may only invoke read/additive ops; every
//! other op — current or future — requires the `Static` principal.

use axum::extract::{Request, State};
use axum::http::{Method, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use subtle::ConstantTimeEq;

use crate::oidc::OidcVerifier;

/// Max accepted `Authorization` token length (cheap CPU-DoS guard — the body
/// limit covers payloads, not headers).
const MAX_TOKEN_LEN: usize = 8 * 1024;

/// Read/additive ops an `Oidc` principal may invoke. Everything not listed is
/// `Static`-only by default (default-deny). Values are canonical op-names =
/// MCP tool names; `rest_route_op` maps REST routes into the same vocabulary.
pub const OIDC_ALLOWLIST: &[&str] = &[
    "search",
    "get_memory",
    "check_database_health",
    "memory_contradictions",
    "find_duplicates",
    "store_memory",
];

/// Authenticated principal, inserted into request extensions by `require_auth`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuthPrincipal {
    Static,
    Oidc,
    Anonymous,
}

/// Whether a service operation may perform shared-state writes. No `Default`:
/// every call site must derive it from the principal (fail-closed by omission).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WritePolicy {
    Full,
    ReadOnly,
}

impl WritePolicy {
    pub fn for_principal(p: AuthPrincipal) -> Self {
        match p {
            AuthPrincipal::Oidc => WritePolicy::ReadOnly,
            AuthPrincipal::Static | AuthPrincipal::Anonymous => WritePolicy::Full,
        }
    }
}

/// Auth configuration + verifier, built once at startup. Cheap to clone.
#[derive(Clone)]
pub struct AuthState {
    pub api_key: Option<String>,
    pub allow_unauthenticated: bool,
    pub oidc: Option<OidcVerifier>,
    pub public_base_url: String,
}

/// Map a REST route to a canonical op-name in the `OIDC_ALLOWLIST` vocabulary.
/// Keyed on `(method, path)` so `GET`/`PATCH` on the same path differ. An
/// unmapped route resolves to a synthetic mutating op → denied for `Oidc`.
pub fn rest_route_op(method: &Method, path: &str) -> &'static str {
    match (method.as_str(), path) {
        ("POST", "/store") => "store_memory",
        ("POST", "/search") => "search",
        ("POST", "/delete") => "delete_memory",
        ("POST", "/relation") => "relation",
        ("POST", "/supersede") => "memory_supersede",
        ("POST", "/contradictions") => "memory_contradictions",
        ("POST", "/duplicates/find") => "find_duplicates",
        ("POST", "/duplicates/merge") => "merge_duplicates",
        ("POST", "/backfill/summaries") => "backfill_summaries",
        ("GET", "/health/detail") => "check_database_health",
        ("GET", p) if p.starts_with("/memories/") => "get_memory",
        ("PATCH", p) if p.starts_with("/memories/") => "patch_memory",
        // Unmapped / unexpected method → fail-closed (not in allowlist).
        _ => "__mutating__",
    }
}

/// True if an `Oidc` principal is permitted to invoke `op`.
pub fn oidc_allows(op: &str) -> bool {
    OIDC_ALLOWLIST.contains(&op)
}

/// Extract the bearer token from an `Authorization` header.
/// RFC 6750 §2.1: the scheme name is case-insensitive (`Bearer`/`bearer`/`BEARER`
/// are all valid); the credential follows whitespace after the scheme.
fn bearer(req: &Request) -> Option<&str> {
    let header = req.headers().get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, token) = header.split_once(|c: char| c.is_ascii_whitespace())?;
    if scheme.eq_ignore_ascii_case("bearer") {
        Some(token.trim_start())
    } else {
        None
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.ct_eq(b).into()
}

async fn authenticate(auth: &AuthState, token: &str) -> Option<AuthPrincipal> {
    // Static key first (cheap, constant-time). A static token is never routed
    // to JWT validation.
    if let Some(ref key) = auth.api_key
        && constant_time_eq(token.as_bytes(), key.as_bytes())
    {
        return Some(AuthPrincipal::Static);
    }
    // OIDC: only attempt for JWT-shaped tokens (exactly two dots).
    if let Some(ref verifier) = auth.oidc
        && token.split('.').count() == 3
    {
        match verifier.validate(token).await {
            Ok(()) => return Some(AuthPrincipal::Oidc),
            // The reason string is already a server-safe &'static str (no
            // token internals leaked). Log so operators can diagnose
            // "claude.ai stopped working" without grepping silence.
            Err(e) => tracing::debug!(reason = %e, "OIDC validation failed"),
        }
    }
    None
}

fn challenge_401(auth: &AuthState) -> Response {
    let www_auth = if auth.oidc.is_some() {
        format!(
            "Bearer resource_metadata=\"{}/.well-known/oauth-protected-resource\"",
            auth.public_base_url
        )
    } else {
        "Bearer".to_string()
    };
    (
        StatusCode::UNAUTHORIZED,
        [(header::WWW_AUTHENTICATE, www_auth)],
    )
        .into_response()
}

fn forbidden_403() -> Response {
    (
        StatusCode::FORBIDDEN,
        axum::Json(serde_json::json!({"error": "forbidden for this principal"})),
    )
        .into_response()
}

/// Dual-mode auth + default-deny REST authz. Layered over the entire protected
/// router. The `/mcp` op-gate is enforced in `mcp_handler` (post-parse), so
/// this middleware only authenticates `/mcp` and sets the principal extension.
pub async fn require_auth(State(auth): State<AuthState>, mut req: Request, next: Next) -> Response {
    // Open mode: only reachable when startup confirmed no auth + the dev flag.
    if auth.api_key.is_none() && auth.oidc.is_none() && auth.allow_unauthenticated {
        req.extensions_mut().insert(AuthPrincipal::Anonymous);
        return next.run(req).await;
    }

    let token = bearer(&req).filter(|t| t.len() <= MAX_TOKEN_LEN);
    let Some(token) = token else {
        return challenge_401(&auth);
    };
    let Some(principal) = authenticate(&auth, token).await else {
        return challenge_401(&auth);
    };

    // REST authz (the /mcp path is gated in mcp_handler after JSON-RPC parse).
    let path = req.uri().path().to_string();
    if path != "/mcp" && principal == AuthPrincipal::Oidc {
        let op = rest_route_op(req.method(), &path);
        if !oidc_allows(op) {
            return forbidden_403();
        }
    }

    req.extensions_mut().insert(principal);
    next.run(req).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `/health/detail` is a pure read, but default-deny resolves any unmapped
    /// route to `__mutating__` — which would 403 an OIDC principal that can
    /// already read substantially the same document through MCP
    /// `check_database_health`. The mapping, not the allowlist, is the fix.
    #[test]
    fn health_detail_is_a_read_for_oidc_principals() {
        let op = rest_route_op(&Method::GET, "/health/detail");
        assert_eq!(op, "check_database_health");
        assert!(oidc_allows(op));

        // The bare probe is not on the protected router at all, so it must
        // stay unmapped — fail-closed if it is ever moved behind auth.
        assert_eq!(rest_route_op(&Method::GET, "/health"), "__mutating__");
    }

    #[test]
    fn write_policy_readonly_only_for_oidc() {
        assert_eq!(
            WritePolicy::for_principal(AuthPrincipal::Oidc),
            WritePolicy::ReadOnly
        );
        assert_eq!(
            WritePolicy::for_principal(AuthPrincipal::Static),
            WritePolicy::Full
        );
        assert_eq!(
            WritePolicy::for_principal(AuthPrincipal::Anonymous),
            WritePolicy::Full
        );
    }

    #[test]
    fn allowlist_is_exactly_the_read_additive_ops() {
        for op in [
            "search",
            "get_memory",
            "check_database_health",
            "memory_contradictions",
            "find_duplicates",
            "store_memory",
        ] {
            assert!(oidc_allows(op), "{op} should be allowed for Oidc");
        }
    }

    #[test]
    fn every_mutating_op_is_denied_for_oidc() {
        // The whole point of default-deny: these (and any future op) are NOT
        // in the allowlist, so an Oidc principal can't invoke them.
        for op in [
            "delete_memory",
            "memory_supersede",
            "merge_duplicates",
            "patch_memory",
            "relation",
            "backfill_summaries",
            "__mutating__",
            "some_future_tool_added_next_year",
        ] {
            assert!(!oidc_allows(op), "{op} must be denied for Oidc");
        }
    }

    #[test]
    fn rest_route_op_distinguishes_methods_on_shared_path() {
        assert_eq!(
            rest_route_op(&Method::GET, "/memories/abc123"),
            "get_memory"
        );
        assert_eq!(
            rest_route_op(&Method::PATCH, "/memories/abc123"),
            "patch_memory"
        );
        // GET is allowed for Oidc, PATCH is not.
        assert!(oidc_allows(rest_route_op(&Method::GET, "/memories/abc123")));
        assert!(!oidc_allows(rest_route_op(
            &Method::PATCH,
            "/memories/abc123"
        )));
    }

    #[test]
    fn rest_route_op_maps_known_routes() {
        assert_eq!(rest_route_op(&Method::POST, "/store"), "store_memory");
        assert_eq!(rest_route_op(&Method::POST, "/search"), "search");
        assert_eq!(rest_route_op(&Method::POST, "/delete"), "delete_memory");
        assert_eq!(
            rest_route_op(&Method::POST, "/duplicates/find"),
            "find_duplicates"
        );
        assert_eq!(
            rest_route_op(&Method::POST, "/duplicates/merge"),
            "merge_duplicates"
        );
        assert_eq!(
            rest_route_op(&Method::POST, "/backfill/summaries"),
            "backfill_summaries"
        );
    }

    #[test]
    fn rest_route_op_unmapped_fails_closed() {
        // An unknown route or unexpected method must not be allowlisted.
        assert!(!oidc_allows(rest_route_op(
            &Method::POST,
            "/some/new/route"
        )));
        assert!(!oidc_allows(rest_route_op(&Method::DELETE, "/memories/x")));
    }

    #[test]
    fn constant_time_eq_basic() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secres"));
        assert!(!constant_time_eq(b"secret", b"secretx"));
        assert!(!constant_time_eq(b"", b"x"));
    }

    // ── authentication path (the dual-mode dispatch + 401 challenge) ─────────

    use crate::testkit::auth_state as state;

    #[tokio::test]
    async fn authenticate_static_key() {
        let auth = state(Some("s3cret"), false);
        assert_eq!(
            authenticate(&auth, "s3cret").await,
            Some(AuthPrincipal::Static)
        );
        assert_eq!(authenticate(&auth, "wrong").await, None);
    }

    #[tokio::test]
    async fn authenticate_valid_oidc_and_refuses_hs256_and_garbage() {
        let auth = state(None, true);
        let good = crate::testkit::mint(
            jsonwebtoken::Algorithm::RS256,
            Some(crate::testkit::KID_RSA),
            &crate::testkit::TestClaims::valid(),
        );
        assert_eq!(authenticate(&auth, &good).await, Some(AuthPrincipal::Oidc));

        // HS256 (alg-confusion downgrade) and a non-JWT must both be refused.
        let hs = crate::testkit::mint(
            jsonwebtoken::Algorithm::HS256,
            Some(crate::testkit::KID_RSA),
            &crate::testkit::TestClaims::valid(),
        );
        assert_eq!(authenticate(&auth, &hs).await, None);
        assert_eq!(authenticate(&auth, "not.a.jwt").await, None);
    }

    #[test]
    fn challenge_advertises_resource_metadata_when_oidc_on() {
        let resp = challenge_401(&state(Some("k"), true));
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        let www = resp
            .headers()
            .get(header::WWW_AUTHENTICATE)
            .unwrap()
            .to_str()
            .unwrap();
        assert!(
            www.contains("https://rs.test/.well-known/oauth-protected-resource"),
            "{www}"
        );
    }

    #[test]
    fn challenge_is_bare_bearer_when_oidc_off() {
        let resp = challenge_401(&state(Some("k"), false));
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            resp.headers().get(header::WWW_AUTHENTICATE).unwrap(),
            "Bearer"
        );
    }

    #[test]
    fn bearer_scheme_is_case_insensitive() {
        let req = axum::http::Request::builder()
            .header(header::AUTHORIZATION, "bearer  tok123")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(bearer(&req), Some("tok123"));

        let not_bearer = axum::http::Request::builder()
            .header(header::AUTHORIZATION, "Basic abc")
            .body(axum::body::Body::empty())
            .unwrap();
        assert_eq!(bearer(&not_bearer), None);
    }
}
