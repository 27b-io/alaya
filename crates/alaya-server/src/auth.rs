//! Dual-mode authentication + default-deny authorization.
//!
//! Three credentials on the protected router: a full static bearer key
//! (service/CLI), a read-only static bearer key (headless service consumers,
//! e.g. radar), and a provider-agnostic OIDC JWT (browser). Authorization is a
//! default-deny allowlist per principal: `Oidc` may only invoke read/additive
//! ops, `StaticReadOnly` only pure reads; every other op — current or future —
//! requires the full `Static` principal.

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

/// Pure-read ops a `StaticReadOnly` principal may invoke. Stricter than
/// `OIDC_ALLOWLIST`: no `store_memory` — the read-only bearer exists for
/// consumers (radar) that must never mutate the corpus, additively or not.
pub const READONLY_ALLOWLIST: &[&str] = &[
    "search",
    "get_memory",
    "check_database_health",
    "memory_contradictions",
    "find_duplicates",
];

/// Authenticated principal, inserted into request extensions by `require_auth`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AuthPrincipal {
    Static,
    StaticReadOnly,
    Oidc,
    Anonymous,
}

impl AuthPrincipal {
    /// Default-deny op gate. `Static` and `Anonymous` (dev open mode) may
    /// invoke anything; restricted principals only their allowlist.
    pub fn allows(self, op: &str) -> bool {
        match self {
            AuthPrincipal::Static | AuthPrincipal::Anonymous => true,
            AuthPrincipal::Oidc => OIDC_ALLOWLIST.contains(&op),
            AuthPrincipal::StaticReadOnly => READONLY_ALLOWLIST.contains(&op),
        }
    }
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
            AuthPrincipal::Oidc | AuthPrincipal::StaticReadOnly => WritePolicy::ReadOnly,
            AuthPrincipal::Static | AuthPrincipal::Anonymous => WritePolicy::Full,
        }
    }

    /// The `read_only` bool threaded into service calls — one derivation for
    /// every gate site, so a new principal can't drift between them.
    pub fn read_only_for(p: AuthPrincipal) -> bool {
        Self::for_principal(p) == WritePolicy::ReadOnly
    }
}

/// Auth configuration + verifier, built once at startup. Cheap to clone.
#[derive(Clone)]
pub struct AuthState {
    pub api_key: Option<String>,
    pub readonly_api_key: Option<String>,
    pub allow_unauthenticated: bool,
    pub oidc: Option<OidcVerifier>,
    pub public_base_url: String,
}

impl AuthState {
    /// True only when NO credential of any kind is configured AND the dev flag
    /// opted in. A readonly key counts as configured auth — otherwise open
    /// mode would hand `Anonymous` (= Full) to the bearer meant as read-only.
    pub fn open_mode(&self) -> bool {
        self.api_key.is_none()
            && self.readonly_api_key.is_none()
            && self.oidc.is_none()
            && self.allow_unauthenticated
    }
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

/// Every canonical op with its write classification, for the read-only
/// auth-config view. `(op, mutating)` — "mutating" means it rewrites or
/// removes shared state (store is additive, so it is not mutating).
pub const ALL_OPS: &[(&str, bool)] = &[
    ("search", false),
    ("get_memory", false),
    ("check_database_health", false),
    ("memory_contradictions", false),
    ("find_duplicates", false),
    ("store_memory", false),
    ("delete_memory", true),
    ("memory_supersede", true),
    ("merge_duplicates", true),
    ("relation", true),
    ("patch_memory", true),
    ("backfill_summaries", true),
];

/// Read-only view of the auth configuration (LAB-1684 AC7): which principals
/// exist, the OIDC issuer/audience, and the principal × op matrix. Contains
/// NO credential material — only presence flags and public identifiers.
pub fn auth_config_view(auth: &AuthState) -> serde_json::Value {
    // The full static principal always has full access, so per-op "static"
    // flags would be constants — the matrix only carries what varies: the
    // two restricted principals (oidc, readonly).
    let ops: Vec<serde_json::Value> = ALL_OPS
        .iter()
        .map(|(op, mutating)| {
            serde_json::json!({
                "op": op,
                "mutating": mutating,
                "oidc": AuthPrincipal::Oidc.allows(op),
                "readonly": AuthPrincipal::StaticReadOnly.allows(op),
            })
        })
        .collect();
    serde_json::json!({
        "static_bearer_configured": auth.api_key.is_some(),
        "readonly_bearer_configured": auth.readonly_api_key.is_some(),
        "oidc": {
            "enabled": auth.oidc.is_some(),
            "issuer": auth.oidc.as_ref().map(|v| v.issuer().to_string()),
            "audience": auth.oidc.as_ref().map(|v| v.audience().to_string()),
        },
        "ops": ops,
    })
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
    // Static keys first (cheap, constant-time). A static token is never routed
    // to JWT validation. Full key is checked before the readonly key; startup
    // refuses equal keys, so a readonly bearer can never resolve to `Static`.
    if let Some(ref key) = auth.api_key
        && constant_time_eq(token.as_bytes(), key.as_bytes())
    {
        return Some(AuthPrincipal::Static);
    }
    if let Some(ref key) = auth.readonly_api_key
        && constant_time_eq(token.as_bytes(), key.as_bytes())
    {
        return Some(AuthPrincipal::StaticReadOnly);
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

/// 403 for an authenticated-but-unauthorized request. Logged so an operator
/// can diagnose a misconfigured consumer (e.g. radar holding the read-only key
/// but calling a mutating route) without grepping silence.
fn forbidden_403(principal: AuthPrincipal, op: &str) -> Response {
    tracing::debug!(?principal, %op, "authorization denied");
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
    if auth.open_mode() {
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
    let path = req.uri().path();
    if path != "/mcp" {
        let op = rest_route_op(req.method(), path);
        if !principal.allows(op) {
            return forbidden_403(principal, op);
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
    /// The read-only static bearer (#80) gets the same view: it is on
    /// `READONLY_ALLOWLIST` because a headless consumer polling health is
    /// exactly the read-only use case.
    #[test]
    fn health_detail_is_a_read_for_restricted_principals() {
        let op = rest_route_op(&Method::GET, "/health/detail");
        assert_eq!(op, "check_database_health");
        assert!(AuthPrincipal::Oidc.allows(op));
        assert!(AuthPrincipal::StaticReadOnly.allows(op));

        // The bare probe is not on the protected router at all, so it must
        // stay unmapped — fail-closed if it is ever moved behind auth.
        assert_eq!(rest_route_op(&Method::GET, "/health"), "__mutating__");
    }

    #[test]
    fn write_policy_readonly_for_restricted_principals() {
        assert_eq!(
            WritePolicy::for_principal(AuthPrincipal::Oidc),
            WritePolicy::ReadOnly
        );
        assert_eq!(
            WritePolicy::for_principal(AuthPrincipal::StaticReadOnly),
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
    fn oidc_allowlist_is_exactly_the_read_additive_ops() {
        for op in [
            "search",
            "get_memory",
            "check_database_health",
            "memory_contradictions",
            "find_duplicates",
            "store_memory",
        ] {
            assert!(
                AuthPrincipal::Oidc.allows(op),
                "{op} should be allowed for Oidc"
            );
        }
    }

    #[test]
    fn readonly_allowlist_is_pure_read() {
        for op in [
            "search",
            "get_memory",
            "check_database_health",
            "memory_contradictions",
            "find_duplicates",
        ] {
            assert!(
                AuthPrincipal::StaticReadOnly.allows(op),
                "{op} should be allowed for StaticReadOnly"
            );
        }
        // Stricter than Oidc: no additive writes either.
        assert!(!AuthPrincipal::StaticReadOnly.allows("store_memory"));
    }

    #[test]
    fn every_mutating_op_is_denied_for_restricted_principals() {
        // The whole point of default-deny: these (and any future op) are NOT
        // in an allowlist, so a restricted principal can't invoke them.
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
            assert!(
                !AuthPrincipal::Oidc.allows(op),
                "{op} must be denied for Oidc"
            );
            assert!(
                !AuthPrincipal::StaticReadOnly.allows(op),
                "{op} must be denied for StaticReadOnly"
            );
        }
    }

    #[test]
    fn full_principals_allow_everything() {
        for op in ["store_memory", "delete_memory", "__mutating__"] {
            assert!(AuthPrincipal::Static.allows(op));
            assert!(AuthPrincipal::Anonymous.allows(op));
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
        // GET is allowed for restricted principals, PATCH is not.
        for p in [AuthPrincipal::Oidc, AuthPrincipal::StaticReadOnly] {
            assert!(p.allows(rest_route_op(&Method::GET, "/memories/abc123")));
            assert!(!p.allows(rest_route_op(&Method::PATCH, "/memories/abc123")));
        }
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
        for p in [AuthPrincipal::Oidc, AuthPrincipal::StaticReadOnly] {
            assert!(!p.allows(rest_route_op(&Method::POST, "/some/new/route")));
            assert!(!p.allows(rest_route_op(&Method::DELETE, "/memories/x")));
        }
    }

    #[test]
    fn auth_config_view_leaks_no_credentials_and_matches_allowlist() {
        let auth = crate::testkit::auth_state(Some("super-secret-bearer"), true);
        let v = auth_config_view(&auth);
        let text = v.to_string();
        assert!(
            !text.contains("super-secret-bearer"),
            "auth-config view must never contain the bearer"
        );
        assert_eq!(v["static_bearer_configured"], true);
        assert_eq!(v["oidc"]["enabled"], true);

        let ops = v["ops"].as_array().unwrap();
        assert_eq!(ops.len(), ALL_OPS.len());
        for op in ops {
            let name = op["op"].as_str().unwrap();
            assert_eq!(
                op["oidc"].as_bool().unwrap(),
                AuthPrincipal::Oidc.allows(name),
                "matrix must mirror Oidc.allows for {name}"
            );
            assert_eq!(
                op["readonly"].as_bool().unwrap(),
                AuthPrincipal::StaticReadOnly.allows(name),
                "matrix must mirror StaticReadOnly.allows for {name}"
            );
        }
        // Every mutating op must be denied to both restricted principals —
        // the read-only invariant the view exists to display. The readonly
        // bearer is stricter still: not even the additive store.
        for op in ops {
            if op["mutating"] == true {
                assert_eq!(op["oidc"], false, "{} must be OIDC-denied", op["op"]);
                assert_eq!(
                    op["readonly"], false,
                    "{} must be readonly-denied",
                    op["op"]
                );
            }
            if op["op"] == "store_memory" {
                assert_eq!(op["readonly"], false, "readonly bearer must not store");
            }
        }
    }

    #[test]
    fn auth_config_view_reports_readonly_bearer_presence_without_the_key() {
        let mut auth = crate::testkit::auth_state(None, false);
        auth.readonly_api_key = Some("readonly-secret".into());
        let v = auth_config_view(&auth);
        assert!(!v.to_string().contains("readonly-secret"));
        assert_eq!(v["static_bearer_configured"], false);
        assert_eq!(v["readonly_bearer_configured"], true);
    }

    /// Drift guard: every op `rest_route_op` can emit (except the fail-closed
    /// sentinel) must appear in `ALL_OPS`, or the auth-config view silently
    /// under-reports the surface it exists to display.
    #[test]
    fn all_ops_covers_every_rest_route_op() {
        let rest_ops = [
            rest_route_op(&Method::POST, "/store"),
            rest_route_op(&Method::POST, "/search"),
            rest_route_op(&Method::POST, "/delete"),
            rest_route_op(&Method::POST, "/relation"),
            rest_route_op(&Method::POST, "/supersede"),
            rest_route_op(&Method::POST, "/contradictions"),
            rest_route_op(&Method::POST, "/duplicates/find"),
            rest_route_op(&Method::POST, "/duplicates/merge"),
            rest_route_op(&Method::POST, "/backfill/summaries"),
            rest_route_op(&Method::GET, "/memories/x"),
            rest_route_op(&Method::PATCH, "/memories/x"),
        ];
        for op in rest_ops {
            assert!(
                ALL_OPS.iter().any(|(name, _)| *name == op),
                "ALL_OPS is missing '{op}' — the auth-config view under-reports"
            );
        }
        // And both allowlists must be subsets of ALL_OPS.
        for op in OIDC_ALLOWLIST.iter().chain(READONLY_ALLOWLIST) {
            assert!(
                ALL_OPS.iter().any(|(name, _)| name == op),
                "ALL_OPS is missing allowlisted op '{op}'"
            );
        }
    }

    #[test]
    fn auth_config_route_is_static_only_by_default_deny() {
        // GET /auth/config is deliberately unmapped in rest_route_op →
        // "__mutating__" → denied for every restricted principal. The
        // read-only bearer reads memories, not the auth configuration.
        let op = rest_route_op(&Method::GET, "/auth/config");
        assert!(!AuthPrincipal::Oidc.allows(op));
        assert!(!AuthPrincipal::StaticReadOnly.allows(op));
        assert!(AuthPrincipal::Static.allows(op));
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
    async fn authenticate_readonly_key() {
        let mut auth = state(Some("full-key"), false);
        auth.readonly_api_key = Some("ro-key".to_string());
        assert_eq!(
            authenticate(&auth, "ro-key").await,
            Some(AuthPrincipal::StaticReadOnly)
        );
        assert_eq!(
            authenticate(&auth, "full-key").await,
            Some(AuthPrincipal::Static)
        );
        assert_eq!(authenticate(&auth, "wrong").await, None);

        // Readonly key works standalone (no full key configured).
        let mut ro_only = state(None, false);
        ro_only.readonly_api_key = Some("ro-key".to_string());
        assert_eq!(
            authenticate(&ro_only, "ro-key").await,
            Some(AuthPrincipal::StaticReadOnly)
        );
    }

    #[test]
    fn readonly_key_disables_open_mode() {
        // Fail-closed: a readonly key + the dev flag must NOT grant Anonymous
        // (= Full) to everyone — the key counts as configured auth.
        let mut auth = state(None, false);
        auth.allow_unauthenticated = true;
        assert!(auth.open_mode());
        auth.readonly_api_key = Some("ro-key".to_string());
        assert!(!auth.open_mode());
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

    // ── middleware wiring: route-level 200 vs 401/403 per bearer ─────────────

    fn test_router(auth: AuthState) -> axum::Router {
        use axum::routing::{get, post};
        async fn ok() -> &'static str {
            "ok"
        }
        axum::Router::new()
            .route("/search", post(ok))
            .route("/store", post(ok))
            .route("/delete", post(ok))
            .route("/supersede", post(ok))
            .route("/relation", post(ok))
            .route("/duplicates/merge", post(ok))
            .route("/memories/{content_hash}", get(ok).patch(ok))
            .layer(axum::middleware::from_fn_with_state(auth, require_auth))
    }

    async fn status_for(
        router: &axum::Router,
        method: Method,
        path: &str,
        token: Option<&str>,
    ) -> StatusCode {
        use tower::ServiceExt;
        let mut builder = axum::http::Request::builder().method(method).uri(path);
        if let Some(t) = token {
            builder = builder.header(header::AUTHORIZATION, format!("Bearer {t}"));
        }
        let req = builder.body(axum::body::Body::empty()).unwrap();
        router.clone().oneshot(req).await.unwrap().status()
    }

    #[tokio::test]
    async fn readonly_bearer_reads_succeed_and_mutations_403() {
        let mut auth = state(Some("full-key"), false);
        auth.readonly_api_key = Some("ro-key".to_string());
        let app = test_router(auth);

        // Reads succeed.
        assert_eq!(
            status_for(&app, Method::POST, "/search", Some("ro-key")).await,
            StatusCode::OK
        );
        assert_eq!(
            status_for(&app, Method::GET, "/memories/abc", Some("ro-key")).await,
            StatusCode::OK
        );

        // Every mutating route → 403 (authenticated but not authorized).
        for (method, path) in [
            (Method::POST, "/store"),
            (Method::POST, "/delete"),
            (Method::POST, "/supersede"),
            (Method::POST, "/relation"),
            (Method::POST, "/duplicates/merge"),
            (Method::PATCH, "/memories/abc"),
            (Method::POST, "/some/unmapped/route"),
        ] {
            assert_eq!(
                status_for(&app, method.clone(), path, Some("ro-key")).await,
                StatusCode::FORBIDDEN,
                "{method} {path} must be 403 for the readonly bearer"
            );
        }

        // The full key is unaffected; a bad key still 401s.
        assert_eq!(
            status_for(&app, Method::POST, "/delete", Some("full-key")).await,
            StatusCode::OK
        );
        assert_eq!(
            status_for(&app, Method::POST, "/search", Some("bogus")).await,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            status_for(&app, Method::POST, "/search", None).await,
            StatusCode::UNAUTHORIZED
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
