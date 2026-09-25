//! OIDC token verification — provider-agnostic OAuth Resource Server side.
//!
//! Discovery, same-origin/HTTPS enforcement, the JWKS cache + cooldown and the
//! whole verify pipeline (alg allowlist → key → signature/`exp`/`aud` →
//! normalised `iss`) live in `alaya-oidc` (shared with ops-console's relying
//! party). This module keeps what only a resource server decides:
//! - the issuer must be https:// (loopback excepted) or startup aborts
//! - `aud` is the canonical resource (`{public_base_url}/mcp`)
//! - a hard max-token-age cap — there is no revocation, so this bounds the
//!   compromise window
//! - rejection reasons reach operator logs only; the client always sees a
//!   generic 401
//!
//! The verifier lives on the axum side (Send+Sync); it never runs behind the
//! service-worker channel.

use std::sync::Arc;

use alaya_oidc::{
    CLOCK_SKEW_LEEWAY_SECS, Error as OidcError, IssuedClaims, Provider, is_loopback_origin,
    origin_of,
};
use serde::Deserialize;

/// Hard cap on accepted token lifetime (`exp - iat`), regardless of issuer.
/// There is no revocation, so this bounds the compromise window.
const MAX_TOKEN_AGE_SECS: u64 = 3600;

#[derive(Deserialize)]
struct Claims {
    iss: String,
    /// Optional per RFC 7519 §4.1.6 — `None` means the IdP omitted it.
    iat: Option<u64>,
    exp: u64,
}

impl IssuedClaims for Claims {
    fn iss(&self) -> &str {
        &self.iss
    }
}

struct Inner {
    provider: Provider,
    audience: String,
}

/// Provider-agnostic JWT verifier. Cheap to clone (Arc-backed).
#[derive(Clone)]
pub struct OidcVerifier {
    inner: Arc<Inner>,
}

impl OidcVerifier {
    /// `issuer` is the configured `OIDC_ISSUER`; `audience` is the canonical
    /// resource (`{public_base_url}/mcp`). Discovery is deferred to first use.
    ///
    /// Aborts on a non-HTTPS issuer (except loopback) — discovery would be
    /// MITM-able over plaintext and the cooldown could be burned trivially.
    pub fn new(issuer: String, audience: String) -> Self {
        let provider = Provider::new(&issuer);
        let is_loopback = origin_of(provider.issuer())
            .map(|origin| is_loopback_origin(&origin))
            .unwrap_or(false);
        if !provider.issuer().starts_with("https://") && !is_loopback {
            panic!("OIDC_ISSUER must be https:// (except loopback): {issuer}");
        }
        Self {
            inner: Arc::new(Inner { provider, audience }),
        }
    }

    /// The configured issuer (normalized), for protected-resource metadata.
    pub fn issuer(&self) -> &str {
        self.inner.provider.issuer()
    }

    /// The configured audience, for the read-only auth-config view.
    pub fn audience(&self) -> &str {
        &self.inner.audience
    }

    /// Validate a bearer token. Returns Ok on a fully-valid token; any failure
    /// is `OidcError::Invalid` and must surface to the client as a generic 401.
    pub async fn validate(&self, token: &str) -> Result<(), OidcError> {
        // Shared pipeline: header allowlist + kid, key (cache → single-flight
        // refetch), signature + `exp`/`aud` for our audience, normalised `iss`.
        let claims: Claims = self
            .inner
            .provider
            .verify(token, &self.inner.audience)
            .await?;

        // Cap the lifetime regardless of issuer (no revocation, so this bounds
        // the compromise window). RFC 7519 §4.1.6 makes `iat` OPTIONAL.
        // Compute `now` once and use it on both branches; if `iat` is present
        // and *in the future*, reject — otherwise `exp.saturating_sub(iat)`
        // would underflow to 0 and silently bypass the cap.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let cap_exceeded = match claims.iat {
            Some(iat) => {
                if iat > now.saturating_add(CLOCK_SKEW_LEEWAY_SECS) {
                    return Err(OidcError::Invalid("iat in the future"));
                }
                claims.exp.saturating_sub(iat) > MAX_TOKEN_AGE_SECS
            }
            None => claims.exp.saturating_sub(now) > MAX_TOKEN_AGE_SECS,
        };
        if cap_exceeded {
            return Err(OidcError::Invalid("token lifetime exceeds cap"));
        }
        Ok(())
    }
}

#[cfg(test)]
impl OidcVerifier {
    /// Build a verifier with a signing key pre-cached so no test reaches the
    /// network. Shared by the `oidc`, `auth`, and `wellknown` tests so the
    /// key-injection logic lives in one place.
    fn test_with_key(jwk: alaya_oidc::Jwk) -> Self {
        let provider = Provider::new(crate::testkit::ISSUER);
        provider.seed_keys([jwk]);
        Self {
            inner: Arc::new(Inner {
                provider,
                audience: crate::testkit::AUDIENCE.to_string(),
            }),
        }
    }

    pub(crate) fn test_with_rsa_key() -> Self {
        Self::test_with_key(alaya_oidc::Jwk {
            kty: "RSA".into(),
            kid: Some(crate::testkit::KID_RSA.into()),
            n: Some(crate::testkit::RSA_N.into()),
            e: Some(crate::testkit::RSA_E.into()),
            x: None,
            y: None,
        })
    }

    pub(crate) fn test_with_ec_key() -> Self {
        Self::test_with_key(alaya_oidc::Jwk {
            kty: "EC".into(),
            kid: Some(crate::testkit::KID_EC.into()),
            n: None,
            e: None,
            x: Some(crate::testkit::EC_X.into()),
            y: Some(crate::testkit::EC_Y.into()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jsonwebtoken::Algorithm;

    // ── validate(): token verification with an injected signing key ──────────
    //
    // These build a verifier with the JWKS key pre-cached (so validate() never
    // touches the network) and exercise the security-critical claim/signature
    // path with real signed tokens minted by `testkit`.

    use crate::testkit::{self, TestClaims, mint};

    fn rsa_verifier() -> OidcVerifier {
        OidcVerifier::test_with_rsa_key()
    }
    fn ec_verifier() -> OidcVerifier {
        OidcVerifier::test_with_ec_key()
    }

    /// Which discovery defect the mock IdP serves, to exercise each rejection
    /// branch independently. (Fieldless; `Copy` so `spawn_idp` can match it
    /// twice without a move.)
    #[derive(Clone, Copy)]
    enum IdpFault {
        /// Correct discovery + JWKS (happy path).
        None,
        /// discovery `issuer` != configured issuer (OIDC Core §4.3).
        Issuer,
        /// `jwks_uri` on a different origin than the issuer (key substitution).
        JwksOrigin,
    }

    /// Spawn a loopback OIDC provider serving discovery + JWKS for the RSA test
    /// key. Returns the base URL (a loopback issuer, so http is accepted).
    async fn spawn_idp(fault: IdpFault) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let issuer = match fault {
            IdpFault::Issuer => "https://attacker.test".to_string(),
            _ => base.clone(),
        };
        let jwks_uri = match fault {
            IdpFault::JwksOrigin => "https://attacker.test/jwks".to_string(),
            _ => format!("{base}/jwks"),
        };
        let discovery = serde_json::json!({ "issuer": issuer, "jwks_uri": jwks_uri }).to_string();
        let jwks = serde_json::json!({
            "keys": [{
                "kty": "RSA", "kid": testkit::KID_RSA,
                "n": testkit::RSA_N, "e": testkit::RSA_E,
            }]
        })
        .to_string();
        let app = axum::Router::new()
            .route(
                "/.well-known/openid-configuration",
                axum::routing::get(move || std::future::ready(discovery.clone())),
            )
            .route(
                "/jwks",
                axum::routing::get(move || std::future::ready(jwks.clone())),
            );
        // The listener is already bound, so the OS accept-backlog absorbs the
        // verifier's connect even before axum's accept loop runs — no sleep.
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        base
    }

    /// Assert the verifier rejected with EXACTLY this reason. Exact-match (not
    /// substring) so a test can't pass on a different-but-also-failing branch
    /// — e.g. "iss" is a substring of "missing kid", and "kid" of both
    /// "missing kid" and "unknown kid (cooldown)". Compared on `Display`, which
    /// is the bare reason for a refused token and a failed provider alike.
    fn assert_invalid(err: OidcError, expected: &str) {
        assert_eq!(err.to_string(), expected, "wrong rejection reason");
    }

    #[tokio::test]
    async fn valid_rs256_token_passes() {
        let v = rsa_verifier();
        let t = mint(
            Algorithm::RS256,
            Some(testkit::KID_RSA),
            &TestClaims::valid(),
        );
        assert!(v.validate(&t).await.is_ok());
    }

    #[tokio::test]
    async fn valid_es256_token_passes() {
        let v = ec_verifier();
        let t = mint(
            Algorithm::ES256,
            Some(testkit::KID_EC),
            &TestClaims::valid(),
        );
        assert!(v.validate(&t).await.is_ok());
    }

    #[tokio::test]
    async fn hs256_symmetric_alg_is_rejected() {
        // The HS256 token is signed with the RSA public modulus as the HMAC
        // secret (the RS256->HS256 confusion an attacker mounts with the known
        // public key — see testkit::mint). Must be refused at the pre-key-lookup
        // allowlist with EXACTLY "alg not allowed" — NOT fall through to
        // build_decoding_key's "alg/key mismatch" (which would also contain "alg").
        let v = rsa_verifier();
        let t = mint(
            Algorithm::HS256,
            Some(testkit::KID_RSA),
            &TestClaims::valid(),
        );
        assert_invalid(v.validate(&t).await.unwrap_err(), "alg not allowed");
    }

    #[tokio::test]
    async fn wrong_audience_is_rejected() {
        let v = rsa_verifier();
        let mut c = TestClaims::valid();
        c.aud = "https://attacker.test/mcp".into();
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(v.validate(&t).await.unwrap_err(), "aud mismatch");
    }

    #[tokio::test]
    async fn wrong_issuer_is_rejected() {
        let v = rsa_verifier();
        let mut c = TestClaims::valid();
        c.iss = "https://attacker.test".into();
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(v.validate(&t).await.unwrap_err(), "iss mismatch");
    }

    #[tokio::test]
    async fn expired_token_is_rejected() {
        let v = rsa_verifier();
        let mut c = TestClaims::valid();
        let n = testkit::now();
        // exp is past `now` by more than the clock-skew leeway → expired.
        c.iat = Some(n - CLOCK_SKEW_LEEWAY_SECS - 600);
        c.exp = n - CLOCK_SKEW_LEEWAY_SECS - 240;
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(v.validate(&t).await.unwrap_err(), "expired");
    }

    #[tokio::test]
    async fn lifetime_over_max_age_cap_is_rejected() {
        let v = rsa_verifier();
        let mut c = TestClaims::valid();
        let n = testkit::now();
        c.iat = Some(n);
        c.exp = n + MAX_TOKEN_AGE_SECS + 1000; // exceeds the hard cap
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(
            v.validate(&t).await.unwrap_err(),
            "token lifetime exceeds cap",
        );
    }

    #[tokio::test]
    async fn future_iat_is_rejected() {
        let v = rsa_verifier();
        let mut c = TestClaims::valid();
        let n = testkit::now();
        c.iat = Some(n + CLOCK_SKEW_LEEWAY_SECS + 240); // beyond leeway
        c.exp = n + CLOCK_SKEW_LEEWAY_SECS + 340;
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(v.validate(&t).await.unwrap_err(), "iat in the future");
    }

    #[tokio::test]
    async fn missing_kid_is_rejected() {
        let v = rsa_verifier();
        let t = mint(Algorithm::RS256, None, &TestClaims::valid());
        assert_invalid(v.validate(&t).await.unwrap_err(), "missing kid");
    }

    #[tokio::test]
    async fn unknown_kid_in_cooldown_is_rejected() {
        let v = rsa_verifier();
        // Cooldown active → an unknown kid fails fast without any outbound fetch.
        v.inner.provider.arm_cooldown();
        let t = mint(Algorithm::RS256, Some("no-such-kid"), &TestClaims::valid());
        assert_invalid(v.validate(&t).await.unwrap_err(), "unknown kid (cooldown)");
    }

    #[tokio::test]
    async fn tampered_signature_is_rejected() {
        let v = rsa_verifier();
        let t = mint(
            Algorithm::RS256,
            Some(testkit::KID_RSA),
            &TestClaims::valid(),
        );
        let parts: Vec<&str> = t.split('.').collect();
        // Flip a char in the signature segment so verification fails.
        let mut sig = parts[2].as_bytes().to_vec();
        let last = sig.len() - 1;
        sig[last] = if sig[last] == b'A' { b'B' } else { b'A' };
        let tampered = format!(
            "{}.{}.{}",
            parts[0],
            parts[1],
            String::from_utf8(sig).unwrap()
        );
        assert!(v.validate(&tampered).await.is_err());
    }

    // ── Live discovery + JWKS fetch against a loopback mock IdP ──────────────
    // Exercises the real network path (OIDC discovery → same-origin jwks_uri →
    // JWKS fetch → signature verify) that runs in production on first request.

    #[tokio::test]
    async fn live_discovery_and_jwks_fetch_validates_token() {
        let base = spawn_idp(IdpFault::None).await;
        let v = OidcVerifier::new(base.clone(), testkit::AUDIENCE.to_string());
        let mut c = TestClaims::valid();
        c.iss = base; // token issuer must equal the configured (loopback) issuer
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert!(
            v.validate(&t).await.is_ok(),
            "token should validate after live discovery + JWKS fetch"
        );
    }

    #[tokio::test]
    async fn live_discovery_issuer_mismatch_is_rejected() {
        let base = spawn_idp(IdpFault::Issuer).await; // discovery advertises a different issuer
        let v = OidcVerifier::new(base.clone(), testkit::AUDIENCE.to_string());
        let mut c = TestClaims::valid();
        c.iss = base;
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(
            v.validate(&t).await.unwrap_err(),
            "discovery issuer mismatch",
        );
    }

    #[tokio::test]
    async fn live_cross_origin_jwks_uri_is_rejected() {
        // discovery `issuer` matches, but `jwks_uri` points at another origin —
        // the key-substitution vector the same-origin check in Provider::discovery() blocks.
        let base = spawn_idp(IdpFault::JwksOrigin).await;
        let v = OidcVerifier::new(base.clone(), testkit::AUDIENCE.to_string());
        let mut c = TestClaims::valid();
        c.iss = base;
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(
            v.validate(&t).await.unwrap_err(),
            "jwks_uri not same-origin",
        );
    }

    // ── alg/key-type confusion: a header alg that doesn't match the resolved
    //    JWK's key type must be refused at build_decoding_key (RS256 and ES256
    //    are both allowlisted, so the header-stage gate does NOT catch this). ──

    #[tokio::test]
    async fn rs256_header_against_ec_key_is_rejected() {
        // kid resolves to the EC JWK, but the header claims RS256.
        let v = ec_verifier();
        let t = mint(
            Algorithm::RS256,
            Some(testkit::KID_EC),
            &TestClaims::valid(),
        );
        assert_invalid(v.validate(&t).await.unwrap_err(), "alg/key mismatch");
    }

    #[tokio::test]
    async fn es256_header_against_rsa_key_is_rejected() {
        let v = rsa_verifier();
        let t = mint(
            Algorithm::ES256,
            Some(testkit::KID_RSA),
            &TestClaims::valid(),
        );
        assert_invalid(v.validate(&t).await.unwrap_err(), "alg/key mismatch");
    }

    #[tokio::test]
    async fn iat_absent_over_cap_is_rejected() {
        // RFC 7519 makes `iat` optional. With no `iat`, the max-age cap is
        // measured against `now` (the branch guarding the saturating-sub
        // underflow). This is the only test that drives the `iat == None` path.
        let v = rsa_verifier();
        let mut c = TestClaims::valid();
        c.iat = None;
        c.exp = testkit::now() + MAX_TOKEN_AGE_SECS + 1000;
        let t = mint(Algorithm::RS256, Some(testkit::KID_RSA), &c);
        assert_invalid(
            v.validate(&t).await.unwrap_err(),
            "token lifetime exceeds cap",
        );
    }
}
