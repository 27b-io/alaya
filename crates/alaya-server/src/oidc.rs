//! OIDC token verification — provider-agnostic OAuth Resource Server side.
//!
//! Validates a bearer JWT against a configured issuer's JWKS:
//! - lazy, non-fatal OIDC discovery (a down provider degrades to 401, never crashes)
//! - redirect-following disabled on discovery + JWKS fetches (blocks key substitution)
//! - `jwks_uri` must be HTTPS and same-origin as the issuer (RFC 8414 §3)
//! - JWKS cache with single-flight refetch + per-issuer cooldown (unknown-`kid`
//!   floods can't drive unbounded outbound fetches)
//! - alg allowlist {RS256, ES256}; `none`/HS* hard-rejected
//! - `iss` (trailing-slash normalized), `aud` (string or array), `exp`, and a
//!   hard max-token-age cap
//!
//! The verifier lives on the axum side (Send+Sync); it never runs behind the
//! service-worker channel.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use tokio::sync::{Mutex, RwLock};

/// Hard cap on accepted token lifetime (`exp - iat`), regardless of issuer.
/// There is no revocation, so this bounds the compromise window.
const MAX_TOKEN_AGE_SECS: u64 = 3600;

/// Minimum interval between JWKS refetches per issuer. An unknown-`kid` flood
/// therefore drives at most ~2 outbound fetches/min.
const JWKS_COOLDOWN: Duration = Duration::from_secs(30);

const CLOCK_SKEW_LEEWAY_SECS: u64 = 60;

#[derive(Debug)]
pub enum OidcError {
    /// Any validation failure. The message is for server logs only — callers
    /// must return a generic 401 without leaking token internals.
    Invalid(&'static str),
}

impl std::fmt::Display for OidcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OidcError::Invalid(m) => write!(f, "{m}"),
        }
    }
}

#[derive(Deserialize)]
struct Discovery {
    /// OIDC Core §4.3 requires the RP verify this equals the configured issuer.
    issuer: String,
    jwks_uri: String,
}

#[derive(Deserialize, Clone)]
struct Jwk {
    kty: String,
    kid: Option<String>,
    n: Option<String>,
    e: Option<String>,
    x: Option<String>,
    y: Option<String>,
}

#[derive(Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

#[derive(Deserialize)]
struct Claims {
    iss: String,
    /// Optional per RFC 7519 §4.1.6 — `None` means the IdP omitted it.
    iat: Option<u64>,
    exp: u64,
}

/// Strip a single trailing slash for issuer comparison.
fn normalize_issuer(s: &str) -> &str {
    s.strip_suffix('/').unwrap_or(s)
}

/// (scheme, host, port) of a URL, for same-origin checks (RFC 6454).
/// Handles IPv6 bracketed literals. Returns None if the URL is not absolute
/// http(s). The port is normalized to the scheme default (443/80) when
/// omitted, so `https://idp` and `https://idp:443` compare equal — but
/// `https://idp:8443` does NOT, blocking same-host different-port forgeries.
fn origin_of(url: &str) -> Option<(String, String, u16)> {
    let (scheme, rest) = url.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    let default_port: u16 = match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    let authority = rest.split('/').next()?;
    let (host, port_opt) = if let Some(inner) = authority.strip_prefix('[') {
        // IPv6 literal: `[host]` or `[host]:port`.
        let end = inner.find(']')?;
        let host = &inner[..end];
        let after = &inner[end + 1..];
        let port = after.strip_prefix(':').and_then(|p| p.parse::<u16>().ok());
        (host, port)
    } else {
        match authority.split_once(':') {
            Some((h, p)) => (h, p.parse::<u16>().ok()),
            None => (authority, None),
        }
    };
    Some((
        scheme,
        host.to_ascii_lowercase(),
        port_opt.unwrap_or(default_port),
    ))
}

struct Inner {
    issuer: String,
    audience: String,
    client: reqwest::Client,
    /// Discovered `jwks_uri`, filled lazily on first use.
    jwks_uri: RwLock<Option<String>>,
    /// kid -> JWK. Swapped wholesale on refetch.
    keys: RwLock<HashMap<String, Jwk>>,
    /// Single-flight fetch gate; the inner `Instant` is the last fetch time
    /// (cooldown). Holding the mutex serializes refetches across requests.
    fetch_gate: Mutex<Instant>,
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
        let normalized = normalize_issuer(&issuer);
        // Parse the issuer host to determine loopback eligibility; a real IP
        // parser defeats `http://127.0.0.1.evil.com` style confusables.
        let is_loopback = origin_of(normalized)
            .map(|(_, host, _)| {
                host == "localhost"
                    || host
                        .parse::<std::net::IpAddr>()
                        .map(|ip| ip.is_loopback())
                        .unwrap_or(false)
            })
            .unwrap_or(false);
        if !normalized.starts_with("https://") && !is_loopback {
            panic!("OIDC_ISSUER must be https:// (except loopback): {issuer}");
        }

        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build OIDC http client");

        // last-fetch seeded in the past so the first real fetch isn't blocked
        // by the cooldown.
        let seeded = Instant::now()
            .checked_sub(JWKS_COOLDOWN * 2)
            .unwrap_or_else(Instant::now);

        Self {
            inner: Arc::new(Inner {
                issuer: normalize_issuer(&issuer).to_string(),
                audience,
                client,
                jwks_uri: RwLock::new(None),
                keys: RwLock::new(HashMap::new()),
                fetch_gate: Mutex::new(seeded),
            }),
        }
    }

    /// The configured issuer (normalized), for protected-resource metadata.
    pub fn issuer(&self) -> &str {
        &self.inner.issuer
    }

    /// Validate a bearer token. Returns Ok on a fully-valid token; any failure
    /// is `OidcError::Invalid` and must surface to the client as a generic 401.
    pub async fn validate(&self, token: &str) -> Result<(), OidcError> {
        // 1. Header: alg allowlist + kid. Rejects `none`/HS* before key lookup.
        let header = decode_header(token).map_err(|_| OidcError::Invalid("bad header"))?;
        if !matches!(header.alg, Algorithm::RS256 | Algorithm::ES256) {
            return Err(OidcError::Invalid("alg not allowed"));
        }
        let kid = header.kid.ok_or(OidcError::Invalid("missing kid"))?;

        // 2. Resolve the signing key (cache → single-flight refetch on miss).
        let jwk = self.key_for_kid(&kid).await?;
        let decoding_key = build_decoding_key(&jwk, header.alg)?;

        // 3. Validate signature + registered claims.
        let mut validation = Validation::new(header.alg);
        validation.set_audience(&[&self.inner.audience]);
        validation.validate_aud = true;
        validation.validate_exp = true;
        validation.leeway = CLOCK_SKEW_LEEWAY_SECS;
        // We validate `iss` manually (trailing-slash normalized), so don't let
        // jsonwebtoken require/exact-match it.
        validation.set_required_spec_claims(&["exp", "aud"]);

        let data =
            decode::<Claims>(token, &decoding_key, &validation).map_err(|e| match e.kind() {
                jsonwebtoken::errors::ErrorKind::InvalidAudience => {
                    OidcError::Invalid("aud mismatch")
                }
                jsonwebtoken::errors::ErrorKind::ExpiredSignature => OidcError::Invalid("expired"),
                jsonwebtoken::errors::ErrorKind::InvalidSignature => {
                    OidcError::Invalid("bad signature")
                }
                _ => OidcError::Invalid("invalid token"),
            })?;

        // 4. Issuer (normalized) + hard max-age cap.
        if normalize_issuer(&data.claims.iss) != self.inner.issuer {
            return Err(OidcError::Invalid("iss mismatch"));
        }
        // Cap the lifetime regardless of issuer (no revocation, so this bounds
        // the compromise window). RFC 7519 §4.1.6 makes `iat` OPTIONAL: when
        // present, cap on the issued lifetime (`exp - iat`); when absent, cap
        // on the remaining lifetime (`exp - now`).
        let cap_exceeded = match data.claims.iat {
            Some(iat) => data.claims.exp.saturating_sub(iat) > MAX_TOKEN_AGE_SECS,
            None => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                data.claims.exp.saturating_sub(now) > MAX_TOKEN_AGE_SECS
            }
        };
        if cap_exceeded {
            return Err(OidcError::Invalid("token lifetime exceeds cap"));
        }
        Ok(())
    }

    /// Look up a key by `kid`; on a miss, single-flight refetch subject to the
    /// per-issuer cooldown.
    async fn key_for_kid(&self, kid: &str) -> Result<Jwk, OidcError> {
        if let Some(jwk) = self.inner.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }

        // Miss: serialize refetch attempts behind the gate.
        let mut last_fetch = self.inner.fetch_gate.lock().await;
        // Double-check: another task may have just populated the cache.
        if let Some(jwk) = self.inner.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }
        if last_fetch.elapsed() < JWKS_COOLDOWN {
            // Cooldown active — refuse to hammer the provider on an unknown kid.
            return Err(OidcError::Invalid("unknown kid (cooldown)"));
        }

        // Set the cooldown BEFORE attempting the fetch. If the IdP is down,
        // every failing refetch must still extend the cooldown — otherwise a
        // bad provider becomes an unbounded outbound-fetch amplifier.
        *last_fetch = Instant::now();
        self.refetch_jwks().await?;

        self.inner
            .keys
            .read()
            .await
            .get(kid)
            .cloned()
            .ok_or(OidcError::Invalid("unknown kid"))
    }

    /// Discover (if needed) and fetch the JWKS, swapping the key cache.
    async fn refetch_jwks(&self) -> Result<(), OidcError> {
        let jwks_uri = self.jwks_uri().await?;
        let resp = self
            .inner
            .client
            .get(&jwks_uri)
            .send()
            .await
            .map_err(|_| OidcError::Invalid("jwks fetch failed"))?;
        if !resp.status().is_success() {
            return Err(OidcError::Invalid("jwks status"));
        }
        let jwks: Jwks = resp
            .json()
            .await
            .map_err(|_| OidcError::Invalid("jwks parse"))?;

        let mut map = HashMap::new();
        for jwk in jwks.keys {
            if let Some(kid) = jwk.kid.clone() {
                map.insert(kid, jwk);
            }
        }
        *self.inner.keys.write().await = map;
        Ok(())
    }

    /// Lazily discover and cache the `jwks_uri`, enforcing same-origin + HTTPS.
    async fn jwks_uri(&self) -> Result<String, OidcError> {
        if let Some(uri) = self.inner.jwks_uri.read().await.clone() {
            return Ok(uri);
        }

        let url = format!("{}/.well-known/openid-configuration", self.inner.issuer);
        let resp = self
            .inner
            .client
            .get(&url)
            .send()
            .await
            .map_err(|_| OidcError::Invalid("discovery failed"))?;
        if !resp.status().is_success() {
            return Err(OidcError::Invalid("discovery status"));
        }
        let disc: Discovery = resp
            .json()
            .await
            .map_err(|_| OidcError::Invalid("discovery parse"))?;

        // OIDC Core §4.3: the discovery doc's `issuer` claim must equal the
        // configured issuer. Prevents a sibling tenant on a shared origin from
        // serving a discovery document that quietly substitutes keys.
        if normalize_issuer(&disc.issuer) != self.inner.issuer {
            return Err(OidcError::Invalid("discovery issuer mismatch"));
        }

        // jwks_uri must be HTTPS and same-origin (RFC 6454: scheme + host + port)
        // as the issuer. Default ports (443/80) are normalized into the tuple,
        // so `https://idp` ≡ `https://idp:443` but `https://idp:8443` does not
        // match — closing the same-host different-port forgery vector.
        let jwks_origin = origin_of(&disc.jwks_uri).ok_or(OidcError::Invalid("jwks_uri form"))?;
        let issuer_origin =
            origin_of(&self.inner.issuer).ok_or(OidcError::Invalid("issuer form"))?;
        if jwks_origin.0 != "https" || jwks_origin != issuer_origin {
            return Err(OidcError::Invalid("jwks_uri not same-origin https"));
        }

        *self.inner.jwks_uri.write().await = Some(disc.jwks_uri.clone());
        Ok(disc.jwks_uri)
    }
}

fn build_decoding_key(jwk: &Jwk, alg: Algorithm) -> Result<DecodingKey, OidcError> {
    match (jwk.kty.as_str(), alg) {
        ("RSA", Algorithm::RS256) => {
            let n = jwk.n.as_deref().ok_or(OidcError::Invalid("rsa n"))?;
            let e = jwk.e.as_deref().ok_or(OidcError::Invalid("rsa e"))?;
            DecodingKey::from_rsa_components(n, e).map_err(|_| OidcError::Invalid("rsa key"))
        }
        ("EC", Algorithm::ES256) => {
            let x = jwk.x.as_deref().ok_or(OidcError::Invalid("ec x"))?;
            let y = jwk.y.as_deref().ok_or(OidcError::Invalid("ec y"))?;
            DecodingKey::from_ec_components(x, y).map_err(|_| OidcError::Invalid("ec key"))
        }
        // alg/key-type mismatch (e.g. HS* against an RSA key) lands here.
        _ => Err(OidcError::Invalid("alg/key mismatch")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_issuer_strips_one_trailing_slash() {
        assert_eq!(normalize_issuer("https://id.27b.io/"), "https://id.27b.io");
        assert_eq!(normalize_issuer("https://id.27b.io"), "https://id.27b.io");
    }

    #[test]
    fn origin_extracts_scheme_host_and_default_port() {
        // No explicit port → scheme's default port.
        assert_eq!(
            origin_of("https://id.27b.io/.well-known/jwks.json"),
            Some(("https".into(), "id.27b.io".into(), 443))
        );
        // Explicit default port → same triple as no port.
        assert_eq!(
            origin_of("https://id.27b.io:443"),
            Some(("https".into(), "id.27b.io".into(), 443))
        );
        // Non-default port → distinct triple (same-host different-port blocked).
        assert_ne!(
            origin_of("https://id.27b.io:8443"),
            origin_of("https://id.27b.io")
        );
        // IPv6 literal with port.
        assert_eq!(
            origin_of("https://[::1]:8443/jwks"),
            Some(("https".into(), "::1".into(), 8443))
        );
        // Unknown scheme.
        assert_eq!(origin_of("ftp://foo"), None);
        assert_eq!(origin_of("not-a-url"), None);
    }
}
