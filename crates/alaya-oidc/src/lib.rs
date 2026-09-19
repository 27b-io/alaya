//! Shared OIDC discovery + JWKS hardening layer.
//!
//! Consumed by `alaya-server` (OAuth resource server: bearer access tokens) and
//! `ops-console` (OIDC relying party: ID tokens after the code exchange). The
//! two copies this replaced drifted by a URL parser within a week (#82); every
//! rule both sides must agree on lives here:
//!
//! - issuer normalisation (one trailing slash) and RFC 6454 origin parsing
//! - discovery over a redirect-disabled, timeout-bounded client; the document's
//!   `issuer` must echo the configured one (OIDC Core §4.3)
//! - `jwks_uri` must be same-origin with the issuer and https (http only for a
//!   loopback issuer, so local dev works); the relying party applies the same
//!   rule to its own endpoints through [`same_origin_https`]
//! - JWKS cache with single-flight refetch and a per-provider cooldown that is
//!   extended *before* the fetch, so an unknown-`kid` flood or a down IdP can't
//!   turn either binary into an outbound-fetch amplifier
//! - one [`Provider::verify`] pipeline: alg allowlist {RS256, ES256} before any
//!   key lookup, JWK → key refusing alg/key-type mismatches, signature plus
//!   `exp` and `aud` with 60 s leeway, then `iss` compared trailing-slash
//!   normalised — never delegated to jsonwebtoken's exact match
//!
//! What stays with the consumer is role-specific: the audience it binds to and
//! the max-token-age cap (server); PKCE, nonce, token exchange (console).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use jsonwebtoken::errors::ErrorKind;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::sync::{Mutex, RwLock};

/// Minimum interval between JWKS refetches per provider. An unknown-`kid` flood
/// therefore drives at most ~2 outbound fetches/min.
const JWKS_COOLDOWN: Duration = Duration::from_secs(30);

/// Clock-skew leeway applied to `exp` (and, server-side, to `iat`).
pub const CLOCK_SKEW_LEEWAY_SECS: u64 = 60;

#[derive(Debug)]
pub enum Error {
    /// Any validation failure. The message is a server-safe `&'static str`
    /// (never token internals); the consumer decides whether it reaches a
    /// client — the server returns a generic 401, the console renders it.
    Invalid(&'static str),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for Error {}

/// Claims a [`Provider::verify`] caller decodes into. The one field the shared
/// pipeline must read is `iss`; everything else is the consumer's.
pub trait IssuedClaims: DeserializeOwned {
    fn iss(&self) -> &str;
}

/// The subset of the OIDC discovery document both roles read.
#[derive(Deserialize, Clone)]
pub struct Discovery {
    /// OIDC Core §4.3 requires the RP verify this equals the configured issuer.
    pub issuer: String,
    pub jwks_uri: String,
    /// Optional because a resource server never uses them and its test IdP
    /// omits them. Only `jwks_uri` is origin-checked here — a relying party
    /// requires both of these at use time and checks them itself.
    pub authorization_endpoint: Option<String>,
    pub token_endpoint: Option<String>,
}

#[derive(Deserialize, Clone)]
pub struct Jwk {
    pub kty: String,
    pub kid: Option<String>,
    pub n: Option<String>,
    pub e: Option<String>,
    pub x: Option<String>,
    pub y: Option<String>,
}

#[derive(Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

/// Strip a single trailing slash for issuer comparison.
fn normalize_issuer(s: &str) -> &str {
    s.strip_suffix('/').unwrap_or(s)
}

/// (scheme, host, port) of an absolute http(s) URL, for same-origin checks
/// (RFC 6454). The port is normalized to the scheme default (443/80) when
/// omitted, so `https://idp` and `https://idp:443` compare equal — but
/// `https://idp:8443` does NOT, blocking same-host different-port forgeries.
/// IPv6 literals come back bare (`::1`, not `[::1]`).
///
/// Parsed with `reqwest::Url`, never by hand: the split-on-`://` parser this
/// replaced read the userinfo of `http://127.0.0.1:@evil.com` as the host
/// (#100).
pub fn origin_of(url: &str) -> Option<(String, String, u16)> {
    let parsed = reqwest::Url::parse(url).ok()?;
    let scheme = parsed.scheme().to_string();
    let default_port: u16 = match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    let host = parsed.host_str()?;
    let host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    Some((
        scheme,
        host.to_ascii_lowercase(),
        parsed.port().unwrap_or(default_port),
    ))
}

/// True when the origin's host is `localhost` or a loopback IP. A real IP
/// parser defeats `http://127.0.0.1.evil.com`-style confusables.
pub fn is_loopback_origin(origin: &(String, String, u16)) -> bool {
    origin.1 == "localhost"
        || origin
            .1
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// `endpoint` must be same-origin with `issuer` and https. http is accepted
/// only when the issuer itself is loopback, so local dev against a local IdP
/// works. A URL with no parseable origin has no origin to match, so it fails
/// the same way. Callers name the endpoint in their own rejection reason.
pub fn same_origin_https(issuer: &str, endpoint: &str) -> Result<(), Error> {
    let same_origin = match (origin_of(issuer), origin_of(endpoint)) {
        (Some(issuer_origin), Some(ep_origin)) => {
            let scheme_ok = ep_origin.0 == "https"
                || (is_loopback_origin(&issuer_origin) && ep_origin.0 == "http");
            scheme_ok && ep_origin == issuer_origin
        }
        _ => false,
    };
    if !same_origin {
        return Err(Error::Invalid("endpoint not same-origin with issuer"));
    }
    Ok(())
}

/// The one validation policy: signature over `alg`, exact `audience`, `exp`
/// and `aud` required, clock-skew leeway. `iss` is deliberately left to
/// [`Provider::verify`] so the comparison is trailing-slash normalised.
fn validation(alg: Algorithm, audience: &str) -> Validation {
    let mut v = Validation::new(alg);
    v.set_audience(&[audience]);
    v.validate_aud = true;
    v.validate_exp = true;
    v.leeway = CLOCK_SKEW_LEEWAY_SECS;
    v.set_required_spec_claims(&["exp", "aud"]);
    v
}

/// One OIDC provider: lazy discovery plus a JWKS cache behind a single-flight,
/// cooled-down refetch. `Send + Sync`; wrap in an `Arc` to share.
pub struct Provider {
    issuer: String,
    http: reqwest::Client,
    /// Discovery document, filled lazily on first use.
    discovery: RwLock<Option<Discovery>>,
    /// kid -> JWK. Swapped wholesale on refetch.
    keys: RwLock<HashMap<String, Jwk>>,
    /// Single-flight fetch gate; the inner `Instant` is the last fetch time
    /// (cooldown). Holding the mutex serializes refetches across requests.
    fetch_gate: Mutex<Instant>,
}

impl Provider {
    /// `issuer` is normalised (one trailing slash stripped). Discovery is
    /// deferred to first use — a down provider degrades to a rejection, never
    /// a startup failure. The client follows no redirects (a 3xx on discovery
    /// or JWKS would otherwise substitute keys) and is timeout-bounded.
    pub fn new(issuer: &str) -> Self {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build OIDC http client");
        // Seeded in the past so the first real fetch isn't blocked by the cooldown.
        let seeded = Instant::now()
            .checked_sub(JWKS_COOLDOWN * 2)
            .unwrap_or_else(Instant::now);
        Provider {
            issuer: normalize_issuer(issuer).to_string(),
            http,
            discovery: RwLock::new(None),
            keys: RwLock::new(HashMap::new()),
            fetch_gate: Mutex::new(seeded),
        }
    }

    /// The configured issuer, normalised.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// The hardened client (no redirects, bounded timeouts) for consumer calls
    /// to the same provider, e.g. the token endpoint.
    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Fetch (once) and cache the discovery document, enforcing the issuer
    /// echo and same-origin-https on `jwks_uri` — the one endpoint both roles
    /// fetch. A resource server never calls the other endpoints, so it must
    /// not reject an IdP (e.g. a split-origin one) over them.
    pub async fn discovery(&self) -> Result<Discovery, Error> {
        if let Some(d) = self.discovery.read().await.clone() {
            return Ok(d);
        }
        let url = format!("{}/.well-known/openid-configuration", self.issuer);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|_| Error::Invalid("discovery failed"))?;
        if !resp.status().is_success() {
            return Err(Error::Invalid("discovery status"));
        }
        let disc: Discovery = resp
            .json()
            .await
            .map_err(|_| Error::Invalid("discovery parse"))?;

        // OIDC Core §4.3: prevents a sibling tenant on a shared origin from
        // serving a discovery document that quietly substitutes keys.
        if normalize_issuer(&disc.issuer) != self.issuer {
            return Err(Error::Invalid("discovery issuer mismatch"));
        }
        same_origin_https(&self.issuer, &disc.jwks_uri)
            .map_err(|_| Error::Invalid("jwks_uri not same-origin"))?;

        *self.discovery.write().await = Some(disc.clone());
        Ok(disc)
    }

    /// Verify a compact JWS against this provider and decode its claims:
    /// header gate (alg allowlist rejects `none`/HS* before any key lookup,
    /// `kid` required), key from the cache or a single-flight refetch, JWK →
    /// key refusing alg/key-type mismatches, signature + registered claims
    /// under [`validation`] for `audience`, then the normalised `iss` check.
    /// Rejection reasons are server-safe literals: `aud mismatch`, `expired`,
    /// `bad signature`, `iss mismatch`, or `invalid token` for anything else.
    /// Role-specific checks (nonce, max-age cap) are the caller's, on the
    /// returned claims.
    pub async fn verify<C: IssuedClaims>(&self, token: &str, audience: &str) -> Result<C, Error> {
        let header = decode_header(token).map_err(|_| Error::Invalid("bad header"))?;
        if !matches!(header.alg, Algorithm::RS256 | Algorithm::ES256) {
            return Err(Error::Invalid("alg not allowed"));
        }
        let kid = header.kid.ok_or(Error::Invalid("missing kid"))?;
        let jwk = self.key_for_kid(&kid).await?;
        let decoding_key = build_decoding_key(&jwk, header.alg)?;

        let data =
            decode::<C>(token, &decoding_key, &validation(header.alg, audience)).map_err(|e| {
                match e.kind() {
                    ErrorKind::InvalidAudience => Error::Invalid("aud mismatch"),
                    ErrorKind::ExpiredSignature => Error::Invalid("expired"),
                    ErrorKind::InvalidSignature => Error::Invalid("bad signature"),
                    _ => Error::Invalid("invalid token"),
                }
            })?;
        if normalize_issuer(data.claims.iss()) != self.issuer {
            return Err(Error::Invalid("iss mismatch"));
        }
        Ok(data.claims)
    }

    /// Look up a key by `kid`; on a miss, single-flight refetch subject to the
    /// per-provider cooldown.
    async fn key_for_kid(&self, kid: &str) -> Result<Jwk, Error> {
        if let Some(jwk) = self.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }

        // Miss: serialize refetch attempts behind the gate.
        let mut last_fetch = self.fetch_gate.lock().await;
        // Double-check: another task may have just populated the cache.
        if let Some(jwk) = self.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }
        if last_fetch.elapsed() < JWKS_COOLDOWN {
            // Cooldown active — refuse to hammer the provider on an unknown kid.
            return Err(Error::Invalid("unknown kid (cooldown)"));
        }

        // Set the cooldown BEFORE attempting the fetch. If the IdP is down,
        // every failing refetch must still extend the cooldown — otherwise a
        // bad provider becomes an unbounded outbound-fetch amplifier.
        *last_fetch = Instant::now();
        self.refetch_jwks().await?;

        self.keys
            .read()
            .await
            .get(kid)
            .cloned()
            .ok_or(Error::Invalid("unknown kid"))
    }

    /// Discover (if needed) and fetch the JWKS, swapping the key cache.
    async fn refetch_jwks(&self) -> Result<(), Error> {
        let jwks_uri = self.discovery().await?.jwks_uri;
        let resp = self
            .http
            .get(&jwks_uri)
            .send()
            .await
            .map_err(|_| Error::Invalid("jwks fetch failed"))?;
        if !resp.status().is_success() {
            return Err(Error::Invalid("jwks status"));
        }
        let jwks: Jwks = resp
            .json()
            .await
            .map_err(|_| Error::Invalid("jwks parse"))?;
        *self.keys.write().await = key_map(jwks.keys);
        Ok(())
    }
}

/// Index JWKs by `kid`; keys without one are unaddressable and dropped.
fn key_map(keys: impl IntoIterator<Item = Jwk>) -> HashMap<String, Jwk> {
    keys.into_iter()
        .filter_map(|jwk| jwk.kid.clone().map(|kid| (kid, jwk)))
        .collect()
}

fn build_decoding_key(jwk: &Jwk, alg: Algorithm) -> Result<DecodingKey, Error> {
    match (jwk.kty.as_str(), alg) {
        ("RSA", Algorithm::RS256) => {
            let n = jwk.n.as_deref().ok_or(Error::Invalid("rsa n"))?;
            let e = jwk.e.as_deref().ok_or(Error::Invalid("rsa e"))?;
            DecodingKey::from_rsa_components(n, e).map_err(|_| Error::Invalid("rsa key"))
        }
        ("EC", Algorithm::ES256) => {
            let x = jwk.x.as_deref().ok_or(Error::Invalid("ec x"))?;
            let y = jwk.y.as_deref().ok_or(Error::Invalid("ec y"))?;
            DecodingKey::from_ec_components(x, y).map_err(|_| Error::Invalid("ec key"))
        }
        // alg/key-type mismatch (e.g. HS* against an RSA key) lands here.
        _ => Err(Error::Invalid("alg/key mismatch")),
    }
}

/// Test seams: let a consumer's unit tests bypass the network. The locks are
/// taken non-blocking because a test holds the only reference.
#[cfg(feature = "test-seams")]
impl Provider {
    /// Pre-populate the discovery cache.
    pub fn seed_discovery(&self, discovery: Discovery) {
        *self.discovery.try_write().expect("uncontended in tests") = Some(discovery);
    }

    /// Pre-populate the key cache (kid-less keys are dropped, as on refetch).
    pub fn seed_keys(&self, keys: impl IntoIterator<Item = Jwk>) {
        *self.keys.try_write().expect("uncontended in tests") = key_map(keys);
    }

    /// Make the cooldown active now, so the next unknown `kid` fails fast
    /// without any outbound fetch.
    pub fn arm_cooldown(&self) {
        *self.fetch_gate.try_lock().expect("uncontended in tests") = Instant::now();
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

    #[test]
    fn origin_reads_the_real_host_behind_userinfo() {
        // The #100 vector: a hand parser split on the first ':' and took the
        // loopback userinfo as the host. The origin is evil.com, so it must NOT
        // pass a same-origin check against a loopback issuer.
        assert_eq!(
            origin_of("http://127.0.0.1:@evil.com"),
            Some(("http".into(), "evil.com".into(), 80))
        );
        assert!(same_origin_https("http://127.0.0.1", "http://127.0.0.1:@evil.com").is_err());
    }

    #[test]
    fn same_origin_rejects_cross_origin_and_port_forgery() {
        assert!(same_origin_https("https://id.27b.io", "https://id.27b.io/jwks").is_ok());
        assert!(same_origin_https("https://id.27b.io", "https://evil.io/jwks").is_err());
        assert!(same_origin_https("https://id.27b.io", "https://id.27b.io:8443/jwks").is_err());
        assert!(same_origin_https("https://id.27b.io", "http://id.27b.io/jwks").is_err());
        // Loopback issuer may use http endpoints (local dev).
        assert!(same_origin_https("http://localhost:8787", "http://localhost:8787/jwks").is_ok());
    }
}
