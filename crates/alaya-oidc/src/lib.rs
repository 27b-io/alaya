//! Shared OIDC discovery + JWKS hardening layer.
//!
//! Consumed by `alaya-server` (OAuth resource server: bearer access tokens) and
//! `ops-console` (OIDC relying party: ID tokens after the code exchange). The
//! two copies this replaced drifted by a URL parser within a week (#82); every
//! rule both sides must agree on lives here:
//!
//! - issuer normalisation (one trailing slash) and RFC 6454 origin parsing
//! - discovery over a redirect-disabled, timeout-bounded client ([`Provider::new`]
//!   builds one; a [`Provider::for_relying_party`] caller owns that); the
//!   document's `issuer` must echo the configured one (OIDC Core §4.3)
//! - every discovery and JWKS body read under a byte cap, so a hostile IdP
//!   response cannot OOM either binary
//! - `jwks_uri` must be same-origin with the issuer and https (http only for a
//!   loopback issuer, so local dev works); a relying party's
//!   `authorization_endpoint` and `token_endpoint` must be present and pass the
//!   same rule. Every rule runs before the document is cached, so a refused
//!   one is never served from the cache; it is fetched afresh once the
//!   discovery cooldown expires
//! - discovery and the JWKS cache each fetch single-flight behind their own
//!   per-provider cooldown, extended *before* the fetch, so a down or refused
//!   IdP or an unknown-`kid` flood can't turn either binary into an
//!   outbound-fetch amplifier
//! - one [`Provider::verify`] pipeline: alg allowlist {RS256, ES256} before any
//!   key lookup, JWK → key refusing alg/key-type mismatches, signature plus
//!   `exp` and `aud` with 60 s leeway, then `iss` compared trailing-slash
//!   normalised — never delegated to jsonwebtoken's exact match
//!
//! What stays with the consumer is role-specific: the audience it binds to and
//! the max-token-age cap (server); PKCE, nonce, token exchange (console). So
//! does logging: [`Error::Provider`] carries the cause, and each consumer
//! decides how loudly to record it.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use jsonwebtoken::errors::ErrorKind;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use tokio::sync::{Mutex, RwLock};

/// Minimum interval between JWKS refetches per provider. An unknown-`kid` flood
/// therefore drives at most ~2 outbound fetches/min.
const JWKS_COOLDOWN: Duration = Duration::from_secs(30);

/// Minimum interval between discovery fetches per provider while no document
/// is cached, so an unauthenticated login flood against a down IdP drives at
/// most ~2 outbound fetches/min. Strictly shorter than [`JWKS_COOLDOWN`]:
/// alaya-server reaches discovery only from a JWKS refetch that has just
/// passed that cooldown, and stamps this gate just after that one, so an
/// equal or longer cooldown here could refuse the refetch.
const DISCOVERY_COOLDOWN: Duration = JWKS_COOLDOWN.saturating_sub(Duration::from_secs(1));
const _: () = assert!(DISCOVERY_COOLDOWN.as_nanos() < JWKS_COOLDOWN.as_nanos());

/// Clock-skew leeway applied to `exp` (and, server-side, to `iat`).
pub const CLOCK_SKEW_LEEWAY_SECS: u64 = 60;

/// Ceiling on a discovery or JWKS body. reqwest reads to EOF with no default
/// cap, and the request timeout bounds the seconds, not the bytes a fast link
/// delivers inside them — so without this a substituted IdP answering with an
/// endless body grows the heap until the OOM killer takes the pod. Honest
/// documents are kilobytes.
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug)]
pub enum Error {
    /// A rule refused the input — the token, in [`Provider::verify`]; an
    /// endpoint, in [`same_origin_https`]. The message is a server-safe
    /// `&'static str` (never token internals); the consumer decides whether it
    /// reaches a client — the server returns a generic 401, the console
    /// renders it.
    Invalid(&'static str),
    /// The provider failed: unreachable, a non-2xx, an oversized or
    /// unparseable body, or a discovery document that broke a rule — or it
    /// failed so recently that it was not asked again. `op` is
    /// the same kind of server-safe literal and is all `Display` renders;
    /// `cause` is for an operator log, never a client.
    Provider { op: &'static str, cause: Cause },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Invalid(op) | Error::Provider { op, .. } => write!(f, "{op}"),
        }
    }
}

/// Why an [`Error::Provider`] happened. Wherever it carries text that text is
/// IdP-influenced, so a consumer that logs it records it escaped (`?`, not
/// `%`) and bounded.
///
/// Closed on purpose: no variant holds a `serde_json::Error`, because its
/// `Display` embeds the offending input — and a body that is a bare JSON
/// string makes the whole body that input. [`ParseFailure`] keeps the shape.
#[derive(Debug)]
pub enum Cause {
    Transport(reqwest::Error),
    Status(reqwest::StatusCode),
    /// The body ran past [`MAX_BODY_BYTES`]; the read was abandoned there.
    TooLarge,
    Parse(ParseFailure),
    /// The discovery-document value that broke a rule — attacker-chosen text.
    Document(String),
    /// The discovery document lacks a field the provider's role requires.
    Missing,
    /// A discovery fetch failed or was refused inside [`DISCOVERY_COOLDOWN`],
    /// so this call made no request.
    Cooldown,
}

/// Where and how a body failed to parse, never what it said: a fieldless
/// enum and two `usize` cannot carry input, so the types are the proof.
#[derive(Debug, Clone, Copy)]
pub struct ParseFailure {
    pub category: serde_json::error::Category,
    pub line: usize,
    pub column: usize,
}

impl From<&serde_json::Error> for ParseFailure {
    fn from(e: &serde_json::Error) -> Self {
        ParseFailure {
            category: e.classify(),
            line: e.line(),
            column: e.column(),
        }
    }
}

/// The reasons one provider fetch can fail with, in pipeline order.
struct FetchOps {
    send: &'static str,
    status: &'static str,
    read: &'static str,
    parse: &'static str,
}

const DISCOVERY_OPS: FetchOps = FetchOps {
    send: "discovery failed",
    status: "discovery status",
    read: "discovery read",
    parse: "discovery parse",
};

const JWKS_OPS: FetchOps = FetchOps {
    send: "jwks fetch failed",
    status: "jwks status",
    read: "jwks read",
    parse: "jwks parse",
};

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
    /// omits them. A [`Provider::for_relying_party`] requires both and
    /// origin-checks them before caching, so from it they are always `Some`.
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
    /// Discovery document, filled lazily on first use. Shared with the
    /// detached fetch that fills it.
    discovery: Arc<RwLock<Option<Discovery>>>,
    /// kid -> JWK. Swapped wholesale on refetch.
    keys: RwLock<HashMap<String, Jwk>>,
    /// Single-flight fetch gate; the inner `Instant` is the last fetch time
    /// (cooldown). Holding the mutex serializes refetches across requests.
    fetch_gate: Mutex<Instant>,
    /// Discovery's own single-flight gate and last fetch time. Not
    /// `fetch_gate`: `key_for_kid` holds that one into `discovery()`, and a
    /// tokio mutex is not reentrant. Lock order is `fetch_gate`, then this,
    /// never the reverse. Shared so a detached fetch holds it to the end.
    discovery_gate: Arc<Mutex<Instant>>,
    /// Relying-party role: discovery also requires `authorization_endpoint`
    /// and `token_endpoint`, same-origin-https with the issuer.
    relying_party: bool,
}

impl Provider {
    /// `issuer` is normalised (one trailing slash stripped). Discovery is
    /// deferred to first use — a down provider degrades to a rejection, never
    /// a startup failure. The client follows no redirects (a 3xx on discovery
    /// or JWKS would otherwise substitute keys) and is timeout-bounded.
    ///
    /// This is the resource-server role: it never calls the authorization or
    /// token endpoint, so it must not reject an IdP (e.g. a split-origin one)
    /// over them.
    pub fn new(issuer: &str) -> Self {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build OIDC http client");
        Self::build(issuer, http, false)
    }

    /// The relying-party role: discovery also requires `authorization_endpoint`
    /// and `token_endpoint`, same-origin-https with the issuer, before the
    /// document is cached — the RP redirects the user to one and posts the
    /// authorization code to the other.
    ///
    /// Over the consumer's own client, for a consumer whose every upstream
    /// shares one builder so a hardening knob added there cannot miss the IdP.
    /// The client MUST refuse redirects and bound its timeouts, as `new`'s
    /// does; passing one is taking that on.
    pub fn for_relying_party(issuer: &str, http: reqwest::Client) -> Self {
        Self::build(issuer, http, true)
    }

    fn build(issuer: &str, http: reqwest::Client, relying_party: bool) -> Self {
        Provider {
            issuer: normalize_issuer(issuer).to_string(),
            http,
            discovery: Arc::new(RwLock::new(None)),
            keys: RwLock::new(HashMap::new()),
            // Seeded in the past so neither cooldown blocks the first real fetch.
            fetch_gate: Mutex::new(cooled_down()),
            discovery_gate: Arc::new(Mutex::new(cooled_down())),
            relying_party,
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
    /// fetch — plus, for a relying party, its two endpoints. Only a document
    /// that passes every rule is cached. Until one is, fetches are
    /// single-flight and at most one per [`DISCOVERY_COOLDOWN`]: inside it, a
    /// call after a failed or refused fetch is refused without a request.
    /// The fetch runs detached from its caller, so a caller that goes away
    /// mid-fetch cannot leave the cooldown armed over an empty cache.
    pub async fn discovery(&self) -> Result<Discovery, Error> {
        if let Some(d) = self.discovery.read().await.clone() {
            return Ok(d);
        }

        // Miss: serialize fetches behind the gate, as `key_for_kid` does.
        let mut last_fetch = self.discovery_gate.clone().lock_owned().await;
        // Double-check: a fetch that succeeded while this call waited must
        // be served, not refused by the cooldown it just set.
        if let Some(d) = self.discovery.read().await.clone() {
            return Ok(d);
        }
        if last_fetch.elapsed() < DISCOVERY_COOLDOWN {
            return Err(Error::Provider {
                op: "discovery (cooldown)",
                cause: Cause::Cooldown,
            });
        }
        // Set BEFORE the fetch, so a failing or refused one extends the
        // cooldown too — the unauthenticated login path must not become an
        // outbound-fetch amplifier while the IdP is down.
        *last_fetch = Instant::now();

        // Spawned, holding the gate until it has cached or failed, with no
        // await between arming the cooldown and the spawn. A caller dropped
        // mid-fetch (a client hanging up on login) must neither abort the
        // fetch, which would arm the cooldown over an empty cache and lock
        // every login out, nor release the gate early, which would let a
        // hang-up flood drive one outbound fetch per request.
        let (http, issuer) = (self.http.clone(), self.issuer.clone());
        let (relying_party, cache) = (self.relying_party, self.discovery.clone());
        let fetch = tokio::spawn(async move {
            let _gate = last_fetch;
            let disc = fetch_discovery(&http, &issuer, relying_party).await?;
            *cache.write().await = Some(disc.clone());
            Ok(disc)
        });
        // Never aborted, so a join error is the task's panic: re-raise it
        // here, as the fetch would have panicked inline.
        fetch
            .await
            .unwrap_or_else(|e| std::panic::resume_unwind(e.into_panic()))
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
        let jwks: Jwks = fetch_json(&self.http, &jwks_uri, &JWKS_OPS).await?;
        *self.keys.write().await = key_map(jwks.keys);
        Ok(())
    }
}

/// GET `issuer`'s discovery document and apply every rule to it: the issuer
/// echo, same-origin-https on `jwks_uri`, and, for a relying party, its two
/// endpoints. The caller caches only what this returns.
async fn fetch_discovery(
    http: &reqwest::Client,
    issuer: &str,
    relying_party: bool,
) -> Result<Discovery, Error> {
    let url = format!("{issuer}/.well-known/openid-configuration");
    let disc: Discovery = fetch_json(http, &url, &DISCOVERY_OPS).await?;

    // OIDC Core §4.3: prevents a sibling tenant on a shared origin from
    // serving a discovery document that quietly substitutes keys.
    if normalize_issuer(&disc.issuer) != issuer {
        return Err(Error::Provider {
            op: "discovery issuer mismatch",
            cause: Cause::Document(disc.issuer),
        });
    }
    // Off the issuer's origin is the substituted-IdP signal; the cause
    // records the value, which the bare reason cannot.
    if same_origin_https(issuer, &disc.jwks_uri).is_err() {
        return Err(Error::Provider {
            op: "jwks_uri not same-origin",
            cause: Cause::Document(disc.jwks_uri),
        });
    }
    if relying_party {
        check_rp_endpoints(issuer, &disc)?;
    }
    Ok(disc)
}

/// GET `url` and parse its JSON body, the body read under
/// [`MAX_BODY_BYTES`]. Every failure is an [`Error::Provider`] tagged
/// with the matching reason from `ops`.
async fn fetch_json<T: DeserializeOwned>(
    http: &reqwest::Client,
    url: &str,
    ops: &FetchOps,
) -> Result<T, Error> {
    let fail = |op, cause| Error::Provider { op, cause };
    let resp = http
        .get(url)
        .send()
        .await
        .map_err(|e| fail(ops.send, Cause::Transport(e)))?;
    if !resp.status().is_success() {
        return Err(fail(ops.status, Cause::Status(resp.status())));
    }
    let body = read_capped(resp).await.map_err(|c| fail(ops.read, c))?;
    serde_json::from_slice(&body).map_err(|e| fail(ops.parse, Cause::Parse((&e).into())))
}

/// Read a body, refusing anything past [`MAX_BODY_BYTES`]. Overrun drops the
/// connection mid-body, so nothing past the cap is ever buffered.
async fn read_capped(mut resp: reqwest::Response) -> Result<Vec<u8>, Cause> {
    let mut buf = Vec::new();
    while let Some(chunk) = resp.chunk().await.map_err(Cause::Transport)? {
        if buf.len() + chunk.len() > MAX_BODY_BYTES {
            return Err(Cause::TooLarge);
        }
        buf.extend_from_slice(&chunk);
    }
    Ok(buf)
}

/// Both relying-party endpoints present and same-origin-https with `issuer`,
/// and `authorization_endpoint` fragment-free. Off-origin is the
/// substituted-IdP signal (OIDC Core §4.3): a hostile `token_endpoint` would
/// receive the authorization code and PKCE verifier.
fn check_rp_endpoints(issuer: &str, disc: &Discovery) -> Result<(), Error> {
    for (missing, off_origin, endpoint) in [
        (
            "discovery missing authorization_endpoint",
            "authorization_endpoint not same-origin",
            &disc.authorization_endpoint,
        ),
        (
            "discovery missing token_endpoint",
            "token_endpoint not same-origin",
            &disc.token_endpoint,
        ),
    ] {
        let Some(endpoint) = endpoint else {
            return Err(Error::Provider {
                op: missing,
                cause: Cause::Missing,
            });
        };
        if same_origin_https(issuer, endpoint).is_err() {
            return Err(Error::Provider {
                op: off_origin,
                cause: Cause::Document(endpoint.clone()),
            });
        }
    }
    // RFC 6749 §3.1: the authorization endpoint URI MUST NOT include a
    // fragment. It is the one endpoint the user's browser is sent to.
    if let Some(endpoint) = &disc.authorization_endpoint
        && reqwest::Url::parse(endpoint).is_ok_and(|u| u.fragment().is_some())
    {
        return Err(Error::Provider {
            op: "authorization_endpoint has a fragment",
            cause: Cause::Document(endpoint.clone()),
        });
    }
    Ok(())
}

/// An instant old enough that neither cooldown blocks a fetch after it.
fn cooled_down() -> Instant {
    Instant::now()
        .checked_sub(JWKS_COOLDOWN * 2)
        .unwrap_or_else(Instant::now)
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
    /// Pre-populate the discovery cache. Unchecked: a seeded document skips
    /// every discovery rule, so seed only what a test means to trust.
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

    /// Expire the discovery cooldown, so the next uncached discovery fetches
    /// at once instead of being refused.
    pub fn expire_discovery_cooldown(&self) {
        *self
            .discovery_gate
            .try_lock()
            .expect("uncontended in tests") = cooled_down();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    /// A substituted IdP answering discovery with a body past the cap is
    /// refused at the read, not buffered: the pod's heap is the target.
    /// Served by hand over loopback so the crate needs no HTTP-server dev-dep.
    #[tokio::test]
    async fn oversized_discovery_body_is_refused_not_buffered() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let issuer = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut req = [0u8; 1024];
            let _request_len = sock.read(&mut req).await.unwrap();
            let len = MAX_BODY_BYTES + 1024;
            let head = format!("HTTP/1.1 200 OK\r\ncontent-length: {len}\r\n\r\n");
            sock.write_all(head.as_bytes()).await.unwrap();
            // The client is meant to hang up mid-body, so the write failing
            // with a reset is the outcome under test, not an error.
            let _hung_up = sock.write_all(&vec![b' '; len]).await;
        });

        let err = Provider::new(&issuer).discovery().await.err();
        server.abort();
        assert!(
            matches!(
                err,
                Some(Error::Provider {
                    op: "discovery read",
                    cause: Cause::TooLarge
                })
            ),
            "{err:?}"
        );
    }

    /// A loopback IdP answering every connection, after `delay`, with the
    /// response `respond` builds for its issuer, then closing it — so each
    /// fetch costs exactly one accepted connection and a reused one cannot
    /// hide a request.
    async fn loopback_idp(
        respond: fn(&str) -> String,
        delay: Duration,
    ) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let issuer = format!("http://{}", listener.local_addr().unwrap());
        let response = respond(&issuer);
        let accepted = Arc::new(AtomicUsize::new(0));
        let count = accepted.clone();
        let server = tokio::spawn(async move {
            loop {
                let (mut sock, _) = listener.accept().await.unwrap();
                count.fetch_add(1, Ordering::SeqCst);
                let mut req = [0u8; 1024];
                let _request_len = sock.read(&mut req).await.unwrap();
                tokio::time::sleep(delay).await;
                sock.write_all(response.as_bytes()).await.unwrap();
                let _closed = sock.shutdown().await;
            }
        });
        (issuer, accepted, server)
    }

    fn http_response(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    /// A resource-server discovery document that passes every rule.
    fn good_discovery(issuer: &str) -> String {
        let doc = serde_json::json!({
            "issuer": issuer,
            "jwks_uri": format!("{issuer}/jwks"),
        });
        http_response("200 OK", &doc.to_string())
    }

    /// A down IdP costs one outbound fetch per cooldown, not one per call:
    /// the unauthenticated login path must not amplify an outage.
    #[tokio::test]
    async fn a_failed_discovery_is_not_retried_inside_the_cooldown() {
        let (issuer, accepted, server) = loopback_idp(
            |_| http_response("500 Internal Server Error", ""),
            Duration::ZERO,
        )
        .await;
        let provider = Provider::new(&issuer);

        let first = provider.discovery().await.err();
        assert!(matches!(first, Some(Error::Provider { .. })), "{first:?}");
        let second = provider.discovery().await.err();
        server.abort();
        assert!(
            matches!(
                second,
                Some(Error::Provider {
                    op: "discovery (cooldown)",
                    cause: Cause::Cooldown
                })
            ),
            "{second:?}"
        );
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    /// The gate guards the miss only: a cached document is served even with
    /// the cooldown active, or one refusal would outlive the fetch that
    /// fixed it.
    #[tokio::test]
    async fn a_cached_discovery_is_served_inside_the_cooldown() {
        let (issuer, accepted, server) = loopback_idp(good_discovery, Duration::ZERO).await;
        let provider = Provider::new(&issuer);

        provider
            .discovery()
            .await
            .expect("first discovery succeeds");
        *provider.discovery_gate.try_lock().unwrap() = Instant::now();
        let cached = provider.discovery().await;
        server.abort();
        assert!(cached.is_ok(), "{:?}", cached.err());
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    /// A caller dropped mid-fetch — a client hanging up on login — neither
    /// aborts the fetch nor leaves the cooldown armed over an empty cache:
    /// the next call is served what the detached fetch cached.
    #[tokio::test]
    async fn a_cancelled_discovery_does_not_lock_out_the_next_call() {
        let (issuer, accepted, server) =
            loopback_idp(good_discovery, Duration::from_millis(200)).await;
        let provider = Provider::new(&issuer);

        let hung_up = tokio::time::timeout(Duration::from_millis(50), provider.discovery()).await;
        assert!(
            hung_up.is_err(),
            "the first caller must be dropped mid-fetch"
        );
        let next = provider.discovery().await;
        server.abort();
        assert!(next.is_ok(), "{:?}", next.err());
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    /// Concurrent misses on a cold cache fetch once, and the call that waited
    /// on the gate is served what the first one cached — not refused by the
    /// cooldown that fetch set.
    #[tokio::test]
    async fn concurrent_cold_discoveries_fetch_once_and_all_succeed() {
        let (issuer, accepted, server) = loopback_idp(good_discovery, Duration::ZERO).await;
        let provider = Provider::new(&issuer);

        let (a, b) = tokio::join!(provider.discovery(), provider.discovery());
        server.abort();
        assert!(a.is_ok() && b.is_ok(), "{:?} / {:?}", a.err(), b.err());
        assert_eq!(accepted.load(Ordering::SeqCst), 1);
    }

    fn rp_discovery(authorization_endpoint: &str) -> Discovery {
        Discovery {
            issuer: "https://id.test".into(),
            jwks_uri: "https://id.test/jwks".into(),
            authorization_endpoint: Some(authorization_endpoint.into()),
            token_endpoint: Some("https://id.test/token".into()),
        }
    }

    #[test]
    fn an_authorization_endpoint_fragment_is_refused() {
        for endpoint in ["https://id.test/authorize#x", "https://id.test/authorize#"] {
            let err = check_rp_endpoints("https://id.test", &rp_discovery(endpoint)).err();
            assert!(
                matches!(
                    err,
                    Some(Error::Provider {
                        op: "authorization_endpoint has a fragment",
                        cause: Cause::Document(_)
                    })
                ),
                "{endpoint}: {err:?}"
            );
        }
    }

    /// RFC 6749 §3.1 allows a query; the console merges its own into it.
    #[test]
    fn an_authorization_endpoint_query_is_accepted() {
        let disc = rp_discovery("https://id.test/authorize?tenant=a");
        assert!(check_rp_endpoints("https://id.test", &disc).is_ok());
    }
}
