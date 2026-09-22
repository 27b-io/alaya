//! OIDC Relying Party — authorization-code flow with PKCE (S256) against
//! id.27b.io, mirroring the hardening posture of alaya-server's resource-side
//! verifier (`crates/alaya-server/src/oidc.rs`):
//!
//! - redirect-following disabled on discovery / JWKS / token fetches
//! - discovery `issuer` must echo the configured issuer (OIDC Core §4.3)
//! - `jwks_uri` and `token_endpoint` must be HTTPS and same-origin with the
//!   issuer (loopback issuer may use http for local dev)
//! - alg allowlist {RS256, ES256}; `none`/HS* rejected before key lookup
//! - JWKS cooldown so an unknown-`kid` flood can't drive unbounded fetches
//! - every IdP body read through `http::body_text`, so an unauthenticated
//!   caller cannot make a hostile IdP response OOM the console
//! - `redirect_uri` is pinned from config; never derived from request headers
//!
//! Client authentication at the token endpoint is `client_secret_basic`
//! (RFC 6749 §2.3.1 — the scheme servers MUST support).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use base64::Engine;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, RwLock};

const JWKS_COOLDOWN: Duration = Duration::from_secs(30);
const CLOCK_SKEW_LEEWAY_SECS: u64 = 60;

#[derive(Debug)]
pub struct OidcRpError(pub &'static str);

impl std::fmt::Display for OidcRpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Log an id_token refusal, then return it — the verification-half twin of
/// `warn_idp_failure`. Named for the side effect: every call emits a warning.
///
/// Everything refused after the token decodes — a bad header, a disallowed
/// alg, an unknown kid, a failed decode, an iss or nonce mismatch, an
/// oversized `sub`, a key the JWK cannot build — returned silently, and
/// under a substituted-IdP threat model those refusals are the highest-signal
/// events this console can observe.
///
/// It sits at each refusal rather than around `exchange_and_verify`'s call
/// site because that call opens with `discovery()`: a wrapper out there fires
/// on every IdP transport and discovery outage as well, and those arms
/// already warn — so each outage logs twice, the second line asserting an
/// id_token was rejected when none was ever received.
///
/// Safe by type: the payload is the `&'static str` the caller wrote, so no
/// IdP-supplied byte can reach the log through it. Use `ok_or_else`, never
/// `ok_or` — the eager form logs a refusal on the success path.
fn warn_rejected(op: &'static str) -> OidcRpError {
    tracing::warn!(op, "oidc: id_token rejected");
    OidcRpError(op)
}

/// The closed set of causes `warn_idp_failure` may print verbatim.
///
/// The cause reaches the pod log, and on these paths it is IdP-influenced,
/// so the set is enumerated rather than left open as `impl Display`. The
/// type this exists to exclude is `serde_json::Error`: it renders the
/// offending input into its own message, and for a token-endpoint body that
/// input can be a live id_token. It has no impl here, so routing a parse
/// failure to the wrong helper stops compiling. That catches the accident,
/// not the act: `String` has an impl — `format!("{which}={endpoint}")` needs
/// one — so `warn_idp_failure(op, e.to_string())` would still build. The
/// claim is that nobody reaches the leak by reflex, not that it is sealed.
trait SafeCause: std::fmt::Display {}
impl SafeCause for reqwest::Error {}
impl SafeCause for reqwest::StatusCode {}
impl SafeCause for String {}
impl SafeCause for &'_ String {}

#[derive(Deserialize, Clone)]
struct Discovery {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
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
struct TokenResponse {
    id_token: String,
}

/// Verified identity claims from the ID token.
#[derive(Deserialize)]
pub struct IdClaims {
    pub sub: String,
    pub iss: String,
    pub nonce: Option<String>,
    pub email: Option<String>,
    pub name: Option<String>,
    pub preferred_username: Option<String>,
}

fn normalize_issuer(s: &str) -> &str {
    s.strip_suffix('/').unwrap_or(s)
}

/// OIDC Core §2 caps the subject identifier: it "MUST NOT exceed 255 ASCII
/// characters in length". Counted in bytes, because the spec says ASCII and
/// bytes are what a log line and a session cookie actually cost.
///
/// Every claim here is IdP-controlled and was otherwise bounded only by the
/// 8 MiB body cap in `http.rs`. `sub` reaches a log line on the rejection
/// path of an unauthenticated route, so a substituted IdP could bill a
/// megabyte of pod log per refused attempt; `warn_idp_failure` capped
/// `cause` for that reason and these siblings were left behind.
///
/// The optional claims fail differently and worse. They carry no spec
/// ceiling, but they ride into the session cookie, and a cookie past the
/// 4 KiB every browser allows per RFC 6265 §6.1 is dropped silently — the
/// callback then lands with no session, redirects to a login the IdP
/// answers immediately, and loops until the browser gives up, with nothing
/// in the log. They borrow this ceiling for that reason, not the spec's.
///
/// What this does NOT bound is the transient allocation: `decode` builds the
/// oversized `String` before either guard runs, so the heap cost up to what
/// an 8 MiB token body decodes to is unchanged. `MAX_BODY_BYTES` is what
/// bounds that; this bounds what survives into a log line and a cookie.
const MAX_CLAIM_BYTES: usize = 255;

fn claim_within_bound(s: &str) -> bool {
    s.len() <= MAX_CLAIM_BYTES
}

/// (scheme, host, port) for RFC 6454 same-origin checks, default ports
/// normalized. Same semantics as alaya-server's `origin_of`.
fn origin_of(u: &str) -> Option<(String, String, u16)> {
    let parsed: url::Url = u.parse().ok()?;
    let scheme = parsed.scheme().to_string();
    let default_port: u16 = match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    Some((
        scheme,
        parsed.host_str()?.to_ascii_lowercase(),
        parsed.port().unwrap_or(default_port),
    ))
}

fn is_loopback_origin(origin: &(String, String, u16)) -> bool {
    origin.1 == "localhost"
        || origin
            .1
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// Endpoint must be same-origin with the issuer and https (http allowed only
/// for a loopback issuer, so local dev against a local IdP works).
fn same_origin_https(issuer: &str, endpoint: &str) -> Result<(), OidcRpError> {
    let issuer_origin = origin_of(issuer).ok_or(OidcRpError("issuer form"))?;
    let ep_origin = origin_of(endpoint).ok_or(OidcRpError("endpoint form"))?;
    let scheme_ok =
        ep_origin.0 == "https" || (is_loopback_origin(&issuer_origin) && ep_origin.0 == "http");
    if !scheme_ok || ep_origin != issuer_origin {
        return Err(OidcRpError("endpoint not same-origin with issuer"));
    }
    Ok(())
}

pub fn pkce_challenge_s256(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

pub struct OidcRp {
    issuer: String,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    http: reqwest::Client,
    discovery: RwLock<Option<Discovery>>,
    keys: RwLock<HashMap<String, Jwk>>,
    fetch_gate: Mutex<Instant>,
}

impl OidcRp {
    pub fn new(
        issuer: String,
        client_id: String,
        client_secret: String,
        redirect_uri: String,
    ) -> Self {
        // Same builder as every other upstream: a hardening knob added to
        // `http::client` must not miss the IdP, which is the one upstream an
        // unauthenticated caller can make the console reach.
        let http = crate::http::client(Duration::from_secs(10));
        let seeded = Instant::now()
            .checked_sub(JWKS_COOLDOWN * 2)
            .unwrap_or_else(Instant::now);
        OidcRp {
            issuer: normalize_issuer(&issuer).to_string(),
            client_id,
            client_secret,
            redirect_uri,
            http,
            discovery: RwLock::new(None),
            keys: RwLock::new(HashMap::new()),
            fetch_gate: Mutex::new(seeded),
        }
    }

    /// Log an IdP failure, then flatten it into the opaque `OidcRpError`
    /// that reaches the page. Named for the side effect: every call emits a
    /// warning. The page deliberately says only "discovery failed"; the pod
    /// log said nothing at all, so a login outage arrived with no issuer, no
    /// operation and no cause to triage from.
    ///
    /// Never the raw body — a token-endpoint body holds the id_token — and
    /// never a deserialize error, which cannot reach this method at all:
    /// `SafeCause` has no impl for `serde_json::Error`, because that error
    /// carries the input inside its own message. Parse failures go through
    /// `warn_idp_parse_failure` instead.
    ///
    /// Two callers do pass a PARSED field of the discovery document — the
    /// echoed issuer, and each endpoint that fails the origin check. Those
    /// are attacker-chosen text, which is what the recording below is about.
    /// `self.issuer` is config rather than IdP-supplied, but it is recorded
    /// the same way regardless: `validate_issuer` refuses userinfo and a
    /// non-https scheme, and says nothing about control characters, so the
    /// old "it is config, therefore safe" argument did not hold up.
    fn warn_idp_failure(&self, op: &'static str, cause: impl SafeCause) -> OidcRpError {
        // Recorded with `?`, not `%`, and that is not cosmetic. Two callers
        // pass a string lifted straight out of the discovery document. The
        // plain-text subscriber writes a field recorded as `Display`
        // verbatim, so one `\n` in it emits a second, wholly attacker-authored
        // line that reads like a real record; recorded as `Debug` the same
        // string is escaped and quoted onto one line. Capped because the only
        // other bound on it is the 8 MiB body cap, which is a log-flood lever
        // — by chars, since a byte split could land mid-codepoint and panic.
        let full = cause.to_string();
        let mut cause: String = full.chars().take(256).collect();
        // Compared by bytes, not chars: `cause` is a char-prefix of `full`, so
        // the two predicates are the same fact, and counting chars walks up to
        // 8 MiB twice on the exact log-flood path this truncation bounds.
        if cause.len() < full.len() {
            // Marked, because a cut URL renders as a complete-looking wrong one.
            cause.push('…');
        }
        tracing::warn!(
            op,
            issuer = ?self.issuer,
            cause = ?cause,
            "oidc: identity provider request failed"
        );
        OidcRpError(op)
    }

    /// The parse-failure twin of `warn_idp_failure`: logs the SHAPE of the
    /// error — category and position — and never its `Display`.
    ///
    /// `serde_json` renders the unexpected value into its message, and the
    /// bodies parsed here are the token endpoint's and the JWKS. A body that
    /// is a bare JSON string — a broken or hostile IdP answering
    /// `"<id_token>"` instead of `{"id_token": "…"}` — makes the WHOLE
    /// string the unexpected value, so `Display` would put a live token in
    /// the pod log (`parse_error_display_carries_the_input` pins this).
    /// Typing every IdP-sourced field as `String` does not save it: that
    /// argument covers the fields, not the top-level value.
    ///
    /// `classify` / `line` / `column` keep syntax vs data vs early EOF, and
    /// where. None of them can carry input: `Category` is a fieldless enum
    /// and the other two are `usize`, so the types are the proof, not a test.
    ///
    /// It is not free. The commonest real failure — the token endpoint
    /// answering `{"error":"invalid_grant"}` — used to log ``missing field
    /// `id_token` ``, which names the problem and is built from the derive's
    /// `&'static str`, so it was never unsafe. `serde_json` offers no way to
    /// tell that arm from the input-bearing ones, so the safe arms lose too.
    fn warn_idp_parse_failure(&self, op: &'static str, e: &serde_json::Error) -> OidcRpError {
        tracing::warn!(
            op,
            issuer = ?self.issuer,
            category = ?e.classify(),
            line = e.line(),
            column = e.column(),
            "oidc: identity provider response did not parse"
        );
        OidcRpError(op)
    }

    async fn discovery(&self) -> Result<Discovery, OidcRpError> {
        if let Some(d) = self.discovery.read().await.clone() {
            return Ok(d);
        }
        let url = format!("{}/.well-known/openid-configuration", self.issuer);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| self.warn_idp_failure("discovery failed", e))?;
        if !resp.status().is_success() {
            return Err(self.warn_idp_failure("discovery status", resp.status()));
        }
        let body = crate::http::body_text("oidc discovery", resp)
            .await
            .map_err(|_| OidcRpError("discovery read"))?;
        let disc: Discovery = serde_json::from_str(&body)
            .map_err(|e| self.warn_idp_parse_failure("discovery parse", &e))?;
        if normalize_issuer(&disc.issuer) != self.issuer {
            return Err(self.warn_idp_failure("discovery issuer mismatch", &disc.issuer));
        }
        // A discovery document pointing an endpoint off the issuer's origin
        // is the substituted-IdP signal (OIDC Core §4.3). Name which one:
        // the bare error cannot say, and this is the rejection an operator
        // most needs a record of.
        for (which, endpoint) in [
            ("jwks_uri", &disc.jwks_uri),
            ("token_endpoint", &disc.token_endpoint),
            ("authorization_endpoint", &disc.authorization_endpoint),
        ] {
            same_origin_https(&self.issuer, endpoint).map_err(|_| {
                self.warn_idp_failure(
                    "endpoint not same-origin with issuer",
                    format!("{which}={endpoint}"),
                )
            })?;
        }
        *self.discovery.write().await = Some(disc.clone());
        Ok(disc)
    }

    /// Build the authorization redirect for a fresh login flow.
    pub async fn authorize_url(
        &self,
        state: &str,
        nonce: &str,
        pkce_verifier: &str,
    ) -> Result<String, OidcRpError> {
        let disc = self.discovery().await?;
        let mut u: url::Url = disc
            .authorization_endpoint
            .parse()
            .map_err(|_| OidcRpError("authorization_endpoint form"))?;
        u.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", &self.redirect_uri)
            .append_pair("scope", "openid profile email")
            .append_pair("state", state)
            .append_pair("nonce", nonce)
            .append_pair("code_challenge", &pkce_challenge_s256(pkce_verifier))
            .append_pair("code_challenge_method", "S256");
        Ok(u.to_string())
    }

    /// Exchange the authorization code, verify the ID token (signature, iss,
    /// aud, exp, nonce) and return the identity claims.
    pub async fn exchange_and_verify(
        &self,
        code: &str,
        pkce_verifier: &str,
        expected_nonce: &str,
    ) -> Result<IdClaims, OidcRpError> {
        let disc = self.discovery().await?;
        let resp = self
            .http
            .post(&disc.token_endpoint)
            .basic_auth(&self.client_id, Some(&self.client_secret))
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", self.redirect_uri.as_str()),
                ("code_verifier", pkce_verifier),
            ])
            .send()
            .await
            .map_err(|e| self.warn_idp_failure("token exchange failed", e))?;
        if !resp.status().is_success() {
            return Err(self.warn_idp_failure("token exchange rejected", resp.status()));
        }
        let body = crate::http::body_text("oidc token response", resp)
            .await
            .map_err(|_| OidcRpError("token response read"))?;
        let tokens: TokenResponse = serde_json::from_str(&body)
            .map_err(|e| self.warn_idp_parse_failure("token response parse", &e))?;

        let claims = self.verify_id_token(&tokens.id_token).await?;
        // Nonce binds the ID token to this login flow (replay defense).
        if claims.nonce.as_deref() != Some(expected_nonce) {
            return Err(warn_rejected("nonce mismatch"));
        }
        Ok(claims)
    }

    async fn verify_id_token(&self, token: &str) -> Result<IdClaims, OidcRpError> {
        let header = decode_header(token).map_err(|_| warn_rejected("bad id_token header"))?;
        if !matches!(header.alg, Algorithm::RS256 | Algorithm::ES256) {
            return Err(warn_rejected("alg not allowed"));
        }
        let kid = header.kid.ok_or_else(|| warn_rejected("missing kid"))?;
        let jwk = self.key_for_kid(&kid).await?;
        let decoding_key = build_decoding_key(&jwk, header.alg)?;

        let mut validation = Validation::new(header.alg);
        // ID token audience is the RP's client_id (OIDC Core §3.1.3.7 #3).
        validation.set_audience(&[&self.client_id]);
        validation.validate_aud = true;
        validation.validate_exp = true;
        validation.leeway = CLOCK_SKEW_LEEWAY_SECS;
        validation.set_required_spec_claims(&["exp", "aud"]);

        let data = decode::<IdClaims>(token, &decoding_key, &validation)
            .map_err(|_| warn_rejected("id_token invalid"))?;
        if normalize_issuer(&data.claims.iss) != self.issuer {
            return Err(warn_rejected("iss mismatch"));
        }
        // The one place every consumer of these claims routes through, so
        // the bound holds for the log sites, the session and the allowlist
        // at once. The error carries the `&'static str` only — refusing an
        // oversized `sub` must not itself log it.
        let mut claims = data.claims;
        if !claim_within_bound(&claims.sub) {
            return Err(warn_rejected("sub exceeds the OIDC Core §2 ceiling"));
        }
        // Dropped rather than refused: these three are display-only and
        // `Session::display_name` already falls back through email to `sub`,
        // so losing an absurd one costs a nicety. Refusing the login instead
        // would hand any IdP with a verbose `name` claim an outage.
        claims.email = claims.email.filter(|s| claim_within_bound(s));
        claims.name = claims.name.filter(|s| claim_within_bound(s));
        claims.preferred_username = claims.preferred_username.filter(|s| claim_within_bound(s));
        Ok(claims)
    }

    async fn key_for_kid(&self, kid: &str) -> Result<Jwk, OidcRpError> {
        if let Some(jwk) = self.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }
        let mut last_fetch = self.fetch_gate.lock().await;
        if let Some(jwk) = self.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }
        if last_fetch.elapsed() < JWKS_COOLDOWN {
            return Err(warn_rejected("unknown kid (cooldown)"));
        }
        // Extend the cooldown before fetching: a down IdP must not turn the
        // console into an outbound-fetch amplifier.
        *last_fetch = Instant::now();

        let disc = self.discovery().await?;
        let resp = self
            .http
            .get(&disc.jwks_uri)
            .send()
            .await
            .map_err(|e| self.warn_idp_failure("jwks fetch failed", e))?;
        if !resp.status().is_success() {
            return Err(self.warn_idp_failure("jwks status", resp.status()));
        }
        let body = crate::http::body_text("oidc jwks", resp)
            .await
            .map_err(|_| OidcRpError("jwks read"))?;
        let jwks: Jwks = serde_json::from_str(&body)
            .map_err(|e| self.warn_idp_parse_failure("jwks parse", &e))?;
        let mut map = HashMap::new();
        for jwk in jwks.keys {
            if let Some(k) = jwk.kid.clone() {
                map.insert(k, jwk);
            }
        }
        *self.keys.write().await = map;

        self.keys
            .read()
            .await
            .get(kid)
            .cloned()
            .ok_or_else(|| warn_rejected("unknown kid"))
    }
}

fn build_decoding_key(jwk: &Jwk, alg: Algorithm) -> Result<DecodingKey, OidcRpError> {
    match (jwk.kty.as_str(), alg) {
        ("RSA", Algorithm::RS256) => {
            let n = jwk.n.as_deref().ok_or_else(|| warn_rejected("rsa n"))?;
            let e = jwk.e.as_deref().ok_or_else(|| warn_rejected("rsa e"))?;
            DecodingKey::from_rsa_components(n, e).map_err(|_| warn_rejected("rsa key"))
        }
        ("EC", Algorithm::ES256) => {
            let x = jwk.x.as_deref().ok_or_else(|| warn_rejected("ec x"))?;
            let y = jwk.y.as_deref().ok_or_else(|| warn_rejected("ec y"))?;
            DecodingKey::from_ec_components(x, y).map_err(|_| warn_rejected("ec key"))
        }
        _ => Err(warn_rejected("alg/key mismatch")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testlog::{LogBuf, separators_in};

    #[test]
    fn pkce_challenge_matches_rfc7636_appendix_b() {
        // RFC 7636 Appendix B test vector.
        assert_eq!(
            pkce_challenge_s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    /// The upstream fact `warn_idp_parse_failure` is built on: `serde_json`
    /// renders the offending input into the error's `Display`, whole and
    /// untruncated. A token endpoint answering with a bare JSON string
    /// (`"<id_token>"` instead of `{"id_token": "…"}`) therefore makes a live
    /// token the unexpected value. Pinned here because it is someone else's
    /// behaviour: if a `serde_json` bump ever stopped echoing the input, this
    /// goes red and the helper is paying for a hazard that no longer exists.
    ///
    /// The converse needs no test — see `warn_idp_parse_failure`, where the
    /// logged fields are a fieldless enum and two `usize`.
    #[test]
    fn parse_error_display_carries_the_input() {
        let token = "eyJhbGciOiJSUzI1NiJ9.SECRET-TOKEN-PAYLOAD.signature";
        // `.err()`, not `unwrap_err()`: `TokenResponse` deliberately has no
        // `Debug` impl — it holds the id_token.
        let e = serde_json::from_str::<TokenResponse>(&format!("\"{token}\""))
            .err()
            .expect("a bare JSON string must not deserialize into the struct");
        assert!(
            e.to_string().contains(token),
            "the hazard this method exists for: {e}"
        );
    }

    /// A substituted IdP must not be able to write its own log records.
    /// `cause` is the one field on this path that carries IdP text — the
    /// discovery `issuer` at the mismatch branch, and the three endpoints
    /// below it — and the plain-text subscriber this binary installs
    /// neutralises nothing in a field recorded as `Display`. The `?` in
    /// `warn_idp_failure` is the whole guard; this is what holds it there.
    #[test]
    fn no_separator_in_an_idp_cause_can_forge_a_log_line() {
        let rp = OidcRp::new(
            "https://id.test".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        // What a hostile discovery document puts in `issuer`: a plausible
        // value, then every separator a log reader might break on, then a
        // complete forged record. `\n` splits a line-oriented ingester,
        // U+2028/U+2029/NEL split a Unicode-aware one, ESC drives a terminal.
        // The oracle below only catches what the input actually carries.
        let hostile =
            "https://id.test\n\r\u{2028}\u{2029}\u{85}\u{1b}[31m  WARN ops_console: all clear"
                .to_string();
        let buf = LogBuf::default();
        {
            let _capture = buf.capture();
            rp.warn_idp_failure("discovery issuer mismatch", &hostile);
        }

        let record = buf.record("identity provider request failed");
        assert!(
            record.contains("\\n") && record.contains("all clear"),
            "the whole hostile cause must be escaped onto this one record: {record}"
        );
        let leaked = separators_in(&record);
        assert!(
            leaked.is_empty(),
            "no separator or control char may reach the log raw: {leaked:?} in {record}"
        );
    }

    /// The warning that says an id_token was rejected must fire when, and
    /// only when, one was.
    ///
    /// It used to sit around `exchange_and_verify`'s call site in `auth.rs`,
    /// which covered the refusals but also every transport and discovery
    /// failure underneath them — those arms warn for themselves, so each IdP
    /// outage logged twice, the second line naming a rejection that never
    /// happened. Under the substituted-IdP threat model the line was added
    /// for, that is the signal it exists to sharpen, degraded.
    #[tokio::test]
    async fn an_id_token_rejection_is_logged_at_the_refusal_and_nowhere_else() {
        let rp = OidcRp::new(
            // Closed port: discovery fails at the transport, so no token is
            // ever received, let alone refused.
            "http://127.0.0.1:1".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );

        let buf = LogBuf::default();
        {
            let _capture = buf.capture();
            assert!(
                rp.exchange_and_verify("code", "verifier", "nonce")
                    .await
                    .is_err()
            );
        }
        let logged = buf.text();
        assert!(
            logged.contains("identity provider request failed"),
            "the transport failure must still be logged: {logged}"
        );
        assert!(
            !logged.contains("id_token rejected"),
            "no id_token was received, so none can have been rejected: {logged}"
        );

        let buf = LogBuf::default();
        {
            let _capture = buf.capture();
            assert!(rp.verify_id_token("not-a-jwt").await.is_err());
        }
        let logged = buf.text();
        assert!(
            logged.contains("id_token rejected") && logged.contains("bad id_token header"),
            "a real refusal must name itself in the log: {logged}"
        );
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

    /// Pins the comparison, not the placement: reaching `verify_id_token`
    /// needs a signed token and a JWKS fixture, which would cost more than
    /// it pins. Placement is safe by construction instead — `decode` into
    /// `IdClaims` appears once, in this private method, whose only caller is
    /// `exchange_and_verify`.
    #[test]
    fn claim_bound_is_the_spec_ceiling() {
        assert!(claim_within_bound(&"a".repeat(255)));
        assert!(!claim_within_bound(&"a".repeat(256)));
        // Bytes, not chars — the ceiling exists to bound what a log line and
        // a cookie cost, so a "fix" to `chars().count()` must fail here.
        assert!(!claim_within_bound(&"é".repeat(128)));
    }

    #[tokio::test]
    async fn authorize_url_pins_redirect_and_carries_pkce() {
        // Discovery is pre-seeded so no network is touched.
        let rp = OidcRp::new(
            "https://id.test".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        *rp.discovery.write().await = Some(Discovery {
            issuer: "https://id.test".into(),
            authorization_endpoint: "https://id.test/authorize".into(),
            token_endpoint: "https://id.test/token".into(),
            jwks_uri: "https://id.test/jwks".into(),
        });
        let u = rp
            .authorize_url("STATE", "NONCE", "VERIFIER")
            .await
            .unwrap();
        assert!(u.contains("redirect_uri=https%3A%2F%2Fconsole.test%2Fauth%2Fcallback"));
        assert!(u.contains("state=STATE"));
        assert!(u.contains("nonce=NONCE"));
        assert!(u.contains("code_challenge_method=S256"));
        assert!(
            !u.contains("secret"),
            "client secret must never be in the authorize URL"
        );
    }
}
