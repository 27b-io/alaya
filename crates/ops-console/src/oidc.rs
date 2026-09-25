//! OIDC Relying Party — authorization-code flow with PKCE (S256) against
//! id.27b.io.
//!
//! Discovery, same-origin/HTTPS enforcement, the JWKS cache + cooldown, the
//! capped discovery/JWKS body reads and the whole ID-token verify pipeline
//! live in `alaya-oidc` (shared with alaya-server's resource-side verifier).
//! This module keeps what only a relying party decides:
//! - PKCE S256 challenge; `state` and `nonce` are supplied by the login flow
//! - `redirect_uri` is pinned from config; never derived from request headers
//! - `authorization_endpoint` and `token_endpoint` must be present and pass
//!   the same same-origin-https rule as `jwks_uri` (the shared layer checks
//!   only `jwks_uri`, because a resource server never calls the other two)
//! - token exchange with `client_secret_basic` (RFC 6749 §2.3.1 — the scheme
//!   servers MUST support), over the console's one upstream client, its body
//!   read through `http::body_text`
//! - ID-token `aud` is this `client_id` (OIDC Core §3.1.3.7 #3); `nonce` must
//!   match the flow (replay defence); claims bounded to the OIDC Core §2 cap
//! - every IdP failure and every id_token refusal logged once, where it
//!   happened — the shared layer returns the cause, this module records it

use std::time::Duration;

use alaya_oidc::{
    Cause, Discovery, Error as OidcError, IssuedClaims, ParseFailure, Provider, same_origin_https,
};
use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};

/// The page-safe reason a login step failed. By the time one exists the
/// failure has been logged, once, by the helper that built it — so, unlike
/// `alaya_oidc::Error`, it carries no cause and no refusal-vs-outage class.
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
/// Every id_token refusal — a bad header, a disallowed alg, an unknown kid, a
/// failed decode, an iss or nonce mismatch, an oversized `sub`, a key the JWK
/// cannot build — comes through here, and under a substituted-IdP threat
/// model those refusals are the highest-signal events this console can
/// observe. The shared verifier's refusals arrive via `logged`; the nonce and
/// `sub` checks call it directly.
///
/// It is never a wrapper around `exchange_and_verify`'s call site, because
/// that call opens with discovery: a wrapper out there fires on every IdP
/// transport and discovery outage as well, and those arms already warn — so
/// each outage logs twice, the second line asserting an id_token was rejected
/// when none was ever received. `logged` keeps the two apart by the class the
/// shared layer returns: `Invalid` is a refusal, `Provider` an outage.
///
/// Safe by type: the payload is a `&'static str` literal, so no IdP-supplied
/// byte can reach the log through it. Use `ok_or_else`, never `ok_or` — the
/// eager form logs a refusal on the success path.
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
/// not the act: `String` has an impl — `Cause::Document` carries one — so
/// `warn_idp_failure(op, e.to_string())` would still build. The
/// claim is that nobody reaches the leak by reflex, not that it is sealed.
trait SafeCause: std::fmt::Display {}
impl SafeCause for reqwest::Error {}
impl SafeCause for reqwest::StatusCode {}
impl SafeCause for String {}
impl SafeCause for &'_ String {}
impl SafeCause for &'static str {}

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

impl IssuedClaims for IdClaims {
    fn iss(&self) -> &str {
        &self.iss
    }
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
/// It is a first cut at that hazard, not the guarantee. The count is RAW
/// bytes, and JSON escaping turns one control byte into six on the way into
/// the cookie: an ordinary subject with `email` and `name` both at this cap
/// issued a ~4.6 KiB `Set-Cookie`. What bounds the encoded artifact is
/// `MAX_SESSION_PLAINTEXT_BYTES` in `session.rs`; what this bounds is the
/// claim — the log line, and how much of it the cookie is asked to carry in
/// the first place.
///
/// What this does NOT bound is the transient allocation: `decode` builds the
/// oversized `String` before either guard runs, so the heap cost up to what
/// an 8 MiB token body decodes to is unchanged. `MAX_BODY_BYTES` is what
/// bounds that; this bounds what survives into a log line and a cookie.
const MAX_CLAIM_BYTES: usize = 255;

fn claim_within_bound(s: &str) -> bool {
    s.len() <= MAX_CLAIM_BYTES
}

pub fn pkce_challenge_s256(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

pub struct OidcRp {
    provider: Provider,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
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
        OidcRp {
            provider: Provider::with_client(&issuer, http),
            client_id,
            client_secret,
            redirect_uri,
        }
    }

    /// Log an IdP failure, then flatten it into the opaque reason that
    /// reaches the page. Named for the side effect: every call emits a
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
    /// The configured issuer is config rather than IdP-supplied, but it is
    /// recorded the same way regardless: `validate_issuer` refuses userinfo
    /// and a non-https scheme, and says nothing about control characters, so
    /// the old "it is config, therefore safe" argument did not hold up.
    fn warn_idp_failure(&self, op: &'static str, cause: impl SafeCause) -> OidcRpError {
        // Recorded with `?`, not `%`, and that is not cosmetic. Two callers
        // pass a string lifted straight out of the discovery document. A
        // plain-text subscriber writes a field recorded as `Display`
        // verbatim, so one `\n` in it emits a second, wholly attacker-authored
        // line that reads like a real record; recorded as `Debug` the same
        // string is escaped and quoted onto one line. The binary's JSON
        // encoder escapes it too (`main.rs` pins that); `?` keeps this call
        // site safe under any encoder. Capped because the only
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
            issuer = ?self.provider.issuer(),
            cause = ?cause,
            "oidc: identity provider request failed"
        );
        OidcRpError(op)
    }

    /// The parse-failure twin of `warn_idp_failure`: logs the SHAPE of the
    /// error — category and position — and never its `Display`.
    ///
    /// `serde_json` renders the unexpected value into its message. The token
    /// response is parsed here; discovery and the JWKS are parsed in
    /// `alaya-oidc` and arrive through `logged` as the same shape. A body that
    /// is a bare JSON string — a broken or hostile IdP answering
    /// `"<id_token>"` instead of `{"id_token": "…"}` — makes the WHOLE
    /// string the unexpected value, so `Display` would put a live token in
    /// the pod log (`parse_error_display_carries_the_input` pins this).
    /// Typing every IdP-sourced field as `String` does not save it: that
    /// argument covers the fields, not the top-level value.
    ///
    /// `ParseFailure` keeps syntax vs data vs early EOF, and where — and, by
    /// type, nothing else (see its doc).
    ///
    /// It is not free. The commonest real failure — the token endpoint
    /// answering `{"error":"invalid_grant"}` — used to log ``missing field
    /// `id_token` ``, which names the problem and is built from the derive's
    /// `&'static str`, so it was never unsafe. `serde_json` offers no way to
    /// tell that arm from the input-bearing ones, so the safe arms lose too.
    fn warn_idp_parse_failure(&self, op: &'static str, e: ParseFailure) -> OidcRpError {
        tracing::warn!(
            op,
            issuer = ?self.provider.issuer(),
            category = ?e.category,
            line = e.line,
            column = e.column,
            "oidc: identity provider response did not parse"
        );
        OidcRpError(op)
    }

    /// Record a shared-layer failure once, by class, at the call that
    /// surfaced it: a refused token through `warn_rejected`, a failed
    /// provider through the IdP helpers with the cause the layer carried
    /// back. The match is exhaustive on purpose — a new `Cause` must pick
    /// its log line before this compiles.
    fn logged(&self, e: OidcError) -> OidcRpError {
        match e {
            OidcError::Invalid(op) => warn_rejected(op),
            OidcError::Provider { op, cause } => match cause {
                Cause::Transport(e) => self.warn_idp_failure(op, e),
                Cause::Status(status) => self.warn_idp_failure(op, status),
                Cause::TooLarge => self.warn_idp_failure(op, "response body exceeded the cap"),
                Cause::Document(value) => self.warn_idp_failure(op, value),
                Cause::Parse(shape) => self.warn_idp_parse_failure(op, shape),
            },
        }
    }

    /// `(authorization_endpoint, token_endpoint)` from discovery, both required
    /// and both same-origin-https with the issuer. Checked together on every
    /// use so a bad `token_endpoint` is refused when the login starts, before
    /// the user is ever redirected.
    async fn endpoints(&self) -> Result<(String, String), OidcRpError> {
        let disc = self
            .provider
            .discovery()
            .await
            .map_err(|e| self.logged(e))?;
        let checked = self.rp_endpoints(disc);
        if checked.is_err() {
            // The shared layer cached this document on its own rules, which
            // do not cover these two endpoints. Forget it, so the next login
            // re-fetches and an IdP that corrects its document recovers
            // without a restart.
            self.provider.forget_discovery().await;
        }
        checked
    }

    fn rp_endpoints(&self, disc: Discovery) -> Result<(String, String), OidcRpError> {
        let authorization = disc.authorization_endpoint.ok_or_else(|| {
            self.warn_idp_failure("discovery missing authorization_endpoint", "absent")
        })?;
        let token = disc
            .token_endpoint
            .ok_or_else(|| self.warn_idp_failure("discovery missing token_endpoint", "absent"))?;
        // A discovery document pointing an endpoint off the issuer's origin
        // is the substituted-IdP signal (OIDC Core §4.3). The cause records
        // the value, which the page reason cannot: this is the rejection an
        // operator most needs a record of.
        for (op, endpoint) in [
            ("authorization_endpoint not same-origin", &authorization),
            ("token_endpoint not same-origin", &token),
        ] {
            same_origin_https(self.provider.issuer(), endpoint)
                .map_err(|_| self.warn_idp_failure(op, endpoint))?;
        }
        Ok((authorization, token))
    }

    /// Build the authorization redirect for a fresh login flow.
    pub async fn authorize_url(
        &self,
        state: &str,
        nonce: &str,
        pkce_verifier: &str,
    ) -> Result<String, OidcRpError> {
        let (authorization_endpoint, _) = self.endpoints().await?;
        let mut u: url::Url = authorization_endpoint
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
        let (_, token_endpoint) = self.endpoints().await?;
        let resp = self
            .provider
            .http()
            .post(&token_endpoint)
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
        // `body_text` logs its own failures, so the refusal is not logged twice.
        let body = crate::http::body_text("oidc token response", resp)
            .await
            .map_err(|_| OidcRpError("token response read"))?;
        let tokens: TokenResponse = serde_json::from_str(&body)
            .map_err(|e| self.warn_idp_parse_failure("token response parse", (&e).into()))?;

        let claims = self.verify_id_token(&tokens.id_token).await?;
        // Nonce binds the ID token to this login flow (replay defense).
        if claims.nonce.as_deref() != Some(expected_nonce) {
            return Err(warn_rejected("nonce mismatch"));
        }
        Ok(claims)
    }

    async fn verify_id_token(&self, token: &str) -> Result<IdClaims, OidcRpError> {
        // ID token audience is the RP's client_id (OIDC Core §3.1.3.7 #3).
        let mut claims: IdClaims = self
            .provider
            .verify(token, &self.client_id)
            .await
            .map_err(|e| self.logged(e))?;
        // The one place every consumer of these claims routes through, so
        // the bound holds for the log sites, the session and the allowlist
        // at once. The error carries the `&'static str` only — refusing an
        // oversized `sub` must not itself log it.
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testlog::{LogBuf, separators_in};
    use alaya_oidc::Jwk;

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
    /// echoed discovery `issuer` and any off-origin endpoint, arriving through
    /// `logged`'s `Cause::Document` arm and `rp_endpoints` — and a plain-text
    /// subscriber neutralises nothing in a field recorded as `Display`. The
    /// `?` in `warn_idp_failure` is the call site's own guard, whatever the
    /// encoder; this is what holds it there.
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

    /// Each refusal must name itself in the log, where it surfaces rather
    /// than around `exchange_and_verify`'s call site in `auth.rs` — see
    /// `warn_rejected`. The converse, that an IdP outage does NOT claim a
    /// rejection, is pinned at route level in `main.rs`.
    #[tokio::test]
    async fn an_id_token_refusal_names_itself_in_the_log() {
        let rp = OidcRp::new(
            "https://id.test".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        let buf = LogBuf::default();
        {
            let _capture = buf.capture();
            // Refused by `decode_header`, before any network call.
            assert!(rp.verify_id_token("not-a-jwt").await.is_err());
        }
        let logged = buf.text();
        assert!(
            logged.contains("id_token rejected") && logged.contains("bad header"),
            "a real refusal must name itself in the log: {logged}"
        );
    }

    /// Pins the comparison itself; the signed-token tests at the bottom of
    /// this module pin what `verify_id_token` does with it. Kept separate
    /// because that harness cannot fail on the bytes-vs-chars distinction,
    /// and this is the assertion that holds it.
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
        rp.provider.seed_discovery(Discovery {
            issuer: "https://id.test".into(),
            authorization_endpoint: Some("https://id.test/authorize".into()),
            token_endpoint: Some("https://id.test/token".into()),
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

    #[tokio::test]
    async fn cross_origin_rp_endpoint_is_refused_before_redirect() {
        // The shared layer checks only jwks_uri; the RP must apply the same
        // rule to its own endpoints before the user is sent anywhere.
        let rp = OidcRp::new(
            "https://id.test".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        rp.provider.seed_discovery(Discovery {
            issuer: "https://id.test".into(),
            authorization_endpoint: Some("https://id.test/authorize".into()),
            token_endpoint: Some("https://evil.test/token".into()),
            jwks_uri: "https://id.test/jwks".into(),
        });
        let err = rp
            .authorize_url("STATE", "NONCE", "VERIFIER")
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "token_endpoint not same-origin");
    }

    /// The shared layer caches discovery on its own rules, which do not cover
    /// the RP endpoints. A document refused on them must not stay cached, or
    /// one bad answer locks every login out until the pod restarts.
    #[tokio::test]
    async fn a_refused_rp_endpoint_is_refetched_not_served_from_cache() {
        // Closed loopback port: the re-fetch fails fast as a transport error,
        // which is the proof it happened.
        let issuer = "http://127.0.0.1:1";
        let rp = OidcRp::new(
            issuer.into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        rp.provider.seed_discovery(Discovery {
            issuer: issuer.into(),
            authorization_endpoint: Some(format!("{issuer}/authorize")),
            token_endpoint: Some("https://evil.test/token".into()),
            jwks_uri: format!("{issuer}/jwks"),
        });
        let first = rp.authorize_url("S", "N", "V").await.unwrap_err();
        assert_eq!(first.to_string(), "token_endpoint not same-origin");
        let second = rp.authorize_url("S", "N", "V").await.unwrap_err();
        assert_eq!(
            second.to_string(),
            "discovery failed",
            "the refused document must be re-fetched, not refused again from cache"
        );
    }

    // --- signed token -> session cookie, measured end to end -------------
    //
    // Why real tokens rather than a hand-built `Session`: the defect these
    // cover lived in the COMPOSITION of `claim_within_bound` and the cookie
    // budget, not in either one, so a test starting after verification would
    // assert the fix on the wrong side of the seam that broke. The expansion
    // chain is documented at `MAX_SESSION_PLAINTEXT_BYTES` in `session.rs`.

    /// RFC 6265 6.1: a user agent should support at least 4096 bytes per
    /// cookie, "as measured by the sum of the length of the cookie's name,
    /// value, and attributes" — which is exactly one `Set-Cookie` header.
    const BROWSER_COOKIE_LIMIT: usize = 4096;

    /// Mint an ES256 id_token and run it through the production verifier.
    /// Discovery is unused and the JWKS is pre-seeded, so nothing touches
    /// the network.
    async fn verify_minted(
        sub: &str,
        email: Option<&str>,
        name: Option<&str>,
    ) -> Result<IdClaims, OidcRpError> {
        use crate::testkit;

        let rp = OidcRp::new(
            testkit::ISSUER.into(),
            testkit::CLIENT_ID.into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        rp.provider.seed_keys([Jwk {
            kty: "EC".into(),
            kid: Some(testkit::KID.into()),
            n: None,
            e: None,
            x: Some(testkit::KEY.x.clone()),
            y: Some(testkit::KEY.y.clone()),
        }]);
        rp.verify_id_token(&testkit::mint_id_token(sub, email, name))
            .await
    }

    /// Issue the session the callback would issue for these claims and hand
    /// back the real `Set-Cookie` header plus the session a browser would
    /// send back on the next request — so "it fits" is measured against a
    /// cookie that is still usable, not merely against a smaller number.
    fn issue_and_replay(claims: IdClaims) -> (String, Option<crate::session::Session>) {
        use axum::http::header::{COOKIE, SET_COOKIE};
        use axum::response::IntoResponse;
        use axum_extra::extract::cookie::{Key, PrivateCookieJar};

        let key = Key::from(&[7u8; 64][..]);
        let sess = crate::session::session_for(claims);
        // `secure = true` is the larger header (`; Secure`), i.e. the shape
        // the deployed console actually emits.
        let jar = crate::session::session_cookie(PrivateCookieJar::new(key.clone()), &sess, true);
        let header = (jar, axum::http::StatusCode::OK)
            .into_response()
            .headers()
            .get(SET_COOKIE)
            .expect("a session cookie is always issued")
            .to_str()
            .expect("the encoded cookie is ascii")
            .to_string();

        let mut replay = axum::http::HeaderMap::new();
        let pair = header.split(';').next().expect("name=value").to_string();
        replay.insert(COOKIE, pair.parse().expect("cookie header"));
        let back = PrivateCookieJar::from_headers(&replay, key);
        (header, crate::session::read_session(&back))
    }

    #[tokio::test]
    async fn control_heavy_claims_at_the_cap_still_fit_the_cookie_limit() {
        let nul = "\u{0}".repeat(MAX_CLAIM_BYTES);
        // The second case is the worst the claim bounds permit: an
        // allowlisted subject could only look like this if an operator
        // pasted it in, but it is what makes the fallback provably
        // sufficient — dropping the two display claims has to be enough
        // whatever `sub` costs.
        for (label, sub) in [
            ("ordinary subject", "ops-operator".to_string()),
            ("subject at the cap too", nul.clone()),
        ] {
            let claims = verify_minted(&sub, Some(&nul), Some(&nul))
                .await
                .expect("claims at the cap verify");
            let (header, session) = issue_and_replay(claims);
            assert!(
                header.len() <= BROWSER_COOKIE_LIMIT,
                "{label}: Set-Cookie is {} bytes, the browser drops it silently",
                header.len()
            );
            assert_eq!(
                session
                    .expect("the issued session must survive a round trip")
                    .sub,
                sub,
                "{label}: identity must not be altered to make the cookie fit"
            );
        }
    }

    #[tokio::test]
    async fn a_full_length_ordinary_profile_keeps_its_display_name() {
        let name = "n".repeat(MAX_CLAIM_BYTES);
        let email = format!("{}@example.test", "e".repeat(MAX_CLAIM_BYTES - 13));
        let claims = verify_minted("ops-operator", Some(&email), Some(&name))
            .await
            .expect("boundary-length text verifies");
        let (header, session) = issue_and_replay(claims);
        assert!(
            header.len() <= BROWSER_COOKIE_LIMIT,
            "Set-Cookie is {} bytes",
            header.len()
        );
        assert_eq!(
            session.expect("usable session").display_name(),
            name,
            "255 bytes of ordinary text is an ordinary profile — dropping it \
             would be a regression dressed as a fix"
        );
    }

    #[tokio::test]
    async fn oversized_claims_are_refused_for_sub_and_dropped_for_the_rest() {
        let over = "a".repeat(MAX_CLAIM_BYTES + 1);
        assert!(
            verify_minted(&over, None, None).await.is_err(),
            "an oversized sub must refuse the login — truncating it would \
             alias two subjects onto one allowlist entry"
        );
        let claims = verify_minted("ops-operator", Some(&over), Some(&over))
            .await
            .expect("oversized display claims degrade, they do not refuse");
        assert!(claims.email.is_none() && claims.name.is_none());
        let (header, session) = issue_and_replay(claims);
        assert!(header.len() <= BROWSER_COOKIE_LIMIT);
        assert_eq!(
            session.expect("usable session").display_name(),
            "ops-operator"
        );
    }
}
