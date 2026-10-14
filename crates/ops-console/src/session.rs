//! Cookie-backed sessions over axum-extra's private (AES-GCM encrypted) jar.
//!
//! Three cookies, all `HttpOnly` + `SameSite=Lax` + `Secure` (when the public
//! URL is https), all encrypted:
//! - `console_session` — authenticated identity + per-session CSRF token.
//! - `console_login`   — transient OIDC flow state (state/nonce/PKCE), 10 min.
//! - `console_flash`   — one-shot result banner after a POST-redirect-GET.
//!
//! A fourth, `console_theme`, carries the operator's colour theme with the
//! same attributes; it outlives the session on purpose (`theme.rs`).
//!
//! Session fixation: the callback always mints a brand-new session cookie
//! (fresh CSRF included) and deletes the flow cookie — nothing minted before
//! authentication survives it.

use axum::extract::FromRequestParts;
use axum::http::request::Parts;
use axum_extra::extract::cookie::{Cookie, PrivateCookieJar, SameSite};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::AppError;
use crate::state::AppState;
use crate::theme::{self, Theme};

pub const SESSION_COOKIE: &str = "console_session";
pub const LOGIN_COOKIE: &str = "console_login";
pub const FLASH_COOKIE: &str = "console_flash";

/// Absolute session lifetime (team session rule: absolute ≤ 12 h). No
/// absolute refresh — an operator re-logs in through the IdP, which is cheap
/// on the tailnet.
const SESSION_TTL_SECS: i64 = 12 * 3600;
/// Idle timeout (team session rule: idle ≤ 15 min). `last_seen` slides on
/// every authenticated request via the `session_refresh` middleware.
pub const IDLE_TTL_SECS: i64 = 15 * 60;
const LOGIN_TTL_SECS: i64 = 600;

/// Plaintext (JSON) ceiling for a cookie value, derived from the encoded
/// artifact rather than from the input.
///
/// RFC 6265 6.1 asks user agents to support at least 4096 bytes per cookie,
/// "as measured by the sum of the length of the cookie's name, value, and
/// attributes"; Chrome and Firefox enforce it, and an oversized cookie is
/// dropped SILENTLY — the callback redirects, the next page has no session,
/// login starts over, and the pod log records a successful login.
///
/// What the browser measures is four expansions away from anything a caller
/// can see, which is why `oidc.rs`'s 255-byte cap on the raw claims did not
/// deliver the guarantee its comment claimed:
///   JSON escaping    — one control byte becomes six, so 255 raw bytes of
///                      claim serialize to 1530.
///   AES-GCM          — plus a 12-byte nonce and a 16-byte tag.
///   base64           — times 4/3.
///   percent-encoding — the jar emits `Cookie::encoded()`, and some base64
///                      characters cost 3 bytes there.
/// 2048 plaintext bytes sits well inside that, and the shrink below always
/// succeeds on its first try. Note what that means for when it fires: three
/// ASCII-graphic claims at the 255-byte cap serialize to ~950 bytes, so no
/// ordinary profile ever reaches this — only escape-heavy IdP input does.
///
/// The arithmetic is not the guarantee. The worst case the claim bounds
/// permit (`sub` at its cap with every byte a control character, both
/// display claims dropped) measures ~2.5 KiB on the wire — and only ever
/// approximately, because the count of percent-encoded characters tracks the
/// random GCM nonce and drifts tens of bytes per run. What holds the bound is
/// `control_heavy_claims_at_the_cap_still_fit_the_cookie_limit`, which
/// measures the real header through the real jar.
const MAX_SESSION_PLAINTEXT_BYTES: usize = 2048;

pub fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// 32 bytes of CSPRNG, base64url — used for state, nonce, CSRF, PKCE verifier.
pub fn random_token() -> String {
    let mut b = [0u8; 32];
    rand::rng().fill_bytes(&mut b);
    use base64::Engine;
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(b)
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Session {
    /// Random session id — the handle logout revokes (`AppState::revoke_session`).
    pub sid: String,
    pub sub: String,
    pub email: Option<String>,
    pub name: Option<String>,
    pub csrf: String,
    /// Absolute expiry (epoch seconds).
    pub exp: i64,
    /// Last activity (epoch seconds); slides on every authenticated request.
    /// No serde default: a cookie without it (pre-idle-timeout format) is
    /// simply invalid and forces a re-login.
    pub last_seen: i64,
    /// Not part of the session cookie: the extractor fills it from the theme
    /// cookie on every request, so every page renders in the operator's theme.
    #[serde(skip)]
    pub theme: Theme,
}

impl Session {
    pub fn display_name(&self) -> &str {
        self.name
            .as_deref()
            .or(self.email.as_deref())
            .unwrap_or(&self.sub)
    }

    /// Constant-time-ish comparison is unnecessary here (the token is a
    /// per-session random value, not a long-lived credential), but reject
    /// empty-vs-empty explicitly so a bug can't make "" a valid token.
    pub fn verify_csrf(&self, submitted: &str) -> Result<(), AppError> {
        if submitted.is_empty() || submitted != self.csrf {
            return Err(AppError::Forbidden("CSRF token mismatch".into()));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct LoginState {
    pub state: String,
    pub nonce: String,
    pub pkce_verifier: String,
    /// Post-login destination (validated: same-site path only).
    pub next: String,
    pub exp: i64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Flash {
    pub kind: String, // "ok" | "error"
    pub msg: String,
}

pub(crate) fn base_cookie(
    name: &'static str,
    value: String,
    secure: bool,
    max_age_secs: i64,
) -> Cookie<'static> {
    let mut c = Cookie::new(name, value);
    c.set_http_only(true);
    c.set_same_site(SameSite::Lax);
    c.set_secure(secure);
    c.set_path("/");
    c.set_max_age(time::Duration::seconds(max_age_secs));
    c
}

/// Removal cookie matching `base_cookie`'s attributes. Removal only takes
/// effect when `Path` matches the stored cookie — a bare `Cookie::from(name)`
/// removal inherits the request's path and silently fails everywhere but `/`
/// (logout would be a no-op on the memory of record's write session).
pub fn removal_cookie(name: &'static str) -> Cookie<'static> {
    Cookie::build(name).path("/").build()
}

pub fn session_cookie(jar: PrivateCookieJar, s: &Session, secure: bool) -> PrivateCookieJar {
    let mut value = serde_json::to_string(s).expect("session serializes");
    if value.len() > MAX_SESSION_PLAINTEXT_BYTES {
        // The two display claims are the only droppable fields. `sub`
        // identifies the session and truncating it would alias two subjects
        // onto one allowlist entry; `sid`, `csrf` and the timestamps are the
        // session. `display_name` falls back name -> email -> sub, so the
        // console greets this operator by their raw subject identifier from
        // here on — the whole visible cost, and the reason `sub` has to stay.
        value = serde_json::to_string(&Session {
            email: None,
            name: None,
            ..s.clone()
        })
        .expect("session serializes");
        tracing::warn!(
            bytes = value.len(),
            "session display claims dropped: the full profile exceeded the cookie budget"
        );
    }
    if value.len() > MAX_SESSION_PLAINTEXT_BYTES {
        // Unreachable while `oidc.rs` caps `sub` at 255 bytes: six bytes per
        // escaped byte plus the fixed fields leaves the fallback ~1.7 KiB.
        // But that guarantee lives in another module and rests on a constant
        // nobody here can see, and dropping the display claims is the only
        // lever this function has — so if it ever stops being enough, the one
        // thing that must not happen again is the failure being invisible.
        tracing::error!(
            bytes = value.len(),
            "session cookie over budget after dropping display claims — the browser may drop it"
        );
    }
    jar.add(base_cookie(SESSION_COOKIE, value, secure, SESSION_TTL_SECS))
}

pub fn login_cookie(jar: PrivateCookieJar, s: &LoginState, secure: bool) -> PrivateCookieJar {
    let value = serde_json::to_string(s).expect("login state serializes");
    jar.add(base_cookie(LOGIN_COOKIE, value, secure, LOGIN_TTL_SECS))
}

pub fn flash_cookie(jar: PrivateCookieJar, f: &Flash, secure: bool) -> PrivateCookieJar {
    let value = serde_json::to_string(f).expect("flash serializes");
    jar.add(base_cookie(FLASH_COOKIE, value, secure, 120))
}

/// Ceiling on a flash message as the cookie carries it: the bytes of its
/// JSON string, escapes included. The flash rides an encrypted cookie, and a
/// cookie over the browser's ~4 KiB limit is dropped silently, taking the
/// message with it. A message can quote upstream text, so a ceiling in
/// characters is not enough: a quote costs two bytes in JSON, a control
/// character six, a non-ASCII character up to four.
const MAX_FLASH_BYTES: usize = 1800;

/// `msg` as a flash carries it: control characters become spaces, and a
/// message whose JSON string would pass `MAX_FLASH_BYTES` is cut there and
/// marked with '…'.
fn flash_text(msg: &str) -> String {
    let cost = |c: char| {
        if matches!(c, '"' | '\\') {
            2
        } else {
            c.len_utf8()
        }
    };
    let text: String = msg
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    if text.chars().map(cost).sum::<usize>() <= MAX_FLASH_BYTES {
        return text;
    }
    let mut out = String::new();
    let mut size = 0;
    for c in text.chars() {
        if size + cost(c) > MAX_FLASH_BYTES - '…'.len_utf8() {
            break;
        }
        size += cost(c);
        out.push(c);
    }
    out.push('…');
    out
}

/// POST-redirect-GET with a one-shot banner: set the flash, then 303 to `to`.
pub fn flash_redirect(
    jar: PrivateCookieJar,
    secure: bool,
    kind: &str,
    msg: String,
    to: &str,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    let msg = flash_text(&msg);
    let jar = flash_cookie(
        jar,
        &Flash {
            kind: kind.into(),
            msg,
        },
        secure,
    );
    (jar, axum::response::Redirect::to(to)).into_response()
}

pub fn read_login(jar: &PrivateCookieJar) -> Option<LoginState> {
    let c = jar.get(LOGIN_COOKIE)?;
    let s: LoginState = serde_json::from_str(c.value()).ok()?;
    (s.exp > now_epoch()).then_some(s)
}

/// The session cookie's session. Its `theme` is `System`: only the `Session`
/// extractor fills it, from the theme cookie.
pub fn read_session(jar: &PrivateCookieJar) -> Option<Session> {
    let c = jar.get(SESSION_COOKIE)?;
    let s: Session = serde_json::from_str(c.value()).ok()?;
    let now = now_epoch();
    // Both limits must hold: absolute lifetime AND idle window.
    (s.exp > now && s.last_seen + IDLE_TTL_SECS > now).then_some(s)
}

/// Read-and-clear the flash. Returns the (possibly modified) jar so the
/// removal reaches the response.
pub fn take_flash(jar: PrivateCookieJar) -> (PrivateCookieJar, Option<Flash>) {
    match jar.get(FLASH_COOKIE) {
        Some(c) => {
            let f: Option<Flash> = serde_json::from_str(c.value()).ok();
            (jar.remove(removal_cookie(FLASH_COOKIE)), f)
        }
        None => (jar, None),
    }
}

/// A fresh session. Its `theme` is `System`: only the `Session` extractor
/// fills it, from the theme cookie.
pub fn new_session(sub: String, email: Option<String>, name: Option<String>) -> Session {
    Session {
        sid: random_token(),
        sub,
        email,
        name,
        csrf: random_token(),
        exp: now_epoch() + SESSION_TTL_SECS,
        last_seen: now_epoch(),
        theme: Theme::System,
    }
}

/// The callback's claims -> session mapping, in one place so a test that
/// drives it is testing what production runs. `preferred_username` is the
/// fallback `name`, matching what `display_name` then prefers.
pub fn session_for(claims: crate::oidc::IdClaims) -> Session {
    new_session(
        claims.sub,
        claims.email,
        claims.name.or(claims.preferred_username),
    )
}

pub fn new_login_state(next: String) -> LoginState {
    LoginState {
        state: random_token(),
        nonce: random_token(),
        pkce_verifier: random_token(),
        next,
        exp: now_epoch() + LOGIN_TTL_SECS,
    }
}

/// Extractor: a valid, unexpired session — or a redirect to login for GETs
/// and a hard 403 for anything else (a POST that lost its session must not
/// be replayed through an auth flow).
impl FromRequestParts<AppState> for Session {
    type Rejection = AppError;

    async fn from_request_parts(parts: &mut Parts, state: &AppState) -> Result<Self, AppError> {
        let jar = PrivateCookieJar::from_request_parts(parts, state)
            .await
            .map_err(|_| AppError::LoginRedirect)?;
        // Logged-out sids are rejected identically to missing sessions.
        match read_session(&jar).filter(|s| !state.is_revoked(&s.sid)) {
            Some(s) => Ok(Session {
                theme: theme::read(&jar),
                ..s
            }),
            None if parts.method == axum::http::Method::GET => Err(AppError::LoginRedirect),
            None => Err(AppError::Forbidden("no session".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    /// The flash survives whatever text it quotes: under the browser's
    /// cookie limit at its worst, as the real jar encodes it.
    #[test]
    fn a_flash_fits_the_cookie_limit_whatever_it_quotes() {
        use axum::http::header::SET_COOKIE;
        use axum_extra::extract::cookie::{Key, PrivateCookieJar};
        let key = Key::from(&[7u8; 64][..]);
        for (label, msg) in [
            ("ascii", "a".repeat(10_000)),
            ("quotes", "\"".repeat(10_000)),
            ("backslashes", "\\".repeat(10_000)),
            ("controls", "\u{1}".repeat(10_000)),
            ("four-byte characters", "🔒".repeat(10_000)),
        ] {
            let resp = flash_redirect(PrivateCookieJar::new(key.clone()), true, "error", msg, "/");
            let header = resp
                .headers()
                .get(SET_COOKIE)
                .unwrap()
                .to_str()
                .unwrap()
                .to_string();
            assert!(
                header.len() <= 4096,
                "{label}: Set-Cookie is {} bytes",
                header.len()
            );
        }
        // Short text passes as it is, control characters aside.
        assert_eq!(flash_text("plain \"quoted\""), "plain \"quoted\"");
        assert_eq!(flash_text("a\nb\u{7}c"), "a b c");
        let cut = flash_text(&"a".repeat(MAX_FLASH_BYTES + 1));
        assert!(
            cut.ends_with('…') && cut.len() <= MAX_FLASH_BYTES,
            "{}",
            cut.len()
        );
        assert_eq!(
            flash_text(&"a".repeat(MAX_FLASH_BYTES)).len(),
            MAX_FLASH_BYTES
        );
    }

    use super::*;

    #[test]
    fn csrf_rejects_empty_and_mismatch() {
        let s = new_session("sub".into(), None, None);
        assert!(s.verify_csrf("").is_err());
        assert!(s.verify_csrf("wrong").is_err());
        assert!(s.verify_csrf(&s.csrf.clone()).is_ok());
    }

    #[test]
    fn random_tokens_are_unique_and_urlsafe() {
        let a = random_token();
        let b = random_token();
        assert_ne!(a, b);
        assert!(
            a.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        );
    }

    fn jar_with(s: &Session) -> PrivateCookieJar {
        let key = axum_extra::extract::cookie::Key::from(&[7u8; 64]);
        session_cookie(PrivateCookieJar::new(key), s, true)
    }

    /// read_session enforces BOTH the absolute lifetime and the idle window
    /// (team session rule: absolute ≤ 12 h, idle ≤ 15 min).
    #[test]
    fn read_session_enforces_idle_and_absolute_limits() {
        let fresh = new_session("sub".into(), None, None);
        assert!(read_session(&jar_with(&fresh)).is_some());

        let mut idle = new_session("sub".into(), None, None);
        idle.last_seen = now_epoch() - IDLE_TTL_SECS - 1;
        assert!(
            read_session(&jar_with(&idle)).is_none(),
            "idle-expired session must be rejected even inside absolute lifetime"
        );

        let mut expired = new_session("sub".into(), None, None);
        expired.exp = now_epoch() - 1;
        assert!(read_session(&jar_with(&expired)).is_none());
    }

    /// The removal cookie must carry `Path=/` to match `base_cookie`'s
    /// attributes — otherwise the browser scopes the removal to the request
    /// path and logout/flash-clear silently no-op off `/`.
    #[test]
    fn removal_cookie_carries_root_path() {
        assert_eq!(removal_cookie(SESSION_COOKIE).path(), Some("/"));
        assert_eq!(removal_cookie(FLASH_COOKIE).path(), Some("/"));
    }
}
