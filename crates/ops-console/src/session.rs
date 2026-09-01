//! Cookie-backed sessions over axum-extra's private (AES-GCM encrypted) jar.
//!
//! Three cookies, all `HttpOnly` + `SameSite=Lax` + `Secure` (when the public
//! URL is https), all encrypted:
//! - `console_session` — authenticated identity + per-session CSRF token.
//! - `console_login`   — transient OIDC flow state (state/nonce/PKCE), 10 min.
//! - `console_flash`   — one-shot result banner after a POST-redirect-GET.
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

fn base_cookie(
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
    let value = serde_json::to_string(s).expect("session serializes");
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

pub fn read_login(jar: &PrivateCookieJar) -> Option<LoginState> {
    let c = jar.get(LOGIN_COOKIE)?;
    let s: LoginState = serde_json::from_str(c.value()).ok()?;
    (s.exp > now_epoch()).then_some(s)
}

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

pub fn new_session(sub: String, email: Option<String>, name: Option<String>) -> Session {
    Session {
        sub,
        email,
        name,
        csrf: random_token(),
        exp: now_epoch() + SESSION_TTL_SECS,
        last_seen: now_epoch(),
    }
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
        match read_session(&jar) {
            Some(s) => Ok(s),
            None if parts.method == axum::http::Method::GET => Err(AppError::LoginRedirect),
            None => Err(AppError::Forbidden("no session".into())),
        }
    }
}

#[cfg(test)]
mod tests {
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
