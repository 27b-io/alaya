//! OIDC login / callback / logout.
//!
//! The callback is the session-fixation boundary: it always mints a brand-new
//! session cookie (fresh CSRF token included) and deletes the transient flow
//! cookie — no pre-authentication cookie value survives login. A subject not
//! on the allowlist gets an explicit 403 (AC1), never a degraded session.

use axum::extract::{Query, State};
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use serde::Deserialize;

use crate::error::AppError;
use crate::routes::safe_next;
use crate::session::{self, Flash, LOGIN_COOKIE, SESSION_COOKIE, new_login_state, read_login};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct LoginQuery {
    #[serde(default)]
    pub next: String,
}

pub async fn login(
    State(state): State<AppState>,
    Query(q): Query<LoginQuery>,
    jar: PrivateCookieJar,
) -> Result<Response, AppError> {
    let login = new_login_state(safe_next(&q.next));
    let url = state
        .oidc
        .authorize_url(&login.state, &login.nonce, &login.pkce_verifier)
        .await
        .map_err(|e| AppError::Upstream(format!("identity provider: {e}")))?;
    let jar = session::login_cookie(jar, &login, state.secure_cookies());
    Ok((jar, Redirect::to(&url)).into_response())
}

#[derive(Deserialize)]
pub struct CallbackQuery {
    pub code: Option<String>,
    pub state: Option<String>,
    pub error: Option<String>,
}

pub async fn callback(
    State(state): State<AppState>,
    Query(q): Query<CallbackQuery>,
    jar: PrivateCookieJar,
) -> Result<Response, AppError> {
    if let Some(err) = q.error {
        // IdP-reported error (user denied, etc). Rendered escaped, but this
        // route is unauthenticated and runs before the state check, so any
        // free text here is attacker-chosen prose on our origin (CWE-451).
        // The raw value goes to the log, where it is diagnosable but not
        // spoofable. Debug-formatted so control characters are escaped and an
        // attacker cannot forge log records with embedded newlines (CWE-117).
        tracing::warn!(error = ?clipped_idp_error(&err), "idp callback error");
        return Err(AppError::Forbidden(format!(
            "identity provider: {}",
            idp_error_detail(&err)
        )));
    }
    let (code, cb_state) = match (q.code, q.state) {
        (Some(c), Some(s)) => (c, s),
        _ => return Err(AppError::BadRequest("missing code/state".into())),
    };

    let login = read_login(&jar)
        .ok_or_else(|| AppError::BadRequest("login flow expired — start again".into()))?;
    if login.state != cb_state {
        return Err(AppError::Forbidden("state mismatch".into()));
    }
    // Before the exchange, not after it: a second callback on this state —
    // a scripted caller resending the original `Cookie:` header ignores the
    // deletion below — is refused without an outbound call to the IdP.
    if !state.consume_login(&login.state, login.exp) {
        // Never log the state itself. A replay and a flow that expired in
        // flight both land here, so neither line may claim a replay.
        tracing::warn!("oidc: login state spent or expired — callback refused");
        return Err(AppError::BadRequest(
            "login flow already used or expired — start again".into(),
        ));
    }

    let claims = state
        .oidc
        .exchange_and_verify(&code, &login.pkce_verifier, &login.nonce)
        .await
        // No warn here: every arm below this call already logs its own, at
        // the refusal that produced it (`warn_rejected`, `warn_idp_failure`,
        // `warn_idp_parse_failure`, `http::body_text`). `warn_rejected` has
        // the reasoning; a wrapper at this level cannot tell a refusal from
        // an outage.
        .map_err(|e| AppError::Forbidden(format!("login failed: {e}")))?;

    // `sub` is recorded with `?`, not `%`, here and everywhere it is logged.
    // It is an unvalidated string straight out of the id_token, logged on
    // the rejection path of an unauthenticated route — so it reaches the log
    // for subjects that are authorized for nothing — and the plain-text
    // subscriber writes a `Display`-recorded field verbatim, so `%` would
    // let a substituted IdP forge whole log records by putting a newline in
    // the subject. `oidc.rs` pins the mechanism with a test, and bounds the
    // length there so neither line can be made arbitrarily long.
    //
    // Default-deny subject allowlist (AC1): explicit 403, nothing minted.
    if !state.config.subject_allowed(&claims.sub) {
        tracing::warn!(sub = ?claims.sub, "login rejected: subject not allowlisted");
        return Err(AppError::Forbidden(
            "this account is not authorized for the console".into(),
        ));
    }

    tracing::info!(sub = ?claims.sub, "console login");
    let sess = session::session_for(claims);
    let secure = state.secure_cookies();
    let jar = jar.remove(session::removal_cookie(LOGIN_COOKIE));
    let jar = session::session_cookie(jar, &sess, secure);
    Ok((jar, Redirect::to(&safe_next(&login.next))).into_response())
}

#[derive(Deserialize)]
pub struct LogoutForm {
    #[serde(default)]
    pub csrf: String,
}

pub async fn logout(
    State(state): State<AppState>,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<LogoutForm>,
) -> Result<Response, AppError> {
    // CSRF-protect logout too — forced logout is a nuisance vector.
    if let Some(sess) = session::read_session(&jar) {
        sess.verify_csrf(&form.csrf)?;
        // Server-side revocation: the cookie is dead even if a concurrent
        // in-flight refresh re-lands it in the browser jar (CWE-613).
        state.revoke_session(&sess.sid, sess.exp);
    }
    let jar = jar.remove(session::removal_cookie(SESSION_COOKIE));
    let jar = session::flash_cookie(
        jar,
        &Flash {
            kind: "ok".into(),
            msg: "Logged out.".into(),
        },
        state.secure_cookies(),
    );
    Ok((jar, Redirect::to("/auth/login")).into_response())
}

/// RFC 6749 §4.1.2.1 error codes. Anything else collapses to a fixed string so
/// `/auth/callback?error=…` cannot put arbitrary prose on the 403 page.
fn idp_error_detail(err: &str) -> &str {
    match err {
        "access_denied"
        | "invalid_request"
        | "unauthorized_client"
        | "unsupported_response_type"
        | "invalid_scope"
        | "server_error"
        | "temporarily_unavailable" => err,
        _ => "login refused",
    }
}

/// Ceiling on the raw IdP error in the log line. The longest code in
/// RFC 6749 §4.1.2.1 and OIDC Core §3.1.2.6 is 26 characters.
const MAX_LOGGED_ERROR_CHARS: usize = 120;

/// The IdP's raw error, clipped for the log line and marked when clipped.
///
/// `error` is caller-controlled on an unauthenticated route: `?` bounds the
/// alphabet, this bounds the length. Logged raw rather than collapsed,
/// because `idp_error_detail` renders a fixed string for every code outside
/// RFC 6749, so a genuine vendor code survives here and — until the request
/// span stops recording the whole URI — nowhere else worth reading. Marked,
/// or a clip reads as the IdP's own value. By chars: a byte split can land
/// mid-codepoint and panic.
fn clipped_idp_error(err: &str) -> String {
    let mut out: String = err.chars().take(MAX_LOGGED_ERROR_CHARS).collect();
    // Exact, and O(1): `out` is by construction a byte prefix of `err`.
    if out.len() < err.len() {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{MAX_LOGGED_ERROR_CHARS, clipped_idp_error, idp_error_detail};

    #[test]
    fn idp_error_detail_passes_spec_codes_and_collapses_everything_else() {
        for code in [
            "access_denied",
            "invalid_request",
            "unauthorized_client",
            "unsupported_response_type",
            "invalid_scope",
            "server_error",
            "temporarily_unavailable",
        ] {
            assert_eq!(idp_error_detail(code), code);
        }
        assert_eq!(
            idp_error_detail("your session expired, sign in again at evil.example"),
            "login refused"
        );
        assert_eq!(idp_error_detail(""), "login refused");
        assert_eq!(idp_error_detail("Access_Denied"), "login refused");
    }

    #[test]
    fn clipped_idp_error_marks_only_what_it_clipped() {
        assert_eq!(clipped_idp_error("access_denied"), "access_denied");

        let flood = "x".repeat(8192);
        let clipped = clipped_idp_error(&flood);
        assert_eq!(clipped.chars().count(), MAX_LOGGED_ERROR_CHARS + 1);
        assert!(clipped.ends_with('…'));

        // At the ceiling in 4-byte codepoints: a byte-compared bound would
        // false-clip this, and a byte slice would panic on it.
        let wide = "🔒".repeat(MAX_LOGGED_ERROR_CHARS);
        assert_eq!(clipped_idp_error(&wide), wide);
    }
}
