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
        // IdP-reported error (user denied, etc). `err` renders escaped.
        return Err(AppError::Forbidden(format!("identity provider: {err}")));
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
