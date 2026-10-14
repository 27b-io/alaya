//! Operator preferences: today only the colour theme (`theme.rs`). They set
//! a cookie and call no upstream.

use axum::extract::State;
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use serde::Deserialize;

use crate::error::AppError;
use crate::routes::{guard_write, return_to};
use crate::session::Session;
use crate::state::AppState;
use crate::theme::{self, Theme};

#[derive(Deserialize)]
pub struct ThemeForm {
    csrf: String,
    theme: String,
    #[serde(default)]
    back: String,
}

/// Save the theme, then go back to the page's navigation entry. The switch
/// carries that entry's path (`ui::Nav`), always a short GET page, so the
/// way back can be neither a POST-only URL nor one too long to accept; it is
/// still held to `return_to` like every other form's.
pub async fn set_theme(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<ThemeForm>,
) -> Result<Response, AppError> {
    guard_write(&state, &session, &form.csrf)?;
    let theme =
        Theme::parse(&form.theme).ok_or_else(|| AppError::BadRequest("unknown theme".into()))?;
    let jar = theme::store(jar, theme, state.secure_cookies());
    Ok((jar, Redirect::to(&return_to(&form.back, "/"))).into_response())
}
