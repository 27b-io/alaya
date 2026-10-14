//! Operator preferences: today only the colour theme (`theme.rs`). They set
//! a cookie and call no upstream.

use axum::extract::State;
use axum::http::{HeaderMap, header};
use axum::response::{IntoResponse, Redirect, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use serde::Deserialize;

use crate::error::AppError;
use crate::routes::{guard_write, safe_next};
use crate::session::Session;
use crate::state::AppState;
use crate::theme::{self, Theme};

#[derive(Deserialize)]
pub struct ThemeForm {
    csrf: String,
    theme: String,
}

/// Save the theme, then go back to the page the switch was on. The switch
/// sits in the shared page header, which does not know the page's own path,
/// so the way back is the `Referer` (sent same-origin by `Referrer-Policy:
/// same-origin`), accepted only when it names this console's origin and then
/// held to `safe_next` like any other redirect target.
pub async fn set_theme(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    headers: HeaderMap,
    axum::Form(form): axum::Form<ThemeForm>,
) -> Result<Response, AppError> {
    guard_write(&state, &session, &form.csrf)?;
    let theme =
        Theme::parse(&form.theme).ok_or_else(|| AppError::BadRequest("unknown theme".into()))?;
    let jar = theme::store(jar, theme, state.secure_cookies());
    let back = referer_path(&headers, &state.config.public_origin());
    Ok((jar, Redirect::to(&back)).into_response())
}

/// The path (and query) of a same-origin `Referer`, or `/`.
fn referer_path(headers: &HeaderMap, origin: &str) -> String {
    headers
        .get(header::REFERER)
        .and_then(|v| v.to_str().ok())
        .and_then(|r| r.strip_prefix(origin))
        .filter(|rest| rest.starts_with('/'))
        .map_or_else(|| "/".to_string(), safe_next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::HeaderValue;

    fn referer(v: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert(header::REFERER, HeaderValue::from_str(v).unwrap());
        h
    }

    #[test]
    fn only_a_same_origin_referer_is_followed() {
        let origin = "https://console.test";
        assert_eq!(
            referer_path(&referer("https://console.test/alaya?q=x"), origin),
            "/alaya?q=x"
        );
        assert_eq!(referer_path(&referer("https://console.test/"), origin), "/");
        // A longer host that starts with the origin is another site.
        assert_eq!(
            referer_path(&referer("https://console.test.evil.com/x"), origin),
            "/"
        );
        assert_eq!(
            referer_path(&referer("https://evil.com/alaya"), origin),
            "/"
        );
        assert_eq!(
            referer_path(&referer("http://console.test/alaya"), origin),
            "/"
        );
        // Still held to safe_next: no way off-site through a crafted path.
        assert_eq!(
            referer_path(&referer("https://console.test//evil.com"), origin),
            "/"
        );
        assert_eq!(referer_path(&HeaderMap::new(), origin), "/");
    }
}
