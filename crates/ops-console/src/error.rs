//! Console error type. Every variant renders a minimal, fully-escaped HTML
//! page — upstream error bodies are treated as untrusted text, never markup.

use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Redirect, Response};
use leptos::prelude::*;

pub enum AppError {
    /// Not logged in — bounce to login (GET only; POSTs get 403 instead).
    LoginRedirect,
    /// Authenticated but not allowed (AC1: explicit 403, never silent).
    Forbidden(String),
    BadRequest(String),
    NotFound(String),
    /// alaya-server (or the IdP) said no / was unreachable.
    Upstream(String),
}

impl AppError {
    /// Human-readable detail, for pages that render an upstream failure
    /// inline (one dark section) instead of failing the whole page.
    pub fn detail(&self) -> &str {
        match self {
            AppError::LoginRedirect => "login required",
            AppError::Forbidden(d)
            | AppError::BadRequest(d)
            | AppError::NotFound(d)
            | AppError::Upstream(d) => d,
        }
    }

    fn status(&self) -> StatusCode {
        match self {
            AppError::LoginRedirect => StatusCode::SEE_OTHER,
            AppError::Forbidden(_) => StatusCode::FORBIDDEN,
            AppError::BadRequest(_) => StatusCode::BAD_REQUEST,
            AppError::NotFound(_) => StatusCode::NOT_FOUND,
            AppError::Upstream(_) => StatusCode::BAD_GATEWAY,
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let status = self.status();
        let (title, detail) = match self {
            AppError::LoginRedirect => return Redirect::to("/auth/login").into_response(),
            AppError::Forbidden(d) => ("403 — forbidden", d),
            AppError::BadRequest(d) => ("400 — bad request", d),
            AppError::NotFound(d) => ("404 — not found", d),
            AppError::Upstream(d) => ("502 — upstream error", d),
        };
        // view! escapes `detail` as a text node; upstream text can't inject
        // markup into the error page.
        let body = view! {
            <div class="mx-auto max-w-lg py-24 px-6 font-sans">
                <h1 class="text-lg font-semibold">{title}</h1>
                <p class="text-muted-foreground text-sm mt-2">{detail}</p>
                <p class="mt-6"><a class="text-primary underline underline-offset-4" href="/">"Back to console"</a></p>
            </div>
        }
        .to_html();
        (
            status,
            Html(crate::ui::document("Error — ops console", body)),
        )
            .into_response()
    }
}

impl AppError {
    /// Transport failure against a named upstream. reqwest errors can embed
    /// the full request URL, so only a one-phrase kind ever reaches a page.
    /// Deliberately not a `From` impl: a bare `?` would have to guess which
    /// upstream failed.
    pub fn transport(what: &str, e: &reqwest::Error) -> Self {
        let kind = if e.is_timeout() {
            "timeout"
        } else if e.is_connect() {
            "connection failed"
        } else {
            "request failed"
        };
        AppError::Upstream(format!("{what}: {kind}"))
    }

    /// A body read that failed part-way or ran past the cap. Overrun is not
    /// a transport fault and must not read as one: an operator told
    /// "connection failed" chases a network that is fine.
    pub fn body(what: &str, e: crate::http::BodyError) -> Self {
        match e {
            crate::http::BodyError::Transport(e) => AppError::transport(what, &e),
            crate::http::BodyError::TooLarge => {
                AppError::Upstream(format!("{what}: response too large"))
            }
        }
    }

    /// Non-2xx from a named upstream. Only the `error` field of a JSON error
    /// body is surfaced, bounded; an arbitrary body (a proxy's HTML page, a
    /// stack trace) never reaches a page.
    pub fn non_success(what: &str, status: StatusCode, body: &str) -> Self {
        let detail: String = serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(|v| {
                v.get("error")?
                    .as_str()
                    .map(|s| s.chars().take(160).collect())
            })
            .unwrap_or_else(|| "unrecognized error body".to_string());
        AppError::Upstream(format!("{what} {status}: {detail}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_success_surfaces_only_a_bounded_json_error_field() {
        let status = StatusCode::UNAUTHORIZED;
        let e = AppError::non_success("lb", status, r#"{"error":"bad key"}"#);
        assert_eq!(e.detail(), "lb 401 Unauthorized: bad key");
        // A proxy's HTML page or a plain-text body never reaches a page.
        let e = AppError::non_success("lb", status, "<html><body>nginx 502</body></html>");
        assert_eq!(e.detail(), "lb 401 Unauthorized: unrecognized error body");
        let long = format!(r#"{{"error":"{}"}}"#, "x".repeat(500));
        assert_eq!(
            AppError::non_success("lb", status, &long).detail().len(),
            "lb 401 Unauthorized: ".len() + 160
        );
    }
}
