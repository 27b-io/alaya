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

impl From<reqwest::Error> for AppError {
    fn from(e: reqwest::Error) -> Self {
        // reqwest errors can embed the full request URL; strip to a safe
        // summary so credentials-adjacent detail never reaches the page.
        let kind = if e.is_timeout() {
            "timeout"
        } else if e.is_connect() {
            "connection failed"
        } else {
            "request failed"
        };
        AppError::Upstream(format!("alaya-server: {kind}"))
    }
}
