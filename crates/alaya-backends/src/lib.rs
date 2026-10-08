mod anthropic;
pub mod embedding;
pub mod graph;
pub mod graph_ref;
pub mod judge;
mod openai;
pub mod qdrant;
pub mod rerank;
pub mod summary;
pub mod traits;

pub use traits::*;

/// Wire protocol an LLM role (summary, judge) speaks. Selected per role at
/// boot (`SUMMARY_PROVIDER`, `JUDGE_PROVIDER`); the prompt, the verdict
/// schema and its validation are the same on both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Provider {
    /// Anthropic Messages API: `POST /v1/messages`, `x-api-key`.
    #[default]
    Anthropic,
    /// Any OpenAI-compatible chat-completions endpoint:
    /// `POST /v1/chat/completions`, `Authorization: Bearer`.
    OpenAi,
}

impl Provider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
        }
    }
}

impl std::str::FromStr for Provider {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "anthropic" => Ok(Self::Anthropic),
            "openai" => Ok(Self::OpenAi),
            _ => Err(format!("expected `anthropic` or `openai`, got {s:?}")),
        }
    }
}

pub(crate) fn redact_reqwest_error(mut error: reqwest::Error) -> String {
    if let Some(url) = error.url_mut() {
        url.set_query(None);
        url.set_fragment(None);
    }
    error.to_string()
}

/// How many bytes of one string a log line or span field keeps: upstream
/// error bodies and caller-supplied labels have no length limit of their own.
pub const LOG_CLIP_BYTES: usize = 256;

/// `s` fit for a log line or span field: its first [`LOG_CLIP_BYTES`] bytes,
/// cut on a char boundary, with the full length noted when anything was cut,
/// and control characters escaped so a value cannot break or forge a line.
/// A span records its fields before the function body runs, so a caller
/// string a span records goes through this even where the body then
/// validates it.
pub fn clip_for_log(s: &str) -> std::borrow::Cow<'_, str> {
    if s.len() <= LOG_CLIP_BYTES && !s.chars().any(char::is_control) {
        return s.into();
    }
    let head = &s[..s.floor_char_boundary(LOG_CLIP_BYTES)];
    let mut out = String::with_capacity(head.len() + 24);
    for c in head.chars() {
        if c.is_control() {
            out.extend(c.escape_default());
        } else {
            out.push(c);
        }
    }
    if head.len() < s.len() {
        out.push_str(&format!("… ({} bytes)", s.len()));
    }
    out.into()
}

/// The body of a non-2xx response, clipped for the error message it goes
/// into: every such message is logged, and an upstream that echoes the
/// request (an OpenAI-compatible proxy's 422 does) would echo memory content.
/// Control characters are dropped, not escaped: a deterministic judge failure
/// stores this text on the edge through `sanitize_reason`, which drops them.
pub(crate) async fn error_body(resp: reqwest::Response) -> String {
    match resp.text().await {
        Ok(body) => {
            let body: String = body.chars().filter(|c| !c.is_control()).collect();
            clip_for_log(&body).into_owned()
        }
        Err(_) => "<unreadable>".into(),
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::Provider;

    #[test]
    fn provider_parses_both_names_and_rejects_anything_else() {
        assert_eq!("anthropic".parse::<Provider>(), Ok(Provider::Anthropic));
        assert_eq!(" OpenAI ".parse::<Provider>(), Ok(Provider::OpenAi));
        assert_eq!(Provider::default(), Provider::Anthropic);
        let err = "litellm".parse::<Provider>().unwrap_err();
        assert!(err.contains("litellm"), "{err}");
        assert!("".parse::<Provider>().is_err());
    }

    #[test]
    fn clip_for_log_bounds_and_escapes() {
        use super::{LOG_CLIP_BYTES, clip_for_log};
        use std::borrow::Cow;

        assert!(matches!(
            clip_for_log("operator:mcp"),
            Cow::Borrowed("operator:mcp")
        ));

        // "é" is two bytes and straddles the cap: the cut lands before it.
        let long = format!("{}é{}", "a".repeat(LOG_CLIP_BYTES - 1), "TAIL");
        let clipped = clip_for_log(&long);
        let head = "a".repeat(LOG_CLIP_BYTES - 1);
        assert_eq!(clipped, format!("{head}… ({} bytes)", long.len()));

        assert_eq!(
            clip_for_log("a\nINFO forged\u{1b}"),
            "a\\nINFO forged\\u{1b}"
        );
    }

    #[tokio::test]
    async fn reqwest_error_redacts_query() {
        #[allow(clippy::disallowed_methods, reason = "test-only client")]
        let error = reqwest::Client::new()
            .get("http://127.0.0.1:1/?api_key=SECRET#fragment")
            .send()
            .await
            .unwrap_err();
        let message = super::redact_reqwest_error(error);
        assert!(message.contains("127.0.0.1"), "url dropped: {message}");
        assert!(!message.contains("SECRET"), "query leaked: {message}");
        assert!(!message.contains("api_key"), "query leaked: {message}");
        assert!(!message.contains("fragment"), "fragment leaked: {message}");
    }
}
