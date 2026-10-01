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

    #[tokio::test]
    async fn reqwest_error_redacts_query() {
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
