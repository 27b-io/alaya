//! SummaryClient — SummaryProvider implementation over the shared LLM
//! transport (Anthropic Messages API or OpenAI-compatible chat completions).

use async_trait::async_trait;

use alaya_types::{AlayaError, Result};

use crate::anthropic::{DEFAULT_REQUEST_TIMEOUT, MessagesTransport, Prompt, truncate_chars};
use crate::{Provider, SummaryProvider};

const SYSTEM_PROMPT: &str = "Summarize the following in one concise sentence of approximately 50 tokens. \
     Return only the summary, no preamble.";

pub struct SummaryClient {
    transport: MessagesTransport,
    model: String,
}

impl SummaryClient {
    /// Fails with `Config` on bearer material that is not a valid header
    /// value (e.g. a trailing newline, #97) or on a client build error; the
    /// error never echoes `api_key` (see `MessagesTransport::for_provider`).
    pub fn new(
        provider: Provider,
        base_url: String,
        model: String,
        api_key: Option<String>,
    ) -> Result<Self> {
        Ok(Self {
            transport: MessagesTransport::for_provider(
                provider,
                base_url,
                api_key,
                DEFAULT_REQUEST_TIMEOUT,
            )?,
            model,
        })
    }
}

#[async_trait(?Send)]
impl SummaryProvider for SummaryClient {
    #[tracing::instrument(skip(self, content), fields(content_len = content.len()))]
    async fn summarize(&self, content: &str) -> Result<String> {
        let prompt = Prompt {
            model: &self.model,
            system: SYSTEM_PROMPT,
            user: truncate_chars(content),
            max_tokens: 100,
            schema: None,
        };

        let parsed = self
            .transport
            .complete(&prompt, AlayaError::Summary)
            .await?;

        parsed
            .text
            .ok_or_else(|| AlayaError::Summary("empty completion".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #97 repro shape: a control character in the bearer. Must be `Config`,
    /// and the message must not echo the key it is rejecting.
    #[test]
    fn new_rejects_control_chars_without_echoing_the_key() {
        let Err(err) = SummaryClient::new(
            Provider::Anthropic,
            "http://anthropic".into(),
            "claude-haiku-4-5-20251001".into(),
            Some("abc\n".into()),
        ) else {
            panic!("a control character in the bearer must be rejected");
        };
        let msg = err.to_string();
        assert!(matches!(err, AlayaError::Config(_)), "{msg}");
        assert!(!msg.contains("abc"), "error echoed the key: {msg}");
    }

    /// LAB-6877 AC-5: the openai wire carries the same prompt as system and
    /// user messages and reads the first non-empty content back.
    #[cfg(not(target_arch = "wasm32"))]
    mod http_openai {
        use super::*;
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        fn client(server: &MockServer) -> SummaryClient {
            SummaryClient::new(
                Provider::OpenAi,
                server.uri(),
                "test-model".into(),
                Some("test-key".into()),
            )
            .unwrap()
        }

        #[tokio::test]
        async fn summary_round_trip() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/chat/completions"))
                .and(header("authorization", "Bearer test-key"))
                .and(body_partial_json(json!({
                    "model": "test-model",
                    "max_completion_tokens": 100,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "the memory"},
                    ],
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{"message": {"content": " A summary. "}, "finish_reason": "stop"}]
                })))
                .expect(1)
                .mount(&server)
                .await;
            assert_eq!(
                client(&server).summarize("the memory").await.unwrap(),
                "A summary."
            );
            let req = &server.received_requests().await.unwrap()[0];
            let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
            assert!(body.get("response_format").is_none(), "{body}");
            assert!(body.get("max_tokens").is_none(), "{body}");
        }

        #[tokio::test]
        async fn failures_are_errors_for_the_caller_to_fail_open_on() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(400).set_body_string("bad"))
                .mount(&server)
                .await;
            let e = client(&server).summarize("x").await.unwrap_err();
            assert!(matches!(e, AlayaError::Summary(_)), "{e:?}");

            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": []})))
                .mount(&server)
                .await;
            let e = client(&server).summarize("x").await.unwrap_err();
            assert!(matches!(e, AlayaError::Summary(_)), "{e:?}");
        }
    }
}
