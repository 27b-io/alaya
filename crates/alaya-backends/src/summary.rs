//! SummaryClient — SummaryProvider implementation over the shared LLM
//! transport (Anthropic Messages API or OpenAI-compatible chat completions).

use async_trait::async_trait;

use alaya_types::{AlayaError, Result};

use crate::anthropic::{DEFAULT_REQUEST_TIMEOUT, MessagesTransport, Prompt, truncate_chars};
use crate::{Provider, SummaryProvider};

const SYSTEM_PROMPT: &str = "Summarize the following in one concise sentence of approximately 50 tokens. \
     Return only the summary, no preamble.";

/// Output budget per wire. The Anthropic wire keeps the 100 it sent before
/// the provider switch (LAB-6877 AC-7). A reasoning model on the OpenAI wire
/// spends `max_completion_tokens` on hidden reasoning first, and 100 can
/// leave no room for the summary (`finish_reason: "length"`, no content).
/// The cap is an upper bound, not a spend: a ~50-token summary costs the same.
const fn max_output_tokens(provider: Provider) -> u32 {
    match provider {
        Provider::Anthropic => 100,
        Provider::OpenAi => 1024,
    }
}

pub struct SummaryClient {
    transport: MessagesTransport,
    model: String,
    max_tokens: u32,
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
            max_tokens: max_output_tokens(provider),
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
            max_tokens: self.max_tokens,
            schema: None,
        };

        let parsed = self
            .transport
            .complete(&prompt, AlayaError::Summary)
            .await?;

        let stop = parsed.stop_reason.unwrap_or_default();
        parsed
            .text
            .ok_or_else(|| AlayaError::Summary(format!("empty completion (stop_reason={stop:?})")))
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

    /// LAB-6877 AC-7: the default wire's summary budget is what it was before
    /// the provider switch; only the openai wire gets reasoning headroom.
    #[test]
    fn budget_is_raised_on_the_openai_wire_only() {
        assert_eq!(max_output_tokens(Provider::Anthropic), 100);
        assert_eq!(max_output_tokens(Provider::OpenAi), 1024);
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
                    "max_completion_tokens": 1024,
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

        /// An OpenAI-compatible proxy's 422 can echo the request, memory
        /// content included. The error, which is logged, keeps the status
        /// and a clipped head of the body, never the whole of it, with its
        /// line breaks dropped as the judge's stored reason drops them.
        #[tokio::test]
        async fn an_echoed_request_body_is_clipped_in_the_error() {
            let echo = format!("{{\n  \"detail\": \"{}CONTENT-TAIL\"\n}}", "x".repeat(1000));
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(422).set_body_string(echo.clone()))
                .mount(&server)
                .await;
            let msg = client(&server)
                .summarize("x")
                .await
                .unwrap_err()
                .to_string();
            assert!(msg.contains("422"), "{msg}");
            assert!(msg.contains("{  \"detail\": \"xxx"), "{msg}");
            assert!(
                msg.contains(&format!("… ({} bytes)", echo.len() - 2)),
                "{msg}"
            );
            assert!(!msg.contains("CONTENT-TAIL"), "{msg}");
            assert!(!msg.contains('\n') && !msg.contains("\\n"), "{msg}");
            assert!(
                msg.len() < crate::LOG_CLIP_BYTES + 128,
                "{} bytes: {msg}",
                msg.len()
            );
        }

        /// Reasoning ate the budget: no content, `finish_reason: "length"`.
        /// The stop reason must reach the error, as it does for the judge.
        #[tokio::test]
        async fn empty_completion_names_the_stop_reason() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                    "choices": [{"message": {"content": null}, "finish_reason": "length"}]
                })))
                .mount(&server)
                .await;
            let e = client(&server).summarize("x").await.unwrap_err();
            assert!(matches!(e, AlayaError::Summary(_)), "{e:?}");
            assert!(e.to_string().contains(r#"stop_reason="length""#), "{e}");
        }
    }
}
