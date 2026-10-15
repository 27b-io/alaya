//! SummaryClient — SummaryProvider implementation over the shared LLM
//! transport (Anthropic Messages API or OpenAI-compatible chat completions).

use async_trait::async_trait;

use alaya_types::{AlayaError, Result};

use crate::anthropic::{DEFAULT_REQUEST_TIMEOUT, MessagesTransport, Prompt, truncate_chars};
use crate::{Provider, SummaryProvider};

const SYSTEM_PROMPT: &str = "Summarize the following in one concise sentence of approximately 50 tokens. \
     Return only the summary, no preamble.";

/// Output budget per wire. A cut-off completion is still stored as the
/// summary, so the cap must clear the longest summary plus any thinking that
/// precedes it: at 100, Haiku 4.5 hit `max_tokens` on 15-18% of a 300-memory
/// sample (at 500 all finished, the longest at 195 tokens). `claude-haiku-5-5`
/// thinks by default: at 500 it still came back empty or cut off on 35 of 274
/// memories, at 1024 on none, the largest reply using 894
/// (`scripts/summary_eval.py`). A reasoning model on the OpenAI wire spends
/// `max_completion_tokens` the same way (`finish_reason: "length"`, no
/// content). The cap is an upper bound, not a spend.
const fn max_output_tokens(provider: Provider) -> u32 {
    match provider {
        Provider::Anthropic => 1024,
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

    /// Both wires leave room for thinking or hidden reasoning before the
    /// summary.
    #[test]
    fn budget_per_wire() {
        assert_eq!(max_output_tokens(Provider::Anthropic), 1024);
        assert_eq!(max_output_tokens(Provider::OpenAi), 1024);
    }

    /// The anthropic wire sends the budget and no `thinking` field, so a
    /// model that thinks by default keeps doing so within the budget, and the
    /// text block after a thinking block is the summary.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn anthropic_wire_sends_the_budget_and_no_thinking_field() {
        use serde_json::json;
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/messages"))
            .and(body_json(json!({
                "model": "test-model",
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": "the memory"}],
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "content": [
                    {"type": "thinking", "thinking": "…", "signature": "s"},
                    {"type": "text", "text": " A summary. "},
                ],
                "stop_reason": "end_turn",
            })))
            .expect(1)
            .mount(&server)
            .await;
        let client = SummaryClient::new(
            Provider::Anthropic,
            server.uri(),
            "test-model".into(),
            Some("test-key".into()),
        )
        .unwrap();
        assert_eq!(client.summarize("the memory").await.unwrap(), "A summary.");
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
