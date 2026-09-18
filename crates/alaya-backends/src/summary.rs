//! SummaryClient — SummaryProvider implementation (Anthropic Messages API).

use async_trait::async_trait;
use serde_json::json;

use alaya_types::{AlayaError, Result};

use crate::SummaryProvider;
use crate::anthropic::{DEFAULT_REQUEST_TIMEOUT, MessagesTransport, truncate_chars};

const SYSTEM_PROMPT: &str = "Summarize the following in one concise sentence of approximately 50 tokens. \
     Return only the summary, no preamble.";

pub struct SummaryClient {
    transport: MessagesTransport,
    model: String,
}

impl SummaryClient {
    /// Fails with `Config` on bearer material that is not a valid header
    /// value (e.g. a trailing newline, #97) or on a client build error; the
    /// error never echoes `api_key` (see `MessagesTransport::new`).
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Result<Self> {
        Ok(Self {
            transport: MessagesTransport::new(base_url, api_key, DEFAULT_REQUEST_TIMEOUT)?,
            model,
        })
    }
}

#[async_trait(?Send)]
impl SummaryProvider for SummaryClient {
    #[tracing::instrument(skip(self, content), fields(content_len = content.len()))]
    async fn summarize(&self, content: &str) -> Result<String> {
        let body = json!({
            "model": self.model,
            "max_tokens": 100,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": truncate_chars(content)}],
        });

        let parsed = self.transport.messages(&body, AlayaError::Summary).await?;

        parsed
            .first_text()
            .ok_or_else(|| AlayaError::Summary("empty response from messages API".into()))
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
}
