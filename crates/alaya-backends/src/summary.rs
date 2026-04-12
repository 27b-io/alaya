//! SummaryClient — SummaryProvider implementation (Anthropic Messages API).

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use alaya_types::{AlayaError, Result};

use crate::SummaryProvider;

const SYSTEM_PROMPT: &str = "Summarize the following in one concise sentence of approximately 50 tokens. \
     Return only the summary, no preamble.";

pub struct SummaryClient {
    client: Client,
    base_url: String,
    model: String,
}

impl SummaryClient {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = api_key {
            headers.insert(
                "x-api-key",
                reqwest::header::HeaderValue::from_str(&key).expect("invalid API key characters"),
            );
        }
        headers.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static("2023-06-01"),
        );

        let builder = Client::builder().default_headers(headers);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .http1_only()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30));

        let client = builder.build().expect("failed to build reqwest client");

        Self {
            client,
            base_url,
            model,
        }
    }
}

/// Maximum content length sent to the LLM. A one-line summary doesn't
/// need the full document — first ~4000 chars give sufficient signal
/// without burning excess input tokens.
const MAX_CONTENT_CHARS: usize = 4000;

#[async_trait(?Send)]
impl SummaryProvider for SummaryClient {
    #[tracing::instrument(skip(self, content), fields(content_len = content.len()))]
    async fn summarize(&self, content: &str) -> Result<String> {
        let url = format!("{}/v1/messages", self.base_url);

        // Truncate at a char boundary to avoid token waste on large memories
        let truncated = if content.len() > MAX_CONTENT_CHARS {
            &content[..content.floor_char_boundary(MAX_CONTENT_CHARS)]
        } else {
            content
        };

        let body = json!({
            "model": self.model,
            "max_tokens": 100,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": truncated}],
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Summary(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            return Err(AlayaError::Summary(format!(
                "messages API returned {status}: {body}"
            )));
        }

        let parsed: MessagesResponse = resp
            .json()
            .await
            .map_err(|e| AlayaError::Summary(format!("failed to parse response: {e}")))?;

        parsed
            .content
            .first()
            .map(|block| block.text.trim().to_string())
            .filter(|s| !s.is_empty())
            .ok_or_else(|| AlayaError::Summary("empty response from messages API".into()))
    }
}

// ─── Response types ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct MessagesResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    text: String,
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_messages_response() {
        let json = r#"{"id":"msg_01","type":"message","role":"assistant","content":[{"type":"text","text":"A concise summary of the content."}],"model":"claude-haiku-4-5-20251001","stop_reason":"end_turn","usage":{"input_tokens":50,"output_tokens":10}}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.content.len(), 1);
        assert_eq!(parsed.content[0].text, "A concise summary of the content.");
    }

    #[test]
    fn parse_messages_response_minimal() {
        let json = r#"{"content":[{"type":"text","text":"Summary."}]}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.content[0].text, "Summary.");
    }

    #[test]
    fn empty_content_blocks() {
        let json = r#"{"content":[]}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.content.is_empty());
    }
}
