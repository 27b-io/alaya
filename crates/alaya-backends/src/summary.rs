//! SummaryClient — SummaryProvider implementation (Anthropic Messages API).

use async_trait::async_trait;
use serde_json::json;

use alaya_types::{AlayaError, Result};

use crate::SummaryProvider;
use crate::anthropic::{MessagesTransport, truncate_chars};

const SYSTEM_PROMPT: &str = "Summarize the following in one concise sentence of approximately 50 tokens. \
     Return only the summary, no preamble.";

pub struct SummaryClient {
    transport: MessagesTransport,
    model: String,
}

impl SummaryClient {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let timeout = crate::anthropic::DEFAULT_REQUEST_TIMEOUT;
        #[cfg(target_arch = "wasm32")]
        let timeout = std::time::Duration::from_secs(30);
        Self {
            transport: MessagesTransport::new(base_url, api_key, timeout),
            model,
        }
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
