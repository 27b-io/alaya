//! Shared raw-HTTP transport for the Anthropic Messages API.
//!
//! Rust has no official Anthropic SDK; this thin `reqwest` wrapper is the
//! sanctioned shape in this repo. `SummaryClient` and `JudgeClient` both ride
//! it so the headers, the timeouts and the wasm32 cfg-gate live in one place.

use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

use alaya_types::{AlayaError, Result};

/// Request timeout for one-line summaries. (Ignored on wasm32, where
/// reqwest has no timeouts — the const still compiles there.)
pub(crate) const DEFAULT_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Maximum content length sent to the LLM per document. A one-line summary
/// or a pairwise verdict doesn't need the full document — the first ~4000
/// chars give sufficient signal without burning excess input tokens.
pub(crate) const MAX_CONTENT_CHARS: usize = 4000;

/// Truncate at a char boundary to avoid token waste on large memories.
pub(crate) fn truncate_chars(content: &str) -> &str {
    if content.len() > MAX_CONTENT_CHARS {
        &content[..content.floor_char_boundary(MAX_CONTENT_CHARS)]
    } else {
        content
    }
}

pub(crate) struct MessagesTransport {
    client: Client,
    base_url: String,
}

impl MessagesTransport {
    /// `request_timeout` only applies natively; reqwest-wasm has no timeouts.
    pub(crate) fn new(
        base_url: String,
        api_key: Option<String>,
        #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
        request_timeout: std::time::Duration,
    ) -> Self {
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
            .timeout(request_timeout);

        let client = builder.build().expect("failed to build reqwest client");

        Self { client, base_url }
    }

    /// POST `/v1/messages`. Failures are classified so callers can tell a
    /// per-request (deterministic) failure from one that has nothing to do
    /// with the request:
    /// - 429 → `AlayaError::RateLimited` (with the `retry-after` hint);
    /// - connect/timeout, 5xx, and auth/model misconfiguration
    ///   (401/403/404) → `AlayaError::Unavailable` (transient);
    /// - 400/413/422 and an unparseable body → `err(..)`, the caller's own
    ///   variant, meaning *this request* will fail the same way again.
    pub(crate) async fn messages(
        &self,
        body: &Value,
        err: fn(String) -> AlayaError,
    ) -> Result<MessagesResponse> {
        let url = format!("{}/v1/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(body)
            .send()
            .await
            .map_err(|e| AlayaError::Unavailable(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let retry_after_secs = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.trim().parse::<u64>().ok());
            return Err(AlayaError::RateLimited { retry_after_secs });
        }
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            let msg = format!("messages API returned {status}: {body}");
            return Err(if is_request_fault(status) {
                err(msg)
            } else {
                AlayaError::Unavailable(msg)
            });
        }

        resp.json::<MessagesResponse>()
            .await
            .map_err(|e| err(format!("failed to parse response: {e}")))
    }
}

/// Statuses that mean the request itself is unacceptable and will be again:
/// 400 invalid_request, 413 request_too_large, 422 unprocessable. Everything
/// else non-2xx (401/403 bad key, 404 bad model, 408/409, 5xx, 529) is not
/// the request's fault.
fn is_request_fault(status: reqwest::StatusCode) -> bool {
    matches!(status.as_u16(), 400 | 413 | 422)
}

// ─── Response types ────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub(crate) struct MessagesResponse {
    pub(crate) content: Vec<ContentBlock>,
    #[serde(default)]
    pub(crate) usage: Usage,
    /// Kept for diagnostics: an empty text response with
    /// `stop_reason = "max_tokens"` means thinking ate the output budget.
    #[serde(default)]
    pub(crate) stop_reason: Option<String>,
}

impl MessagesResponse {
    /// First non-empty text block, trimmed. Non-text blocks (e.g. thinking)
    /// deserialize with an empty `text` and are skipped.
    pub(crate) fn first_text(&self) -> Option<String> {
        self.content
            .iter()
            .map(|block| block.text.trim())
            .find(|t| !t.is_empty())
            .map(str::to_string)
    }
}

#[derive(Deserialize)]
pub(crate) struct ContentBlock {
    #[serde(default)]
    pub(crate) text: String,
}

/// Token usage. `input_tokens` on the wire counts only uncached input; the
/// cache fields carry the rest when a proxy or the API applies prompt
/// caching, so spend accounting must sum all three.
#[derive(Deserialize, Default, Clone, Copy, Debug)]
pub(crate) struct Usage {
    #[serde(default)]
    pub(crate) input_tokens: u64,
    #[serde(default)]
    pub(crate) cache_creation_input_tokens: u64,
    #[serde(default)]
    pub(crate) cache_read_input_tokens: u64,
    #[serde(default)]
    pub(crate) output_tokens: u64,
}

impl Usage {
    /// Every billable input token, cached or not (list-price upper bound).
    pub(crate) fn total_input_tokens(&self) -> u64 {
        self.input_tokens + self.cache_creation_input_tokens + self.cache_read_input_tokens
    }
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
        assert_eq!(
            parsed.first_text().as_deref(),
            Some("A concise summary of the content.")
        );
        assert_eq!(parsed.usage.input_tokens, 50);
        assert_eq!(parsed.usage.output_tokens, 10);
    }

    #[test]
    fn request_fault_statuses_are_the_deterministic_ones() {
        use reqwest::StatusCode as S;
        for s in [
            S::BAD_REQUEST,
            S::PAYLOAD_TOO_LARGE,
            S::UNPROCESSABLE_ENTITY,
        ] {
            assert!(is_request_fault(s), "{s}");
        }
        for s in [
            S::UNAUTHORIZED,
            S::FORBIDDEN,
            S::NOT_FOUND,
            S::REQUEST_TIMEOUT,
            S::INTERNAL_SERVER_ERROR,
            S::BAD_GATEWAY,
            S::SERVICE_UNAVAILABLE,
        ] {
            assert!(!is_request_fault(s), "{s}");
        }
    }

    #[test]
    fn usage_sums_cached_input_classes() {
        let json = r#"{"content":[{"type":"text","text":"x"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"cache_creation_input_tokens":2000,"cache_read_input_tokens":300,"output_tokens":80}}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.usage.total_input_tokens(), 2301);
        assert_eq!(parsed.stop_reason.as_deref(), Some("end_turn"));
    }

    #[test]
    fn parse_messages_response_minimal() {
        let json = r#"{"content":[{"type":"text","text":"Summary."}]}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.first_text().as_deref(), Some("Summary."));
        assert_eq!(parsed.usage.input_tokens, 0);
    }

    #[test]
    fn empty_content_blocks() {
        let json = r#"{"content":[]}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert!(parsed.first_text().is_none());
    }

    #[test]
    fn non_text_blocks_are_skipped() {
        let json = r#"{"content":[{"type":"thinking","thinking":""},{"type":"text","text":" {\"a\":1} "}]}"#;
        let parsed: MessagesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.first_text().as_deref(), Some("{\"a\":1}"));
    }

    #[test]
    fn truncate_respects_char_boundary() {
        let s = "é".repeat(MAX_CONTENT_CHARS);
        let t = truncate_chars(&s);
        assert!(t.len() <= MAX_CONTENT_CHARS);
        assert!(t.chars().all(|c| c == 'é'));
        assert_eq!(truncate_chars("short"), "short");
    }
}
