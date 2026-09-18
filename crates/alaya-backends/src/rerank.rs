// RerankClient — RerankingService implementation (TEI `/rerank` endpoint)

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result};

use crate::RerankingService;

pub struct RerankClient {
    client: Client,
    base_url: String,
    top_n: usize,
    timeout: Duration,
}

impl RerankClient {
    /// Fails with `Config` on bearer material that is not a valid header
    /// value (e.g. a trailing newline, #97) or on a client build error. The
    /// error must never echo `api_key`: `InvalidHeaderValue` carries no
    /// payload and neither message below interpolates the key.
    pub fn new(
        base_url: String,
        top_n: usize,
        api_key: Option<String>,
        timeout: Duration,
    ) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = api_key {
            let val =
                reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")).map_err(|e| {
                    AlayaError::Config(format!(
                        "rerank api key is not valid HTTP header material: {e}"
                    ))
                })?;
            headers.insert(reqwest::header::AUTHORIZATION, val);
        }

        // No connect_timeout: any value <= the budget fires in the same tick
        // as the call-site tokio timer on a blackholed connect and steals its
        // log line. The per-request total timeout in `rerank()` bounds the
        // connect phase too.
        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| AlayaError::Config(format!("rerank HTTP client: {e}")))?;

        Ok(Self {
            client,
            base_url,
            top_n,
            timeout,
        })
    }
}

#[async_trait(?Send)]
impl RerankingService for RerankClient {
    #[tracing::instrument(skip(self, texts), fields(n = texts.len()))]
    async fn rerank(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/rerank", self.base_url);
        let body = serde_json::json!({
            "query": query,
            "texts": texts,
            "truncate": true,
            "raw_scores": false,
        });

        // Native: +1s margin so the call-site `tokio::time::timeout` in
        // service.rs fires first and logs "rerank timed out" — the ticket's
        // post-deploy check greps for that line; reqwest is the backstop that
        // releases the socket. wasm32 has no tokio timer, so this per-request
        // timeout (a fetch abort timer) is the sole bound — no margin there.
        #[cfg(not(target_arch = "wasm32"))]
        let deadline = self.timeout + Duration::from_secs(1);
        #[cfg(target_arch = "wasm32")]
        let deadline = self.timeout;

        let resp = self
            .client
            .post(url.as_str())
            .json(&body)
            .timeout(deadline)
            .send()
            .await
            .map_err(|e| AlayaError::Rerank(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            return Err(AlayaError::Rerank(format!(
                "rerank API returned {status}: {body}"
            )));
        }

        let parsed: Vec<RerankItem> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Rerank(format!("failed to parse response: {e}")))?;

        // TEI returns items sorted by score desc; remap to input order.
        if parsed.len() != texts.len() {
            return Err(AlayaError::Rerank(format!(
                "rerank returned {} scores for {} texts",
                parsed.len(),
                texts.len()
            )));
        }

        let mut scores = vec![0.0_f32; texts.len()];
        let mut seen = vec![false; texts.len()];
        for item in parsed {
            if item.index >= texts.len() {
                return Err(AlayaError::Rerank(format!(
                    "rerank index {} out of range for {} texts",
                    item.index,
                    texts.len()
                )));
            }
            if seen[item.index] {
                return Err(AlayaError::Rerank(format!(
                    "rerank returned duplicate index {} (would silently overwrite)",
                    item.index
                )));
            }
            seen[item.index] = true;
            scores[item.index] = item.score;
        }
        Ok(scores)
    }

    fn top_n(&self) -> usize {
        self.top_n
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }
}

// --- Response types (private) ---

#[derive(Deserialize)]
struct RerankItem {
    index: usize,
    score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #97 repro shape: a control character in the bearer. Must be `Config`,
    /// and the message must not echo the key it is rejecting.
    #[test]
    fn new_rejects_control_chars_without_echoing_the_key() {
        let Err(err) = RerankClient::new(
            "http://tei".into(),
            20,
            Some("abc\n".into()),
            std::time::Duration::from_secs(5),
        ) else {
            panic!("a control character in the bearer must be rejected");
        };
        let msg = err.to_string();
        assert!(matches!(err, AlayaError::Config(_)), "{msg}");
        assert!(!msg.contains("abc"), "error echoed the key: {msg}");
    }

    #[test]
    fn parse_rerank_response_remaps_indices() {
        let json = r#"[
            {"index": 2, "score": 0.9},
            {"index": 0, "score": 0.5},
            {"index": 1, "score": 0.1}
        ]"#;
        let parsed: Vec<RerankItem> = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.len(), 3);

        // Mirror the remap logic in rerank()
        let mut scores = vec![0.0_f32; 3];
        for item in parsed {
            scores[item.index] = item.score;
        }
        assert_eq!(scores, vec![0.5, 0.1, 0.9]);
    }

    #[test]
    fn top_n_is_returned() {
        let client = RerankClient::new(
            "http://localhost:8089".to_string(),
            20,
            None,
            std::time::Duration::from_millis(5000),
        )
        .unwrap();
        assert_eq!(client.top_n(), 20);
    }

    #[test]
    fn duplicate_index_in_remap_is_rejected() {
        // Mirror the bounds + dedup check from the rerank() implementation.
        let n = 3;
        let mut scores = vec![0.0_f32; n];
        let mut seen = vec![false; n];
        let items = [(1usize, 0.5_f32), (1usize, 0.9_f32)];
        let mut got_dup_error = false;
        for (idx, score) in items {
            assert!(idx < n);
            if seen[idx] {
                got_dup_error = true;
                break;
            }
            seen[idx] = true;
            scores[idx] = score;
        }
        assert!(got_dup_error, "duplicate index must trigger an error path");
    }
}
