// RerankClient — RerankingService implementation (TEI `/rerank` endpoint)

use std::time::Duration;

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result};

use crate::RerankingService;

/// TEI `/rerank`-backed [`RerankingService`].
///
/// On native, `rerank()` sets no client-side timer and is unbounded on its
/// own — see the call-site comment there for why a second timer would steal
/// the wrong log line on a late poll. Every call MUST be wrapped by the
/// caller in `tokio::time::timeout(self.timeout(), …)`; see
/// [`RerankingService::timeout`] for the budget. wasm32 has no tokio timer
/// and bounds its own transport instead.
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

        // No client-side timers on native (see `rerank()`): a connect_timeout
        // at or below the budget fires in the same tick as the call-site
        // tokio timer on a blackholed connect and steals its log line, and
        // the call-site timer bounds the connect phase anyway.
        let client = Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| {
                AlayaError::Config(format!(
                    "rerank HTTP client: {}",
                    crate::redact_reqwest_error(e)
                ))
            })?;

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

        let req = self.client.post(url.as_str()).json(&body);
        // Native deliberately sets NO client-side timer: the call-site
        // `tokio::time::timeout` in service.rs is the bound. It polls this
        // future before its own deadline, so a response that has already
        // arrived is used even on a late poll, and dropping the future on
        // elapse aborts the request and closes its connection. A second
        // timer inside reqwest is checked *before* the socket
        // (`PendingRequest::poll`), so on a late poll it would discard a
        // completed response and report real errors as timeouts. wasm32 has
        // no tokio timer, so there this per-request deadline (a fetch abort
        // timer) is the sole bound.
        #[cfg(target_arch = "wasm32")]
        let req = req.timeout(self.timeout);

        let resp = req
            .send()
            .await
            .map_err(|e| AlayaError::Rerank(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
            return Err(AlayaError::Rerank(format!(
                "rerank API returned {status}: {body}"
            )));
        }

        let parsed: Vec<RerankItem> = resp.json().await.map_err(|e| {
            AlayaError::Rerank(format!(
                "failed to parse response: {}",
                crate::redact_reqwest_error(e)
            ))
        })?;

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
