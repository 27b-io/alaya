// RerankClient — RerankingService implementation (TEI `/rerank` endpoint)

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result};

use crate::RerankingService;

pub struct RerankClient {
    client: Client,
    base_url: String,
    top_n: usize,
}

impl RerankClient {
    pub fn new(base_url: String, top_n: usize, api_key: Option<String>) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(key) = api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                    .expect("invalid API key characters"),
            );
        }

        let builder = Client::builder().default_headers(headers);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30));

        let client = builder.build().expect("failed to build reqwest client");

        Self {
            client,
            base_url,
            top_n,
        }
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

        let resp = self
            .client
            .post(url.as_str())
            .json(&body)
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
        let client = RerankClient::new("http://localhost:8089".to_string(), 20, None);
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
