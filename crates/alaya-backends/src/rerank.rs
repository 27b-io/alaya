// RerankClient — RerankingService implementation (TEI `/rerank` endpoint)
// HostedRerankClient — RerankingService over the Cohere-family wire shape
//   (Voyage / Cohere / Jina / LiteLLM `/rerank`). Throwaway A/B adapter for
//   LAB-3831; not for production without the expert-panel gate.

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result};

use crate::RerankingService;

/// Shared reqwest client: optional bearer auth, 5 s connect / 30 s total.
fn build_client(api_key: Option<&str>) -> Client {
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

    builder.build().expect("failed to build reqwest client")
}

/// Vendors return items sorted by score desc, each tagged with its input
/// index. Remap to input order; reject short/long/duplicate/out-of-range
/// results rather than silently zero-filling or overwriting.
fn remap_scores(items: impl IntoIterator<Item = (usize, f32)>, n: usize) -> Result<Vec<f32>> {
    let mut scores = vec![0.0_f32; n];
    let mut seen = vec![false; n];
    let mut count = 0usize;
    for (index, score) in items {
        count += 1;
        if index >= n {
            return Err(AlayaError::Rerank(format!(
                "rerank index {index} out of range for {n} texts"
            )));
        }
        if seen[index] {
            return Err(AlayaError::Rerank(format!(
                "rerank returned duplicate index {index} (would silently overwrite)"
            )));
        }
        seen[index] = true;
        scores[index] = score;
    }
    if count != n {
        return Err(AlayaError::Rerank(format!(
            "rerank returned {count} scores for {n} texts"
        )));
    }
    Ok(scores)
}

async fn check_status(resp: reqwest::Response) -> Result<reqwest::Response> {
    if resp.status().is_success() {
        return Ok(resp);
    }
    let status = resp.status();
    let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
    Err(AlayaError::Rerank(format!(
        "rerank API returned {status}: {body}"
    )))
}

pub struct RerankClient {
    client: Client,
    base_url: String,
    top_n: usize,
}

impl RerankClient {
    pub fn new(base_url: String, top_n: usize, api_key: Option<String>) -> Self {
        Self {
            client: build_client(api_key.as_deref()),
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
        let resp = check_status(resp).await?;

        let parsed: Vec<RerankItem> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Rerank(format!("failed to parse response: {e}")))?;

        remap_scores(parsed.into_iter().map(|i| (i.index, i.score)), texts.len())
    }

    fn top_n(&self) -> usize {
        self.top_n
    }
}

/// Hosted reranker over the Cohere-family shape:
///   POST {url}  {query, documents, model}  ->  {data|results: [{index, relevance_score}]}
/// `url` is the full endpoint (e.g. `https://api.voyageai.com/v1/rerank`).
/// No `top_k`/`top_n` is sent: Ālaya needs a score for every candidate.
pub struct HostedRerankClient {
    client: Client,
    url: String,
    model: String,
    top_n: usize,
}

impl HostedRerankClient {
    pub fn new(url: String, model: String, top_n: usize, api_key: &str) -> Self {
        Self {
            client: build_client(Some(api_key)),
            url,
            model,
            top_n,
        }
    }
}

#[async_trait(?Send)]
impl RerankingService for HostedRerankClient {
    #[tracing::instrument(skip(self, texts), fields(n = texts.len(), model = %self.model))]
    async fn rerank(&self, query: &str, texts: &[&str]) -> Result<Vec<f32>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let body = serde_json::json!({
            "query": query,
            "documents": texts,
            "model": self.model,
        });

        let resp = self
            .client
            .post(self.url.as_str())
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Rerank(e.to_string()))?;
        let resp = check_status(resp).await?;

        let parsed: HostedResponse = resp
            .json()
            .await
            .map_err(|e| AlayaError::Rerank(format!("failed to parse response: {e}")))?;

        remap_scores(
            parsed
                .data
                .into_iter()
                .map(|i| (i.index, i.relevance_score)),
            texts.len(),
        )
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

#[derive(Deserialize)]
struct HostedResponse {
    /// Voyage uses `data`; Cohere / Jina / LiteLLM use `results`.
    #[serde(alias = "results")]
    data: Vec<HostedItem>,
}

#[derive(Deserialize)]
struct HostedItem {
    index: usize,
    relevance_score: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_reorders_by_index() {
        let scores = remap_scores([(2, 0.9), (0, 0.5), (1, 0.1)], 3).unwrap();
        assert_eq!(scores, vec![0.5, 0.1, 0.9]);
    }

    #[test]
    fn remap_rejects_duplicate_index() {
        assert!(remap_scores([(1, 0.5), (1, 0.9)], 2).is_err());
    }

    #[test]
    fn remap_rejects_out_of_range_and_short() {
        assert!(remap_scores([(3, 0.5)], 3).is_err());
        assert!(remap_scores([(0, 0.5)], 2).is_err());
    }

    #[test]
    fn top_n_is_returned() {
        let client = RerankClient::new("http://localhost:8089".to_string(), 20, None);
        assert_eq!(client.top_n(), 20);
        let hosted =
            HostedRerankClient::new("http://localhost:8089/v1/rerank".into(), "m".into(), 7, "k");
        assert_eq!(hosted.top_n(), 7);
    }

    #[test]
    fn hosted_response_accepts_voyage_and_cohere_keys() {
        let voyage: HostedResponse =
            serde_json::from_str(r#"{"object":"list","data":[{"index":1,"relevance_score":0.4}],"usage":{"total_tokens":8}}"#)
                .unwrap();
        let cohere: HostedResponse =
            serde_json::from_str(r#"{"results":[{"index":1,"relevance_score":0.4}]}"#).unwrap();
        assert_eq!(voyage.data[0].index, 1);
        assert_eq!(cohere.data[0].relevance_score, 0.4);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod wire_tests {
    use super::*;
    use wiremock::matchers::{body_partial_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn hosted_client_sends_cohere_family_body_and_remaps() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/rerank"))
            .and(header("authorization", "Bearer test-key"))
            .and(body_partial_json(serde_json::json!({
                "query": "q",
                "documents": ["a", "b", "c"],
                "model": "rerank-3-lite",
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "object": "list",
                "data": [
                    {"index": 2, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.5},
                    {"index": 1, "relevance_score": 0.1}
                ],
                "model": "rerank-3-lite",
                "usage": {"total_tokens": 8}
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = HostedRerankClient::new(
            format!("{}/v1/rerank", server.uri()),
            "rerank-3-lite".into(),
            20,
            "test-key",
        );
        let scores = client.rerank("q", &["a", "b", "c"]).await.unwrap();
        assert_eq!(scores, vec![0.5, 0.1, 0.9]);
    }

    #[tokio::test]
    async fn hosted_client_surfaces_http_errors() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;
        let client = HostedRerankClient::new(server.uri(), "m".into(), 20, "k");
        let err = client.rerank("q", &["a"]).await.unwrap_err().to_string();
        assert!(err.contains("429"), "{err}");
    }
}
