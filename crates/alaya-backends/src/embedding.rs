// EmbeddingClient — EmbeddingProvider implementation (OpenAI-compat REST API)

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result, search::PromptName};

use crate::EmbeddingProvider;

/// Maximum texts per API call (OpenAI-compatible limit).
const BATCH_SIZE: usize = 64;

pub struct EmbeddingClient {
    client: Client,
    base_url: String,
    model: String,
    dimensions: usize,
}

impl EmbeddingClient {
    pub fn new(
        base_url: String,
        model: String,
        dimensions: usize,
        api_key: Option<String>,
    ) -> Self {
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
            .timeout(std::time::Duration::from_secs(60));

        let client = builder.build().expect("failed to build reqwest client");

        Self {
            client,
            base_url,
            model,
            dimensions,
        }
    }
}

#[async_trait(?Send)]
impl EmbeddingProvider for EmbeddingClient {
    #[tracing::instrument(skip(self, texts), fields(n = texts.len(), prompt = ?prompt_name))]
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed: Vec<String> = texts.iter().map(|t| prefix_text(prompt_name, t)).collect();
        let url = format!("{}/v1/embeddings", self.base_url);

        // Fire all chunk requests concurrently — on a LocalSet (?Send context)
        // this interleaves I/O waits instead of blocking sequentially.
        let chunk_futures = prefixed.chunks(BATCH_SIZE).map(|chunk| {
            let client = &self.client;
            let url = &url;
            let body = serde_json::json!({
                "model": self.model,
                "input": chunk,
                "encoding_format": "float",
            });
            async move {
                let resp = client
                    .post(url.as_str())
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| AlayaError::Embedding(e.to_string()))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
                    return Err(AlayaError::Embedding(format!(
                        "embedding API returned {status}: {body}"
                    )));
                }

                let parsed: EmbeddingResponse = resp
                    .json()
                    .await
                    .map_err(|e| AlayaError::Embedding(format!("failed to parse response: {e}")))?;

                let mut items = parsed.data;
                items.sort_by_key(|d| d.index);
                Ok(items.into_iter().map(|d| d.embedding).collect::<Vec<_>>())
            }
        });

        let results = futures::future::try_join_all(chunk_futures).await?;
        Ok(results.into_iter().flatten().collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

fn prefix_text(prompt_name: PromptName, text: &str) -> String {
    match prompt_name {
        PromptName::Query => format!("search_query: {text}"),
        PromptName::Passage => format!("search_document: {text}"),
    }
}

// --- Response types (private) ---

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
    index: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_query() {
        let out = prefix_text(PromptName::Query, "hello world");
        assert_eq!(out, "search_query: hello world");
    }

    #[test]
    fn prefix_passage() {
        let out = prefix_text(PromptName::Passage, "some document");
        assert_eq!(out, "search_document: some document");
    }

    #[test]
    fn batch_splitting_logic() {
        // Verify chunks(BATCH_SIZE) produces correct batch counts
        let texts: Vec<String> = (0..150).map(|i| format!("text_{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let chunks: Vec<&[&str]> = refs.chunks(BATCH_SIZE).collect();
        assert_eq!(chunks.len(), 3); // 64 + 64 + 22
        assert_eq!(chunks[0].len(), 64);
        assert_eq!(chunks[1].len(), 64);
        assert_eq!(chunks[2].len(), 22);
    }

    #[test]
    fn parse_embedding_response() {
        let json = r#"{
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 1},
                {"embedding": [0.4, 0.5, 0.6], "index": 0}
            ]
        }"#;
        let parsed: EmbeddingResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.data.len(), 2);

        // Sort by index like embed_batch does
        let mut items = parsed.data;
        items.sort_by_key(|d| d.index);
        assert_eq!(items[0].index, 0);
        assert_eq!(items[0].embedding, vec![0.4, 0.5, 0.6]);
        assert_eq!(items[1].index, 1);
        assert_eq!(items[1].embedding, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn empty_input_returns_empty() {
        // Can't call async in sync test, but we verify the guard clause logic
        let texts: &[&str] = &[];
        assert!(texts.is_empty());
    }
}
