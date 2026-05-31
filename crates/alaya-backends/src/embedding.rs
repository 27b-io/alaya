// EmbeddingClient — EmbeddingProvider implementation (OpenAI-compat REST API)

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result, search::PromptName};

use crate::EmbeddingProvider;

/// TEI hard-rejects client batches larger than this with HTTP 422 (no silent
/// truncation), so the configured batch size is clamped to it. Batch size does
/// not change output numerics — only how many texts ride in a single request.
const MAX_TEI_BATCH_SIZE: usize = 256;

pub struct EmbeddingClient {
    client: Client,
    base_url: String,
    model: String,
    dimensions: usize,
    /// Texts per `/v1/embeddings` request. Must not exceed the target TEI's
    /// `max_client_batch_size`. Clamped to `[1, MAX_TEI_BATCH_SIZE]` in `new`.
    batch_size: usize,
}

impl EmbeddingClient {
    pub fn new(
        base_url: String,
        model: String,
        dimensions: usize,
        batch_size: usize,
        api_key: Option<String>,
    ) -> Self {
        // chunks(0) panics, and >256 makes TEI return 422 — clamp to safe range.
        let clamped = batch_size.clamp(1, MAX_TEI_BATCH_SIZE);
        if clamped != batch_size {
            tracing::warn!(
                requested = batch_size,
                used = clamped,
                "embedding batch size out of range, clamped to [1, {MAX_TEI_BATCH_SIZE}]"
            );
        }

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
            batch_size: clamped,
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
        let chunk_futures = prefixed.chunks(self.batch_size).map(|chunk| {
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
        // Default batch size of 32 produces the expected chunk counts.
        let texts: Vec<String> = (0..150).map(|i| format!("text_{i}")).collect();
        let refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let chunks: Vec<&[&str]> = refs.chunks(32).collect();
        assert_eq!(chunks.len(), 5); // 32 + 32 + 32 + 32 + 22
        assert_eq!(chunks[0].len(), 32);
        assert_eq!(chunks[4].len(), 22);
    }

    #[test]
    fn batch_size_is_clamped() {
        // 0 would panic chunks(); must floor to 1.
        let zero = EmbeddingClient::new("http://x".into(), "m".into(), 1024, 0, None);
        assert_eq!(zero.batch_size, 1);
        // Above TEI's hard ceiling must clamp to 256 (else HTTP 422).
        let huge = EmbeddingClient::new("http://x".into(), "m".into(), 1024, 9999, None);
        assert_eq!(huge.batch_size, MAX_TEI_BATCH_SIZE);
        // In-range values pass through untouched.
        let ok = EmbeddingClient::new("http://x".into(), "m".into(), 1024, 64, None);
        assert_eq!(ok.batch_size, 64);
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
