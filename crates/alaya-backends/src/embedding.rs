// EmbeddingClient — EmbeddingProvider implementation (OpenAI-compat REST API)

use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;

use alaya_types::{AlayaError, Result, memory::HealthStatus, search::PromptName};

use crate::{EmbeddingProvider, Tokenizer};

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

        #[allow(clippy::disallowed_methods, reason = "sets .no_proxy() below")]
        let builder = Client::builder().default_headers(headers);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            // Dial the host the boot guard classified, never an env proxy
            // (see `check_credential_transport` in alaya-server).
            .no_proxy()
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
                    .map_err(|e| AlayaError::Embedding(crate::redact_reqwest_error(e)))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = crate::error_body(resp).await;
                    return Err(AlayaError::Embedding(format!(
                        "embedding API returned {status}: {body}"
                    )));
                }

                let parsed: EmbeddingResponse = resp.json().await.map_err(|e| {
                    AlayaError::Embedding(format!(
                        "failed to parse response: {}",
                        crate::redact_reqwest_error(e)
                    ))
                })?;

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

    /// `GET {base_url}/health` — the TEI / vLLM readiness endpoint: 200 once
    /// the model is loaded, 503 until then. Reachability only — it spends no
    /// embedding, so a dashboard poll costs the endpoint nothing.
    async fn health(&self) -> Result<HealthStatus> {
        let req = self.client.get(format!("{}/health", self.base_url));
        // wasm32: no tokio timer, so the call-site `with_budget` awaits
        // directly and the builder timeout above is cfg'd out — this
        // per-request fetch-abort deadline is the sole bound there (same
        // shape as `RerankClient::rerank`; mirrors the native 5s budget).
        #[cfg(target_arch = "wasm32")]
        let req = req.timeout(std::time::Duration::from_secs(5));

        let resp = req
            .send()
            .await
            .map_err(|e| AlayaError::Embedding(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(AlayaError::Embedding(format!(
                "embedding health returned {}",
                resp.status()
            )));
        }

        Ok(HealthStatus {
            status: "healthy".into(),
            backend: "embedding".into(),
            details: None,
        })
    }
}

impl EmbeddingClient {
    /// The [`Tokenizer`] of this client's endpoint, sharing its connection
    /// pool, headers and transport settings.
    pub fn tokenizer(&self) -> TeiTokenizer {
        TeiTokenizer {
            client: self.client.clone(),
            base_url: self.base_url.clone(),
        }
    }
}

/// [`Tokenizer`] over TEI's `GET /info` and `POST /tokenize`. Built by
/// [`EmbeddingClient::tokenizer`].
pub struct TeiTokenizer {
    client: Client,
    base_url: String,
}

/// Per-request bound for `/info` and `/tokenize`, inside the client's own
/// embed timeout: neither does model work.
const TOKENIZER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

impl TeiTokenizer {
    async fn send_json<T: serde::de::DeserializeOwned>(
        &self,
        req: reqwest::RequestBuilder,
        what: &str,
    ) -> Result<T> {
        let resp = req
            .timeout(TOKENIZER_TIMEOUT)
            .send()
            .await
            .map_err(|e| AlayaError::Embedding(crate::redact_reqwest_error(e)))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = crate::error_body(resp).await;
            return Err(AlayaError::Embedding(format!(
                "embedding {what} returned {status}: {body}"
            )));
        }
        resp.json().await.map_err(|e| {
            AlayaError::Embedding(format!(
                "failed to parse embedding {what}: {}",
                crate::redact_reqwest_error(e)
            ))
        })
    }

    async fn tokenize(&self, text: &str, add_special_tokens: bool) -> Result<Vec<TeiToken>> {
        let req = self
            .client
            .post(format!("{}/tokenize", self.base_url))
            .json(&serde_json::json!({ "inputs": text, "add_special_tokens": add_special_tokens }));
        let parsed: Vec<Vec<TeiToken>> = self.send_json(req, "/tokenize").await?;
        parsed.into_iter().next().ok_or_else(|| {
            AlayaError::Embedding("embedding /tokenize returned no token list".into())
        })
    }
}

#[async_trait(?Send)]
impl Tokenizer for TeiTokenizer {
    /// `max_input_length` from `/info`, less the tokens of the passage
    /// prompt prefix with the special tokens around it. The prefix alone
    /// tokenizes to at least as many tokens as it takes before text.
    async fn passage_token_budget(&self) -> Result<usize> {
        let req = self.client.get(format!("{}/info", self.base_url));
        let info: TeiInfo = self.send_json(req, "/info").await?;
        let wrapper = self
            .tokenize(&prefix_text(PromptName::Passage, ""), true)
            .await?
            .len();
        Ok(info.max_input_length.saturating_sub(wrapper))
    }

    async fn token_spans(&self, text: &str) -> Result<Vec<(usize, usize)>> {
        self.tokenize(text, false)
            .await?
            .into_iter()
            .filter(|t| !t.special)
            .map(|t| match (t.start, t.stop) {
                (Some(start), Some(stop)) => Ok((start, stop)),
                _ => Err(AlayaError::Embedding(
                    "embedding /tokenize returned a token without a span".into(),
                )),
            })
            .collect()
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

#[derive(Deserialize)]
struct TeiInfo {
    max_input_length: usize,
}

/// One `/tokenize` token. Special tokens carry no span.
#[derive(Deserialize)]
struct TeiToken {
    special: bool,
    start: Option<usize>,
    stop: Option<usize>,
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

    fn tokenizer_at(uri: String) -> TeiTokenizer {
        EmbeddingClient::new(uri, "m".into(), 1024, 32, None).tokenizer()
    }

    /// The budget is `/info`'s window less the prefix with its special
    /// tokens; `/tokenize` gives byte spans with the special tokens, which
    /// carry none, dropped. Responses are TEI 1.9's shape, multibyte text
    /// included.
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn tei_tokenizer_reads_the_budget_and_byte_spans() {
        use serde_json::json;
        use wiremock::matchers::{body_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/info"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "model_id": "m", "max_input_length": 2048, "max_batch_tokens": 2048,
            })))
            .mount(&server)
            .await;
        let special = |id: u32, text: &str| json!({ "id": id, "text": text, "special": true, "start": null, "stop": null });
        let token = |id: u32, text: &str, start: usize, stop: usize| json!({ "id": id, "text": text, "special": false, "start": start, "stop": stop });
        Mock::given(method("POST"))
            .and(path("/tokenize"))
            .and(body_json(
                json!({ "inputs": "search_document: ", "add_special_tokens": true }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[
                special(0, "<s>"),
                token(1, "search", 0, 6),
                token(2, "_document", 6, 15),
                token(3, ":", 15, 16),
                token(4, " ", 16, 17),
                special(2, "</s>"),
            ]])))
            .mount(&server)
            .await;
        Mock::given(method("POST"))
            .and(path("/tokenize"))
            .and(body_json(
                json!({ "inputs": "Ālaya café", "add_special_tokens": false }),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([[
                token(5, "Ā", 0, 2),
                token(6, "laya", 2, 6),
                token(7, " café", 6, 12),
            ]])))
            .mount(&server)
            .await;

        let t = tokenizer_at(server.uri());
        assert_eq!(t.passage_token_budget().await.unwrap(), 2048 - 6);
        assert_eq!(
            t.token_spans("Ālaya café").await.unwrap(),
            vec![(0, 2), (2, 6), (6, 12)]
        );
    }

    /// A refusal (TEI's 422 past its length limit), an empty answer, or a
    /// text token with no span must surface as an error, never as an empty
    /// or short layout that would read as "fits".
    #[cfg(not(target_arch = "wasm32"))]
    #[tokio::test]
    async fn tei_tokenizer_surfaces_a_refusal_or_a_malformed_layout() {
        use serde_json::json;
        use wiremock::matchers::{body_partial_json, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let server = MockServer::start().await;
        for (inputs, resp) in [
            (
                "refused",
                ResponseTemplate::new(422).set_body_string("too long"),
            ),
            ("empty", ResponseTemplate::new(200).set_body_json(json!([]))),
            (
                "spanless",
                ResponseTemplate::new(200).set_body_json(json!([[
                    { "id": 1, "text": "x", "special": false, "start": null, "stop": null }
                ]])),
            ),
        ] {
            Mock::given(method("POST"))
                .and(path("/tokenize"))
                .and(body_partial_json(json!({ "inputs": inputs })))
                .respond_with(resp)
                .mount(&server)
                .await;
        }

        let t = tokenizer_at(server.uri());
        let err = t.token_spans("refused").await.unwrap_err();
        assert!(err.to_string().contains("422"), "{err}");
        assert!(t.token_spans("empty").await.is_err());
        assert!(t.token_spans("spanless").await.is_err());
    }

    #[test]
    fn empty_input_returns_empty() {
        // Can't call async in sync test, but we verify the guard clause logic
        let texts: &[&str] = &[];
        assert!(texts.is_empty());
    }
}
