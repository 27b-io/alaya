mod anthropic;
pub mod embedding;
pub mod graph;
pub mod graph_ref;
pub mod judge;
pub mod qdrant;
pub mod rerank;
pub mod summary;
pub mod traits;

pub use traits::*;

pub(crate) fn redact_reqwest_error(mut error: reqwest::Error) -> String {
    if let Some(url) = error.url_mut() {
        url.set_query(None);
        url.set_fragment(None);
    }
    error.to_string()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    #[tokio::test]
    async fn reqwest_error_redacts_query() {
        let error = reqwest::Client::new()
            .get("http://127.0.0.1:1/?api_key=SECRET#fragment")
            .send()
            .await
            .unwrap_err();
        let message = super::redact_reqwest_error(error);
        assert!(message.contains("127.0.0.1"), "url dropped: {message}");
        assert!(!message.contains("SECRET"), "query leaked: {message}");
        assert!(!message.contains("api_key"), "query leaked: {message}");
        assert!(!message.contains("fragment"), "fragment leaked: {message}");
    }
}
