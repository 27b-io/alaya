//! Trusted-service HTTP client for alaya-server's REST API (D2: the console
//! backend holds the static bearer; the browser never sees it).
//!
//! Responses stay `serde_json::Value` — the console renders fields
//! defensively instead of mirroring another service's output types, so an
//! additive upstream change can't break the UI.

use serde_json::{Value, json};

use crate::error::AppError;

#[derive(Clone)]
pub struct AlayaClient {
    base: url::Url,
    bearer: String,
    http: reqwest::Client,
}

/// Extract a safe error string from an upstream error body.
fn upstream_error(status: reqwest::StatusCode, body: &str) -> AppError {
    let detail = serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v.get("error").and_then(|e| e.as_str()).map(String::from))
        .unwrap_or_else(|| "unrecognized error body".to_string());
    AppError::Upstream(format!("alaya-server {status}: {detail}"))
}

impl AlayaClient {
    pub fn new(base: url::Url, bearer: String) -> Self {
        let http = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .expect("failed to build alaya http client");
        AlayaClient { base, bearer, http }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base.as_str().trim_end_matches('/'))
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value, AppError> {
        let resp = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.bearer)
            .json(&body)
            .send()
            .await?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(upstream_error(status, &text));
        }
        serde_json::from_str(&text)
            .map_err(|_| AppError::Upstream("alaya-server returned non-JSON".into()))
    }

    async fn get(&self, path: &str) -> Result<Value, AppError> {
        let resp = self
            .http
            .get(self.url(path))
            .bearer_auth(&self.bearer)
            .send()
            .await?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(AppError::NotFound("memory not found".into()));
        }
        let text = resp.text().await?;
        if !status.is_success() {
            return Err(upstream_error(status, &text));
        }
        serde_json::from_str(&text)
            .map_err(|_| AppError::Upstream("alaya-server returned non-JSON".into()))
    }

    pub async fn search(&self, params: Value) -> Result<Value, AppError> {
        self.post("/search", params).await
    }

    pub async fn store(&self, params: Value) -> Result<Value, AppError> {
        self.post("/store", params).await
    }

    pub async fn get_memory(&self, content_hash: &str) -> Result<Value, AppError> {
        self.get(&format!("/memories/{content_hash}")).await
    }

    pub async fn delete(&self, content_hash: &str) -> Result<Value, AppError> {
        self.post("/delete", json!({ "content_hash": content_hash }))
            .await
    }

    pub async fn supersede(
        &self,
        old_hash: &str,
        new_hash: &str,
        reason: &str,
    ) -> Result<Value, AppError> {
        self.post(
            "/supersede",
            json!({ "old_hash": old_hash, "new_hash": new_hash, "reason": reason }),
        )
        .await
    }

    pub async fn relation(
        &self,
        action: &str,
        content_hash: &str,
        target_hash: Option<&str>,
        relation_type: Option<&str>,
    ) -> Result<Value, AppError> {
        self.post(
            "/relation",
            json!({
                "action": action,
                "content_hash": content_hash,
                "target_hash": target_hash,
                "relation_type": relation_type,
            }),
        )
        .await
    }

    pub async fn contradictions(&self, limit: usize) -> Result<Value, AppError> {
        self.post("/contradictions", json!({ "limit": limit }))
            .await
    }

    pub async fn find_duplicates(
        &self,
        similarity_threshold: f64,
        limit: usize,
    ) -> Result<Value, AppError> {
        self.post(
            "/duplicates/find",
            json!({ "similarity_threshold": similarity_threshold, "limit": limit }),
        )
        .await
    }

    pub async fn merge_duplicates(
        &self,
        canonical_hash: &str,
        duplicate_hashes: &[String],
        reason: &str,
        dry_run: bool,
    ) -> Result<Value, AppError> {
        self.post(
            "/duplicates/merge",
            json!({
                "canonical_hash": canonical_hash,
                "duplicate_hashes": duplicate_hashes,
                "reason": reason,
                "dry_run": dry_run,
            }),
        )
        .await
    }

    /// alaya-server split `/health` (LAB-2481): the bare probe carries only
    /// `status`; the memory count the home card renders lives on the
    /// authenticated detail view. The console holds the bearer anyway.
    pub async fn health(&self) -> Result<Value, AppError> {
        self.get("/health/detail").await
    }

    /// AC7: read-only auth-state view (OIDC config + principal tool matrix).
    pub async fn auth_config(&self) -> Result<Value, AppError> {
        self.get("/auth/config").await
    }
}
