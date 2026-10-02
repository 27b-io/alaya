//! Trusted-service HTTP client for alaya-server's REST API (D2: the console
//! backend holds the static bearer; the browser never sees it).
//!
//! Responses stay `serde_json::Value` — the console renders fields
//! defensively instead of mirroring another service's output types, so an
//! additive upstream change can't break the UI.

use serde_json::{Value, json};

use crate::error::AppError;

/// `AlayaError::NotFound`'s `safe_message` on alaya-server.
const NOT_FOUND_MESSAGE: &str = "Resource not found";

/// The one non-null resolution the server accepts.
pub const KEEP_BOTH: &str = "keep_both";

#[derive(Clone)]
pub struct AlayaClient {
    base: url::Url,
    bearer: String,
    http: reqwest::Client,
}

impl AlayaClient {
    pub fn new(base: url::Url, bearer: String) -> Self {
        let http = crate::http::client(std::time::Duration::from_secs(60));
        AlayaClient { base, bearer, http }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base.as_str().trim_end_matches('/'))
    }

    async fn post(&self, path: &str, body: Value) -> Result<Value, AppError> {
        // Surfaced, or the operator gets a green flash for a write that
        // never happened (panel, LAB-3885).
        op_failure(self.post_json(path, body).await?)
    }

    /// `post` without the `success: false` guard, for the one caller that
    /// reads a typed `status` first.
    async fn post_json(&self, path: &str, body: Value) -> Result<Value, AppError> {
        let resp = self
            .http
            .post(self.url(path))
            .bearer_auth(&self.bearer)
            .json(&body)
            .send()
            .await
            .map_err(|e| AppError::transport("alaya-server", &e))?;
        let status = resp.status();
        let text = crate::http::body_text("alaya-server", resp)
            .await
            .map_err(|e| AppError::body("alaya-server", e))?;
        if !status.is_success() {
            return Err(AppError::non_success("alaya-server", status, &text));
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
            .await
            .map_err(|e| AppError::transport("alaya-server", &e))?;
        let status = resp.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(AppError::NotFound("memory not found".into()));
        }
        let text = crate::http::body_text("alaya-server", resp)
            .await
            .map_err(|e| AppError::body("alaya-server", e))?;
        if !status.is_success() {
            return Err(AppError::non_success("alaya-server", status, &text));
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

    /// Reverse a wrong supersession (LAB-6876). The two typed
    /// `success: false` answers are outcomes the operator acts on, so they
    /// are read before the guard; every other answer still goes through it.
    pub async fn unsupersede(
        &self,
        content_hash: &str,
        reason: &str,
    ) -> Result<Unsupersede, AppError> {
        let body = self
            .post_json(
                "/unsupersede",
                json!({
                    "content_hash": content_hash,
                    "reason": reason,
                    "unsuperseded_via": "operator:console",
                }),
            )
            .await?;
        let success = body.get("success").and_then(Value::as_bool);
        match (success, body.get("status").and_then(Value::as_str)) {
            (Some(true), Some("unsuperseded")) => Ok(Unsupersede::Reversed {
                superseded_again: body.get("now_superseded_by").is_some_and(|v| !v.is_null()),
            }),
            (Some(false), Some("not_superseded")) => Ok(Unsupersede::NotSuperseded),
            (Some(false), Some("superseded_by_changed")) => Ok(Unsupersede::SupersededByChanged),
            _ => {
                op_failure(body)?;
                // A reversal is reported only when the server says it landed.
                Err(AppError::Upstream(
                    "alaya-server: unrecognized unsupersede answer".into(),
                ))
            }
        }
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

    /// Stamp (`Some("keep_both")`) or clear (`None`) a CONTRADICTS pair's
    /// operator resolution (LAB-3885). Non-destructive either way: neither
    /// memory is touched, and a stamped pair leaves the default queue.
    /// `resolved_via` is fixed to the console's tag — the server records it
    /// verbatim, and requires it on a clear too.
    pub async fn set_resolution(
        &self,
        memory_a_hash: &str,
        memory_b_hash: &str,
        resolution: Option<&str>,
    ) -> Result<Value, AppError> {
        self.post(
            "/contradictions/resolution",
            json!({
                "memory_a_hash": memory_a_hash,
                "memory_b_hash": memory_b_hash,
                "resolution": resolution,
                "resolved_via": "operator:console",
            }),
        )
        .await
    }

    /// One queue page. Empty `verdicts` lets the server apply its default
    /// filter (contradiction, supersession, unjudged).
    pub async fn contradictions(
        &self,
        limit: usize,
        offset: usize,
        include_resolved: bool,
        verdicts: &[String],
    ) -> Result<Value, AppError> {
        let mut body = json!({
            "limit": limit,
            "offset": offset,
            "include_resolved": include_resolved,
        });
        if !verdicts.is_empty() {
            body["verdicts"] = json!(verdicts);
        }
        self.post("/contradictions", body).await
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

    /// `GET /stats` (LAB-6881): corpus and contradiction-judge aggregates.
    /// A worker deadline answers `200 {"success": false}`, which must read
    /// as a failure, never as an empty document.
    pub async fn stats(&self) -> Result<Value, AppError> {
        op_failure(self.get("/stats").await?)
    }
}

/// What `POST /unsupersede` answered.
pub enum Unsupersede {
    /// Reversed. `superseded_again`: a supersession landed straight after
    /// the reversal (a non-null `now_superseded_by`).
    Reversed { superseded_again: bool },
    /// Nothing to reverse.
    NotSuperseded,
    /// Superseded again before the reversal ran; nothing was reversed.
    SupersededByChanged,
}

/// Op-level failures come back `200 {"success": false, "error": …}`; turn
/// one into an error so no caller renders it as a result.
fn op_failure(body: Value) -> Result<Value, AppError> {
    if body.get("success").and_then(Value::as_bool) == Some(false) {
        let detail = body
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("operation failed");
        // The server reports a missing target only as this fixed
        // `safe_message`, in a 200 body. Typed, because Reopen expects
        // it for a pair with no reverse edge.
        if detail == NOT_FOUND_MESSAGE {
            return Err(AppError::NotFound(format!("alaya-server: {detail}")));
        }
        return Err(AppError::Upstream(format!("alaya-server: {detail}")));
    }
    Ok(body)
}
