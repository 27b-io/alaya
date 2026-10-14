//! Client-store module — the admin API client.
//!
//! The client store holds the LB's ordinary clients: a document that grants
//! (clients, keys, policies) and a set of restrictions that deny (disabled
//! names and revoked key digests). Every LB instance polls it. This module
//! is the console's only path to it: one URL, `{CLIENT_STORE_URL}/_clients/admin`,
//! with the writer token as `Authorization: Bearer`. `GET` reads everything;
//! `POST` carries either restriction ops or document ops, never both.
//!
//! The console still has no write path to the LB itself: `lb.rs` only GETs.

use std::time::Duration;

use axum::http::StatusCode;
use serde::Serialize;
use serde_json::{Value, json};

use crate::error::AppError;
use crate::http::{self, join, json_body, text_body};

/// The one path the store's writer token opens. The store answers 404 to any
/// other, so this constant is the whole of the console's write surface there.
pub const ADMIN_PATH: &str = "/_clients/admin";

/// The store reads its two objects one after the other, up to 10 s each, so
/// a slow `GET` legitimately takes 20 s.
const TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct StoreClient {
    base: url::Url,
    token: String,
    http: reqwest::Client,
}

/// A restriction op: the kill switch. Each op carries its own actor and
/// reason, which the store keeps on the record it writes.
#[derive(Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum RestrictionOp {
    Disable {
        name: String,
        actor: String,
        reason: String,
    },
    /// `expected_version` is the version of the name's record as the console
    /// read it, or 0 when it read none. The store answers 409 if the record
    /// changed since, so a stale form can never undo a newer `disable`.
    Enable {
        name: String,
        expected_version: u64,
        actor: String,
        reason: String,
    },
}

impl RestrictionOp {
    pub fn name(&self) -> &str {
        match self {
            RestrictionOp::Disable { name, .. } | RestrictionOp::Enable { name, .. } => name,
        }
    }
}

/// What a `POST` came back with: the store's status and its JSON answer. The
/// store puts the status in the body too, and every answer it writes is JSON;
/// a non-JSON answer is an edge or proxy page and is an error here.
pub struct Answer {
    pub status: StatusCode,
    pub body: Value,
}

impl StoreClient {
    pub fn new(base: url::Url, token: String) -> Self {
        StoreClient {
            base,
            token,
            http: http::client(TIMEOUT),
        }
    }

    fn admin_url(&self) -> String {
        join(&self.base, ADMIN_PATH)
    }

    /// `GET /_clients/admin`: the document, the restrictions with the part
    /// polls are served now, CF's report, history and the stacks that count.
    pub async fn read(&self) -> Result<Value, AppError> {
        let req = self.http.get(self.admin_url()).bearer_auth(&self.token);
        json_body("client store", req).await
    }

    /// `POST /_clients/admin` with `{"restrictions": ops}`. Returns the store's
    /// answer whatever its status, because 409, 422 and 503 each carry the
    /// store's own account of what it refused, which the console shows.
    pub async fn restrict(&self, ops: &[RestrictionOp]) -> Result<Answer, AppError> {
        self.post(&json!({ "restrictions": ops })).await
    }

    async fn post(&self, body: &Value) -> Result<Answer, AppError> {
        let req = self
            .http
            .post(self.admin_url())
            .bearer_auth(&self.token)
            .json(body);
        let (status, text) = text_body("client store", req).await?;
        match serde_json::from_str::<Value>(&text) {
            Ok(body) if body.is_object() => Ok(Answer { status, body }),
            // The edge blocks the admin path from every IP but one, and its
            // block page is HTML: say what most likely happened, without
            // echoing a body that is not the store's.
            _ => Err(AppError::Upstream(format!(
                "client store {status}: the answer was not the store's JSON (blocked at the edge?)"
            ))),
        }
    }
}

/// The store's refusal, one line per error: `code: message`. Only the two
/// fields the store's error shape defines are read, each clipped, so an
/// upstream answer cannot flood a page or the flash cookie.
pub fn refusal_lines(answer: &Value) -> Vec<String> {
    const MAX_LINES: usize = 8;
    const MAX_CHARS: usize = 200;
    let errors = answer
        .get("errors")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut lines: Vec<String> = errors
        .iter()
        .take(MAX_LINES)
        .map(|e| {
            let code = e.get("code").and_then(Value::as_str).unwrap_or("error");
            let message = e.get("message").and_then(Value::as_str).unwrap_or("");
            crate::routes::clip(&format!("{code}: {message}"), MAX_CHARS)
        })
        .collect();
    if errors.len() > MAX_LINES {
        lines.push(format!("…and {} more", errors.len() - MAX_LINES));
    }
    if lines.is_empty() {
        // A gateway refusal carries `{"error": …}` instead of `errors`.
        let gateway = answer
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("no detail");
        lines.push(crate::routes::clip(gateway, MAX_CHARS));
    }
    lines
}

/// Whether the store refused with this error code (anywhere in `errors`).
pub fn has_code(answer: &Value, code: &str) -> bool {
    answer
        .get("errors")
        .and_then(Value::as_array)
        .is_some_and(|errs| {
            errs.iter()
                .any(|e| e.get("code").and_then(Value::as_str) == Some(code))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restriction_ops_serialise_to_the_stores_shape() {
        let ops = vec![
            RestrictionOp::Enable {
                name: "spare".into(),
                expected_version: 1_791_698_400_000_000,
                actor: "ops@example.test".into(),
                reason: "make room".into(),
            },
            RestrictionOp::Disable {
                name: "acme".into(),
                actor: "ops@example.test".into(),
                reason: "leaked key".into(),
            },
        ];
        assert_eq!(
            json!({ "restrictions": ops }),
            json!({"restrictions": [
                {"op": "enable", "name": "spare", "expected_version": 1_791_698_400_000_000_u64,
                 "actor": "ops@example.test", "reason": "make room"},
                {"op": "disable", "name": "acme", "actor": "ops@example.test", "reason": "leaked key"}
            ]})
        );
    }

    #[test]
    fn refusal_lines_read_only_code_and_message_and_stay_bounded() {
        let answer = json!({"status": 409, "errors": [
            {"code": "enable.stale", "message": "acme changed since it was read", "name": "acme", "version": 7}
        ]});
        assert_eq!(
            refusal_lines(&answer),
            ["enable.stale: acme changed since it was read"]
        );
        assert!(has_code(&answer, "enable.stale"));
        assert!(!has_code(&answer, "disabled.cap"));
        let many: Vec<Value> = (0..20)
            .map(|i| json!({"code": "op.kind", "message": "x".repeat(500 + i)}))
            .collect();
        let lines = refusal_lines(&json!({ "errors": many }));
        assert_eq!(lines.len(), 9);
        assert_eq!(lines[8], "…and 12 more");
        assert!(lines[0].chars().count() <= 201, "{}", lines[0]);
        // The gateway's own shape.
        assert_eq!(
            refusal_lines(&json!({"error": "unauthorized"})),
            ["unauthorized"]
        );
    }
}
