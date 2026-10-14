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
use sha2::Digest;

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

    /// The state the op's record is written in.
    pub fn state(&self) -> crate::effect::RecordState {
        match self {
            RestrictionOp::Disable { .. } => crate::effect::RecordState::Disabled,
            RestrictionOp::Enable { .. } => crate::effect::RecordState::Enabled,
        }
    }
}

/// A document op: a change to the clients the store grants. The console
/// sends one per request.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum DocOp {
    Create {
        name: String,
        #[serde(flatten)]
        policy: Policy,
    },
    /// Sets all four policy fields to what the form shows.
    Update {
        name: String,
        #[serde(flatten)]
        policy: Policy,
    },
    /// Leaves the client's keys unrevoked, so a `restore` can undo it.
    Delete { name: String },
    /// The store mints the key and returns it once, in the commit's answer.
    AddKey { name: String, label: String },
    /// Revokes a key for good, by its digest.
    RevokeKey { digest: String },
    /// Returns the document to a version in history.
    Restore { version: u64 },
}

/// A client's policy. `None` is no budget or no ceiling; an empty list is
/// every model or no pins.
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Policy {
    pub daily_budget: Option<u64>,
    pub utilization_limit: Option<f64>,
    pub models: Vec<String>,
    pub preferred_endpoints: Vec<String>,
}

/// A document request. `expected_version` is the document version the
/// operator's page read: the store answers 409 if it moved since, so a
/// resubmitted form never writes twice. `dry_run` validates and returns the
/// diff without writing anything.
#[derive(Serialize)]
pub struct DocRequest<'a> {
    pub expected_version: u64,
    pub actor: &'a str,
    pub reason: &'a str,
    pub dry_run: bool,
    pub ops: &'a [DocOp],
}

/// A committed document write, read strictly from the store's 200.
#[derive(Debug)]
pub struct Commit {
    pub version: u64,
    pub digest: String,
    pub keys: Vec<MintedKey>,
}

/// A key the store minted, as its answer carries it. The only place a key
/// exists in the console: it is rendered once and dropped, never logged.
pub struct MintedKey {
    pub client: String,
    pub id: String,
    pub key: String,
    pub digest: String,
}

/// No key material in a Debug line, whatever prints it.
impl std::fmt::Debug for MintedKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MintedKey")
            .field("client", &self.client)
            .field("id", &self.id)
            .field("digest", &self.digest)
            .finish_non_exhaustive()
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

    /// `POST /_clients/admin` with `{"document": request}`, answered like
    /// `restrict`. A commit's 200 may carry minted keys: see `loggable`.
    pub async fn document(&self, request: &DocRequest<'_>) -> Result<Answer, AppError> {
        self.post(&json!({ "document": request })).await
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
            // Every answer the store writes is a JSON object; anything else
            // is not the store's, and only its status is worth showing.
            _ => Err(AppError::Upstream(format!(
                "client store {status}: the answer was not the store's JSON"
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

/// The members of a store answer the write log keeps. Anything else is left
/// out, so a member the store might add later can never carry key material
/// into a log line.
const LOGGED: [&str; 11] = [
    "status",
    "errors",
    "error",
    "dry_run",
    "version",
    "digest",
    "diff",
    "merged_after_commit",
    "results",
    "records_version",
    "message",
];

/// The answer as the write log may hold it: the members in `LOGGED`, and
/// each minted key reduced to its client, id and digest.
pub fn loggable(answer: &Value) -> Value {
    let mut out = serde_json::Map::new();
    for k in LOGGED {
        if let Some(v) = answer.get(k) {
            out.insert(k.to_string(), v.clone());
        }
    }
    if let Some(keys) = answer.get("keys").and_then(Value::as_array) {
        let keys = keys
            .iter()
            .map(|k| json!({"client": k.get("client"), "id": k.get("id"), "digest": k.get("digest")}))
            .collect();
        out.insert("keys".to_string(), Value::Array(keys));
    }
    Value::Object(out)
}

/// Read a commit's 200 strictly. Every key must be one the request asked
/// for, in order, and hash to the digest the store kept, so the page never
/// shows a key that cannot authenticate. Anything else is an error: the write
/// committed, but its answer cannot be trusted to say what it wrote.
pub fn commit_of(answer: &Value, ops: &[DocOp]) -> Result<Commit, String> {
    let version = answer
        .get("version")
        .and_then(Value::as_u64)
        .filter(|v| *v >= 1)
        .ok_or("no version")?;
    let digest = answer
        .get("digest")
        .and_then(Value::as_str)
        .filter(|d| crate::effect::is_hex64(d))
        .ok_or("no digest")?
        .to_string();
    let wanted: Vec<&str> = ops
        .iter()
        .filter_map(|op| match op {
            DocOp::AddKey { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    // A request with no add_key has no key to lose, so an answer may leave
    // the list out; one that asked for a key must carry it.
    let keys = match answer.get("keys") {
        Some(Value::Array(keys)) => keys.as_slice(),
        None if wanted.is_empty() => &[],
        _ => return Err("no keys list".into()),
    };
    if keys.len() != wanted.len() {
        return Err(format!(
            "{} keys for {} add_key ops",
            keys.len(),
            wanted.len()
        ));
    }
    let mut minted = Vec::new();
    for (k, client) in keys.iter().zip(wanted) {
        let field = |f: &str| k.get(f).and_then(Value::as_str).unwrap_or_default();
        let (key, digest) = (field("key"), field("digest"));
        if field("client") != client {
            return Err("a key names another client than its op".into());
        }
        if !is_key_id(field("id")) || !is_minted_key(key) || !crate::effect::is_hex64(digest) {
            return Err("a key is malformed".into());
        }
        if crate::effect::hex(&sha2::Sha256::digest(key.as_bytes())) != digest {
            return Err(format!("key {} does not hash to its digest", field("id")));
        }
        minted.push(MintedKey {
            client: client.to_string(),
            id: field("id").to_string(),
            key: key.to_string(),
            digest: digest.to_string(),
        });
    }
    Ok(Commit {
        version,
        digest,
        keys: minted,
    })
}

/// `k` and 7 lowercase hex digits: the store's key ids.
fn is_key_id(s: &str) -> bool {
    s.len() == 8
        && s.starts_with('k')
        && s[1..]
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// 32 random bytes, base64url without padding: 43 characters.
fn is_minted_key(s: &str) -> bool {
    s.len() == 43
        && s.bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
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
    fn document_ops_serialise_to_the_stores_shape() {
        let policy = Policy {
            daily_budget: Some(600_000_000),
            utilization_limit: None,
            models: vec!["claude-*".into()],
            preferred_endpoints: vec![],
        };
        let ops = [
            DocOp::Create {
                name: "acme".into(),
                policy: policy.clone(),
            },
            DocOp::Update {
                name: "acme".into(),
                policy,
            },
            DocOp::Delete {
                name: "acme".into(),
            },
            DocOp::AddKey {
                name: "acme".into(),
                label: "2026-10".into(),
            },
            DocOp::RevokeKey {
                digest: "a".repeat(64),
            },
            DocOp::Restore { version: 7 },
        ];
        let req = DocRequest {
            expected_version: 1_791_700_000_123,
            actor: "ops@example.test",
            reason: "rotation",
            dry_run: true,
            ops: &ops,
        };
        assert_eq!(
            json!({ "document": req }),
            json!({"document": {
                "expected_version": 1_791_700_000_123_u64, "actor": "ops@example.test",
                "reason": "rotation", "dry_run": true,
                "ops": [
                    {"op": "create", "name": "acme", "daily_budget": 600_000_000, "utilization_limit": null,
                     "models": ["claude-*"], "preferred_endpoints": []},
                    {"op": "update", "name": "acme", "daily_budget": 600_000_000, "utilization_limit": null,
                     "models": ["claude-*"], "preferred_endpoints": []},
                    {"op": "delete", "name": "acme"},
                    {"op": "add_key", "name": "acme", "label": "2026-10"},
                    {"op": "revoke_key", "digest": "a".repeat(64)},
                    {"op": "restore", "version": 7}
                ]
            }})
        );
    }

    /// A key and its digest as the store mints them.
    fn minted(client: &str) -> Value {
        let key = {
            use base64::Engine;
            base64::engine::general_purpose::URL_SAFE_NO_PAD.encode([7u8; 32])
        };
        let digest = crate::effect::hex(&sha2::Sha256::digest(key.as_bytes()));
        json!({"client": client, "id": format!("k{}", &digest[..7]), "key": key, "digest": digest})
    }

    #[test]
    fn a_commit_is_read_strictly_and_its_keys_must_hash_to_their_digest() {
        let ops = [DocOp::AddKey {
            name: "acme".into(),
            label: String::new(),
        }];
        let answer = json!({"status": 200, "version": 9, "digest": "b".repeat(64), "diff": [], "keys": [minted("acme")]});
        let c = commit_of(&answer, &ops).unwrap();
        assert_eq!((c.version, c.digest.as_str()), (9, "b".repeat(64).as_str()));
        assert_eq!(c.keys.len(), 1);
        assert_eq!(c.keys[0].key.len(), 43);
        // Debug never prints the key.
        assert!(!format!("{c:?}").contains(&c.keys[0].key));

        // A key whose digest is not its hash cannot authenticate: refused.
        let mut lying = answer.clone();
        lying["keys"][0]["digest"] = json!("c".repeat(64));
        assert!(
            commit_of(&lying, &ops)
                .unwrap_err()
                .contains("does not hash")
        );
        // A key for another client than the op named.
        let mut other = answer.clone();
        other["keys"][0] = minted("other");
        assert!(commit_of(&other, &ops).is_err());
        // A key missing, or one nobody asked for.
        let mut none = answer.clone();
        none["keys"] = json!([]);
        assert!(commit_of(&none, &ops).is_err());
        assert!(
            commit_of(
                &answer,
                &[DocOp::Delete {
                    name: "acme".into()
                }]
            )
            .is_err()
        );
        // No version or digest: unreadable.
        assert!(commit_of(&json!({"keys": []}), &[]).is_err());
        // No keys list: fine with no key asked for, never with one.
        let plain = json!({"status": 200, "version": 9, "digest": "b".repeat(64)});
        assert!(
            commit_of(
                &plain,
                &[DocOp::Delete {
                    name: "acme".into()
                }]
            )
            .is_ok()
        );
        assert!(commit_of(&plain, &ops).is_err());
        let mut odd = plain.clone();
        odd["keys"] = json!({"acme": "x"});
        assert!(
            commit_of(
                &odd,
                &[DocOp::Delete {
                    name: "acme".into()
                }]
            )
            .is_err()
        );
    }

    #[test]
    fn the_loggable_answer_carries_no_key() {
        let answer = json!({"status": 200, "version": 9, "digest": "b".repeat(64), "keys": [minted("acme")]});
        let key = answer["keys"][0]["key"].as_str().unwrap().to_string();
        let logged = loggable(&answer).to_string();
        assert!(!logged.contains(&key), "{logged}");
        assert!(logged.contains("\"client\":\"acme\"") && logged.contains(&"b".repeat(64)));
        // An answer with no keys passes through unchanged.
        let refusal = json!({"status": 409, "errors": [{"code": "version.stale", "message": "x"}]});
        assert_eq!(loggable(&refusal), refusal);
        // A member the console does not know is left out, whatever it holds.
        let odd = json!({"status": 200, "version": 9, "secret": key, "keys": {"key": key}});
        let logged = loggable(&odd).to_string();
        assert!(!logged.contains(&key), "{logged}");
        assert!(logged.contains("\"version\":9"), "{logged}");
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
