//! HTTP handlers for alaya-bridge.
//!
//! All handlers share `AppState` via axum's `State` extractor.
//! Cypher queries are dispatched through `exec_query` which serializes
//! parameters as a `CYPHER key=value` prefix and uses `--compact` mode.

pub mod consolidation;
pub mod contradictions;
pub mod edges;
pub mod health;
pub mod hebbian;
pub mod nodes;

use std::collections::HashMap;

use axum::http::StatusCode;
use serde_json::Value;

use crate::{AppState, resp::FalkorResult};

// ─── Redis execution ──────────────────────────────────────────────────────────

/// Execute a Cypher query against FalkorDB via Redis.
///
/// Builds: `GRAPH.QUERY graph_name "CYPHER k=v ... query" --compact`
/// or:    `GRAPH.RO_QUERY graph_name "CYPHER k=v ... query" --compact`
pub async fn exec_query(
    state: &AppState,
    cypher: &str,
    params: HashMap<String, Value>,
    readonly: bool,
) -> Result<FalkorResult, StatusCode> {
    let cmd_name = if readonly {
        "GRAPH.RO_QUERY"
    } else {
        "GRAPH.QUERY"
    };

    let cypher_with_params = if params.is_empty() {
        cypher.to_string()
    } else {
        let param_str = params
            .iter()
            .map(|(k, v)| format!("{}={}", k, value_to_cypher_literal(v)))
            .collect::<Vec<_>>()
            .join(" ");
        format!("CYPHER {} {}", param_str, cypher)
    };

    let mut cmd = redis::cmd(cmd_name);
    cmd.arg(&state.graph_name);
    cmd.arg(&cypher_with_params);
    cmd.arg("--compact");

    let mut conn = state.redis.clone();
    let raw: redis::Value = match cmd.query_async(&mut conn).await {
        Ok(v) => v,
        // FalkorDB creates a graph lazily on the first write, so on a fresh
        // deployment (no memories stored yet) read queries hit a missing key
        // and fail with "Invalid graph operation on empty key". That's an
        // empty graph, not a failure — return an empty result so the 5s
        // health-check polls stay quiet. Gated on `readonly` so a write error
        // can never be swallowed as an empty success. (LAB-373 / alaya#32)
        Err(e) if readonly && is_empty_graph_error(&e) => {
            tracing::debug!(
                cmd = cmd_name,
                graph = state.graph_name.as_str(),
                "graph key absent (empty graph); returning empty result"
            );
            return Ok(FalkorResult {
                columns: Vec::new(),
                result_set: Vec::new(),
                stats: HashMap::new(),
            });
        }
        Err(e) => {
            // Log enough context to identify which query failed without
            // dumping params (may include user content).
            let cypher_preview: String = cypher.chars().take(120).collect();
            tracing::error!(
                cmd = cmd_name,
                graph = state.graph_name.as_str(),
                cypher = %cypher_preview,
                error = %e,
                "FalkorDB query failed"
            );
            return Err(StatusCode::BAD_GATEWAY);
        }
    };

    FalkorResult::parse(&raw).map_err(|e| {
        tracing::error!("FalkorDB parse error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })
}

/// True when a FalkorDB error signals a not-yet-created graph key.
///
/// FalkorDB creates a graph lazily on the first write; until then any read
/// answers with `-ERR Invalid graph operation on empty key`. redis-rs parses
/// the `ERR` prefix into a `ResponseError` carrying the message, so gate on
/// that kind first — a non-response error (e.g. an I/O failure) whose message
/// happens to contain "empty key" must never be swallowed as an empty graph —
/// then substring-match the distinctive "empty key". (LAB-373)
fn is_empty_graph_error(err: &redis::RedisError) -> bool {
    err.kind() == redis::ErrorKind::ResponseError && err.to_string().contains("empty key")
}

// ─── Cypher literal serialization ────────────────────────────────────────────

/// Convert a `serde_json::Value` to a Cypher literal for the `CYPHER` prefix.
///
/// - String → `'escaped'` (single-quoted, internal single quotes escaped)
/// - Number → bare numeric literal
/// - Bool   → `true` / `false`
/// - Null   → `null`
/// - Array  → `[elem, ...]` (elements recursively converted)
/// - Object → not supported in Cypher params; falls back to `null`
pub fn value_to_cypher_literal(v: &Value) -> String {
    match v {
        Value::String(s) => {
            let escaped = s.replace('\\', "\\\\").replace('\'', "\\'");
            format!("'{escaped}'")
        }
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        Value::Array(arr) => {
            let elems: Vec<String> = arr.iter().map(value_to_cypher_literal).collect();
            format!("[{}]", elems.join(", "))
        }
        Value::Object(_) => "null".to_string(),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn string_is_single_quoted() {
        assert_eq!(value_to_cypher_literal(&json!("hello")), "'hello'");
    }

    #[test]
    fn string_escapes_single_quote() {
        assert_eq!(value_to_cypher_literal(&json!("it's")), "'it\\'s'");
    }

    #[test]
    fn string_escapes_backslash() {
        assert_eq!(value_to_cypher_literal(&json!("a\\b")), "'a\\\\b'");
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn number_bare() {
        assert_eq!(value_to_cypher_literal(&json!(3.14)), "3.14");
        assert_eq!(value_to_cypher_literal(&json!(42)), "42");
    }

    #[test]
    fn bool_lowercase() {
        assert_eq!(value_to_cypher_literal(&json!(true)), "true");
        assert_eq!(value_to_cypher_literal(&json!(false)), "false");
    }

    #[test]
    fn null_literal() {
        assert_eq!(value_to_cypher_literal(&Value::Null), "null");
    }

    #[test]
    fn array_recursive() {
        assert_eq!(value_to_cypher_literal(&json!(["a", "b"])), "['a', 'b']");
    }

    #[test]
    fn empty_graph_error_detected() {
        // FalkorDB v4 returns `-ERR Invalid graph operation on empty key` for a
        // read against a graph key that doesn't exist yet (fresh deployment,
        // LAB-373). redis-rs parses the `ERR` prefix as a ResponseError whose
        // detail carries the message — verified live against falkordb:v4.18.6.
        let err = redis::RedisError::from((
            redis::ErrorKind::ResponseError,
            "response error",
            "Invalid graph operation on empty key".to_string(),
        ));
        assert!(is_empty_graph_error(&err));
    }

    #[test]
    fn genuine_errors_not_treated_as_empty_graph() {
        // A real query failure must still surface, not be swallowed as empty.
        let syntax = redis::RedisError::from((
            redis::ErrorKind::ResponseError,
            "response error",
            "errMsg: Invalid input 'X': expected a query".to_string(),
        ));
        assert!(!is_empty_graph_error(&syntax));

        let io = redis::RedisError::from((redis::ErrorKind::IoError, "connection reset by peer"));
        assert!(!is_empty_graph_error(&io));
    }

    #[test]
    fn non_response_error_mentioning_empty_key_not_swallowed() {
        // Only a server ResponseError may be treated as an empty graph. A
        // non-response error whose message happens to contain "empty key"
        // (e.g. an I/O failure surfaced mid-read) must still propagate.
        let io = redis::RedisError::from((
            redis::ErrorKind::IoError,
            "io error",
            "stream closed while reading empty key".to_string(),
        ));
        assert!(!is_empty_graph_error(&io));
    }
}
