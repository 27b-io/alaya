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
    let raw: redis::Value = cmd.query_async(&mut conn).await.map_err(|e| {
        tracing::error!("Redis error: {e}");
        StatusCode::BAD_GATEWAY
    })?;

    FalkorResult::parse(&raw).map_err(|e| {
        tracing::error!("FalkorDB parse error: {e}");
        StatusCode::INTERNAL_SERVER_ERROR
    })
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
}
