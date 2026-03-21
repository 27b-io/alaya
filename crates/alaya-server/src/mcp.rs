//! MCP Streamable HTTP transport — JSON-RPC 2.0 over HTTP POST.
//!
//! Single `/mcp` endpoint handles: initialize, tools/list, tools/call, ping.
//! Stateless — each request is self-contained, no session management.

use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::sync::oneshot;

use alaya_core::deduplication::CanonicalStrategy;
use alaya_core::service::{RelationParams, SearchParams, StoreParams};

use crate::{Cmd, ServiceHandle};

// ─── JSON-RPC types ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct JsonRpcRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

#[derive(Serialize)]
struct JsonRpcResponse {
    jsonrpc: &'static str,
    id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcErrorBody>,
}

#[derive(Serialize)]
struct JsonRpcErrorBody {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

impl JsonRpcResponse {
    fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    fn error(id: Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(JsonRpcErrorBody {
                code,
                message: message.into(),
                data: None,
            }),
        }
    }
}

// ─── MCP handler ────────────────────────────────────────────────────────────

const MAX_BODY_SIZE: usize = 1_048_576; // 1MB

pub async fn mcp_handler(
    headers: HeaderMap,
    handle: axum::extract::State<ServiceHandle>,
    body: axum::body::Bytes,
) -> Response {
    // Validate Content-Type
    let ct = headers
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !ct.contains("application/json") {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }

    // Body size check
    if body.len() > MAX_BODY_SIZE {
        return StatusCode::PAYLOAD_TOO_LARGE.into_response();
    }

    // Check if client wants SSE
    let wants_sse = headers
        .get("accept")
        .and_then(|v| v.to_str().ok())
        .map(|a| a.contains("text/event-stream"))
        .unwrap_or(false);

    // Parse JSON-RPC request
    let req: JsonRpcRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            let resp = JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}"));
            return make_response(resp, wants_sse);
        }
    };

    // Validate jsonrpc version
    if req.jsonrpc != "2.0" {
        let resp = JsonRpcResponse::error(
            req.id.unwrap_or(Value::Null),
            -32600,
            "Invalid Request: jsonrpc must be \"2.0\"",
        );
        return make_response(resp, wants_sse);
    }

    let id = req.id.unwrap_or(Value::Null);

    // Notifications (no id) → 202 Accepted
    if id.is_null()
        && matches!(
            req.method.as_str(),
            "initialized" | "notifications/cancelled"
        )
    {
        return StatusCode::ACCEPTED.into_response();
    }

    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(id),
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tools_call(id, req.params, &handle).await,
        "ping" => JsonRpcResponse::success(id, json!({})),
        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", req.method)),
    };

    make_response(resp, wants_sse)
}

/// Format response as SSE or plain JSON depending on client preference.
fn make_response(resp: JsonRpcResponse, sse: bool) -> Response {
    let json_str = serde_json::to_string(&resp).unwrap_or_else(|_| "{}".to_string());

    if sse {
        let body = format!("event: message\ndata: {json_str}\n\n");
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/event-stream")],
            body,
        )
            .into_response()
    } else {
        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            json_str,
        )
            .into_response()
    }
}

// ─── Method handlers ────────────────────────────────────────────────────────

fn handle_initialize(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": { "listChanged": false }
            },
            "serverInfo": {
                "name": "alaya",
                "version": env!("CARGO_PKG_VERSION")
            }
        }),
    )
}

fn handle_tools_list(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(id, json!({ "tools": tool_schemas() }))
}

async fn handle_tools_call(
    id: Value,
    params: Option<Value>,
    handle: &ServiceHandle,
) -> JsonRpcResponse {
    let params = match params {
        Some(p) => p,
        None => {
            return JsonRpcResponse::error(id, -32602, "Missing params for tools/call");
        }
    };

    let tool_name = match params.get("name").and_then(|n| n.as_str()) {
        Some(n) => n.to_string(),
        None => {
            return JsonRpcResponse::error(id, -32602, "Missing 'name' in tools/call params");
        }
    };

    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| json!({}));

    let result = dispatch_tool(&tool_name, arguments, handle).await;

    match result {
        Ok(value) => {
            // MCP wraps tool results in content array
            let text = serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string());
            JsonRpcResponse::success(
                id,
                json!({
                    "content": [{
                        "type": "text",
                        "text": text
                    }]
                }),
            )
        }
        Err((code, msg)) => JsonRpcResponse::error(id, code, msg),
    }
}

// ─── Tool dispatch ──────────────────────────────────────────────────────────

async fn dispatch_tool(
    name: &str,
    args: Value,
    handle: &ServiceHandle,
) -> Result<Value, (i32, String)> {
    match name {
        "store_memory" => {
            let params: StoreParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Store { params, reply: tx })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "search" => {
            let params: SearchParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Search { params, reply: tx })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "delete_memory" => {
            let hash = args
                .get("content_hash")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing content_hash".to_string()))?
                .to_string();
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Delete { hash, reply: tx })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "check_database_health" => {
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Health { reply: tx })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "relation" => {
            let params: RelationParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Relation { params, reply: tx })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "memory_supersede" => {
            let old = args
                .get("old_id")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing old_id".to_string()))?
                .to_string();
            let new = args
                .get("new_id")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing new_id".to_string()))?
                .to_string();
            let reason = args
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Supersede {
                    old_hash: old,
                    new_hash: new,
                    reason,
                    reply: tx,
                })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "memory_contradictions" => {
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::Contradictions { limit, reply: tx })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "find_duplicates" => {
            let threshold = args
                .get("similarity_threshold")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.95);
            let limit = args.get("limit").and_then(|v| v.as_u64()).unwrap_or(100) as usize;
            let strategy: CanonicalStrategy = args
                .get("strategy")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::FindDuplicates {
                    threshold,
                    limit,
                    strategy,
                    reply: tx,
                })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        "merge_duplicates" => {
            let canonical = args
                .get("canonical_hash")
                .and_then(|v| v.as_str())
                .ok_or((-32602, "Missing canonical_hash".to_string()))?
                .to_string();
            let dupes: Vec<String> = args
                .get("duplicate_hashes")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .ok_or((-32602, "Missing duplicate_hashes".to_string()))?;
            let reason = args
                .get("reason")
                .and_then(|v| v.as_str())
                .unwrap_or("Merged by deduplication")
                .to_string();
            let dry_run = args
                .get("dry_run")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let (tx, rx) = oneshot::channel();
            handle
                .tx
                .send(Cmd::MergeDuplicates {
                    canonical,
                    duplicates: dupes,
                    reason,
                    dry_run,
                    reply: tx,
                })
                .await
                .map_err(|_| (-32000, "Service unavailable".to_string()))?;
            rx.await
                .map_err(|_| (-32000, "Service dropped".to_string()))
        }
        _ => Err((-32601, format!("Unknown tool: {name}"))),
    }
}

// ─── Tool schemas ───────────────────────────────────────────────────────────

fn tool_schemas() -> Value {
    json!([
        {
            "name": "store_memory",
            "description": "Store a new memory for future semantic retrieval. Content is vectorized for similarity search. Salience scoring and contradiction detection are computed automatically.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Text to store (embedded for semantic search)" },
                    "tags": { "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}, {"type": "null"}], "description": "Labels — accepts [\"tag1\", \"tag2\"] or \"tag1,tag2\"" },
                    "memory_type": { "type": "string", "enum": ["note", "decision", "task", "reference"], "default": "note" },
                    "metadata": { "anyOf": [{"type": "object"}, {"type": "null"}], "description": "Structured data. Special key: importance (float 0.0-1.0)" },
                    "client_hostname": { "anyOf": [{"type": "string"}, {"type": "null"}] },
                    "summary": { "anyOf": [{"type": "string"}, {"type": "null"}], "description": "One-line summary (~50 tokens). Auto-generated if omitted." },
                    "dedup_threshold": { "anyOf": [{"type": "number"}, {"type": "null"}], "description": "If set, skip storage when nearest neighbor similarity >= threshold (0.0-1.0)" }
                },
                "required": ["content"]
            }
        },
        {
            "name": "search",
            "description": "Search and retrieve memories. Consolidates all retrieval modes into one tool.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "default": "", "description": "Natural language search query" },
                    "mode": { "type": "string", "enum": ["hybrid", "scan", "similar", "tag", "recent"], "default": "hybrid" },
                    "tags": { "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}] },
                    "match_all": { "type": "boolean", "default": false },
                    "k": { "type": "integer", "default": 10, "description": "Max results for scan/similar modes" },
                    "page": { "type": "integer", "default": 1 },
                    "page_size": { "type": "integer", "default": 10 },
                    "min_similarity": { "anyOf": [{"type": "number"}, {"type": "null"}], "default": 0.3 },
                    "output": { "type": "string", "enum": ["full", "summary", "both"], "default": "full" },
                    "memory_type": { "anyOf": [{"type": "string"}, {"type": "null"}] },
                    "encoding_context": { "anyOf": [{"type": "object"}, {"type": "null"}] },
                    "include_superseded": { "type": "boolean", "default": false },
                    "min_trust_score": { "anyOf": [{"type": "number"}, {"type": "null"}] }
                }
            }
        },
        {
            "name": "delete_memory",
            "description": "Permanently delete a specific memory by its unique identifier.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": { "type": "string", "description": "Unique identifier returned from store_memory or search results" }
                },
                "required": ["content_hash"]
            }
        },
        {
            "name": "check_database_health",
            "description": "Check memory database health and get storage statistics.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "relation",
            "description": "Manage typed relationships between memories in the knowledge graph.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["create", "get", "delete"] },
                    "content_hash": { "type": "string", "description": "Content hash of the primary/source memory" },
                    "target_hash": { "anyOf": [{"type": "string"}, {"type": "null"}], "description": "Required for create/delete" },
                    "relation_type": { "anyOf": [{"type": "string"}, {"type": "null"}], "enum": ["RELATES_TO", "PRECEDES", "CONTRADICTS"], "description": "Required for create/delete" }
                },
                "required": ["action", "content_hash"]
            }
        },
        {
            "name": "memory_supersede",
            "description": "Mark one memory as superseded by another, resolving a contradiction.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "old_id": { "type": "string", "description": "Content hash of the memory being superseded" },
                    "new_id": { "type": "string", "description": "Content hash of the newer memory" },
                    "reason": { "type": "string", "default": "" }
                },
                "required": ["old_id", "new_id"]
            }
        },
        {
            "name": "memory_contradictions",
            "description": "List unresolved contradiction pairs for review and resolution.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 20, "description": "Max contradiction pairs to return" }
                }
            }
        },
        {
            "name": "find_duplicates",
            "description": "Scan memories for near-duplicates using embedding cosine similarity.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "similarity_threshold": { "type": "number", "default": 0.95 },
                    "limit": { "type": "integer", "default": 500 },
                    "strategy": { "type": "string", "enum": ["keep_newest", "keep_oldest", "keep_most_accessed"], "default": "keep_newest" }
                }
            }
        },
        {
            "name": "merge_duplicates",
            "description": "Supersede duplicate memories in favour of a canonical one.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "canonical_hash": { "type": "string" },
                    "duplicate_hashes": { "type": "array", "items": {"type": "string"} },
                    "reason": { "type": "string", "default": "Merged by deduplication" },
                    "dry_run": { "type": "boolean", "default": false }
                },
                "required": ["canonical_hash", "duplicate_hashes"]
            }
        }
    ])
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_jsonrpc_request() {
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.method, "initialize");
        assert_eq!(req.id, Some(json!(1)));
    }

    #[test]
    fn initialize_response() {
        let resp = handle_initialize(json!(1));
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["result"]["protocolVersion"], "2025-03-26");
        assert!(v["result"]["capabilities"]["tools"].is_object());
        assert_eq!(v["result"]["serverInfo"]["name"], "alaya");
    }

    #[test]
    fn tools_list_returns_9_tools() {
        let resp = handle_tools_list(json!(2));
        let v = serde_json::to_value(&resp).unwrap();
        let tools = v["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 9);
    }

    #[test]
    fn tools_list_has_all_names() {
        let schemas = tool_schemas();
        let names: Vec<&str> = schemas
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"store_memory"));
        assert!(names.contains(&"search"));
        assert!(names.contains(&"delete_memory"));
        assert!(names.contains(&"check_database_health"));
        assert!(names.contains(&"relation"));
        assert!(names.contains(&"memory_supersede"));
        assert!(names.contains(&"memory_contradictions"));
        assert!(names.contains(&"find_duplicates"));
        assert!(names.contains(&"merge_duplicates"));
    }

    #[test]
    fn error_response_format() {
        let resp = JsonRpcResponse::error(json!(99), -32601, "Method not found");
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["error"]["code"], -32601);
        assert_eq!(v["error"]["message"], "Method not found");
        assert!(v["result"].is_null());
    }

    #[test]
    fn invalid_jsonrpc_version() {
        let raw = r#"{"jsonrpc":"1.0","id":1,"method":"test"}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert_ne!(req.jsonrpc, "2.0");
    }

    #[test]
    fn store_memory_schema_has_required_content() {
        let schemas = tool_schemas();
        let store = schemas
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "store_memory")
            .unwrap();
        let required = store["inputSchema"]["required"].as_array().unwrap();
        assert!(required.contains(&json!("content")));
    }
}
