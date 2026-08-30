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
use alaya_core::service::{OutputMode, RelationParams, SearchParams, StoreParams};

use crate::auth::{AuthPrincipal, oidc_allows};
use crate::{CmdInner, ServiceHandle};

// ─── Typed param structs for MCP dispatch ───────────────────────────────────

#[derive(Deserialize)]
struct DeleteParams {
    content_hash: String,
}

#[derive(Deserialize)]
struct GetMemoryParams {
    content_hash: String,
    #[serde(default)]
    output: OutputMode,
}

#[derive(Deserialize)]
struct SupersedeParams {
    old_id: String,
    new_id: String,
    #[serde(default)]
    reason: String,
}

#[derive(Deserialize)]
struct ContradictionsParams {
    #[serde(default = "default_contradictions_limit")]
    limit: usize,
}
fn default_contradictions_limit() -> usize {
    20
}

#[derive(Deserialize)]
struct FindDuplicatesParams {
    #[serde(default = "default_find_dup_threshold")]
    similarity_threshold: f64,
    #[serde(default = "default_find_dup_limit")]
    limit: usize,
    #[serde(default)]
    strategy: CanonicalStrategy,
}
fn default_find_dup_threshold() -> f64 {
    0.95
}
fn default_find_dup_limit() -> usize {
    500
}

#[derive(Deserialize)]
struct MergeDuplicatesParams {
    canonical_hash: String,
    duplicate_hashes: Vec<String>,
    #[serde(default = "default_merge_reason")]
    reason: String,
    #[serde(default)]
    dry_run: bool,
}
fn default_merge_reason() -> String {
    "Merged by deduplication".into()
}

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

fn is_accepted_notification(id: &Value, method: &str) -> bool {
    id.is_null()
        && matches!(
            method,
            "initialized" | "notifications/initialized" | "notifications/cancelled"
        )
}

pub async fn mcp_handler(
    headers: HeaderMap,
    axum::Extension(principal): axum::Extension<AuthPrincipal>,
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
    if is_accepted_notification(&id, req.method.as_str()) {
        return StatusCode::ACCEPTED.into_response();
    }

    // Default-deny authorization: an Oidc principal may only call allowlisted
    // tools. Enforced here, axum-side, before any channel dispatch.
    if req.method == "tools/call" && principal == AuthPrincipal::Oidc {
        let tool = req
            .params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");
        if !oidc_allows(tool) {
            let resp = JsonRpcResponse::error(id, -32001, "forbidden for this principal");
            return make_response(resp, wants_sse);
        }
    }

    let resp = match req.method.as_str() {
        "initialize" => handle_initialize(id),
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tools_call(id, req.params, &handle, principal).await,
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

/// Model-facing usage hint returned in the `initialize` result. The MCP spec
/// defines `instructions` as a hint to improve the LLM's understanding of the
/// server. This is the single DRY home for the cross-tool hash convention —
/// without it the rule is only stated per-input-field and never as a whole.
const SERVER_INSTRUCTIONS: &str = "Ālaya is a persistent semantic memory service (vector search + knowledge graph).

Memory identifiers: every memory is keyed by `content_hash`, a full 64-character SHA-256 hex string. `store_memory` and `search` return this hash on every result — pass it back verbatim to `get_memory`, `memory_supersede`, `delete_memory`, `relation`, and `merge_duplicates`. Never truncate or abbreviate it; the 8-character prefixes shown in log lines and display output are not valid identifiers and will be rejected.

Inspect before mutating: use `get_memory` to read a memory by hash before `memory_supersede` or `delete_memory`.";

fn handle_initialize(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "tools": { "listChanged": false }
            },
            // SHA-qualified so an MCP client can tell two releases of the same
            // crate version apart — bare CARGO_PKG_VERSION could not (#70).
            "serverInfo": {
                "name": "alaya",
                "version": crate::build_info::version_qualified()
            },
            "instructions": SERVER_INSTRUCTIONS
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
    principal: AuthPrincipal,
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

    let result = dispatch_tool(&tool_name, arguments, handle, principal).await;

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
    principal: AuthPrincipal,
) -> Result<Value, (i32, String)> {
    let read_only = matches!(principal, AuthPrincipal::Oidc);
    match name {
        "store_memory" => {
            let params: StoreParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Store {
                        params,
                        read_only,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "search" => {
            let params: SearchParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Search {
                        params,
                        read_only,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "delete_memory" => {
            let p: DeleteParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Delete {
                        hash: p.content_hash,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "get_memory" => {
            let p: GetMemoryParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::GetMemory {
                        hash: p.content_hash,
                        output: p.output,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "check_database_health" => {
            let (tx, rx) = oneshot::channel();
            handle.call_rpc(CmdInner::Health { reply: tx }, rx).await
        }
        "relation" => {
            let params: RelationParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(CmdInner::Relation { params, reply: tx }, rx)
                .await
        }
        "memory_supersede" => {
            let p: SupersedeParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Supersede {
                        old_hash: p.old_id,
                        new_hash: p.new_id,
                        reason: p.reason,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "memory_contradictions" => {
            let p: ContradictionsParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Contradictions {
                        limit: p.limit,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "find_duplicates" => {
            let p: FindDuplicatesParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::FindDuplicates {
                        threshold: p.similarity_threshold,
                        limit: p.limit,
                        strategy: p.strategy,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "merge_duplicates" => {
            let p: MergeDuplicatesParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::MergeDuplicates {
                        canonical: p.canonical_hash,
                        duplicates: p.duplicate_hashes,
                        reason: p.reason,
                        dry_run: p.dry_run,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        _ => Err((-32601, format!("Unknown tool: {name}"))),
    }
}

// ─── Tool schemas ───────────────────────────────────────────────────────────

fn tool_schemas() -> Value {
    json!([
        {
            "name": "store_memory",
            "description": "Store a new memory for future semantic retrieval. Content is vectorized for similarity search. Salience scoring and contradiction detection are computed automatically. Returns the new memory's full 64-char content_hash — the identifier used by get_memory, memory_supersede, delete_memory, and relation.",
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
            "description": "Search and retrieve memories. Consolidates all retrieval modes into one tool. Each result includes a full 64-char content_hash — pass it to get_memory for an exact re-fetch, or to memory_supersede/delete_memory/relation.",
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
                    "min_trust_score": { "anyOf": [{"type": "number"}, {"type": "null"}] },
                    "cursor": { "anyOf": [{"type": "number"}, {"type": "null"}], "description": "Cursor for recent mode pagination. Pass next_cursor from previous response." }
                }
            }
        },
        {
            "name": "get_memory",
            "description": "Retrieve a single memory by its exact content_hash. Returns {\"found\": true, \"memory\": {...}} or {\"found\": false}. Use this to inspect a memory before supersede/delete/relation when you already hold its hash — e.g. from search results or a memory_contradictions report. Superseded memories are returned; check metadata.superseded_by.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash returned by store_memory or search results. Do not pass truncated display/log prefixes."
                    },
                    "output": { "type": "string", "enum": ["full", "summary", "both"], "default": "full" }
                },
                "required": ["content_hash"]
            }
        },
        {
            "name": "delete_memory",
            "description": "Permanently delete a specific memory by its unique identifier.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash returned by store_memory or search results. Do not pass truncated display/log prefixes."
                    }
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
                    "content_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash of the primary/source memory. Do not pass truncated display/log prefixes."
                    },
                    "target_hash": {
                        "anyOf": [
                            { "type": "string", "minLength": 64, "maxLength": 64, "pattern": "^[0-9a-f]{64}$" },
                            { "type": "null" }
                        ],
                        "description": "Full 64-char SHA-256 hex content_hash of the target memory. Required for create/delete."
                    },
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
                    "old_id": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash of the memory being superseded. Do not pass truncated display/log prefixes."
                    },
                    "new_id": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash of the newer memory. Do not pass truncated display/log prefixes."
                    },
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

    fn test_handle() -> ServiceHandle {
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        ServiceHandle { tx }
    }

    fn mcp_headers() -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            header::ACCEPT,
            "application/json, text/event-stream".parse().unwrap(),
        );
        headers
    }

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

        // serverInfo.version must lead with the crate version so the
        // SHA-qualified form stays `<semver>+<sha>` (#70) — this catches a
        // regression to a bare SHA or a SHA-first format. The qualification
        // itself is tested in build_info::tests::qualify_appends_sha_as_build_metadata,
        // since a test binary compiles with no ALAYA_GIT_SHA to qualify with.
        let reported = v["result"]["serverInfo"]["version"]
            .as_str()
            .expect("serverInfo.version must be a string");
        assert!(
            reported.starts_with(crate::build_info::version()),
            "serverInfo.version must begin with the crate version, got {reported}"
        );

        // instructions must carry the cross-tool hash convention
        let instructions = v["result"]["instructions"]
            .as_str()
            .expect("initialize result must include instructions");
        assert!(instructions.contains("content_hash"));
        assert!(
            instructions.contains("64"),
            "instructions must state the full hash length"
        );
        assert!(
            instructions.contains("get_memory"),
            "instructions must mention the retrieval tool"
        );
    }

    #[test]
    fn tools_list_returns_10_tools() {
        let resp = handle_tools_list(json!(2));
        let v = serde_json::to_value(&resp).unwrap();
        let tools = v["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 10);
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
        assert!(names.contains(&"get_memory"));
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

    #[tokio::test]
    async fn notifications_initialized_returns_empty_accepted() {
        let response = mcp_handler(
            mcp_headers(),
            axum::Extension(AuthPrincipal::Static),
            axum::extract::State(test_handle()),
            axum::body::Bytes::from(
                serde_json::to_vec(&json!({
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {}
                }))
                .unwrap(),
            ),
        )
        .await;

        assert_eq!(response.status(), StatusCode::ACCEPTED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(body.is_empty());
    }

    #[test]
    fn accepted_notifications_include_spec_initialized_method() {
        assert!(is_accepted_notification(&Value::Null, "initialized"));
        assert!(is_accepted_notification(
            &Value::Null,
            "notifications/initialized"
        ));
        assert!(is_accepted_notification(
            &Value::Null,
            "notifications/cancelled"
        ));
        assert!(!is_accepted_notification(
            &json!(1),
            "notifications/initialized"
        ));
        assert!(!is_accepted_notification(&Value::Null, "tools/list"));
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

    #[test]
    fn hash_fields_enforce_full_sha256_length() {
        // Regression guard: clients have repeatedly sent truncated 7/8/11-char
        // hashes from log displays. minLength/maxLength + a lowercase-hex
        // pattern on the schema are the first line of defense before requests
        // hit the server-side validator (validate_content_hash).
        const HEX64: &str = "^[0-9a-f]{64}$";
        let schemas = tool_schemas();
        let by_name = |name: &str| -> Value {
            schemas
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == name)
                .unwrap_or_else(|| panic!("tool {name} missing"))
                .clone()
        };

        for (tool, field) in [
            ("delete_memory", "content_hash"),
            ("get_memory", "content_hash"),
            ("relation", "content_hash"),
            ("memory_supersede", "old_id"),
            ("memory_supersede", "new_id"),
        ] {
            let schema = by_name(tool);
            let prop = &schema["inputSchema"]["properties"][field];
            assert_eq!(
                prop["minLength"], 64,
                "{tool}.{field} must enforce minLength: 64"
            );
            assert_eq!(
                prop["maxLength"], 64,
                "{tool}.{field} must enforce maxLength: 64"
            );
            assert_eq!(
                prop["pattern"], HEX64,
                "{tool}.{field} must enforce lowercase-hex pattern"
            );
        }

        // target_hash is nullable so the constraint lives inside anyOf[0]
        let relation = by_name("relation");
        let target = &relation["inputSchema"]["properties"]["target_hash"]["anyOf"][0];
        assert_eq!(target["minLength"], 64, "target_hash must enforce length");
        assert_eq!(target["pattern"], HEX64, "target_hash must enforce pattern");
        assert_eq!(target["maxLength"], 64, "target_hash must enforce length");
    }
}
