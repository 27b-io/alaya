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
use alaya_core::service::{
    MAX_UNSUPERSEDE_REASON_LEN, OutputMode, Refutation, RelationParams, SearchParams, StoreParams,
};
use alaya_types::memory::{MAX_CITATION_LEN, MAX_CITATIONS, validate_content_hash};

use alaya_types::graph::Resolution;

use crate::auth::{AuthPrincipal, WritePolicy};
use crate::{CmdInner, ServiceHandle, require_present};

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
struct UnsupersedeParams {
    content_hash: String,
    reason: String,
}

#[derive(Deserialize)]
struct ConfirmParams {
    content_hash: String,
}

#[derive(Deserialize)]
struct ContradictionsParams {
    #[serde(default = "default_contradictions_limit")]
    limit: usize,
    #[serde(default)]
    include_resolved: bool,
    #[serde(default)]
    verdicts: Option<Vec<String>>,
    #[serde(default)]
    offset: usize,
}
fn default_contradictions_limit() -> usize {
    20
}

/// `resolution` is required on the wire: `"keep_both"` stamps, explicit
/// `null` clears; an absent key is a -32602, never a silent clear.
#[derive(Deserialize)]
struct ResolveContradictionParams {
    memory_a_hash: String,
    memory_b_hash: String,
    #[serde(deserialize_with = "require_present")]
    resolution: Option<Resolution>,
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

    let method = record_rpc_span(&req);

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
    if is_accepted_notification(&id, method) {
        return StatusCode::ACCEPTED.into_response();
    }

    // Default-deny authorization: a restricted principal may only call its
    // allowlisted tools. Enforced here, axum-side, before any channel dispatch.
    if method == "tools/call" {
        let tool = req
            .params
            .as_ref()
            .and_then(|p| p.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("");
        if !principal.allows(tool) {
            tracing::debug!(?principal, tool = %alaya_backends::clip_for_log(tool), "authorization denied");
            let resp = JsonRpcResponse::error(id, -32001, "forbidden for this principal");
            return make_response(resp, wants_sse);
        }
    }

    // Dispatch on the bounded name, so a method missing from
    // `KNOWN_METHODS` is unreachable rather than silently traced `unknown`.
    let resp = match method {
        "initialize" => handle_initialize(id),
        "tools/list" => handle_tools_list(id),
        "tools/call" => handle_tools_call(id, req.params, &handle, principal).await,
        "ping" => JsonRpcResponse::success(id, json!({})),
        _ => JsonRpcResponse::error(id, -32601, format!("Method not found: {}", req.method)),
    };

    make_response(resp, wants_sse)
}

/// Methods this endpoint answers; anything else is `unknown`, so neither a
/// span name nor `mcp_handler`'s dispatch ever sees a client's string.
const KNOWN_METHODS: &[&str] = &[
    "initialize",
    "initialized",
    "notifications/initialized",
    "notifications/cancelled",
    "tools/list",
    "tools/call",
    "ping",
];

/// Every tool `tools/list` advertises; pinned to `tool_schemas` by a test.
const TOOL_NAMES: &[&str] = &[
    "store_memory",
    "search",
    "get_memory",
    "delete_memory",
    "check_database_health",
    "relation",
    "memory_supersede",
    "memory_unsupersede",
    "memory_confirm",
    "memory_refute",
    "memory_contradictions",
    "resolve_contradiction",
    "find_duplicates",
    "merge_duplicates",
];

fn known(table: &[&'static str], s: &str) -> &'static str {
    table
        .iter()
        .find(|k| **k == s)
        .copied()
        .unwrap_or("unknown")
}

/// Name the request span (`request_span` in `main.rs`, current here) per the
/// OTel MCP conventions: `{mcp.method.name}`, or `tools/call {tool}`. Only
/// fixed tokens are recorded — never the arguments. Returns the bounded
/// method name the handler dispatches on.
fn record_rpc_span(req: &JsonRpcRequest) -> &'static str {
    let span = tracing::Span::current();
    let method = known(KNOWN_METHODS, &req.method);
    span.record("mcp.method.name", method);
    if method != "tools/call" {
        span.record("otel.name", method);
        return method;
    }
    let tool = req
        .params
        .as_ref()
        .and_then(|p| p.get("name"))
        .and_then(Value::as_str)
        .map_or("unknown", |n| known(TOOL_NAMES, n));
    span.record("gen_ai.tool.name", tool);
    span.record("otel.name", format!("tools/call {tool}"));
    method
}

/// Format response as SSE or plain JSON depending on client preference, and
/// record a JSON-RPC error's code on the request span — the code only: the
/// message can echo the client's method or tool name.
fn make_response(resp: JsonRpcResponse, sse: bool) -> Response {
    if let Some(e) = &resp.error {
        tracing::Span::current().record("rpc.response.status_code", e.code.to_string());
    }
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

Memory identifiers: every memory is keyed by `content_hash`, a full 64-character SHA-256 hex string. `store_memory` and `search` return this hash on every result — pass it back verbatim to `get_memory`, `memory_supersede`, `memory_unsupersede`, `memory_confirm`, `memory_refute`, `delete_memory`, `relation`, and `merge_duplicates`. Never truncate or abbreviate it; the 8-character prefixes shown in log lines and display output are not valid identifiers and will be rejected.

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
    let read_only = WritePolicy::read_only_for(principal);
    match name {
        "store_memory" => {
            let params: StoreParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            params
                .validate()
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
                        as_of: None,
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
        "memory_unsupersede" => {
            let p: UnsupersedeParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Unsupersede {
                        hash: p.content_hash,
                        reason: p.reason,
                        // Fixed, like resolve_contradiction's: a tool call
                        // cannot record itself as the console or a person.
                        via: "operator:mcp".to_string(),
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "memory_confirm" => {
            let p: ConfirmParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            if !validate_content_hash(&p.content_hash) {
                return Err((-32602, "Invalid params: invalid content_hash format".into()));
            }
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Confirm {
                        hash: p.content_hash,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "memory_refute" => {
            // `content_hash` names the wrong memory; the rest of the
            // arguments are the correction, refused whole on an unknown key.
            let mut args = args;
            let hash = args
                .as_object_mut()
                .and_then(|o| o.remove("content_hash"))
                .and_then(|v| v.as_str().map(String::from))
                .ok_or_else(|| {
                    (
                        -32602,
                        "Invalid params: content_hash is required".to_string(),
                    )
                })?;
            if !validate_content_hash(&hash) {
                return Err((-32602, "Invalid params: invalid content_hash format".into()));
            }
            let refutation: Refutation = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            refutation
                .validate()
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::Refute {
                        hash,
                        refutation,
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
                        offset: p.offset,
                        include_resolved: p.include_resolved,
                        verdicts: p.verdicts,
                        reply: tx,
                    },
                    rx,
                )
                .await
        }
        "resolve_contradiction" => {
            let p: ResolveContradictionParams = serde_json::from_value(args)
                .map_err(|e| (-32602, format!("Invalid params: {e}")))?;
            let (tx, rx) = oneshot::channel();
            handle
                .call_rpc(
                    CmdInner::ResolveContradiction {
                        memory_a_hash: p.memory_a_hash,
                        memory_b_hash: p.memory_b_hash,
                        resolution: p.resolution,
                        // The MCP surface is the operator's: fixed tag, not
                        // caller-supplied, so a tool call cannot impersonate
                        // the console or an engine run.
                        resolved_via: "operator:mcp".to_string(),
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

/// The `citations` input of store_memory and memory_refute.
fn citations_schema() -> Value {
    json!({
        "anyOf": [
            {
                "type": "array",
                "maxItems": MAX_CITATIONS,
                "items": { "type": "string", "maxLength": MAX_CITATION_LEN }
            },
            { "type": "null" }
        ],
        "description": "The sources this memory rests on, returned on every hit at metadata.citations. Each is <owner>/<repo>@<sha>:<path>, optionally :<line> or :<line>-<line>, with a 7-40 char hex sha, or an https:// URL with a host (a PR, an issue comment). A malformed one refuses the call."
    })
}

fn tool_schemas() -> Value {
    json!([
        {
            "name": "store_memory",
            "description": "Store a new memory for future semantic retrieval. Content is vectorized for similarity search. Salience scoring and contradiction detection are computed automatically. Returns the new memory's full 64-char content_hash — the identifier used by get_memory, memory_supersede, delete_memory, and relation — and `neighbours`: up to 5 nearest live memories ({content_hash, similarity, memory_type, summary}), absent when no neighbour search ran.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Text to store (embedded for semantic search)" },
                    "tags": { "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}, {"type": "null"}], "description": "Labels — accepts [\"tag1\", \"tag2\"] or \"tag1,tag2\"" },
                    "memory_type": { "type": "string", "enum": ["note", "decision", "task", "reference"], "default": "note" },
                    "metadata": { "anyOf": [{"type": "object"}, {"type": "null"}], "description": "Structured data. Special key: importance (float 0.0-1.0)" },
                    "client_hostname": { "anyOf": [{"type": "string"}, {"type": "null"}] },
                    "summary": { "anyOf": [{"type": "string"}, {"type": "null"}], "description": "One-line summary (~50 tokens). Auto-generated if omitted." },
                    "dedup_threshold": { "anyOf": [{"type": "number"}, {"type": "null"}], "description": "If set, skip storage when nearest neighbor similarity >= threshold (0.0-1.0)" },
                    "citations": citations_schema()
                },
                "required": ["content"]
            }
        },
        {
            "name": "search",
            "description": "Search and retrieve memories. Consolidates all retrieval modes into one tool. Each result includes a full 64-char content_hash — pass it to get_memory for an exact re-fetch, or to memory_supersede/delete_memory/relation. A result may carry newer_related: [{content_hash, verdict_confidence}], newer memories the contradiction judge (an LLM) picked over this one. Nobody has reviewed that pick. Read both memories: this one often holds detail the newer one lacks.",
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
            "description": "Retrieve a single memory by its exact content_hash. Returns {\"found\": true, \"memory\": {...}} or {\"found\": false}. Use this to inspect a memory before supersede/delete/relation when you already hold its hash — e.g. from search results or a memory_contradictions report. Superseded memories are returned; check metadata.superseded_by. With full or both output, memory.supersession_reason says why it was superseded, and memory.supersession_log lists the supersessions reversed by memory_unsupersede, oldest first (who, when, why); absent when none. memory.newer_related, when present, lists newer memories the contradiction judge (an LLM) picked over this one, unreviewed: read both.",
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
            "description": "Mark one memory as superseded by another, resolving a contradiction destructively: the old memory leaves default search results (it stays retrievable with an audit trail) and every pair it is in leaves the contradiction queue. Use only when one memory misleads. When both are true (verdict coexist) or the pair is detector noise (verdict unrelated), call resolve_contradiction with resolution keep_both instead — nothing is superseded. A wrong supersession is reversed with memory_unsupersede.",
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
            "name": "memory_unsupersede",
            "description": "Reverse a memory_supersede (or merge_duplicates) that was wrong: the memory returns to default search results and its contradiction pairs return to the queue. The reversal is recorded on the memory with your reason, the prior superseded_by and supersession_reason; the pair with the former survivor is stamped keep_both (resolved_via unsupersede; a pair already stamped keeps its stamp) so the judge never re-applies it. Only that memory changes: in a chain A→B→C, unsuperseding B leaves A superseded by B. A memory that is not superseded returns {success: false, status: not_superseded}; one superseded again while this ran returns {success: false, status: superseded_by_changed}. Recorded as unsuperseded_via operator:mcp.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash of the superseded memory to restore. Do not pass truncated display/log prefixes."
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_UNSUPERSEDE_REASON_LEN,
                        "description": "Why the supersession was wrong. Required: it is the audit record."
                    }
                },
                "required": ["content_hash", "reason"]
            }
        },
        {
            "name": "memory_confirm",
            "description": "Record that you re-checked a memory against its sources and it still holds: sets metadata.last_confirmed_at to the server's time, which every hit then returns. Nothing else changes; it has no effect on ranking. A missing memory returns {success: false, status: not_found}; a superseded one returns {success: false, status: superseded, superseded_by}: check the survivor instead.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash of the memory you re-checked. Do not pass truncated display/log prefixes."
                    }
                },
                "required": ["content_hash"]
            }
        },
        {
            "name": "memory_refute",
            "description": "File a correction against a wrong memory in one call. The correction is stored like store_memory (never skipped as a duplicate, never overwriting a memory); a live memory already holding its exact text is reused (existing: true), and any tags, memory_type, summary or citations you send must match it. By default the pair is linked correction -> wrong as CONTRADICTS: both stay searchable and the pair enters the contradiction queue, unless it was settled already (keep_both). Pass supersede: true only when the correction carries every still-true claim of the wrong memory; it is then superseded by the correction. Your reason is stored either way, as supersession_reason or as the link's refute_reason, and returned; a retry with another reason is a conflict naming the stored one. Every outcome is a typed status. Nothing written: not_found, superseded (refute its survivor), correction_superseded, correction_exists, correction_unreadable, correction_deleted. After the store: conflict, correction_superseded, correction_deleted, or partial with failed_step (retry the same refute; a failed supersede can also be finished with memory_supersede).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "Full 64-char SHA-256 hex content_hash of the wrong memory. Do not pass truncated display/log prefixes."
                    },
                    "content": { "type": "string", "minLength": 1, "description": "The correction's text" },
                    "tags": { "anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "string"}, {"type": "null"}], "description": "Labels — accepts [\"tag1\", \"tag2\"] or \"tag1,tag2\"" },
                    "memory_type": { "type": "string", "enum": ["note", "decision", "task", "reference"], "default": "note" },
                    "summary": { "anyOf": [{"type": "string"}, {"type": "null"}], "description": "One-line summary. Auto-generated if omitted." },
                    "citations": citations_schema(),
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": MAX_UNSUPERSEDE_REASON_LEN,
                        "description": "Why the memory is wrong. Required: it is the audit record."
                    },
                    "supersede": { "type": "boolean", "default": false, "description": "Supersede the wrong memory instead of linking it. Only when the correction carries every still-true claim of it." }
                },
                "required": ["content_hash", "content", "reason"],
                "additionalProperties": false
            }
        },
        {
            "name": "memory_contradictions",
            "description": "List CONTRADICTS pairs with the judge's verdict per pair (contradiction | supersession | coexist | unrelated | unjudged), the recommended survivor hash and a one-line reason. Pairs with a superseded endpoint, or stamped keep_both (by resolve_contradiction, or by memory_unsupersede as resolved_via unsupersede), are hidden unless include_resolved is true. Returns {pairs: [{memory_a_hash, memory_b_hash, confidence, created_at, memory_a_content, memory_b_content, memory_a_superseded, memory_b_superseded, verdict, verdict_reason, survivor, verdict_confidence, verdict_model, judged_at, resolution, resolved_at, resolved_via}], total, next_offset}. Page with offset = next_offset until it is null. To act on a pair: memory_supersede when one memory misleads (verdict supersession/contradiction); resolve_contradiction keep_both when both are true or the pair is noise (verdict coexist/unrelated) — non-destructive and reversible.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "default": 20, "description": "Max contradiction pairs to fetch (1-500)" },
                    "offset": { "type": "integer", "default": 0, "description": "Pairs to skip; pass the previous response's next_offset to page" },
                    "include_resolved": { "type": "boolean", "default": false, "description": "Also return resolved pairs: one endpoint superseded, or the pair stamped keep_both" },
                    "verdicts": { "type": "array", "items": { "type": "string", "enum": alaya_types::graph::Verdict::ALL.map(|v| v.as_str()) }, "description": "Only pairs with these verdicts. Default: contradiction, supersession, unjudged" }
                }
            }
        },
        {
            "name": "resolve_contradiction",
            "description": "Resolve a CONTRADICTS pair from memory_contradictions without superseding or deleting anything. resolution keep_both stamps the pair as settled (both memories are true, or the pair is detector noise): it leaves the default queue, both memories stay searchable, the judge's verdict stays on the edge. resolution null clears the stamp and the pair returns to the queue. Recorded as resolved_via operator:mcp with a server-set resolved_at. Pass memory_a_hash and memory_b_hash exactly as memory_contradictions returned them (the edge is directed). Prefer this over memory_supersede for verdict coexist or unrelated.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "memory_a_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "memory_a_hash of the pair, verbatim from memory_contradictions. Do not pass truncated display/log prefixes."
                    },
                    "memory_b_hash": {
                        "type": "string",
                        "minLength": 64,
                        "maxLength": 64,
                        "pattern": "^[0-9a-f]{64}$",
                        "description": "memory_b_hash of the pair, verbatim from memory_contradictions. Do not pass truncated display/log prefixes."
                    },
                    "resolution": {
                        "anyOf": [
                            { "type": "string", "enum": alaya_types::graph::Resolution::ALL.map(|r| r.as_str()) },
                            { "type": "null" }
                        ],
                        "description": "keep_both to resolve; null to undo a previous keep_both. Required — an omitted key is rejected, never treated as a clear."
                    }
                },
                "required": ["memory_a_hash", "memory_b_hash", "resolution"]
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

    /// A span names only tools `tools/list` advertises; a tool added there
    /// without joining `TOOL_NAMES` would trace as `tools/call unknown`.
    #[test]
    fn tool_names_match_tool_schemas() {
        let schemas = tool_schemas();
        let advertised: Vec<&str> = schemas
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap())
            .collect();
        assert_eq!(advertised, TOOL_NAMES);
    }

    /// A caller-set supersession marker or novelty value is -32602 and never
    /// reaches the worker, so nothing is stored (LAB-6891).
    #[tokio::test]
    async fn store_memory_refuses_reserved_metadata_key() {
        for key in ["superseded_by", "nearest_similarity"] {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let handle = ServiceHandle { tx };
            let args = json!({
                "content": "live fact",
                "metadata": {key: "b".repeat(64)},
            });
            let (code, msg) = dispatch_tool("store_memory", args, &handle, AuthPrincipal::Static)
                .await
                .expect_err("reserved key must be refused");
            assert_eq!(code, -32602);
            assert!(msg.contains(&format!("metadata.{key}")), "{msg}");
            assert!(rx.try_recv().is_err(), "nothing may be dispatched");
        }
    }

    /// Malformed citations and a caller-set `last_confirmed_at` are -32602
    /// on store, as are a bad hash or correction on confirm and refute; none
    /// reaches the worker (LAB-7721).
    #[tokio::test]
    async fn citation_and_verb_refusals_are_invalid_params_and_never_dispatch() {
        let hash = "a".repeat(64);
        let cases = [
            (
                "store_memory",
                json!({"content": "x", "citations": ["see PR"]}),
            ),
            (
                "store_memory",
                json!({"content": "x", "metadata": {"citations": ["see PR"]}}),
            ),
            (
                "store_memory",
                json!({"content": "x", "metadata": {"last_confirmed_at": 1.0}}),
            ),
            ("memory_confirm", json!({"content_hash": "abc"})),
            ("memory_confirm", json!({})),
            ("memory_refute", json!({"content": "x", "reason": "y"})),
            (
                "memory_refute",
                json!({"content_hash": "abc", "content": "x", "reason": "y"}),
            ),
            (
                "memory_refute",
                json!({"content_hash": hash, "content": "x"}),
            ),
            (
                "memory_refute",
                json!({"content_hash": hash, "content": "x", "reason": "y", "metadata": {}}),
            ),
            (
                "memory_refute",
                json!({"content_hash": hash, "content": "x", "reason": "y", "citations": ["see PR"]}),
            ),
        ];
        for (tool, args) in cases {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let handle = ServiceHandle { tx };
            let (code, msg) = dispatch_tool(tool, args.clone(), &handle, AuthPrincipal::Static)
                .await
                .expect_err("refused");
            assert_eq!(code, -32602, "{tool} {args}: {msg}");
            assert!(rx.try_recv().is_err(), "{tool} {args} was dispatched");
        }
    }

    #[tokio::test]
    async fn refute_dispatches_the_hash_apart_from_the_correction() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let handle = ServiceHandle { tx };
        let hash = "a".repeat(64);
        let args =
            json!({"content_hash": hash, "content": "fixed", "reason": "wrong", "tags": "a,b"});
        let call = tokio::spawn(async move {
            dispatch_tool("memory_refute", args, &handle, AuthPrincipal::Static).await
        });
        let cmd = rx.recv().await.expect("dispatched");
        let CmdInner::Refute {
            hash: h,
            refutation,
            reply,
        } = cmd.inner
        else {
            panic!("expected CmdInner::Refute");
        };
        assert_eq!(h, hash);
        assert_eq!(refutation.content, "fixed");
        assert!(!refutation.supersede, "link is the default");
        assert_eq!(
            refutation.tags,
            Some(vec!["a".to_string(), "b".to_string()])
        );
        // The link's reason comes back as the worker answers it (LAB-9265).
        let linked = json!({"success": true, "status": "contradicts", "refute_reason": "wrong"});
        reply.send(linked.clone()).unwrap();
        assert_eq!(call.await.unwrap().unwrap(), linked);
    }

    /// A link already filed with another reason is a typed result, not a
    /// JSON-RPC error: the stored reason reaches the caller (LAB-9265).
    #[tokio::test]
    async fn refute_link_reason_conflict_is_a_result_carrying_the_stored_reason() {
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        let handle = ServiceHandle { tx };
        let hash = "a".repeat(64);
        let args = json!({"content_hash": hash, "content": "fixed", "reason": "new why"});
        let call = tokio::spawn(async move {
            dispatch_tool("memory_refute", args, &handle, AuthPrincipal::Static).await
        });
        let CmdInner::Refute { reply, .. } = rx.recv().await.expect("dispatched").inner else {
            panic!("expected CmdInner::Refute");
        };
        let conflict = json!({
            "success": false,
            "status": "conflict",
            "correction_hash": "b".repeat(64),
            "refute_reason": "first why",
        });
        reply.send(conflict.clone()).unwrap();
        assert_eq!(call.await.unwrap().unwrap(), conflict);
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
    fn tools_list_returns_14_tools() {
        let resp = handle_tools_list(json!(2));
        let v = serde_json::to_value(&resp).unwrap();
        let tools = v["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 14);
    }

    #[test]
    fn resolve_contradiction_params_require_resolution_key_and_list_it_required() {
        let a = "a".repeat(64);
        let b = "b".repeat(64);
        let set: ResolveContradictionParams = serde_json::from_value(
            json!({"memory_a_hash": a, "memory_b_hash": b, "resolution": "keep_both"}),
        )
        .unwrap();
        assert_eq!(set.resolution, Some(Resolution::KeepBoth));
        let clear: ResolveContradictionParams = serde_json::from_value(
            json!({"memory_a_hash": a, "memory_b_hash": b, "resolution": null}),
        )
        .unwrap();
        assert_eq!(clear.resolution, None);
        assert!(
            serde_json::from_value::<ResolveContradictionParams>(
                json!({"memory_a_hash": a, "memory_b_hash": b})
            )
            .is_err(),
            "an absent key must not read as a clear"
        );
        assert!(
            serde_json::from_value::<ResolveContradictionParams>(
                json!({"memory_a_hash": a, "memory_b_hash": b, "resolution": "keep_both", "resolved_via": "engine:x"})
            )
            .is_ok(),
            "an unknown key is ignored; the MCP surface fixes resolved_via itself"
        );

        let schema = tool_schemas()
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "resolve_contradiction")
            .cloned()
            .unwrap();
        assert!(
            schema["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .contains(&json!("resolution"))
        );
        assert_eq!(
            schema["inputSchema"]["properties"]["resolution"]["anyOf"][0]["enum"],
            json!(["keep_both"])
        );
        // The two tool descriptions name the non-destructive path.
        for name in ["memory_contradictions", "memory_supersede"] {
            let d = tool_schemas()
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == name)
                .unwrap()["description"]
                .as_str()
                .unwrap()
                .to_string();
            assert!(d.contains("resolve_contradiction"), "{name}: {d}");
            assert!(d.contains("keep_both"), "{name}: {d}");
        }
    }

    /// The reason is the audit record, so the wire requires it; who reversed
    /// it is fixed by the surface, never taken from the caller.
    #[test]
    fn unsupersede_params_require_reason_and_ignore_a_caller_via() {
        let h = "a".repeat(64);
        assert!(
            serde_json::from_value::<UnsupersedeParams>(json!({"content_hash": h})).is_err(),
            "a missing reason is -32602"
        );
        let p: UnsupersedeParams = serde_json::from_value(
            json!({"content_hash": h, "reason": "wrong merge", "unsuperseded_via": "operator:console"}),
        )
        .unwrap();
        assert_eq!(p.reason, "wrong merge");

        let schema = tool_schemas()
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "memory_unsupersede")
            .cloned()
            .unwrap();
        assert_eq!(
            schema["inputSchema"]["required"],
            json!(["content_hash", "reason"])
        );
        assert!(
            schema["inputSchema"]["properties"]
                .get("unsuperseded_via")
                .is_none()
        );
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

    /// The handler dispatches on `KNOWN_METHODS`' bounded name, so a method
    /// left out of that table answers -32601 (or a notification stops being
    /// accepted) here instead of only tracing as `unknown`.
    #[tokio::test]
    async fn every_answered_method_is_in_known_methods() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        let call = |body: Value| {
            mcp_handler(
                headers.clone(),
                axum::Extension(AuthPrincipal::Static),
                axum::extract::State(test_handle()),
                axum::body::Bytes::from(body.to_string()),
            )
        };

        for method in ["initialize", "tools/list", "ping"] {
            let resp = call(json!({ "jsonrpc": "2.0", "id": 1, "method": method })).await;
            let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
                .await
                .unwrap();
            let v: Value = serde_json::from_slice(&body).unwrap();
            assert!(v.get("result").is_some(), "{method}: {v}");
        }
        for method in [
            "initialized",
            "notifications/initialized",
            "notifications/cancelled",
        ] {
            let resp = call(json!({ "jsonrpc": "2.0", "method": method })).await;
            assert_eq!(resp.status(), StatusCode::ACCEPTED, "{method}");
        }
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

    /// `as_of` is a REST-only replay field: the MCP `search` tool neither
    /// lists it nor reads it (`SearchParams` has no such field).
    #[test]
    fn search_schema_does_not_offer_as_of() {
        let schemas = tool_schemas();
        let search = schemas
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "search")
            .unwrap();
        assert!(search["inputSchema"]["properties"].get("as_of").is_none());
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
            ("memory_unsupersede", "content_hash"),
            ("memory_confirm", "content_hash"),
            ("memory_refute", "content_hash"),
            ("resolve_contradiction", "memory_a_hash"),
            ("resolve_contradiction", "memory_b_hash"),
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
