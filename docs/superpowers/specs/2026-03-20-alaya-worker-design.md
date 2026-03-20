# alaya-worker: Cloudflare Worker Entry Point

*Design spec for the WASM Worker deployment of Ālaya memory service.*

## Context

Ālaya's core crates (alaya-types, alaya-backends, alaya-core) compile to `wasm32-unknown-unknown`. A native server (alaya-server) is deployed on lab k3s. This spec adds a Cloudflare Worker entry point — full-stack MemoryService running at the edge with `worker::Fetch` as the HTTP transport.

The `worker::Fetch` pattern is proven in cachekit-rs (`crates/cachekit/src/backend/workers.rs`), which implements the same trait-with-conditional-compilation approach on `worker = "0.4"`.

## Architecture

### HttpClient Trait

New module `alaya-backends/src/http.rs` — a minimal HTTP abstraction:

```rust
#[async_trait(?Send)]
pub trait HttpClient {
    async fn request(
        &self,
        method: Method,
        url: &str,
        headers: &[(&str, &str)],
        body: Option<&[u8]>,
    ) -> Result<HttpResponse>;
}

pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

pub enum Method { Get, Post, Put, Delete, Head }
```

Two implementations, gated by `cfg(target_arch)`:

| Impl | Target | HTTP primitive | Feature flag |
|------|--------|----------------|-------------|
| `ReqwestHttpClient` | native | `reqwest::Client` | default |
| `WorkerHttpClient` | wasm32 | `worker::Fetch` | `workers` |

`WorkerHttpClient` maps all five HTTP methods including `Method::Post` (the cachekit-rs reference only needed GET/PUT/DELETE/HEAD — Ālaya uses POST extensively for Qdrant search, bridge calls, and embeddings).

### Backend Client Refactor

All three clients (`QdrantClient`, `EmbeddingClient`, `GraphHttpClient`) change from owning a `reqwest::Client` to accepting `Arc<dyn HttpClient>`:

```rust
use std::sync::Arc;

pub struct QdrantClient {
    http: Arc<dyn HttpClient>,  // was: reqwest::Client
    base_url: String,
    collection: String,
}
```

`Arc<dyn HttpClient>` instead of `Box<dyn HttpClient>` because multiple backend clients share the same HTTP client instance. `Arc::clone()` is cheap; on Workers (single-threaded) the atomic refcount cost is zero.

HTTP call sites change from reqwest convenience methods to explicit serialization:

```rust
// Before: self.client.post(url).json(&body).send().await
// After:
let bytes = serde_json::to_vec(&body)?;
let resp = self.http.request(
    Method::Post, &url,
    &[("content-type", "application/json")],
    Some(&bytes),
).await?;
let data: T = serde_json::from_slice(&resp.body)?;
```

The logic (filter construction, UUID generation, response parsing) stays identical.

### MCP Dispatch Extraction

Move reusable MCP protocol logic from `alaya-server/src/mcp.rs` into `alaya-core/src/mcp.rs`:

- `tool_schemas() -> Value` — 9 tool JSON schemas (pure data)
- `dispatch_tool_direct(svc: &MemoryService, name: &str, args: Value) -> Result<Value>` — direct tool dispatch, calls MemoryService methods synchronously. Used by Worker.
- `format_jsonrpc_response(id: Value, result: Result<Value>) -> Vec<u8>` — JSON-RPC 2.0 response formatting
- `parse_jsonrpc_request(body: &[u8]) -> Result<(Value, String, Option<Value>)>` — parse id, method, params

`alaya-server` keeps its channel-based `dispatch_tool` wrapper that sends `Cmd` over mpsc — it calls `dispatch_tool_direct` inside the `service_worker` loop. `alaya-worker` calls `dispatch_tool_direct` directly (single-threaded, no channel needed).

### Worker Entry Point

New crate `crates/alaya-worker/`:

```rust
use worker::*;
use std::sync::Arc;

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    if req.method() != Method::Post || req.path() != "/mcp" {
        return Response::error("Not Found", 404);
    }

    let http: Arc<dyn HttpClient> = Arc::new(WorkerHttpClient::new());

    let qdrant_url = env.var("QDRANT_URL")?.to_string();
    let collection = env.var("QDRANT_COLLECTION")
        .map(|v| v.to_string())
        .unwrap_or_else(|_| "memories_arctic1024".into());
    let embedding_url = env.var("EMBEDDING_URL")?.to_string();
    let graph_url = env.var("GRAPH_URL")?.to_string();
    let graph_api_key = env.var("GRAPH_API_KEY")
        .map(|v| v.to_string())
        .unwrap_or_default();

    let qdrant = QdrantClient::new(http.clone(), qdrant_url, collection, None);
    let embeddings = EmbeddingClient::new(http.clone(), embedding_url, ...);
    let graph1 = GraphHttpClient::new(http.clone(), graph_url.clone(), &graph_api_key);
    let graph2 = GraphHttpClient::new(http.clone(), graph_url.clone(), &graph_api_key);
    let graph3 = GraphHttpClient::new(http.clone(), graph_url, &graph_api_key);

    let svc = MemoryService::new(
        Box::new(qdrant),
        Box::new(embeddings),
        Box::new(graph1),   // GraphService
        Box::new(graph2),   // HebbianService
        Box::new(graph3),   // ConsolidationService
    );

    let body = req.bytes().await?;
    let wants_sse = req.headers().get("Accept")
        .map(|a| a.contains("text/event-stream"))
        .unwrap_or(false);

    let (id, method, params) = parse_jsonrpc_request(&body)?;
    let result = match method.as_str() {
        "initialize" => Ok(initialize_response()),
        "tools/list" => Ok(tool_schemas_response()),
        "tools/call" => dispatch_tool_direct(&svc, params).await,
        "ping" => Ok(json!({})),
        _ => Err(method_not_found(&method)),
    };

    let response_bytes = format_jsonrpc_response(id, result, wants_sse);
    Response::from_bytes(response_bytes)
}
```

MemoryService is constructed per-request. This is cheap — no connection pools, no state, just struct allocation with `Arc` references to the shared HTTP client. The Worker runtime reuses the isolate across requests, but we don't rely on that.

Config from Worker environment bindings (same var names as alaya-server):
- `QDRANT_URL`, `QDRANT_COLLECTION`, `EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `GRAPH_URL`, `GRAPH_API_KEY`

### Timestamp Utility

`std::time::SystemTime::now()` appears in both `alaya-core/src/service.rs` and `alaya-backends/src/qdrant.rs` (in `increment_access_count`). Extract a shared utility into `alaya-types/src/time.rs`:

```rust
pub fn current_timestamp() -> f64 {
    #[cfg(target_arch = "wasm32")]
    { js_sys::Date::now() / 1000.0 }

    #[cfg(not(target_arch = "wasm32"))]
    { std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64() }
}
```

Add `js-sys` as conditional dependency on `alaya-types`:
```toml
[target.'cfg(target_arch = "wasm32")'.dependencies]
js-sys = "0.3"
```

Both `alaya-core` and `alaya-backends` import `alaya_types::time::current_timestamp()`.

### Backend Topology

For Topology A (current): Worker → Cloudflare Tunnel → lab k3s backends.

```
CF Edge (Worker)
    │
    ├─→ tunnel → qdrant.mcp.svc:6333
    ├─→ tunnel → tei.mcp.svc:80
    └─→ tunnel → alaya-bridge.mcp.svc:3000
```

Tunnel hostnames configured in `wrangler.toml` vars. Backends unchanged.

## Scope

### In scope
- `HttpClient` trait + `ReqwestHttpClient` + `WorkerHttpClient`
- Refactor 3 backend clients to use `Arc<dyn HttpClient>`
- Extract MCP dispatch into `alaya-core/src/mcp.rs` (direct + channel variants)
- `alaya-worker` crate with `#[event(fetch)]` entry
- `current_timestamp()` utility in `alaya-types` with `js-sys` conditional dep
- `wrangler.toml` for deployment
- WASM compile gate in CI

### Out of scope
- cachekit-rs integration (separate workstream)
- Prajna integration
- OTLP tracing on Worker (no OTel WASM support)
- Durable Objects / KV state
- Authentication (tunnel provides network-level security for Topology A; if Worker is exposed on a public route, auth becomes critical — separate spec)

## Testing

- Unit: `HttpClient` implementations with mock responses
- Compile: `cargo build -p alaya-worker --target wasm32-unknown-unknown` in CI
- Existing: all 168 unit tests + 5 integration tests must keep passing after refactor
- Manual: `wrangler dev` against tunnel-exposed lab backends

## Files Changed

| File | Change |
|------|--------|
| `crates/alaya-types/src/time.rs` | **New** — `current_timestamp()` with cfg(wasm32) branch |
| `crates/alaya-types/src/lib.rs` | Add `pub mod time` |
| `crates/alaya-types/Cargo.toml` | Add `js-sys` conditional dep |
| `crates/alaya-backends/src/http.rs` | **New** — HttpClient trait, Method, HttpResponse |
| `crates/alaya-backends/src/http_reqwest.rs` | **New** — ReqwestHttpClient impl |
| `crates/alaya-backends/src/http_worker.rs` | **New** — WorkerHttpClient impl (cfg wasm32, feature workers) |
| `crates/alaya-backends/src/lib.rs` | Add `pub mod http` + conditional re-exports |
| `crates/alaya-backends/src/qdrant.rs` | Replace `reqwest::Client` with `Arc<dyn HttpClient>` |
| `crates/alaya-backends/src/embedding.rs` | Same refactor |
| `crates/alaya-backends/src/graph.rs` | Same refactor |
| `crates/alaya-backends/Cargo.toml` | Add `workers` feature flag + conditional deps |
| `crates/alaya-core/src/mcp.rs` | **New** — tool_schemas, dispatch_tool_direct, JSON-RPC helpers |
| `crates/alaya-core/src/service.rs` | Use `alaya_types::time::current_timestamp()` |
| `crates/alaya-core/src/lib.rs` | Add `pub mod mcp` |
| `crates/alaya-server/src/mcp.rs` | Delegate to `alaya_core::mcp` for schemas + dispatch |
| `crates/alaya-server/src/main.rs` | Update client construction with Arc<dyn HttpClient> |
| `crates/alaya-worker/Cargo.toml` | **New** — worker crate |
| `crates/alaya-worker/src/lib.rs` | **New** — Worker entry point |
| `Cargo.toml` | Add `worker` to workspace deps |
| `.github/workflows/ci.yml` | Add WASM compile check for alaya-worker |
| `wrangler.toml` | **New** — Worker deployment config |
