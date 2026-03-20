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

### Backend Client Refactor

All three clients (`QdrantClient`, `EmbeddingClient`, `GraphHttpClient`) change from owning a `reqwest::Client` to accepting `Box<dyn HttpClient>`:

```rust
pub struct QdrantClient {
    http: Box<dyn HttpClient>,  // was: reqwest::Client
    base_url: String,
    collection: String,
}
```

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

- `tool_schemas() -> Value` — 9 tool JSON schemas
- `handle_jsonrpc(svc: &MemoryService, body: &[u8], wants_sse: bool) -> Vec<u8>` — full JSON-RPC parse + dispatch + format
- `dispatch_tool(svc: &MemoryService, name: &str, args: Value) -> Result<Value>` — tool name → MemoryService method

`alaya-server` calls `handle_jsonrpc` via the channel bridge (unchanged behavior). `alaya-worker` calls it directly (single-threaded, no channel needed).

### Worker Entry Point

New crate `crates/alaya-worker/`:

```rust
use worker::*;

#[event(fetch)]
async fn fetch(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    if req.method() != Method::Post || req.path() != "/mcp" {
        return Response::error("Not Found", 404);
    }

    let http = Box::new(WorkerHttpClient::new());
    let qdrant = QdrantClient::new(http.clone(), env.var("QDRANT_URL")?, ...);
    let embeddings = EmbeddingClient::new(http.clone(), env.var("EMBEDDING_URL")?, ...);
    let graph = GraphHttpClient::new(http.clone(), env.var("GRAPH_URL")?, ...);
    let svc = MemoryService::new(Box::new(qdrant), Box::new(embeddings), ...);

    let body = req.bytes().await?;
    let wants_sse = req.headers().get("accept")?.contains("text/event-stream");
    let response_bytes = alaya_core::mcp::handle_jsonrpc(&svc, &body, wants_sse).await;

    Response::from_bytes(response_bytes)
}
```

Config from Worker environment bindings (same var names as alaya-server):
- `QDRANT_URL`, `QDRANT_COLLECTION`, `EMBEDDING_URL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSIONS`, `GRAPH_URL`, `GRAPH_API_KEY`

### SystemTime Fix

`current_timestamp()` in `alaya-core/src/service.rs` uses `std::time::SystemTime::now()`. On Workers, use `js_sys::Date::now()`:

```rust
fn current_timestamp() -> f64 {
    #[cfg(target_arch = "wasm32")]
    { js_sys::Date::now() / 1000.0 }

    #[cfg(not(target_arch = "wasm32"))]
    { std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64() }
}
```

Add `js-sys` as conditional dependency: `[target.'cfg(target_arch = "wasm32")'.dependencies] js-sys = "0.3"`.

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
- Refactor 3 backend clients to use `HttpClient`
- Extract MCP dispatch into `alaya-core/src/mcp.rs`
- `alaya-worker` crate with `#[event(fetch)]` entry
- `js-sys` conditional dep for WASM timestamps
- `wrangler.toml` for deployment
- WASM compile gate in CI

### Out of scope
- cachekit-rs integration (separate workstream)
- Prajna integration
- OTLP tracing on Worker (no OTel WASM support)
- Durable Objects / KV state
- Authentication (tunnel provides network-level security for Topology A)

## Testing

- Unit: `HttpClient` implementations with mock responses
- Compile: `cargo build -p alaya-worker --target wasm32-unknown-unknown` in CI
- Existing: all 168 unit tests + 5 integration tests must keep passing after refactor
- Manual: `wrangler dev` against tunnel-exposed lab backends

## Files Changed

| File | Change |
|------|--------|
| `crates/alaya-backends/src/http.rs` | **New** — HttpClient trait, Method, HttpResponse |
| `crates/alaya-backends/src/http_reqwest.rs` | **New** — ReqwestHttpClient impl |
| `crates/alaya-backends/src/http_worker.rs` | **New** — WorkerHttpClient impl (cfg wasm32) |
| `crates/alaya-backends/src/lib.rs` | Add `pub mod http` + conditional re-exports |
| `crates/alaya-backends/src/qdrant.rs` | Replace `reqwest::Client` with `Box<dyn HttpClient>` |
| `crates/alaya-backends/src/embedding.rs` | Same refactor |
| `crates/alaya-backends/src/graph.rs` | Same refactor |
| `crates/alaya-backends/Cargo.toml` | Add `worker` feature flag + conditional deps |
| `crates/alaya-core/src/mcp.rs` | **New** — extracted MCP dispatch logic |
| `crates/alaya-core/src/service.rs` | `current_timestamp()` cfg branch |
| `crates/alaya-core/src/lib.rs` | Add `pub mod mcp` |
| `crates/alaya-core/Cargo.toml` | Add `js-sys` conditional dep |
| `crates/alaya-server/src/mcp.rs` | Delegate to `alaya_core::mcp` |
| `crates/alaya-server/src/main.rs` | Update client construction |
| `crates/alaya-worker/Cargo.toml` | **New** — worker crate |
| `crates/alaya-worker/src/lib.rs` | **New** — Worker entry point |
| `Cargo.toml` | Add `worker` to workspace deps |
| `.github/workflows/ci.yml` | Add WASM compile check for alaya-worker |
| `wrangler.toml` | **New** — Worker deployment config |
