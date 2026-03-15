# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Ālaya — Rust MCP Memory Service

Rust rewrite of the mcp-memory-service API layer. Deployed on k3s as a native server (not CF Workers — deferred). Shares Qdrant + FalkorDB backends with the Python service during migration.

## Spec

`../mcp-memory-service/docs/superpowers/specs/2026-03-14-alaya-rust-worker-design.md`

## Architecture

6-crate workspace (5 implemented, 1 deferred):

| Crate | Target | Status | Purpose |
|-------|--------|--------|---------|
| **alaya-types** | wasm32 + native | Done | Shared types: Memory, Edge, SearchMode, AlayaError, PayloadFilter |
| **alaya-bridge** | native only | Done | FalkorDB typed RPC bridge (axum + redis), 18 endpoints |
| **alaya-backends** | wasm32 + native | Done | Trait definitions + HTTP clients (Qdrant, Embedding, Graph) |
| **alaya-core** | wasm32 + native | Done | MemoryService orchestration (all 9 MCP tools), 5 integration tests |
| **alaya-server** | native only | Done | REST API + MCP Streamable HTTP (axum, channel-based, 9 endpoints + /mcp) |
| **alaya-worker** | wasm32 | Deferred | CF Worker entry point (reqwest-wasm unreliable on Workers, native server sufficient) |

```
crates/
├── alaya-types/src/
│   ├── lib.rs          # Re-exports
│   ├── error.rs        # AlayaError (6 variants, JSON-RPC codes, safe_message())
│   ├── graph.rs        # UserRelationType, SystemRelationType, Edge, Neighbor, etc.
│   ├── memory.rs       # Memory (15 fields), ScoredMemory, ScrollResult, MetadataUpdate
│   └── search.rs       # SearchMode (5 modes), PromptName, PayloadFilter
├── alaya-bridge/src/
│   ├── lib.rs           # Library target (re-exports for integration tests)
│   ├── main.rs          # Binary entry — reads REDIS_URL/GRAPH_NAME, starts axum + queue
│   ├── routes.rs        # 18 HTTP endpoints, auth middleware on API routes
│   ├── auth.rs          # Bearer token middleware (GRAPH_API_KEY env)
│   ├── cypher.rs        # Typed Cypher query builders (17 functions, all parameterized)
│   ├── resp.rs          # FalkorDB RESP parser (compact + non-compact modes)
│   ├── queue.rs         # Hebbian write queue (LPUSH/BRPOP, rate-limited 100 ops/sec)
│   └── handlers/
│       ├── mod.rs       # exec_query() + value_to_cypher_literal()
│       ├── nodes.rs     # POST /nodes/ensure, /nodes/delete
│       ├── edges.rs     # POST /edges/create, /edges/get, /edges/delete, /edges/create-system
│       ├── health.rs    # GET /health (unauth), GET /stats
│       ├── hebbian.rs   # POST /hebbian/{neighbors,spreading,boosts-within,strengthen}
│       ├── contradictions.rs  # POST /contradictions/{all,for}
│       └── consolidation.rs   # POST /consolidation/{decay-all,decay-stale,prune,orphans}
├── alaya-backends/src/
│   ├── lib.rs           # Re-exports
│   ├── traits.rs        # VectorStorage, EmbeddingProvider, GraphService, HebbianService, ConsolidationService
│   ├── qdrant.rs        # QdrantClient — Qdrant REST API (WASM-compat)
│   ├── embedding.rs     # EmbeddingClient — OpenAI-compat /v1/embeddings
│   └── graph.rs         # GraphHttpClient — bridge HTTP wrapper (3 trait impls)
├── alaya-core/src/
│   ├── lib.rs           # Re-exports
│   ├── service.rs       # MemoryService — all 9 MCP tools orchestrated
│   ├── hashing.rs       # SHA-256 content hashing
│   ├── hybrid_search.rs # RRF, adaptive alpha, keyword extraction, recency decay
│   ├── interference.rs  # Contradiction detection (negation, antonym, temporal)
│   ├── salience.rs      # Salience scoring + boost
│   ├── deduplication.rs # Cosine similarity, UnionFind, duplicate clustering
│   ├── spaced_repetition.rs # Spacing quality + boost
│   ├── provenance.rs    # Trust scoring, provenance building
│   └── encoding_context.rs  # Context capture + similarity
└── alaya-server/src/
    ├── main.rs          # Native REST + MCP server (axum, channel-based)
    ├── mcp.rs           # MCP Streamable HTTP (JSON-RPC 2.0, SSE, protocol 2025-03-26)
    └── telemetry.rs     # OTLP tracing to Phoenix (optional, graceful fallback)
```

## Commands

```bash
cargo build --workspace                                    # Build all (native)
cargo build -p alaya-types --target wasm32-unknown-unknown # WASM check (types)
cargo build -p alaya-backends --target wasm32-unknown-unknown # WASM check (backends)
cargo build -p alaya-core --target wasm32-unknown-unknown  # WASM check (core)
cargo test --workspace                                     # Unit tests (168)
cargo test -p alaya-core --test integration                # Skip (needs backends)
cargo clippy --workspace -- -D warnings                    # Lint
cargo fmt --all -- --check                                 # Format
cargo run -p alaya-bridge                                  # Run bridge (needs REDIS_URL)
cargo run -p alaya-server                                  # Run server (needs QDRANT_URL + EMBEDDING_URL + GRAPH_URL)

# Integration tests (need real backends — lab k3s cluster IPs or port-forwards)
QDRANT_URL=http://10.43.119.230:6333 \
EMBEDDING_URL=http://10.43.242.167 \
  cargo test -p alaya-core --test integration -- --test-threads=1

# Bridge integration tests (need FalkorDB)
REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test '*' -- --test-threads=1
```

## Quality Gates

Every commit must pass: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`, WASM gate.
Pre-commit hooks via prek enforce all four + gitleaks + trailing whitespace.

## Deployment (lab k3s)

Both services deployed in `mcp` namespace via `lab/k8s/mcp/alaya.yaml`:

| Pod | Image | Port | Connects to |
|-----|-------|------|-------------|
| alaya-bridge | ghcr.io/27b-io/alaya:latest | 8080 (svc: 3000) | FalkorDB (recsys:6379) |
| alaya-server | ghcr.io/27b-io/alaya:latest | 3001 | Qdrant (mcp:6333), TEI (mcp:80), bridge (mcp:3000) |

CI pushes to ghcr.io on every main push. Network policies restrict egress to named backends + DNS.

## Environment Variables

### Bridge
```bash
REDIS_URL=redis://falkordb.recsys.svc:6379
GRAPH_NAME=memory
GRAPH_API_KEY=               # empty = no auth
RUST_LOG=alaya_bridge=info
```

### Server
```bash
QDRANT_URL=http://qdrant:6333            # required
QDRANT_COLLECTION=memories_arctic1024    # default
EMBEDDING_URL=http://tei:80              # required
EMBEDDING_MODEL=Snowflake/snowflake-arctic-embed-l-v2.0
EMBEDDING_DIMENSIONS=1024
GRAPH_URL=http://alaya-bridge:3000       # required
GRAPH_API_KEY=
LISTEN_ADDR=0.0.0.0:3001
RUST_LOG=alaya_server=info
OTEL_EXPORTER_OTLP_ENDPOINT=http://phoenix-svc.recsys.svc:6006  # optional
OTEL_SERVICE_NAME=alaya-server
```

## Key Design Decisions

- **`?Send` traits** — All backend traits use `#[async_trait(?Send)]` for WASM compat. The native server bridges this via a channel-based architecture: axum (multi-threaded, Send+Sync) sends commands over mpsc to MemoryService running on a dedicated LocalSet thread.
- **UUID from content_hash** — `Uuid::parse_str(&hash[..32])` — takes first 32 hex chars, NOT uuid5. Must match Python `uuid.UUID(hash[:32])` for data compatibility.
- **Superseded filtering** — Done at application layer, NOT Qdrant filter level. Qdrant's `is_null` on nested payload fields is unreliable without explicit indexes.
- **Graph operations are non-fatal** — All graph calls (spreading activation, Hebbian, interference) use `unwrap_or_default()`. Service degrades gracefully when FalkorDB is down.
- **reqwest default-features = false** — Workspace-level and per-crate. Uses `rustls-tls` on native, bare `json` on wasm32. Prevents OpenSSL dependency in containers.
- **MCP protocol 2025-03-26** — SSE response format (`event: message\ndata: {...}\n\n`) when client sends `Accept: text/event-stream`. Plain JSON otherwise.

## Conventions

- Guard clauses over nesting
- Parameterized Cypher only — no string interpolation from external input
- Relation types use compile-time enums with `cypher_label()` methods
- Hop depths are `u8` capped at 3 via `.clamp(1, 3)`
- All errors sanitized before external exposure (`safe_message()`)
- Single-hop Hebbian queries use direct edge match (avoids FalkorDB Path/Edge type ambiguity)
- Multi-hop queries use `relationships(e)` to extract edge list from Path type

## Known FalkorDB Wire-Format Behaviors

These were discovered during integration testing and are NOT documented in FalkorDB's public docs:

1. **Compact mode double-quotes strings** — all strings wrapped in `"..."`, must strip
2. **Compact mode typed cells** — each cell is `[type_id, value]` not bare value
3. **Variable-length paths bind as Path** — `e` in `*1..N` is a Path type, use `relationships(e)` to get edge list
4. **Single-length paths bind as Edge** — `*1..1` binds `e` as bare Edge, not List — `ALL(r IN e ...)` fails

## Implementation Roadmap

1. ~~**Backend HTTP clients**~~ — Done (alaya-backends: QdrantClient, EmbeddingClient, GraphHttpClient)
2. ~~**alaya-core**~~ — Done (MemoryService: all 9 tools, 7 algorithm modules)
3. ~~**Native REST server**~~ — Done (alaya-server: 9 REST endpoints, channel-based axum)
4. ~~**Integration testing**~~ — Done (5 tests against real Qdrant + TEI on lab k3s)
5. ~~**MCP transport**~~ — Done (JSON-RPC 2.0 + SSE, protocol 2025-03-26, 9 tool schemas)
6. ~~**Deployment**~~ — Done (k3s manifests, CI → ghcr.io, network policies)
7. **OTLP tracing** — Wired but degraded (reqwest async client issue in container, falls back to stderr)
8. **Prajna integration** — Replace writer.rs qdrant-client with Ālaya HTTP calls
9. **cachekit-rs integration** — Embedding cache for edge performance (immutable, content-addressed)
