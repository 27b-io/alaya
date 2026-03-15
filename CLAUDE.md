# CLAUDE.md

Ālaya — Rust/WASM MCP Memory Worker

Rust rewrite of the mcp-memory-service API layer, targeting Cloudflare Workers (WASM).

## Spec

`../mcp-memory-service/docs/superpowers/specs/2026-03-14-alaya-rust-worker-design.md`

## Architecture

5-crate workspace (5 implemented, 1 future):

| Crate | Target | Status | Purpose |
|-------|--------|--------|---------|
| **alaya-types** | wasm32 + native | Done | Shared types: Memory, Edge, SearchMode, AlayaError, PayloadFilter |
| **alaya-bridge** | native only | Done | FalkorDB typed RPC bridge (axum + redis), 18 endpoints |
| **alaya-backends** | wasm32 + native | Done | Trait definitions + HTTP clients (Qdrant, Embedding, Graph) |
| **alaya-core** | wasm32 + native | Done | MemoryService orchestration (all 9 MCP tools), 5 integration tests |
| **alaya-server** | native only | Done | REST API wrapping MemoryService (axum, 9 endpoints) |
| **alaya-worker** | wasm32 | Future | CF Worker entry point + MCP Streamable HTTP transport |

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
└── alaya-core/src/
    ├── lib.rs           # Re-exports
    ├── service.rs       # MemoryService — all 9 MCP tools orchestrated
    ├── hashing.rs       # SHA-256 content hashing
    ├── hybrid_search.rs # RRF, adaptive alpha, keyword extraction, recency decay
    ├── interference.rs  # Contradiction detection (negation, antonym, temporal)
    ├── salience.rs      # Salience scoring + boost
    ├── deduplication.rs # Cosine similarity, UnionFind, duplicate clustering
    ├── spaced_repetition.rs # Spacing quality + boost
    ├── provenance.rs    # Trust scoring, provenance building
    └── encoding_context.rs  # Context capture + similarity
└── alaya-server/src/
    └── main.rs          # Native REST server (axum, channel-based, 9 endpoints)
```

## Commands

```bash
cargo build --workspace                                    # Build all (native)
cargo build -p alaya-types --target wasm32-unknown-unknown # WASM check (types)
cargo build -p alaya-backends --target wasm32-unknown-unknown # WASM check (backends)
cargo build -p alaya-core --target wasm32-unknown-unknown  # WASM check (core)
cargo test --workspace                                     # Unit tests (156)
cargo clippy --workspace -- -D warnings                    # Lint
cargo fmt --all -- --check                                 # Format
cargo run -p alaya-bridge                                  # Run bridge

# Integration tests (need FalkorDB)
docker run -d --name falkordb -p 6379:6379 falkordb/falkordb:latest
REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test '*' -- --test-threads=1
```

## Quality Gates

Every commit must pass: `cargo fmt`, `cargo clippy -D warnings`, `cargo test`

## Environment Variables (Bridge)

```bash
REDIS_URL=redis://localhost:6379    # FalkorDB connection
GRAPH_NAME=memory                   # FalkorDB graph name (default: memory)
GRAPH_API_KEY=                      # Bearer token (empty = no auth)
RUST_LOG=alaya_bridge=info          # Logging level
```

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

Per the design spec, remaining phases:

1. ~~**Backend HTTP clients**~~ — Done (alaya-backends: QdrantClient, EmbeddingClient, GraphHttpClient)
2. ~~**alaya-core**~~ — Done (MemoryService: all 9 tools, 7 algorithm modules)
3. ~~**Native REST server**~~ — Done (alaya-server: 9 REST endpoints, channel-based axum)
4. ~~**Integration testing**~~ — Done (5 tests against real Qdrant + TEI on lab k3s)
5. **MCP transport** — JSON-RPC 2.0 over fetch, tool definitions matching Python service
6. **alaya-worker** — CF Worker entry point, wrangler.toml, KV bindings for config
7. **Prajna integration** — Replace writer.rs qdrant-client with Ālaya HTTP calls
8. **Deployment** — k3s manifests for bridge + server, Cloudflare Worker for edge
