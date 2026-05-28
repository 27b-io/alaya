# Ālaya

Rust MCP memory service. Semantic memory storage and retrieval over Qdrant (vectors) and FalkorDB (knowledge graph), exposed as both a REST API and an [MCP](https://modelcontextprotocol.io) server (Streamable HTTP, protocol 2025-03-26).

## Quick Start

```bash
cp .env.example .env
docker compose up --build -d
```

First run pulls images and downloads the embedding model (~2 GB total). TEI takes 1-2 minutes to warm up — the server waits for it automatically.

```
http://localhost:3001/health   # server
http://localhost:3001/mcp      # MCP endpoint
http://localhost:8080/health   # bridge
```

## Architecture

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│   Qdrant    │   │  FalkorDB   │   │     TEI     │
│  (vectors)  │   │   (graph)   │   │ (embeddings)│
│   :6333     │   │   :6379     │   │    :80      │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       │          ┌──────┴──────┐          │
       │          │alaya-bridge │          │
       │          │   :8080     │          │
       │          └──────┬──────┘          │
       │                 │                 │
       └────────┬────────┴────────┬────────┘
                │  alaya-server   │
                │     :3001       │
                └─────────────────┘
```

Six Rust crates, five compiled:

| Crate | Purpose |
|-------|---------|
| `alaya-types` | Shared types (Memory, Edge, SearchMode, AlayaError) |
| `alaya-backends` | Trait definitions + HTTP clients (Qdrant, Embedding, Graph, Summary) |
| `alaya-core` | MemoryService orchestration — all 9 MCP tools, hybrid search, interference detection |
| `alaya-bridge` | FalkorDB typed RPC bridge (axum + Redis, Cypher query builders) |
| `alaya-server` | REST API + MCP Streamable HTTP (axum, channel-based) |

`alaya-types`, `alaya-backends`, and `alaya-core` compile to both native and `wasm32-unknown-unknown`.

## MCP Tools

Connect any MCP client to `http://localhost:3001/mcp` (Streamable HTTP with SSE).

| Tool | Description |
|------|-------------|
| `store_memory` | Store content with tags, metadata, and optional auto-summary |
| `search` | Hybrid semantic + tag-boosted retrieval (5 modes: hybrid, scan, similar, tag, recent) |
| `delete_memory` | Delete by content hash |
| `relation` | Create / get / delete typed edges (RELATES_TO, PRECEDES, CONTRADICTS) |
| `memory_supersede` | Mark one memory as superseded by another |
| `memory_contradictions` | List unresolved contradiction pairs |
| `find_duplicates` | Cosine similarity scan for near-duplicate memories |
| `merge_duplicates` | Supersede duplicates in favour of a canonical memory |
| `check_database_health` | Backend health and storage stats |

## REST API

All endpoints accept/return JSON. Auth via `Authorization: Bearer <ALAYA_API_KEY>` when configured.

| Method | Path | Description |
|--------|------|-------------|
| POST | `/store` | Store a memory |
| POST | `/search` | Search memories |
| POST | `/delete` | Delete a memory |
| POST | `/relation` | Manage graph edges |
| POST | `/supersede` | Supersede a memory |
| POST | `/contradictions` | List contradictions |
| POST | `/duplicates/find` | Find duplicates |
| POST | `/duplicates/merge` | Merge duplicates |
| PATCH | `/memories/{hash}` | Update memory metadata |
| POST | `/backfill/summaries` | Batch-generate missing summaries |
| GET | `/health` | Health check (no auth) |
| POST | `/mcp` | MCP JSON-RPC endpoint (no auth) |

## Configuration

Copy `.env.example` to `.env`. All settings have sensible defaults for local dev.

| Variable | Default | Description |
|----------|---------|-------------|
| `QDRANT_URL` | — (required) | Qdrant HTTP endpoint |
| `QDRANT_COLLECTION` | `memories_arctic1024` | Vector collection name |
| `QDRANT_API_KEY` | — | Optional Qdrant auth |
| `EMBEDDING_URL` | — (required) | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | Model name |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensionality |
| `GRAPH_URL` | — (required) | Bridge endpoint |
| `GRAPH_API_KEY` | — | Bridge auth token |
| `ALAYA_API_KEY` | — | Server auth token (empty = no auth) |
| `LISTEN_ADDR` | `0.0.0.0:3001` | Server bind address |
| `REDIS_CACHE_URL` | — | L2 embedding cache (optional) |
| `SUMMARY_URL` | — | Anthropic Messages API URL (optional) |
| `SUMMARY_API_KEY` | — | Required if `SUMMARY_URL` is set |
| `SUMMARY_MODEL` | `claude-haiku-4-5-20251001` | Summary model |
| `RERANK_URL` | — | TEI `/rerank` endpoint (empty = rerank disabled) |
| `RERANK_API_KEY` | — | Optional bearer token for `RERANK_URL` |
| `RERANK_TOP_N` | `20` | How many top RRF candidates to rerank |
| `OIDC_ISSUER` | — | OAuth Resource Server issuer URL (empty = no OAuth) |

## Development

### Prerequisites

- Rust 1.87+ (see `rust-toolchain.toml`)
- `wasm32-unknown-unknown` target: `rustup target add wasm32-unknown-unknown`
- Docker + Docker Compose (for local backends)

### Build and test

```bash
cargo build --workspace
cargo test --workspace                    # 229 unit tests
cargo clippy --workspace -- -D warnings
cargo fmt --all -- --check
```

### WASM compatibility gate

```bash
cargo build -p alaya-types    --target wasm32-unknown-unknown
cargo build -p alaya-backends --target wasm32-unknown-unknown
cargo build -p alaya-core     --target wasm32-unknown-unknown
```

### Integration tests

Require live Qdrant + TEI backends (lab k3s or local docker-compose):

```bash
QDRANT_URL=http://localhost:6333 \
EMBEDDING_URL=http://localhost:8888 \
  cargo test -p alaya-core --test integration -- --test-threads=1
```

### Pre-commit hooks

Installed via `.pre-commit-config.yaml` — enforces fmt, clippy, tests, WASM gate, gitleaks, and trailing whitespace on every commit.

## Docker Compose Services

| Service | Image | Host Port | Purpose |
|---------|-------|-----------|---------|
| `falkordb` | `falkordb/falkordb` | 6379 | Graph database (Redis + FalkorDB module) |
| `qdrant` | `qdrant/qdrant` | 6333, 6334 | Vector database |
| `tei` | `ghcr.io/huggingface/text-embeddings-inference:cpu-latest` | 8888 | Embedding inference (CPU) |
| `alaya-bridge` | Built from `Dockerfile` | 8080 | FalkorDB RPC bridge |
| `alaya-server` | Built from `Dockerfile` | 3001 | REST + MCP server |
| `tei-rerank` | `ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest` | 8889 | Cross-encoder reranker (opt-in via `--profile rerank`) |

### Cross-encoder reranker

The reranker re-scores the top-N RRF candidates from `search_hybrid` and reorders them — boosts recall@5 ~0.94 → ~0.99 on LongMemEval. It's opt-in because the second TEI container adds ~1GB to the first-run download.

```bash
docker compose --profile rerank up -d
# then in .env, uncomment:
#   RERANK_URL=http://tei-rerank:80
docker compose up -d alaya-server   # re-create with new env
```

The server gracefully degrades to RRF order if `RERANK_URL` is unset or unreachable.

For GPU-accelerated embeddings, swap the TEI image:

```yaml
tei:
  image: ghcr.io/huggingface/text-embeddings-inference:latest
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

## License

MIT
