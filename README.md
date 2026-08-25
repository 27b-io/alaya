# Ālaya

Long-term memory for LLM agents — store what matters, retrieve it months later by meaning, and never trip over your own contradictions.

A single Rust service over Qdrant (vectors) and FalkorDB (knowledge graph). It speaks both plain REST and [MCP](https://modelcontextprotocol.io) (Streamable HTTP, protocol 2025-03-26), so the same store answers your agents and your scripts.

## Why this exists

Vector search alone gives an agent a fuzzy lookup table. It returns whatever is closest in embedding space — including stale facts, near-duplicates, and statements that flatly contradict each other — with no notion of which memory superseded which, or how the memories relate. Ālaya is built to be the memory layer you can actually trust over months of writes: it retrieves by meaning, notices when two memories disagree, lets you resolve the conflict while keeping an audit trail, and reasons over the relationships between memories instead of treating each one as an island.

- **It finds the right memory even when your words don't match** — hybrid retrieval fuses semantic vectors with keyword signal (Reciprocal Rank Fusion), so "why did we switch package managers" finds the note that says "migrated to pnpm." *(hybrid RRF retrieval)*
- **It catches when your memories disagree** — conflicting facts are flagged automatically on write; you resolve them with one call and the old answer stays auditable instead of silently vanishing. *(contradiction detection — negation, antonym, temporal cues — plus supersede)*
- **Related memories pull each other up** — a relationship graph (RELATES_TO / PRECEDES / CONTRADICTS) lets one strong hit surface its neighbors, so retrieving one fact brings back the context around it. *(Hebbian spreading-activation)*
- **What you mark important, and what you revisit, ranks higher** — relevance is weighted by how important a memory is and how often it's accessed, not just raw cosine distance. *(salience + spaced-repetition boosts)*
- **It knows where a memory came from and how much to trust it** — provenance and a trust score let you filter out low-confidence sources at query time. *(provenance / trust scoring)*
- **It won't drown you in duplicates** — near-duplicate detection clusters and merges redundant memories on demand. *(cosine-similarity dedup + merge)*
- **One service, two protocols** — the same storage answers MCP tool calls (for agents) and plain REST (for scripts and backfill). *(dual protocol over one endpoint)*
- **Small to run** — a single Docker stack with a ~150 MB server binary that degrades gracefully when a backend blips (graph calls are non-fatal). *(operational footprint)*

## Proof

Measured on **LongMemEval** (`longmemeval_s_cleaned`, 500 multi-session QA items), using the standard LongMemEval **reset-per-question** protocol — each question is scored against only its own haystack in an otherwise-empty index.

We report **hit-rate@5**: the fraction of questions where at least one ground-truth session lands in the top 5 results. This is the *any-correct-in-top-k* metric (a binary per-question hit, averaged across questions) — we label it hit-rate, **not** classical recall, because that is what the harness computes (`recall_at_k` in `benchmarks/longmemeval_bench.py`: `float(any(cid in top_k ...))`).

| Configuration              | hit-rate@5 | hit-rate@10 |
|:---------------------------|-----------:|------------:|
| Hybrid (RRF)               | 0.916      | 0.964       |
| Hybrid + cross-encoder     | 0.986      | 0.988       |

Both rows are the same 500 LongMemEval items against the same live server, identical except for `RERANK_URL`. The cross-encoder re-scores the top-20 RRF candidates as (query, document) pairs and reorders them. n=500, 95% CI [0.971, 0.993]; paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).

Cross-encoder reranking lifts hit-rate@5 from 0.916 to 0.986 on the live server — +7.0 points, paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).

Numbers above are from harness `benchmarks/longmemeval_bench.py` at commit `8b01f85`, run on `2026-05-30`, against the shipped server build. **[Reproduce →](benchmarks/README.md)**

## Quick Start

```bash
cp .env.example .env
docker compose up --build -d
```

First run pulls images and downloads the embedding model (~2 GB total). TEI takes 1-2 minutes to warm up — the server waits for it automatically.

```text
http://localhost:3001/health   # server
http://localhost:3001/mcp      # MCP endpoint
http://localhost:8080/health   # bridge
```

> [!WARNING]
> **Auth is fail-closed.** The server won't start without auth, so the dev Compose opts into open mode on `localhost` (`DANGEROUSLY_ALLOW_UNAUTHENTICATED=true`, automatically refused on any non-private origin). Before exposing this anywhere, set `ALAYA_API_KEY` in `.env` — that enables `Authorization: Bearer` auth and disables the dev-open flag. See [Self-hosting → hardening](docs/quickstart-selfhost.md).

The default Compose stack is **5 containers** (Qdrant, FalkorDB, TEI, `alaya-bridge`, `alaya-server`); the opt-in reranker (`--profile rerank`) adds a sixth.

### Next steps

- **Connect an MCP client** → [docs/quickstart-mcp.md](docs/quickstart-mcp.md)
- **Run your own server** → [docs/quickstart-selfhost.md](docs/quickstart-selfhost.md)
- **Full MCP tool reference** → [docs/mcp-tools.md](docs/mcp-tools.md)
- **Full REST API reference** → [docs/rest-api.md](docs/rest-api.md)
- **Claude Code memory hooks (Stop + PreCompact)** → [integrations/claude-code](integrations/claude-code/README.md)

## Architecture

```text
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
|:------|:--------|
| `alaya-types` | Shared types (Memory, Edge, SearchMode, AlayaError) |
| `alaya-backends` | Trait definitions + HTTP clients (Qdrant, Embedding, Graph, Summary) |
| `alaya-core` | MemoryService orchestration — all 10 MCP tools, hybrid search, interference detection |
| `alaya-bridge` | FalkorDB typed RPC bridge (axum + Redis, Cypher query builders) |
| `alaya-server` | REST API + MCP Streamable HTTP (axum, channel-based) |

`alaya-types`, `alaya-backends`, and `alaya-core` compile to both native and `wasm32-unknown-unknown`.

## MCP Tools

Connect any MCP client to `http://localhost:3001/mcp` (Streamable HTTP with SSE). Full parameter reference: [docs/mcp-tools.md](docs/mcp-tools.md).

| Tool | Description |
|:-----|:------------|
| `store_memory` | Store content with tags, metadata, and optional auto-summary |
| `search` | Hybrid semantic + tag-boosted retrieval (5 modes: hybrid, scan, similar, tag, recent) |
| `get_memory` | Fetch a single memory by content hash |
| `delete_memory` | Delete by content hash |
| `relation` | Create / get / delete typed edges (RELATES_TO, PRECEDES, CONTRADICTS) |
| `memory_supersede` | Mark one memory as superseded by another |
| `memory_contradictions` | List unresolved contradiction pairs |
| `find_duplicates` | Cosine similarity scan for near-duplicate memories |
| `merge_duplicates` | Supersede duplicates in favour of a canonical memory |
| `check_database_health` | Backend health and storage stats |

## REST API

All endpoints accept/return JSON. Auth via `Authorization: Bearer ${ALAYA_API_KEY}` when configured (the dev Compose runs open on localhost — see [Quick Start](#quick-start)). Full reference: [docs/rest-api.md](docs/rest-api.md).

| Method | Path | Description |
|:-------|:-----|:------------|
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
| POST | `/mcp` | MCP JSON-RPC endpoint |

## Configuration

Copy `.env.example` to `.env`. All settings have sensible defaults for local dev.

| Variable | Default | Description |
|:---------|:--------|:------------|
| `QDRANT_URL` | — (required) | Qdrant HTTP endpoint |
| `QDRANT_COLLECTION` | `memories_arctic1024` | Vector collection name |
| `QDRANT_API_KEY` | — | Optional Qdrant auth |
| `EMBEDDING_URL` | — (required) | OpenAI-compatible embeddings endpoint |
| `EMBEDDING_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | Model name |
| `EMBEDDING_DIMENSIONS` | `1024` | Vector dimensionality |
| `GRAPH_URL` | — (required) | Bridge endpoint |
| `GRAPH_API_KEY` | — | Bridge auth token |
| `ALAYA_API_KEY` | — | Server auth token. Empty → server won't boot unless `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` (localhost dev only) |
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
|:--------|:------|:----------|:--------|
| `falkordb` | `falkordb/falkordb` | 6379 | Graph database (Redis + FalkorDB module) |
| `qdrant` | `qdrant/qdrant` | 6333, 6334 | Vector database |
| `tei` | `ghcr.io/huggingface/text-embeddings-inference:${TEI_TAG:-cpu-arm64-latest}` | 8888 | Embedding inference (CPU) |
| `alaya-bridge` | Built from `Dockerfile` | 8080 | FalkorDB RPC bridge |
| `alaya-server` | Built from `Dockerfile` | 3001 | REST + MCP server |
| `tei-rerank` | `ghcr.io/huggingface/text-embeddings-inference:cpu-arm64-latest` | 8889 | Cross-encoder reranker (opt-in via `--profile rerank`) |

The default stack runs **5 containers**; `--profile rerank` brings the sixth (`tei-rerank`) online.

### Cross-encoder reranker

The reranker re-scores the top-N RRF candidates from `search_hybrid` and reorders them — see [Proof](#proof) for the measured effect. It's opt-in because the second TEI container adds ~1GB to the first-run download.

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
