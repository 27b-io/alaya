# Quickstart: Self-hosting

Bring up Ālaya on your own machine. The recommended path is `docker compose` — one command brings up all four required services. There's also a "build from source" path for contributors and air-gapped environments.

## Prerequisites

- Docker 24+ and Docker Compose v2 (or Podman + podman-compose).
- About **3 GB of free disk** for the first run (TEI model download is the bulk; subsequent runs reuse the volume).
- An **x86_64 or ARM64** host. Apple Silicon works out of the box; on Intel/AMD, set `TEI_TAG=cpu-1.9.3` in `.env`.

## Bring up the stack

```bash
git clone https://github.com/27b-io/alaya.git
cd alaya
cp .env.example .env
docker compose up -d
```

That starts five containers (six with `--profile rerank`):

> [!WARNING]
> **Auth is fail-closed.** The server won't boot without auth, so the dev Compose opts into open mode on `localhost` (`DANGEROUSLY_ALLOW_UNAUTHENTICATED=true`, automatically refused on any non-private origin). Set `ALAYA_API_KEY` in `.env` before exposing it — that enables `Authorization: Bearer` auth and disables the dev-open flag.

| Service | Purpose | Default host port |
|:--|:--|:--|
| `falkordb` | Graph database (Redis + FalkorDB module) — also doubles as the L2 embedding cache | `6379` |
| `qdrant` | Vector database | `6333` (HTTP), `6334` (gRPC) |
| `tei` | Embedding inference (Snowflake Arctic embed v2.0 by default) | `8888` |
| `alaya-bridge` | FalkorDB RPC bridge | `8080` |
| `alaya-server` | REST + MCP server — this is what your clients talk to | `3001` |

First boot takes a few minutes — TEI has to download the ~600 MB embedding model and Ālaya has to build the Rust binaries. Watch progress with `docker compose logs -f`.

## The vector collection is created automatically

`alaya-server` ensures its Qdrant collection exists at startup, so a fresh Qdrant volume — even after `docker compose down -v` — accepts writes with no manual step. The collection name comes from `QDRANT_COLLECTION` (`memories_arctic1024` by default, for the Snowflake Arctic embed model) and is created with `EMBEDDING_DIMENSIONS`-wide Cosine vectors.

> [!WARNING]
> Qdrant has no auth in the default compose; this assumes localhost — don't expose :6333.

## Verify

```bash
curl http://localhost:3001/health
```

Expect:

```json
{"status":"healthy"}
```

`/health` is the unauthenticated probe and carries only `status`. For
per-backend detail, memory count and build identity:

```bash
curl -H "Authorization: Bearer $ALAYA_API_KEY" \
  http://localhost:3001/health/detail
```

Then exercise the API end-to-end:

```bash
# Store
# Authorization header only when a key is set — omit on a localhost no-auth dev box.
curl -fsS -H "Authorization: Bearer ${ALAYA_API_KEY}" \
  -X POST http://localhost:3001/store \
  -H 'Content-Type: application/json' \
  -d '{"content":"hello memory","tags":["demo"]}'
# → {"content_hash":"e3b0c4...","stored":true,...}

# Search
# Authorization header only when a key is set — omit on a localhost no-auth dev box.
curl -fsS -H "Authorization: Bearer ${ALAYA_API_KEY}" \
  -X POST http://localhost:3001/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"memory","mode":"hybrid"}'
# → [{"content":"hello memory","similarity":0.78,...}]
```

You now have a working memory service. To wire MCP clients to it, see [Quickstart: MCP clients](./quickstart-mcp.md).

## Enable the cross-encoder reranker (optional)

Re-scores the top-N RRF candidates from `search` and reorders them. Empirically lifts recall@5 from ~0.94 to ~0.99 on the LongMemEval benchmark. Costs ~1 GB extra disk for a second TEI container loading the reranker model.

```bash
docker compose --profile rerank up -d
```

Then in `.env`, uncomment:

```ini
RERANK_URL=http://tei-rerank:80
```

And re-create `alaya-server` so it picks up the new env:

```bash
docker compose up -d alaya-server
```

The server gracefully degrades to plain RRF ordering if `RERANK_URL` is unset or unreachable, so this is fully opt-in.

## Configuration

Every knob lives in `.env`. The interesting ones:

| Variable | Default | Notes |
|:--|:--|:--|
| `ALAYA_API_KEY` | empty | Bearer token clients must send. Empty is fail-closed: the server won't boot unless the dev Compose sets `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` (localhost only). **Set this for any non-local deployment.** |
| `GRAPH_API_KEY` | empty | Bridge bearer token. Empty → bridge won't boot unless `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` (the dev Compose sets it). |
| `QDRANT_COLLECTION` | `memories_arctic1024` | Name of the collection the server auto-creates at startup. |
| `TEI_MODEL` | `Snowflake/snowflake-arctic-embed-l-v2.0` | Embedding model. Change this and you'll need a new collection with the matching dimensionality. |
| `EMBEDDING_DIMENSIONS` | `1024` | Must match `TEI_MODEL`'s output. |
| `REDIS_CACHE_URL` | `redis://falkordb:6379` | L2 embedding cache. Reuses falkordb's Redis layer by default. |
| `RERANK_URL` | empty | Set to `http://tei-rerank:80` to enable rerank (requires `--profile rerank`). |
| `RERANK_TOP_N` | `20` | How many top RRF candidates to re-score. |
| `OIDC_ISSUER` | empty | Set to your IdP's issuer URL to enable OAuth Resource Server mode. See [MCP quickstart → OAuth](./quickstart-mcp.md#oauth-optional). |
| `SUMMARY_URL` | empty | Set + provide `SUMMARY_API_KEY` to auto-generate one-line summaries via Anthropic's Messages API. |

The complete list — including the bridge-side variables — lives in `CLAUDE.md` under "Environment Variables".

## Operational notes

- **Data persistence:** Qdrant vectors, FalkorDB graph, and TEI model cache live in named docker volumes (`qdrant_data`, `falkordb_data`, `tei_cache`). `docker compose down` keeps them; `docker compose down -v` wipes everything.
- **Backups:** dump Qdrant with [snapshot API](https://qdrant.tech/documentation/concepts/snapshots/) and FalkorDB with `redis-cli SAVE` → copy `dump.rdb`. The two backends must be restored together or hashes won't line up.
- **Resource use (idle, single user):** ~150 MB RAM for `alaya-server`, ~300 MB for `qdrant`, ~600 MB for `tei` (or ~1.6 GB with rerank), ~50 MB for `falkordb`. Embedding requests dominate CPU under load — TEI is the thing to scale first.
- **Logging:** `RUST_LOG=alaya_server=info,alaya_bridge=info` by default. Bump to `debug` for request-level tracing.

## Updating

```bash
git pull
docker compose pull          # if you use prebuilt images
docker compose up -d --build # if you build locally
```

Migrations: none for the Rust server itself, but if you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS` you'll need to re-embed all stored memories. There's a helper script at `scripts/migrate_from_mcp.py` that demonstrates the bulk re-embed flow against an external collection.

## Build from source (no Docker)

If you'd rather run native binaries:

```bash
# 1. Bring up the three external backends however you like — falkordb, qdrant, tei.

# 2. Point at them and run:
export REDIS_URL=redis://localhost:6379
cargo run -p alaya-bridge &

export QDRANT_URL=http://localhost:6333
export EMBEDDING_URL=http://localhost:8888
export GRAPH_URL=http://localhost:8080
cargo run -p alaya-server
```

The same env vars apply. `CLAUDE.md` documents the full Rust workflow (build commands, integration test setup, quality gates).

## Production-ish hardening checklist

Before pointing the internet at this:

- [ ] Set `ALAYA_API_KEY` and `GRAPH_API_KEY` to non-empty values.
- [ ] Put `alaya-server` behind a TLS-terminating reverse proxy (the server speaks plain HTTP).
- [ ] Lock down the docker-compose ports — by default Qdrant, FalkorDB, and the bridge are exposed on `0.0.0.0`; bind to `127.0.0.1` or drop the `ports:` mapping for any service the public doesn't need.
- [ ] If you want delegated auth, set `OIDC_ISSUER` and put your IdP in front.
- [ ] Snapshot Qdrant + FalkorDB on a schedule.
- [ ] Watch `alaya-server` logs for the circuit-breaker messages — they mean the L2 cache or a backend is flapping.
