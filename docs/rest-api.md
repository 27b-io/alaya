# REST API reference

Ālaya's REST API and MCP tools cover the same operations against the same storage. Use REST for scripts, backfill jobs, and any client that doesn't speak MCP. Use [MCP](./mcp-tools.md) for LLM agents.

All endpoints below assume the server is reachable at `http://localhost:3001` — substitute your own base URL.

## Authentication

Set `ALAYA_API_KEY` on the server to require authentication. Clients then send:

```http
Authorization: Bearer <your-api-key>
```

Auth is **fail-closed**: with no `ALAYA_API_KEY` (and no `OIDC_ISSUER`) the server refuses to boot — unless `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` is set, which the dev Compose does for `localhost` only (it is refused on any non-private `PUBLIC_BASE_URL`). Set `ALAYA_API_KEY` before exposing the server; that enables bearer auth and disables the dev-open flag.

If the server has `OIDC_ISSUER` set, clients use an OAuth access token instead of a fixed key. See [MCP quickstart → OAuth](./quickstart-mcp.md#oauth-optional) — the same flow applies to REST clients.

Failed auth returns `401 Unauthorized` with a `WWW-Authenticate: Bearer …` header pointing at the protected-resource metadata when OAuth is enabled.

## Content types

- All request bodies are `application/json`.
- Responses are `application/json`.
- The maximum request body size is **1 MB**. Larger payloads return `413 Payload Too Large`.

## Endpoint summary

| Method | Path | Purpose | Auth? |
|:--|:--|:--|:-:|
| `GET`  | `/health` | Liveness probe — status only | no |
| `GET`  | `/health/detail` | Backend health, capacity, build identity | yes |
| `POST` | `/store` | Add a memory | yes |
| `POST` | `/search` | Retrieve memories | yes |
| `GET`  | `/memories/{content_hash}` | Fetch one memory | yes |
| `PATCH`| `/memories/{content_hash}` | Update fields on one memory | yes |
| `POST` | `/delete` | Hard-delete a memory | yes |
| `POST` | `/relation` | Manage graph edges | yes |
| `POST` | `/supersede` | Mark old → new | yes |
| `POST` | `/contradictions` | List unresolved contradictions | yes |
| `POST` | `/duplicates/find` | Scan for near-duplicates | yes |
| `POST` | `/duplicates/merge` | Supersede a duplicate cluster | yes |
| `POST` | `/backfill/summaries` | Generate missing summaries | yes |
| `POST` | `/mcp` | MCP JSON-RPC entry point | yes |
| `GET`  | `/.well-known/oauth-protected-resource[/mcp]` | OAuth resource metadata (404 unless `OIDC_ISSUER` set) | no |

## `GET /health`

Unauthenticated liveness + readiness probe. Returns the status word and nothing
else.

```bash
curl http://localhost:3001/health
```

```json
{ "status": "healthy" }
```

`status` is `healthy` when every backend is reachable, `degraded` (HTTP 200) when
a backend is down — restarting the pod won't fix Qdrant — and `unhealthy`
(HTTP 503) when the service worker has stalled, so a liveness probe restarts it.

Probers read the HTTP code, so a k8s `httpGet` probe and `curl -sf .../health`
both work against this endpoint unchanged.

## `GET /health/detail`

Authenticated. The full operational document — per-backend health, worker state,
memory count and build identity.

```bash
curl -H "Authorization: Bearer $ALAYA_API_KEY" \
  http://localhost:3001/health/detail
```

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "git_sha": "2f9c1a4b6d8e0f2a4c6e8b0d2f4a6c8e0b2d4f6a",
  "built_at": "2026-08-09T11:22:33Z",
  "backend": "qdrant",
  "worker": { "state": "ok", "stalled": false, "last_progress_age_s": 3 },
  "vector_health": { "status": "green" },
  "graph_health": { "status": "healthy" },
  "total_memories": 1247
}
```

Same status and HTTP-code mapping as `/health`.

> [!NOTE]
> These fields were served by the unauthenticated `/health` in earlier builds.
> They are live capacity, outage state and — on the failure path — in-cluster
> backend URLs, so they now sit behind the same bearer auth as every other
> read. If you probed `/health` for anything other than `status`, point that
> reader at `/health/detail` and give it a token.

### Build identity

`version`, `git_sha` and `built_at` answer "is build X live?" without cluster
access. `version` is the crate semver; `git_sha` is the commit the binary was
built from; `built_at` is an RFC3339 timestamp. The last two are `null` for any
build that didn't pass them (a plain `cargo build`, or `docker build` without
`--build-arg`) — absence is never an error. Verify a rollout with:

```bash
curl -s -H "Authorization: Bearer $ALAYA_API_KEY" \
  http://localhost:3001/health/detail | jq -r .git_sha   # == git rev-parse HEAD
```

CI images always carry the full 40-hex SHA. A build that passes an abbreviation
reports that prefix, so compare with `startswith` rather than equality if you
accept locally-built images.

> [!NOTE]
> Before v0.1.0's build-identity change, `version` carried the git SHA. It now
> carries the crate semver — read `git_sha` for the commit. The MCP
> `initialize` response reports both together as `serverInfo.version`
> (`0.1.0+<sha>`, semver build metadata).

## `POST /store`

Embed and persist text. Body schema matches MCP's `store_memory` arguments:

```http
POST /store
Content-Type: application/json

{
  "content": "Migrated frontend from npm to pnpm because of lockfile churn.",
  "tags": ["frontend", "tooling"],
  "memory_type": "decision",
  "metadata": {"importance": 0.7}
}
```

Required: `content`. Optional: `tags`, `memory_type` (`note`|`decision`|`task`|`reference`), `metadata`, `client_hostname`, `summary`, `dedup_threshold`.

**Response:**

```json
{
  "content_hash": "a3f4e891b27c5d6e0123456789abcdef0123456789abcdef0123456789abcdef",
  "stored": true,
  "duplicate_of": null,
  "salience": 0.62
}
```

`duplicate_of` is non-null when `dedup_threshold` was set and the new memory's nearest neighbour exceeded it — no write happened.

## `POST /search`

Body schema matches MCP's `search` arguments (see [MCP tool reference → search](./mcp-tools.md#search) for the full param table).

```bash
curl -X POST http://localhost:3001/search \
  -H "Authorization: Bearer $ALAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"query":"why did we switch package managers?","mode":"hybrid","page_size":5}'
```

**Response:** array of memories sorted by relevance.

```json
[
  {
    "content_hash": "a3f4...",
    "content": "Migrated frontend from npm to pnpm…",
    "similarity": 0.82,
    "summary": null,
    "tags": ["frontend", "tooling"],
    "memory_type": "decision",
    "metadata": {"importance": 0.7},
    "created_at": "2026-05-12T09:31:04Z",
    "salience": 0.62,
    "rank": 1
  }
]
```

## `GET /memories/{content_hash}`

Fetch a single memory by exact hash. Query string `?output=full|summary|both` controls payload shape.

```bash
curl -H "Authorization: Bearer $ALAYA_API_KEY" \
     "http://localhost:3001/memories/a3f4e891.../"
```

| Status | Body |
|:--|:--|
| `200 OK` | `{ "found": true, "memory": {...} }` |
| `404 Not Found` | `{ "found": false }` |
| `400 Bad Request` | `{ "error": "invalid content_hash format" }` — hash isn't 64 lowercase hex chars |

Superseded memories return `200` with `memory.metadata.superseded_by` populated.

## `PATCH /memories/{content_hash}`

Update mutable fields on one memory. At least one field must be present.

```http
PATCH /memories/a3f4...
Content-Type: application/json

{
  "summary": "Switched to pnpm — lockfile churn was killing CI cache.",
  "tags": ["frontend", "tooling", "pnpm"]
}
```

Updatable fields: `summary`, `tags`, `metadata`. Content and `content_hash` are immutable by design — to change content, store a new memory and supersede the old.

Changing `summary` also drops the stored summary embedding (the hybrid-search boost vector) so the two never disagree; the boost returns when the summary is next generated server-side.

| Status | Meaning |
|:--|:--|
| `200 OK` | Updated. Response echoes the new state. |
| `400` | Empty patch, invalid hash, or validation failure. |
| `404` | Memory doesn't exist. |

## `POST /delete`

```http
POST /delete
Content-Type: application/json

{"content_hash": "a3f4e891..."}
```

Hard delete. Response: `{ "deleted": true }`. **No tombstone, no recovery** — use `/supersede` if you might want history.

## `POST /relation`

Manage typed graph edges. Same shape as MCP's `relation` tool.

```http
POST /relation
Content-Type: application/json

{
  "action": "create",
  "content_hash": "a3f4...",
  "target_hash": "b71c...",
  "relation_type": "PRECEDES"
}
```

`action`: `create`, `get`, or `delete`. Relation types: `RELATES_TO`, `PRECEDES`, `CONTRADICTS`.

## `POST /supersede`

> [!IMPORTANT]
> **Field-name divergence.** `supersede` takes `old_hash`/`new_hash` on REST (`POST /supersede`) and `old_id`/`new_id` in MCP (`memory_supersede`). The **values are identical** — full 64-char content hashes; only the field names differ. (`relation` and `delete` use `content_hash`/`target_hash` on *both* protocols — no divergence.)

See also [MCP: `memory_supersede`](./mcp-tools.md#memory_supersede).

```http
POST /supersede
Content-Type: application/json

{
  "old_hash": "a3f4...",
  "new_hash": "c0de...",
  "reason": "Reverted the pnpm migration."
}
```

`reason` is optional. Returns `{ "superseded": true, "old_hash": "...", "new_hash": "..." }`.

## `POST /contradictions`

```http
POST /contradictions
Content-Type: application/json

{"limit": 20}
```

Returns up to `limit` unresolved contradiction pairs.

## `POST /duplicates/find`

Plan-only. Doesn't mutate.

```http
POST /duplicates/find
Content-Type: application/json

{
  "similarity_threshold": 0.95,
  "limit": 500,
  "strategy": "keep_newest"
}
```

`strategy`: `keep_newest`, `keep_oldest`, or `keep_most_accessed` — picks the canonical entry in each cluster. Response: array of clusters with `canonical_hash` and `duplicate_hashes[]`.

## `POST /duplicates/merge`

Apply the plan from `/duplicates/find`. **Use `dry_run: true` first.**

```http
POST /duplicates/merge
Content-Type: application/json

{
  "canonical_hash": "a3f4...",
  "duplicate_hashes": ["b71c...", "c0de..."],
  "reason": "Merged by deduplication sweep 2026-05-28",
  "dry_run": false
}
```

## `POST /backfill/summaries`

For deployments that turned on `SUMMARY_URL` after the corpus already existed: generates summaries for up to `limit` memories that don't have one yet.

```http
POST /backfill/summaries
Content-Type: application/json

{"limit": 100}
```

`limit` defaults to `100`. Use sparingly — this calls your summary provider once per memory.

## `POST /mcp`

The MCP JSON-RPC entry point. Most users will reach this via an MCP client rather than direct REST, but it's a normal endpoint you can `curl`:

```bash
curl -X POST http://localhost:3001/mcp \
  -H "Authorization: Bearer $ALAYA_API_KEY" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

The `Accept` header is required by the MCP Streamable HTTP spec — without `text/event-stream` the server can't switch to SSE if the response needs it.

See the [MCP tool reference](./mcp-tools.md) for the available `tools/call` methods.

## Errors

REST endpoints use HTTP status codes plus a JSON body:

| Status | When |
|:--|:--|
| `400` | Malformed JSON, invalid `content_hash`, missing required field. Body: `{"error": "..."}`. |
| `401` | Missing or wrong bearer token. |
| `403` | Authenticated but the token is read-only (OAuth scope) and the endpoint mutates. |
| `404` | Memory doesn't exist (`get_memory`, `patch_memory`). |
| `413` | Request body over 1 MB. |
| `429` | Rate limited (only when running behind a rate-limiting proxy). |
| `500` | Backend failure — Qdrant/FalkorDB/TEI unreachable, embedding timeout. Body is sanitized; check server logs for detail. |
| `503` | Server's internal work queue is saturated. Retry with backoff. |
