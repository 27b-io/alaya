# REST API reference

Ālaya's REST API and MCP tools cover the same operations against the same storage. Use REST for scripts, backfill jobs, and any client that doesn't speak MCP. Use [MCP](./mcp-tools.md) for LLM agents.

All endpoints below assume the server is reachable at `http://localhost:3001` — substitute your own base URL.

## Authentication

Set `ALAYA_API_KEY` (and/or `ALAYA_READONLY_API_KEY`, below) on the server to require authentication. Clients then send:

```http
Authorization: Bearer <your-api-key>
```

Auth is **fail-closed**: with no credential configured (`ALAYA_API_KEY`, `ALAYA_READONLY_API_KEY`, and `OIDC_ISSUER` all empty) the server refuses to boot — unless `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` is set, which the dev Compose does for `localhost` only (it is refused on any non-private `PUBLIC_BASE_URL`). Set a static key before exposing the server; either one enables bearer auth and disables the dev-open flag (a read-only deployment may set just `ALAYA_READONLY_API_KEY`).

If the server has `OIDC_ISSUER` set, clients use an OAuth access token instead of a fixed key. See [MCP quickstart → OAuth](./quickstart-mcp.md#oauth-optional) — the same flow applies to REST clients.

### Read-only bearer (optional)

Set `ALAYA_READONLY_API_KEY` (must differ from `ALAYA_API_KEY`) to mint a second static bearer for headless service consumers that must never mutate the corpus (e.g. a read-only dashboard). It authenticates the same way but is authorized for pure reads only — `POST /search`, `GET /memories/{content_hash}`, `POST /contradictions`, `POST /duplicates/find`, `GET /health/detail` — and receives `403 Forbidden` on every mutating route, including `POST /store`:

```bash
# succeeds
curl -H "Authorization: Bearer $ALAYA_READONLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "deploy checklist"}' http://localhost:3001/search

# 403 — read-only bearer cannot mutate
curl -H "Authorization: Bearer $ALAYA_READONLY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content_hash": "…"}' http://localhost:3001/delete
```

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
| `POST` | `/contradictions` | List contradiction pairs with judge verdicts | yes |
| `POST` | `/contradictions/resolution` | Keep both memories of a pair, or undo that | yes |
| `POST` | `/duplicates/find` | Scan for near-duplicates | yes |
| `POST` | `/duplicates/merge` | Supersede a duplicate cluster | yes |
| `POST` | `/backfill/summaries` | Generate missing summaries | yes |
| `POST` | `/backfill/contradictions` | Judge unjudged contradiction pairs | yes |
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

`status` is `healthy` when the service worker is live and Qdrant is reachable,
`degraded` (HTTP 200) when Qdrant is down — restarting the pod won't fix Qdrant —
and `unhealthy` (HTTP 503) when the service worker has stalled, so a liveness
probe restarts it. The bare probe's verdict ignores the graph and it never
probes the embedding endpoint; both verdicts are reported only on
`/health/detail`.

Probers read the HTTP code, so a k8s `httpGet` probe and `curl -sf .../health`
both work against this endpoint unchanged.

This route does no backend I/O per request. The worker-stall check (the only
input to the 503) is read live; the Qdrant verdict behind `healthy` vs
`degraded` is refreshed by an internal task every 30s and may lag a Qdrant
outage by up to ~40s. Neither the HTTP code nor any restart decision depends on
it. Use `/health/detail` for a live per-backend view.

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
  "embedding_health": { "status": "healthy" },
  "total_memories": 1247
}
```

Same HTTP-code mapping as `/health`. `status` additionally folds in the
embedding probe: with the embedding endpoint down, `/health/detail` reports
`degraded` and `embedding_health` carries the reason —

```json
  "status": "degraded",
  "embedding_health": {
    "status": "unhealthy",
    "error": "error sending request for url (http://embeddings/health)"
  }
```

A stalled worker still reports `unhealthy` (503) regardless: an embedding
outage is not fixed by a restart, a wedged worker is.

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

{"limit": 20, "offset": 0, "include_resolved": false, "verdicts": ["contradiction", "supersession", "unjudged"]}
```

| Field | Default | Notes |
|:--|:--|:--|
| `limit` | `20` | Page size (1–500), newest edge first. |
| `offset` | `0` | Page cursor: pass back the previous response's `next_offset`. |
| `include_resolved` | `false` | `false` hides resolved pairs: either memory superseded, or the pair stamped `keep_both` via [`POST /contradictions/resolution`](#post-contradictionsresolution). The filter runs in the graph (an incoming `SUPERSEDES` edge or `e.resolution` is the resolved state), so a run of resolved pairs at the top of the queue never hides the rest. |
| `verdicts` | `["contradiction","supersession","unjudged"]` | Only pairs whose judge verdict is in the list. `unjudged` = no verdict yet, or a pair the judge could not classify (see `verdict_reason`). Add `coexist` / `unrelated` to see everything. Unknown values are a `400`. |

Each pair carries the lexical detector's `confidence` plus the judge's advisory verdict (LAB-3283). A verdict never mutates a memory; `survivor` is a recommendation for `POST /supersede`.

```json
{
  "success": true,
  "pairs": [
    {
      "memory_a_hash": "a3f4...",
      "memory_b_hash": "c0de...",
      "confidence": 0.7,
      "created_at": 1786288228.58,
      "memory_a_content": "Switched to pnpm — lockfile churn was killing CI cache.",
      "memory_b_content": "We use npm; pnpm was reverted.",
      "memory_a_superseded": false,
      "memory_b_superseded": false,
      "verdict": "supersession",
      "verdict_reason": "B records the pnpm rollback that replaces A's switch.",
      "survivor": "c0de...",
      "verdict_confidence": 0.91,
      "verdict_model": "claude-haiku-4-5-20251001",
      "judged_at": 1789000000.0,
      "resolution": null,
      "resolved_at": null,
      "resolved_via": null
    }
  ],
  "total": 1,
  "next_offset": null
}
```

`verdict` is one of `contradiction`, `supersession`, `coexist`, `unrelated`, or `unjudged`. An edge the judge has never seen has `null` for the other verdict fields; an edge the judge failed on deterministically (unparseable or empty answer, request rejected, endpoint missing) is `unjudged` with `verdict_reason` starting `unjudged:` and `verdict_model` set. `next_offset` is set whenever the graph page was full (an exactly-full last page yields a cursor to an empty page) and `null` once the server knows nothing follows. A page can hold fewer than `limit` pairs when Qdrant marks a memory superseded that the graph does not yet know about; the cursor still advances. Pure read — the read-only bearer may call it.

`resolution` / `resolved_at` / `resolved_via` carry the `keep_both` stamp written by `POST /contradictions/resolution` (all `null` when unresolved; only visible with `include_resolved: true`). A pair leaves the default page one of three ways: **supersede** (`POST /supersede` — destructive with an audit trail, no un-supersede), **keep both** (`POST /contradictions/resolution` — non-destructive, reversed by the same route with `resolution: null`), or **hidden by the verdict filter** (the default `verdicts` omit `coexist` / `unrelated`; nothing is written).

## `POST /contradictions/resolution`

Resolve a pair from `POST /contradictions` **without superseding or deleting anything**. `"keep_both"` stamps the `memory_a_hash -> memory_b_hash` `CONTRADICTS` edge so the pair leaves the default queue while both memories stay searchable and the judge's verdict stays put; `null` clears the stamp and the pair returns. This route is the only writer of the stamp — `POST /relation` cannot set it and the judge never touches it. A `CONTRADICTS` edge carrying a verdict or a resolution also cannot be deleted through `POST /relation` (`delete`): it is the queue item and its audit trail, and the call fails with `Edge carries a verdict or resolution; resolve it (keep_both / supersede) instead of deleting`.

```http
POST /contradictions/resolution
Content-Type: application/json

{"memory_a_hash": "a3f4...", "memory_b_hash": "c0de...", "resolution": "keep_both", "resolved_via": "operator:console"}
```

| Field | Required | Notes |
|:--|:-:|:--|
| `memory_a_hash`, `memory_b_hash` | ✓ | Verbatim from the `POST /contradictions` row — the edge is directed. |
| `resolution` | ✓ | `"keep_both"` to resolve, `null` to undo. The key must be present: an absent key is rejected (`422`), never read as a clear. |
| `resolved_via` | ✓ | Who resolved, recorded verbatim (1–128 chars): `operator:console`, `engine:<run-id>`, … The MCP tool fixes this to `operator:mcp`. |

`resolved_at` is set by the server. Returns `{ "success": true, "memory_a_hash", "memory_b_hash", "resolution", "resolved_at", "resolved_via" }` — the last three `null` after a clear. No `CONTRADICTS` edge in that direction returns `{ "success": false, "error": "Resource not found" }` (same shape as `/supersede`); nothing is created. The stamp sits on the directed edge you named, but the queue treats the pair as resolved when either direction carries one — a re-store of the older memory re-detects the pair the other way round, and that fresh edge must not undo the operator's call. Mutating — static bearer only.

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

## `POST /backfill/contradictions`

Judge `CONTRADICTS` pairs that have no verdict yet (LAB-3283). Operator-only, same auth class as `/backfill/summaries` — the read-only bearer gets `403`.

```http
POST /backfill/contradictions
Content-Type: application/json

{"limit": 100, "rejudge": false}
```

Blocks until the pass completes and returns what happened:

```json
{"queued": 100, "judged": 95, "persisted": 95, "marked": 2, "unjudged": 3, "input_tokens": 231044, "output_tokens": 7112}
```

`limit` defaults to `100` (max 500 per call). At most 4 judge calls are in flight — a cap shared with the judging that runs after each `POST /store` — and a `429` from the provider is retried with backoff (honouring `retry-after`). Resolved pairs (an endpoint with a `SUPERSEDES` edge) are not judged.

Failures are classified so one poison pair cannot stall the pass or re-bill forever:

- **deterministic** (unparseable or empty answer, request rejected with 400/413/422) → the edge is marked `verdict = unjudged` with `verdict_reason = "unjudged: <error>"` and the configured model; counted in `marked`, skipped by later passes, visible on `POST /contradictions` under `verdicts: ["unjudged"]`.
- **transient** (timeout, 5xx, 429 after retries, upstream misconfiguration, endpoint missing) → nothing is written; counted in `unjudged`, retried next pass.
- `judged - persisted` is verdicts produced that did not land on an edge (graph blip); they are retried next pass.

Re-running is idempotent: only edges with no verdict at all are selected. **Switching `JUDGE_MODEL`** does not touch existing verdicts; run with `"rejudge": true` to also re-annotate every edge whose `verdict_model` differs from the configured model **and every `unjudged` marker** (the operator's way to retry deterministic failures after a fix), paging with `limit` until `queued` is `0`. A marker never overwrites a real verdict — if a re-judge fails on a pair that already has one, the old verdict stands. Verdicts are written onto the graph edge only; no memory record is modified.

One pass at a time: a second call while one is running returns `{"success": false, "error": "backfill already running"}`. The HTTP reply waits at most 630 s, so keep `limit` around 200 per call; a pass that outlives the reply still runs to completion and the next call is refused until it finishes.

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
| `403` | Authenticated, but the principal is not authorized for this endpoint: the `ALAYA_READONLY_API_KEY` bearer on anything but a pure read, or an OIDC bearer on a mutating route (delete / supersede / contradictions/resolution / merge / relation / patch / backfill). OAuth scopes are not evaluated. |
| `404` | Memory doesn't exist (`get_memory`, `patch_memory`). |
| `413` | Request body over 1 MB. |
| `429` | Rate limited (only when running behind a rate-limiting proxy). |
| `500` | Backend failure — Qdrant/FalkorDB/TEI unreachable, embedding timeout. Body is sanitized; check server logs for detail. |
| `503` | Server's internal work queue is saturated. Retry with backoff. |
