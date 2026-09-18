# MCP tool reference

Ten tools, grouped by what they do. The authoritative JSON schemas live in [`crates/alaya-server/src/mcp.rs`](../crates/alaya-server/src/mcp.rs) — if anything below disagrees with the source, the source wins.

For wiring an MCP client to a running server, see [Quickstart: MCP clients](./quickstart-mcp.md). For the REST equivalents (different field names in a few places), see [REST API reference](./rest-api.md).

## The `content_hash` convention

Almost every tool below either returns or takes a `content_hash`. Two rules that will save you a debugging session:

1. **Always pass the full 64-character SHA-256 hex string.** Log lines and UI displays often truncate to 8 chars (`a3f4e891`) — those are display-only and will be rejected with `-32602 invalid params`.
2. **Pass exactly what `store_memory` or `search` gave you back.** Don't lowercase, don't strip prefixes, don't recompute from content. The hash uniquely identifies a stored memory.

## Tools at a glance

| Tool | Purpose | Mutates? |
|:--|:--|:-:|
| [`store_memory`](#store_memory) | Add a new memory | ✓ |
| [`search`](#search) | Retrieve memories — hybrid, vector, tag, recent, or full scan | |
| [`get_memory`](#get_memory) | Fetch one memory by exact `content_hash` | |
| [`delete_memory`](#delete_memory) | Hard-delete a memory | ✓ |
| [`check_database_health`](#check_database_health) | Backend health + storage stats | |
| [`relation`](#relation) | Create / read / delete typed edges between memories | ✓ (`create`/`delete`) |
| [`memory_supersede`](#memory_supersede) | Mark old memory as superseded by new | ✓ |
| [`memory_contradictions`](#memory_contradictions) | List contradiction pairs with judge verdicts | |
| [`resolve_contradiction`](#resolve_contradiction) | Keep both memories of a contradiction pair — non-destructive, reversible | ✓ (edge stamp only) |
| [`find_duplicates`](#find_duplicates) | Scan for near-duplicate memories | |
| [`merge_duplicates`](#merge_duplicates) | Supersede a cluster of duplicates in favour of one canonical | ✓ |

---

## `store_memory`

Embed text and persist it. Returns the new memory's `content_hash`.

| Param | Type | Required | Default | Notes |
|:--|:--|:-:|:--|:--|
| `content` | string | ✓ | | The text to store. Embedded for semantic search. |
| `tags` | string[] or string | | | Labels for tag-mode search. Accepts `["a","b"]` or `"a,b"`. |
| `memory_type` | enum | | `note` | One of `note`, `decision`, `task`, `reference`. Used by `memory_type` filter on `search`. |
| `metadata` | object | | | Arbitrary structured data. Special key: `importance` (float 0–1) boosts salience. |
| `client_hostname` | string | | | Tagged on the memory for provenance / multi-host setups. |
| `summary` | string | | | One-line summary (~50 tokens). Auto-generated if `SUMMARY_URL` is configured on the server. |
| `dedup_threshold` | number | | | If set, skip storage when nearest neighbour cosine similarity ≥ threshold. Use `0.95` for near-exact dedup. |

**Returns:** `{ "content_hash": "<64-hex>", "stored": true, "duplicate_of": null | "<hash>", ... }`

**Example:**

```json
{
  "name": "store_memory",
  "arguments": {
    "content": "Migrated frontend from npm to pnpm because of lockfile churn.",
    "tags": ["frontend", "tooling"],
    "memory_type": "decision",
    "metadata": {"importance": 0.7}
  }
}
```

---

## `search`

The one retrieval tool. Mode selects the algorithm.

| Param | Type | Default | Notes |
|:--|:--|:--|:--|
| `query` | string | `""` | Natural-language query. Required for `hybrid` and `similar`. |
| `mode` | enum | `hybrid` | One of `hybrid`, `scan`, `similar`, `tag`, `recent`. |
| `tags` | string[] | | Required for `tag` mode; optional filter on others. |
| `match_all` | bool | `false` | Tag mode: AND vs OR semantics. |
| `k` | int | `10` | Result cap for `scan` and `similar`. |
| `page` | int | `1` | Pagination for `hybrid` and `tag`. |
| `page_size` | int | `10` | |
| `min_similarity` | number | `0.3` | Drop results below this cosine similarity. |
| `output` | enum | `full` | `full` (content), `summary` (one-line), or `both`. |
| `memory_type` | string | | Restrict to one of the 4 types. |
| `encoding_context` | object | | Context-similarity reranking — pass the same shape used when storing. |
| `include_superseded` | bool | `false` | Set to `true` to see history of resolved contradictions. |
| `min_trust_score` | number | | Drop results below this provenance trust score. |
| `cursor` | number | | `recent` mode pagination — pass `next_cursor` from the previous response. |

**Modes:**

| Mode | What it does |
|:--|:--|
| `hybrid` | Vector + keyword fused with [RRF](https://qdrant.tech/documentation/concepts/hybrid-queries/), optionally re-scored by a cross-encoder if `RERANK_URL` is set. **Default — use this unless you have a reason not to.** |
| `similar` | Pure vector similarity. Faster than `hybrid` but ignores keyword signal. |
| `tag` | Filter by tag(s). With `match_all=true`, all tags must match. |
| `recent` | Reverse chronological. Use `cursor` to paginate. |
| `scan` | Full collection scan, ordered by creation time. Expensive — use only for admin / migration. |

**Returns:** array of `{ content_hash, content, similarity, summary?, metadata, ... }`.

**Example:**

```json
{
  "name": "search",
  "arguments": {
    "query": "why did we switch package managers?",
    "mode": "hybrid",
    "page_size": 5
  }
}
```

---

## `get_memory`

Fetch a single memory by its exact hash. Returns `{"found": false}` if it doesn't exist — superseded memories still return `{"found": true}` and you can check `metadata.superseded_by`.

| Param | Type | Required | Default | Notes |
|:--|:--|:-:|:--|:--|
| `content_hash` | string | ✓ | | Full 64-char SHA-256 hex. |
| `output` | enum | | `full` | `full`, `summary`, or `both`. |

**Use when** you already hold a hash — typically from a `search` result, a `memory_contradictions` pair, or a `find_duplicates` cluster — and want to re-inspect it without searching again.

**Example:**

```json
{
  "name": "get_memory",
  "arguments": {
    "content_hash": "a3f4e891b27c5d6e0123456789abcdef0123456789abcdef0123456789abcdef"
  }
}
```

---

## `delete_memory`

Hard-delete. Vector is removed from Qdrant, node from FalkorDB. **No tombstone, no recovery** — use `memory_supersede` if you might want history.

| Param | Type | Required |
|:--|:--|:-:|
| `content_hash` | string | ✓ |

**Returns:** `{ "deleted": true }` or an error.

---

## `check_database_health`

No params. Returns backend status and per-backend stats — vector count, graph node count, embedding endpoint reachability. Cheap; use it as a liveness signal from automation.

```json
{ "name": "check_database_health", "arguments": {} }
```

---

## `relation`

Manage typed edges between two memories in the knowledge graph. One tool with three actions.

| Param | Type | Required | Notes |
|:--|:--|:-:|:--|
| `action` | enum | ✓ | `create`, `get`, or `delete`. |
| `content_hash` | string | ✓ | Source memory hash. For `get`, all outgoing edges of this node are returned. |
| `target_hash` | string | for `create`/`delete` | Target memory hash. |
| `relation_type` | enum | for `create`/`delete` | One of `RELATES_TO`, `PRECEDES`, `CONTRADICTS`. |

**Relation types:**

| Type | Meaning |
|:--|:--|
| `RELATES_TO` | Generic association. Used by spreading-activation to boost search results. |
| `PRECEDES` | Temporal ordering — `A PRECEDES B` means A happened before B. |
| `CONTRADICTS` | Explicit contradiction. Surfaces in `memory_contradictions`. Resolve with `resolve_contradiction` (keep both) or `memory_supersede` (one memory misleads). `delete` refuses a `CONTRADICTS` edge that already carries a judge verdict or a resolution — it is the queue item and its audit trail. |

**Example (create):**

```json
{
  "name": "relation",
  "arguments": {
    "action": "create",
    "content_hash": "a3f4...e891",
    "target_hash": "b71c...0042",
    "relation_type": "PRECEDES"
  }
}
```

---

## `memory_supersede`

Mark `old_id` as superseded by `new_id`. The old memory stays in storage (so the history is auditable) but is filtered out of default `search` results, and every contradiction pair it is in leaves the queue. Use this to resolve a contradiction when one memory misleads. It is the destructive option — there is no un-supersede — so when both memories are true (verdict `coexist`) or the pair is detector noise (`unrelated`), use [`resolve_contradiction`](#resolve_contradiction) instead.

| Param | Type | Required | Default |
|:--|:--|:-:|:--|
| `old_id` | string | ✓ | |
| `new_id` | string | ✓ | |
| `reason` | string | | `""` |

> [!IMPORTANT]
> **Field-name divergence.** `supersede` takes `old_hash`/`new_hash` on REST (`POST /supersede`) and `old_id`/`new_id` in MCP (`memory_supersede`). The **values are identical** — full 64-char content hashes; only the field names differ. (`relation` and `delete` use `content_hash`/`target_hash` on *both* protocols — no divergence.)

See [REST: `POST /supersede`](rest-api.md#post-supersede) for the REST equivalent.

**Example:**

```json
{
  "name": "memory_supersede",
  "arguments": {
    "old_id": "a3f4...e891",
    "new_id": "c0de...beef",
    "reason": "Switched from pnpm back to npm after migration was reverted."
  }
}
```

---

## `memory_contradictions`

List pairs of memories the contradiction detector has flagged (via negation, antonym, or temporal cues), each with the LLM judge's advisory verdict. The lexical detector over-fires on same-project progress notes, so triage on `verdict`: `contradiction` and `supersession` are worth a look, `coexist` / `unrelated` are detector noise. Pairs already resolved — by `memory_supersede` or by `resolve_contradiction` — are hidden by default.

| Param | Type | Default | Notes |
|:--|:--|:--|:--|
| `limit` | int | `20` | Page size (1–500), newest first. |
| `offset` | int | `0` | Page cursor — pass back the previous response's `next_offset` until it is `null`. |
| `include_resolved` | bool | `false` | `true` also returns resolved pairs: one memory superseded, or the pair stamped `keep_both`. Filtered in the graph, so paging always reaches the unresolved pairs. |
| `verdicts` | string[] | `["contradiction","supersession","unjudged"]` | Only pairs whose verdict is in the list. `unjudged` = not judged yet, or a pair the judge could not classify (`verdict_reason` says why). |

**Returns:** `{ success, pairs: [ ... ], total, next_offset }` where each pair is:

| Field | Notes |
|:--|:--|
| `memory_a_hash`, `memory_b_hash` | The flagged pair (`a` is the newer memory that triggered detection). |
| `confidence` | Lexical detector confidence (`0.8` antonym, `0.7` negation/temporal). |
| `memory_a_content`, `memory_b_content` | Summary, or the first 200 chars of content. |
| `memory_a_superseded`, `memory_b_superseded` | Whether that memory is already superseded. |
| `verdict` | `contradiction` \| `supersession` \| `coexist` \| `unrelated` \| `unjudged`. |
| `verdict_reason` | One line from the judge; `null` when never judged, `unjudged: <error>` when the judge failed on this pair. |
| `survivor` | `content_hash` the judge recommends keeping, or `null`. Advisory — pass it to `memory_supersede` yourself. |
| `verdict_confidence`, `verdict_model`, `judged_at` | Judge self-reported confidence (0–1), model id, epoch seconds. |
| `resolution`, `resolved_at`, `resolved_via` | `keep_both` stamp from `resolve_contradiction`, when it was set (epoch seconds, server clock) and by whom (`operator:mcp`, `operator:console`, …). All `null` on an unresolved pair. Only visible with `include_resolved: true`. |

The judge never writes to a memory: verdicts live on the graph edge, and resolution stays a human/agent call. A pair leaves the default page one of three ways:

| Exit | Verb | Destructive? | Reverse |
|:--|:--|:-:|:--|
| Supersede | `memory_supersede` — the loser leaves default search, a `SUPERSEDES` edge is written | yes (audit trail kept) | none — store the memory again |
| Keep both | `resolve_contradiction` `keep_both` — stamps the edge, both memories stay live | no | `resolve_contradiction` with `resolution: null` |
| Hidden by filter | default `verdicts` omit `coexist` / `unrelated` — nothing is written | no | pass `verdicts` including them |

---

## `resolve_contradiction`

Resolve a pair from `memory_contradictions` **without superseding or deleting anything**. `resolution: "keep_both"` stamps the pair as settled — both memories are true, or the pair is detector noise — so it leaves the default queue while both memories stay searchable and the judge's verdict stays on the edge. `resolution: null` clears the stamp and the pair returns to the queue. This is the only way to write the stamp: `relation` cannot set it and the judge never touches it, so no agent's ordinary graph write can retire a pair from the human queue.

| Param | Type | Required | Notes |
|:--|:--|:-:|:--|
| `memory_a_hash` | string | ✓ | Verbatim from the `memory_contradictions` row — the edge is directed, so the order matters. |
| `memory_b_hash` | string | ✓ | Verbatim from the same row. |
| `resolution` | `"keep_both"` \| `null` | ✓ | The key must be present: an omitted key is an invalid-params error, never a silent clear. |

The stamp is recorded as `resolved_via: "operator:mcp"` with a server-set `resolved_at`; the MCP surface does not accept a caller-supplied `resolved_via`. Prefer this over `memory_supersede` for verdict `coexist` or `unrelated`.

**Returns:** `{ success, memory_a_hash, memory_b_hash, resolution, resolved_at, resolved_via }` — the last three `null` after a clear. A pair with no `CONTRADICTS` edge in that direction is a not-found error; nothing is created.

**Example:**

```json
{
  "name": "resolve_contradiction",
  "arguments": {
    "memory_a_hash": "a3f4...e891",
    "memory_b_hash": "c0de...beef",
    "resolution": "keep_both"
  }
}
```

---

## `find_duplicates`

Scan up to `limit` memories for near-duplicates by embedding cosine similarity. **Does not mutate anything** — it's a planning tool. To act on the clusters it finds, call `merge_duplicates`.

| Param | Type | Default | Notes |
|:--|:--|:--|:--|
| `similarity_threshold` | number | `0.95` | Cosine similarity cutoff for "near-duplicate". |
| `limit` | int | `500` | Max memories to scan. |
| `strategy` | enum | `keep_newest` | Which memory in a cluster to mark canonical: `keep_newest`, `keep_oldest`, or `keep_most_accessed`. |

**Returns:** clusters, each with a `canonical_hash` and a list of `duplicate_hashes`.

---

## `merge_duplicates`

Supersede each `duplicate_hashes` entry with `canonical_hash`. Typically called after inspecting `find_duplicates` output.

| Param | Type | Required | Default |
|:--|:--|:-:|:--|
| `canonical_hash` | string | ✓ | |
| `duplicate_hashes` | string[] | ✓ | |
| `reason` | string | | `Merged by deduplication` |
| `dry_run` | bool | | `false` |

**Use `dry_run: true` first.** It returns the same shape as a real merge but doesn't write — useful to confirm you got the cluster right before destroying.

---

## Errors

All tools return JSON-RPC error codes. The ones you'll see most:

| Code | Meaning |
|:--|:--|
| `-32600` | Invalid request — usually a missing `jsonrpc` or `method`. |
| `-32601` | Method (tool) not found — check the name. |
| `-32602` | Invalid params — wrong type, missing required field, or a truncated `content_hash`. |
| `-32603` | Internal error — backend issue, see server logs. |

Backend failures (Qdrant unreachable, embedding timeout) come back as `-32603` with a sanitized message — Ālaya never leaks backend internals to clients.
