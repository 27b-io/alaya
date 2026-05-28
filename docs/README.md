# Ālaya Documentation

Ālaya is a persistent semantic-memory service: an MCP server that stores text, vector-searches it back, tracks relationships between memories, and resolves contradictions.

It speaks two protocols against the same storage layer:

- **MCP** (Model Context Protocol) at `POST /mcp` — for Claude Code, Claude Desktop, and other MCP clients.
- **REST** at `POST /store`, `POST /search`, etc. — for direct HTTP integration, scripts, and backfill jobs.

## Getting started

| You want to… | Read |
|---|---|
| Connect an MCP client (Claude Code / Desktop) to a running server | [Quickstart: MCP clients](./quickstart-mcp.md) |
| Run your own server with `docker compose` | [Quickstart: Self-hosting](./quickstart-selfhost.md) |

## Reference

| Topic | Read |
|---|---|
| All 10 MCP tools — schemas, params, examples | [MCP tool reference](./mcp-tools.md) |
| REST endpoints — auth, request/response shapes | [REST API reference](./rest-api.md) |

## How it works (one paragraph)

You write text via `store_memory`. Ālaya embeds it (via [TEI](https://github.com/huggingface/text-embeddings-inference)), stores the vector in [Qdrant](https://qdrant.tech) and the metadata in [FalkorDB](https://www.falkordb.com), and computes a salience score plus a SHA-256 `content_hash` you'll reference everywhere. Later you `search` with a natural-language query and get back the most relevant memories — by default using hybrid retrieval (vector + keyword, fused with RRF) and optional cross-encoder rerank. Contradictions are detected automatically; you resolve them with `memory_supersede`. Relationships between memories (`RELATES_TO`, `PRECEDES`, `CONTRADICTS`) are stored as a graph and used to boost spreading-activation results.

## Source of truth

These docs describe the released `main` branch. The authoritative tool schemas live in [`crates/alaya-server/src/mcp.rs`](../crates/alaya-server/src/mcp.rs) — if a doc and the code disagree, the code wins and the docs need a fix.
