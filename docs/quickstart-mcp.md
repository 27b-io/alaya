# Quickstart: MCP clients

Point a Model Context Protocol client (Claude Code, Claude Desktop, etc.) at a running Ālaya server. Assumes the server is already up — if not, see [Self-hosting](./quickstart-selfhost.md) first.

## You need

- A running Ālaya server reachable at some URL (e.g. `http://localhost:3001` for local docker compose, or `https://alaya.example.com` for a hosted deployment).
- The `ALAYA_API_KEY` for that server. Auth is fail-closed: the dev Compose opts into open mode on `localhost` (`DANGEROUSLY_ALLOW_UNAUTHENTICATED=true`, refused on any non-private origin), so a local dev box needs no key. Set `ALAYA_API_KEY` before exposing the server — that enables `Authorization: Bearer` auth and disables the dev-open flag.
- An MCP client. The examples below are for Claude Code and Claude Desktop.

## Claude Code

Claude Code uses MCP server configurations in `~/.claude/mcp_servers.json` (global) or per-project. The MCP endpoint is the server's base URL plus `/mcp`:

```json
{
  "mcpServers": {
    "alaya": {
      "type": "http",
      "url": "http://localhost:3001/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_ALAYA_API_KEY"
      }
    }
  }
}
```

Omit the `Authorization` header when `ALAYA_API_KEY` is empty.

Restart Claude Code. The 10 Ālaya tools (`store_memory`, `search`, `get_memory`, …) appear in the tool list. Try:

> Store a note: "Switching to pnpm for the frontend — npm's lockfile churn is killing CI cache hits."

## Claude Desktop

Claude Desktop reads `claude_desktop_config.json` (location depends on your OS — see Anthropic's docs). Same shape:

```json
{
  "mcpServers": {
    "alaya": {
      "type": "http",
      "url": "https://alaya.example.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_ALAYA_API_KEY"
      }
    }
  }
}
```

Omit the `Authorization` header when `ALAYA_API_KEY` is empty (a localhost dev box in open mode).

Restart Claude Desktop. The Ālaya tools show up under the connections panel.

## OAuth (optional)

If the server was started with `OIDC_ISSUER` set, it acts as an [RFC 9728 OAuth Protected Resource](https://datatracker.ietf.org/doc/rfc9728/): clients discover the authorization server via `GET /.well-known/oauth-protected-resource/mcp` and then perform a standard OAuth 2.1 PKCE flow against your IdP (e.g. Auth0, Keycloak, claude.ai).

MCP clients that support OAuth (claude.ai's hosted MCP, for example) will:

1. Probe `/.well-known/oauth-protected-resource/mcp` and receive `{"resource": "...", "authorization_servers": ["<your-issuer>"], ...}`.
2. Send the user through your IdP's login flow.
3. Attach the resulting access token as `Authorization: Bearer <token>` on every request.

When `OIDC_ISSUER` is **unset**, the well-known endpoints return `404` and clients fall back to whatever bearer token you give them — typically a fixed `ALAYA_API_KEY`.

## Verify

The cheapest "is the server reachable from this client?" check is the unauthenticated health endpoint:

```bash
curl http://localhost:3001/health
# {"status":"ok","qdrant":"ok","graph":"ok","embedding":"ok",...}
```

If you set an API key, also test it works:

```bash
curl -H "Authorization: Bearer $ALAYA_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
     -H "Accept: application/json, text/event-stream" \
     http://localhost:3001/mcp
```

You should see all 10 tool schemas come back.

## First memory, end-to-end

From inside an MCP-enabled session, ask the model to:

1. Store a memory.
2. Search for it.
3. Inspect the result.

```text
You: Use the alaya store_memory tool to save: "Project alpha uses Rust 1.87 with the channel-based axum architecture — see CLAUDE.md."

[model calls store_memory → returns content_hash: "a3f4...e891"]

You: Now search alaya for "rust axum project"
[model calls search → returns the memory above with similarity score ~0.6]
```

That's it — you're integrated. From here, see the [MCP tool reference](./mcp-tools.md) for the full surface.

## Troubleshooting

| Symptom | Likely cause |
|:--|:--|
| Client shows "0 tools" | `Authorization` header missing or wrong; check `GET /health` works from the client's host |
| Tools listed but calls return `401 Unauthorized` | `ALAYA_API_KEY` mismatch; server logs the expected prefix on startup |
| `503 service unavailable` on `/store` or `/search` | Backend down (Qdrant, TEI, or FalkorDB); check `GET /health/detail` (authenticated) for which one |
| MCP client tries OAuth instead of bearer | Server has `OIDC_ISSUER` set; either configure OAuth on the client or unset `OIDC_ISSUER` on the server for dev |
