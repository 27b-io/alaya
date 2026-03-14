# CLAUDE.md

Ālaya — Rust/WASM MCP Memory Worker

Rust rewrite of the mcp-memory-service API layer, targeting Cloudflare Workers (WASM).

## Architecture

4-crate workspace:
- **alaya-types** — Shared types (compiles to wasm32-unknown-unknown)
- **alaya-bridge** — FalkorDB typed RPC bridge (native only: tokio, redis, axum)
- **alaya-core** — MemoryService orchestration (future)
- **alaya-worker** — CF Worker entry point (future)

## Spec

`../mcp-memory-service/docs/superpowers/specs/2026-03-14-alaya-rust-worker-design.md`

## Commands

```bash
# Build all (native)
cargo build --workspace

# Build WASM target (types only for now)
cargo build -p alaya-types --target wasm32-unknown-unknown

# Test
cargo test --workspace

# Lint
cargo clippy --workspace -- -D warnings

# Format check
cargo fmt --all -- --check

# Run bridge
cargo run -p alaya-bridge
```

## Quality Gates

Every commit must pass:
```bash
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo test --workspace
```

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
- Hop depths are `u8` capped at 3
- All errors sanitized before external exposure (`safe_message()`)
