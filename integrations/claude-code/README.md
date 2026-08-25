# Ālaya Memory Hooks (Claude Code plugin)

Two hooks that make Claude Code sessions persist into an [Ālaya](../../README.md) long-term
memory server:

- **Stop** (`scripts/alaya-session-save.sh`) — async, after each turn ends. Extracts 0–3
  memories from the transcript via an LLM and POSTs them to Ālaya's `/store` endpoint, with
  gates against saving too often and a dedup threshold to catch LLM rephrases of the same fact.
- **PreCompact** (`scripts/alaya-precompact.sh`) — blocks compaction (exit 2) unless a
  `mcp__alaya__store_memory` call shows up in the last 500 transcript lines, so valuable
  context doesn't get compacted away unsaved. Fails closed (blocks) if the transcript can't be read or parsed — treated the same as "no recent save."

## Prerequisites

- `bash`
- `jq`
- `curl`
- `python3` (used by the PreCompact hook to parse the hook's stdin JSON)
- An Ālaya server reachable from wherever `claude` runs, with `ALAYA_API_KEY` auth enabled
- An OpenAI chat-completions-compatible LLM endpoint for transcript extraction (the Stop hook
  calls this once per save; it does not have to be the same provider as the agent itself)

## Install

### Option A — marketplace + plugin install (recommended)

```
/plugin marketplace add 27b-io/alaya
/plugin install alaya-memory-hooks@alaya
```

or non-interactively:

```bash
claude plugin marketplace add 27b-io/alaya
claude plugin install alaya-memory-hooks@alaya
```

### Option B — local testing without installing

```bash
claude --plugin-dir /path/to/alaya/integrations/claude-code
```

Runs the hooks straight from a checkout — useful while editing them.

Either way, validate the manifest any time you change it:

```bash
claude plugin validate integrations/claude-code --strict
```

## Configure

The hooks read plain environment variables — nothing is hardcoded. Copy the values you need
from [`config/alaya-hook.env.example`](config/alaya-hook.env.example) into your shell profile,
an env file you `source` before launching `claude`, or the top-level `"env"` key in
`~/.claude/settings.json`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `ALAYA_URL` | yes | — | Ālaya `/store` endpoint |
| `ALAYA_API_KEY` / `ALAYA_API_KEY_CMD` | yes | — | Ālaya bearer token, direct value or a command that prints it |
| `ALAYA_LLM_URL` | yes | — | OpenAI-chat-completions-compatible endpoint used for extraction |
| `ALAYA_LLM_API_KEY` / `ALAYA_LLM_API_KEY_CMD` | yes | — | LLM gateway key, direct value or a command that prints it |
| `ALAYA_LLM_MODEL` | no | `claude-haiku-4-5` | Extraction model |
| `ALAYA_MIN_DURATION_SECS` | no | `120` | Skip sessions shorter than this |
| `ALAYA_MIN_NEW_MESSAGES` | no | `5` | Skip unless N+ new user messages since the last save |
| `ALAYA_COOLDOWN_SECS` | no | `300` | Skip if saved within this many seconds |
| `ALAYA_MEMORY_CAP` | no | `3` | Max memories stored per Stop |
| `ALAYA_DEDUP_THRESHOLD` | no | `0.70` | `dedup_threshold` sent to Ālaya's store call |
| `ALAYA_IMPORTANCE` | no | `0.7` | `metadata.importance` sent to Ālaya's store call |
| `ALAYA_CLIENT_HOSTNAME` | no | `claude-code-hook` | `client_hostname` provenance tag |
| `ALAYA_HOOK_STATE_DIR` | no | `~/.cache/alaya-hook` | Per-session save state, resolved-secret cache, `failures.log` |
| `ALAYA_SECRET_CACHE_MINUTES` | no | `720` | How long a `_CMD`-resolved secret is cached |
| `ALAYA_PRECOMPACT_RECENT_LINES` | no | `500` | PreCompact: how far back to scan for a recent save |

**Value-or-command secrets:** `ALAYA_API_KEY` and `ALAYA_LLM_API_KEY` can each be set directly,
or sourced from a command via the `_CMD` variant (e.g. `ALAYA_API_KEY_CMD='op read op://vault/item/field'`
for a 1Password service account). The command's output is cached under `ALAYA_HOOK_STATE_DIR`
for `ALAYA_SECRET_CACHE_MINUTES`, and a stale cache is served if the command fails — so saves
keep working through a secret manager's rate-limit windows. Delete the cache file (named after
the variable, e.g. `alaya-api-key`) to force a refresh after a key rotation.

**Missing configuration is a silent no-op, not a hook error:** if `ALAYA_URL`, `ALAYA_LLM_URL`,
or either secret can't be resolved, the Stop hook logs one line to
`$ALAYA_HOOK_STATE_DIR/failures.log` and exits 0 — it never surfaces an error into the session.
Check that file first when a save doesn't happen.

## Migrating from a manual `~/.claude/hooks/` setup

If you were running these as loose scripts registered by hand in `~/.claude/settings.json`
(the pre-plugin setup), after installing the plugin:

1. Remove the `Stop` and `PreCompact` entries pointing at your old script paths (e.g.
   `~/.claude/hooks/alaya-session-save.sh`, `~/.claude/hooks/alaya-precompact.sh`) from
   `~/.claude/settings.json` — otherwise both the old and new copies fire on every Stop/PreCompact.
2. Move whatever hardcoded endpoint/key/vault values lived in your old scripts into the env vars
   above.
3. Delete (or keep as a backup) the old script files — the plugin no longer reads them.

## Testing

```bash
scripts/test-secret-cache.sh
```

Proves the value-or-command secret resolver (`_resolve_secret` in `alaya-session-save.sh`):
cold fetch calls the resolver once, a warm cache calls it zero times, the cache file is written
`0600`, a stale cache is served if the resolver starts failing, and the hook script still parses.

## Troubleshooting

- **No memories are showing up** — check `$ALAYA_HOOK_STATE_DIR/failures.log` (default
  `~/.cache/alaya-hook/failures.log`). Every failure path (missing config, unresolved secret,
  dead LLM endpoint, failed store call) logs one line there — see LAB-170: a silent failure
  here once cost 5 days of unsaved memories.
- **PreCompact keeps blocking** — it only unblocks once `mcp__alaya__store_memory` appears in
  the last `ALAYA_PRECOMPACT_RECENT_LINES` transcript lines; call it manually, or lower the
  Stop hook's gates (`ALAYA_MIN_NEW_MESSAGES`, `ALAYA_COOLDOWN_SECS`) so it fires sooner.
- **`claude plugin validate` fails** — run it with `--strict` locally before pushing; it flags
  unrecognized fields and missing metadata that the runtime otherwise tolerates silently.
