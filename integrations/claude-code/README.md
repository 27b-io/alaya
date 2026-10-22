# Ālaya Memory Hooks (Claude Code plugin)

Two hooks that make Claude Code sessions persist into an [Ālaya](../../README.md) long-term
memory server:

- **Stop** (`scripts/alaya-session-save.sh`) — after each turn ends. Extracts 0–3
  memories from the transcript via an LLM and POSTs them to Ālaya's `/store` endpoint, with
  gates against saving too often and a dedup threshold to catch LLM rephrases of the same fact.
  The hook returns within milliseconds and does the save in a detached copy of itself, so the
  save survives the CLI exiting right after its last turn, as SDK-driven runs do.
- **PreCompact** (`scripts/alaya-precompact.sh`) — trigger `auto` (proactive and reactive
  compaction) always passes: Claude Code never shows Claude the block message of an automatic
  compaction, so a block there only skips compaction until the session hits the context limit.
  Every other trigger (a manual `/compact`, idle compaction, which Claude Code sends as
  `manual`, or a missing or unknown one) is blocked (exit 2) unless a
  `mcp__alaya__store_memory` call shows up in the last 500 transcript lines, so valuable
  context doesn't get compacted away unsaved. When nothing qualifies, Claude writes
  `ALAYA_NOTHING_TO_SAVE` alone as the last line of its reply instead (nothing after it in that
  turn, not even a tool call); only Claude's own text
  counts, never tool results, user turns or tool inputs, and a mention of the marker inside
  a sentence, a quote or a code-block example does not count. Fails closed (blocks) if the transcript or the hook input can't
  be read or parsed — treated the same as "no recent save." Individual malformed transcript
  lines are skipped, not fatal, but one that follows the marker clears it, so a parse error
  can only block.

## Prerequisites

- `bash`
- `jq`
- `curl`
- `python3` (PreCompact stdin parsing; Stop hook timestamp parsing on macOS/BSD, where GNU `date -d` is unavailable)
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
| `ALAYA_SECRET_CACHE_MINUTES` | no | `720` | How long a `_CMD`-resolved secret is cached; `0` writes no cache file (see below) |
| `ALAYA_PRECOMPACT_RECENT_LINES` | no | `500` | PreCompact: how far back to scan for a recent save |

**Transport security:** both endpoints receive a bearer token on every call. Use `https://`
for anything that crosses a network you don't fully trust. Plain `http://` is only appropriate
when the transport itself is trusted or encrypted underneath — loopback, an in-cluster service
address (e.g. `http://alaya-server.mcp.svc:3001/store`), or a WireGuard/Tailscale underlay.
The hooks do not enforce this; the endpoint choice is yours.

**Value-or-command secrets:** `ALAYA_API_KEY` and `ALAYA_LLM_API_KEY` can each be set directly,
or sourced from a command via the `_CMD` variant (e.g. `ALAYA_API_KEY_CMD='op read op://vault/item/field'`
for a 1Password service account). The command's output is cached under `ALAYA_HOOK_STATE_DIR`
for `ALAYA_SECRET_CACHE_MINUTES`, and a stale cache is served if the command fails — so saves
keep working through a secret manager's rate-limit windows. A failed command is retried at most
once every 15 minutes, not on every Stop, so retries don't pile onto a limit that is already
rejecting them. Delete the cache file (named after the variable, e.g. `alaya-api-key`) to force a
refresh after a key rotation; if a refresh failed in the last 15 minutes, delete its `.attempt`
marker too.

The cache is a plain-text file (mode 0600). Set `ALAYA_SECRET_CACHE_MINUTES=0` to keep no copy at
all: the command then runs on every Stop, nothing is written under `ALAYA_HOOK_STATE_DIR` but
`failures.log`, every Stop deletes both cache files and their `.attempt` markers left from an
earlier setting, and a failed command (non-zero exit, even with output, or empty output) skips the
save (logged as `unresolved`) with no stale fallback and no backoff. Use
it when the command does its own caching and failure handling, so the hook's copy would be a second,
unencrypted one.

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
`0600`, a stale cache is served if the resolver starts failing, a failing resolver is retried at
most once per 15 minutes (with or without a stale cache, and never leaves an empty cache file),
and the hook script still parses.

```bash
scripts/test-prompt-extraction.sh [transcript.jsonl]
```

Runs the real Stop hook against a stub `curl` and proves what reaches the extractor: prompts are
read whether they are strings (terminal sessions) or lists of text blocks (SDK and agent
sessions), while tool results, `isMeta` injections such as skill bodies, subagent (`isSidechain`)
task prompts, `[Request interrupted by user]` markers, and `<wrapper>` messages like
`<system-reminder>` are left out. A typed prompt that starts with `<` is left out too. In a
Multica issue run, each per-turn template is reduced to the human comment it quotes: the
extractor sees that request and no template text, and comments quoted from agents or the
platform, and wakeups, add nothing. The save gate still counts every prompt line. Pass a real
transcript to also run a smoke replay. It checks only that a non-zero message count reaches the
extractor, not which lines those are, and it turns off the duration and message-count save gates
so a short or still-running session counts too.

```bash
scripts/test-stop-lifetime.sh
```

Starts the real Stop hook the way a CLI exit treats it: the parent closes the hook's stdin, then
kills the hook's process group the moment the hook returns (or after a second if it does not),
while the stub extractor takes two seconds. It
proves the store call still happens and the state file is written, with and without `setsid(1)`
on `PATH`, that the hook's output pipe closes as soon as the hook returns (the detached copy
holds neither stdout nor stderr, so it cannot hold the CLI), and that `hooks.json` registers
the Stop hook synchronously.

```bash
scripts/test-precompact-gate.sh
```

Runs the real PreCompact hook against fixture transcripts. It proves trigger `auto` passes
and manual or unknown triggers are gated, and that the gate passes on a recent
`store_memory` call or on `ALAYA_NOTHING_TO_SAVE` alone as the last line of Claude's own reply
text, and still blocks when the marker is only mentioned inside a sentence or shown in a fenced
example or followed by more text or a tool call in the same turn, or appears only in a tool
result, a user turn, a tool input, a quoted file, a thinking block or a subagent reply, when the
transcript is missing or holds no parseable line, and when the marker has scrolled out of the
window (with a control showing the same marker passes inside it).

## Troubleshooting

- **No memories are showing up** — check `$ALAYA_HOOK_STATE_DIR/failures.log` (default
  `~/.cache/alaya-hook/failures.log`). Every failure path (missing config or `jq`, unresolved
  or hung secret command, dead LLM endpoint, unexpected LLM output shape, failed store call)
  logs one line there — see LAB-170: a silent failure here once cost 5 days of unsaved memories.
- **PreCompact keeps blocking** — trigger `auto` (proactive and reactive compaction) always
  passes. Idle compaction arrives as `manual`, so the gate applies to it as it does to a
  manual `/compact`. On a manual `/compact` the block message reaches you, not Claude. The gate
  unblocks once an `mcp__alaya__store_memory` tool
  call, or an assistant reply whose final line is `ALAYA_NOTHING_TO_SAVE` alone (Claude's
  deliberate "nothing worth saving"), appears in the last `ALAYA_PRECOMPACT_RECENT_LINES`
  transcript lines. The marker counts only in Claude's own reply text, so a file, tool result
  or user message that contains it cannot open the gate. This is deliberate: the two hooks are
  independent persistence paths, and the PreCompact gate wants Claude itself to decide before
  context is compacted away — a curated save via the Ālaya MCP server, or the marker. The Stop
  hook's automatic REST extraction satisfies neither and is not meant to. Ask Claude to save or
  to state the marker (the block message, shown to you on a manual `/compact`, says how), or disable the gate by removing the
  PreCompact hook if you don't run the Ālaya MCP server.
- **`claude plugin validate` fails** — run it with `--strict` locally before pushing; it flags
  unrecognized fields and missing metadata that the runtime otherwise tolerates silently.
