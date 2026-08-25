#!/usr/bin/env bash
# PreCompact hook: block compaction unless Claude has saved memories recently.
#
# Recency rule: at least one mcp__alaya__store_memory tool_use call within
# the last N lines (default 500) of the transcript. Roughly: "did Claude save
# at any point in the recent few turns?"
#
# Exit 0 = allow, exit 2 = block (stderr injected into Claude's context).
# Fails CLOSED (blocks) when the transcript can't be found or parsed, same as
# a genuine "no recent save" — an unreadable transcript never bypasses the gate.

set -u
RECENT_LINES=${ALAYA_PRECOMPACT_RECENT_LINES:-500}

input=$(cat)

# Extract transcript_path from the hook's stdin JSON. Use python3 — universally available.
transcript=$(printf '%s' "$input" | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get("transcript_path", ""))
except Exception:
    pass
' 2>/dev/null)

if [ -n "$transcript" ] && [ -r "$transcript" ]; then
    # Scan the last $RECENT_LINES of the transcript for any store_memory tool_use.
    # Match on the literal "name":"mcp__alaya__store_memory" substring inside a JSONL line.
    # This is fast (tail + grep) and avoids false positives from text mentions, which
    # would show up as "input":{"command":"... mcp__alaya__store_memory ..."} not as
    # the tool_use name field.
    if tail -n "$RECENT_LINES" "$transcript" 2>/dev/null \
        | grep -q '"name":"mcp__alaya__store_memory"'; then
        exit 0
    fi
fi

# No recent save (or couldn't determine) — block and prompt Claude to save.
cat >&2 <<'PROMPT'
MEMORY SAVE REQUIRED before compaction. Call mcp__alaya__store_memory (up to 3 calls) to persist valuable knowledge from this session. Save ONLY:

1. DECISIONS — architectural/design choices and their rationale
2. DISCOVERIES — non-obvious findings, gotchas, undocumented behaviors
3. BLOCKERS — unresolved issues or follow-up work needed

For each: set memory_type (decision/note/reference/task), add tags [project-name, topic], write a one-line summary, set dedup_threshold: 0.85. Skip routine/trivial work. If nothing worth saving, proceed without saving.
PROMPT

exit 2
