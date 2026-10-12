#!/usr/bin/env bash
# PreCompact hook: block compaction unless Claude has saved memories recently,
# or has said on purpose that nothing qualifies.
#
# Recency rule: within the last N lines (default 500) of the transcript, either
# at least one mcp__alaya__store_memory tool_use call, or an assistant text block
# containing ALAYA_NOTHING_TO_SAVE. Without the second path a session with nothing
# worth saving could never compact, or would store filler just to pass.
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
    # Match the "name" key with optional JSON whitespace around the colon — the
    # transcript is compact JSON today, but legal serializations may add spaces.
    # Forgery from message *content* stays impossible either way: the pattern
    # requires raw double quotes, which JSON string-encoding always escapes to \"
    # inside content, so only a genuine tool_use name field can match.
    if tail -n "$RECENT_LINES" "$transcript" 2>/dev/null \
        | grep -Eq '"name"[[:space:]]*:[[:space:]]*"mcp__alaya__store_memory"'; then
        exit 0
    fi
    # Deliberate no-save: the marker counts only in a text block of a main-thread
    # assistant entry. User turns, tool results, tool_use inputs and file contents
    # never count, so text Claude reads or quotes cannot open the gate; subagent
    # (sidechain) replies are excluded because a subagent echoes what it was fed.
    if tail -n "$RECENT_LINES" "$transcript" 2>/dev/null | python3 -c '
import json, sys
for line in sys.stdin:
    try:
        e = json.loads(line)
        if e.get("type") != "assistant" or e.get("isSidechain"):
            continue
        for b in e["message"]["content"]:
            if b.get("type") == "text" and "ALAYA_NOTHING_TO_SAVE" in b.get("text", ""):
                sys.exit(0)
    except (ValueError, KeyError, TypeError, AttributeError):
        continue
sys.exit(1)
' 2>/dev/null; then
        exit 0
    fi
fi

# No recent save (or couldn't determine) — block and prompt Claude to save.
cat >&2 <<'PROMPT'
MEMORY CHECK before compaction. Review this session for reusable knowledge. If any qualifies, call mcp__alaya__store_memory (up to 3 calls) to persist it. Save ONLY:

1. DECISIONS — architectural/design choices and their rationale (memory_type: decision)
2. LESSONS — non-obvious findings, gotchas, undocumented behaviours, plus the fix (memory_type: reference)

Rules:
- One lesson or decision per memory, citing its evidence (a file:line, command or link).
- Never store run state: ticket status, PR state, progress, blockers, follow-up work or handoff narration. The issue tracker holds those.
- Before each store, search Alaya for the lesson's key terms. If a live memory already says it, store nothing.
- Add tags [project-name, topic], a one-line summary, and dedup_threshold: 0.85.
- Storing nothing is a normal outcome. If nothing qualifies, write ALAYA_NOTHING_TO_SAVE in your reply (plain text, not inside a tool call) and proceed; that marker lets the next compaction through.
PROMPT

exit 2
