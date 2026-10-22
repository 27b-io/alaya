#!/usr/bin/env bash
# PreCompact hook: block a manual /compact unless Claude has saved memories
# recently, or has said on purpose that nothing qualifies.
#
# Recency rule: within the last N lines (default 500) of the transcript, either
# at least one mcp__alaya__store_memory tool_use call, or an assistant text block
# containing ALAYA_NOTHING_TO_SAVE. Without the second path a session with nothing
# worth saving could never compact, or would store filler just to pass.
#
# Exit 0 = allow, exit 2 = block (stderr shown for a manual /compact).
# Fails CLOSED (blocks) when the transcript can't be found or parsed, same as
# a genuine "no recent save" — an unreadable transcript never bypasses the gate.
#
# Always allows trigger "auto". Claude Code never shows Claude the block message
# of an automatic compaction: it skips the compaction and the conversation keeps
# growing, so Claude cannot save or answer the gate and the session runs into
# the context limit ("Prompt is too long"). Blocking there loses the session
# instead of protecting it.

set -u
RECENT_LINES=${ALAYA_PRECOMPACT_RECENT_LINES:-500}

input=$(cat)

# Read a top-level field from the hook's stdin JSON. Use python3 — universally available.
field() {
    printf '%s' "$input" | python3 -c '
import json, sys
try:
    print(json.load(sys.stdin).get(sys.argv[1], ""))
except Exception:
    pass
' "$1" 2>/dev/null
}

[ "$(field trigger)" = "auto" ] && exit 0

transcript=$(field transcript_path)

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
    # Deliberate no-save: the marker counts only as the last non-empty line of the
    # last block of a main-thread assistant turn — the turn ends there, at the next
    # user entry that is not a tool result, or at the end of the tail; a tool result
    # means the turn went on, so it clears the marker. User turns, tool results, tool_use
    # inputs and file contents never count, so text Claude reads cannot open the
    # gate; subagent (sidechain) replies are excluded because a subagent echoes
    # what it was fed. Claude Code writes each block of a reply as its own line,
    # so any later assistant block (more text, a fence's closing line, a tool
    # call) clears the marker: mentioning, quoting or showing it is not a decision.
    # A malformed or unparseable later line clears it too, so parse errors block.
    if tail -n "$RECENT_LINES" "$transcript" 2>/dev/null | python3 -c '
import json, sys
ended_with_marker = False
for line in sys.stdin:
    try:
        e = json.loads(line)
        if not isinstance(e, dict):
            raise TypeError
    except (ValueError, TypeError):
        ended_with_marker = False  # unknown line: it may have been a later block
        continue
    if e.get("isSidechain"):
        continue
    if e.get("type") == "user":
        try:
            content = e["message"]["content"]
            tool_result = not isinstance(content, str) and any(b["type"] == "tool_result" for b in content)
        except (KeyError, TypeError, AttributeError):
            tool_result = True  # unreadable: never confirms a turn end
        if ended_with_marker and not tool_result:
            sys.exit(0)
        ended_with_marker = False  # a tool result means the turn went on past the marker
        continue
    if e.get("type") != "assistant":
        continue
    ended_with_marker = False  # any later assistant entry clears it, even a malformed one
    try:
        for b in e["message"]["content"]:
            last_line = b["text"].strip().rsplit("\n", 1)[-1].strip() if b.get("type") == "text" else ""
            ended_with_marker = last_line == "ALAYA_NOTHING_TO_SAVE"
    except (KeyError, TypeError, AttributeError):
        ended_with_marker = False
sys.exit(0 if ended_with_marker else 1)
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
- Storing nothing is a normal outcome. If nothing qualifies, end your reply with the marker ALAYA_NOTHING_TO_SAVE alone on its own line (plain reply text: no quotes, backticks or tool call) and proceed; that line lets the next compaction through.
PROMPT

exit 2
