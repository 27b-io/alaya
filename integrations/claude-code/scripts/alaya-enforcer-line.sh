#!/usr/bin/env bash
# PreToolUse hook: in a Multica agent run, refuse a memory store, or a refute's
# correction, whose content has no line starting "Enforcer:".
#
# The Multica agent skills require every stored memory to name the check that
# enforces its lesson. A rule that lives only in skill text is forgotten; this
# hook turns it into a refusal at the moment of the store.
#
# Scope: only Multica runs. The run is recognised by MULTICA_TASK_ID or
# MULTICA_AGENT_ID, which the Multica daemon exports into the process of every
# run it starts. A hook inherits that environment whichever prompt started the
# run and whichever agent in it makes the call, so subagent calls and scheduled
# runs are covered, which a match on the transcript's first prompt would miss.
# Every other session passes untouched: the rule is not theirs.
#
# Exit 0 = allow, exit 2 = refuse (stderr is shown to Claude as the reason).
# Fails OPEN: when the hook input cannot be read or parsed, or jq is missing,
# the store goes through and one line lands in failures.log. Failing closed
# would stop every memory store of every Multica run on a hook fault, and a
# memory missing a line costs less than a memory lost.

set -u
[[ -z "${MULTICA_TASK_ID:-}" && -z "${MULTICA_AGENT_ID:-}" ]] && exit 0

STATE_DIR="${ALAYA_HOOK_STATE_DIR:-$HOME/.cache/alaya-hook}"
_pass_logged() {
    mkdir -p "$STATE_DIR" 2>/dev/null || true
    printf '%s enforcer-line: %s, store allowed unchecked\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" \
        >> "$STATE_DIR/failures.log" 2>/dev/null || true
    exit 0
}

command -v jq >/dev/null 2>&1 || _pass_logged "config: jq not found on PATH"

# has-line | no-line | not-a-store. The three tools that write a new memory all
# carry its text in "content": the two stores, and memory_refute, whose
# correction is stored like any other memory. hooks.json matches exactly these,
# so not-a-store is a guard against a widened matcher, not an expected path.
verdict=$(jq -r '
    .tool_name as $t
    | if $t != "mcp__alaya__store_memory" and $t != "mcp__alaya__memory_refute" and $t != "mcp__unified-memory__store"
    then "not-a-store"
    elif (.tool_input.content | type) != "string" then error("tool_input.content is not a string")
    elif .tool_input.content | test("(^|\n)Enforcer:") then "has-line"
    else "no-line" end' 2>/dev/null) || _pass_logged "cannot read the hook input"

case "$verdict" in
    has-line | not-a-store) exit 0 ;;
    no-line) ;;
    *) _pass_logged "cannot read the hook input" ;; # empty stdin: jq prints nothing and exits 0
esac

cat >&2 <<'MSG'
Refused: this memory has no Enforcer line. In a Multica run every stored memory
names the check that enforces it (the progress-update skill's memory rules).
Add a line that starts with "Enforcer:" and names that check: a test, a hook, a
lint rule, a CI gate or a gating ticket. If no check exists yet, file a ticket
for one and write "Enforcer: LAB-N (none yet)". Then store the memory again.
MSG
exit 2
