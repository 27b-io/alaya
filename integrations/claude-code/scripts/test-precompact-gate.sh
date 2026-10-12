#!/usr/bin/env bash
# Black-box test of the PreCompact gate (alaya-precompact.sh). It must pass (exit 0) on a
# recent store_memory call or on ALAYA_NOTHING_TO_SAVE in Claude's own reply text, and block
# (exit 2) everywhere else — above all when the marker only arrives in content Claude read,
# so nothing outside Claude's own words can open the gate.
#
#   scripts/test-precompact-gate.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-precompact.sh"
T=$(mktemp -d) || exit 1; trap 'rm -rf "$T"' EXIT
RC=0
M=ALAYA_NOTHING_TO_SAVE

user()      { jq -cn --argjson c "$1" '{type:"user", message:{role:"user", content:$c}}'; }
assistant() { jq -cn --argjson c "$1" --argjson x "${2:-null}" \
  '{type:"assistant", message:{role:"assistant", content:$c}} + ($x // {})'; }
base() { # an ordinary exchange with no store and no marker
  user '"please fix the bug"'
  assistant '[{"type":"text","text":"Fixed it."}]'
}

# expect <name> <want-exit> <transcript-path>
expect() {
  printf '{"session_id":"s1","transcript_path":"%s"}' "$3" | bash "$HOOK" >/dev/null 2>&1
  local got=$?
  if [[ $got -eq $2 ]]; then echo "ok    $1"; else echo "FAIL  $1: want exit $2, got $got"; RC=1; fi
}
# case <name> <want-exit> — the transcript is read from stdin
case_() { cat > "$T/$1.jsonl"; expect "$1" "$2" "$T/$1.jsonl"; }

base | case_ "no store, no marker: blocks" 2

{ base; assistant "[{\"type\":\"text\",\"text\":\"Nothing qualifies. $M\"}]"; } \
  | case_ "deliberate skip in assistant text: passes" 0

{ base; assistant '[{"type":"tool_use","id":"t1","name":"mcp__alaya__store_memory","input":{"content":"x"}}]'; } \
  | case_ "store_memory tool_use: passes" 0

# forgeries: the marker reaches the transcript through something Claude did not write
{ base; user "[{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":\"$M\"}]"; } \
  | case_ "marker in a tool result: blocks" 2
{ base; user "[{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":[{\"type\":\"text\",\"text\":\"$M\"}]}]"; } \
  | case_ "marker in a tool result text block: blocks" 2
{ base; user "\"$M\""; } | case_ "marker in a user string turn: blocks" 2
{ base; user "[{\"type\":\"text\",\"text\":\"$M\"}]"; } | case_ "marker in a user text block: blocks" 2
{ base; assistant "[{\"type\":\"tool_use\",\"id\":\"t1\",\"name\":\"Write\",\"input\":{\"content\":\"$M\"}}]"; } \
  | case_ "marker in a tool_use input: blocks" 2
{ base; user "[{\"type\":\"tool_result\",\"tool_use_id\":\"t1\",\"content\":\"1\\t# notes\\n2\\t$M\\n\"}]"
  jq -cn --arg m "$M" '{type:"user", toolUseResult:{file:{filePath:"/x.md", content:$m}},
    message:{role:"user", content:[{type:"tool_result", tool_use_id:"t2", content:"ok"}]}}'; } \
  | case_ "marker in a quoted file's contents: blocks" 2
{ base; assistant "[{\"type\":\"thinking\",\"thinking\":\"$M\"}]"; } | case_ "marker in a thinking block: blocks" 2
{ base; assistant "[{\"type\":\"text\",\"text\":\"$M\"}]" '{"isSidechain":true}'; } \
  | case_ "marker in a subagent (sidechain) reply: blocks" 2
{ base; jq -cn --arg m "$M" '{type:"system", content:$m}'; } | case_ "marker in a system entry: blocks" 2

# fail closed
expect "missing transcript: blocks" 2 "$T/does-not-exist.jsonl"
printf 'not json {{{\n' > "$T/garbage.jsonl"; expect "unparseable transcript: blocks" 2 "$T/garbage.jsonl"
if printf 'not json' | bash "$HOOK" >/dev/null 2>&1; then echo "FAIL  unparseable hook input: passed"; RC=1
else echo "ok    unparseable hook input: blocks"; fi

# the window: a marker older than the last N lines no longer counts
{ assistant "[{\"type\":\"text\",\"text\":\"$M\"}]"; base; base; } > "$T/old.jsonl"
printf '{"session_id":"s1","transcript_path":"%s"}' "$T/old.jsonl" \
  | ALAYA_PRECOMPACT_RECENT_LINES=2 bash "$HOOK" >/dev/null 2>&1
if [[ $? -eq 2 ]]; then echo "ok    marker outside the window: blocks"; else echo "FAIL  marker outside the window: passed"; RC=1; fi

# the block message tells Claude about the marker and keeps "storing nothing is normal"
msg=$(printf '{"transcript_path":"%s"}' "$T/no store, no marker: blocks.jsonl" | bash "$HOOK" 2>&1 >/dev/null)
for want in "$M" "Storing nothing is a normal outcome"; do
  if grep -qF -- "$want" <<< "$msg"; then echo "ok    prompt names [$want]"; else echo "FAIL  prompt lacks [$want]"; RC=1; fi
done

[[ $RC -eq 0 ]] && echo "ALL PASS: store or assistant-text marker opens the gate; no forged marker does; failures block"
exit $RC
