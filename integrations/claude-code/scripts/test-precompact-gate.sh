#!/usr/bin/env bash
# Black-box test of the PreCompact gate (alaya-precompact.sh). It must pass (exit 0) on
# trigger "auto", on a recent store_memory call or on ALAYA_NOTHING_TO_SAVE in Claude's own
# reply text, and block (exit 2) everywhere else — above all when the marker only arrives in
# content Claude read, so nothing outside Claude's own words can open the gate.
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

# expect <name> <want-exit> <transcript-path> [trigger]
expect() {
  local trigger=""; [[ -n "${4:-}" ]] && trigger="\"trigger\":\"$4\","
  printf '{"session_id":"s1",%s"transcript_path":"%s"}' "$trigger" "$3" | bash "$HOOK" >/dev/null 2>&1
  local got=$?
  # case_ runs in a pipeline subshell, where RC=1 is lost: the marker file reaches the parent
  if [[ $got -eq $2 ]]; then echo "ok    $1"; else echo "FAIL  $1: want exit $2, got $got"; RC=1; : > "$T/failed"; fi
}
# case <name> <want-exit> — the transcript is read from stdin
case_() { cat > "$T/$1.jsonl"; expect "$1" "$2" "$T/$1.jsonl"; }

base | case_ "no store, no marker: blocks" 2

{ base; assistant "[{\"type\":\"text\",\"text\":\"Nothing qualifies.\\n\\n$M\"}]"; } \
  | case_ "deliberate skip on its own line in assistant text: passes" 0

# Claude explaining or quoting the rule is not a decision: the marker must stand alone on a line
{ base; assistant "[{\"type\":\"text\",\"text\":\"The README says the gate accepts $M when nothing qualifies; I have not checked the session yet.\"}]"; } \
  | case_ "marker mentioned in an assistant sentence: blocks" 2
{ base; assistant "[{\"type\":\"text\",\"text\":\"Nothing qualifies. $M\"}]"; } \
  | case_ "marker trailing a sentence: blocks" 2
{ base; assistant "[{\"type\":\"text\",\"text\":\"The flag is \`$M\`.\\n\`$M\`\"}]"; } \
  | case_ "marker in backticks: blocks" 2
# a standalone marker counts only as the last line of its block: a fenced example or one followed by prose blocks
say() { jq -cn --arg t "$1" '{type:"assistant", message:{role:"assistant", content:[{type:"text", text:$t}]}}'; }
{ base; say "Here is the README example; I have not checked whether to store anything:
\`\`\`text
$M
\`\`\`
This is documentation, not my decision."; } | case_ "marker alone inside a fenced example: blocks" 2
{ base; say "Example:
\`\`\`
$M
\`\`\`"; } | case_ "marker alone in a fence that ends the reply: blocks" 2
{ base; say "$M
That is what you would write if nothing qualified."; } | case_ "own-line marker followed by prose: blocks" 2
{ base; say "$(printf 'Reviewed the session: nothing qualifies.\n\n  %s  \r\n \n' "$M")"; } | case_ "marker as the last line with surrounding whitespace: passes" 0

# the marker must end the assistant's turn: Claude Code writes each block of a reply as its own
# line, so a later block of the same turn (text or a tool call) means the marker was not the decision
EX="Here is the standalone no-save marker from the docs, as an example, not my decision:\\n$M"
READ='{"type":"tool_use","id":"t9","name":"Read","input":{"file_path":"/x"}}'
{ base; assistant "[{\"type\":\"text\",\"text\":\"$EX\"},$READ]"; } \
  | case_ "marker block then a tool call in the same entry: blocks" 2
{ base; assistant "[{\"type\":\"text\",\"text\":\"$EX\"}]"; assistant "[$READ]"; } \
  | case_ "marker block then a tool call on the next line: blocks" 2
{ base; assistant "[{\"type\":\"text\",\"text\":\"$EX\"}]"; assistant "[$READ]"
  user '[{"type":"tool_result","tool_use_id":"t9","content":"file body"}]'; } \
  | case_ "marker block, tool call, tool result: blocks" 2
{ base; assistant "[{\"type\":\"text\",\"text\":\"$EX\"}]"; say "That was only an example."; } \
  | case_ "marker block then another text block: blocks" 2
# a tool result is not a turn end: the turn goes on past it
TR='[{"type":"tool_result","tool_use_id":"t9","content":"file body"}]'
{ base; assistant "[$READ,{\"type\":\"text\",\"text\":\"$EX\"}]"; user "$TR"; } \
  | case_ "tool call then marker block in one entry, then its tool result: blocks" 2
{ base; assistant "[$READ]"; assistant "[{\"type\":\"text\",\"text\":\"$EX\"}]"; user "$TR"; } \
  | case_ "tool call, marker block, then its tool result: blocks" 2
{ base; say "Nothing qualifies.
$M"; user '"next task please"'; assistant '[{"type":"text","text":"On it."}]'; } \
  | case_ "marker ending an earlier turn, then more turns: passes" 0
{ base; say "Nothing qualifies.
$M"; assistant '[{"type":"text","text":"x"}]' '{"isSidechain":true}'; } \
  | case_ "marker turn end not cleared by a sidechain entry: passes" 0

# a malformed or unparseable later line must clear the marker too: a parse error never opens the gate
# (labels stay plain: a case name becomes the fixture path inside the hook's JSON input)
malformed() { # <label> <raw line>
  { base; assistant "[{\"type\":\"text\",\"text\":\"$EX\"}]"; printf '%s\n' "$2"; user '"next"'; } \
    | case_ "marker then $1: blocks" 2; }
malformed "assistant entry with null content"     '{"type":"assistant","message":{"content":null}}'
malformed "assistant entry with string content"   '{"type":"assistant","message":{"content":"x"}}'
malformed "assistant entry with no message"       '{"type":"assistant"}'
malformed "assistant text block with a non-string text" '{"type":"assistant","message":{"content":[{"type":"text","text":7}]}}'
malformed "assistant entry with a null block"     '{"type":"assistant","message":{"content":[null]}}'
malformed "an unparseable line"                   'not json {'
malformed "a JSON line that is not an object"     '[1,2]'
malformed "user entry with no message"            '{"type":"user"}'

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

# only trigger "auto" bypasses the gate; manual, missing and unknown triggers are gated
base > "$T/trig.jsonl"
expect "auto trigger, no store, no marker: passes" 0 "$T/trig.jsonl" auto
expect "auto trigger, missing transcript: passes" 0 "$T/does-not-exist.jsonl" auto
expect "manual trigger, no store, no marker: blocks" 2 "$T/trig.jsonl" manual
expect "unknown trigger, no store, no marker: blocks" 2 "$T/trig.jsonl" bogus

# fail closed
expect "missing transcript: blocks" 2 "$T/does-not-exist.jsonl"
printf 'not json {{{\n' > "$T/garbage.jsonl"; expect "unparseable transcript: blocks" 2 "$T/garbage.jsonl"
printf 'not json' | bash "$HOOK" >/dev/null 2>&1; got=$?
if [[ $got -eq 2 ]]; then echo "ok    unparseable hook input: blocks"
else echo "FAIL  unparseable hook input: want exit 2, got $got"; RC=1; fi

# the window: a marker older than the last N lines no longer counts
{ assistant "[{\"type\":\"text\",\"text\":\"$M\"}]"; base; base; } > "$T/old.jsonl"
window() { # <lines> — exit code of the hook over old.jsonl with that window
  printf '{"session_id":"s1","transcript_path":"%s"}' "$T/old.jsonl" \
    | ALAYA_PRECOMPACT_RECENT_LINES="$1" bash "$HOOK" >/dev/null 2>&1; }
window 2; got=$?
if [[ $got -eq 2 ]]; then echo "ok    marker outside the window: blocks"; else echo "FAIL  marker outside the window: got $got"; RC=1; fi
if window 5; then echo "ok    same marker inside the window: passes"; else echo "FAIL  same marker inside the window: blocked"; RC=1; fi

# the block message (shown to the user on a manual /compact) names the marker, keeps "storing nothing is normal", and the
# prompt's own mention of the marker is inline, so Claude quoting it cannot open the gate
base > "$T/msg.jsonl"
msg=$(printf '{"transcript_path":"%s"}' "$T/msg.jsonl" | bash "$HOOK" 2>&1 >/dev/null)
for want in "end your reply with the marker $M alone on its own line" "Storing nothing is a normal outcome"; do
  if grep -qF -- "$want" <<< "$msg"; then echo "ok    prompt names [$want]"; else echo "FAIL  prompt lacks [$want]"; RC=1; fi
done

[[ -e "$T/failed" ]] && RC=1
[[ $RC -eq 0 ]] && echo "ALL PASS: auto compaction, a store or an assistant-text marker opens the gate; no forged marker does; failures block"
exit $RC
