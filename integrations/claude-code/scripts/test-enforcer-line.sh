#!/usr/bin/env bash
# The Enforcer line, on both paths a memory takes. The PreToolUse hook
# (alaya-enforcer-line.sh) must refuse (exit 2) a memory store or refute with no line
# starting "Enforcer:" in a Multica run, pass one that has the line, and pass
# every store outside a Multica run. When it cannot read its input it must pass
# and log, never refuse. The Stop hook's extractor must be asked for the line,
# and a reply that carries it must reach the store with the line intact.
#
#   scripts/test-enforcer-line.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-enforcer-line.sh"
SAVE="$HERE/alaya-session-save.sh"
T=$(mktemp -d) || exit 1; trap 'rm -rf "$T"' EXIT
JQ_DIR=$(dirname "$(command -v jq)")
PY_DIR=$(dirname "$(command -v python3)") # env -i keeps only PATH: python3 may live outside /usr/bin
RC=0
ok()   { echo "ok    $1"; }
fail() { echo "FAIL  $1"; RC=1; }

# --- PreToolUse hook ---
store() { jq -cn --arg t "$1" --arg c "$2" '{hook_event_name:"PreToolUse", tool_name:$t, tool_input:{content:$c, tags:["t"]}}'; }

# expect <name> <want-exit> <want failures.log lines> <stdin> [VAR=value...]
# The refusal must name the rule and the escape; a pass must print nothing.
expect() {
  local home="$T/home" got err logs=0
  rm -rf "$home"; mkdir -p "$home"
  err=$(printf '%s' "$4" | env -i HOME="$home" PATH="${GATE_PATH:-$JQ_DIR:/usr/bin:/bin}" "${@:5}" bash "$HOOK" 2>&1 >/dev/null)
  got=$?
  [[ -f "$home/.cache/alaya-hook/failures.log" ]] && logs=$(( $(wc -l < "$home/.cache/alaya-hook/failures.log") ))
  if [[ $got -ne $2 ]]; then fail "$1: want exit $2, got $got ($err)"
  elif [[ $logs -ne $3 ]]; then fail "$1: want $3 failures.log lines, got $logs"
  elif [[ $2 -eq 2 ]] && ! { grep -qF "progress-update" <<< "$err" && grep -qF "Enforcer: LAB-N (none yet)" <<< "$err"; }; then
    fail "$1: the refusal does not name the rule and the escape: [$err]"
  elif [[ $2 -eq 0 && -n "$err" ]]; then fail "$1: a pass printed [$err]"
  else ok "$1"; fi
}

ALAYA=mcp__alaya__store_memory
UNIFIED=mcp__unified-memory__store
REFUTE=mcp__alaya__memory_refute # its correction is stored like any other memory
WITH=$'A lesson, with its evidence.\nEnforcer: scripts/test-enforcer-line.sh'
WITHOUT='A lesson, with its evidence.'

for tool in "$ALAYA" "$UNIFIED" "$REFUTE"; do
  expect "Multica run, $tool, no line: refused" 2 0 "$(store "$tool" "$WITHOUT")" MULTICA_TASK_ID=t1 MULTICA_AGENT_ID=a1
  expect "Multica run, $tool, with the line: passes" 0 0 "$(store "$tool" "$WITH")" MULTICA_TASK_ID=t1 MULTICA_AGENT_ID=a1
  expect "outside a Multica run, $tool, no line: passes" 0 0 "$(store "$tool" "$WITHOUT")"
done
expect "only MULTICA_TASK_ID set, no line: refused" 2 0 "$(store "$ALAYA" "$WITHOUT")" MULTICA_TASK_ID=t1
expect "only MULTICA_AGENT_ID set, no line: refused" 2 0 "$(store "$ALAYA" "$WITHOUT")" MULTICA_AGENT_ID=a1
expect "empty MULTICA variables are not a Multica run: passes" 0 0 "$(store "$ALAYA" "$WITHOUT")" MULTICA_TASK_ID= MULTICA_AGENT_ID=
M=(MULTICA_TASK_ID=t1)
expect "the line is the whole content: passes" 0 0 "$(store "$ALAYA" "Enforcer: LAB-1 (none yet)")" "${M[@]}"
expect "the line after a CRLF: passes" 0 0 "$(store "$ALAYA" $'A lesson.\r\nEnforcer: a test')" "${M[@]}"
expect "the line in the middle: passes" 0 0 "$(store "$ALAYA" $'A lesson.\nEnforcer: a test\nMore.')" "${M[@]}"
# the word inside a sentence, indented, in another case or without its colon is not the line
expect "Enforcer: inside a sentence: refused" 2 0 "$(store "$ALAYA" "The rule says Enforcer: goes last.")" "${M[@]}"
expect "indented line: refused" 2 0 "$(store "$ALAYA" $'A lesson.\n  Enforcer: a test')" "${M[@]}"
expect "lower-case line: refused" 2 0 "$(store "$ALAYA" $'A lesson.\nenforcer: a test')" "${M[@]}"
expect "no colon: refused" 2 0 "$(store "$ALAYA" $'A lesson.\nEnforcer a test')" "${M[@]}"
# the line in another field does not count: content is what is stored and searched
expect "the line only in summary: refused" 2 0 \
  "$(jq -cn --arg t "$ALAYA" '{tool_name:$t, tool_input:{content:"A lesson.", summary:"Enforcer: a test"}}')" "${M[@]}"
# fail open and say so: a hook fault must not stop every store of every run
expect "input is not JSON: passes, logged" 0 1 "not json" "${M[@]}"
expect "empty input: passes, logged" 0 1 "" "${M[@]}"
expect "content is not a string: passes, logged" 0 1 \
  "$(jq -cn --arg t "$ALAYA" '{tool_name:$t, tool_input:{content:["A lesson."]}}')" "${M[@]}"
expect "no content field: passes, logged" 0 1 "$(jq -cn --arg t "$ALAYA" '{tool_name:$t, tool_input:{}}')" "${M[@]}"
mkdir -p "$T/nojq"; for b in bash cat date mkdir; do ln -s "$(command -v $b)" "$T/nojq/$b"; done
GATE_PATH="$T/nojq" expect "jq missing: passes, logged" 0 1 "$(store "$ALAYA" "$WITHOUT")" "${M[@]}"
GATE_PATH="$T/nojq" expect "jq missing outside a Multica run: passes, nothing logged" 0 0 "$(store "$ALAYA" "$WITHOUT")"
# any other tool is not this hook's business, whatever its input holds
expect "another tool, no line: passes" 0 0 "$(store mcp__alaya__search "$WITHOUT")" "${M[@]}"
expect "a tool whose name only starts like a store's: passes" 0 0 "$(store mcp__alaya__store "$WITHOUT")" "${M[@]}"

# hooks.json sends the three tools that write a memory, and nothing else, to the hook
if jq -e '[.hooks.PreToolUse[] | select(.hooks | any(.command | contains("alaya-enforcer-line.sh"))) | .matcher] | sort
    == ["mcp__alaya__memory_refute", "mcp__alaya__store_memory", "mcp__unified-memory__store"]' "$HERE/../hooks/hooks.json" >/dev/null
then ok "hooks.json: PreToolUse sends the three memory-writing tools to the hook"
else fail "hooks.json: PreToolUse does not match exactly the three memory-writing tools"; fi

# --- Stop hook extractor ---
# stub curl: keep the extraction request and answer it with the reply in
# $CAP/reply; keep each store POST's body (the hook reads the code via -w '%{http_code}')
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'STUB'
#!/bin/sh
prev=; body=
for a; do [ "$prev" = "-d" ] && body=$a; prev=$a; done
case "$*" in
  *llm.test*) cat "${body#@}" > "$CAP/llm.json"; jq -Rsc '{choices:[{message:{content:.}}]}' "$CAP/reply" ;;
  *) printf '%s\n' "$body" >> "$CAP/stores.jsonl"; printf 200 ;;
esac
STUB
chmod +x "$T/bin/curl"

# a transcript that clears every save gate: old enough, 5 prompt lines
for i in 1 2 3 4 5; do
  jq -nc --arg t "prompt $i" '{type:"user", timestamp:"2020-01-01T00:00:00Z", message:{role:"user", content:$t}}'
done > "$T/transcript.jsonl"

CAP="$T/cap"; mkdir -p "$T/shome" "$CAP"
NAMED=$'Lesson one, with its evidence.\nEnforcer: scripts/test-enforcer-line.sh'
NONE=$'Lesson two, with its evidence.\nEnforcer: none named'
jq -nc --arg a "$NAMED" --arg b "$NONE" \
  '[{content:$a, memory_type:"reference", tags:["t"], summary:"s"}, {content:$b, memory_type:"decision", tags:["t"], summary:"s"}]' > "$CAP/reply"
printf '{"session_id":"s1","transcript_path":"%s"}' "$T/transcript.jsonl" \
  | (cd "$T" && env -i HOME="$T/shome" CAP="$CAP" PATH="$T/bin:$JQ_DIR:$PY_DIR:/usr/bin:/bin" \
      ALAYA_URL=http://alaya.test/store ALAYA_LLM_URL=http://llm.test/v1/chat/completions \
      ALAYA_LLM_API_KEY=k ALAYA_API_KEY=k ALAYA_HOOK_STATE_DIR="$T/shome/state" _ALAYA_HOOK_FOREGROUND=1 \
      bash "$SAVE") >/dev/null 2>&1

if jq -e '.messages[0].content | (contains("starting \"Enforcer:\"") and contains("\"Enforcer: none named\""))' "$CAP/llm.json" >/dev/null 2>&1
then ok "extractor prompt asks for the Enforcer line and its none-named form"
else fail "extractor prompt does not ask for the Enforcer line"; fi

if jq -se --arg a "$NAMED" --arg b "$NONE" 'map(.content) == [$a, $b]' "$CAP/stores.jsonl" >/dev/null 2>&1
then ok "extractor reply: both memories stored with the Enforcer line intact"
else fail "extractor reply: stored content lost the Enforcer line: $(cat "$CAP/stores.jsonl" 2>/dev/null) (log: $(cat "$T/shome/state/failures.log" 2>/dev/null))"; fi

[[ $RC -eq 0 ]] && echo "PASS" || echo "FAILED"
exit $RC
