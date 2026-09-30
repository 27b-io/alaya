#!/usr/bin/env bash
# Black-box test of what alaya-session-save.sh hands the extractor. A prompt is
# a string in terminal sessions but a list of text blocks in SDK/agent sessions;
# reading strings only made every SDK-driven run save nothing. Runs the real
# hook under `env -i` with a scratch HOME/state dir, direct secret values and a
# stub curl first on PATH, so nothing reaches a real LLM, Ālaya, or secret store.
#
#   scripts/test-prompt-extraction.sh                   # fixture cases
#   scripts/test-prompt-extraction.sh <transcript>      # plus replay a real transcript
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-session-save.sh"
T=$(mktemp -d) || exit 1; trap 'rm -rf "$T"' EXIT
JQ_DIR=$(dirname "$(command -v jq)")
RC=0

# stub curl: keep the extraction request and answer it with one memory; answer
# the store POST with a 200 (the hook reads the code via -w '%{http_code}')
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'STUB'
#!/bin/sh
prev=; body=
for a; do [ "$prev" = "-d" ] && body=$a; prev=$a; done
case "$*" in
  *llm.test*) cat "${body#@}" > "$CAP/llm.json"
    echo '{"choices":[{"message":{"content":"[{\"content\":\"stub memory\"}]"}}]}' ;;
  *) printf 200 ;;
esac
STUB
chmod +x "$T/bin/curl"

# run_hook <transcript> [VAR=value...] — fresh HOME per run. Leaves the extraction
# request in cap/llm.json and the message-line count the hook checkpointed in cap/total.
run_hook() {
  local home="$T/home" cap="$T/cap" tr="$1"
  [[ "$tr" == /* ]] || tr="$PWD/$tr" # the hook runs in $T, not the caller's dir
  rm -rf "$home" "$cap"; mkdir -p "$home" "$cap"
  printf '{"session_id":"s1","transcript_path":"%s"}' "$tr" \
    | (cd "$T" && env -i HOME="$home" CAP="$cap" PATH="$T/bin:$JQ_DIR:/usr/bin:/bin" \
        ALAYA_URL=http://alaya.test/store ALAYA_LLM_URL=http://llm.test/v1/chat/completions \
        ALAYA_LLM_API_KEY=k ALAYA_API_KEY=k ALAYA_HOOK_STATE_DIR="$home/state" "${@:2}" \
        bash "$HOOK") >/dev/null 2>&1
  sed -n 2p "$home/state/s1" > "$cap/total" 2>/dev/null
}

ctx() { jq -r '.messages[1].content' "$T/cap/llm.json" 2>/dev/null; }
has()   { if ctx | grep -qF -- "$2"; then echo "ok    $1"; else echo "FAIL  $1: missing [$2]"; RC=1; fi; }
lacks() { # no extraction call at all proves nothing about what it would have leaked
  if [[ ! -s "$T/cap/llm.json" ]]; then echo "FAIL  $1: no extraction call"; RC=1
  elif ctx | grep -qF -- "$2"; then echo "FAIL  $1: leaked [$2]"; RC=1; else echo "ok    $1"; fi; }
total() {
  got=$(cat "$T/cap/total" 2>/dev/null)
  if [[ "$got" == "$2" ]]; then echo "ok    $1"; else echo "FAIL  $1: want $2 lines, got [${got:-no save}]"; RC=1; fi
}

# transcript entries — the timestamp is old enough to clear the 120s duration gate
entry() { # <content-json> [extra-fields-json]
  jq -c '{type:"user", timestamp:"2020-01-01T00:00:00Z", message:{role:"user", content:.}} + ($x // {})' \
    --argjson x "${2:-null}" <<< "$1"; }
REMINDER="<system-reminder>\nREMINDER-BODY-TEXT\n</system-reminder>"
noise() {
  entry '[{"type":"tool_result","tool_use_id":"t1","content":"TOOL-RESULT-TEXT"}]'
  entry '[{"type":"text","text":"Base directory for this skill: /s\n\nSKILL-BODY-TEXT"}]' '{"isMeta":true}'
  entry '[{"type":"text","text":"[Request interrupted by user]"}]'
  entry '[{"type":"text","text":"[Request interrupted by user for tool use]"}]'
  # a subagent's task prompt: the parent agent wrote it, not the user
  entry '[{"type":"text","text":"SIDECHAIN-TASK-TEXT"}]' '{"isSidechain":true}'
  # the tool input matches the hook's '"type":"user"' grep; only its .type check keeps this out of the count
  jq -cn '{type:"assistant", message:{role:"assistant", content:[
    {type:"tool_use", id:"t2", name:"x", input:{type:"user"}}, {type:"text", text:"assistant closing words"}]}}'
}

# terminal session: prompts are strings
{
  for n in one two three four; do entry "\"string prompt $n\""; done
  entry '"[Request interrupted? no, string prompt five]"' # typed; only looks like a marker
  entry "\"$REMINDER\""
  noise
} > "$T/string.jsonl"
run_hook "$T/string.jsonl"
for n in one two three four five; do has "string: prompt $n extracted" "string prompt $n"; done
lacks "string: <system-reminder> body dropped" "REMINDER-BODY-TEXT"
lacks "string: tool result dropped"           "TOOL-RESULT-TEXT"
lacks "string: isMeta skill body dropped"     "SKILL-BODY-TEXT"
lacks "string: interrupt marker dropped"      "[Request interrupted by user"
lacks "string: sidechain task prompt dropped" "SIDECHAIN-TASK-TEXT"
has   "string: last assistant text kept"      "assistant closing words"
total "string: 5 message lines counted" 5

# SDK/agent session: prompts are block lists, a reminder can ride along as its own block
{
  for n in one two three four; do entry "[{\"type\":\"text\",\"text\":\"block prompt $n\"}]"; done
  entry "[{\"type\":\"text\",\"text\":\"block prompt five\"},{\"type\":\"text\",\"text\":\"$REMINDER\"}]"
  noise
} > "$T/blocks.jsonl"
run_hook "$T/blocks.jsonl"
for n in one two three four five; do has "blocks: prompt $n extracted" "block prompt $n"; done
lacks "blocks: <system-reminder> block dropped" "REMINDER-BODY-TEXT"
lacks "blocks: tool result dropped"             "TOOL-RESULT-TEXT"
lacks "blocks: isMeta skill body dropped"       "SKILL-BODY-TEXT"
lacks "blocks: interrupt marker dropped"        "[Request interrupted by user"
lacks "blocks: sidechain task prompt dropped"   "SIDECHAIN-TASK-TEXT"
total "blocks: 5 message lines counted" 5

# optional replay of a real transcript: a smoke test that it reaches the extractor with a
# non-zero count. It cannot know which lines are real prompts; the fixtures above prove that.
# Save gates off, so a short or still-running session measures extraction, not eligibility.
if [[ -n "${1:-}" ]]; then
  run_hook "$1" ALAYA_MIN_DURATION_SECS=0 ALAYA_MIN_NEW_MESSAGES=1
  got=$(cat "$T/cap/total" 2>/dev/null)
  if [[ "${got:-0}" -gt 0 ]]; then echo "smoke replay: $got message lines reached the extractor from $(basename "$1") (count only)"
  else echo "FAIL  replay: nothing extracted from $1"; RC=1; fi
fi

[[ $RC -eq 0 ]] && echo "ALL PASS: string and block-list prompts extracted; tool results, isMeta and sidechain entries, interrupt markers and <wrapper> bodies excluded"
exit $RC
