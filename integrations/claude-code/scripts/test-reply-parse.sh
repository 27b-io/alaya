#!/usr/bin/env bash
# What the Stop hook makes of the extractor's reply. The extractor often wraps
# the array in a fence and explains it in prose: a fenced [] followed by
# "Nothing here is reusable" was logged as invalid JSON. The hook stores
# memories only when the reply, fence lines aside, is one JSON array and
# nothing else: prose can hold anything, an invented example included, and a
# false memory is worse than a skipped save, which is logged. Runs the real
# hook in the foreground under `env -i` with a stub curl that answers the
# extraction call with each reply below, and checks what was stored and what
# failures.log says.
#
#   scripts/test-reply-parse.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-session-save.sh"
T=$(mktemp -d) || exit 1; trap 'rm -rf "$T"' EXIT
JQ_DIR=$(dirname "$(command -v jq)")
PY_DIR=$(dirname "$(command -v python3)") # env -i keeps only PATH: python3 may live outside /usr/bin
RC=0

# stub curl: answer the extraction call with the reply in $CAP/reply; count
# store POSTs and answer them with a 200 (the hook reads the code via -w '%{http_code}')
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'STUB'
#!/bin/sh
case "$*" in
  *llm.test*) jq -Rsc '{choices:[{message:{content:.}}]}' "$CAP/reply" ;;
  *) echo x >> "$CAP/stores"; printf 200 ;;
esac
STUB
chmod +x "$T/bin/curl"

# a transcript that clears every save gate: old enough, 5 prompt lines
for i in 1 2 3 4 5; do
  jq -nc --arg t "prompt $i" '{type:"user", timestamp:"2020-01-01T00:00:00Z", message:{role:"user", content:$t}}'
done > "$T/transcript.jsonl"

# case_ <label> <stores wanted> <failures.log line wanted, or empty for none> <reply> [PATH prefix]
# (an argument, not stdin: a piped case_ runs in a subshell and loses RC)
case_() {
  local home="$T/home" cap="$T/cap" stores log
  rm -rf "$home" "$cap"; mkdir -p "$home" "$cap"
  printf '%s' "$4" > "$cap/reply"
  printf '{"session_id":"s1","transcript_path":"%s"}' "$T/transcript.jsonl" \
    | (cd "$T" && env -i HOME="$home" CAP="$cap" PATH="${5:-}$T/bin:$JQ_DIR:$PY_DIR:/usr/bin:/bin" \
        ALAYA_URL=http://alaya.test/store ALAYA_LLM_URL=http://llm.test/v1/chat/completions \
        ALAYA_LLM_API_KEY=k ALAYA_API_KEY=k ALAYA_HOOK_STATE_DIR="$home/state" _ALAYA_HOOK_FOREGROUND=1 \
        bash "$HOOK") >/dev/null 2>&1
  stores=0; [[ -f "$cap/stores" ]] && stores=$(( $(wc -l < "$cap/stores") ))
  log=$(cat "$home/state/failures.log" 2>/dev/null)
  if [[ $stores -ne $2 ]]; then echo "FAIL  $1: want $2 stores, got $stores (log: $log)"; RC=1
  elif [[ -z "$3" && -n "$log" ]]; then echo "FAIL  $1: want no log line, got [$log]"; RC=1
  elif [[ -n "$3" ]] && ! { [[ $(wc -l <<< "$log") -eq 1 ]] && grep -qF -- "$3" <<< "$log"; }; then
    echo "FAIL  $1: want one log line with [$3], got [$log]"; RC=1
  else echo "ok    $1"; fi
}

MEM='{"content":"a lesson","memory_type":"reference","tags":["t"],"summary":"s"}'
FENCE='```'

# nothing to save: every value is [] whatever surrounds it, a quiet exit
case_ "fenced [] then prose: nothing to save, nothing logged" 0 "" \
  "$(printf '%sjson\n[]\n%s\n\nThis session is process narration. Nothing here is reusable.\n' "$FENCE" "$FENCE")"
case_ "bare [] then prose: nothing to save, nothing logged" 0 "" \
  "$(printf '[]\nNo memories qualify.\n')"
case_ "two [] arrays: nothing to save, nothing logged" 0 "" \
  "$(printf '%sjson\n[]\n%s\nStill nothing:\n[]\n' "$FENCE" "$FENCE")"
case_ "inline [] in prose: nothing to save, nothing logged" 0 "" \
  "$(printf 'Actual extraction result: []\n')"
# stored: the reply is one array of objects, fenced or not
case_ "bare array: stored" 1 "" \
  "$(printf '[%s]' "$MEM")"
case_ "fenced array of two: both stored" 2 "" \
  "$(printf '%sjson\n[\n  %s,\n  %s\n]\n%s\n' "$FENCE" "$MEM" "$MEM" "$FENCE")"
case_ "indented fenced array: stored" 1 "" \
  "$(printf '  %sjson\n  [%s]\n  %s\n' "$FENCE" "$MEM" "$FENCE")"
case_ "CRLF fenced array: stored" 1 "" \
  "$(printf '%sjson\r\n[%s]\r\n%s\r\n' "$FENCE" "$MEM" "$FENCE")"
# a non-empty result with prose or another value beside it is logged, never stored:
# the prose may say it is an example, and storing an example is a false memory
case_ "fenced array then prose: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf '%sjson\n[%s]\n%s\n\nIt is new.\n' "$FENCE" "$MEM" "$FENCE")"
case_ "prose then bare array: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf 'Here are the memories:\n[%s]\n' "$MEM")"
case_ "markdown link then fenced array: logged" 0 "LLM output not stored:" \
  "$(printf 'Filed:\n[LAB-123](https://example.test/issue) covers the bug.\n%sjson\n[%s]\n%s\n' "$FENCE" "$MEM" "$FENCE")"
case_ "fenced command then fenced array: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf 'Verified with:\n%ssh\nnpm test\n%s\n%sjson\n[%s]\n%s\n' "$FENCE" "$FENCE" "$FENCE" "$MEM" "$FENCE")"
case_ "two non-empty arrays: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf '[%s]\n[%s]\n' "$MEM" "$MEM")"
case_ "illustrative array then the actual fenced []: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf 'For illustration only, not a fact from this session:\n%sjson\n[%s]\n%s\nActual extraction result; nothing qualifies:\n%sjson\n[]\n%s\n' \
    "$FENCE" "$MEM" "$FENCE" "$FENCE" "$FENCE")"
case_ "the actual fenced [] then an illustrative array: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf '%sjson\n[]\n%s\nFor illustration only, not a fact from this session:\n%sjson\n[%s]\n%s\n' \
    "$FENCE" "$FENCE" "$FENCE" "$MEM" "$FENCE")"
case_ "illustrative array then an inline actual []: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf 'For illustration only:\n%sjson\n[%s]\n%s\nActual extraction result: []\n' "$FENCE" "$MEM" "$FENCE")"
case_ "inline actual [] then an illustrative array: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf 'Actual extraction result: []\nFor illustration only:\n%sjson\n[%s]\n%s\n' "$FENCE" "$MEM" "$FENCE")"
case_ "illustrative array beside an actual {\"memories\":[]}: logged" 0 "LLM output not stored: a non-empty result with prose or another value beside it" \
  "$(printf 'For illustration only:\n[%s]\nActual:\n{"memories":[]}\n' "$MEM")"
# not an array: an object alone, pretty-printed or not; nothing nested inside counts
case_ "object envelope: logged as not an array" 0 "valid JSON but not an array" \
  "$(printf '{"memories":[%s]}\n' "$MEM")"
case_ "pretty envelope with a nested []: logged as not an array" 0 "valid JSON but not an array" \
  "$(printf '{\n  "memories": [%s],\n  "tags":\n  []\n}\n' "$MEM")"
case_ "pretty envelope with the array on its own line: logged as not an array" 0 "valid JSON but not an array" \
  "$(printf '{\n  "memories":\n  [\n    %s\n  ]\n}\n' "$MEM")"
# a reply with no JSON value is logged once, on one line, with the reply itself
case_ "prose only: logged with the reply" 0 "LLM output not valid JSON: I could not find anything. Sorry." \
  "$(printf 'I could not find anything.\nSorry.\n')"
# malformed JSON is never a quiet empty result, even when a [] inside it decodes
M='LLM output not stored: malformed JSON in the reply'
case_ "truncated array with a nested []: logged" 0 "$M" \
  "$(printf '[{"content":"a lesson","memory_type":"reference","tags":[]')"
case_ "truncated string holding []: logged" 0 "$M" \
  "$(printf '[{"content":"see [] and more')"
case_ "single-quoted array with a nested []: logged" 0 "$M" \
  "$(printf "[{'content': 'x', 'tags': []}]")"
case_ "markdown link beside a fenced []: logged" 0 "$M" \
  "$(printf '%sjson\n[]\n%s\nSee [LAB-123](https://example.test/issue).\n' "$FENCE" "$FENCE")"
# a truncated array may hold whole objects, so its message varies: logged with the reply, not stored
case_ "truncated array: logged with the reply" 0 ': [{"content":"a lesson","memory_type"' \
  "$(printf '[%s, {"content":"cut sho' "$MEM")"

# the reply file gone before the parser reads it is logged as that, not as bad model output
mkdir -p "$T/rmreply"
# shellcheck disable=SC2016 # $3 and $@ belong to the wrapper, not this shell
printf '#!/bin/sh\nrm -f "$3"\nexec %s "$@"\n' "$(command -v python3)" > "$T/rmreply/python3"
chmod +x "$T/rmreply/python3"
case_ "reply file gone before the parse: logged as a read failure" 0 "No such file or directory" \
  "$(printf '[%s]' "$MEM")" "$T/rmreply:"

[[ $RC -eq 0 ]] && echo "ALL PASS: the Stop hook stores a reply only when it is one JSON array, and logs the rest"
exit $RC
