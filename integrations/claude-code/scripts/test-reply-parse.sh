#!/usr/bin/env bash
# What the Stop hook makes of the extractor's reply. The extractor often wraps
# the array in a fence and explains it in prose, before or after: a fenced []
# followed by "Nothing here is reusable" was logged as invalid JSON, and an
# array with prose around it was dropped. Runs
# the real hook in the foreground under `env -i` with a stub curl that answers
# the extraction call with each reply below, and checks what was stored and
# what failures.log says.
#
#   scripts/test-reply-parse.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-session-save.sh"
T=$(mktemp -d) || exit 1; trap 'rm -rf "$T"' EXIT
JQ_DIR=$(dirname "$(command -v jq)")
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

# case_ <label> <stores wanted> <failures.log line wanted, or empty for none> <reply>
# (an argument, not stdin: a piped case_ runs in a subshell and loses RC)
case_() {
  local home="$T/home" cap="$T/cap" stores log
  rm -rf "$home" "$cap"; mkdir -p "$home" "$cap"
  printf '%s' "$4" > "$cap/reply"
  printf '{"session_id":"s1","transcript_path":"%s"}' "$T/transcript.jsonl" \
    | (cd "$T" && env -i HOME="$home" CAP="$cap" PATH="$T/bin:$JQ_DIR:/usr/bin:/bin" \
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

case_ "fenced [] then prose: nothing to save, nothing logged" 0 "" \
  "$(printf '%sjson\n[]\n%s\n\nThis session is process narration. Nothing here is reusable.\n' "$FENCE" "$FENCE")"
case_ "bare [] then prose: nothing to save, nothing logged" 0 "" \
  "$(printf '[]\nNo memories qualify.\n')"
case_ "fenced array then prose: both stored" 2 "" \
  "$(printf '%sjson\n[\n  %s,\n  %s\n]\n%s\n\nBoth are new.\n' "$FENCE" "$MEM" "$MEM" "$FENCE")"
case_ "prose then fenced array: stored" 1 "" \
  "$(printf 'Here are the memories:\n\n%sjson\n[%s]\n%s\n' "$FENCE" "$MEM" "$FENCE")"
# prose before the fence that itself opens with [ is not the payload: the fenced block is
case_ "bracket-led link then fenced array: stored" 1 "" \
  "$(printf 'Filed:\n[LAB-123](https://example.test/issue) covers the bug.\n\n%sjson\n[%s]\n%s\n' "$FENCE" "$MEM" "$FENCE")"
case_ "bracket-led [] remark then fenced array: stored" 1 "" \
  "$(printf '[] means no memories; this session does have one:\n%sjson\n[%s]\n%s\n' "$FENCE" "$MEM" "$FENCE")"
# a fence in the explanation is not the payload: the leading bare array is
case_ "bare array then prose with a fenced command: stored" 1 "" \
  "$(printf '[%s]\nTo check it, run:\n%ssh\nnpm test\n%s\n' "$MEM" "$FENCE" "$FENCE")"
case_ "bare array then prose with a fenced []: stored" 1 "" \
  "$(printf '[%s]\nAn empty result would look like:\n%sjson\n[]\n%s\n' "$MEM" "$FENCE" "$FENCE")"
case_ "fenced command then fenced array: stored" 1 "" \
  "$(printf 'Verified with:\n%ssh\nnpm test\n%s\n%sjson\n[%s]\n%s\n' "$FENCE" "$FENCE" "$FENCE" "$MEM" "$FENCE")"
# a footnote-style line that opens with a JSON value but goes on is prose
case_ "footnote line then fenced array: stored" 1 "" \
  "$(printf '[1] see the ticket.\n%sjson\n[%s]\n%s\n' "$FENCE" "$MEM" "$FENCE")"
case_ "CRLF fenced array: stored" 1 "" \
  "$(printf '%sjson\r\n[%s]\r\n%s\r\n' "$FENCE" "$MEM" "$FENCE")"
case_ "indented fenced array then prose: stored" 1 "" \
  "$(printf '  %sjson\n  [%s]\n  %s\nDone.\n' "$FENCE" "$MEM" "$FENCE")"
case_ "prose then bare array: stored" 1 "" \
  "$(printf 'Here are the memories:\n[%s]\n' "$MEM")"
case_ "bare array: stored" 1 "" \
  "$(printf '[%s]' "$MEM")"
# a reply with no JSON value is logged once, on one line, with the reply itself
case_ "prose only: logged with the reply" 0 "LLM output not valid JSON: I could not find anything. Sorry." \
  "$(printf 'I could not find anything.\nSorry.\n')"
case_ "truncated array: logged with the reply" 0 'LLM output not valid JSON: [{"content":"a lesson"' \
  "$(printf '[%s, {"content":"cut sho' "$MEM")"
case_ "object envelope: logged as not an array" 0 "valid JSON but not an array" \
  "$(printf '{"memories":[%s]}\n' "$MEM")"

[[ $RC -eq 0 ]] && echo "ALL PASS: the Stop hook reads the extractor's array and ignores prose around it"
exit $RC
