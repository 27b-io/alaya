#!/usr/bin/env bash
# The Stop hook's save must outlive the CLI. Claude Code sends SIGTERM to a
# Stop hook still running when the CLI exits, and an SDK-driven CLI exits
# within seconds of its last result, so a hook that extracts and stores in the
# foreground is killed at a run's final Stop and saves nothing. This starts the
# real hook the way that exit treats it: a parent writes the hook input, closes
# the hook's stdin and, the moment the hook returns (or after one second if it
# does not), kills the hook's whole process group. The extractor answers after
# two seconds (stub curl, as in test-prompt-extraction.sh), so a foreground
# save, or a copy not yet in its own session when the hook returns, is killed.
# Then it asserts the store call happened and the state file was written, and
# that the hook's stdout and stderr, read through a pipe as the CLI reads them,
# reached EOF at once: a copy that kept either would hold the CLI until it ends.
#
#   scripts/test-stop-lifetime.sh
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-session-save.sh"
T=$(mktemp -d) || exit 1; trap 'rm -rf "$T"' EXIT
RC=0

# stub curl: the extraction call sleeps 2s then returns one memory; the store
# POST keeps its body and answers 200 (the hook reads the code via -w '%{http_code}')
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'STUB'
#!/bin/sh
prev=; body=
for a; do [ "$prev" = "-d" ] && body=$a; prev=$a; done
case "$*" in
  *llm.test*) sleep 2
    echo '{"choices":[{"message":{"content":"[{\"content\":\"stub memory\"}]"}}]}' ;;
  *) printf '%s' "$body" > "$CAP/store.json"; printf 200 ;;
esac
STUB
chmod +x "$T/bin/curl"

# a transcript that clears every save gate: old enough, 5 prompt lines
for i in 1 2 3 4 5; do
  jq -nc --arg t "prompt $i" '{type:"user", timestamp:"2020-01-01T00:00:00Z", message:{role:"user", content:$t}}'
done > "$T/transcript.jsonl"
jq -nc '{type:"assistant", message:{content:[{type:"text", text:"closing words"}]}}' >> "$T/transcript.jsonl"

# PATH for one run: the stub, then every tool the hook uses, linked one by one
# so that a run can leave setsid out and take the macOS/BSD fallback
tools() { # <dir> <tool...>
  mkdir -p "$1"; ln -sf "$T/bin/curl" "$1/curl"
  local t p
  for t in "${@:2}"; do p=$(command -v "$t") && ln -sf "$p" "$1/$t"; done
}
BASE=(bash env cat jq date mkdir rm grep head tail wc sed basename dirname git mktemp find seq
  printf touch timeout python3 sleep kill)
tools "$T/path-setsid" "${BASE[@]}" setsid
tools "$T/path-nosetsid" "${BASE[@]}"

now_ms() { python3 -c 'import time; print(int(time.time() * 1000))'; }
PY=$(command -v python3)
# setsid(1) for the harness itself, portable: a new session and process group
NEW_GROUP=("$PY" -c 'import os, sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])')

# run <label> <path-dir> — the "CLI": start the hook in its own process group,
# feed it the input and close its stdin, read its stdout and stderr through a
# pipe as the CLI does, kill the group the moment the hook returns, give the
# pipe until 1s after the start to reach EOF, then wait up to 15s for the save
run() {
  local home="$T/home-$1" cap="$T/cap-$1" out="$T/out-$1" pid start dur reader eof held
  mkdir -p "$home" "$cap"
  mkfifo "$out"
  # the CLI's end of the hook's output: EOF comes only when every holder of the
  # pipe has closed it, so a copy that keeps it open holds the CLI
  cat "$out" >/dev/null & reader=$!
  start=$(now_ms)
  printf '{"session_id":"s1","transcript_path":"%s"}' "$T/transcript.jsonl" \
    | (cd "$T" && exec "${NEW_GROUP[@]}" env -i HOME="$home" CAP="$cap" PATH="$2" \
        ALAYA_URL=http://alaya.test/store ALAYA_LLM_URL=http://llm.test/v1/chat/completions \
        ALAYA_LLM_API_KEY=k ALAYA_API_KEY=k ALAYA_HOOK_STATE_DIR="$home/state" \
        bash "$HOOK") >"$out" 2>&1 &
  pid=$!
  # kill the group the moment the hook returns, or after 1s if it does not:
  # a copy still in the hook's group when it returns dies here
  ( sleep 1; kill -TERM -- "-$pid" ) 2>/dev/null & watchdog=$!
  wait "$pid" 2>/dev/null
  kill -TERM -- "-$pid" 2>/dev/null
  dur=$(( $(now_ms) - start ))
  kill "$watchdog" 2>/dev/null; wait "$watchdog" 2>/dev/null
  # bounded: a pipe still held 1s after the start is a failure, and the reader
  # is stopped, so a copy that holds the pipe for long cannot hang the test
  while kill -0 "$reader" 2>/dev/null && (( $(now_ms) - start < 1000 )); do sleep 0.02; done
  eof=$(( $(now_ms) - start )); held=0
  if kill -0 "$reader" 2>/dev/null; then held=1; kill "$reader" 2>/dev/null; fi
  wait "$reader" 2>/dev/null
  for _ in $(seq 1 30); do [[ -s "$home/state/s1" ]] && break; sleep 0.5; done

  if [[ -s "$cap/store.json" ]] && jq -e '.content == "stub memory"' "$cap/store.json" >/dev/null 2>&1
  then echo "ok    $1: store call made after the CLI exit"; else echo "FAIL  $1: no store call"; RC=1; fi
  if [[ "$(sed -n 2p "$home/state/s1" 2>/dev/null)" == 5 ]]
  then echo "ok    $1: state file written"; else echo "FAIL  $1: no state file"; RC=1; fi
  # a synchronous hook must not hold the CLI: it returns before the extraction ends
  if [[ $dur -lt 1000 ]]; then echo "ok    $1: hook returned in ${dur}ms"
  else echo "FAIL  $1: hook still running after ${dur}ms"; RC=1; fi
  # nor hold its output pipe: the copy must not keep the hook's stdout or stderr
  if [[ $held -eq 0 ]]; then echo "ok    $1: hook output closed within ${eof}ms"
  else echo "FAIL  $1: hook output still open after ${eof}ms"; RC=1; fi
}
if command -v setsid >/dev/null 2>&1; then
  run setsid "$T/path-setsid"
else
  echo "skip  setsid: not installed here (the no-setsid fallback runs below)"
fi
run no-setsid "$T/path-nosetsid"

# a synchronous registration: an async hook is killed with the CLI before it can detach
if jq -e '.hooks.Stop[].hooks[] | select(.command | contains("alaya-session-save.sh")) | .async != true' \
  "$HERE/../hooks/hooks.json" >/dev/null; then echo "ok    hooks.json: Stop hook is synchronous"
else echo "FAIL  hooks.json: Stop hook is registered async"; RC=1; fi

[[ $RC -eq 0 ]] && echo "ALL PASS: the Stop hook's save survives the CLI's exit"
exit $RC
