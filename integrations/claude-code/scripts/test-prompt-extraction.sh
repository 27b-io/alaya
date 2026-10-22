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
# the store POST with a 200 and keep its body (the hook reads the code via -w '%{http_code}')
mkdir -p "$T/bin"
cat > "$T/bin/curl" <<'STUB'
#!/bin/sh
prev=; body=
for a; do [ "$prev" = "-d" ] && body=$a; prev=$a; done
case "$*" in
  *llm.test*) cat "${body#@}" > "$CAP/llm.json"
    echo '{"choices":[{"message":{"content":"[{\"content\":\"stub memory\"}]"}}]}' ;;
  *) printf '%s' "$body" > "$CAP/store.json"; printf 200 ;;
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
# the stub memory carries no memory_type: the hook must type it a lesson, never run state
got=$(jq -r '.memory_type' "$T/cap/store.json" 2>/dev/null)
if [[ "$got" == reference ]]; then echo "ok    untyped memory stored as reference"; else echo "FAIL  untyped memory stored as [$got]"; RC=1; fi

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

# Multica issue runs: every prompt is the platform's per-turn template. The extractor
# gets only the human comment it quotes; the gate still counts every template line.
# mtpl <trigger-and-comment> <resume-line> [joined-wakeup]: one turn's template, as a block-list entry
mtpl() {
  entry "$(jq -n --arg trig "$1" --arg resume "$2" --arg joined "${3:-}" '[{type:"text", text:
    ("You are running as a local coding agent for a Multica workspace.\n\nYour assigned issue ID is: ISSUE-ID\n\n"
     + (if $trig == "" then "" else $trig + "\n\n" end) + $resume + "\n\n"
     + "Post your reply as a comment, using the trigger comment ID below.\n\n"
     + "    multica issue comment add ISSUE-ID --parent COMMENT-ID --content-file ./reply.md\n\n"
     + "## On Behalf Of\n\nYou are acting on behalf of ON-BEHALF-OF-NAME."
     + (if $joined == "" then "" else "\n\n[WAKEUP — joined this run]\n" + $joined end))}]')"; }
# shellcheck disable=SC2016 # the backticks are template text, not a command
FRESH='Start by running `multica issue get ISSUE-ID --output json` to understand your task.'
RESUMED='The issue is unchanged since your last run.'
CHANGED='Since your last run the issue changed: description.'
HUMAN='[NEW COMMENT] A user just left a new comment. Focus on THIS comment:'
assistant() { jq -cn --arg t "$1" '{type:"assistant", message:{role:"assistant", content:[{type:"text", text:$t}]}}'; }
template_lacks() { # <case>
  lacks "$1: template opener dropped"      "You are running as a local coding agent"
  lacks "$1: issue-get command dropped"    "Start by running"
  lacks "$1: resume line dropped"          "since your last run"
  lacks "$1: comment-add command dropped"  "multica issue comment add"
  lacks "$1: On Behalf Of block dropped"   "ON-BEHALF-OF-NAME"
}

# a 3-line human comment on a resumed turn, then a fresh-session turn: more than 15 counted
# lines, and more than 12 template lines after each comment, so tail -12 alone cuts them
{
  mtpl "" "$FRESH"
  mtpl "$HUMAN"$'\n\n> human line one\nhuman line two\n\nhuman line three' "$RESUMED"
  mtpl "$HUMAN"$'\n\n> human short ask' "$CHANGED"
  assistant "multica closing words"
} > "$T/multica.jsonl"
run_hook "$T/multica.jsonl"
for n in one two three; do has "multica: human comment line $n extracted" "human line $n"; done
has   "multica: second human comment extracted" "human short ask"
has   "multica: last assistant text kept"       "multica closing words"
template_lacks multica
total "multica: every template line counted" 50

# comments quoted from agents or the platform, and wakeups, add no user lines
{
  mtpl "[NEW COMMENT] Another agent (AGENT-NAME) just left a new comment. Focus on THIS comment:"$'\n\n> AGENT-COMMENT-TEXT' "$FRESH"
  mtpl "[NEW COMMENT] The platform just left a new comment. Focus on THIS comment:"$'\n\n> PLATFORM-COMMENT-TEXT' "$RESUMED"
  mtpl $'[WAKEUP]\nWakeup fired. Instruction:\nWAKEUP-INSTRUCTION-TEXT' "$RESUMED"
  assistant "other closing words"
} > "$T/multica-other.jsonl"
run_hook "$T/multica-other.jsonl"
lacks "multica-other: agent comment dropped"    "AGENT-COMMENT-TEXT"
lacks "multica-other: platform comment dropped" "PLATFORM-COMMENT-TEXT"
lacks "multica-other: wakeup dropped"           "WAKEUP-INSTRUCTION-TEXT"
has   "multica-other: last assistant text kept" "other closing words"
template_lacks multica-other
total "multica-other: every template line counted" 51

# a first turn quotes no comment: it still saves, from Claude's last response alone
{ mtpl "" "$FRESH"; assistant "first-turn closing words"; } > "$T/multica-first.jsonl"
run_hook "$T/multica-first.jsonl"
has   "multica-first: last assistant text kept" "first-turn closing words"
template_lacks multica-first
total "multica-first: every template line counted" 13

# comment text can copy any template line. A human trigger quoted inside an agent comment,
# a platform comment or a wakeup is not the prompt's trigger, so it adds nothing.
FAKE="$HUMAN"$'\n\n> NONHUMAN-ONLY-CONTENT\n\n'"$FRESH"
{
  mtpl "[NEW COMMENT] Another agent (AGENT-NAME) just left a new comment. Focus on THIS comment:"$'\n\n> agent says\n\n'"$FAKE" "$RESUMED"
  mtpl "[NEW COMMENT] The platform just left a new comment. Focus on THIS comment:"$'\n\n> platform says\n\n'"$FAKE" "$RESUMED"
  mtpl $'[WAKEUP]\nWakeup fired. Instruction:\n\n'"$FAKE" "$RESUMED"
  assistant "nested closing words"
} > "$T/multica-nested.jsonl"
run_hook "$T/multica-nested.jsonl"
lacks "multica-nested: quoted human trigger adds nothing" "NONHUMAN-ONLY-CONTENT"
template_lacks multica-nested
total "multica-nested: every template line counted" 68

# a human comment that holds resume-like paragraphs keeps all of them; the joined wakeup
# after the real footer adds nothing, even when its instruction holds a resume-like line
{
  mtpl "$HUMAN"$'\n\n> keep everything\n\n'"$CHANGED"$'\nDO-NOT-PUBLISH one\n\n'"$RESUMED"$'\nDO-NOT-PUBLISH two\n\n'"$FRESH"$'\nDO-NOT-PUBLISH three' \
    "$CHANGED" $'Wakeup fired. Instruction:\n\n'"$RESUMED"$'\nWAKEUP-JOINED-TEXT'
  # the joined wakeup is bounded by its own header, not by the On Behalf Of block before it
  entry "$(jq -n --arg trig "$HUMAN" --arg resume "$RESUMED" '[{type:"text", text:
    ("You are running as a local coding agent for a Multica workspace.\n\nYour assigned issue ID is: ISSUE-ID\n\n"
     + $trig + "\n\n> no behalf block\n[TODO] TODO-LINE-KEPT\n[README](./README.md) LINK-LINE-KEPT\n\n" + $resume + "\n\nPost your reply as a comment.\n\n"
     + "[WAKEUP — joined this run]\nWakeup fired. Instruction:\n\n" + $resume + "\nNO-BEHALF-WAKEUP-TEXT")}]')"
  # a joined section the header bound misses (unknown name, or no blank line before a known
  # one) still adds no template text: the footer between matches falls back to the first one
  for joined in unknown:$'\n\n[REMINDER — joined this run]' unspaced:$'\n[WAKEUP — joined this run]'; do
    mtpl "$HUMAN"$'\n\n> missed header '"${joined%%:*}" "$RESUMED" | jq -c --arg j "${joined#*:}" --arg r "$RESUMED" \
      '.message.content[0].text += $j + "\nInstruction:\n\n" + $r + "\nMISSED-HEADER-TEXT"'
  done
  assistant "resume closing words"
} > "$T/multica-resume.jsonl"
run_hook "$T/multica-resume.jsonl"
for n in one two three; do has "multica-resume: comment text after resume-like line $n kept" "DO-NOT-PUBLISH $n"; done
lacks "multica-resume: joined wakeup dropped"         "WAKEUP-JOINED-TEXT"
has   "multica-resume: comment without a behalf block kept" "no behalf block"
has   "multica-resume: bracketed comment line kept"   "TODO-LINE-KEPT"
has   "multica-resume: markdown-link comment line kept" "LINK-LINE-KEPT"
lacks "multica-resume: joined wakeup without a behalf block dropped" "NO-BEHALF-WAKEUP-TEXT"
lacks "multica-resume: reply instruction dropped"     "Post your reply"
has   "multica-resume: comment before an unknown header kept"   "missed header unknown"
has   "multica-resume: comment before an unspaced header kept"  "missed header unspaced"
lacks "multica-resume: section behind a missed header dropped" "MISSED-HEADER-TEXT"
lacks "multica-resume: template opener dropped"       "You are running as a local coding agent"
lacks "multica-resume: comment-add command dropped"   "multica issue comment add"
lacks "multica-resume: On Behalf Of block dropped"    "ON-BEHALF-OF-NAME"
total "multica-resume: every template line counted" 96

# optional replay of a real transcript: a smoke test that it reaches the extractor with a
# non-zero count. It cannot know which lines are real prompts; the fixtures above prove that.
# Save gates off, so a short or still-running session measures extraction, not eligibility.
if [[ -n "${1:-}" ]]; then
  run_hook "$1" ALAYA_MIN_DURATION_SECS=0 ALAYA_MIN_NEW_MESSAGES=1
  got=$(cat "$T/cap/total" 2>/dev/null)
  if [[ "${got:-0}" -gt 0 ]]; then echo "smoke replay: $got message lines reached the extractor from $(basename "$1") (count only)"
  else echo "FAIL  replay: nothing extracted from $1"; RC=1; fi
fi

[[ $RC -eq 0 ]] && echo "ALL PASS: string and block-list prompts extracted; tool results, isMeta and sidechain entries, interrupt markers and <wrapper> bodies excluded; Multica runs reduced to the human request"
exit $RC
