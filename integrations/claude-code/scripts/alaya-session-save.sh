#!/usr/bin/env bash
# Stop hook: async memory extraction from transcript → Alaya REST API.
# Fires after Claude stops. Reads transcript, calls an LLM for structured
# extraction, POSTs memories with dedup. Never blocks the UX.
#
# All values below are configuration, not hardcoded endpoints — see
# ../config/alaya-hook.env.example and the plugin README for the full list.
#
# Guards against duplicate saves:
#   1. ALAYA_MIN_NEW_MESSAGES — skip unless N+ new user messages since last save
#   2. ALAYA_COOLDOWN_SECS — skip if saved within N seconds
#   3. ALAYA_DEDUP_THRESHOLD on the Alaya POST — catches LLM rephrases of same facts
set -euo pipefail
umask 077 # every file this hook creates (cache, state, log) is owner-only

MIN_DURATION_SECS="${ALAYA_MIN_DURATION_SECS:-120}"
MIN_NEW_MESSAGES="${ALAYA_MIN_NEW_MESSAGES:-5}"
COOLDOWN_SECS="${ALAYA_COOLDOWN_SECS:-300}"
MEMORY_CAP="${ALAYA_MEMORY_CAP:-3}"
CLIENT_HOSTNAME="${ALAYA_CLIENT_HOSTNAME:-claude-code-hook}"
STATE_DIR="${ALAYA_HOOK_STATE_DIR:-$HOME/.cache/alaya-hook}"
SECRET_CACHE_MINUTES="${ALAYA_SECRET_CACHE_MINUTES:-720}"
LLM_MODEL="${ALAYA_LLM_MODEL:-claude-haiku-4-5}"

# Reject a non-numeric override instead of feeding it to jq --argjson later
# (a bad value there would silently drop every extracted memory).
_numeric_or_default() { # <value> <default>
    [[ "$1" =~ ^[0-9]+(\.[0-9]+)?$ ]] && printf '%s' "$1" || printf '%s' "$2"
}
DEDUP_THRESHOLD=$(_numeric_or_default "${ALAYA_DEDUP_THRESHOLD:-0.70}" 0.70)
IMPORTANCE=$(_numeric_or_default "${ALAYA_IMPORTANCE:-0.7}" 0.7)

# The gate vars feed integer [[ -lt ]] comparisons — a non-numeric override
# ("5m") would error the test falsy and silently disable the gate entirely.
# 10# strips leading zeros ("08" would otherwise be rejected as bad octal).
_int_or_default() { # <value> <default>
    [[ "$1" =~ ^[0-9]+$ ]] && printf '%s' "$((10#$1))" || printf '%s' "$2"
}
MIN_DURATION_SECS=$(_int_or_default "$MIN_DURATION_SECS" 120)
MIN_NEW_MESSAGES=$(_int_or_default "$MIN_NEW_MESSAGES" 5)
COOLDOWN_SECS=$(_int_or_default "$COOLDOWN_SECS" 300)
MEMORY_CAP=$(_int_or_default "$MEMORY_CAP" 3)
SECRET_CACHE_MINUTES=$(_int_or_default "$SECRET_CACHE_MINUTES" 720)

mkdir -p "$STATE_DIR" 2>/dev/null || true
_log_failure() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" >> "$STATE_DIR/failures.log" 2>/dev/null || true; }
_skip() { _log_failure "$1"; exit 0; }

# GNU `date -d` doesn't exist on BSD/macOS date; python3 (already a documented
# prerequisite, used by the PreCompact hook) parses ISO-8601 portably.
_iso_to_epoch() {
    date -d "$1" +%s 2>/dev/null && return
    python3 -c "import sys,datetime; print(int(datetime.datetime.fromisoformat(sys.argv[1].replace('Z','+00:00')).timestamp()))" "$1" 2>/dev/null
}

# --- Gate: required endpoints must be configured ---
# Absent config is a logged no-op, never a hook error surfaced into the session.
[[ -z "${ALAYA_URL:-}" ]] && _skip "config: ALAYA_URL not set, skipping save"
[[ -z "${ALAYA_LLM_URL:-}" ]] && _skip "config: ALAYA_LLM_URL not set, skipping save"
command -v jq >/dev/null 2>&1 || _skip "config: jq not found on PATH, skipping save"

# Disable errexit for everything downstream: pipelines that use head would die
# to SIGPIPE under pipefail, and any parse failure past this point must reach
# a logged skip, not an unlogged errexit death (LAB-170).
set +e

INPUT=$(cat)
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
TRANSCRIPT=$(echo "$INPUT" | jq -r '.transcript_path // empty' 2>/dev/null)

# --- Gate: need a transcript ---
[[ -z "$TRANSCRIPT" || ! -f "$TRANSCRIPT" ]] && exit 0

# --- Gate: skip short sessions ---
START_ISO=$(jq -r 'select(.timestamp != null) | .timestamp' "$TRANSCRIPT" 2>/dev/null | head -1)
[[ -z "$START_ISO" ]] && exit 0
START_EPOCH=$(_iso_to_epoch "$START_ISO")
[[ -z "$START_EPOCH" ]] && _skip "cannot parse transcript timestamp '$START_ISO' (no working date/python3)"
NOW_EPOCH=$(date +%s)
DURATION=$((NOW_EPOCH - START_EPOCH))
[[ $DURATION -lt $MIN_DURATION_SECS ]] && exit 0

_HOOKDIR=$(mktemp -d)
trap 'rm -rf "$_HOOKDIR"' EXIT
PROJECT=$(basename "$(pwd)")
BRANCH=$(git branch --show-current 2>/dev/null || echo "n/a")

# --- Extract user messages (string content only, skip XML system tags) ---
grep '"type":"user"' "$TRANSCRIPT" \
    | jq -r 'select(.message.content | type == "string") | .message.content' 2>/dev/null \
    | grep -v '^<' \
    > "$_HOOKDIR/all_messages.txt" || true

TOTAL=$(wc -l < "$_HOOKDIR/all_messages.txt" 2>/dev/null || echo 0)
[[ $TOTAL -eq 0 ]] && exit 0

# --- Gate: exchange counting + cooldown ---
# State is per-session — different sessions save independently
STATE_FILE="$STATE_DIR/${SESSION_ID:-default}"

LAST_SAVE_EPOCH=0
LAST_SAVE_COUNT=0
if [[ -f "$STATE_FILE" ]]; then
    LAST_SAVE_EPOCH=$(sed -n '1p' "$STATE_FILE" 2>/dev/null || echo 0)
    LAST_SAVE_COUNT=$(sed -n '2p' "$STATE_FILE" 2>/dev/null || echo 0)
    # A torn/garbage state file must degrade to "no previous save", not kill
    # the hook with an unlogged arithmetic error under set -u.
    [[ "$LAST_SAVE_EPOCH" =~ ^[0-9]+$ ]] || LAST_SAVE_EPOCH=0
    [[ "$LAST_SAVE_COUNT" =~ ^[0-9]+$ ]] || LAST_SAVE_COUNT=0
fi

# Skip if not enough new messages since last save
NEW_MESSAGES=$((TOTAL - LAST_SAVE_COUNT))
[[ $NEW_MESSAGES -lt $MIN_NEW_MESSAGES ]] && exit 0

# Skip if saved too recently
ELAPSED=$((NOW_EPOCH - LAST_SAVE_EPOCH))
[[ $ELAPSED -lt $COOLDOWN_SECS ]] && exit 0

# --- Build context window: first message + recent messages ---
if [[ $TOTAL -le 15 ]]; then
    head -c 32768 "$_HOOKDIR/all_messages.txt" > "$_HOOKDIR/context.txt"
else
    {
        echo "=== SESSION TOPIC ==="
        head -1 "$_HOOKDIR/all_messages.txt"
        echo ""
        echo "=== RECENT USER MESSAGES ==="
        tail -12 "$_HOOKDIR/all_messages.txt"
    } > "$_HOOKDIR/context.txt"
fi

# --- Last assistant response (richest source of decisions/conclusions) ---
LAST_ASSISTANT=$(grep '"type":"assistant"' "$TRANSCRIPT" \
    | jq -r 'select(.message.content | type == "array") | .message.content[] | select(.type == "text") | .text' 2>/dev/null \
    | tail -60) || true
LAST_ASSISTANT="${LAST_ASSISTANT:0:3000}"

if [[ -n "$LAST_ASSISTANT" ]]; then
    {
        echo ""
        echo "=== CLAUDE'S LAST RESPONSE ==="
        echo "$LAST_ASSISTANT"
    } >> "$_HOOKDIR/context.txt"
fi

# --- Resolve a secret from either a direct env var or a `_CMD` sourcing
# command, with a cached result so a slow secret manager (1Password, vault,
# etc.) isn't invoked on every Stop. On resolver failure, serve the last
# cached value rather than dropping the save (LAB-1663). ---
# `rm` the matching cache file under STATE_DIR to force a refresh after a
# key rotation.
_resolve_secret() { # <env-var-name> <cache-file>
    local name="$1" cache="$2" direct cmd_var cmd val
    direct="${!name:-}"
    if [[ -n "$direct" ]]; then
        printf '%s' "$direct"
        return 0
    fi
    cmd_var="${name}_CMD"
    cmd="${!cmd_var:-}"
    [[ -z "$cmd" ]] && return 1
    if [[ -z "$(find "$cache" -mmin -"$SECRET_CACHE_MINUTES" 2>/dev/null)" ]]; then
        # Resolver's own stderr (e.g. "op: command not found", vault-not-found)
        # goes to failures.log instead of /dev/null — it's the one piece of
        # info that explains WHICH failure mode this is.
        # Bound the resolver: a hung secret manager (locked 1Password app,
        # vault network hang) would otherwise ride to the hook's 60s timeout
        # and die by SIGKILL with no log line. With a bound, the failure
        # falls through to the caller's logged "unresolved" skip.
        if command -v timeout >/dev/null 2>&1; then
            val=$(timeout 15 bash -c "$cmd" 2>>"$STATE_DIR/failures.log")
        else
            val=$(eval "$cmd" 2>>"$STATE_DIR/failures.log")
        fi
        [[ -n "$val" ]] && printf '%s' "$val" > "$cache"
    fi
    [[ -r "$cache" ]] && cat "$cache"
}

LLM_API_KEY=$(_resolve_secret ALAYA_LLM_API_KEY "$STATE_DIR/llm-api-key") || true
[[ -z "$LLM_API_KEY" ]] && _skip "config: ALAYA_LLM_API_KEY unresolved, skipping save"

# Alaya bearer — server auth is fail-closed. Missing key means no save.
ALAYA_API_KEY=$(_resolve_secret ALAYA_API_KEY "$STATE_DIR/alaya-api-key") || true
[[ -z "$ALAYA_API_KEY" ]] && _skip "config: ALAYA_API_KEY unresolved, skipping save"

# --- LLM call: memory-focused structured extraction ---
read -r -d '' SYSTEM_PROMPT <<'SYS' || true
You extract reusable knowledge from Claude Code sessions.
Output a JSON array of 0-3 memories worth recalling in 6 months.

Types: "decision" (choices+rationale), "note" (discoveries, gotchas), "reference" (commands, configs, endpoints), "task" (unfinished work, blockers).

Each object: {"content":"self-contained description","memory_type":"...","tags":["project","topic"],"summary":"one-line under 50 tokens"}

Rules:
- Only extract NEW information — skip topics likely already saved from earlier in this session
- Skip routine work, greetings, process narration, code dumps
- Skip generic programming advice any senior dev would know
- Skip descriptions of tooling internals (how hooks/scripts/configs work) — that's in the code
- Skip bug reports or issues that have been filed — that's in the issue tracker
- Content must stand alone — no "in this session" or "we discussed" references
- Tags: always include project name, plus relevant technologies/topics
- If nothing worth remembering, output []
SYS

DURATION_MIN=$((DURATION / 60))

jq -n \
    --rawfile ctx "$_HOOKDIR/context.txt" \
    --arg model "$LLM_MODEL" \
    --arg sys "$SYSTEM_PROMPT" \
    --arg proj "$PROJECT" \
    --arg branch "$BRANCH" \
    --arg dur "${DURATION_MIN}m" \
    '{
        model: $model,
        messages: [
            {role: "system", content: $sys},
            {role: "user", content: ("Project: " + $proj + " (" + $branch + ", " + $dur + ")\n\n" + $ctx)}
        ],
        max_tokens: 800,
        temperature: 0.2
    }' > "$_HOOKDIR/llm_payload.json" 2>/dev/null

RAW=$(curl -s --max-time 30 "$ALAYA_LLM_URL" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $LLM_API_KEY" \
    -d @"$_HOOKDIR/llm_payload.json" 2>/dev/null) || true
RESPONSE=$(printf '%s' "$RAW" | jq -r '.choices[0].message.content // empty' 2>/dev/null) || true

# A dead LLM gate must never be silent again (LAB-170: 5 lost days taught us that)
[[ -z "$RESPONSE" ]] && _skip "LLM extraction failed model=$LLM_MODEL resp=${RAW:0:300}"

# --- Parse LLM output (strip markdown fences if present) ---
printf '%s\n' "$RESPONSE" | sed 's/^```[a-zA-Z]*//;/^$/d' > "$_HOOKDIR/memories.json"

# -1 = valid JSON but not an array (output-envelope drift, e.g. {"memories":[...]})
# — must be logged, or a model/gateway change silently stops all saves (LAB-170).
# An empty [] is a legitimate "nothing worth saving" and stays a quiet exit.
MEMORY_COUNT=$(jq 'if type == "array" then length else -1 end' "$_HOOKDIR/memories.json" 2>&1) \
    || _skip "LLM output not valid JSON: ${MEMORY_COUNT:0:200}"
[[ "$MEMORY_COUNT" -eq -1 ]] && _skip "LLM output valid JSON but not an array: $(head -c 200 "$_HOOKDIR/memories.json")"
[[ "$MEMORY_COUNT" -eq 0 ]] && exit 0

# Cap even if the LLM returned more
[[ "$MEMORY_COUNT" -gt "$MEMORY_CAP" ]] && MEMORY_COUNT=$MEMORY_CAP

# --- POST each memory to Alaya ---
SAVED=0
for i in $(seq 0 $((MEMORY_COUNT - 1))); do
    MEMORY=$(jq --argjson idx "$i" '.[$idx]' "$_HOOKDIR/memories.json" 2>/dev/null)
    [[ -z "$MEMORY" || "$MEMORY" == "null" ]] && continue

    # Validate required field
    CONTENT=$(echo "$MEMORY" | jq -r '.content // empty' 2>/dev/null)
    [[ -z "$CONTENT" ]] && continue

    # Enrich with provenance, dedup threshold, importance
    PAYLOAD=$(echo "$MEMORY" | jq \
        --arg hostname "$CLIENT_HOSTNAME" \
        --argjson dedup "$DEDUP_THRESHOLD" \
        --argjson importance "$IMPORTANCE" \
        '{
            content: .content,
            memory_type: (.memory_type // "note"),
            tags: ((.tags // []) + ["auto-save"] | unique),
            summary: (.summary // null),
            client_hostname: $hostname,
            dedup_threshold: $dedup,
            metadata: {importance: $importance}
        }' 2>/dev/null) || continue

    HTTP_CODE=$(curl -s -o /dev/null --max-time 5 -w '%{http_code}' \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $ALAYA_API_KEY" \
        -d "$PAYLOAD" \
        "$ALAYA_URL" 2>/dev/null) || HTTP_CODE="000"
    if [[ "$HTTP_CODE" == 2?? ]]; then
        SAVED=$((SAVED + 1))
    else
        _log_failure "alaya store http $HTTP_CODE for memory $i (url=$ALAYA_URL)"
    fi
done

# --- Update state: record this save ---
if [[ $SAVED -gt 0 ]]; then
    printf '%s\n%s\n' "$NOW_EPOCH" "$TOTAL" > "$STATE_FILE" 2>/dev/null || true
else
    _log_failure "alaya store failed: $MEMORY_COUNT memories extracted, 0 saved (url=$ALAYA_URL)"
fi
