#!/usr/bin/env bash
# Shim test for _resolve_secret in alaya-session-save.sh (LAB-1663 pattern,
# generalized from a hardcoded `op read` call to any value-or-command source).
# Proves: cold fetch = 1 resolver call; warm = 0 resolver calls; 0600 cache
# perms; stale cache served when the resolver fails; hook still parses.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-session-save.sh"

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

printf '%s\n' '#!/bin/bash' \
  'echo $(( $(cat "$RESOLVECOUNT" 2>/dev/null || echo 0) + 1 )) > "$RESOLVECOUNT"' \
  '[[ -f "$RESOLVEFAIL" ]] && exit 1' \
  'echo "sekrit-value"' > "$T/resolver"
chmod +x "$T/resolver"

# extract the real function from the shipped hook
sed -n '/^_resolve_secret()/,/^}/p' "$HOOK" > "$T/fn.sh"
[[ -s "$T/fn.sh" ]] || { echo "FAIL: function not found in hook"; exit 1; }

export RESOLVECOUNT="$T/count" RESOLVEFAIL="$T/fail.flag"
export SECRET_CACHE_MINUTES=720
export STATE_DIR="$T"
export TEST_SECRET_CMD="$T/resolver"
source "$T/fn.sh"
umask 077 # the real hook sets this globally; the function itself no longer does
C="$T/cachefile"

v1=$(_resolve_secret TEST_SECRET "$C"); c1=$(cat "$RESOLVECOUNT")
v2=$(_resolve_secret TEST_SECRET "$C"); c2=$(cat "$RESOLVECOUNT")
# GNU stat -c / BSD-macOS stat -f; touch -t (portable) with the 13-hours-ago
# timestamp computed via python3 — GNU-only `touch -d '13 hours ago'` isn't.
perms=$(stat -c %a "$C" 2>/dev/null || stat -f %Lp "$C")
STALE=$(python3 -c 'import datetime; print((datetime.datetime.now()-datetime.timedelta(hours=13)).strftime("%Y%m%d%H%M"))')
touch -t "$STALE" "$C"; touch "$RESOLVEFAIL"   # stale cache + resolver now failing
v3=$(_resolve_secret TEST_SECRET "$C"); c3=$(cat "$RESOLVECOUNT")

[[ "$v1" == "sekrit-value" ]] || { echo "FAIL cold value: $v1"; exit 1; }
[[ "$c1" == 1 ]]              || { echo "FAIL cold count: $c1"; exit 1; }
[[ "$v2" == "sekrit-value" && "$c2" == 1 ]] || { echo "FAIL warm: v=$v2 c=$c2 (resolver called again)"; exit 1; }
[[ "$perms" == 600 ]]         || { echo "FAIL perms: $perms"; exit 1; }
[[ "$v3" == "sekrit-value" && "$c3" == 2 ]] || { echo "FAIL stale-fallback: v=$v3 c=$c3"; exit 1; }
bash -n "$HOOK" || { echo "FAIL syntax"; exit 1; }

# Direct-value path bypasses the command sourcing entirely
export TEST_SECRET="direct-value"
v4=$(_resolve_secret TEST_SECRET "$C"); c4=$(cat "$RESOLVECOUNT")
[[ "$v4" == "direct-value" && "$c4" == 2 ]] || { echo "FAIL direct-value: v=$v4 c=$c4"; exit 1; }

echo "ALL PASS: cold=1 call, warm=0 calls, 0600 perms, stale cache on resolver failure, direct-value bypass, hook syntax clean"
