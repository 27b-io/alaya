#!/usr/bin/env bash
# Shim test for _resolve_secret in alaya-session-save.sh (LAB-1663 pattern,
# generalized from a hardcoded `op read` call to any value-or-command source).
# Proves: cold fetch = 1 resolver call; warm = 0 resolver calls; 0600 cache
# perms; stale cache served when the resolver fails; a failing resolver retried
# at most once per 15 min (stale and cold); SECRET_CACHE_MINUTES=0 persists
# nothing; hook still parses.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOK="$HERE/alaya-session-save.sh"

T=$(mktemp -d); trap 'rm -rf "$T"' EXIT

printf '%s\n' '#!/bin/bash' \
  'echo $(( $(cat "$RESOLVECOUNT" 2>/dev/null || echo 0) + 1 )) > "$RESOLVECOUNT"' \
  '[[ -f "$RESOLVEFAIL" ]] && exit 1' \
  'echo "sekrit-value"' > "$T/resolver"
chmod +x "$T/resolver"

# extract the real functions from the shipped hook
sed -n '/^_run_secret_cmd()/,/^}/p;/^_resolve_secret()/,/^}/p' "$HOOK" > "$T/fn.sh"
[[ $(grep -c '^_run_secret_cmd()\|^_resolve_secret()' "$T/fn.sh") == 2 ]] || { echo "FAIL: functions not found in hook"; exit 1; }

export RESOLVECOUNT="$T/count" RESOLVEFAIL="$T/fail.flag"
export SECRET_CACHE_MINUTES=720
export STATE_DIR="$T"
unset TEST_SECRET # an inherited value would short-circuit the command-source path
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
touch -t "$STALE" "$C" "$C.attempt"; touch "$RESOLVEFAIL"   # 13 h later: stale cache + resolver now failing
v3=$(_resolve_secret TEST_SECRET "$C"); c3=$(cat "$RESOLVECOUNT")

[[ "$v1" == "sekrit-value" ]] || { echo "FAIL cold value: $v1"; exit 1; }
[[ "$c1" == 1 ]]              || { echo "FAIL cold count: $c1"; exit 1; }
[[ "$v2" == "sekrit-value" && "$c2" == 1 ]] || { echo "FAIL warm: v=$v2 c=$c2 (resolver called again)"; exit 1; }
[[ "$perms" == 600 ]]         || { echo "FAIL perms: $perms"; exit 1; }
[[ "$v3" == "sekrit-value" && "$c3" == 2 ]] || { echo "FAIL stale-fallback: v=$v3 c=$c3"; exit 1; }
bash -n "$HOOK" || { echo "FAIL syntax"; exit 1; }

# Retry backoff: a failing resolver is re-run at most once per 15 min, not on
# every Stop — each retry against a rate-limited secret manager spends quota.
for _ in 1 2 3 4 5 6 7 8 9; do v=$(_resolve_secret TEST_SECRET "$C"); done
c6=$(cat "$RESOLVECOUNT")
[[ "$v" == "sekrit-value" && "$c6" == 2 ]] || { echo "FAIL backoff (stale): v=$v c=$c6, want stale value and no retry"; exit 1; }
RETRY=$(python3 -c 'import datetime; print((datetime.datetime.now()-datetime.timedelta(minutes=16)).strftime("%Y%m%d%H%M"))')
touch -t "$RETRY" "$C.attempt"
v=$(_resolve_secret TEST_SECRET "$C"); c7=$(cat "$RESOLVECOUNT")
[[ "$v" == "sekrit-value" && "$c7" == 3 ]] || { echo "FAIL backoff expiry: v=$v c=$c7, want one retry after 15 min"; exit 1; }
# Cold cache + failing resolver: one call, and no empty cache file left behind
C2="$T/cold-cache"
for _ in 1 2 3 4 5 6 7 8 9 10; do v=$(_resolve_secret TEST_SECRET "$C2") || true; done
c8=$(cat "$RESOLVECOUNT")
[[ -z "$v" && "$c8" == 4 && ! -e "$C2" ]] || { echo "FAIL backoff (cold): v=$v c=$c8 cache-exists=$([[ -e $C2 ]] && echo y || echo n)"; exit 1; }
rm -f "$RESOLVEFAIL"
# A successful refresh clears the marker, so deleting the cache after a key
# rotation refetches on the next call instead of waiting out the backoff.
C3="$T/rotate-cache"
_resolve_secret TEST_SECRET "$C3" >/dev/null; rm -f "$C3"
v=$(_resolve_secret TEST_SECRET "$C3") || true; c9=$(cat "$RESOLVECOUNT")
[[ "$v" == "sekrit-value" && "$c9" == 6 ]] || { echo "FAIL rotation refetch: v=$v c=$c9, want an immediate refetch"; exit 1; }

# Direct-value path bypasses the command sourcing entirely
export TEST_SECRET="direct-value"
v4=$(_resolve_secret TEST_SECRET "$C"); c4=$(cat "$RESOLVECOUNT")
[[ "$v4" == "direct-value" && "$c4" == "$c9" ]] || { echo "FAIL direct-value: v=$v4 c=$c4"; exit 1; }

# SECRET_CACHE_MINUTES=0, in its own state dir: the command runs on every call,
# nothing is written but failures.log, and a failure returns non-zero with no
# stale fallback, even when the failing command printed something.
unset TEST_SECRET
Z="$T/zero"; mkdir "$Z"
zero() { STATE_DIR="$Z" SECRET_CACHE_MINUTES=0 _resolve_secret TEST_SECRET "$Z/zero-cache"; }
zfiles() { find "$Z" -mindepth 1 ! -name failures.log | sort | tr '\n' ' '; }
before=$(cat "$RESOLVECOUNT")
v=$(zero); c=$(cat "$RESOLVECOUNT")
[[ "$v" == "sekrit-value" && "$c" == $((before + 1)) && -z "$(zfiles)" ]] \
  || { echo "FAIL zero (success): v=$v calls=$((c - before)) files=$(zfiles)"; exit 1; }
v=$(zero); c=$(cat "$RESOLVECOUNT")
[[ "$v" == "sekrit-value" && "$c" == $((before + 2)) && -z "$(zfiles)" ]] \
  || { echo "FAIL zero (no cache): v=$v calls=$((c - before)) files=$(zfiles)"; exit 1; }
touch "$RESOLVEFAIL"
rc=0; v=$(zero) || rc=$?; c=$(cat "$RESOLVECOUNT")
[[ -z "$v" && "$rc" != 0 && "$c" == $((before + 3)) && -z "$(zfiles)" ]] \
  || { echo "FAIL zero (failure): v=$v rc=$rc calls=$((c - before)) files=$(zfiles)"; exit 1; }
rc=0; v=$(zero) || rc=$?; c=$(cat "$RESOLVECOUNT")
[[ -z "$v" && "$rc" != 0 && "$c" == $((before + 4)) ]] \
  || { echo "FAIL zero (no backoff): v=$v rc=$rc calls=$((c - before))"; exit 1; }
rm -f "$RESOLVEFAIL"
printf '%s\n' '#!/bin/bash' 'printf partial-secret; exit 42' > "$T/partial"; chmod +x "$T/partial"
rc=0; v=$(TEST_SECRET_CMD="$T/partial" zero) || rc=$?
[[ -z "$v" && "$rc" != 0 && -z "$(zfiles)" ]] \
  || { echo "FAIL zero (non-zero exit with output): v=$v rc=$rc files=$(zfiles)"; exit 1; }

# Whole hook: with 0, key files and markers left by an earlier setting are purged
# before any gate can skip the save (here the unset ALAYA_URL gate, which exits
# before either secret is resolved). Above 0 they are kept.
seed() { for f in llm-api-key alaya-api-key; do printf 'old-plaintext' > "$Z/$f"; touch "$Z/$f.attempt"; done; }
runhook() { env -u ALAYA_URL ALAYA_HOOK_STATE_DIR="$Z" ALAYA_SECRET_CACHE_MINUTES="$1" bash "$HOOK" </dev/null; }
seed; runhook 0
[[ -z "$(zfiles)" ]] || { echo "FAIL zero (hook purge): files=$(zfiles)"; exit 1; }
seed; runhook 720
[[ "$(zfiles)" == "$Z/alaya-api-key $Z/alaya-api-key.attempt $Z/llm-api-key $Z/llm-api-key.attempt " ]] \
  || { echo "FAIL non-zero (hook kept cache): files=$(zfiles)"; exit 1; }

# python3 watchdog branch (macOS/BSD path, where timeout(1) doesn't exist):
# with timeout hidden from PATH, a hanging resolver must be killed at the
# bound and logged — not ride to the hook's 60s SIGKILL. Linux CI would
# otherwise never execute this branch.
printf '%s\n' '#!/bin/bash' 'sleep 30' > "$T/hangs"; chmod +x "$T/hangs"
mkdir "$T/nobin"
for c in bash python3 find cat sleep touch; do ln -s "$(command -v "$c")" "$T/nobin/$c"; done
start=$(date +%s)
v5=$(env PATH="$T/nobin" STATE_DIR="$T" SECRET_CACHE_MINUTES=720 _RESOLVER_TIMEOUT_SECS=2 \
    HANG_SECRET_CMD="$T/hangs" bash -c "source '$T/fn.sh'; _resolve_secret HANG_SECRET '$T/cache-hang'") || true
took=$(( $(date +%s) - start ))
[[ -z "$v5" && "$took" -le 15 ]] || { echo "FAIL watchdog: v='$v5' took=${took}s"; exit 1; }
grep -q 'timed out' "$T/failures.log" || { echo "FAIL watchdog: no timeout line in failures.log"; exit 1; }

echo "ALL PASS: cold=1 call, warm=0 calls, 0600 perms, stale cache on resolver failure, 15-min retry backoff (stale and cold), direct-value bypass, CACHE_MINUTES=0 persists nothing and purges leftovers, watchdog bound without timeout(1), hook syntax clean"
