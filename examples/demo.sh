#!/usr/bin/env bash
# examples/demo.sh — Ālaya 60-second walkthrough: automatic contradiction
# detection + supersession resolution, end to end against the REAL REST API.
#
# The story (5 steps, ~60s):
#   0. health           — confirm the server (and its graph backend) are up
#   1. store A          — record a policy fact
#   2. store B          — record the opposite fact; the server AUTO-DETECTS the
#                         contradiction on write and echoes it inline under
#                         `interference.contradictions` (no separate "analyze" call)
#   3. /contradictions  — surface the CONTRADICTS edge the store wrote to the graph
#   4. /supersede       — resolve it: B supersedes A (REST: old_hash / new_hash)
#   5. /search          — A is now hidden by default; include_superseded brings it
#                         back, stamped with superseded_by + the reason
#
# Requires: curl, jq.
#
# Auth — no literal tokens, ever:
#   The Ālaya server is fail-closed: it refuses to boot unless one of
#   ALAYA_API_KEY, OIDC_ISSUER, or DANGEROUSLY_ALLOW_UNAUTHENTICATED=true is set.
#   There is no silent no-auth default. "No auth" therefore means the operator
#   explicitly booted with DANGEROUSLY_ALLOW_UNAUTHENTICATED=true (only sane on a
#   private/loopback origin). This script sends an Authorization header ONLY when
#   ALAYA_API_KEY is exported — otherwise it sends none at all (not an empty one).
#
# Usage:
#   # Against a server booted with ALAYA_API_KEY:
#   ALAYA_URL=http://localhost:3001 ALAYA_API_KEY=your-key ./examples/demo.sh
#
#   # Against a local no-auth dev server (DANGEROUSLY_ALLOW_UNAUTHENTICATED=true):
#   ALAYA_URL=http://localhost:3001 ./examples/demo.sh
#
# Env:
#   ALAYA_URL       base URL of the REST server (default http://localhost:3001)
#   ALAYA_API_KEY   bearer token; if unset, no Authorization header is sent
#
# This is the REST surface (alaya-server). The MCP surface (POST /mcp, JSON-RPC)
# uses different field names for supersede (old_id/new_id) — do not copy them
# here; REST uses old_hash/new_hash.

set -euo pipefail

ALAYA_URL="${ALAYA_URL:-http://localhost:3001}"

command -v jq   >/dev/null 2>&1 || { echo "demo.sh: need jq on PATH";   exit 1; }
command -v curl >/dev/null 2>&1 || { echo "demo.sh: need curl on PATH"; exit 1; }

# Auth header assembled into an array so it is ENTIRELY ABSENT when no key is
# set — never an empty or placeholder "Authorization:" header.
AUTH=()
if [ -n "${ALAYA_API_KEY:-}" ]; then
  AUTH=(-H "Authorization: Bearer ${ALAYA_API_KEY}")
  echo "auth: bearer token (ALAYA_API_KEY is set)"
else
  echo "auth: none — assuming the server was booted with DANGEROUSLY_ALLOW_UNAUTHENTICATED=true"
  echo "      (set ALAYA_API_KEY to talk to a hardened server)"
fi

# POST helper: api <path> <json-body>
# -f makes any non-2xx fatal under `set -e`, so a 401 (wrong/missing key) or an
# empty graph fails LOUDLY at the offending step instead of silently.
api() {
  curl -fsS --connect-timeout 5 --max-time 30 "${AUTH[@]}" -H 'Content-Type: application/json' \
       -X POST "${ALAYA_URL}${1}" -d "${2}"
}

# --- 0. health -------------------------------------------------------------
# GET, unauthenticated — /health is not on the protected router. Full detail
# (total_memories, graph_health, …) requires auth or dev open mode (alaya#75);
# this demo targets the open-mode dev compose, so the fields are present.
# graph_health must not be "unhealthy": /contradictions (step 3) reads from FalkorDB; if the
# graph is down, store still works and the inline interference still appears,
# but the CONTRADICTS edge can't be surfaced and step 3 returns pairs: [].
echo "== 0. health =="
curl -fsS "${ALAYA_URL}/health" | jq '{status, total_memories, graph_health: .graph_health.status}'

# --- 1. store fact A (the original policy) ----------------------------------
echo "== 1. store A (the original fact) =="
A=$(api /store '{
  "content": "Authentication is required for all API endpoints; every request must carry a bearer token.",
  "tags": ["auth", "api", "policy"],
  "memory_type": "decision",
  "metadata": {"importance": 0.9}
}')
echo "$A" | jq '{content_hash, created, message}'
A_HASH=$(echo "$A" | jq -r '.content_hash')

# --- 2. store fact B (contradicts A) ---------------------------------------
# Wording is deliberately engineered to trip the detector AND stay close to A:
#   * same subject ("Authentication ... API endpoints ... bearer tokens") keeps
#     cosine(A,B) >= 0.7 — the gate for interference detection to run at all;
#   * "is not required" gives the negation-asymmetry signal;
#   * "no longer enforce" gives the temporal-supersession signal.
# Detection is AUTOMATIC on store — the contradiction is computed in-process and
# the CONTRADICTS graph edge is written SYNCHRONOUSLY before this call returns.
# The signals are echoed inline under interference.contradictions (each entry:
# existing_hash, signal_type, confidence, detail).
echo "== 2. store B (contradicts A — server auto-detects on write) =="
B=$(api /store '{
  "content": "Authentication is not required for public API endpoints; we no longer enforce bearer tokens on them.",
  "tags": ["auth", "api", "policy"],
  "memory_type": "decision",
  "metadata": {"importance": 0.95}
}')
echo "$B" | jq '{content_hash, interference}'
B_HASH=$(echo "$B" | jq -r '.content_hash')

# Sanity gate: if B's content drifted < 0.7 cosine from A, or lost the
# negation/temporal cue, NO edge was written and step 3 would silently return
# pairs: []. Catch it here, at the source, rather than blaming the graph.
if [ "$(echo "$B" | jq '.interference.contradictions | length')" -eq 0 ]; then
  echo "demo.sh: WARNING — store B reported no inline contradiction signals." >&2
  echo "         B may have drifted too far from A (< 0.7 cosine) or lost its" >&2
  echo "         negation/temporal cue; step 3 will likely show pairs: []." >&2
fi

# --- 3. surface the contradiction ------------------------------------------
# Reads the CONTRADICTS edges from the graph (FalkorDB). Key is `pairs`, hashes
# are `memory_a_hash`/`memory_b_hash` (NOT the inline shape's `existing_hash`).
# Edge direction is new->existing, so memory_a_hash is typically B and
# memory_b_hash is A — the demo prints both and assumes no ordering.
echo "== 3. list contradictions =="
api /contradictions '{"limit": 20}' \
  | jq '{total, pairs: [.pairs[] | {memory_a_hash, memory_b_hash, confidence, memory_a_superseded, memory_b_superseded}]}'

# --- 4. resolve: B supersedes A --------------------------------------------
# REST field names are old_hash / new_hash (the MCP tool uses old_id / new_id —
# do not confuse them). Sets metadata.superseded_by = B and supersession_reason
# on A, and writes a SUPERSEDES edge B->A. old_hash == new_hash is rejected.
echo "== 4. supersede A with B =="
api /supersede "$(jq -nc --arg old "$A_HASH" --arg new "$B_HASH" \
     '{old_hash: $old, new_hash: $new, reason: "Policy updated: public endpoints no longer require auth"}')" \
  | jq '{success, superseded, superseded_by, reason}'

# --- 5. search reflects the resolution -------------------------------------
# Default search filters out superseded memories (application-layer, on
# metadata.superseded_by), so only B surfaces.
echo "== 5a. default search — superseded A is hidden, only B surfaces =="
api /search '{"query": "is authentication required for the API?", "mode": "hybrid", "page_size": 5}' \
  | jq '{total, results: [.results[] | {content_hash, content, score}]}'

# include_superseded:true brings A back, now carrying superseded_by + reason.
echo "== 5b. include_superseded — A returns, now stamped superseded_by =="
api /search '{"query": "is authentication required for the API?", "mode": "hybrid", "page_size": 5, "include_superseded": true}' \
  | jq --arg a "$A_HASH" \
      '.results[] | select(.content_hash == $a)
                  | {content_hash, superseded_by: .metadata.superseded_by, reason: .metadata.supersession_reason}'

echo "== done =="
