#!/usr/bin/env bash
# Screenshot every console page at phone, tablet and desktop width (390, 768
# and 1280px), light and dark, and fail if any page scrolls sideways: the
# visual evidence a UI change attaches to its review.
#
# The pages are rendered by an ignored test against the same fake upstreams
# the test suite uses, so this needs no running console, no login and no
# network. scripts/snapshot.py then screenshots them with headless Chrome.
#
# Output, in target/snapshots/ at the workspace root:
#   <page>.<theme>.html            the rendered pages
#   png/<page>.<theme>.<width>.png one full-page screenshot each
#   sheets/<page>.png              390 and 1280, light and dark, side by side
#
# Usage: scripts/snapshot.sh
# Needs: cargo, python3 and Chrome (set CHROME if it is not on PATH as
# google-chrome or chromium).
set -euo pipefail

cd "$(dirname "$0")/.."
out="$(cd ../.. && pwd)/target/snapshots"
rm -rf "${out:?}"
mkdir -p "$out"

SNAPSHOT_DIR="$out" cargo test -p ops-console --bin ops-console -- \
  --ignored --exact tests::snapshot::write_snapshots
python3 scripts/snapshot.py "$out"
