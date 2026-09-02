#!/usr/bin/env bash
# Regenerate static/console.css from style/input.css with the Tailwind v4
# standalone CLI. The generated file is checked in so the Rust build (and the
# container image) never needs node/tailwind at build time.
#
# Usage: scripts/build-css.sh [path-to-tailwindcss-binary]
set -euo pipefail

cd "$(dirname "$0")/.."

TAILWIND="${1:-tailwindcss}"
if ! command -v "$TAILWIND" >/dev/null 2>&1; then
  echo "tailwindcss CLI not found. Install the standalone binary:" >&2
  echo "  https://github.com/tailwindlabs/tailwindcss/releases (tailwindcss-linux-x64)" >&2
  exit 1
fi

"$TAILWIND" --input style/input.css --output static/console.css --minify
echo "wrote static/console.css ($(wc -c < static/console.css) bytes)"
