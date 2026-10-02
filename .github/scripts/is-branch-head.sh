#!/usr/bin/env bash
# Decide whether this run may move the branch's moving image tags (main, latest).
#
# Two pushes close together build concurrently, and the older commit's build can
# finish last. So the moving tags follow commit order, not build-finish order:
# only a run whose commit is still the branch head moves them. Any answer other
# than a clean "identical" or "ahead" fails the job, so an unreadable order never
# moves the tags.
#
# Env: SHA (this run's commit), REPO (owner/name), BRANCH, GH_TOKEN.
# Writes current=true|false to $GITHUB_OUTPUT (stdout when unset, for local replay).
set -euo pipefail

: "${SHA:?}" "${REPO:?}" "${BRANCH:?}"
out="${GITHUB_OUTPUT:-/dev/stdout}"

# base...head = SHA...BRANCH: "identical" means SHA is the head,
# "ahead" means the branch has newer commits on top of SHA.
status="$(gh api "repos/${REPO}/compare/${SHA}...${BRANCH}" --jq .status)" || {
  echo "::error::Compare API read failed for ${SHA}...${BRANCH}; not moving tags."
  exit 1
}

case "${status}" in
  identical)
    echo "current=true" >>"${out}"
    ;;
  ahead)
    echo "::notice::${SHA} is no longer the head of ${BRANCH}; leaving the ${BRANCH} and latest tags to the newer commit's run."
    echo "current=false" >>"${out}"
    ;;
  *)
    echo "::error::Cannot establish order of ${SHA} against ${BRANCH} (compare status: '${status}'); not moving tags."
    exit 1
    ;;
esac
