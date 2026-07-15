# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""One-shot migration for issue #54: supersessions written as a LITERAL flat payload
key named "metadata.superseded_by" (qdrant set-payload does not nest dotted map keys).

Per affected point, two TARGETED operations — no full-payload writes, so concurrent
writers to unrelated fields are never clobbered:

  1. key-scoped set-payload writes the nested metadata.superseded_by (an existing
     nested value wins; the flat key is only the fallback)
  2. delete_payload with a QUOTED path segment ("metadata.superseded_by") removes the
     literal dotted key — quoting is qdrant's escape for literal dots in JsonPath
     (verified empirically; the unquoted form deletes the NESTED value instead)

    kubectl -n mcp port-forward svc/qdrant 6333:6333 &
    uv run scripts/migrate_flat_superseded.py            # dry run: report only
    uv run scripts/migrate_flat_superseded.py --write    # backup + migrate + verify
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import httpx

DEFAULT_COLLECTION = "memories_arctic1024"
FLAT_KEY = "metadata.superseded_by"
COLLECTION = (
    DEFAULT_COLLECTION  # overridden by --collection (also enables integration tests)
)


def scroll_all(c: httpx.Client) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    offset: Any = None
    while True:
        body: dict[str, Any] = {
            "limit": 256,
            "with_payload": True,
            "with_vector": False,
        }
        if offset is not None:
            body["offset"] = offset
        r = c.post(f"/collections/{COLLECTION}/points/scroll", json=body)
        r.raise_for_status()
        res = r.json()["result"]
        points.extend(res["points"])
        offset = res.get("next_page_offset")
        if offset is None:
            return points


def fetch_payload(c: httpx.Client, point_id: int | str) -> dict[str, Any] | None:
    r = c.post(
        f"/collections/{COLLECTION}/points",
        json={"ids": [point_id], "with_payload": True},
    )
    r.raise_for_status()
    points = r.json()["result"]
    return points[0]["payload"] if points else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()
    global COLLECTION  # noqa: PLW0603 — one-shot script, simplest way to thread the override
    COLLECTION = args.collection

    with httpx.Client(base_url=args.qdrant_url, timeout=60) as c:
        points = scroll_all(c)
        # presence, not truthiness: a falsey flat value still needs cleaning
        affected = [p for p in points if FLAT_KEY in p["payload"]]
        print(f"scanned {len(points)} points: {len(affected)} flat-key casualties")
        if not affected:
            print("nothing to migrate")
            return 0
        if not args.write:
            for p in affected[:10]:
                print(
                    f"  would fix {p['payload'].get('content_hash', '?')[:12]} → {str(p['payload'][FLAT_KEY])[:12]}"
                )
            print("dry run — re-run with --write to apply")
            return 0

        backup = f"flat_superseded_backup_{time.strftime('%Y%m%d%H%M%S')}.json"
        with open(backup, "w") as f:
            json.dump(affected, f)
        print(f"backed up {len(affected)} full payloads → {backup}")

        expected: dict[int | str, str] = {}
        for p in affected:
            # re-fetch right before writing: decide the winning value from live state
            fresh = fetch_payload(c, p["id"])
            if fresh is None or FLAT_KEY not in fresh:
                continue  # vanished or already fixed since the scan
            want = (fresh.get("metadata") or {}).get("superseded_by") or fresh[FLAT_KEY]
            # targeted nested write — touches ONLY metadata.superseded_by
            c.post(
                f"/collections/{COLLECTION}/points/payload?wait=true",
                json={
                    "payload": {"superseded_by": want},
                    "key": "metadata",
                    "points": [p["id"]],
                },
            ).raise_for_status()
            # targeted flat-key delete — quoted segment = literal dotted key
            c.post(
                f"/collections/{COLLECTION}/points/payload/delete?wait=true",
                json={"keys": [f'"{FLAT_KEY}"'], "points": [p["id"]]},
            ).raise_for_status()
            expected[p["id"]] = want
        print(f"migrated {len(expected)} points")

        # verify per point (aggregate counts race with concurrent supersedes): every
        # migrated id must have its flat key gone (presence check) and the exact
        # nested value written; plus one full sweep for surviving flat keys anywhere
        bad = 0
        for pid, want in expected.items():
            now = fetch_payload(c, pid) or {}
            if (
                FLAT_KEY in now
                or (now.get("metadata") or {}).get("superseded_by") != want
            ):
                print(f"  VERIFY FAILED for point {pid}")
                bad += 1
        flat2 = sum(1 for p in scroll_all(c) if FLAT_KEY in p["payload"])
        print(f"verify: per-point failures={bad}, flat keys remaining={flat2}")
        return 0 if bad == 0 and flat2 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
