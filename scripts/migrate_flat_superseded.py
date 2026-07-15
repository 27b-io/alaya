# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""One-shot migration for issue #54: supersessions written as a LITERAL flat payload
key named "metadata.superseded_by" (qdrant set-payload does not nest dotted map keys).

For every affected point: back up the full payload, then overwrite the payload with a
corrected copy — flat key removed, value merged into the nested metadata object.
PUT /points/payload replaces the whole payload, which is the only way to delete a
flat key containing dots (delete_payload interprets dots as nesting too).

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

COLLECTION = "memories_arctic1024"
FLAT_KEY = "metadata.superseded_by"


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


def corrected_payload(payload: dict[str, Any]) -> dict[str, Any]:
    fixed = {k: v for k, v in payload.items() if k != FLAT_KEY}
    meta = dict(fixed.get("metadata") or {})
    # an existing nested value wins; the flat key is only the fallback — never
    # clobber already-correct data (no such overlap existed in the 2026-07-15 run)
    meta.setdefault("superseded_by", payload[FLAT_KEY])
    fixed["metadata"] = meta
    return fixed


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
    ap.add_argument("--write", action="store_true", help="apply (default: dry run)")
    args = ap.parse_args()

    with httpx.Client(base_url=args.qdrant_url, timeout=60) as c:
        points = scroll_all(c)
        affected = [p for p in points if p["payload"].get(FLAT_KEY)]
        nested = sum(
            1
            for p in points
            if (p["payload"].get("metadata") or {}).get("superseded_by")
        )
        print(
            f"scanned {len(points)} points: {len(affected)} flat-key casualties, {nested} already nested"
        )
        if not affected:
            print("nothing to migrate")
            return 0
        if not args.write:
            for p in affected[:10]:
                print(
                    f"  would fix {p['payload'].get('content_hash', '?')[:12]} → {p['payload'][FLAT_KEY][:12]}"
                )
            print("dry run — re-run with --write to apply")
            return 0

        backup = f"flat_superseded_backup_{time.strftime('%Y%m%d%H%M%S')}.json"
        with open(backup, "w") as f:
            json.dump(affected, f)
        print(f"backed up {len(affected)} full payloads → {backup}")

        # expected nested value per migrated point id — verified individually below
        expected: dict[int | str, str] = {}
        for p in affected:
            # re-fetch immediately before the overwrite: the full-payload PUT (needed to
            # remove the dotted key) would otherwise clobber any field a live writer
            # touched (e.g. access_count) between the scan and this write
            fresh = fetch_payload(c, p["id"])
            if fresh is None or not fresh.get(FLAT_KEY):
                continue  # vanished or already fixed since the scan
            fixed = corrected_payload(fresh)
            r = c.put(
                f"/collections/{COLLECTION}/points/payload?wait=true",
                json={"payload": fixed, "points": [p["id"]]},
            )
            r.raise_for_status()
            expected[p["id"]] = fixed["metadata"]["superseded_by"]
        print(f"migrated {len(expected)} points")

        # verify per point (aggregate counts race with concurrent supersedes): every
        # migrated id must have its flat key gone and the exact nested value written;
        # plus one full sweep asserting no flat key survives anywhere
        bad = 0
        for pid, want in expected.items():
            now = fetch_payload(c, pid) or {}
            if (
                now.get(FLAT_KEY)
                or (now.get("metadata") or {}).get("superseded_by") != want
            ):
                print(f"  VERIFY FAILED for point {pid}")
                bad += 1
        flat2 = sum(1 for p in scroll_all(c) if p["payload"].get(FLAT_KEY))
        print(f"verify: per-point failures={bad}, flat keys remaining={flat2}")
        return 0 if bad == 0 and flat2 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
