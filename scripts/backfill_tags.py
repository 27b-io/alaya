#!/usr/bin/env python3
"""Fix mangled tags caused by bug #17.

When tags were sent as a stringified JSON array (e.g. '["a","b"]'),
the deserializer comma-split the string, producing garbage like:
  ['["a"', '"b"', '"c"]']

This script scrolls all memories, detects the pattern, reconstructs
the correct tags, and PATCHes them back via alaya-server.

Usage:
    pip install httpx

    # Dry run (default — report only):
    python3 scripts/backfill_tags.py

    # Apply fixes:
    python3 scripts/backfill_tags.py --apply

    # Override endpoint:
    ALAYA_URL=http://10.43.61.94:3001 python3 scripts/backfill_tags.py --apply
"""

import argparse
import asyncio
import json
import os

import httpx

ALAYA_URL = os.environ.get("ALAYA_URL", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "memories_arctic1024")

if not ALAYA_URL or not QDRANT_URL:
    print(
        "Error: ALAYA_URL and QDRANT_URL must be set.\n\n"
        "  ALAYA_URL=http://<alaya-server>:3001 \\\n"
        "  QDRANT_URL=http://<qdrant>:6333 \\\n"
        "    python3 scripts/backfill_tags.py [--apply]\n"
    )
    raise SystemExit(1)
PAGE_SIZE = 100


def is_mangled(tags: list[str]) -> bool:
    """Detect the #17 mangling pattern: tags contain brackets or embedded quotes."""
    if not tags:
        return False
    return any(t.startswith("[") or t.startswith('"') or t.endswith("]") for t in tags)


def reconstruct_tags(mangled: list[str]) -> list[str] | None:
    """Reconstruct original tags from mangled comma-split fragments.

    Join fragments back into a single string and parse as JSON array.
    """
    joined = ",".join(mangled)
    # Strip outer brackets if present (they always are in the #17 pattern)
    try:
        parsed = json.loads(joined)
        if isinstance(parsed, list) and all(isinstance(t, str) for t in parsed):
            # Order-preserving dedup matching service's deserialize_tags
            seen: set[str] = set()
            result = []
            for t in parsed:
                t = t.strip()
                if t and t not in seen:
                    seen.add(t)
                    result.append(t)
            return result or None
    except json.JSONDecodeError:
        pass
    return None


async def scroll_all(client: httpx.AsyncClient) -> list[dict]:
    """Scroll all points from Qdrant."""
    memories = []
    offset = None
    while True:
        body = {"limit": PAGE_SIZE, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset

        resp = await client.post(
            f"{QDRANT_URL}/collections/{COLLECTION}/points/scroll",
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()["result"]

        for point in data.get("points", []):
            payload = point.get("payload", {})
            memories.append(
                {
                    "id": point["id"],
                    "content_hash": payload.get("content_hash", ""),
                    "tags": payload.get("tags", []),
                    "content_preview": payload.get("content", "")[:80],
                }
            )

        offset = data.get("next_page_offset")
        if not offset or not data.get("points"):
            break

    return memories


_AUTH_HEADERS: dict[str, str] = {}
_token = os.environ.get("ALAYA_API_KEY", "")
if _token:
    _AUTH_HEADERS["Authorization"] = f"Bearer {_token}"


async def patch_tags(
    client: httpx.AsyncClient, content_hash: str, tags: list[str]
) -> bool:
    """PATCH memory tags via alaya-server."""
    resp = await client.patch(
        f"{ALAYA_URL}/memories/{content_hash}",
        json={"tags": tags},
        headers=_AUTH_HEADERS,
    )
    return resp.status_code == 200


async def main(apply: bool) -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        print(f"Scrolling all memories from {QDRANT_URL}...")
        memories = await scroll_all(client)
        print(f"Total memories: {len(memories)}")

        affected = []
        for m in memories:
            if is_mangled(m["tags"]):
                fixed = reconstruct_tags(m["tags"])
                if fixed:
                    affected.append(
                        {
                            "content_hash": m["content_hash"],
                            "mangled": m["tags"],
                            "fixed": fixed,
                            "preview": m["content_preview"],
                        }
                    )
                else:
                    print(
                        f"  WARN: could not reconstruct tags for {m['content_hash'][:8]}: {m['tags']}"
                    )

        print(f"\nAffected memories: {len(affected)}")

        if not affected:
            print("Nothing to fix.")
            return

        for item in affected:
            h = item["content_hash"][:8]
            print(f"\n  {h}: {item['mangled']}")
            print(f"     -> {item['fixed']}")
            print(f"     preview: {item['preview']}")

        if not apply:
            print(f"\nDry run complete. Run with --apply to fix {len(affected)} memories.")
            return

        print(f"\nApplying fixes to {len(affected)} memories...")
        ok = 0
        fail = 0
        for item in affected:
            success = await patch_tags(client, item["content_hash"], item["fixed"])
            if success:
                ok += 1
                print(f"  OK  {item['content_hash'][:8]}")
            else:
                fail += 1
                print(f"  FAIL {item['content_hash'][:8]}")

        print(f"\nDone: {ok} fixed, {fail} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix mangled tags (bug #17)")
    parser.add_argument(
        "--apply", action="store_true", help="Apply fixes (default: dry run)"
    )
    args = parser.parse_args()
    asyncio.run(main(args.apply))
