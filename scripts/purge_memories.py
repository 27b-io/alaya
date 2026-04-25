#!/usr/bin/env python3
"""Purge memories by tag or content prefix.

Scrolls Qdrant, collects matching hashes, deletes through alaya-server.

Usage:
    pip install httpx tenacity

    # Delete all GT Overseer spam:
    python3 scripts/purge_memories.py --tag gas-town

    # Delete by content prefix:
    python3 scripts/purge_memories.py --prefix "GT Overseer run at"

    # Dry run (count only, no deletes):
    python3 scripts/purge_memories.py --tag gas-town --dry-run
"""

import argparse
import asyncio
import time

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

ALAYA_URL = "http://10.43.61.94:3001"
QDRANT_URL = "http://10.43.119.230:6333"
QDRANT_COLLECTION = "memories_arctic1024"
CONCURRENCY = 10

RETRYABLE = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)


async def scroll_matching(
    client: httpx.AsyncClient,
    tag: str | None,
    prefix: str | None,
) -> list[str]:
    """Scroll Qdrant and collect content_hashes matching the filter."""
    hashes = []
    offset = None

    while True:
        body = {"limit": 100, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset

        resp = await client.post(
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll",
            json=body,
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        points = data.get("result", {}).get("points", [])
        if not points:
            break

        for p in points:
            payload = p.get("payload", {})
            content = payload.get("content", "")
            tags = payload.get("tags", [])
            content_hash = payload.get("content_hash")
            if not content_hash:
                continue

            match = False
            if tag and tag in tags:
                match = True
            if prefix and content.startswith(prefix):
                match = True
            if match:
                hashes.append(content_hash)

        next_offset = data.get("result", {}).get("next_page_offset")
        if next_offset is None:
            break
        offset = next_offset

        if len(hashes) % 5000 == 0 and len(hashes) > 0:
            print(f"  found {len(hashes)} matches so far...", flush=True)

    return hashes


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def delete_memory(client: httpx.AsyncClient, content_hash: str) -> bool:
    resp = await client.post(
        f"{ALAYA_URL}/delete",
        json={"content_hash": content_hash},
        timeout=10.0,
    )
    if resp.status_code >= 500 or resp.status_code == 429:
        resp.raise_for_status()
    return resp.status_code == 200


async def worker(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    stats: dict,
):
    while True:
        content_hash = await queue.get()
        if content_hash is None:
            break
        try:
            ok = await delete_memory(client, content_hash)
            if ok:
                stats["deleted"] += 1
            else:
                stats["errors"] += 1
        except Exception as e:
            print(f"    [{content_hash[:8]}] failed: {e}", flush=True)
            stats["errors"] += 1

        stats["processed"] += 1
        n = stats["processed"]
        if n % 500 == 0 or n == stats["total"]:
            elapsed = time.time() - stats["start"]
            rate = n / max(elapsed, 1)
            eta = (stats["total"] - n) / max(rate, 0.01)
            print(
                f"  [{n}/{stats['total']}] "
                f"deleted={stats['deleted']} err={stats['errors']} "
                f"rate={rate:.0f}/s eta={eta/60:.1f}m",
                flush=True,
            )
        queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Purge memories by tag or content prefix")
    parser.add_argument("--tag", help="Delete memories with this tag")
    parser.add_argument("--prefix", help="Delete memories whose content starts with this")
    parser.add_argument("--dry-run", action="store_true", help="Count matches without deleting")
    args = parser.parse_args()

    if not args.tag and not args.prefix:
        parser.error("at least one of --tag or --prefix is required")

    print("Purge memories")
    if args.tag:
        print(f"  tag:    {args.tag}")
    if args.prefix:
        print(f"  prefix: {args.prefix!r}")
    print(f"  dry-run: {args.dry_run}")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Scrolling Qdrant for matches...", flush=True)
        hashes = await scroll_matching(client, args.tag, args.prefix)
        print(f"Found {len(hashes)} matching memories")

        if not hashes:
            print("Nothing to delete!")
            return

        if args.dry_run:
            print(f"\nDry run — would delete {len(hashes)} memories")
            return

        print(f"\nDeleting {len(hashes)} memories...", flush=True)
        stats = {
            "processed": 0,
            "deleted": 0,
            "errors": 0,
            "total": len(hashes),
            "start": time.time(),
        }

        queue: asyncio.Queue = asyncio.Queue()
        for h in hashes:
            queue.put_nowait(h)

        workers = [
            asyncio.create_task(worker(queue, client, stats))
            for _ in range(CONCURRENCY)
        ]

        await queue.join()

        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - stats["start"]
        print()
        print(f"Done in {elapsed/60:.1f}m")
        print(f"  deleted: {stats['deleted']}")
        print(f"  errors:  {stats['errors']}")
        print(f"  rate:    {stats['total']/max(elapsed,1):.0f}/s")


if __name__ == "__main__":
    asyncio.run(main())
