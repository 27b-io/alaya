#!/usr/bin/env python3
"""Backfill all memory summaries via Haiku.

Scrolls every memory in Qdrant directly (cursor-based), generates a
one-line summary via anthropic-lb (Haiku), and PATCHes it back through
alaya-server. Resumable via a progress file.

Usage:
    pip install httpx tenacity

    # Port-forward:
    kubectl -n mcp port-forward svc/alaya-server 3099:3001 &
    kubectl -n mcp port-forward svc/anthropic-lb 8082:8082 &
    kubectl -n mcp port-forward svc/qdrant 6333:6333 &

    ALAYA_URL=http://localhost:3099 \
    SUMMARY_URL=http://localhost:8082 \
    QDRANT_URL=http://localhost:6333 \
    python3 scripts/backfill_summaries.py
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ─── Config ─────────────────────────────────────────────────────────────────

ALAYA_URL = os.environ.get("ALAYA_URL", "http://localhost:3099")
SUMMARY_URL = os.environ.get("SUMMARY_URL", "http://localhost:8082")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "memories_arctic1024")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "claude-haiku-4-5-20251001")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))
THROTTLE_DELAY = float(os.environ.get("THROTTLE_DELAY", "0.2"))
PROGRESS_FILE = Path(os.environ.get("PROGRESS_FILE", "backfill_summaries_progress.json"))

SYSTEM_PROMPT = (
    "Summarize the following in one concise sentence of approximately 50 tokens. "
    "Return only the summary, no preamble."
)
MAX_CONTENT_CHARS = 4000

# ─── Progress tracking ──────────────────────────────────────────────────────


def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("completed", []))
    return set()


def save_progress(completed: set[str], total: int, errors: int):
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "completed": sorted(completed),
                "total_processed": len(completed),
                "total_memories": total,
                "errors": errors,
                "updated_at": time.time(),
            },
            indent=2,
        )
    )


# ─── Qdrant scroll (cursor-based, no re-scan) ──────────────────────────────


async def scroll_all_qdrant(client: httpx.AsyncClient) -> list[dict]:
    """Scroll all memories from Qdrant directly using cursor pagination."""
    memories = []
    offset = None

    while True:
        body = {
            "limit": 100,
            "with_payload": True,
            "with_vector": False,
        }
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
            content = payload.get("content")
            content_hash = payload.get("content_hash")
            if content and content_hash:
                memories.append({
                    "content": content,
                    "content_hash": content_hash,
                    "summary": payload.get("summary"),
                })

        next_offset = data.get("result", {}).get("next_page_offset")
        if next_offset is None:
            break
        offset = next_offset

        if len(memories) % 5000 == 0:
            print(f"  scrolled {len(memories)} memories...", flush=True)

    return memories


# ─── API calls with retries ────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def generate_summary(client: httpx.AsyncClient, content: str) -> str | None:
    """Call anthropic-lb to generate a summary."""
    truncated = content[:MAX_CONTENT_CHARS]

    resp = await client.post(
        f"{SUMMARY_URL}/v1/messages",
        json={
            "model": SUMMARY_MODEL,
            "max_tokens": 100,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": truncated}],
        },
        headers={
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        timeout=30.0,
    )

    if resp.status_code in (429, 529):
        resp.raise_for_status()

    if not resp.is_success:
        return None

    data = resp.json()
    blocks = data.get("content", [])
    if blocks:
        text = blocks[0].get("text", "").strip()
        return text if text else None
    return None


@retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout)),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def patch_summary(
    client: httpx.AsyncClient, content_hash: str, summary: str
) -> bool:
    """PATCH the summary back to alaya-server."""
    resp = await client.patch(
        f"{ALAYA_URL}/memories/{content_hash}",
        json={"summary": summary},
        timeout=10.0,
    )
    return resp.status_code == 200


# ─── Worker with throttling ────────────────────────────────────────────────


async def process_memory(
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    memory: dict,
    completed: set[str],
    stats: dict,
):
    content_hash = memory["content_hash"]
    h = content_hash[:8]

    async with sem:
        await asyncio.sleep(THROTTLE_DELAY)

        try:
            summary = await generate_summary(client, memory["content"])
            if summary:
                ok = await patch_summary(client, content_hash, summary)
                if ok:
                    completed.add(content_hash)
                    stats["success"] += 1
                else:
                    print(f"    [{h}] patch failed", flush=True)
                    stats["errors"] += 1
            else:
                print(f"    [{h}] empty summary", flush=True)
                stats["errors"] += 1
        except Exception as e:
            print(f"    [{h}] failed after retries: {e}", flush=True)
            stats["errors"] += 1

        stats["processed"] += 1
        n = stats["processed"]
        if n % 100 == 0 or n == stats["total"]:
            elapsed = time.time() - stats["start"]
            rate = n / max(elapsed, 1)
            eta = (stats["total"] - n) / max(rate, 0.01)
            print(
                f"  [{n}/{stats['total']}] "
                f"ok={stats['success']} err={stats['errors']} "
                f"rate={rate:.1f}/s eta={eta / 60:.0f}m",
                flush=True,
            )
            save_progress(completed, stats["total"], stats["errors"])


# ─── Main ───────────────────────────────────────────────────────────────────


async def main():
    print("Backfill summaries (full regeneration)")
    print(f"  qdrant:     {QDRANT_URL}/{QDRANT_COLLECTION}")
    print(f"  alaya:      {ALAYA_URL}")
    print(f"  summary:    {SUMMARY_URL}")
    print(f"  model:      {SUMMARY_MODEL}")
    print(f"  concurrency: {CONCURRENCY}")
    print(f"  throttle:   {THROTTLE_DELAY}s")
    print()

    completed = load_progress()
    if completed:
        print(f"Resuming — {len(completed)} already done")

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Scrolling all memories from Qdrant...", flush=True)
        memories = await scroll_all_qdrant(client)
        print(f"Found {len(memories)} total memories")

        to_process = [m for m in memories if m["content_hash"] not in completed]
        print(f"  {len(to_process)} need processing ({len(completed)} already done)")
        print()

        if not to_process:
            print("Nothing to do!")
            return

        stats = {
            "processed": 0,
            "success": 0,
            "errors": 0,
            "total": len(to_process),
            "start": time.time(),
        }
        sem = asyncio.Semaphore(CONCURRENCY)

        tasks = [
            process_memory(sem, client, m, completed, stats) for m in to_process
        ]
        await asyncio.gather(*tasks)

        elapsed = time.time() - stats["start"]
        save_progress(completed, len(memories), stats["errors"])

        print()
        print(f"Done in {elapsed / 60:.1f}m")
        print(f"  processed: {stats['processed']}")
        print(f"  success:   {stats['success']}")
        print(f"  errors:    {stats['errors']}")
        print(f"  rate:      {stats['processed'] / max(elapsed, 1):.1f}/s")


if __name__ == "__main__":
    asyncio.run(main())
