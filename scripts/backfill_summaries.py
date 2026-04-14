#!/usr/bin/env python3
"""Backfill all memory summaries via Haiku.

Scrolls every memory in Qdrant directly (cursor-based), generates a
one-line summary via anthropic-lb (Haiku), and PATCHes it back through
alaya-server. Resumable via a progress file.

Usage:
    pip install httpx tenacity

    # Full backfill (summary + embedding):
    python3 scripts/backfill_summaries.py

    # Embeddings only (skip Haiku, use existing summaries):
    python3 scripts/backfill_summaries.py --embeddings-only

    # Override endpoints:
    ALAYA_URL=http://10.43.61.94:3001 \
    python3 scripts/backfill_summaries.py
"""

import argparse
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

ALAYA_URL = os.environ.get("ALAYA_URL", "http://10.43.61.94:3001")
SUMMARY_URL = os.environ.get("SUMMARY_URL", "http://192.168.0.10:8082")
EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://10.43.242.167")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://10.43.119.230:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "memories_arctic1024")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "claude-haiku-4-5-20251001")
SUMMARY_MODEL_COMPLEX = os.environ.get("SUMMARY_MODEL_COMPLEX", "claude-sonnet-4-6")
COMPLEXITY_THRESHOLD = int(os.environ.get("COMPLEXITY_THRESHOLD", "200"))  # chars
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "Snowflake/snowflake-arctic-embed-l-v2.0")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))
THROTTLE_DELAY = float(os.environ.get("THROTTLE_DELAY", "0.2"))
PROGRESS_FILE = Path(os.environ.get("PROGRESS_FILE", "backfill_summaries_progress.json"))

SYSTEM_PROMPT = (
    "Summarize the following in one concise sentence of approximately 50 tokens. "
    "Return only the summary, no preamble."
)
MAX_CONTENT_CHARS = 4000

# Retryable exception types (network errors + mid-response resets)
RETRYABLE = (
    httpx.HTTPStatusError,
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
)

# ─── Progress tracking ──────────────────────────────────────────────────────


def load_progress() -> set[str]:
    if not PROGRESS_FILE.exists():
        return set()
    try:
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("completed", []))
    except (json.JSONDecodeError, ValueError):
        print("  warning: corrupt progress file, starting fresh", flush=True)
        return set()


def save_progress(completed: set[str], total: int, errors: int):
    """Atomic write: tmp file then rename."""
    tmp = PROGRESS_FILE.with_suffix(".tmp")
    tmp.write_text(
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
    tmp.rename(PROGRESS_FILE)


# ─── Qdrant scroll (cursor-based, with retry) ──────────────────────────────


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

        resp = await _scroll_request(client, body)
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


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _scroll_request(client: httpx.AsyncClient, body: dict) -> httpx.Response:
    resp = await client.post(
        f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points/scroll",
        json=body,
        timeout=30.0,
    )
    # 429 = rate limited, retry via tenacity
    # Other 4xx = client error, fail fast
    if 400 <= resp.status_code < 500 and resp.status_code != 429:
        raise RuntimeError(f"Qdrant scroll {resp.status_code}: {resp.text[:200]}")
    resp.raise_for_status()
    return resp


# ─── API calls with retries ────────────────────────────────────────────────


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def generate_summary(client: httpx.AsyncClient, content: str) -> str | None:
    """Call anthropic-lb to generate a summary. Promotes to Sonnet for complex content."""
    truncated = content[:MAX_CONTENT_CHARS]
    model = SUMMARY_MODEL_COMPLEX if len(content) > COMPLEXITY_THRESHOLD else SUMMARY_MODEL

    resp = await client.post(
        f"{SUMMARY_URL}/v1/messages",
        json={
            "model": model,
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

    # Retry all 5xx + 429
    if resp.status_code >= 500 or resp.status_code == 429:
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
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_embedding(client: httpx.AsyncClient, text: str) -> list[float] | None:
    """Generate embedding via TEI (OpenAI-compatible endpoint)."""
    resp = await client.post(
        f"{EMBEDDING_URL}/v1/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "input": [f"search_document: {text}"],
            "encoding_format": "float",
        },
        timeout=30.0,
    )
    if resp.status_code >= 500 or resp.status_code == 429:
        resp.raise_for_status()
    if not resp.is_success:
        return None
    data = resp.json()
    items = data.get("data", [])
    if items:
        return items[0].get("embedding")
    return None


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    wait=wait_exponential(multiplier=0.5, min=1, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def patch_summary(
    client: httpx.AsyncClient,
    content_hash: str,
    summary: str,
    summary_embedding: list[float] | None = None,
) -> bool:
    """PATCH the summary + embedding back to alaya-server."""
    payload: dict = {"summary": summary}
    if summary_embedding is not None:
        payload["summary_embedding"] = summary_embedding
    resp = await client.patch(
        f"{ALAYA_URL}/memories/{content_hash}",
        json=payload,
        timeout=10.0,
    )
    if resp.status_code >= 500 or resp.status_code == 429:
        resp.raise_for_status()
    return resp.status_code == 200


# ─── Worker ─────────────────────────────────────────────────────────────────


async def process_memory(
    client: httpx.AsyncClient,
    memory: dict,
    completed: set[str],
    stats: dict,
    embeddings_only: bool = False,
):
    content_hash = memory["content_hash"]
    h = content_hash[:8]

    await asyncio.sleep(THROTTLE_DELAY)

    try:
        if embeddings_only:
            summary = memory.get("summary")
            if not summary:
                print(f"    [{h}] no existing summary, skipping", flush=True)
                stats["errors"] += 1
                stats["processed"] += 1
                return
        else:
            summary = await generate_summary(client, memory["content"])
            if not summary:
                print(f"    [{h}] empty summary", flush=True)
                stats["errors"] += 1
                stats["processed"] += 1
                return

        embedding = await generate_embedding(client, summary)
        if embedding is None:
            print(f"    [{h}] embedding failed, skipping", flush=True)
            stats["errors"] += 1
            stats["processed"] += 1
            return
        ok = await patch_summary(client, content_hash, summary, embedding)
        if ok:
            completed.add(content_hash)
            stats["success"] += 1
        else:
            print(f"    [{h}] patch failed", flush=True)
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
        save_progress(completed, stats["total_memories"], stats["errors"])


async def worker(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    completed: set[str],
    stats: dict,
    embeddings_only: bool = False,
):
    """Pull memories from queue and process until sentinel."""
    while True:
        memory = await queue.get()
        if memory is None:
            break
        await process_memory(client, memory, completed, stats, embeddings_only)
        queue.task_done()


# ─── Main ───────────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser(description="Backfill memory summaries via Haiku")
    parser.add_argument(
        "--embeddings-only",
        action="store_true",
        help="Skip Haiku — only generate embeddings for existing summaries",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore progress file — re-summarise and re-embed everything",
    )
    args = parser.parse_args()

    mode = "embeddings only" if args.embeddings_only else "full (summary + embedding)"
    print(f"Backfill summaries ({mode})")
    print(f"  qdrant:      {QDRANT_URL}/{QDRANT_COLLECTION}")
    print(f"  alaya:       {ALAYA_URL}")
    if not args.embeddings_only:
        print(f"  summary:     {SUMMARY_URL}")
        print(f"  model:       {SUMMARY_MODEL} (< {COMPLEXITY_THRESHOLD} chars)")
        print(f"  model:       {SUMMARY_MODEL_COMPLEX} (>= {COMPLEXITY_THRESHOLD} chars)")
    print(f"  embedding:   {EMBEDDING_URL}")
    print(f"  concurrency: {CONCURRENCY}")
    print(f"  throttle:    {THROTTLE_DELAY}s")
    print()

    completed = set() if args.force else load_progress()
    if args.force:
        print("Force mode — ignoring progress file")
    elif completed:
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
            "total_memories": len(memories),
            "start": time.time(),
        }

        # Queue-based worker pool — constant memory regardless of batch size
        queue: asyncio.Queue = asyncio.Queue()
        for m in to_process:
            queue.put_nowait(m)

        workers = [
            asyncio.create_task(worker(queue, client, completed, stats, args.embeddings_only))
            for _ in range(CONCURRENCY)
        ]

        # Wait for all items to be processed
        await queue.join()

        # Send sentinel to stop workers
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)

        elapsed = time.time() - stats["start"]
        save_progress(completed, stats["total_memories"], stats["errors"])

        print()
        print(f"Done in {elapsed / 60:.1f}m")
        print(f"  processed: {stats['processed']}")
        print(f"  success:   {stats['success']}")
        print(f"  errors:    {stats['errors']}")
        print(f"  rate:      {stats['processed'] / max(elapsed, 1):.1f}/s")


if __name__ == "__main__":
    asyncio.run(main())
