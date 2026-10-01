#!/usr/bin/env python3
"""Migrate memories from Python mcp-memory-service to Alaya.

Scrolls source Qdrant (Python service), re-ingests each memory through
Alaya's /store endpoint for full e2e processing (re-embedding, graph
node creation, salience scoring, provenance).

Idempotent — content hashing means identical content upserts safely.

`metadata.superseded_by` is reserved: /store refuses it. A source memory's
supersession is recreated through /supersede once every memory is stored,
so it lands with its SUPERSEDES edge and reason.

Usage:
    python3 scripts/migrate_from_mcp.py

Environment:
    SOURCE_QDRANT_URL  (default: http://localhost:6333)
    SOURCE_COLLECTION  (default: memories)
    ALAYA_URL          (default: http://localhost:3001)
    ALAYA_API_KEY      (default: empty)
    CONCURRENCY        (default: 5)
    BATCH_SIZE         (default: 50)
    DRY_RUN            (default: false)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass

import httpx
from qdrant_client import QdrantClient

SOURCE_QDRANT_URL = os.environ.get("SOURCE_QDRANT_URL", "http://localhost:6333")
SOURCE_COLLECTION = os.environ.get("SOURCE_COLLECTION", "memories")
ALAYA_URL = os.environ.get("ALAYA_URL", "http://localhost:3001")
ALAYA_API_KEY = os.environ.get("ALAYA_API_KEY", "")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "50"))
DRY_RUN = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")

if CONCURRENCY <= 0:
    sys.exit(f"CONCURRENCY must be > 0, got {CONCURRENCY}")
if BATCH_SIZE <= 0:
    sys.exit(f"BATCH_SIZE must be > 0, got {BATCH_SIZE}")

METADATA_POINT_PREFIX = "00000000-0000-0000-0000-"


@dataclass
class Stats:
    scrolled: int = 0
    stored: int = 0
    existed: int = 0
    skipped_no_content: int = 0
    failed: int = 0
    superseded: int = 0
    supersede_failed: int = 0


@dataclass
class Supersession:
    index: int  # into memories
    source_new_hash: str
    reason: str


async def store_memory(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    memory: dict,
    stats: Stats,
) -> str | None:
    """Store one memory; its Alaya content hash, or None on failure."""
    async with sem:
        try:
            r = await client.post("/store", json=memory)
            body = r.json()
            if r.status_code == 200 and body.get("success"):
                if body.get("created"):
                    stats.stored += 1
                else:
                    stats.existed += 1
                return body["content_hash"]
            else:
                stats.failed += 1
                if stats.failed <= 5:
                    msg = body.get("error", r.text[:100])
                    print(f"  FAIL: {msg}")
        except Exception as e:
            stats.failed += 1
            if stats.failed <= 5:
                print(f"  ERR: {e}")
        return None


async def supersede_memory(
    client: httpx.AsyncClient,
    old_hash: str | None,
    new_hash: str | None,
    s: Supersession,
    stats: Stats,
) -> None:
    def fail(msg: str) -> None:
        stats.supersede_failed += 1
        if stats.supersede_failed <= 5:
            print(f"  SUPERSEDE FAIL: {msg}")

    if old_hash is None:
        fail(f"memory #{s.index} was not stored")
        return
    if new_hash is None:
        fail(f"{old_hash}: superseding memory {s.source_new_hash} was not stored")
        return
    try:
        r = await client.post(
            "/supersede",
            json={"old_hash": old_hash, "new_hash": new_hash, "reason": s.reason},
        )
        body = r.json()
        if r.status_code == 200 and body.get("success"):
            stats.superseded += 1
        else:
            fail(f"{old_hash}: {body.get('error', r.text[:100])}")
    except Exception as e:
        fail(f"{old_hash}: {e}")


async def run() -> None:
    print(f"Source:      {SOURCE_QDRANT_URL} / {SOURCE_COLLECTION}")
    print(f"Target:      {ALAYA_URL}")
    print(f"Concurrency: {CONCURRENCY}, Batch: {BATCH_SIZE}")
    if DRY_RUN:
        print("DRY RUN — no writes")
    print()

    # ── Phase 1: Scroll source ──────────────────────────────────────
    print("Phase 1: Scrolling source Qdrant...")
    qclient = QdrantClient(url=SOURCE_QDRANT_URL, timeout=30)
    stats = Stats()
    memories: list[dict] = []
    source_hashes: list[str | None] = []  # aligned with memories
    supersessions: list[Supersession] = []
    next_offset = None

    while True:
        points, next_offset = qclient.scroll(
            collection_name=SOURCE_COLLECTION,
            limit=BATCH_SIZE,
            with_payload=True,
            with_vectors=False,
            offset=next_offset,
        )
        if not points:
            break

        for p in points:
            pid = str(p.id)
            if pid.startswith(METADATA_POINT_PREFIX):
                continue

            payload = p.payload or {}
            content = payload.get("content")
            if not content:
                stats.skipped_no_content += 1
                continue

            entry = {"content": content}

            tags = payload.get("tags")
            if tags:
                entry["tags"] = tags

            memory_type = payload.get("memory_type")
            if memory_type:
                entry["memory_type"] = memory_type

            # Preserve metadata, emotional_valence, provenance.
            # Use `is not None` so legitimate falsy values (e.g. epoch 0,
            # neutral 0.0 valence) survive the copy.
            metadata = dict(payload.get("metadata") or {})
            superseded_by = metadata.pop("superseded_by", None)
            if superseded_by:
                supersessions.append(
                    Supersession(
                        index=len(memories),
                        source_new_hash=str(superseded_by),
                        reason=payload.get("supersession_reason")
                        or "migrated from mcp-memory-service",
                    )
                )
            if payload.get("emotional_valence") is not None:
                metadata["emotional_valence"] = payload["emotional_valence"]
            if payload.get("created_at") is not None:
                metadata["original_created_at"] = payload["created_at"]
            if payload.get("updated_at") is not None:
                metadata["original_updated_at"] = payload["updated_at"]
            if metadata:
                entry["metadata"] = metadata

            memories.append(entry)
            source_hashes.append(payload.get("content_hash"))

        if next_offset is None:
            break

    stats.scrolled = len(memories)
    print(
        f"  Found {stats.scrolled} memories ({stats.skipped_no_content} skipped — no content)"
    )
    print(f"  {len(supersessions)} supersessions to recreate")

    if DRY_RUN:
        for m in memories[:5]:
            tags = m.get("tags", [])
            print(f"  [{','.join(tags[:3])}] {m['content'][:80]}...")
        if len(memories) > 5:
            print(f"  ... and {len(memories) - 5} more")
        return

    # ── Phase 2: Store to Alaya ─────────────────────────────────────
    print(f"\nPhase 2: Storing {stats.scrolled} memories to Alaya...")
    t0 = time.monotonic()
    sem = asyncio.Semaphore(CONCURRENCY)

    headers = {}
    if ALAYA_API_KEY:
        headers["Authorization"] = f"Bearer {ALAYA_API_KEY}"

    async with httpx.AsyncClient(
        base_url=ALAYA_URL,
        headers=headers,
        timeout=60.0,
    ) as client:
        stored_hashes: list[str | None] = []  # aligned with memories
        for batch_start in range(0, len(memories), BATCH_SIZE):
            batch = memories[batch_start : batch_start + BATCH_SIZE]
            tasks = [store_memory(client, sem, m, stats) for m in batch]
            stored_hashes.extend(await asyncio.gather(*tasks))

            done = min(batch_start + BATCH_SIZE, len(memories))
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  {done}/{stats.scrolled} ({rate:.1f}/s)"
                f" — stored={stats.stored} existed={stats.existed} failed={stats.failed}"
            )

        # ── Phase 3: Recreate supersessions ─────────────────────────
        # Runs after every store, so each superseding memory exists. Alaya
        # hashes raw content and the source may not, so a source hash is
        # translated through the memory that carried it.
        print(f"\nPhase 3: Recreating {len(supersessions)} supersessions...")
        to_alaya: dict[str, str] = {}
        for src, dst in zip(source_hashes, stored_hashes, strict=True):
            if dst is not None:
                to_alaya[dst] = dst
                if src:
                    to_alaya[src] = dst
        for s in supersessions:
            await supersede_memory(
                client,
                stored_hashes[s.index],
                to_alaya.get(s.source_new_hash),
                s,
                stats,
            )
        print(f"  superseded={stats.superseded} failed={stats.supersede_failed}")

    total_elapsed = time.monotonic() - t0

    # ── Phase 4: Verify ─────────────────────────────────────────────
    print("\nPhase 4: Verifying...")
    try:
        r = httpx.get(f"{ALAYA_URL}/health/detail", headers=headers, timeout=10)
        health = r.json()
        target_count = health.get("total_memories", "?")
        status = health.get("status", "?")
        print(f"  Alaya health: {status}, total memories: {target_count}")
    except Exception as e:
        print(f"  Health check failed: {e}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"Migration complete in {total_elapsed:.1f}s")
    print(f"  Source memories:  {stats.scrolled}")
    print(f"  Stored:           {stats.stored}")
    print(f"  Already existed:  {stats.existed}")
    print(f"  Failed:           {stats.failed}")
    print(f"  Skipped (empty):  {stats.skipped_no_content}")
    print(f"  Superseded:       {stats.superseded}")
    print(f"  Supersede failed: {stats.supersede_failed}")

    if stats.failed or stats.supersede_failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
