#!/usr/bin/env python3
"""Migrate memories from Python mcp-memory-service to Alaya.

Scrolls source Qdrant (Python service), re-ingests each memory through
Alaya's /store endpoint for full e2e processing (re-embedding, graph
node creation, salience scoring, provenance).

Idempotent — content hashing means identical content upserts safely.

`metadata.superseded_by` is reserved: /store refuses it. A source memory's
supersession is recreated through /supersede once every memory is stored,
so it lands with its SUPERSEDES edge and reason.

Alaya keys a memory by its raw content; the source keys it by normalised
content plus metadata, so several source memories can share one Alaya
record. Every supersession is resolved to Alaya record keys before any
write. A superseded memory whose content a live one also holds is not
stored, and its supersession is skipped: the content is live in the source.
Superseded copies of one content that name different survivors cannot share
one record; they are not stored, are reported, and fail the run.

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
import hashlib
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
    supersede_skipped: int = 0
    supersede_failed: int = 0
    conflicted: int = 0


@dataclass
class Supersession:
    index: int  # into memories
    source_new_hash: str
    reason: str


def alaya_hash(content: str) -> str:
    """Alaya's record key: SHA-256 of the raw content, not normalised."""
    return hashlib.sha256(content.encode()).hexdigest()


async def store_memory(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    memory: dict,
    key: str,
    stats: Stats,
) -> bool:
    """Store one memory under its expected Alaya key; whether it landed there."""

    def fail(msg: str) -> bool:
        stats.failed += 1
        if stats.failed <= 5:
            print(f"  FAIL: {msg}")
        return False

    async with sem:
        try:
            r = await client.post("/store", json=memory)
            body = r.json()
        except (httpx.HTTPError, ValueError) as e:
            return fail(str(e))
    if not (r.status_code == 200 and body.get("success") and body.get("content_hash")):
        return fail(body.get("error") or r.text[:100])
    # Supersessions are planned on locally computed keys; a different
    # key means Alaya's hashing changed and that plan is wrong.
    if body["content_hash"] != key:
        return fail(f"stored as {body['content_hash']}, expected {key}")
    if body.get("created"):
        stats.stored += 1
    else:
        stats.existed += 1
    return True


async def supersede_memory(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    old_hash: str,
    new_hash: str,
    s: Supersession,
    stored: set[str],
    stats: Stats,
) -> None:
    def fail(msg: str) -> None:
        stats.supersede_failed += 1
        if stats.supersede_failed <= 5:
            print(f"  SUPERSEDE FAIL: {msg}")

    if old_hash not in stored:
        fail(f"{old_hash} was not stored")
        return
    if new_hash not in stored:
        fail(f"{old_hash}: superseding memory {s.source_new_hash} was not stored")
        return
    async with sem:
        try:
            r = await client.post(
                "/supersede",
                json={"old_hash": old_hash, "new_hash": new_hash, "reason": s.reason},
            )
            body = r.json()
        except (httpx.HTTPError, ValueError) as e:
            fail(f"{old_hash}: {e}")
            return
    if r.status_code == 200 and body.get("success"):
        stats.superseded += 1
    else:
        fail(f"{old_hash}: {body.get('error') or r.text[:100]}")


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
    empty_markers = 0
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
            has_marker = "superseded_by" in metadata
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
            elif has_marker:
                # The source filters on a truthy marker, so an empty one is
                # live there and migrates live.
                empty_markers += 1
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
    if empty_markers:
        print(
            f"  {empty_markers} memories carry an empty superseded_by: live, migrated live"
        )

    # Plan every supersession on Alaya record keys, so source memories that
    # collapse onto one record are settled before anything is written.
    keys = [alaya_hash(m["content"]) for m in memories]
    to_alaya = {k: k for k in keys}
    to_alaya.update((src, k) for src, k in zip(source_hashes, keys, strict=True) if src)
    superseded = {s.index for s in supersessions}
    live = {k for i, k in enumerate(keys) if i not in superseded}
    by_record: dict[str, list[Supersession]] = {}
    for s in supersessions:
        by_record.setdefault(keys[s.index], []).append(s)

    shadowed: set[int] = set()
    conflicts: dict[str, list[Supersession]] = {}
    plan: list[tuple[str, str, Supersession]] = []  # one per superseded record
    for old, group in by_record.items():
        # Storing a shadowed copy would re-store the live record with its
        # tags and metadata, and superseding it would hide live content.
        if old in live:
            shadowed.update(s.index for s in group)
            stats.supersede_skipped += len(group)
            continue
        # A survivor with the same content is this record: nothing to do.
        survivors = {to_alaya.get(s.source_new_hash, s.source_new_hash) for s in group}
        survivors.discard(old)
        # One record holds one marker, so it cannot keep both histories.
        if len(survivors) > 1:
            conflicts[old] = group
            continue
        stats.supersede_skipped += len(group) - len(survivors)
        if survivors:
            plan.append((old, survivors.pop(), group[0]))

    if shadowed:
        print(f"  {len(shadowed)} superseded memories hold live content: not stored")
        for i in sorted(shadowed)[:5]:
            print(f"    {source_hashes[i] or f'memory #{i}'}")
    conflicted = {s.index for group in conflicts.values() for s in group}
    stats.conflicted = len(conflicted)
    if conflicts:
        print(
            f"  {stats.conflicted} superseded memories share content but name"
            " different survivors: not stored"
        )
        for old, group in list(conflicts.items())[:5]:
            print(f"    {old}:")
            for s in group:
                print(
                    f"      {source_hashes[s.index] or f'memory #{s.index}'}"
                    f" -> {s.source_new_hash}"
                )

    if DRY_RUN:
        for m in memories[:5]:
            tags = m.get("tags", [])
            print(f"  [{','.join(tags[:3])}] {m['content'][:80]}...")
        if len(memories) > 5:
            print(f"  ... and {len(memories) - 5} more")
        return

    # ── Phase 2: Store to Alaya ─────────────────────────────────────
    to_store = [i for i in range(len(memories)) if i not in shadowed | conflicted]
    print(f"\nPhase 2: Storing {len(to_store)} memories to Alaya...")
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
        stored: set[str] = set()
        for batch_start in range(0, len(to_store), BATCH_SIZE):
            batch = to_store[batch_start : batch_start + BATCH_SIZE]
            tasks = [
                store_memory(client, sem, memories[i], keys[i], stats) for i in batch
            ]
            for i, ok in zip(batch, await asyncio.gather(*tasks), strict=True):
                if ok:
                    stored.add(keys[i])

            done = min(batch_start + BATCH_SIZE, len(to_store))
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"  {done}/{len(to_store)} ({rate:.1f}/s)"
                f" — stored={stats.stored} existed={stats.existed} failed={stats.failed}"
            )

        # ── Phase 3: Recreate supersessions ─────────────────────────
        # Runs after every store, so each superseding record exists.
        print(f"\nPhase 3: Recreating {len(plan)} supersessions...")
        await asyncio.gather(
            *(
                supersede_memory(client, sem, old, new, s, stored, stats)
                for old, new, s in plan
            )
        )
        print(
            f"  superseded={stats.superseded} skipped={stats.supersede_skipped}"
            f" failed={stats.supersede_failed}"
        )

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
    print(f"  Supersede skipped: {stats.supersede_skipped}")
    print(f"  Supersede failed: {stats.supersede_failed}")
    print(f"  Conflicted:       {stats.conflicted}")

    if stats.failed or stats.supersede_failed or stats.conflicted:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
