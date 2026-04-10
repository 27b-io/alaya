#!/usr/bin/env python3
"""Backfill FalkorDB graph nodes and SUPERSEDES edges from Qdrant.

Scrolls all memories in Qdrant, creates a graph node for each via the
alaya-bridge /nodes/ensure endpoint, then rebuilds SUPERSEDES edges
from qdrant metadata.superseded_by fields.

Idempotent — safe to re-run. MERGE semantics on both nodes and edges.

Usage:
    kubectl -n mcp port-forward svc/qdrant 6333:6333 &
    kubectl -n mcp port-forward svc/alaya-bridge 3000:3000 &
    python3 scripts/backfill_graph.py

Environment:
    QDRANT_URL      (default: http://localhost:6333)
    BRIDGE_URL      (default: http://localhost:3000)
    COLLECTION      (default: memories_arctic1024)
    CONCURRENCY     (default: 20)
    BATCH_SIZE      (default: 250)
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field

import httpx
from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://localhost:3000")
COLLECTION = os.environ.get("COLLECTION", "memories_arctic1024")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "20"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "250"))

# Internal metadata point used by mcp-memory-service — skip it
METADATA_POINT_PREFIX = "00000000-0000-0000-0000-"


@dataclass
class Stats:
    scrolled: int = 0
    nodes_created: int = 0
    nodes_existed: int = 0
    nodes_failed: int = 0
    edges_created: int = 0
    edges_existed: int = 0
    edges_failed: int = 0
    supersede_pairs: list[dict] = field(default_factory=list)


async def ensure_node(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    content_hash: str,
    created_at: float,
    stats: Stats,
) -> None:
    async with sem:
        try:
            r = await client.post(
                "/nodes/ensure",
                json={"content_hash": content_hash, "created_at": created_at},
            )
            if r.status_code == 200:
                if r.json().get("created"):
                    stats.nodes_created += 1
                else:
                    stats.nodes_existed += 1
            else:
                stats.nodes_failed += 1
                if stats.nodes_failed <= 5:
                    print(f"  WARN node {content_hash[:16]}...: {r.status_code} {r.text[:100]}")
        except Exception as e:
            stats.nodes_failed += 1
            if stats.nodes_failed <= 5:
                print(f"  ERR node {content_hash[:16]}...: {e}")


async def create_supersedes_edge(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    source: str,
    target: str,
    created_at: float,
    stats: Stats,
) -> None:
    async with sem:
        try:
            r = await client.post(
                "/edges/create-system",
                json={
                    "source": source,
                    "target": target,
                    "relation_type": "SUPERSEDES",
                    "created_at": created_at,
                },
            )
            if r.status_code == 200:
                if r.json().get("created"):
                    stats.edges_created += 1
                else:
                    stats.edges_existed += 1
            else:
                stats.edges_failed += 1
                if stats.edges_failed <= 5:
                    print(f"  WARN edge {source[:12]}→{target[:12]}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            stats.edges_failed += 1
            if stats.edges_failed <= 5:
                print(f"  ERR edge {source[:12]}→{target[:12]}: {e}")


async def run() -> None:
    print(f"Qdrant:     {QDRANT_URL}")
    print(f"Bridge:     {BRIDGE_URL}")
    print(f"Collection: {COLLECTION}")
    print(f"Concurrency: {CONCURRENCY}, Batch: {BATCH_SIZE}")
    print()

    # --- Phase 1: Scroll qdrant ---
    print("Phase 1: Scrolling qdrant...")
    qclient = QdrantClient(url=QDRANT_URL, timeout=30)
    stats = Stats()
    sem = asyncio.Semaphore(CONCURRENCY)

    memories: list[tuple[str, float, dict]] = []  # (hash, created_at, metadata)
    next_offset = None

    while True:
        points, next_offset = qclient.scroll(
            collection_name=COLLECTION,
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
            content_hash = payload.get("content_hash")
            created_at = payload.get("created_at")

            if not content_hash or not created_at:
                continue

            metadata = payload.get("metadata") or {}
            memories.append((content_hash, float(created_at), metadata))

        if next_offset is None:
            break

    stats.scrolled = len(memories)
    print(f"  Scrolled {stats.scrolled} memories")

    # --- Phase 2: Create nodes ---
    print(f"\nPhase 2: Creating graph nodes ({stats.scrolled} total)...")
    t0 = time.monotonic()

    async with httpx.AsyncClient(base_url=BRIDGE_URL, timeout=10.0) as client:
        for batch_start in range(0, len(memories), BATCH_SIZE):
            batch = memories[batch_start : batch_start + BATCH_SIZE]
            tasks = [
                ensure_node(client, sem, h, ts, stats)
                for h, ts, _ in batch
            ]
            await asyncio.gather(*tasks)

            done = min(batch_start + BATCH_SIZE, len(memories))
            elapsed = time.monotonic() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  {done}/{stats.scrolled} ({rate:.0f}/s) — created={stats.nodes_created} existed={stats.nodes_existed} failed={stats.nodes_failed}")

    node_elapsed = time.monotonic() - t0
    print(f"  Done in {node_elapsed:.1f}s")

    # --- Phase 3: Collect and create SUPERSEDES edges ---
    for content_hash, created_at, metadata in memories:
        superseded_by = metadata.get("superseded_by")
        if superseded_by and isinstance(superseded_by, str) and len(superseded_by) == 64:
            # SUPERSEDES direction: new → old (source=superseder, target=superseded)
            stats.supersede_pairs.append({
                "source": superseded_by,
                "target": content_hash,
                "created_at": created_at,
            })

    if stats.supersede_pairs:
        print(f"\nPhase 3: Creating {len(stats.supersede_pairs)} SUPERSEDES edges...")
        t1 = time.monotonic()

        async with httpx.AsyncClient(base_url=BRIDGE_URL, timeout=10.0) as client:
            for batch_start in range(0, len(stats.supersede_pairs), BATCH_SIZE):
                batch = stats.supersede_pairs[batch_start : batch_start + BATCH_SIZE]
                tasks = [
                    create_supersedes_edge(client, sem, e["source"], e["target"], e["created_at"], stats)
                    for e in batch
                ]
                await asyncio.gather(*tasks)

        edge_elapsed = time.monotonic() - t1
        print(f"  Done in {edge_elapsed:.1f}s")
    else:
        print("\nPhase 3: No SUPERSEDES metadata found — skipping")

    # --- Summary ---
    total_elapsed = time.monotonic() - t0
    print(f"\n{'=' * 50}")
    print(f"Backfill complete in {total_elapsed:.1f}s")
    print(f"  Memories scrolled:  {stats.scrolled}")
    print(f"  Nodes created:      {stats.nodes_created}")
    print(f"  Nodes existed:      {stats.nodes_existed}")
    print(f"  Nodes failed:       {stats.nodes_failed}")
    print(f"  Edges created:      {stats.edges_created}")
    print(f"  Edges existed:      {stats.edges_existed}")
    print(f"  Edges failed:       {stats.edges_failed}")

    if stats.nodes_failed or stats.edges_failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(run())
