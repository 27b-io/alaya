#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Alaya x LongMemEval Benchmark
================================

Evaluates Alaya's retrieval against the LongMemEval benchmark (500 questions).
Replicates MemPalace's methodology for direct comparison.

For each question:
1. Reset Qdrant collections (fresh per question)
2. Store haystack sessions through Alaya's /store endpoint
3. Search through Alaya's /search endpoint (hybrid or similar mode)
4. Score retrieval against ground-truth answer sessions

Usage:
    uv run benchmarks/longmemeval_bench.py --limit 20
    uv run benchmarks/longmemeval_bench.py --mode hybrid
    uv run benchmarks/longmemeval_bench.py --mode raw
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime

import httpx
from _stop_words import STOP_WORDS


# Force unbuffered stdout for piped contexts (uv run, background tasks)
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_ALAYA_URL = "http://localhost:3001"
DEFAULT_QDRANT_URL = (
    "http://localhost:16333"  # isolated bench Qdrant; NEVER a shared 6333 stack
)
COLLECTION = "bench_memories"
TAG_COLLECTION = "bench_memories_tags"
EMBEDDING_DIM = 1024

LME_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LME_CACHE = "/tmp/longmemeval_s_cleaned.json"


# ── Data-safety guards ───────────────────────────────────────────────────────
# This harness DELETEs COLLECTION/TAG_COLLECTION on EVERY question (clean slate
# per question). That is only safe against the isolated, tmpfs-backed bench
# Qdrant. Refuse to run unless the target is unmistakably that stack:
#   1. --qdrant-url uses the remapped bench host port (never the default 6333).
#   2. Both collection names carry the bench_ prefix.
# DATA IS SACRED — fail closed before any destructive call.
BENCH_QDRANT_PORTS = {"16333"}
BENCH_COLLECTION_PREFIX = "bench_"


def assert_bench_safe(qdrant_url: str) -> None:
    """Abort before any destructive Qdrant call unless pointed at the bench stack."""
    from urllib.parse import urlparse

    port = str(urlparse(qdrant_url).port or "")
    if port not in BENCH_QDRANT_PORTS:
        sys.exit(
            f"\n  REFUSING TO RUN: --qdrant-url={qdrant_url!r} is not an isolated "
            f"bench port {sorted(BENCH_QDRANT_PORTS)}.\n"
            "  This harness DELETES collections per question — point it at the\n"
            "  benchmarks/docker-compose.yml Qdrant (host port 16333), never a\n"
            "  shared 6333 stack.\n"
        )
    for coll in (COLLECTION, TAG_COLLECTION):
        if not coll.startswith(BENCH_COLLECTION_PREFIX):
            sys.exit(
                f"\n  REFUSING TO RUN: collection {coll!r} lacks the "
                f"{BENCH_COLLECTION_PREFIX!r} prefix.\n"
                "  Destructive per-question resets are only permitted on bench_* "
                "collections.\n"
            )


# ── Metrics ──────────────────────────────────────────────────────────────────


def dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in ranked_ids[:k]]
    # IDCG from ground truth: min(k, num_relevant) ones at the top
    n_relevant = min(k, len(correct_ids))
    ideal = [1.0] * n_relevant + [0.0] * (k - n_relevant)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def recall_at_k(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    """Hit-rate@k: 1.0 if ANY correct id is in the top-k, else 0.0.

    This is the LongMemEval/MemPalace convention — a binary per-question hit,
    averaged across questions. It is NOT classical recall@k (the fraction of
    relevant items retrieved). Reported honestly as "hit-rate@k" in the docs;
    the `recall_*` key names are retained only for result-file compatibility.
    """
    top_k = set(ranked_ids[:k])
    return float(any(cid in top_k for cid in correct_ids))


# ── Data ─────────────────────────────────────────────────────────────────────


def download_data(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  Downloading LongMemEval data to {path}...")
        urllib.request.urlretrieve(LME_URL, path)
        print("  Done.")
    with open(path) as f:
        return json.load(f)


def build_session_doc(session: list[dict]) -> str:
    """Join user turns from a session into one document (same as MemPalace)."""
    user_turns = [t["content"] for t in session if t["role"] == "user"]
    return "\n".join(user_turns)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Qdrant direct (collection management only) ──────────────────────────────


class QdrantAdmin:
    """Direct Qdrant REST for collection lifecycle (not an Alaya concern)."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30)

    def reset_collections(self):
        """Delete and recreate both collections for a clean slate."""
        for coll in (COLLECTION, TAG_COLLECTION):
            resp = self.client.delete(f"{self.base}/collections/{coll}")
            if resp.status_code not in (200, 404):
                resp.raise_for_status()
            resp = self.client.put(
                f"{self.base}/collections/{coll}",
                json={
                    "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"},
                },
            )
            if resp.status_code not in (200, 409):
                resp.raise_for_status()

    def close(self):
        self.client.close()


# ── Alaya client ─────────────────────────────────────────────────────────────


class DuplicateMemory(Exception):
    """Raised when Alaya reports a dedup hit (HTTP 200, duplicate=true)."""


class AlayaClient:
    """Talks to Alaya's REST API."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=300)

    def health(self) -> dict:
        r = self.client.get(f"{self.base}/health")
        r.raise_for_status()
        return r.json()

    def store(self, content: str, tags: list[str], metadata: dict) -> dict:
        r = self.client.post(
            f"{self.base}/store",
            json={
                "content": content,
                "tags": tags,
                "memory_type": "reference",
                "metadata": metadata,
            },
        )
        r.raise_for_status()
        body = r.json()
        if body.get("duplicate"):
            raise DuplicateMemory(body.get("content_hash", ""))
        if body.get("success") is False or "error" in body:
            msg = body.get("error", body.get("message", "unknown store failure"))
            raise RuntimeError(f"Alaya store failed: {msg}")
        return body

    def search(self, query: str, mode: str = "hybrid", k: int = 10) -> dict:
        r = self.client.post(
            f"{self.base}/search",
            json={
                "query": query,
                "mode": mode,
                "k": k,
                "page_size": k,
                "include_superseded": True,
            },
        )
        r.raise_for_status()
        body = r.json()
        if "error" in body:
            raise RuntimeError(f"Alaya search error: {body['error']}")
        return body

    def close(self):
        self.client.close()


# ── Keyword extraction (for tags) ───────────────────────────────────────────


def extract_tags(text: str, max_tags: int = 5) -> list[str]:
    """Extract distinctive keywords from text for use as tags."""
    import re

    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    # Count frequencies, skip stop words
    freq: dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and len(w) >= 3:
            freq[w] = freq.get(w, 0) + 1
    # Sort by frequency descending, take top N
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:max_tags]]


# ── Stratified sampling ──────────────────────────────────────────────────────


def stratified_sample(data: list[dict], n: int) -> list[dict]:
    """Deterministic balanced sample across question types — NO randomness.

    Round-robin: walk the types in fixed (sorted) order, taking the next unused
    entry of each type per pass, preserving dataset order within a type. Same
    input + n => byte-identical output, so an A/B (rerank on vs off) sees the
    IDENTICAL question set and any delta is attributable to rerank alone.
    """
    by_type: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_type[entry["question_type"]].append(entry)

    types = sorted(by_type)
    cursors = {t: 0 for t in types}
    sampled: list[dict] = []

    while len(sampled) < n:
        progressed = False
        for t in types:
            if len(sampled) >= n:
                break
            if cursors[t] < len(by_type[t]):
                sampled.append(by_type[t][cursors[t]])
                cursors[t] += 1
                progressed = True
        if not progressed:  # every type exhausted
            break

    return sampled


# ── Benchmark ────────────────────────────────────────────────────────────────


def run_question(
    entry: dict,
    alaya: AlayaClient,
    qdrant: QdrantAdmin,
    mode: str,
    top_ks: list[int],
) -> dict:
    """Run one LongMemEval question through Alaya. Returns metrics."""

    question = entry["question"]
    answer_ids = set(entry["answer_session_ids"])
    sessions = entry["haystack_sessions"]
    session_ids = entry["haystack_session_ids"]
    dates = entry["haystack_dates"]

    # 1. Reset collections
    qdrant.reset_collections()

    # 2. Store each session through Alaya
    hash_to_session_ids: dict[str, list[str]] = {}
    stored = 0

    for session, sess_id, date in zip(sessions, session_ids, dates, strict=True):
        doc = build_session_doc(session)
        if not doc.strip():
            continue

        tags = extract_tags(doc)
        chash = content_hash(doc)
        hash_to_session_ids.setdefault(chash, []).append(sess_id)

        try:
            alaya.store(
                content=doc,
                tags=tags,
                metadata={"session_id": sess_id, "session_date": date},
            )
            stored += 1
        except DuplicateMemory:
            pass  # Same content already stored — expected for duplicate sessions
        except (httpx.HTTPStatusError, RuntimeError) as e:
            return {"error": f"store failure for {sess_id}: {e}"}

    if stored == 0:
        return {"error": "no sessions stored"}

    # 3. Search through Alaya
    search_mode = "similar" if mode == "raw" else "hybrid"
    max_k = max(top_ks)

    resp = alaya.search(query=question, mode=search_mode, k=max_k)
    results = resp.get("results", [])

    # 4. Map results back to session IDs
    ranked_session_ids = []
    for r in results:
        mem = r.get("memory", r)
        chash = mem.get("content_hash", "")
        for sid in hash_to_session_ids.get(chash, []):
            if sid not in ranked_session_ids:
                ranked_session_ids.append(sid)

    # 5. Score
    metrics = {
        "question_id": entry["question_id"],
        "question_type": entry["question_type"],
        "question": question,
        "answer_session_ids": list(answer_ids),
        "ranked_session_ids": ranked_session_ids[:max_k],
        "stored": stored,
        "retrieved": len(results),
    }

    for k in top_ks:
        metrics[f"recall_{k}"] = recall_at_k(ranked_session_ids, answer_ids, k)
        metrics[f"ndcg_{k}"] = ndcg_at_k(ranked_session_ids, answer_ids, k)

    return metrics


def run_benchmark(args):
    assert_bench_safe(args.qdrant_url)  # fail closed before any destructive call

    print(f"\n{'=' * 60}")
    print("  Alaya x LongMemEval Benchmark")
    print(f"{'=' * 60}")
    print(f"  Alaya:     {args.alaya_url}")
    print(f"  Qdrant:    {args.qdrant_url}")
    print(f"  Mode:      {args.mode}")
    print(f"  Top-k:     {args.top_k}")
    print(f"{'─' * 60}")

    # Load data
    data = download_data(args.data)
    if args.stratified:
        data = stratified_sample(data, args.stratified)
        print(f"  Sampling: stratified {args.stratified}")
    elif args.limit:
        data = data[: args.limit]
    print(f"  Questions: {len(data)}")

    # Connect
    alaya = AlayaClient(args.alaya_url)
    qdrant = QdrantAdmin(args.qdrant_url)

    # Connectivity check
    try:
        alaya.health()
        print("  Health:    connected")
    except Exception as e:
        print(f"\n  ERROR: Cannot reach Alaya at {args.alaya_url}: {e}")
        print("  Is the docker-compose stack running?")
        print(
            "    cd benchmarks && EMBEDDING_URL=http://10.43.242.167 docker compose up -d"
        )
        sys.exit(1)

    print(f"{'─' * 60}\n")

    top_ks = [int(k) for k in args.top_k.split(",")]

    # Output path determined up front so a crashed run can resume into it.
    if args.out:
        out_path = args.out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = f"benchmarks/results_alaya_{args.mode}_top{top_ks[0]}_{ts}.jsonl"

    # Resume: load any questions already completed in a prior (crashed) run.
    all_metrics: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)
    done_ids: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    m = json.loads(line)
                except json.JSONDecodeError:
                    # A crash mid-write can leave a truncated final line; skip it
                    # rather than aborting the whole resume.
                    print(f"  Resume:    skipping malformed line in {out_path}")
                    continue
                all_metrics.append(m)
                per_type[m["question_type"]].append(m)
                done_ids.add(m["question_id"])
        if done_ids:
            print(f"  Resume:    {len(done_ids)} question(s) already in {out_path}")

    t0 = time.monotonic()
    errors = 0
    # Append + flush per question: a crash at hour 5 keeps every completed row.
    out_f = open(out_path, "a")

    for i, entry in enumerate(data):
        if entry["question_id"] in done_ids:
            continue
        qt0 = time.monotonic()
        metrics = run_question(entry, alaya, qdrant, args.mode, top_ks)

        if "error" in metrics:
            errors += 1
            continue

        all_metrics.append(metrics)
        per_type[metrics["question_type"]].append(metrics)
        out_f.write(json.dumps(metrics) + "\n")
        out_f.flush()
        qt_elapsed = time.monotonic() - qt0

        # Progress
        if (i + 1) % 5 == 0 or i == len(data) - 1:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate if rate > 0 else 0
            if all_metrics:
                recall_parts = "  ".join(
                    f"R@{k}={sum(m[f'recall_{k}'] for m in all_metrics) / len(all_metrics):.3f}"
                    for k in top_ks
                    if f"recall_{k}" in all_metrics[0]
                )
            else:
                recall_parts = "no results yet"
            print(
                f"  [{i + 1:4}/{len(data)}]"
                f"  {recall_parts}"
                f"  {qt_elapsed:.1f}s/q  ETA {eta:.0f}s"
            )

    elapsed = time.monotonic() - t0
    out_f.close()
    alaya.close()
    qdrant.close()

    if not all_metrics:
        print("\n  No results. Check stack health.")
        sys.exit(1)

    # ── Summary ──────────────────────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed / len(all_metrics):.2f}s/question)")
    print(f"  Questions: {len(all_metrics)} (errors: {errors})")
    print()

    for k in top_ks:
        avg_recall = sum(m[f"recall_{k}"] for m in all_metrics) / len(all_metrics)
        avg_ndcg = sum(m[f"ndcg_{k}"] for m in all_metrics) / len(all_metrics)
        print(f"  R@{k:<3}  {avg_recall:.4f}    NDCG@{k:<3} {avg_ndcg:.4f}")

    print(f"\n  PER-TYPE BREAKDOWN (R@{top_ks[0]}):")
    for qtype in sorted(per_type.keys()):
        items = per_type[qtype]
        avg = sum(m[f"recall_{top_ks[0]}"] for m in items) / len(items)
        print(f"    {qtype:30} R@{top_ks[0]}={avg:.3f}  ({len(items)}q)")

    print(f"\n  {'─' * 56}")
    print("  MemPalace reference: R@5=0.966 (raw), R@5=0.984 (hybrid v4)")
    print(f"{'=' * 60}\n")

    # ── Write results ────────────────────────────────────────────────────
    # Per-question rows were already appended+flushed to out_path during the run.
    print(f"  Results: {out_path}")

    sampling = (
        f"stratified:{args.stratified}"
        if args.stratified
        else (f"first:{args.limit}" if args.limit else "full")
    )

    # Summary JSON
    summary_path = out_path.replace(".jsonl", "_summary.json")
    summary = {
        "system": "alaya",
        "mode": args.mode,
        "metric": "hit-rate@k (1.0 if any correct session in top-k, averaged; NOT classical recall@k)",
        "model": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "dimensions": EMBEDDING_DIM,
        "questions": len(all_metrics),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "alaya_url": args.alaya_url,
        "qdrant_url": args.qdrant_url,
        "sampling": sampling,
        "rerank_note": args.rerank_note,
    }
    for k in top_ks:
        summary[f"recall_{k}"] = round(
            sum(m[f"recall_{k}"] for m in all_metrics) / len(all_metrics), 4
        )
        summary[f"ndcg_{k}"] = round(
            sum(m[f"ndcg_{k}"] for m in all_metrics) / len(all_metrics), 4
        )
    summary["per_type"] = {}
    for qtype, items in sorted(per_type.items()):
        summary["per_type"][qtype] = {
            "count": len(items),
            f"recall_{top_ks[0]}": round(
                sum(m[f"recall_{top_ks[0]}"] for m in items) / len(items), 4
            ),
        }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alaya x LongMemEval Benchmark")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="DEBUG: first N questions (one question type only — NOT for headline "
        "or A/B numbers; use --stratified for those)",
    )
    parser.add_argument(
        "--stratified",
        type=int,
        default=None,
        help="Stratified sample of N questions across all 6 types",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "hybrid"],
        default="hybrid",
        help="Search mode: raw (vector only) or hybrid (RRF pipeline)",
    )
    parser.add_argument(
        "--top-k", default="5,10", help="Comma-separated k values (default: 5,10)"
    )
    parser.add_argument(
        "--alaya-url", default=DEFAULT_ALAYA_URL, help="Alaya server URL"
    )
    parser.add_argument(
        "--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL for resets"
    )
    parser.add_argument("--data", default=LME_CACHE, help="LongMemEval JSON path")
    parser.add_argument("--out", default=None, help="Output JSONL path")
    parser.add_argument(
        "--rerank-note",
        default="",
        help="Provenance string recorded verbatim in the summary "
        "(e.g. 'on:bge-reranker-v2-m3:top_n=20' or 'off'). The startup log "
        "'cross-encoder reranker enabled' only proves rerank is CONFIGURED; to prove "
        "it EXECUTED, watch the reranker TEI's te_predict_count climb (see README).",
    )
    args = parser.parse_args()

    run_benchmark(args)
