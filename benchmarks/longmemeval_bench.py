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
from pathlib import Path

import httpx


# Force unbuffered stdout for piped contexts (uv run, background tasks)
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

# ── Config ───────────────────────────────────────────────────────────────────

DEFAULT_ALAYA_URL = "http://localhost:3001"
DEFAULT_QDRANT_URL = "http://localhost:6333"
COLLECTION = "bench_memories"
TAG_COLLECTION = "bench_memories_tags"
EMBEDDING_DIM = 1024

LME_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LME_CACHE = "/tmp/longmemeval_s_cleaned.json"


# ── Metrics ──────────────────────────────────────────────────────────────────


def dcg(relevances: list[float], k: int) -> float:
    score = 0.0
    for i, rel in enumerate(relevances[:k]):
        score += rel / math.log2(i + 2)
    return score


def ndcg_at_k(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
    relevances = [1.0 if rid in correct_ids else 0.0 for rid in ranked_ids[:k]]
    ideal = sorted(relevances, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(relevances, k) / idcg


def recall_at_k(ranked_ids: list[str], correct_ids: set[str], k: int) -> float:
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
            # Delete (ignore errors — may not exist)
            try:
                self.client.delete(f"{self.base}/collections/{coll}")
            except Exception:
                pass
            # Create (ignore "already exists")
            try:
                self.client.put(
                    f"{self.base}/collections/{coll}",
                    json={
                        "vectors": {"size": EMBEDDING_DIM, "distance": "Cosine"},
                    },
                )
            except Exception:
                pass

    def close(self):
        self.client.close()


# ── Alaya client ─────────────────────────────────────────────────────────────


class AlayaClient:
    """Talks to Alaya's REST API."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.client = httpx.Client(timeout=60)

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
        return r.json()

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
        return r.json()

    def close(self):
        self.client.close()


# ── Keyword extraction (for tags) ───────────────────────────────────────────

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "about", "up", "it",
    "its", "me", "my", "we", "our", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "what", "which", "who", "this",
    "that", "these", "those", "am", "also", "like", "want", "know", "think",
    "make", "go", "get", "take", "see", "come", "look", "find", "give",
    "tell", "say", "ask", "use", "try", "work", "call", "keep", "let",
    "begin", "seem", "help", "show", "hear", "play", "run", "move", "live",
    "believe", "bring", "happen", "write", "provide", "sit", "stand",
    "lose", "pay", "meet", "include", "continue", "set", "learn", "change",
    "lead", "understand", "watch", "follow", "stop", "create", "speak",
    "read", "allow", "add", "spend", "grow", "open", "walk", "win", "offer",
    "remember", "love", "consider", "appear", "buy", "wait", "serve", "die",
    "send", "expect", "build", "stay", "fall", "cut", "reach", "kill",
    "remain", "suggest", "raise", "pass", "sell", "require", "report",
    "decide", "pull", "really", "much", "actually", "pretty", "something",
    "anything", "everything", "thing", "things", "got", "going", "been",
    "being", "well", "still", "even", "back", "after", "long", "right",
    "good", "new", "first", "last", "great", "little", "many", "own", "old",
    "big", "high", "different", "small", "large", "next", "early", "young",
    "important", "few", "public", "bad", "sure", "able", "feel",
}


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
    """Sample n questions proportionally across question types."""
    import random

    random.seed(42)  # reproducible
    by_type: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_type[entry["question_type"]].append(entry)

    total = len(data)
    sampled = []
    for qtype, entries in sorted(by_type.items()):
        k = max(1, round(len(entries) * n / total))
        sampled.extend(random.sample(entries, min(k, len(entries))))

    random.shuffle(sampled)
    return sampled[:n]


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
    hash_to_session_id: dict[str, str] = {}
    stored = 0

    for session, sess_id, date in zip(sessions, session_ids, dates):
        doc = build_session_doc(session)
        if not doc.strip():
            continue

        tags = extract_tags(doc)
        chash = content_hash(doc)
        hash_to_session_id[chash] = sess_id

        try:
            alaya.store(
                content=doc,
                tags=tags,
                metadata={"session_id": sess_id, "session_date": date},
            )
            stored += 1
        except httpx.HTTPStatusError as e:
            # Duplicate content hash — same doc text across sessions
            if e.response.status_code == 409:
                pass
            else:
                print(f"    Store error for {sess_id}: {e}", file=sys.stderr)

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
        sid = hash_to_session_id.get(chash)
        if sid and sid not in ranked_session_ids:
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
        print("    cd benchmarks && EMBEDDING_URL=http://10.43.242.167 docker compose up -d")
        sys.exit(1)

    print(f"{'─' * 60}\n")

    top_ks = [int(k) for k in args.top_k.split(",")]
    all_metrics: list[dict] = []
    per_type: dict[str, list[dict]] = defaultdict(list)
    t0 = time.monotonic()
    errors = 0

    for i, entry in enumerate(data):
        qt0 = time.monotonic()
        metrics = run_question(entry, alaya, qdrant, args.mode, top_ks)

        if "error" in metrics:
            errors += 1
            continue

        all_metrics.append(metrics)
        per_type[metrics["question_type"]].append(metrics)
        qt_elapsed = time.monotonic() - qt0

        # Progress
        if (i + 1) % 5 == 0 or i == len(data) - 1:
            avg_r5 = sum(m["recall_5"] for m in all_metrics) / len(all_metrics)
            avg_r10 = sum(m.get("recall_10", 0) for m in all_metrics) / len(all_metrics)
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            eta = (len(data) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i + 1:4}/{len(data)}]"
                f"  R@5={avg_r5:.3f}  R@10={avg_r10:.3f}"
                f"  {qt_elapsed:.1f}s/q  ETA {eta:.0f}s"
            )

    elapsed = time.monotonic() - t0
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
    print(f"  MemPalace reference: R@5=0.966 (raw), R@5=0.984 (hybrid v4)")
    print(f"{'=' * 60}\n")

    # ── Write results ────────────────────────────────────────────────────

    if args.out:
        out_path = args.out
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        out_path = f"benchmarks/results_alaya_{args.mode}_top{top_ks[0]}_{ts}.jsonl"

    with open(out_path, "w") as f:
        for m in all_metrics:
            f.write(json.dumps(m) + "\n")
    print(f"  Results: {out_path}")

    # Summary JSON
    summary_path = out_path.replace(".jsonl", "_summary.json")
    summary = {
        "system": "alaya",
        "mode": args.mode,
        "model": "Snowflake/snowflake-arctic-embed-l-v2.0",
        "dimensions": EMBEDDING_DIM,
        "questions": len(all_metrics),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
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
    parser.add_argument("--limit", type=int, default=None, help="Run first N questions")
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
    parser.add_argument("--top-k", default="5,10", help="Comma-separated k values (default: 5,10)")
    parser.add_argument("--alaya-url", default=DEFAULT_ALAYA_URL, help="Alaya server URL")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL, help="Qdrant URL for resets")
    parser.add_argument("--data", default=LME_CACHE, help="LongMemEval JSON path")
    parser.add_argument("--out", default=None, help="Output JSONL path")
    args = parser.parse_args()

    run_benchmark(args)
