#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["gepa", "httpx", "numpy", "litellm"]
# ///
"""
GEPA optimizer for Alaya's hybrid search parameters.

Phase 1 — Pre-embed:
    uv run benchmarks/optimize_params.py precompute

Phase 2 — Optimize:
    uv run benchmarks/optimize_params.py optimize

Phase 3 — Validate:
    uv run benchmarks/optimize_params.py validate --params results.json
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

import httpx
import numpy as np
from _stop_words import STOP_WORDS

sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)

# ── Config ───────────────────────────────────────────────────────────────────

EMBEDDING_URL = os.environ.get("EMBEDDING_URL", "http://10.43.242.167")
EMBEDDING_MODEL = "Snowflake/snowflake-arctic-embed-l-v2.0"
EMBEDDING_DIM = 1024
BATCH_SIZE = 6  # TEI on CPU limits to max_batch_requests=8

LME_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
LME_CACHE = "/tmp/longmemeval_s_cleaned.json"
EMBED_CACHE_DIR = Path("benchmarks/cache")
EMBED_CACHE_ARRAYS = EMBED_CACHE_DIR / "lme_embeddings.npz"
EMBED_CACHE_META = EMBED_CACHE_DIR / "lme_embeddings_meta.json"
# Legacy pickle cache (auto-migrated on first load)
_LEGACY_CACHE = EMBED_CACHE_DIR / "lme_embeddings.pkl"

# ── Seed parameters (current Alaya defaults) ─────────────────────────────────

SEED_PARAMS = {
    "rrf_k": 20,
    "rrf_blend_weight": 0.4,
    "alpha_small": 0.72,
    "alpha_medium": 0.7,
    "alpha_large": 0.8,
    "alpha_tag_threshold": 5,
    "alpha_tag_factor": 1.2,
    "boost_salience": 0.15,
    "boost_spacing": 0.10,
    "boost_summary": 0.15,
    "boost_graph": 0.10,
    "boost_hebbian": 0.10,
    "recency_decay_lambda": 0.01,
    "tag_only_base_score": 0.02,
    "score_cap": 1.5,
}

# ── Data helpers ─────────────────────────────────────────────────────────────


def download_data(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"  Downloading LongMemEval data to {path}...")
        urllib.request.urlretrieve(LME_URL, path)
    with open(path) as f:
        return json.load(f)


def build_session_doc(session: list[dict]) -> str:
    return "\n".join(t["content"] for t in session if t["role"] == "user")


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def extract_keywords(text: str, existing_tags: set[str] | None = None) -> list[str]:
    """Extract keywords mirroring Alaya's Rust hybrid_search::extract_query_keywords.

    - Split on non-alphanumeric (not just alpha — keeps digits like Rust)
    - Generate hyphenated compounds from adjacent pairs
    - Filter to existing_tags if provided (matches Rust's optional tag filter)
    """
    tokens = [
        t.lower()
        for t in re.split(r"[^a-zA-Z0-9]+", text)
        if len(t) >= 2 and t.lower() not in STOP_WORDS
    ]

    keywords = list(tokens)
    # Adjacent hyphenated compounds (same as Rust windows(2))
    for a, b in pairwise(tokens):
        keywords.append(f"{a}-{b}")

    # Filter to existing tags if provided
    if existing_tags is not None:
        keywords = [k for k in keywords if k in existing_tags]

    # Deduplicate preserving order
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def extract_tags(text: str, max_tags: int = 30) -> list[str]:
    """Extract distinctive keywords for tag matching. More tags = better recall."""
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    freq: dict[str, int] = {}
    for w in words:
        if w not in STOP_WORDS and len(w) >= 3:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:max_tags]]


def stratified_split(data: list[dict], dev_n: int = 50, seed: int = 42):
    """Split into dev (stratified) and val (rest)."""
    import random

    rng = random.Random(seed)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for entry in data:
        by_type[entry["question_type"]].append(entry)

    total = len(data)
    dev_ids = set()
    dev = []
    for _, entries in sorted(by_type.items()):
        k = max(1, round(len(entries) * dev_n / total))
        sample = rng.sample(entries, min(k, len(entries)))
        dev.extend(sample)
        dev_ids.update(e["question_id"] for e in sample)

    dev = dev[:dev_n]
    dev_ids = {e["question_id"] for e in dev}
    val = [e for e in data if e["question_id"] not in dev_ids]
    return dev, val


# ── TEI embedding ────────────────────────────────────────────────────────────


def embed_batch(
    client: httpx.Client, texts: list[str], prefix: str
) -> list[np.ndarray]:
    """Embed texts via TEI. prefix is 'search_query' or 'search_document'."""
    all_embeddings = []
    prefixed = [f"{prefix}: {t}" for t in texts]

    for i in range(0, len(prefixed), BATCH_SIZE):
        batch = prefixed[i : i + BATCH_SIZE]
        resp = client.post(
            f"{EMBEDDING_URL}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": batch, "encoding_format": "float"},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda x: x["index"])
        if len(data) != len(batch) or [d["index"] for d in data] != list(
            range(len(batch))
        ):
            raise ValueError(
                f"TEI returned {len(data)} embeddings for batch of {len(batch)}"
            )
        for d in data:
            all_embeddings.append(np.array(d["embedding"], dtype=np.float32))
        # Delay between batches — TEI on CPU with ONNX is fragile under load
        if i + BATCH_SIZE < len(prefixed):
            time.sleep(0.2)

    return all_embeddings


# ── Phase 1: Pre-embed ──────────────────────────────────────────────────────


def precompute(args):
    print("=" * 60)
    print("  Phase 1: Pre-embed all LongMemEval sessions")
    print("=" * 60)

    data = download_data(args.data)
    print(f"  Questions: {len(data)}")
    print(f"  TEI:       {EMBEDDING_URL}")

    client = httpx.Client(timeout=120)
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Resumable: load existing cache if present
    try:
        cache = load_cache()
        print(f"  Resuming from {len(cache)} cached questions")
    except FileNotFoundError:
        cache = {}

    t0 = time.monotonic()
    embedded_this_run = 0

    for i, entry in enumerate(data):
        qid = entry["question_id"]

        # Skip already cached
        if qid in cache:
            continue

        sessions = entry["haystack_sessions"]
        session_ids = entry["haystack_session_ids"]
        dates = entry["haystack_dates"]
        question = entry["question"]

        # Build docs and tags
        docs = []
        valid_session_ids = []
        valid_dates = []
        doc_tags = []
        for session, sid, date in zip(sessions, session_ids, dates, strict=True):
            doc = build_session_doc(session)
            if not doc.strip():
                continue
            docs.append(doc)
            valid_session_ids.append(sid)
            valid_dates.append(date)
            doc_tags.append(extract_tags(doc))

        if not docs:
            continue

        # Embed docs (retry with backoff on transient failures)
        success = False
        for attempt in range(5):
            try:
                doc_embeddings = embed_batch(client, docs, "search_document")
                query_embedding = embed_batch(client, [question], "search_query")[0]
                success = True
                break
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.ReadError) as e:
                wait = 10 * (2**attempt)  # 10, 20, 40, 80, 160s
                print(f"\n  TEI failed (attempt {attempt + 1}/5): {e}")
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
        if not success:
            print(f"  TEI failed after 5 retries at question {i + 1}.")
            print(f"  Saving {len(cache)} cached questions. Re-run to resume.")
            save_cache(cache)
            sys.exit(1)

        doc_emb_matrix = np.stack(doc_embeddings)

        cache[qid] = {
            "question": question,
            "question_type": entry["question_type"],
            "question_date": entry.get("question_date", ""),
            "answer_session_ids": entry["answer_session_ids"],
            "session_ids": valid_session_ids,
            "dates": valid_dates,
            "tags": doc_tags,
            "doc_embeddings": doc_emb_matrix,
            "query_embedding": query_embedding,
        }
        embedded_this_run += 1

        # Save checkpoint every 50 questions
        if embedded_this_run % 50 == 0:
            save_cache(cache)

        total_done = len(cache)
        if total_done % 25 == 0 or i == len(data) - 1:
            elapsed = time.monotonic() - t0
            rate = embedded_this_run / elapsed if elapsed > 0 else 0
            remaining = len(data) - total_done
            eta = remaining / rate if rate > 0 else 0
            print(
                f"  [{total_done:4}/{len(data)}]  {rate:.2f} q/s  ETA {eta / 60:.0f}m"
            )

    client.close()

    # Final save
    save_cache(cache)

    elapsed = time.monotonic() - t0
    size_mb = EMBED_CACHE_ARRAYS.stat().st_size / 1024 / 1024
    print(
        f"\n  Cached {len(cache)} questions to {EMBED_CACHE_ARRAYS} ({size_mb:.1f} MB)"
    )
    print(f"  Embedded this run: {embedded_this_run}")
    print(f"  Time: {elapsed:.0f}s")


def save_cache(cache: dict) -> None:
    """Save cache as JSON metadata + npz arrays (no pickle)."""
    arrays = {}
    meta = {}
    for qid, entry in cache.items():
        meta[qid] = {
            k: v
            for k, v in entry.items()
            if k not in ("doc_embeddings", "query_embedding")
        }
        arrays[f"{qid}__doc"] = entry["doc_embeddings"]
        arrays[f"{qid}__query"] = entry["query_embedding"]
    np.savez_compressed(EMBED_CACHE_ARRAYS, **arrays)
    with open(EMBED_CACHE_META, "w") as f:
        json.dump(meta, f, separators=(",", ":"))


def load_cache() -> dict:
    """Load cache from npz + JSON. Auto-migrates legacy pickle on first load."""
    if EMBED_CACHE_ARRAYS.exists() and EMBED_CACHE_META.exists():
        try:
            arrays = np.load(EMBED_CACHE_ARRAYS, allow_pickle=False)
            with open(EMBED_CACHE_META) as f:
                meta = json.load(f)
            cache = {}
            for qid, entry in meta.items():
                entry["doc_embeddings"] = arrays[f"{qid}__doc"]
                entry["query_embedding"] = arrays[f"{qid}__query"]
                cache[qid] = entry
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            raise RuntimeError(
                "Corrupt cache checkpoint — delete lme_embeddings.npz and "
                "lme_embeddings_meta.json, then rerun precompute."
            ) from exc
        return cache

    if EMBED_CACHE_ARRAYS.exists() and not EMBED_CACHE_META.exists():
        # Could be partial new-format write or legacy pickle named .npz.
        # Peek at magic bytes: real npz starts with PK (zip), pickle doesn't.
        with open(EMBED_CACHE_ARRAYS, "rb") as f:
            magic = f.read(2)
        if magic == b"PK":
            # Real npz without json sidecar — partial write, discard
            print("  Removing partial cache checkpoint (missing metadata sidecar)")
            EMBED_CACHE_ARRAYS.unlink()
        else:
            # Legacy pickle disguised as .npz — migrate
            import pickle

            with open(EMBED_CACHE_ARRAYS, "rb") as f:
                cache = pickle.load(f)
            print(f"  Migrating legacy pickle cache ({len(cache)} questions)...")
            EMBED_CACHE_ARRAYS.rename(_LEGACY_CACHE)
            save_cache(cache)
            print(f"  Migrated to {EMBED_CACHE_ARRAYS} + {EMBED_CACHE_META}")
            return cache

    raise FileNotFoundError(
        f"No cache found at {EMBED_CACHE_ARRAYS}. Run precompute first."
    )


# ── Scoring pipeline (pure Python, configurable params) ──────────────────────


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def cosine_sim_batch(query: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Cosine similarity between query (1d) and docs (2d matrix). Returns 1d array."""
    dot = docs @ query
    doc_norms = np.linalg.norm(docs, axis=1)
    q_norm = np.linalg.norm(query)
    denom = doc_norms * q_norm
    denom = np.where(denom == 0, 1.0, denom)
    return dot / denom


def get_adaptive_alpha(
    corpus_size: int, matching_tag_count: int, params: dict
) -> float:
    if corpus_size < 500:
        base = params["alpha_small"]
    elif corpus_size < 5000:
        base = params["alpha_medium"]
    else:
        base = params["alpha_large"]

    if matching_tag_count >= params["alpha_tag_threshold"]:
        return max(0.0, min(1.0, 1.0 - params["alpha_tag_factor"] * (1.0 - base)))
    return base


def rrf_score(rank: int, k: int) -> float:
    return 1.0 / (k + rank)


def score_question(cached_q: dict, params: dict) -> dict:
    """
    Score one question using cached embeddings and candidate params.
    Returns metrics + diagnostics.
    """
    query_emb = cached_q["query_embedding"]
    doc_embs = cached_q["doc_embeddings"]
    session_ids = cached_q["session_ids"]
    answer_ids = set(cached_q["answer_session_ids"])
    tags_per_doc = cached_q["tags"]
    n_docs = len(session_ids)

    if n_docs == 0:
        return {"recall_5": 0.0, "recall_10": 0.0, "error": "no docs"}

    # ── Vector ranking (truncated to fetch_size like Rust/Qdrant) ────
    fetch_size = min(int(params.get("fetch_size", 50)), n_docs)
    cosines = cosine_sim_batch(query_emb, doc_embs)
    vector_order = np.argsort(-cosines)  # descending
    vector_ranks = {
        int(idx): rank + 1 for rank, idx in enumerate(vector_order[:fetch_size])
    }

    # ── Tag ranking ──────────────────────────────────────────────────
    # Build tag set (mirrors Rust's get_all_tags → filter to existing)
    all_tags = {tag for tags in tags_per_doc for tag in tags}
    query_keywords = set(extract_keywords(cached_q["question"], existing_tags=all_tags))
    tag_scores = []
    for doc_idx, tags in enumerate(tags_per_doc):
        overlap = len(query_keywords & set(tags))
        tag_scores.append((doc_idx, overlap))
    tag_scores.sort(key=lambda x: -x[1])
    tag_ranks = {
        idx: rank + 1 for rank, (idx, score) in enumerate(tag_scores) if score > 0
    }

    # ── RRF fusion ───────────────────────────────────────────────────
    rrf_k = int(params["rrf_k"])
    # Rust passes n_keywords (matched query keyword count), not document count
    alpha = get_adaptive_alpha(n_docs, len(query_keywords), params)

    candidates = vector_ranks.keys() | tag_ranks.keys()
    fused = []
    for idx in candidates:
        v_rrf = (
            rrf_score(vector_ranks.get(idx, n_docs + 1), rrf_k)
            if idx in vector_ranks
            else 0.0
        )
        t_rrf = (
            rrf_score(tag_ranks.get(idx, n_docs + 1), rrf_k)
            if idx in tag_ranks
            else 0.0
        )
        combined = alpha * v_rrf + (1.0 - alpha) * t_rrf
        display = (
            float(cosines[idx])
            if idx in vector_ranks
            else params["tag_only_base_score"]
        )
        fused.append((idx, combined, display))

    fused.sort(key=lambda x: -x[1])

    # ── Final scoring (matches Rust fix) ─────────────────────────────
    # Blend normalized RRF rank signal with cosine display_score.
    max_rrf = max((r for _, r, _ in fused), default=1e-9)
    if max_rrf < 1e-9:
        max_rrf = 1e-9
    if "rrf_blend_weight" not in params:
        raise KeyError("candidate missing required key: rrf_blend_weight")
    blend_w = float(params["rrf_blend_weight"])
    if not math.isfinite(blend_w) or not (0.0 <= blend_w <= 1.0):
        raise ValueError(f"rrf_blend_weight={blend_w} out of range [0.0, 1.0]")

    scored = []
    for idx, rrf_combined, display in fused:
        rrf_norm = rrf_combined / max_rrf
        score = blend_w * rrf_norm + (1.0 - blend_w) * display
        score = min(score, params["score_cap"])
        scored.append((idx, score))

    scored.sort(key=lambda x: -x[1])

    # ── Evaluate ─────────────────────────────────────────────────────
    ranked_sids = [session_ids[idx] for idx, _ in scored]

    top5 = set(ranked_sids[:5])
    top10 = set(ranked_sids[:10])
    r5 = float(any(aid in top5 for aid in answer_ids))
    r10 = float(any(aid in top10 for aid in answer_ids))

    # NDCG@10 — IDCG from ground truth, not from retrieved subset
    relevances = [1.0 if sid in answer_ids else 0.0 for sid in ranked_sids[:10]]
    dcg_val = sum(r / math.log2(i + 2) for i, r in enumerate(relevances))
    n_relevant = min(10, len(answer_ids))
    idcg_val = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))
    ndcg10 = dcg_val / idcg_val if idcg_val > 0 else 0.0

    # Find where the correct answer actually ranked
    correct_rank = None
    for rank, sid in enumerate(ranked_sids):
        if sid in answer_ids:
            correct_rank = rank + 1
            break

    # Diagnostics for misses
    diag = {
        "question_type": cached_q["question_type"],
        "recall_5": r5,
        "recall_10": r10,
        "ndcg_10": ndcg10,
        "correct_rank": correct_rank,
        "n_docs": n_docs,
        "alpha_used": alpha,
        "matching_tags": len(query_keywords),
    }

    if r5 == 0.0 and correct_rank is not None:
        # Detailed miss diagnostics — derive from the session that set correct_rank
        correct_idx = int(scored[correct_rank - 1][0])
        rank1_idx = int(scored[0][0])
        diag["miss"] = {
            "question": cached_q["question"][:120],
            "correct_cosine": float(cosines[correct_idx]),
            "rank1_cosine": float(cosines[rank1_idx]),
            "correct_vector_rank": vector_ranks.get(correct_idx),
            "correct_tag_rank": tag_ranks.get(correct_idx),
            "correct_tags": tags_per_doc[correct_idx],
            "rank1_tags": tags_per_doc[rank1_idx],
        }

    return diag


# ── GEPA evaluator ───────────────────────────────────────────────────────────


def make_evaluator(cache: dict):
    """Create a GEPA evaluator that scores params against a question."""

    def evaluator(candidate_str: str, example: dict) -> tuple[float, dict]:
        # Parse candidate params from string (GEPA passes strings)
        try:
            params = json.loads(candidate_str)
        except (json.JSONDecodeError, TypeError):
            return 0.0, {"error": "invalid JSON params"}

        qid = example["question_id"]
        if qid not in cache:
            return 0.0, {"error": f"question {qid} not in cache"}

        try:
            result = score_question(cache[qid], params)
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
            return 0.0, {"error": "invalid candidate", "detail": str(e)}

        # Primary metric: NDCG@10 (continuous, gives gradient signal)
        # R@5 is binary (0/1) which makes GEPA unable to distinguish
        # candidates that fix one question but break another.
        score = result["ndcg_10"]

        # ASI: diagnostic side information
        asi = {
            "R@5": result["recall_5"],
            "R@10": result["recall_10"],
            "NDCG@10": result["ndcg_10"],
            "question_type": result["question_type"],
            "correct_rank": result["correct_rank"],
            "alpha_used": result["alpha_used"],
            "matching_tags": result["matching_tags"],
        }
        if "miss" in result:
            asi["miss_detail"] = result["miss"]

        return score, asi

    return evaluator


# ── Phase 2: Optimize ────────────────────────────────────────────────────────


def optimize(args):
    from gepa.optimize_anything import (
        optimize_anything,
        GEPAConfig,
        EngineConfig,
        ReflectionConfig,
    )

    print("=" * 60)
    print("  Phase 2: GEPA Optimization")
    print("=" * 60)

    cache = load_cache()
    print(f"  Loaded {len(cache)} cached questions")

    data = download_data(args.data)

    # Verify cache covers the dataset
    data_ids = {e["question_id"] for e in data}
    missing = data_ids - cache.keys()
    if missing:
        print(f"  ERROR: cache missing {len(missing)} questions. Run precompute first.")
        print(f"  Examples: {sorted(missing)[:5]}")
        sys.exit(1)

    dev, val = stratified_split(data, dev_n=args.dev_size)
    print(f"  Dev set:  {len(dev)} questions")
    print(f"  Val set:  {len(val)} questions")

    # Build dataset for GEPA (list of examples)
    dev_examples = [{"question_id": e["question_id"]} for e in dev]
    val_examples = [{"question_id": e["question_id"]} for e in val]

    evaluator = make_evaluator(cache)

    # Score baseline first
    if not dev_examples or not val_examples:
        print("\n  ERROR: empty dev or val set after split. Check --dev-size and data.")
        sys.exit(1)

    print("\n  Baseline (current Alaya params):")
    baseline_scores = []
    for ex in dev_examples:
        s, _ = evaluator(json.dumps(SEED_PARAMS), ex)
        baseline_scores.append(s)
    baseline_dev = sum(baseline_scores) / len(baseline_scores)
    print(f"    Dev NDCG@10:  {baseline_dev:.4f}")

    baseline_val_scores = []
    for ex in val_examples:
        s, _ = evaluator(json.dumps(SEED_PARAMS), ex)
        baseline_val_scores.append(s)
    baseline_val = sum(baseline_val_scores) / len(baseline_val_scores)
    print(f"    Val NDCG@10:  {baseline_val:.4f}")

    # LiteLLM gateway config
    litellm_api_base = args.api_base
    litellm_api_key = args.api_key
    reflection_lm = args.model

    if litellm_api_key:
        os.environ["OPENAI_API_KEY"] = litellm_api_key
    if litellm_api_base:
        os.environ["OPENAI_API_BASE"] = litellm_api_base

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_evals,
            max_candidate_proposals=args.max_proposals,
            parallel=False,  # evaluator is sub-ms, no parallelism needed
            display_progress_bar=True,
            seed=42,
        ),
        reflection=ReflectionConfig(
            reflection_lm=reflection_lm,
            reflection_minibatch_size=args.minibatch,
            skip_perfect_score=False,
        ),
    )

    print("\n  Starting GEPA optimization...")
    print(f"    Max evals:  {args.max_evals}")
    print(f"    Minibatch:  {args.minibatch}")
    print(f"    Model:      {reflection_lm}")
    print(f"{'─' * 60}\n")

    result = optimize_anything(
        seed_candidate=json.dumps(SEED_PARAMS, indent=2),
        evaluator=evaluator,
        dataset=dev_examples,
        valset=val_examples,
        objective=(
            "Maximize NDCG@10 (and thus R@5) on the LongMemEval retrieval benchmark. "
            "The candidate is a JSON dict of scoring parameters for a hybrid search system. "
            "Key levers: rrf_k controls rank fusion smoothing (lower = sharper ranking), "
            "alpha_small controls vector vs tag weight (higher = more vector, less tags), "
            "score_cap limits final scores (>1.0 removes the cap), "
            "tag_only_base_score is the display score for tag-only matches. "
            "The miss_detail in ASI shows WHY specific questions fail — use it. "
            "All values must be valid floats/ints. Do not add or remove keys. "
            "Return ONLY the JSON dict, no explanation."
        ),
        background=(
            "This is a retrieval benchmark (LongMemEval). For each question, ~53 conversation "
            "sessions are embedded and ranked. The system blends RRF rank fusion (vector + "
            "keyword tags) with raw cosine similarity via rrf_blend_weight. "
            "R@5 = is the correct session in the top 5? Target: >=0.95. "
            "The corpus is always <500, so alpha_small is the active alpha. "
            "Salience/spacing/context/summary/graph/hebbian boosts are zero (fresh data). "
            "ACTIVE PARAMS: rrf_blend_weight (0=pure cosine, 1=pure RRF), rrf_k (smoothing), "
            "alpha_small (vector vs tag weight), alpha_tag_threshold/factor, "
            "tag_only_base_score, score_cap."
        ),
        config=config,
    )

    # Extract best candidate
    best_params_str = result.best_candidate
    try:
        best_params = json.loads(best_params_str)
    except (json.JSONDecodeError, TypeError, ValueError):
        best_params = SEED_PARAMS
        print("  WARNING: Could not parse best candidate, using seed params")

    # Score on full val set
    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")

    val_scores = []
    val_per_type = defaultdict(list)
    for ex in val_examples:
        s, asi = evaluator(json.dumps(best_params), ex)
        val_scores.append(s)
        val_per_type[asi["question_type"]].append(s)

    opt_val = sum(val_scores) / len(val_scores) if val_scores else 0.0
    print(f"  Baseline Val NDCG@10:  {baseline_val:.4f}")
    print(f"  Optimized Val NDCG@10: {opt_val:.4f}  ({opt_val - baseline_val:+.4f})")
    print("\n  Per-type (val):")
    for qtype in sorted(val_per_type.keys()):
        scores = val_per_type[qtype]
        avg = sum(scores) / len(scores)
        print(f"    {qtype:30} NDCG@10={avg:.3f}  ({len(scores)}q)")

    # Save results
    out_path = args.out or "benchmarks/optimized_params.json"
    output = {
        "seed_params": SEED_PARAMS,
        "optimized_params": best_params,
        "baseline_dev_ndcg10": baseline_dev,
        "baseline_val_ndcg10": baseline_val,
        "optimized_val_ndcg10": opt_val,
        "improvement": opt_val - baseline_val,
        "val_per_type": {k: sum(v) / len(v) for k, v in val_per_type.items()},
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {out_path}")
    print(f"{'=' * 60}")


# ── Phase 3: Validate ────────────────────────────────────────────────────────


def validate(args):
    print("=" * 60)
    print("  Phase 3: Validate optimized params")
    print("=" * 60)

    cache = load_cache()
    data = download_data(args.data)
    data_ids = {e["question_id"] for e in data}
    cache_ids = set(cache.keys())
    missing = data_ids - cache_ids
    if missing:
        print(f"  ERROR: cache missing {len(missing)} questions. Run precompute first.")
        sys.exit(1)
    extra = cache_ids - data_ids
    if extra:
        print(f"  Pruning {len(extra)} stale cache entries not in dataset")
        for eid in extra:
            del cache[eid]
    n_cached = len(cache)
    with open(args.params) as f:
        result = json.load(f)
    params = result["optimized_params"]
    print(f"  Params: {args.params}")

    all_scores = []
    per_type = defaultdict(list)
    for cached_q in cache.values():
        diag = score_question(cached_q, params)
        all_scores.append(diag["recall_5"])
        per_type[diag["question_type"]].append(diag)

    r5 = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n  {n_cached}q R@5: {r5:.4f}")
    print("\n  Per-type:")
    for qtype in sorted(per_type.keys()):
        items = per_type[qtype]
        avg = sum(d["recall_5"] for d in items) / len(items)
        r10 = sum(d["recall_10"] for d in items) / len(items)
        print(f"    {qtype:30} R@5={avg:.3f}  R@10={r10:.3f}  ({len(items)}q)")

    print("\n  MemPalace reference: R@5=0.966 (raw), R@5=0.984 (hybrid v4)")
    print(f"{'=' * 60}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GEPA optimizer for Alaya search params"
    )
    sub = parser.add_subparsers(dest="command")

    # Precompute
    p_pre = sub.add_parser("precompute", help="Pre-embed all LongMemEval sessions")
    p_pre.add_argument("--data", default=LME_CACHE)

    # Optimize
    p_opt = sub.add_parser("optimize", help="Run GEPA optimization")
    p_opt.add_argument("--data", default=LME_CACHE)
    p_opt.add_argument("--dev-size", type=int, default=50)
    p_opt.add_argument(
        "--max-evals", type=int, default=50000, help="Max metric calls (budget)"
    )
    p_opt.add_argument(
        "--max-proposals", type=int, default=20, help="Max LLM iterations"
    )
    p_opt.add_argument("--minibatch", type=int, default=10)
    p_opt.add_argument(
        "--model",
        default=os.environ.get("GEPA_MODEL", "openai/gpt-5.4"),
        help="LiteLLM model (or set GEPA_MODEL env var)",
    )
    p_opt.add_argument(
        "--api-base",
        default=os.environ.get("GEPA_API_BASE"),
        help="LiteLLM API base URL (or set GEPA_API_BASE env var)",
    )
    p_opt.add_argument(
        "--api-key", default=None, help="API key (or set OPENAI_API_KEY)"
    )
    p_opt.add_argument("--out", default=None)

    # Validate
    p_val = sub.add_parser("validate", help="Validate params on full dataset")
    p_val.add_argument("--params", required=True, help="Path to optimized_params.json")
    p_val.add_argument("--data", default=LME_CACHE)

    args = parser.parse_args()
    if args.command == "precompute":
        precompute(args)
    elif args.command == "optimize":
        optimize(args)
    elif args.command == "validate":
        validate(args)
    else:
        parser.print_help()
