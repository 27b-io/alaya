#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "httpx", "litellm", "sentence-transformers", "torch"]
# ///
"""
Cross-encoder rerank validation on LongMemEval.

Pipeline:
  1. Load cached embeddings + LongMemEval text
  2. Run current Alaya scoring (RRF+blend, faithful Python re-impl) to get top-N candidates
  3. Rerank top-N via cross-encoder (BAAI/bge-reranker-v2-m3)
  4. Compare R@5 / R@10 vs baseline

Sweeps:
  - rerank_top_n ∈ {10, 20, 30}  — how deep to rerank
  - baseline vs reranked         — A/B isolation
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from optimize_params import (  # noqa: E402
    build_session_doc,
    cosine_sim_batch,
    extract_keywords,
    get_adaptive_alpha,
    rrf_score,
)

CACHE = "benchmarks/cache/lme_embeddings.npz"
LME = "/tmp/longmemeval_s_cleaned.json"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_TOP_NS = [10, 20, 30]

PARAMS = {
    "rrf_k": 20,
    "alpha_small": 0.72,
    "alpha_medium": 0.7,
    "alpha_large": 0.8,
    "alpha_tag_threshold": 5,
    "alpha_tag_factor": 1.2,
    "tag_only_base_score": 0.02,
    "score_cap": 1.5,
    "fetch_size": 50,
    "rrf_blend_weight": 0.4,
}


def rank_question(cached_q: dict, params: dict) -> list[tuple[int, float]]:
    """Return [(doc_idx, alaya_score), ...] sorted by Alaya's current scoring."""
    query_emb = cached_q["query_embedding"]
    doc_embs = cached_q["doc_embeddings"]
    session_ids = cached_q["session_ids"]
    tags_per_doc = cached_q["tags"]
    n_docs = len(session_ids)

    if n_docs == 0:
        return []

    fetch_size = min(int(params.get("fetch_size", 50)), n_docs)
    cosines = cosine_sim_batch(query_emb, doc_embs)
    vector_order = np.argsort(-cosines)
    vector_ranks = {
        int(idx): rank + 1 for rank, idx in enumerate(vector_order[:fetch_size])
    }

    all_tags = {tag for tags in tags_per_doc for tag in tags}
    query_keywords = set(extract_keywords(cached_q["question"], existing_tags=all_tags))
    tag_scores = [
        (doc_idx, len(query_keywords & set(tags)))
        for doc_idx, tags in enumerate(tags_per_doc)
    ]
    tag_scores.sort(key=lambda x: -x[1])
    tag_ranks = {
        idx: rank + 1 for rank, (idx, score) in enumerate(tag_scores) if score > 0
    }

    rrf_k = int(params["rrf_k"])
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

    max_rrf = max((r for _, r, _ in fused), default=1e-9) or 1e-9
    blend_w = float(params["rrf_blend_weight"])
    scored = []
    for idx, rrf_combined, display in fused:
        rrf_norm = rrf_combined / max_rrf
        score = blend_w * rrf_norm + (1.0 - blend_w) * display
        score = min(score, params["score_cap"])
        scored.append((idx, score))

    scored.sort(key=lambda x: -x[1])
    return scored


def metrics_for_ranking(
    ranked_sids: list[str], answer_ids: set[str]
) -> dict[str, float]:
    top5 = set(ranked_sids[:5])
    top10 = set(ranked_sids[:10])
    correct_rank = next(
        (i + 1 for i, sid in enumerate(ranked_sids) if sid in answer_ids), None
    )
    return {
        "recall_5": float(any(aid in top5 for aid in answer_ids)),
        "recall_10": float(any(aid in top10 for aid in answer_ids)),
        "correct_rank": correct_rank,
    }


def main() -> None:
    print("Loading data...")
    cache = np.load(CACHE, allow_pickle=True)
    qids = list(cache.keys())
    cached = {
        qid: (
            cache[qid].item()
            if hasattr(cache[qid], "item") and getattr(cache[qid], "shape", None) == ()
            else cache[qid]
        )
        for qid in qids
    }

    lme = {e["question_id"]: e for e in json.load(open(LME))}
    print(f"  cached questions: {len(cached)}")
    print(f"  LME questions:    {len(lme)}")

    # Build per-question session_id → document mapping
    sid_to_doc: dict[str, dict[str, str]] = {}
    for qid, e in lme.items():
        if qid not in cached:
            continue
        sid_to_doc[qid] = {}
        for sess, sid in zip(
            e["haystack_sessions"], e["haystack_session_ids"], strict=True
        ):
            doc = build_session_doc(sess)
            if doc.strip():
                sid_to_doc[qid][sid] = doc

    # Load reranker
    print(f"\nLoading reranker: {RERANKER_MODEL} ...")
    t_load = time.monotonic()
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
    print(f"  loaded in {time.monotonic() - t_load:.1f}s")

    # Pre-compute Alaya rankings for all 500q (cheap, reuses cache)
    print("\nComputing Alaya baseline rankings...")
    t0 = time.monotonic()
    baseline_rankings: dict[str, list[tuple[str, float]]] = {}
    for qid in qids:
        if qid not in lme or qid not in sid_to_doc:
            continue
        q = cached[qid]
        scored = rank_question(q, PARAMS)
        session_ids = q["session_ids"]
        baseline_rankings[qid] = [(session_ids[idx], score) for idx, score in scored]
    print(
        f"  baseline computed in {time.monotonic() - t0:.1f}s for {len(baseline_rankings)}q"
    )

    # Baseline metrics
    print("\nBaseline (current Alaya scoring):")
    base_r5 = 0.0
    base_r10 = 0.0
    base_per_type = defaultdict(list)
    for qid, ranked in baseline_rankings.items():
        gt = set(lme[qid]["answer_session_ids"])
        sids = [s for s, _ in ranked]
        m = metrics_for_ranking(sids, gt)
        base_r5 += m["recall_5"]
        base_r10 += m["recall_10"]
        base_per_type[lme[qid]["question_type"]].append(m["recall_5"])
    base_r5 /= len(baseline_rankings)
    base_r10 /= len(baseline_rankings)
    print(f"  R@5={base_r5:.4f}  R@10={base_r10:.4f}")

    # Rerank sweep
    per_topn_results: dict[int, dict] = {}
    for top_n in RERANK_TOP_NS:
        print(f"\n--- Reranking top-{top_n} ---")
        t_rr = time.monotonic()

        # Build all (query, doc) pairs across all questions for this top_n
        # Track which pair belongs to which question/candidate position
        pairs: list[tuple[str, str]] = []
        pair_index: list[tuple[str, int]] = []  # (qid, candidate_index_in_baseline)
        for qid, ranked in baseline_rankings.items():
            query = lme[qid]["question"]
            top_candidates = ranked[:top_n]
            docs_for_qid = sid_to_doc[qid]
            for cand_i, (sid, _score) in enumerate(top_candidates):
                doc = docs_for_qid.get(sid, "")
                if not doc:
                    continue
                pairs.append((query, doc))
                pair_index.append((qid, cand_i))

        # Batched cross-encoder scoring
        print(f"  scoring {len(pairs)} (query, doc) pairs ...")
        rerank_scores = reranker.predict(pairs, batch_size=64, show_progress_bar=True)

        # Group scores by question, rerank candidates
        scores_by_qid: dict[str, dict[int, float]] = defaultdict(dict)
        for (qid, cand_i), score in zip(pair_index, rerank_scores, strict=True):
            scores_by_qid[qid][cand_i] = float(score)

        # Build reranked ranking per question
        reranked_r5 = 0.0
        reranked_r10 = 0.0
        per_type = defaultdict(list)
        for qid, ranked in baseline_rankings.items():
            top_candidates = ranked[:top_n]
            tail = ranked[top_n:]
            cand_scores = scores_by_qid.get(qid, {})
            reordered_top = sorted(
                enumerate(top_candidates),
                key=lambda x: -cand_scores.get(x[0], -1e9),
            )
            new_top = [c for _, c in reordered_top]
            new_ranking = new_top + tail
            sids = [s for s, _ in new_ranking]
            gt = set(lme[qid]["answer_session_ids"])
            m = metrics_for_ranking(sids, gt)
            reranked_r5 += m["recall_5"]
            reranked_r10 += m["recall_10"]
            per_type[lme[qid]["question_type"]].append(m["recall_5"])

        reranked_r5 /= len(baseline_rankings)
        reranked_r10 /= len(baseline_rankings)
        elapsed = time.monotonic() - t_rr
        print(f"  reranked R@5={reranked_r5:.4f}  R@10={reranked_r10:.4f}")
        print(
            f"  delta vs baseline: R@5 {reranked_r5 - base_r5:+.4f}  R@10 {reranked_r10 - base_r10:+.4f}"
        )
        print(
            f"  elapsed: {elapsed:.1f}s ({elapsed * 1000 / len(baseline_rankings):.0f}ms/q amortized)"
        )

        per_topn_results[top_n] = {
            "recall_5": reranked_r5,
            "recall_10": reranked_r10,
            "per_type": {t: sum(v) / len(v) for t, v in per_type.items()},
            "elapsed_s": elapsed,
        }

    # Final summary
    print(f"\n{'=' * 64}")
    print("  SUMMARY")
    print(f"{'=' * 64}")
    print(f"  Baseline (no rerank):     R@5={base_r5:.4f}  R@10={base_r10:.4f}")
    for top_n, res in per_topn_results.items():
        print(
            f"  +rerank top-{top_n:<3}            R@5={res['recall_5']:.4f}  R@10={res['recall_10']:.4f}  "
            f"({(res['recall_5'] - base_r5) * 100:+.2f}pp)"
        )
    print("\n  MemPalace reference:      R@5=0.966")

    # Per-type breakdown for best top-N
    best_n = max(per_topn_results, key=lambda n: per_topn_results[n]["recall_5"])
    print(f"\n  Per-type R@5 (best top-N = {best_n}):")
    print(f"  {'type':<32} {'baseline':>10} {'reranked':>10} {'Δ':>8}")
    for t in sorted(base_per_type.keys()):
        b = sum(base_per_type[t]) / len(base_per_type[t])
        r = per_topn_results[best_n]["per_type"].get(t, 0.0)
        print(f"  {t:<32} {b:>10.4f} {r:>10.4f} {(r - b) * 100:>+7.2f}pp")

    # Persist results
    out_path = "benchmarks/rerank_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": RERANKER_MODEL,
                "baseline": {
                    "recall_5": base_r5,
                    "recall_10": base_r10,
                    "per_type": {t: sum(v) / len(v) for t, v in base_per_type.items()},
                },
                "sweep": per_topn_results,
            },
            f,
            indent=2,
        )
    print(f"\n  Results written to: {out_path}")


if __name__ == "__main__":
    main()
