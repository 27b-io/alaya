#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "httpx", "litellm"]
# ///
"""
Bisect rrf_blend_weight on cached LongMemEval embeddings.

Uses optimize_params.py's score_question (faithful Python re-impl of the
Rust scoring pipeline) to sweep blend weights without rebuilding the image.
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from optimize_params import score_question  # noqa: E402

CACHE = "benchmarks/cache/lme_embeddings.npz"

# Current Rust constants (post-PR-#28), with rrf_blend_weight swept.
BASE = {
    "rrf_k": 20,
    "alpha_small": 0.72,
    "alpha_medium": 0.7,
    "alpha_large": 0.8,
    "alpha_tag_threshold": 5,
    "alpha_tag_factor": 1.2,
    "tag_only_base_score": 0.02,
    "score_cap": 1.5,
    "fetch_size": 50,
    "boost_salience": 0.15,
    "boost_spacing": 0.10,
    "boost_summary": 0.15,
    "boost_graph": 0.10,
    "boost_hebbian": 0.10,
    "recency_decay_lambda": 0.01,
}

SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]


def main() -> None:
    d = np.load(CACHE, allow_pickle=True)
    qids = list(d.keys())
    cached = {
        qid: (
            d[qid].item()
            if hasattr(d[qid], "item") and getattr(d[qid], "shape", None) == ()
            else d[qid]
        )
        for qid in qids
    }
    print(f"Loaded {len(cached)} cached questions")

    per_blend: dict[float, list[dict]] = {}
    for blend in SWEEP:
        params = {**BASE, "rrf_blend_weight": blend}
        results = []
        for qid in qids:
            q = cached[qid]
            m = score_question(q, params)
            m["question_id"] = qid
            m["question_type"] = q["question_type"]
            results.append(m)
        per_blend[blend] = results

    # Summary table
    print(f"\n{'blend':>6}  {'R@5':>6}  {'R@10':>6}  {'Δ vs 0.4':>10}")
    print("-" * 36)
    baseline = per_blend[0.4]
    base_r5 = sum(r["recall_5"] for r in baseline) / len(baseline)
    for blend in SWEEP:
        rs = per_blend[blend]
        r5 = sum(r["recall_5"] for r in rs) / len(rs)
        r10 = sum(r["recall_10"] for r in rs) / len(rs)
        delta = (r5 - base_r5) * 100
        marker = "  ← current" if blend == 0.4 else ""
        print(f"{blend:>6.2f}  {r5:>6.3f}  {r10:>6.3f}  {delta:>+9.1f}pp{marker}")

    # Per-type at blend=0.0 vs 0.4 (regression detection)
    print("\nPer-type R@5 (blend=0.0 vs 0.4 vs 0.5):")
    type_r5: dict[float, dict[str, list[float]]] = {
        b: defaultdict(list) for b in [0.0, 0.4, 0.5]
    }
    for blend in [0.0, 0.4, 0.5]:
        for r in per_blend[blend]:
            type_r5[blend][r["question_type"]].append(r["recall_5"])

    types = sorted(type_r5[0.4].keys())
    print(f"  {'type':<32} {'0.0':>6} {'0.4':>6} {'0.5':>6}")
    for t in types:
        avg = {b: sum(type_r5[b][t]) / len(type_r5[b][t]) for b in [0.0, 0.4, 0.5]}
        n = len(type_r5[0.4][t])
        print(f"  {t:<32} {avg[0.0]:>6.3f} {avg[0.4]:>6.3f} {avg[0.5]:>6.3f}  ({n}q)")

    # Find questions that moved when going from 0.0 → 0.4 (PR #28 effect)
    print("\nQuestions PR #28 (blend 0.0→0.4) MOVED:")
    by_id_0 = {r["question_id"]: r for r in per_blend[0.0]}
    by_id_4 = {r["question_id"]: r for r in per_blend[0.4]}
    gained, lost = 0, 0
    for qid in by_id_0:
        a, b = by_id_0[qid]["recall_5"], by_id_4[qid]["recall_5"]
        if a < b:
            gained += 1
        elif a > b:
            lost += 1
    print(f"  blend 0.0 → 0.4: gained={gained}, lost={lost}, net={gained - lost}")

    # Also blend 0.4 → 0.5
    by_id_5 = {r["question_id"]: r for r in per_blend[0.5]}
    gained, lost = 0, 0
    for qid in by_id_4:
        a, b = by_id_4[qid]["recall_5"], by_id_5[qid]["recall_5"]
        if a < b:
            gained += 1
        elif a > b:
            lost += 1
    print(f"  blend 0.4 → 0.5: gained={gained}, lost={lost}, net={gained - lost}")

    # Dump optimum
    best = max(SWEEP, key=lambda b: sum(r["recall_5"] for r in per_blend[b]))
    best_r5 = sum(r["recall_5"] for r in per_blend[best]) / len(per_blend[best])
    print(f"\nOptimum: blend={best}  R@5={best_r5:.3f}")


if __name__ == "__main__":
    main()
