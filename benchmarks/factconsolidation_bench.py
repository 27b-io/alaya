#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "pyarrow"]
# ///
"""
Alaya x MemoryAgentBench FactConsolidation
==========================================

FactConsolidation is the conflict-resolution task of MemoryAgentBench
(arXiv 2507.05257, data `ai-hyz/MemoryAgentBench`, split `Conflict_Resolution`).
A numbered list of facts is memorised in order; later facts overwrite earlier
ones, and each question must be answered from the newest fact. Single-hop (sh)
questions ask one fact; multi-hop (mh) questions chain several.

Protocol, at the shortest length, 6k (sh and mh share one context):
  retrieve  Reset the bench collections, store every numbered fact as its own
            Alaya memory (serial number kept in the text), then hybrid-search
            each question for the top-k memories. Needs the bench stack.
  read      Ask the reader with MemoryAgentBench's own rag-agent query prompt,
            the retrieved memories inserted as "Memory i:" in search order.
  judge     Score with the same LongMemEval judge prompt and judge model used for
            the LongMemEval QA table (the headline metric here), and with the
            benchmark's own substring exact match, which is lenient: it also
            counts an answer named anywhere in the reader's reasoning.
  report    Accuracy per sub-task, both metrics, with Wilson 95% intervals.

MemoryAgentBench's RAG agents store the context as multi-sentence chunks. Here
each fact is its own Alaya memory, which is how an agent would write these
facts, so the result is not comparable with the paper's RAG rows.

Usage (from benchmarks/, bench stack up):
    uv run factconsolidation_bench.py all --run-dir qa_runs/x/factconsolidation
"""

import argparse
import json
import os
import re
import shutil
import string
import sys
import urllib.request
from collections import defaultdict

import pyarrow.parquet as pq

from longmemeval_bench import (
    DEFAULT_ALAYA_URL,
    DEFAULT_QDRANT_URL,
    AlayaClient,
    QdrantAdmin,
    assert_bench_safe,
    extract_tags,
)
from longmemeval_qa import (
    DEFAULT_JUDGE,
    DEFAULT_READER,
    ClaudeCLI,
    bind_stage,
    get_anscheck_prompt,
    judge_label,
    load_json,
    load_jsonl,
    record_calls,
    retrieval_fingerprint,
    run_parallel,
    score,
)

MAB_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
MAB_URL = (
    f"https://huggingface.co/datasets/ai-hyz/MemoryAgentBench/resolve/{MAB_REVISION}"
    "/data/Conflict_Resolution-00000-of-00001.parquet"
)
MAB_CACHE = "/tmp/memoryagentbench_conflict_resolution.parquet"
DEFAULT_K = 10
LENGTH = "6k"

# MemoryAgentBench utils/templates.py, the rag-agent query for fact consolidation,
# verbatim (MIT licensed). {question} is filled per question.
QUERY_TEMPLATE = "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:"
# utils/templates.py system message; methods/embedding_retriever.py prefixes the
# retrieved memories to the query as "Memory i:" blocks, in search order.
SYSTEM = "You are a helpful assistant that can read the context and memorize it for future retrieval."
FACT_LINE = re.compile(r"^\d+\. ")


# ── Metric (MemoryAgentBench utils/eval_other_utils.py) ──────────────────────


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_output(prediction: str) -> str:
    """The text after 'Answer:', else the first line (the benchmark scores both)."""
    m = re.search(r"Answer:\s*(.*)", prediction, re.DOTALL)
    return (m.group(1) if m else prediction).strip().split("\n")[0]


def substring_exact_match(prediction: str, answers: list[str]) -> bool:
    preds = {normalize_answer(prediction), normalize_answer(parse_output(prediction))}
    return any(normalize_answer(a) in p for a in answers for p in preds)


# ── Data ─────────────────────────────────────────────────────────────────────


def load_rows(path: str, length: str = LENGTH) -> tuple[list[str], list[dict]]:
    """Facts and questions for one length; sh and mh share the same context."""
    if not os.path.exists(path):
        print(f"  Downloading MemoryAgentBench Conflict_Resolution to {path}...")
        try:
            with (
                urllib.request.urlopen(MAB_URL, timeout=60) as resp,
                open(path + ".tmp", "wb") as f,
            ):
                shutil.copyfileobj(resp, f)
        except OSError as e:  # URLError and socket timeouts are OSErrors
            sys.exit(f"  Download of {MAB_URL} failed: {e}")
        os.replace(path + ".tmp", path)  # never cache a partial download
    table = pq.read_table(path).to_pylist()
    rows = {r["metadata"]["source"]: r for r in table}
    sh, mh = (
        rows[f"factconsolidation_sh_{length}"],
        rows[f"factconsolidation_mh_{length}"],
    )
    if sh["context"] != mh["context"]:
        sys.exit(
            f"  sh and mh contexts differ at {length}; this script assumes one shared context"
        )
    facts = [line for line in sh["context"].split("\n") if FACT_LINE.match(line)]
    questions = []
    for task, row in (("single-hop", sh), ("multi-hop", mh)):
        for i, (q, ans) in enumerate(
            zip(row["questions"], row["answers"], strict=True)
        ):
            answers = [ans] if isinstance(ans, str) else list(ans)
            questions.append(
                {
                    "question_id": f"{task}-{i}",
                    "task": task,
                    "question": q,
                    "answers": answers,
                }
            )
    return facts, questions


# ── Stages ───────────────────────────────────────────────────────────────────


def stage_retrieve(args) -> None:
    assert_bench_safe(args.qdrant_url)  # fail closed before the destructive reset
    facts, questions = load_rows(args.data)
    out = os.path.join(args.run_dir, "retrieval.jsonl")
    if os.path.exists(out):
        print(f"  retrieve: {out} exists; delete it to re-run retrieval")
        return
    alaya, qdrant = AlayaClient(args.alaya_url), QdrantAdmin(args.qdrant_url)
    try:
        alaya.health()
        qdrant.reset_collections()
        print(
            f"  retrieve: storing {len(facts)} facts, then {len(questions)} searches (k={args.k})"
        )
        for fact in facts:
            alaya.store(content=fact, tags=extract_tags(fact), metadata={})
        # Written whole, then renamed, so a crash never leaves a short file that a
        # rerun would take as finished retrieval.
        with open(out + ".tmp", "w") as f:
            for q in questions:
                resp = alaya.search(query=q["question"], mode="hybrid", k=args.k)
                texts = [
                    r.get("memory", r).get("content", "")
                    for r in resp.get("results", [])
                ]
                f.write(json.dumps({**q, "retrieved": texts[: args.k]}) + "\n")
        os.replace(out + ".tmp", out)
    finally:
        alaya.close()
        qdrant.close()


def stage_read(args, cli: ClaudeCLI) -> int:
    retrieval = load_jsonl(os.path.join(args.run_dir, "retrieval.jsonl"))
    config = {
        "model": args.reader,
        "k": args.k,
        "retrieval_sha256": retrieval_fingerprint(retrieval, "retrieved"),
    }
    bind_stage(args.run_dir, "read", **config)
    out = os.path.join(args.run_dir, "hypotheses.jsonl")
    done = {r["question_id"] for r in load_jsonl(out, missing_ok=True)}
    todo = [r for r in retrieval if r["question_id"] not in done]
    print(f"  read: {len(todo)} to do (reader {args.reader})")

    def one(r: dict) -> dict:
        memories = "\n".join(
            f"Memory {i + 1}:\n{t}" for i, t in enumerate(r["retrieved"])
        )
        prompt = memories + "\n" + QUERY_TEMPLATE.format(question=r["question"])
        return {
            "question_id": r["question_id"],
            "hypothesis": cli.ask(args.reader, prompt, system=SYSTEM),
        }

    failures = run_parallel(todo, one, out, args.workers, "read")
    record_calls(args.run_dir, "read", cli, **config)
    return failures


def stage_judge(args, cli: ClaudeCLI) -> int:
    by_id = {
        r["question_id"]: r
        for r in load_jsonl(os.path.join(args.run_dir, "retrieval.jsonl"))
    }
    hyps = load_jsonl(os.path.join(args.run_dir, "hypotheses.jsonl"))
    bind_stage(args.run_dir, "judge", model=args.judge)
    out = os.path.join(args.run_dir, "judge_labels.jsonl")
    done = {r["question_id"] for r in load_jsonl(out, missing_ok=True)}
    todo = [h for h in hyps if h["question_id"] not in done]
    print(f"  judge: {len(todo)} to do (judge {args.judge})")

    def one(h: dict) -> dict:
        q = by_id[h["question_id"]]
        # LongMemEval's generic fact-recall judge prompt; the gold answer is the
        # newest fact, which the question alone does not reveal.
        prompt = get_anscheck_prompt(
            "single-session-user",
            q["question"],
            " / ".join(q["answers"]),
            h["hypothesis"],
        )
        reply = cli.ask(args.judge, prompt)
        return {
            **h,
            "substring_exact_match": substring_exact_match(
                h["hypothesis"], q["answers"]
            ),
            "autoeval_label": {
                "model": args.judge,
                "label": judge_label(reply),
                "raw": reply.strip()[:200],
            },
        }

    failures = run_parallel(todo, one, out, args.workers, "judge")
    record_calls(args.run_dir, "judge", cli, model=args.judge)
    return failures


def stage_report(args) -> dict:
    by_id = {
        r["question_id"]: r
        for r in load_jsonl(os.path.join(args.run_dir, "retrieval.jsonl"))
    }
    labels = load_jsonl(os.path.join(args.run_dir, "judge_labels.jsonl"))
    calls_path = os.path.join(args.run_dir, "calls.json")
    calls = load_json(calls_path)
    ids = [r["question_id"] for r in labels]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        sys.exit(
            f"  {len(ids)} labels ({len(set(ids))} distinct) for {len(by_id)} questions; finish the judge stage"
        )
    if "read" not in calls or "judge" not in calls:
        sys.exit(f"  {calls_path} lacks the read or judge stage")
    if calls["read"].get("retrieval_sha256") != retrieval_fingerprint(
        list(by_id.values()), "retrieved"
    ):
        sys.exit("  retrieval.jsonl is not the retrieval this run's reader read")
    if {r["autoeval_label"]["model"] for r in labels} != {calls["judge"]["model"]}:
        sys.exit(f"  labels are not all from judge {calls['judge']['model']}")
    em: dict[str, list[int]] = defaultdict(list)
    judged: dict[str, list[int]] = defaultdict(list)
    for row in labels:
        task = by_id[row["question_id"]]["task"]
        em[task].append(int(row["substring_exact_match"]))
        judged[task].append(int(row["autoeval_label"]["label"]))
    summary_path = os.path.join(args.run_dir, "summary.json")
    summary = {
        "dataset": f"MemoryAgentBench Conflict_Resolution factconsolidation_{{sh,mh}}_{LENGTH} @ {MAB_REVISION}",
        "ingest": "one Alaya memory per numbered fact, hybrid search",
        "k": calls["read"].get("k"),
        "reader": calls["read"]["model"],
        "judge": calls["judge"]["model"],
        "retrieval_note": args.retrieval_note
        or load_json(summary_path).get("retrieval_note", ""),
        "claude_calls": calls,
        "judge_accuracy": {t: score(sum(v), len(v)) for t, v in sorted(judged.items())},
        "substring_exact_match_lenient": {
            t: score(sum(v), len(v)) for t, v in sorted(em.items())
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: summary[k]
                for k in ("judge_accuracy", "substring_exact_match_lenient")
            },
            indent=2,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MemoryAgentBench FactConsolidation over Alaya"
    )
    parser.add_argument("stage", choices=["retrieve", "read", "judge", "report", "all"])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data", default=MAB_CACHE)
    parser.add_argument("--alaya-url", default=DEFAULT_ALAYA_URL)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument(
        "--retrieval-note",
        default="",
        help="Retrieval config recorded verbatim in the summary",
    )
    parser.add_argument("--reader", default=DEFAULT_READER)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    failures = 0
    if args.stage in ("retrieve", "all"):
        stage_retrieve(args)
    if args.stage in ("read", "all"):
        failures += stage_read(args, ClaudeCLI())
    if args.stage in ("judge", "all"):
        failures += stage_judge(args, ClaudeCLI())
    if failures:
        sys.exit(f"  {failures} call(s) failed; rerun the same command to resume them.")
    if args.stage in ("report", "all"):
        stage_report(args)


if __name__ == "__main__":
    main()
