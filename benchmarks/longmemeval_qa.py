#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Alaya x LongMemEval: end-to-end QA accuracy
============================================

Adds a reader and a judge on top of the retrieval run written by
`longmemeval_bench.py --out`, so Alaya gets the number memory systems are usually
compared on: question-answering accuracy, not only retrieval hit-rate.

Three stages, each resumable and each reading the previous stage's file:

  read   For every question, take its top-k retrieved sessions (full sessions,
         both roles, sorted by date), build LongMemEval's own reading prompt
         (flat-session, JSON history, chain-of-thought) and ask the reader model.
         Writes {question_id, hypothesis} jsonl in the format LongMemEval's
         `evaluate_qa.py` reads.
  judge  Scores every hypothesis with LongMemEval's own per-question-type judge
         prompts (copied verbatim below) and writes `evaluate_qa.py`'s label
         format: the hypothesis row plus autoeval_label {model, label}.
  report Accuracy overall, per question type and for abstention, each with n and
         a Wilson 95% interval; strict recall_all@5 beside hit-rate@5.

Reader and judge both run through the Claude Code CLI in print mode
(`claude -p --bare`, no tools, an empty system prompt), so the judge differs from
the paper's GPT-4o. The prompts are LongMemEval's; only the model behind them is
not.

Usage (from benchmarks/, after a retrieval run):
    uv run longmemeval_qa.py read   --retrieval results_on_500q.jsonl --run-dir qa_runs/x
    uv run longmemeval_qa.py judge  --run-dir qa_runs/x
    uv run longmemeval_qa.py report --retrieval results_on_500q.jsonl --run-dir qa_runs/x
    uv run longmemeval_qa.py all    --retrieval results_on_500q.jsonl --run-dir qa_runs/x
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

DEFAULT_READER = "claude-sonnet-5-5"
DEFAULT_JUDGE = "claude-opus-5-5"
DEFAULT_K = 10
LME_CACHE = "/tmp/longmemeval_s_cleaned.json"

# The reader and judge see only the prompt: no tools, no project context, and an
# empty system prompt, matching evaluate_qa.py's single user message.
CLAUDE_ARGS = [
    "-p",
    "--bare",
    "--tools",
    "",
    "--no-session-persistence",
    "--output-format",
    "json",
]
SYSTEM_PROMPT = ""


# ── Claude CLI ───────────────────────────────────────────────────────────────


class ClaudeCLI:
    """One `claude -p` call per prompt; counts calls so a run can report them."""

    def __init__(self, binary: str = "claude", retries: int = 4, timeout: int = 600):
        self.binary = binary
        self.retries = retries
        self.timeout = timeout
        self.calls = 0
        self._lock = threading.Lock()

    def version(self) -> str:
        out = subprocess.run(
            [self.binary, "--version"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()

    def ask(self, model: str, prompt: str, system: str = SYSTEM_PROMPT) -> str:
        cmd = [self.binary, *CLAUDE_ARGS, "--model", model, "--system-prompt", system]
        last_err = ""
        for attempt in range(self.retries + 1):
            with self._lock:
                self.calls += 1
            try:
                proc = subprocess.run(
                    cmd,
                    check=False,
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                body = json.loads(proc.stdout) if proc.stdout.strip() else {}
                if (
                    proc.returncode == 0
                    and not body.get("is_error")
                    and isinstance(body.get("result"), str)
                ):
                    return body["result"]
                last_err = f"exit {proc.returncode}: {body.get('result') or proc.stderr.strip()[:300]}"
            except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
                last_err = repr(e)
            time.sleep(min(60, 5 * 2**attempt))
        raise RuntimeError(
            f"claude -p failed after {self.retries + 1} attempts: {last_err}"
        )


# ── LongMemEval prompts ──────────────────────────────────────────────────────
# Copied verbatim from LongMemEval (https://github.com/xiaowu0162/LongMemEval,
# commit 9e0b455): the reading prompt from src/generation/run_generation.py
# (flat-session retrieval, JSON history format, chain-of-thought) and the judge
# prompts from get_anscheck_prompt in src/evaluation/evaluate_qa.py.
# LongMemEval is MIT licensed, Copyright (c) 2024 Di Wu.

READER_TEMPLATE = "I will give you several history chats between you and a user. Please answer the question based on the relevant chat history. Answer the question step by step: first extract all the relevant information, and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ["single-session-user", "single-session-assistant", "multi-session"]:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "temporal-reasoning":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "knowledge-update":
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == "single-session-preference":
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


def judge_label(eval_response: str) -> bool:
    """evaluate_qa.py's rule: the label is true iff the reply contains 'yes'."""
    return "yes" in eval_response.strip().lower()


def build_reader_prompt(entry: dict, ranked_session_ids: list[str], k: int) -> str:
    """LongMemEval's flat-session reading prompt over the top-k retrieved sessions.

    Retrieval indexes user turns only (longmemeval_bench.build_session_doc), but the
    reader gets each retrieved session whole, as in LongMemEval's own retrieval
    runs: single-session-assistant answers live in the assistant turns.
    """
    by_id = {
        sid: (date, session)
        for sid, date, session in zip(
            entry["haystack_session_ids"],
            entry["haystack_dates"],
            entry["haystack_sessions"],
            strict=True,
        )
    }
    chunks = []
    for sid in ranked_session_ids[:k]:
        date, session = by_id[sid]
        turns = [
            {key: v for key, v in turn.items() if key != "has_answer"}
            for turn in session
        ]
        chunks.append((date, turns))
    chunks.sort(key=lambda c: c[0])
    history = ""
    for i, (date, turns) in enumerate(chunks):
        history += "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
            i + 1, date, "\n" + json.dumps(turns)
        )
    return READER_TEMPLATE.format(history, entry["question_date"], entry["question"])


# ── Files ────────────────────────────────────────────────────────────────────


def load_jsonl(path: str, missing_ok: bool = False) -> list[dict]:
    """Rows of a jsonl file. A missing file is an error unless missing_ok (resume)."""
    if not os.path.exists(path):
        if missing_ok:
            return []
        sys.exit(f"  Not found: {path}")
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_dataset(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        sys.exit(
            f"  LongMemEval data not found at {path}; run longmemeval_bench.py first (it downloads it)."
        )
    with open(path) as f:
        return {e["question_id"]: e for e in json.load(f)}


def run_parallel(todo: list, fn, out_path: str, workers: int, label: str) -> int:
    """Run fn over todo with a thread pool, appending each result row as it lands."""
    failures = 0
    done = 0
    with open(out_path, "a") as out_f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): item for item in todo}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except RuntimeError as e:
                failures += 1
                print(f"  {label} FAILED: {e}", file=sys.stderr)
                continue
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {label}: {done}/{len(todo)}")
    return failures


# ── Stages ───────────────────────────────────────────────────────────────────


def paths(run_dir: str) -> dict[str, str]:
    return {
        "hyp": os.path.join(run_dir, "hypotheses.jsonl"),
        "labels": os.path.join(run_dir, "judge_labels.jsonl"),
        "calls": os.path.join(run_dir, "calls.json"),
        "summary": os.path.join(run_dir, "qa_summary.json"),
    }


def record_calls(run_dir: str, stage: str, cli: ClaudeCLI, model: str, **extra) -> None:
    """Accumulate per-stage call counts across resumed runs; the report reads the
    models (and the reader's k) from here, so they describe the run, not argv."""
    p = paths(run_dir)["calls"]
    calls = load_json(p)
    prev = calls.get(stage, {})
    if prev and prev.get("model") != model:
        sys.exit(
            f"  {p} records {stage} model {prev['model']}, not {model}; use a new --run-dir"
        )
    calls[stage] = {
        "model": model,
        "calls": prev.get("calls", 0) + cli.calls,
        "cli": cli.version(),
        **extra,
    }
    with open(p, "w") as f:
        json.dump(calls, f, indent=2)
        f.write("\n")


def stage_read(args, cli: ClaudeCLI) -> int:
    data = load_dataset(args.data)
    retrieval = load_jsonl(args.retrieval)
    if not retrieval:
        sys.exit(f"  No retrieval rows in {args.retrieval}")
    out = paths(args.run_dir)["hyp"]
    done = {r["question_id"] for r in load_jsonl(out, missing_ok=True)}
    todo = [r for r in retrieval if r["question_id"] not in done]
    print(
        f"  read: {len(todo)} to do, {len(done)} already in {out} (reader {args.reader}, k={args.k})"
    )

    def one(r: dict) -> dict:
        prompt = build_reader_prompt(
            data[r["question_id"]], r["ranked_session_ids"], args.k
        )
        return {
            "question_id": r["question_id"],
            "hypothesis": cli.ask(args.reader, prompt),
        }

    failures = run_parallel(todo, one, out, args.workers, "read")
    record_calls(args.run_dir, "read", cli, args.reader, k=args.k)
    return failures


def stage_judge(args, cli: ClaudeCLI) -> int:
    data = load_dataset(args.data)
    hyps = load_jsonl(paths(args.run_dir)["hyp"])
    out = paths(args.run_dir)["labels"]
    done = {r["question_id"] for r in load_jsonl(out, missing_ok=True)}
    todo = [h for h in hyps if h["question_id"] not in done]
    print(
        f"  judge: {len(todo)} to do, {len(done)} already in {out} (judge {args.judge})"
    )

    def one(h: dict) -> dict:
        e = data[h["question_id"]]
        prompt = get_anscheck_prompt(
            e["question_type"],
            e["question"],
            e["answer"],
            h["hypothesis"],
            abstention="_abs" in h["question_id"],
        )
        reply = cli.ask(args.judge, prompt)
        return {
            **h,
            "autoeval_label": {
                "model": args.judge,
                "label": judge_label(reply),
                "raw": reply.strip()[:200],
            },
        }

    failures = run_parallel(todo, one, out, args.workers, "judge")
    record_calls(args.run_dir, "judge", cli, args.judge)
    return failures


def wilson(hits: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(centre - half, 4), round(centre + half, 4))


def score(hits: int, n: int) -> dict:
    return {
        "n": n,
        "correct": hits,
        "accuracy": round(hits / n, 4) if n else None,
        "ci95": wilson(hits, n),
    }


def recall_all_at_k(ranked: list[str], correct: list[str], k: int) -> float:
    """Strict retrieval: 1.0 only if EVERY ground-truth session is in the top k."""
    top = set(ranked[:k])
    return float(all(c in top for c in correct))


def stage_report(args) -> dict:
    """Score a finished run. Reads only the run's own files, never the dataset."""
    p = paths(args.run_dir)
    labels = load_jsonl(p["labels"])
    retrieval = load_jsonl(args.retrieval)
    calls = load_json(p["calls"])

    # The labels must be exactly one per retrieved question, from this run's judge.
    ids = [r["question_id"] for r in labels]
    if len(ids) != len(set(ids)) or set(ids) != {r["question_id"] for r in retrieval}:
        sys.exit(
            f"  {p['labels']} has {len(ids)} rows ({len(set(ids))} distinct) for "
            f"{len(retrieval)} retrieved questions; finish or redo the judge stage"
        )
    reader, judge = calls.get("read", {}), calls.get("judge", {})
    if not reader or not judge:
        sys.exit(f"  {p['calls']} lacks the read or judge stage")
    if {r["autoeval_label"]["model"] for r in labels} != {judge["model"]}:
        sys.exit(f"  labels in {p['labels']} are not all from judge {judge['model']}")

    # Grouped as LongMemEval's print_qa_metrics.py: every question in its
    # question_type (abstention questions included), and the _abs questions
    # again in their own abstention row.
    qtype = {r["question_id"]: r["question_type"] for r in retrieval}
    per_type: dict[str, list[int]] = defaultdict(list)
    abstention: list[int] = []
    for row in labels:
        hit = int(row["autoeval_label"]["label"])
        per_type[qtype[row["question_id"]]].append(hit)
        if "_abs" in row["question_id"]:
            abstention.append(hit)
    all_hits = [h for v in per_type.values() for h in v]

    # LongMemEval's retrieval evaluation skips abstention questions: their
    # answer sessions do not contain the asked-for fact.
    ret = [r for r in retrieval if "_abs" not in r["question_id"]]
    hit5 = sum(r["recall_5"] for r in ret)
    all5 = sum(
        recall_all_at_k(r["ranked_session_ids"], r["answer_session_ids"], 5)
        for r in ret
    )

    summary = {
        "dataset": "LongMemEval-S (longmemeval_s_cleaned.json)",
        "reader": reader["model"],
        "judge": judge["model"],
        "k_sessions": reader.get("k"),
        "reader_prompt": "LongMemEval flat-session, JSON history, chain-of-thought; full sessions sorted by date",
        "judge_prompt": "LongMemEval evaluate_qa.get_anscheck_prompt (per question type + abstention)",
        "grouping": "LongMemEval print_qa_metrics.py: abstention questions in their type and in the abstention row",
        "retrieval_file": os.path.basename(args.retrieval),
        "retrieval_note": args.retrieval_note
        or load_json(p["summary"]).get("retrieval_note", ""),
        "claude_calls": calls,
        "qa": {
            "overall": score(sum(all_hits), len(all_hits)),
            "task_averaged": round(
                sum(sum(v) / len(v) for v in per_type.values()) / len(per_type), 4
            ),
            "per_type": {t: score(sum(v), len(v)) for t, v in sorted(per_type.items())},
            "abstention": score(sum(abstention), len(abstention)),
        },
        "retrieval_excluding_abstention": {
            "hit_rate@5": score(int(hit5), len(ret)),
            "recall_all@5": score(int(all5), len(ret)),
        },
    }
    with open(p["summary"], "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    print(json.dumps(summary["qa"], indent=2))
    print(json.dumps(summary["retrieval_excluding_abstention"], indent=2))
    print(f"  Summary: {p['summary']}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LongMemEval QA accuracy over an Alaya retrieval run"
    )
    parser.add_argument("stage", choices=["read", "judge", "report", "all"])
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Directory for hypotheses, judge labels and summary",
    )
    parser.add_argument(
        "--retrieval",
        help="longmemeval_bench.py --out jsonl (needed by read and report)",
    )
    parser.add_argument(
        "--retrieval-note",
        default="",
        help="Retrieval config recorded verbatim in the summary",
    )
    parser.add_argument("--data", default=LME_CACHE, help="LongMemEval-S JSON")
    parser.add_argument("--reader", default=DEFAULT_READER)
    parser.add_argument("--judge", default=DEFAULT_JUDGE)
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Retrieved sessions given to the reader",
    )
    parser.add_argument(
        "--workers", type=int, default=6, help="Concurrent claude -p calls"
    )
    args = parser.parse_args()

    if args.stage in ("read", "report", "all") and not args.retrieval:
        parser.error("--retrieval is required for read, report and all")
    os.makedirs(args.run_dir, exist_ok=True)

    failures = 0
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
