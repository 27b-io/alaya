#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare summary models on the golden set's memories, graded blind by a stronger model.

Every memory the contradiction golden fixture names
(``crates/alaya-core/tests/fixtures/contradiction_golden.json``) is fetched from a
live Ālaya and summarised by each arm with the server's summary request: the
system prompt is read from ``crates/alaya-backends/src/summary.rs``, the content
is cut to 4000 characters and the timeout is 30 s, as in ``anthropic.rs``. The
arms vary only the model, ``max_tokens`` and ``thinking`` of the body
``anthropic_body`` builds.

    A  claude-haiku-4-5-20251001  max_tokens 500   (baseline)
    B  claude-haiku-5-5           max_tokens 500
    C  claude-haiku-5-5           max_tokens 500   thinking {type: disabled}
    D  claude-haiku-5-5           max_tokens 1024

A summary call is never retried: the server does not retry, so an error counts.
Calls run interleaved across arms, so every arm sees the same endpoint load.

The grader sees one memory and its non-empty summaries in shuffled order under
neutral labels, so it never knows the arm, and answers three yes/no checks per
summary: faithful (no claim the content does not support), main point (captures
the content's central point) and one sentence. Rates are over memories fetched,
so an empty summary counts as a failure.

An arm passes when no summary is empty, errored or stopped on max_tokens (the
server stores a cut-off summary as is, so text alone is not enough), its faithful and
main-point rates are each within 0.03 of arm A's, and its p95 latency is under
10 s.

Run (any Anthropic Messages endpoint; keys via the environment):

    ALAYA_URL=http://localhost:13001 ALAYA_API_KEY=... \\
    SUMMARY_URL=https://api.anthropic.com SUMMARY_API_KEY=... \\
      uv run scripts/summary_eval.py [--grader claude-opus-5-5] [--limit N]

Memory contents are fetched at run time and never written to disk. The report,
printed to stdout, quotes failing summaries with their memory hashes.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "crates/alaya-core/tests/fixtures/contradiction_golden.json"
SUMMARY_RS = REPO / "crates/alaya-backends/src/summary.rs"

# Mirrors of anthropic.rs.
MAX_CONTENT_CHARS = 4000
SUMMARY_TIMEOUT_S = 30
ANTHROPIC_VERSION = "2023-06-01"

ARMS = {
    "A": {"model": "claude-haiku-4-5-20251001", "max_tokens": 500},
    "B": {"model": "claude-haiku-5-5", "max_tokens": 500},
    "C": {
        "model": "claude-haiku-5-5",
        "max_tokens": 500,
        "thinking": {"type": "disabled"},
    },
    "D": {"model": "claude-haiku-5-5", "max_tokens": 1024},
}
BASELINE = "A"
MAX_RATE_DROP = 0.03
MAX_P95_S = 10.0

CONCURRENCY = 4
GRADER_TIMEOUT_S = 180
GRADER_MAX_TOKENS = 8192
GRADER_ATTEMPTS = 3
SEED = 9587

GRADER_SYSTEM = """You grade one-sentence summaries of a stored note against the note itself.
For each summary answer three yes/no questions:
- faithful: every claim in the summary is supported by the note. A summary that \
adds a fact, a number, a cause or a conclusion the note does not state is not faithful. \
Leaving detail out is not a faithfulness failure.
- main_point: the summary captures the note's central point, the thing a reader \
would most need to know, not a side detail.
- one_sentence: the summary is a single sentence, with no preamble, list or second sentence.
Judge each summary on its own. Return one grade per summary id."""

GRADER_SCHEMA = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "faithful": {"type": "boolean"},
                    "main_point": {"type": "boolean"},
                    "one_sentence": {"type": "boolean"},
                },
                "required": ["id", "faithful", "main_point", "one_sentence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["grades"],
    "additionalProperties": False,
}


def system_prompt() -> str:
    """The summary system prompt, read from summary.rs (one source of truth)."""
    src = SUMMARY_RS.read_text()
    m = re.search(r'const SYSTEM_PROMPT: &str = "(.*?)";', src, re.DOTALL)
    if not m:
        sys.exit(f"SYSTEM_PROMPT not found in {SUMMARY_RS}")
    # A Rust string's trailing backslash drops the newline and the next line's indent.
    return re.sub(r"\\\n\s*", "", m.group(1))


def env(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        sys.exit(f"{key} must be set")
    return val


@dataclass
class Call:
    status: str  # "ok", "empty" or "error"
    text: str = ""
    stop_reason: str = ""
    latency_s: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""


@dataclass
class Grade:
    faithful: bool
    main_point: bool
    one_sentence: bool


def post_json(
    url: str, headers: dict[str, str], body: dict, timeout: float
) -> tuple[int, dict | str]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


def messages(base: str, key: str, body: dict, timeout: float) -> Call:
    headers = {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    start = time.monotonic()
    try:
        code, resp = post_json(
            f"{base.rstrip('/')}/v1/messages", headers, body, timeout
        )
    except (
        OSError,
        http.client.HTTPException,
        ValueError,
    ) as e:  # timeout, reset, bad body: the server fails open
        return Call(
            "error",
            latency_s=time.monotonic() - start,
            error=f"{type(e).__name__}: {e}"[:200],
        )
    latency = time.monotonic() - start
    if code != 200 or not isinstance(resp, dict):
        return Call("error", latency_s=latency, error=f"HTTP {code}: {resp}"[:300])
    usage = resp.get("usage") or {}
    # First non-empty text block, trimmed, as MessagesResponse::first_text.
    text = next(
        (
            b.get("text", "").strip()
            for b in resp.get("content", [])
            if b.get("type") == "text" and b.get("text", "").strip()
        ),
        "",
    )
    return Call(
        "ok" if text else "empty",
        text=text,
        stop_reason=resp.get("stop_reason") or "",
        latency_s=latency,
        # The endpoint may cache the prompt; cached reads are input all the same.
        input_tokens=sum(
            usage.get(k) or 0
            for k in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        ),
        output_tokens=usage.get("output_tokens", 0),
    )


def fetch_memories(url: str, key: str, hashes: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for h in hashes:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/memories/{h}",
            headers={"authorization": f"Bearer {key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                env_ = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if (
                e.code != 404
            ):  # a deleted memory is a 404; anything else is a broken run
                sys.exit(f"fetching memory {h}: HTTP {e.code}")
            env_ = {}
        except (OSError, ValueError) as e:
            sys.exit(f"fetching memory {h}: {type(e).__name__}: {e}")
        mem = env_.get("memory") if env_.get("found") else None
        if mem and mem.get("content"):
            out[h] = mem["content"]
        else:
            print(f"missing memory {h} (deleted since labelling?)", file=sys.stderr)
    return out


def grade(
    base: str, key: str, model: str, content: str, summaries: dict[str, str]
) -> tuple[dict[str, Grade], Call]:
    """Grade one memory's summaries. Keys of `summaries` are neutral labels."""
    listing = "\n".join(f"{sid}: {text}" for sid, text in summaries.items())
    body = {
        "model": model,
        "max_tokens": GRADER_MAX_TOKENS,
        "system": GRADER_SYSTEM,
        "messages": [
            {
                "role": "user",
                "content": f"<note>\n{content}\n</note>\n\n<summaries>\n{listing}\n</summaries>",
            }
        ],
        "output_config": {"format": {"type": "json_schema", "schema": GRADER_SCHEMA}},
    }
    spent = Call("ok")  # token totals across attempts, retries included
    error = ""
    for attempt in range(GRADER_ATTEMPTS):
        call = messages(base, key, body, GRADER_TIMEOUT_S)
        spent.input_tokens += call.input_tokens
        spent.output_tokens += call.output_tokens
        error = call.error or f"stop_reason={call.stop_reason}"
        if call.status == "ok":
            try:
                grades = {
                    g["id"]: Grade(g["faithful"], g["main_point"], g["one_sentence"])
                    for g in json.loads(call.text)["grades"]
                }
            except (ValueError, KeyError, TypeError) as e:
                error = f"unparseable grade: {e}"
            else:
                if set(grades) == set(summaries):
                    return grades, spent
                error = f"grade ids {sorted(grades)} != {sorted(summaries)}"
        time.sleep(2**attempt)
    raise RuntimeError(f"grader failed after {GRADER_ATTEMPTS} attempts: {error}")


def pct(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grader", default="claude-opus-5-5")
    ap.add_argument("--limit", type=int, help="first N memories only (smoke run)")
    ap.add_argument(
        "--arms",
        default="".join(ARMS),
        help="arms to run, e.g. ABC; must include the baseline",
    )
    args = ap.parse_args()
    arms = [arm for arm in ARMS if arm in args.arms.upper()]
    if BASELINE not in arms:
        sys.exit(f"--arms must include the baseline {BASELINE}")

    alaya_url, alaya_key = env("ALAYA_URL"), env("ALAYA_API_KEY")
    llm_url, llm_key = env("SUMMARY_URL"), env("SUMMARY_API_KEY")
    system = system_prompt()

    pairs = json.loads(FIXTURE.read_text())["pairs"]
    hashes = sorted({h for p in pairs for h in (p["a"], p["b"])})[: args.limit]
    memories = fetch_memories(alaya_url, alaya_key, hashes)
    truncated = sum(len(c) > MAX_CONTENT_CHARS for c in memories.values())
    print(
        f"fixture hashes={len(hashes)} fetched={len(memories)} truncated(>{MAX_CONTENT_CHARS} chars)={truncated}"
    )

    def summarise(job: tuple[str, str]) -> tuple[str, str, Call]:
        h, arm = job
        body = {
            "model": ARMS[arm]["model"],
            "max_tokens": ARMS[arm]["max_tokens"],
            "system": system,
            "messages": [{"role": "user", "content": memories[h][:MAX_CONTENT_CHARS]}],
        }
        if "thinking" in ARMS[arm]:
            body["thinking"] = ARMS[arm]["thinking"]
        return h, arm, messages(llm_url, llm_key, body, SUMMARY_TIMEOUT_S)

    rng = random.Random(SEED)
    jobs = [(h, arm) for h in memories for arm in arms]
    rng.shuffle(jobs)
    results: dict[str, dict[str, Call]] = {arm: {} for arm in arms}
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for i, (h, arm, call) in enumerate(pool.map(summarise, jobs), 1):
            results[arm][h] = call
            if i % 100 == 0:
                print(f"  summarised {i}/{len(jobs)}", file=sys.stderr)

    def grade_one(h: str) -> tuple[str, dict[str, Grade], Call]:
        graded = [arm for arm in arms if results[arm][h].status == "ok"]
        random.Random(f"{SEED}:{h}").shuffle(graded)
        labels = {f"S{i + 1}": arm for i, arm in enumerate(graded)}
        if not labels:
            return h, {}, Call("ok")
        grades, call = grade(
            llm_url,
            llm_key,
            args.grader,
            memories[h],
            {sid: results[a][h].text for sid, a in labels.items()},
        )
        return h, {labels[sid]: g for sid, g in grades.items()}, call

    grades: dict[str, dict[str, Grade]] = {arm: {} for arm in arms}
    grader_calls: list[Call] = []
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for i, (h, by_arm, call) in enumerate(pool.map(grade_one, memories), 1):
            grader_calls.append(call)
            for arm, g in by_arm.items():
                grades[arm][h] = g
            if i % 50 == 0:
                print(f"  graded {i}/{len(memories)}", file=sys.stderr)

    n = len(memories)
    rows = {}
    for arm, calls in results.items():
        cs = list(calls.values())
        failed = [c for c in cs if c.status != "ok"]
        by_reason: dict[str, int] = {}
        for c in failed:
            reason = c.stop_reason if c.status == "empty" else c.error.split(":")[0]
            key = f"{c.status}:{reason or '?'}"
            by_reason[key] = by_reason.get(key, 0) + 1
        g = grades[arm].values()
        lat = [c.latency_s for c in cs]
        ok_text = [len(c.text) for c in cs if c.status == "ok"]
        rows[arm] = {
            "n": len(cs),
            "failed": len(failed),
            # Text came back but the budget cut it off: a summary ending mid-sentence.
            "cut": sum(c.status == "ok" and c.stop_reason == "max_tokens" for c in cs),
            "by_reason": by_reason,
            "faithful": sum(x.faithful for x in g) / n,
            "main_point": sum(x.main_point for x in g) / n,
            "one_sentence": sum(x.one_sentence for x in g) / n,
            "mean_out": statistics.mean(c.output_tokens for c in cs),
            "max_out": max(c.output_tokens for c in cs),
            "p50": pct(lat, 0.50),
            "p95": pct(lat, 0.95),
            "mean_chars": statistics.mean(ok_text) if ok_text else 0.0,
            "in_tok": sum(c.input_tokens for c in cs),
            "out_tok": sum(c.output_tokens for c in cs),
        }

    base = rows[BASELINE]
    print(
        f"\n== arms (rates over {n} memories fetched; an empty or errored summary counts as a failure)"
    )
    print(
        f"{'arm':<4}{'model':<28}{'max_tok':>8}{'think':>9}{'fail':>6}{'cut':>5}{'faith':>8}{'main':>8}{'1sent':>8}"
        f"{'out_tok':>9}{'max':>6}{'p50 s':>8}{'p95 s':>8}{'chars':>7}  pass  failures by stop_reason"
    )
    for arm, r in rows.items():
        spec = ARMS[arm]
        passes = arm != BASELINE and (
            r["failed"] == 0
            and r["cut"] == 0
            and r["faithful"] >= base["faithful"] - MAX_RATE_DROP
            and r["main_point"] >= base["main_point"] - MAX_RATE_DROP
            and r["p95"] < MAX_P95_S
        )
        r["pass"] = passes
        print(
            f"{arm:<4}{spec['model']:<28}{spec['max_tokens']:>8}{spec.get('thinking', {}).get('type', '-'):>9}"
            f"{r['failed']:>6}{r['cut']:>5}{r['faithful']:>8.3f}{r['main_point']:>8.3f}{r['one_sentence']:>8.3f}"
            f"{r['mean_out']:>9.1f}{r['max_out']:>6}{r['p50']:>8.2f}{r['p95']:>8.2f}{r['mean_chars']:>7.0f}  "
            f"{'base' if arm == BASELINE else ('YES' if passes else 'no'):<5} {r['by_reason'] or '-'}"
        )

    print("\n== failing summaries, first 5 per candidate arm")
    for arm in [a for a in ("B", "C", "D") if a in results]:
        shown = 0
        for h, c in results[arm].items():
            g = grades[arm].get(h)
            if c.status != "ok":
                why = c.error or f"stop_reason={c.stop_reason}"
            else:
                failed = [
                    k
                    for k in ("faithful", "main_point", "one_sentence")
                    if not getattr(g, k)
                ]
                if c.stop_reason == "max_tokens":
                    failed.insert(0, "cut at max_tokens")
                why = ",".join(failed)
            if not why:
                continue
            print(f"[{arm}] {h}  {why}\n    {c.text!r}")
            shown += 1
            if shown == 5:
                break

    summ_in = sum(r["in_tok"] for r in rows.values())
    summ_out = sum(r["out_tok"] for r in rows.values())
    g_in = sum(c.input_tokens for c in grader_calls)
    g_out = sum(c.output_tokens for c in grader_calls)
    print(
        f"\ntokens: summaries in={summ_in} out={summ_out}  grader in={g_in} out={g_out}  "
        f"total={summ_in + summ_out + g_in + g_out}"
    )


if __name__ == "__main__":
    main()
