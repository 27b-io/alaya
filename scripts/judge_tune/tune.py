#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["anthropic>=1.5,<2", "gepa>=0.1.4,<0.2", "httpx>=0.27"]
# ///
"""Tune the contradiction judge's system prompt with GEPA against the golden set.

The judge call here is wire-identical to ``JudgeClient``
(``crates/alaya-backends/src/judge.rs``): the candidate prompt goes in the
``system`` slot, the pair is rendered by a port of ``render_pair``, the reply is
constrained to the same JSON schema via structured output, and the output cap
is the same. Scoring is the Rust golden harness's
(``crates/alaya-core/tests/golden_judge.rs``): the verdict class must match the
label, and a supersession must also name the labelled survivor. The seed prompt
is read from ``judge.rs`` so there is one source of truth.

Subcommands (see README.md):

    split   write split.json (stratified by label, fixed seed)
    tune    run GEPA on the train split; validation pairs are only ever scored
    eval    score one prompt file on val / train / all pairs, list disagreements

Memory contents are fetched from Ālaya at run time; this script writes none of
them to disk. The run directory (gitignored) holds model output about them:
candidate prompts, verdict reasons and GEPA's own logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anthropic
import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURE = REPO / "crates/alaya-core/tests/fixtures/contradiction_golden.json"
JUDGE_RS = REPO / "crates/alaya-backends/src/judge.rs"
SPLIT_FILE = HERE / "split.json"

# ── Mirrors of judge.rs / anthropic.rs ───────────────────────────────────────
CLASSES = ("contradiction", "supersession", "coexist", "unrelated")
CONFLICT = {"contradiction", "supersession"}
MAX_OUTPUT_TOKENS = 4096
MAX_CONTENT_CHARS = 4000
MAX_REASON_CHARS = 200
JUDGE_TIMEOUT_S = 90
CONCURRENCY = 4
MINIBATCH = 6  # the seed scores ~0.9; GEPA's default 3 is all-correct most rounds
TRAIN_FRAC = 0.6

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(CLASSES)},
        "survivor": {
            "anyOf": [{"type": "string", "enum": ["a", "b"]}, {"type": "null"}]
        },
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["verdict", "survivor", "reason", "confidence"],
    "additionalProperties": False,
}

# USD per million tokens, list price (input, output). Cached input is priced
# at the full input rate: an upper bound, never an under-count.
PRICE_PER_MTOK = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

# The quality bar the judge must clear before promotion.
BAR_PRECISION = 0.90
BAR_COEXIST_ESCALATION = 0.10

# Text every candidate must keep. A candidate missing any of it, longer than
# twice the seed, naming infrastructure, or quoting a golden memory is scored
# 0 without a single judge call.
CONSERVATIVE_SENTENCE = (
    'Be conservative: call "supersession" or "contradiction" only when a reader '
    "relying on the older memory today would be misled."
)
REQUIRED_PHRASES = (
    *(f'"{c}"' for c in CLASSES),
    "survivor",
    "reason",
    "confidence",
    CONSERVATIVE_SENTENCE,
)
INFRA_RE = re.compile(
    r"\b[\w-]+\.(?:io|net|com|org|dev|local|internal|svc)\b"  # hostnames
    r"|\b(?:falkordb|qdrant|k3s|k8s|kubectl|namespace|proxy)\b",
    re.I,
)
# Leakage. A 12-char window is the natural unit for "quoted a memory", but
# against ~600k chars of memory text it also flags ordinary English (the seed
# prompt itself shares 101 such windows; rejected candidates shared phrases
# like " an earlier "). The gate therefore uses 40-char verbatim spans plus
# identifier-shaped tokens; the 12-char count is still measured and reported.
LEAK_WINDOW = 40
LITERAL_WINDOW = 12
IDENT_RE = re.compile(
    r"\b[A-Z]{2,}-\d+\b"  # ticket ids
    r"|#\d{2,}\b"  # issue / PR refs
    r"|\b\d{4}-\d{2}-\d{2}\b"  # dates
    r"|\b[0-9a-f]{8,}\b"  # hashes
    r"|https?://\S+"  # urls
    r"|\b\w+_\w+\b"  # snake_case names
    r"|\bv?\d+\.\d+(?:\.\d+)?\b"  # versions
)

REFLECTION_TEMPLATE = """I gave an assistant this system prompt for judging whether two memories from an engineering team's long-term memory store conflict:
```
<curr_param>
```

Below are pairs the assistant judged, its verdict for each, and feedback saying whether the verdict matched the label and why:
```
<side_info>
```

Write an improved system prompt. Hard constraints, each checked mechanically; a prompt that breaks one is discarded unread and costs you the round:
- Keep the four verdict names "supersession", "contradiction", "coexist" and "unrelated", each in double quotes; keep the survivor / reason / confidence output contract; keep this sentence verbatim: Be conservative: call "supersession" or "contradiction" only when a reader relying on the older memory today would be misled.
- LENGTH: the current prompt is <curr_len> characters. Yours must be under <max_len> characters, about <max_words> words. Proposals over that were discarded. Edit and tighten the existing text; do not append sections or worked examples.
- Do not quote memory content: no project or ticket names, dates, numbers, hashes, people, hostnames, file or system names. Describe the general pattern in your own words.
- No hostnames or infrastructure names of any kind.

Read the feedback for the recurring reasons a verdict was wrong, and change the prompt so the assistant gets those cases right without losing the ones it already gets right. Provide the new prompt within ``` blocks."""  # noqa: E501


def reflection_template(seed: str) -> str:
    """Template with the length budget filled in (10 % under the hard cap)."""
    max_len = int(2 * len(seed) * 0.9)
    return (
        REFLECTION_TEMPLATE.replace("<curr_len>", str(len(seed)))
        .replace("<max_len>", str(max_len))
        .replace("<max_words>", str(max_len // 6))
    )


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def env(key: str) -> str:
    value = os.environ.get(key)
    if not value:
        sys.exit(f"{key} must be set")
    return value


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ── Fixture, split, seed prompt ──────────────────────────────────────────────


@dataclass(frozen=True)
class Pair:
    id: int
    a: str
    b: str
    label: str
    survivor: str | None
    source: str

    def survivor_letter(self) -> str | None:
        if self.survivor is None:
            return None
        return "A" if self.survivor == self.a else "B"


def load_fixture() -> list[Pair]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        Pair(i, p["a"], p["b"], p["label"], p.get("survivor"), p.get("source", ""))
        for i, p in enumerate(raw["pairs"])
    ]


def fixture_sha() -> str:
    return hashlib.sha256(FIXTURE.read_bytes()).hexdigest()


def make_split(pairs: list[Pair], seed: int, train_frac: float) -> dict:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for p in pairs:
        by_label[p.label].append(p.id)
    train: list[int] = []
    val: list[int] = []
    counts = {}
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        k = round(len(ids) * train_frac)
        train += ids[:k]
        val += ids[k:]
        counts[label] = {"train": k, "val": len(ids) - k}
    return {
        "fixture_sha256": fixture_sha(),
        "seed": seed,
        "train_frac": train_frac,
        "counts": counts,
        "train": sorted(train),
        "val": sorted(val),
    }


def load_split(pairs: list[Pair]) -> tuple[list[Pair], list[Pair]]:
    if not SPLIT_FILE.exists():
        sys.exit(f"{SPLIT_FILE} missing: run `tune.py split` first")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split["fixture_sha256"] != fixture_sha():
        sys.exit("split.json was cut from a different fixture; rerun `split --force`")
    by_id = {p.id: p for p in pairs}
    train = [by_id[i] for i in split["train"]]
    val = [by_id[i] for i in split["val"]]
    if set(split["train"]) & set(split["val"]) or len(train) + len(val) != len(pairs):
        sys.exit("split.json does not partition the fixture")
    return train, val


RUST_ESCAPES = {
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "0": "\0",
    '"': '"',
    "'": "'",
    "\\": "\\",
}


def unescape_rust(literal: str) -> str:
    """The body of a Rust `"..."` literal as the string it compiles to."""
    literal = re.sub(r"\\\n\s*", "", literal)  # `\` + newline continues the line
    unknown = set(re.findall(r"\\(.)", literal)) - set(RUST_ESCAPES)
    if unknown:  # passing one through would tune a prompt the server never sends
        sys.exit(
            f"unsupported escapes in SYSTEM_PROMPT: {sorted(unknown)}; extend RUST_ESCAPES"
        )
    return re.sub(r"\\(.)", lambda e: RUST_ESCAPES[e.group(1)], literal)


def seed_prompt() -> str:
    """The `SYSTEM_PROMPT` literal in judge.rs, unescaped."""
    src = JUDGE_RS.read_text(encoding="utf-8")
    m = re.search(r'const SYSTEM_PROMPT: &str = "(.*?)";\n', src, re.S)
    if not m:
        sys.exit(f"SYSTEM_PROMPT not found in {JUDGE_RS}")
    prompt = unescape_rust(m.group(1))
    missing = [p for p in REQUIRED_PHRASES if p not in prompt]
    if missing:
        sys.exit(f"seed prompt in judge.rs lacks required text: {missing}")
    return prompt


# ── Memories ─────────────────────────────────────────────────────────────────


def fetch_memories(pairs: list[Pair]) -> dict[str, dict]:
    url = env("ALAYA_URL").rstrip("/")
    headers = {"Authorization": f"Bearer {env('ALAYA_API_KEY')}"}
    hashes = sorted({h for p in pairs for h in (p.a, p.b)})
    memories: dict[str, dict] = {}
    transport = httpx.HTTPTransport(retries=3)
    with httpx.Client(timeout=60, headers=headers, transport=transport) as client:
        for h in hashes:
            try:
                resp = client.get(f"{url}/memories/{h}")
                resp.raise_for_status()
                envelope = resp.json()
            except (httpx.HTTPError, ValueError) as e:  # transport, status, non-JSON
                sys.exit(f"GET /memories/{h}: {e}")
            if not envelope.get("found") or not envelope.get("memory"):
                sys.exit(f"memory {h} is missing (deleted since labelling?)")
            memories[h] = envelope["memory"]
    log(f"fetched {len(memories)} memories for {len(pairs)} pairs")
    return memories


def describe(label: str, m: dict) -> str:
    tags = ", ".join(m["tags"]) if m.get("tags") else "-"
    content = m["content"][:MAX_CONTENT_CHARS]
    return (
        f"Memory {label} (recorded_at={m['created_at']:.0f}; "
        f"type: {m['memory_type']}; tags: {tags}):\n{content}"
    )


def render_pair(a: dict, b: dict) -> str:
    """Port of `judge::render_pair`. Keep byte-identical to the Rust."""
    days = (b["created_at"] - a["created_at"]) / 86_400.0
    if abs(days) < 1.0:
        order = "A and B were recorded within a day of each other"
    elif days > 0.0:
        order = f"A was recorded {days:.0f} days BEFORE B"
    else:
        order = f"A was recorded {-days:.0f} days AFTER B"
    return f"{order}.\n\n{describe('A', a)}\n\n{describe('B', b)}"


# ── Judge call, validation, scoring ──────────────────────────────────────────


def sanitize_reason(s: str) -> str:
    kept = "".join(ch for ch in s if unicodedata.category(ch) != "Cc")
    return kept.strip()[:MAX_REASON_CHARS]


def unjudged(error: str, tokens: tuple[int, int]) -> dict:
    return {
        "verdict": "unjudged",
        "survivor": None,
        "reason": sanitize_reason(error),
        "confidence": None,
        "tokens": tokens,
    }


def validate(raw: Any, tokens: tuple[int, int]) -> dict:
    """Port of `RawVerdict::validate`; failures become `unjudged`."""
    if not isinstance(raw, dict):
        return unjudged("verdict is not a JSON object", tokens)
    verdict, survivor, conf = (
        raw.get("verdict"),
        raw.get("survivor"),
        raw.get("confidence"),
    )
    if verdict not in CLASSES:
        return unjudged(f"unknown verdict {verdict!r}", tokens)
    if survivor not in ("a", "b", None):
        return unjudged(f"unknown survivor {survivor!r}", tokens)
    if (
        isinstance(conf, bool)
        or not isinstance(conf, (int, float))
        or not 0.0 <= conf <= 1.0
    ):
        return unjudged(f"confidence {conf!r} outside 0.0..=1.0", tokens)
    if verdict == "supersession" and survivor is None:
        return unjudged("supersession without a survivor", tokens)
    if verdict in ("coexist", "unrelated"):
        survivor = None
    return {
        "verdict": verdict,
        "survivor": survivor,
        "reason": sanitize_reason(str(raw.get("reason", ""))),
        "confidence": float(conf),
        "tokens": tokens,
    }


def judge_pair(
    client: anthropic.Anthropic, model: str, prompt: str, a: dict, b: dict
) -> dict:
    # An API error that survives the SDK's retries aborts the run: an
    # infrastructure fault must not be scored as a prompt failure.
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=prompt,
        messages=[{"role": "user", "content": render_pair(a, b)}],
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    )
    u = resp.usage
    tokens = (
        (u.input_tokens or 0)
        + (u.cache_creation_input_tokens or 0)
        + (u.cache_read_input_tokens or 0),
        u.output_tokens or 0,
    )
    text = next(
        (
            blk.text.strip()
            for blk in resp.content
            if blk.type == "text" and blk.text.strip()
        ),
        None,
    )
    if text is None:
        return unjudged(f"empty response (stop_reason={resp.stop_reason})", tokens)
    try:
        raw = json.loads(text)
    except ValueError as e:
        return unjudged(f"not JSON (stop_reason={resp.stop_reason}): {e}", tokens)
    return validate(raw, tokens)


def predicted_survivor(pair: Pair, v: dict) -> str | None:
    return {"a": pair.a, "b": pair.b}.get(v["survivor"])


def score(pair: Pair, v: dict) -> float:
    if v["verdict"] != pair.label:
        return 0.0
    if pair.label == "supersession":
        return 1.0 if predicted_survivor(pair, v) == pair.survivor else 0.0
    return 1.0


def ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def metrics(rows: list[tuple[Pair, dict]], model: str) -> dict:
    matrix = Counter((v["verdict"], p.label) for p, v in rows)

    def row(pred: str) -> int:
        return sum(c for (pr, _), c in matrix.items() if pr == pred)

    def col(label: str) -> int:
        return sum(c for (_, lab), c in matrix.items() if lab == label)

    tp = matrix[("supersession", "supersession")]
    surv_rows = [
        (p, v)
        for p, v in rows
        if p.label == "supersession" and v["verdict"] == "supersession"
    ]
    surv_ok = sum(1 for p, v in surv_rows if predicted_survivor(p, v) == p.survivor)
    coexist_esc = sum(matrix[(k, "coexist")] for k in CONFLICT)
    unrelated_esc = sum(matrix[(k, "unrelated")] for k in CONFLICT)
    conf: dict[str, list[float]] = defaultdict(list)
    for p, v in rows:
        if v["confidence"] is not None:
            conf[p.label].append(v["confidence"])
    tin = sum(v["tokens"][0] for _, v in rows)
    tout = sum(v["tokens"][1] for _, v in rows)
    pin, pout = price(model)
    n = len(rows)
    precision = ratio(tp, row("supersession"))
    escalation = ratio(coexist_esc, col("coexist"))
    return {
        "pairs": n,
        "accuracy": sum(score(p, v) for p, v in rows) / n if n else None,
        "precision_supersession": precision,
        "recall_supersession": ratio(tp, col("supersession")),
        "coexist_to_conflict": escalation,
        "coexist_escalated": f"{coexist_esc}/{col('coexist')}",
        "unrelated_to_conflict": ratio(unrelated_esc, col("unrelated")),
        "unrelated_escalated": f"{unrelated_esc}/{col('unrelated')}",
        "survivor_accuracy": ratio(surv_ok, len(surv_rows)),
        "mean_confidence_by_label": {
            k: sum(v) / len(v) for k, v in sorted(conf.items())
        },
        "unjudged": row("unjudged"),
        "tokens_in_per_pair": tin / n if n else None,
        "tokens_out_per_pair": tout / n if n else None,
        "usd_per_pair": (tin * pin + tout * pout) / 1e6 / n if n else None,
        "meets_bar": precision is not None
        and escalation is not None
        and precision >= BAR_PRECISION
        and escalation <= BAR_COEXIST_ESCALATION,
        "confusion_pred_label": {
            f"{pr}->{lab}": c for (pr, lab), c in sorted(matrix.items())
        },
    }


def disagreements(rows: list[tuple[Pair, dict]]) -> list[dict]:
    return [
        {
            "a": p.a,
            "b": p.b,
            "label": p.label,
            "label_survivor": p.survivor_letter(),
            "verdict": v["verdict"],
            "survivor": v["survivor"],
            "confidence": v["confidence"],
            "reason": v["reason"],
        }
        for p, v in rows
        if score(p, v) < 1.0
    ]


# ── Feedback for the reflection model ────────────────────────────────────────

LABEL_RULE = {
    "supersession": (
        "one memory was superseded by the other: the survivor updates or replaces "
        "the loser's claim about the same thing, so a reader relying on the loser "
        "today would be misled"
    ),
    "coexist": (
        "both memories are true at once: snapshots of the same thread or programme "
        "at different times, a plan and its outcome, or different facets of one "
        "topic. Recording history is intended; neither replaces the other"
    ),
    "unrelated": "they share vocabulary or a project name only; neither bears on the other's claim",
    "contradiction": "both claim to be current and cannot both be true",
}
SOURCE_NOTE = {
    "resolved-by-operator": (
        "an operator resolved this pair by superseding one memory with the other "
        "(verified: the loser's superseded_by points at the survivor)"
    ),
    "queue-2026-09-10": "a reviewer read the unresolved queue and labelled the pair",
}


def feedback(pair: Pair, v: dict) -> str:
    survivor = f", survivor Memory {pair.survivor_letter()}" if pair.survivor else ""
    source = SOURCE_NOTE.get(pair.source, pair.source or "the golden set")
    if v["verdict"] == "unjudged":
        return (
            f"No valid verdict was produced ({v['reason']}). The label is "
            f"{pair.label}{survivor}. Answer with the JSON schema only."
        )
    if v["verdict"] == pair.label and score(pair, v) == 1.0:
        return f"Correct: {pair.label}{survivor}. {LABEL_RULE[pair.label]}."
    if v["verdict"] == pair.label:
        pred = "A" if v["survivor"] == "a" else "B"
        return (
            f"Right class, wrong survivor: you named Memory {pred}; the label names "
            f"Memory {pair.survivor_letter()} as the one that stays current. "
            f"Source: {source}."
        )
    text = (
        f"Wrong: you said {v['verdict']} (confidence {v['confidence']:.2f}); the "
        f"label is {pair.label}{survivor}. Source: {source}. Why: "
        f"{LABEL_RULE[pair.label]}."
    )
    if pair.label == "coexist" and v["verdict"] in CONFLICT:
        text += (
            " Calling a later snapshot of the same work a supersession hides the "
            "earlier snapshot from search and erases true history."
        )
    if pair.label == "supersession" and v["verdict"] not in CONFLICT:
        text += (
            " The two are not independent snapshots: the survivor changed the "
            "claim, and the loser now misleads."
        )
    return text


# ── Prompt hygiene ───────────────────────────────────────────────────────────


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower())


def windows(text: str, n: int) -> set[str]:
    t = normalize(text)
    return {t[i : i + n] for i in range(len(t) - n + 1)}


class Hygiene:
    def __init__(self, seed: str, memories: dict[str, dict]):
        self.seed = seed
        self.max_len = 2 * len(seed)
        self.contents = [normalize(m["content"]) for m in memories.values()]
        self.seed_idents = set(IDENT_RE.findall(seed))

    def leak_hits(
        self, prompt: str, n: int = LEAK_WINDOW, ignore_seed: bool = True
    ) -> set[str]:
        """`n`-char windows of the prompt that occur in any golden memory.

        With `ignore_seed`, windows already present in the seed prompt (text we
        authored) are not counted, so the check targets what tuning added.
        """
        wins = windows(prompt, n)
        if ignore_seed:
            wins -= windows(self.seed, n)
        hits: set[str] = set()
        if not wins:
            return hits
        for c in self.contents:
            for i in range(len(c) - n + 1):
                w = c[i : i + n]
                if w in wins:
                    hits.add(w)
        return hits

    def ident_hits(self, prompt: str) -> set[str]:
        """Identifier-shaped tokens added by tuning that occur in a memory."""
        added = set(IDENT_RE.findall(prompt)) - self.seed_idents
        return {t for t in added if any(t.lower() in c for c in self.contents)}

    def problems(self, prompt: str) -> list[str]:
        out = []
        missing = [p for p in REQUIRED_PHRASES if p not in prompt]
        if missing:
            out.append(f"missing required text: {missing}")
        if len(prompt) > self.max_len:
            out.append(f"length {len(prompt)} chars > 2x seed ({self.max_len})")
        infra = INFRA_RE.findall(prompt)
        if infra:
            out.append(f"infrastructure names: {sorted(set(infra))}")
        hits = self.leak_hits(prompt)
        if hits:
            out.append(f"{len(hits)} golden-memory spans of {LEAK_WINDOW}+ chars")
        idents = self.ident_hits(prompt)
        if idents:
            out.append(f"{len(idents)} identifiers from golden memories")
        return out


# ── Spend ledger ─────────────────────────────────────────────────────────────


def price(model: str) -> tuple[float, float]:
    for prefix, p in PRICE_PER_MTOK.items():
        if model.startswith(prefix):
            return p
    sys.exit(f"no list price for {model}; add it to PRICE_PER_MTOK")


class Ledger:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])

    def add(self, role: str, model: str, tokens_in: int, tokens_out: int) -> None:
        row = self.rows[(role, model)]
        row[0] += 1
        row[1] += tokens_in
        row[2] += tokens_out

    def usd(self, role: str | None = None) -> float:
        total = 0.0
        for (r, model), (_, tin, tout) in self.rows.items():
            if role in (None, r):
                pin, pout = price(model)
                total += (tin * pin + tout * pout) / 1e6
        return total

    def summary(self) -> dict:
        return {
            f"{r}:{m}": {
                "calls": c,
                "tokens_in": tin,
                "tokens_out": tout,
                "usd": round(self.usd(r), 4),
            }
            for (r, m), (c, tin, tout) in sorted(self.rows.items())
        } | {"total_usd": round(self.usd(), 4)}


# ── GEPA adapter, reflection model, stopper ──────────────────────────────────


def judge_batch(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    memories: dict,
    batch: list[Pair],
) -> list[dict]:
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        return list(
            pool.map(
                lambda p: judge_pair(
                    client, model, prompt, memories[p.a], memories[p.b]
                ),
                batch,
            )
        )


class JudgeAdapter:
    # GEPA reads these directly: None means "use the default proposer".
    propose_new_texts = None

    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        memories: dict[str, dict],
        train: list[Pair],
        val: list[Pair],
        hygiene: Hygiene,
        ledger: Ledger,
        max_usd: float,
    ):
        self.client, self.model, self.memories = client, model, memories
        self.train_ids = {p.id for p in train}
        self.val_ids = {p.id for p in val}
        self.pairs = {p.id: p for p in (*train, *val)}
        self.validated: set[str] = set()
        self.hygiene, self.ledger, self.max_usd = hygiene, ledger, max_usd
        self.prompts: dict[str, str] = {}
        self.records: dict[str, dict[int, dict]] = defaultdict(dict)
        self.val_evals: list[tuple[str, dict]] = []
        self.judge_calls = 0
        self.rejected = 0

    def evaluate(
        self, batch: list[Pair], candidate: dict[str, str], capture_traces: bool = False
    ):
        from gepa.core.adapter import EvaluationBatch

        prompt = candidate["system_prompt"]
        key = sha(prompt)
        self.prompts[key] = prompt
        problems = self.hygiene.problems(prompt)
        if problems:
            self.rejected += 1
            log(f"candidate {key[:8]} rejected unread: {'; '.join(problems)}")
            v = unjudged("prompt rejected: " + "; ".join(problems), (0, 0))
            traj = (
                [{"pair": p, "verdict": v} for p in batch] if capture_traces else None
            )
            return EvaluationBatch(
                outputs=[v] * len(batch),
                scores=[0.0] * len(batch),
                trajectories=traj,
                num_metric_calls=0,
            )
        # Stoppers run between iterations; this guard bounds the overshoot of
        # one full validation pass at 10 % over the cap.
        if self.ledger.usd() > self.max_usd * 1.1:
            raise RuntimeError(
                f"spend guard: ${self.ledger.usd():.2f} > 1.1 x ${self.max_usd}"
            )
        verdicts = judge_batch(self.client, self.model, prompt, self.memories, batch)
        for p, v in zip(batch, verdicts):
            self.ledger.add("judge", self.model, *v["tokens"])
            self.records[key][p.id] = v
        self.judge_calls += len(batch)
        scores = [score(p, v) for p, v in zip(batch, verdicts)]
        self.note_validation(key)
        traj = (
            [
                {"pair": p, "verdict": v, "score": s}
                for p, v, s in zip(batch, verdicts, scores)
            ]
            if capture_traces
            else None
        )
        return EvaluationBatch(
            outputs=verdicts,
            scores=scores,
            trajectories=traj,
            num_metric_calls=len(batch),
        )

    def note_validation(self, key: str) -> None:
        """Record a candidate's validation metrics once every val pair is judged.

        Coverage, not batch identity: GEPA may hand the validation set over in
        pieces, and the report must not depend on how it batches.
        """
        if key in self.validated or not self.val_ids <= self.records[key].keys():
            return
        self.validated.add(key)
        rows = [(self.pairs[i], self.records[key][i]) for i in sorted(self.val_ids)]
        m = metrics(rows, self.model)
        self.val_evals.append((key, m))
        log(
            f"val eval #{len(self.val_evals)} {key[:8]}: acc={fmt(m['accuracy'])} "
            f"P(sup)={fmt(m['precision_supersession'])} "
            f"coexist->conflict={m['coexist_escalated']} "
            f"bar={'MET' if m['meets_bar'] else 'not met'} "
            f"spend=${self.ledger.usd():.2f}"
        )

    def dump_records(self, path: Path) -> None:
        """Every verdict of the run, one JSON line per (candidate, pair)."""
        with path.open("w", encoding="utf-8") as f:
            for key, by_pair in self.records.items():
                for pid, v in sorted(by_pair.items()):
                    row = {
                        "prompt_sha": key,
                        "pair": pid,
                        "label": self.pairs[pid].label,
                    }
                    row |= {
                        k: v[k] for k in ("verdict", "survivor", "confidence", "reason")
                    }
                    row["score"] = score(self.pairs[pid], v)
                    f.write(json.dumps(row) + "\n")

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        records = []
        for t in eval_batch.trajectories or []:
            pair: Pair = t["pair"]
            if pair.id not in self.train_ids:
                raise AssertionError(
                    f"validation pair {pair.id} reached the reflection model"
                )
            v = t["verdict"]
            records.append(
                {
                    "Inputs": {
                        "pair": render_pair(
                            self.memories[pair.a], self.memories[pair.b]
                        )
                    },
                    "Generated Outputs": json.dumps(
                        {
                            k: v[k]
                            for k in ("verdict", "survivor", "reason", "confidence")
                        }
                    ),
                    "Feedback": feedback(pair, v),
                }
            )
        return {"system_prompt": records}


class Reflector:
    def __init__(self, client: anthropic.Anthropic, model: str, ledger: Ledger):
        self.client, self.model, self.ledger = client, model, ledger

    def __call__(self, prompt) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=16_000,
            messages=[{"role": "user", "content": prompt}],
        )
        u = resp.usage
        self.ledger.add(
            "reflection",
            self.model,
            (u.input_tokens or 0)
            + (u.cache_creation_input_tokens or 0)
            + (u.cache_read_input_tokens or 0),
            u.output_tokens or 0,
        )
        if resp.stop_reason != "end_turn":
            log(f"reflection stop_reason={resp.stop_reason}")
        return "".join(blk.text for blk in resp.content if blk.type == "text")


def qualifies(m: dict, seed_acc: float) -> bool:
    """A tuned candidate is landable when it meets both bars without losing
    accuracy against the seed."""
    return bool(m["meets_bar"]) and (m["accuracy"] or 0.0) >= seed_acc


class Stopper:
    """Stop on spend, or once a *tuned* candidate is landable.

    The seed never stops the run: with 14 coexist pairs the bar resolves in
    steps of one pair, so the seed clearing it on the split says more about
    the split than the prompt (it escalated 5/35 on the full set).
    """

    def __init__(self, adapter: JudgeAdapter, ledger: Ledger, max_usd: float):
        self.adapter, self.ledger, self.max_usd = adapter, ledger, max_usd
        self.reason: str | None = None

    def __call__(self, state) -> bool:
        if self.ledger.usd() >= self.max_usd:
            self.reason = "budget"
            log(f"stop: spend ${self.ledger.usd():.2f} >= ${self.max_usd}")
            return True
        evals = self.adapter.val_evals
        if len(evals) < 2:
            return False
        seed_acc = evals[0][1]["accuracy"]
        if any(qualifies(m, seed_acc) for _, m in evals[1:]):
            self.reason = "bar"
            log("stop: a tuned candidate meets both bars on the validation split")
            return True
        return False


def choose_best(adapter: JudgeAdapter) -> tuple[str, dict]:
    """The most accurate landable tuned candidate; failing that, the most
    accurate candidate of all (seed included), so the report shows what
    tuning did. Ties go to the earliest."""
    evals = adapter.val_evals
    seed_acc = evals[0][1]["accuracy"]
    landable = [e for e in evals[1:] if qualifies(e[1], seed_acc)]
    return max(landable or evals, key=lambda e: e[1]["accuracy"] or 0.0)


# ── Commands ─────────────────────────────────────────────────────────────────


def make_client(url: str, key: str) -> anthropic.Anthropic:
    # No redirects: httpx strips Authorization on a cross-origin redirect but
    # not x-api-key, so a redirecting endpoint could take the key elsewhere.
    # Same policy as the Rust transport.
    return anthropic.Anthropic(
        base_url=url,
        api_key=key,
        timeout=JUDGE_TIMEOUT_S,
        max_retries=2,
        http_client=anthropic.DefaultHttpxClient(follow_redirects=False),
    )


def cmd_split(args) -> None:
    if SPLIT_FILE.exists() and not args.force:
        sys.exit(f"{SPLIT_FILE} exists; pass --force to overwrite")
    pairs = load_fixture()
    split = make_split(pairs, args.seed, TRAIN_FRAC)
    SPLIT_FILE.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {SPLIT_FILE}: {split['counts']}")


def fmt(x: object) -> str:
    if isinstance(x, float):
        return f"{x:.3f}"
    return json.dumps(x) if isinstance(x, dict) else str(x)


def write_report(out: Path, report: dict) -> None:
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    lines = [f"# Judge prompt tune: {report['judge_model']}", ""]
    lines += [f"- {k}: {v}" for k, v in report["run"].items()]
    lines += ["", "## Validation split (never seen by the reflection model)", ""]
    keys = [k for k in report["seed_val"] if k != "confusion_pred_label"]
    lines += ["| metric | seed | best |", "|---|---|---|"]
    for k in keys:
        s, b = report["seed_val"].get(k), report["best_val"].get(k)
        lines.append(f"| {k} | {fmt(s)} | {fmt(b)} |")
    lines += [
        "",
        "## Spend",
        "",
        "```",
        json.dumps(report["spend"], indent=2),
        "```",
        "",
    ]
    lines += [
        "## Candidates with a full validation pass",
        "",
        "| # | prompt | acc | P(sup) | coexist->conflict | bar |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(report["candidates"], 1):
        m = c["val"]
        lines.append(
            f"| {i} | {c['sha'][:8]} | {fmt(m['accuracy'])} | {fmt(m['precision_supersession'])} | "
            f"{m['coexist_escalated']} | {'MET' if m['meets_bar'] else '-'} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_tune(args) -> None:
    import gepa

    out = HERE / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_fixture()
    train, val = load_split(pairs)
    seed = seed_prompt()
    judge_model = os.environ.get("JUDGE_MODEL", "claude-sonnet-5")
    reflection_model = os.environ.get("REFLECTION_MODEL", "claude-opus-5")
    price(judge_model), price(reflection_model)  # fail before the first paid call
    memories = fetch_memories(pairs)
    hygiene = Hygiene(seed, memories)
    literal_hits = hygiene.leak_hits(seed, LITERAL_WINDOW, ignore_seed=False)
    log(
        f"seed prompt: {len(seed)} chars; literal {LITERAL_WINDOW}-char overlaps "
        f"with golden memories: {len(literal_hits)}"
    )
    ledger = Ledger()
    judge_url, judge_key = env("JUDGE_URL"), env("JUDGE_API_KEY")
    judge_client = make_client(judge_url, judge_key)
    refl_url = os.environ.get("REFLECTION_URL")
    refl_key = os.environ.get("REFLECTION_API_KEY")
    if bool(refl_url) != bool(refl_key):  # never send one origin's key to another
        sys.exit("set REFLECTION_URL and REFLECTION_API_KEY together, or neither")
    reflection_client = make_client(
        refl_url or judge_url, refl_key or judge_key
    ).with_options(timeout=600)
    adapter = JudgeAdapter(
        judge_client, judge_model, memories, train, val, hygiene, ledger, args.max_usd
    )
    stopper = Stopper(adapter, ledger, args.max_usd)
    log(
        f"tune: judge={judge_model} reflection={reflection_model} train={len(train)} val={len(val)} "
        f"minibatch={MINIBATCH} max_metric_calls={args.max_metric_calls} max_usd={args.max_usd}"
    )
    started = time.time()
    error = None
    result = None
    # No cache_evaluation: GEPA keys its cache by positional example id, and the
    # train and validation loaders share that id space, so a child scored on
    # train pair 3 would be credited with that score for validation pair 3.
    try:
        result = gepa.optimize(
            seed_candidate={"system_prompt": seed},
            trainset=train,
            valset=val,
            adapter=adapter,
            reflection_lm=Reflector(reflection_client, reflection_model, ledger),
            reflection_prompt_template=reflection_template(seed),
            reflection_minibatch_size=MINIBATCH,
            max_metric_calls=args.max_metric_calls,
            stop_callbacks=stopper,
            perfect_score=1.0,
            skip_perfect_score=True,
            seed=args.seed,
            run_dir=str(out / "gepa"),
            track_best_outputs=True,
            display_progress_bar=False,
            raise_on_exception=True,
        )
    except (Exception, KeyboardInterrupt) as e:  # report what we have, then fail
        error = f"{type(e).__name__}: {e}"
        log(f"gepa aborted: {error}")
    # Paid verdicts and spend are written before any exit path.
    adapter.dump_records(out / "records.jsonl")
    (out / "spend.json").write_text(json.dumps(ledger.summary(), indent=2) + "\n")
    if not adapter.val_evals:
        sys.exit(f"no validation pass completed ({error})")
    _, seed_val = adapter.val_evals[0]
    best_key, best_val = choose_best(adapter)
    best_prompt = adapter.prompts[best_key]
    (out / "best_prompt.txt").write_text(best_prompt, encoding="utf-8")
    (out / "seed_prompt.txt").write_text(seed, encoding="utf-8")
    if hygiene.problems(best_prompt):  # evaluate() scores such a prompt 0; cannot win
        sys.exit(f"best prompt fails hygiene: {hygiene.problems(best_prompt)}")
    stopped_on = stopper.reason or (
        "metric_calls"
        if adapter.judge_calls >= args.max_metric_calls
        else error or "gepa"
    )
    report = {
        "judge_model": judge_model,
        "reflection_model": reflection_model,
        "run": {
            "stopped_on": stopped_on,
            "error": error,
            "judge_calls": adapter.judge_calls,
            "candidates_validated": len(adapter.val_evals),
            "candidates_rejected_by_hygiene": adapter.rejected,
            "gepa_best_is_report_best": bool(result)
            and sha(result.best_candidate["system_prompt"]) == best_key,
            "seed_len": len(seed),
            "best_len": len(best_prompt),
            "seed_literal_12char_overlaps": len(literal_hits),
            "best_literal_12char_overlaps": len(
                hygiene.leak_hits(best_prompt, LITERAL_WINDOW, ignore_seed=False)
            ),
            "best_12char_overlaps_beyond_seed": len(
                hygiene.leak_hits(best_prompt, LITERAL_WINDOW)
            ),
            "minutes": round((time.time() - started) / 60, 1),
            "minibatch": MINIBATCH,
            "gepa_seed": args.seed,
        },
        "seed_val": seed_val,
        "best_val": best_val,
        "best_sha": best_key,
        "spend": ledger.summary(),
        "candidates": [{"sha": k, "val": m} for k, m in adapter.val_evals],
    }
    write_report(out, report)
    log(f"report: {out / 'report.md'}; best prompt: {out / 'best_prompt.txt'}")
    if error:
        sys.exit(1)


def cmd_eval(args) -> None:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    model = os.environ.get("JUDGE_MODEL", "claude-sonnet-5")
    price(model)  # fail before the first paid call
    pairs = load_fixture()
    train, val = load_split(pairs)
    chosen = {"val": val, "train": train, "all": pairs}[args.pairs]
    memories = fetch_memories(pairs)
    hygiene = Hygiene(seed_prompt(), memories)
    problems = hygiene.problems(prompt)
    if problems:  # eval measures; landing is gated in judge.rs and by a human
        log("prompt hygiene problems: " + "; ".join(problems))
    client = make_client(env("JUDGE_URL"), env("JUDGE_API_KEY"))
    ledger = Ledger()
    verdicts = judge_batch(client, model, prompt, memories, chosen)
    for v in verdicts:
        ledger.add("judge", model, *v["tokens"])
    rows = list(zip(chosen, verdicts))
    result = {
        "judge_model": model,
        "pairs": args.pairs,
        "prompt_sha256": sha(prompt),
        "hygiene_problems": problems,
        "metrics": metrics(rows, model),
        "spend": ledger.summary(),
        "disagreements": disagreements(rows),
    }
    out = HERE / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"eval_{args.pairs}_{sha(prompt)[:8]}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k != "disagreements"}, indent=2)
    )
    print(f"disagreements: {len(result['disagreements'])} (see {path})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split", help="write split.json")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_split)
    t = sub.add_parser("tune", help="run GEPA")
    t.add_argument(
        "--run", required=True, help="run name; outputs go under runs/<name>/"
    )
    t.add_argument("--max-metric-calls", type=int, default=600)
    t.add_argument("--max-usd", type=float, default=30.0)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(fn=cmd_tune)
    e = sub.add_parser("eval", help="score one prompt file")
    e.add_argument("--prompt-file", required=True)
    e.add_argument("--pairs", choices=("val", "train", "all"), default="val")
    e.add_argument("--run", default="eval", help="outputs go under runs/<name>/")
    e.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
