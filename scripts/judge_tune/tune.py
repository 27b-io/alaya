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
    eval    judge val / train / all golden pairs, or resolved spot-check rows,
            over k passes with one judge (Anthropic, OpenAI-compatible or Jev),
            optionally on scrubbed text, with 95 % Wilson intervals
    rows    resolve spot-check rows given as hash prefixes to full pairs
    compare score every judge of a run against the current fixture, from
            their saved verdicts alone: per-class rates, agreement, vote value

Memory contents are fetched from Ālaya at run time; this script writes none of
them to disk. The run directory (gitignored) holds model output about them:
candidate prompts, verdict reasons and GEPA's own logs.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import importlib
import ipaddress
import json
import math
import os
import random
import re
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
SPEND_CHECK_EVERY = 20  # eval checks --max-usd between chunks of this many calls

# Decoding regimes for `eval --regime`: extra Messages API parameters on top of
# the production request. `default` is what JudgeClient sends (no `thinking`,
# so claude-sonnet-5 runs adaptive thinking). `thinking-off` disables it.
# Neither sets `temperature`: claude-sonnet-5 rejects any value but the
# default 1.0, with or without thinking (400 "`temperature` is deprecated for
# this model", checked 2026-10-01), so a temperature-0 regime cannot be sent.
REGIMES: dict[str, dict] = {
    "default": {},
    "thinking-off": {"thinking": {"type": "disabled"}},
}
ABSTAIN = "abstain"

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
# at the full input rate: an upper bound, never an under-count. The OpenAI
# and Gemini rows are a LiteLLM proxy's `/model/info` prices (2026-10-03);
# their output counts include reasoning tokens. Jev bills input only.
PRICE_PER_MTOK = {
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "gpt-6-sol": (2.0, 10.0),
    "gemini-3.8-flash": (0.75, 3.75),
    "jev": (0.042, 0.0),
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
    # A trailing CR or space would reach an HTTP header, and httpx echoes an
    # illegal header value, key included, in its error message.
    if not all("!" <= c <= "~" for c in value):
        sys.exit(f"{key} holds whitespace or a non-printable character")
    return value


# ── Egress ───────────────────────────────────────────────────────────────────
# Every origin that gets a key or memory text, judged as alaya-server judges
# JUDGE_URL at boot: https, or plain http only to a cluster-local host. The
# check certifies the URL's host, so it holds only while every client dials
# that host itself: each one is built with trust_env=False, or an HTTP_PROXY /
# ALL_PROXY from the environment would carry the key and the text elsewhere.

PRIVATE_NETS = tuple(
    ipaddress.ip_network(n)
    for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
)
ANTHROPIC_ORIGIN = "https://api.anthropic.com"


def is_cluster_local(url: str) -> bool:
    """Port of alaya-server's `is_cluster_local`: a private or loopback IP,
    localhost, `*.svc`, `*.svc.cluster.local`, `*.internal`, or a single-label
    service name. The host is the one the client connects to, never userinfo."""
    host = urlsplit(url).hostname or ""
    if host == "localhost" or host.endswith(
        (".svc", ".svc.cluster.local", ".internal")
    ):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # "16843009" and "0x01010101" have no dot, yet the resolver dials them
        # as IPv4 addresses, and alaya-server's URL parser reads them so too.
        try:
            ip = ipaddress.IPv4Address(socket.inet_aton(host))
        except OSError:
            return bool(host) and "." not in host
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_loopback or any(ip in net for net in PRIVATE_NETS)


def egress_url(key: str) -> str:
    """An API origin from the environment that may carry a key: https, or
    http to a cluster-local host only."""
    url = env(key)
    scheme = urlsplit(url).scheme
    if scheme == "https" or (scheme == "http" and is_cluster_local(url)):
        return url
    sys.exit(f"{key} must be https, or http to a cluster-local host")


def origin(url: str) -> str:
    u = urlsplit(url)
    return f"{u.scheme}://{u.hostname}:{u.port or {'https': 443, 'http': 80}.get(u.scheme)}"


def unscrubbed_ok(model: str, url: str) -> None:
    """Unscrubbed pairs go only to a Claude model at an origin approved for
    them: Anthropic's API, or one named in UNSCRUBBED_JUDGE_ORIGINS
    (comma-separated, the production judge's own proxy). The model alone is
    no proof: a third-party proxy can serve a `claude-*` name."""
    if not model.startswith("claude-"):
        sys.exit(f"{model} would read unscrubbed memory text; only claude-* may")
    listed = os.environ.get("UNSCRUBBED_JUDGE_ORIGINS", "").split(",")
    allowed = {origin(ANTHROPIC_ORIGIN)} | {
        origin(o.strip()) for o in listed if o.strip()
    }
    if origin(url) not in allowed:
        sys.exit(
            f"unscrubbed pairs may go only to {sorted(allowed)}, not {origin(url)}; "
            "name the production judge's origin in UNSCRUBBED_JUDGE_ORIGINS"
        )


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
    sub_label: str | None = None
    tags: tuple[str, ...] = ()
    # A label still awaiting adjudication: scored per pass, kept out of the
    # headline bars.
    disputed: bool = False

    def survivor_letter(self) -> str | None:
        if self.survivor is None:
            return None
        return "A" if self.survivor == self.a else "B"


def load_fixture() -> list[Pair]:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return [
        Pair(
            i,
            p["a"],
            p["b"],
            p["label"],
            p.get("survivor"),
            p.get("source", ""),
            p.get("sub_label"),
            tuple(p.get("tags", ())),
            "dispute" in p,
        )
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

    def one(m: re.Match) -> str:
        c = m.group(1)
        if c is None:  # `\` + newline: rustc drops it and the next line's ASCII indent
            return ""
        if c not in RUST_ESCAPES:  # a pass-through would tune a prompt never sent
            sys.exit(
                f"unsupported Rust escape {m.group(0)!r}; "
                f"unescape_rust knows {sorted(RUST_ESCAPES)}"
            )
        return RUST_ESCAPES[c]

    # One pass, left to right, so `\\` before a newline is a backslash, not a continuation.
    return re.sub(r"\\(?:\n[ \t\r\n]*|(.))", one, literal)


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


def alaya_client() -> httpx.Client:
    return httpx.Client(
        base_url=egress_url("ALAYA_URL").rstrip("/"),
        headers={"Authorization": f"Bearer {env('ALAYA_API_KEY')}"},
        timeout=120,
        transport=httpx.HTTPTransport(retries=3),
        trust_env=False,  # no proxy from the environment: see Egress
    )


def fetch_memories(pairs: list[Pair]) -> dict[str, dict]:
    hashes = sorted({h for p in pairs for h in (p.a, p.b)})
    memories: dict[str, dict] = {}
    with alaya_client() as client:
        for h in hashes:
            try:
                resp = client.get(f"/memories/{h}")
                if resp.status_code == 404:  # the server maps found:false to 404
                    sys.exit(f"memory {h} is missing (deleted since labelling?)")
                resp.raise_for_status()
                memory = resp.json().get("memory")
            except httpx.HTTPStatusError as e:  # str(e) echoes the URL, userinfo too
                sys.exit(f"GET /memories/{h}: HTTP {e.response.status_code}")
            except (httpx.HTTPError, ValueError) as e:  # never str(e): see env()
                sys.exit(f"GET /memories/{h}: {type(e).__name__}")
            if not memory:
                sys.exit(f"GET /memories/{h}: 200 without a memory body")
            memories[h] = memory
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


# ── Scrub ────────────────────────────────────────────────────────────────────
# Five classes of text are replaced by a typed placeholder before a pair is
# rendered for any judge but the unscrubbed control run. Rules run in order: a
# private-key block spans lines, a URL holds a host and an email a domain, so
# each goes before the patterns that would split it. No placeholder matches a
# rule, so scrubbing twice changes nothing and `Scrubber.leaks` can re-check
# rendered text.

HOST_TLDS = (
    "com|net|org|io|ai|dev|app|cloud|goog|co|tech|xyz|info|biz|edu|gov|us|uk|au|nz"
    "|de|ca|local|localhost|internal|svc|lan|home|arpa"
)
SECRET_TOKENS = (
    r"sk-[\w-]{20,}"  # Anthropic, OpenAI and LiteLLM keys
    r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}"
    r"|xox[abprs]-[A-Za-z0-9-]{10,}"
    r"|(?:AKIA|ASIA)[0-9A-Z]{16}|AIza[\w-]{35}|GOCSPX-[\w-]{20,}"
    r"|glpat-[\w-]{20,}|hf_[A-Za-z0-9]{30,}|npm_[A-Za-z0-9]{36}|pypi-[\w-]{50,}"
    r"|[sr]k_(?:live|test)_\w{20,}|whsec_[A-Za-z0-9+/=]{20,}"
    r"|ops_[\w-]{20,}|tskey-[A-Za-z0-9-]{10,}"
    r"|eyJ[\w-]{8,}\.[\w-]{8,}\.[\w-]{8,}"  # JWT
)
# A key name that names a secret: API_KEY, client_secret, x-api-key. The
# look-behind starts it at the head of a name and the possessive `++` never
# gives a character back, so a long run costs linear time, not quadratic.
SECRET_NAME = (
    r"(?<![\w.-])(?=[\w.-]*?(?:api[_-]?key|account[_-]?key|[_-]key|token|secret"
    r"|passw(?:or)?d|passphrase|pwd|credential))[\w.-]++"
)
# What may close a key before its separator: a quote, a bracket or both, as
# in os.environ['X_KEY'] = '...'. Every rule that finds a key's separator
# takes it from here, so none of them can miss a closer another accepts.
KEY_CLOSE = r"[\"']?\]?"
SECRET_KEY_NAME = rf"{SECRET_NAME}{KEY_CLOSE}\s*(?:=>|[:=])\s*[\"']?"
# What a URL may hold; a URL ends at the first character outside it.
URL_CHAR = r"[^\s<>\"'`)\]]"
SEPARATOR = r"(?:=>|[:=])"
# A secret-named key that ends a URL: a quote, bracket or space before its
# separator, or a separator the URL cannot run past. Its value lies outside
# the URL (`.../DB_PASSWORD: x`, `Get "...?token=x": err`), so the URL rule
# takes the key, the separator and the value too. A key whose value stays
# inside (`?api_key=v`, a password with `=` padding before `@host`) ends
# nothing; and a URL is never cut short, since its last run may be the secret.
URL_ENDING_KEY = rf"{SECRET_NAME}(?:(?=[\"'\]\s]){KEY_CLOSE}\s*+{SEPARATOR}|{SEPARATOR}(?!{URL_CHAR}))"
# A value's characters, and a placeholder a rule before the key rules wrote.
VALUE_CHAR = r"[^\s\"'<>,;]"
PLACEHOLDER = r"(?:<(?:secret|url|email|ip)>)"
# A value that ran into the next key's name took that key's separator with it,
# leaving the next value with no key in front: the key rules take it too. Only
# a secret-named key's value may sit on a later line (`.../DB_PASSWORD:\n  x`);
# any other chain stays on its line, so a value ending in `=` padding or `:`
# takes nothing from the lines after it. `>?`: the separator may be `=>`.
SECRET_KEY_ENDS = (
    "key", "keys", "token", "tokens", "secret", "secrets", "password", "passwords",
    "passwd", "pwd", "passphrase", "credential", "credentials",
)  # fmt: skip
ENDS_SECRET_KEY = "|".join(f"(?<={w})" for w in SECRET_KEY_ENDS)
# The value took the separator too, and with it a `]` (a value never holds a
# quote, so only the bracket of KEY_CLOSE can come along).
ENDS_SECRET_KEY_SEP = "|".join(f"(?<={w}[:=])|(?<={w}\\][:=])" for w in SECRET_KEY_ENDS)
NEXT_VALUES = (
    rf"(?:(?:(?:{ENDS_SECRET_KEY_SEP})>?\s*+"
    rf"|(?:{ENDS_SECRET_KEY}){KEY_CLOSE}\s*+(?:=>|[:=])\s*+"
    r"|(?<=[:=])>?[ \t]*+|[ \t]++(?:=>|[:=])[ \t]*+)[\"']?"
    rf"(?:{VALUE_CHAR}|{PLACEHOLDER})++)*+"
)
# (class, pattern). A `keep` group survives in front of the placeholder.
SCRUB_RULES: tuple[tuple[str, re.Pattern], ...] = (
    (
        "secret",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----.*?"
            r"(?:-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----|\Z)",
            re.S,
        ),
    ),
    (  # a scheme may follow a dash, as in ${VAR:-redis://...}; a secret key
        # that ends the URL comes along with its value (see URL_ENDING_KEY)
        "url",
        re.compile(
            r"(?<![a-z0-9+.-])(?P<keep>[0-9+.-]*+)[a-z][a-z0-9+.-]*+://"
            rf"(?={URL_CHAR})(?:(?!{URL_ENDING_KEY}){URL_CHAR})*+"
            rf"(?:(?={URL_ENDING_KEY}){SECRET_NAME}{KEY_CLOSE}\s*+{SEPARATOR}\s*+[\"']?"
            rf"(?:{VALUE_CHAR}|{PLACEHOLDER})*+{NEXT_VALUES})?",
            re.I,
        ),
    ),
    # The domain ends in letters, so a package pin (`pkg@0.1.4`) is no email.
    (
        "email",
        re.compile(r"(?<![\w.+-])[\w.+-]++@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b"),
    ),
    (
        "ip",
        re.compile(
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
            r"|\b(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}\b"
            r"|\b(?:[0-9a-f]{1,4}:){2,6}:(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4})*)?",
            re.I,
        ),
    ),
    ("secret", re.compile(rf"\b(?:{SECRET_TOKENS})")),
    ("secret", re.compile(r"(?P<keep>\bBearer\s+)[\w.~+/=-]{8,}")),
    (
        "secret",
        re.compile(
            r"(?P<keep>\bauthorization[\"']?\s*[:=]\s*[\"']?\w+\s+)[^\s\"'<>]{8,}",
            re.I,
        ),
    ),
    (  # the user-and-password argument of curl -u or --user
        "secret",
        re.compile(r"(?P<keep>(?:^|\s)(?:-u\s*|--user[=\s]\s*)[^\s:<>]+:)[^\s<>]{6,}"),
    ),
    (  # a command-line flag naming a secret, then its value: --password VALUE
        "secret",
        re.compile(
            r"(?P<keep>(?<![\w-])--(?=[\w-]*?(?:key|token|secret|passw(?:or)?d))"
            r"[\w-]++[=\s]\s*[\"']?)[^\s\"'<>]{6,}",
            re.I,
        ),
    ),
    # A secret-named key's value: first one that stops at a bracket, so a call
    # such as `token => login(password: "...")` cannot hide the inner key; then
    # any value, brackets included. The flag rule runs first: a placeholder it
    # wrote inside a key's value after them would change on a second scrub.
    (
        "secret",
        re.compile(
            rf"(?P<keep>{SECRET_KEY_NAME})[^\s\"'<>,;()\[\]{{}}]{{8,}}{NEXT_VALUES}",
            re.I,
        ),
    ),
    (  # ...and what is left of a value an earlier placeholder split: the
        # tail a bracket left, or the rest around an IP, URL or email in it
        "secret",
        re.compile(
            rf"(?P<keep>{SECRET_KEY_NAME})(?:"
            rf"(?:{VALUE_CHAR}++{PLACEHOLDER}++|{PLACEHOLDER}++{VALUE_CHAR})"
            rf"(?:{VALUE_CHAR}++|{PLACEHOLDER})*+|{VALUE_CHAR}{{8,}}){NEXT_VALUES}",
            re.I,
        ),
    ),
    (
        "host",
        re.compile(
            # At the head of a name only, so a long dotted run is linear; a
            # leading dash (`-hdb.example.com`) or dot (`*.example.net`) is skipped.
            rf"(?<![a-z0-9.-])(?P<keep>-*+)\.?(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.++)+"
            rf"(?:{HOST_TLDS})\b"
            r"(?!-|\.[a-z0-9])",
            re.I,
        ),
    ),
)
SCRUB_CLASSES = ("host", "ip", "url", "email", "secret")
# Only for the fail-closed check: a secret key and its value right after a URL
# placeholder mean some rule cut a URL short, leaving a value no key rule saw.
ORPHANED_VALUE = re.compile(
    rf"<url>{SECRET_NAME}[\"']?\]?\s*+{SEPARATOR}\s*+[\"']?{VALUE_CHAR}{{8,}}", re.I
)


class Scrubber:
    """Replace the scrub classes with `<class>` placeholders, counting each.

    A host name with no dot cannot be told from a word by pattern, so bare
    names (a machine list, kept out of git) come in through `host_names`.
    """

    def __init__(self, host_names: Iterable[str] = ()) -> None:
        self.rules = list(SCRUB_RULES)
        names = sorted({n.strip() for n in host_names if n.strip()}, key=len)
        if names:
            alt = "|".join(map(re.escape, reversed(names)))  # longest first
            self.rules.append(("host", re.compile(rf"\b(?:{alt})\b", re.I)))
        self.counts: Counter[str] = Counter()

    def __call__(self, text: str) -> str:
        for cls, pattern in self.rules:
            text, n = pattern.subn(
                lambda m, cls=cls: (m.groupdict().get("keep") or "") + f"<{cls}>",
                text,
            )
            self.counts[cls] += n
        return text

    def memory(self, m: dict) -> dict:
        """`m` with its content and tags scrubbed: the fields `describe` sends."""
        return m | {
            "content": self(m["content"]),
            "tags": [self(t) for t in m.get("tags") or ()],
        }

    def leaks(self, text: str) -> list[str]:
        """Classes some rule still matches in `text`: empty when it is clean."""
        found = {cls for cls, pattern in self.rules if pattern.search(text)}
        if ORPHANED_VALUE.search(text):
            found.add("secret")
        return sorted(found)


def read_file(path: str | Path) -> str:
    """A file's text, or an exit that names the file."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        sys.exit(f"cannot read {path}: {e}")


def parse_json(text: str, where: str) -> Any:
    """`text` as JSON, or an exit that names where it came from."""
    try:
        return json.loads(text)
    except ValueError as e:
        sys.exit(f"{where}: not JSON ({e})")


def read_host_names(path: str | None) -> list[str]:
    if not path:
        return []
    lines = read_file(path).splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


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


def request_params(model: str, regime: str = "default") -> dict:
    """Every Messages API parameter the judge sends except `system` and `messages`."""
    return {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "output_config": {"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
        **REGIMES[regime],
    }


def judge_pair(
    client: anthropic.Anthropic,
    model: str,
    prompt: str,
    text: str,
    regime: str = "default",
    *,
    refusal_fails: bool = False,
) -> dict:
    """Judge one rendered pair. An API error propagates: the caller decides
    whether it aborts the run (tune) or fails the pair (eval). A refusal's text
    is read as production reads it, unless `refusal_fails` makes the refusal an
    `ApiError`, as eval treats a refusal on every wire."""
    resp = client.messages.create(
        system=prompt,
        messages=[{"role": "user", "content": text}],
        **request_params(model, regime),
    )
    u = resp.usage
    tokens = (
        (u.input_tokens or 0)
        + (u.cache_creation_input_tokens or 0)
        + (u.cache_read_input_tokens or 0),
        u.output_tokens or 0,
    )
    if refusal_fails and resp.stop_reason == "refusal":
        raise ApiError("model refused (stop_reason=refusal)", tokens)
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


# ── Judges for eval: Anthropic, OpenAI-compatible, Jev ───────────────────────
# Each is a function from a rendered pair to a verdict dict. `retried` wraps
# it: an API error is retried once, and a second one makes a `failed` verdict,
# which eval counts and keeps out of every metric. A model reply that is not a
# valid verdict stays `unjudged`, scored wrong, as before.

FAILED = "failed"
RETRY_SLEEP_S = 5
JUDGES = ("anthropic", "openai", "jev")


class ApiError(Exception):
    """No verdict, through no fault of the reply's content: transport, HTTP
    status, content filter, refusal or a malformed typed answer. Carries the
    tokens of a call that returned, so they are still booked."""

    def __init__(self, msg: str, tokens: tuple[int, int] = (0, 0)) -> None:
        super().__init__(sanitize_reason(msg))
        self.tokens = tokens


class AuthError(ApiError):
    """401 / 403: a configuration fault every later call would repeat."""


def retried(judge: Callable[[str], dict]) -> Callable[[str], dict]:
    def call(text: str) -> dict:
        tin = tout = 0
        for attempt in range(2):
            try:
                v = judge(text)
            except AuthError:
                raise
            except ApiError as e:
                tin, tout, error = tin + e.tokens[0], tout + e.tokens[1], str(e)
                if attempt == 0:
                    time.sleep(RETRY_SLEEP_S)
                continue
            return v | {"tokens": (tin + v["tokens"][0], tout + v["tokens"][1])}
        return unjudged(error, (tin, tout)) | {"verdict": FAILED}

    return call


def anthropic_judge(
    client: anthropic.Anthropic, model: str, prompt: str, regime: str
) -> Callable[[str], dict]:
    def judge(text: str) -> dict:
        try:
            return judge_pair(client, model, prompt, text, regime, refusal_fails=True)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as e:
            raise AuthError(f"HTTP {e.status_code}") from e
        except anthropic.APIStatusError as e:
            raise ApiError(f"HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIError as e:
            raise ApiError(type(e).__name__) from e

    return judge


def http_client(url: str, key: str) -> httpx.Client:
    # No redirects, as in make_client: one could carry the bearer elsewhere;
    # no proxy from the environment: see Egress.
    return httpx.Client(
        base_url=url.rstrip("/"),
        headers={"Authorization": f"Bearer {key}"},
        timeout=JUDGE_TIMEOUT_S,
        follow_redirects=False,
        trust_env=False,
    )


def post_json(http: httpx.Client, path: str, body: dict) -> dict:
    try:
        resp = http.post(path, json=body)
    except httpx.HTTPError as e:  # never str(e): see env()
        raise ApiError(type(e).__name__) from e
    if resp.status_code in (401, 403):
        raise AuthError(f"HTTP {resp.status_code}")
    if resp.status_code != 200:  # a gateway content-filter block lands here too
        raise ApiError(f"HTTP {resp.status_code}: {resp.text}")
    try:
        return resp.json()
    except ValueError as e:
        raise ApiError("HTTP 200 with a non-JSON body") from e


def openai_body(model: str, prompt: str, text: str) -> dict:
    """The request `openai.rs` sends: strict json_schema response_format,
    `max_completion_tokens`, no `temperature`."""
    return {
        "model": model,
        "max_completion_tokens": MAX_OUTPUT_TOKENS,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "verdict",
                "strict": True,
                "schema": VERDICT_SCHEMA,
            },
        },
    }


def openai_verdict(body: dict) -> dict:
    """Port of `ChatResponse::into_completion`, then `validate`. A content
    filter or a refusal is an API error here (retried, then a failed pair);
    production records it as unjudged."""
    u = body.get("usage") or {}
    tokens = (u.get("prompt_tokens") or 0, u.get("completion_tokens") or 0)
    choice = next(iter(body.get("choices") or ()), None)
    if choice is None:
        return unjudged("no choices in the response", tokens)
    message, finish = choice.get("message") or {}, choice.get("finish_reason")
    if finish == "content_filter":
        raise ApiError("stopped by the content filter", tokens)
    if (message.get("refusal") or "").strip():
        raise ApiError(f"model refused (finish_reason={finish})", tokens)
    text = (message.get("content") or "").strip()
    if not text:
        return unjudged(f"empty response (finish_reason={finish})", tokens)
    try:
        raw = json.loads(text)
    except ValueError as e:
        return unjudged(f"not JSON (finish_reason={finish}): {e}", tokens)
    return validate(raw, tokens)


def openai_judge(http: httpx.Client, model: str, prompt: str) -> Callable[[str], dict]:
    def judge(text: str) -> dict:
        return openai_verdict(
            post_json(http, "/v1/chat/completions", openai_body(model, prompt, text))
        )

    return judge


# Jev answers typed questions about a `state` (here, the rendered pair) with
# probabilities and writes no text. The criteria restate the judge prompt's
# four classes; `hide_a` / `hide_b` ask the auto-apply question directly.
JEV_MODEL = "jev-1.13.0"  # pinned: the `jev-latest` alias moves without notice
# A fixed origin and a key of its own, so no other vendor's key can reach it.
TYPESAFE_URL = "https://api.typesafe.ai"


def safe_to_hide(x: str, y: str) -> dict:
    return {
        "type": "noul",
        "instructions": (
            f"Is it safe to hide Memory {x} entirely from search, keeping only "
            f"Memory {y}?"
        ),
        "criteria": {
            "true": (
                f"Memory {y} covers or corrects everything in Memory {x} that a "
                f"reader still needs, so nothing true and useful is lost when "
                f"Memory {x} is hidden."
            ),
            "false": (
                f"Memory {x} still holds something true and useful that Memory "
                f"{y} does not: a claim, finding, history or context a reader "
                "would lose."
            ),
        },
    }


JEV_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "The state holds two memories, Memory A and Memory B, from an "
            "engineering team's long-term memory store, and says which was "
            "recorded first. Which one relation holds between them?"
        ),
        "criteria": {
            "contradiction": (
                "Both claim to be current and cannot both be true, and recency "
                "alone does not settle which is right."
            ),
            "supersession": (
                "The newer memory updates or replaces a claim the older one makes "
                "about the same thing (a fact, state, decision or plan that "
                "changed), so a reader relying on the older memory today would "
                "be misled."
            ),
            "coexist": (
                "Both are true at once: progress snapshots of the same work at "
                "different times, a plan and its later outcome, a decision and "
                "the analysis behind it, or different facets of one topic."
            ),
            "unrelated": (
                "They merely share vocabulary or a project name; neither bears "
                "on the other's claim."
            ),
        },
    },
    "survivor": {
        "type": "choice",
        "instructions": (
            "If one memory replaces a claim the other makes, which memory stays "
            "current?"
        ),
        "criteria": {
            "a": "Memory A stays current; it replaces a claim Memory B makes.",
            "b": "Memory B stays current; it replaces a claim Memory A makes.",
            "neither": "Neither memory replaces a claim the other makes.",
        },
    },
    "hide_a": safe_to_hide("A", "B"),
    "hide_b": safe_to_hide("B", "A"),
}


def jev_verdict(body: dict) -> dict:
    """A verdict from Jev's answers. The class is the most probable one; a
    supersession names the likelier of a and b as survivor (it must name one).
    No metric reads another class's survivor; the probabilities are kept."""
    u = body.get("usage") or {}
    tokens = (u.get("input_tokens") or 0, u.get("output_tokens") or 0)
    try:
        answers = body["answers"]
        probs = {c: float(answers["verdict"]["probabilities"][c]) for c in CLASSES}
        surv = {
            s: float(answers["survivor"]["probabilities"][s])
            for s in ("a", "b", "neither")
        }
        hide = {s: float(answers[f"hide_{s}"]["noul"]) for s in ("a", "b")}
    except (KeyError, TypeError, ValueError) as e:
        raise ApiError(f"malformed answer: {type(e).__name__} {e}", tokens) from e
    verdict = max(CLASSES, key=probs.__getitem__)
    survivor = None
    if verdict == "supersession":
        survivor = "a" if surv["a"] >= surv["b"] else "b"
    return {
        "verdict": verdict,
        "survivor": survivor,
        "reason": "",
        "confidence": probs[verdict],
        "tokens": tokens,
        "model_version": body.get("model"),
        "probs": probs,
        "survivor_probs": surv,
        "hide": hide,
    }


def jev_judge(http: httpx.Client, model: str) -> Callable[[str], dict]:
    def judge(text: str) -> dict:
        body = {"state": text, "model": model, "questions": JEV_QUESTIONS}
        return jev_verdict(post_json(http, "/v1/systemone", body))

    return judge


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


# ── k-pass consensus and interval scoring (eval) ─────────────────────────────

Z95 = 1.96


def wilson(k: int, n: int) -> tuple[float, float] | None:
    """95 % Wilson score interval for k successes in n trials."""
    if n == 0:
        return None
    p = k / n
    d = 1 + Z95 * Z95 / n
    centre = (p + Z95 * Z95 / (2 * n)) / d
    half = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": ratio(k, n), "ci95": wilson(k, n)}


def vote_key(v: dict) -> tuple[str, str | None]:
    """What a pass decided: the class, plus the survivor when it would supersede."""
    return v["verdict"], v["survivor"] if v["verdict"] == "supersession" else None


def consensus(votes: list[dict]) -> dict:
    """The k passes' common verdict, or an abstention.

    Any dissent abstains (so at k = 3 a 2-1 split goes to the operator), and so
    does a pass that produced no verdict. An abstention is never a correct
    non-supersession: it scores 0 against every label.
    """
    keys = {vote_key(v) for v in votes}
    if len(keys) != 1 or votes[0]["verdict"] == "unjudged":
        return {"verdict": ABSTAIN, "survivor": None}
    verdict, survivor = keys.pop()
    return {"verdict": verdict, "survivor": survivor}


def consensus_metrics(rows: list[tuple[Pair, dict]]) -> dict:
    """Headline rates over consensus verdicts; definitions in README.md."""
    by_label: dict[str, list[tuple[Pair, dict]]] = defaultdict(list)
    for p, v in rows:
        by_label[p.label].append((p, v))
    coexist, sup = by_label["coexist"], by_label["supersession"]
    unrelated, contradiction = by_label["unrelated"], by_label["contradiction"]
    coexist_sup = sum(1 for _, v in coexist if v["verdict"] == "supersession")
    called = [(p, v) for p, v in sup if v["verdict"] == "supersession"]
    wrong = sum(1 for p, v in called if predicted_survivor(p, v) != p.survivor)
    predicted = sum(1 for _, v in rows if v["verdict"] == "supersession")

    def escalated(group: list[tuple[Pair, dict]]) -> dict:
        return rate(sum(1 for _, v in group if v["verdict"] in CONFLICT), len(group))

    return {
        "pairs": len(rows),
        "by_label": dict(sorted(Counter(p.label for p, _ in rows).items())),
        "abstained": sum(1 for _, v in rows if v["verdict"] == ABSTAIN),
        "false_supersede_rate": rate(coexist_sup + wrong, len(coexist) + wrong),
        "false_supersedes": {
            "coexist_called_supersession": coexist_sup,
            "wrong_survivor": wrong,
        },
        "coexist_to_conflict": escalated(coexist),
        "unrelated_to_conflict": escalated(unrelated),
        "recall_supersession": rate(len(called), len(sup)),
        "precision_supersession": rate(len(called), predicted),
        "survivor_accuracy": rate(len(called) - wrong, len(called)),
        "yield": rate(int(sum(score(p, v) for p, v in rows)), len(rows)),
        # n = 0 while no pair carries the label: the class is unmeasured.
        "contradiction": rate(
            sum(1 for _, v in contradiction if v["verdict"] == "contradiction"),
            len(contradiction),
        ),
    }


def label_views(pair: Pair, newer: dict[int, str]) -> dict[str, Pair]:
    """The pair as labelled, then with a `partial` pair read as a supersession
    by the newer memory (it corrects a claim in the older one)."""
    flipped = pair
    if "partial" in pair.tags:
        flipped = dataclasses.replace(
            pair, label="supersession", survivor=newer[pair.id]
        )
    return {"partial_as_coexist": pair, "partial_as_supersession": flipped}


VIEWS = ("partial_as_coexist", "partial_as_supersession")


def both_ways(rows: list[tuple[Pair, dict]], newer: dict[int, str]) -> dict:
    """Headline metrics with `partial` pairs counted both ways."""
    return {
        view: consensus_metrics([(label_views(p, newer)[view], v) for p, v in rows])
        for view in VIEWS
    }


def unanimity(decided: list[dict]) -> dict:
    """Share of pairs whose passes all agreed on a verdict: the default arm's
    variance diagnostic.

    Never a comparison metric across regimes: a near-deterministic regime
    scores ~100 % on it by construction.
    """
    return rate(sum(1 for c in decided if c["verdict"] != ABSTAIN), len(decided))


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
    "operator": (
        "an operator resolved this pair by superseding one memory with the other "
        "(verified: the loser's superseded_by points at the survivor)"
    ),
    "queue-read": "a reviewer read the unresolved queue and labelled the pair",
    "agreed": "a blind full-text reading under the rule agreed with the earlier label",
    "adjudicated": "the label was settled by adjudication after reviewers disagreed",
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
    regime: str = "default",
) -> list[dict]:
    # An API error aborts the tune: an infrastructure fault must not be scored
    # as a prompt failure.
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        return list(
            pool.map(
                lambda p: judge_pair(
                    client,
                    model,
                    prompt,
                    render_pair(memories[p.a], memories[p.b]),
                    regime,
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


def make_client(url: str, key: str, max_retries: int = 2) -> anthropic.Anthropic:
    # No redirects: httpx strips Authorization on a cross-origin redirect but
    # not x-api-key, so a redirecting endpoint could take the key elsewhere.
    # Same policy as the Rust transport, which also ignores proxies from the
    # environment (see Egress). The SDK's default client mounts HTTP(S)_PROXY
    # and ALL_PROXY itself, whatever trust_env says, unless it is handed a
    # transport, and it speaks its own httpx flavour (httpx2 since 1.11): the
    # transport comes from the package its client class is built on.
    base = next(
        c
        for c in anthropic.DefaultHttpxClient.__mro__
        if c.__module__.split(".")[0] in ("httpx", "httpx2")
    )
    lib = importlib.import_module(base.__module__.split(".")[0])
    return anthropic.Anthropic(
        base_url=url,
        api_key=key,
        timeout=JUDGE_TIMEOUT_S,
        max_retries=max_retries,
        http_client=anthropic.DefaultHttpxClient(
            follow_redirects=False, trust_env=False, transport=lib.HTTPTransport()
        ),
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
    judge_url, judge_key = egress_url("JUDGE_URL"), env("JUDGE_API_KEY")
    refl_url = (
        egress_url("REFLECTION_URL") if os.environ.get("REFLECTION_URL") else None
    )
    refl_key = (
        env("REFLECTION_API_KEY") if os.environ.get("REFLECTION_API_KEY") else None
    )
    if bool(refl_url) != bool(refl_key):  # never send one origin's key to another
        sys.exit("set REFLECTION_URL and REFLECTION_API_KEY together, or neither")
    unscrubbed_ok(judge_model, judge_url)  # both models read raw pairs
    unscrubbed_ok(reflection_model, refl_url or judge_url)
    memories = fetch_memories(pairs)
    hygiene = Hygiene(seed, memories)
    literal_hits = hygiene.leak_hits(seed, LITERAL_WINDOW, ignore_seed=False)
    log(
        f"seed prompt: {len(seed)} chars; literal {LITERAL_WINDOW}-char overlaps "
        f"with golden memories: {len(literal_hits)}"
    )
    ledger = Ledger()
    judge_client = make_client(judge_url, judge_key)
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


def judge_passes(
    judge: Callable[[str], dict],
    model: str,
    texts: list[str],
    passes: int,
    ledger: Ledger,
    max_usd: float,
    votes: list[list[dict]],
) -> None:
    """Fill `votes[pair]` with `passes` independent verdicts per rendered pair.

    The caller owns `votes`, so verdicts already paid for survive an abort:
    every call that returned is booked before an error is re-raised. A chunk
    whose every call failed aborts too: that is an outage, not failed pairs.
    Spend is checked between chunks of SPEND_CHECK_EVERY calls, so the cap is
    overshot by at most one chunk, and the first chunk's cost is projected over
    the whole run, so a run the cap cannot cover stops after one chunk rather
    than mid-pass with nothing scorable.
    """
    total, done = passes * len(texts), 0
    for k in range(passes):
        for i in range(0, len(texts), SPEND_CHECK_EVERY):
            if ledger.usd() >= max_usd:
                raise RuntimeError(
                    f"spend cap: ${ledger.usd():.2f} >= ${max_usd} in pass {k + 1}"
                )
            chunk = texts[i : i + SPEND_CHECK_EVERY]
            with ThreadPoolExecutor(CONCURRENCY) as pool:
                futures = [pool.submit(judge, t) for t in chunk]
            errors = [f.exception() for f in futures if f.exception() is not None]
            booked = []
            for j, f in enumerate(futures):
                if f.exception() is None:
                    v = f.result()
                    ledger.add("judge", model, *v["tokens"])
                    votes[i + j].append(v)
                    booked.append(v)
            if errors:
                raise errors[0]
            if len(chunk) > 1 and all(v["verdict"] == FAILED for v in booked):
                raise RuntimeError(
                    f"all {len(chunk)} calls of a chunk failed: {booked[0]['reason']}"
                )
            done += len(chunk)
            projected = ledger.usd() / done * total
            # 10 % headroom, so a run the projection admits does not trip the cap
            # late; a run that fit in its first chunk is already paid for.
            if done == len(chunk) and done < total and projected * 1.1 > max_usd:
                raise RuntimeError(
                    f"spend cap: {total} calls project to ${projected:.2f}, "
                    f"too close to ${max_usd} (10 % headroom)"
                )
        log(f"pass {k + 1}/{passes} done: spend ${ledger.usd():.2f}")


def newer_memory(pair: Pair, memories: dict[str, dict]) -> str:
    a, b = memories[pair.a], memories[pair.b]
    return pair.a if a["created_at"] >= b["created_at"] else pair.b


def consensus_disagreements(
    pairs: list[Pair], votes: list[list[dict]], decided: list[dict]
) -> list[dict]:
    """Every pair whose consensus differs from its label, abstentions included."""
    return [
        {
            "a": p.a,
            "b": p.b,
            "label": p.label,
            "sub_label": p.sub_label,
            "tags": list(p.tags),
            "label_survivor": p.survivor_letter(),
            "disputed_label": p.disputed,
            "verdict": c["verdict"],
            "survivor": c["survivor"],
            "votes": [
                f"{v['verdict']}:{v['survivor']}" if v["survivor"] else v["verdict"]
                for v in vs
            ],
            "reasons": [v["reason"] for v in vs],
        }
        for p, vs, c in zip(pairs, votes, decided)
        if score(p, c) < 1.0
    ]


SPEND_LOG = "spend_log.jsonl"
ROWS_FILE = "rows.json"
LOCK_FILE = ".eval.lock"


def read_jsonl(path: Path) -> list[dict]:
    """Every line of `path` as JSON. A torn line (an aborted write) exits
    naming it: skipped, it would undercount spend or drop verdicts."""
    lines = read_file(path).splitlines()
    return [
        parse_json(line, f"{path}:{n}")
        for n, line in enumerate(lines, 1)
        if line.strip()
    ]


def spend_log(out: Path) -> list[dict]:
    """One line per eval in the run directory; the run's cap covers them all."""
    path = out / SPEND_LOG
    return read_jsonl(path) if path.exists() else []


def load_rows(out: Path) -> list[Pair]:
    """Resolved spot-check rows (see `rows`) as unlabelled pairs."""
    path = out / ROWS_FILE
    if not path.exists():
        sys.exit(f"{path} missing: run `tune.py rows` first")
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Pair(r["row"], r["a"], r["b"], "", None, "spotcheck") for r in rows]


def make_judge(
    judge: str, model: str, prompt: str, regime: str, url: str
) -> Callable[[str], dict]:
    if judge == "jev":
        http = http_client(url, env("TYPESAFE_API_KEY"))
        return retried(jev_judge(http, model))
    key = env("JUDGE_API_KEY")
    if judge == "anthropic":  # no SDK retries: `retried` retries once, as for all
        client = make_client(url, key, max_retries=0)
        return retried(anthropic_judge(client, model, prompt, regime))
    return retried(openai_judge(http_client(url, key), model, prompt))


def request_shape(judge: str, model: str, regime: str) -> dict:
    """Everything the judge sends but the system prompt and the pair."""
    if judge == "anthropic":
        return request_params(model, regime)
    if judge == "openai":
        body = openai_body(model, "", "")
        return {k: v for k, v in body.items() if k != "messages"}
    return {"model": model, "questions": JEV_QUESTIONS}


def record(p: Pair, k: int, v: dict, newer: str) -> dict:
    row = {"pair": p.id, "a": p.a, "b": p.b, "newer": newer, "pass": k + 1}
    row |= {"label": p.label} | {
        key: v[key] for key in ("verdict", "survivor", "confidence", "reason")
    }
    extra = ("model_version", "probs", "survivor_probs", "hide")
    return row | {key: v[key] for key in extra if key in v}


def render_all(
    chosen: list[Pair], memories: dict[str, dict], scrubber: Scrubber | None
) -> list[str]:
    """Every chosen pair as the judge sees it. Scrubbed, it fails closed: the
    run exits before any request while a rendered pair still matches a rule."""
    sent = {h for p in chosen for h in (p.a, p.b)}
    shown = {h: scrubber.memory(memories[h]) if scrubber else memories[h] for h in sent}
    texts = [render_pair(shown[p.a], shown[p.b]) for p in chosen]
    if scrubber:
        leaked = Counter(c for t in texts for c in scrubber.leaks(t))
        if leaked:
            sys.exit(f"scrub check failed, nothing sent: still matching {dict(leaked)}")
        log(f"scrubbed {len(sent)} memories: {dict(scrubber.counts)}")
    return texts


def cmd_eval(args) -> None:
    if args.judge != "anthropic" and not args.scrub:
        # Unscrubbed text goes only where the production judge already sends it.
        sys.exit(f"--judge {args.judge} needs --scrub")
    if args.judge != "anthropic" and args.regime != "default":
        sys.exit("--regime applies to the anthropic judge only")
    prompt = read_file(args.prompt_file) if args.prompt_file else seed_prompt()
    model = args.model or {
        "anthropic": os.environ.get("JUDGE_MODEL", "claude-sonnet-5"),
        "jev": JEV_MODEL,
    }.get(args.judge)
    if not model:
        sys.exit(f"--judge {args.judge} needs --model")
    price(model)  # fail before the first paid call
    # Every egress check runs before the first memory is fetched.
    judge_url = TYPESAFE_URL if args.judge == "jev" else egress_url("JUDGE_URL")
    if not args.scrub:
        unscrubbed_ok(model, judge_url)
    out = HERE / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / LOCK_FILE).open("w")  # held until exit: one eval per run at a time
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"another eval is running in {out}; the run's cap needs them in turn")
    cap = args.max_usd - sum(e["total_usd"] for e in spend_log(out))
    if cap <= 0:
        sys.exit(f"run {args.run} has already spent its ${args.max_usd} cap")
    if args.pairs == "rows":
        chosen = load_rows(out)
        memories = fetch_memories(chosen)
    else:
        pairs = load_fixture()
        train, val = load_split(pairs)
        chosen = {"val": val, "train": train, "all": pairs}[args.pairs]
        memories = fetch_memories(pairs)
    problems = []
    if args.judge != "jev":
        problems = Hygiene(seed_prompt(), memories).problems(prompt)
        if problems:  # eval measures; landing is gated in judge.rs and by a human
            log("prompt hygiene problems: " + "; ".join(problems))
    scrubber = Scrubber(read_host_names(args.host_names)) if args.scrub else None
    if scrubber and args.judge != "jev" and scrubber.leaks(prompt):
        # The system prompt goes out too, to whichever model the wire reaches,
        # and a tuned one was written from raw pairs.
        sys.exit(
            f"the prompt matches scrub rules {scrubber.leaks(prompt)}; nothing sent"
        )
    texts = render_all(chosen, memories, scrubber)
    judge = make_judge(args.judge, model, prompt, args.regime, judge_url)
    instructions = (
        json.dumps(JEV_QUESTIONS, sort_keys=True) if args.judge == "jev" else prompt
    )
    scrub = "scrubbed" if scrubber else "raw"
    stem = (
        f"eval_{args.pairs}_{args.judge}_{model}_{scrub}_{args.regime}"
        f"_k{args.passes}_{sha(instructions)[:8]}"
    )
    # What this eval sends, written before the first request: an abort or an
    # outage later cannot drop it from the report.
    sent = {"judge": args.judge, "model": model, "scrubbed": bool(scrubber)}
    sent["pairs"] = [[p.a, p.b] for p in chosen]
    (out / f"{stem}_sent.json").write_text(json.dumps(sent) + "\n", encoding="utf-8")
    log(
        f"eval: {args.judge}:{model}, {len(chosen)} pairs x {args.passes} passes, "
        f"{scrub}, regime={args.regime}, run cap left ${cap:.2f}"
    )
    newer = {p.id: newer_memory(p, memories) for p in chosen}
    votes: list[list[dict]] = [[] for _ in chosen]
    ledger = Ledger()
    try:
        judge_passes(judge, model, texts, args.passes, ledger, cap, votes)
    finally:  # paid verdicts and spend are written before any exit path
        with (out / SPEND_LOG).open("a", encoding="utf-8") as f:
            line = {"stem": stem, "judge": args.judge, "model": model}
            f.write(json.dumps(line | ledger.summary()) + "\n")
        with (out / f"{stem}_records.jsonl").open("w", encoding="utf-8") as f:
            for p, vs in zip(chosen, votes):
                for k, v in enumerate(vs):
                    f.write(json.dumps(record(p, k, v, newer[p.id])) + "\n")
    failed = set()
    for p, vs in zip(chosen, votes):
        reasons = [v["reason"] for v in vs if v["verdict"] == FAILED]
        if reasons:
            failed.add(p.id)
            log(f"failed pair {p.id}: {reasons[0]}")
    result = {
        "judge": args.judge,
        "judge_model": model,
        "model_versions": sorted(
            {v["model_version"] for vs in votes for v in vs if v.get("model_version")}
        ),
        "pairs": args.pairs,
        "regime": args.regime,
        "passes": args.passes,
        "request": request_shape(args.judge, model, args.regime),
        "instructions_sha256": sha(instructions),
        "fixture_sha256": fixture_sha(),
        "hygiene_problems": problems,
        "scrubbed": bool(scrubber),
        "scrub_counts": dict(scrubber.counts) if scrubber else None,
        "sent": {
            "pairs": len(chosen),
            "memories": len({h for p in chosen for h in (p.a, p.b)}),
        },
        "failed_pairs": len(failed),
        "records": f"{stem}_records.jsonl",
        "spend": ledger.summary(),
    }
    if args.pairs != "rows":
        kept = [(p, vs) for p, vs in zip(chosen, votes) if p.id not in failed]
        decided = [consensus(vs) for _, vs in kept]
        headline = [(p, c) for (p, _), c in zip(kept, decided) if not p.disputed]
        result |= {
            "headline": {
                "excluded_disputed": len(kept) - len(headline),
                "excluded_failed": len(failed),
                **both_ways(headline, newer),
            },
            "unanimity": unanimity(decided)
            if args.regime == "default" and args.passes > 1
            else None,
            "per_pass": [
                metrics([(p, vs[k]) for p, vs in kept], model)
                for k in range(args.passes)
            ],
            "disagreements": consensus_disagreements(
                [p for p, _ in kept], [vs for _, vs in kept], decided
            ),
        }
    path = out / f"{stem}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    brief = ("disagreements", "per_pass", "request")
    print(json.dumps({k: v for k, v in result.items() if k not in brief}, indent=2))
    print(f"disagreements: {len(result.get('disagreements', []))} (see {path})")


# ── Spot-check rows ──────────────────────────────────────────────────────────
# A spot-check sheet names each row's memories by hash prefix. `rows` resolves
# them through the contradiction queue, resolved pairs included: the edge
# carries the production verdict, survivor and confidence the row was drawn on.


def contradiction_pages() -> Iterable[dict]:
    """Every CONTRADICTS pair, whatever its verdict, newest first."""
    offset = 0
    with alaya_client() as client:
        while offset is not None:
            body = {
                "limit": 500,
                "offset": offset,
                "include_resolved": True,
                "verdicts": [*CLASSES, "unjudged"],
            }
            try:
                resp = client.post("/contradictions", json=body)
                resp.raise_for_status()
                page = resp.json()
            except httpx.HTTPStatusError as e:  # str(e) echoes the URL
                sys.exit(f"POST /contradictions: HTTP {e.response.status_code}")
            except (httpx.HTTPError, ValueError) as e:  # never str(e): see env()
                sys.exit(f"POST /contradictions: {type(e).__name__}")
            yield from page["pairs"]
            offset = page.get("next_offset")


def check_prefixes(rows: Any) -> list[dict]:
    """Spot-check rows in the shape `resolve_rows` reads: each a unique int
    `row` and a pair of distinct, non-empty lowercase-hex `survivor` and
    `loser` prefixes (an empty one matches every hash) that no other row
    names. Exits on the first bad row."""
    if not isinstance(rows, list):
        sys.exit("--prefixes must hold a JSON list of rows")
    seen_rows: set[int] = set()
    seen_pairs: set[frozenset[str]] = set()
    for i, r in enumerate(rows):
        ok = (
            isinstance(r, dict)
            and type(r.get("row")) is int
            and r["row"] not in seen_rows
            and all(
                isinstance(r.get(k), str) and re.fullmatch(r"[0-9a-f]+", r[k])
                for k in ("survivor", "loser")
            )
            and r["survivor"] != r["loser"]
            and frozenset((r["survivor"], r["loser"])) not in seen_pairs
        )
        if not ok:
            sys.exit(
                f"--prefixes entry {i}: want a unique int `row` and distinct "
                "lowercase-hex `survivor` and `loser` prefixes no other row names"
            )
        seen_rows.add(r["row"])
        seen_pairs.add(frozenset((r["survivor"], r["loser"])))
    return rows


def resolve_rows(
    rows: list[dict], edges: Iterable[dict]
) -> tuple[list[dict], list[dict], list[dict]]:
    """Match each row's `survivor` and `loser` prefixes to one memory pair.

    Returns (resolved, ambiguous, unmatched). A row whose prefixes fit two
    different memory pairs is ambiguous: dropped, never guessed. When both
    directions of one pair carry an edge, the one recording the row's
    supersession is used.
    """
    want = {frozenset((r["survivor"], r["loser"])): r for r in rows}
    width = {len(r["survivor"]) for r in rows} | {len(r["loser"]) for r in rows}
    if len(width) > 1:
        sys.exit(f"rows mix prefix lengths {sorted(width)}")
    n = width.pop() if width else 0
    hits: dict[int, list[dict]] = defaultdict(list)
    for e in edges:
        r = want.get(frozenset((e["memory_a_hash"][:n], e["memory_b_hash"][:n])))
        if r is not None:
            hits[r["row"]].append(e)
    resolved, ambiguous, unmatched = [], [], []
    for r in rows:
        edges_r = hits.get(r["row"], [])
        if not edges_r:
            unmatched.append(r)
            continue
        if (
            len({frozenset((e["memory_a_hash"], e["memory_b_hash"])) for e in edges_r})
            > 1
        ):
            ambiguous.append(r)
            continue
        e = next(
            (
                e
                for e in edges_r
                if e["verdict"] == "supersession"
                and (e["survivor"] or "").startswith(r["survivor"])
            ),
            edges_r[0],
        )
        a, b = e["memory_a_hash"], e["memory_b_hash"]
        full = {a[:n]: a, b[:n]: b}
        resolved.append(
            r
            | {
                "a": a,
                "b": b,
                "survivor": full[r["survivor"]],
                "loser": full[r["loser"]],
                "prod_verdict": e["verdict"],
                "prod_survivor": e["survivor"],
                "prod_confidence": e["verdict_confidence"],
                "prod_model": e["verdict_model"],
            }
        )
    return resolved, ambiguous, unmatched


def cmd_rows(args) -> None:
    out = HERE / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    wanted = check_prefixes(parse_json(read_file(args.prefixes), args.prefixes))
    resolved, ambiguous, unmatched = resolve_rows(wanted, contradiction_pages())
    (out / ROWS_FILE).write_text(json.dumps(resolved, indent=1) + "\n")
    log(
        f"rows: {len(resolved)} of {len(wanted)} resolved; dropped "
        f"{sorted(r['row'] for r in ambiguous)} ambiguous, "
        f"{sorted(r['row'] for r in unmatched)} unmatched"
    )


# ── Compare: every judge of a run, rescored from its saved verdicts ──────────

AUTO_APPLY = 0.90  # the proposed auto-apply threshold on judge confidence
HIDE_THRESHOLDS = (0.5, 0.7, 0.9)


def kappa(x: list[str], y: list[str]) -> float | None:
    """Cohen's kappa of two raters over the same items; None when undefined."""
    n = len(x)
    if n == 0:
        return None
    observed = sum(1 for i, j in zip(x, y) if i == j) / n
    cx, cy = Counter(x), Counter(y)
    expected = sum(cx[k] * cy[k] for k in cx) / (n * n)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def decide(votes: list[dict]) -> dict | None:
    """A judge's one-pass decision on a pair, or None when the pass failed."""
    v = votes[0]
    if v["verdict"] == FAILED:
        return None
    return consensus(votes) | {k: v[k] for k in ("confidence", "hide") if k in v}


def load_evals(out: Path) -> dict[str, dict[str, dict]]:
    """ "golden" or "rows" -> judge name -> {"result", "votes": (a, b) -> votes}."""
    evals: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sorted(out.glob("eval_*.json")):
        if path.name.endswith(("_sent.json", "_spend.json")):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("pairs") not in ("all", "rows"):
            continue
        if "judge" not in result or result["passes"] != 1:
            log(f"compare skips {path.name}: it scores one-pass judge-seam evals only")
            continue
        stem = result["records"].removesuffix("_records.jsonl")
        if not (out / f"{stem}_sent.json").exists():  # the Sent table would undercount
            sys.exit(f"{path.name} has no {stem}_sent.json")
        kind = "rows" if result["pairs"] == "rows" else "golden"
        name = result["judge_model"] + ("" if result["scrubbed"] else ":raw")
        if name in evals[kind]:
            sys.exit(f"two {kind} evals for {name} in {out}; keep one")
        votes: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for r in read_jsonl(out / result["records"]):
            votes[(r["a"], r["b"])].append(r)
        evals[kind][name] = {"result": result, "votes": votes}
    return evals


def jev_name(evals: dict[str, dict]) -> str | None:
    return next((n for n, ev in evals.items() if ev["result"]["judge"] == "jev"), None)


def class_rates(rows: list[tuple[Pair, dict]]) -> dict:
    out = {}
    for c in CLASSES:
        tp = sum(1 for p, d in rows if p.label == c and d["verdict"] == c)
        out[c] = {
            "precision": rate(tp, sum(1 for _, d in rows if d["verdict"] == c)),
            "recall": rate(tp, sum(1 for p, _ in rows if p.label == c)),
        }
    return out


def hide_rates(rows: list[tuple[Pair, dict]]) -> dict:
    """Safe-to-hide as a classifier over endpoints (positive: the losing
    memory of a supersession pair) at each threshold, plus how well it ranks
    whatever the threshold (ROC AUC, ties half) and each group's median."""
    pos, neg = [], []
    for p, d in rows:
        for side, h in (("a", p.a), ("b", p.b)):
            loser = p.label == "supersession" and p.survivor != h
            (pos if loser else neg).append(d["hide"][side])
    thresholds = {}
    for t in HIDE_THRESHOLDS:
        tp, fp = sum(x >= t for x in pos), sum(x >= t for x in neg)
        thresholds[str(t)] = {
            "precision": rate(tp, tp + fp),
            "recall": rate(tp, len(pos)),
        }
    wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return {
        "thresholds": thresholds,
        "auc": wins / (len(pos) * len(neg)) if pos and neg else None,
        "median_loser": statistics.median(pos) if pos else None,
        "median_other": statistics.median(neg) if neg else None,
    }


def loser_side(d: dict) -> str:
    return "b" if d["survivor"] == "a" else "a"


def auto_apply(
    pairs: list[Pair],
    primary: dict[int, dict],
    agrees: Callable[[Pair, dict], bool] | None = None,
) -> dict:
    """The pairs auto-apply would supersede: the primary judge says
    supersession at confidence >= AUTO_APPLY, and the second vote agrees.
    Correct means the label is supersession with the same survivor."""
    applied = [
        p
        for p in pairs
        if (d := primary.get(p.id))
        and d["verdict"] == "supersession"
        and (d["confidence"] or 0.0) >= AUTO_APPLY
        and (agrees is None or agrees(p, d))
    ]
    correct = [
        p
        for p in applied
        if p.label == "supersession"
        and predicted_survivor(p, primary[p.id]) == p.survivor
    ]
    positives = sum(1 for p in pairs if p.id in primary and p.label == "supersession")
    return {
        "applied": len(applied),
        "correct": len(correct),
        "precision": rate(len(correct), len(applied)),
        "recall": rate(len(correct), positives),
    }


def second_votes(
    decided: dict[str, dict[int, dict]], primary: str, jev: str | None
) -> dict[str, tuple[str, Callable[[Pair, dict], bool]]]:
    """rule -> (the judge casting the vote, agrees(pair, primary decision)).
    A rule is scored only over pairs its judge decided: a failed vote is
    left out, as everywhere, never counted as a veto."""
    out = {}
    for name, dec in decided.items():
        if name == primary or name.endswith(":raw"):
            continue
        out[name] = (
            name,
            lambda p, d, dec=dec: (
                dec[p.id]["verdict"] == "supersession"
                and dec[p.id]["survivor"] == d["survivor"]
            ),
        )
        if name == jev:
            for t in HIDE_THRESHOLDS:
                out[f"{name} hide>={t}"] = (
                    name,
                    lambda p, d, dec=dec, t=t: dec[p.id]["hide"][loser_side(d)] >= t,
                )
    return out


def agreement(decided: dict[str, dict[Any, dict]]) -> list[dict]:
    out = []
    names = sorted(decided)
    for i, x in enumerate(names):
        for y in names[i + 1 :]:
            common = sorted(decided[x].keys() & decided[y].keys())
            vx = [decided[x][k]["verdict"] for k in common]
            vy = [decided[y][k]["verdict"] for k in common]
            same = sum(1 for i, j in zip(vx, vy) if i == j)
            out.append(
                {
                    "judges": [x, y],
                    "n": len(common),
                    "raw": rate(same, len(common)),
                    "kappa": kappa(vx, vy),
                }
            )
    return out


def fixture_provenance() -> dict:
    data = FIXTURE.read_bytes()
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "log", "-1", "--format=%H", "--", str(FIXTURE)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        log(f"fixture commit unknown: {e}")
        commit = None
    return {
        "commit": commit or None,
        "git_blob": blob,
        "sha256": fixture_sha(),
        "rule_version": json.loads(data)["rule_version"],
    }


def golden_section(evals: dict[str, dict], primary: str) -> dict:
    fixture = load_fixture()
    pairs = [p for p in fixture if not p.disputed]  # as eval's headline
    newer: dict[int, str] = {}
    decided: dict[str, dict[int, dict]] = {}
    failed: dict[str, int] = {}
    for name, ev in evals.items():
        dec = {}
        for p in pairs:
            if votes := ev["votes"].get((p.a, p.b)):
                newer[p.id] = votes[0]["newer"]
                if (d := decide(votes)) is not None:
                    dec[p.id] = d
        decided[name] = dec
        failed[name] = sum(1 for p in pairs if (p.a, p.b) in ev["votes"]) - len(dec)
    if primary not in decided:
        sys.exit(f"--primary {primary} has no golden eval here: {sorted(decided)}")
    jev = jev_name(evals)
    views = {
        view: [label_views(p, newer)[view] for p in pairs if p.id in newer]
        for view in VIEWS
    }
    judges = {}
    for name, dec in decided.items():
        scored = {
            view: [(p, dec[p.id]) for p in views[view] if p.id in dec] for view in VIEWS
        }
        judges[name] = {
            "scored": len(scored[VIEWS[0]]),
            "failed": failed[name],
            "headline": both_ways(scored[VIEWS[0]], newer),
            "classes": {view: class_rates(scored[view]) for view in VIEWS},
        }
        if name == jev:
            judges[name]["safe_to_hide"] = {
                view: hide_rates(scored[view]) for view in VIEWS
            }
    prim = decided[primary]
    seconds = second_votes(decided, primary, jev)
    vote_value = {
        view: {"alone": auto_apply(views[view], prim)}
        | {
            rule: auto_apply(
                [p for p in views[view] if p.id in decided[j]], prim, agrees
            )
            for rule, (j, agrees) in seconds.items()
        }
        for view in VIEWS
    }
    overlap = {}
    if jev:
        jd = decided[jev]
        for view in VIEWS:
            common = [p for p in views[view] if p.id in prim and p.id in jd]
            wrong_p = {p.id for p in common if score(p, prim[p.id]) == 0}
            wrong_j = {p.id for p in common if score(p, jd[p.id]) == 0}
            overlap[view] = {
                "pairs": len(common),
                "primary_wrong": len(wrong_p),
                "jev_wrong": len(wrong_j),
                "both_wrong": len(wrong_p & wrong_j),
            }
    return {
        "excluded_disputed": len(fixture) - len(pairs),
        "judges": judges,
        "vote_value": vote_value,
        "error_overlap": overlap,
        "agreement": agreement(decided),
    }


def short(d: dict | None, row: dict) -> str:
    if d is None:
        return "failed"
    conf = f" {d['confidence']:.2f}" if d.get("confidence") is not None else ""
    if d["verdict"] != "supersession":
        return d["verdict"] + conf
    kept = row["a"] if d["survivor"] == "a" else row["b"]
    return ("S same" if kept == row["survivor"] else "S flipped") + conf


def rows_section(out: Path, evals: dict[str, dict]) -> dict:
    rows = json.loads((out / ROWS_FILE).read_text(encoding="utf-8"))
    decided = {
        name: {
            r["row"]: d
            for r in rows
            if (votes := ev["votes"].get((r["a"], r["b"])))
            and (d := decide(votes)) is not None
        }
        for name, ev in evals.items()
    }
    jev = jev_name(evals)
    plain = {"row", "a", "b", "survivor", "loser", "conf"}
    table = []
    for r in rows:
        j = decided[jev].get(r["row"]) if jev else None
        table.append(
            {
                "row": r["row"],
                "survivor": r["survivor"][:12],
                "loser": r["loser"][:12],
                "prod_confidence": r["prod_confidence"],
                "marks": {
                    k: v
                    for k, v in r.items()
                    if k not in plain and not k.startswith("prod_")
                },
                "verdicts": {
                    n: short(dec.get(r["row"]), r) for n, dec in decided.items()
                },
                "jev_hide_loser": j["hide"]["a" if r["loser"] == r["a"] else "b"]
                if j
                else None,
            }
        )
    return {"rows": len(rows), "agreement": agreement(decided), "table": table}


def spend_by_vendor(out: Path) -> dict:
    totals: dict[str, float] = defaultdict(float)
    for entry in spend_log(out):
        totals[entry["judge"]] += entry["total_usd"]
    return {k: round(v, 4) for k, v in sorted(totals.items())} | {
        "total": round(sum(totals.values()), 4)
    }


def sent_by_vendor(out: Path) -> dict:
    """Distinct pairs and memories each vendor was sent, from what every eval
    records before its first request, aborted evals included."""
    sent: dict[str, dict[str, set]] = defaultdict(
        lambda: {"pairs": set(), "memories": set(), "unscrubbed_pairs": set()}
    )
    for path in sorted(out.glob("eval_*_sent.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        v = sent[s["judge"]]
        for a, b in s["pairs"]:
            v["pairs"].add((a, b))
            v["memories"] |= {a, b}
            if not s["scrubbed"]:
                v["unscrubbed_pairs"].add((a, b))
    return {k: {f: len(x) for f, x in v.items()} for k, v in sorted(sent.items())}


def scrub_section(evals: dict[str, dict[str, dict]]) -> dict:
    """Replacements per class, per pair set. Scrubbing is deterministic, so
    different counts mean judges saw different text: that aborts."""
    out = {}
    for kind, kind_evals in evals.items():
        results = [
            ev["result"] for ev in kind_evals.values() if ev["result"]["scrubbed"]
        ]
        if not results:
            continue
        if len({json.dumps(r["scrub_counts"], sort_keys=True) for r in results}) > 1:
            sys.exit(
                f"{kind} evals were scrubbed differently: different --host-names "
                "or scrub rules?"
            )
        counts = results[0]["scrub_counts"]
        out[kind] = {"memories": results[0]["sent"]["memories"]} | {
            c: counts.get(c, 0) for c in SCRUB_CLASSES
        }
    return out


def cmd_compare(args) -> None:
    out = HERE / "runs" / args.run
    evals = load_evals(out)
    if not evals:
        sys.exit(f"no one-pass `--pairs all` or `--pairs rows` evals in {out}")
    report = {
        "fixture": fixture_provenance(),
        "primary": args.primary,
        "auto_apply_confidence": AUTO_APPLY,
        "spend_usd": spend_by_vendor(out),
        "sent": sent_by_vendor(out),
        "scrub": scrub_section(evals),
        "golden": golden_section(evals["golden"], args.primary)
        if evals.get("golden")
        else None,
        "rows": rows_section(out, evals["rows"]) if evals.get("rows") else None,
    }
    (out / "compare.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "compare.md").write_text(compare_markdown(report), encoding="utf-8")
    log(f"compare: {out / 'compare.md'}")


def rate_text(r: dict | None) -> str:
    if not r or r["n"] == 0:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{r['k']}/{r['n']} = {r['rate']:.3f} [{lo:.2f}, {hi:.2f}]"


def num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def md_table(head: list[str], rows: Iterable[Iterable]) -> list[str]:
    return [
        "",
        "| " + " | ".join(head) + " |",
        "|---" * len(head) + "|",
        *("| " + " | ".join(map(str, r)) + " |" for r in rows),
    ]


def agreement_table(agree: list[dict]) -> list[str]:
    return md_table(
        ["judges", "n", "raw", "kappa"],
        (
            [" vs ".join(a["judges"]), a["n"], rate_text(a["raw"]), num(a["kappa"])]
            for a in agree
        ),
    )


def compare_markdown(rep: dict) -> str:
    fx = rep["fixture"]
    lines = [
        "# Cross-judge comparison",
        "",
        f"- fixture: commit {fx['commit']}, git blob {fx['git_blob']}, "
        f"rule_version {fx['rule_version']}",
        f"- auto-apply: {rep['primary']} supersession at confidence >= "
        f"{rep['auto_apply_confidence']}",
        f"- spend USD: {json.dumps(rep['spend_usd'])}",
        "",
        "## Sent",
    ]
    lines += md_table(
        ["vendor", "distinct pairs", "distinct memories", "unscrubbed pairs"],
        (
            [k, v["pairs"], v["memories"], v["unscrubbed_pairs"]]
            for k, v in rep["sent"].items()
        ),
    )
    lines += ["", "## Scrub replacements"]
    lines += md_table(
        ["set", "memories", *SCRUB_CLASSES],
        (
            [k, c["memories"], *(c[x] for x in SCRUB_CLASSES)]
            for k, c in rep["scrub"].items()
        ),
    )
    g = rep["golden"]
    if g:
        headline = (
            "false_supersede_rate",
            "coexist_to_conflict",
            "survivor_accuracy",
            "yield",
        )
        for view in VIEWS:
            lines += [
                "",
                f"## Golden set, {view} ({g['excluded_disputed']} disputed out)",
            ]
            lines += md_table(
                ["judge", "scored", "failed", *headline],
                (
                    [
                        n,
                        j["scored"],
                        j["failed"],
                        *(rate_text(j["headline"][view][k]) for k in headline),
                    ]
                    for n, j in g["judges"].items()
                ),
            )
            lines += md_table(
                ["judge", *(f"{m} {c}" for c in CLASSES for m in ("P", "R"))],
                (
                    [
                        n,
                        *(
                            rate_text(j["classes"][view][c][m])
                            for c in CLASSES
                            for m in ("precision", "recall")
                        ),
                    ]
                    for n, j in g["judges"].items()
                ),
            )
            for n, j in g["judges"].items():
                if "safe_to_hide" in j:
                    h = j["safe_to_hide"][view]
                    lines += [
                        "",
                        f"{n} safe-to-hide over endpoints: ROC AUC {num(h['auc'])}, "
                        f"median loser {num(h['median_loser'])}, "
                        f"median other {num(h['median_other'])}",
                    ]
                    lines += md_table(
                        ["threshold", "precision", "recall"],
                        (
                            [t, rate_text(r["precision"]), rate_text(r["recall"])]
                            for t, r in h["thresholds"].items()
                        ),
                    )
            lines += ["", "Auto-apply set:"]
            lines += md_table(
                ["rule", "applied", "correct", "precision", "recall"],
                (
                    [
                        "primary alone" if r == "alone" else f"AND {r}",
                        a["applied"],
                        a["correct"],
                        rate_text(a["precision"]),
                        rate_text(a["recall"]),
                    ]
                    for r, a in g["vote_value"][view].items()
                ),
            )
            if g["error_overlap"]:
                lines += [
                    "",
                    f"Error overlap with Jev: {json.dumps(g['error_overlap'][view])}",
                ]
        lines += ["", "## Agreement, golden", *agreement_table(g["agreement"])]
    r = rep["rows"]
    if r:
        lines += [
            "",
            f"## Spot-check rows ({r['rows']})",
            *agreement_table(r["agreement"]),
        ]
        names = list(r["table"][0]["verdicts"]) if r["table"] else []
        lines += md_table(
            [
                "row",
                "marks",
                "survivor",
                "loser",
                "prod conf",
                *names,
                "Jev hide loser",
            ],
            (
                [
                    t["row"],
                    ",".join(f"{k}={v}" for k, v in t["marks"].items()),
                    f"`{t['survivor']}`",
                    f"`{t['loser']}`",
                    t["prod_confidence"],
                    *(t["verdicts"][n] for n in names),
                    num(t["jev_hide_loser"]),
                ]
                for t in r["table"]
            ),
        )
    return "\n".join(lines) + "\n"


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {n}")
    return n


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
    e = sub.add_parser("eval", help="judge golden pairs or spot-check rows")
    e.add_argument("--prompt-file", help="system prompt; default judge.rs's")
    e.add_argument("--pairs", choices=("val", "train", "all", "rows"), default="val")
    e.add_argument("--run", default="eval", help="outputs go under runs/<name>/")
    e.add_argument("--judge", choices=JUDGES, default="anthropic")
    e.add_argument("--model", help="default JUDGE_MODEL (anthropic), jev-1.13.0")
    e.add_argument(
        "--scrub", action="store_true", help="replace hosts, IPs, URLs, emails, secrets"
    )
    e.add_argument("--host-names", help="file of bare host names to scrub too")
    e.add_argument(
        "--passes",
        type=positive_int,
        default=1,
        help="verdicts per pair; any dissent abstains",
    )
    e.add_argument("--regime", choices=sorted(REGIMES), default="default")
    e.add_argument(
        "--max-usd", type=float, default=15.0, help="cap on the run's total spend"
    )
    e.set_defaults(fn=cmd_eval)
    r = sub.add_parser("rows", help="resolve spot-check rows from hash prefixes")
    r.add_argument("--run", required=True, help="writes runs/<name>/rows.json")
    r.add_argument(
        "--prefixes", required=True, help='JSON [{"row", "survivor", "loser"}]'
    )
    r.set_defaults(fn=cmd_rows)
    c = sub.add_parser("compare", help="score a run's judges from saved verdicts")
    c.add_argument("--run", required=True)
    c.add_argument("--primary", default="claude-sonnet-5", help="the auto-apply judge")
    c.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
