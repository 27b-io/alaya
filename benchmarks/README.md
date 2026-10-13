# Ālaya Retrieval Benchmarks

How we measure Ālaya's retrieval quality, and how to reproduce it.

Everything in this directory runs against a **live Ālaya server** over its real
`/store` and `/search` HTTP endpoints — not a simulation, not a re-implementation
of the scoring code. The numbers in the [results table](#results) come from the
same binary you deploy. The one exception (an offline algorithm-validation
script) is labelled as such and kept out of the live-server rows.

The harness is [`longmemeval_bench.py`](longmemeval_bench.py). The dataset is
[LongMemEval](#dataset-provenance). The reproduction is three commands.

---

## Scope

This benchmark answers one question: **when an agent asks Ālaya for a memory,
how often is the right one in the top results?**

It does this end to end:

1. Bring up a real Ālaya server (REST + the same Qdrant / FalkorDB / embedding
   backends used in production).
2. For each LongMemEval question, store that question's haystack of past
   sessions via `POST /store`, then issue the question via `POST /search`.
3. Score whether a ground-truth session landed in the top *k* results.

No mocked storage, no cached scoring, no offline re-implementation in the
headline numbers. If a number is in the [results table](#results) under a
"live server" row, it was produced by the shipped server answering real HTTP
requests.

---

## Metric

We report **hit-rate@k**.

> **hit-rate@k** — the fraction of questions for which at least one
> ground-truth answer session appears in the top *k* retrieved results.

It is implemented as `recall_at_k` in
[`longmemeval_bench.py`](longmemeval_bench.py):

```python
def recall_at_k(ranked_ids, correct_ids, k):
    top_k = set(ranked_ids[:k])
    return float(any(cid in top_k for cid in correct_ids))   # 1.0 or 0.0, then averaged
```

This is a **binary any-correct-in-top-k** measure, averaged across questions.

We deliberately call it **hit-rate, not recall@k**. Classical recall@k divides
the number of relevant items found by the *total* number of relevant items; this
function returns `1.0` the moment a *single* correct id appears in the top *k*,
regardless of how many relevant items exist. The Python function is named
`recall_at_k` for historical reasons, but its body is hit-rate — so we label it
honestly everywhere it is reported. (The function name is historical; the
semantics, not the identifier, are what we report.)

We also report **NDCG@k** (`ndcg_at_k` in [`longmemeval_bench.py`](longmemeval_bench.py)),
which rewards ranking the correct session higher within the top *k*.

---

## Protocol disclosure

Ālaya is scored under the **LongMemEval-standard single-question (reset-per-question)
protocol**, and we disclose it up front rather than burying it.

> Each question gets a **fresh corpus**. Before scoring a question, the harness
> deletes and recreates the Qdrant collections, stores *only that question's*
> haystack sessions, then searches. Each LongMemEval item is self-contained, so
> this matches the dataset's intended evaluation and the methodology of the
> systems we compare ourselves against.

What this means for honest reading of the numbers:

- The reported hit-rate is measured against a clean, single-question corpus —
  **not** with the full knowledge graph, contradiction load, or cross-question
  interference active. We do not claim the headline number reflects retrieval
  under a large, contradictory, long-lived store. It reflects retrieval quality
  on the standard benchmark, period.
- **Timing is embedder-bound.** Most of each question's wall-clock is
  re-embedding its haystack, so total runtime scales with your embedding
  endpoint's throughput, not with Ālaya. Our published run used a GPU TEI
  (RTX 4090, fp16) and measured **~1.4 s/question — ~11 min for the full 500**;
  a CPU embedding endpoint is far slower (we measured ~40–70 s/question, i.e.
  several hours for 500q). The `/search` call itself is a small fraction either
  way. Size your expectations to your embedder.

---

## Dataset provenance

> **LongMemEval (cleaned)** — `longmemeval_s_cleaned.json`, pulled from Hugging
> Face: <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned>.

- **500 multi-session QA items** across **6 question types**:
  `single-session-user`, `single-session-assistant`, `single-session-preference`,
  `multi-session`, `temporal-reasoning`, `knowledge-update`.
- Each item carries a haystack of prior chat sessions plus a question whose
  ground-truth answer lives in one or more of those sessions.
- The harness downloads it once and caches it to `/tmp/longmemeval_s_cleaned.json`
  (`LME_CACHE` in [`longmemeval_bench.py`](longmemeval_bench.py)); subsequent runs read
  the cache, but first check its SHA-256 against the pinned `LME_SHA256` and refuse a
  mismatch.

---

## One-command reproduction

Three blocks, in order: bring up the isolated bench stack, run a stratified A/B
smoke (rerank on vs off), then the full headline run.

> [!CAUTION]
> **The harness `DELETE`s and recreates its Qdrant collections (`bench_memories`,
> `bench_memories_tags`) on _every question_.** Point `--qdrant-url` at the
> **isolated bench Qdrant only** — never at a Qdrant that holds real data. The
> bench stack publishes Qdrant on host port **`16333`** (not the default `6333`)
> precisely so it cannot collide with another stack, and the harness fails closed
> if `--qdrant-url` does not point at the bench port. Do not defeat that guard.

### 1. Bring up the isolated bench stack

Build the pinned local image, then start the stack. The bench Qdrant is
`tmpfs`-backed and ephemeral, published on host port **`16333`**:

```bash
# From the repo root — stamp the commit into /health/detail so results are
# auditable. Read it back with
# `curl -s localhost:3001/health/detail | jq -r .git_sha` (the `version`
# field carries the crate semver, not the commit). The bench stack runs
# unauthenticated, so no token is needed there.
docker build -t localhost/alaya:bench --build-arg GIT_SHA="$(git rev-parse HEAD)" .

cd benchmarks
# EMBEDDING_URL must serve Snowflake/snowflake-arctic-embed-l-v2.0 at 1024-d.
# A reachable TEI endpoint (lab or local) avoids a cold-start model pull.
EMBEDDING_URL=http://<your-tei-host> docker compose up -d

docker compose ps   # wait until every service is healthy
```

> [!WARNING]
> The bench stack runs **unauthenticated by design** — it sets
> `DANGEROUSLY_ALLOW_UNAUTHENTICATED=true` with a `localhost` origin so the
> (otherwise fail-closed) server will boot without a key. That is fine because
> it is isolated and ephemeral; **it must not be internet-reachable.** For any
> non-bench deployment, set `ALAYA_API_KEY` in `.env` (which enables bearer auth
> and disables the dev-open flag) before exposing the server. See the
> [self-hosting hardening guide](../docs/quickstart-selfhost.md).

### 2. Smoke A/B (rerank on vs off), stratified so the delta is meaningful

Use **`--stratified N`**, **not** `--limit N`.

- `--stratified N` draws a deterministic, balanced sample across all 6 question
  types (round-robin, no randomness), so the rerank-on and rerank-off arms see
  the **identical** question set and the delta is attributable to rerank alone.
- `--limit N` takes the *first* N questions, which in this dataset are all one
  question type — useless for an A/B delta. It exists for quick debugging only.

```bash
# --- rerank OFF arm: server started WITHOUT RERANK_URL ---
RERANK_URL= docker compose up -d alaya
# VERIFY it is actually off (grep the server's own startup log):
docker compose logs alaya | grep -E "cross-encoder rerank disabled"

uv run longmemeval_bench.py --stratified 24 \
  --qdrant-url http://localhost:16333 \
  --rerank-note "off" \
  --out results_rerank_off_smoke_24q.jsonl

# --- rerank ON arm: server started WITH RERANK_URL ---
docker compose up -d alaya
# VERIFY rerank is configured (startup log):
docker compose logs alaya | grep -E "cross-encoder reranker enabled"

uv run longmemeval_bench.py --stratified 24 \
  --qdrant-url http://localhost:16333 \
  --rerank-note "on:BAAI/bge-reranker-v2-m3:top_n=20" \
  --out results_rerank_on_smoke_24q.jsonl

# VERIFY rerank actually EXECUTED: the reranker TEI's request counter must climb
# during the run (rerank success is silent in alaya's logs; only failures warn).
docker compose exec tei-rerank curl -s http://localhost:80/metrics | grep '^te_predict_count'
docker compose logs alaya | grep -cE "rerank failed|rerank timed out"   # must be 0
```

**Why verification is not optional.** Rerank runs only if the *server* has
`RERANK_URL` set; the harness cannot read the server's environment. A rerank
*failure* logs `rerank failed (non-fatal); using RRF order`, and a call past
`RERANK_TIMEOUT_MS` normally logs `rerank timed out (non-fatal); using RRF
order` — though if a transport error lands in the same late poll as the
expired timer, it logs `rerank failed` instead (with `elapsed_ms >=
budget_ms`). Both silently fall back to RRF — so a misconfigured or too-slow
"ON" arm can look identical to OFF. And rerank
*success* is **silent** in alaya's logs (only failures warn), so the authoritative
proof it executed is the reranker TEI's own request counter:

- `cross-encoder reranker enabled` in the alaya startup log proves it is *configured*.
- The reranker TEI's `te_predict_count` (`/metrics`) **climbing during the run**
  proves it *executed*; it stays flat on the OFF arm.
- **Zero** `rerank failed` / `rerank timed out` lines in the alaya log proves it
  never silently degraded. A CPU reranker over 20 pairs can exceed the 5 s default;
  raise `RERANK_TIMEOUT_MS` rather than publish a run that fell back to RRF.

Do not trust an A/B delta whose "rerank ON" arm can't show the counter climb.
`--rerank-note` records your operator-verified config verbatim in the summary.

The smoke delta is `hit-rate@5(on) − hit-rate@5(off)` on the same 24 questions
(~1 min per arm on a GPU embedder; ~15–25 min on a CPU embedder).

### 3. Full headline run

Rerank ON, all 500 questions (**~11 min on a GPU embedder**; several hours on a
CPU embedder — run under `tmux`/`nohup` if slow):

```bash
docker compose up -d alaya
docker compose logs alaya | grep "cross-encoder reranker enabled"

uv run longmemeval_bench.py --stratified 500 \
  --qdrant-url http://localhost:16333 \
  --rerank-note "on:BAAI/bge-reranker-v2-m3:top_n=20" \
  --out results_rerank_on_full_500q.jsonl

# Teardown — ephemeral tmpfs Qdrant, nothing persisted.
docker compose down -v
```

### 4. QA accuracy (reader + judge)

[`longmemeval_qa.py`](longmemeval_qa.py) reads a retrieval file from step 3 and
needs no stack. [`factconsolidation_bench.py`](factconsolidation_bench.py) stores
its own facts, so run it while the stack is still up. Both call the Claude Code
CLI as `claude -p --bare`, which authenticates only with `ANTHROPIC_API_KEY` (or an
`apiKeyHelper`), so set that first. The `report` commands below call no model and
need neither the key nor the dataset.

```bash
# LongMemEval-S: reader, judge and report from saved retrieval results.
uv run longmemeval_qa.py all \
  --retrieval results_rerank_on_full_500q.jsonl \
  --retrieval-note "hybrid, rerank on (BAAI/bge-reranker-v2-m3, top_n=20), keyword leg tags" \
  --run-dir qa_runs/my-run

# MemoryAgentBench FactConsolidation (6k): store, retrieve, read, judge, report.
uv run factconsolidation_bench.py all \
  --retrieval-note "hybrid, rerank on (BAAI/bge-reranker-v2-m3, top_n=20), keyword leg tags" \
  --run-dir qa_runs/my-run/factconsolidation
```

Every stage appends one row per question and skips rows already written, so a
rerun of the same command resumes after a failure. To recompute the published
scores from the committed labels without calling a model:

```bash
uv run longmemeval_qa.py report --retrieval qa_runs/2026-10-13/retrieval_500q.jsonl --run-dir qa_runs/2026-10-13
uv run factconsolidation_bench.py report --run-dir qa_runs/2026-10-13/factconsolidation
```

---

## Config

| Flag / env | Default | Meaning |
|:--|:--|:--|
| `--stratified N` | — | Deterministic balanced sample of N across all 6 types. **Use this for A/B and headline runs.** |
| `--limit N` | — | First N questions (one type only — debug use, **not** for A/B or published numbers). |
| `--mode {raw,hybrid}` | `hybrid` | `hybrid` = full RRF pipeline; `raw` = vector-only. |
| `--top-k` | `5,10` | Comma-separated *k* values to score. |
| `--alaya-url` | `http://localhost:3001` | Live server under test. |
| `--qdrant-url` | `http://localhost:16333` | Qdrant the harness **resets per question**. Must be the isolated bench port. |
| `--rerank-note` | — | Operator-verified rerank config, recorded verbatim in the summary (e.g. `on:BAAI/bge-reranker-v2-m3:top_n=20`). |
| `--out` | timestamped | Output JSONL (`*_summary.json` written alongside). |
| server `RERANK_URL` | unset | Set on the **server** to enable the cross-encoder reranker; verify in logs. |
| server `RERANK_TOP_N` | `20` | How many RRF candidates the cross-encoder re-scores. |
| server `RERANK_TIMEOUT_MS` | `5000` | Budget per rerank call (ms). Past it the query falls back to RRF and logs `rerank timed out` (or `rerank failed` with `elapsed_ms >= budget_ms`, if a transport error lands in the same late poll) — the validity gate above must count zero. |

---

## Results

The numbers below are from a verified live-server run — measured against the
shipped server with rerank state confirmed in the logs (provenance follows the
table). A figure ships here **only** after that verification.

| Configuration | Questions | hit-rate@5 | hit-rate@10 | NDCG@5 |
|:--|--:|--:|--:|--:|
| Hybrid (RRF), live server | `500` | `0.916` | `0.964` | `0.792` |
| Hybrid + cross-encoder, live server | `500` | `0.986` | `0.988` | `0.920` |

Provenance for the rows above: commit `8b01f85`, date `2026-05-30`, embedding
model `Snowflake/snowflake-arctic-embed-l-v2.0` (1024-d, embedder fp16 on GPU (RTX 4090); production runs fp32 — sub-pp difference), rerank model
`BAAI/bge-reranker-v2-m3`, `RERANK_TOP_N=20`,
rerank-fired-in-logs: `yes (TEI predict-count 1084->11084 on rerank-on, flat on rerank-off; 0 "rerank failed")`, headline framing: `Cross-encoder reranking lifts hit-rate@5 from 0.916 to 0.986 on the live server — +7.0 points, paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).`,
confidence interval: `95% CI [0.971, 0.993]; paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed)` (baseline hit-rate@5 `0.916`).

> [!IMPORTANT]
> **Offline algorithm validation (cached embeddings, NOT the live server).**
> [`rerank_sweep.py`](rerank_sweep.py) re-implements Ālaya's RRF+blend scoring in
> Python and replays it over **cached embeddings** to validate the *reranking
> algorithm* in isolation — its own docstring states it runs the "current Alaya
> scoring (RRF+blend, faithful Python re-impl)". It does **not** exercise the
> shipped Rust server, so its figures must never be merged into the live-server
> rows above or quoted as the headline. For the record, in that offline sweep the
> cross-encoder at `top_n=20` pushed hit-rate@5 to the high-0.9s — read it as
> evidence the algorithm works, not as a live-server result.

---

## QA accuracy

End to end: Ālaya retrieves, a reader model answers from what it retrieved, and a
judge model scores the answer.

> [!WARNING]
> **Not comparable with published LongMemEval QA numbers.** The reader is a
> frontier model (`claude-sonnet-5-5`) reasoning step by step over the top 10
> retrieved sessions in full, with rerank on; the paper's readers are GPT-4o-class.
> The judge is `claude-opus-5-5`, not the paper's GPT-4o, though its prompts are
> LongMemEval's own. This is one run, and run-to-run variance was not measured.

| LongMemEval-S | n | Correct | Accuracy | 95% CI |
|:--|--:|--:|--:|:--|
| **Overall** | 500 | 465 | **0.930** | [0.904, 0.949] |
| Task-averaged | | | 0.950 | |
| knowledge-update | 78 | 74 | 0.949 | [0.875, 0.980] |
| multi-session | 133 | 115 | 0.865 | [0.796, 0.913] |
| single-session-assistant | 56 | 56 | 1.000 | [0.936, 1.000] |
| single-session-preference | 30 | 30 | 1.000 | [0.887, 1.000] |
| single-session-user | 70 | 68 | 0.971 | [0.902, 0.992] |
| temporal-reasoning | 133 | 122 | 0.917 | [0.858, 0.953] |
| abstention | 30 | 27 | 0.900 | [0.744, 0.965] |

Rows are grouped as LongMemEval's `print_qa_metrics.py` groups them: each of the
30 abstention questions (ids ending `_abs`) counts in its own question type and
again in the abstention row. Task-averaged is the mean of the six type rows.
Intervals are Wilson 95% intervals.

The same run's retrieval, on the 470 questions that are not abstention questions
(LongMemEval's retrieval evaluation skips those):

| Retrieval, same run | n | Correct | Rate | 95% CI |
|:--|--:|--:|--:|:--|
| hit-rate@5 (any ground-truth session in the top 5) | 470 | 464 | 0.987 | [0.972, 0.994] |
| recall_all@5 (every ground-truth session in the top 5) | 470 | 428 | 0.911 | [0.881, 0.933] |

On all 500 questions the run's hit-rate@5 is 0.986, the same as the published
headline above.

**How it was run.**

- **Retrieval:** hybrid search, cross-encoder rerank on (`BAAI/bge-reranker-v2-m3`,
  `RERANK_TOP_N=20`), keyword leg `tags` (the default), embedder
  `Snowflake/snowflake-arctic-embed-l-v2.0`. Server built from commit `4d63bad`.
  Run on `2026-10-13`.
- **Reader:** `claude-sonnet-5-5`, given the top **k = 10** retrieved sessions.
  Retrieval indexes user turns only, but the reader gets each retrieved session
  whole (both roles), sorted by date, in LongMemEval's own reading prompt
  (flat-session, JSON history, chain-of-thought).
- **Judge:** `claude-opus-5-5`, with LongMemEval's per-question-type judge prompts
  (`get_anscheck_prompt` in its `evaluate_qa.py`) and its rule that an answer is
  correct when the judge's reply contains "yes".
- **Calls:** both models run through `claude -p` with no tools and an empty system
  prompt (Claude Code `2.1.285`): 500 reader calls and 500 judge calls, recorded
  in `calls.json`. The CLI exposes no temperature setting.

### FactConsolidation

FactConsolidation is the conflict-resolution task of
[MemoryAgentBench](https://arxiv.org/abs/2507.05257): a numbered list of facts in
which later facts overwrite earlier ones, and questions that must be answered from
the newest fact. Single-hop questions ask one fact; multi-hop questions chain
several. We ran the shortest length, 6k (455 facts, 100 questions per sub-task),
with the same reader and judge.

| FactConsolidation 6k | n | Judge accuracy | 95% CI | Substring EM (lenient) |
|:--|--:|--:|:--|--:|
| single-hop | 100 | **0.99** | [0.946, 0.998] | 0.99 |
| multi-hop | 100 | **0.25** | [0.176, 0.343] | 0.27 |

Each numbered fact is stored as its own Ālaya memory, and each question retrieves
the top 10 by hybrid search. The reader gets them with MemoryAgentBench's own query
prompt for this task. The benchmark's RAG agents store the context as multi-sentence
chunks instead, so these rows are not comparable with the paper's. Judge accuracy is
the headline. The benchmark's substring exact match is shown for reference but is
lenient: it also matches an answer the reader only named while reasoning, which is
the whole of the multi-hop gap. Multi-hop is low because the question names only the
first entity in the chain, so a single search cannot find the facts about the
entities after it.

The hypotheses, judge labels and call counts for both tables are in
[`qa_runs/2026-10-13/`](qa_runs/2026-10-13/), beside the retrieval results they were
read from, so the scores can be recomputed (see
[step 4](#4-qa-accuracy-reader--judge)).

---

## Continuous integration

No CI job runs this benchmark. [`ci.yml`](../.github/workflows/ci.yml) builds and
tests the code; it never starts a bench stack or runs `longmemeval_bench.py`. Every
number in the [results table](#results) comes from a deliberate manual run, with its
provenance recorded beside it.

## Ranking changes

This harness cannot judge a ranking change on its own. It stores every session as a
`reference` memory stamped with the ingestion time, and keeps the session's real date
only in `metadata.session_date`, which ranking never reads. A change to recency decay,
memory-type weighting or popularity boosts therefore leaves its numbers unchanged.

A ranking change (scoring, fusion, decay, boosts, rerank) is instead judged on a
read-only replay of labelled real agent queries against a deployment's live corpus.
That query set is private, because it is drawn from operators' own memories, so public
CI cannot run it. A ranking-change PR posts the replay's deltas against `main` (recall@5,
recall@10 and MRR, each with a 95% confidence interval) on its tracking ticket before
the change is switched on in a deployment.
