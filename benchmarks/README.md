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

It is implemented at
[`longmemeval_bench.py:79`](longmemeval_bench.py#L79) as:

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

We also report **NDCG@k** ([`longmemeval_bench.py:66`](longmemeval_bench.py#L66)),
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
  ([`longmemeval_bench.py:53`](longmemeval_bench.py#L53)); subsequent runs read
  the cache.

---

## One-command reproduction

Three blocks, in order: bring up the isolated bench stack, run a stratified A/B
smoke (rerank on vs off), then the full headline run.

> [!WARNING]
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
# From the repo root — stamp the commit into /health so results are auditable.
docker build -t localhost/alaya:bench --build-arg GIT_SHA="$(git rev-parse --short HEAD)" .

cd benchmarks
# EMBEDDING_URL must serve Snowflake/snowflake-arctic-embed-l-v2.0 at 1024-d.
# A reachable TEI endpoint (lab or local) avoids a cold-start model pull.
EMBEDDING_URL=http://<your-tei-host> docker compose up -d

docker compose ps   # wait until every service is healthy
```

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
curl -s http://<your-rerank-tei-host>/metrics | grep '^te_predict_count'
docker compose logs alaya | grep -c "rerank failed"   # must be 0
```

**Why verification is not optional.** Rerank runs only if the *server* has
`RERANK_URL` set; the harness cannot read the server's environment. A rerank
*failure* logs `rerank failed (non-fatal); using RRF order` and silently falls
back to RRF — so a misconfigured "ON" arm can look identical to OFF. And rerank
*success* is **silent** in alaya's logs (only failures warn), so the authoritative
proof it executed is the reranker TEI's own request counter:

- `cross-encoder reranker enabled` in the alaya startup log proves it is *configured*.
- The reranker TEI's `te_predict_count` (`/metrics`) **climbing during the run**
  proves it *executed*; it stays flat on the OFF arm.
- **Zero** `rerank failed` lines in the alaya log proves it never silently degraded.

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

---

## Config

| Flag / env | Default | Meaning |
|---|---|---|
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

---

## Results

Every numeric cell below is a placeholder pending the verified live-server run.
A figure ships here **only** after it is measured against the shipped server with
rerank state verified in the logs.

| Configuration | Questions | hit-rate@5 | hit-rate@10 | NDCG@5 |
|---|---|---|---|---|
| Hybrid (RRF), live server | `500` | `0.916` | `0.964` | `0.792` |
| Hybrid + cross-encoder, live server | `500` | `0.986` | `0.988` | `0.920` |

Provenance for the rows above: commit `8b01f85`, date `2026-05-30`, embedding
model `Snowflake/snowflake-arctic-embed-l-v2.0` (1024-d, embedder fp16 on GPU (RTX 4090); production runs fp32 — sub-pp difference), rerank model
`BAAI/bge-reranker-v2-m3`, `RERANK_TOP_N=20`,
rerank-fired-in-logs: `yes (TEI predict-count 1084->11084 on rerank-on, flat on rerank-off; 0 "rerank failed")`, headline framing: `Cross-encoder reranking lifts hit-rate@5 from 0.916 to 0.986 on the live server — +7.0 points, paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).`,
confidence interval: `95% CI [0.971, 0.993]; paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed)` (baseline hit-rate@5 `0.916`).

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

## Continuous integration

> A reduced run (`--stratified 50`) runs in CI on a schedule against an
> ephemeral stack. It gates on hit-rate@5 not regressing below `0.95`.

CI does **not** assert the headline number — at `n=50` the confidence interval
is too wide (≈ ±6–7 pp) to publish a decimal, and the full 500-question run is
manual (~6 h). The scheduled stratified run exists to **catch regressions**: a
retrieval change that drops hit-rate@5 below the floor fails the gate. The
[results table](#results) above is populated only by a deliberate full run, with
its provenance recorded alongside it.
