# Ālaya — Launch Copy (skeleton)

---

## 0. Channels & sequencing (orientation, not copy)

| Asset | Channel | Number-bearing? |
|:--|:--|:--|
| Tagline (§1) | Product Hunt name/tagline | No — number-free |
| First comment / maker's note (§2) | Product Hunt first comment | Yes — proof sentence |
| Reddit post (§3) | r/rust · r/LocalLLaMA · r/selfhosted (one body, swappable title) | Yes — proof sentence + hit-rate@5 |

The proof-line in §4 is the **single source of truth** for the headline sentence.
§2 and §3 use that text verbatim — do not paraphrase the number anywhere else,
drift between assets is the failure mode this layout prevents.

---

## 1. Product Hunt tagline — [NUMBER-FREE]

**Ship this:**

> **Memory for AI agents that catches its own contradictions**

(56 chars — within PH's ~60-char tagline limit. Outcome-first: leads with the
rare capability, not "vector store" / "semantic memory" mechanism. Number-free.)

Backup taglines, same constraints (pick at JOIN if the lead reads wrong in
context — all ≤60, all number-free, all outcome-first):

- `Long-term memory for agents that won't contradict itself` (56)
- `Memory for LLM agents that catches when its facts disagree` (58)
- `Self-hosted memory for AI agents that resolves its contradictions` (65 — over by 5; trim only if PH allows)

---

## 2. Product Hunt — first comment / maker's note

> Hi PH — maker here.
>
> I built Ālaya because every "memory layer" I tried for my agents was just a
> vector store with a nicer name. They'd happily store two facts that flatly
> contradict each other — "auth is required on every endpoint" and "we dropped
> auth on the public ones" — and then hand back whichever one cosine similarity
> ranked highest that day, with no notion of which fact superseded which.
>
> Ālaya is a single-binary memory service (Rust) that does the three things I
> actually needed: it **finds the right memory even when the words don't match**
> (hybrid vector + keyword retrieval), it **notices when two memories disagree**
> and lets you resolve the conflict with one call while the old answer stays
> auditable (contradiction detection + supersede), and it **reasons over a small
> relationship graph** so one strong hit can pull up its neighbors. It speaks
> both MCP (for agents) and plain REST (for scripts and backfill) over one
> endpoint, and the whole default stack is five containers you bring up with one
> `docker compose up`.
>
> On retrieval quality:
> Cross-encoder reranking lifts hit-rate@5 from 0.916 to 0.986 on the live server — +7.0 points, paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).
>
> The benchmark is fully reproducible: same harness, dataset, commit and date,
> documented step-by-step against a live server (not a simulation) in
> `benchmarks/README.md`. I'd genuinely rather you reproduce the number than take
> my word for it.
>
> It's MIT-licensed and self-hostable — no account, no SaaS, runs on your own
> box. Happy to answer questions about the architecture, the metric, or the
> benchmark method.
>
> Honesty footnote for the comment (keep adjacent, do not inline a number):
> the headline is **0.986** hit-rate@5 on LongMemEval. That metric is *hit-rate@5* —
> "did at least one correct session land in the top 5", averaged over questions —
> a deliberately lenient measure I label as hit-rate, not recall. The method,
> including the per-question reset protocol, is disclosed in `benchmarks/README.md`.

---

## 3. Reddit post — [re-titlable]

### 3a. Swappable title line (pick ONE — body below is unchanged either way)

- **r/rust:** `Ālaya: a Rust memory service for LLM agents — hybrid retrieval + a contradiction-resolving knowledge graph, MCP and REST`
- **r/LocalLLaMA:** `Self-hosted long-term memory for local agents: hybrid recall + automatic contradiction detection, reproducible LongMemEval benchmark`
- **r/selfhosted:** `Ālaya: self-hosted memory for AI agents — one docker compose, MCP + REST, no SaaS, MIT`

> Title swap rule: the three titles above are interchangeable with **zero body
> edits**. The body never names a subreddit and never repeats the title's framing,
> so it drops cleanly into any of the three.

### 3b. Body (one post, technical-credible, opens with the problem)

> **The problem.** If you've wired long-term memory into an agent, you've hit
> this: the "memory layer" is a vector database with a wrapper. It stores
> whatever you give it and retrieves by cosine similarity. Two issues show up
> fast. First, retrieval misses when the query wording diverges from the stored
> wording — ask "why did we switch package managers" and it can't find the note
> that says "migrated to pnpm". Second, and worse, it has **no concept of facts
> disagreeing**: store "auth is required everywhere", later store "we no longer
> require auth on public endpoints", and both sit in the index forever. The
> retriever returns whichever one ranks highest; nothing represents that the two
> conflict.
>
> **What I built.** Ālaya is a memory service written in Rust that treats those
> two problems as first-class:
>
> - **Hybrid retrieval** fuses semantic vectors with keyword signal (RRF), so it
>   finds the right memory even when your words don't match the stored ones.
> - **Contradiction detection** runs automatically on write — negation, antonym,
>   and temporal cues ("no longer", "switched from") flag conflicting facts. You
>   resolve a conflict with one call (`supersede`); the superseded answer stays
>   in the store, auditable, marked with what replaced it.
> - **A relationship graph** (RELATES_TO / PRECEDES / CONTRADICTS) lets one
>   strong hit surface its neighbors via spreading activation.
> - **Salience, spaced-repetition, provenance/trust, and dedup** shape ranking
>   beyond raw cosine distance.
> - **One service, two protocols:** the same storage answers MCP tool calls (for
>   agents) and plain REST (for scripts/backfill). Ten MCP tools; nine REST
>   endpoints plus `/mcp`.
> - **Small to run:** the default stack is five containers (six if you enable the
>   optional cross-encoder reranker with `--profile rerank`); the server image is
>   ~150 MB and degrades gracefully when a backend blips.
>
> **Does it actually retrieve well?** This is the part I want scrutinized, not
> trusted:
>
> Cross-encoder reranking lifts hit-rate@5 from 0.916 to 0.986 on the live server — +7.0 points, paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).
>
> The metric is **0.986** hit-rate@5 measured on **LongMemEval** (`longmemeval_s_cleaned`,
> 500 multi-session QA items) against a **live server** over its real `/store` and
> `/search` endpoints. I report *hit-rate@5* — the fraction of questions where at
> least one correct session lands in the top 5 — and I deliberately call it
> hit-rate, not recall@5: it's a binary any-correct-in-top-k measure, the lenient
> one. The full method — dataset
> provenance, the per-question reset protocol, the exact commit and date, and a
> one-command reproduction — is in `benchmarks/README.md`.
>
> **Try it.** MIT-licensed, self-hostable, no account:
>
> ```bash
> git clone https://github.com/27b-io/alaya && cd alaya
> cp .env.example .env
> docker compose up --build -d        # 5 containers; first TEI start pulls the embed model
> ```
>
> Heads-up on auth: **the server is fail-closed** — it won't boot without auth,
> so the dev Compose opts into open mode on `localhost` (refused on any public
> origin). Set `ALAYA_API_KEY` in `.env` before exposing it anywhere (that turns
> on `Authorization: Bearer` auth and disables the dev-open flag):
>
> ```bash
> # Authorization header only when a key is configured; omit it on a no-auth dev box.
> curl -fsS -H "Authorization: Bearer ${ALAYA_API_KEY}" \
>   -H 'Content-Type: application/json' \
>   -X POST http://localhost:3001/search \
>   -d '{"query":"is auth required for the API?","mode":"hybrid"}'
> ```
>
> There's a 60-second `examples/demo.sh` that stores two contradicting facts,
> shows the server detecting the conflict on write, resolves it with `supersede`,
> and proves the superseded fact drops out of search — the contradiction story
> end-to-end against the real REST API.
>
> Repo: https://github.com/27b-io/alaya · Benchmark + reproduction: `benchmarks/README.md`
>
> Happy to get into the architecture, the `?Send`-trait WASM design, or the
> FalkorDB wire-format details.

---

## 4. Proof-line and quarantine note

### Proof-line (Variant A — rerank-delta)

> On LongMemEval, the default hybrid retrieval reaches **0.916**
> hit-rate@5; turning on the optional cross-encoder reranker (re-scoring the
> top-20 candidates) lifts it to **0.986** — same 500 items,
> same server, identical except `RERANK_URL`. (n=500, 95% CI [0.971, 0.993]; paired McNemar p≈5.5e-10 (36 questions fixed, 1 regressed).)

- Both numbers come from the **shipped live server**, not the offline validator.
- `0.916` = rerank-off arm; `0.986` = rerank-on arm — the delta is
  attributable to rerank alone (identical deterministic question set).

### Offline algorithm validation — NOT a launch headline (quarantine note)

> **Offline algorithm validation (cached embeddings, not the live server):** a
> faithful Python re-implementation of the scoring (`benchmarks/rerank_sweep.py`,
> RRF + blend over cached embeddings) pushed hit-rate@5 to ~0.99 with the
> cross-encoder. This validates the *algorithm*, not the shipped Rust server, and
> is **never** presented as the live launch number. It exists here only so the
> distinction is explicit; cite it only with this exact label, if at all.
