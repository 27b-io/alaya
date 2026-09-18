# Judge prompt tuning

Tunes the contradiction judge's `SYSTEM_PROMPT` (`crates/alaya-backends/src/judge.rs`)
with [GEPA](https://github.com/gepa-ai/gepa) against the golden set
(`crates/alaya-core/tests/fixtures/contradiction_golden.json`).

The Python judge call is wire-identical to `JudgeClient`: same `system` slot,
same user message (a port of `render_pair`), same JSON schema through
structured output, same output cap. Scoring is the Rust golden harness's:
verdict class must match the label, and a supersession must also name the
labelled survivor. The seed prompt is parsed out of `judge.rs`, so there is
one source of truth; the Rust harness (`cargo test -p alaya-core --test
golden_judge -- --ignored`) is the confirmatory run for any prompt landed here.

## Environment

Secrets come from the environment only. Nothing here reads a `.env` or writes
a key.

| variable | meaning |
|---|---|
| `ALAYA_URL` | Ālaya REST origin; the script GETs `/memories/{hash}` |
| `ALAYA_API_KEY` | bearer for that origin (read access is all it needs) |
| `JUDGE_URL` | Anthropic Messages API origin, as `JUDGE_URL` for the server |
| `JUDGE_API_KEY` | key for `JUDGE_URL` |
| `JUDGE_MODEL` | default `claude-sonnet-5` |
| `REFLECTION_URL`, `REFLECTION_API_KEY` | default to the `JUDGE_*` values; set both or neither, a key is never sent to another origin |
| `REFLECTION_MODEL` | default `claude-opus-5` |

## Run

```bash
# 1. Cut the split once (stratified by label, seed 0, ~60/40). Committed.
uv run scripts/judge_tune/tune.py split

# 2. Tune. Validation pairs are only ever scored; the harness raises if one
#    reaches the reflection model. Stops on the bar, the call cap or the
#    spend cap, whichever first.
uv run scripts/judge_tune/tune.py tune --run $(date +%Y%m%d)

# 3. Score a prompt on val / train / all pairs and list every disagreement.
uv run scripts/judge_tune/tune.py eval \
  --prompt-file scripts/judge_tune/runs/<run>/best_prompt.txt --pairs all --run <run>
```

Every output lands under `scripts/judge_tune/runs/<name>/`, which is gitignored:
`tune` writes `report.md`, `report.json`, `records.jsonl` (every verdict),
`spend.json`, `best_prompt.txt` and GEPA's own state and logs; `eval` writes
`eval_<pairs>_<sha>.json`. The harness writes no memory content to disk, but the
run directory holds model output about it (candidate prompts, verdict reasons),
so it stays out of git.

## What the tune enforces

- **Budget.** `--max-metric-calls` (600) and `--max-usd` (30) cover judge and
  reflection calls. Spend is computed from usage as
  `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` at
  list input price plus output tokens at list output price, so it is an upper
  bound. A candidate whose full validation pass would overshoot the cap by more
  than 10 % aborts the run. An Anthropic API error that survives the SDK's
  retries also aborts it: an infrastructure fault is not a prompt failure and
  must not be scored as one. Every abort still writes `records.jsonl`,
  `spend.json` and, once the seed has a validation pass, the report. A call
  that times out is billed by the provider but cannot be ledgered; with two
  retries that is at most three calls per abort.
- **Bar.** Precision(supersession) ≥ 0.90 and coexist→conflict ≤ 0.10 on the
  validation split. A tuned candidate is *landable* when it meets both without
  losing accuracy against the seed; the first landable candidate stops the run,
  and the report's best is the most accurate landable one. The seed alone never
  stops the run, because 14 coexist pairs resolve the bar in steps of one pair.
  With no landable candidate the report shows the most accurate candidate of
  all, so a negative result still says what tuning did.
- **Prompt hygiene.** A candidate is scored 0 without a judge call if it drops
  any of the four quoted verdict names, the survivor / reason / confidence
  contract or the conservative-bias sentence; is longer than twice the seed;
  names a hostname or infrastructure; or quotes a golden memory. Quoting means
  a 40-character span, or an identifier-shaped token (ticket id, date, hash,
  URL, snake_case name, version), that the seed did not contain and some
  golden memory does. The report also gives the 12-character overlap counts:
  against ~600k characters of memory text that window flags ordinary English
  (the seed itself shares 101 such windows), so it is measured, not gated.
- **Minibatch.** 6 rather than GEPA's 3 (a constant in `tune.py`): the seed
  already scores about 0.9, so a 3-pair minibatch is all-correct most of the
  time and teaches nothing.

One seed, one split. If a result looks split-sensitive, say so in the report
rather than re-cutting.
