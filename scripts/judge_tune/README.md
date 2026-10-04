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
| `ALAYA_URL` | Ālaya REST origin; the script GETs `/memories/{hash}` and, for `rows`, POSTs the read-only `/contradictions` |
| `ALAYA_API_KEY` | bearer for that origin (read access is all it needs) |
| `JUDGE_URL` | the judge's API origin: the Anthropic Messages API, or an OpenAI-compatible endpoint for `eval --judge openai`. Like `ALAYA_URL` and `REFLECTION_URL`, it must be https, or plain http to a cluster-local host (the rule alaya-server applies to its own `JUDGE_URL`) |
| `JUDGE_API_KEY` | key for `JUDGE_URL`, read once at start |
| `TYPESAFE_API_KEY` | key for `eval --judge jev`, which only ever calls TypeSafe's fixed origin |
| `UNSCRUBBED_JUDGE_ORIGINS` | comma-separated origins, besides `https://api.anthropic.com`, that may receive unscrubbed pairs: the production judge's own proxy. Kept out of git |
| `JUDGE_MODEL` | default `claude-sonnet-5` for the Anthropic judge; `eval --model` overrides it |
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
#    --passes k judges every pair k times; --regime picks the decoding
#    regime; --max-usd caps judge spend for the run.
uv run scripts/judge_tune/tune.py eval \
  --prompt-file scripts/judge_tune/runs/<run>/best_prompt.txt --pairs all --run <run> \
  --passes 3 --regime default --max-usd 15

# 4. Cross-judge: the same scrubbed pairs to several judges, then compare.
#    --prompt-file defaults to judge.rs's SYSTEM_PROMPT. --scrub needs
#    gitleaks on PATH, and --host-names <file> or --no-host-names.
uv run scripts/judge_tune/tune.py rows --run <run> --prefixes <rows.json>
uv run scripts/judge_tune/tune.py eval --judge jev --scrub --host-names <file> \
  --pairs all --run <run>
uv run scripts/judge_tune/tune.py eval --judge openai --model <id> --scrub \
  --host-names <file> --pairs rows --run <run>
uv run scripts/judge_tune/tune.py compare --run <run>
```

Every output lands under `scripts/judge_tune/runs/<name>/`, which is gitignored:
`tune` writes `report.md`, `report.json`, `records.jsonl` (every verdict),
`spend.json`, `best_prompt.txt` and GEPA's own state and logs; `eval` writes
`eval_<pairs>_<judge>_<model>_<scrubbed|raw>_<regime>_k<passes>_<sha>.json`
plus `_records.jsonl` (every verdict of every pass, keyed by both hashes) and
`_sent.json` (every pair it sends, written before the first request), and
appends its spend to the run's `spend_log.jsonl`; `rows`
writes `rows.json`; `compare` writes `compare.json` and `compare.md`. The
harness writes no memory content to disk, but the run directory holds model
output about it (candidate prompts, verdict reasons, hash prefixes), so it
stays out of git.

## Cross-judge eval

`eval --judge` picks the judge; every judge gets the same rendered pair.

- `anthropic` (default): the production request, as above.
- `openai`: what `crates/alaya-backends/src/openai.rs` sends, `POST
  /v1/chat/completions` with a strict `json_schema` response format,
  `max_completion_tokens` and no `temperature`, so its numbers predict that
  wire. `--model` is required.
- `jev`: TypeSafe's typed-question API (`POST /v1/systemone`), model
  `jev-1.13.0` pinned. Jev writes no text: the committed `JEV_QUESTIONS` ask
  for a probability per verdict class, a survivor (a / b / neither) and, per
  memory, whether it is safe to hide that whole memory. The class is the most
  probable one; a supersession takes the likelier of a and b as survivor.

`--scrub` replaces host names, IP addresses, URLs (`op://` included), email
addresses and secrets (API keys, tokens, passwords, bearer strings,
private-key blocks, password hashes, and any run of 16 or more letters and
digits mixing upper case, lower case and digits) with `<host>`, `<ip>`,
`<url>`, `<email>` and `<secret>` in each memory's content and tags, before
the 4,000-character cut. The host rule needs a known suffix (`.com`, `.svc`,
`.local`, ...), so a bare name or a short in-cluster name such as
`service.namespace` comes in through `--host-names <file>`, one per line,
kept out of git. `--scrub` takes that file or an explicit `--no-host-names`,
and the eval records which as `host_names` in its JSON and its `_sent.json`.

Before any request leaves, a gate checks every rendered pair, and the system
prompt for any judge that gets one. gitleaks, run with its default rules,
must find nothing, and no scrub rule may still match. gitleaks shares no rule
with the scrubber, so a secret of a shape gitleaks knows stops the run even
where the scrubber missed it; the scrub-rule check catches one rule undoing
another. Either finding stops the run with nothing sent, and so does a
missing `gitleaks` binary, which `--scrub` needs on `PATH`. The gate is a
second, independent detector, not a proof: a secret of a shape neither
knows still goes out. A finding may be a false positive, as gitleaks'
generic rule fires on some prose; the way past one is to scrub more, never
to skip the gate: add a rule, or put the flagged word in the `--host-names`
file, which scrubs it as `<host>`.

`openai` and `jev` refuse to run without `--scrub`. An
unscrubbed run (the control that measures what the scrub changes) goes only
to a `claude-*` model at an approved origin, `https://api.anthropic.com` or
one listed in `UNSCRUBBED_JUDGE_ORIGINS`; a model name alone proves nothing,
since any proxy can serve one. `tune` holds its judge and reflection
endpoints to the same rule.
These checks certify the host each URL names, so every client dials it
directly: `HTTP_PROXY`, `HTTPS_PROXY` and `ALL_PROXY` are ignored, as
alaya-server ignores them.

An API error, a content-filter block or a refusal is retried once; a second
one makes the pair `failed`: counted, kept out of every metric, never scored
as a verdict. A 401 or 403, or a chunk in which every call failed, aborts the
run. A reply that is not a valid verdict is still `unjudged` and scores wrong.
`--max-usd` caps the run directory's total spend across every `eval` in it;
a lock keeps a second `eval` in the same run from starting meanwhile.

`rows --prefixes <file>` resolves spot-check rows, given as
`[{"row", "survivor", "loser"}]` hash prefixes, through `/contradictions`
with resolved pairs included. A row whose prefixes fit two memory pairs is
dropped, never guessed. Extra keys on a row are carried into the report as
marks. `eval --pairs rows` then judges them.

`compare --run <name>` rescores every one-pass `--pairs all` and `--pairs
rows` eval in the run from its records against the current fixture, with no
API call, so a relabelled fixture costs nothing to rescore. It skips any other
eval in the run, with a log line. It writes `compare.json` and a readable
`compare.md`. A positive for
Jev's safe-to-hide answer is the losing memory of a supersession pair. The
auto-apply rule is `--primary` supersession at confidence >= 0.90, alone and
with each other judge's agreement as a second vote; a pair the second judge
failed on is left out of that rule, never counted as a veto.

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

## The fixture

`contradiction_golden.json` holds hashes and labels only. Each pair records
what an operator did with it (`action`, `action_survivor`) apart from what the
label rule says (`label`, `sub_label`, `survivor`, `tags`, `source`). The
fixture's own `fields`, `rule_version` and `label_rule` keys define them. The
Rust harness reads `label`, `survivor` and `source` and ignores the rest.

## What eval reports

`eval --passes k` judges every pair k times. A pair's verdict is the passes'
common verdict; any dissent abstains (at k = 3 a 2-1 split goes to the
operator), and so does a pass with no valid verdict. Per-pass scoring is the
Rust harness's (`per_pass` in the output). The headline scores the consensus,
skips `dispute` pairs, and gives every rate a 95 % Wilson interval:

- **false-supersede rate**: coexist pairs given a supersession verdict, plus
  supersession pairs given the wrong survivor (that write hides the memory
  that should stay), over coexist pairs plus those wrong-survivor pairs;
- **coexist to conflict**: the coexist bar, as in the Rust harness;
- **recall(supersession)**, **precision(supersession)** and **survivor
  accuracy**; an abstention is a recall miss;
- **yield**: pairs the consensus resolves correctly, over all pairs; an
  abstention is a yield loss, never a correct non-supersession.

Every headline is computed twice, with `partial` pairs counted as coexist and
as a supersession by the newer memory. The `contradiction` class has `n = 0`,
that is unmeasured, while no pair carries the label.

`--max-usd` is checked between chunks of 20 calls. The first chunk's cost is
projected over the whole run, so a run the cap cannot cover with 10 % headroom
stops after one chunk. Every call that returned is booked before an abort.

`--regime default` is the production request (no `thinking` parameter, so
`claude-sonnet-5` runs adaptive thinking). `--regime thinking-off` adds
`thinking: {"type": "disabled"}`. Neither sets `temperature`:
`claude-sonnet-5` rejects any value but the default 1.0, with or without
thinking, so a temperature-0 regime cannot be sent. The output records the
exact request parameters. The unanimity rate (pairs whose passes all agree on
a verdict) is reported for the default regime at k ≥ 2 only, as its variance
diagnostic: a near-deterministic regime scores close to 100 % on it by
construction, so it never compares regimes.
