# Results: GEPA tune of the judge prompt for `claude-sonnet-5` (2026-09-14)

**Outcome: negative.** No tuned candidate met both bars on the validation
split, and the best tuned candidate by accuracy bought supersession recall with
coexist escalation. `SYSTEM_PROMPT` in `judge.rs` is unchanged; this round
lands only the model default (`JUDGE_MODEL=claude-sonnet-5`), the harness and
this note.

## Setup

- Golden set: 174 pairs (114 supersession / 35 coexist / 25 unrelated),
  stratified 60/40 with seed 0 (`split.json`): train 104 (68/21/15),
  validation 70 (46/14/10). Validation pairs were never shown to the
  reflection model (asserted in the harness).
- Judge `claude-sonnet-5`, wire-identical to `JudgeClient`; reflection
  `claude-opus-5`. Minibatch 6, GEPA seed 0. Invocation, in the current CLI:
  `uv run scripts/judge_tune/tune.py tune --max-metric-calls 450 --run <name>`
  with `JUDGE_URL`/`JUDGE_API_KEY` pointing at the Messages API and the model
  env vars at their defaults. The fixture header was reworded after these runs;
  `split.json` was re-cut for the new hash and the train/val ids are unchanged.
- Bar: precision(supersession) ≥ 0.90 and coexist→conflict ≤ 0.10 on the
  validation split.
- The valid run: 528 metric calls against a cap of 450 (GEPA checks the cap
  between iterations, so one validation pass overshoots), 9 reflection calls,
  6 candidates validated, 3 rejected unread by the hygiene gate, 18 minutes,
  USD 6.20. It stopped on the call cap, not on the bar.

## Validation split (70 pairs): seed vs best tuned candidate

| metric | seed | best tuned |
|---|---|---|
| accuracy | 0.871 | 0.886 |
| precision(supersession) | 0.955 | 0.938 |
| recall(supersession) | 0.913 | 0.978 |
| coexist→conflict | 2/14 = 0.143 | 3/14 = 0.214 |
| unrelated→conflict | 0/10 | 0/10 |
| survivor accuracy | 1.000 | 1.000 |
| mean confidence coexist / supersession / unrelated | 0.805 / 0.893 / 0.908 | 0.752 / 0.926 / 0.870 |
| unjudged | 0 | 0 |
| tokens in / out per pair | 2888 / 251 | 3255 / 269 |
| USD per pair (list price) | 0.0083 | 0.0092 |
| meets bar | no | no |

All six validated candidates: accuracy 0.771–0.886, precision 0.882–0.955,
coexist→conflict 2/14–6/14. None reached 1/14.

## Full set (174 pairs; the tuned candidate saw the train half during tuning)

| metric | seed (fresh pass) | best tuned |
|---|---|---|
| accuracy | 0.787 | 0.828 |
| precision(supersession) | 0.979 (93/95) | 0.922 (107/116) |
| recall(supersession) | 0.816 | 0.939 |
| coexist→conflict | 2/35 = 0.057 | 8/35 = 0.229 |
| unrelated→conflict | 0/25 | 1/25 |
| survivor accuracy | 1.000 | 1.000 |
| unjudged | 0 | 0 |
| tokens in / out per pair | 2916 / 290 | 3283 / 287 |
| USD per pair (list price) | 0.0087 | 0.0094 |
| meets bar | yes | no |

Seed confusion (predicted → labelled): supersession→coexist 20,
supersession→contradiction 1, coexist→supersession 2, coexist→unrelated 9,
unrelated→coexist 5. The judge errs conservative: its main miss is calling an
operator-resolved supersession a coexisting snapshot.

## Why tuning did not help

- **The labels pull one surface pattern two ways.** The 114 supersessions are
  pairs an operator resolved, mostly successive snapshots of one thread. The 35
  coexist pairs are also successive snapshots of one thread, read as both true.
  Every candidate that recovered more supersessions escalated more coexist
  pairs; the two bars move together, in opposite directions.
- **The bar cannot be resolved at this sample size.** Five validation passes of
  the byte-identical seed gave coexist→conflict 1/14 once and 2/14 four times,
  straddling the 0.10 line. On the full set the same prompt gave 5/35 on
  2026-09-10 and 2/35 on 2026-09-14. One sampled verdict decides pass or fail.

## Hygiene of the best tuned candidate

2527 characters (seed 1374, cap 2748); four quoted verdict names, the
survivor / reason / confidence contract and the conservative-bias sentence
kept; no hostnames; 0 spans of 40+ characters and 0 identifier tokens shared
with any golden memory. 78 new 12-character overlaps, all ordinary English
(the untouched seed shares 101 such windows with the memory text).

## Spend

| run | judge calls | USD |
|---|---|---|
| valid tune run (above) | 528 + 9 reflections | 6.20 |
| seed, full 174-pair pass | 174 | 1.52 |
| best tuned, full 174-pair pass | 174 | 1.64 |
| first attempt, stopped when the seed cleared the bar on the split (stop rule since fixed) | 70 | 0.60 |
| three aborted harness runs (missing adapter attribute; over-strict hygiene gate; GEPA evaluation-cache id collision between train and validation) | ≈1200 + 18 reflections | ≈13.7 |
| **total** | | **≈23.7** |

List price throughout, cached input counted at the full input rate.

## Options

1. Grow the coexist class to 100+ pairs so a 10 % bar resolves to one pair.
2. Score each pair over k passes (majority or mean) so one sampled verdict
   cannot flip the bar.
3. Accept the seed's 0.06–0.14 coexist escalation for the advisory phase,
   where verdicts annotate edges and never write memories.
4. Review the 20 operator supersessions the judge calls coexist: if operators
   were tidying snapshots rather than correcting claims, the label rule and the
   prompt's "recording history is intended" disagree, and one of them has to
   move.
