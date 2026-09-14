#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["anthropic>=1.5,<2", "gepa>=0.1.4,<0.2", "httpx>=0.27"]
# ///
"""Offline checks for tune.py: `uv run scripts/judge_tune/test_tune.py`.

No network, no keys. Mirrors the unit tests in judge.rs so the Python port
and the Rust stay in step.
"""

import tune


def mem(content: str, created_at: float, tags=(), memory_type="note") -> dict:
    return {
        "content": content,
        "created_at": created_at,
        "tags": list(tags),
        "memory_type": memory_type,
    }


def verdict(**over) -> dict:
    base = {
        "verdict": "coexist",
        "survivor": None,
        "reason": "r",
        "confidence": 0.5,
        "tokens": (0, 0),
    }
    return base | over


def main() -> None:
    seed = tune.seed_prompt()
    assert seed.startswith("You judge whether two memories"), seed[:60]
    assert seed.endswith("your probability that the verdict is correct."), seed[-60:]
    assert "\\n" not in seed and '\\"' not in seed, "Rust escapes left in the seed"

    # render_pair: the cases judge.rs tests.
    a, b = mem("content a", 1000.0), mem("content b", 1000.0 + 3 * 86_400)
    s = tune.render_pair(a, b)
    assert "A was recorded 3 days BEFORE B" in s, s
    assert "Memory A (recorded_at=1000; type: note; tags: -):\ncontent a" in s, s
    assert "A was recorded 3 days AFTER B" in tune.render_pair(b, a)
    assert "within a day of each other" in tune.render_pair(a, mem("x", 4600.0))
    tagged = mem("c" * 5000, 7.0, tags=("t1", "t2"), memory_type="decision")
    d = tune.describe("B", tagged)
    assert "type: decision; tags: t1, t2" in d
    assert d.endswith("c" * tune.MAX_CONTENT_CHARS) and len(d.split("\n", 1)[1]) == 4000

    # validate: RawVerdict::validate.
    ok = tune.validate(
        verdict(verdict="supersession", survivor="b", confidence=0.9), (1, 2)
    )
    assert (ok["verdict"], ok["survivor"], ok["tokens"]) == (
        "supersession",
        "b",
        (1, 2),
    )
    bad = [
        verdict(verdict="supersession", survivor=None),
        verdict(verdict="maybe"),
        verdict(survivor="c"),
        verdict(verdict="Supersession", survivor="a"),
        verdict(confidence=1.5),
        verdict(confidence=-0.1),
        verdict(confidence=float("nan")),
        verdict(confidence=True),
        verdict(verdict="unjudged"),
        "I think B wins.",
    ]
    for raw in bad:
        assert tune.validate(raw, (0, 0))["verdict"] == "unjudged", raw
    assert tune.validate(verdict(survivor="a"), (0, 0))["survivor"] is None
    clipped = tune.validate(verdict(reason="x" * 500), (0, 0))
    assert len(clipped["reason"]) == tune.MAX_REASON_CHARS
    ctl = tune.validate(verdict(reason="line one\nx -> y\t\r\x1b[0m"), (0, 0))
    assert ctl["reason"] == "line onex -> y[0m", ctl["reason"]

    # score: class match, survivor required for supersession.
    pair = tune.Pair(
        0, "a" * 64, "b" * 64, "supersession", "b" * 64, "resolved-by-operator"
    )
    assert tune.score(pair, ok) == 1.0
    assert tune.score(pair, ok | {"survivor": "a"}) == 0.0
    assert tune.score(pair, verdict()) == 0.0
    co = tune.Pair(1, "c" * 64, "d" * 64, "coexist", None, "queue-2026-09-10")
    assert tune.score(co, verdict()) == 1.0
    assert tune.score(co, verdict(verdict="unjudged")) == 0.0

    # metrics and the bar.
    rows = [
        (pair, ok),
        (co, verdict()),
        (co, verdict(verdict="supersession", survivor="a")),
    ]
    m = tune.metrics(rows, "claude-sonnet-5")
    assert m["precision_supersession"] == 0.5 and m["recall_supersession"] == 1.0
    assert m["coexist_to_conflict"] == 0.5 and m["coexist_escalated"] == "1/2"
    assert m["survivor_accuracy"] == 1.0 and m["meets_bar"] is False
    assert (
        tune.metrics([(pair, ok), (co, verdict())], "claude-sonnet-5")["meets_bar"]
        is True
    )
    assert len(tune.disagreements(rows)) == 1

    # feedback names the label, the source and the rule; never the memory.
    fb = tune.feedback(
        co, verdict(verdict="supersession", survivor="a", confidence=0.8)
    )
    assert (
        "label is coexist" in fb
        and "unresolved queue" in fb
        and "erases true history" in fb
    )
    assert tune.feedback(pair, ok).startswith(
        "Correct: supersession, survivor Memory B"
    )
    assert "wrong survivor" in tune.feedback(pair, ok | {"survivor": "a"})

    # hygiene: required text, length, infrastructure, leakage beyond the seed.
    quote = "the quick brown fox jumps over the lazy dog while the cat sleeps"
    h = tune.Hygiene(seed, {"m": mem(f"Note XY-4242: {quote}. See #77.", 0.0)})
    assert h.problems(seed) == []
    leak = h.problems(seed + f"\nRecall that {quote}.")
    assert len(leak) == 1 and "golden-memory spans" in leak[0], leak
    assert h.problems(seed + "\nRecall the quick brown fox.") == []
    ident = h.problems(seed + "\nAs in XY-4242 and #77.")
    assert len(ident) == 1 and "2 identifiers" in ident[0], ident
    assert h.problems(seed + "\nUnlike XY-9999.") == []
    assert (
        "missing required text" in h.problems(seed.replace('"coexist"', "coexist"))[0]
    )
    assert "infrastructure names" in h.problems(seed + " ask judge.internal.svc")[0]
    assert "2x seed" in h.problems(seed + " x" * len(seed))[0]
    assert h.leak_hits(seed, 12, ignore_seed=False) == set()
    tpl = tune.reflection_template(seed)
    assert str(len(seed)) in tpl and "<curr_len>" not in tpl and "<side_info>" in tpl

    # split: deterministic, stratified, a partition.
    pairs = tune.load_fixture()
    s1, s2 = tune.make_split(pairs, 0, 0.6), tune.make_split(pairs, 0, 0.6)
    assert s1 == s2 and not set(s1["train"]) & set(s1["val"])
    assert len(s1["train"]) + len(s1["val"]) == len(pairs)
    assert s1["counts"]["coexist"] == {"train": 21, "val": 14}, s1["counts"]

    # stopper: the seed never stops the run; a tuned candidate must clear the
    # bar without losing accuracy to the seed.
    class Stub:
        val_evals: list = []

    stub = Stub()
    stopper = tune.Stopper(stub, tune.Ledger(), 30.0)
    stub.val_evals = [("seed", {"meets_bar": True, "accuracy": 0.87})]
    assert stopper(None) is False
    stub.val_evals.append(("c1", {"meets_bar": True, "accuracy": 0.86}))
    assert stopper(None) is False
    stub.val_evals.append(("c2", {"meets_bar": False, "accuracy": 0.95}))
    assert stopper(None) is False
    stub.val_evals.append(("c3", {"meets_bar": True, "accuracy": 0.87}))
    assert stopper(None) is True and stopper.reason == "bar"
    assert tune.choose_best(stub)[0] == "c3"  # landable beats the more accurate c2
    stub.val_evals = [
        ("seed", {"meets_bar": False, "accuracy": 0.87}),
        ("c1", {"meets_bar": True, "accuracy": 0.80}),
        ("c2", {"meets_bar": False, "accuracy": 0.95}),
    ]
    assert tune.choose_best(stub)[0] == "c2"  # nothing landable: most accurate

    # adapter: validation metrics come from coverage of every val pair, however
    # GEPA batches the set; the train half never counts.
    all_pairs = tune.load_fixture()
    tr, va = tune.load_split(all_pairs)
    adapter = tune.JudgeAdapter(
        None, "claude-sonnet-5", {}, tr, va, tune.Hygiene(seed, {}), tune.Ledger(), 30.0
    )
    real = tune.judge_batch
    try:
        tune.judge_batch = lambda c, m, pr, mem, batch: [
            {
                "verdict": p.label,
                "survivor": (
                    None if p.survivor is None else ("a" if p.survivor == p.a else "b")
                ),
                "reason": "r",
                "confidence": 0.9,
                "tokens": (10, 1),
            }
            for p in batch
        ]
        adapter.evaluate(tr[:6], {"system_prompt": seed})
        assert adapter.val_evals == []
        adapter.evaluate(va[:30], {"system_prompt": seed})
        assert adapter.val_evals == []
        adapter.evaluate(va[30:], {"system_prompt": seed})
        assert (
            len(adapter.val_evals) == 1 and adapter.val_evals[0][1]["accuracy"] == 1.0
        )
        adapter.evaluate(va, {"system_prompt": seed})  # a repeat does not double count
        assert len(adapter.val_evals) == 1
    finally:
        tune.judge_batch = real

    # GEPA's proposer dereferences this attribute without getattr.
    assert tune.JudgeAdapter.propose_new_texts is None

    # ledger: every input token class counts.
    ledger = tune.Ledger()
    ledger.add("judge", "claude-sonnet-5", 1_000_000, 100_000)
    assert abs(ledger.usd() - 3.0) < 1e-9
    print("ok")


if __name__ == "__main__":
    main()
