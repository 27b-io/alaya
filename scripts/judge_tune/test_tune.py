#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["anthropic>=1.5,<2", "gepa>=0.1.4,<0.2", "httpx>=0.27"]
# ///
"""Offline checks for tune.py: `uv run scripts/judge_tune/test_tune.py`.

No network, no keys. Mirrors the unit tests in judge.rs so the Python port
and the Rust stay in step.
"""

import json
import sys

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


# The 22 adjudicated pairs, exactly as settled, one per line:
# a b label sub_label survivor(a|b|-) tags(comma-separated|-).
# In every one the fixture's `a` is the adjudication sheet's A.
ADJUDICATED = """
f5f27c48ec51455ea9976b98f8607c6f91279eeb1d43b50d82c0fc807dd9610c 3d0099a0c08710f038277b6f47b79c5a21e5ca5b3adf552aa6ea49cb65f1a8ba coexist P - -
b36a28f8f47bf1cd36de2f3bf61486fe77ede9f896696e0c346c1881cbc8f94c 081daaea208fb950e4c97f3f99ec678a10a9e1279bc3b0a55a4706b73200f485 coexist P - partial
8567cc97472a3c59829a079d5eed76761033a7e1bf8910210eacc3d31e6db7d4 a2b7efdf8d0d79a733b3e3884158564df9793db8ad1a7dc391c437f1266b28c4 coexist P - partial
8567cc97472a3c59829a079d5eed76761033a7e1bf8910210eacc3d31e6db7d4 79c1fd606a0b65697f18cea3be4e17534c11344c3bafead8cd1a21ca97893059 coexist P - -
3aea554c37b21a977a1c4a97fae59dc4058acad677a82a88f2e93560fa64e3af 446fd099282063fb624860a470b51bbea4777c454c9cb56c2f26323682eb3957 supersession S a -
b13e54a2c9218765a46c422ad71af94625ca85872ecb58a0a4ab9b5b6806cdfe 63729f3c7c842deaf715bb3967e6e0b2e1fb50d5d8c5b5ae85cdb3153a43ee1c coexist P - -
de6f8f9d12c3f66a1f52c9809b577db6b60a923daf1f94c873e00d47b63bbde2 297163b8442653d8ade8f22bb5b219cd65acdf16fb727c0aa8fb0490fa8cb852 coexist C - -
bc2ee64acb0801edb993ce24124ee3b1e7d0b2c79f9e3e892b3783aaab1af80b 44a140e1a74270c657ec35f3523c6c7f629976d9ee378241e2f54b470f9f2d35 coexist C - -
b1a52fb8fa5b20df1325bb296c0c6d1419d088ae6dbee07cee54a012d1b43481 0f38b61b40618fdc55b444ff284467e4a1c89f7e496421f20fd0a05e9ab195f8 coexist C - near-duplicate
ab318874cb6ab3b5ecad02c9f9f98938c709ab4451ba014e837b968c34aaedbe 8a2be445ba91f36484baa85ad5bce8ca7c87d5bda9c6c144703fc872a37e3c09 coexist P - -
40907745d7f8d1cd3c40bee483d3c51b73e888f83cba2b3c8094416af2a4754a 1a79dafe9ae93eb702b588fa46499ff74be4770170c5c958808a4888d15ec973 coexist P - -
a382df6513b5ece48ea0fefa0f46a5dd008a3ad45b436afb9b09a03f8124c754 3144a312170d226d9fdaef7df2ca5f72198cb6fddad5ddc513d16c32111d6ee4 coexist P - -
570f969c0996a6c6981fdc98c455e53a08905b715cc97280e52a0adee4166be7 492c2477bf1c21cd2e37d3e4bcbba4b161734b099350929dc82d77c55024078d supersession S a -
4371d8322c9257e379ff4c17f9054c9b3d2edd2ba24f92efd8cbd7698f330619 2585a206049568c31413418ed2a4c998c445886171b4e10d3bafded56dacc801 coexist P - -
017449d19803190c26aedbfd0ba6b48f3396d58a5cd5986c9e0fd9b5f4ff8539 2e0903c45058e1e689aa3ae4f68676aca3d429d4f892bef3c2a8425dc37f12cf coexist P - -
19317381ce1131a8dd5ce90325b39838e2ecd39443ddc47d2805da99fa674fc8 4a711c131ec9f2d277f7816291d7b4eb811b65c813b6b9255b229055041ff3a9 coexist P - partial
d045b299b8cba0f9ccb7e131d658fe8a991d263f9f757395f6797ed877dc2ee6 8571827c29db36cccfd6f175a3ab1e9d845024ff853db46c8645929c0c1944b1 coexist P - -
ddd443f04175a37831231bb29351ee24197a4fe734831fe61bbbcbdf46852dcb 7deabd93163f6b662cfba02e36c5a6d39df3df269a60fda2709e73c9b0e5a510 coexist P - partial
dc25b2b20ff37234017b6c7c5d2c5241ef0929b9907f8b26c3710cb3f17cd3e3 41e2e5fde2c9da31c6d06805436265538ad5785c9558c0010970bf909cb8a5ae supersession S a -
3d22859dc47d8c64d94467cac366e9e358258174ff4e250b563c219532d6a8cd 8bf437781ac6bc07370cc087dbf6c872e90bc6e3eb31d662293ba7f091cf80f6 coexist P - -
7481fc252fee94e540b1ab506ad83cacf934ee04b5699dcd7cda5c26eb9ed64f 6ae3e1a09864c5de87fe5b08681189e725adf5fed00503d8f9c94b913af924b2 coexist P - -
6a3f2194b5ea3d20a4cce25c944be88186e3512b9d70c3f75ab53214482c6ec7 d7dd699202e1eafc03baa9735f08ef3cd3ff39a6b1962780226dac9ed31c827d supersession S b newer-revoked
"""
SUB_LABELS = {
    "supersession": {"S"},
    "coexist": {"P", "C"},
    "unrelated": set(),
    "contradiction": set(),
}
SOURCES = {"operator", "queue-read", "agreed", "adjudicated"}
TAGS = {"partial", "near-duplicate", "newer-revoked"}


def check_fixture() -> None:
    """Schema v2 holds for every pair, and the 22 adjudicated pairs are exact."""
    raw = json.loads(tune.FIXTURE.read_text(encoding="utf-8"))
    assert raw["rule_version"] == 2
    assert {"supersession", "coexist", "revocation", "main_point"} <= raw[
        "label_rule"
    ].keys()
    seen = set()
    for p in raw["pairs"]:
        key = frozenset((p["a"], p["b"]))
        assert len(key) == 2 and key not in seen, p
        seen.add(key)
        assert p["source"] in SOURCES and set(p["tags"]) <= TAGS, p
        assert p["sub_label"] is None or p["sub_label"] in SUB_LABELS[p["label"]], p
        assert (p["survivor"] in (p["a"], p["b"])) == (p["label"] == "supersession"), p
        assert p["action"] in ("superseded", "none"), p
        assert (p["action_survivor"] in (p["a"], p["b"])) == (
            p["action"] == "superseded"
        ), p
        assert p["source"] != "operator" or p["action"] == "superseded", p
        assert "partial" not in p["tags"] or p["label"] == "coexist", p
        if "dispute" in p:
            assert p["dispute"]["label"] in SUB_LABELS, p
    by_pair = {(p["a"], p["b"]): p for p in raw["pairs"]}
    rows = [line.split() for line in ADJUDICATED.strip().splitlines()]
    assert len(rows) == 22
    for a, b, label, sub, surv, tags in rows:
        tags = [] if tags == "-" else tags.split(",")
        p = by_pair[(a, b)]
        want = {"a": a, "b": b}.get(surv)
        got = (p["label"], p["sub_label"], p["survivor"], p["tags"], p["source"])
        assert got == (label, sub, want, tags, "adjudicated"), (a[:8], got)
    assert sum(p["source"] == "adjudicated" for p in raw["pairs"]) == len(rows)
    assert sum("partial" in p.tags for p in tune.load_fixture()) == 4


def check_consensus_scoring() -> None:
    """k-pass consensus, abstain charging, both-ways partial scoring, intervals."""
    sup = verdict(verdict="supersession", survivor="b", confidence=0.9)
    co, unj = verdict(), verdict(verdict="unjudged", confidence=None)
    assert tune.consensus([sup, sup, sup]) == {
        "verdict": "supersession",
        "survivor": "b",
    }
    assert tune.consensus([co, co, co])["verdict"] == "coexist"
    for votes in (
        [sup, sup, co],
        [sup, sup, sup | {"survivor": "a"}],
        [co, co, unj],
        [unj] * 3,
    ):
        assert tune.consensus(votes)["verdict"] == tune.ABSTAIN, votes
    k1 = verdict(
        verdict="contradiction", survivor="a"
    )  # survivor is moot off supersession
    assert tune.consensus([k1, k1 | {"survivor": None}])["verdict"] == "contradiction"

    a, b = "a" * 64, "b" * 64
    s_pair = tune.Pair(0, a, b, "supersession", b, "operator", "S")
    c_pair = tune.Pair(1, "c" * 64, "d" * 64, "coexist", None, "agreed", "P")
    partial = tune.Pair(
        2, "e" * 64, "f" * 64, "coexist", None, "adjudicated", "P", ("partial",)
    )
    abstain = {"verdict": tune.ABSTAIN, "survivor": None}
    call_b = {"verdict": "supersession", "survivor": "b"}
    # An abstention is a recall miss and a yield loss, never a correct non-S;
    # on a coexist pair it is a yield loss but not a false supersede.
    m = tune.consensus_metrics([(s_pair, abstain), (c_pair, abstain)])
    assert (
        m["recall_supersession"]["k"] == 0
        and m["yield"]["k"] == 0
        and m["abstained"] == 2
    )
    assert m["false_supersede_rate"]["k"] == 0 and m["false_supersede_rate"]["n"] == 1
    # A wrong survivor is a false supersede.
    m = tune.consensus_metrics(
        [
            (s_pair, {"verdict": "supersession", "survivor": "a"}),
            (c_pair, {"verdict": "coexist", "survivor": None}),
        ]
    )
    assert m["false_supersedes"] == {
        "coexist_called_supersession": 0,
        "wrong_survivor": 1,
    }
    assert (m["false_supersede_rate"]["k"], m["false_supersede_rate"]["n"]) == (1, 2)
    assert (
        m["recall_supersession"]["rate"] == 1.0
        and m["survivor_accuracy"]["rate"] == 0.0
    )
    # `partial` both ways: superseding toward the newer memory is a false
    # supersede when partial counts as coexist, a hit when it counts as S.
    newer = {0: b, 1: "d" * 64, 2: "f" * 64}
    bw = tune.both_ways([(s_pair, call_b), (partial, call_b)], newer)
    pc, ps = bw["partial_as_coexist"], bw["partial_as_supersession"]
    assert (
        pc["by_label"] == {"coexist": 1, "supersession": 1}
        and pc["false_supersede_rate"]["k"] == 1
    )
    assert ps["by_label"] == {"supersession": 2} and ps["recall_supersession"]["k"] == 2
    assert ps["false_supersede_rate"]["n"] == 0 and ps["yield"]["k"] == 2
    assert pc["contradiction"].startswith("unmeasured")
    # Wilson: 2/35 spans about 1.6 % to 18.6 %; 0/n has a lower bound of 0.
    lo, hi = tune.wilson(2, 35)
    assert abs(lo - 0.0158) < 5e-4 and abs(hi - 0.1858) < 5e-4, (lo, hi)
    assert tune.wilson(0, 110)[0] == 0.0 and tune.wilson(0, 0) is None
    assert tune.unanimity([[sup, sup, sup], [sup, co, co]]) == tune.rate(1, 2)

    # Regimes: default is the production request; neither sets temperature.
    assert "thinking" not in tune.request_params("claude-sonnet-5", "default")
    off = tune.request_params("claude-sonnet-5", "thinking-off")
    assert off["thinking"] == {"type": "disabled"} and "temperature" not in off
    assert tune.newer_memory(s_pair, {a: mem("x", 1.0), b: mem("y", 2.0)}) == b

    # Spend cap: checked between chunks; the verdicts already paid for survive.
    real = tune.judge_batch
    try:
        tune.judge_batch = lambda c, m, pr, mm, batch, regime: [
            verdict(tokens=(250_000, 0)) for _ in batch
        ]
        pairs = [tune.Pair(i, a, b, "coexist", None, "agreed") for i in range(45)]
        votes: list = [[] for _ in pairs]
        ledger = tune.Ledger()
        try:
            tune.judge_passes(
                None,
                "claude-sonnet-5",
                "p",
                {},
                pairs,
                3,
                "default",
                ledger,
                15.0,
                votes,
            )
        except RuntimeError as e:
            assert "spend cap" in str(e), e
        else:
            raise AssertionError("spend cap must abort")
        assert (
            sum(map(len, votes)) == 2 * tune.SPEND_CHECK_EVERY and ledger.usd() >= 15.0
        )
        votes = [[] for _ in pairs]
        tune.judge_passes(
            None,
            "claude-sonnet-5",
            "p",
            {},
            pairs,
            3,
            "default",
            tune.Ledger(),
            1e9,
            votes,
        )
        assert all(len(vs) == 3 for vs in votes)
    finally:
        tune.judge_batch = real


def main() -> None:
    if not __debug__:  # every check below is an assert; -O would strip them all
        sys.exit("run this file without -O")
    seed = tune.seed_prompt()
    assert seed.startswith("You judge whether two memories"), seed[:60]
    assert seed.endswith("your probability that the verdict is correct."), seed[-60:]
    assert "\\n" not in seed and '\\"' not in seed, "Rust escapes left in the seed"
    assert tune.unescape_rust('a\\n\\"b\\"\\\\ \\\n    c') == 'a\n"b"\\ c'
    # `\\` then a newline is a backslash, not a continuation; rustc skips ASCII indent only.
    assert tune.unescape_rust("a\\\\\nb") == "a\\\nb"
    assert tune.unescape_rust("a\\\n \u00a0b") == "a\u00a0b"
    try:
        tune.unescape_rust("tab\\tok \\x41 crash")
    except SystemExit as e:  # an escape the map lacks must exit, never pass through
        assert "unsupported Rust escape '\\\\x'" in str(e), e
    else:
        raise AssertionError("unknown escape must exit")

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
    pair = tune.Pair(0, "a" * 64, "b" * 64, "supersession", "b" * 64, "operator")
    assert tune.score(pair, ok) == 1.0
    assert tune.score(pair, ok | {"survivor": "a"}) == 0.0
    assert tune.score(pair, verdict()) == 0.0
    co = tune.Pair(1, "c" * 64, "d" * 64, "coexist", None, "queue-read")
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
    assert s1["counts"]["coexist"] == {"train": 31, "val": 20}, s1["counts"]

    check_fixture()
    check_consensus_scoring()

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
