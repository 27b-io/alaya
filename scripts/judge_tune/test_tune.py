#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["anthropic>=1.5,<2", "gepa>=0.1.4,<0.2", "httpx>=0.27"]
# ///
"""Offline checks for tune.py: `uv run scripts/judge_tune/test_tune.py`.

No network, no keys. Mirrors the unit tests in judge.rs so the Python port
and the Rust stay in step.
"""

import argparse
import itertools
import json
import socket
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


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
    assert pc["contradiction"]["n"] == 0 and pc["contradiction"]["rate"] is None
    # Wilson: 2/35 spans about 1.6 % to 18.6 %; 0/n has a lower bound of 0.
    lo, hi = tune.wilson(2, 35)
    assert abs(lo - 0.0158) < 5e-4 and abs(hi - 0.1858) < 5e-4, (lo, hi)
    assert tune.wilson(0, 110)[0] == 0.0 and tune.wilson(0, 0) is None
    decided = [tune.consensus(vs) for vs in ([sup] * 3, [sup, co, co], [unj] * 3)]
    assert tune.unanimity(decided) == tune.rate(1, 3)

    # Regimes: default is the production request; neither sets temperature.
    assert "thinking" not in tune.request_params("claude-sonnet-5", "default")
    off = tune.request_params("claude-sonnet-5", "thinking-off")
    assert off["thinking"] == {"type": "disabled"} and "temperature" not in off
    assert tune.newer_memory(s_pair, {a: mem("x", 1.0), b: mem("y", 2.0)}) == b

    check_judge_passes()


def check_judge_passes() -> None:
    """Spend cap, cost projection, API errors and outages in `judge_passes`."""
    texts = [f"{i:064x}" for i in range(45)]
    calls = itertools.count()

    def run(cost, max_usd: float) -> tuple[list, tune.Ledger, str | None]:
        votes: list = [[] for _ in texts]
        ledger = tune.Ledger()

        def judge(t):
            return cost(t, next(calls))

        try:
            tune.judge_passes(
                judge, "claude-sonnet-5", texts, 3, ledger, max_usd, votes
            )
        except RuntimeError as e:
            return votes, ledger, str(e)
        return votes, ledger, None

    # A run the cap cannot cover stops after its first chunk ($0.50 a call,
    # 135 calls, $15 cap).
    votes, ledger, err = run(lambda t, n: verdict(tokens=(250_000, 0)), 15.0)
    assert err and "project" in err, err
    assert (
        sum(map(len, votes))
        == tune.SPEND_CHECK_EVERY
        == ledger.rows[("judge", "claude-sonnet-5")][0]
    )
    # A run projected just under the cap (cap = 1.05x the projection) is refused
    # up front, not killed late.
    calls = itertools.count()
    votes, ledger, err = run(lambda t, n: verdict(tokens=(5_000, 0)), 1.35 * 1.05)
    assert err and "project" in err, err
    # A run whose cost rises later still stops on the hard cap, at a chunk edge.
    calls = itertools.count()
    votes, ledger, err = run(
        lambda t, n: verdict(tokens=(5_000 if n < 20 else 500_000, 0)), 15.0
    )
    assert err and ">= $15.0" in err and sum(map(len, votes)) == 40, err

    # An error is re-raised only after every call that returned is booked.
    def flaky(t):
        if t == texts[7]:
            raise tune.AuthError("HTTP 401")
        return verdict(tokens=(1_000, 10))

    votes: list = [[] for _ in texts]
    ledger = tune.Ledger()
    try:
        tune.judge_passes(flaky, "claude-sonnet-5", texts, 3, ledger, 1e9, votes)
    except tune.AuthError:
        pass
    else:
        raise AssertionError("an auth error must abort the run")
    assert sum(map(len, votes)) == tune.SPEND_CHECK_EVERY - 1 and not votes[7]
    assert ledger.rows[("judge", "claude-sonnet-5")][0] == tune.SPEND_CHECK_EVERY - 1
    # A failed pair is booked and kept; a chunk of nothing but failures aborts.
    gone = verdict(verdict=tune.FAILED, reason="HTTP 503", tokens=(0, 0))
    votes, ledger, err = run(lambda t, n: gone if t == texts[3] else verdict(), 1e9)
    assert err is None and votes[3][0]["verdict"] == tune.FAILED
    votes, ledger, err = run(lambda t, n: gone, 1e9)
    assert err and "all 20 calls of a chunk failed: HTTP 503" in err, err
    # A run that fits in one chunk is never refused after it has been paid.
    votes, ledger = [[] for _ in texts[:20]], tune.Ledger()

    def paid(t):
        return verdict(tokens=(25_000, 0))

    tune.judge_passes(paid, "claude-sonnet-5", texts[:20], 1, ledger, 1.05, votes)
    assert all(len(vs) == 1 for vs in votes) and abs(ledger.usd() - 1.0) < 1e-9
    # ... but with passes left after the first chunk, the projection applies.
    votes, ledger = [[] for _ in texts[:20]], tune.Ledger()
    try:
        tune.judge_passes(paid, "claude-sonnet-5", texts[:20], 2, ledger, 1.5, votes)
    except RuntimeError as e:
        assert "project" in str(e), e
    else:
        raise AssertionError(
            "a run of 2 passes projected at $2 must stop under a $1.50 cap"
        )
    assert all(len(vs) == 1 for vs in votes) and abs(ledger.usd() - 1.0) < 1e-9
    # Uncapped, every pair gets every pass.
    votes, _, err = run(lambda t, n: verdict(), 1e9)
    assert err is None and all(len(vs) == 3 for vs in votes)
    assert tune.positive_int("3") == 3
    try:
        tune.positive_int("0")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("--passes 0 must be rejected")


def check_scrub() -> None:
    """Each scrub class becomes its placeholder, counted; nothing survives the
    check; scrubbing is idempotent. Key-shaped test strings are assembled at
    run time so the repo's secret scanners never see one in the source."""
    key_block = (
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA\n"
        "-----END " + "OPENSSH PRIVATE KEY-----"
    )
    cases = {
        "host": [
            "svc at store-api.data.svc:3001 now",
            "the gateway.corp.example.net front",
            "pushed to crates.io",
            "behind iap at app.cloud.goog",
            "resolves *.example.net and {svc}.dev.example.io",
            "mysql -hdb.prod.example.com -uroot",
            "see...api.example.com for details",
            "box build-box-01 rebooted",
        ],
        "ip": [
            "ClusterIP 192.0.2.10:3001",
            "v6 2001:db8:a1e0::1 and 2001:db8:0:0:0:0:2:1",
        ],
        "url": [
            "see https://example.com/a?b=c) for more",
            "key at op://vault/item/field",
            "redis://cache:6379/0",
            "REDIS_URL=${REDIS_URL:-redis://:pw@cache:6379/0}",
        ],
        "email": ["mail ops+alerts@example.org today"],
        "secret": [
            "header Authorization: Bearer " + "abc" * 6,
            "export API_KEY=" + "z9" * 10,
            "token: " + "q" * 12,
            # Prefixed and suffixed key names, as env vars and configs write them.
            "ALAYA_API_KEY=" + "0f" * 16,
            "GRAPH_API_KEY: " + "ab12" * 4,
            "DB_PASSWORD=" + "hunter2hunter2",
            '"client_secret": "' + "c5" * 10 + '"',
            "refresh_token = " + "r7" * 8,
            "Authorization: Basic " + "dXNlcjpw" * 3,
            "curl -u admin:" + "pa55word",
            "curl --user admin:" + "pa55word",
            "mysql --password " + "s3cr3tpass",
            "private_key: " + "k3" * 8,
            "ENCRYPTION_KEY=" + "e4" * 8,
            "passphrase: " + "p5" * 8,
            "AccountName=x;AccountKey=" + "Zm9v" * 6 + "==",
            "'apikey' => '" + "a6" * 8 + "'",
            "wh" + "sec_" + "W" * 24,
            "mysql --password '" + "Zx9Qw8Er7Ty6" + "'",
            'cli --api-key "' + "Zx9Qw8Er7Ty6" + '"',
            "os.environ['OPENAI_API_KEY'] = '" + "o8" * 8 + "'",
            'config["password"] = "' + "Zx9Qw8Er7Ty6" + '"',
            "n" + "pm_" + "B" * 36,
            "-----BEGIN PGP "
            + "PRIVATE KEY BLOCK-----\nlQOYBF\n-----END PGP PRIVATE KEY BLOCK-----",
            "sk-" + "ant-" + "x1" * 15,
            "gh" + "p_" + "A" * 36,
            "jwt eyJ" + "a" * 10 + ".eyJ" + "b" * 10 + "." + "c" * 12,
            key_block,
        ],
    }
    for cls, texts in cases.items():
        for text in texts:
            s = tune.Scrubber(["build-box-01"])
            out = s(text)
            assert f"<{cls}>" in out and s.counts[cls] >= 1, (cls, text, out)
            assert s.leaks(out) == [] and s.leaks(text), (cls, text, out)
            assert s(out) == out, (cls, out)  # idempotent
    s = tune.Scrubber()
    assert s("Bearer " + "abc" * 6) == "Bearer <secret>"
    assert s(key_block + " tail") == "<secret> tail"
    assert s("ssh -i key2 " + key_block[:40]) == "ssh -i key2 <secret>"  # cut block
    # A bracket inside a value cannot leave the value's tail behind.
    for text in (
        "DB_PASSWORD=Xk9#mQ2!vR7(pL4zQ8w",
        "f(auth_token=zC&x0JA95mJ[B#@YlL2BoR)",
    ):
        out = s(text)
        assert "pL4zQ8w" not in out and "YlL2BoR" not in out, out
        assert s(out) == out and s.leaks(out) == [], out
    # Nor can an IP, URL or email an earlier rule replaced inside the value.
    # A run of 8 or more before it is a `<secret>` already: the tail goes too.
    for text in (
        "API_KEY=" + "x9-1.2.3.4-Zq8Lm",
        "PASSWORD=" + "x9-ops@example.org-Zq8Lm",
        "TOKEN=" + "x9-1.2.3.4-5.6.7.8-Zq8Lm",
        "SECRET=" + "1.2.3.4-Zq8Lm",
        "DB_PASSWORD=" + "hunter2hunter2-10.0.0.5-Zq8Lm",
        "SECRET=" + "longprefix9@ops@example.org-Zq8Lm",
    ):
        out = s(text)
        assert "x9" not in out and "Zq8Lm" not in out and "hunter2" not in out, out
        assert out.endswith("=<secret>") and s(out) == out and s.leaks(out) == [], out
    # A value that runs into the next key's name takes that key's value too.
    for text in (
        "API_TOKEN=" + "10.0.0.5/db_password: Pa[ss]w0rd",
        "secret_key=" + "10.0.0.5&&PASSWORD => abc(Pa55w0rd)",
        "token=" + "abcdefgh/password: Pa55w0rd99",
        "token=" + "abcdefgh/password: x/api_key: Pa55w0rd99",
    ):
        out = s(text)
        assert "Pa" not in out, out
        assert s(out) == out and s.leaks(out) == [], out
    # A placeholder a later rule writes, or a `>` after one, keeps the scrub
    # idempotent: a second pass would read as a leak and stop the eval.
    for text in ("echo TOKEN=" + "abcdefgh(1)>out.txt", 'TOKEN="--key=,abcdef"'):
        out = s(text)
        assert s(out) == out and s.leaks(out) == [], out
    # A URL cannot swallow the secret-named key after it, leaving the value
    # with no key in front of it.
    for text in (
        "API_TOKEN=abcdefgh/http://intra/DB_PASSWORD: " + "Pa55w0rd99",
        "see http://intra/DB_PASSWORD : " + "Pa55w0rd99",
        "https://h.example.com/p?api_key=" + "Pa55w0rd99" + "&x=1",
        "redis://cache:6379/0?password=" + "Pa55w0rd99",
        "token=abcdefgh/password=>" + "Pa55w0rd99",
        "API_TOKEN=abcdefgh/http://intraDB_PASSWORD=>" + "Pa55w0rd99",
        # A key inside the URL stays in it, even one that looks like a key:
        # base64 padding after `pwd` in a password, a port after a host.
        "s3://user:ae8qx3WuOWbJxnvPZpwdAhUKEh==@db.example.local:5432/" + "Pa55w0rd99",
        "proxy_pass http://yg.pwdcqm.svc:17368/" + "Pa55w0rd99",
        # A URL is never cut short: its last run may be the secret itself, as
        # in the error messages Go and Python print.
        'Get "https://api.example.com/v1/x?api_key=Ab3pwd' + 'Pa55w0rd99": dial tcp',
        "fetch https://api.example.com/v1?access_token=" + "Pa55w0rd99" + ": HTTP 403",
        "webhook https://hooks.example.com/services/T0/B0/xQ9pwd"
        + "Pa55w0rd99"
        + ": 404",
        # Anything between the key and its separator ends the URL at the key.
        "http://intra/DB_PASSWORD :" + "Pa55w0rd99",
        "http://intra/DB_PASSWORD\t=" + "Pa55w0rd99",
        'http://intra/DB_PASSWORD"=' + "Pa55w0rd99",
    ):
        out = s(text)
        assert "Pa55w0rd99" not in out, (text, out)
        assert s(out) == out and s.leaks(out) == [], out
    # The check also fails closed on a value a URL cut left behind.
    assert s.leaks("<url>DB_PASSWORD: " + "Pa55w0rd99") == ["secret"]
    assert s.leaks("<url>DB_PASSWORD: <secret>") == []
    # A value ending in `=` padding or `:` takes nothing from the next line.
    padded = "SECRET=YWJj" + "ZGVmZ2hpams="
    assert s(padded + "\nNote: the rotation") == "SECRET=<secret>\nNote: the rotation"
    assert (
        s(padded + "\n\nnext paragraph here")
        == "SECRET=<secret>\n\nnext paragraph here"
    )
    # A word that holds `key` is no secret name: SECRET_NAME needs `_key` or `-key`.
    assert s("SECRET=YWJjZGVm/monkey:\nnext line") == "SECRET=<secret>\nnext line"
    # ...but a value that ran into a secret key still takes that key's value
    # from the next line.
    # The key may close with what SECRET_KEY_NAME accepts, and its separator
    # may follow on the next line.
    for text in (
        "token=abcdefgh/password:\n  ",
        "token=abcdefgh/DB_PASSWORD :\n\t",
        "token=abcdefgh/DB_PASSWORD]:\n",
        'token=abcdefgh/DB_PASSWORD"]:\n',
        "token=abcdefgh/DB_PASSWORD']=>\n",
        "token=abcdefgh/DB_PASSWORD\n: ",
        # A secret word anywhere in the name, as SECRET_NAME reads it.
        "token=abcdefgh/DB_PASSWORD_PROD:\n  ",
        "token=abcdefgh/SECRET_KEY_ID:\n",
        "token=abcdefgh/API_TOKEN_V2]:\n",
        "token=abcdefgh/password_hash\n: ",
        # ...at any length, as SECRET_NAME has no limit either.
        "token=abcdefgh/X_API_KEY_APPLICATION_PRODUCTION_EU_WEST_01:\n",
        "token=abcdefgh/DB_PASSWORD_" + "X" * 300 + "]:\n",
    ):
        out = s(text + "Pa55w0rd99")
        assert "Pa55w0rd99" not in out and s.leaks(out) == [], out
    # A next-line value may end in, or be, another such key: the chain goes whole.
    for text in (
        "password:\nYWJjZGVmZ2hp/api_key:\n",
        "token=abcdefgh/password:\nYWJjZGVmZ2hp/api_key:\n",
        "password:\nab/SECRET:\nYWJjZGVmZ2hp/x-api-key\n= ",
        "password:\nmy_token_value:\n",
        "secrets:\n  db_password:\n    value: ",
        # ...however an earlier rule's value ran into the key.
        "http://intra/DB_PASSWORD: abcdefgh/api_key:\n",
        "Authorization: Bearer abcdefgh/api_key:\n",
        "authorization: Basic abcdefgh/api_key=\n",
        "curl -u user:abcdefgh/api_key:\n",
        "cli --password abcdefgh/api_key:\n",
    ):
        out = s(text + "Pa55w0rd99")
        assert "Pa55w0rd99" not in out and "YWJj" not in out, out
        assert s(out) == out and s.leaks(out) == [], out
    # A secret-named value cannot swallow the next secret-named key.
    out = s(':auth_token => login(password: "' + "Zx9Qw8Er7Ty6" + '")')
    assert "Zx9Qw8Er7Ty6" not in out and s.leaks(out) == [], out
    # Every rule stays near linear on long adversarial runs (some were
    # quadratic or worse: 64k characters took minutes).
    for run in (
        "x",
        "a.",
        "1.",
        "_key",
        "token",
        "--key",
        "-",
        "a@",
        "ab:",
        "pwd:\n",
        "pwd:\nx ",
    ):
        text = (run * 64_000)[:64_000]
        start = tune.time.perf_counter()
        s(text)
        took = tune.time.perf_counter() - start
        assert took < 1.0, (run, took)
    # Prose, file names, versions, times and hashes are not a scrub class.
    plain = (
        "Ray ruled on tune.py and judge.rs (v1.13.0) at 12:30:45; max_tokens=4096, "
        "token budget 5, std::fs, hash 9999d3f16a2030def3b3479ef273318194c8e03f, "
        "e.g. the bearer token; lab node; pinned cachekit@0.1.4 and action@v3.2.0; "
        "systemctl --user restart gateway"
    )
    assert s(plain) == plain and s.leaks(plain) == [], s(plain)
    # Bare names come only from the host list, longest first, as whole words.
    s = tune.Scrubber(["box", "box-wsl"])
    assert s("box-wsl and box, not boxing") == "<host> and <host>, not boxing"
    assert s.counts == Counter(host=2)
    try:
        tune.read_host_names("no-such-host-list.txt")
    except SystemExit as e:
        assert "no-such-host-list.txt" in str(e), e
    else:
        raise AssertionError("a missing host list must exit, naming it")
    # The rendered pair carries scrubbed content and tags, and checks clean.
    s = tune.Scrubber()
    a = s.memory(mem("db at 10.0.0.5 via db.internal", 0.0, tags=("x.svc", "plain")))
    assert a["content"] == "db at <ip> via <host>" and a["tags"] == ["<host>", "plain"]
    text = tune.render_pair(a, s.memory(mem("ok", 86_400.0 * 2)))
    assert s.leaks(text) == [] and "tags: <host>, plain" in text, text
    assert s.counts == Counter(ip=1, host=2)

    # Keys reach HTTP headers: whitespace would make httpx echo them in errors.
    real_env = dict(tune.os.environ)
    try:
        tune.os.environ["X_KEY"] = "abc\r"
        try:
            tune.env("X_KEY")
        except SystemExit as e:
            assert "non-printable" in str(e) and "abc" not in str(e)
        else:
            raise AssertionError("a key with a CR must be refused")
    finally:
        tune.os.environ.clear()
        tune.os.environ.update(real_env)
    check_egress(real_env)
    check_no_env_proxy(real_env)


def check_no_env_proxy(real_env: dict) -> None:
    """Every client that carries a key or memory text dials the certified
    host itself: with HTTP_PROXY and ALL_PROXY pointing at a local listener,
    none reaches it. A control client that trusts the environment does, so the
    probe is live."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    listener.settimeout(0.2)
    proxy = f"http://127.0.0.1:{listener.getsockname()[1]}"
    target = "http://judge-under-test.invalid:8082"  # never resolves directly

    def proxied(call) -> int:
        try:
            call()
        except (tune.httpx.HTTPError, tune.anthropic.APIError):
            pass  # no server answers: only where the request went matters
        hits = 0
        while True:
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                return hits
            conn.close()
            hits += 1

    try:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            tune.os.environ[var] = proxy
        tune.os.environ["ALAYA_URL"] = "http://alaya-under-test.svc:3001"
        tune.os.environ["ALAYA_API_KEY"] = "test-only"
        control = tune.httpx.Client(timeout=2)
        assert proxied(lambda: control.get(target)) == 1, "the probe must see a proxy"
        assert proxied(lambda: tune.http_client(target, "t").get("/")) == 0
        assert proxied(lambda: tune.alaya_client().get("/health")) == 0
        sdk = tune.make_client(target, "t", max_retries=0).with_options(timeout=2)
        call = lambda: sdk.messages.create(  # noqa: E731
            model="claude-sonnet-5",
            max_tokens=1,
            messages=[{"role": "user", "content": "x"}],
        )
        assert proxied(call) == 0
    finally:
        listener.close()
        tune.os.environ.clear()
        tune.os.environ.update(real_env)


def check_egress(real_env: dict) -> None:
    """A key or memory text leaves only over https, or plain http to a
    cluster-local host (alaya-server's boot rule); unscrubbed pairs go only to
    a Claude model at an approved origin."""
    # alaya-server's `cluster_local_accepts_service_dns_and_private_hosts_only`.
    for ok in (
        "http://anthropic-lb:8082",
        "http://alaya-bridge:3000",
        "http://alaya-server.mcp.svc:3001",
        "http://localhost:8082",
        "http://10.43.144.201:8082",
        "http://[::1]:8082",
        "http://[fd12::1]:80",
        "http://u:p@anthropic-lb:8082",
        "http://2130706433:8082",  # 127.0.0.1, as the resolver reads it
        "http://0x0a000001",  # 10.0.0.1
    ):
        assert tune.is_cluster_local(ok), ok
    for no in (
        "http://api.anthropic.com",
        "http://proxy.example.net:8082",
        "http://1.2.3.4",
        "http://100.64.0.1:8082",  # shared address space is not private
        "http://u:p@api.anthropic.com",
        "http://anthropic-lb:8082@api.anthropic.com",  # userinfo never poses as host
        "http://[2606:4700::1111]",
        "http://[::ffff:1.2.3.4]",
        "http://16843009",  # 1.1.1.1 to the resolver, though it has no dot
        "http://0x01010101",
    ):
        assert not tune.is_cluster_local(no), no

    def exits(fn, *args) -> str:
        try:
            fn(*args)
        except SystemExit as e:
            return str(e)
        raise AssertionError(f"{fn.__name__}{args} must exit")

    try:
        for url in ("https://proxy.example.net", "http://anthropic-lb:8082"):
            tune.os.environ["X_URL"] = url
            assert tune.egress_url("X_URL") == url
        for url in (
            "http://proxy.example.net",
            "ftp://anthropic-lb",
            "anthropic-lb:8082",
        ):
            tune.os.environ["X_URL"] = url
            assert "must be https" in exits(tune.egress_url, "X_URL"), url
        tune.os.environ.pop("UNSCRUBBED_JUDGE_ORIGINS", None)
        tune.unscrubbed_ok("claude-sonnet-5", "https://api.anthropic.com")
        tune.unscrubbed_ok("claude-sonnet-5", "https://api.anthropic.com:443/")
        # A claude-* name behind an unapproved origin is no Anthropic judge.
        assert "UNSCRUBBED_JUDGE_ORIGINS" in exits(
            tune.unscrubbed_ok, "claude-sonnet-5", "https://proxy.example.net"
        )
        assert "only claude-*" in exits(
            tune.unscrubbed_ok, "gemini-3.8-flash", "https://api.anthropic.com"
        )
        tune.os.environ["UNSCRUBBED_JUDGE_ORIGINS"] = (
            "http://anthropic-lb:8082, https://lb.example.net"
        )
        tune.unscrubbed_ok("claude-sonnet-5", "http://anthropic-lb:8082/")
        tune.unscrubbed_ok("claude-opus-5", "https://lb.example.net:443")
        exits(tune.unscrubbed_ok, "claude-sonnet-5", "https://lb.example.net:8443")
        exits(tune.unscrubbed_ok, "gpt-6-sol", "https://lb.example.net")
    finally:
        tune.os.environ.clear()
        tune.os.environ.update(real_env)


def check_judges() -> None:
    """The judge seam: retry once, then fail; OpenAI and Jev replies."""
    tries = []

    def flaky(t):
        tries.append(t)
        if len(tries) == 1:
            raise tune.ApiError("HTTP 503", (5, 1))
        return verdict(tokens=(10, 2))

    real_sleep = tune.time.sleep
    tune.time.sleep = lambda s: None
    try:
        v = tune.retried(flaky)("t")
        assert v["verdict"] == "coexist" and v["tokens"] == (15, 3) and len(tries) == 2

        def down(t):
            raise tune.ApiError("HTTP 400: content filter " + "x" * 300, (7, 0))

        v = tune.retried(down)("t")
        assert v["verdict"] == tune.FAILED and v["tokens"] == (14, 0)
        assert v["reason"].startswith("HTTP 400: content filter")
        assert len(v["reason"]) == tune.MAX_REASON_CHARS

        def locked(t):
            raise tune.AuthError("HTTP 401")

        try:
            tune.retried(locked)("t")
        except tune.AuthError:
            pass
        else:
            raise AssertionError("an auth error must propagate")
    finally:
        tune.time.sleep = real_sleep

    # The request openai.rs sends: no temperature, strict schema.
    body = tune.openai_body("gpt-6-sol", "sys", "pair")
    assert "temperature" not in body and "max_tokens" not in body
    assert body["max_completion_tokens"] == tune.MAX_OUTPUT_TOKENS
    assert body["response_format"]["json_schema"]["strict"] is True
    assert [m["role"] for m in body["messages"]] == ["system", "user"]

    def chat(content=None, finish="stop", refusal=None, usage=True):
        msg = {"role": "assistant", "content": content, "refusal": refusal}
        out = {"choices": [{"message": msg, "finish_reason": finish}]}
        if usage:
            out["usage"] = {"prompt_tokens": 100, "completion_tokens": 20}
        return out

    good = json.dumps(verdict(verdict="supersession", survivor="a", confidence=0.9))
    v = tune.openai_verdict(chat(good))
    assert (v["verdict"], v["survivor"], v["tokens"]) == (
        "supersession",
        "a",
        (100, 20),
    )
    assert tune.openai_verdict(chat(good, usage=False))["tokens"] == (0, 0)
    assert tune.openai_verdict(chat(None))["verdict"] == "unjudged"
    assert tune.openai_verdict(chat("not json"))["verdict"] == "unjudged"
    assert tune.openai_verdict({"choices": []})["verdict"] == "unjudged"
    for bad in (chat(good, finish="content_filter"), chat(None, refusal="no")):
        try:
            tune.openai_verdict(bad)
        except tune.ApiError as e:
            assert e.tokens == (100, 20), e.tokens
        else:
            raise AssertionError("a content filter or refusal is an API error")

    def fake_client(stop: str, text: str = "") -> SimpleNamespace:
        usage = SimpleNamespace(
            input_tokens=50,
            output_tokens=3,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        )
        blk = SimpleNamespace(type="text", text=text)
        resp = SimpleNamespace(stop_reason=stop, usage=usage, content=[blk])
        return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))

    def ask(client) -> dict:
        return tune.anthropic_judge(client, "claude-sonnet-5", "sys", "default")("p")

    assert ask(fake_client("end_turn", good))["verdict"] == "supersession"
    try:
        ask(fake_client("refusal", good))  # even a parseable verdict
    except tune.ApiError as e:
        assert e.tokens == (50, 3) and "refused" in str(e), e
    else:
        raise AssertionError("an Anthropic refusal is an API error in eval")
    # tune reads a refusal's text as production does.
    v = tune.judge_pair(fake_client("refusal", good), "claude-sonnet-5", "sys", "p")
    assert v["verdict"] == "supersession" and v["tokens"] == (50, 3), v

    def jev(verdicts, survivors, hide=(0.2, 0.8)):
        return {
            "model": tune.JEV_MODEL,
            "answers": {
                "verdict": {"type": "choice", "probabilities": verdicts},
                "survivor": {"type": "choice", "probabilities": survivors},
                "hide_a": {"type": "noul", "noul": hide[0]},
                "hide_b": {"type": "noul", "noul": hide[1]},
            },
            "usage": {"input_tokens": 900, "output_tokens": 40},
        }

    probs = {
        "contradiction": 0.05,
        "supersession": 0.6,
        "coexist": 0.3,
        "unrelated": 0.05,
    }
    v = tune.jev_verdict(jev(probs, {"a": 0.2, "b": 0.1, "neither": 0.7}))
    # A supersession names a survivor even when "neither" leads.
    assert (v["verdict"], v["survivor"], v["confidence"]) == ("supersession", "a", 0.6)
    assert v["hide"] == {"a": 0.2, "b": 0.8} and v["tokens"] == (900, 40)
    # A Jev verdict is one `validate` accepts, as every judge's must be.
    assert v["model_version"] == tune.JEV_MODEL
    assert tune.validate(v, v["tokens"]) == {
        k: v[k] for k in ("verdict", "survivor", "reason", "confidence", "tokens")
    }
    probs = {
        "contradiction": 0.7,
        "supersession": 0.1,
        "coexist": 0.1,
        "unrelated": 0.1,
    }
    v = tune.jev_verdict(jev(probs, {"a": 0.2, "b": 0.1, "neither": 0.7}))
    assert (v["verdict"], v["survivor"]) == ("contradiction", None)
    assert v["survivor_probs"]["neither"] == 0.7
    try:
        tune.jev_verdict({"answers": {"verdict": {}}, "usage": {"input_tokens": 9}})
    except tune.ApiError as e:
        assert e.tokens == (9, 0) and "malformed" in str(e)
    else:
        raise AssertionError("a malformed Jev answer is an API error")
    assert set(tune.JEV_QUESTIONS) == {"verdict", "survivor", "hide_a", "hide_b"}
    assert list(tune.JEV_QUESTIONS["verdict"]["criteria"]) == list(tune.CLASSES)


def check_rows() -> None:
    """Prefix rows resolve to exactly one memory pair, or are dropped."""

    def edge(a, b, verdict="supersession", survivor=None, conf=0.95):
        return {
            "memory_a_hash": a,
            "memory_b_hash": b,
            "verdict": verdict,
            "survivor": survivor,
            "verdict_confidence": conf,
            "verdict_model": "claude-sonnet-5",
        }

    h = {k: k * 64 for k in "abcdef"}
    clash = "e" * 12 + "0" * 52  # shares e's 12-char prefix
    rows = [
        {"row": 1, "survivor": "a" * 12, "loser": "b" * 12, "flag": True},
        {"row": 2, "survivor": "c" * 12, "loser": "d" * 12},
        {"row": 3, "survivor": "e" * 12, "loser": "f" * 12},
        {"row": 4, "survivor": "9" * 12, "loser": "8" * 12},
    ]
    edges = [
        edge(h["b"], h["a"], "coexist", None, 0.6),  # the reverse direction
        edge(h["a"], h["b"], "supersession", h["a"], 0.97),
        edge(h["d"], h["c"], "supersession", h["c"], 0.91),
        edge(h["e"], h["f"]),
        edge(clash, h["f"]),
    ]
    done, ambiguous, unmatched = tune.resolve_rows(rows, edges)
    assert [r["row"] for r in done] == [1, 2]
    assert [r["row"] for r in ambiguous] == [3] and [r["row"] for r in unmatched] == [4]
    r1, r2 = done
    assert (r1["a"], r1["b"], r1["survivor"], r1["loser"]) == (
        h["a"],
        h["b"],
        h["a"],
        h["b"],
    )
    assert r1["prod_confidence"] == 0.97 and r1["flag"] is True
    assert (r2["a"], r2["survivor"], r2["loser"]) == (h["d"], h["c"], h["d"])
    assert tune.check_prefixes(rows) == rows
    ok = {"row": 5, "survivor": "a" * 12, "loser": "b" * 12}
    for bad in (
        {"rows": rows},
        [7],
        [ok, ok],  # a row number twice
        [ok | {"row": "5"}],
        [ok | {"row": True}],
        [{"row": 5, "survivor": "a" * 12}],
        [ok | {"loser": ""}],  # an empty prefix matches every hash
        [ok | {"loser": "B" * 12}],  # hashes are lowercase hex
        [ok | {"loser": "a" * 12}],
        [ok, ok | {"row": 6, "survivor": ok["loser"], "loser": ok["survivor"]}],
    ):
        try:
            tune.check_prefixes(bad)
        except SystemExit as e:
            assert "--prefixes" in str(e), e
        else:
            raise AssertionError(f"bad prefixes must exit: {bad}")


def check_compare() -> None:
    """Agreement, safe-to-hide rates and the auto-apply vote."""
    assert tune.kappa(["s", "c", "s", "c"], ["s", "c", "s", "c"]) == 1.0
    assert tune.kappa(["s", "s", "c", "c"], ["s", "c", "s", "c"]) == 0.0
    assert tune.kappa(["s", "s"], ["s", "s"]) is None and tune.kappa([], []) is None
    k = tune.kappa(list("sssscc"), list("ssscsc"))
    assert abs(k - 0.25) < 1e-12, k  # po 4/6, pe (4*4 + 2*2) / 36
    assert tune.decide([verdict(verdict=tune.FAILED)]) is None
    assert tune.decide([verdict(verdict="unjudged")])["verdict"] == tune.ABSTAIN
    d = tune.decide([verdict(confidence=0.9, hide={"a": 0.3, "b": 0.8})])
    assert d["confidence"] == 0.9 and d["hide"] == {"a": 0.3, "b": 0.8}

    a, b, c, e = ("a" * 64, "b" * 64, "c" * 64, "e" * 64)
    s1 = tune.Pair(0, a, b, "supersession", a, "operator")  # loser b
    s2 = tune.Pair(1, c, e, "supersession", e, "operator")  # loser c
    co = tune.Pair(2, "1" * 64, "2" * 64, "coexist", None, "agreed")
    pairs = [s1, s2, co]

    def sup(surv: str, conf: float = 0.95) -> dict:
        return {"verdict": "supersession", "survivor": surv, "confidence": conf}

    primary = {0: sup("a"), 1: sup("b", 0.80), 2: sup("a")}
    alone = tune.auto_apply(pairs, primary)
    assert (alone["applied"], alone["correct"]) == (2, 1)
    assert alone["recall"]["n"] == 2 and alone["precision"]["rate"] == 0.5
    jev = {0: {"hide": {"a": 0.1, "b": 0.95}}, 2: {"hide": {"a": 0.2, "b": 0.6}}}
    seconds = tune.second_votes({"p": primary, "jev-1.13.0": jev}, "p", "jev-1.13.0")
    assert sorted(seconds) == ["jev-1.13.0"] + [
        f"jev-1.13.0 hide>={t}" for t in tune.HIDE_THRESHOLDS
    ]
    judge, agrees = seconds["jev-1.13.0 hide>=0.7"]
    voted = tune.auto_apply([p for p in pairs if p.id in jev], primary, agrees)
    assert judge == "jev-1.13.0" and (voted["applied"], voted["correct"]) == (1, 1)

    rows = [(s1, {"hide": {"a": 0.1, "b": 0.8}}), (co, {"hide": {"a": 0.85, "b": 0.2}})]
    r = tune.hide_rates(rows)
    at7 = r["thresholds"]["0.7"]
    assert (at7["precision"]["k"], at7["precision"]["n"], at7["recall"]["n"]) == (
        1,
        2,
        1,
    )
    assert r["thresholds"]["0.9"]["precision"]["n"] == 0
    # The one loser (0.8) outranks 2 of the 3 other endpoints.
    assert r["auc"] == 2 / 3 and r["median_loser"] == 0.8 and r["median_other"] == 0.2
    assert tune.hide_rates(rows[1:])["auc"] is None
    agree = tune.agreement({"x": {0: verdict(), 1: verdict()}, "y": {1: verdict()}})
    assert agree == [
        {"judges": ["x", "y"], "n": 1, "raw": tune.rate(1, 1), "kappa": None}
    ]
    assert len(tune.fixture_provenance()["git_blob"]) == 40
    check_compare_run()


def check_compare_run() -> None:
    """compare end to end over a fake run directory: a failed second vote is
    left out, never a veto; sent counts include an eval that aborted."""
    pairs = tune.load_fixture()
    sups = [p for p in pairs if p.label == "supersession"]
    flaky = sups[0]  # the second judge fails here

    def vote(p, failed=False, hide=None):
        v = {"verdict": p.label, "survivor": p.survivor_letter(), "confidence": 0.95}
        v["survivor"] = v["survivor"] and v["survivor"].lower()
        if failed:
            v = {"verdict": tune.FAILED, "survivor": None, "confidence": None}
        if hide is not None:
            v["hide"] = hide
        return v | {"a": p.a, "b": p.b, "newer": p.b, "reason": ""}

    def write_eval(out, stem, judge, model, votes, scrubbed=True, counts=None):
        (out / f"{stem}_records.jsonl").write_text(
            "".join(json.dumps(v) + "\n" for v in votes)
        )
        result = {"judge": judge, "judge_model": model, "pairs": "all", "passes": 1}
        result |= {"scrubbed": scrubbed, "records": f"{stem}_records.jsonl"}
        result |= {"scrub_counts": counts or {"host": 3}, "sent": {"memories": 274}}
        (out / f"{stem}.json").write_text(json.dumps(result))
        sent = {
            "judge": judge,
            "scrubbed": scrubbed,
            "pairs": [[p.a, p.b] for p in pairs],
        }
        (out / f"{stem}_sent.json").write_text(json.dumps(sent))

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_eval(
            out, "eval_s", "anthropic", "claude-sonnet-5", [vote(p) for p in pairs]
        )
        write_eval(
            out, "eval_g", "openai", "gpt-6-sol", [vote(p, p is flaky) for p in pairs]
        )
        hide = {"a": 0.9, "b": 0.9}
        write_eval(
            out, "eval_j", "jev", "jev-1.13.0", [vote(p, hide=hide) for p in pairs]
        )
        # An eval that aborted before its result: still counted as sent.
        aborted = {"judge": "jev", "scrubbed": True, "pairs": [["f" * 64, "e" * 64]]}
        (out / "eval_x_sent.json").write_text(json.dumps(aborted))
        (out / tune.SPEND_LOG).write_text(
            json.dumps({"judge": "openai", "total_usd": 1.5})
            + "\n"
            + json.dumps({"judge": "jev", "total_usd": 0.25})
            + "\n"
        )
        # README step 3's k-pass eval, or an older one with no judge, can share
        # the run directory; compare skips it.
        for stem, result in (("eval_k3", {"judge": "anthropic"}), ("eval_old", {})):
            result |= {"pairs": "all", "passes": 3, "records": f"{stem}_records.jsonl"}
            (out / f"{stem}.json").write_text(json.dumps(result))
        evals = tune.load_evals(out)
        g = tune.golden_section(evals["golden"], "claude-sonnet-5")
        vv = g["vote_value"]["partial_as_coexist"]
        assert vv["alone"]["recall"]["k"] == vv["alone"]["recall"]["n"] == len(sups)
        # The failed pair leaves the gpt-6-sol rule's denominator; no veto.
        assert vv["gpt-6-sol"]["recall"]["n"] == len(sups) - 1
        assert vv["gpt-6-sol"]["recall"]["k"] == len(sups) - 1
        assert g["judges"]["gpt-6-sol"]["failed"] == 1
        assert vv["jev-1.13.0 hide>=0.9"]["applied"] == len(sups)
        sent = tune.sent_by_vendor(out)
        assert sent["jev"]["pairs"] == len(pairs) + 1 and sent["openai"][
            "pairs"
        ] == len(pairs)
        assert sent["anthropic"]["unscrubbed_pairs"] == 0
        assert tune.spend_by_vendor(out) == {"jev": 0.25, "openai": 1.5, "total": 1.75}
        assert tune.scrub_section(evals)["golden"]["host"] == 3
        assert "## Golden set" in tune.compare_markdown(
            {
                "fixture": tune.fixture_provenance(),
                "primary": "claude-sonnet-5",
                "auto_apply_confidence": tune.AUTO_APPLY,
                "spend_usd": {},
                "sent": sent,
                "scrub": {},
                "golden": g,
                "rows": None,
            }
        )
        try:
            tune.golden_section(evals["golden"], "claude-sonnet-5-typo")
        except SystemExit:
            pass
        else:
            raise AssertionError("an unknown --primary must exit")
        # A scored eval that never recorded what it sent would undercount.
        (out / "eval_j_sent.json").rename(out / "eval_j_sent.bak")
        try:
            tune.load_evals(out)
        except SystemExit as e:
            assert "_sent.json" in str(e), e
        else:
            raise AssertionError("an eval without its _sent.json must exit")
        (out / "eval_j_sent.bak").rename(out / "eval_j_sent.json")
        # Different scrub counts mean different text reached different judges.
        write_eval(out, "eval_r", "anthropic", "claude-opus-5", [], counts={"host": 4})
        try:
            tune.scrub_section(tune.load_evals(out))
        except SystemExit:
            pass
        else:
            raise AssertionError("mismatched scrub counts must exit")
        # An aborted append can tear the last spend line: the cap must not
        # undercount, so the run stops and names the line.
        with (out / tune.SPEND_LOG).open("a", encoding="utf-8") as f:
            f.write('{"judge": "jev", "tot')
        try:
            tune.spend_log(out)
        except SystemExit as e:
            assert f"{tune.SPEND_LOG}:3" in str(e), e
        else:
            raise AssertionError("a torn spend line must exit")
        with (out / "eval_s_records.jsonl").open("a", encoding="utf-8") as f:
            f.write('{"pair": 0, "a": "')
        try:
            tune.load_evals(out)
        except SystemExit as e:
            assert "eval_s_records.jsonl:" in str(e), e
        else:
            raise AssertionError("a torn records line must exit")


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
    check_scrub()
    check_judges()
    check_rows()
    check_compare()

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
