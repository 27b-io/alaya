//! Integration test for `POST /contradictions/stats` (LAB-6881) against a
//! real FalkorDB: every aggregate, and the queue's resolved definition — a
//! stamp in either direction or a superseded endpoint. Skipped when
//! `REDIS_URL` is unset; see `integration_contradictions.rs` for a local run.

mod common;

use std::sync::Arc;

use alaya_bridge::{cypher, handlers::contradictions};
use alaya_types::graph::{
    ContradictionStatsQuery, DayTally, EdgeVerdict, ReasonTally, Resolution, SystemRelationType,
    UserRelationType, Verdict, VerdictCount, VerdictTally,
};
use axum::{Json, extract::State};

const DAY: f64 = 86_400.0;
/// UTC midnight, day 20_000 since the epoch.
const T0: f64 = 20_000.0 * DAY;

fn hash(i: usize) -> String {
    format!("{i:064x}")
}

async fn edge(ctx: &common::TestContext, a: &str, b: &str) {
    for h in [a, b] {
        ctx.exec_tuple(cypher::ensure_node(h, T0)).await;
    }
    ctx.exec_tuple(cypher::create_typed_edge(
        a,
        b,
        UserRelationType::Contradicts,
        T0,
        Some(0.7),
    ))
    .await;
}

async fn judge(
    ctx: &common::TestContext,
    a: &str,
    b: &str,
    verdict: Verdict,
    reason: &str,
    at: f64,
) {
    ctx.exec_tuple(cypher::set_contradiction_verdict(
        a,
        b,
        &EdgeVerdict {
            verdict,
            verdict_survivor: None,
            verdict_reason: reason.into(),
            verdict_confidence: 0.9,
            verdict_model: "m".into(),
            judged_at: at,
        },
    ))
    .await;
}

async fn keep_both(ctx: &common::TestContext, a: &str, b: &str) {
    ctx.exec_tuple(cypher::set_contradiction_resolution(
        a,
        b,
        Some(Resolution::KeepBoth),
        "operator:test",
        T0,
    ))
    .await;
}

fn tally(verdict: Option<&str>, resolved: bool, count: usize) -> VerdictTally {
    VerdictTally {
        verdict: verdict.map(str::to_string),
        resolved,
        count,
    }
}

fn sorted<T: Clone, K: Ord>(v: &[T], key: impl Fn(&T) -> K) -> Vec<T> {
    let mut v = v.to_vec();
    v.sort_by_key(|t| key(t));
    v
}

#[tokio::test]
async fn stats_count_every_edge_once_under_the_queue_resolved_definition() -> anyhow::Result<()> {
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    let h: Vec<String> = (0..20).map(hash).collect();

    // Open supersession, 2-char reason: degenerate.
    edge(&ctx, &h[0], &h[1]).await;
    judge(&ctx, &h[0], &h[1], Verdict::Supersession, "ok", T0).await;
    // Coexist stamped keep_both on itself; padded `Placeholder`: degenerate.
    edge(&ctx, &h[2], &h[3]).await;
    judge(
        &ctx,
        &h[2],
        &h[3],
        Verdict::Coexist,
        "  Placeholder ",
        T0 + DAY,
    )
    .await;
    keep_both(&ctx, &h[2], &h[3]).await;
    // A pair settled by a stamp on the REVERSE edge only: both directions
    // count as resolved, neither twice.
    edge(&ctx, &h[4], &h[5]).await;
    judge(
        &ctx,
        &h[4],
        &h[5],
        Verdict::Coexist,
        "both hold, years apart",
        T0 + DAY,
    )
    .await;
    edge(&ctx, &h[5], &h[4]).await;
    judge(
        &ctx,
        &h[5],
        &h[4],
        Verdict::Unrelated,
        "shared words only",
        T0,
    )
    .await;
    keep_both(&ctx, &h[5], &h[4]).await;
    // Never judged, endpoint superseded: resolved backlog.
    edge(&ctx, &h[6], &h[7]).await;
    ctx.exec_tuple(cypher::ensure_node(&h[19], T0)).await;
    ctx.exec_tuple(cypher::create_system_edge(
        &h[19],
        &h[7],
        SystemRelationType::Supersedes,
        T0,
    ))
    .await;
    // Never judged, open: the backlog.
    edge(&ctx, &h[8], &h[9]).await;
    // Stored failures. A short marker reason is not a degenerate judge reason.
    for (i, reason) in [
        (10, "unjudged: boom"),
        (12, "unjudged: boom"),
        (14, "unjudged:"),
    ] {
        edge(&ctx, &h[i], &h[i + 1]).await;
        judge(
            &ctx,
            &h[i],
            &h[i + 1],
            Verdict::Unjudged,
            reason,
            T0 + 2.0 * DAY,
        )
        .await;
    }
    // Judged before the window: counted everywhere but per-day.
    edge(&ctx, &h[16], &h[17]).await;
    judge(
        &ctx,
        &h[16],
        &h[17],
        Verdict::Contradiction,
        "a long enough reason",
        T0 - DAY,
    )
    .await;

    let state = Arc::new(common::build_state(&ctx.conn, &ctx.graph_name));
    let Json(stats) = contradictions::stats(
        State(state),
        Json(ContradictionStatsQuery { judged_since: T0 }),
    )
    .await
    .expect("stats");

    let key = |t: &VerdictTally| (t.verdict.clone(), t.resolved);
    assert_eq!(
        sorted(&stats.verdicts, key),
        sorted(
            &[
                tally(Some("supersession"), false, 1),
                tally(Some("coexist"), true, 2),
                tally(Some("unrelated"), true, 1),
                tally(Some("contradiction"), false, 1),
                tally(Some("unjudged"), false, 3),
                tally(None, true, 1),
                tally(None, false, 1),
            ],
            key
        ),
    );
    assert_eq!(
        stats.failures,
        [
            ReasonTally {
                reason: "unjudged: boom".into(),
                count: 2
            },
            ReasonTally {
                reason: "unjudged:".into(),
                count: 1
            },
        ],
        "most frequent first"
    );
    let day = |d: i64, v: &str, n: usize| DayTally {
        day: 20_000 + d,
        verdict: Some(v.into()),
        count: n,
    };
    let day_key = |t: &DayTally| (t.day, t.verdict.clone());
    assert_eq!(
        sorted(&stats.judged_per_day, day_key),
        sorted(
            &[
                day(0, "supersession", 1),
                day(0, "unrelated", 1),
                day(1, "coexist", 2),
                day(2, "unjudged", 3),
            ],
            day_key
        ),
    );
    let vc = |v: &str, n: usize| VerdictCount {
        verdict: v.into(),
        count: n,
    };
    assert_eq!(
        sorted(&stats.degenerate_reasons, |t| t.verdict.clone()),
        [vc("coexist", 1), vc("supersession", 1)],
    );

    ctx.cleanup().await;
    Ok(())
}

/// A fresh deployment has no graph key yet: every aggregate is empty, not
/// an error.
#[tokio::test]
async fn stats_on_an_absent_graph_are_empty() -> anyhow::Result<()> {
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    let state = Arc::new(common::build_state(&ctx.conn, &ctx.graph_name));
    let Json(stats) = contradictions::stats(
        State(state),
        Json(ContradictionStatsQuery { judged_since: 0.0 }),
    )
    .await
    .expect("stats");
    assert_eq!(stats, Default::default());
    Ok(())
}
