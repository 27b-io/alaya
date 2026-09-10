//! Integration tests for the contradiction-pair selection Cypher against a
//! real FalkorDB (LAB-3283 AC-5/AC-6 as amended after review). Skipped when
//! `REDIS_URL` is unset. Run locally with a throwaway FalkorDB, e.g.
//! `podman run -d -p 16380:6379 falkordb/falkordb:v4.18.6` then
//! `REDIS_URL=redis://localhost:16380 cargo test -p alaya-bridge --test integration_contradictions`.

mod common;

use alaya_bridge::cypher;
use alaya_types::graph::{
    ContradictionQuery, EdgeVerdict, SystemRelationType, UserRelationType, Verdict,
};

fn hash(i: usize) -> String {
    format!("{i:064x}")
}

/// `n` CONTRADICTS pairs `(a_i)->(b_i)`, `i = 0` newest. The newest
/// `resolved_newest` pairs get a SUPERSEDES edge onto `b_i` (what
/// `mark_superseded` writes), i.e. they are resolved graph-side.
async fn seed_pairs(ctx: &common::TestContext, n: usize, resolved_newest: usize) {
    for i in 0..n {
        let (a, b) = (hash(1000 + i), hash(2000 + i));
        let ts = 1_800_000_000.0 - i as f64;
        ctx.exec_tuple(cypher::ensure_node(&a, ts)).await;
        ctx.exec_tuple(cypher::ensure_node(&b, ts)).await;
        ctx.exec_tuple(cypher::create_typed_edge(
            &a,
            &b,
            UserRelationType::Contradicts,
            ts,
            Some(0.7),
        ))
        .await;
        if i < resolved_newest {
            // A third memory supersedes b_i.
            let winner = hash(3000 + i);
            ctx.exec_tuple(cypher::ensure_node(&winner, ts + 1.0)).await;
            ctx.exec_tuple(cypher::create_system_edge(
                &winner,
                &b,
                SystemRelationType::Supersedes,
                ts + 1.0,
            ))
            .await;
        }
    }
}

fn a_hashes(result: &alaya_bridge::FalkorResult) -> Vec<String> {
    result
        .result_set
        .iter()
        .map(|row| row[0].as_str().unwrap().to_string())
        .collect()
}

/// AC-6 (amended): more resolved pairs at the top than one page holds, and
/// `limit = N` still returns N unresolved pairs; `SKIP` pages through them.
#[tokio::test]
async fn exclude_resolved_reaches_past_a_resolved_run_larger_than_the_page() -> anyhow::Result<()> {
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    // 12 pairs, the newest 8 resolved; page size 3.
    seed_pairs(&ctx, 12, 8).await;

    let page = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 3,
            exclude_resolved: true,
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&page), [hash(1008), hash(1009), hash(1010)]);

    let page2 = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 3,
            skip: 3,
            exclude_resolved: true,
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&page2), [hash(1011)], "only 4 unresolved exist");

    // Without the filter the same page is the resolved run.
    let all = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 3,
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&all), [hash(1000), hash(1001), hash(1002)]);

    ctx.cleanup().await;
    Ok(())
}

/// AC-5 (amended): a persisted `unjudged` marker leaves the backfill's
/// NULL selection (no poison-pill loop) but still reads as `unjudged` on the
/// read surface, and the re-judge selection picks up other models' edges.
#[tokio::test]
async fn needs_judging_skips_marked_failures_and_rejudge_selects_other_models() -> anyhow::Result<()>
{
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    seed_pairs(&ctx, 4, 0).await;

    // Pair 1: judged by model X. Pair 2: deterministic failure marker by X.
    for (i, verdict, reason) in [
        (1usize, Verdict::Coexist, "both true"),
        (2, Verdict::Unjudged, "unjudged: verdict is not valid JSON"),
    ] {
        let r = ctx
            .exec_tuple(cypher::set_contradiction_verdict(
                &hash(1000 + i),
                &hash(2000 + i),
                &EdgeVerdict {
                    verdict,
                    verdict_survivor: None,
                    verdict_reason: reason.into(),
                    verdict_confidence: 0.0,
                    verdict_model: "model-x".into(),
                    judged_at: 1.0,
                },
            ))
            .await;
        assert_eq!(r.count(), Some(1), "edge {i} must match");
    }

    let needs = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            needs_judging: true,
            ..Default::default()
        }))
        .await;
    assert_eq!(
        a_hashes(&needs),
        [hash(1000), hash(1003)],
        "judged and marked edges are excluded from the NULL selection"
    );

    let rejudge = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            needs_judging: true,
            rejudge_model: Some("model-y".into()),
            ..Default::default()
        }))
        .await;
    assert_eq!(
        a_hashes(&rejudge),
        [hash(1000), hash(1001), hash(1002), hash(1003)],
        "a model switch re-selects every edge not judged by the new model"
    );
    let same_model = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            needs_judging: true,
            rejudge_model: Some("model-x".into()),
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&same_model), [hash(1000), hash(1003)]);

    // Read surface: `unjudged` matches NULL and the marker; the marker
    // carries its reason (column 6) so the operator sees why.
    let unjudged = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            verdicts: Some(vec!["unjudged".into()]),
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&unjudged), [hash(1000), hash(1002), hash(1003)]);
    let marked_row = unjudged
        .result_set
        .iter()
        .find(|r| r[0].as_str() == Some(hash(1002).as_str()))
        .unwrap();
    assert_eq!(marked_row[4].as_str(), Some("unjudged"));
    assert_eq!(
        marked_row[6].as_str(),
        Some("unjudged: verdict is not valid JSON")
    );

    ctx.cleanup().await;
    Ok(())
}
