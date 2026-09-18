//! Integration tests for the contradiction-pair selection and resolution
//! Cypher against a real FalkorDB (LAB-3283 AC-5/AC-6 as amended after
//! review; LAB-3885 AC-7 keep_both). Skipped when
//! `REDIS_URL` is unset. Run locally with a throwaway FalkorDB, e.g.
//! `podman run -d -p 16380:6379 falkordb/falkordb:v4.18.6` then
//! `REDIS_URL=redis://localhost:16380 cargo test -p alaya-bridge --test integration_contradictions`.

mod common;

use alaya_bridge::cypher;
use alaya_types::graph::{
    ContradictionQuery, EdgeVerdict, Resolution, SystemRelationType, UserRelationType, Verdict,
};
use serde_json::{Value, json};

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
            rejudge_model: Some("model-y".into()),
            ..Default::default()
        }))
        .await;
    assert_eq!(
        a_hashes(&rejudge),
        [hash(1000), hash(1001), hash(1002), hash(1003)],
        "a model switch re-selects every edge not judged by the new model"
    );
    // Same model: real verdicts stay, but the marker is retried on opt-in.
    let same_model = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            rejudge_model: Some("model-x".into()),
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&same_model), [hash(1000), hash(1002), hash(1003)]);

    // A marker never clobbers a real verdict (store-path spawn racing a
    // backfill, or a failed re-judge): pair 1 keeps `coexist`.
    let r = ctx
        .exec_tuple(cypher::set_contradiction_verdict(
            &hash(1001),
            &hash(2001),
            &EdgeVerdict {
                verdict: Verdict::Unjudged,
                verdict_survivor: None,
                verdict_reason: "unjudged: late failure".into(),
                verdict_confidence: 0.0,
                verdict_model: "model-y".into(),
                judged_at: 2.0,
            },
        ))
        .await;
    assert_eq!(r.count(), Some(0), "marker must not match a judged edge");
    // …but a real verdict overwrites a marker (pair 2), and a marker may
    // replace an older marker (pair 3 gets one).
    for (i, verdict, reason, conf) in [
        (2usize, Verdict::Unrelated, "shared vocabulary", 0.8),
        (
            3,
            Verdict::Unjudged,
            "unjudged: verdict is not valid JSON",
            0.0,
        ),
    ] {
        let r = ctx
            .exec_tuple(cypher::set_contradiction_verdict(
                &hash(1000 + i),
                &hash(2000 + i),
                &EdgeVerdict {
                    verdict,
                    verdict_survivor: None,
                    verdict_reason: reason.into(),
                    verdict_confidence: conf,
                    verdict_model: "model-y".into(),
                    judged_at: 2.0,
                },
            ))
            .await;
        assert_eq!(r.count(), Some(1), "edge {i}");
    }
    let judged = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            verdicts: Some(vec!["coexist".into(), "unrelated".into()]),
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&judged), [hash(1001), hash(1002)]);

    // Read surface: `unjudged` matches NULL and a marker; a marker carries
    // its reason (column 6) so the operator sees why.
    let unjudged = ctx
        .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
            limit: 10,
            verdicts: Some(vec!["unjudged".into()]),
            ..Default::default()
        }))
        .await;
    assert_eq!(a_hashes(&unjudged), [hash(1000), hash(1003)]);
    let marked_row = unjudged
        .result_set
        .iter()
        .find(|r| r[0].as_str() == Some(hash(1003).as_str()))
        .unwrap();
    assert_eq!(marked_row[4].as_str(), Some("unjudged"));
    assert_eq!(
        marked_row[6].as_str(),
        Some("unjudged: verdict is not valid JSON")
    );

    ctx.cleanup().await;
    Ok(())
}

/// Panel CRIT: every edge written by one `store` shares `created_at`, and
/// FalkorDB orders ties differently per SKIP/LIMIT window. With the pair
/// tiebreak, paging over a tie group returns every edge exactly once.
#[tokio::test]
async fn paging_over_created_at_ties_is_stable_and_complete() -> anyhow::Result<()> {
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    // 30 pairs in three tie groups of 10.
    let ts0 = 1_800_000_000.0_f64;
    for i in 0..30usize {
        let (a, b) = (hash(1000 + i), hash(2000 + i));
        let ts = ts0 - (i / 10) as f64;
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
    }
    for page_size in [4usize, 7] {
        let mut seen = std::collections::BTreeSet::new();
        let mut skip = 0;
        loop {
            let page = ctx
                .exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
                    limit: page_size,
                    skip,
                    ..Default::default()
                }))
                .await;
            let rows = a_hashes(&page);
            for h in &rows {
                assert!(
                    seen.insert(h.clone()),
                    "duplicate {h} at skip {skip} (page {page_size})"
                );
            }
            if rows.len() < page_size {
                break;
            }
            skip += page_size;
        }
        assert_eq!(
            seen.len(),
            30,
            "page size {page_size} must reach every edge once"
        );
    }

    ctx.cleanup().await;
    Ok(())
}

/// The default queue: `exclude_resolved`, one page of 10.
async fn default_queue(ctx: &common::TestContext) -> alaya_bridge::FalkorResult {
    ctx.exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
        limit: 10,
        exclude_resolved: true,
        ..Default::default()
    }))
    .await
}

/// Every pair (`include_resolved`), one page of 10.
async fn everything(ctx: &common::TestContext) -> alaya_bridge::FalkorResult {
    ctx.exec_tuple(cypher::get_all_contradictions(&ContradictionQuery {
        limit: 10,
        ..Default::default()
    }))
    .await
}

/// Cells 10..=12 (`e.resolution, e.resolved_at, e.resolved_via`) of the row
/// whose `a.content_hash` is `a`.
fn resolution_cells(result: &alaya_bridge::FalkorResult, a: &str) -> (Value, Value, Value) {
    let row = result
        .result_set
        .iter()
        .find(|r| r[0].as_str() == Some(a))
        .unwrap_or_else(|| panic!("no row for {a}"));
    (row[10].clone(), row[11].clone(), row[12].clone())
}

/// LAB-3885 AC-7: `keep_both` stamps the edge, the default queue drops the
/// pair, `include_resolved` still lists it with its stamp, and clearing
/// puts it back. The stamp survives what the store path does to the edge —
/// a re-store of either endpoint re-runs the detector, which `MERGE`s the
/// same CONTRADICTS edge (`ON CREATE SET` only) — and a later verdict write
/// on the same edge. The spec could not prove the MERGE claim on paper;
/// this is the proof.
#[tokio::test]
async fn keep_both_leaves_the_default_queue_survives_merge_and_is_reversible() -> anyhow::Result<()>
{
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    seed_pairs(&ctx, 3, 0).await;
    let (a, b) = (hash(1001), hash(2001));

    // Unresolved: the pair is in the default queue, cells are null.
    assert_eq!(
        a_hashes(&default_queue(&ctx).await),
        [hash(1000), hash(1001), hash(1002)]
    );
    assert_eq!(
        resolution_cells(&everything(&ctx).await, &a),
        (Value::Null, Value::Null, Value::Null)
    );

    // Set: MATCH-only, one edge.
    let r = ctx
        .exec_tuple(cypher::set_contradiction_resolution(
            &a,
            &b,
            Some(Resolution::KeepBoth),
            "operator:console",
            42.0,
        ))
        .await;
    assert_eq!(r.count(), Some(1));
    assert_eq!(
        a_hashes(&default_queue(&ctx).await),
        [hash(1000), hash(1002)],
        "a kept pair leaves the default queue"
    );
    let all = everything(&ctx).await;
    assert_eq!(
        a_hashes(&all),
        [hash(1000), hash(1001), hash(1002)],
        "include_resolved still lists it"
    );
    assert_eq!(
        resolution_cells(&all, &a),
        (json!("keep_both"), json!(42.0), json!("operator:console"))
    );
    assert_eq!(
        resolution_cells(&all, &hash(1000)),
        (Value::Null, Value::Null, Value::Null),
        "the stamp is per edge"
    );

    // No-create: stamping a pair with no edge matches nothing and makes nothing.
    let r = ctx
        .exec_tuple(cypher::set_contradiction_resolution(
            &hash(1009),
            &hash(2009),
            Some(Resolution::KeepBoth),
            "operator:console",
            42.0,
        ))
        .await;
    assert_eq!(r.count(), Some(0));
    assert_eq!(everything(&ctx).await.result_set.len(), 3);

    // MERGE preservation: the store path re-detects the pair on a re-store
    // and MERGEs the edge with ON CREATE SET. The stamp — and the original
    // created_at / confidence — must survive.
    let r = ctx
        .exec_tuple(cypher::create_typed_edge(
            &a,
            &b,
            UserRelationType::Contradicts,
            99.0,
            Some(0.8),
        ))
        .await;
    assert_eq!(r.count(), Some(1), "MERGE matched the existing edge");
    let all = everything(&ctx).await;
    assert_eq!(all.result_set.len(), 3, "MERGE created no second edge");
    assert_eq!(
        resolution_cells(&all, &a),
        (json!("keep_both"), json!(42.0), json!("operator:console")),
        "re-store MERGE preserves e.resolution*"
    );
    let row = all
        .result_set
        .iter()
        .find(|r| r[0].as_str() == Some(a.as_str()))
        .unwrap();
    assert_eq!(
        row[2],
        json!(0.7),
        "ON CREATE SET did not fire: confidence kept"
    );
    assert_eq!(
        row[3],
        json!(1_800_000_000.0 - 1.0),
        "ON CREATE SET did not fire: created_at kept"
    );

    // A verdict written after the stamp touches its own namespace only.
    let r = ctx
        .exec_tuple(cypher::set_contradiction_verdict(
            &a,
            &b,
            &EdgeVerdict {
                verdict: Verdict::Coexist,
                verdict_survivor: None,
                verdict_reason: "both true".into(),
                verdict_confidence: 0.9,
                verdict_model: "model-x".into(),
                judged_at: 50.0,
            },
        ))
        .await;
    assert_eq!(r.count(), Some(1));
    let all = everything(&ctx).await;
    assert_eq!(
        resolution_cells(&all, &a),
        (json!("keep_both"), json!(42.0), json!("operator:console"))
    );
    let row = all
        .result_set
        .iter()
        .find(|r| r[0].as_str() == Some(a.as_str()))
        .unwrap();
    assert_eq!(row[4], json!("coexist"));
    assert_eq!(
        a_hashes(&default_queue(&ctx).await),
        [hash(1000), hash(1002)],
        "still resolved after the verdict"
    );

    // AC-6: the delete guard sees a verdict or a resolution; the bare pair
    // (1000) is free to delete.
    let locked = ctx
        .exec_tuple(cypher::count_locked_contradiction(&a, &b))
        .await;
    assert_eq!(locked.count(), Some(1));
    let free = ctx
        .exec_tuple(cypher::count_locked_contradiction(&hash(1000), &hash(2000)))
        .await;
    assert_eq!(free.count(), Some(0));

    // Clear: all three cells null, the pair is back in the default queue,
    // the verdict is untouched — and the guard still holds on the verdict.
    let r = ctx
        .exec_tuple(cypher::set_contradiction_resolution(&a, &b, None, "", 0.0))
        .await;
    assert_eq!(r.count(), Some(1));
    assert_eq!(
        a_hashes(&default_queue(&ctx).await),
        [hash(1000), hash(1001), hash(1002)],
        "clearing is the reverse path"
    );
    let all = everything(&ctx).await;
    assert_eq!(
        resolution_cells(&all, &a),
        (Value::Null, Value::Null, Value::Null)
    );
    let row = all
        .result_set
        .iter()
        .find(|r| r[0].as_str() == Some(a.as_str()))
        .unwrap();
    assert_eq!(row[4], json!("coexist"), "clear does not touch the verdict");
    let locked = ctx
        .exec_tuple(cypher::count_locked_contradiction(&a, &b))
        .await;
    assert_eq!(locked.count(), Some(1), "a judged edge stays guarded");

    ctx.cleanup().await;
    Ok(())
}
