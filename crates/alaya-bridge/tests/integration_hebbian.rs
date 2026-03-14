//! Integration tests — Hebbian learning and consolidation against a real FalkorDB instance.
//!
//! Gate: skipped when `REDIS_URL` is not set (CI has no FalkorDB).
//! Run with: `REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test integration_hebbian`

mod common;

use alaya_bridge::cypher;

// ─── test hashes ─────────────────────────────────────────────────────────────

const H1: &str = "1111000000000000000000000000000000000000000000000000000000001111";
const H2: &str = "2222000000000000000000000000000000000000000000000000000000002222";
const H3: &str = "3333000000000000000000000000000000000000000000000000000000003333";
const H4: &str = "4444000000000000000000000000000000000000000000000000000000004444";

async fn seed_nodes(ctx: &common::TestContext) {
    let ts = 1_710_000_000.0_f64;
    for hash in [H1, H2, H3, H4] {
        ctx.exec_tuple(cypher::ensure_node(hash, ts)).await;
    }
}

// ─── get_neighbors ────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_neighbors_returns_hebbian_targets() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;

    // Manually plant a HEBBIAN edge H1→H2 with weight=0.5.
    ctx.exec(
        "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
         MERGE (a)-[e:HEBBIAN]->(b) \
         ON CREATE SET e.weight = 0.5, e.co_access_count = 1, \
                       e.created_at = 1710000000.0, e.last_co_access = 1710000000.0",
        [
            ("src".to_string(), serde_json::json!(H1)),
            ("dst".to_string(), serde_json::json!(H2)),
        ]
        .into(),
        false,
    )
    .await;

    let result = ctx
        .exec_tuple(cypher::get_neighbors(H1, 1, 0.1, 10))
        .await;

    assert!(!result.result_set.is_empty(), "must find H2 as a neighbor of H1");

    let first_hash = result.result_set[0][0].as_str().unwrap_or("");
    assert_eq!(first_hash, H2, "neighbor hash must be H2");

    let path_weight = result.result_set[0][1].as_f64().unwrap_or(0.0);
    assert!(
        (path_weight - 0.5).abs() < 1e-6,
        "path_weight must be ~0.5, got {path_weight}"
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── spreading_activation ────────────────────────────────────────────────────

#[tokio::test]
async fn spreading_activation_reaches_non_seed_nodes() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;

    // H1→H2 (w=0.8), H2→H3 (w=0.6)
    for (src, dst, w) in [(H1, H2, 0.8_f64), (H2, H3, 0.6_f64)] {
        ctx.exec(
            "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
             MERGE (a)-[e:HEBBIAN]->(b) \
             ON CREATE SET e.weight = $w, e.co_access_count = 1, \
                           e.created_at = 1710000000.0, e.last_co_access = 1710000000.0",
            [
                ("src".to_string(), serde_json::json!(src)),
                ("dst".to_string(), serde_json::json!(dst)),
                ("w".to_string(), serde_json::json!(w)),
            ]
            .into(),
            false,
        )
        .await;
    }

    // Seed from H1 — should reach H2 and H3.
    let result = ctx
        .exec_tuple(cypher::spreading_activation(&[H1], 2))
        .await;

    assert!(
        result.result_set.len() >= 2,
        "spreading activation must reach at least H2 and H3"
    );

    let reached: Vec<&str> = result.result_set
        .iter()
        .map(|r| r[0].as_str().unwrap_or(""))
        .collect();
    assert!(reached.contains(&H2), "must reach H2");
    assert!(reached.contains(&H3), "must reach H3");

    // H4 was not seeded with an edge so should not appear.
    assert!(!reached.contains(&H4), "H4 must not be reached");

    ctx.cleanup().await;
    Ok(())
}

// ─── hebbian_boosts_within ────────────────────────────────────────────────────

#[tokio::test]
async fn hebbian_boosts_within_finds_mutual_edges() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;

    // Plant H1→H2 (w=0.4) and H2→H1 (w=0.6).
    for (src, dst, w) in [(H1, H2, 0.4_f64), (H2, H1, 0.6_f64)] {
        ctx.exec(
            "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
             MERGE (a)-[e:HEBBIAN]->(b) \
             ON CREATE SET e.weight = $w, e.co_access_count = 1, \
                           e.created_at = 1710000000.0, e.last_co_access = 1710000000.0",
            [
                ("src".to_string(), serde_json::json!(src)),
                ("dst".to_string(), serde_json::json!(dst)),
                ("w".to_string(), serde_json::json!(w)),
            ]
            .into(),
            false,
        )
        .await;
    }

    let result = ctx
        .exec_tuple(cypher::hebbian_boosts_within(&[H1, H2]))
        .await;

    assert!(!result.result_set.is_empty(), "must find edges within the set");

    // Each row: [hash, max_weight].  H1 outgoing max = 0.4, H2 outgoing max = 0.6.
    let max_weights: Vec<f64> = result.result_set
        .iter()
        .map(|r| r[1].as_f64().unwrap_or(0.0))
        .collect();
    assert!(
        max_weights.iter().any(|&w| (w - 0.6).abs() < 1e-6),
        "max_weight 0.6 must appear for H2"
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── strengthen_edge ─────────────────────────────────────────────────────────

#[tokio::test]
async fn strengthen_edge_increases_weight_on_second_call() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;

    let ts = 1_710_000_000.0_f64;

    // First call — creates the edge with init_weight=0.1.
    ctx.exec_tuple(cypher::strengthen_edge(H1, H2, 0.1, 0.15, 1.0, 1.0, ts))
        .await;

    // Second call — should increase weight via LTP formula.
    ctx.exec_tuple(cypher::strengthen_edge(H1, H2, 0.1, 0.15, 1.0, 1.0, ts + 1.0))
        .await;

    let result = ctx
        .exec(
            "MATCH (a:Memory {content_hash: $src})-[e:HEBBIAN]->(b:Memory {content_hash: $dst}) \
             RETURN e.weight, e.co_access_count",
            [
                ("src".to_string(), serde_json::json!(H1)),
                ("dst".to_string(), serde_json::json!(H2)),
            ]
            .into(),
            true,
        )
        .await;

    assert!(!result.result_set.is_empty(), "HEBBIAN edge must exist");

    let weight = result.result_set[0][0].as_f64().expect("weight must be float");
    assert!(
        weight > 0.1,
        "weight must have increased above initial 0.1, got {weight}"
    );

    let co_count = result.result_set[0][1].as_i64().expect("co_access_count must be int");
    assert_eq!(co_count, 2, "co_access_count must be 2 after two strengthen calls");

    ctx.cleanup().await;
    Ok(())
}

// ─── decay_all_edges ─────────────────────────────────────────────────────────

#[tokio::test]
async fn decay_all_edges_reduces_weight() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;

    let initial_weight = 0.8_f64;
    let ts = 1_710_000_000.0_f64;

    // Plant a HEBBIAN edge with known weight.
    ctx.exec(
        "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
         MERGE (a)-[e:HEBBIAN]->(b) \
         ON CREATE SET e.weight = $w, e.co_access_count = 1, \
                       e.created_at = $ts, e.last_co_access = $ts",
        [
            ("src".to_string(), serde_json::json!(H1)),
            ("dst".to_string(), serde_json::json!(H2)),
            ("w".to_string(), serde_json::json!(initial_weight)),
            ("ts".to_string(), serde_json::json!(ts)),
        ]
        .into(),
        false,
    )
    .await;

    // Apply 50% decay.
    let decay = 0.5_f64;
    ctx.exec_tuple(cypher::decay_all_edges(decay, 1000)).await;

    let result = ctx
        .exec(
            "MATCH (a:Memory {content_hash: $src})-[e:HEBBIAN]->(b:Memory {content_hash: $dst}) \
             RETURN e.weight",
            [
                ("src".to_string(), serde_json::json!(H1)),
                ("dst".to_string(), serde_json::json!(H2)),
            ]
            .into(),
            true,
        )
        .await;

    assert!(!result.result_set.is_empty(), "HEBBIAN edge must still exist after decay");

    let weight = result.result_set[0][0].as_f64().expect("weight must be float");
    assert!(
        (weight - initial_weight * decay).abs() < 1e-5,
        "weight must be ~{}, got {weight}",
        initial_weight * decay
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── prune_weak_edges ─────────────────────────────────────────────────────────

#[tokio::test]
async fn prune_weak_edges_removes_below_threshold() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;

    let ts = 1_710_000_000.0_f64;

    // H1→H2 with weight=0.02 (weak), H3→H4 with weight=0.9 (strong).
    for (src, dst, w) in [(H1, H2, 0.02_f64), (H3, H4, 0.9_f64)] {
        ctx.exec(
            "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
             MERGE (a)-[e:HEBBIAN]->(b) \
             ON CREATE SET e.weight = $w, e.co_access_count = 1, \
                           e.created_at = $ts, e.last_co_access = $ts",
            [
                ("src".to_string(), serde_json::json!(src)),
                ("dst".to_string(), serde_json::json!(dst)),
                ("w".to_string(), serde_json::json!(w)),
                ("ts".to_string(), serde_json::json!(ts)),
            ]
            .into(),
            false,
        )
        .await;
    }

    // Prune edges with weight < 0.05.
    ctx.exec_tuple(cypher::prune_weak_edges(0.05, 1000)).await;

    // Weak edge must be gone.
    let weak = ctx
        .exec(
            "MATCH (a:Memory {content_hash: $src})-[e:HEBBIAN]->(b:Memory {content_hash: $dst}) \
             RETURN count(e)",
            [
                ("src".to_string(), serde_json::json!(H1)),
                ("dst".to_string(), serde_json::json!(H2)),
            ]
            .into(),
            true,
        )
        .await;
    assert_eq!(weak.count(), Some(0), "weak edge (w=0.02) must have been pruned");

    // Strong edge must survive.
    let strong = ctx
        .exec(
            "MATCH (a:Memory {content_hash: $src})-[e:HEBBIAN]->(b:Memory {content_hash: $dst}) \
             RETURN count(e)",
            [
                ("src".to_string(), serde_json::json!(H3)),
                ("dst".to_string(), serde_json::json!(H4)),
            ]
            .into(),
            true,
        )
        .await;
    assert_eq!(strong.count(), Some(1), "strong edge (w=0.9) must survive pruning");

    ctx.cleanup().await;
    Ok(())
}
