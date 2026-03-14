//! Integration tests — typed edge lifecycle against a real FalkorDB instance.
//!
//! Gate: skipped when `REDIS_URL` is not set (CI has no FalkorDB).
//! Run with: `REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test integration_edges`

mod common;

use alaya_bridge::cypher;
use alaya_types::graph::{Direction, SystemRelationType, UserRelationType};

// ─── test hashes (64-char hex, pass validate_content_hash) ───────────────────

const HASH_A: &str = "aaaa0000000000000000000000000000000000000000000000000000000000aaaa";
const HASH_B: &str = "bbbb0000000000000000000000000000000000000000000000000000000000bbbb";
const HASH_C: &str = "cccc0000000000000000000000000000000000000000000000000000000000cccc";

/// Seed two Memory nodes into `ctx` so edge queries have something to work with.
async fn seed_nodes(ctx: &common::TestContext) {
    let ts = 1_710_000_000.0_f64;
    ctx.exec_tuple(cypher::ensure_node(HASH_A, ts)).await;
    ctx.exec_tuple(cypher::ensure_node(HASH_B, ts)).await;
    ctx.exec_tuple(cypher::ensure_node(HASH_C, ts)).await;
}

// ─── create + get ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn create_and_get_relates_to_edge() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;
    let ts = 1_710_000_010.0_f64;

    // Create a RELATES_TO edge A→B.
    ctx.exec_tuple(cypher::create_typed_edge(
        HASH_A,
        HASH_B,
        UserRelationType::RelatesTo,
        ts,
        None,
    ))
    .await;

    // Retrieve outgoing RELATES_TO edges from A.
    let result = ctx
        .exec_tuple(cypher::get_typed_edges(
            HASH_A,
            UserRelationType::RelatesTo,
            Direction::Outgoing,
            10,
        ))
        .await;

    assert!(!result.result_set.is_empty(), "must find at least one edge");
    // Each row: [a.content_hash, b.content_hash, e.created_at]
    let dst = result.result_set[0][1].as_str().unwrap_or("");
    assert_eq!(dst, HASH_B, "destination must be HASH_B");

    ctx.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn create_contradicts_edge_stores_confidence() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;
    let ts = 1_710_000_020.0_f64;
    let confidence = 0.85;

    ctx.exec_tuple(cypher::create_typed_edge(
        HASH_A,
        HASH_B,
        UserRelationType::Contradicts,
        ts,
        Some(confidence),
    ))
    .await;

    // Query the edge directly to verify confidence was stored.
    let result = ctx
        .exec(
            "MATCH (a:Memory {content_hash: $src})-[e:CONTRADICTS]->(b:Memory {content_hash: $dst}) \
             RETURN e.confidence",
            [
                ("src".to_string(), serde_json::json!(HASH_A)),
                ("dst".to_string(), serde_json::json!(HASH_B)),
            ]
            .into(),
            true,
        )
        .await;

    assert!(!result.result_set.is_empty(), "CONTRADICTS edge must exist");
    let stored_conf = result.result_set[0][0]
        .as_f64()
        .expect("confidence must be a float");
    assert!(
        (stored_conf - confidence).abs() < 1e-6,
        "confidence mismatch: got {stored_conf}"
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── direction ────────────────────────────────────────────────────────────────

#[tokio::test]
async fn get_edges_direction_incoming_vs_outgoing() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;
    let ts = 1_710_000_030.0_f64;

    // A→B  PRECEDES
    ctx.exec_tuple(cypher::create_typed_edge(
        HASH_A,
        HASH_B,
        UserRelationType::Precedes,
        ts,
        None,
    ))
    .await;

    // Outgoing from A: should include the edge.
    let out_a = ctx
        .exec_tuple(cypher::get_typed_edges(
            HASH_A,
            UserRelationType::Precedes,
            Direction::Outgoing,
            10,
        ))
        .await;
    assert!(
        !out_a.result_set.is_empty(),
        "A must have outgoing PRECEDES edge"
    );

    // Incoming to A: should be empty (nothing points to A yet).
    let in_a = ctx
        .exec_tuple(cypher::get_typed_edges(
            HASH_A,
            UserRelationType::Precedes,
            Direction::Incoming,
            10,
        ))
        .await;
    assert!(
        in_a.result_set.is_empty(),
        "A must have no incoming PRECEDES edge"
    );

    // Incoming to B: should find the A→B edge.
    let in_b = ctx
        .exec_tuple(cypher::get_typed_edges(
            HASH_B,
            UserRelationType::Precedes,
            Direction::Incoming,
            10,
        ))
        .await;
    assert!(
        !in_b.result_set.is_empty(),
        "B must have incoming PRECEDES edge"
    );

    // Both on B: should also find it.
    let both_b = ctx
        .exec_tuple(cypher::get_typed_edges(
            HASH_B,
            UserRelationType::Precedes,
            Direction::Both,
            10,
        ))
        .await;
    assert!(
        !both_b.result_set.is_empty(),
        "Both direction on B must find the edge"
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── delete_typed_edge ────────────────────────────────────────────────────────

#[tokio::test]
async fn delete_typed_edge_removes_it() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;
    let ts = 1_710_000_040.0_f64;

    ctx.exec_tuple(cypher::create_typed_edge(
        HASH_A,
        HASH_B,
        UserRelationType::RelatesTo,
        ts,
        None,
    ))
    .await;

    // Delete it.
    ctx.exec_tuple(cypher::delete_typed_edge(
        HASH_A,
        HASH_B,
        UserRelationType::RelatesTo,
    ))
    .await;

    // Verify gone.
    let result = ctx
        .exec_tuple(cypher::get_typed_edges(
            HASH_A,
            UserRelationType::RelatesTo,
            Direction::Outgoing,
            10,
        ))
        .await;
    assert!(
        result.result_set.is_empty(),
        "edge must be gone after delete"
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── system edge (SUPERSEDES) ────────────────────────────────────────────────

#[tokio::test]
async fn create_supersedes_system_edge() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;
    let ts = 1_710_000_050.0_f64;

    ctx.exec_tuple(cypher::create_system_edge(
        HASH_A,
        HASH_B,
        SystemRelationType::Supersedes,
        ts,
    ))
    .await;

    // Verify the edge exists.
    let result = ctx
        .exec(
            "MATCH (a:Memory {content_hash: $src})-[e:SUPERSEDES]->(b:Memory {content_hash: $dst}) \
             RETURN count(e)",
            [
                ("src".to_string(), serde_json::json!(HASH_A)),
                ("dst".to_string(), serde_json::json!(HASH_B)),
            ]
            .into(),
            true,
        )
        .await;

    assert_eq!(result.count(), Some(1), "SUPERSEDES edge must exist");

    ctx.cleanup().await;
    Ok(())
}

// ─── contradictions ───────────────────────────────────────────────────────────

#[tokio::test]
async fn get_all_contradictions_returns_created_pairs() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    seed_nodes(&ctx).await;
    let ts = 1_710_000_060.0_f64;

    ctx.exec_tuple(cypher::create_typed_edge(
        HASH_A,
        HASH_B,
        UserRelationType::Contradicts,
        ts,
        Some(0.9),
    ))
    .await;
    ctx.exec_tuple(cypher::create_typed_edge(
        HASH_B,
        HASH_C,
        UserRelationType::Contradicts,
        ts,
        Some(0.7),
    ))
    .await;

    let result = ctx.exec_tuple(cypher::get_all_contradictions(50)).await;

    assert!(
        result.result_set.len() >= 2,
        "must find both CONTRADICTS pairs"
    );

    ctx.cleanup().await;
    Ok(())
}
