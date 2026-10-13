//! Integration tests — node lifecycle against a real FalkorDB instance.
//!
//! Gate: skipped when `REDIS_URL` is not set (CI has no FalkorDB).
//! Run with: `REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test integration_nodes`

mod common;

use alaya_bridge::cypher;

// ─── ensure_node ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn ensure_node_creates_new_node() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };

    let hash = "aabbcc001122334455667788990011223344556677889900112233445566778899";
    let ts = 1_710_000_000.0_f64;

    // Create the node.
    let result = ctx.exec_tuple(cypher::ensure_node(hash, ts)).await;

    // "Nodes created: 1" must appear in stats.
    let nodes_created: u64 = result
        .stats
        .get("Nodes created")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    assert_eq!(nodes_created, 1, "expected one new node to be created");

    // Verify via a count query.
    let count_result = ctx
        .exec(
            "MATCH (m:Memory {content_hash: $hash}) RETURN count(m)",
            [("hash".to_string(), serde_json::json!(hash))].into(),
            true,
        )
        .await;
    assert_eq!(
        count_result.count(),
        Some(1),
        "node must exist after ensure"
    );

    ctx.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn ensure_node_is_idempotent() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };

    let hash = "bb00112233445566778899001122334455667788990011223344556677889900aa";
    let ts = 1_710_000_001.0_f64;

    // Create twice.
    ctx.exec_tuple(cypher::ensure_node(hash, ts)).await;
    let second = ctx.exec_tuple(cypher::ensure_node(hash, ts)).await;

    // Second call must not report a new node.
    let nodes_created: u64 = second
        .stats
        .get("Nodes created")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    assert_eq!(
        nodes_created, 0,
        "second ensure_node must not create a duplicate"
    );

    // Count must still be 1.
    let count_result = ctx
        .exec(
            "MATCH (m:Memory {content_hash: $hash}) RETURN count(m)",
            [("hash".to_string(), serde_json::json!(hash))].into(),
            true,
        )
        .await;
    assert_eq!(
        count_result.count(),
        Some(1),
        "must still have exactly one node"
    );

    ctx.cleanup().await;
    Ok(())
}

// ─── delete_node ─────────────────────────────────────────────────────────────

#[tokio::test]
async fn delete_node_removes_existing_node() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };

    let hash = "cc1122334455667788990011223344556677889900112233445566778899001122";
    let ts = 1_710_000_002.0_f64;

    ctx.exec_tuple(cypher::ensure_node(hash, ts)).await;

    // Delete it.
    ctx.exec_tuple(cypher::delete_node(hash)).await;

    // Verify gone.
    let count_result = ctx
        .exec(
            "MATCH (m:Memory {content_hash: $hash}) RETURN count(m)",
            [("hash".to_string(), serde_json::json!(hash))].into(),
            true,
        )
        .await;
    assert_eq!(
        count_result.count(),
        Some(0),
        "node must be gone after delete"
    );

    ctx.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn delete_node_nonexistent_is_silent() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };

    // Delete a hash that was never created — must not error.
    let hash = "dd334455667788990011223344556677889900112233445566778899001122334455";
    ctx.exec_tuple(cypher::delete_node(hash)).await;

    // Graph should still be empty (or contain only other nodes from other tests,
    // but this ephemeral graph was freshly created so there's nothing).
    ctx.cleanup().await;
    Ok(())
}

// ─── gone-memory node sweep (LAB-9265) ────────────────────────────────────────

/// Paging in hash order; a mark that `ensure_node` clears; a delete that
/// takes only nodes still carrying its own token, with their edges.
#[tokio::test]
async fn sweep_marks_and_deletes_only_still_marked_nodes() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };
    let h = |c: char| c.to_string().repeat(64);
    let (a, b, c) = (h('a'), h('b'), h('c'));
    for x in [&a, &b, &c] {
        ctx.exec_tuple(cypher::ensure_node(x, 1.0)).await;
    }
    ctx.exec_tuple(cypher::create_typed_edge(
        &b,
        &a,
        alaya_types::graph::UserRelationType::Contradicts,
        1.0,
        None,
    ))
    .await;
    let hashes = |r: &alaya_bridge::FalkorResult| -> Vec<String> {
        r.result_set
            .iter()
            .filter_map(|row| row.first()?.as_str().map(str::to_string))
            .collect()
    };

    let first = ctx.exec_tuple(cypher::list_memory_nodes("", 2)).await;
    assert_eq!(hashes(&first), [a.clone(), b.clone()]);
    let rest = ctx.exec_tuple(cypher::list_memory_nodes(&b, 2)).await;
    assert_eq!(hashes(&rest), std::slice::from_ref(&c));

    let marked = ctx
        .exec_tuple(cypher::sweep_mark_memory_nodes(&[&b, &c], "run-1"))
        .await;
    assert_eq!(marked.count(), Some(2));
    // A store re-ensures c after the mark: c is alive again.
    ctx.exec_tuple(cypher::ensure_node(&c, 2.0)).await;
    let other = ctx
        .exec_tuple(cypher::sweep_delete_marked_nodes(&[&b, &c], "run-2"))
        .await;
    assert_eq!(
        other.count(),
        Some(0),
        "another run's token deletes nothing"
    );
    let deleted = ctx
        .exec_tuple(cypher::sweep_delete_marked_nodes(&[&b, &c], "run-1"))
        .await;
    assert_eq!(deleted.count(), Some(1), "only b still carries the mark");

    let left = ctx.exec_tuple(cypher::list_memory_nodes("", 10)).await;
    assert_eq!(hashes(&left), [a.clone(), c.clone()]);
    let edges = ctx
        .exec(
            "MATCH ()-[e:CONTRADICTS]->() RETURN count(e)",
            Default::default(),
            true,
        )
        .await;
    assert_eq!(edges.count(), Some(0), "DETACH took b's edge");

    ctx.cleanup().await;
    Ok(())
}
