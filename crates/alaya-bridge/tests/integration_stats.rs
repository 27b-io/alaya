//! Integration tests — reads against a not-yet-created graph key (LAB-373).
//!
//! FalkorDB creates a graph lazily on the first write. Until then any read
//! answers with "Invalid graph operation on empty key". `exec_query` must
//! treat that as an empty graph (empty result, no error) so a fresh
//! deployment's 5s health-check polls stay quiet.
//!
//! Gate: skipped when `REDIS_URL` is not set (CI has no FalkorDB).
//! Run with: `REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test integration_stats`

mod common;

use std::collections::HashMap;

use alaya_bridge::cypher;

#[tokio::test]
async fn stats_query_on_empty_graph_returns_empty_not_error() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };

    // ctx.graph_name has never been written to, so the FalkorDB key is absent.
    // Before the fix this propagated BAD_GATEWAY and `exec` panicked; now it
    // yields an empty result set, which the /stats handler maps to zero counts.
    let result = ctx.exec_tuple(cypher::get_graph_stats_union()).await;
    assert!(
        result.result_set.is_empty(),
        "stats query on an empty graph must return no rows, got {:?}",
        result.result_set
    );

    ctx.cleanup().await;
    Ok(())
}

#[tokio::test]
async fn read_on_absent_graph_is_empty_not_error() -> anyhow::Result<()> {
    let ctx = match common::TestContext::new().await {
        Some(c) => c,
        None => return Ok(()),
    };

    // Any read handler (nodes, edges, hebbian, …) routes through exec_query,
    // so the empty-key handling covers them all — not just /stats.
    let result = ctx.exec("MATCH (n) RETURN n", HashMap::new(), true).await;
    assert!(
        result.result_set.is_empty(),
        "read on an absent graph must return no rows"
    );

    ctx.cleanup().await;
    Ok(())
}
