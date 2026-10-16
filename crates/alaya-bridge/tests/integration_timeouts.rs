//! Connection and response timeouts (LAB-9869).
//!
//! The mute-peer test needs no backend and always runs. The rest need a real
//! FalkorDB and skip when `REDIS_URL` is not set.
//! Run with: `REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test integration_timeouts`

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use alaya_bridge::{
    CONNECT_TIMEOUT, RESPONSE_TIMEOUT, connect, cypher, query_config, queue_config,
};
use alaya_types::graph::CoAccessPair;

const H1: &str = "1111000000000000000000000000000000000000000000000000000000009869";
const H2: &str = "2222000000000000000000000000000000000000000000000000000000009869";

/// Attempts are 1 + `CONNECT_RETRIES` (2), with at most 2 s of backoff after
/// each failed one; 2 s of slack on top.
const CONNECT_BOUND: Duration = Duration::from_secs(3 * 5 + 2 * 2 + 2);

fn client() -> Option<redis::Client> {
    let url = std::env::var("REDIS_URL").ok()?;
    Some(redis::Client::open(url.as_str()).expect("Invalid REDIS_URL"))
}

/// The 2026-10-16 outage shape: the peer accepts TCP and never answers.
/// Both connections must fail with a timeout inside the bound, not hang.
#[tokio::test]
async fn connect_to_a_mute_peer_times_out() -> anyhow::Result<()> {
    assert_eq!(
        CONNECT_TIMEOUT,
        Duration::from_secs(5),
        "update CONNECT_BOUND"
    );

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let port = listener.local_addr()?.port();
    tokio::spawn(async move {
        // Hold every accepted socket open and never write to it.
        let mut held = Vec::new();
        while let Ok((sock, _)) = listener.accept().await {
            held.push(sock);
        }
    });

    let client = redis::Client::open(format!("redis://127.0.0.1:{port}"))?;
    let started = Instant::now();
    let (query, queue) = tokio::time::timeout(CONNECT_BOUND, async {
        tokio::join!(
            connect(&client, query_config(), "query"),
            connect(&client, queue_config(), "queue")
        )
    })
    .await
    .expect("connect hung past its bound");

    for (role, result) in [("query", query), ("queue", queue)] {
        let err = result
            .err()
            .unwrap_or_else(|| panic!("{role}: connected to a mute peer"));
        assert!(err.is_timeout(), "{role}: expected a timeout, got {err}");
    }
    // At least one full attempt ran: the timeout fired, nothing short-circuited.
    assert!(started.elapsed() >= CONNECT_TIMEOUT);
    Ok(())
}

/// A query that never gets a reply errors after `RESPONSE_TIMEOUT`, while the
/// queue connection rides out a blocking pop longer than that deadline.
#[tokio::test]
async fn query_reply_times_out_but_queue_blocking_pop_does_not() -> anyhow::Result<()> {
    let Some(client) = client() else {
        eprintln!("REDIS_URL not set — skipping integration test");
        return Ok(());
    };
    let mut query = connect(&client, query_config(), "query").await?;
    let mut queue = connect(&client, queue_config(), "queue").await?;
    let key = format!("lab9869:never:{}", std::process::id());
    let pop_secs = RESPONSE_TIMEOUT.as_secs_f64() + 1.0;

    let started = Instant::now();
    // BLPOP with timeout 0 blocks forever: the query connection's reply never comes.
    let stuck = async {
        redis::cmd("BLPOP")
            .arg(&key)
            .arg(0)
            .query_async::<Option<(String, String)>>(&mut query)
            .await
    };
    let pop = async {
        redis::cmd("BRPOP")
            .arg(&key)
            .arg(pop_secs)
            .query_async::<Option<(String, String)>>(&mut queue)
            .await
    };
    let (stuck, pop) =
        tokio::time::timeout(RESPONSE_TIMEOUT * 3, async { tokio::join!(stuck, pop) })
            .await
            .expect("a reply deadline did not fire");

    let err = stuck.expect_err("query connection waited forever for a reply");
    assert!(err.is_timeout(), "expected a timeout, got {err}");
    assert_eq!(pop?, None, "queue connection must outlast RESPONSE_TIMEOUT");
    assert!(started.elapsed() >= Duration::from_secs_f64(pop_secs));
    Ok(())
}

/// The Hebbian consumer still drains the queue over its timeout-exempt connection.
#[tokio::test]
async fn hebbian_consumer_drains_the_queue() -> anyhow::Result<()> {
    let Some(ctx) = common::TestContext::new().await else {
        return Ok(());
    };
    let client = client().expect("REDIS_URL is set");
    for hash in [H1, H2] {
        ctx.exec_tuple(cypher::ensure_node(hash, 1_710_000_000.0))
            .await;
    }

    let state = Arc::new(common::build_state(&ctx.conn, &ctx.graph_name));
    alaya_bridge::queue::spawn_consumer(state, connect(&client, queue_config(), "queue").await?);

    let pair = CoAccessPair {
        src: H1.into(),
        dst: H2.into(),
        spacing_quality: 1.0,
        timestamp: 1_710_000_000.0,
    };
    let mut conn = ctx.conn.clone();
    redis::cmd("LPUSH")
        .arg("alaya:hebbian:queue")
        .arg(serde_json::to_string(&pair)?)
        .query_async::<()>(&mut conn)
        .await?;

    let edge = "MATCH (:Memory {content_hash: $src})-[e:HEBBIAN]->(:Memory {content_hash: $dst}) RETURN e.weight";
    let params = [
        ("src".to_string(), serde_json::json!(H1)),
        ("dst".to_string(), serde_json::json!(H2)),
    ];
    let deadline = Instant::now() + Duration::from_secs(10);
    let found = loop {
        if !ctx
            .exec(edge, params.clone().into(), true)
            .await
            .result_set
            .is_empty()
        {
            break true;
        }
        if Instant::now() > deadline {
            break false;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    ctx.cleanup().await;
    assert!(found, "consumer did not write the HEBBIAN edge within 10 s");
    Ok(())
}
