//! Connection and response timeouts (LAB-9869).
//!
//! The mute-peer test needs no backend and always runs. The rest need a real
//! FalkorDB and skip when `REDIS_URL` is not set.
//! Run with: `REDIS_URL=redis://localhost:6379 cargo test -p alaya-bridge --test integration_timeouts`

mod common;

use std::sync::Arc;
use std::time::{Duration, Instant};

use alaya_bridge::{
    CONNECT_TIMEOUT, QUEUE_RESPONSE_TIMEOUT, RESPONSE_TIMEOUT, connect, cypher, query_config,
    queue_config,
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

/// Send a blocking pop on `conn` and time its outcome. The server-side block
/// is finite, so a connection the test abandons is released, not leaked.
async fn timed_pop(
    conn: &mut redis::aio::ConnectionManager,
    key: &str,
    block: Duration,
) -> (redis::RedisResult<Option<(String, String)>>, Duration) {
    let started = Instant::now();
    let result = redis::cmd("BRPOP")
        .arg(key)
        .arg(block.as_secs_f64())
        .query_async(conn)
        .await;
    (result, started.elapsed())
}

/// Each connection errors at its own reply deadline when no reply comes, and
/// the consumer's own BRPOP (1 s) is not cut off by the queue deadline.
#[tokio::test]
async fn replies_time_out_at_each_deadline_but_the_consumer_pop_does_not() -> anyhow::Result<()> {
    let Some(client) = client() else {
        eprintln!("REDIS_URL not set — skipping integration test");
        return Ok(());
    };
    let key = format!("lab9869:never:{}", std::process::id());
    let past = |d: Duration| d + Duration::from_secs(2);
    let mut query = connect(&client, query_config(), "query").await?;
    let mut queue = connect(&client, queue_config(), "queue").await?;
    let mut consumer = connect(&client, queue_config(), "queue").await?;

    let ((query, query_took), (queue, queue_took), (pop, _)) =
        tokio::time::timeout(RESPONSE_TIMEOUT * 2, async {
            tokio::join!(
                timed_pop(&mut query, &key, past(RESPONSE_TIMEOUT)),
                timed_pop(&mut queue, &key, past(QUEUE_RESPONSE_TIMEOUT)),
                timed_pop(&mut consumer, &key, Duration::from_secs(1)),
            )
        })
        .await
        .expect("a reply deadline did not fire");

    for (role, result, took, deadline) in [
        ("query", query, query_took, RESPONSE_TIMEOUT),
        ("queue", queue, queue_took, QUEUE_RESPONSE_TIMEOUT),
    ] {
        let err = result.expect_err(&format!("{role}: a pop past its deadline returned"));
        assert!(err.is_timeout(), "{role}: expected a timeout, got {err}");
        assert!(
            took >= deadline && took < past(deadline),
            "{role}: timed out after {took:?}"
        );
    }
    assert_eq!(pop?, None, "the consumer's 1 s BRPOP must complete");
    Ok(())
}

/// The Hebbian consumer still drains the queue over its own connection.
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
