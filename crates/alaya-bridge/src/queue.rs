//! Background consumer for the Hebbian write queue.
//!
//! The `/hebbian/strengthen` handler LPUSH'es serialized `CoAccessPair` items
//! to `alaya:hebbian:queue`.  This module BRPOP's them and executes the
//! corresponding `strengthen_edge` Cypher queries against FalkorDB.

use std::sync::Arc;

use alaya_types::graph::CoAccessPair;

use crate::{AppState, cypher, handlers};

const QUEUE_KEY: &str = "alaya:hebbian:queue";
const INITIAL_WEIGHT: f64 = 0.1;
const STRENGTHEN_RATE: f64 = 0.15;
const MAX_WEIGHT: f64 = 1.0;
const BRPOP_TIMEOUT_SECS: f64 = 1.0;
const MAX_OPS_PER_SEC: u32 = 100;

/// Spawn the Hebbian write-queue consumer as a background tokio task.
pub fn spawn_consumer(state: Arc<AppState>) {
    tokio::spawn(async move {
        consumer_loop(&state).await;
    });
    tracing::info!("Hebbian write-queue consumer started (queue={QUEUE_KEY})");
}

async fn consumer_loop(state: &AppState) {
    let mut conn = state.redis.clone();
    let mut ops_this_second: u32 = 0;
    let mut second_start = tokio::time::Instant::now();

    loop {
        // Rate limiting: cap throughput to MAX_OPS_PER_SEC.
        if ops_this_second >= MAX_OPS_PER_SEC {
            let elapsed = second_start.elapsed();
            if elapsed < tokio::time::Duration::from_secs(1) {
                tokio::time::sleep(tokio::time::Duration::from_secs(1) - elapsed).await;
            }
            ops_this_second = 0;
            second_start = tokio::time::Instant::now();
        }

        // Reset counter each second.
        if second_start.elapsed() >= tokio::time::Duration::from_secs(1) {
            ops_this_second = 0;
            second_start = tokio::time::Instant::now();
        }

        // BRPOP with timeout — blocks until an item arrives or timeout expires.
        let result: Result<Option<(String, String)>, redis::RedisError> = redis::cmd("BRPOP")
            .arg(QUEUE_KEY)
            .arg(BRPOP_TIMEOUT_SECS)
            .query_async(&mut conn)
            .await;

        match result {
            Ok(Some((_key, raw))) => match serde_json::from_str::<CoAccessPair>(&raw) {
                Ok(pair) => {
                    let spacing_mod = 0.5 + 0.5 * pair.spacing_quality.clamp(0.0, 1.0);
                    let (cypher, params, readonly) = cypher::strengthen_edge(
                        &pair.src,
                        &pair.dst,
                        INITIAL_WEIGHT,
                        STRENGTHEN_RATE,
                        MAX_WEIGHT,
                        spacing_mod,
                        pair.timestamp,
                    );
                    if let Err(e) = handlers::exec_query(state, &cypher, params, readonly).await {
                        tracing::error!("Hebbian strengthen failed: {e:?}");
                    }
                    ops_this_second += 1;
                }
                Err(e) => {
                    tracing::error!("Failed to deserialize queue item: {e}");
                }
            },
            Ok(None) => {
                // Timeout, no items — loop back.
            }
            Err(e) => {
                tracing::error!("BRPOP error: {e}");
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }
    }
}
