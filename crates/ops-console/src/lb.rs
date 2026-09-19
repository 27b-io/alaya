//! anthropic-lb monitoring module — upstream clients (LAB-1964).
//!
//! Read-only by design: the console RENDERS LB state; budgets and limits
//! change through GitOps only. There is no write path in this module and
//! none may be added here — the LB exposes no admin write API, and a
//! mutation surface would need its own security review first.
//!
//! Two upstreams, both called server-side; credentials never reach the
//! browser:
//! - the LB's `/_stats`, authenticated with an operator client key sent as
//!   `x-api-key` (the LB's admin gate does not accept `Authorization: Bearer`);
//! - a Prometheus-compatible query API for the 7-day budget-burn history,
//!   read off the LB's own fleet-wide `anthropic_cluster_budget_used` gauge
//!   — no bespoke history store, no dashboard embeds.

use std::collections::{BTreeMap, HashMap};
use std::time::Duration;

use serde_json::Value;

use crate::error::AppError;
use crate::http;

/// Columns in the burn history: today plus the six preceding UTC days.
const DAYS: usize = 7;

/// Daily peak of the fleet-wide budget gauge, collapsed across scrape
/// sources: every LB replica publishes the same shared aggregate, so their
/// series agree and `max by (client)` de-duplicates rather than sums. The
/// 23h58m window evaluated at 23:59:00 covers (00:01, 23:59]: the LB
/// refreshes this gauge from shared state on a 5 s tick and the scraper
/// samples every ~15 s, so the first sample after midnight can still carry
/// the previous day's total — a plain `[1d]` window would credit yesterday's
/// peak to today.
const BURN_QUERY: &str = "max by (client) (max_over_time(anthropic_cluster_budget_used[23h58m]))";

fn join(base: &url::Url, path: &str) -> String {
    format!("{}{path}", base.as_str().trim_end_matches('/'))
}

/// Send and read a JSON body. Transport errors collapse to a one-phrase kind
/// (they can embed the request URL); error bodies are surfaced as plain text
/// (the page renders them as text nodes, never markup), truncated. Query
/// errors arrive as non-2xx on the Prometheus API, so this is the only error
/// path a caller needs.
async fn json_body(what: &str, req: reqwest::RequestBuilder) -> Result<Value, AppError> {
    let resp = req
        .send()
        .await
        .map_err(|e| AppError::transport(what, &e))?;
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| AppError::transport(what, &e))?;
    if !status.is_success() {
        let detail: String = text.chars().take(160).collect();
        return Err(AppError::Upstream(format!("{what} {status}: {detail}")));
    }
    serde_json::from_str(&text).map_err(|_| AppError::Upstream(format!("{what} returned non-JSON")))
}

#[derive(Clone)]
pub struct LbClient {
    base: url::Url,
    api_key: String,
    http: reqwest::Client,
}

impl LbClient {
    pub fn new(base: url::Url, api_key: String) -> Self {
        LbClient {
            base,
            api_key,
            http: http::client(Duration::from_secs(20)),
        }
    }

    /// `GET /_stats`: endpoints (upstream accounts), per-client budgets
    /// (replica-local mirror plus the fleet-wide `cluster.budget_usage`),
    /// consumers and sessions.
    pub async fn stats(&self) -> Result<Value, AppError> {
        let req = self
            .http
            .get(join(&self.base, "/_stats"))
            .header("x-api-key", &self.api_key);
        json_body("anthropic-lb", req).await
    }
}

fn f64_of(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

/// Today's per-client `(used, limit)` from a `/_stats` body. Prefers the
/// fleet-wide Redis aggregate; falls back to this replica's local mirror —
/// which resets on pod restart and sees one replica's traffic — when the
/// aggregate is absent OR empty (the LB emits `budget_usage: {}` when its
/// Redis read failed). The flag is `true` when the numbers are fleet-wide;
/// callers must say so when it is not.
pub fn live_budgets(stats: &Value) -> (BTreeMap<String, (f64, f64)>, bool) {
    let pairs = |v: Option<&Value>, used: &str, limit: &str| -> BTreeMap<String, (f64, f64)> {
        v.and_then(Value::as_object)
            .into_iter()
            .flatten()
            .map(|(k, b)| (k.clone(), (f64_of(b, used), f64_of(b, limit))))
            .collect()
    };
    let fleet = pairs(stats.pointer("/cluster/budget_usage"), "used", "limit");
    if !fleet.is_empty() {
        return (fleet, true);
    }
    let local = pairs(stats.get("client_budgets"), "used_today", "daily_limit");
    // Nothing budgeted anywhere is not a degraded state — no warning to raise.
    let fleet_true = local.is_empty();
    (local, fleet_true)
}

#[derive(Clone)]
pub struct MetricsClient {
    base: url::Url,
    http: reqwest::Client,
}

/// Daily budget burn per client, oldest day first.
pub struct DailyBurn {
    /// UTC midnight (epoch seconds) that starts each column's day.
    pub day_starts: Vec<i64>,
    /// client → one value per day; `None` = no sample in that day's window.
    pub by_client: BTreeMap<String, Vec<Option<f64>>>,
}

/// Evaluation instants for `BURN_QUERY`: 23:59:00 UTC of each of the last
/// `DAYS` days, today last. Today's instant lies in the future; the store
/// evaluates the window up to "now", which yields today's running total.
fn burn_evals(now: i64) -> Vec<i64> {
    let today = now.div_euclid(86_400) * 86_400;
    (0..DAYS as i64)
        .rev()
        .map(|k| today - k * 86_400 + 86_400 - 60)
        .collect()
}

/// Align a `query_range` result to the evaluation instants. Sample values
/// arrive as strings (Prometheus wire format); a missing instant stays
/// `None` rather than becoming a fictitious zero.
fn parse_daily_burn(body: &Value, evals: &[i64]) -> BTreeMap<String, Vec<Option<f64>>> {
    let mut out = BTreeMap::new();
    let series = body
        .pointer("/data/result")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    for s in series {
        let Some(client) = s.pointer("/metric/client").and_then(Value::as_str) else {
            continue;
        };
        let samples: HashMap<i64, f64> = s
            .get("values")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|p| {
                let ts = p.get(0)?.as_f64()?.round() as i64;
                let v = p.get(1)?.as_str()?.parse::<f64>().ok()?;
                Some((ts, v))
            })
            .collect();
        out.insert(
            client.to_string(),
            evals.iter().map(|t| samples.get(t).copied()).collect(),
        );
    }
    out
}

impl MetricsClient {
    pub fn new(base: url::Url) -> Self {
        MetricsClient {
            base,
            http: http::client(Duration::from_secs(20)),
        }
    }

    pub async fn daily_burn(&self) -> Result<DailyBurn, AppError> {
        let evals = burn_evals(crate::session::now_epoch());
        let (start, end) = (evals[0], evals[DAYS - 1]);
        let req = self
            .http
            .get(join(&self.base, "/api/v1/query_range"))
            .query(&[
                ("query", BURN_QUERY.to_string()),
                ("start", start.to_string()),
                ("end", end.to_string()),
                ("step", "86400".to_string()),
            ]);
        let body = json_body("metrics", req).await?;
        Ok(DailyBurn {
            day_starts: evals.iter().map(|t| t + 60 - 86_400).collect(),
            by_client: parse_daily_burn(&body, &evals),
        })
    }
}

/// Humanised token count for table cells; the exact value goes in `title`.
pub fn fmt_tokens(n: f64) -> String {
    if n >= 1e9 {
        format!("{:.2}B", n / 1e9)
    } else if n >= 1e6 {
        format!("{:.1}M", n / 1e6)
    } else if n >= 1e3 {
        format!("{:.0}K", n / 1e3)
    } else {
        format!("{n:.0}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Pinned against a live run (2026-09-19): now 08:09Z → instants 23:59Z
    /// of 09-13 … 09-19, and every column day is a midnight.
    #[test]
    fn burn_evals_are_seven_consecutive_2359_utc_instants() {
        let evals = burn_evals(1_789_826_976);
        assert_eq!(evals.len(), DAYS);
        assert_eq!(evals[0], 1_789_343_940);
        assert_eq!(evals[DAYS - 1], 1_789_862_340);
        for w in evals.windows(2) {
            assert_eq!(w[1] - w[0], 86_400);
        }
        for t in &evals {
            assert_eq!((t + 60) % 86_400, 0, "{t} is not 23:59:00 UTC");
        }
    }

    #[test]
    fn parse_daily_burn_aligns_samples_and_keeps_gaps_as_none() {
        let evals = [100_i64, 200, 300];
        let body = json!({
            "status": "success",
            "data": { "result": [
                { "metric": { "client": "kody" },
                  "values": [[100, "22030571"], [200, "18792677"], [300, "17913067"]] },
                // A day with no sample must not render as zero.
                { "metric": { "client": "radar" }, "values": [[100.0, "4069"], [300, "460214"]] },
                // Series without a client label are ignored (defensive).
                { "metric": {}, "values": [[100, "1"]] }
            ]}
        });
        let out = parse_daily_burn(&body, &evals);
        assert_eq!(out.len(), 2);
        assert_eq!(
            out["kody"],
            vec![Some(22_030_571.0), Some(18_792_677.0), Some(17_913_067.0)]
        );
        assert_eq!(out["radar"], vec![Some(4_069.0), None, Some(460_214.0)]);
    }

    #[test]
    fn live_budgets_prefers_fleet_and_falls_back_on_absent_or_empty_aggregate() {
        let mirror = json!({"kody": {"daily_limit": 50000000, "used_today": 1}});
        // Fleet aggregate present: it wins over the mirror.
        let (m, fleet) = live_budgets(&json!({
            "client_budgets": mirror,
            "cluster": {"budget_usage": {"kody": {"limit": 50000000, "used": 17913067}}}
        }));
        assert!(fleet);
        assert_eq!(m["kody"], (17_913_067.0, 50_000_000.0));
        // Redis read failed: the LB emits an EMPTY aggregate — fall back.
        let (m, fleet) = live_budgets(&json!({
            "client_budgets": mirror,
            "cluster": {"redis_connected": false, "budget_usage": {}}
        }));
        assert!(!fleet);
        assert_eq!(m["kody"], (1.0, 50_000_000.0));
        // No cluster info at all (Redis unconfigured / before first tick).
        let (m, fleet) = live_budgets(&json!({"client_budgets": mirror}));
        assert!(!fleet && m.len() == 1);
        // Nothing budgeted anywhere is not a degraded state.
        let (m, fleet) = live_budgets(&json!({"client_budgets": null}));
        assert!(fleet && m.is_empty());
    }

    #[test]
    fn fmt_tokens_humanises() {
        assert_eq!(fmt_tokens(0.0), "0");
        assert_eq!(fmt_tokens(999.0), "999");
        assert_eq!(fmt_tokens(460_214.0), "460K");
        assert_eq!(fmt_tokens(17_913_067.0), "17.9M");
        // The largest configured daily limit is in the billions.
        assert_eq!(fmt_tokens(15_000_000_000.0), "15.00B");
    }
}
