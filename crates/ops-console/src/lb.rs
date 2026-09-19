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

use crate::error::{AppError, reqwest_kind};

/// Columns in the burn history: today plus the six preceding UTC days.
pub const DAYS: usize = 7;

/// Daily peak of the fleet-wide budget gauge, collapsed across scrape
/// sources: every LB replica publishes the same shared aggregate, so their
/// series agree and `max by (client)` de-duplicates rather than sums. The
/// 23h58m window evaluated at 23:59:00 covers (00:01, 23:59]: the LB
/// refreshes this gauge from shared state on a 5 s tick and the scraper
/// samples every ~15 s, so the first sample after midnight can still carry
/// the previous day's total — a plain `[1d]` window would credit yesterday's
/// peak to today.
const BURN_QUERY: &str = "max by (client) (max_over_time(anthropic_cluster_budget_used[23h58m]))";

fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(20))
        // Never follow a redirect with the operator key attached: a 3xx must
        // not be able to carry the credential off-host.
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .expect("failed to build anthropic-lb http client")
}

fn join(base: &url::Url, path: &str) -> String {
    format!("{}{path}", base.as_str().trim_end_matches('/'))
}

/// JSON body of a successful response. Error bodies are surfaced as plain
/// text (the page renders them as text nodes, never markup), truncated.
async fn json_body(what: &str, resp: reqwest::Response) -> Result<Value, AppError> {
    let status = resp.status();
    let text = resp
        .text()
        .await
        .map_err(|e| AppError::Upstream(format!("{what}: {}", reqwest_kind(&e))))?;
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
            http: http_client(),
        }
    }

    /// `GET /_stats`: endpoints (upstream accounts), per-client budgets
    /// (replica-local mirror plus the fleet-true `cluster.budget_usage`),
    /// consumers and sessions.
    pub async fn stats(&self) -> Result<Value, AppError> {
        let resp = self
            .http
            .get(join(&self.base, "/_stats"))
            .header("x-api-key", &self.api_key)
            .send()
            .await
            .map_err(|e| AppError::Upstream(format!("anthropic-lb: {}", reqwest_kind(&e))))?;
        json_body("anthropic-lb", resp).await
    }
}

#[derive(Clone)]
pub struct MetricsClient {
    base: url::Url,
    http: reqwest::Client,
}

/// Daily budget burn per client, oldest day first.
pub struct DailyBurn {
    /// UTC midnight (epoch seconds) of each column's day.
    pub days: Vec<i64>,
    /// client → one value per day; `None` = no sample in that day's window.
    pub by_client: BTreeMap<String, Vec<Option<f64>>>,
}

/// Evaluation instants for `BURN_QUERY`: 23:59:00 UTC of each of the last
/// `DAYS` days, today last. Today's instant lies in the future; the store
/// evaluates the window up to "now", which yields today's running total.
pub fn burn_evals(now: i64) -> Vec<i64> {
    let today = now.div_euclid(86_400) * 86_400;
    (0..DAYS as i64)
        .rev()
        .map(|k| today - k * 86_400 + 86_400 - 60)
        .collect()
}

/// Align a `query_range` result to the evaluation instants. Sample values
/// arrive as strings (Prometheus wire format); a missing instant stays
/// `None` rather than becoming a fictitious zero.
pub fn parse_daily_burn(body: &Value, evals: &[i64]) -> BTreeMap<String, Vec<Option<f64>>> {
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
            http: http_client(),
        }
    }

    pub async fn daily_burn(&self, now: i64) -> Result<DailyBurn, AppError> {
        let evals = burn_evals(now);
        let (start, end) = (evals[0], evals[DAYS - 1]);
        let resp = self
            .http
            .get(join(&self.base, "/api/v1/query_range"))
            .query(&[
                ("query", BURN_QUERY.to_string()),
                ("start", start.to_string()),
                ("end", end.to_string()),
                ("step", "86400".to_string()),
            ])
            .send()
            .await
            .map_err(|e| AppError::Upstream(format!("metrics: {}", reqwest_kind(&e))))?;
        let body = json_body("metrics", resp).await?;
        if body.get("status").and_then(Value::as_str) != Some("success") {
            let err = body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("query failed");
            return Err(AppError::Upstream(format!("metrics: {err}")));
        }
        Ok(DailyBurn {
            days: evals.iter().map(|t| t + 60 - 86_400).collect(),
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

    /// Pinned against a live VictoriaMetrics run (2026-09-19): now 08:09Z →
    /// instants 23:59Z of 09-13 … 09-19, and every column day is a midnight.
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
    fn fmt_tokens_humanises() {
        assert_eq!(fmt_tokens(0.0), "0");
        assert_eq!(fmt_tokens(999.0), "999");
        assert_eq!(fmt_tokens(460_214.0), "460K");
        assert_eq!(fmt_tokens(17_913_067.0), "17.9M");
        assert_eq!(fmt_tokens(15_000_000_000.0), "15.00B");
    }
}
