//! anthropic-lb module — read-only monitoring pane (LAB-1964).
//!
//! GET only. Renders per-client budget burn vs budget (live from `/_stats`,
//! 7-day history from the metrics store) and per-account utilisation /
//! headroom. Every card names its source; limits render as "TOML, GitOps".
//! There is no edit flow here by design and the LB exposes no admin write
//! route — the console has no write route to the LB and this module must
//! not grow one.
//!
//! Only fleet-wide numbers are rendered. `/_stats` also carries
//! process-local counters (per-consumer request rates, per-endpoint burn
//! rates and token totals); through a Service with several replicas those
//! describe one random pod, so they are deliberately left out.

use std::collections::{BTreeMap, BTreeSet};

use axum::extract::State;
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde_json::Value;

use crate::error::AppError;
use crate::lb::{DailyBurn, fmt_tokens, live_budgets};
use crate::routes::{fmt_epoch, vf, vs};
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

const TITLE: &str = "anthropic-lb — ops console";

pub async fn pane(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);

    let Some(lb) = state.lb.as_ref() else {
        let content = view! {
            <Card>
                <CardHeader>
                    <CardTitle>"anthropic-lb — module not configured"</CardTitle>
                    <CardDescription>
                        "Set LB_URL, LB_API_KEY and METRICS_URL on the console deployment (all three, or none) to enable read-only LB monitoring."
                    </CardDescription>
                </CardHeader>
            </Card>
        };
        return Ok((jar, Html(page(TITLE, &session, flash, content))));
    };

    // Sections are independent: a dark metrics store must not hide live
    // headroom, and an LB outage must not hide the burn history.
    let (stats, burn) = tokio::join!(lb.client.stats(), lb.metrics.daily_burn());

    let content = view! {
        <div class="space-y-6">
            {fleet_card(&stats)}
            {budgets_card(&stats, &burn)}
            {accounts_card(&stats)}
        </div>
    };
    Ok((jar, Html(page(TITLE, &session, flash, content))))
}

// ─── Rendering helpers ──────────────────────────────────────────────────────

/// Optional integer for a cell, or a dash — never a fictitious zero.
fn int_or_dash(v: Option<&Value>) -> String {
    v.and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}

fn ratio_pct(r: Option<f64>) -> String {
    r.map(|x| format!("{:.0}%", x * 100.0))
        .unwrap_or_else(|| "—".into())
}

/// `MM-DD` column header for a UTC-midnight epoch.
fn fmt_day(midnight: i64) -> String {
    fmt_epoch(midnight as f64)
        .get(5..10)
        .unwrap_or("—")
        .to_string()
}

fn unavailable(what: &str, e: &AppError) -> impl IntoView + use<> {
    let msg = format!("{what}: {}", e.detail());
    view! {
        <p class="text-sm mb-4">
            <span class=badge(BadgeKind::Destructive)>"unavailable"</span>
            " "{msg}
        </p>
    }
}

// ─── Fleet: strategy, replicas, shared state ────────────────────────────────

fn fleet_card(stats: &Result<Value, AppError>) -> impl IntoView + use<> {
    let body = match stats {
        Err(e) => Either::Left(unavailable("anthropic-lb /_stats", e)),
        Ok(s) => {
            let strategy = vs(s, "strategy");
            let cluster = s.get("cluster");
            let replicas = int_or_dash(cluster.and_then(|c| c.get("replicas_seen")));
            let (redis_class, redis_text) = match cluster
                .and_then(|c| c.get("redis_connected"))
                .and_then(Value::as_bool)
            {
                Some(true) => (badge(BadgeKind::Success), "connected"),
                Some(false) => (badge(BadgeKind::Destructive), "disconnected"),
                None => (badge(BadgeKind::Muted), "no cluster info"),
            };
            let headroom = int_or_dash(s.pointer("/aggregate/total_headroom_requests"));
            // serde_json maps iterate in key order — no sort needed.
            let transport = cluster
                .and_then(|c| c.get("transport_errors"))
                .and_then(Value::as_object)
                .map(|m| {
                    m.iter()
                        .map(|(k, v)| format!("{k} {}", v.as_u64().unwrap_or(0)))
                        .collect::<Vec<_>>()
                        .join(" · ")
                })
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "—".into());

            Either::Right(view! {
                <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                    <div>
                        <dt class="text-muted-foreground text-xs">"Strategy"</dt>
                        <dd>{strategy}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Replicas seen"</dt>
                        <dd>{replicas}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Shared state (Redis)"</dt>
                        <dd><span class=redis_class>{redis_text}</span></dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Pooled headroom (requests)"</dt>
                        <dd>{headroom}</dd>
                    </div>
                    <div class="col-span-2 sm:col-span-4">
                        <dt class="text-muted-foreground text-xs">"Upstream transport errors (fleet, cumulative)"</dt>
                        <dd>{transport}</dd>
                    </div>
                </dl>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Fleet"</CardTitle>
                <CardDescription>
                    "Live from anthropic-lb /_stats, fetched server-side with the operator credential; fleet-wide values only. Read-only: routing strategy and limits are TOML, GitOps."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Per-client budget burn ─────────────────────────────────────────────────

fn budgets_card(
    stats: &Result<Value, AppError>,
    burn: &Result<DailyBurn, AppError>,
) -> impl IntoView + use<> {
    let (live, fleet_true) = match stats {
        Ok(s) => live_budgets(s),
        Err(_) => (BTreeMap::new(), true),
    };
    let history = burn.as_ref().ok();
    let day_starts: Vec<i64> = history.map(|b| b.day_starts.clone()).unwrap_or_default();
    // Union: a client de-budgeted this week keeps its history columns.
    let clients: BTreeSet<String> = live
        .keys()
        .cloned()
        .chain(
            history
                .into_iter()
                .flat_map(|b| b.by_client.keys().cloned()),
        )
        .collect();

    // Component children are boxed 'static closures: hand them owned values.
    let day_heads = day_starts
        .iter()
        .map(|d| {
            let label = fmt_day(*d);
            view! { <TableHead>{label}</TableHead> }
        })
        .collect_view();

    let rows = clients
        .iter()
        .map(|c| {
            let live_cell = match live.get(c) {
                Some(&(used, limit)) => {
                    let pct = ratio_pct((limit > 0.0).then(|| used / limit));
                    Either::Left(view! {
                        <div class="flex items-center gap-2 whitespace-nowrap">
                            <progress class="h-2 w-24" value=format!("{used:.0}") max=format!("{limit:.0}")></progress>
                            <span class="tabular-nums" title=format!("{used:.0} / {limit:.0} tokens")>
                                {fmt_tokens(used)}" / "{fmt_tokens(limit)}" ("{pct}")"
                            </span>
                        </div>
                    })
                }
                None => Either::Right(view! { <span class="text-muted-foreground">"—"</span> }),
            };
            let series = history
                .and_then(|b| b.by_client.get(c))
                .cloned()
                .unwrap_or_else(|| vec![None; day_starts.len()]);
            let cells = series
                .into_iter()
                .map(|v| {
                    let (label, exact) = match v {
                        Some(x) => (fmt_tokens(x), format!("{x:.0}")),
                        None => ("—".to_string(), String::new()),
                    };
                    view! { <TableCell><span class="tabular-nums" title=exact>{label}</span></TableCell> }
                })
                .collect_view();
            let name = c.clone();
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{name}</span></TableCell>
                    <TableCell>{live_cell}</TableCell>
                    {cells}
                </TableRow>
            }
        })
        .collect_view();

    let live_note = stats
        .as_ref()
        .err()
        .map(|e| unavailable("live budgets (anthropic-lb /_stats)", e));
    let mirror_note = (!fleet_true).then(|| {
        view! {
            <p class="text-sm mb-4">
                <span class=badge(BadgeKind::Warning)>"replica-local"</span>
                " Live numbers come from one replica's mirror (fleet aggregate unavailable): they reset on pod restart and undercount the fleet."
            </p>
        }
    });
    let history_note = burn
        .as_ref()
        .err()
        .map(|e| unavailable("7-day history (metrics store)", e));

    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Per-client budget burn"</CardTitle>
                <CardDescription>
                    "Live: today's fleet-wide burn vs daily limit (anthropic-lb /_stats, Redis aggregate). History: daily peak of anthropic_cluster_budget_used per UTC day from the metrics store; today's column is running. Limits are TOML, GitOps — nothing here edits them."
                </CardDescription>
            </CardHeader>
            <CardContent>
                {live_note}
                {mirror_note}
                {history_note}
                <TableWrapper><Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>"Client"</TableHead>
                            <TableHead>"Today (live) — used / limit"</TableHead>
                            {day_heads}
                        </TableRow>
                    </TableHeader>
                    <TableBody>{rows}</TableBody>
                </Table></TableWrapper>
            </CardContent>
        </Card>
    }
}

// ─── Upstream accounts: utilisation + headroom ──────────────────────────────

fn status_of(e: &Value) -> (String, String) {
    if let Some(secs) = e.get("hard_limited_remaining_secs").and_then(Value::as_u64) {
        return (
            badge(BadgeKind::Destructive),
            format!("hard-limited {secs}s"),
        );
    }
    let (s5, s7) = (vs(e, "status_5h"), vs(e, "status_7d"));
    // Neither window reported: say so, never a fictitious "allowed".
    if s5.is_empty() && s7.is_empty() {
        return (badge(BadgeKind::Muted), "unknown".into());
    }
    let throttled = |s: &str| !s.is_empty() && s != "allowed";
    if throttled(&s5) || throttled(&s7) {
        return (badge(BadgeKind::Warning), format!("{s5} / {s7}"));
    }
    (badge(BadgeKind::Success), "allowed".into())
}

fn accounts_card(stats: &Result<Value, AppError>) -> impl IntoView + use<> {
    let body = match stats {
        Err(e) => Either::Left(unavailable("accounts (anthropic-lb /_stats)", e)),
        Ok(s) => {
            let mut endpoints: Vec<&Value> = s
                .get("endpoints")
                .and_then(Value::as_array)
                .map(|a| a.iter().collect())
                .unwrap_or_default();
            // Hottest first, then by name for a stable order.
            endpoints.sort_by(|a, b| {
                vf(b, "utilization_7d")
                    .total_cmp(&vf(a, "utilization_7d"))
                    .then_with(|| vs(a, "name").cmp(&vs(b, "name")))
            });
            let rows = endpoints
                .into_iter()
                .map(|e| {
                    let name = vs(e, "name");
                    let protocol = vs(e, "protocol");
                    let passthrough = e
                        .get("passthrough")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                        .then(
                            || view! { <span class=badge(BadgeKind::Muted)>"passthrough"</span> },
                        );
                    let priority = int_or_dash(e.get("priority"));
                    let u5 = ratio_pct(e.get("utilization_5h").and_then(Value::as_f64));
                    let u7 = ratio_pct(e.get("utilization_7d").and_then(Value::as_f64));
                    let (status_class, status_text) = status_of(e);
                    let headroom = int_or_dash(e.get("headroom_requests"));
                    let reset_5h = e
                        .get("reset_5h")
                        .and_then(Value::as_f64)
                        .map(fmt_epoch)
                        .unwrap_or_else(|| "—".into());
                    view! {
                        <TableRow>
                            <TableCell>
                                <div class="flex items-center gap-2">
                                    <span class="font-mono text-xs">{name}</span>
                                    {passthrough}
                                </div>
                            </TableCell>
                            <TableCell>{protocol}</TableCell>
                            <TableCell>{priority}</TableCell>
                            <TableCell><span class="tabular-nums">{u5}</span></TableCell>
                            <TableCell><span class="tabular-nums">{u7}</span></TableCell>
                            <TableCell><span class=status_class>{status_text}</span></TableCell>
                            <TableCell><span class="tabular-nums">{headroom}</span></TableCell>
                            <TableCell><span class="whitespace-nowrap">{reset_5h}</span></TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            Either::Right(view! {
                <TableWrapper><Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>"Account"</TableHead>
                            <TableHead>"Protocol"</TableHead>
                            <TableHead>"Priority"</TableHead>
                            <TableHead>"5h util"</TableHead>
                            <TableHead>"7d util"</TableHead>
                            <TableHead>"Status"</TableHead>
                            <TableHead>"Headroom (req)"</TableHead>
                            <TableHead>"5h reset (UTC)"</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>{rows}</TableBody>
                </Table></TableWrapper>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Upstream accounts"</CardTitle>
                <CardDescription>
                    "Per account: Anthropic 5-hour / 7-day window utilisation, routing status and remaining requests, hottest first. Endpoints and priorities are TOML, GitOps."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Shape of a live `/_stats` (2026-09-19), synthetic values.
    fn sample_stats() -> Value {
        json!({
            "strategy": "sticky-weighted-v2",
            "aggregate": {
                "total_headroom_requests": null,
                "consumers": {"multica-runtime": {"requests_per_minute": 6.83, "share": 0.991}}
            },
            // Replica-local mirror: deliberately stale (1 token) so the test
            // proves the fleet-wide aggregate below wins.
            "client_budgets": {"kody": {"daily_limit": 50000000, "used_today": 1, "remaining": 49999999}},
            "cluster": {
                "redis_connected": true,
                "replicas_seen": 3,
                "budget_usage": {
                    "kody": {"limit": 50000000, "used": 17913067},
                    "alaya": {"limit": 20000000, "used": 1516943}
                },
                "transport_errors": {"timeout": 706, "other": 4103}
            },
            "endpoints": [
                {"name": "acct-cool", "priority": 2, "protocol": "anthropic",
                 "utilization_5h": 0.0, "utilization_7d": 0.21,
                 "status_5h": "allowed", "status_7d": "allowed",
                 "headroom_requests": 0, "burn_rate": {"last_1h": 0.0}, "reset_5h": 1789833600},
                {"name": "acct-hot", "priority": 1, "protocol": "anthropic", "passthrough": true,
                 "utilization_5h": 0.9, "utilization_7d": 0.97,
                 "status_5h": "allowed_warning", "status_7d": "allowed",
                 "hard_limited_remaining_secs": 120,
                 "burn_rate": {"last_1h": 3.5}, "reset_5h": 1789830000}
            ]
        })
    }

    fn sample_burn() -> DailyBurn {
        let mut by_client = BTreeMap::new();
        by_client.insert(
            "kody".to_string(),
            vec![Some(22_030_571.0), None, Some(17_913_067.0)],
        );
        DailyBurn {
            day_starts: vec![1_789_603_200, 1_789_689_600, 1_789_776_000],
            by_client,
        }
    }

    #[test]
    fn budgets_card_prefers_fleet_wide_usage_and_renders_history() {
        let html = budgets_card(&Ok(sample_stats()), &Ok(sample_burn())).to_html();
        // Fleet-wide 17.9M, not the replica-local mirror's 1 token. (Adjacent
        // text nodes carry SSR markers between them — assert per fragment.)
        assert!(html.contains("17913067 / 50000000 tokens"), "{html}");
        assert!(html.contains("17.9M") && html.contains("50.0M") && html.contains("36%"));
        assert!(
            html.contains("<progress value=\"17913067\" max=\"50000000\""),
            "{html}"
        );
        assert!(!html.contains("replica-local"));
        // Union of live + history clients; a gap renders as a dash.
        assert!(html.contains("alaya"));
        assert!(html.contains("22.0M") && html.contains(">—<"), "{html}");
        assert!(html.contains("09-17") && html.contains("09-19"), "{html}");
        assert!(html.contains("TOML, GitOps"));
        assert!(!html.contains("<form"));
    }

    /// The LB emits `budget_usage: {}` when its Redis read failed: the card
    /// must fall back to the mirror AND say so, never render blanks under a
    /// "fleet-wide" heading.
    #[test]
    fn budgets_card_falls_back_to_the_mirror_and_flags_it() {
        let mut stats = sample_stats();
        stats["cluster"]["budget_usage"] = json!({});
        stats["cluster"]["redis_connected"] = json!(false);
        let html = budgets_card(&Ok(stats), &Ok(sample_burn())).to_html();
        assert!(html.contains("replica-local"), "{html}");
        assert!(
            html.contains("<progress value=\"1\" max=\"50000000\""),
            "{html}"
        );
    }

    #[test]
    fn accounts_card_sorts_hottest_first_and_flags_limits() {
        let html = accounts_card(&Ok(sample_stats())).to_html();
        let hot = html.find("acct-hot").expect("hot row");
        let cool = html.find("acct-cool").expect("cool row");
        assert!(hot < cool, "hottest account must render first");
        assert!(html.contains("hard-limited 120s"));
        assert!(html.contains("passthrough"));
        assert!(html.contains("97%"));
    }

    /// Process-local counters (consumers, burn rates) must not render: through
    /// a multi-replica Service they describe one random pod.
    #[test]
    fn fleet_card_renders_only_fleet_wide_state() {
        let html = fleet_card(&Ok(sample_stats())).to_html();
        assert!(html.contains("sticky-weighted-v2"));
        assert!(html.contains(">connected<"), "{html}");
        assert!(html.contains("other 4103 · timeout 706"), "{html}");
        assert!(!html.contains("multica-runtime"), "{html}");
    }

    #[test]
    fn status_prefers_hard_limit_then_window_status() {
        let (_, t) = status_of(&json!({"hard_limited_remaining_secs": 42, "status_5h": "allowed"}));
        assert_eq!(t, "hard-limited 42s");
        let (_, t) = status_of(&json!({"status_5h": "allowed_warning", "status_7d": "allowed"}));
        assert_eq!(t, "allowed_warning / allowed");
        let (_, t) = status_of(&json!({"status_5h": "allowed", "status_7d": null}));
        assert_eq!(t, "allowed");
        let (_, t) = status_of(&json!({"name": "acct-new"}));
        assert_eq!(t, "unknown");
    }

    #[test]
    fn ratio_pct_never_invents_a_number() {
        assert_eq!(ratio_pct(None), "—");
        assert_eq!(ratio_pct(Some(0.29)), "29%");
        assert_eq!(ratio_pct((0.0_f64 > 0.0).then(|| 5.0 / 0.0)), "—");
    }
}
