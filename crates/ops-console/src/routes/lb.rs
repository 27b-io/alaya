//! anthropic-lb module — read-only monitoring pane (LAB-1964).
//!
//! GET only. Renders per-client budget burn vs budget (live from `/_stats`,
//! 7-day history from the metrics store) and per-account utilisation. Every
//! card names its source; limits render as "TOML, GitOps".
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
            <div class="space-y-6">
                {header()}
                <NotConfigured>
                    "Set LB_URL, LB_API_KEY and METRICS_URL on the console deployment (all three, or none) to enable read-only LB monitoring."
                </NotConfigured>
            </div>
        };
        return Ok((jar, Html(page(TITLE, Nav::Lb, &session, flash, content))));
    };

    // Sections are independent: a dark metrics store must not hide live
    // utilisation, and an LB outage must not hide the burn history.
    let (stats, burn) = tokio::join!(lb.client.stats(), lb.metrics.daily_burn());

    let content = view! {
        <div class="space-y-6">
            {header()}
            {fleet_card(&stats)}
            {budgets_card(&stats, &burn)}
            {accounts_card(&stats)}
        </div>
    };
    Ok((jar, Html(page(TITLE, Nav::Lb, &session, flash, content))))
}

fn header() -> impl IntoView {
    view! {
        <PageHeader
            title="anthropic-lb"
            description="Read-only monitoring of the load balancer: fleet state, per-client budget burn and upstream account utilisation. Limits are TOML, GitOps."
        />
    }
}

// ─── Rendering helpers ──────────────────────────────────────────────────────

/// Optional integer for a cell, or a dash — never a fictitious zero.
fn int_or_dash(v: Option<&Value>) -> String {
    v.and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}

fn ratio_pct(r: Option<f64>) -> String {
    r.filter(|x| x.is_finite())
        .map(|x| format!("{:.0}%", x * 100.0))
        .unwrap_or_else(|| "—".into())
}

/// A token figure for a cell, or a dash: never a fictitious zero.
fn tokens(v: Option<f64>) -> String {
    v.map_or_else(|| "—".into(), fmt_tokens)
}

/// The exact figure for a cell's `title`, or a dash.
fn exact(v: Option<f64>) -> String {
    v.filter(|x| x.is_finite())
        .map_or_else(|| "—".into(), |x| format!("{x:.0}"))
}

/// `MM-DD` column header for a UTC-midnight epoch.
fn fmt_day(midnight: i64) -> String {
    fmt_epoch(midnight as f64)
        .get(5..10)
        .unwrap_or("—")
        .to_string()
}

// ─── Fleet: strategy, replicas, shared state ────────────────────────────────

/// Transport errors by kind, most first: a count that is not a number is
/// "no data", never a zero. An empty map is no errors; no map is no
/// cluster info, so the ranking says it has no data.
fn transport_errors(cluster: Option<&Value>) -> impl IntoView + use<> {
    let errors = cluster
        .and_then(|c| c.get("transport_errors"))
        .and_then(Value::as_object);
    if errors.is_some_and(|m| m.is_empty()) {
        return Either::Left(
            view! { <p class="text-sm text-muted-foreground">"None recorded."</p> },
        );
    }
    let mut rows: Vec<Part> = errors
        .into_iter()
        .flatten()
        .map(|(kind, n)| Part::new(kind.clone(), n.as_f64(), Colour::SERIES[0]))
        .collect();
    // serde_json maps iterate in key order, which the stable sort keeps for ties.
    rows.sort_by(|a, b| b.value.unwrap_or(-1.0).total_cmp(&a.value.unwrap_or(-1.0)));
    Either::Right(view! { <HBarList label="Upstream transport errors" rows=rows /> })
}

fn fleet_card(stats: &Result<Value, AppError>) -> impl IntoView + use<> {
    let body = match stats {
        Err(e) => Either::Left(unavailable("anthropic-lb /_stats", e.detail())),
        Ok(s) => {
            let strategy = vs(s, "strategy");
            let cluster = s.get("cluster");
            let replicas = int_or_dash(cluster.and_then(|c| c.get("replicas_seen")));
            let (redis, redis_text) = match cluster
                .and_then(|c| c.get("redis_connected"))
                .and_then(Value::as_bool)
            {
                Some(true) => (Status::Ok, "connected"),
                Some(false) => (Status::Error, "disconnected"),
                None => (Status::Off, "no cluster info"),
            };
            let transport = transport_errors(cluster);
            Either::Right(view! {
                <StatGrid>
                    <Stat label="Strategy" text=true>{strategy}</Stat>
                    // `replicas_seen` counts heartbeating instances in the shared
                    // coordination store, which more than one deployment writes to —
                    // it is not this Deployment's replica count, and reading it as
                    // one sends people hunting a phantom pod (LAB-4554).
                    <Stat label="Heartbeating instances (shared store)">{replicas}</Stat>
                    <Stat label="Shared state (Redis)">
                        <StatusBadge status=redis label=redis_text />
                    </Stat>
                    // No pooled-headroom tile: see the note above accounts_card (LAB-4554).
                </StatGrid>
                <h3 class="text-sm font-medium mt-6 mb-2">"Upstream transport errors (fleet, cumulative)"</h3>
                {transport}
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

    let day_labels: Vec<String> = day_starts.iter().map(|d| fmt_day(*d)).collect();
    // A client's history, a dash per day it has none; `None` while the
    // metrics store is dark, so no chart claims to know the week.
    let series_of = |c: &str| -> Option<Vec<Option<f64>>> {
        history.map(|b| {
            b.by_client
                .get(c)
                .cloned()
                .unwrap_or_else(|| vec![None; day_starts.len()])
        })
    };

    let tiles = clients
        .iter()
        .map(|c| {
            let live_meter = match live.get(c) {
                Some(&(used, limit)) => {
                    let label = format!("{c}: live budget");
                    Either::Left(
                        view! { <Meter label=label used=used limit=limit fmt=fmt_tokens /> },
                    )
                }
                None => Either::Right(
                    view! { <p class="text-sm text-muted-foreground">"no live figure"</p> },
                ),
            };
            let trend = series_of(c).map(|values| {
                let (label, x) = (format!("{c}: daily burn"), day_labels.clone());
                view! {
                    <div>
                        <p class="mt-3 text-xs text-muted-foreground">"Daily burn, 7 days"</p>
                        <Sparkline label=label values=values x=x fmt=fmt_tokens />
                    </div>
                }
            });
            let name = c.clone();
            view! {
                <li class="min-w-0 space-y-2 rounded-lg border bg-card p-4">
                    <p class="font-mono text-xs wrap-break-word">{name}</p>
                    {live_meter}
                    {trend}
                </li>
            }
        })
        .collect_view();

    // Component children are boxed 'static closures: hand them owned values.
    let day_heads = day_labels
        .iter()
        .map(|label| {
            let label = label.clone();
            view! { <TableHead secondary=true>{label}</TableHead> }
        })
        .collect_view();

    let rows = clients
        .iter()
        .map(|c| {
            let live_cell = match live.get(c) {
                Some(&(used, limit)) => {
                    let pct = ratio_pct(used.zip(limit.filter(|l| *l > 0.0)).map(|(u, l)| u / l));
                    let title = format!("{} / {} tokens", exact(used), exact(limit));
                    let text = format!("{} / {} ({pct})", tokens(used), tokens(limit));
                    Either::Left(view! {
                        <span class="tabular-nums whitespace-nowrap" title=title>{text}</span>
                    })
                }
                None => Either::Right(view! { <span class="text-muted-foreground">"—"</span> }),
            };
            let cells = series_of(c)
                .unwrap_or_else(|| vec![None; day_starts.len()])
                .into_iter()
                .map(|v| {
                    let (label, title) = (tokens(v), exact(v));
                    view! {
                        <TableCell secondary=true>
                            <span class="tabular-nums" title=title>{label}</span>
                        </TableCell>
                    }
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

    // Nothing to chart when no source named a client: the notices say why.
    let charts = (!clients.is_empty()).then(|| {
        view! {
        <ul class="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">{tiles}</ul>
        <details class="mt-6">
            <summary class="text-sm cursor-pointer">"Daily figures"</summary>
            <div class="mt-3">
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
            </div>
        </details>
        }
    });

    let live_note = stats
        .as_ref()
        .err()
        .map(|e| view! { <div class="mb-4">{unavailable("live budgets (anthropic-lb /_stats)", e.detail())}</div> });
    let mirror_note = (!fleet_true).then(|| {
        view! {
            <div class="mb-4">
                <Notice status=Status::Warn label="replica-local">
                    "Live numbers come from one replica's mirror (fleet aggregate unavailable): they reset on pod restart and undercount the fleet."
                </Notice>
            </div>
        }
    });
    let history_note = burn
        .as_ref()
        .err()
        .map(|e| view! { <div class="mb-4">{unavailable("7-day history (metrics store)", e.detail())}</div> });

    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Per-client budget burn"</CardTitle>
                <CardDescription>
                    "Live: today's fleet-wide burn vs daily limit (anthropic-lb /_stats, Redis aggregate). History: daily peak of anthropic_cluster_budget_used per UTC day from the metrics store, today's still running. Each client's meter turns amber at 80% and red at its limit; the daily figures fold out below. Limits are TOML, GitOps — nothing here edits them."
                </CardDescription>
            </CardHeader>
            <CardContent>
                {live_note}
                {mirror_note}
                {history_note}
                {charts}
            </CardContent>
        </Card>
    }
}

// ─── Upstream accounts: utilisation ─────────────────────────────────────────

fn status_of(e: &Value) -> (Status, String) {
    if let Some(secs) = e.get("hard_limited_remaining_secs").and_then(Value::as_u64) {
        return (Status::Error, format!("hard-limited {secs}s"));
    }
    let (s5, s7) = (vs(e, "status_5h"), vs(e, "status_7d"));
    // Neither window reported: say so, never a fictitious "allowed".
    if s5.is_empty() && s7.is_empty() {
        return (Status::Off, "unknown".into());
    }
    let throttled = |s: &str| !s.is_empty() && s != "allowed";
    if throttled(&s5) || throttled(&s7) {
        return (Status::Warn, format!("{s5} / {s7}"));
    }
    (Status::Ok, "allowed".into())
}

/// Where an account's utilisation meter turns amber and red: the floors the
/// LB itself reads its upstream statuses as, `allowed_warning` at 0.80 and
/// `throttled` at 0.98 (anthropic-lb's `utilization.rs`). The status badge
/// still carries the upstream's own word, which the meter never overrides.
const UTIL_WARN: f64 = 0.80;
const UTIL_CRIT: f64 = 0.98;

// Deliberately no request-headroom column, and none of the `headroom_requests`
// / `total_headroom_requests` fields `/_stats` publishes: Anthropic's headers
// carry token/claim utilisation, not a request-count limit, so they are always
// null or 0 here (measured live twice, seven weeks apart — LAB-4554).
// Utilisation is the signal that exists. Don't re-add the column.
fn accounts_card(stats: &Result<Value, AppError>) -> impl IntoView + use<> {
    let body = match stats {
        Err(e) => Either::Left(unavailable("accounts (anthropic-lb /_stats)", e.detail())),
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
                        .then(|| view! { <StatusBadge status=Status::Off label="passthrough" /> });
                    let priority = int_or_dash(e.get("priority"));
                    let util = |window: &str| {
                        let label = format!("{name}: {window} utilisation");
                        let used = e.get(format!("utilization_{window}").as_str()).and_then(Value::as_f64);
                        view! {
                            <div class="min-w-24">
                                <Meter label=label used=used limit=Some(1.0) warn=UTIL_WARN crit=UTIL_CRIT />
                            </div>
                        }
                    };
                    let (u5, u7) = (util("5h"), util("7d"));
                    let (status, status_text) = status_of(e);
                    let reset_5h = e
                        .get("reset_5h")
                        .and_then(Value::as_f64)
                        .map(fmt_epoch)
                        .unwrap_or_else(|| "—".into());
                    view! {
                        <TableRow>
                            <TableCell>
                                <div class="flex flex-wrap items-center gap-2">
                                    <span class="font-mono text-xs wrap-break-word">{name}</span>
                                    {passthrough}
                                </div>
                            </TableCell>
                            <TableCell secondary=true>{protocol}</TableCell>
                            <TableCell secondary=true>{priority}</TableCell>
                            <TableCell>{u5}</TableCell>
                            <TableCell>{u7}</TableCell>
                            <TableCell><StatusBadge status=status label=status_text /></TableCell>
                            <TableCell secondary=true><span class="whitespace-nowrap">{reset_5h}</span></TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            Either::Right(view! {
                <TableWrapper><Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>"Account"</TableHead>
                            <TableHead secondary=true>"Protocol"</TableHead>
                            <TableHead secondary=true>"Priority"</TableHead>
                            <TableHead>"5h util"</TableHead>
                            <TableHead>"7d util"</TableHead>
                            <TableHead>"Status"</TableHead>
                            <TableHead secondary=true>"5h reset (UTC)"</TableHead>
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
                    "Per account: Anthropic 5-hour / 7-day window utilisation and routing status, hottest first. Endpoints and priorities are TOML, GitOps."
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
                 "burn_rate": {"last_1h": 0.0}, "reset_5h": 1789833600},
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
            html.contains("aria-label=\"kody: live budget: 17.9M / 50.0M, 36%\""),
            "{html}"
        );
        assert!(!html.contains("replica-local"));
        // Union of live + history clients; a gap renders as a dash in the
        // table and as no data in the trend, never as a zero.
        assert!(html.contains("alaya"));
        assert!(html.contains("22.0M") && html.contains(">—<"), "{html}");
        assert!(
            html.contains("aria-label=\"kody: daily burn: 3 points from 09-17 to 09-19; latest 17.9M; highest 22.0M; lowest 17.9M; no data for 1\""),
            "{html}"
        );
        // alaya has a live budget and no history: its trend says so.
        assert!(html.contains("alaya: daily burn: no data"), "{html}");
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
            html.contains("aria-label=\"kody: live budget: 1 / 50.0M, &lt;1%\""),
            "{html}"
        );
    }

    /// A dark source keeps its named notice, and no chart stands in for it
    /// with zeros: no meter without `/_stats`, no trend without the metrics
    /// store.
    #[test]
    fn budgets_card_draws_nothing_for_a_dark_source() {
        let down = || AppError::Upstream("connection refused".into());
        let html = budgets_card(&Err(down()), &Ok(sample_burn())).to_html();
        assert!(
            html.contains("live budgets (anthropic-lb /_stats)"),
            "{html}"
        );
        assert!(!html.contains("live budget:"), "{html}");
        assert!(html.contains("kody: daily burn: 3 points"), "{html}");
        let html = budgets_card(&Ok(sample_stats()), &Err(down())).to_html();
        assert!(html.contains("7-day history (metrics store)"), "{html}");
        assert!(!html.contains("daily burn"), "{html}");
        assert!(
            html.contains("kody: live budget: 17.9M / 50.0M, 36%"),
            "{html}"
        ); // Both dark: the two notices, and no empty grid or table under them.
        let html = budgets_card(&Err(down()), &Err(down())).to_html();
        assert!(
            html.contains("live budgets (anthropic-lb /_stats)"),
            "{html}"
        );
        assert!(html.contains("7-day history (metrics store)"), "{html}");
        assert!(
            !html.contains("Daily figures") && !html.contains("<ul"),
            "{html}"
        );
    }

    /// A budget the LB reports without a figure says "no data", not 0%.
    #[test]
    fn a_budget_missing_its_used_figure_is_no_data() {
        let mut stats = sample_stats();
        stats["cluster"]["budget_usage"]["kody"] = json!({"limit": 50000000});
        let html = budgets_card(&Ok(stats), &Ok(sample_burn())).to_html();
        assert!(
            html.contains("kody: live budget: — / 50.0M · no data"),
            "{html}"
        );
        assert!(html.contains(">— / 50.0M (—)<"), "{html}");
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
        // Utilisation is a meter on the LB's own status floors: 0.97 is
        // past the warning floor, short of the throttled one.
        assert!(
            html.contains("aria-label=\"acct-hot: 7d utilisation: 97%, warning\""),
            "{html}"
        );
        assert!(
            html.contains("acct-hot: 5h utilisation: 90%, warning"),
            "{html}"
        );
        assert!(html.contains("acct-cool: 5h utilisation: 0%\""), "{html}");
        // LAB-4554: request headroom is structurally empty upstream. Matched
        // case-insensitively so a re-add under any label trips the guard.
        assert!(!html.to_lowercase().contains("headroom"), "{html}");
    }

    /// Process-local counters (consumers, burn rates) must not render: through
    /// a multi-replica Service they describe one random pod.
    #[test]
    fn fleet_card_renders_only_fleet_wide_state() {
        let html = fleet_card(&Ok(sample_stats())).to_html();
        assert!(html.contains("sticky-weighted-v2"));
        assert!(html.contains(">connected<"), "{html}");
        // Transport errors ranked, most first.
        assert!(
            html.contains("aria-label=\"Upstream transport errors: other 4,103, timeout 706\""),
            "{html}"
        );
        assert!(!html.contains("multica-runtime"), "{html}");
        // LAB-4554: the pooled request-headroom tile is gone for good, and the
        // instance count says what it counts rather than "Replicas seen".
        assert!(!html.to_lowercase().contains("headroom"), "{html}");
        assert!(
            html.contains("Heartbeating instances (shared store)"),
            "{html}"
        );
        assert!(!html.contains("Replicas seen"), "{html}");
    }

    /// No errors is not no data, and no cluster info is not no errors.
    #[test]
    fn transport_errors_tell_none_from_unknown() {
        let html = transport_errors(Some(&json!({"transport_errors": {}}))).to_html();
        assert!(html.contains("None recorded."), "{html}");
        let html = transport_errors(None).to_html();
        assert!(
            html.contains("no data") && !html.contains("None recorded."),
            "{html}"
        );
        let html = transport_errors(Some(
            &json!({"transport_errors": {"a": 1, "b": "x", "c": 5}}),
        ))
        .to_html();
        assert!(
            html.contains("Upstream transport errors: c 5, a 1, b no data"),
            "{html}"
        );
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
