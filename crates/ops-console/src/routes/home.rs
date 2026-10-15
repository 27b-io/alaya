//! Console home: one card per module. Multi-tenant from day one (LAB-1641
//! constraint A): the Ālaya curation module and the anthropic-lb and selecta
//! read-only panes.

use axum::extract::State;
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde_json::Value;

use crate::error::AppError;
use crate::lb::{Budget, fmt_tokens, live_budgets};
use crate::routes;
use crate::routes::health::DAY_COLUMNS;
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

pub async fn home(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);

    // Every probe is informational — a degraded upstream must not take the
    // console home page down with it. Fetched concurrently.
    let lb_probe = async {
        match state.lb.as_ref() {
            Some(lb) => Some(lb.client.stats().await),
            None => None,
        }
    };
    // The one read that both proves the token and says what matters most.
    let selecta_probe = async {
        match state.selecta.as_ref() {
            Some(s) => Some(
                s.list_tasks("awaiting_approval", routes::selecta::AWAITING_SHOWN)
                    .await,
            ),
            None => None,
        }
    };
    let (health, stats, lb_stats, selecta_awaiting) = tokio::join!(
        state.alaya.health(),
        state.alaya.stats(),
        lb_probe,
        selecta_probe
    );
    let (status, detail) = match &health {
        Ok(h) => (
            h.get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_string(),
            "alaya-server is answering.".to_string(),
        ),
        Err(e) => ("unreachable".to_string(), e.detail().to_string()),
    };
    let alaya_status = match status.as_str() {
        "healthy" => Status::Ok,
        "degraded" => Status::Warn,
        _ => Status::Error,
    };
    // With alaya-server down the notice above already says so.
    let kpis = match (&health, &stats) {
        (_, Ok(s)) => Some(Either::Left(alaya_kpis(s))),
        (Ok(_), Err(e)) => Some(Either::Right(view! {
            <div class="mt-4">{unavailable("alaya-server GET /stats", e.detail())}</div>
        })),
        (Err(_), Err(_)) => None,
    };
    let alaya_card = view! {
        <Notice status=alaya_status label=status>{detail}</Notice>
        {kpis}
        {open("/alaya")}
    };

    let lb_card = match &lb_stats {
        None => Either::Left(view! {
            <NotConfigured>"Set LB_URL, LB_API_KEY and METRICS_URL on the console to enable."</NotConfigured>
        }),
        Some(probe) => {
            let (status, label, summary) = match probe {
                Ok(s) => {
                    let accounts = s
                        .get("endpoints")
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len);
                    let budgeted = live_budgets(s).0.len();
                    (
                        Status::Ok,
                        "reachable",
                        format!("{accounts} accounts · {budgeted} budgeted clients"),
                    )
                }
                Err(e) => (Status::Error, "unreachable", e.detail().to_string()),
            };
            let budgets = probe.as_ref().ok().map(tightest_budgets);
            Either::Right(view! {
                <Notice status=status label=label>{summary}</Notice>
                {budgets}
                {open("/lb")}
            })
        }
    };

    let selecta_card = match &selecta_awaiting {
        None => Either::Left(view! {
            <NotConfigured>"Set SELECTA_URL and SELECTA_API_KEY on the console to enable."</NotConfigured>
        }),
        Some(probe) => {
            let mut awaiting = None;
            let (status, label, summary) = match probe {
                Ok(rows) => {
                    let no_pr = rows.iter().filter(|r| r.approval_pr().is_none()).count();
                    // A task with no PR cannot be approved: worth a look,
                    // though selecta itself is answering.
                    let status = if no_pr > 0 { Status::Warn } else { Status::Ok };
                    // A full probe means the list was cut, not that it is exact.
                    let full = rows.len() >= routes::selecta::AWAITING_SHOWN as usize;
                    awaiting = Some((rows.len(), no_pr, full));
                    let mut summary = format!(
                        "{}{} awaiting your approval",
                        rows.len(),
                        if full { "+" } else { "" }
                    );
                    if no_pr > 0 {
                        summary.push_str(&format!(" · {no_pr} with no PR"));
                    }
                    (status, "reachable", summary)
                }
                // "error", not "unreachable": a 401 or a tool error arrives
                // over a healthy network. The detail names the kind.
                Err(e) => (Status::Error, "error", e.detail().to_string()),
            };
            let mix = awaiting.map(|(n, no_pr, full)| {
                let parts = vec![
                    Part::new(
                        "with an approval PR",
                        Some((n - no_pr) as f64),
                        Colour::SERIES[0],
                    ),
                    Part::new("no PR", Some(no_pr as f64), Colour::SERIES[1]),
                ];
                // A cut list breaks down only the rows returned, newest first.
                let label = if full {
                    format!("The newest {n} tasks awaiting approval, by whether they have a PR")
                } else {
                    "Tasks awaiting approval, by whether they have a PR".to_string()
                };
                view! {
                    <div class="mt-4">
                        <StackedBar label=label parts=parts />
                    </div>
                }
            });
            Either::Right(view! {
                <Notice status=status label=label>{summary}</Notice>
                {mix}
                {open("/selecta")}
            })
        }
    };

    let content = view! {
        <div class="space-y-6">
            <PageHeader
                title="Overview"
                description="Each module this console runs, and whether its upstream is answering."
            />
            <div class="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
                <Card>
                    <CardHeader>
                        <CardTitle>"Ālaya — memory curation"</CardTitle>
                        <CardDescription>
                            "Browse, search and curate the memory corpus: supersede, delete, merge duplicates, relations, contradictions."
                        </CardDescription>
                    </CardHeader>
                    <CardContent>{alaya_card}</CardContent>
                </Card>
                <Card>
                    <CardHeader>
                        <CardTitle>"anthropic-lb — monitoring"</CardTitle>
                        <CardDescription>
                            "Read-only budget burn and account utilisation for the load balancer. Limits are TOML, GitOps."
                        </CardDescription>
                    </CardHeader>
                    <CardContent>{lb_card}</CardContent>
                </Card>
                <Card>
                    <CardHeader>
                        <CardTitle>"selecta — what it is doing"</CardTitle>
                        <CardDescription>
                            "Read-only: failures and why, tasks waiting on your approval, and whether selecta is parked, fenced or stalled. Approvals happen on GitHub."
                        </CardDescription>
                    </CardHeader>
                    <CardContent>{selecta_card}</CardContent>
                </Card>
            </div>
        </div>
    };

    Ok((
        jar,
        Html(page("ops console", Nav::Home, &session, flash, content)),
    ))
}

/// A module card's way into its module.
fn open(href: &'static str) -> impl IntoView {
    view! {
        <a href=href class=format!("mt-4 {}", btn_sm(Btn::Outline))>
            "Open module"
        </a>
    }
}

/// Ālaya's key values: the memory count and the judge backlog, with the
/// pairs judged each day under it. Read from `GET /stats`, whose count is
/// null when it failed; `/health/detail` reports a failed count as 0.
fn alaya_kpis(stats: &Value) -> impl IntoView + use<> {
    let count = |v: Option<&Value>| v.and_then(Value::as_u64).map(|n| n as f64);
    let figure = |v: Option<f64>| v.map_or_else(|| "—".to_string(), fmt_count);
    let memories = count(stats.pointer("/memories/total"));
    let backlog = count(stats.pointer("/contradictions/by_verdict/never_judged/open"));
    let days = stats
        .pointer("/contradictions/judged_per_day")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    // A day's total is unknown when any verdict's count is missing or bad:
    // the health pane's own columns, so a partial day is never a small one.
    let judged = days
        .iter()
        .map(|d| {
            let counts = d.get("counts");
            DAY_COLUMNS
                .iter()
                .map(|(key, ..)| count(counts.and_then(|c| c.get(*key))))
                .sum::<Option<f64>>()
        })
        .collect::<Vec<_>>();
    let x = days
        .iter()
        .map(|d| {
            d.get("date")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string()
        })
        .collect::<Vec<_>>();
    view! {
        <dl class="mt-4 grid grid-cols-2 gap-3">
            <Stat label="Memories">{figure(memories)}</Stat>
            <Stat label="Judge backlog">
                {figure(backlog)}
                <Sparkline label="Pairs judged per day" values=judged x=x />
            </Stat>
        </dl>
    }
}

/// The LB's clients nearest their daily limit, each a meter, tightest first.
fn tightest_budgets(stats: &Value) -> impl IntoView + use<> {
    const SHOWN: usize = 3;
    let (budgets, fleet) = live_budgets(stats);
    let mut rows: Vec<_> = budgets.into_iter().collect();
    // The meter's own share rule: a client with none (a figure missing or
    // negative, or a zero limit) sorts last, and its meter says "no data".
    let rank = |&(used, limit): &Budget| share(used, limit).unwrap_or(f64::NEG_INFINITY);
    rows.sort_by(|(_, a), (_, b)| rank(b).total_cmp(&rank(a)));
    let more = rows.len().saturating_sub(SHOWN);
    let meters = rows
        .into_iter()
        .take(SHOWN)
        .map(|(client, (used, limit))| {
            let label = format!("{client}: live budget");
            view! {
                <li class="min-w-0 space-y-1">
                    <p class="font-mono text-xs wrap-break-word">{client}</p>
                    <Meter label=label used=used limit=limit fmt=fmt_tokens />
                </li>
            }
        })
        .collect_view();
    let note = match (more, fleet) {
        (0, true) => None,
        (0, false) => {
            Some("One replica's figures: the fleet-wide ones are unavailable.".to_string())
        }
        (n, true) => Some(format!("{n} more on the module page.")),
        (n, false) => Some(format!(
            "{n} more on the module page. One replica's figures: the fleet-wide ones are unavailable."
        )),
    };
    view! {
        <ul class="mt-4 space-y-3">{meters}</ul>
        {note.map(|n| view! { <p class="mt-2 text-xs text-muted-foreground">{n}</p> })}
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn html(v: impl IntoView) -> String {
        v.into_view().to_html()
    }

    /// The tightest budgets come first, three at most, and the rest are
    /// counted; a client with a zero limit has no share and sorts last.
    #[test]
    fn the_lb_card_meters_the_tightest_budgets_first() {
        let budget = |used: u64, limit: u64| json!({ "used": used, "limit": limit });
        let stats = json!({ "cluster": { "budget_usage": {
            "loose": budget(1, 100),
            "tight": budget(95, 100),
            "unlimited": budget(5, 0),
            "middling": budget(50, 100),
            // A limit with no used figure has no share either.
            "unreported": json!({ "limit": 100 }),
            "over": budget(130, 100),
        }}});
        let out = html(tightest_budgets(&stats));
        let order: Vec<usize> = ["over", "tight", "middling"]
            .iter()
            .map(|c| {
                out.find(&format!(r#"aria-label="{c}: live budget"#))
                    .expect(c)
            })
            .collect();
        assert!(order.is_sorted(), "{out}");
        assert!(
            !out.contains("loose") && !out.contains("unlimited") && !out.contains("unreported"),
            "{out}"
        );
        assert!(out.contains("3 more on the module page."), "{out}");
        assert!(!out.contains("One replica"), "{out}");
    }

    /// A client with no share sorts below one with a share of nothing: a
    /// missing figure is not a zero, and a negative one is not a share.
    /// Their names sort first, so a tie with `idle` would put them on top.
    #[test]
    fn a_client_with_no_share_sorts_below_an_idle_one() {
        let stats = json!({ "cluster": { "budget_usage": {
            "absent": { "limit": 100 },
            "broken": { "used": -5, "limit": 100 },
            "idle": { "used": 0, "limit": 100 },
        }}});
        let out = html(tightest_budgets(&stats));
        let order: Vec<usize> = ["idle", "absent", "broken"]
            .iter()
            .map(|c| {
                out.find(&format!(r#"aria-label="{c}: live budget"#))
                    .expect(c)
            })
            .collect();
        assert!(order.is_sorted(), "{out}");
        assert!(
            out.contains("broken: live budget: — / 100 · no data"),
            "{out}"
        );
    }

    /// Figures from one replica's mirror say so.
    #[test]
    fn the_lb_card_says_when_its_figures_are_one_replicas() {
        let stats = json!({
            "client_budgets": { "kody": { "daily_limit": 100, "used_today": 40 } },
        });
        let out = html(tightest_budgets(&stats));
        assert!(out.contains("kody: live budget: 40 / 100, 40%"), "{out}");
        assert!(
            out.contains("the fleet-wide ones are unavailable."),
            "{out}"
        );
    }

    /// The memory count and backlog come from `/stats`; a null there is a
    /// dash, never a 0, and a day with a bad count is a gap in the trend.
    #[test]
    fn alaya_kpis_read_stats_and_never_invent_a_zero() {
        let day = |date: &str, coexist: Value| {
            json!({ "date": date, "counts": {
                "contradiction": 0, "supersession": 0, "coexist": coexist, "unrelated": 2, "unjudged": 0,
            }})
        };
        let stats = json!({
            "memories": { "total": 5_234 },
            "contradictions": {
                "by_verdict": { "never_judged": { "open": 1_716, "resolved": 97 } },
                "judged_per_day": [day("2026-09-30", json!(148)), day("2026-10-01", json!("x"))],
            },
        });
        let out = html(alaya_kpis(&stats));
        assert!(out.contains(">5,234<") && out.contains(">1,716"), "{out}");
        assert!(
            out.contains("Pairs judged per day: 2 points from 2026-09-30 to 2026-10-01; latest no data; highest 150"),
            "{out}"
        );

        let out = html(alaya_kpis(&json!({ "memories": { "total": null } })));
        assert_eq!(out.matches(">—<").count(), 2, "{out}");
        assert!(!out.contains(">0<"), "{out}");
    }

    /// A day missing any verdict's count has an unknown total: a partial
    /// `counts` is not a smaller day, and an empty one is not a quiet day.
    #[test]
    fn a_day_missing_a_verdict_count_is_a_gap_in_the_trend() {
        let full = json!({ "date": "2026-09-30", "counts": {
            "contradiction": 1, "supersession": 2, "coexist": 3, "unrelated": 4, "unjudged": 5,
        }});
        let stats = json!({ "contradictions": { "judged_per_day": [
            full,
            { "date": "2026-10-01", "counts": { "coexist": 2 } },
            { "date": "2026-10-02", "counts": {} },
        ]}});
        let out = html(alaya_kpis(&stats));
        assert!(
            out.contains("Pairs judged per day: 3 points from 2026-09-30 to 2026-10-02; latest no data; highest 15; lowest 15; no data for 2"),
            "{out}"
        );
    }
}
