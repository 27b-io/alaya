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
use crate::lb::live_budgets;
use crate::routes;
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
    let (health, lb_stats, selecta_awaiting) =
        tokio::join!(state.alaya.health(), lb_probe, selecta_probe);
    let health = health.ok();
    let (status, memories) = match &health {
        Some(h) => (
            h.get("status")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_string(),
            h.get("storage")
                .and_then(|s| s.get("total_memories"))
                .or_else(|| h.get("total_memories"))
                .and_then(|n| n.as_u64())
                .map(|n| n.to_string())
                .unwrap_or_else(|| "—".into()),
        ),
        None => ("unreachable".to_string(), "—".into()),
    };
    let alaya_status = match status.as_str() {
        "healthy" => Status::Ok,
        "degraded" => Status::Warn,
        _ => Status::Error,
    };
    let alaya_card = view! {
        <Notice status=alaya_status label=status>{format!("{memories} memories")}</Notice>
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
            Either::Right(view! {
                <Notice status=status label=label>{summary}</Notice>
                {open("/lb")}
            })
        }
    };

    let selecta_card = match &selecta_awaiting {
        None => Either::Left(view! {
            <NotConfigured>"Set SELECTA_URL and SELECTA_API_KEY on the console to enable."</NotConfigured>
        }),
        Some(probe) => {
            let (status, label, summary) = match probe {
                Ok(rows) => {
                    let no_pr = rows.iter().filter(|r| r.approval_pr().is_none()).count();
                    // A task with no PR cannot be approved: worth a look,
                    // though selecta itself is answering.
                    let status = if no_pr > 0 { Status::Warn } else { Status::Ok };
                    // A full probe means the list was cut, not that it is exact.
                    let full = rows.len() >= routes::selecta::AWAITING_SHOWN as usize;
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
            Either::Right(view! {
                <Notice status=status label=label>{summary}</Notice>
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
