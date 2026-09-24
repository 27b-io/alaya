//! Console home: one card per module. Two-tenant from day one (LAB-1641
//! constraint A): the Ālaya curation module and the anthropic-lb read-only
//! monitoring pane.

use axum::extract::State;
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde_json::Value;

use crate::error::AppError;
use crate::lb::live_budgets;
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

pub async fn home(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);

    // Both probes are informational — a degraded upstream must not take the
    // console home page down with it. Fetched concurrently.
    let lb_probe = async {
        match state.lb.as_ref() {
            Some(lb) => Some(lb.client.stats().await),
            None => None,
        }
    };
    let (health, lb_stats) = tokio::join!(state.alaya.health(), lb_probe);
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
    let status_badge = match status.as_str() {
        "healthy" => badge(BadgeKind::Success),
        "degraded" => badge(BadgeKind::Warning),
        _ => badge(BadgeKind::Destructive),
    };

    let lb_card = match &lb_stats {
        None => Either::Left(view! {
            <div class="text-sm">
                <span class=badge(BadgeKind::Muted)>"not configured"</span>
                <p class="text-muted-foreground mt-2">
                    "Set LB_URL, LB_API_KEY and METRICS_URL on the console to enable."
                </p>
            </div>
        }),
        Some(probe) => {
            let (class, label, summary) = match probe {
                Ok(s) => {
                    let accounts = s
                        .get("endpoints")
                        .and_then(Value::as_array)
                        .map_or(0, Vec::len);
                    let budgeted = live_budgets(s).0.len();
                    (
                        badge(BadgeKind::Success),
                        "reachable",
                        format!("{accounts} accounts · {budgeted} budgeted clients"),
                    )
                }
                Err(e) => (
                    badge(BadgeKind::Destructive),
                    "unreachable",
                    e.detail().to_string(),
                ),
            };
            Either::Right(view! {
                <div class="flex items-center gap-3 text-sm mb-4">
                    <span class=class>{label}</span>
                    <span class="text-muted-foreground">{summary}</span>
                </div>
                <a href="/lb" class=btn(Btn::Default)>"Open module"</a>
            })
        }
    };

    let content = view! {
        <div class="grid gap-6 sm:grid-cols-2">
            <Card>
                <CardHeader>
                    <CardTitle>"Ālaya — memory curation"</CardTitle>
                    <CardDescription>
                        "Browse, search and curate the memory corpus: supersede, delete, merge duplicates, relations, contradictions."
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div class="flex items-center gap-3 text-sm mb-4">
                        <span class=status_badge>{status}</span>
                        <span class="text-muted-foreground">{memories}" memories"</span>
                    </div>
                    <a href="/alaya" class=btn(Btn::Default)>"Open module"</a>
                </CardContent>
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
        </div>
    };

    Ok((jar, Html(page("ops console", &session, flash, content))))
}
