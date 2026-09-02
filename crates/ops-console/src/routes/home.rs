//! Console home: one card per module. The skeleton is two-tenant from day
//! one (LAB-1641 constraint A) — the anthropic-lb pane lands as the second
//! route module under LAB-1964; here it renders as a staged placeholder.

use axum::extract::State;
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::prelude::*;

use crate::error::AppError;
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

pub async fn home(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);

    // Health is informational — a degraded alaya-server must not take the
    // console home page down with it.
    let health = state.alaya.health().await.ok();
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
                        "Read-only budget burn and account headroom for the load balancer."
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <span class=badge(BadgeKind::Muted)>"staged — LAB-1964"</span>
                </CardContent>
            </Card>
        </div>
    };

    Ok((jar, Html(page("ops console", &session, flash, content))))
}
