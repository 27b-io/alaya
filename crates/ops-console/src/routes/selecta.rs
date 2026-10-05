//! selecta module — read-only pane (LAB-6674).
//!
//! GET only. One screen: is selecta alive (heartbeat, lease, run budget),
//! what is waiting on the operator (gated tasks and their approval PRs), what
//! failed and why, task counts per state, and download health. Every card
//! names its source.
//!
//! There is no form here except the shell's logout, and there must never be
//! one: approvals are GitHub PR merges, so a gated task renders only as a
//! link to its PR. The console's selecta credential cannot enqueue work
//! anyway — but this module must not grow a route that tries.
//!
//! Fail loud: a section whose source is down or whose payload does not parse
//! renders as an error naming the tool, never as an empty list.

use axum::extract::{Path, State};
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;

use crate::error::AppError;
use crate::lb::fmt_tokens;
use crate::routes::{fmt_epoch, unavailable};
use crate::selecta::{
    DOWNLOAD_HEALTH, GET_TASK, Gauges, HealthSnapshot, LIST_TASKS, Liveness, STATES, TORN_SENTINEL,
    TaskRow, github_pr, valid_task_id,
};
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

const TITLE: &str = "selecta — ops console";
const FAILED_SHOWN: u32 = 20;
/// Also the home card's probe size: it says "N+" when a list fills it.
pub const AWAITING_SHOWN: u32 = 100;

fn not_configured() -> impl IntoView + use<> {
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"selecta — module not configured"</CardTitle>
                <CardDescription>
                    "Set SELECTA_URL and SELECTA_API_KEY (a selecta read-only token) on the console deployment, both or neither, to enable the read-only selecta pane."
                </CardDescription>
            </CardHeader>
        </Card>
    }
}

pub async fn pane(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(selecta) = state.selecta.as_ref() else {
        return Ok((jar, Html(page(TITLE, &session, flash, not_configured()))));
    };

    // Independent sources: any one dark leaves the others up.
    let (gauges, live, awaiting, failed, health) = tokio::join!(
        selecta.gauges(),
        selecta.liveness(),
        selecta.list_tasks("awaiting_approval", AWAITING_SHOWN),
        selecta.list_tasks("failed", FAILED_SHOWN),
        selecta.download_health(),
    );

    let content = view! {
        <div class="space-y-6">
            {runner_card(&live, &gauges)}
            {approvals_card(&awaiting, &gauges)}
            {failures_card(&failed, &gauges)}
            {states_card(&gauges)}
            {health_card(&health)}
        </div>
    };
    Ok((jar, Html(page(TITLE, &session, flash, content))))
}

pub async fn task(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    Path(id): Path<String>,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    if !valid_task_id(&id) {
        return Err(AppError::BadRequest("invalid selecta task id".into()));
    }
    let (jar, flash) = take_flash(jar);
    let Some(selecta) = state.selecta.as_ref() else {
        return Ok((jar, Html(page(TITLE, &session, flash, not_configured()))));
    };
    let body = match selecta.get_task(&id).await {
        Err(e) => Either::Left(unavailable("task detail", &e)),
        Ok(t) => {
            let pr = pr_cell(t.pr_url.as_deref(), t.state == "awaiting_approval");
            let audit = t.audit.unwrap_or_else(|| "—".into());
            let result_path = t.result_path.unwrap_or_else(|| "—".into());
            Either::Right(view! {
                <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm mb-6">
                    <div><dt class="text-muted-foreground text-xs">"Verb"</dt><dd class="font-mono text-xs">{t.verb}</dd></div>
                    <div><dt class="text-muted-foreground text-xs">"State"</dt><dd>{t.state}</dd></div>
                    <div><dt class="text-muted-foreground text-xs">"Tier"</dt><dd>{t.tier}</dd></div>
                    <div><dt class="text-muted-foreground text-xs">"Approval PR"</dt><dd>{pr}</dd></div>
                    <div class="col-span-2 sm:col-span-4"><dt class="text-muted-foreground text-xs">"Result path"</dt><dd class="font-mono text-xs break-all">{result_path}</dd></div>
                </dl>
                <h3 class="text-sm font-medium mb-2">"Audit"</h3>
                <pre class="text-xs whitespace-pre-wrap break-words rounded-md border bg-muted p-4">{audit}</pre>
            })
        }
    };
    let id_text = id.clone();
    let content = view! {
        <p class="text-sm mb-4"><a class="text-primary underline underline-offset-4" href="/selecta">"← selecta"</a></p>
        <Card>
            <CardHeader>
                <CardTitle>"Task "<span class="font-mono text-sm">{id_text}</span></CardTitle>
                <CardDescription>
                    "Live from selecta's " {GET_TASK} " (read-only mount). The audit is selecta's own scrubbed text."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    };
    Ok((jar, Html(page(TITLE, &session, flash, content))))
}

// ─── Rendering helpers ──────────────────────────────────────────────────────

/// `1h 02m`, `4m 05s`, `12s`.
fn fmt_secs(s: f64) -> String {
    let s = s.max(0.0).round() as u64;
    match (s / 3600, (s % 3600) / 60, s % 60) {
        (0, 0, sec) => format!("{sec}s"),
        (0, m, sec) => format!("{m}m {sec:02}s"),
        (h, m, _) => format!("{h}h {m:02}m"),
    }
}

/// A count for a cell; the torn-read sentinel renders as what it is.
fn count_or_torn(v: f64, fmt: impl Fn(f64) -> String) -> String {
    if v >= TORN_SENTINEL {
        "unreadable (torn ledger read)".into()
    } else {
        fmt(v)
    }
}

/// The approval-PR cell. Only a well-formed GitHub PR URL becomes a link —
/// a link, never a button: the merge on GitHub is the approval. A gated task
/// with no PR cannot be approved at all, so it is flagged as an error.
fn pr_cell(pr_url: Option<&str>, awaiting: bool) -> impl IntoView + use<> {
    match pr_url {
        Some(raw) => match github_pr(raw) {
            Some(href) => {
                let label = href
                    .rsplit('/')
                    .next()
                    .map(|n| format!("PR #{n}"))
                    .unwrap_or_default();
                Either::Left(view! {
                    <a class="text-primary underline underline-offset-4 whitespace-nowrap" href=href rel="noreferrer">{label}</a>
                })
            }
            None => {
                let text = raw.to_string();
                Either::Right(Either::Left(view! {
                    <span class=badge(BadgeKind::Destructive)>"unrecognised PR link"</span>
                    " "<span class="font-mono text-xs break-all">{text}</span>
                }))
            }
        },
        None if awaiting => Either::Right(Either::Right(Either::Left(view! {
            <span class=badge(BadgeKind::Destructive)>"no PR — cannot be approved"</span>
        }))),
        None => Either::Right(Either::Right(Either::Right(view! {
            <span class="text-muted-foreground">"—"</span>
        }))),
    }
}

/// `newest N of M` when the list was cut, from the all-time state count.
fn shown_of(shown: usize, gauges: &Result<Gauges, AppError>, state: &str) -> Option<String> {
    let total = gauges.as_ref().ok()?.queue_depth.get(state).copied()? as usize;
    (total > shown).then(|| format!("Showing the newest {shown} of {total}."))
}

fn task_link(id: &str) -> impl IntoView + use<> {
    let href = format!("/selecta/task/{id}");
    view! { <a class="text-primary underline underline-offset-4 whitespace-nowrap" href=href>"details"</a> }
}

// ─── Runner: heartbeat, lease, run budget, daily caps ───────────────────────

fn runner_card(
    live: &Result<Liveness, AppError>,
    gauges: &Result<Gauges, AppError>,
) -> impl IntoView + use<> {
    let heartbeat = match live {
        Err(e) => Either::Left(unavailable("heartbeat", e)),
        Ok(l) => {
            let seq = l.seq.map(|s| format!("seq {s}")).unwrap_or_default();
            let (class, text) = if l.advancing {
                (badge(BadgeKind::Success), "advancing".to_string())
            } else {
                (
                    badge(BadgeKind::Destructive),
                    format!(
                        "stalled — {}",
                        l.detail.clone().unwrap_or_else(|| "no reason given".into())
                    ),
                )
            };
            Either::Right(view! {
                <span class=class>{text}</span>" "<span class="text-muted-foreground tabular-nums">{seq}</span>
            })
        }
    };

    let body = match gauges {
        Err(e) => Either::Left(unavailable("lease and budget", e)),
        Ok(g) => {
            let (lease_class, lease_text) = if g.lease_remaining_s > 0.0 {
                (
                    badge(BadgeKind::Success),
                    format!("held, {} left", fmt_secs(g.lease_remaining_s)),
                )
            } else {
                (
                    badge(BadgeKind::Destructive),
                    "not held — irreversible work is refused".to_string(),
                )
            };
            let b = &g.budget;
            let (park_class, park_text) = if b.parked {
                (
                    badge(BadgeKind::Destructive),
                    format!("parked — {}", b.reason),
                )
            } else {
                (badge(BadgeKind::Success), "not parked".to_string())
            };
            let tokens = count_or_torn(b.tokens, fmt_tokens);
            let wallclock = count_or_torn(b.wallclock_s, fmt_secs);
            let toolcalls = count_or_torn(b.toolcalls, |v| format!("{v:.0}"));
            let caps = g
                .day_caps
                .iter()
                .map(|(cap, (used, limit))| {
                    let name = cap.clone();
                    let cell = match (used, limit) {
                        (Some(u), Some(l)) => {
                            let (u, l) = (*u, *l);
                            Either::Left(view! {
                                <div class="flex items-center gap-2 whitespace-nowrap">
                                    <progress class="h-2 w-24" value=format!("{u:.0}") max=format!("{l:.0}")></progress>
                                    <span class="tabular-nums">{format!("{u:.0} / {l:.0}")}</span>
                                </div>
                            })
                        }
                        // Half a pair is a selecta bug; show what exists.
                        (u, l) => {
                            let text = format!(
                                "{} / {}",
                                u.map_or("—".into(), |v| format!("{v:.0}")),
                                l.map_or("—".into(), |v| format!("{v:.0}"))
                            );
                            Either::Right(view! { <span class="tabular-nums">{text}</span> })
                        }
                    };
                    view! {
                        <TableRow>
                            <TableCell><span class="font-mono text-xs">{name}</span></TableCell>
                            <TableCell>{cell}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            let caps_body = if g.day_caps.is_empty() {
                Either::Left(view! {
                    <p class="text-sm text-muted-foreground">"No daily caps seeded yet."</p>
                })
            } else {
                Either::Right(view! {
                    <TableWrapper><Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Daily cap"</TableHead>
                                <TableHead>"Used today / limit"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>{caps}</TableBody>
                    </Table></TableWrapper>
                })
            };
            Either::Right(view! {
                <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm mb-6">
                    <div class="col-span-2">
                        <dt class="text-muted-foreground text-xs">"Lease"</dt>
                        <dd><span class=lease_class>{lease_text}</span></dd>
                    </div>
                    <div class="col-span-2">
                        <dt class="text-muted-foreground text-xs">"Run budget"</dt>
                        <dd><span class=park_class>{park_text}</span></dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Tokens consumed"</dt>
                        <dd class="tabular-nums">{tokens}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Wall clock consumed"</dt>
                        <dd class="tabular-nums">{wallclock}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Tool calls consumed"</dt>
                        <dd class="tabular-nums">{toolcalls}</dd>
                    </div>
                </dl>
                {caps_body}
            })
        }
    };

    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Runner"</CardTitle>
                <CardDescription>
                    "Heartbeat: selecta's own /healthz verdict. Lease, run budget and daily caps: selecta's /metrics. selecta exports what the run consumed but not its run caps; a cap shows in the parked reason once it is hit. Daily caps are consumed against their limits."
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div class="text-sm mb-4">
                    <span class="text-muted-foreground text-xs mr-2">"Heartbeat"</span>{heartbeat}
                </div>
                {body}
            </CardContent>
        </Card>
    }
}

// ─── Waiting on the operator: gated tasks and their approval PRs ────────────

fn approvals_card(
    awaiting: &Result<Vec<TaskRow>, AppError>,
    gauges: &Result<Gauges, AppError>,
) -> impl IntoView + use<> {
    let body = match awaiting {
        Err(e) => Either::Left(unavailable(LIST_TASKS, e)),
        Ok(rows) if rows.is_empty() => Either::Right(Either::Left(view! {
            <p class="text-sm text-muted-foreground">"Nothing is waiting on you."</p>
        })),
        Ok(rows) => {
            let no_pr = rows.iter().filter(|r| r.approval_pr().is_none()).count();
            let no_pr_note = (no_pr > 0).then(|| {
                let msg = format!(
                    " {no_pr} gated task(s) have no usable approval PR, so nothing can approve them."
                );
                view! {
                    <p class="text-sm mb-4">
                        <span class=badge(BadgeKind::Destructive)>"error"</span>{msg}
                    </p>
                }
            });
            let more = shown_of(rows.len(), gauges, "awaiting_approval")
                .map(|m| view! { <p class="text-sm text-muted-foreground mb-4">{m}</p> });
            let trs = rows
                .iter()
                .map(|r| {
                    let verb = r.verb.clone();
                    let tier = r.tier.clone();
                    let when = fmt_epoch(r.enqueued_at);
                    let pr = pr_cell(r.pr_url.as_deref(), true);
                    let link = task_link(&r.task_id);
                    view! {
                        <TableRow>
                            <TableCell><span class="font-mono text-xs">{verb}</span></TableCell>
                            <TableCell>{tier}</TableCell>
                            <TableCell><span class="whitespace-nowrap">{when}</span></TableCell>
                            <TableCell>{pr}</TableCell>
                            <TableCell>{link}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            Either::Right(Either::Right(view! {
                {no_pr_note}
                {more}
                <TableWrapper><Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>"Verb"</TableHead>
                            <TableHead>"Tier"</TableHead>
                            <TableHead>"Enqueued (UTC)"</TableHead>
                            <TableHead>"Approval"</TableHead>
                            <TableHead>""</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>{trs}</TableBody>
                </Table></TableWrapper>
            }))
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Waiting on you"</CardTitle>
                <CardDescription>
                    "Tasks in awaiting_approval, newest first (selecta_list_tasks). Approving one means merging its PR on GitHub — the console only links to it."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Recent failures ────────────────────────────────────────────────────────

fn failures_card(
    failed: &Result<Vec<TaskRow>, AppError>,
    gauges: &Result<Gauges, AppError>,
) -> impl IntoView + use<> {
    let body = match failed {
        Err(e) => Either::Left(unavailable(LIST_TASKS, e)),
        Ok(rows) if rows.is_empty() => Either::Right(Either::Left(view! {
            <p class="text-sm text-muted-foreground">"No failed tasks."</p>
        })),
        Ok(rows) => {
            let more = shown_of(rows.len(), gauges, "failed")
                .map(|m| view! { <p class="text-sm text-muted-foreground mb-4">{m}</p> });
            let trs = rows
                .iter()
                .map(|r| {
                    let verb = r.verb.clone();
                    let when = fmt_epoch(r.enqueued_at);
                    let audit = r.audit.clone().unwrap_or_else(|| "—".into());
                    let link = task_link(&r.task_id);
                    view! {
                        <TableRow>
                            <TableCell><span class="font-mono text-xs">{verb}</span></TableCell>
                            <TableCell><span class="whitespace-nowrap">{when}</span></TableCell>
                            <TableCell><span class="text-xs break-words">{audit}</span></TableCell>
                            <TableCell>{link}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            Either::Right(Either::Right(view! {
                {more}
                <TableWrapper><Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>"Verb"</TableHead>
                            <TableHead>"Enqueued (UTC)"</TableHead>
                            <TableHead>"Audit (excerpt)"</TableHead>
                            <TableHead>""</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>{trs}</TableBody>
                </Table></TableWrapper>
            }))
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Recent failures"</CardTitle>
                <CardDescription>
                    "The newest failed tasks (selecta_list_tasks) with the first 200 characters of each audit; details shows the whole audit."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Task counts per state ──────────────────────────────────────────────────

fn states_card(gauges: &Result<Gauges, AppError>) -> impl IntoView + use<> {
    let body = match gauges {
        Err(e) => Either::Left(unavailable("task counts", e)),
        Ok(g) => {
            // selecta omits zero-count states: every state renders, absent = 0.
            let tiles = STATES
                .iter()
                .map(|s| {
                    let n = g.queue_depth.get(*s).copied().unwrap_or(0.0);
                    let class = match *s {
                        "failed" | "aborted_stale" if n > 0.0 => "text-destructive",
                        "awaiting_approval" if n > 0.0 => "text-warning-dark",
                        _ => "",
                    };
                    let (label, count) = (s.to_string(), format!("{n:.0}"));
                    view! {
                        <div>
                            <dt class="text-muted-foreground text-xs font-mono">{label}</dt>
                            <dd class=format!("tabular-nums text-lg {class}")>{count}</dd>
                        </div>
                    }
                })
                .collect_view();
            Either::Right(view! {
                <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">{tiles}</dl>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Tasks by state"</CardTitle>
                <CardDescription>
                    "queue_depth from selecta's /metrics. All time: selecta never deletes a queue row, so done and failed only grow."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Download health ────────────────────────────────────────────────────────

fn health_card(health: &Result<HealthSnapshot, AppError>) -> impl IntoView + use<> {
    let body = match health {
        Err(e) => Either::Left(unavailable(DOWNLOAD_HEALTH, e)),
        Ok(h) => {
            let alerts = if h.alerts.is_empty() {
                Either::Left(
                    view! { <p class="text-sm text-muted-foreground mb-4">"No alerts."</p> },
                )
            } else {
                let items = h
                    .alerts
                    .iter()
                    .map(|a| {
                        let text = a.clone();
                        view! {
                            <li><span class=badge(BadgeKind::Warning)>"alert"</span>" "{text}</li>
                        }
                    })
                    .collect_view();
                Either::Right(view! { <ul class="text-sm space-y-1 mb-4">{items}</ul> })
            };
            // serde_json maps → BTreeMap: already in key order.
            let tiles = h
                .signals
                .iter()
                .map(|(k, v)| {
                    let (name, value) = (k.clone(), format!("{v}"));
                    view! {
                        <div>
                            <dt class="text-muted-foreground text-xs font-mono">{name}</dt>
                            <dd class="tabular-nums">{value}</dd>
                        </div>
                    }
                })
                .collect_view();
            Either::Right(view! {
                {alerts}
                <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">{tiles}</dl>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Download health"</CardTitle>
                <CardDescription>
                    "selecta_download_health: selecta's signals per download client, and its alerts. A client selecta could not reach has no signals and an alert naming it."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}
