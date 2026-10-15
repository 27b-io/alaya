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
use crate::routes::fmt_epoch;
use crate::selecta::{
    DOWNLOAD_HEALTH, GET_TASK, Gauges, HealthSnapshot, LIST_TASKS, Liveness, STATES, TORN_SENTINEL,
    TaskRow, github_pr, valid_task_id,
};
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

const TITLE: &str = "selecta — ops console";
/// The states a task ends in. selecta keeps every row, so their counts are
/// all time.
const TERMINAL: [&str; 3] = ["done", "aborted_stale", "failed"];
const FAILED_SHOWN: u32 = 20;
/// Also the home card's probe size: it says "N+" when a list fills it.
pub const AWAITING_SHOWN: u32 = 100;

fn header() -> impl IntoView {
    view! {
        <PageHeader
            title="selecta"
            description="What selecta is doing: whether it is alive, what waits on your approval, what failed and why, and download health. Read-only: approvals happen on GitHub."
        />
    }
}

fn not_configured() -> impl IntoView + use<> {
    view! {
        <div class="space-y-6">
            {header()}
            <NotConfigured>
                "Set SELECTA_URL and SELECTA_API_KEY (a selecta read-only token) on the console deployment, both or neither, to enable the read-only selecta pane."
            </NotConfigured>
        </div>
    }
}

pub async fn pane(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(selecta) = state.selecta.as_ref() else {
        return Ok((
            jar,
            Html(page(TITLE, Nav::Selecta, &session, flash, not_configured())),
        ));
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
            {header()}
            {runner_card(&live, &gauges)}
            {approvals_card(&awaiting, &gauges)}
            {failures_card(&failed, &gauges)}
            {states_card(&gauges)}
            {health_card(&health)}
        </div>
    };
    Ok((
        jar,
        Html(page(TITLE, Nav::Selecta, &session, flash, content)),
    ))
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
        return Ok((
            jar,
            Html(page(TITLE, Nav::Selecta, &session, flash, not_configured())),
        ));
    };
    let body = match selecta.get_task(&id).await {
        Err(e) => Either::Left(unavailable("task detail", e.detail())),
        Ok(t) => {
            let pr = pr_cell(t.pr_url.as_deref(), t.state == "awaiting_approval");
            let audit = t.audit.unwrap_or_else(|| "—".into());
            let result_path = t.result_path.unwrap_or_else(|| "—".into());
            Either::Right(view! {
                <StatGrid>
                    <Stat label="Verb" text=true><span class="font-mono">{t.verb}</span></Stat>
                    <Stat label="State">{t.state}</Stat>
                    <Stat label="Tier">{t.tier}</Stat>
                    <Stat label="Approval PR" text=true>{pr}</Stat>
                </StatGrid>
                <h3 class="text-sm font-medium mt-6 mb-2">"Result path"</h3>
                <p class="font-mono text-xs break-all">{result_path}</p>
                <h3 class="text-sm font-medium mt-6 mb-2">"Audit"</h3>
                <pre class="text-xs whitespace-pre-wrap break-words rounded-md border bg-muted p-4">{audit}</pre>
            })
        }
    };
    let content = view! {
        <div class="space-y-6">
            <p class="text-sm"><Link href="/selecta">"← selecta"</Link></p>
            <PageHeader
                title=format!("Task {id}")
                description=format!("Live from selecta's {GET_TASK} (read-only mount). The audit is selecta's own scrubbed text.")
            />
            <Card>
                <CardContent>{body}</CardContent>
            </Card>
        </div>
    };
    Ok((
        jar,
        Html(page(TITLE, Nav::Selecta, &session, flash, content)),
    ))
}

// ─── Rendering helpers ──────────────────────────────────────────────────────

/// `1h 02m`, `4m 05s`, `12s`; a dash for a value that is not a number.
fn fmt_secs(s: f64) -> String {
    if !s.is_finite() {
        return "—".into();
    }
    let s = s.max(0.0).round() as u64;
    match (s / 3600, (s % 3600) / 60, s % 60) {
        (0, 0, sec) => format!("{sec}s"),
        (0, m, sec) => format!("{m}m {sec:02}s"),
        (h, m, _) => format!("{h}h {m:02}m"),
    }
}

/// A count for a cell; the torn-read sentinel renders as what it is.
fn count_or_torn(v: f64, fmt: impl Fn(f64) -> String) -> String {
    // An infinity is not the sentinel: `fmt_count` and its kin print it as a dash.
    if v.is_finite() && v >= TORN_SENTINEL {
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
                    <span class="whitespace-nowrap">
                        <Link href=href external=true>{label}</Link>
                    </span>
                })
            }
            None => {
                let text = raw.to_string();
                Either::Right(Either::Left(view! {
                    <StatusBadge status=Status::Error label="unrecognised PR link" />
                    " "<span class="font-mono text-xs break-all">{text}</span>
                }))
            }
        },
        None if awaiting => Either::Right(Either::Right(Either::Left(view! {
            <StatusBadge status=Status::Error label="no PR — cannot be approved" />
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
    view! { <Link href=href>"details"</Link> }
}

// ─── Runner: heartbeat, lease, run budget, daily caps ───────────────────────

fn runner_card(
    live: &Result<Liveness, AppError>,
    gauges: &Result<Gauges, AppError>,
) -> impl IntoView + use<> {
    let heartbeat = match live {
        Err(e) => Either::Left(unavailable("heartbeat", e.detail())),
        Ok(l) => {
            let seq = l.seq.map(|s| format!(", seq {s}")).unwrap_or_default();
            let (status, text) = if l.advancing {
                (Status::Ok, "advancing".to_string())
            } else {
                (
                    Status::Error,
                    format!(
                        "stalled — {}",
                        l.detail.clone().unwrap_or_else(|| "no reason given".into())
                    ),
                )
            };
            Either::Right(view! {
                <Notice status=status label=text>{format!("Heartbeat{seq}.")}</Notice>
            })
        }
    };

    let body = match gauges {
        Err(e) => Either::Left(unavailable("lease and budget", e.detail())),
        Ok(g) => {
            let (lease, lease_text) = if !g.lease_remaining_s.is_finite() {
                (Status::Off, "unknown".to_string())
            } else if g.lease_remaining_s > 0.0 {
                (
                    Status::Ok,
                    format!("held, {} left", fmt_secs(g.lease_remaining_s)),
                )
            } else {
                (
                    Status::Error,
                    "not held — irreversible work is refused".to_string(),
                )
            };
            let b = &g.budget;
            let (park, park_text) = if b.parked {
                (Status::Error, format!("parked — {}", b.reason))
            } else {
                (Status::Ok, "not parked".to_string())
            };
            let tokens = count_or_torn(b.tokens, fmt_tokens);
            let wallclock = count_or_torn(b.wallclock_s, fmt_secs);
            let toolcalls = count_or_torn(b.toolcalls, fmt_count);
            let caps = g
                .day_caps
                .iter()
                .map(|(cap, (used, limit))| {
                    let name = cap.clone();
                    // Half a pair is a selecta bug: the meter shows what
                    // exists and draws no share.
                    let label = format!("{cap}: used today");
                    let cell = view! {
                        <div class="min-w-24">
                            <Meter label=label used=*used limit=*limit fmt=fmt_count />
                        </div>
                    };
                    view! {
                        <TableRow>
                            <TableCell><span class="font-mono text-xs wrap-break-word">{name}</span></TableCell>
                            <TableCell>{cell}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            let caps_body = if g.day_caps.is_empty() {
                Either::Left(view! { <EmptyState>"No daily caps seeded yet."</EmptyState> })
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
                <div class="mb-6">
                    <StatGrid>
                        <Stat label="Lease"><StatusBadge status=lease label=lease_text /></Stat>
                        <Stat label="Run budget"><StatusBadge status=park label=park_text /></Stat>
                        <Stat label="Tokens consumed">{tokens}</Stat>
                        <Stat label="Wall clock consumed">{wallclock}</Stat>
                        <Stat label="Tool calls consumed">{toolcalls}</Stat>
                    </StatGrid>
                </div>
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
                <div class="mb-4">{heartbeat}</div>
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
        Err(e) => Either::Left(unavailable(LIST_TASKS, e.detail())),
        Ok(rows) if rows.is_empty() => Either::Right(Either::Left(view! {
            <EmptyState>"Nothing is waiting on you."</EmptyState>
        })),
        Ok(rows) => {
            let no_pr = rows.iter().filter(|r| r.approval_pr().is_none()).count();
            let no_pr_note = (no_pr > 0).then(|| {
                let msg = format!(
                    "{no_pr} gated task(s) have no usable approval PR, so nothing can approve them."
                );
                view! {
                    <div class="mb-4">
                        <Notice status=Status::Error label="error">{msg}</Notice>
                    </div>
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
                            <TableCell><span class="font-mono text-xs wrap-break-word">{verb}</span></TableCell>
                            <TableCell secondary=true>{tier}</TableCell>
                            <TableCell secondary=true><span class="whitespace-nowrap">{when}</span></TableCell>
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
                            <TableHead secondary=true>"Tier"</TableHead>
                            <TableHead secondary=true>"Enqueued (UTC)"</TableHead>
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
        Err(e) => Either::Left(unavailable(LIST_TASKS, e.detail())),
        Ok(rows) if rows.is_empty() => Either::Right(Either::Left(view! {
            <EmptyState>"No failed tasks."</EmptyState>
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
                            <TableCell><span class="font-mono text-xs wrap-break-word">{verb}</span></TableCell>
                            <TableCell secondary=true><span class="whitespace-nowrap">{when}</span></TableCell>
                            <TableCell><span class="text-xs wrap-break-word">{audit}</span></TableCell>
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
                            <TableHead secondary=true>"Enqueued (UTC)"</TableHead>
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
        Err(e) => Either::Left(unavailable("task counts", e.detail())),
        Ok(g) => {
            // A state this console does not know gets a tile too, under an
            // error: hidden, it would read as 0.
            let unknown: Vec<&str> = g
                .queue_depth
                .keys()
                .map(String::as_str)
                .filter(|s| !STATES.contains(s))
                .collect();
            let unknown_note = (!unknown.is_empty()).then(|| {
                let msg = format!(
                    "selecta reports queue states this console does not know: {}.",
                    unknown.join(", ")
                );
                view! {
                    <div class="mb-4">
                        <Notice status=Status::Error label="error">{msg}</Notice>
                    </div>
                }
            });
            // selecta omits zero-count states: every state renders, absent = 0.
            let depth = |s: &str| g.queue_depth.get(s).copied().unwrap_or(0.0);
            // The live states only: done, aborted_stale and failed count all
            // time and would leave nothing else visible.
            let live = STATES
                .iter()
                .filter(|s| !TERMINAL.contains(s))
                .enumerate()
                .map(|(i, s)| Part::new(*s, Some(depth(s)), Colour::SERIES[i]))
                .collect::<Vec<_>>();
            let tiles = STATES
                .iter()
                .chain(&unknown)
                .map(|s| {
                    let n = depth(s);
                    let class = match *s {
                        "failed" | "aborted_stale" if n > 0.0 => "text-destructive",
                        "awaiting_approval" if n > 0.0 => "text-warning-dark",
                        _ => "",
                    };
                    let (label, n) = (s.to_string(), fmt_count(n));
                    view! { <Stat label=label><span class=class>{n}</span></Stat> }
                })
                .collect_view();
            Either::Right(view! {
                {unknown_note}
                <div class="mb-6">
                    <h3 class="text-sm font-medium mb-2">"In flight"</h3>
                    <StackedBar label="Tasks in flight by state" parts=live />
                </div>
                <StatGrid>{tiles}</StatGrid>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Tasks by state"</CardTitle>
                <CardDescription>
                    "queue_depth from selecta's /metrics. The bar holds the tasks still in flight; the tiles count all time, since selecta never deletes a queue row, so done, aborted_stale and failed only grow."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Download health ────────────────────────────────────────────────────────

/// A health signal as selecta reports it (`812.5`, `1`), or a dash for a
/// value that is not a number. The signals mix units (gigabytes, queue
/// lengths, up flags), so they stay tiles: no one axis would hold them.
fn signal(v: f64) -> String {
    if v.is_finite() {
        format!("{v}")
    } else {
        "—".into()
    }
}

fn health_card(health: &Result<HealthSnapshot, AppError>) -> impl IntoView + use<> {
    let body = match health {
        Err(e) => Either::Left(unavailable(DOWNLOAD_HEALTH, e.detail())),
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
                        view! { <Notice status=Status::Warn label="alert">{text}</Notice> }
                    })
                    .collect_view();
                Either::Right(view! { <div class="space-y-2 mb-4">{items}</div> })
            };
            // serde_json maps → BTreeMap: already in key order.
            let tiles = h
                .signals
                .iter()
                .map(|(k, v)| {
                    let (name, value) = (k.clone(), signal(*v));
                    view! { <Stat label=name>{value}</Stat> }
                })
                .collect_view();
            Either::Right(view! {
                {alerts}
                <StatGrid>{tiles}</StatGrid>
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
