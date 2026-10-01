//! Ālaya health pane (LAB-6881) — `/alaya/health`.
//!
//! GET only, plain tables, no JS. Renders alaya-server's `GET /stats`: the
//! contradiction judge's verdict mix, stored failures, backlog, daily-cap
//! usage and the corpus by edge type. A section whose source is down shows
//! an "unavailable" banner, never zeros, so an outage cannot pass for an
//! empty corpus.

use axum::extract::State;
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde_json::Value;

use crate::error::AppError;
use crate::routes::{fmt_epoch, vf, vs};
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

const TITLE: &str = "Ālaya health — ops console";

/// Verdict rows in display order: `(key in by_verdict, label, the
/// contradictions page's `verdict` filter)`. A stored failure and a pair
/// never judged share the `unjudged` filter, which matches both; an
/// unrecognised verdict has no filter that selects it.
const VERDICT_ROWS: [(&str, &str, Option<&str>); 7] = [
    ("contradiction", "contradiction", Some("contradiction")),
    ("supersession", "supersession", Some("supersession")),
    ("coexist", "coexist", Some("coexist")),
    ("unrelated", "unrelated", Some("unrelated")),
    ("unjudged", "stored judge failure", Some("unjudged")),
    ("never_judged", "never judged (backlog)", Some("unjudged")),
    ("unrecognised", "unrecognised verdict", None),
];

/// Columns of the per-day table, in the server's key names.
const DAY_COLUMNS: [&str; 6] = [
    "contradiction",
    "supersession",
    "coexist",
    "unrelated",
    "unjudged",
    "unrecognised",
];

/// The four classes the judge answers with — the degenerate-reason rows.
const JUDGE_CLASSES: [&str; 4] = ["contradiction", "supersession", "coexist", "unrelated"];

pub async fn pane(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let content = match state.alaya.stats().await {
        Err(e) => Either::Left(view! {
            <Card>
                <CardHeader>
                    <CardTitle>"Ālaya health"</CardTitle>
                </CardHeader>
                <CardContent>{unavailable("alaya-server GET /stats", e.detail())}</CardContent>
            </Card>
        }),
        Ok(s) => Either::Right(view! {
            <div class="space-y-6">
                {summary_card(&s)}
                {verdicts_card(&s)}
                {failures_card(&s)}
                {per_day_card(&s)}
                {degenerate_card(&s)}
                {corpus_card(&s)}
            </div>
        }),
    };
    Ok((jar, Html(page(TITLE, &session, flash, content))))
}

// ─── Rendering helpers ──────────────────────────────────────────────────────

/// A count cell, or a dash for a missing or mistyped field — never a
/// fictitious zero.
fn count(v: Option<&Value>) -> String {
    v.and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}

fn unavailable(what: &str, detail: &str) -> impl IntoView + use<> {
    let msg = format!("{what}: {detail}");
    view! {
        <p class="text-sm" role="alert">
            <span class=badge(BadgeKind::Destructive)>"unavailable"</span>
            " "{msg}
        </p>
    }
}

/// `section` when the server sent it, else the banner. The server nulls a
/// section whose source failed and says why in `errors`.
fn section<'a>(s: &'a Value, key: &str) -> Option<&'a Value> {
    s.get(key).filter(|v| v.is_object())
}

fn link(href: String, text: String) -> impl IntoView + use<> {
    view! { <a class="text-primary underline-offset-4 hover:underline" href=href>{text}</a> }
}

// ─── Summary: freshness, errors, judge daily cap ────────────────────────────

fn summary_card(s: &Value) -> impl IntoView + use<> {
    let generated = fmt_epoch(vf(s, "generated_at"));
    let errors: Vec<String> = s
        .get("errors")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let cap = s.get("judge_daily_cap");
    let cap_value = match cap.and_then(|c| c.get("cap")) {
        Some(Value::Null) => "no judge configured".to_string(),
        v => count(v),
    };
    let admitted = count(cap.and_then(|c| c.get("admitted_today")));
    let day = cap.map(|c| vs(c, "utc_day")).unwrap_or_default();
    let memories = count(section(s, "memories").and_then(|m| m.get("total")));
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Ālaya health"</CardTitle>
                <CardDescription>{format!("Generated {generated} UTC")}</CardDescription>
            </CardHeader>
            <CardContent>
                <div class="space-y-2 mb-4">
                    {errors
                        .into_iter()
                        .map(|e| view! {
                            <p class="text-sm" role="alert">
                                <span class=badge(BadgeKind::Destructive)>"unavailable"</span>
                                " "{e}
                            </p>
                        })
                        .collect_view()}
                </div>
                <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
                    <div>
                        <dt class="text-muted-foreground text-xs">"Memories (vector store)"</dt>
                        <dd>{memories}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Judge daily cap"</dt>
                        <dd>{cap_value}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"Admitted today"</dt>
                        <dd>{admitted}</dd>
                    </div>
                    <div>
                        <dt class="text-muted-foreground text-xs">"UTC day"</dt>
                        <dd>{day}</dd>
                    </div>
                </dl>
                <p class="text-xs text-muted-foreground mt-3">
                    "The daily cap bounds store-path judge calls per alaya-server process; with several replicas this is the one that answered. Operator backfill is not counted."
                </p>
            </CardContent>
        </Card>
    }
}

// ─── Verdict mix ────────────────────────────────────────────────────────────

fn verdicts_card(s: &Value) -> impl IntoView + use<> {
    let c = section(s, "contradictions");
    let body = match c.and_then(|c| c.get("by_verdict")) {
        None => Either::Left(unavailable("contradiction stats", "see the errors above")),
        Some(by) => {
            let rows = VERDICT_ROWS
                .iter()
                .map(|(key, label, filter)| {
                    let row = by.get(*key);
                    let open = row.and_then(|r| r.get("open"));
                    let resolved = row.and_then(|r| r.get("resolved"));
                    let total = match (
                        open.and_then(Value::as_u64),
                        resolved.and_then(Value::as_u64),
                    ) {
                        (Some(o), Some(r)) => (o + r).to_string(),
                        _ => "—".into(),
                    };
                    let cell = |n: String, resolved: bool| match filter {
                        Some(f) if n != "—" => {
                            let href = if resolved {
                                format!("/alaya/contradictions?verdict={f}&resolved=1")
                            } else {
                                format!("/alaya/contradictions?verdict={f}")
                            };
                            Either::Left(link(href, n))
                        }
                        _ => Either::Right(n),
                    };
                    let (label, open, resolved) = (
                        label.to_string(),
                        cell(count(open), false),
                        cell(count(resolved), true),
                    );
                    view! {
                        <TableRow>
                            <TableCell>{label}</TableCell>
                            <TableCell>{open}</TableCell>
                            <TableCell>{resolved}</TableCell>
                            <TableCell>{total}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            Either::Right(view! {
                <TableWrapper>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Verdict"</TableHead>
                                <TableHead>"Open"</TableHead>
                                <TableHead>"Resolved"</TableHead>
                                <TableHead>"Total"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>{rows}</TableBody>
                    </Table>
                </TableWrapper>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Judge verdicts"</CardTitle>
                <CardDescription>
                    "Every CONTRADICTS edge by its judge verdict. Resolved means an endpoint was superseded or the pair was stamped keep both, in either direction. Never judged is the backlog waiting for a backfill; it and stored failures share the unjudged filter."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Stored judge failures ──────────────────────────────────────────────────

fn failures_card(s: &Value) -> impl IntoView + use<> {
    let f = section(s, "contradictions").and_then(|c| c.get("failures"));
    let body = match f {
        None => Either::Left(unavailable("contradiction stats", "see the errors above")),
        Some(f) => {
            let top = f
                .get("top")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let rows = top
                .iter()
                .map(|r| {
                    let (reason, n) = (vs(r, "reason"), count(r.get("count")));
                    view! {
                        <TableRow>
                            <TableCell><span class="font-mono text-xs">{reason}</span></TableCell>
                            <TableCell>{n}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            let other = count(f.get("other"));
            let total = count(f.get("total"));
            Either::Right(view! {
                <TableWrapper>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Reason"</TableHead>
                                <TableHead>"Pairs"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {rows}
                            <TableRow>
                                <TableCell>"other"</TableCell>
                                <TableCell>{other}</TableCell>
                            </TableRow>
                            <TableRow>
                                <TableCell><span class="font-medium">"total"</span></TableCell>
                                <TableCell><span class="font-medium">{total}</span></TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </TableWrapper>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Stored judge failures"</CardTitle>
                <CardDescription>
                    "Deterministic failures persisted as verdict unjudged, by full reason, ten most frequent. Transient failures (rate limits, timeouts) are retried and stored nowhere, so they are not here."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Judged per day ─────────────────────────────────────────────────────────

fn per_day_card(s: &Value) -> impl IntoView + use<> {
    let days = section(s, "contradictions").and_then(|c| c.get("judged_per_day"));
    let body = match days.and_then(Value::as_array) {
        None => Either::Left(unavailable("contradiction stats", "see the errors above")),
        Some(days) => {
            let rows = days
                .iter()
                .map(|d| {
                    let counts = d.get("counts");
                    let cells = DAY_COLUMNS
                        .iter()
                        .map(|k| {
                            let n = count(counts.and_then(|c| c.get(*k)));
                            view! { <TableCell>{n}</TableCell> }
                        })
                        .collect_view();
                    let date = vs(d, "date");
                    view! {
                        <TableRow>
                            <TableCell>{date}</TableCell>
                            {cells}
                        </TableRow>
                    }
                })
                .collect_view();
            let heads = DAY_COLUMNS
                .iter()
                .map(|k| {
                    let k = k.to_string();
                    view! { <TableHead>{k}</TableHead> }
                })
                .collect_view();
            Either::Right(view! {
                <TableWrapper>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"UTC day"</TableHead>
                                {heads}
                            </TableRow>
                        </TableHeader>
                        <TableBody>{rows}</TableBody>
                    </Table>
                </TableWrapper>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Pairs judged per day"</CardTitle>
                <CardDescription>
                    "Last 14 UTC days by the verdict each edge carries now, from judged_at. A re-judged edge counts once, on its latest day."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Degenerate reasons ─────────────────────────────────────────────────────

fn degenerate_card(s: &Value) -> impl IntoView + use<> {
    let c = section(s, "contradictions");
    let body = match c.and_then(|c| c.get("degenerate_reasons")) {
        None => Either::Left(unavailable("contradiction stats", "see the errors above")),
        Some(d) => {
            let by = c.and_then(|c| c.get("by_verdict"));
            let rows = JUDGE_CLASSES
                .iter()
                .map(|k| {
                    let row = by.and_then(|b| b.get(*k));
                    let judged = match (
                        row.and_then(|r| r.get("open")).and_then(Value::as_u64),
                        row.and_then(|r| r.get("resolved")).and_then(Value::as_u64),
                    ) {
                        (Some(o), Some(r)) => (o + r).to_string(),
                        _ => "—".into(),
                    };
                    let (verdict, n) = (k.to_string(), count(d.get(*k)));
                    view! {
                        <TableRow>
                            <TableCell>{verdict}</TableCell>
                            <TableCell>{n}</TableCell>
                            <TableCell>{judged}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            Either::Right(view! {
                <TableWrapper>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Verdict"</TableHead>
                                <TableHead>"Degenerate reasons"</TableHead>
                                <TableHead>"Judged"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>{rows}</TableBody>
                    </Table>
                </TableWrapper>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Degenerate judge reasons"</CardTitle>
                <CardDescription>
                    "Judged edges whose trimmed reason is under 10 characters or reads placeholder. Display only: nothing acts on this count."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Corpus ─────────────────────────────────────────────────────────────────

fn corpus_card(s: &Value) -> impl IntoView + use<> {
    let body = match section(s, "graph") {
        None => Either::Left(unavailable("graph stats", "see the errors above")),
        Some(g) => {
            let edges = g
                .get("edge_counts")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            // serde_json maps iterate in key order — no sort needed.
            let rows = edges
                .iter()
                .map(|(k, v)| {
                    let (label, n) = (k.clone(), count(Some(v)));
                    view! {
                        <TableRow>
                            <TableCell><span class="font-mono text-xs">{label}</span></TableCell>
                            <TableCell>{n}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            let nodes = count(g.get("node_count"));
            Either::Right(view! {
                <TableWrapper>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Graph"</TableHead>
                                <TableHead>"Count"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow>
                                <TableCell>"nodes"</TableCell>
                                <TableCell>{nodes}</TableCell>
                            </TableRow>
                            {rows}
                        </TableBody>
                    </Table>
                </TableWrapper>
            })
        }
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Corpus"</CardTitle>
                <CardDescription>"Graph nodes and edges by type."</CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}
