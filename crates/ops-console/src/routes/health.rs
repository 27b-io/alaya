//! Ālaya health pane (LAB-6881) — `/alaya/health`.
//!
//! GET only, plain tables, no JS. Renders alaya-server's `GET /stats`: the
//! answering pod's self-check and failure counters (LAB-4026), the
//! contradiction judge's verdict mix, stored failures, backlog, daily-cap
//! usage, the novelty of each day's writes and the corpus by edge type. A
//! section whose source is down shows an "unavailable" notice, never zeros,
//! so an outage cannot pass for an empty corpus; a failing self-check is an
//! alert too.

use axum::extract::State;
use axum::response::Html;
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde_json::Value;

use crate::error::AppError;
use crate::routes::{fmt_epoch, vs};
use crate::session::{Session, take_flash};
use crate::state::AppState;
use crate::ui::*;

const TITLE: &str = "Judge health — ops console";

/// Verdict rows in display order: `(key in by_verdict, label, the
/// contradictions page's `verdict` filter, colour)`. A stored failure and a
/// pair never judged share the `unjudged` filter, which matches both. Each
/// verdict keeps its series slot in every chart on the page.
const VERDICT_ROWS: [(&str, &str, &str, Colour); 6] = [
    (
        "contradiction",
        "contradiction",
        "contradiction",
        Colour::SERIES[0],
    ),
    (
        "supersession",
        "supersession",
        "supersession",
        Colour::SERIES[1],
    ),
    ("coexist", "coexist", "coexist", Colour::SERIES[2]),
    ("unrelated", "unrelated", "unrelated", Colour::SERIES[3]),
    (
        "unjudged",
        "stored judge failure",
        "unjudged",
        Colour::SERIES[4],
    ),
    (
        "never_judged",
        "never judged (backlog)",
        "unjudged",
        Colour::SERIES[5],
    ),
];

/// Columns of the judged-per-day chart and table: `(the server's key in
/// counts, heading, colour)`, the colours `VERDICT_ROWS`' own. The home
/// card's daily total needs every one of these keys.
pub(crate) const DAY_COLUMNS: [(&str, &str, Colour); 5] = [
    ("contradiction", "contradiction", Colour::SERIES[0]),
    ("supersession", "supersession", Colour::SERIES[1]),
    ("coexist", "coexist", Colour::SERIES[2]),
    ("unrelated", "unrelated", Colour::SERIES[3]),
    ("unjudged", "unjudged", Colour::SERIES[4]),
];

/// Columns of the write-novelty chart and table, by nearest-neighbour
/// similarity. The bands are ordered, so they take steps of one hue, the
/// closest the darkest; "none" is no band at all.
const NOVELTY_COLUMNS: [(&str, &str, Colour); 5] = [
    ("0.95+", "≥ 0.95", Colour::RAMP[4]),
    ("0.85-0.95", "0.85–0.95", Colour::RAMP[3]),
    ("0.70-0.85", "0.70–0.85", Colour::RAMP[2]),
    ("below_0.70", "< 0.70", Colour::RAMP[1]),
    ("none", "none", Colour::OTHER),
];

/// The four classes the judge answers with — the degenerate-reason rows.
const JUDGE_CLASSES: [&str; 4] = ["contradiction", "supersession", "coexist", "unrelated"];

pub async fn pane(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let body = match state.alaya.stats().await {
        Err(e) => Either::Left(unavailable("alaya-server GET /stats", e.detail())),
        Ok(s) => Either::Right(view! {
            {errors(&s)}
            {summary(&s)}
            {pod_card(&s)}
            {verdicts_card(&s)}
            {failures_card(&s)}
            {per_day_card(&s)}
            {novelty_card(&s)}
            {degenerate_card(&s)}
            {corpus_card(&s)}
        }),
    };
    let content = view! {
        <div class="space-y-6">
            <PageHeader
                title="Judge health"
                description="alaya-server's own figures: the contradiction judge's verdicts, failures, backlog and daily cap, the novelty of each day's writes, the corpus by edge type, and the answering pod's self-check."
            />
            {body}
        </div>
    };
    Ok((
        jar,
        Html(page(TITLE, Nav::Health, &session, flash, content)),
    ))
}

// ─── Rendering helpers ──────────────────────────────────────────────────────

/// A count cell, or a dash for a missing or mistyped field — never a
/// fictitious zero.
fn count(v: Option<&Value>) -> String {
    v.and_then(Value::as_u64)
        .map(|n| n.to_string())
        .unwrap_or_else(|| "—".into())
}

/// A count as a chart value: `None`, which a chart draws as no data, for a
/// missing or mistyped field — never a fictitious zero.
fn num(v: Option<&Value>) -> Option<f64> {
    v.and_then(Value::as_u64).map(|n| n as f64)
}

/// The exact figures under a chart, folded away: operators read numbers.
fn folded(table: impl IntoView + 'static) -> impl IntoView {
    view! {
        <details class="mt-4">
            <summary class="cursor-pointer text-sm text-muted-foreground">"Table"</summary>
            <div class="mt-3">{table}</div>
        </details>
    }
}

/// `section` when the server sent it, else `None` and the card says it is
/// unavailable. The server nulls a section whose source failed and says why
/// in `errors`.
fn section<'a>(s: &'a Value, key: &str) -> Option<&'a Value> {
    s.get(key).filter(|v| v.is_object())
}

/// The server's own account of each section it could not read, one notice
/// each, above everything else on the page.
fn errors(s: &Value) -> impl IntoView + use<> {
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
    (!errors.is_empty()).then(|| {
        let notices = errors
            .into_iter()
            .map(|e| view! { <Notice status=Status::Error label="unavailable">{e}</Notice> })
            .collect_view();
        view! { <div class="space-y-2">{notices}</div> }
    })
}

// ─── Pod: self-check and failure counters (LAB-4026) ─────────────────────────

/// Failure counters in display order: `(key in pod.failures, label)`.
const FAILURE_ROWS: [(&str, &str); 3] = [
    ("embedding", "embedding calls"),
    ("rerank", "rerank fallbacks to RRF order"),
    ("store", "stores"),
];

fn pod_card(s: &Value) -> impl IntoView + use<> {
    let Some(pod) = section(s, "pod") else {
        return Either::Left(view! {
            <Card>
                <CardHeader>
                    <CardTitle>"Self-check"</CardTitle>
                </CardHeader>
                <CardContent>
                    {unavailable("self-check and failure counters", "this alaya-server does not report them")}
                </CardContent>
            </Card>
        });
    };
    let sc = pod.get("selfcheck");
    let enabled = sc.and_then(|c| c.get("enabled")).and_then(Value::as_bool);
    let last = sc.and_then(|c| c.get("last")).filter(|l| l.is_object());
    let streak = count(sc.and_then(|c| c.get("consecutive_failures")));
    let state = match (enabled, last) {
        (Some(false), _) => Either::Left(view! {
            <Notice status=Status::Off label="disabled">
                "SELFCHECK_QUERY and SELFCHECK_EXPECT_HASH are unset on this pod."
            </Notice>
        }),
        (_, None) => Either::Left(view! {
            <Notice status=Status::Off label="not run yet">
                "The first check runs at startup."
            </Notice>
        }),
        (_, Some(l)) if l.get("ok").and_then(Value::as_bool) == Some(true) => {
            let msg = format!(
                "Last run {} UTC; rerank: {}.",
                fmt_epoch(l["at"].as_f64().unwrap_or(0.0)),
                vs(l, "rerank"),
            );
            Either::Right(Either::Left(view! {
                <Notice status=Status::Ok label="passing">{msg}</Notice>
            }))
        }
        (_, Some(l)) => {
            let msg = format!(
                "Self-check failing at step {}: {} ({} consecutive; last run {} UTC).",
                vs(l, "failing_step"),
                vs(l, "error"),
                streak,
                fmt_epoch(l["at"].as_f64().unwrap_or(0.0)),
            );
            Either::Right(Either::Right(view! {
                <Notice status=Status::Error label="failing">{msg}</Notice>
            }))
        }
    };
    let rows = FAILURE_ROWS
        .iter()
        .map(|(key, label)| {
            let (label, n) = (
                label.to_string(),
                count(pod.get("failures").and_then(|f| f.get(*key))),
            );
            view! {
                <TableRow>
                    <TableCell>{label}</TableCell>
                    <TableCell>{n}</TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let (name, started) = (
        vs(pod, "name"),
        fmt_epoch(pod.get("started_at").and_then(Value::as_f64).unwrap_or(0.0)),
    );
    Either::Right(view! {
        <Card>
            <CardHeader>
                <CardTitle>"Self-check"</CardTitle>
                <CardDescription>
                    {format!("Pod {name}, up since {started} UTC. ")}
                    "Each pod embeds a fixed query and runs it as a read-only search every few minutes; the counts below run from the pod's start. With several replicas this is the pod that answered."
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div class="mb-4">{state}</div>
                <TableWrapper>
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Failed"</TableHead>
                                <TableHead>"Since start"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>{rows}</TableBody>
                    </Table>
                </TableWrapper>
            </CardContent>
        </Card>
    })
}

// ─── Summary: vector total, judge daily cap ─────────────────────────────────

fn summary(s: &Value) -> impl IntoView + use<> {
    let cap = s.get("judge_daily_cap");
    let admitted = num(cap.and_then(|c| c.get("admitted_today")));
    // A null cap is no judge. A cap of 0 is a setting the meter shows as no
    // share, as it does a missing one.
    let judge = match cap.and_then(|c| c.get("cap")) {
        Some(Value::Null) => Either::Left("no judge configured"),
        limit => Either::Right(view! {
            <Meter label="Judge admitted today against its daily cap" used=admitted limit=num(limit) fmt=fmt_count />
        }),
    };
    let day = cap
        .map(|c| vs(c, "utc_day"))
        .filter(|d| !d.is_empty())
        .unwrap_or_else(|| "—".into());
    let memories = count(section(s, "memories").and_then(|m| m.get("total")));
    view! {
        <div class="space-y-2">
            <StatGrid>
                <Stat label="Memories (vector store)">{memories}</Stat>
                <Stat label="Judge admitted today / daily cap">{judge}</Stat>
                <Stat label="UTC day">{day}</Stat>
            </StatGrid>
            <p class="text-xs text-muted-foreground">
                "The daily cap bounds store-path judge calls per alaya-server process; with several replicas this is the one that answered. Operator backfill is not counted."
            </p>
        </div>
    }
}

// ─── Verdict mix ────────────────────────────────────────────────────────────

fn verdicts_card(s: &Value) -> impl IntoView + use<> {
    let c = section(s, "contradictions");
    let body = match c.and_then(|c| c.get("by_verdict")) {
        None => Either::Left(unavailable("contradiction stats", "see the errors above")),
        Some(by) => {
            let parts = |state: &str| {
                VERDICT_ROWS
                    .iter()
                    .map(|(key, label, _, colour)| {
                        Part::new(
                            *label,
                            num(by.get(*key).and_then(|r| r.get(state))),
                            *colour,
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let (open, resolved) = (parts("open"), parts("resolved"));
            let rows = VERDICT_ROWS
                .iter()
                .map(|(key, label, filter, _)| {
                    let row = by.get(*key);
                    let open = row.and_then(|r| r.get("open"));
                    let resolved = row.and_then(|r| r.get("resolved"));
                    // A dash is a missing count: nothing to link to.
                    let cell = |n: String, resolved: bool| match n.as_str() {
                        "—" => Either::Right(n),
                        _ if resolved => {
                            let href = format!("/alaya/contradictions?verdict={filter}&resolved=1");
                            Either::Left(view! { <Link href=href>{n}</Link> })
                        }
                        _ => {
                            let href = format!("/alaya/contradictions?verdict={filter}");
                            Either::Left(view! { <Link href=href>{n}</Link> })
                        }
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
                        </TableRow>
                    }
                })
                .collect_view();
            // The table stays open: its counts link to the queue.
            Either::Right(view! {
                <div class="space-y-6">
                    <div class="grid gap-6 md:grid-cols-2">
                        <div class="min-w-0 space-y-2">
                            <p class="text-sm font-medium">"Open"</p>
                            <StackedBar label="Open pairs by verdict" parts=open />
                        </div>
                        <div class="min-w-0 space-y-2">
                            <p class="text-sm font-medium">"Resolved"</p>
                            <StackedBar label="Resolved pairs by verdict" parts=resolved />
                        </div>
                    </div>
                    <TableWrapper>
                        <Table>
                            <TableHeader>
                                <TableRow>
                                    <TableHead>"Verdict"</TableHead>
                                    <TableHead>"Open"</TableHead>
                                    <TableHead>"Resolved"</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>{rows}</TableBody>
                        </Table>
                    </TableWrapper>
                </div>
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
            // Every failure is a stored judge failure: the verdict's colour.
            let ranked = top
                .iter()
                .map(|r| Part::new(vs(r, "reason"), num(r.get("count")), Colour::SERIES[4]))
                .chain([Part::new("other", num(f.get("other")), Colour::OTHER)])
                .collect::<Vec<_>>();
            let table = view! {
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
                        </TableBody>
                    </Table>
                </TableWrapper>
            };
            Either::Right(view! {
                <HBarList label="Stored judge failures by reason" rows=ranked />
                {folded(table)}
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
        Some(days) => Either::Right(view! {
            {day_chart("Pairs judged per day by verdict", days, &DAY_COLUMNS)}
            {folded(day_table(days, &DAY_COLUMNS))}
        }),
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

/// `[{date, counts}]` as bars: one per UTC day, labelled `MM-DD`, stacked
/// by column.
fn day_chart(
    label: &'static str,
    days: &[Value],
    columns: &'static [(&'static str, &'static str, Colour)],
) -> impl IntoView + use<> {
    let x = days
        .iter()
        .map(|d| {
            let date = vs(d, "date");
            date.get(5..).map_or_else(|| date.clone(), str::to_string)
        })
        .collect();
    let series = columns
        .iter()
        .map(|(key, heading, colour)| {
            let values = days
                .iter()
                .map(|d| num(d.get("counts").and_then(|c| c.get(*key))))
                .collect();
            Series::new(*heading, values, *colour)
        })
        .collect();
    view! { <BarSeries label=label x=x series=series /> }
}

/// `[{date, counts}]` as a table: one row per UTC day, one column per count.
fn day_table(
    days: &[Value],
    columns: &'static [(&'static str, &'static str, Colour)],
) -> impl IntoView + use<> {
    let rows = days
        .iter()
        .map(|d| {
            let counts = d.get("counts");
            let cells = columns
                .iter()
                .map(|(k, _, _)| {
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
    let heads = columns
        .iter()
        .map(|(_, label, _)| view! { <TableHead>{*label}</TableHead> })
        .collect_view();
    view! {
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
    }
}

// ─── Write novelty ──────────────────────────────────────────────────────────

fn novelty_card(s: &Value) -> impl IntoView + use<> {
    let days = section(s, "writes").and_then(|w| w.get("novelty"));
    let body = match days.and_then(Value::as_array) {
        None => Either::Left(unavailable("write novelty", "see the errors above")),
        Some(days) => Either::Right(view! {
            {day_chart("Memories created per day by similarity to their nearest neighbour", days, &NOVELTY_COLUMNS)}
            {folded(day_table(days, &NOVELTY_COLUMNS))}
        }),
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Write novelty"</CardTitle>
                <CardDescription>
                    "Memories created per UTC day over the last 14 days, by their similarity to the nearest live memory when stored. None means no live neighbour was found, no neighbour search ran (read-only or failed), or the memory predates the measure. A re-store recomputes the value and keeps the creation day."
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
}

// ─── Degenerate reasons ─────────────────────────────────────────────────────

fn degenerate_card(s: &Value) -> impl IntoView + use<> {
    let d = section(s, "contradictions").and_then(|c| c.get("degenerate_reasons"));
    let body = match d {
        None => Either::Left(unavailable("contradiction stats", "see the errors above")),
        Some(d) => {
            let rows = JUDGE_CLASSES
                .iter()
                .map(|k| {
                    let (verdict, n) = (k.to_string(), count(d.get(*k)));
                    view! {
                        <TableRow>
                            <TableCell>{verdict}</TableCell>
                            <TableCell>{n}</TableCell>
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
            let parts = edge_parts(&edges);
            let headline = format!("{nodes} nodes");
            let table = view! {
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
            };
            Either::Right(view! {
                <p class="mb-3 text-sm tabular-nums">{headline}</p>
                <StackedBar label="Graph edges by type" parts=parts />
                {folded(table)}
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

/// Edge counts by type, in key order, each type on its series slot. Past
/// eight types the rest fold into one "other" part, whose count is unknown
/// when any of theirs is.
fn edge_parts(edges: &serde_json::Map<String, Value>) -> Vec<Part> {
    let slots = Colour::SERIES.len();
    let shown = if edges.len() > slots {
        slots - 1
    } else {
        slots
    };
    let mut parts: Vec<Part> = edges
        .iter()
        .take(shown)
        .zip(Colour::SERIES)
        .map(|((kind, n), colour)| Part::new(kind.clone(), num(Some(n)), colour))
        .collect();
    if edges.len() > shown {
        let rest = edges.values().skip(shown).map(|n| num(Some(n))).sum();
        parts.push(Part::new("other", rest, Colour::OTHER));
    }
    parts
}
