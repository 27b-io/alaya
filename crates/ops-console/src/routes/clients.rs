//! Client-store module — managing the LB's clients through the client
//! store's admin API. Two kinds of write:
//! - the kill switch, here: `disable` and `enable`, restriction ops that deny
//!   or re-admit every key of a name on every LB instance within a poll,
//!   whatever state the store's document is in;
//! - document ops, in `document`: clients, their policies and their keys.
//!
//! Writes follow the console's pattern: CSRF token and Origin check, then
//! the subject allowlist re-checked on every write, then POST-redirect-GET.
//! After a write the console shows, per stack, whether it is in effect, and
//! refreshes until it is or until it has said why not (`effect.rs`).
//!
//! No metrics-store value can stop or delay a restriction op. Metrics values
//! are untrusted input, so the handlers that send one never read the metrics
//! store first.

mod document;

pub use document::{commit, history, preview};

use std::collections::{BTreeMap, BTreeSet};

use axum::extract::{RawQuery, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::Digest;

use crate::effect::{
    self, Carried, Doc, Op, RecordState, Report, ServedPart, Sight, Verdict, is_name,
};
use crate::error::AppError;
use crate::fleet::{self, Fleet};
use crate::routes::{clip, fmt_epoch};
use crate::session::{Flash, Session, flash_redirect, take_flash};
use crate::state::{AppState, StoreModule};
use crate::store::{Answer, RestrictionOp, has_code, loggable, refusal_lines};
use crate::ui::*;

const TITLE: &str = "LB clients — ops console";

/// The store refuses a request that would leave more names disabled than
/// this, and more than it found.
const DISABLED_CAP: usize = 1024;

/// The store's ceilings on a restriction op's free text, in characters.
const MAX_ACTOR_CHARS: usize = 64;
const MAX_REASON_CHARS: usize = 200;

/// A request carries at most this many ops.
const MAX_OPS: usize = 100;

/// Why nothing is confirmed when the store's answer names no stack that counts.
const NO_STACKS: &str = "the store's answer named no stacks, so nothing can be confirmed";

/// How long the effect page keeps refreshing: readers poll every 5 s and the
/// metrics store scrapes every 30 s, so a write is normally confirmed within
/// a minute. After this the page stays put and says what is not in effect.
const REFRESH_FOR_SECS: i64 = 120;
const REFRESH_EVERY: &str = "5";

/// Ceiling on a write-log field built from a request or an answer. The
/// answer is upstream text bounded only by the body cap.
const MAX_LOGGED_CHARS: usize = 16 * 1024;

// ─── One judgement-grade read ───────────────────────────────────────────────

/// The store and the fleet read for one judgement: `GET`, the metrics read,
/// then `GET` again. CF and k3s are judged against the first `GET`, and a
/// confirmation is shown only if what is judged did not move by the second:
/// the served part's tag for a restriction op, the document's digest for a
/// document write. Instances are never compared with something the store had
/// already replaced.
struct Reading {
    store: Value,
    fleet: Fleet,
    tag_stable: bool,
    doc_stable: bool,
}

fn part_tag(v: &Value) -> Option<&str> {
    v.pointer("/restrictions/part/tag").and_then(Value::as_str)
}

fn doc_digest(v: &Value) -> Option<&str> {
    v.get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))
        .and_then(|d| d.get("digest"))
        .and_then(Value::as_str)
}

async fn read_all(m: &StoreModule) -> Result<Reading, AppError> {
    let store = m.client.read().await?;
    let fleet = fleet::read(&m.metrics).await;
    let again = m.client.read().await;
    let again = again.as_ref().ok();
    let same = |f: fn(&Value) -> Option<&str>| matches!((f(&store), again.and_then(f)), (Some(a), Some(b)) if a == b);
    let (tag_stable, doc_stable) = (same(part_tag), same(doc_digest));
    Ok(Reading {
        store,
        fleet,
        tag_stable,
        doc_stable,
    })
}

/// The stacks that count, as the store names them: CF, plus each stack that
/// holds a reader token.
fn stacks_of(store: &Value) -> Vec<String> {
    store
        .get("stacks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .filter(|s| is_name(s))
        .map(str::to_string)
        .collect()
}

/// CF's state from the store's report. `None` when the document side of the
/// store did not answer: the report lives there.
fn cf_report(store: &Value) -> Option<Report> {
    let doc = store.get("document")?;
    if doc.get("status").and_then(Value::as_str) != Some("ok") {
        return None;
    }
    let cf = doc.get("cf")?;
    let hex = |p: &str| {
        cf.pointer(p)
            .and_then(Value::as_str)
            .filter(|h| effect::is_hex64(h))
            .map(str::to_string)
    };
    Some(Report {
        running: cf.get("running").and_then(Value::as_bool) == Some(true),
        set_hash: hex("/report/set_hash"),
        digest: hex("/report/digest"),
    })
}

/// How each stack that counts is seen. A stack the metrics store lists but
/// the store does not count is a reader the console cannot account for, so
/// every stack judged from the metrics store reads unknown while it is there.
fn sights(store: &Value, fleet: &Fleet) -> Vec<(String, Sight)> {
    let named = stacks_of(store);
    let instances = fleet.stacks.as_ref().map_err(Clone::clone).and_then(|m| {
        let extra: Vec<&str> = m
            .keys()
            .filter(|s| !named.contains(s))
            .map(String::as_str)
            .collect();
        if extra.is_empty() {
            Ok(m)
        } else {
            Err(format!(
                "the metrics store lists readers of stacks the store does not count ({}), so their \
                 instances cannot be accounted for",
                extra.join(", ")
            ))
        }
    });
    named
        .iter()
        .map(|stack| {
            let sight = if stack == "cf" {
                Sight::Report(cf_report(store))
            } else {
                Sight::Instances(
                    instances
                        .clone()
                        .map(|m| m.get(stack).cloned().unwrap_or_default()),
                )
            };
            (stack.clone(), sight)
        })
        .collect()
}

fn served_part(store: &Value) -> Result<ServedPart, String> {
    effect::parse_part(store.pointer("/restrictions/part"))
}

// ─── The restrictions table ─────────────────────────────────────────────────

/// A row of the store's restrictions table, read defensively: an edited row
/// still renders, with what could not be read marked as such.
struct TableRecord {
    name: String,
    state: Option<RecordState>,
    version: Option<u64>,
    actor: String,
    reason: String,
    at: String,
}

fn table_records(store: &Value) -> Vec<TableRecord> {
    let s = |r: &Value, k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let mut rows: Vec<TableRecord> = store
        .pointer("/restrictions/records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|r| TableRecord {
            name: s(r, "name"),
            state: r
                .get("state")
                .and_then(Value::as_str)
                .and_then(RecordState::parse),
            version: r.get("version").and_then(Value::as_u64),
            actor: s(r, "actor"),
            reason: s(r, "reason"),
            at: s(r, "at"),
        })
        .collect();
    // The store sends them in no defined order: disabled first, then by name.
    rows.sort_by(|a, b| {
        (a.state != Some(RecordState::Disabled), &a.name)
            .cmp(&(b.state != Some(RecordState::Disabled), &b.name))
    });
    rows
}

fn disabled_count(rows: &[TableRecord]) -> usize {
    rows.iter()
        .filter(|r| r.state == Some(RecordState::Disabled))
        .count()
}

/// A record version (microseconds) as a UTC minute, the exact value in the
/// title.
fn fmt_micros(v: u64) -> String {
    fmt_epoch(v as f64 / 1e6)
}

// ─── The page ───────────────────────────────────────────────────────────────

/// The LB clients page's header, above its cards.
fn header() -> impl IntoView {
    view! {
        <PageHeader
            title="LB clients"
            description="The kill switch, and the clients the store's document grants with their policies and keys. Every write needs a reason, recorded with your identity."
        />
    }
}

fn not_configured() -> impl IntoView {
    view! {
        <div class="space-y-6">
            {header()}
            <NotConfigured>
                "Set CLIENT_STORE_URL and CLIENT_STORE_TOKEN (both, or neither) on the console deployment, with the anthropic-lb module configured, to manage LB clients through the client store."
            </NotConfigured>
        </div>
    }
}

/// A sub-page's header: its title, what it is for, and the way back.
fn sub_header(title: String, description: String) -> impl IntoView {
    view! {
        <PageHeader title=title description=description>
            <Link href="/lb/clients">"Back to LB clients"</Link>
        </PageHeader>
    }
}

pub async fn pane(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    RawQuery(raw): RawQuery,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))));
    };
    // `?disable=<name>` prefills the disable form (a client row's button, or
    // the return from a refused request).
    let prefill = prefill_of(raw.as_deref());
    let content = match read_all(m).await {
        // The store could not be read, but a disable needs no read: the form
        // stays, so a degraded store never hides the kill switch.
        Err(e) => Either::Left(view! {
            <div class="space-y-6">
                {header()}
                {unavailable("client store", e.detail())}
                {kill_switch_card(&session.csrf, &prefill)}
            </div>
        }),
        Ok(r) => Either::Right(store_view(&r, &session.csrf, &prefill)),
    };
    Ok((jar, Html(page_html(&session, flash, content))))
}

/// `GET /lb/clients/disable`: the disable form and nothing else. It reads
/// neither the store nor the metrics store, so it loads at once however
/// either is doing: the page to keep for an emergency.
pub async fn disable_page(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    RawQuery(raw): RawQuery,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    if state.store.as_ref().is_none() {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))));
    }
    let prefill = prefill_of(raw.as_deref());
    let content = view! {
        <div class="space-y-6">
            {header()}
            {kill_switch_card(&session.csrf, &prefill)}
        </div>
    };
    Ok((jar, Html(page_html(&session, flash, content))))
}

fn prefill_of(raw: Option<&str>) -> String {
    url::form_urlencoded::parse(raw.unwrap_or_default().as_bytes())
        .find(|(k, _)| k == "disable")
        .map(|(_, v)| v.into_owned())
        .filter(|v| is_name(v))
        .unwrap_or_default()
}

fn kill_switch_card(csrf: &str, prefill: &str) -> impl IntoView + use<> {
    let form = disable_form(csrf, prefill, false, &[], &BTreeMap::new());
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Disable a name"</CardTitle>
                <CardDescription>
                    "Refuses every key of the name on every LB instance within a poll. Sent without reading anything first. If the store answers that the disabled set is full, the LB clients page lets you enable names in the same request."
                </CardDescription>
            </CardHeader>
            <CardContent>{form}</CardContent>
        </Card>
    }
}

fn capitalise(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

fn page_html(session: &Session, flash: Option<Flash>, content: impl IntoView) -> String {
    page(TITLE, Nav::LbClients, session, flash, content)
}

/// Whether a stack holds a write, as a badge, with why when it does not.
fn verdict_view(v: &Verdict) -> impl IntoView + use<> {
    let (status, label, why) = match v {
        Verdict::InEffect => (Status::Ok, "in effect", None),
        Verdict::NotInEffect(why) => (Status::Warn, "not in effect", Some(why.clone())),
        Verdict::Unknown(why) => (Status::Off, "unknown", Some(why.clone())),
    };
    view! {
        <StatusBadge status=status label=label />
        {why.map(|w| view! { <p class="mt-1 text-xs wrap-break-word">{w}</p> })}
    }
}

fn store_view(r: &Reading, csrf: &str, prefill: &str) -> impl IntoView + use<> {
    let part = served_part(&r.store);
    let doc = effect::served_doc(&r.store);
    let sights = sights(&r.store, &r.fleet);
    let rows = table_records(&r.store);
    let unmatched = r.fleet.unmatched.clone().unwrap_or_default();
    view! {
        <div class="space-y-6">
            {header()}
            {warnings(r, &part)}
            {stacks_card(&part, &doc, &sights, r)}
            {restrictions_card(&rows, &part, &unmatched, csrf, prefill)}
            {document::clients_card(&r.store, csrf)}
        </div>
    }
}

/// Everything that should stop an operator reading the rest as healthy.
fn warnings(r: &Reading, part: &Result<ServedPart, String>) -> impl IntoView + use<> {
    let mut items: Vec<(Status, String)> = Vec::new();
    match part {
        Err(e) => items.push((
            Status::Error,
            format!("Restrictions part unreadable: {e}. Nothing reads as in effect."),
        )),
        Ok(p) if !p.full => items.push((Status::Error, effect::PARTIAL.to_string())),
        Ok(_) => {}
    }
    match &r.fleet.behind {
        Ok(b) if !b.is_empty() => items.push((
            Status::Error,
            format!(
                "Restrictions behind on {}: those instances hold a disable or revoked digest the store lost. Re-send the name's disable (or enable), or reseed with restrictions_import.",
                b.iter().cloned().collect::<Vec<_>>().join(", ")
            ),
        )),
        Ok(_) => {}
        Err(e) => items.push((Status::Warn, format!("Restrictions-behind alerts unreadable (metrics store: {e})."))),
    }
    // A failed read of the readers' unmatched names hides the only flag on a
    // mistyped disable, which still reads as in effect: say it is missing.
    if let Err(e) = &r.fleet.unmatched {
        items.push((
            Status::Warn,
            format!("Unmatched disabled names unreadable (metrics store: {e}): a mistyped disable is not flagged."),
        ));
    }
    if stacks_of(&r.store).is_empty() {
        items.push((Status::Error, format!("{}.", capitalise(NO_STACKS))));
    }
    let doc = r.store.get("document");
    if doc.and_then(|d| d.get("status")).and_then(Value::as_str) != Some("ok") {
        let e = doc
            .and_then(|d| d.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("no detail");
        items.push((
            Status::Warn,
            format!(
                "The document side of the store did not answer ({}). Restriction ops still work.",
                clip(e, 160)
            ),
        ));
    }
    if let Some(frozen) = doc
        .and_then(|d| d.get("frozen"))
        .and_then(Value::as_array)
        .filter(|f| !f.is_empty())
    {
        let reasons: Vec<String> = frozen
            .iter()
            .filter_map(Value::as_str)
            .map(|s| clip(s, 160))
            .collect();
        items.push((
            Status::Warn,
            format!(
                "The store refuses document writes: {}. Restriction ops still work.",
                reasons.join("; ")
            ),
        ));
    }
    if r.store.get("pitr_window").and_then(Value::as_bool) == Some(true) {
        items.push((Status::Warn, "A point-in-time recovery window is open: the store takes no document write but pitr. Restriction ops still work.".into()));
    }
    if doc
        .and_then(|d| d.get("recovery_refused"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        items.push((Status::Error, "A recovery was refused: the store serves no document to readers until an import or another recovery commits. Restriction ops still work.".into()));
    }
    match &r.fleet.store_alerts {
        Ok(who) if !who.is_empty() => items.push((
            Status::Error,
            format!(
                "Instances report the store behind or edited: {}. Every document write needs a typed confirmation while they do. Restriction ops still work.",
                clip(&who.iter().cloned().collect::<Vec<_>>().join(", "), 600)
            ),
        )),
        Ok(_) => {}
        Err(e) => items.push((Status::Warn, format!("Store behind or edited alerts unreadable (metrics store: {e})."))),
    }
    let list = items
        .into_iter()
        .map(|(status, text)| view! { <Notice status=status label="warning">{text}</Notice> })
        .collect_view();
    view! { <div class="space-y-2">{list}</div> }
}

fn stacks_card(
    part: &Result<ServedPart, String>,
    doc: &Result<Option<Doc>, String>,
    sights: &[(String, Sight)],
    r: &Reading,
) -> impl IntoView + use<> {
    let rows = sights
        .iter()
        .map(|(stack, sight)| {
            let restrictions = match part {
                Ok(p) => effect::steady(effect::holds(p, sight), r.tag_stable, effect::MOVED),
                Err(e) => Verdict::Unknown(format!("the served part is unreadable: {e}")),
            };
            let held_doc = match doc {
                Ok(Some(d)) => effect::steady(
                    effect::holds_doc(&d.digest, sight),
                    r.doc_stable,
                    effect::DOC_MOVED,
                ),
                Ok(None) => Verdict::Unknown("the store holds no document".into()),
                Err(e) => Verdict::Unknown(e.clone()),
            };
            let stack = stack.clone();
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{stack}</span></TableCell>
                    <TableCell>{verdict_view(&restrictions)}</TableCell>
                    <TableCell>{verdict_view(&held_doc)}</TableCell>
                </TableRow>
            }
        })
        .collect_view();
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Stacks"</CardTitle>
                <CardDescription>
                    "Whether each stack holds the restrictions and the document the store serves now: CF from its report to the store, every other stack from its instances in the metrics store. Restrictions are in effect only when every instance reports exactly the served set hash, and the document only when every instance reports exactly the served digest."
                </CardDescription>
            </CardHeader>
            <CardContent>
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Stack"</TableHead>
                        <TableHead>"Served restrictions"</TableHead>
                        <TableHead>"Served document"</TableHead>
                    </TableRow></TableHeader>
                    <TableBody>{rows}</TableBody>
                </Table></TableWrapper>
            </CardContent>
        </Card>
    }
}

fn restrictions_card(
    rows: &[TableRecord],
    part: &Result<ServedPart, String>,
    unmatched: &BTreeMap<String, BTreeSet<String>>,
    csrf: &str,
    prefill: &str,
) -> impl IntoView + use<> {
    let table = rows
        .iter()
        .map(|r| {
            let served = match (part, r.state, r.version) {
                (Ok(p), Some(s), Some(v)) => p.records.get(&r.name) == Some(&(s, v)),
                _ => false,
            };
            let served = if served { "served" } else { "not in the served part" };
            let (state, state_text) = match r.state {
                Some(RecordState::Disabled) => (Status::Error, "disabled"),
                Some(RecordState::Enabled) => (Status::Ok, "enabled"),
                None => (Status::Off, "unreadable"),
            };
            let flag = unmatched.get(&r.name).filter(|_| r.state == Some(RecordState::Disabled)).map(|stacks| {
                let text = format!("matches no client on {} — mistyped?", clip(&stacks.iter().cloned().collect::<Vec<_>>().join(", "), 200));
                view! { <StatusBadge status=Status::Warn label=text /> }
            });
            let (version_text, version_title) = match r.version {
                Some(v) => (fmt_micros(v), v.to_string()),
                None => ("unreadable".to_string(), String::new()),
            };
            let action = match (r.state, r.version) {
                (Some(RecordState::Disabled), Some(v)) => Either::Left(enable_form(csrf, &r.name, v)),
                (Some(RecordState::Enabled), _) => Either::Right(Either::Left(disable_form_inline(csrf, &r.name))),
                _ => Either::Right(Either::Right(view! { <span class="text-xs text-muted-foreground">"the record is unreadable: repair the store"</span> })),
            };
            let (name, actor, reason, at) = (r.name.clone(), r.actor.clone(), r.reason.clone(), r.at.clone());
            view! {
                <TableRow>
                    <TableCell>
                        <div class="flex flex-wrap items-center gap-1">
                            <span class="font-mono text-xs wrap-break-word">{name}</span>
                            {flag}
                        </div>
                    </TableCell>
                    <TableCell><StatusBadge status=state label=state_text /></TableCell>
                    <TableCell secondary=true><span class="tabular-nums whitespace-nowrap" title=version_title>{version_text}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs wrap-break-word">{actor}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs wrap-break-word">{reason}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs whitespace-nowrap">{at}</span></TableCell>
                    <TableCell><span class="text-xs">{served}</span></TableCell>
                    <TableCell>{action}</TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let full = disabled_count(rows) >= DISABLED_CAP;
    // Component children are boxed 'static closures: build them first.
    let form = disable_form(csrf, prefill, full, rows, unmatched);
    let unrecorded = enable_unrecorded_form(csrf);
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Restrictions (kill switch)"</CardTitle>
                <CardDescription>
                    "disable refuses every key of a name on every instance, TOML keys of an ordinary client included; enable undoes it. Both reach every polling instance within a poll, whatever state the store's document is in, and nothing read from the metrics store can stop them. Each needs a reason, recorded with your identity. "
                    <Link href="/lb/clients/disable">"The disable page"</Link>
                    " loads without reading anything, for when the store or the metrics store is slow."
                </CardDescription>
            </CardHeader>
            <CardContent>
                {form}
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Name"</TableHead>
                        <TableHead>"State"</TableHead>
                        <TableHead secondary=true>"Version (UTC)"</TableHead>
                        <TableHead secondary=true>"Actor"</TableHead>
                        <TableHead secondary=true>"Reason"</TableHead>
                        <TableHead secondary=true>"At"</TableHead>
                        <TableHead>"Served"</TableHead>
                        <TableHead>""</TableHead>
                    </TableRow></TableHeader>
                    <TableBody>{table}</TableBody>
                </Table></TableWrapper>
                {unrecorded}
            </CardContent>
        </Card>
    }
}

/// The disable form. When the disabled set is full it also asks which
/// disabled names to enable in the same request, listing first those the
/// readers report matching nothing.
fn disable_form(
    csrf: &str,
    prefill: &str,
    full: bool,
    rows: &[TableRecord],
    unmatched: &BTreeMap<String, BTreeSet<String>>,
) -> impl IntoView + use<> {
    let chooser = full.then(|| {
        let mut disabled: Vec<(&TableRecord, bool)> = rows
            .iter()
            .filter(|r| r.state == Some(RecordState::Disabled) && r.version.is_some())
            .map(|r| (r, unmatched.contains_key(&r.name)))
            .collect();
        disabled.sort_by_key(|(r, matches_nothing)| (!matches_nothing, r.name.clone()));
        let boxes = disabled
            .into_iter()
            .map(|(r, matches_nothing)| {
                let value = format!("{}:{}", r.name, r.version.unwrap_or_default());
                let label = if matches_nothing {
                    format!("{} (matches no client)", r.name)
                } else {
                    r.name.clone()
                };
                view! {
                    <label class="flex items-center gap-2 text-xs">
                        <input type="checkbox" name="also_enable" value=value />
                        <span class="font-mono break-all">{label}</span>
                    </label>
                }
            })
            .collect_view();
        view! {
            <div class="mt-3">
                <div class="mb-2">
                    <Notice status=Status::Warn label="disabled set full">
                        "The store refuses a request that leaves more than 1,024 names disabled and more than it found. Pick names to enable in the same request:"
                    </Notice>
                </div>
                <div class="grid grid-cols-2 sm:grid-cols-4 gap-1 max-h-64 overflow-y-auto">{boxes}</div>
            </div>
        }
    });
    let csrf = csrf.to_string();
    let prefill = prefill.to_string();
    view! {
        <form method="post" action="/lb/clients/restrict" class="mb-6" id="disable">
            <input type="hidden" name="csrf" value=csrf />
            <input type="hidden" name="action" value="disable" />
            <div class="flex flex-col gap-3 sm:flex-row sm:items-end">
                <div class="sm:w-56">
                    <label class="text-xs text-muted-foreground" for="disable_name">"Name"</label>
                    <input class=INPUT_CLASS id="disable_name" name="name" value=prefill required maxlength="63" placeholder="client name" />
                </div>
                <div class="sm:flex-1">
                    <label class="text-xs text-muted-foreground" for="disable_reason">"Reason"</label>
                    <input class=INPUT_CLASS id="disable_reason" name="reason" required maxlength="200" placeholder="why this name is cut off" />
                </div>
                <button type="submit" class=btn(Btn::Destructive)>"Disable"</button>
            </div>
            {chooser}
        </form>
    }
}

fn disable_form_inline(csrf: &str, name: &str) -> impl IntoView + use<> {
    let (csrf, name) = (csrf.to_string(), name.to_string());
    view! {
        <form method="post" action="/lb/clients/restrict" class="flex items-center gap-2">
            <input type="hidden" name="csrf" value=csrf />
            <input type="hidden" name="action" value="disable" />
            <input type="hidden" name="name" value=name />
            <input class=row_input() name="reason" required maxlength="200" placeholder="reason" />
            <button type="submit" class=btn_sm(Btn::Destructive)>"Disable"</button>
        </form>
    }
}

/// `enable` carries the version of the name's record as this page read it,
/// so a newer `disable` made since gets 409 instead of being undone.
fn enable_form(csrf: &str, name: &str, version: u64) -> impl IntoView + use<> {
    let (csrf, name, version) = (csrf.to_string(), name.to_string(), version.to_string());
    view! {
        <form method="post" action="/lb/clients/restrict" class="flex items-center gap-2">
            <input type="hidden" name="csrf" value=csrf />
            <input type="hidden" name="action" value="enable" />
            <input type="hidden" name="name" value=name />
            <input type="hidden" name="expected_version" value=version />
            <input class=row_input() name="reason" required maxlength="200" placeholder="reason" />
            <button type="submit" class=btn_sm(Btn::Outline)>"Enable"</button>
        </form>
    }
}

/// `enable` of a name the store holds no record of writes one, which releases
/// a reader still holding a disable the store lost.
fn enable_unrecorded_form(csrf: &str) -> impl IntoView + use<> {
    let csrf = csrf.to_string();
    view! {
        <details class="mt-6">
            <summary class="text-sm cursor-pointer">"Enable a name the store holds no record of"</summary>
            <p class="text-xs text-muted-foreground my-2">
                "Writes an enabled record, which releases an instance still holding a disable the store lost (restrictions behind). A name that has a record gets 409: use its row instead."
            </p>
            <form method="post" action="/lb/clients/restrict" class="flex flex-col gap-3 sm:flex-row sm:items-end">
                <input type="hidden" name="csrf" value=csrf />
                <input type="hidden" name="action" value="enable" />
                <input type="hidden" name="expected_version" value="0" />
                <input class=INPUT_CLASS name="name" required maxlength="63" placeholder="client name" />
                <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" />
                <button type="submit" class=btn(Btn::Outline)>"Enable"</button>
            </form>
        </details>
    }
}

// ─── Writes ─────────────────────────────────────────────────────────────────

/// `routes::guard_write` for a client-store write, which puts its refusal
/// on the write log like every other attempt.
fn guard(state: &AppState, session: &Session, csrf: &str, kind: &str) -> Result<(), AppError> {
    crate::routes::guard_write(state, session, csrf)
        .inspect_err(|e| refused_locally(session, kind, e.detail()))
}

/// A write refused before anything was sent to the store, logged with the
/// subject on the write log like every other attempt.
fn refused_locally(session: &Session, kind: &str, why: &str) {
    tracing::warn!(
        target: "ops_console::write_log",
        sub = ?session.sub,
        kind,
        why = %clip(why, 300),
        "client-store write refused before sending"
    );
}

/// The actor the store records on each op: the session's email when the IdP
/// sent one that fits the store's rule, else its subject, else a digest of
/// the subject. The rule is 1 to 64 characters with no control character.
/// It always yields one, because the kill switch must never fail on how a
/// name is spelled; the write log carries the full subject either way.
fn actor_of(session: &Session) -> String {
    let fits = |s: &str| {
        !s.is_empty() && s.chars().count() <= MAX_ACTOR_CHARS && !s.chars().any(char::is_control)
    };
    session
        .email
        .as_deref()
        .filter(|e| fits(e))
        .or(Some(session.sub.as_str()).filter(|s| fits(s)))
        .map(str::to_string)
        .unwrap_or_else(|| {
            let d = sha2::Sha256::digest(session.sub.as_bytes());
            format!("sub-sha256:{}", &effect::hex(&d)[..16])
        })
}

/// A reason as the store takes it: whitespace runs collapsed to one space
/// (so no newline or tab reaches it), 1 to 200 characters.
fn clean_reason(raw: &str) -> Result<String, String> {
    let r = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let n = r.chars().count();
    if n == 0 {
        return Err("a reason is required: the store records it on the op".into());
    }
    if n > MAX_REASON_CHARS || r.chars().any(char::is_control) {
        return Err(format!(
            "a reason is at most {MAX_REASON_CHARS} characters, with no control characters"
        ));
    }
    Ok(r)
}

/// Whether a request may or may not have applied: anything but a 200 or a
/// 4xx refusal. The store answers 503 `store.unavailable` to any failure of
/// an object's call, a connection dropped after the commit included, and no
/// 5xx proves that nothing was written; nor does a status the store never
/// sends for these requests. The write log and the page both read it through
/// here, so they can never disagree.
fn outcome_unknown(a: &Answer) -> bool {
    !(a.status == StatusCode::OK || a.status.is_client_error())
}

/// One line per write attempt, refusals and upstream failures included, with
/// the user's subject. Request and answer are re-serialised and clipped, and
/// the answer passes through `loggable`, so no minted key is ever logged.
fn write_log(sub: &str, kind: &str, request: &Value, outcome: Result<&Answer, &AppError>) {
    let request = clip(&request.to_string(), MAX_LOGGED_CHARS);
    let answer = |a: &Answer| clip(&loggable(&a.body).to_string(), MAX_LOGGED_CHARS);
    match outcome {
        Ok(a) if outcome_unknown(a) => tracing::warn!(
            target: "ops_console::write_log",
            sub = ?sub,
            kind,
            request = %request,
            status = a.status.as_u16(),
            answer = %answer(a),
            "client-store write outcome unknown"
        ),
        Ok(a) if a.status.is_success() => tracing::info!(
            target: "ops_console::write_log",
            sub = ?sub,
            kind,
            request = %request,
            status = a.status.as_u16(),
            answer = %answer(a),
            "client-store write committed"
        ),
        Ok(a) => tracing::warn!(
            target: "ops_console::write_log",
            sub = ?sub,
            kind,
            request = %request,
            status = a.status.as_u16(),
            answer = %answer(a),
            "client-store write refused"
        ),
        Err(e) => tracing::warn!(
            target: "ops_console::write_log",
            sub = ?sub,
            kind,
            request = %request,
            error = %e.detail(),
            "client-store write outcome unknown"
        ),
    }
}

#[derive(Deserialize)]
pub struct RestrictForm {
    #[serde(default)]
    csrf: String,
    #[serde(default)]
    action: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    expected_version: String,
    /// `name:version` of each disabled name to enable in the same request,
    /// chosen when the disabled set is full.
    #[serde(default)]
    also_enable: Vec<String>,
}

fn back_to(name: &str) -> String {
    format!(
        "/lb/clients?disable={}#disable",
        url::form_urlencoded::byte_serialize(name.as_bytes()).collect::<String>()
    )
}

pub async fn restrict(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum_extra::extract::Form(form): axum_extra::extract::Form<RestrictForm>,
) -> Result<Response, AppError> {
    let m = store_of(&state)?;
    guard(&state, &session, &form.csrf, RESTRICTIONS)?;
    let secure = state.secure_cookies();
    let ops = match restriction_ops(&form, &actor_of(&session)) {
        Ok(ops) => ops,
        Err(msg) => {
            // A refusal made here, before anything is sent, is an attempt too.
            refused_locally(&session, RESTRICTIONS, &msg);
            return Ok(flash_redirect(jar, secure, "error", msg, "/lb/clients"));
        }
    };
    let export = {
        let ops = ops.clone();
        move |m: &StoreModule, body: &Value| {
            for op in written_ops(body, &ops).unwrap_or_default() {
                m.committed.record(&op.name, op.version);
            }
        }
    };
    let body = json!({ "restrictions": ops });
    let answer = send_recorded(m, &session.sub, RESTRICTIONS, body, export).await;
    let (kind, msg, to) = restrict_outcome(m, &ops, answer).await;
    Ok(flash_redirect(jar, secure, kind, msg, &to))
}

/// The client-store module, or a 404 for a route of a module not configured.
fn store_of(state: &AppState) -> Result<&StoreModule, AppError> {
    state
        .store
        .as_ref()
        .ok_or_else(|| AppError::NotFound("the client-store module is not configured".into()))
}

/// Send a write and record it, in a task of its own: a write the store
/// commits must reach the write log and the committed labels even when the
/// operator's connection drops and the handler is dropped with it. `export`
/// runs on a 200's body and exports what it can read as committed. The
/// answer, minted keys included, comes back to the handler and nowhere else.
async fn send_recorded(
    m: &StoreModule,
    sub: &str,
    kind: &'static str,
    body: Value,
    export: impl FnOnce(&StoreModule, &Value) + Send + 'static,
) -> Result<Answer, AppError> {
    let (m, sub) = (m.clone(), sub.to_string());
    tokio::spawn(async move {
        let answer = m.client.post(&body).await;
        write_log(&sub, kind, &body, answer.as_ref());
        if let Ok(a) = &answer
            && a.status == StatusCode::OK
        {
            export(&m, &a.body);
        }
        answer
    })
    .await
    .unwrap_or_else(|e| Err(AppError::Upstream(format!("the write's task failed ({e})"))))
}

/// How a write's answer reads, for every kind of write.
enum Outcome {
    /// Whether it applied is unknown: the answer was lost, or is one that no
    /// refusal or commit gives (`outcome_unknown`). Why, in words.
    Unknown(String),
    /// A 4xx: the store refused, and wrote nothing.
    Refused { status: StatusCode, body: Value },
    /// A 200.
    Committed(Value),
}

fn outcome(answer: Result<Answer, AppError>) -> Outcome {
    match answer {
        Err(e) => Outcome::Unknown(format!("the store's answer was lost ({})", e.detail())),
        Ok(a) if outcome_unknown(&a) => Outcome::Unknown(format!(
            "the store answered {}: {}",
            a.status.as_u16(),
            refusal_lines(&a.body).join("; ")
        )),
        Ok(a) if a.status == StatusCode::OK => Outcome::Committed(a.body),
        Ok(a) => Outcome::Refused {
            status: a.status,
            body: a.body,
        },
    }
}

/// A refusal in words: what was refused, its status and the store's errors.
fn refused_text(what: &str, status: StatusCode, body: &Value) -> String {
    format!(
        "{what} ({}): {}.",
        status.as_u16(),
        refusal_lines(body).join("; ")
    )
}

/// The write log's kind for restriction ops.
const RESTRICTIONS: &str = "restrictions";

/// The ops a restrict form asks for: the enables chosen when the disabled
/// set is full, then the form's own `disable` or `enable`, last.
fn restriction_ops(form: &RestrictForm, actor: &str) -> Result<Vec<RestrictionOp>, String> {
    let name = form.name.trim().to_string();
    if !is_name(&name) {
        return Err("A name is [a-z0-9] then [a-z0-9_-], at most 63 characters.".into());
    }
    let reason = clean_reason(&form.reason)?;
    let mut ops: Vec<RestrictionOp> = Vec::new();
    for pick in &form.also_enable {
        let parsed = pick
            .split_once(':')
            .and_then(|(n, v)| Some((n, v.parse::<u64>().ok()?)))
            .filter(|(n, _)| is_name(n));
        let Some((n, v)) = parsed else {
            return Err("A name to enable in the same request is malformed.".into());
        };
        ops.push(RestrictionOp::Enable {
            name: n.to_string(),
            expected_version: v,
            actor: actor.to_string(),
            reason: reason.clone(),
        });
    }
    ops.push(match form.action.as_str() {
        "disable" => RestrictionOp::Disable {
            name,
            actor: actor.to_string(),
            reason,
        },
        "enable" => RestrictionOp::Enable {
            name,
            expected_version: form
                .expected_version
                .trim()
                .parse()
                .map_err(|_| "enable needs the version of the record it read.")?,
            actor: actor.to_string(),
            reason,
        },
        _ => return Err("The action is disable or enable.".into()),
    });
    if ops.len() > MAX_OPS {
        return Err(format!("A request carries at most {MAX_OPS} ops."));
    }
    Ok(ops)
}

/// What the page says after a restrict request, and where it goes: the
/// effect page after a commit, the form again after a full-set refusal, the
/// clients page otherwise. The committed labels are exported where the
/// request is sent, and only for a commit whose answer reads in full.
async fn restrict_outcome(
    m: &StoreModule,
    ops: &[RestrictionOp],
    answer: Result<Answer, AppError>,
) -> (&'static str, String, String) {
    let back = "/lb/clients".to_string();
    match outcome(answer) {
        Outcome::Unknown(why) => ("error", unknown_outcome(m, ops, &why).await, back),
        Outcome::Committed(body) => {
            let done = match written_ops(&body, ops) {
                Ok(done) => done,
                Err(e) => {
                    // Committed, but which records it wrote, at which versions,
                    // is unreadable: nothing can be confirmed or exported.
                    let mut msg = format!(
                        "The store committed the request, but its answer could not be read \
                         ({e}), so nothing is confirmed or exported as committed."
                    );
                    msg.push_str(&now_holds(m, ops).await);
                    return ("error", msg, back);
                }
            };
            let summary = done
                .iter()
                .map(|op| format!("{} {}", op.state.as_str(), op.name))
                .collect::<Vec<_>>()
                .join(", ");
            let msg = if summary.is_empty() {
                "Nothing changed: every name was already in that state.".to_string()
            } else {
                format!("Committed: {summary}. Confirming on every stack…")
            };
            ("ok", msg, effect_href(&done, None))
        }
        Outcome::Refused { status, body } => {
            let mut msg = refused_text("The store refused the request", status, &body);
            if status == StatusCode::CONFLICT {
                // A 409 means the console's picture was stale: read the store
                // again before reporting, and say what it holds now.
                msg.push_str(&now_holds(m, ops).await);
            }
            let to = match ops.last() {
                Some(last) if has_code(&body, "disabled.cap") => back_to(last.name()),
                _ => back,
            };
            ("error", msg, to)
        }
    }
}

/// A request that may or may not have applied: its answer was lost, or the
/// store's call to its restrictions object failed, possibly after the commit.
/// Say so and read the store to show what it holds now. Nothing is exported
/// as committed, since the console cannot prove it wrote the record; an
/// unmatched-install alert on its own write is the safe side of that doubt,
/// and the write log holds the attempt.
async fn unknown_outcome(m: &StoreModule, ops: &[RestrictionOp], why: &str) -> String {
    let mut msg = format!(
        "Whether the request applied is unknown: {why}. Check the restrictions before \
         sending it again."
    );
    msg.push_str(&now_holds(m, ops).await);
    msg
}

/// After a 409: what the store holds now for each name the request named.
async fn now_holds(m: &StoreModule, ops: &[RestrictionOp]) -> String {
    let store = match m.client.read().await {
        Ok(s) => s,
        Err(e) => return format!(" Re-reading the store failed too: {}.", e.detail()),
    };
    let rows = table_records(&store);
    let names: BTreeSet<&str> = ops.iter().map(RestrictionOp::name).collect();
    let now: Vec<String> = names
        .into_iter()
        .map(|n| match rows.iter().find(|r| r.name == n) {
            Some(r) => format!(
                "{n} is {} at {} by {} ({})",
                r.state.map(RecordState::as_str).unwrap_or("unreadable"),
                r.version
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "?".into()),
                clip(&r.actor, 64),
                clip(&r.reason, 80)
            ),
            None => format!("{n} has no record"),
        })
        .collect();
    format!(" Re-read the store: {}.", now.join("; "))
}

/// The ops a 200 actually wrote, read strictly: one readable result per op
/// sent, in order, each naming the name and state of the op it answers. An
/// `enable` of a name already enabled writes nothing and returns the existing
/// record, so there is nothing new to confirm or to export as committed. An
/// answer that cannot be read in full is an error, never "nothing changed".
fn written_ops(answer: &Value, sent: &[RestrictionOp]) -> Result<Vec<Op>, String> {
    let results = answer
        .get("results")
        .and_then(Value::as_array)
        .ok_or("no results")?;
    if results.len() != sent.len() {
        return Err(format!("{} results for {} ops", results.len(), sent.len()));
    }
    let mut done = Vec::new();
    for (r, asked) in results.iter().zip(sent) {
        let op = (|| {
            Some(Op {
                name: r.get("name")?.as_str().filter(|n| is_name(n))?.to_string(),
                state: RecordState::parse(r.get("state")?.as_str()?)?,
                version: r.get("version")?.as_u64()?,
            })
        })();
        let op = match op {
            Some(op) if op.name == asked.name() && op.state == asked.state() => op,
            Some(op) => {
                return Err(format!(
                    "a result names {} {} for an op on {} {}",
                    op.state.as_str(),
                    op.name,
                    asked.state().as_str(),
                    asked.name()
                ));
            }
            None => return Err("a result is malformed".into()),
        };
        match r.get("written").and_then(Value::as_bool) {
            Some(true) => done.push(op),
            Some(false) => {}
            None => return Err("a result is malformed".into()),
        }
    }
    Ok(done)
}

// ─── After a write: in effect per stack ─────────────────────────────────────

/// `/lb/clients/effect?op=<state>.<name>.<version>…&doc=<version>.<digest>&at=<epoch s>`.
/// What to confirm rides the URL, taken from the write's own result, so the
/// page needs no state of its own and survives a console restart.
fn effect_href(ops: &[Op], doc: Option<&Doc>) -> String {
    let mut qs = url::form_urlencoded::Serializer::new(String::new());
    for op in ops {
        qs.append_pair(
            "op",
            &format!("{}.{}.{}", op.state.as_str(), op.name, op.version),
        );
    }
    if let Some(d) = doc {
        qs.append_pair("doc", &format!("{}.{}", d.version, d.digest));
    }
    qs.append_pair("at", &crate::session::now_epoch().to_string());
    format!("/lb/clients/effect?{}", qs.finish())
}

/// What the effect page confirms, read from its URL.
struct Wanted {
    ops: Vec<Op>,
    doc: Option<Doc>,
    at: i64,
}

fn parse_effect_query(raw: Option<&str>) -> Result<Wanted, AppError> {
    let mut wanted = Wanted {
        ops: Vec::new(),
        doc: None,
        at: 0,
    };
    for (k, v) in url::form_urlencoded::parse(raw.unwrap_or("").as_bytes()) {
        match &*k {
            "op" => {
                let mut it = v.splitn(3, '.');
                let op = (|| {
                    let state = RecordState::parse(it.next()?)?;
                    let name = it.next().filter(|n| is_name(n))?.to_string();
                    let version = it.next()?.parse().ok()?;
                    Some(Op {
                        name,
                        state,
                        version,
                    })
                })()
                .ok_or_else(|| AppError::BadRequest("malformed op".into()))?;
                wanted.ops.push(op);
            }
            "doc" => {
                let doc = v
                    .split_once('.')
                    .and_then(|(version, digest)| {
                        Some(Doc {
                            version: version.parse().ok()?,
                            digest: digest.to_string(),
                        })
                    })
                    .filter(|d| effect::is_hex64(&d.digest))
                    .ok_or_else(|| AppError::BadRequest("malformed document".into()))?;
                wanted.doc = Some(doc);
            }
            "at" => wanted.at = v.parse().unwrap_or(0),
            _ => {}
        }
    }
    if wanted.ops.len() > MAX_OPS {
        return Err(AppError::BadRequest(format!("at most {MAX_OPS} ops")));
    }
    Ok(wanted)
}

/// One thing the effect page confirms: its title, and its judgement or the
/// sentence saying why none could be made.
type Block = (String, Result<effect::Judged, String>);

pub async fn effect(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    RawQuery(raw): RawQuery,
) -> Result<Response, AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))).into_response());
    };
    let wanted = parse_effect_query(raw.as_deref())?;
    let now_micros = (crate::session::now_epoch().max(0) as u64) * 1_000_000;
    let reading = match read_all(m).await {
        Ok(r) => r,
        Err(e) => {
            let content = view! {
                <div class="space-y-6">
                    {sub_header("In effect?".into(), "Whether every stack holds the write.".into())}
                    {unavailable("client store", e.detail())}
                </div>
            };
            return Ok((jar, Html(page_html(&session, flash, content))).into_response());
        }
    };
    let part = served_part(&reading.store);
    let sights = sights(&reading.store, &reading.fleet);
    // An answer that names no stack would judge everything on an empty
    // table and read as settled: nothing is confirmed instead.
    let no_stacks = sights
        .is_empty()
        .then(|| format!("{}.", capitalise(NO_STACKS)));
    let mut blocks: Vec<Block> = wanted
        .ops
        .iter()
        .filter(|op| !effect::aged_out(op, now_micros))
        .map(|op| {
            let title = format!("{} {} — version {}", op.state.as_str(), op.name, op.version);
            let j = match (&part, &no_stacks) {
                (Err(e), _) => Err(format!(
                    "The served part is unreadable, so nothing is confirmed: {e}"
                )),
                (_, Some(why)) => Err(why.clone()),
                (Ok(p), None) => Ok(effect::judge_op(op, p, &sights, reading.tag_stable)),
            };
            (title, j)
        })
        .collect();
    if let Some(d) = &wanted.doc {
        let title = format!(
            "document version {} — digest {}",
            d.version,
            crate::routes::short_digest(&d.digest)
        );
        let j = match (effect::served_doc(&reading.store), &no_stacks) {
            (Err(e), _) => Err(format!(
                "The store's document is unreadable, so nothing is confirmed: {e}"
            )),
            (_, Some(why)) => Err(why.clone()),
            (Ok(served), None) => Ok(effect::judge_doc(
                d,
                served.as_ref(),
                &sights,
                reading.doc_stable,
            )),
        };
        blocks.push((title, j));
    }
    // Keep polling while something can still become in effect on its own.
    let settling = blocks.iter().any(|(_, j)| match j {
        Ok(j) => {
            !matches!(j.carried, Carried::Superseded(_))
                && j.stacks.iter().any(|(_, v)| *v != Verdict::InEffect)
        }
        Err(_) => true,
    });
    // `at` comes from the URL: a time in the future, or one so far back that
    // the subtraction would overflow, never keeps a page refreshing.
    let now = crate::session::now_epoch();
    let at = wanted.at;
    let refresh = settling && at <= now && now.saturating_sub(at) < REFRESH_FOR_SECS;
    let note = if refresh {
        "Refreshing every 5 s until every stack is in effect (readers poll every 5 s; the metrics store trails by up to a minute)."
    } else {
        "Not refreshing: everything is settled, or two minutes have passed. Reload to check again."
    };
    let content = view! {
        <div class="space-y-6">
            {sub_header("In effect?".into(), note.into())}
            {warnings(&reading, &part)}
            {effect_card(&blocks)}
        </div>
    };
    let html = Html(page_html(&session, flash, content));
    let mut resp = (jar, html).into_response();
    if refresh {
        resp.headers_mut()
            .insert(header::REFRESH, HeaderValue::from_static(REFRESH_EVERY));
    }
    Ok(resp)
}

fn effect_card(blocks: &[Block]) -> impl IntoView + use<> {
    let blocks_view = blocks
        .iter()
        .map(|(title, j)| {
            let title = title.clone();
            let body = match j {
                Err(e) => Either::Left(view! { <p class="text-sm">{e.clone()}</p> }),
                Ok(j) => {
                    let head = match &j.carried {
                        Carried::Superseded(v) => Some(format!("Superseded: a newer write, at {v}, replaced this one. It is history, not in effect.")),
                        _ => None,
                    };
                    let rows = j
                        .stacks
                        .iter()
                        .map(|(stack, v)| {
                            let (stack, verdict) = (stack.clone(), verdict_view(v));
                            view! {
                                <TableRow>
                                    <TableCell><span class="font-mono text-xs">{stack}</span></TableCell>
                                    <TableCell>{verdict}</TableCell>
                                </TableRow>
                            }
                        })
                        .collect_view();
                    Either::Right(view! {
                        {head.map(|h| view! {
                            <div class="mb-2">
                                <Notice status=Status::Off label="superseded">{h}</Notice>
                            </div>
                        })}
                        <TableWrapper><Table>
                            <TableHeader><TableRow>
                                <TableHead>"Stack"</TableHead>
                                <TableHead>"State"</TableHead>
                            </TableRow></TableHeader>
                            <TableBody>{rows}</TableBody>
                        </Table></TableWrapper>
                    })
                }
            };
            view! {
                <div class="mb-6">
                    <h2 class="text-sm font-medium mb-2 font-mono break-all">{title}</h2>
                    {body}
                </div>
            }
        })
        .collect_view();
    let empty = blocks.is_empty().then(|| {
        view! { <EmptyState>"Nothing to confirm: the request wrote no record (already in that state, or older than the store keeps)."</EmptyState> }
    });
    view! {
        <Card>
            <CardContent>
                {empty}
                {blocks_view}
            </CardContent>
        </Card>
    }
}
