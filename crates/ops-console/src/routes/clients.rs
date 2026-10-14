//! Client-store module — managing the LB's clients through the client
//! store's admin API. This slice carries the kill switch: `disable` and
//! `enable`, which deny or re-admit every key of a name on every LB instance
//! within a poll, whatever state the store's document is in.
//!
//! Writes follow the console's pattern: CSRF token and Origin check, then
//! the subject allowlist re-checked on every write, then POST-redirect-GET.
//! After a write the console shows, per stack, whether it is in effect, and
//! refreshes until it is or until it has said why not (`effect.rs`).
//!
//! No metrics-store value can stop or delay a restriction op. Metrics values
//! are untrusted input, so the handlers that send one never read the metrics
//! store first.

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

use crate::effect::{self, Carried, Op, RecordState, Report, ServedPart, Sight, Verdict, is_name};
use crate::error::AppError;
use crate::fleet::{self, Fleet};
use crate::routes::{clip, fmt_epoch, unavailable};
use crate::session::{Flash, Session, flash_redirect, take_flash};
use crate::state::{AppState, StoreModule};
use crate::store::{Answer, RestrictionOp, has_code, refusal_lines};
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
/// confirmation is shown only if the served part's tag did not move by the
/// second, so instances are never compared with a part the store had already
/// replaced.
struct Reading {
    store: Value,
    fleet: Fleet,
    tag_stable: bool,
}

fn part_tag(v: &Value) -> Option<&str> {
    v.pointer("/restrictions/part/tag").and_then(Value::as_str)
}

async fn read_all(m: &StoreModule) -> Result<Reading, AppError> {
    let store = m.client.read().await?;
    let fleet = fleet::read(&m.metrics).await;
    let again = m.client.read().await;
    let tag_stable = matches!(
        (part_tag(&store), again.as_ref().ok().and_then(part_tag)),
        (Some(a), Some(b)) if a == b
    );
    Ok(Reading {
        store,
        fleet,
        tag_stable,
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
    Some(Report {
        running: cf.get("running").and_then(Value::as_bool) == Some(true),
        set_hash: cf
            .pointer("/report/set_hash")
            .and_then(Value::as_str)
            .filter(|h| effect::is_hex64(h))
            .map(str::to_string),
    })
}

fn sights(store: &Value, fleet: &Fleet) -> Vec<(String, Sight)> {
    stacks_of(store)
        .into_iter()
        .map(|stack| {
            let sight = if stack == "cf" {
                Sight::Report(cf_report(store))
            } else {
                Sight::Instances(
                    fleet
                        .stacks
                        .clone()
                        .map(|m| m.get(&stack).cloned().unwrap_or_default()),
                )
            };
            (stack, sight)
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

fn not_configured() -> impl IntoView {
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"LB clients — module not configured"</CardTitle>
                <CardDescription>
                    "Set CLIENT_STORE_URL and CLIENT_STORE_TOKEN (both, or neither) on the console deployment, with the anthropic-lb module configured, to manage LB clients through the client store."
                </CardDescription>
            </CardHeader>
        </Card>
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
            {unavailable("client store", &e)}
            {kill_switch_card(&session.csrf, &prefill)}
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
    let content = kill_switch_card(&session.csrf, &prefill);
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
    page(TITLE, session, flash, content)
}

fn verdict_badge(v: &Verdict) -> (String, &'static str, String) {
    match v {
        Verdict::InEffect => (badge(BadgeKind::Success), "in effect", String::new()),
        Verdict::NotInEffect(why) => (badge(BadgeKind::Warning), "not in effect", why.clone()),
        Verdict::Unknown(why) => (badge(BadgeKind::Muted), "unknown", why.clone()),
    }
}

fn store_view(r: &Reading, csrf: &str, prefill: &str) -> impl IntoView + use<> {
    let part = served_part(&r.store);
    let sights = sights(&r.store, &r.fleet);
    let rows = table_records(&r.store);
    let unmatched = r.fleet.unmatched.clone().unwrap_or_default();
    view! {
        <div class="space-y-6">
            {warnings(r, &part)}
            {stacks_card(&part, &sights, r.tag_stable)}
            {restrictions_card(&rows, &part, &unmatched, csrf, prefill)}
            {clients_card(&r.store)}
        </div>
    }
}

/// Everything that should stop an operator reading the rest as healthy.
fn warnings(r: &Reading, part: &Result<ServedPart, String>) -> impl IntoView + use<> {
    let mut items: Vec<(String, String)> = Vec::new();
    let destructive = badge(BadgeKind::Destructive);
    let warning = badge(BadgeKind::Warning);
    match part {
        Err(e) => items.push((
            destructive.clone(),
            format!("Restrictions part unreadable: {e}. Nothing reads as in effect."),
        )),
        Ok(p) if !p.full => items.push((destructive.clone(), effect::PARTIAL.to_string())),
        Ok(_) => {}
    }
    match &r.fleet.behind {
        Ok(b) if !b.is_empty() => items.push((
            destructive.clone(),
            format!(
                "Restrictions behind on {}: those instances hold a disable or revoked digest the store lost. Re-send the name's disable (or enable), or reseed with restrictions_import.",
                b.iter().cloned().collect::<Vec<_>>().join(", ")
            ),
        )),
        Ok(_) => {}
        Err(e) => items.push((warning.clone(), format!("Restrictions-behind alerts unreadable (metrics store: {e})."))),
    }
    // A failed read of the readers' unmatched names hides the only flag on a
    // mistyped disable, which still reads as in effect: say it is missing.
    if let Err(e) = &r.fleet.unmatched {
        items.push((
            warning.clone(),
            format!("Unmatched disabled names unreadable (metrics store: {e}): a mistyped disable is not flagged."),
        ));
    }
    if stacks_of(&r.store).is_empty() {
        items.push((destructive.clone(), format!("{}.", capitalise(NO_STACKS))));
    }
    let doc = r.store.get("document");
    if doc.and_then(|d| d.get("status")).and_then(Value::as_str) != Some("ok") {
        let e = doc
            .and_then(|d| d.get("error"))
            .and_then(Value::as_str)
            .unwrap_or("no detail");
        items.push((
            warning.clone(),
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
            warning.clone(),
            format!(
                "The store refuses document writes: {}. Restriction ops still work.",
                reasons.join("; ")
            ),
        ));
    }
    if r.store.get("pitr_window").and_then(Value::as_bool) == Some(true) {
        items.push((warning.clone(), "A point-in-time recovery window is open: the store takes no document write but pitr. Restriction ops still work.".into()));
    }
    let list = items
        .into_iter()
        .map(|(class, text)| view! { <p class="text-sm"><span class=class>"warning"</span>" "{text}</p> })
        .collect_view();
    view! { <div class="space-y-2">{list}</div> }
}

fn stacks_card(
    part: &Result<ServedPart, String>,
    sights: &[(String, Sight)],
    tag_stable: bool,
) -> impl IntoView + use<> {
    let rows = sights
        .iter()
        .map(|(stack, sight)| {
            let v = match part {
                Ok(p) => effect::steady(effect::holds(p, sight), tag_stable),
                Err(e) => Verdict::Unknown(format!("the served part is unreadable: {e}")),
            };
            let (class, label, why) = verdict_badge(&v);
            let stack = stack.clone();
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{stack}</span></TableCell>
                    <TableCell><span class=class>{label}</span></TableCell>
                    <TableCell><span class="text-xs">{why}</span></TableCell>
                </TableRow>
            }
        })
        .collect_view();
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Stacks"</CardTitle>
                <CardDescription>
                    "Whether each stack holds the restrictions the store serves now: CF from its report to the store, every other stack from its instances' set hash in the metrics store. In effect only when every instance reports exactly the served set."
                </CardDescription>
            </CardHeader>
            <CardContent>
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Stack"</TableHead>
                        <TableHead>"Served restrictions"</TableHead>
                        <TableHead>"Why"</TableHead>
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
            let (state_class, state_text) = match r.state {
                Some(RecordState::Disabled) => (badge(BadgeKind::Destructive), "disabled"),
                Some(RecordState::Enabled) => (badge(BadgeKind::Success), "enabled"),
                None => (badge(BadgeKind::Muted), "unreadable"),
            };
            let flag = unmatched.get(&r.name).filter(|_| r.state == Some(RecordState::Disabled)).map(|stacks| {
                let text = format!("matches no client on {} — mistyped?", stacks.iter().cloned().collect::<Vec<_>>().join(", "));
                view! { <span class=badge(BadgeKind::Warning)>{text}</span> }
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
                    <TableCell><span class="font-mono text-xs">{name}</span>" "{flag}</TableCell>
                    <TableCell><span class=state_class>{state_text}</span></TableCell>
                    <TableCell><span class="tabular-nums whitespace-nowrap" title=version_title>{version_text}</span></TableCell>
                    <TableCell><span class="text-xs">{actor}</span></TableCell>
                    <TableCell><span class="text-xs">{reason}</span></TableCell>
                    <TableCell><span class="text-xs whitespace-nowrap">{at}</span></TableCell>
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
                    <a class="underline underline-offset-4" href="/lb/clients/disable">"The disable page"</a>
                    " loads without reading anything, for when the store or the metrics store is slow."
                </CardDescription>
            </CardHeader>
            <CardContent>
                {form}
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Name"</TableHead>
                        <TableHead>"State"</TableHead>
                        <TableHead>"Version (UTC)"</TableHead>
                        <TableHead>"Actor"</TableHead>
                        <TableHead>"Reason"</TableHead>
                        <TableHead>"At"</TableHead>
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
                        <span class="font-mono">{label}</span>
                    </label>
                }
            })
            .collect_view();
        view! {
            <div class="mt-3">
                <p class="text-sm mb-2">
                    <span class=badge(BadgeKind::Warning)>"disabled set full"</span>
                    " The store refuses a request that leaves more than 1,024 names disabled and more than it found. Pick names to enable in the same request:"
                </p>
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
            <div class="flex items-end gap-3">
                <div class="w-56">
                    <label class="text-xs text-muted-foreground" for="disable_name">"Name"</label>
                    <input class=INPUT_CLASS id="disable_name" name="name" value=prefill required maxlength="63" placeholder="client name" />
                </div>
                <div class="flex-1">
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
            <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" />
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
            <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" />
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
            <form method="post" action="/lb/clients/restrict" class="flex items-end gap-3">
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

/// The document's clients, read-only here, each with a shortcut to the
/// disable form.
fn clients_card(store: &Value) -> impl IntoView + use<> {
    let doc = store.get("document");
    let version = doc
        .and_then(|d| d.get("version"))
        .and_then(Value::as_u64)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "—".into());
    let clients: Vec<Value> = doc
        .and_then(|d| d.get("clients"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let rows = clients
        .iter()
        .map(|c| {
            let name = c.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            let budget = c
                .get("daily_budget")
                .and_then(Value::as_u64)
                .map(|b| crate::lb::fmt_tokens(b as f64))
                .unwrap_or_else(|| "—".into());
            let keys: Vec<String> = c
                .get("keys")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|k| {
                    let id = k.get("id").and_then(Value::as_str).unwrap_or("?");
                    let label = k.get("label").and_then(Value::as_str).unwrap_or("");
                    format!("{id} {label}").trim().to_string()
                })
                .collect();
            let keys = if keys.is_empty() { "—".to_string() } else { keys.join(", ") };
            let href = format!("/lb/clients/disable?disable={}", url::form_urlencoded::byte_serialize(name.as_bytes()).collect::<String>());
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{name}</span></TableCell>
                    <TableCell><span class="tabular-nums">{budget}</span></TableCell>
                    <TableCell><span class="text-xs font-mono">{keys}</span></TableCell>
                    <TableCell><a class="text-xs underline underline-offset-4" href=href>"disable…"</a></TableCell>
                </TableRow>
            }
        })
        .collect_view();
    view! {
        <Card>
            <CardHeader>
                <CardTitle>{format!("Clients (document version {version})")}</CardTitle>
                <CardDescription>"The store's document: the clients it grants, with their keys (ids and labels only)."</CardDescription>
            </CardHeader>
            <CardContent>
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Client"</TableHead>
                        <TableHead>"Daily budget"</TableHead>
                        <TableHead>"Keys"</TableHead>
                        <TableHead>""</TableHead>
                    </TableRow></TableHeader>
                    <TableBody>{rows}</TableBody>
                </Table></TableWrapper>
            </CardContent>
        </Card>
    }
}

// ─── Writes ─────────────────────────────────────────────────────────────────

/// Every store write passes here before anything else: the session's CSRF
/// token, then the subject allowlist as configured now. The allowlist is
/// otherwise read only at login, so without this a subject removed from it
/// would keep writing for the rest of its session. The Origin check has
/// already run, as middleware, before any handler.
pub fn guard_write(state: &AppState, session: &Session, csrf: &str) -> Result<(), AppError> {
    if let Err(e) = session.verify_csrf(csrf) {
        refused_locally(session, "CSRF token mismatch");
        return Err(e);
    }
    if !state.config.subject_allowed(&session.sub) {
        refused_locally(session, "subject no longer allowlisted");
        return Err(AppError::Forbidden(
            "this subject is no longer allowed to write".into(),
        ));
    }
    Ok(())
}

/// A write refused before anything was sent to the store, logged with the
/// subject on the write log like every other attempt.
fn refused_locally(session: &Session, why: &str) {
    tracing::warn!(
        target: "ops_console::write_log",
        sub = ?session.sub,
        kind = "restrictions",
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

/// One line per write attempt, refusals and upstream failures included, with
/// the user's subject. Request and answer are re-serialised and clipped.
fn write_log(session: &Session, kind: &str, request: &Value, outcome: Result<&Answer, &AppError>) {
    let request = clip(&request.to_string(), MAX_LOGGED_CHARS);
    match outcome {
        Ok(a) if a.status.is_success() => tracing::info!(
            target: "ops_console::write_log",
            sub = ?session.sub,
            kind,
            request = %request,
            status = a.status.as_u16(),
            answer = %clip(&a.body.to_string(), MAX_LOGGED_CHARS),
            "client-store write committed"
        ),
        Ok(a) => tracing::warn!(
            target: "ops_console::write_log",
            sub = ?session.sub,
            kind,
            request = %request,
            status = a.status.as_u16(),
            answer = %clip(&a.body.to_string(), MAX_LOGGED_CHARS),
            "client-store write refused"
        ),
        Err(e) => tracing::warn!(
            target: "ops_console::write_log",
            sub = ?session.sub,
            kind,
            request = %request,
            error = %e.detail(),
            "client-store write failed upstream"
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
    let Some(m) = state.store.as_ref() else {
        return Err(AppError::NotFound(
            "the client-store module is not configured".into(),
        ));
    };
    guard_write(&state, &session, &form.csrf)?;
    let secure = state.secure_cookies();
    // A refusal made here, before anything is sent, is an attempt too.
    let refuse = |jar: PrivateCookieJar, msg: String| -> Result<Response, AppError> {
        refused_locally(&session, &msg);
        Ok(flash_redirect(jar, secure, "error", msg, "/lb/clients"))
    };

    let name = form.name.trim().to_string();
    if !is_name(&name) {
        return refuse(
            jar,
            "A name is [a-z0-9] then [a-z0-9_-], at most 63 characters.".into(),
        );
    }
    let reason = match clean_reason(&form.reason) {
        Ok(r) => r,
        Err(e) => return refuse(jar, e),
    };
    let actor = actor_of(&session);
    let mut ops: Vec<RestrictionOp> = Vec::new();
    for pick in &form.also_enable {
        let parsed = pick
            .split_once(':')
            .and_then(|(n, v)| Some((n, v.parse::<u64>().ok()?)))
            .filter(|(n, _)| is_name(n));
        let Some((n, v)) = parsed else {
            return refuse(
                jar,
                "A name to enable in the same request is malformed.".into(),
            );
        };
        ops.push(RestrictionOp::Enable {
            name: n.to_string(),
            expected_version: v,
            actor: actor.clone(),
            reason: reason.clone(),
        });
    }
    let last = match form.action.as_str() {
        "disable" => RestrictionOp::Disable {
            name: name.clone(),
            actor: actor.clone(),
            reason: reason.clone(),
        },
        "enable" => match form.expected_version.trim().parse() {
            Ok(v) => RestrictionOp::Enable {
                name: name.clone(),
                expected_version: v,
                actor: actor.clone(),
                reason: reason.clone(),
            },
            Err(_) => {
                return refuse(
                    jar,
                    "enable needs the version of the record it read.".into(),
                );
            }
        },
        _ => return refuse(jar, "The action is disable or enable.".into()),
    };
    ops.push(last);
    if ops.len() > MAX_OPS {
        return refuse(jar, format!("A request carries at most {MAX_OPS} ops."));
    }

    let request = json!({ "restrictions": ops });
    let answer = m.client.restrict(&ops).await;
    write_log(&session, "restrictions", &request, answer.as_ref());
    let answer = match answer {
        Ok(a) => a,
        Err(e) => {
            let msg = unknown_outcome(
                m,
                &ops,
                &format!("the store's answer was lost ({})", e.detail()),
            )
            .await;
            return Ok(flash_redirect(jar, secure, "error", msg, "/lb/clients"));
        }
    };
    // The store answers 503 to any failure of the restrictions object's call,
    // a connection dropped after its commit included: that is a lost answer too.
    if answer.status == StatusCode::SERVICE_UNAVAILABLE
        && has_code(&answer.body, "store.unavailable")
    {
        let why = format!(
            "the store answered 503: {}",
            refusal_lines(&answer.body).join("; ")
        );
        let msg = unknown_outcome(m, &ops, &why).await;
        return Ok(flash_redirect(jar, secure, "error", msg, "/lb/clients"));
    }
    if answer.status == StatusCode::OK {
        let done = match written_ops(&answer.body, ops.len()) {
            Ok(done) => done,
            Err(e) => {
                // Committed, but which records it wrote, at which versions, is
                // unreadable: nothing can be confirmed or exported as committed.
                let mut msg = format!(
                    "The store committed the request, but its answer could not be read ({e}), \
                     so nothing is confirmed or exported as committed."
                );
                msg.push_str(&now_holds(m, &ops).await);
                return Ok(flash_redirect(jar, secure, "error", msg, "/lb/clients"));
            }
        };
        for op in &done {
            m.committed.record(&op.name, op.version);
        }
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
        return Ok(flash_redirect(jar, secure, "ok", msg, &effect_href(&done)));
    }
    let mut msg = format!(
        "The store refused the request ({}): {}.",
        answer.status.as_u16(),
        refusal_lines(&answer.body).join("; ")
    );
    if answer.status == StatusCode::CONFLICT {
        // A 409 means the console's picture was stale: read the store again
        // before reporting, and say what it holds now.
        msg.push_str(&now_holds(m, &ops).await);
    }
    let to = if has_code(&answer.body, "disabled.cap") {
        back_to(&name)
    } else {
        "/lb/clients".to_string()
    };
    Ok(flash_redirect(jar, secure, "error", msg, &to))
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
/// sent, in order. An `enable` of a name already enabled writes nothing and
/// returns the existing record, so there is nothing new to confirm or to
/// export as committed. An answer that cannot be read in full is an error,
/// never "nothing changed".
fn written_ops(answer: &Value, sent: usize) -> Result<Vec<Op>, String> {
    let results = answer
        .get("results")
        .and_then(Value::as_array)
        .ok_or("no results")?;
    if results.len() != sent {
        return Err(format!("{} results for {sent} ops", results.len()));
    }
    let mut done = Vec::new();
    for r in results {
        let op = (|| {
            Some(Op {
                name: r.get("name")?.as_str().filter(|n| is_name(n))?.to_string(),
                state: RecordState::parse(r.get("state")?.as_str()?)?,
                version: r.get("version")?.as_u64()?,
            })
        })();
        match (op, r.get("written").and_then(Value::as_bool)) {
            (Some(op), Some(true)) => done.push(op),
            (Some(_), Some(false)) => {}
            _ => return Err("a result is malformed".into()),
        }
    }
    Ok(done)
}

// ─── After a write: in effect per stack ─────────────────────────────────────

/// `/lb/clients/effect?op=<state>.<name>.<version>…&at=<epoch s>`. The ops
/// ride the URL, taken from the write's own result, so the page needs no
/// state of its own and survives a console restart.
fn effect_href(ops: &[Op]) -> String {
    let mut qs = url::form_urlencoded::Serializer::new(String::new());
    for op in ops {
        qs.append_pair(
            "op",
            &format!("{}.{}.{}", op.state.as_str(), op.name, op.version),
        );
    }
    qs.append_pair("at", &crate::session::now_epoch().to_string());
    format!("/lb/clients/effect?{}", qs.finish())
}

fn parse_effect_query(raw: Option<&str>) -> Result<(Vec<Op>, i64), AppError> {
    let mut ops = Vec::new();
    let mut at = 0;
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
                ops.push(op);
            }
            "at" => at = v.parse().unwrap_or(0),
            _ => {}
        }
    }
    if ops.len() > MAX_OPS {
        return Err(AppError::BadRequest(format!("at most {MAX_OPS} ops")));
    }
    Ok((ops, at))
}

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
    let (ops, at) = parse_effect_query(raw.as_deref())?;
    let now_micros = (crate::session::now_epoch().max(0) as u64) * 1_000_000;
    let ops: Vec<Op> = ops
        .into_iter()
        .filter(|op| !effect::aged_out(op, now_micros))
        .collect();
    let reading = match read_all(m).await {
        Ok(r) => r,
        Err(e) => {
            let content = unavailable("client store", &e);
            return Ok((jar, Html(page_html(&session, flash, content))).into_response());
        }
    };
    let part = served_part(&reading.store);
    let sights = sights(&reading.store, &reading.fleet);
    let judged: Vec<(Op, Result<effect::Judged, String>)> = ops
        .into_iter()
        .map(|op| {
            // An answer that names no stack would judge each op on an empty
            // table and read as settled: it is unreadable instead.
            let j = match &part {
                Err(e) => Err(e.clone()),
                Ok(_) if sights.is_empty() => Err(NO_STACKS.to_string()),
                Ok(p) => Ok(effect::judge_op(&op, p, &sights, reading.tag_stable)),
            };
            (op, j)
        })
        .collect();
    // Keep polling while some op can still become in effect on its own.
    let settling = judged.iter().any(|(_, j)| match j {
        Ok(j) => {
            !matches!(j.carried, Carried::Superseded(_))
                && j.stacks.iter().any(|(_, v)| *v != Verdict::InEffect)
        }
        Err(_) => true,
    });
    // `at` comes from the URL: a time in the future, or one so far back that
    // the subtraction would overflow, never keeps a page refreshing.
    let now = crate::session::now_epoch();
    let refresh = settling && at <= now && now.saturating_sub(at) < REFRESH_FOR_SECS;
    let content = view! {
        <div class="space-y-6">
            {warnings(&reading, &part)}
            {effect_card(&judged, refresh)}
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

fn effect_card(
    judged: &[(Op, Result<effect::Judged, String>)],
    refresh: bool,
) -> impl IntoView + use<> {
    let blocks = judged
        .iter()
        .map(|(op, j)| {
            let title = format!("{} {} — version {}", op.state.as_str(), op.name, op.version);
            let body = match j {
                Err(e) => Either::Left(view! { <p class="text-sm">"The served part is unreadable, so nothing is confirmed: "{e.clone()}</p> }),
                Ok(j) => {
                    let head = match &j.carried {
                        Carried::Superseded(v) => Some(format!("Superseded: a newer record of {} in the other state, at {v}, replaced this op. It is history, not in effect.", op.name)),
                        _ => None,
                    };
                    let rows = j
                        .stacks
                        .iter()
                        .map(|(stack, v)| {
                            let (class, label, why) = verdict_badge(v);
                            let stack = stack.clone();
                            view! {
                                <TableRow>
                                    <TableCell><span class="font-mono text-xs">{stack}</span></TableCell>
                                    <TableCell><span class=class>{label}</span></TableCell>
                                    <TableCell><span class="text-xs">{why}</span></TableCell>
                                </TableRow>
                            }
                        })
                        .collect_view();
                    Either::Right(view! {
                        {head.map(|h| view! { <p class="text-sm mb-2"><span class=badge(BadgeKind::Muted)>"superseded"</span>" "{h}</p> })}
                        <TableWrapper><Table>
                            <TableHeader><TableRow>
                                <TableHead>"Stack"</TableHead>
                                <TableHead>"State"</TableHead>
                                <TableHead>"Why"</TableHead>
                            </TableRow></TableHeader>
                            <TableBody>{rows}</TableBody>
                        </Table></TableWrapper>
                    })
                }
            };
            view! {
                <div class="mb-6">
                    <p class="text-sm font-medium mb-2 font-mono">{title}</p>
                    {body}
                </div>
            }
        })
        .collect_view();
    let note = if refresh {
        "Refreshing every 5 s until every stack is in effect (readers poll every 5 s; the metrics store trails by up to a minute)."
    } else {
        "Not refreshing: every op is settled, or two minutes have passed. Reload to check again."
    };
    let empty = judged.is_empty().then(|| {
        view! { <p class="text-sm">"Nothing to confirm: the request wrote no record (already in that state, or older than the store keeps)."</p> }
    });
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"In effect?"</CardTitle>
                <CardDescription>{note}</CardDescription>
            </CardHeader>
            <CardContent>
                {empty}
                {blocks}
                <p class="text-sm"><a class="underline underline-offset-4" href="/lb/clients">"Back to LB clients"</a></p>
            </CardContent>
        </Card>
    }
}
