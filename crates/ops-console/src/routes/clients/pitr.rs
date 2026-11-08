//! Point-in-time recovery of the document. `import`, the last resort, is a
//! document op and lives in `document`; the reseed of the restrictions lives
//! in `reseed`.
//!
//! Point-in-time recovery runs in the store. `pitr` leaves a recovery marker
//! and restarts the store's document object, which finishes or refuses the
//! recovery on its next start. The console's part is everything around that
//! call. It lists the documents to return to, from its own committed labels
//! and the store's history, and derives the restore time from when the
//! picked one was committed, never from its version, which can run ahead of
//! the clock. It shows the inputs that open the recovery window, and sends
//! `pitr` only once the open window restores to a time between the pick's
//! commit and the next. Then it reads the marker, and exports the recovery
//! commit as committed only once the operator accepts its diff, so the
//! unmatched-install alert fires on a recovery nobody looked at. No metrics
//! value can stop a `pitr`: an empty answer leaves only the store's history
//! to pick from, each pick behind a typed confirmation.

use axum::extract::{RawQuery, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    Block, Outcome, REFRESH_EVERY, REFRESH_FOR_SECS, Reading, actor_of, capitalise, clean_reason,
    effect_card, effect_href, guard, not_configured, outcome, page_html, read_all, refused_locally,
    refused_text, send_recorded, sights, store_of, sub_header,
};
use crate::committed::{self, Label};
use crate::effect::{self, Doc, is_hex64};
use crate::error::AppError;
use crate::routes::{clip, short_digest};
use crate::session::{Session, flash_redirect, now_epoch, take_flash};
use crate::state::{AppState, StoreModule};
use crate::store::{Answer, DocOp, DocRequest, has_code, refusal_lines};
use crate::ui::*;
use crate::writelog;

/// The write log's kinds for a pitr and its acceptance.
const PITR: &str = "pitr";
const PITR_ACCEPT: &str = "pitr_accept";

/// How long after the dispatch the window's deadline is suggested. The store
/// ignores a deadline more than an hour after its own deploy.
const UNTIL_MINUTES: u64 = 50;

// ─── Times ──────────────────────────────────────────────────────────────────

/// A UTC time as the store writes its `at` fields and its window's times,
/// `YYYY-MM-DDTHH:MM:SS[.fff]Z`, in Unix milliseconds. Anything else is
/// `None`: a time the console cannot read bounds no restore.
fn parse_ms(s: &str) -> Option<u64> {
    let b = s.as_bytes();
    let shape = b.len() >= 20
        && [(4, b'-'), (7, b'-'), (10, b'T'), (13, b':'), (16, b':')]
            .iter()
            .all(|&(i, c)| b[i] == c)
        && b.last() == Some(&b'Z');
    if !shape {
        return None;
    }
    let num = |from: usize, to: usize| -> Option<u32> {
        let part = s.get(from..to)?;
        part.bytes()
            .all(|c| c.is_ascii_digit())
            .then(|| part.parse().ok())?
    };
    let millis = match s.get(19..s.len() - 1)? {
        "" => 0,
        f if f.starts_with('.') && (2..=4).contains(&f.len()) => {
            let digits = &f[1..];
            let n: u32 = digits
                .bytes()
                .all(|c| c.is_ascii_digit())
                .then(|| digits.parse().ok())??;
            n * 10u32.pow(4 - f.len() as u32)
        }
        _ => return None,
    };
    let month = time::Month::try_from(u8::try_from(num(5, 7)?).ok()?).ok()?;
    let date = time::Date::from_calendar_date(
        i32::try_from(num(0, 4)?).ok()?,
        month,
        u8::try_from(num(8, 10)?).ok()?,
    )
    .ok()?;
    let clock = time::Time::from_hms_milli(
        u8::try_from(num(11, 13)?).ok()?,
        u8::try_from(num(14, 16)?).ok()?,
        u8::try_from(num(17, 19)?).ok()?,
        u16::try_from(millis).ok()?,
    )
    .ok()?;
    let t = time::PrimitiveDateTime::new(date, clock).assume_utc();
    u64::try_from(t.unix_timestamp_nanos() / 1_000_000).ok()
}

/// Unix milliseconds as the recovery window's inputs take them:
/// `YYYY-MM-DDTHH:MM:SS.mmmZ`, UTC.
fn iso_ms(ms: u64) -> String {
    let Ok(t) = time::OffsetDateTime::from_unix_timestamp_nanos(i128::from(ms) * 1_000_000) else {
        return "—".into();
    };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        t.year(),
        u8::from(t.month()),
        t.day(),
        t.hour(),
        t.minute(),
        t.second(),
        t.millisecond()
    )
}

// ─── Targets ────────────────────────────────────────────────────────────────

/// A document to return to.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Target {
    version: u64,
    digest: String,
    /// When the store committed it, as the earliest and latest Unix ms it
    /// can have been. A history row's `at` is exact. A committed label's
    /// sample is the console's clock when it saw the commit, in whole
    /// seconds, so it is taken as a second either side.
    committed: (u64, u64),
    /// Among the console's committed labels.
    labelled: bool,
    /// Among the store's history rows.
    in_history: bool,
}

impl Target {
    fn key(&self) -> String {
        format!("{}.{}", self.version, self.digest)
    }
}

/// The documents to return to, oldest commit first: the console's committed
/// labels over 30 days and the store's history rows, merged on version and
/// digest. A history row an edit rewrote matches no label, so it stands
/// apart, found only in history.
fn targets(labels: &[Label], store: &Value) -> Vec<Target> {
    let mut out: Vec<Target> = Vec::new();
    let history = store
        .pointer("/document/history")
        .and_then(Value::as_array)
        .into_iter()
        .flatten();
    for row in history {
        let version = row.get("version").and_then(Value::as_u64);
        let digest = row
            .get("digest")
            .and_then(Value::as_str)
            .filter(|d| is_hex64(d));
        let at = row.get("at").and_then(Value::as_str).and_then(parse_ms);
        let (Some(version), Some(digest), Some(at)) = (version, digest, at) else {
            continue;
        };
        if out
            .iter()
            .any(|t| t.version == version && t.digest == digest)
        {
            continue;
        }
        out.push(Target {
            version,
            digest: digest.to_string(),
            committed: (at, at),
            labelled: false,
            in_history: true,
        });
    }
    // `committed::read` keeps only labels whose sample is a time the console
    // could have seen, so this arithmetic cannot overflow.
    for l in labels {
        match out
            .iter_mut()
            .find(|t| t.version == l.version && t.digest == l.digest)
        {
            Some(t) => t.labelled = true,
            None => out.push(Target {
                version: l.version,
                digest: l.digest.clone(),
                committed: ((l.secs - 1) * 1000, (l.secs + 1) * 1000),
                labelled: true,
                in_history: false,
            }),
        }
    }
    out.sort_by_key(|t| (t.committed.0, t.version));
    out
}

/// The restore times that return to a pick: after its commit, and before the
/// next commit when one is known.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interval {
    after: u64,
    before: Option<u64>,
}

impl Interval {
    fn contains(&self, ms: u64) -> bool {
        ms > self.after && self.before.is_none_or(|b| ms < b)
    }

    /// The restore time to dispatch: midway to the next commit, so a
    /// bookmark the platform takes only approximately still lands on the
    /// pick; a second after the commit when no later commit is known, since
    /// the wipe or edit being recovered from came later.
    fn suggested(&self) -> u64 {
        match self.before {
            Some(b) => self.after + (b - self.after) / 2,
            None => self.after + 1000,
        }
    }
}

/// The pick's interval, or `None` when the next commit is too close to it
/// for a time to fall between them. The next commit is the earliest of the
/// later versions: versions order commits, and a history row an edit
/// rewrote keeps its version and time, so it is never the pick's next. The
/// labels count too, since a commit a recovery rewound away has left the
/// history but not the store's past. A later version found only in the
/// labels, which are untrusted input like every metrics value, narrows the
/// interval but never shuts it: no metrics value can stop a pitr.
fn interval(targets: &[Target], i: usize) -> Option<Interval> {
    let pick = &targets[i];
    let after = pick.committed.1;
    let before = targets
        .iter()
        .filter(|n| n.version > pick.version && (n.in_history || n.committed.0 > after + 1))
        .map(|n| n.committed.0)
        .min();
    match before {
        Some(b) if b <= after + 1 => None,
        _ => Some(Interval { after, before }),
    }
}

/// The recovery window, as the store's document side reports it.
struct Window {
    open: bool,
    usable: bool,
    problem: Option<String>,
    restore_to: Option<u64>,
    closes: Option<String>,
}

fn window(store: &Value) -> Option<Window> {
    let w = store
        .get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))?
        .get("pitr_window")?;
    let s = |k: &str| w.get(k).and_then(Value::as_str).map(str::to_string);
    Some(Window {
        open: w.get("open").and_then(Value::as_bool) == Some(true),
        usable: w.get("usable").and_then(Value::as_bool) == Some(true),
        problem: s("problem"),
        restore_to: s("restore_to").as_deref().and_then(parse_ms),
        closes: s("closes"),
    })
}

/// Why `pitr` cannot be sent for this pick now, or `None` when it can: the
/// window must be open and usable, and restore to a time inside the pick's
/// interval, or the restore lands on another document and the store refuses
/// the recovery.
fn window_fault(w: Option<&Window>, iv: Option<Interval>) -> Option<String> {
    let Some(iv) = iv else {
        return Some(
            "The next commit is too close to this one for a restore time to fall between them: \
             pick another document."
                .into(),
        );
    };
    let Some(w) = w else {
        return Some(
            "The store's document side did not answer, so the recovery window cannot be read."
                .into(),
        );
    };
    if !w.open {
        return Some("No recovery window is open: dispatch it, then reload this page.".into());
    }
    if !w.usable {
        let why = w
            .problem
            .clone()
            .unwrap_or_else(|| "no reason given".into());
        return Some(format!(
            "The open recovery window cannot be used: {}.",
            clip(&why, 200)
        ));
    }
    match w.restore_to {
        Some(t) if iv.contains(t) => None,
        Some(t) => Some(format!(
            "The open window restores to {}, outside this pick's restore times: dispatch the window \
             again with the pitr_at below.",
            iso_ms(t)
        )),
        None => Some("The open window names no restore time the console can read.".into()),
    }
}

/// `target=<version>.<digest>`, as the pick links carry it.
fn parse_target(s: &str) -> Option<(u64, String)> {
    let (v, d) = s.trim().split_once('.')?;
    let d = d.to_ascii_lowercase();
    Some((v.parse().ok()?, d)).filter(|(_, d)| is_hex64(d))
}

fn query_param(raw: Option<&str>, key: &str) -> Option<String> {
    url::form_urlencoded::parse(raw.unwrap_or_default().as_bytes())
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.into_owned())
}

/// The version every recovery form pins: the store's addressable version, 0
/// when it holds none.
fn pinned_version(store: &Value) -> Result<u64, String> {
    effect::served_doc(store).map(|d| d.map(|d| d.version).unwrap_or(0))
}

// ─── The PITR page ──────────────────────────────────────────────────────────

const PITR_ABOUT: &str = "Rewinds the store's document to one the console committed, after a wipe, a rollback or an edit made outside the API, up to 30 days back. It never touches the restrictions, so no disable is lifted. Pick the document to return to; the console derives the restore time from when it was committed, never from its version. Open the recovery window with the inputs shown, then send pitr. The store finishes or refuses the recovery itself.";

pub async fn pitr_page(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    RawQuery(raw): RawQuery,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))));
    };
    let header = sub_header("Point-in-time recovery".into(), PITR_ABOUT.into());
    let (store, labels) = tokio::join!(m.client.read(), committed::read(&m.metrics));
    let store = match store {
        Ok(s) => s,
        Err(e) => {
            let content = view! {
                <div class="space-y-6">{header}{unavailable("client store", e.detail())}</div>
            };
            return Ok((jar, Html(page_html(&session, flash, content))));
        }
    };
    let pick = query_param(raw.as_deref(), "target").and_then(|t| parse_target(&t));
    let content = view! {
        <div class="space-y-6">
            {header}
            {recovery_state(&store)}
            {targets_card(&store, &labels, pick.as_ref())}
            {pick.map(|p| pick_card(&store, &labels, &p, &session.csrf))}
        </div>
    };
    Ok((jar, Html(page_html(&session, flash, content))))
}

/// The store's state as a recovery needs it: its version, why it refuses
/// writes, the window, and every recovery marker that is not done.
fn recovery_state(store: &Value) -> impl IntoView + use<> {
    let mut lines: Vec<(Status, String)> = Vec::new();
    match pinned_version(store) {
        Ok(v) => lines.push((
            Status::Off,
            format!("The store is at document version {v}."),
        )),
        Err(e) => lines.push((
            Status::Error,
            format!("The store's document cannot be read: {}.", clip(&e, 160)),
        )),
    }
    let doc = store.get("document");
    for f in doc
        .and_then(|d| d.get("frozen"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        lines.push((
            Status::Warn,
            format!("It refuses document writes: {}.", clip(f, 160)),
        ));
    }
    if doc
        .and_then(|d| d.get("recovery_refused"))
        .and_then(Value::as_bool)
        == Some(true)
    {
        lines.push((
            Status::Error,
            "A recovery was refused: the store serves no document to readers until an import or another recovery commits.".into(),
        ));
    }
    match window(store) {
        Some(w) if w.open => {
            let to = w
                .restore_to
                .map(iso_ms)
                .unwrap_or_else(|| "a time it cannot read".into());
            let closes = w.closes.clone().unwrap_or_else(|| "—".into());
            let what = if w.usable {
                format!("A recovery window is open: it restores to {to} and closes at {closes}.")
            } else {
                format!(
                    "A recovery window is open but cannot be used: {}.",
                    clip(w.problem.as_deref().unwrap_or("no reason given"), 200)
                )
            };
            lines.push((Status::Warn, what));
        }
        Some(_) => lines.push((Status::Off, "No recovery window is open.".into())),
        None => {}
    }
    for m in store
        .get("recoveries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let s = |k: &str| clip(m.get(k).and_then(Value::as_str).unwrap_or(""), 200);
        let id = s("id");
        let line = format!(
            "Recovery {} is {}{}",
            id,
            s("state"),
            m.get("reason")
                .and_then(Value::as_str)
                .map(|r| format!(": {}", clip(r, 200)))
                .unwrap_or_default()
        );
        lines.push((Status::Warn, line));
    }
    let items = lines
        .into_iter()
        .map(|(status, text)| view! { <Notice status=status label="store">{text}</Notice> })
        .collect_view();
    view! {
        <Card>
            <CardHeader><CardTitle>"The store"</CardTitle></CardHeader>
            <CardContent><div class="space-y-2">{items}</div></CardContent>
        </Card>
    }
}

fn targets_card(
    store: &Value,
    labels: &Result<Vec<Label>, String>,
    pick: Option<&(u64, String)>,
) -> impl IntoView + use<> {
    let list = targets(labels.as_deref().unwrap_or_default(), store);
    let labels_note = match labels {
        Ok(l) if l.is_empty() => Some(
            "The metrics store shows no committed label from the last 30 days: only the store's history is listed, and each pick needs its version typed."
                .to_string(),
        ),
        Ok(_) => None,
        Err(e) => Some(format!(
            "The committed labels could not be read ({}): only the store's history is listed, and each pick needs its version typed.",
            clip(e, 200)
        )),
    }
    .map(|n| view! { <div class="mb-3"><Notice status=Status::Warn label="labels">{n}</Notice></div> });
    let rows = list
        .iter()
        .rev()
        .map(|t| {
            let (status, source) = match (t.labelled, t.in_history) {
                (true, true) => (Status::Ok, "console and history"),
                (true, false) => (Status::Ok, "console"),
                _ => (Status::Warn, "history only: typed confirmation"),
            };
            let picked = pick.is_some_and(|(v, d)| *v == t.version && *d == t.digest);
            let href = format!("/lb/clients/pitr?target={}#pick", t.key());
            let action = if picked {
                view! { <StatusBadge status=Status::Ok label="picked" /> }.into_any()
            } else {
                view! { <Link href=href>"Pick…"</Link> }.into_any()
            };
            let (when, exact) = (iso_ms(t.committed.0), t.version.to_string());
            let (short, digest) = (short_digest(&t.digest), t.digest.clone());
            view! {
                <TableRow>
                    <TableCell><span class="tabular-nums">{exact}</span></TableCell>
                    <TableCell secondary=true><span class="font-mono text-xs" title=digest>{short}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs whitespace-nowrap">{when}</span></TableCell>
                    <TableCell><StatusBadge status=status label=source /></TableCell>
                    <TableCell>{action}</TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let body = if list.is_empty() {
        view! { <EmptyState>"Nothing to return to: neither the committed labels nor the store's history name a document. Import is the way back."</EmptyState> }.into_any()
    } else {
        view! {
            <TableWrapper><Table>
                <TableHeader><TableRow>
                    <TableHead>"Version"</TableHead>
                    <TableHead secondary=true>"Digest"</TableHead>
                    <TableHead secondary=true>"Committed (UTC)"</TableHead>
                    <TableHead>"From"</TableHead>
                    <TableHead>""</TableHead>
                </TableRow></TableHeader>
                <TableBody>{rows}</TableBody>
            </Table></TableWrapper>
        }
        .into_any()
    };
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Documents to return to"</CardTitle>
                <CardDescription>"The documents the console committed in the last 30 days, from its committed labels, and the store's newest history rows, newest first. A row found only in history could have been rewritten by an edit, so its pick needs its version typed."</CardDescription>
            </CardHeader>
            <CardContent>{labels_note}{body}</CardContent>
        </Card>
    }
}

fn pick_card(
    store: &Value,
    labels: &Result<Vec<Label>, String>,
    (version, digest): &(u64, String),
    csrf: &str,
) -> impl IntoView + use<> {
    let list = targets(labels.as_deref().unwrap_or_default(), store);
    let Some(i) = list
        .iter()
        .position(|t| t.version == *version && t.digest == *digest)
    else {
        return view! {
            <Card><CardContent>
                <Notice status=Status::Error label="pick">"That document is not among the documents to return to: pick one from the list."</Notice>
            </CardContent></Card>
        }
        .into_any();
    };
    let t = &list[i];
    let iv = interval(&list, i);
    let w = window(store);
    let what = format!(
        "Return to version {} (digest {}), committed at {}.",
        t.version,
        short_digest(&t.digest),
        iso_ms(t.committed.0)
    );
    let times = match iv {
        Some(Interval {
            after,
            before: Some(b),
        }) => format!(
            "The restore time must fall after {} and before the next commit, at {}, and before the wipe or edit being recovered from if that came first.",
            iso_ms(after),
            iso_ms(b)
        ),
        Some(Interval {
            after,
            before: None,
        }) => format!(
            "The restore time must fall after {}. No later commit is known, so it must also fall before the wipe or edit being recovered from.",
            iso_ms(after)
        ),
        None => "The next commit is too close to this one for a restore time to fall between them."
            .into(),
    };
    let dispatch = iv.map(|iv| {
        let until = (now_epoch().max(0) as u64 + UNTIL_MINUTES * 60) * 1000;
        let inputs = format!(
            "-f pitr_at={} -f pitr_until={}",
            iso_ms(iv.suggested()),
            iso_ms(until)
        );
        view! {
            <div class="space-y-1">
                <p class="text-sm">"Open the window: dispatch the store's deploy workflow on main with these inputs. The point-in-time recovery runbook has the command, and the window lapses within the hour."</p>
                <pre class="font-mono text-xs break-all whitespace-pre-wrap rounded-md border p-2"><code>{inputs}</code></pre>
            </div>
        }
    });
    let action = match (window_fault(w.as_ref(), iv), pinned_version(store)) {
        (Some(why), _) => {
            view! { <Notice status=Status::Warn label="window">{why}</Notice> }.into_any()
        }
        (None, Err(e)) => {
            let why = format!(
                "The store's document cannot be read ({}), so no pitr can be pinned to its version.",
                clip(&e, 160)
            );
            view! { <Notice status=Status::Error label="store">{why}</Notice> }.into_any()
        }
        (None, Ok(expected)) => send_form(csrf, t, expected).into_any(),
    };
    view! {
        <Card>
            <CardHeader><CardTitle>"The pick"</CardTitle></CardHeader>
            <CardContent>
                <div class="space-y-3" id="pick">
                    <p class="text-sm">{what}</p>
                    <p class="text-sm">{times}</p>
                    {dispatch}
                    {action}
                </div>
            </CardContent>
        </Card>
    }
    .into_any()
}

fn send_form(csrf: &str, t: &Target, expected: u64) -> impl IntoView + use<> {
    let confirm = (!t.labelled).then(|| {
        let warning = format!(
            "Version {} is found only in the store's history, which an edit could have rewritten.",
            t.version
        );
        let label = format!("Type {} to confirm.", t.version);
        view! {
            <div class="space-y-1">
                <Notice status=Status::Error label="confirm">{warning}</Notice>
                <label class="text-xs text-muted-foreground" for="confirm_version">{label}</label>
                <input class=INPUT_CLASS id="confirm_version" name="confirm_version" required autocomplete="off" />
            </div>
        }
    });
    let (csrf, target, expected) = (csrf.to_string(), t.key(), expected.to_string());
    view! {
        <form method="post" action="/lb/clients/pitr" class="space-y-3">
            <input type="hidden" name="csrf" value=csrf />
            <input type="hidden" name="target" value=target />
            <input type="hidden" name="expected_version" value=expected />
            {confirm}
            <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" />
            <button type="submit" class=btn(Btn::Destructive)>"Send pitr"</button>
        </form>
    }
}

// ─── Sending pitr ───────────────────────────────────────────────────────────

#[derive(Deserialize, Default)]
pub struct PitrForm {
    #[serde(default)]
    csrf: String,
    #[serde(default)]
    target: String,
    #[serde(default)]
    expected_version: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    confirm_version: String,
}

pub async fn pitr_send(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<PitrForm>,
) -> Result<Response, AppError> {
    let m = store_of(&state)?;
    guard(&state, &session, &form.csrf, PITR)?;
    let secure = state.secure_cookies();
    // Back to the pick only as parsed: the raw field could carry anything,
    // and a redirect's location must be a valid header value.
    let back_to = match parse_target(&form.target) {
        Some((v, d)) => format!("/lb/clients/pitr?target={v}.{d}#pick"),
        None => "/lb/clients/pitr".to_string(),
    };
    let back = |jar, msg: String, to: &str| flash_redirect(jar, secure, "error", msg, to);
    let refuse = |jar, msg: String| {
        refused_locally(&session, PITR, &msg);
        back(jar, msg, &back_to)
    };
    let Some((version, digest)) = parse_target(&form.target) else {
        return Ok(refuse(
            jar,
            "The form names no document to return to.".into(),
        ));
    };
    let Ok(expected) = form.expected_version.trim().parse::<u64>() else {
        return Ok(refuse(
            jar,
            "The form carries no document version: reload the page.".into(),
        ));
    };
    let reason = match clean_reason(&form.reason) {
        Ok(r) => r,
        Err(msg) => return Ok(refuse(jar, msg)),
    };
    // Everything is read again: a page is never trusted to still be true.
    let (store, labels) = tokio::join!(m.client.read(), committed::read(&m.metrics));
    let store = match store {
        Ok(s) => s,
        Err(e) => {
            return Ok(refuse(
                jar,
                format!("The store could not be read: {}.", e.detail()),
            ));
        }
    };
    let list = targets(labels.as_deref().unwrap_or_default(), &store);
    let Some(i) = list
        .iter()
        .position(|t| t.version == version && t.digest == digest)
    else {
        return Ok(refuse(
            jar,
            "That document is no longer among the documents to return to: pick again.".into(),
        ));
    };
    if let Some(why) = window_fault(window(&store).as_ref(), interval(&list, i)) {
        return Ok(refuse(jar, format!("Nothing was sent. {why}")));
    }
    if !list[i].labelled && form.confirm_version.trim() != version.to_string() {
        return Ok(refuse(
            jar,
            format!(
                "Version {version} is found only in the store's history. Nothing was sent: type \
                 {version} to confirm."
            ),
        ));
    }
    let actor = actor_of(&session);
    let ops = [DocOp::Pitr {
        expected_restore_digest: digest,
    }];
    let body = json!({ "document": DocRequest {
        expected_version: expected,
        actor: &actor,
        reason: &reason,
        dry_run: false,
        ops: &ops,
    } });
    // Nothing is exported as committed here: the recovery commit is exported
    // only once the operator accepts its diff.
    let answer = send_recorded(m, &session.sub, PITR, body, |_, _| {}).await;
    let (kind, msg, to) = pitr_outcome(m, answer).await;
    let to = to.unwrap_or(back_to);
    Ok(flash_redirect(jar, secure, kind, msg, &to))
}

/// What the page says after `pitr`, and where it goes: the marker's page on
/// a 202, the pick otherwise. A `pitr` ends with the store's document object
/// restarting, so its call answers 202 with the marker it left.
async fn pitr_outcome(
    m: &StoreModule,
    answer: Result<Answer, AppError>,
) -> (&'static str, String, Option<String>) {
    if let Ok(a) = &answer
        && a.status == StatusCode::ACCEPTED
    {
        let id = a
            .body
            .pointer("/marker/id")
            .and_then(Value::as_str)
            .filter(|id| is_marker_id(id));
        return match id {
            Some(id) => (
                "ok",
                format!("pitr sent: the store restarted to finish or refuse recovery {id}. Reading its marker…"),
                Some(marker_href(id)),
            ),
            None => (
                "error",
                "The store accepted the pitr, but its answer names no marker: read the markers on this page before sending pitr again.".into(),
                None,
            ),
        };
    }
    // The store's two 503s that say no pitr ran: send it again.
    if let Ok(a) = &answer
        && a.status == StatusCode::SERVICE_UNAVAILABLE
        && (has_code(&a.body, "pitr.failed") || has_code(&a.body, "store.unavailable"))
    {
        let msg = format!(
            "Nothing was scheduled ({}). Send pitr again while the window is open.",
            refusal_lines(&a.body).join("; ")
        );
        return ("error", msg, None);
    }
    let msg = match outcome(answer) {
        Outcome::Unknown(why) => format!(
            "Whether the pitr was scheduled is unknown: {why}. Read the markers on this page before sending it again."
        ),
        Outcome::Accepted(_) => {
            "The store answered the pitr with a 200, which a pitr never gets: read the markers on this page before sending it again.".into()
        }
        Outcome::Refused { status, body } => {
            let mut msg = refused_text("The store refused the pitr", status, &body);
            if status == StatusCode::CONFLICT {
                msg.push_str(&super::document::doc_now(m).await);
            }
            msg
        }
    };
    ("error", msg, None)
}

/// A recovery id: 128 random bits, in lowercase hex.
fn is_marker_id(s: &str) -> bool {
    s.len() == 32
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

fn marker_href(id: &str) -> String {
    format!("/lb/clients/pitr/marker?id={id}&at={}", now_epoch())
}

// ─── The marker ─────────────────────────────────────────────────────────────

const MARKER_ABOUT: &str = "How a point-in-time recovery ended. The store judges its marker when its document object next starts: done means the restore landed on the picked document and the store committed it, failed that nothing was restored, refused that the restore landed elsewhere.";

/// A history row of a recovery: the recovery commit, and any follow-up
/// commit of the same recovery that merged a digest taken meanwhile.
#[derive(Clone, Debug, PartialEq, Eq)]
struct RecoveryRow {
    version: u64,
    digest: String,
    role: String,
    diff: Vec<Value>,
    /// When the store committed it, in Unix ms; `None` when its row names no
    /// time the console can read.
    at: Option<u64>,
}

/// The rows a recovery committed, oldest first, and the row before them:
/// the document the restore landed on. History lists rows newest first.
fn recovery_rows(store: &Value, id: &str) -> (Vec<RecoveryRow>, Option<(u64, String)>) {
    let rows: Vec<&Value> = store
        .pointer("/document/history")
        .and_then(Value::as_array)
        .map(|r| r.iter().collect())
        .unwrap_or_default();
    let mine: Vec<usize> = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.get("recovery_id").and_then(Value::as_str) == Some(id))
        .map(|(i, _)| i)
        .collect();
    let out: Vec<RecoveryRow> = mine
        .iter()
        .rev()
        .filter_map(|&i| {
            let r = rows[i];
            Some(RecoveryRow {
                version: r.get("version")?.as_u64()?,
                digest: r
                    .get("digest")?
                    .as_str()
                    .filter(|d| is_hex64(d))?
                    .to_string(),
                role: r
                    .get("role")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
                diff: r
                    .get("diff")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default(),
                at: r.get("at").and_then(Value::as_str).and_then(parse_ms),
            })
        })
        .collect();
    let restored = mine.last().and_then(|&i| rows.get(i + 1)).and_then(|r| {
        Some((
            r.get("version")?.as_u64()?,
            r.get("digest")?.as_str()?.to_string(),
        ))
    });
    (out, restored)
}

fn find_marker<'a>(store: &'a Value, id: &str) -> Result<Option<&'a Value>, String> {
    let restrictions = store.get("restrictions");
    if restrictions
        .and_then(|r| r.get("status"))
        .and_then(Value::as_str)
        != Some("ok")
    {
        return Err(
            "the restrictions side of the store did not answer, so its markers cannot be read"
                .into(),
        );
    }
    Ok(restrictions
        .and_then(|r| r.get("markers"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find(|m| m.get("id").and_then(Value::as_str) == Some(id)))
}

pub async fn pitr_marker(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    RawQuery(raw): RawQuery,
) -> Result<Response, AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))).into_response());
    };
    let id = query_param(raw.as_deref(), "id")
        .filter(|id| is_marker_id(id))
        .ok_or_else(|| AppError::BadRequest("malformed recovery id".into()))?;
    let at: i64 = query_param(raw.as_deref(), "at")
        .and_then(|a| a.parse().ok())
        .unwrap_or(0);
    let header = sub_header("Recovery".into(), MARKER_ABOUT.into());
    let reading = match read_all(m).await {
        Ok(r) => r,
        Err(e) => {
            let content = view! {
                <div class="space-y-6">{header}{unavailable("client store", e.detail())}</div>
            };
            return Ok((jar, Html(page_html(&session, flash, content))).into_response());
        }
    };
    let rewound = rewound(m, &reading.store, &id).await;
    let (body, settling) = marker_view(&reading, &id, rewound.as_ref(), &session.csrf);
    let now = now_epoch();
    let refresh = settling && at <= now && now.saturating_sub(at) < REFRESH_FOR_SECS;
    let content = view! { <div class="space-y-6">{header}{body}</div> };
    let mut resp = (jar, Html(page_html(&session, flash, content))).into_response();
    if refresh {
        resp.headers_mut()
            .insert(header::REFRESH, HeaderValue::from_static(REFRESH_EVERY));
    }
    Ok(resp)
}

/// What a finished recovery's rewind changed: the store's bytes for its
/// newest commit, diffed against the newest document the console committed.
/// The recovery's own rows cannot show it: each diffs against the document
/// the restore landed on. `None` unless the marker is done.
async fn rewound(m: &StoreModule, store: &Value, id: &str) -> Option<Against> {
    let done = find_marker(store, id)
        .ok()
        .flatten()
        .and_then(|marker| marker.get("state"))
        .and_then(Value::as_str)
        == Some("done");
    if !done {
        return None;
    }
    let (rows, _) = recovery_rows(store, id);
    let newest = rows.last()?;
    let text = store.pointer("/document/text").and_then(Value::as_str);
    Some(match text {
        Some(t) if serves(store, newest) && writelog::digest_of(t) == newest.digest => {
            super::document::against_newest(m, t).await
        }
        _ => Err(
            "the store no longer serves the recovery's commit, or sent no bytes that hash to it"
                .into(),
        ),
    })
}

/// A document's diff against the newest document the console committed, or
/// why there is none.
type Against = Result<(writelog::Logged, Vec<Value>), String>;

/// Whether the store's document is this recovery row's.
fn serves(store: &Value, row: &RecoveryRow) -> bool {
    let doc = store.get("document");
    let current = (
        doc.and_then(|d| d.get("version")).and_then(Value::as_u64),
        doc.and_then(|d| d.get("digest")).and_then(Value::as_str),
    );
    current == (Some(row.version), Some(row.digest.as_str()))
}

/// The marker's page body, and whether it may still change on its own.
fn marker_view(r: &Reading, id: &str, rewound: Option<&Against>, csrf: &str) -> (AnyView, bool) {
    let marker = match find_marker(&r.store, id) {
        Err(why) => {
            let why = format!("{}: reading again.", capitalise(&why));
            return (
                view! { <Notice status=Status::Warn label="marker">{why}</Notice> }.into_any(),
                true,
            );
        }
        Ok(None) => {
            return (
                view! { <Notice status=Status::Error label="marker">"The store holds no recovery marker with this id."</Notice> }.into_any(),
                false,
            );
        }
        Ok(Some(m)) => m,
    };
    let s = |k: &str| clip(marker.get(k).and_then(Value::as_str).unwrap_or(""), 300);
    let reason = s("reason");
    let picked = s("expected_restore_digest");
    let head = format!(
        "Recovery {id}, to the document with digest {}, started {}.",
        short_digest(&picked),
        s("created_at")
    );
    let head = view! { <p class="text-sm">{head}</p> };
    match s("state").as_str() {
        "pending" => {
            let doc_down =
                r.store.pointer("/document/status").and_then(Value::as_str) != Some("ok");
            let why = if doc_down {
                "Pending: the store's document object could not start, so it has not judged the marker yet. Reading again."
            } else {
                "Pending: the store judges the marker when its document object next starts, which this read starts. Reading again."
            };
            (
                view! { {head} <Notice status=Status::Warn label="pending">{why}</Notice> }
                    .into_any(),
                true,
            )
        }
        "failed" => {
            let why = format!(
                "Failed: nothing was restored ({reason}). Send pitr again while the window is open, or close the window and import."
            );
            (view! { {head} <Notice status=Status::Error label="failed">{why}</Notice> <Link href="/lb/clients/pitr">"Back to point-in-time recovery"</Link> }.into_any(), false)
        }
        "refused" => {
            let why = format!(
                "Refused: {reason}. The store serves no document to readers until an import or another recovery commits. Dispatch the window again with another restore time and send pitr again, or close the window and import."
            );
            (view! { {head} <Notice status=Status::Error label="refused">{why}</Notice> <Link href="/lb/clients/pitr">"Back to point-in-time recovery"</Link> }.into_any(), false)
        }
        "done" => (
            done_view(r, id, &picked, rewound, csrf, head.into_any()),
            false,
        ),
        other => {
            let why = format!("The marker is in a state the console does not know: {other}.");
            (
                view! { {head} <Notice status=Status::Error label="marker">{why}</Notice> }
                    .into_any(),
                false,
            )
        }
    }
}

/// A finished recovery: the commits it made with their diffs, whether the
/// document it restored is the one picked, how each stack holds the
/// newest, and the form that accepts it.
fn done_view(
    r: &Reading,
    id: &str,
    picked: &str,
    rewound: Option<&Against>,
    csrf: &str,
    head: AnyView,
) -> AnyView {
    let (rows, restored) = recovery_rows(&r.store, id);
    let Some(newest) = rows.last().cloned() else {
        return view! {
            {head}
            <Notice status=Status::Warn label="done">"Done, but the recovery's commit is not among the store's newest history rows, so its diff cannot be shown or accepted here."</Notice>
        }
        .into_any();
    };
    let restored_note = match restored {
        Some((v, d)) if d == picked => (
            Status::Ok,
            format!(
                "The restore landed on the picked document: version {v}, digest {}.",
                short_digest(&d)
            ),
        ),
        Some((v, d)) => (
            Status::Error,
            format!(
                "The row before the recovery commit is version {v}, digest {}, not the picked document: compare the diffs with care.",
                short_digest(&d)
            ),
        ),
        None => (
            Status::Warn,
            "The row before the recovery commit is not among the store's newest history rows."
                .into(),
        ),
    };
    let (status, text) = restored_note;
    let commits = rows
        .iter()
        .map(|row| {
            let title = format!(
                "version {} — digest {} ({})",
                row.version,
                short_digest(&row.digest),
                if row.role == "recovery" {
                    "the recovery commit"
                } else {
                    "a follow-up commit that merged a revoked digest"
                }
            );
            let lines = super::document::diff_lines(&row.diff)
                .into_iter()
                .map(|l| view! { <li class="font-mono text-xs break-all">{l}</li> })
                .collect_view();
            let empty = row
                .diff
                .is_empty()
                .then(|| view! { <li class="text-xs">"no change to any client"</li> });
            view! {
                <div class="space-y-1">
                    <h3 class="text-sm font-medium font-mono break-all">{title}</h3>
                    <ul class="space-y-1">{lines}{empty}</ul>
                </div>
            }
        })
        .collect_view();
    let doc = Doc {
        version: newest.version,
        digest: newest.digest.clone(),
    };
    let sights = sights(&r.store, &r.fleet);
    let judged: Vec<Block> = vec![(
        format!(
            "document version {} — digest {}",
            doc.version,
            short_digest(&doc.digest)
        ),
        effect::served_doc(&r.store)
            .map(|served| effect::judge_doc(&doc, served.as_ref(), &sights, r.doc_stable)),
    )];
    let accept = accept_form(csrf, id, &rows);
    // The rewind itself, with a warning for each name it drops that is still
    // in a stack's TOML: on that stack the name goes back to its TOML policy
    // and keys.
    let rewound = rewound.map(|a| {
        view! {
            <div class="space-y-2">
                <p class="text-sm">"Each commit above diffs against the document the restore landed on. What the rewind changed shows against the newest document the console committed:"</p>
                {super::document::against_view(a, &r.fleet.toml)}
            </div>
        }
    });
    view! {
        {head}
        <Notice status=Status::Ok label="done">"Done: the restore landed and the store committed it, with every revoked key merged."</Notice>
        <Notice status=status label="restored">{text}</Notice>
        <div class="space-y-4">{commits}</div>
        {rewound}
        {effect_card(&judged)}
        <div class="space-y-2">
            <p class="text-sm">"The recovery's versions are not exported as committed until you accept them, so the unmatched-install alert fires on any instance that installs them meanwhile. Accept once the diff against the picked document is what you meant."</p>
            {accept}
        </div>
    }
    .into_any()
}

fn accept_form(csrf: &str, id: &str, rows: &[RecoveryRow]) -> impl IntoView + use<> {
    let hidden = rows
        .iter()
        .map(|r| {
            let v = format!("{}.{}", r.version, r.digest);
            view! { <input type="hidden" name="row" value=v /> }
        })
        .collect_view();
    let (csrf, id) = (csrf.to_string(), id.to_string());
    view! {
        <form method="post" action="/lb/clients/pitr/accept" class="space-y-3">
            <input type="hidden" name="csrf" value=csrf />
            <input type="hidden" name="id" value=id />
            {hidden}
            <button type="submit" class=btn(Btn::Default)>"Accept the recovery"</button>
        </form>
    }
}

// ─── Accepting a recovery ───────────────────────────────────────────────────

#[derive(Deserialize, Default)]
pub struct AcceptForm {
    #[serde(default)]
    csrf: String,
    #[serde(default)]
    id: String,
    /// `version.digest` of each recovery row the operator saw, oldest first.
    #[serde(default)]
    row: Vec<String>,
}

pub async fn pitr_accept(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum_extra::extract::Form(form): axum_extra::extract::Form<AcceptForm>,
) -> Result<Response, AppError> {
    let m = store_of(&state)?;
    guard(&state, &session, &form.csrf, PITR_ACCEPT)?;
    let secure = state.secure_cookies();
    let id = form.id.trim().to_string();
    let back_to = if is_marker_id(&id) {
        marker_href(&id)
    } else {
        "/lb/clients/pitr".into()
    };
    let refuse = |jar, msg: String| {
        refused_locally(&session, PITR_ACCEPT, &msg);
        flash_redirect(jar, secure, "error", msg, &back_to)
    };
    if !is_marker_id(&id) {
        return Ok(refuse(jar, "The form names no recovery.".into()));
    }
    let store = match m.client.read().await {
        Ok(s) => s,
        Err(e) => {
            return Ok(refuse(
                jar,
                format!("The store could not be read: {}.", e.detail()),
            ));
        }
    };
    let state_of = find_marker(&store, &id).map(|m| {
        m.and_then(|m| m.get("state"))
            .and_then(Value::as_str)
            .map(str::to_string)
    });
    match state_of {
        Ok(Some(s)) if s == "done" => {}
        Ok(Some(s)) => {
            return Ok(refuse(
                jar,
                format!("Recovery {id} is {s}, not done: there is nothing to accept."),
            ));
        }
        Ok(None) => {
            return Ok(refuse(
                jar,
                format!("The store holds no recovery marker {id}."),
            ));
        }
        Err(why) => return Ok(refuse(jar, format!("Nothing was accepted: {why}."))),
    }
    let (rows, _) = recovery_rows(&store, &id);
    let now: Vec<String> = rows
        .iter()
        .map(|r| format!("{}.{}", r.version, r.digest))
        .collect();
    let seen: Vec<String> = form
        .row
        .iter()
        .map(|r| r.trim().to_ascii_lowercase())
        .collect();
    if rows.is_empty() || now != seen {
        return Ok(refuse(
            jar,
            "The recovery's commits are not the ones on the page you accepted from: review them again.".into(),
        ));
    }
    // Each is exported as of when the store committed it, never when it was
    // accepted: a label's time is read back as its commit time, the bound a
    // later point-in-time recovery's restore time is derived from.
    let mut at = Vec::with_capacity(rows.len());
    for r in &rows {
        let Some(ms) = r.at else {
            return Ok(refuse(
                jar,
                format!(
                    "The store's history names no commit time the console can read for version {}: nothing was accepted.",
                    r.version
                ),
            ));
        };
        at.push(i64::try_from(ms / 1000).unwrap_or(i64::MAX));
    }
    for (r, secs) in rows.iter().zip(at) {
        m.committed.document_at(r.version, &r.digest, secs);
    }
    let newest = rows.last().expect("rows is not empty");
    let sub = clip(&session.sub, 256);
    tracing::info!(
        target: "ops_console::write_log",
        sub = ?sub,
        kind = PITR_ACCEPT,
        recovery = %id,
        versions = %now.join(","),
        "client-store recovery accepted"
    );
    // The recovery commit's bytes reach the console only through the store's
    // view: logged whole when they are the newest row's, and only while the
    // write log holds no readable copy. Every copy is read back with the
    // rest, and three of the largest document run past what a read takes,
    // so accepting again must not log it again; an acceptance whose copy
    // never became readable logs it on the next.
    let logged = writelog::document(&m.metrics, newest.version, &newest.digest)
        .await
        .is_ok();
    if logged {
        tracing::info!(
            target: "ops_console::write_log",
            version = newest.version,
            digest = %newest.digest,
            "client-store document already on the write log"
        );
    } else if serves(&store, newest) {
        let text = store.pointer("/document/text").and_then(Value::as_str);
        writelog::log_committed(newest.version, &newest.digest, text);
    } else {
        tracing::warn!(
            target: "ops_console::write_log",
            version = newest.version,
            digest = %newest.digest,
            "client-store document not logged: the store has moved on from the recovery's commit"
        );
    }
    let doc = Doc {
        version: newest.version,
        digest: newest.digest.clone(),
    };
    let msg = format!(
        "Accepted recovery {id}: version {} is exported as committed. Confirming on every stack…",
        newest.version
    );
    Ok(flash_redirect(
        jar,
        secure,
        "ok",
        msg,
        &effect_href(&[], Some(&doc)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const D1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
    const D2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
    const D3: &str = "3333333333333333333333333333333333333333333333333333333333333333";

    fn label(version: u64, digest: &str, secs: u64) -> Label {
        Label {
            version,
            digest: digest.to_string(),
            secs,
        }
    }

    fn store_with(history: Value) -> Value {
        json!({"document": {"status": "ok", "version": 0, "history": history}})
    }

    /// Labels and history merge on version and digest; a history row an edit
    /// rewrote stands apart, found only in history.
    #[test]
    fn targets_merge_the_labels_with_the_history() {
        let store = store_with(json!([
            {"version": 3_000_000, "digest": D3, "at": "2026-11-08T10:00:00.250Z"},
            {"version": 2_000_000, "digest": D2, "at": "2026-11-08T09:00:00.000Z"},
        ]));
        let labels = [
            label(2_000_000, D2, 1_793_000_000),
            label(1_000_000, D1, 1_792_990_000),
        ];
        let t = targets(&labels, &store);
        let summary: Vec<(u64, bool, bool)> = t
            .iter()
            .map(|t| (t.version, t.labelled, t.in_history))
            .collect();
        assert_eq!(
            summary,
            [
                (1_000_000, true, false),
                (2_000_000, true, true),
                (3_000_000, false, true)
            ]
        );
        // A history row's time is exact; a label's is a second either side.
        assert_eq!(t[2].committed, (1_794_132_000_250, 1_794_132_000_250));
        assert_eq!(t[0].committed, (1_792_989_999_000, 1_792_990_001_000));
        // The same version with another digest in history: two targets.
        let edited = store_with(
            json!([{"version": 1_000_000, "digest": D3, "at": "2026-11-08T08:00:00.000Z"}]),
        );
        let t = targets(&[label(1_000_000, D1, 1_792_990_000)], &edited);
        assert_eq!(t.len(), 2);
        assert!(t.iter().any(|t| t.digest == D3 && !t.labelled));
    }

    /// The restore time comes from when the document was committed, never
    /// from its version, which can run ten minutes ahead of the clock.
    #[test]
    fn the_restore_time_comes_from_the_commit_time_never_the_version() {
        let at = "2026-11-08T10:00:00.000Z";
        let at_ms = parse_ms(at).unwrap();
        let ahead = at_ms + 600_000; // a version ten minutes ahead of its commit
        let store = store_with(json!([
            {"version": ahead + 60_000, "digest": D2, "at": "2026-11-08T10:01:00.000Z"},
            {"version": ahead, "digest": D1, "at": at},
        ]));
        let t = targets(&[], &store);
        let iv = interval(&t, 0).unwrap();
        assert_eq!(
            iv,
            Interval {
                after: at_ms,
                before: Some(at_ms + 60_000)
            }
        );
        assert_eq!(iv.suggested(), at_ms + 30_000);
        assert!(iv.suggested() < ahead);
        assert_eq!(iso_ms(iv.suggested()), "2026-11-08T10:00:30.000Z");
        // The newest: a second after its commit.
        let iv = interval(&t, 1).unwrap();
        assert_eq!((iv.before, iv.suggested()), (None, at_ms + 61_000));
        // A next commit within a millisecond leaves no time between them.
        let close = store_with(json!([
            {"version": 2, "digest": D2, "at": "2026-11-08T10:00:00.001Z"},
            {"version": 1, "digest": D1, "at": at},
        ]));
        assert_eq!(interval(&targets(&[], &close), 0), None);
    }

    /// The next commit is the next version. A history row an edit rewrote
    /// keeps its version and time, so it never shuts the original's interval.
    /// A later version found only in the labels narrows an interval, as a
    /// commit a recovery rewound away must, but never shuts it.
    #[test]
    fn the_next_commit_is_the_next_version_and_no_label_shuts_an_interval() {
        let at = "2026-11-08T10:00:00.000Z";
        let at_ms = parse_ms(at).unwrap();
        let secs = at_ms / 1000;
        let later = "2026-11-08T10:05:00.000Z";
        // The console committed (5, D1); an edit rewrote its row to D2,
        // keeping its version and time.
        let edited = store_with(json!([{"version": 5, "digest": D2, "at": at}]));
        let t = targets(&[label(5, D1, secs)], &edited);
        let pick = t.iter().position(|t| t.digest == D1).unwrap();
        assert_eq!(
            interval(&t, pick),
            Some(Interval {
                after: (secs + 1) * 1000,
                before: None
            })
        );
        // A label-only version a second on would shut the interval: it does
        // not bound it.
        let history = store_with(json!([
            {"version": 7, "digest": D2, "at": later},
            {"version": 5, "digest": D1, "at": at},
        ]));
        let t = targets(&[label(6, D3, secs + 1)], &history);
        let pick = t.iter().position(|t| t.version == 5).unwrap();
        assert_eq!(
            interval(&t, pick),
            Some(Interval {
                after: at_ms,
                before: parse_ms(later)
            })
        );
        // A minute on, it is a commit a recovery rewound away, gone from the
        // history: it bounds the pick.
        let t = targets(&[label(6, D3, secs + 61)], &history);
        let pick = t.iter().position(|t| t.version == 5).unwrap();
        assert_eq!(
            interval(&t, pick),
            Some(Interval {
                after: at_ms,
                before: Some((secs + 60) * 1000)
            })
        );
        // The same holds for a pick found only in the labels.
        let labels = [label(5, D1, secs), label(6, D3, secs + 2)];
        let t = targets(&labels, &store_with(json!([])));
        assert_eq!(
            interval(&t, 0),
            Some(Interval {
                after: (secs + 1) * 1000,
                before: None
            })
        );
        // A history version that close does shut it.
        let close = store_with(json!([
            {"version": 6, "digest": D3, "at": "2026-11-08T10:00:00.001Z"},
            {"version": 5, "digest": D1, "at": at},
        ]));
        assert_eq!(interval(&targets(&[], &close), 0), None);
    }

    #[test]
    fn the_window_must_restore_inside_the_picks_interval() {
        let iv = Some(Interval {
            after: 1_000,
            before: Some(5_000),
        });
        let w = |open, usable, to: Option<u64>| Window {
            open,
            usable,
            problem: Some("the restore time is in the future".into()),
            restore_to: to,
            closes: None,
        };
        assert_eq!(window_fault(Some(&w(true, true, Some(3_000))), iv), None);
        for (window, why) in [
            (w(false, false, None), "No recovery window is open"),
            (
                w(true, false, Some(3_000)),
                "cannot be used: the restore time is in the future",
            ),
            (
                w(true, true, Some(5_000)),
                "outside this pick's restore times",
            ),
            (
                w(true, true, Some(1_000)),
                "outside this pick's restore times",
            ),
            (w(true, true, None), "names no restore time"),
        ] {
            let fault = window_fault(Some(&window), iv).unwrap();
            assert!(fault.contains(why), "{fault}");
        }
        assert!(window_fault(None, iv).unwrap().contains("did not answer"));
        assert!(
            window_fault(Some(&w(true, true, Some(3_000))), None)
                .unwrap()
                .contains("too close")
        );
        assert_eq!(
            parse_ms("2026-11-08T10:00:30.500Z").map(iso_ms).as_deref(),
            Some("2026-11-08T10:00:30.500Z")
        );
    }

    #[test]
    fn a_recovery_is_its_rows_and_the_row_before_them() {
        let id = "a".repeat(32);
        let store = store_with(json!([
            {"version": 9, "digest": D3, "role": "store", "recovery_id": id, "diff": [{"change": "digest revoked", "digest": D1}]},
            {"version": 8, "digest": D2, "role": "recovery", "recovery_id": id, "diff": [], "at": "2026-11-08T10:00:00.250Z"},
            {"version": 5, "digest": D1, "role": "writer", "recovery_id": null, "diff": []},
        ]));
        let (rows, restored) = recovery_rows(&store, &id);
        assert_eq!(rows.iter().map(|r| r.version).collect::<Vec<_>>(), [8, 9]);
        assert_eq!(rows[0].role, "recovery");
        assert_eq!(
            rows.iter().map(|r| r.at).collect::<Vec<_>>(),
            [parse_ms("2026-11-08T10:00:00.250Z"), None]
        );
        assert_eq!(restored, Some((5, D1.to_string())));
        assert_eq!(recovery_rows(&store, &"b".repeat(32)), (vec![], None));
    }

    #[test]
    fn a_target_is_named_by_version_and_digest() {
        assert_eq!(
            parse_target(&format!("12.{}", D1.to_uppercase())),
            Some((12, D1.to_string()))
        );
        for bad in ["12", "x.abc", "12.abc", ""] {
            assert_eq!(parse_target(bad), None, "{bad}");
        }
        assert!(is_marker_id(&"0f".repeat(16)));
        assert!(!is_marker_id(&"0F".repeat(16)) && !is_marker_id("abc"));
    }

    /// A time the store writes reads back exactly, and anything else is no
    /// time at all.
    #[test]
    fn times_read_only_the_stores_shape() {
        assert_eq!(parse_ms("1970-01-01T00:00:01Z"), Some(1_000));
        assert_eq!(
            parse_ms("2026-11-08T10:00:00.5Z"),
            parse_ms("2026-11-08T10:00:00.500Z")
        );
        for bad in [
            "2026-11-08 10:00:00Z",
            "2026-11-08T10:00:00+00:00",
            "2026-13-08T10:00:00Z",
            "2026-11-08T10:00:00.1234Z",
            "2026-11-08T1+:00:00Z",
            "",
        ] {
            assert_eq!(parse_ms(bad), None, "{bad}");
        }
    }
}
