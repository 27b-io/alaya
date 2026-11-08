//! The reseed of the restrictions: `restrictions_import`, the break-glass
//! after the restrictions table was wiped or partly lost.
//!
//! It sends the restrictions sets the operator collected from the readers'
//! logs. It adds denials and never lifts one, so it has no preview and no
//! guard. The typed override, which marks the restrictions complete without
//! CF's set, is recorded on the write log (`writelog::logged_request`), and
//! the answer's names to re-enable by hand are shown in full, on the page
//! that answers the form. That page cannot be a redirect: the names can run
//! past what a flash cookie holds. So a reloaded answer re-sends the form,
//! and the override marks whatever the table holds at that time complete.
//!
//! A form is therefore sent once, and only over the table it was filled in
//! over. It carries a digest of that table (`table_of`) and the time it was
//! filled in. It is refused when the table differs, or when anything may
//! have been written to the store since that time, its own send included:
//! a write and a wipe can bring the table back to the very state it was
//! filled in over, and the form would then be sent over a write its
//! operator never saw. Writes since are read from this process's own mark
//! and from the write log, which survives a restart; a write log that
//! cannot be read refuses the form, and leaves the table as it is.

use std::collections::BTreeSet;
use std::sync::atomic::Ordering;

use axum::extract::State;
use axum::response::{Html, IntoResponse, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    Outcome, actor_of, clean_reason, guard, not_configured, outcome, page_html, refused_locally,
    refused_text, send_recorded, store_of, sub_header,
};
use crate::effect::{self, is_hex64};
use crate::error::AppError;
use crate::routes::clip;
use crate::session::{Session, flash_redirect, now_us, take_flash};
use crate::state::{AppState, StoreModule};
use crate::store::Answer;
use crate::ui::*;
use crate::writelog;

/// The write log's kind for the reseed.
const RESEED: &str = "restrictions_import";

/// What the operator types to mark the restrictions complete without CF's
/// set: the store's own phrase, which it refuses unless typed exactly.
const OVERRIDE: &str = "mark the restrictions complete without the CF set";

/// Set slots on the reseed form: CF's set and the newest of each other
/// reader's, with room to spare.
const SLOTS: usize = 8;

/// How long a form stays good: a day, well inside the write log's 30 days,
/// whose lines are what tells a write since it was filled in.
const MAX_AGE_US: u64 = 24 * 3600 * 1_000_000;

/// Why a form that something was written after is refused.
const WROTE_SINCE: &str = "Something may have been written to the store since this form was filled \
     in, this form's own send included: nothing was sent. Review the table below and fill the form \
     in again.";

/// The restrictions table a reseed form is filled in over, as a digest of
/// what the store's view says of it: every record, every revoked digest and
/// the integrity record. Any write, wipe or reseed changes it, the reseed's
/// own included, so a form sent again over a table its operator did not see
/// is refused. The store holds it, so a console restart loses nothing.
fn table_of(store: &Value) -> Result<String, String> {
    let r = store
        .get("restrictions")
        .filter(|r| r.get("status").and_then(Value::as_str) == Some("ok"))
        .ok_or("the restrictions side of the store did not answer")?;
    let table = json!({
        "records": r.get("records"),
        "revoked_digests": r.get("revoked_digests"),
        "integrity": r.get("integrity"),
    });
    Ok(writelog::digest_of(&table.to_string()))
}

// ─── The reseed ─────────────────────────────────────────────────────────────

const RESEED_ABOUT: &str = "restrictions_import: the break-glass after the restrictions table was wiped or partly lost. Supply every set marked complete that a reader logged, CF's above all. The store merges them name by name as readers do, never below a name's current record, writes only denials, and marks the table complete again when one set matches CF's latest complete report and survives whole. It never lifts a disable. A form is sent once, within a day, and only over the table it was filled in over with nothing written since: otherwise, fill it in again.";

pub async fn reseed_page(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))));
    };
    let header = sub_header("Reseed the restrictions".into(), RESEED_ABOUT.into());
    let body = match m.client.read().await {
        Ok(store) => {
            let form = match table_of(&store) {
                Ok(table) => reseed_form(&session.csrf, &table, now_us()).into_any(),
                Err(why) => {
                    let why =
                        format!("No form can be bound to the table, so none is offered: {why}.");
                    view! { <Notice status=Status::Error label="reseed">{why}</Notice> }.into_any()
                }
            };
            view! { {reseed_state(&store)} {form} }.into_any()
        }
        Err(e) => unavailable("client store", e.detail()).into_any(),
    };
    let content = view! {
        <div class="space-y-6">
            {header}
            {body}
        </div>
    };
    Ok((jar, Html(page_html(&session, flash, content))))
}

/// What the reseed needs to know from the store: whether the table matches
/// its integrity record, how it was last marked complete, and CF's latest
/// complete report, whose set must be among those supplied.
fn reseed_state(store: &Value) -> impl IntoView + use<> {
    let mut lines: Vec<(Status, String)> = Vec::new();
    let r = store.get("restrictions");
    if r.and_then(|r| r.get("status")).and_then(Value::as_str) != Some("ok") {
        lines.push((
            Status::Error,
            "The restrictions side of the store did not answer: a reseed fails until it does."
                .into(),
        ));
    } else {
        match r.and_then(|r| r.pointer("/integrity/matches")).and_then(Value::as_bool) {
            Some(true) => lines.push((Status::Ok, "The table matches its integrity record: nothing needs a reseed.".into())),
            _ => lines.push((Status::Error, "The table does not match its integrity record: fresh readers serve no ordinary client until a reseed marks it complete.".into())),
        }
        if let Some(e) = r
            .and_then(|r| r.pointer("/integrity/established"))
            .filter(|e| !e.is_null())
        {
            let s = |k: &str| clip(e.get(k).and_then(Value::as_str).unwrap_or(""), 120);
            lines.push((
                Status::Off,
                format!(
                    "Last marked complete via {} by {} at {}: {}.",
                    s("via"),
                    s("actor"),
                    s("at"),
                    s("reason")
                ),
            ));
        }
    }
    match store.pointer("/document/cf/complete").filter(|c| !c.is_null()) {
        Some(c) => {
            let s = |k: &str| clip(c.get(k).and_then(Value::as_str).unwrap_or(""), 80);
            lines.push((Status::Off, format!(
                "CF's latest complete report holds the set with hash {}, reported at {}. Find CF's restrictions log line with that hash: supplied whole, it marks the table complete.",
                s("set_hash"), s("at")
            )));
        }
        None => lines.push((Status::Warn, "The store holds no complete CF report: only the typed override can mark the table complete.".into())),
    }
    let items = lines
        .into_iter()
        .map(|(status, text)| view! { <Notice status=status label="restrictions">{text}</Notice> })
        .collect_view();
    view! {
        <Card>
            <CardHeader><CardTitle>"The restrictions"</CardTitle></CardHeader>
            <CardContent><div class="space-y-2">{items}</div></CardContent>
        </Card>
    }
}

fn reseed_form(csrf: &str, table: &str, issued: u64) -> impl IntoView + use<> {
    let slots = (1..=SLOTS)
        .map(|i| {
            let title = format!("Set {i}");
            let value = i.to_string();
            view! {
                <fieldset class="space-y-1 rounded-md border p-3">
                    <legend class="text-xs text-muted-foreground px-1">{title}</legend>
                    <textarea class=TEXTAREA_CLASS name="set_text" rows="3" spellcheck="false" placeholder="its set text, the log lines' parts joined in order"></textarea>
                    <input class=INPUT_CLASS name="set_hash" maxlength="64" placeholder="its set hash, 64 hex" autocomplete="off" />
                    <label class="flex items-center gap-2 text-xs">
                        <input type="checkbox" name="set_complete" value=value />
                        "its log line marks it complete"
                    </label>
                </fieldset>
            }
        })
        .collect_view();
    let override_label = format!(
        "Only when no complete CF report exists, or its set can no longer be rebuilt from the logs: type \"{OVERRIDE}\" to mark the table complete without CF's set."
    );
    let (csrf, table, issued) = (csrf.to_string(), table.to_string(), issued.to_string());
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Sets"</CardTitle>
                <CardDescription>"Each set as its log lines carry it: the chunks of one hash joined in order, with that hash, and only a set marked complete. Empty slots are ignored."</CardDescription>
            </CardHeader>
            <CardContent>
                <form method="post" action="/lb/clients/reseed" class="space-y-3">
                    <input type="hidden" name="csrf" value=csrf />
                    <input type="hidden" name="table" value=table />
                    <input type="hidden" name="issued" value=issued />
                    <div class="grid gap-3 md:grid-cols-2">{slots}</div>
                    <div class="space-y-1">
                        <Notice status=Status::Warn label="override">"The override marks the table complete while disables may be missing: fresh readers then serve those names until someone finds the gap, and running readers keep raising restrictions-behind meanwhile. Every use is recorded."</Notice>
                        <label class="text-xs text-muted-foreground" for="override">{override_label}</label>
                        <input class=INPUT_CLASS id="override" name="override" autocomplete="off" />
                    </div>
                    <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" />
                    <button type="submit" class=btn(Btn::Destructive)>"Send restrictions_import"</button>
                </form>
            </CardContent>
        </Card>
    }
}

#[derive(Deserialize, Default)]
pub struct ReseedForm {
    #[serde(default)]
    csrf: String,
    #[serde(default)]
    set_text: Vec<String>,
    #[serde(default)]
    set_hash: Vec<String>,
    /// The slots, numbered from 1, whose set is marked complete.
    #[serde(default)]
    set_complete: Vec<String>,
    #[serde(default, rename = "override")]
    override_phrase: String,
    #[serde(default)]
    reason: String,
    /// The table the form was filled in over (`table_of`).
    #[serde(default)]
    table: String,
    /// When the form was filled in, in Unix microseconds.
    #[serde(default)]
    issued: String,
}

/// The sets a reseed form supplies, as the store takes them. A form sends
/// line breaks as CRLF whatever was pasted, so they are read back as LF: a
/// set's lines end in LF alone. A slot with neither text nor hash is empty.
fn sets_of(form: &ReseedForm) -> Result<Vec<Value>, String> {
    let complete: BTreeSet<usize> = form
        .set_complete
        .iter()
        .filter_map(|c| c.trim().parse().ok())
        .collect();
    let mut sets = Vec::new();
    for (i, text) in form.set_text.iter().enumerate() {
        let slot = i + 1;
        let text = text.replace("\r\n", "\n");
        let hash = form
            .set_hash
            .get(i)
            .map(|h| h.trim().to_ascii_lowercase())
            .unwrap_or_default();
        if text.is_empty() && hash.is_empty() {
            continue;
        }
        if !is_hex64(&hash) {
            return Err(format!(
                "Set {slot} names no set hash: 64 hex characters, from its log line."
            ));
        }
        if writelog::digest_of(&text) != hash {
            return Err(format!(
                "Set {slot}'s text does not hash to its set hash: a part is missing or out of order."
            ));
        }
        if !complete.contains(&slot) {
            return Err(format!(
                "Set {slot} is not marked complete: the store takes only sets their log lines mark complete."
            ));
        }
        sets.push(json!({"text": text, "hash": hash, "complete": true}));
    }
    if sets.is_empty() {
        return Err("Supply at least one set.".into());
    }
    Ok(sets)
}

pub async fn reseed(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum_extra::extract::Form(form): axum_extra::extract::Form<ReseedForm>,
) -> Result<Response, AppError> {
    let m = store_of(&state)?;
    guard(&state, &session, &form.csrf, RESEED)?;
    let secure = state.secure_cookies();
    let refuse = |jar, msg: String| {
        refused_locally(&session, RESEED, &msg);
        flash_redirect(jar, secure, "error", msg, "/lb/clients/reseed")
    };
    let reason = match clean_reason(&form.reason) {
        Ok(r) => r,
        Err(msg) => return Ok(refuse(jar, msg)),
    };
    let sets = match sets_of(&form) {
        Ok(s) => s,
        Err(msg) => return Ok(refuse(jar, msg)),
    };
    let typed = form.override_phrase.trim();
    if !typed.is_empty() && typed != OVERRIDE {
        return Ok(refuse(
            jar,
            format!("The override is typed exactly: \"{OVERRIDE}\". Nothing was sent."),
        ));
    }
    let Ok(issued) = form.issued.trim().parse::<u64>() else {
        return Ok(refuse(
            jar,
            "The form carries no time it was filled in: reload the page and fill it in again."
                .into(),
        ));
    };
    if now_us().saturating_sub(issued) > MAX_AGE_US {
        return Ok(refuse(
            jar,
            "The form was filled in more than a day ago: fill it in again.".into(),
        ));
    }
    // Read again: the form is sent only over the table it was filled in over.
    let now = match m.client.read().await {
        Ok(store) => table_of(&store),
        Err(e) => Err(format!("the store could not be read ({})", e.detail())),
    };
    match now {
        Ok(table) if !form.table.is_empty() && table == form.table => {}
        Ok(_) => {
            return Ok(refuse(
                jar,
                "The restrictions table changed since this form was filled in: a write, a wipe \
                 or another reseed came between. Nothing was sent: review the table below and \
                 fill the form in again."
                    .into(),
            ));
        }
        Err(why) => {
            return Ok(refuse(
                jar,
                format!(
                    "The table this form was filled in over cannot be checked, so nothing was \
                     sent: {}.",
                    clip(&why, 300)
                ),
            ));
        }
    }
    if m.wrote_at.load(Ordering::Relaxed) >= issued {
        return Ok(refuse(jar, WROTE_SINCE.into()));
    }
    match writelog::wrote_since(&m.metrics, issued).await {
        Ok(false) => {}
        Ok(true) => return Ok(refuse(jar, WROTE_SINCE.into())),
        Err(why) => {
            return Ok(refuse(
                jar,
                format!(
                    "Whether anything was written to the store since this form was filled in \
                     cannot be read from the write log, so nothing was sent: {}.",
                    clip(&why, 300)
                ),
            ));
        }
    }
    let mut import = json!({ "actor": actor_of(&session), "reason": reason, "sets": sets });
    if !typed.is_empty() {
        import["override"] = json!(OVERRIDE);
    }
    // The records it writes are the console's to export as committed: an
    // instance installs any it did not hold, re-stamped ones included.
    let export = |m: &StoreModule, body: &Value| {
        let (written, _) = written_of(body).unwrap_or_default();
        for (name, version) in written {
            m.committed.record(&name, version);
        }
    };
    let body = json!({ "restrictions_import": import });
    let answer = send_recorded(m, &session.sub, RESEED, body, export).await;
    let content = reseed_result(answer);
    Ok((jar, Html(page_html(&session, None, content))).into_response())
}

/// The reseed's answer, in full: what it wrote, whether the table is marked
/// complete again, and every name to re-enable by hand.
fn reseed_result(answer: Result<Answer, AppError>) -> impl IntoView + use<> {
    let header = sub_header("Reseed: the store's answer".into(), RESEED_ABOUT.into());
    let body = match outcome(answer) {
        Outcome::Unknown(why) => {
            let why = format!(
                "Whether the reseed applied is unknown: {why}. It is safe to send again: it only adds denials."
            );
            view! { <Notice status=Status::Error label="unknown">{why}</Notice> }.into_any()
        }
        Outcome::Refused { status, body } => {
            let why = refused_text("The store refused the reseed", status, &body);
            view! { <Notice status=Status::Error label="refused">{why}</Notice> }.into_any()
        }
        Outcome::Accepted(body) => reseed_answer(&body).into_any(),
    };
    view! {
        <div class="space-y-6">
            {header}
            <Card><CardContent><div class="space-y-3">{body}</div></CardContent></Card>
            <p class="text-sm"><Link href="/lb/clients/reseed">"Back to the reseed"</Link>" · "<Link href="/lb/clients">"LB clients"</Link></p>
        </div>
    }
}

/// The records a reseed's answer lists as written, each a name and its
/// version, and how many of its entries name neither. `None` when the answer
/// holds no list.
fn written_of(body: &Value) -> Option<(Vec<(String, u64)>, usize)> {
    let list = body.get("written")?.as_array()?;
    let read: Vec<(String, u64)> = list
        .iter()
        .filter_map(|w| {
            let name = w.get("name")?.as_str().filter(|n| effect::is_name(n))?;
            Some((name.to_string(), w.get("version")?.as_u64()?))
        })
        .collect();
    let unread = list.len() - read.len();
    Some((read, unread))
}

fn reseed_answer(body: &Value) -> impl IntoView + use<> {
    let strs = |k: &str| -> Vec<String> {
        body.get(k)
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|s| clip(s, 80))
            .collect()
    };
    let (integrity_status, integrity) = match body.get("integrity").and_then(Value::as_str) {
        Some("cf_set") => (Status::Ok, "The table is complete again: one set matched CF's latest complete report and survived whole.".to_string()),
        Some("override") => (Status::Warn, "The table was marked complete by the typed override, without CF's set. The write log records it.".to_string()),
        _ => (Status::Warn, "The denials were added, but the table is not marked complete: no set matched CF's latest complete report, or it did not survive whole. Add the missing sets and send again: it is safe to repeat.".to_string()),
    };
    let reenable = strs("reenable_by_hand");
    let reenable_view = (!reenable.is_empty()).then(|| {
        let names = reenable.join(", ");
        let text = format!(
            "Re-enable by hand, after checking each should be enabled: {names}. Each was disabled in CF's set while another set held a newer enable, which was lost with the table, and the reseed never lifts a disable."
        );
        view! { <Notice status=Status::Error label="re-enable">{text}</Notice> }
    });
    // A count the answer does not name is never read as zero.
    let written = match written_of(body) {
        None => "Its answer names no count of the records it wrote.".to_string(),
        Some((w, 0)) if w.is_empty() => {
            "It wrote no record: every disable it was given was already in the table.".into()
        }
        Some((w, unread)) => {
            let mut names: Vec<String> = w.iter().map(|(n, v)| format!("{n} at {v}")).collect();
            if unread > 0 {
                names.push(format!("{unread} the console cannot read"));
            }
            format!(
                "It wrote {}: {}.",
                w.len() + unread,
                clip(&names.join(", "), 2000)
            )
        }
    };
    let added = match body.get("digests_added").and_then(Value::as_u64) {
        Some(n) => format!("It added {n} revoked digests"),
        None => "Its answer names no count of the revoked digests it added".into(),
    };
    let ignored = strs("digests_ignored");
    let digests = if ignored.is_empty() {
        format!("{added}.")
    } else {
        format!(
            "{added}, and ignored {} found in neither the document nor its history: {}.",
            ignored.len(),
            clip(&ignored.join(", "), 1000)
        )
    };
    let unavailable_note = (body.get("client_store").and_then(Value::as_str) == Some("unavailable")).then(|| {
        view! { <Notice status=Status::Warn label="client store">"The store's document side did not answer, so neither CF's set nor the digests could be checked: bring it back, then send the same reseed again."</Notice> }
    });
    view! {
        <Notice status=integrity_status label="integrity">{integrity}</Notice>
        {reenable_view}
        {unavailable_note}
        <p class="text-sm">{written}</p>
        <p class="text-sm">{digests}</p>
        <p class="text-sm">"The reseed is done when every running reader's restrictions-behind alert has cleared. For a name a reader still disputes, send that name's disable, or an enable if it should be enabled."</p>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reseed_sets_are_checked_against_their_hashes_before_sending() {
        let text = "r acme disabled 1791700000123000\n";
        let hash = writelog::digest_of(text);
        let form = |texts: &[&str], hashes: &[&str], complete: &[&str]| ReseedForm {
            set_text: texts.iter().map(|s| s.to_string()).collect(),
            set_hash: hashes.iter().map(|s| s.to_string()).collect(),
            set_complete: complete.iter().map(|s| s.to_string()).collect(),
            ..ReseedForm::default()
        };
        // CRLF from the form is read back as LF; empty slots are skipped; an
        // empty set (the first reseed) is a set.
        let crlf = text.replace('\n', "\r\n");
        let empty = writelog::digest_of("");
        let sets = sets_of(&form(
            &[&crlf, "", ""],
            &[&hash.to_uppercase(), "", &empty],
            &["1", "3"],
        ))
        .unwrap();
        assert_eq!(
            sets,
            [
                json!({"text": text, "hash": hash, "complete": true}),
                json!({"text": "", "hash": empty, "complete": true})
            ]
        );
        for (f, why) in [
            (form(&[text], &[&hash], &[]), "not marked complete"),
            (form(&[&text[1..]], &[&hash], &["1"]), "does not hash"),
            (form(&[text], &["abc"], &["1"]), "names no set hash"),
            (form(&["", ""], &["", ""], &[]), "at least one set"),
        ] {
            let err = sets_of(&f).unwrap_err();
            assert!(err.contains(why), "{err}");
        }
    }
}
