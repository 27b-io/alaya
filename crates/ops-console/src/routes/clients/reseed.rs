//! The reseed of the restrictions: `restrictions_import`, the break-glass
//! after the restrictions table was wiped or partly lost.
//!
//! It sends the restrictions sets the operator collected from the readers'
//! logs. It adds denials and never lifts one, so it has no preview and no
//! guard. The typed override, which marks the restrictions complete without
//! CF's set, is recorded on the write log (`writelog::logged_request`), and
//! the answer's names to re-enable by hand are shown in full, on the page
//! that answers the form. That page cannot be a redirect: the names can run
//! past what a flash cookie holds. So each form is sent once, and a reloaded
//! answer is refused rather than sent again: the override marks whatever the
//! table holds at the time complete, and a table damaged since must not be.

use std::collections::BTreeSet;

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
use crate::session::{Session, flash_redirect, take_flash};
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

/// How many sent reseed forms are remembered: far more than are ever open.
const SENT_KEPT: usize = 256;

/// Spend a reseed form's one-time token; true only the first time. Tested
/// and recorded under one lock, so a double submit sends once.
fn spend(m: &StoreModule, once: &str) -> bool {
    let mut sent = m.sent_reseeds.lock().expect("reseed lock poisoned");
    if sent.iter().any(|s| s == once) {
        return false;
    }
    if sent.len() == SENT_KEPT {
        sent.pop_front();
    }
    sent.push_back(once.to_string());
    true
}

// ─── The reseed ─────────────────────────────────────────────────────────────

const RESEED_ABOUT: &str = "restrictions_import: the break-glass after the restrictions table was wiped or partly lost. Supply every set marked complete that a reader logged, CF's above all. The store merges them name by name as readers do, never below a name's current record, writes only denials, and marks the table complete again when one set matches CF's latest complete report and survives whole. It never lifts a disable. Each form is sent once: to send another, reload this page.";

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
    let state_view = match m.client.read().await {
        Ok(store) => reseed_state(&store).into_any(),
        Err(e) => unavailable("client store", e.detail()).into_any(),
    };
    let content = view! {
        <div class="space-y-6">
            {header}
            {state_view}
            {reseed_form(&session.csrf)}
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

fn reseed_form(csrf: &str) -> impl IntoView + use<> {
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
    let (csrf, once) = (csrf.to_string(), crate::session::random_token());
    view! {
        <Card>
            <CardHeader>
                <CardTitle>"Sets"</CardTitle>
                <CardDescription>"Each set as its log lines carry it: the chunks of one hash joined in order, with that hash, and only a set marked complete. Empty slots are ignored."</CardDescription>
            </CardHeader>
            <CardContent>
                <form method="post" action="/lb/clients/reseed" class="space-y-3">
                    <input type="hidden" name="csrf" value=csrf />
                    <input type="hidden" name="once" value=once />
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
    /// The form's one-time token (`spend`).
    #[serde(default)]
    once: String,
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
    if form.once.is_empty() || !spend(m, &form.once) {
        return Ok(refuse(
            jar,
            "This reseed form was already sent, and its answer shown then: nothing was sent \
             again. To send another, fill in the form below."
                .into(),
        ));
    }
    let mut import = json!({ "actor": actor_of(&session), "reason": reason, "sets": sets });
    if !typed.is_empty() {
        import["override"] = json!(OVERRIDE);
    }
    // The records it writes are the console's to export as committed: an
    // instance installs any it did not hold, re-stamped ones included.
    let export = |m: &StoreModule, body: &Value| {
        for w in body
            .get("written")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let (Some(name), Some(version)) = (
                w.get("name")
                    .and_then(Value::as_str)
                    .filter(|n| effect::is_name(n)),
                w.get("version").and_then(Value::as_u64),
            ) {
                m.committed.record(name, version);
            }
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
    let written: Vec<String> = body
        .get("written")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|w| {
            Some(format!(
                "{} at {}",
                clip(w.get("name")?.as_str()?, 63),
                w.get("version")?.as_u64()?
            ))
        })
        .collect();
    let written = if written.is_empty() {
        "It wrote no record: every disable it was given was already in the table.".to_string()
    } else {
        format!(
            "It wrote {}: {}.",
            written.len(),
            clip(&written.join(", "), 2000)
        )
    };
    let added = body
        .get("digests_added")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ignored = strs("digests_ignored");
    let digests = if ignored.is_empty() {
        format!("It added {added} revoked digests.")
    } else {
        format!(
            "It added {added} revoked digests, and ignored {} found in neither the document nor its history: {}.",
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
