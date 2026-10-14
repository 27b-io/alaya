//! Document ops: the clients the store grants, their policies and their keys.
//!
//! Every write is two steps. The preview sends the op as a `dry_run`, pinned
//! to the document version the operator's page read, and shows the store's
//! diff with whatever the metrics store says about it. The commit sends the
//! same op, pinned to the same version, so a resubmitted form gets 409 from
//! the store and never writes twice: that is what stops a reloaded key page
//! from minting a second key.
//!
//! What the metrics store may decide here, and nothing more: an instance
//! reporting the store behind or edited puts a typed confirmation in front of
//! every document write, and a `restore` is refused unless the target row
//! checks out against the (version, digest) pairs instances installed in the
//! last 30 days, or the operator types its version where it cannot be
//! checked. A metrics store that cannot answer refuses a `restore` and holds
//! up nothing else. Both checks run again at commit: a preview is never
//! trusted to still be true.
//!
//! A minted key exists only in the commit's answer. It is rendered once, on a
//! page sent `Cache-Control: no-store`, and never logged, never put in the
//! flash cookie and never put in a URL.

use std::collections::{BTreeMap, BTreeSet};

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use super::{
    actor_of, clean_reason, effect_href, guard, not_configured, outcome_unknown, page_html,
    refused_locally, write_log,
};
use crate::confirm::{self, Needed, RowCheck};
use crate::effect::{Doc, is_hex64, is_name};
use crate::error::AppError;
use crate::fleet;
use crate::routes::{clip, fmt_epoch, unavailable};
use crate::session::{Session, flash_redirect, take_flash};
use crate::state::{AppState, StoreModule};
use crate::store::{Commit, DocOp, DocRequest, Policy, commit_of, refusal_lines};
use crate::ui::*;

/// The write log's kind for document ops.
const DOCUMENT: &str = "document";

/// Most changes one diff renders; the rest are counted.
const MAX_DIFF_LINES: usize = 100;

/// Most ops a history row names, and most keys a client row lists; the rest
/// are counted. A valid document holds at most two keys per client, but the
/// page renders whatever the store sends.
const MAX_OPS_SHOWN: usize = 20;
const MAX_KEYS_SHOWN: usize = 8;

/// The fields of every document form. Each form fills those its op reads;
/// the commit form carries them back exactly as the preview parsed them.
#[derive(Deserialize, Default)]
pub struct DocForm {
    #[serde(default)]
    csrf: String,
    #[serde(default)]
    op: String,
    #[serde(default)]
    expected_version: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    label: String,
    #[serde(default)]
    digest: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    daily_budget: String,
    #[serde(default)]
    utilization_limit: String,
    #[serde(default)]
    models: String,
    #[serde(default)]
    preferred_endpoints: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    confirm_signal: String,
    #[serde(default)]
    confirm_version: String,
}

impl DocForm {
    fn typed(&self, field: &str) -> &str {
        match field {
            "confirm_signal" => &self.confirm_signal,
            "confirm_version" => &self.confirm_version,
            _ => "",
        }
    }
}

/// A document form as the store takes it: the op, the version the page
/// read, and the reason. Only types are checked here; the store validates
/// the rest, every rule at once, and the preview shows what it refuses.
fn parse(f: &DocForm) -> Result<(DocOp, u64, String), String> {
    let expected = f
        .expected_version
        .trim()
        .parse::<u64>()
        .map_err(|_| "The form carries no document version: reload the page.".to_string())?;
    let name = || {
        let n = f.name.trim();
        if is_name(n) {
            Ok(n.to_string())
        } else {
            Err("A name is [a-z0-9] then [a-z0-9_-], at most 63 characters.".to_string())
        }
    };
    let op = match f.op.as_str() {
        "create" => DocOp::Create {
            name: name()?,
            policy: policy(f)?,
        },
        "update" => DocOp::Update {
            name: name()?,
            policy: policy(f)?,
        },
        "delete" => DocOp::Delete { name: name()? },
        "add_key" => DocOp::AddKey {
            name: name()?,
            label: f.label.trim().to_string(),
        },
        "revoke_key" => {
            let digest = f.digest.trim().to_ascii_lowercase();
            if !is_hex64(&digest) {
                return Err("revoke_key names a key by its digest: 64 hex characters.".into());
            }
            DocOp::RevokeKey { digest }
        }
        "restore" => DocOp::Restore {
            version: f
                .version
                .trim()
                .parse()
                .ok()
                .filter(|v| *v >= 1)
                .ok_or("restore names a version in history.")?,
        },
        _ => return Err("Unknown document op.".into()),
    };
    Ok((op, expected, clean_reason(&f.reason)?))
}

/// A policy from its four form fields. An empty budget or ceiling is none;
/// models and pins are lists separated by commas or spaces.
fn policy(f: &DocForm) -> Result<Policy, String> {
    let daily_budget = match f.daily_budget.trim() {
        "" => None,
        s => Some(
            s.replace('_', "")
                .parse::<u64>()
                .map_err(|_| "A daily budget is a whole number of tokens, or empty for none.")?,
        ),
    };
    let utilization_limit =
        match f.utilization_limit.trim() {
            "" => None,
            // JSON has no NaN or infinity: serialised, either would become null,
            // which reads as no ceiling at all.
            s => Some(s.parse::<f64>().ok().filter(|x| x.is_finite()).ok_or(
                "A utilization limit is a number above 0 and at most 1, or empty for none.",
            )?),
        };
    let list = |s: &str| -> Vec<String> {
        s.split(|c: char| c == ',' || c.is_whitespace())
            .filter(|x| !x.is_empty())
            .map(str::to_string)
            .collect()
    };
    Ok(Policy {
        daily_budget,
        utilization_limit,
        models: list(&f.models),
        preferred_endpoints: list(&f.preferred_endpoints),
    })
}

/// The op as form fields, the inverse of `parse`, so the commit sends
/// exactly what the preview showed.
fn op_fields(op: &DocOp) -> Vec<(&'static str, String)> {
    let policy = |p: &Policy| {
        vec![
            (
                "daily_budget",
                p.daily_budget.map(|b| b.to_string()).unwrap_or_default(),
            ),
            (
                "utilization_limit",
                p.utilization_limit
                    .map(|u| u.to_string())
                    .unwrap_or_default(),
            ),
            ("models", p.models.join(",")),
            ("preferred_endpoints", p.preferred_endpoints.join(",")),
        ]
    };
    match op {
        DocOp::Create { name, policy: p } => {
            let mut f = vec![("op", "create".into()), ("name", name.clone())];
            f.extend(policy(p));
            f
        }
        DocOp::Update { name, policy: p } => {
            let mut f = vec![("op", "update".into()), ("name", name.clone())];
            f.extend(policy(p));
            f
        }
        DocOp::Delete { name } => vec![("op", "delete".into()), ("name", name.clone())],
        DocOp::AddKey { name, label } => vec![
            ("op", "add_key".into()),
            ("name", name.clone()),
            ("label", label.clone()),
        ],
        DocOp::RevokeKey { digest } => {
            vec![("op", "revoke_key".into()), ("digest", digest.clone())]
        }
        DocOp::Restore { version } => {
            vec![("op", "restore".into()), ("version", version.to_string())]
        }
    }
}

fn op_title(op: &DocOp) -> String {
    match op {
        DocOp::Create { name, .. } => format!("create {name}"),
        DocOp::Update { name, .. } => format!("update {name}'s policy"),
        DocOp::Delete { name } => format!("delete {name}"),
        DocOp::AddKey { name, .. } => format!("add a key to {name}"),
        DocOp::RevokeKey { digest } => format!("revoke key {}…", &digest[..12]),
        DocOp::Restore { version } => format!("restore version {version}"),
    }
}

/// A document version (Unix milliseconds) as a UTC minute.
fn fmt_ms(v: u64) -> String {
    fmt_epoch(v as f64 / 1e3)
}

/// One line per change in a store diff, bounded.
fn diff_lines(diff: &[Value]) -> Vec<String> {
    let s = |c: &Value, k: &str| c.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let short = |d: String| format!("{}…", clip(&d, 12).trim_end_matches('…'));
    let mut lines: Vec<String> = diff
        .iter()
        .take(MAX_DIFF_LINES)
        .map(|c| {
            let client = s(c, "client");
            let line = match s(c, "change").as_str() {
                "created" => format!("{client}: created"),
                "deleted" => format!("{client}: deleted (its keys stay unrevoked)"),
                "policy" => format!(
                    "{client}: {} {} → {}",
                    s(c, "field"),
                    c.get("from").map(Value::to_string).unwrap_or_default(),
                    c.get("to").map(Value::to_string).unwrap_or_default()
                ),
                "key added" => format!("{client}: key added {} ({})", s(c, "id"), s(c, "label")),
                "key removed" => format!("{client}: key removed {}", s(c, "id")),
                "digest revoked" => format!("digest revoked {}", short(s(c, "digest"))),
                _ => c.to_string(),
            };
            clip(&line, 300)
        })
        .collect();
    if diff.len() > MAX_DIFF_LINES {
        lines.push(format!("…and {} more", diff.len() - MAX_DIFF_LINES));
    }
    lines
}

fn diff_of(answer: &Value) -> &[Value] {
    answer
        .get("diff")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default()
}

// ─── What stands between an op and its commit ───────────────────────────────

/// Read now, at the preview and again at the commit.
struct Gate {
    /// Typed confirmations the commit needs.
    needed: Vec<Needed>,
    /// Shown on the preview; they ask for nothing.
    notes: Vec<String>,
    /// Why the op is not sent at all: a `restore` whose row the check
    /// refused, or could not check.
    refused: Option<String>,
    /// Ordinary clients still in a stack's TOML, for the preview's warnings.
    toml: Result<BTreeMap<String, BTreeSet<String>>, String>,
}

async fn gate(m: &StoreModule, op: &DocOp) -> Gate {
    let restore = match op {
        DocOp::Restore { version } => Some(*version),
        _ => None,
    };
    let (fleet, check) = tokio::join!(fleet::read(&m.metrics), async {
        match restore {
            Some(v) => Some(restore_check(m, v).await),
            None => None,
        }
    });
    let mut needed = Vec::new();
    let mut notes = Vec::new();
    let mut refused = None;
    needed.extend(confirm::for_signals(&fleet.store_alerts));
    if let (Some(version), Some(check)) = (restore, check) {
        match check {
            Ok(RowCheck::Checked) => notes.push(format!(
                "Version {version} checks out: instances installed it with its history row's digest."
            )),
            Ok(RowCheck::Unchecked(why)) => needed.push(Needed::Unchecked { version, why }),
            Ok(RowCheck::Refused(why)) => refused = Some(format!("The restore is refused: {why}.")),
            Err(why) => {
                refused = Some(format!(
                    "The restore cannot be checked, so it is refused: {why}."
                ))
            }
        }
    }
    Gate {
        needed,
        notes,
        refused,
        toml: fleet.toml,
    }
}

/// The `restore` row check, on a fresh read of the store's history and of
/// the pairs instances installed.
async fn restore_check(m: &StoreModule, version: u64) -> Result<RowCheck, String> {
    let (store, installed) = tokio::join!(m.client.read(), fleet::installed(&m.metrics));
    let store = store.map_err(|e| format!("the store could not be read ({})", e.detail()))?;
    let installed = installed.map_err(|e| format!("the metrics store could not be read ({e})"))?;
    confirm::row_check(version, &confirm::history_row(&store, version)?, &installed)
}

/// A name the write drops that is still in a stack's TOML goes back to its
/// TOML policy and keys on that stack: say so on the preview.
fn toml_notes(
    diff: &[Value],
    toml: &Result<BTreeMap<String, BTreeSet<String>>, String>,
) -> Vec<String> {
    let dropped: Vec<&str> = diff
        .iter()
        .filter(|c| c.get("change").and_then(Value::as_str) == Some("deleted"))
        .filter_map(|c| c.get("client").and_then(Value::as_str))
        .collect();
    if dropped.is_empty() {
        return Vec::new();
    }
    match toml {
        Err(e) => vec![format!(
            "The readers' TOML clients are unreadable (metrics store: {e}): a name this drops that \
             is still in a stack's TOML goes back to its TOML policy and keys there."
        )],
        Ok(t) => dropped
            .into_iter()
            .filter_map(|n| {
                let stacks = clip(
                    &t.get(n)?.iter().cloned().collect::<Vec<_>>().join(", "),
                    300,
                );
                Some(format!(
                    "{n} is still in the TOML of {stacks}: on those stacks it goes back to its \
                     TOML policy and keys."
                ))
            })
            .collect(),
    }
}

// ─── Preview ────────────────────────────────────────────────────────────────

pub async fn preview(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<DocForm>,
) -> Result<Response, AppError> {
    let Some(m) = state.store.as_ref() else {
        return Err(AppError::NotFound(
            "the client-store module is not configured".into(),
        ));
    };
    guard(&state, &session, &form.csrf, DOCUMENT)?;
    let secure = state.secure_cookies();
    let back = |jar, msg: String| flash_redirect(jar, secure, "error", msg, "/lb/clients");
    let (op, expected, reason) = match parse(&form) {
        Ok(p) => p,
        Err(msg) => {
            refused_locally(&session, DOCUMENT, &msg);
            return Ok(back(jar, msg));
        }
    };
    let actor = actor_of(&session);
    let ops = [op];
    let request = DocRequest {
        expected_version: expected,
        actor: &actor,
        reason: &reason,
        dry_run: true,
        ops: &ops,
    };
    let answer = match m.client.document(&request).await {
        Ok(a) => a,
        Err(e) => return Ok(back(jar, format!("The preview failed: {}.", e.detail()))),
    };
    let op = &ops[0];
    if answer.status != StatusCode::OK {
        let what = if answer.status.is_client_error() {
            "The store refused the preview"
        } else {
            "The preview failed"
        };
        let mut msg = format!(
            "{what} ({}): {}.",
            answer.status.as_u16(),
            refusal_lines(&answer.body).join("; ")
        );
        if answer.status == StatusCode::CONFLICT {
            msg.push_str(&doc_now(m).await);
        }
        return Ok(back(jar, msg));
    }
    // A preview is shown only for an answer that is a dry run's. One that
    // carries a version or keys is a commit: the store wrote something the
    // operator never confirmed, so it is logged and reported like any write
    // whose outcome is unknown.
    let dry = answer.body.get("dry_run").and_then(Value::as_bool) == Some(true);
    if !dry || answer.body.get("version").is_some() || answer.body.get("keys").is_some() {
        write_log(
            &session.sub,
            DOCUMENT,
            &json!({ "document": request }),
            Ok(&answer),
        );
        let why = "the store answered the preview as if it were a commit";
        return Ok(back(jar, unknown(m, op, why).await));
    }
    let gate = gate(m, op).await;
    let content = preview_view(op, expected, &reason, &answer.body, &gate, &session.csrf);
    Ok((jar, Html(page_html(&session, None, content))).into_response())
}

fn preview_view(
    op: &DocOp,
    expected: u64,
    reason: &str,
    answer: &Value,
    gate: &Gate,
    csrf: &str,
) -> impl IntoView + use<> {
    let diff = diff_of(answer);
    let lines = diff_lines(diff)
        .into_iter()
        .map(|l| view! { <li class="font-mono text-xs">{l}</li> })
        .collect_view();
    let cf = if answer.pointer("/cf/running").and_then(Value::as_bool) == Some(true) {
        "CF has reported in the last 15 s."
    } else {
        "CF has not reported in the last 15 s: the write still commits, and CF takes it on its next poll."
    };
    let mut notes = gate.notes.clone();
    notes.extend(toml_notes(diff, &gate.toml));
    let notes = notes
        .into_iter()
        .map(|n| view! { <p class="text-sm"><span class=badge(BadgeKind::Warning)>"warning"</span>" "{n}</p> })
        .collect_view();
    let action = match (&gate.refused, diff.is_empty()) {
        (Some(why), _) => view! {
            <p class="text-sm"><span class=badge(BadgeKind::Destructive)>"refused"</span>" "{why.clone()}</p>
        }
        .into_any(),
        (None, true) => view! { <p class="text-sm">"Nothing would change, so there is nothing to commit."</p> }.into_any(),
        (None, false) => commit_form(op, expected, reason, &gate.needed, csrf).into_any(),
    };
    let title = format!("Preview: {}", op_title(op));
    let pinned = format!(
        "Nothing is written yet. The commit is pinned to document version {expected}: if the store moves on first, it answers 409 and writes nothing, so a resubmitted form never writes twice."
    );
    view! {
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>{pinned}</CardDescription>
            </CardHeader>
            <CardContent>
                <div class="space-y-4">
                    <ul class="space-y-1">{lines}</ul>
                    <p class="text-xs text-muted-foreground">{cf}</p>
                    <div class="space-y-2">{notes}</div>
                    {action}
                    <p class="text-sm"><a class="underline underline-offset-4" href="/lb/clients">"Back to LB clients"</a></p>
                </div>
            </CardContent>
        </Card>
    }
}

fn commit_form(
    op: &DocOp,
    expected: u64,
    reason: &str,
    needed: &[Needed],
    csrf: &str,
) -> impl IntoView + use<> {
    let hidden = op_fields(op)
        .into_iter()
        .map(|(k, v)| view! { <input type="hidden" name=k value=v /> })
        .collect_view();
    let confirms = needed
        .iter()
        .map(|n| {
            let (warning, field, phrase) = (n.warning(), n.field(), n.phrase());
            let label = format!("Type {phrase} to confirm.");
            view! {
                <div class="space-y-1">
                    <p class="text-sm"><span class=badge(BadgeKind::Destructive)>"confirm"</span>" "{warning}</p>
                    <label class="text-xs text-muted-foreground" for=field>{label}</label>
                    <input class=INPUT_CLASS id=field name=field required autocomplete="off" />
                </div>
            }
        })
        .collect_view();
    let style = match op {
        DocOp::Delete { .. } | DocOp::RevokeKey { .. } | DocOp::Restore { .. } => Btn::Destructive,
        _ => Btn::Default,
    };
    let (csrf, expected, reason) = (csrf.to_string(), expected.to_string(), reason.to_string());
    view! {
        <form method="post" action="/lb/clients/commit" class="space-y-3">
            <input type="hidden" name="csrf" value=csrf />
            <input type="hidden" name="expected_version" value=expected />
            <input type="hidden" name="reason" value=reason />
            {hidden}
            {confirms}
            <button type="submit" class=btn(style)>"Commit"</button>
        </form>
    }
}

// ─── Commit ─────────────────────────────────────────────────────────────────

pub async fn commit(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<DocForm>,
) -> Result<Response, AppError> {
    let Some(m) = state.store.as_ref() else {
        return Err(AppError::NotFound(
            "the client-store module is not configured".into(),
        ));
    };
    guard(&state, &session, &form.csrf, DOCUMENT)?;
    let secure = state.secure_cookies();
    let back = |jar, msg: String| flash_redirect(jar, secure, "error", msg, "/lb/clients");
    // A refusal made here, before anything is sent, is an attempt too.
    let refuse = |jar, msg: String| {
        refused_locally(&session, DOCUMENT, &msg);
        back(jar, msg)
    };
    let (op, expected, reason) = match parse(&form) {
        Ok(p) => p,
        Err(msg) => return Ok(refuse(jar, msg)),
    };
    let gate = gate(m, &op).await;
    if let Some(why) = gate.refused {
        return Ok(refuse(jar, why));
    }
    if let Some(n) = confirm::missing(&gate.needed, |f| form.typed(f)) {
        let msg = format!(
            "{} Nothing was sent: preview again, and type {} to confirm.",
            n.warning(),
            n.phrase()
        );
        return Ok(refuse(jar, msg));
    }
    let actor = actor_of(&session);
    let ops = [op];
    // Sent and recorded in a task of its own: a write the store commits must
    // reach the write log and the committed labels even when the operator's
    // connection drops and this handler is dropped with it. The answer comes
    // back here, minted keys included, and nowhere else.
    let answer = {
        let (m, sub, ops) = (m.clone(), session.sub.clone(), ops.clone());
        tokio::spawn(async move {
            let request = DocRequest {
                expected_version: expected,
                actor: &actor,
                reason: &reason,
                dry_run: false,
                ops: &ops,
            };
            let answer = m.client.document(&request).await;
            write_log(
                &sub,
                DOCUMENT,
                &json!({ "document": request }),
                answer.as_ref(),
            );
            let committed = answer.as_ref().ok().filter(|a| a.status == StatusCode::OK);
            if let Some(Ok(c)) = committed.map(|a| commit_of(&a.body, &ops)) {
                m.committed.document(c.version, &c.digest);
            }
            answer
        })
        .await
        .unwrap_or_else(|e| Err(AppError::Upstream(format!("the write's task failed ({e})"))))
    };
    let op = &ops[0];
    let answer = match answer {
        Ok(a) if !outcome_unknown(&a) => a,
        Ok(a) => {
            let why = format!(
                "the store answered {}: {}",
                a.status.as_u16(),
                refusal_lines(&a.body).join("; ")
            );
            return Ok(back(jar, unknown(m, op, &why).await));
        }
        Err(e) => {
            let why = format!("the store's answer was lost ({})", e.detail());
            return Ok(back(jar, unknown(m, op, &why).await));
        }
    };
    if answer.status != StatusCode::OK {
        let mut msg = format!(
            "The store refused the write ({}): {}.",
            answer.status.as_u16(),
            refusal_lines(&answer.body).join("; ")
        );
        if answer.status == StatusCode::CONFLICT {
            // The page the form came from was stale: say what the store holds.
            if matches!(op, DocOp::AddKey { .. }) {
                msg.push_str(
                    " If the form was sent twice, the first send may have minted a key that was \
                     never shown: compare the client's keys with the ones you hold, and revoke \
                     any other.",
                );
            }
            msg.push_str(&doc_now(m).await);
        }
        return Ok(back(jar, msg));
    }
    let c = match commit_of(&answer.body, &ops) {
        Ok(c) => c,
        Err(e) => {
            let mut msg = format!(
                "The store committed the write, but its answer could not be read ({e}), so \
                 nothing is confirmed or exported as committed."
            );
            if matches!(op, DocOp::AddKey { .. }) {
                msg.push_str(" Its key cannot be shown: revoke the new key and add another.");
            }
            msg.push_str(&doc_now(m).await);
            return Ok(back(jar, msg));
        }
    };
    // Exported as committed where it was sent. When the store follows a
    // commit with a catch-up commit of its own (`merged_after_commit`), the
    // answer carries the catch-up's version and digest, which readers
    // converge on. The first is not exported: an instance that installed it
    // in between raises the unmatched-install alert, which is the safe side.
    let doc = Doc {
        version: c.version,
        digest: c.digest.clone(),
    };
    let to = effect_href(&[], Some(&doc));
    if !c.keys.is_empty() {
        return Ok(key_page(&session, &c, &to));
    }
    let msg = format!(
        "Committed document version {}: {}. Confirming on every stack…",
        c.version,
        op_title(op)
    );
    Ok(flash_redirect(jar, secure, "ok", msg, &to))
}

/// A write that may or may not have applied. Nothing is exported as
/// committed: an unmatched-install alert on the console's own write is the
/// safe side of that doubt, and the write log holds the attempt.
async fn unknown(m: &StoreModule, op: &DocOp, why: &str) -> String {
    let mut msg = format!(
        "Whether the write applied is unknown: {why}. Check the history before sending it again."
    );
    if matches!(op, DocOp::AddKey { .. }) {
        msg.push_str(
            " If it committed, its key was never shown: revoke the new key, whose id the history \
             shows, and add another.",
        );
    }
    msg.push_str(&doc_now(m).await);
    msg
}

/// What the store's document side holds now, read again after a 409 or an
/// unknown outcome.
async fn doc_now(m: &StoreModule) -> String {
    let store = match m.client.read().await {
        Ok(s) => s,
        Err(e) => return format!(" Re-reading the store failed too: {}.", e.detail()),
    };
    let Some(doc) = store
        .get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))
    else {
        return " Re-read the store: its document side did not answer.".into();
    };
    let version = doc
        .get("version")
        .and_then(Value::as_u64)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "unreadable".into());
    let mut out = format!(" Re-read the store: it is at document version {version}");
    let frozen: Vec<String> = doc
        .get("frozen")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(|f| clip(f, 160))
        .collect();
    if !frozen.is_empty() {
        out.push_str(&format!(
            ", and refuses document writes: {}",
            frozen.join("; ")
        ));
    }
    if store.get("pitr_window").and_then(Value::as_bool) == Some(true) {
        out.push_str(", with a point-in-time recovery window open");
    }
    out.push('.');
    out
}

/// The minted key, once. The page is the commit's own answer, never a
/// redirect, so the key reaches no URL and no cookie; `no-store` keeps it out
/// of every cache. Reloading re-sends the commit, which the store refuses
/// with 409 because its pinned version is gone.
fn key_page(session: &Session, c: &Commit, effect: &str) -> Response {
    let rows = c
        .keys
        .iter()
        .map(|k| {
            let (client, id, key, digest) = (
                k.client.clone(),
                k.id.clone(),
                k.key.clone(),
                k.digest.clone(),
            );
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{client}</span></TableCell>
                    <TableCell><span class="font-mono text-xs">{id}</span></TableCell>
                    <TableCell><code class="font-mono text-sm break-all select-all">{key}</code></TableCell>
                    <TableCell><span class="font-mono text-xs break-all">{digest}</span></TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let committed = format!("Committed document version {}.", c.version);
    let effect = effect.to_string();
    let content = view! {
        <Card>
            <CardHeader>
                <CardTitle>"New key — shown once"</CardTitle>
                <CardDescription>
                    "Store it in 1Password now. The console keeps no copy: this page is not cached, and reloading it re-sends the write, which the store refuses because the document has moved on."
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div class="space-y-4">
                    <TableWrapper><Table>
                        <TableHeader><TableRow>
                            <TableHead>"Client"</TableHead>
                            <TableHead>"Key id"</TableHead>
                            <TableHead>"Key"</TableHead>
                            <TableHead>"Digest"</TableHead>
                        </TableRow></TableHeader>
                        <TableBody>{rows}</TableBody>
                    </Table></TableWrapper>
                    <p class="text-sm">{committed}" Rotation with overlap: once every consumer reads the new key, revoke the old one from the client's row."</p>
                    <p class="text-sm"><a class="underline underline-offset-4" href=effect>"Confirm the new document is in effect"</a></p>
                </div>
            </CardContent>
        </Card>
    };
    let mut resp = Html(page_html(session, None, content)).into_response();
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

// ─── History ────────────────────────────────────────────────────────────────

pub async fn history(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))));
    };
    let content = match m.client.read().await {
        Err(e) => unavailable("client store", &e).into_any(),
        Ok(store) => history_view(&store, &session.csrf).into_any(),
    };
    Ok((jar, Html(page_html(&session, flash, content))))
}

/// An op from a history row, in a few words. Rows never hold a key.
fn op_summary(op: &Value) -> String {
    let kind = op.get("op").and_then(Value::as_str).unwrap_or("?");
    let target = op
        .get("name")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| {
            op.get("digest")
                .and_then(Value::as_str)
                .map(|d| format!("{}…", clip(d, 12).trim_end_matches('…')))
        })
        .or_else(|| {
            op.get("version")
                .and_then(Value::as_u64)
                .map(|v| v.to_string())
        })
        .unwrap_or_default();
    clip(format!("{kind} {target}").trim(), 120)
}

fn history_view(store: &Value, csrf: &str) -> impl IntoView + use<> {
    let Some(doc) = store
        .get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))
    else {
        return view! {
            <p class="text-sm"><span class=badge(BadgeKind::Destructive)>"unavailable"</span>" The document side of the store did not answer, so its history cannot be read."</p>
        }
        .into_any();
    };
    let current = doc.get("version").and_then(Value::as_u64).unwrap_or(0);
    let rows: Vec<Value> = doc
        .get("history")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let shown: BTreeSet<u64> = rows
        .iter()
        .filter_map(|r| r.get("version").and_then(Value::as_u64))
        .collect();
    let older: Vec<u64> = doc
        .get("history_versions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .filter(|v| !shown.contains(v))
        .collect();
    let table = rows
        .iter()
        .map(|r| {
            // A row is the store's text, and an edit can put anything in it:
            // every field is clipped, and the ops are counted past a few.
            let s = |k: &str, max: usize| clip(r.get(k).and_then(Value::as_str).unwrap_or(""), max);
            let version = r.get("version").and_then(Value::as_u64);
            let (when, exact) = version
                .map(|v| (fmt_ms(v), v.to_string()))
                .unwrap_or_else(|| ("unreadable".into(), String::new()));
            let digest = s("digest", 64);
            let short = format!("{}…", clip(&digest, 12).trim_end_matches('…'));
            let who = format!("{} ({})", s("actor", 64), s("role", 16));
            let all_ops: Vec<String> = r
                .get("ops")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(op_summary)
                .collect();
            let mut ops = all_ops.iter().take(MAX_OPS_SHOWN).cloned().collect::<Vec<_>>().join(", ");
            if all_ops.len() > MAX_OPS_SHOWN {
                ops.push_str(&format!(", and {} more", all_ops.len() - MAX_OPS_SHOWN));
            }
            let changes = diff_lines(r.get("diff").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default())
                .into_iter()
                .map(|l| view! { <li>{l}</li> })
                .collect_view();
            let action = match version {
                Some(v) if v == current => view! { <span class=badge(BadgeKind::Success)>"current"</span> }.into_any(),
                Some(v) => restore_form(csrf, current, v).into_any(),
                None => view! { <span class="text-xs">"—"</span> }.into_any(),
            };
            let (at, reason) = (s("at", 40), s("reason", 200));
            view! {
                <TableRow>
                    <TableCell><span class="tabular-nums whitespace-nowrap" title=exact>{when}</span></TableCell>
                    <TableCell><span class="font-mono text-xs" title=digest>{short}</span></TableCell>
                    <TableCell><span class="text-xs whitespace-nowrap">{at}</span></TableCell>
                    <TableCell><span class="text-xs">{who}</span></TableCell>
                    <TableCell><span class="text-xs">{reason}</span></TableCell>
                    <TableCell><span class="text-xs font-mono">{ops}</span></TableCell>
                    <TableCell><ul class="text-xs font-mono">{changes}</ul></TableCell>
                    <TableCell>{action}</TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let older_view = (!older.is_empty()).then(|| {
        let summary = format!("{} older versions in history", older.len());
        let list = older
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let form = restore_by_version_form(csrf, current);
        view! {
            <details class="mt-6">
                <summary class="text-sm cursor-pointer">{summary}</summary>
                <p class="text-xs text-muted-foreground my-2">"The store shows a digest only for its 50 newest rows, so a restore of an older version cannot be checked against what instances installed: its preview asks you to type the version."</p>
                <p class="text-xs font-mono break-all mb-3">{list}</p>
                {form}
            </details>
        }
    });
    let title = format!("Document history (current version {current})");
    view! {
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>
                    "Every document the store committed, newest first, with what each changed. Restoring one previews the change first, and is refused if instances installed that version with another digest: its row was edited after the fact. Restore never brings back a revoked key. "
                    <a class="underline underline-offset-4" href="/lb/clients">"Back to LB clients"</a>
                </CardDescription>
            </CardHeader>
            <CardContent>
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Version (UTC)"</TableHead>
                        <TableHead>"Digest"</TableHead>
                        <TableHead>"At"</TableHead>
                        <TableHead>"By"</TableHead>
                        <TableHead>"Reason"</TableHead>
                        <TableHead>"Ops"</TableHead>
                        <TableHead>"Changes"</TableHead>
                        <TableHead>""</TableHead>
                    </TableRow></TableHeader>
                    <TableBody>{table}</TableBody>
                </Table></TableWrapper>
                {older_view}
            </CardContent>
        </Card>
    }
    .into_any()
}

/// The hidden fields every document form carries: the session's CSRF token,
/// the document version the page read, and the op.
fn pinned(csrf: &str, expected: u64, op: &'static str) -> impl IntoView + use<> {
    let (csrf, expected) = (csrf.to_string(), expected.to_string());
    view! {
        <input type="hidden" name="csrf" value=csrf />
        <input type="hidden" name="expected_version" value=expected />
        <input type="hidden" name="op" value=op />
    }
}

fn reason_input() -> impl IntoView {
    view! { <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" /> }
}

fn restore_form(csrf: &str, current: u64, version: u64) -> impl IntoView + use<> {
    let version = version.to_string();
    view! {
        <form method="post" action="/lb/clients/preview" class="flex items-center gap-2">
            {pinned(csrf, current, "restore")}
            <input type="hidden" name="version" value=version />
            {reason_input()}
            <button type="submit" class=btn_sm(Btn::Outline)>"Restore…"</button>
        </form>
    }
}

fn restore_by_version_form(csrf: &str, current: u64) -> impl IntoView + use<> {
    view! {
        <form method="post" action="/lb/clients/preview" class="flex items-end gap-3">
            {pinned(csrf, current, "restore")}
            <input class=INPUT_CLASS name="version" required inputmode="numeric" placeholder="version" />
            {reason_input()}
            <button type="submit" class=btn(Btn::Outline)>"Restore…"</button>
        </form>
    }
}

// ─── The clients card ───────────────────────────────────────────────────────

/// The document's clients, each with its policy and keys, and every
/// document op as a form that previews first.
pub(super) fn clients_card(store: &Value, csrf: &str) -> impl IntoView + use<> {
    let Some(doc) = store
        .get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))
    else {
        return clients_shell(
            "Clients".into(),
            view! { <p class="text-sm">"The document side of the store did not answer, so document ops are unavailable. Restriction ops still work."</p> }
                .into_any(),
        );
    };
    let version = doc.get("version").and_then(Value::as_u64).unwrap_or(0);
    if version == 0 {
        return clients_shell(
            "Clients".into(),
            view! { <p class="text-sm">"The store holds no document at a version a write can address, so it takes no document op but the recovery ops: an import, or a point-in-time recovery."</p> }
                .into_any(),
        );
    }
    let clients: Vec<Value> = doc
        .get("clients")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let rows = clients
        .iter()
        .map(|c| client_row(c, csrf, version))
        .collect_view();
    let create = create_form(csrf, version);
    let revoke = revoke_by_digest_form(csrf, version);
    let body = view! {
        <TableWrapper><Table>
            <TableHeader><TableRow>
                <TableHead>"Client"</TableHead>
                <TableHead>"Daily budget"</TableHead>
                <TableHead>"Utilization"</TableHead>
                <TableHead>"Models"</TableHead>
                <TableHead>"Pins"</TableHead>
                <TableHead>"Keys"</TableHead>
                <TableHead>""</TableHead>
            </TableRow></TableHeader>
            <TableBody>{rows}</TableBody>
        </Table></TableWrapper>
        {create}
        {revoke}
    }
    .into_any();
    clients_shell(format!("Clients (document version {version})"), body)
}

fn clients_shell(title: String, body: AnyView) -> AnyView {
    view! {
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>
                    "The clients the store's document grants, with their policies and keys (ids, labels and digests: the store keeps no key). Every change previews the store's diff first, pinned to the version this page read. "
                    <a class="underline underline-offset-4" href="/lb/clients/history">"History and restore"</a>
                </CardDescription>
            </CardHeader>
            <CardContent>{body}</CardContent>
        </Card>
    }
    .into_any()
}

/// A JSON value as a form shows it: a string as itself, a number as its
/// text, null as empty.
fn field_text(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

fn list_text(v: Option<&Value>, sep: &str) -> String {
    v.and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join(sep)
}

fn client_row(c: &Value, csrf: &str, version: u64) -> impl IntoView + use<> {
    let name = c
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let budget = c
        .get("daily_budget")
        .and_then(Value::as_u64)
        .map(|b| crate::lb::fmt_tokens(b as f64))
        .unwrap_or_else(|| "—".into());
    let budget_exact = field_text(c.get("daily_budget"));
    let util = Some(field_text(c.get("utilization_limit")))
        .filter(|u| !u.is_empty())
        .unwrap_or_else(|| "—".into());
    let models = Some(clip(&list_text(c.get("models"), ", "), 300))
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "all".into());
    let pins = Some(clip(&list_text(c.get("preferred_endpoints"), ", "), 300))
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| "—".into());
    let keys: Vec<Value> = c
        .get("keys")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let more_keys = (keys.len() > MAX_KEYS_SHOWN).then(|| {
        let more = format!("and {} more", keys.len() - MAX_KEYS_SHOWN);
        view! { <li class="text-xs">{more}</li> }
    });
    let key_list = keys
        .iter()
        .take(MAX_KEYS_SHOWN)
        .map(|k| {
            let s = |f: &str, max: usize| clip(k.get(f).and_then(Value::as_str).unwrap_or(""), max);
            let text = format!(
                "{} {} ({})",
                s("id", 16),
                s("label", 64),
                s("created_at", 40)
            );
            let digest = s("digest", 64);
            let form = revoke_form(csrf, version, &digest);
            view! {
                <li class="flex items-center gap-2 mb-1">
                    <span class="font-mono text-xs" title=digest>{text}</span>
                    {form}
                </li>
            }
        })
        .collect_view();
    let manage = manage_forms(c, &name, keys.len(), csrf, version);
    let disable = format!(
        "/lb/clients/disable?disable={}",
        url::form_urlencoded::byte_serialize(name.as_bytes()).collect::<String>()
    );
    view! {
        <TableRow>
            <TableCell><span class="font-mono text-xs">{name}</span></TableCell>
            <TableCell><span class="tabular-nums" title=budget_exact>{budget}</span></TableCell>
            <TableCell><span class="tabular-nums">{util}</span></TableCell>
            <TableCell><span class="text-xs font-mono">{models}</span></TableCell>
            <TableCell><span class="text-xs font-mono">{pins}</span></TableCell>
            <TableCell><ul>{key_list}{more_keys}</ul></TableCell>
            <TableCell>
                {manage}
                <a class="text-xs underline underline-offset-4" href=disable>"disable…"</a>
            </TableCell>
        </TableRow>
    }
}

/// The policy, key and delete forms of one client, folded away.
fn manage_forms(
    c: &Value,
    name: &str,
    key_count: usize,
    csrf: &str,
    version: u64,
) -> impl IntoView + use<> {
    let policy = policy_inputs(
        &field_text(c.get("daily_budget")),
        &field_text(c.get("utilization_limit")),
        &list_text(c.get("models"), ","),
        &list_text(c.get("preferred_endpoints"), ","),
    );
    let add_key = if key_count >= 2 {
        view! { <p class="text-xs text-muted-foreground">"Two keys: revoke one before adding another."</p> }.into_any()
    } else {
        let name = name.to_string();
        view! {
            <form method="post" action="/lb/clients/preview" class="flex items-end gap-2">
                {pinned(csrf, version, "add_key")}
                <input type="hidden" name="name" value=name />
                <input class=INPUT_CLASS name="label" maxlength="64" placeholder="key label, e.g. 2026-10 rotation" />
                {reason_input()}
                <button type="submit" class=btn_sm(Btn::Outline)>"Add key…"</button>
            </form>
        }
        .into_any()
    };
    let (n1, n2) = (name.to_string(), name.to_string());
    view! {
        <details class="mb-1">
            <summary class="text-xs cursor-pointer underline underline-offset-4">"manage…"</summary>
            <div class="space-y-3 mt-2 min-w-96">
                <form method="post" action="/lb/clients/preview" class="space-y-2">
                    {pinned(csrf, version, "update")}
                    <input type="hidden" name="name" value=n1 />
                    {policy}
                    {reason_input()}
                    <button type="submit" class=btn_sm(Btn::Outline)>"Update policy…"</button>
                </form>
                {add_key}
                <form method="post" action="/lb/clients/preview" class="flex items-end gap-2">
                    {pinned(csrf, version, "delete")}
                    <input type="hidden" name="name" value=n2 />
                    {reason_input()}
                    <button type="submit" class=btn_sm(Btn::Destructive)>"Delete…"</button>
                </form>
            </div>
        </details>
    }
}

fn policy_inputs(budget: &str, util: &str, models: &str, pins: &str) -> impl IntoView + use<> {
    let (budget, util, models, pins) = (
        budget.to_string(),
        util.to_string(),
        models.to_string(),
        pins.to_string(),
    );
    view! {
        <div class="grid grid-cols-2 gap-2">
            <input class=INPUT_CLASS name="daily_budget" value=budget inputmode="numeric" placeholder="daily budget (tokens; empty = none)" />
            <input class=INPUT_CLASS name="utilization_limit" value=util inputmode="decimal" placeholder="utilization limit (0–1; empty = none)" />
            <input class=INPUT_CLASS name="models" value=models placeholder="models, comma-separated (empty = all)" />
            <input class=INPUT_CLASS name="preferred_endpoints" value=pins placeholder="pinned endpoints, comma-separated" />
        </div>
    }
}

fn revoke_form(csrf: &str, version: u64, digest: &str) -> impl IntoView + use<> {
    let digest = digest.to_string();
    view! {
        <form method="post" action="/lb/clients/preview" class="flex items-center gap-1">
            {pinned(csrf, version, "revoke_key")}
            <input type="hidden" name="digest" value=digest />
            <input class=INPUT_CLASS name="reason" required maxlength="200" placeholder="reason" />
            <button type="submit" class=btn_sm(Btn::Destructive)>"Revoke…"</button>
        </form>
    }
}

fn create_form(csrf: &str, version: u64) -> impl IntoView + use<> {
    let policy = policy_inputs("", "", "", "");
    view! {
        <details class="mt-6">
            <summary class="text-sm cursor-pointer">"Create a client"</summary>
            <form method="post" action="/lb/clients/preview" class="space-y-2 mt-2">
                {pinned(csrf, version, "create")}
                <input class=INPUT_CLASS name="name" required maxlength="63" placeholder="client name" />
                {policy}
                {reason_input()}
                <button type="submit" class=btn(Btn::Default)>"Create…"</button>
            </form>
        </details>
    }
}

/// `revoke_key` by digest, for a key no longer in the document: one a
/// `delete` left unrevoked, which a `restore` would bring back.
fn revoke_by_digest_form(csrf: &str, version: u64) -> impl IntoView + use<> {
    view! {
        <details class="mt-3">
            <summary class="text-sm cursor-pointer">"Revoke a key by digest"</summary>
            <p class="text-xs text-muted-foreground my-2">"For a key the document no longer holds, such as one a delete left unrevoked: revoked here, no restore can bring it back. The store takes only a digest from the document or its history."</p>
            <form method="post" action="/lb/clients/preview" class="flex items-end gap-3">
                {pinned(csrf, version, "revoke_key")}
                <input class=INPUT_CLASS name="digest" required maxlength="64" placeholder="key digest (64 hex)" />
                {reason_input()}
                <button type="submit" class=btn(Btn::Destructive)>"Revoke…"</button>
            </form>
        </details>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn form(pairs: &[(&str, &str)]) -> DocForm {
        let mut f = DocForm::default();
        for (k, v) in pairs {
            let v = v.to_string();
            match *k {
                "op" => f.op = v,
                "expected_version" => f.expected_version = v,
                "name" => f.name = v,
                "label" => f.label = v,
                "digest" => f.digest = v,
                "version" => f.version = v,
                "daily_budget" => f.daily_budget = v,
                "utilization_limit" => f.utilization_limit = v,
                "models" => f.models = v,
                "preferred_endpoints" => f.preferred_endpoints = v,
                "reason" => f.reason = v,
                _ => panic!("no field {k}"),
            }
        }
        f
    }

    /// The commit form carries the op exactly as the preview parsed it.
    #[test]
    fn op_fields_round_trip_through_parse() {
        let ops = [
            DocOp::Create {
                name: "acme".into(),
                policy: Policy {
                    daily_budget: Some(600_000_000),
                    utilization_limit: Some(0.9),
                    models: vec!["claude-*".into(), "gpt-5".into()],
                    preferred_endpoints: vec!["west".into()],
                },
            },
            DocOp::Update {
                name: "acme".into(),
                policy: Policy {
                    daily_budget: None,
                    utilization_limit: Some(1.0),
                    models: vec![],
                    preferred_endpoints: vec![],
                },
            },
            DocOp::Delete {
                name: "acme".into(),
            },
            DocOp::AddKey {
                name: "acme".into(),
                label: "2026-10 rotation".into(),
            },
            DocOp::RevokeKey {
                digest: "a".repeat(64),
            },
            DocOp::Restore { version: 9 },
        ];
        for op in ops {
            let mut pairs: Vec<(&str, String)> = op_fields(&op);
            pairs.push(("expected_version", "7".into()));
            pairs.push(("reason", "r".into()));
            let pairs: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (*k, v.as_str())).collect();
            let (back, expected, reason) = parse(&form(&pairs)).unwrap();
            assert_eq!(back, op);
            assert_eq!((expected, reason.as_str()), (7, "r"));
        }
    }

    #[test]
    fn parse_checks_types_and_leaves_the_rules_to_the_store() {
        let ok = |pairs: &[(&str, &str)]| parse(&form(pairs));
        let base = [("expected_version", "7"), ("reason", "x")];
        let with = |extra: &[(&str, &str)]| {
            let mut p = base.to_vec();
            p.extend_from_slice(extra);
            ok(&p)
        };
        // Lists split on commas and spaces; an empty budget is none.
        let (op, _, _) = with(&[
            ("op", "create"),
            ("name", "acme"),
            ("models", "a, b  c"),
            ("daily_budget", "1_000"),
        ])
        .unwrap();
        let DocOp::Create { policy, .. } = op else {
            panic!()
        };
        assert_eq!(policy.models, ["a", "b", "c"]);
        assert_eq!(policy.daily_budget, Some(1000));
        // A budget of 0 is the store's to refuse, with its own rule id.
        assert!(with(&[("op", "create"), ("name", "acme"), ("daily_budget", "0")]).is_ok());
        // Not numbers, or not finite: refused here, never sent as null.
        for bad in [
            ("daily_budget", "lots"),
            ("utilization_limit", "NaN"),
            ("utilization_limit", "inf"),
        ] {
            assert!(
                with(&[("op", "create"), ("name", "acme"), bad]).is_err(),
                "{bad:?}"
            );
        }
        assert!(with(&[("op", "delete"), ("name", "Bad Name")]).is_err());
        assert!(with(&[("op", "revoke_key"), ("digest", "abc")]).is_err());
        assert!(with(&[("op", "restore"), ("version", "0")]).is_err());
        assert!(with(&[("op", "pitr")]).is_err());
        // No pinned version, or no reason: refused.
        assert!(ok(&[("op", "delete"), ("name", "acme"), ("reason", "x")]).is_err());
        assert!(
            ok(&[
                ("op", "delete"),
                ("name", "acme"),
                ("expected_version", "7")
            ])
            .is_err()
        );
        // A digest in capitals is the same digest.
        let (op, _, _) = with(&[("op", "revoke_key"), ("digest", &"A".repeat(64))]).unwrap();
        assert_eq!(
            op,
            DocOp::RevokeKey {
                digest: "a".repeat(64)
            }
        );
    }

    #[test]
    fn the_diff_renders_one_bounded_line_per_change() {
        let diff = vec![
            json!({"client": "acme", "change": "created"}),
            json!({"client": "acme", "change": "policy", "field": "daily_budget", "from": null, "to": 5}),
            json!({"client": "acme", "change": "key added", "id": "k1234567", "label": "l", "digest": "d"}),
            json!({"client": "acme", "change": "key removed", "id": "k1234567", "digest": "d"}),
            json!({"client": "acme", "change": "deleted"}),
            json!({"change": "digest revoked", "digest": "0123456789abcdef"}),
        ];
        assert_eq!(
            diff_lines(&diff),
            [
                "acme: created",
                "acme: daily_budget null → 5",
                "acme: key added k1234567 (l)",
                "acme: key removed k1234567",
                "acme: deleted (its keys stay unrevoked)",
                "digest revoked 0123456789ab…",
            ]
        );
        let many = vec![json!({"client": "x", "change": "created"}); MAX_DIFF_LINES + 3];
        let lines = diff_lines(&many);
        assert_eq!(lines.len(), MAX_DIFF_LINES + 1);
        assert_eq!(lines[MAX_DIFF_LINES], "…and 3 more");
    }

    #[test]
    fn toml_notes_name_each_dropped_name_still_in_a_stacks_toml() {
        let diff = vec![
            json!({"client": "acme", "change": "deleted"}),
            json!({"client": "beta", "change": "deleted"}),
            json!({"client": "gamma", "change": "created"}),
        ];
        let toml: BTreeMap<String, BTreeSet<String>> = [(
            "acme".to_string(),
            ["cf".to_string(), "k3s".to_string()].into(),
        )]
        .into();
        let notes = toml_notes(&diff, &Ok(toml));
        assert_eq!(notes.len(), 1);
        assert!(
            notes[0].starts_with("acme is still in the TOML of cf, k3s"),
            "{}",
            notes[0]
        );
        // Unreadable: one warning that says what it could hide.
        let notes = toml_notes(&diff, &Err("down".into()));
        assert!(
            notes.len() == 1 && notes[0].contains("unreadable"),
            "{notes:?}"
        );
        // Nothing dropped: nothing to say, readable or not.
        assert!(toml_notes(&diff[2..], &Err("down".into())).is_empty());
    }
}
