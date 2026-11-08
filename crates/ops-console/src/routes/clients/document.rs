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
//! reporting the store behind or edited, or a signal that cannot be told,
//! puts a typed confirmation in front of every document write; and a
//! `restore` goes only if its row checks out against the (version, digest)
//! pairs instances installed in the last 30 days, or is too old for that
//! record and the operator types its version. A metrics store that cannot
//! answer refuses a `restore`, and asks for the typed confirmation on every
//! other document write, which then goes. Both checks run again at commit: a
//! preview is never trusted to still be true.
//!
//! A minted key exists only in the commit's answer. It is rendered once, on a
//! page sent `Cache-Control: no-store`, and never logged, never put in the
//! flash cookie and never put in a URL.
//!
//! `import` is the last resort, when point-in-time recovery cannot help. Its
//! preview starts from a source, not from bytes: a document the console
//! committed, read back from its write log, or text pasted from an instance's
//! install log or written by hand. From then on its exact bytes ride the
//! commit form, as base64, with their digest. Its guard is the 30-day install
//! record, judged on the version and the digest of the bytes: an empty record
//! refuses it, a version below the newest installed one needs each newer
//! version it discards typed, and anything at or above it but the very
//! document instances installed there needs its version typed. The preview
//! also shows its diff against the newest document the console committed,
//! confirmed by this process's own exports or a committed label, never a
//! write-log line alone.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use base64::Engine;
use base64::engine::general_purpose::STANDARD;

use super::{
    Outcome, RECOVERY_REFUSED, WINDOW_OPEN, actor_of, clean_reason, effect_href, guard,
    not_configured, outcome, page_html, pinned_version, refused_locally, refused_text,
    send_recorded, store_conditions, store_of, sub_header, write_log,
};
use crate::committed;
use crate::confirm::{self, ImportCheck, Needed, RowCheck};
use crate::effect::{self, Doc, is_hex64, is_name};
use crate::error::AppError;
use crate::fleet;
use crate::routes::{clip, fmt_epoch, short_digest};
use crate::session::{Session, flash_redirect, take_flash};
use crate::state::{AppState, StoreModule};
use crate::store::{Commit, DocOp, DocRequest, Policy, commit_of};
use crate::ui::*;
use crate::writelog;

/// The write log's kind for document ops.
const DOCUMENT: &str = "document";

/// Most changes one diff renders; the rest are counted.
const MAX_DIFF_LINES: usize = 100;

/// Most ops a history row names, and most keys a client row lists; the rest
/// are counted. A valid document holds at most two keys per client, but the
/// page renders whatever the store sends.
const MAX_OPS_SHOWN: usize = 20;
const MAX_KEYS_SHOWN: usize = 8;

/// The largest document the store takes, in bytes.
const MAX_DOCUMENT_BYTES: usize = 2 * 1024 * 1024;

/// Most logged documents the import page offers.
const MAX_LOGGED_SHOWN: usize = 10;

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
    /// A restore's commit: the digest of the row its preview's dry run named.
    #[serde(default)]
    expected_digest: String,
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
    #[serde(default)]
    confirm_discard: String,
    /// An import's preview: where its document comes from, `log` (by
    /// `version` and `digest`) or `paste` (in `document`).
    #[serde(default)]
    source: String,
    /// An import's pasted document: its exact text, or its base64.
    #[serde(default)]
    document: String,
    /// An import's document as the commit form carries it, in base64, so
    /// that no form encoding can touch its bytes.
    #[serde(default)]
    text_b64: String,
}

impl DocForm {
    fn typed(&self, field: &str) -> &str {
        match field {
            "confirm_signal" => &self.confirm_signal,
            "confirm_version" => &self.confirm_version,
            "confirm_discard" => &self.confirm_discard,
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
            expected_digest: match f.expected_digest.trim() {
                "" => None,
                d if is_hex64(d) => Some(d.to_string()),
                _ => return Err("A restore's row digest is 64 hex characters.".into()),
            },
        },
        "import" => {
            let text = STANDARD
                .decode(f.text_b64.trim())
                .ok()
                .and_then(|b| String::from_utf8(b).ok())
                .filter(|t| !t.is_empty())
                .ok_or("The form carries no document to import: preview the import again.")?;
            let digest = writelog::digest_of(&text);
            // The commit names the digest its preview showed: bytes that no
            // longer hash to it are not the document the operator checked.
            let named = f.expected_digest.trim();
            if !named.is_empty() && named != digest {
                return Err(
                    "The document no longer hashes to the digest its preview named: preview the \
                     import again."
                        .into(),
                );
            }
            DocOp::Import {
                text,
                expected_digest: digest,
            }
        }
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
        DocOp::Restore {
            version,
            expected_digest,
        } => {
            let mut f = vec![("op", "restore".into()), ("version", version.to_string())];
            f.extend(expected_digest.clone().map(|d| ("expected_digest", d)));
            f
        }
        DocOp::Import {
            text,
            expected_digest,
        } => vec![
            ("op", "import".into()),
            ("text_b64", STANDARD.encode(text.as_bytes())),
            ("expected_digest", expected_digest.clone()),
        ],
        // Sent from the recovery page, never through a document form.
        DocOp::Pitr {
            expected_restore_digest,
        } => vec![
            ("op", "pitr".into()),
            ("expected_restore_digest", expected_restore_digest.clone()),
        ],
    }
}

fn op_title(op: &DocOp) -> String {
    match op {
        DocOp::Create { name, .. } => format!("create {name}"),
        DocOp::Update { name, .. } => format!("update {name}'s policy"),
        DocOp::Delete { name } => format!("delete {name}"),
        DocOp::AddKey { name, .. } => format!("add a key to {name}"),
        DocOp::RevokeKey { digest } => format!("revoke key {}", short_digest(digest)),
        DocOp::Restore { version, .. } => format!("restore version {version}"),
        DocOp::Import {
            text,
            expected_digest,
        } => match import_version(text) {
            Ok(v) => format!("import version {v} ({})", short_digest(expected_digest)),
            Err(_) => format!("import the document {}", short_digest(expected_digest)),
        },
        DocOp::Pitr {
            expected_restore_digest,
        } => format!(
            "point-in-time recovery to {}",
            short_digest(expected_restore_digest)
        ),
    }
}

/// The version a document to import names, which the guard compares.
fn import_version(text: &str) -> Result<u64, String> {
    writelog::document_version(text)
        .ok_or_else(|| "the document names no version the guard can compare".to_string())
}

/// A document version (Unix milliseconds) as a UTC minute.
fn fmt_ms(v: u64) -> String {
    fmt_epoch(v as f64 / 1e3)
}

/// One line per change in a store diff, bounded.
pub(super) fn diff_lines(diff: &[Value]) -> Vec<String> {
    let s = |c: &Value, k: &str| c.get(k).and_then(Value::as_str).unwrap_or("").to_string();
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
                "digest revoked" => format!("digest revoked {}", short_digest(&s(c, "digest"))),
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
    /// refused, or could not check, and an `import` the guard refuses.
    refused: Option<String>,
    /// Ordinary clients still in a stack's TOML, for the preview's warnings.
    toml: Result<BTreeMap<String, BTreeSet<String>>, String>,
}

async fn gate(m: &StoreModule, op: &DocOp) -> Gate {
    // A restore and an import are judged on the 30-day install record.
    let judged = matches!(op, DocOp::Restore { .. } | DocOp::Import { .. });
    let (fleet, installed) = tokio::join!(fleet::read(&m.metrics), async {
        match judged {
            true => fleet::installed(&m.metrics)
                .await
                .map_err(|e| format!("the metrics store could not be read ({e})")),
            false => Ok(BTreeMap::new()),
        }
    });
    let mut needed = Vec::new();
    let mut notes = Vec::new();
    let mut refused = None;
    needed.extend(confirm::for_signals(&fleet.store_alerts));
    match op {
        DocOp::Restore {
            version,
            expected_digest,
        } => {
            let version = *version;
            // The `restore` row check: the digest of the row the store
            // restores, against the pairs instances installed. The digest is
            // the one the store's own dry run named, and the commit sends it
            // back, so the store refuses the restore if that row changes after
            // this check. The wording for none is true at both steps: a
            // preview whose dry run named no digest, and a commit whose form
            // carries none.
            let check = match expected_digest {
                None => Err(
                    "it names no digest for the row it restores, so the store could not refuse \
                     that row if it changed before the commit"
                        .to_string(),
                ),
                Some(d) => installed.and_then(|i| confirm::row_check(version, d, &i)),
            };
            match check {
                Ok(RowCheck::Checked) => notes.push(format!(
                    "Version {version} checks out: instances installed it with the digest of the \
                     row this restores."
                )),
                Ok(RowCheck::Unchecked(why)) => needed.push(Needed::Unchecked { version, why }),
                Ok(RowCheck::Refused(why)) => {
                    refused = Some(format!("The restore is refused: {why}."))
                }
                Err(why) => {
                    refused = Some(format!(
                        "The restore cannot be checked, so it is refused: {why}."
                    ))
                }
            }
        }
        DocOp::Import { text, .. } => {
            // Judged on the digest of the bytes themselves, never on what the
            // form names: the store checks the form's against the bytes too.
            let digest = writelog::digest_of(text);
            let check = import_version(text)
                .and_then(|v| Ok((v, confirm::import_check(v, &digest, &installed?)?)));
            match check {
                Ok((v, ImportCheck::Newest)) => notes.push(format!(
                    "Version {v} is the newest document instances installed in the last 30 days: \
                     they installed it with this digest."
                )),
                Ok((version, ImportCheck::Unvouched(why))) => {
                    needed.push(Needed::Unchecked { version, why })
                }
                Ok((version, ImportCheck::Below(newer))) => {
                    needed.push(Needed::Discard { version, newer })
                }
                Err(why) => refused = Some(format!("The import is refused: {why}.")),
            }
        }
        _ => {}
    }
    Gate {
        needed,
        notes,
        refused,
        toml: fleet.toml,
    }
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
    axum::Form(mut form): axum::Form<DocForm>,
) -> Result<Response, AppError> {
    let m = store_of(&state)?;
    guard(&state, &session, &form.csrf, DOCUMENT)?;
    let secure = state.secure_cookies();
    let back = |jar, msg: String| flash_redirect(jar, secure, "error", msg, "/lb/clients");
    // An import's preview names a source; its bytes ride the form from here.
    let mut source = None;
    if form.op == "import" && !form.source.is_empty() {
        match import_source(m, &form).await {
            Ok((text, from)) => {
                form.text_b64 = STANDARD.encode(text.as_bytes());
                source = Some(from);
            }
            Err(msg) => {
                refused_locally(&session, DOCUMENT, &msg);
                return Ok(back(jar, msg));
            }
        }
    }
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
    }
    .body();
    let answer = m.client.post(&request).await;
    // A preview is shown only for an answer that is a dry run's. One that
    // carries a version or keys is a commit: the store wrote something the
    // operator never confirmed, so it is logged and reported like any write
    // whose outcome is unknown.
    let commit_like = |body: &Value| {
        body.get("dry_run").and_then(Value::as_bool) != Some(true)
            || body.get("version").is_some()
            || body.get("keys").is_some()
    };
    if let Ok(a) = &answer
        && a.status == StatusCode::OK
        && commit_like(&a.body)
    {
        write_log(&session.sub, DOCUMENT, &request, Ok(a));
        let why = "the store answered the preview as if it were a commit";
        return Ok(back(jar, unknown(m, &ops[0], why).await));
    }
    let body = match outcome(answer) {
        Outcome::Accepted(body) => body,
        Outcome::Unknown(why) => return Ok(back(jar, format!("The preview failed: {why}."))),
        Outcome::Refused { status, body } => {
            let mut msg = refused_text("The store refused the preview", status, &body);
            if status == StatusCode::CONFLICT {
                msg.push_str(&doc_now(m).await);
            }
            return Ok(back(jar, msg));
        }
    };
    // A restore commits only the row its preview showed: the digest the
    // store's dry run names for that row rides the commit form, and the
    // store refuses the commit if the row changed since. A dry run that
    // names none leaves the op without one, which `gate` refuses; one that
    // names something that is not a digest is the store's answer gone wrong.
    let op = match &ops[0] {
        DocOp::Restore { version, .. } => {
            let named = body.get("restore_digest");
            let expected_digest = match named.and_then(Value::as_str) {
                Some(d) if is_hex64(d) => Some(d.to_string()),
                None if named.is_none() => None,
                _ => {
                    let why = "The store's dry run named a malformed digest for the row the \
                               restore would restore, so the restore cannot be bound to that row: \
                               nothing can be committed.";
                    return Ok(back(jar, why.into()));
                }
            };
            DocOp::Restore {
                version: *version,
                expected_digest,
            }
        }
        op => op.clone(),
    };
    let gate = gate(m, &op).await;
    let against = match &op {
        DocOp::Import { text, .. } => Some(against_newest(m, text).await),
        _ => None,
    };
    let content = preview_view(
        &op,
        expected,
        &reason,
        &body,
        &gate,
        Imported {
            source,
            against: against.as_ref(),
        },
        &session.csrf,
    );
    Ok((jar, Html(page_html(&session, None, content))).into_response())
}

/// What an import's preview shows beyond the store's dry run: where its
/// document came from, and its diff against the newest document the console
/// committed. Both `None` for every other op.
struct Imported<'a> {
    source: Option<String>,
    against: Option<&'a Result<(u64, Vec<Value>), String>>,
}

fn preview_view(
    op: &DocOp,
    expected: u64,
    reason: &str,
    answer: &Value,
    gate: &Gate,
    imported: Imported,
    csrf: &str,
) -> impl IntoView + use<> {
    let diff = diff_of(answer);
    let lines = diff_lines(diff)
        .into_iter()
        .map(|l| view! { <li class="font-mono text-xs break-all">{l}</li> })
        .collect_view();
    let import = matches!(op, DocOp::Import { .. });
    // An import's dry run diffs against what the store holds now, which may
    // be nothing at all: say so, and set the console's own diff beside it.
    let store_heading = import.then(|| {
        view! { <h3 class="text-sm font-medium">"Against what the store holds now"</h3> }
    });
    let source = imported
        .source
        .map(|s| view! { <p class="text-sm">{format!("Source: {s}.")}</p> });
    let against = imported.against.map(|a| against_view(a, &gate.toml));
    let cf = if answer.pointer("/cf/running").and_then(Value::as_bool) == Some(true) {
        "CF has reported in the last 15 s."
    } else {
        "CF has not reported in the last 15 s: the write still commits, and CF takes it on its next poll."
    };
    let mut notes = gate.notes.clone();
    notes.extend(toml_notes(diff, &gate.toml));
    let notes = notes
        .into_iter()
        .map(|n| view! { <Notice status=Status::Warn label="warning">{n}</Notice> })
        .collect_view();
    // An import commits a new version even with no client changed: that is
    // what repairs a wiped, behind or edited store.
    let action = match (&gate.refused, diff.is_empty() && !import) {
        (Some(why), _) => {
            let why = why.clone();
            view! { <Notice status=Status::Error label="refused">{why}</Notice> }.into_any()
        }
        (None, true) => view! {
            <EmptyState>"Nothing would change, so there is nothing to commit."</EmptyState>
        }
        .into_any(),
        (None, false) => commit_form(op, expected, reason, &gate.needed, csrf).into_any(),
    };
    let title = format!("Preview: {}", op_title(op));
    let pinned = format!(
        "Nothing is written yet. The commit is pinned to document version {expected}: if the store moves on first, it answers 409 and writes nothing, so a resubmitted form never writes twice."
    );
    view! {
        <div class="space-y-6">
            {sub_header(title, pinned)}
            <Card>
                <CardContent>
                    <div class="space-y-4">
                        {source}
                        {store_heading}
                        <ul class="space-y-1">{lines}</ul>
                        {against}
                        <p class="text-xs text-muted-foreground">{cf}</p>
                        <div class="space-y-2">{notes}</div>
                        {action}
                    </div>
                </CardContent>
            </Card>
        </div>
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
    let confirms = confirm_inputs(needed);
    let style = match op {
        DocOp::Delete { .. }
        | DocOp::RevokeKey { .. }
        | DocOp::Restore { .. }
        | DocOp::Import { .. } => Btn::Destructive,
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

/// A form's typed confirmations, each under its warning.
pub(super) fn confirm_inputs(needed: &[Needed]) -> impl IntoView + use<> {
    needed
        .iter()
        .map(|n| {
            let (warning, field, phrase) = (n.warning(), n.field(), n.phrase());
            let label = format!("Type {phrase} to confirm.");
            view! {
                <div class="space-y-1">
                    <Notice status=Status::Error label="confirm">{warning}</Notice>
                    <label class="text-xs text-muted-foreground" for=field>{label}</label>
                    <input class=INPUT_CLASS id=field name=field required autocomplete="off" />
                </div>
            }
        })
        .collect_view()
}

// ─── Commit ─────────────────────────────────────────────────────────────────

pub async fn commit(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<DocForm>,
) -> Result<Response, AppError> {
    let m = store_of(&state)?;
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
    let request = DocRequest {
        expected_version: expected,
        actor: &actor,
        reason: &reason,
        dry_run: false,
        ops: &ops,
    };
    // Exported as committed, and logged whole, where it is sent. When the
    // store follows a commit with a catch-up commit of its own
    // (`merged_after_commit`), the answer carries the catch-up's version,
    // digest and bytes, which readers converge on. The first is not
    // exported: an instance that installed it in between raises the
    // unmatched-install alert, which is the safe side. Whether its document
    // reached the write log comes back for the operator.
    let logged = Arc::new(AtomicBool::new(false));
    let export = {
        let (ops, logged) = (ops.clone(), Arc::clone(&logged));
        move |m: &StoreModule, body: &Value| {
            if let Ok(c) = commit_of(body, &ops) {
                m.committed.document(c.version, &c.digest);
                let text = body.get("text").and_then(Value::as_str);
                let done = writelog::log_committed(c.version, &c.digest, text);
                logged.store(done, Ordering::Relaxed);
            }
        }
    };
    let answer = send_recorded(m, &session.sub, DOCUMENT, request.body(), export).await;
    let op = &ops[0];
    let body = match outcome(answer) {
        Outcome::Accepted(body) => body,
        Outcome::Unknown(why) => return Ok(back(jar, unknown(m, op, &why).await)),
        Outcome::Refused { status, body } => {
            let mut msg = refused_text("The store refused the write", status, &body);
            if status == StatusCode::CONFLICT {
                // The page the form came from was stale, or the row a restore
                // previewed has changed: say what the store holds now.
                if matches!(op, DocOp::AddKey { .. }) {
                    msg.push_str(
                        " If the form was sent twice, the first send may have minted a key that \
                         was never shown: compare the client's keys with the ones you hold, and \
                         revoke any other.",
                    );
                }
                msg.push_str(&doc_now(m).await);
            }
            return Ok(back(jar, msg));
        }
    };
    let c = match commit_of(&body, &ops) {
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
    let doc = Doc {
        version: c.version,
        digest: c.digest.clone(),
    };
    let to = effect_href(&[], Some(&doc));
    // The task that sent the write has finished, so its answer is final.
    let unlogged = (!logged.load(Ordering::Relaxed)).then_some(UNLOGGED);
    if !c.keys.is_empty() {
        return Ok(key_page(&session, &c, &to, unlogged));
    }
    let msg = format!(
        "Committed document version {}: {}.{} Confirming on every stack…",
        c.version,
        op_title(op),
        unlogged.map(|u| format!(" {u}")).unwrap_or_default()
    );
    Ok(flash_redirect(jar, secure, "ok", msg, &to))
}

/// What the operator is told of a commit whose document is not on the write
/// log, where an import after a wipe looks for it first.
const UNLOGGED: &str = "Its document is not on the write log: the store's answer carries no bytes \
     that hash to it. Should an import need it, paste it from an instance's install log.";

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
pub(super) async fn doc_now(m: &StoreModule) -> String {
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
    let mut out = format!(" Re-read the store: it is at document version {version}.");
    let c = store_conditions(&store);
    for line in [
        c.frozen_line(),
        c.recovery_refused.then(|| RECOVERY_REFUSED.to_string()),
        c.window_open.then(|| WINDOW_OPEN.to_string()),
    ]
    .into_iter()
    .flatten()
    {
        out.push(' ');
        out.push_str(&line);
    }
    out
}

/// The minted key, once. The page is the commit's own answer, never a
/// redirect, so the key reaches no URL and no cookie; `no-store` keeps it out
/// of every cache. Reloading re-sends the commit, which the store refuses
/// with 409 because its pinned version is gone.
fn key_page(session: &Session, c: &Commit, effect: &str, unlogged: Option<&str>) -> Response {
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
                    <TableCell secondary=true><span class="font-mono text-xs break-all">{digest}</span></TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let committed = format!("Committed document version {}.", c.version);
    let unlogged = unlogged
        .map(str::to_string)
        .map(|u| view! { <Notice status=Status::Warn label="write log">{u}</Notice> });
    let effect = effect.to_string();
    let content = view! {
        <div class="space-y-6">
            {sub_header(
                "New key — shown once".into(),
                "Store it in 1Password now. The console keeps no copy: this page is not cached, and reloading it re-sends the write, which the store refuses because the document has moved on.".into(),
            )}
            <Card>
                <CardContent>
                    <div class="space-y-4">
                        <TableWrapper><Table>
                            <TableHeader><TableRow>
                                <TableHead>"Client"</TableHead>
                                <TableHead>"Key id"</TableHead>
                                <TableHead>"Key"</TableHead>
                                <TableHead secondary=true>"Digest"</TableHead>
                            </TableRow></TableHeader>
                            <TableBody>{rows}</TableBody>
                        </Table></TableWrapper>
                        <p class="text-sm">{committed}" Rotation with overlap: once every consumer reads the new key, revoke the old one from the client's row."</p>
                        {unlogged}
                        <p class="text-sm"><Link href=effect>"Confirm the new document is in effect"</Link></p>
                    </div>
                </CardContent>
            </Card>
        </div>
    };
    let mut resp = Html(page_html(session, None, content)).into_response();
    resp.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    resp
}

// ─── Import ─────────────────────────────────────────────────────────────────

/// An import's document, from the source its preview names: a document the
/// console committed, from its write log, or pasted text. Returns its exact
/// text and where it came from, in words.
async fn import_source(m: &StoreModule, f: &DocForm) -> Result<(String, String), String> {
    match f.source.as_str() {
        "log" => {
            let version = f
                .version
                .trim()
                .parse::<u64>()
                .map_err(|_| "A logged document is named by its version.".to_string())?;
            let digest = f.digest.trim().to_ascii_lowercase();
            let text = writelog::document(&m.metrics, version, &digest)
                .await
                .map_err(|e| format!("The logged document cannot be imported: {e}."))?;
            Ok((text, "the console's write log".into()))
        }
        "paste" => {
            let (text, from) = pasted(&f.document)?;
            // An install log names each document's digest beside its base64:
            // given, it is the digest the import is checked against, and a
            // paste that lost or changed a byte does not hash to it. A
            // document written by hand has none.
            let named = f.digest.trim().to_ascii_lowercase();
            if named.is_empty() {
                return Ok((
                    text,
                    format!("{from}, with no logged digest: the console hashed its bytes itself"),
                ));
            }
            let digest = writelog::digest_of(&text);
            if digest != named {
                return Err(format!(
                    "The pasted document does not hash to the digest given: it hashes to {}, and \
                     the log names {}. Paste its parts again, whole and in order.",
                    short_digest(&digest),
                    short_digest(&named)
                ));
            }
            Ok((text, format!("{from}, matching its logged digest")))
        }
        _ => Err("An import names its source: a logged document, or pasted text.".into()),
    }
}

/// A pasted document: its JSON text, or its base64 as an instance's install
/// log carries it, the log lines' parts joined. A form sends line breaks as
/// CRLF whatever was pasted, so they are read back as LF; JSON text is
/// trimmed, which changes no document; base64 loses every space, so wrapped
/// lines join.
fn pasted(raw: &str) -> Result<(String, String), String> {
    let raw = raw.replace("\r\n", "\n");
    let trimmed = raw.trim();
    let (text, from) = if trimmed.starts_with('{') {
        (trimmed.to_string(), "pasted text")
    } else {
        let b64: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
        let text = STANDARD
            .decode(b64)
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .ok_or("The pasted document is neither JSON text nor base64 of a UTF-8 text.")?;
        (text, "pasted base64")
    };
    if text.is_empty() {
        return Err("Paste the document to import.".into());
    }
    if text.len() > MAX_DOCUMENT_BYTES {
        return Err(format!(
            "The pasted document is {} bytes; the store takes at most {MAX_DOCUMENT_BYTES}.",
            text.len()
        ));
    }
    Ok((text, from.into()))
}

/// A document's diff against the newest document the console committed, as
/// its write log holds it: anything this diff shows, the document changes.
/// An import is checked against it, and so is a finished point-in-time
/// recovery. The newest commit is named first, and only its own bytes are
/// the base: an older copy standing in for a missing one would show no
/// change for a document that undoes the newest commit.
pub(super) async fn against_newest(
    m: &StoreModule,
    text: &str,
) -> Result<(u64, Vec<Value>), String> {
    let labels = committed::read(&m.metrics).await;
    let (version, digest) =
        newest_known(m, labels.as_deref().unwrap_or_default()).ok_or_else(|| match &labels {
            Ok(_) => "no document the console committed is known: this process exported none, \
                      and the metrics store shows no committed label from the last 30 days"
                .to_string(),
            Err(e) => format!(
                "no document the console committed is known: this process exported none, and \
                 the committed labels could not be read ({e})"
            ),
        })?;
    let base = writelog::document(&m.metrics, version, &digest)
        .await
        .map_err(|e| {
            format!(
                "the newest document the console committed, version {version}, cannot be read \
                 from the write log ({e}), and no older copy stands in for it"
            )
        })?;
    let json = |t: &str| {
        serde_json::from_str::<Value>(t).map_err(|_| "a document is not JSON".to_string())
    };
    let changes = client_diff(&json(&base)?, &json(text)?);
    Ok((version, changes))
}

/// The newest document the console committed, by version and digest: the
/// newest this process exported, or else the newest a committed label names.
fn newest_known(m: &StoreModule, labels: &[committed::Label]) -> Option<(u64, String)> {
    m.committed.newest().or_else(|| {
        labels
            .iter()
            .max_by_key(|l| l.version)
            .map(|l| (l.version, l.digest.clone()))
    })
}

fn labelled(labels: &[committed::Label], d: &writelog::Logged) -> bool {
    labels
        .iter()
        .any(|l| l.version == d.version && l.digest == d.digest)
}

/// The diff against the newest document the console committed, with a
/// warning for each name it drops that is still in a stack's TOML.
pub(super) fn against_view(
    against: &Result<(u64, Vec<Value>), String>,
    toml: &Result<BTreeMap<String, BTreeSet<String>>, String>,
) -> AnyView {
    let body = match against {
        Ok((version, changes)) if changes.is_empty() => view! {
            <p class="text-sm">{format!(
                "No change from the newest document the console committed: version {version}."
            )}</p>
        }
        .into_any(),
        Ok((version, changes)) => {
            let items = diff_lines(changes)
                .into_iter()
                .map(|l| view! { <li class="font-mono text-xs break-all">{l}</li> })
                .collect_view();
            let what =
                format!("Against the newest document the console committed: version {version}.");
            let notes = toml_notes(changes, toml)
                .into_iter()
                .map(|n| view! { <Notice status=Status::Warn label="toml">{n}</Notice> })
                .collect_view();
            view! {
                <p class="text-sm">{what}</p>
                <ul class="space-y-1">{items}</ul>
                <div class="space-y-2">{notes}</div>
            }
            .into_any()
        }
        Err(why) => {
            let why = format!(
                "No diff against the newest document the console committed: {why}. Check the \
                 document against the store's history and the instances' install logs."
            );
            view! { <Notice status=Status::Warn label="no diff">{why}</Notice> }.into_any()
        }
    };
    view! {
        <div class="space-y-2">
            <h3 class="text-sm font-medium">"Against the newest document the console committed"</h3>
            {body}
        </div>
    }
    .into_any()
}

/// What changed from one document to another, in the shape of the store's
/// own diff, which `diff_lines` renders. The store diffs only against what
/// it holds now; an import is checked against what the console committed.
fn client_diff(before: &Value, after: &Value) -> Vec<Value> {
    const POLICY: [&str; 4] = [
        "daily_budget",
        "utilization_limit",
        "models",
        "preferred_endpoints",
    ];
    let clients = |d: &Value| -> Vec<(String, Value)> {
        d.get("clients")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|c| Some((c.get("name")?.as_str()?.to_string(), c.clone())))
            .collect()
    };
    let list = |v: &Value, k: &str| {
        v.get(k)
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
    };
    let (was, now) = (clients(before), clients(after));
    let mut out = Vec::new();
    for (name, c) in &now {
        let Some((_, old)) = was.iter().find(|(n, _)| n == name) else {
            out.push(json!({"client": name, "change": "created"}));
            continue;
        };
        for field in POLICY {
            if old.get(field) != c.get(field) {
                out.push(json!({"client": name, "change": "policy", "field": field,
                                "from": old.get(field), "to": c.get(field)}));
            }
        }
        let (old_keys, new_keys) = (list(old, "keys"), list(c, "keys"));
        let has =
            |keys: &[Value], k: &Value| keys.iter().any(|o| o.get("digest") == k.get("digest"));
        for k in new_keys.iter().filter(|k| !has(&old_keys, k)) {
            out.push(
                json!({"client": name, "change": "key added", "id": k.get("id"),
                            "label": k.get("label"), "digest": k.get("digest")}),
            );
        }
        for k in old_keys.iter().filter(|k| !has(&new_keys, k)) {
            out.push(json!({"client": name, "change": "key removed", "id": k.get("id"), "digest": k.get("digest")}));
        }
    }
    for (name, _) in was.iter().filter(|(n, _)| !now.iter().any(|(m, _)| m == n)) {
        out.push(json!({"client": name, "change": "deleted"}));
    }
    let revoked = list(before, "revoked_digests");
    for d in list(after, "revoked_digests")
        .into_iter()
        .filter(|d| !revoked.contains(d))
    {
        out.push(json!({"change": "digest revoked", "digest": d}));
    }
    out
}

/// What the import page is for.
const IMPORT: &str = "The last resort, for when point-in-time recovery cannot help: the store's object was deleted, or the wipe is one PITR cannot reach. An import replaces the document with one the console committed, from its write log, or with text pasted from an instance's install log or written by hand. The store takes one only at version 0, or while it is behind, edited or refused. It merges every revoked key into the import, so none comes back.";

pub async fn import_page(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let Some(m) = state.store.as_ref() else {
        return Ok((jar, Html(page_html(&session, flash, not_configured()))));
    };
    let (store, logged, installed, labels) = tokio::join!(
        m.client.read(),
        writelog::logged(&m.metrics),
        fleet::installed(&m.metrics),
        committed::read(&m.metrics)
    );
    // Each logged copy, marked as the console committed it or not, and the
    // newest commit, whose copy every preview diffs against.
    let newest = newest_known(m, labels.as_deref().unwrap_or_default());
    let logged = logged.map(|l| {
        let labels = labels.as_deref().unwrap_or_default();
        let offered = l
            .docs
            .iter()
            .map(|d| Offered {
                committed: m.committed.holds(d.version, &d.digest) || labelled(labels, d),
                newest: newest
                    .as_ref()
                    .is_some_and(|(v, g)| *v == d.version && *g == d.digest),
                twin: l
                    .docs
                    .iter()
                    .any(|o| o.version == d.version && o.digest != d.digest),
                doc: d.clone(),
            })
            .collect();
        (offered, l.malformed)
    });
    let labels = labels.map(|_| ());
    let content = import_view(store, logged, labels, newest, installed, &session.csrf);
    Ok((jar, Html(page_html(&session, flash, content))))
}

/// A logged document as the import page offers it.
struct Offered {
    doc: writelog::Logged,
    /// This process exported it as committed, or a committed label names it.
    committed: bool,
    /// The newest such: what every import preview diffs against.
    newest: bool,
    /// The write log names its version with another digest too, which only a
    /// fault or a write from outside the console makes.
    twin: bool,
}

fn import_view(
    store: Result<Value, AppError>,
    logged: Result<(Vec<Offered>, usize), String>,
    labels: Result<(), String>,
    newest: Option<(u64, String)>,
    installed: Result<BTreeMap<u64, BTreeSet<String>>, String>,
    csrf: &str,
) -> impl IntoView + use<> {
    let header = sub_header("Import a document".into(), IMPORT.into());
    // The version every import form pins: the store's addressable version,
    // 0 when it holds none. Unknown, and no form is offered.
    let expected = store
        .as_ref()
        .map_err(|e| e.detail().to_string())
        .and_then(pinned_version);
    let state = match (&store, &expected) {
        (Ok(s), Ok(v)) => {
            let c = store_conditions(s);
            let mut lines = vec![format!("The store is at document version {v}.")];
            lines.extend(c.frozen_line());
            if c.recovery_refused {
                lines.push(RECOVERY_REFUSED.into());
            }
            if c.window_open {
                lines.push(WINDOW_OPEN.into());
            }
            let items = lines
                .into_iter()
                .map(|l| view! { <li>{l}</li> })
                .collect_view();
            view! { <ul class="text-sm space-y-1">{items}</ul> }.into_any()
        }
        // `expected` is an error whenever the store's read is one.
        _ => {
            let e = expected.as_ref().err().cloned().unwrap_or_default();
            let why = format!(
                "The store's document cannot be read ({}), so no import can be pinned to its version.",
                clip(&e, 160)
            );
            view! { <Notice status=Status::Error label="store">{why}</Notice> }.into_any()
        }
    };
    let guard = match &installed {
        Err(e) => view! {
            <Notice status=Status::Error label="guard">{format!("The metrics store could not be read ({}): every import is refused until it answers.", clip(e, 200))}</Notice>
        }
        .into_any(),
        Ok(i) => match i.keys().next_back() {
            None => view! {
                <Notice status=Status::Error label="guard">"The metrics store shows no document installed in the last 30 days, which reads as unknown: every import is refused."</Notice>
            }
            .into_any(),
            Some(high) => view! {
                <p class="text-sm">{format!("The newest version instances installed in the last 30 days is {high}. An import below it needs each newer version it discards typed. One at or above it needs its version typed unless instances installed that exact document.")}</p>
            }
            .into_any(),
        },
    };
    let expected = expected.ok();
    let malformed = logged.as_ref().map(|l| l.1).unwrap_or(0);
    let malformed = (malformed > 0).then(|| {
        let text = format!(
            "The write log also holds {malformed} document lines that name no version and digest, which only a fault or a write from outside the console makes: they are left out."
        );
        view! { <div class="mb-3"><Notice status=Status::Warn label="write log">{text}</Notice></div> }
    });
    let labels = labels.err().map(|e| {
        let text = format!(
            "The committed labels could not be read ({}): only the documents this console process exported are marked as committed.",
            clip(&e, 200)
        );
        view! { <div class="mb-3"><Notice status=Status::Warn label="labels">{text}</Notice></div> }
    });
    // The newest commit's copy missing from the list: no preview can diff
    // against it, and nothing older stands in for it.
    let listed = |v: u64, d: &str| {
        logged
            .as_ref()
            .is_ok_and(|l| l.0.iter().any(|o| o.doc.version == v && o.doc.digest == d))
    };
    let unlisted = newest.filter(|(v, d)| logged.is_ok() && !listed(*v, d)).map(|(v, _)| {
        let text = format!(
            "The newest document the console committed, version {v}, is not among the documents the write log lists: no import preview can show its diff against it. Paste it from an instance's install log if you have it."
        );
        view! { <div class="mb-3"><Notice status=Status::Warn label="newest">{text}</Notice></div> }
    });
    let rows = match logged.map(|l| l.0) {
        Err(e) => view! {
            <Notice status=Status::Warn label="write log">{format!("The write log could not be read ({}): paste a document below instead.", clip(&e, 200))}</Notice>
        }
        .into_any(),
        Ok(docs) if docs.is_empty() => view! {
            <EmptyState>"The write log holds no document from the last 30 days: paste one below, from an instance's install log or written by hand."</EmptyState>
        }
        .into_any(),
        Ok(docs) => {
            let rows = docs
                .iter()
                .take(MAX_LOGGED_SHOWN)
                .map(|o| {
                    let d = &o.doc;
                    let form = expected.map(|v| logged_import_form(csrf, v, d));
                    let badges = [
                        o.newest.then_some((Status::Ok, "newest committed")),
                        (!o.committed).then_some((Status::Warn, "no committed label")),
                        o.twin.then_some((Status::Error, "another digest at this version")),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|(status, label)| view! { <StatusBadge status=status label=label /> })
                    .collect_view();
                    let (when, exact, short, digest, at) = (fmt_ms(d.version), d.version.to_string(), short_digest(&d.digest), d.digest.clone(), d.at.clone());
                    view! {
                        <TableRow>
                            <TableCell><span class="tabular-nums whitespace-nowrap" title=exact>{when}</span> {badges}</TableCell>
                            <TableCell secondary=true><span class="font-mono text-xs" title=digest>{short}</span></TableCell>
                            <TableCell secondary=true><span class="text-xs whitespace-nowrap">{at}</span></TableCell>
                            <TableCell>{form}</TableCell>
                        </TableRow>
                    }
                })
                .collect_view();
            view! {
                <TableWrapper><Table>
                    <TableHeader><TableRow>
                        <TableHead>"Version (UTC)"</TableHead>
                        <TableHead secondary=true>"Digest"</TableHead>
                        <TableHead secondary=true>"Logged"</TableHead>
                        <TableHead>""</TableHead>
                    </TableRow></TableHeader>
                    <TableBody>{rows}</TableBody>
                </Table></TableWrapper>
            }
            .into_any()
        }
    };
    let paste = expected.map(|v| paste_import_form(csrf, v));
    view! {
        <div class="space-y-6">
            {header}
            <Card>
                <CardHeader><CardTitle>"The store and the guard"</CardTitle></CardHeader>
                <CardContent><div class="space-y-3">{state}{guard}</div></CardContent>
            </Card>
            <Card>
                <CardHeader>
                    <CardTitle>"Documents the console committed"</CardTitle>
                    <CardDescription>"From the console's write log, highest version first. The newest the console committed is the one to import, unless it is the damage being recovered from, and every preview diffs against it. A copy without a committed label, or whose version the log names with another digest too, is not one the console committed as it stands: only a fault, or a write to the log store from outside the console, makes one."</CardDescription>
                </CardHeader>
                <CardContent>{labels}{unlisted}{malformed}{rows}</CardContent>
            </Card>
            <Card>
                <CardHeader>
                    <CardTitle>"Paste a document"</CardTitle>
                    <CardDescription>"Its exact JSON text, or its base64 as an instance's install log carries it, with the parts of a long one joined in order, and the digest the log names beside it. A document written by hand has no logged digest, and is checked like any other."</CardDescription>
                </CardHeader>
                <CardContent>{paste}</CardContent>
            </Card>
        </div>
    }
}

fn logged_import_form(csrf: &str, expected: u64, d: &writelog::Logged) -> impl IntoView + use<> {
    let (version, digest) = (d.version.to_string(), d.digest.clone());
    view! {
        <form method="post" action="/lb/clients/preview" class="flex items-center gap-2">
            {pinned(csrf, expected, "import")}
            <input type="hidden" name="source" value="log" />
            <input type="hidden" name="version" value=version />
            <input type="hidden" name="digest" value=digest />
            {reason_input(row_input())}
            <button type="submit" class=btn_sm(Btn::Outline)>"Preview import…"</button>
        </form>
    }
}

fn paste_import_form(csrf: &str, expected: u64) -> impl IntoView + use<> {
    view! {
        <form method="post" action="/lb/clients/preview" class="space-y-2">
            {pinned(csrf, expected, "import")}
            <input type="hidden" name="source" value="paste" />
            <textarea class=TEXTAREA_CLASS name="document" rows="8" required spellcheck="false" placeholder="{\"schema\":1, …} or its base64"></textarea>
            <input class=INPUT_CLASS name="digest" autocomplete="off" spellcheck="false" maxlength="64" placeholder="its digest, as the install log names it (none for a document written by hand)" />
            {reason_input(INPUT_CLASS.into())}
            <button type="submit" class=btn(Btn::Outline)>"Preview import…"</button>
        </form>
    }
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
        Err(e) => view! {
            <div class="space-y-6">
                {sub_header("Document history".into(), HISTORY.into())}
                {unavailable("client store", e.detail())}
            </div>
        }
        .into_any(),
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
        .or_else(|| op.get("digest").and_then(Value::as_str).map(short_digest))
        .or_else(|| {
            op.get("version")
                .and_then(Value::as_u64)
                .map(|v| v.to_string())
        })
        .unwrap_or_default();
    clip(format!("{kind} {target}").trim(), 120)
}

/// What the history page is for.
const HISTORY: &str = "Every document the store committed, newest first, with what each changed. Restoring one previews the change first and commits only the row the preview showed. It is refused if instances installed that version with another digest, since the row was edited after the fact, and needs its version typed if no instance installed it in the last 30 days. Restore never brings back a revoked key.";

fn history_view(store: &Value, csrf: &str) -> impl IntoView + use<> {
    let Some(doc) = store
        .get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))
    else {
        return view! {
            <div class="space-y-6">
                {sub_header("Document history".into(), HISTORY.into())}
                {unavailable("document history", "the document side of the store did not answer")}
            </div>
        }
        .into_any();
    };
    // The version every restore form pins. Unreadable, or a store at no
    // addressable version: the rows still show, and no form is offered.
    let (current, unreadable) = match effect::served_doc(store) {
        Ok(Some(d)) => (Some(d.version), None),
        Ok(None) => (None, None),
        Err(e) => (
            None,
            Some(format!(
                "The store's document is unreadable ({}), so nothing can be restored.",
                clip(&e, 160)
            )),
        ),
    };
    let unreadable = unreadable.map(|why| {
        view! {
            <div class="mb-3">
                <Notice status=Status::Error label="unreadable">{why}</Notice>
            </div>
        }
    });
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
            let short = short_digest(&digest);
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
                .map(|l| view! { <li class="wrap-break-word">{l}</li> })
                .collect_view();
            let action = match (version, current) {
                (Some(v), Some(c)) if v == c => view! { <StatusBadge status=Status::Ok label="current" /> }.into_any(),
                (Some(v), Some(c)) => restore_form(csrf, c, v).into_any(),
                _ => view! { <span class="text-xs">"—"</span> }.into_any(),
            };
            let (at, reason) = (s("at", 40), s("reason", 200));
            view! {
                <TableRow>
                    <TableCell><span class="tabular-nums whitespace-nowrap" title=exact>{when}</span></TableCell>
                    <TableCell secondary=true><span class="font-mono text-xs" title=digest>{short}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs whitespace-nowrap">{at}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs wrap-break-word">{who}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs wrap-break-word">{reason}</span></TableCell>
                    <TableCell secondary=true><span class="text-xs font-mono wrap-break-word">{ops}</span></TableCell>
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
        let form = current.map(|c| restore_by_version_form(csrf, c));
        view! {
            <details class="mt-6">
                <summary class="text-sm cursor-pointer">{summary}</summary>
                <p class="text-xs text-muted-foreground my-2">"Rows past the 50 newest are listed by version only. A restore of one is checked like any other: its preview names the row's digest, and it is refused if instances installed that version with another digest."</p>
                <p class="text-xs font-mono break-all mb-3">{list}</p>
                {form}
            </details>
        }
    });
    let title = match current {
        Some(c) => format!("Document history (current version {c})"),
        None => "Document history (nothing can be restored now)".to_string(),
    };
    view! {
        <div class="space-y-6">
            {sub_header(title, HISTORY.into())}
            <Card>
                <CardContent>
                    {unreadable}
                    <TableWrapper><Table>
                        <TableHeader><TableRow>
                            <TableHead>"Version (UTC)"</TableHead>
                            <TableHead secondary=true>"Digest"</TableHead>
                            <TableHead secondary=true>"At"</TableHead>
                            <TableHead secondary=true>"By"</TableHead>
                            <TableHead secondary=true>"Reason"</TableHead>
                            <TableHead secondary=true>"Ops"</TableHead>
                            <TableHead>"Changes"</TableHead>
                            <TableHead>""</TableHead>
                        </TableRow></TableHeader>
                        <TableBody>{table}</TableBody>
                    </Table></TableWrapper>
                    {older_view}
                </CardContent>
            </Card>
        </div>
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

/// The reason every document op carries, in `class`: `INPUT_CLASS` in a
/// form of its own, `row_input()` in a table row.
fn reason_input(class: String) -> impl IntoView {
    view! { <input class=class name="reason" required maxlength="200" placeholder="reason" /> }
}

fn restore_form(csrf: &str, current: u64, version: u64) -> impl IntoView + use<> {
    let version = version.to_string();
    view! {
        <form method="post" action="/lb/clients/preview" class="flex items-center gap-2">
            {pinned(csrf, current, "restore")}
            <input type="hidden" name="version" value=version />
            {reason_input(row_input())}
            <button type="submit" class=btn_sm(Btn::Outline)>"Restore…"</button>
        </form>
    }
}

fn restore_by_version_form(csrf: &str, current: u64) -> impl IntoView + use<> {
    view! {
        <form method="post" action="/lb/clients/preview" class="flex flex-col gap-3 sm:flex-row sm:items-end">
            {pinned(csrf, current, "restore")}
            <input class=INPUT_CLASS name="version" required inputmode="numeric" placeholder="version" />
            {reason_input(INPUT_CLASS.into())}
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
            unavailable(
                "document ops",
                "the document side of the store did not answer. Restriction ops still work.",
            )
            .into_any(),
        );
    };
    let version = match effect::served_doc(store) {
        Ok(Some(d)) => d.version,
        Ok(None) => {
            return clients_shell(
                "Clients".into(),
                view! {
                    <Notice status=Status::Warn label="no document">
                        "The store holds no document at a version a write can address, so it takes no document op but the recovery ops: "
                        <Link href="/lb/clients/pitr">"a point-in-time recovery"</Link>
                        ", or "
                        <Link href="/lb/clients/import">"an import"</Link>
                        "."
                    </Notice>
                }
                .into_any(),
            );
        }
        Err(e) => {
            let why = format!(
                "the store's document is unreadable ({}). Restriction ops still work.",
                clip(&e, 160)
            );
            return clients_shell(
                "Clients".into(),
                unavailable("document ops", &why).into_any(),
            );
        }
    };
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
                <TableHead secondary=true>"Utilization"</TableHead>
                <TableHead secondary=true>"Models"</TableHead>
                <TableHead secondary=true>"Pins"</TableHead>
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
                    <Link href="/lb/clients/history">"History and restore"</Link>
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
                <li class="flex flex-wrap items-center gap-2 mb-1">
                    <span class="font-mono text-xs wrap-break-word" title=digest>{text}</span>
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
            <TableCell><span class="font-mono text-xs wrap-break-word">{name}</span></TableCell>
            <TableCell><span class="tabular-nums" title=budget_exact>{budget}</span></TableCell>
            <TableCell secondary=true><span class="tabular-nums">{util}</span></TableCell>
            <TableCell secondary=true><span class="text-xs font-mono wrap-break-word">{models}</span></TableCell>
            <TableCell secondary=true><span class="text-xs font-mono wrap-break-word">{pins}</span></TableCell>
            <TableCell><ul>{key_list}{more_keys}</ul></TableCell>
            <TableCell>
                {manage}
                <span class="text-xs"><Link href=disable>"disable…"</Link></span>
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
            <form method="post" action="/lb/clients/preview" class="flex flex-col gap-2">
                {pinned(csrf, version, "add_key")}
                <input type="hidden" name="name" value=name />
                <input class=INPUT_CLASS name="label" maxlength="64" placeholder="key label, e.g. 2026-10 rotation" />
                {reason_input(INPUT_CLASS.into())}
                <button type="submit" class=btn_sm(Btn::Outline)>"Add key…"</button>
            </form>
        }
        .into_any()
    };
    let (n1, n2) = (name.to_string(), name.to_string());
    view! {
        <details class="mb-1">
            <summary class="text-xs cursor-pointer underline underline-offset-4">"manage…"</summary>
            <div class="space-y-3 mt-2 min-w-64 sm:min-w-96">
                <form method="post" action="/lb/clients/preview" class="space-y-2">
                    {pinned(csrf, version, "update")}
                    <input type="hidden" name="name" value=n1 />
                    {policy}
                    {reason_input(INPUT_CLASS.into())}
                    <button type="submit" class=btn_sm(Btn::Outline)>"Update policy…"</button>
                </form>
                {add_key}
                <form method="post" action="/lb/clients/preview" class="flex flex-col gap-2">
                    {pinned(csrf, version, "delete")}
                    <input type="hidden" name="name" value=n2 />
                    {reason_input(INPUT_CLASS.into())}
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
        <div class="grid gap-2 sm:grid-cols-2">
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
            {reason_input(row_input())}
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
                {reason_input(INPUT_CLASS.into())}
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
            <form method="post" action="/lb/clients/preview" class="flex flex-col gap-3 sm:flex-row sm:items-end">
                {pinned(csrf, version, "revoke_key")}
                <input class=INPUT_CLASS name="digest" required maxlength="64" placeholder="key digest (64 hex)" />
                {reason_input(INPUT_CLASS.into())}
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
                "expected_digest" => f.expected_digest = v,
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
            DocOp::Restore {
                version: 9,
                expected_digest: None,
            },
            DocOp::Restore {
                version: 9,
                expected_digest: Some("c".repeat(64)),
            },
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
        assert!(
            with(&[
                ("op", "restore"),
                ("version", "9"),
                ("expected_digest", "abc")
            ])
            .is_err()
        );
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
