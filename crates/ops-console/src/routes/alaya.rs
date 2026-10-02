//! Ālaya module: memory browse/search, detail, and the full curation set —
//! supersede, delete, merge-duplicates, relations, contradiction resolution
//! (AC2–AC7). Every mutation is a POST form carrying the session CSRF token;
//! results follow POST-redirect-GET with a flash banner.

use axum::extract::{Path, Query, RawQuery, State};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::AppError;
use crate::routes::{clip, fmt_epoch, short_hash, validate_hash, vf, vs};
use crate::session::{Flash, Session, flash_cookie, take_flash};
use crate::state::AppState;
use crate::ui::*;

// ─── Value helpers (defensive rendering over upstream JSON) ────────────────

/// Ceiling on an unusable `content_hash` answer in the log line. Capped
/// because it is upstream text bounded only by `MAX_BODY_BYTES`, so logging
/// it whole hands a compromised upstream a megabyte of pod log per attempt.
/// The cap is 64, a well-formed hash's own length, so an unmarked cut of a
/// 65-char answer whose first 64 are valid hex would log as a perfect hash on
/// a line saying there was no usable one — reading as though `validate_hash`
/// had rejected a good hash, and inviting a manual supersede onto an
/// upstream-chosen target. `clip` marks the cut.
const MAX_LOGGED_HASH_CHARS: usize = 64;

fn excerpt(v: &Value, max: usize) -> String {
    let text = v
        .get("summary")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| v.get("content").and_then(|x| x.as_str()))
        .unwrap_or("");
    clip(text, max)
}

fn is_superseded(v: &Value) -> bool {
    v.get("metadata")
        .and_then(|m| m.get("superseded_by"))
        .map(|s| !s.is_null())
        .unwrap_or(false)
}

fn flash_redirect(
    jar: PrivateCookieJar,
    secure: bool,
    kind: &str,
    msg: String,
    to: &str,
) -> Response {
    let jar = flash_cookie(
        jar,
        &Flash {
            kind: kind.into(),
            msg,
        },
        secure,
    );
    (jar, Redirect::to(to)).into_response()
}

fn memory_href(hash: &str) -> String {
    format!("/alaya/memory/{hash}")
}

// ─── Browse / search (AC2) ──────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct BrowseQuery {
    #[serde(default)]
    q: String,
    mode: Option<String>,
    #[serde(default)]
    memory_type: String,
    #[serde(default)]
    tags: String,
    include_superseded: Option<String>,
    #[serde(default = "one")]
    page: usize,
    cursor: Option<f64>,
}

fn one() -> usize {
    1
}

const PAGE_SIZE: usize = 20;

/// The browse modes the console offers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Hybrid,
    Scan,
    Recent,
    Tag,
}

/// The inputs a mode's server path reads. `include_superseded` is read by
/// every mode, so it is not listed.
#[derive(Debug, PartialEq, Eq)]
struct Applies {
    query: bool,
    memory_type: bool,
    tags: bool,
}

impl Mode {
    const ALL: [Mode; 4] = [Mode::Hybrid, Mode::Scan, Mode::Recent, Mode::Tag];

    fn as_str(self) -> &'static str {
        match self {
            Mode::Hybrid => "hybrid",
            Mode::Scan => "scan",
            Mode::Recent => "recent",
            Mode::Tag => "tag",
        }
    }

    fn parse(s: &str) -> Result<Self, AppError> {
        Mode::ALL
            .into_iter()
            .find(|m| m.as_str() == s)
            .ok_or_else(|| {
                AppError::BadRequest("unknown mode (expected hybrid, scan, recent or tag)".into())
            })
    }

    /// Read off alaya-core's `search_hybrid` / `search_scan` /
    /// `search_recent` / `search_tag`. Hybrid matches tags from the query's
    /// own keywords, never from `tags`; tag mode never reads `memory_type`;
    /// only hybrid reads the query.
    fn applies(self) -> Applies {
        let (query, memory_type, tags) = match self {
            Mode::Hybrid => (true, true, false),
            Mode::Scan | Mode::Recent => (false, true, false),
            Mode::Tag => (false, false, true),
        };
        Applies {
            query,
            memory_type,
            tags,
        }
    }
}

/// One browse request, cut down to what its mode applies. Every link the
/// page offers is built from it, so no link carries an input its mode would
/// drop, and `ignored` names what the request carried that the mode drops.
struct BrowseView {
    mode: Mode,
    q: String,
    memory_type: String,
    tags: String,
    include_superseded: bool,
    page: usize,
    cursor: Option<f64>,
    ignored: Vec<&'static str>,
}

impl BrowseView {
    fn new(q: BrowseQuery) -> Result<Self, AppError> {
        let mode = match q.mode.as_deref() {
            Some(m) if !m.is_empty() => Mode::parse(m)?,
            _ if q.q.trim().is_empty() => Mode::Scan,
            _ => Mode::Hybrid,
        };
        let applies = mode.applies();
        let mut ignored = Vec::new();
        let mut keep = |applied: bool, name: &'static str, value: String| {
            if applied {
                return value;
            }
            if !value.trim().is_empty() {
                ignored.push(name);
            }
            String::new()
        };
        let query = keep(applies.query, "query", q.q);
        let memory_type = keep(applies.memory_type, "type", q.memory_type);
        let tags = keep(applies.tags, "tags", q.tags);
        // Recent pages by cursor and the rest by page number; each mode
        // ignores the other's.
        let (page, cursor) = if mode == Mode::Recent {
            if q.page > 1 {
                ignored.push("page");
            }
            (1, q.cursor)
        } else {
            if q.cursor.is_some() {
                ignored.push("cursor");
            }
            (q.page.max(1), None)
        };
        Ok(BrowseView {
            mode,
            q: query,
            memory_type,
            tags,
            include_superseded: q.include_superseded.is_some(),
            page,
            cursor,
            ignored,
        })
    }

    /// This view in `mode`, on its first page.
    fn in_mode(&self, mode: Mode) -> BrowseView {
        BrowseView {
            mode,
            q: self.q.clone(),
            memory_type: self.memory_type.clone(),
            tags: self.tags.clone(),
            include_superseded: self.include_superseded,
            page: 1,
            cursor: None,
            ignored: Vec::new(),
        }
    }

    fn href(&self, page: usize, cursor: Option<f64>) -> String {
        let applies = self.mode.applies();
        let mut qs = url::form_urlencoded::Serializer::new(String::new());
        qs.append_pair("mode", self.mode.as_str());
        for (applied, key, value) in [
            (applies.query, "q", &self.q),
            (applies.memory_type, "memory_type", &self.memory_type),
            (applies.tags, "tags", &self.tags),
        ] {
            if applied && !value.is_empty() {
                qs.append_pair(key, value);
            }
        }
        if self.include_superseded {
            qs.append_pair("include_superseded", "on");
        }
        if page > 1 {
            qs.append_pair("page", &page.to_string());
        }
        if let Some(c) = cursor {
            qs.append_pair("cursor", &c.to_string());
        }
        format!("/alaya?{}", qs.finish())
    }

    /// The input a mode cannot run without, when it is missing: the server
    /// would refuse the search, so the page asks for it instead.
    fn missing_input(&self) -> Option<&'static str> {
        match self.mode {
            Mode::Hybrid if self.q.trim().is_empty() => {
                Some("Enter a query to run a hybrid search.")
            }
            Mode::Tag if self.tags.trim().is_empty() => Some("Enter one or more tags."),
            _ => None,
        }
    }

    /// The search body: only what the mode applies, so the request says
    /// exactly what the server will do.
    fn upstream(&self) -> Value {
        let mut params = json!({
            "mode": self.mode.as_str(),
            "page": self.page,
            "page_size": PAGE_SIZE,
            "include_superseded": self.include_superseded,
            "output": "both",
        });
        if !self.q.is_empty() {
            params["query"] = json!(self.q);
        }
        if !self.memory_type.is_empty() {
            params["memory_type"] = json!(self.memory_type);
        }
        if !self.tags.trim().is_empty() {
            params["tags"] = json!(self.tags);
        }
        if let Some(c) = self.cursor {
            params["cursor"] = json!(c);
        }
        params
    }
}

/// Paging links and the count line, from the server's own paging fields.
/// Next appears only when the server says there is more.
struct Pager {
    summary: String,
    prev: Option<String>,
    next: Option<String>,
    first: Option<String>,
}

fn pager(view: &BrowseView, res: &Value, shown: usize) -> Pager {
    let has_more = res.get("has_more").and_then(Value::as_bool) == Some(true);
    let page = view.page;
    let prev = (page > 1).then(|| view.href(page - 1, None));
    let next = has_more.then(|| view.href(page + 1, None));
    match view.mode {
        // A cursor only goes forward without JS: "first page", never Prev.
        Mode::Recent => Pager {
            summary: match view.cursor {
                Some(c) => format!("{shown} memories created before {}", fmt_epoch(c)),
                None => format!("{shown} newest memories"),
            },
            prev: None,
            next: res
                .get("next_cursor")
                .and_then(Value::as_f64)
                .filter(|_| has_more)
                .map(|c| view.href(1, Some(c))),
            first: view.cursor.map(|_| view.href(1, None)),
        },
        // Hybrid ranks a bounded candidate pool, not the corpus: `total` is
        // that pool, so it is labelled as one.
        Mode::Hybrid => Pager {
            summary: match res.get("total").and_then(Value::as_u64) {
                Some(total) => {
                    let pages = res.get("total_pages").and_then(Value::as_u64).unwrap_or(1);
                    format!("Top {total} candidates for this query · page {page} of {pages}")
                }
                None => format!("{shown} results · page {page}"),
            },
            prev,
            next,
            first: None,
        },
        Mode::Scan | Mode::Tag => Pager {
            summary: format!("{shown} memories · page {page}"),
            prev,
            next,
            first: None,
        },
    }
}

fn notice(label: &'static str, msg: String) -> impl IntoView + use<> {
    view! {
        <p class="text-sm" role="alert">
            <span class=badge(BadgeKind::Warning)>{label}</span>
            " "{msg}
        </p>
    }
}

fn result_row(m: &Value) -> impl IntoView + use<> {
    let hash = vs(m, "content_hash");
    let mtype = vs(m, "memory_type");
    let text = excerpt(m, 140);
    let tags: Vec<String> = m
        .get("tags")
        .and_then(|t| t.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .take(4)
                .collect()
        })
        .unwrap_or_default();
    let created = fmt_epoch(vf(m, "created_at"));
    let superseded = is_superseded(m);
    view! {
        <TableRow>
            <TableCell><HashLink hash=hash /></TableCell>
            <TableCell><span class=badge(BadgeKind::Secondary)>{mtype}</span></TableCell>
            <TableCell>
                <span class="text-sm">{text}</span>
                {superseded.then(|| view! {
                    <span class=format!("ml-2 {}", badge(BadgeKind::Warning))>"superseded"</span>
                })}
            </TableCell>
            <TableCell>
                <div class="flex flex-wrap gap-1">
                    {tags.into_iter().map(|t| view! {
                        <span class=badge(BadgeKind::Muted)>{t}</span>
                    }).collect_view()}
                </div>
            </TableCell>
            <TableCell><span class="text-xs text-muted-foreground whitespace-nowrap">{created}</span></TableCell>
        </TableRow>
    }
}

/// The search form for `view`'s mode: only the inputs that mode applies.
/// Without JS a form cannot re-shape itself when a select changes, so the
/// mode is picked by link and the form posts it back hidden.
fn browse_form(view: &BrowseView) -> impl IntoView + use<> {
    let applies = view.mode.applies();
    let tabs = Mode::ALL
        .into_iter()
        .map(|m| {
            let (class, current) = if m == view.mode {
                (btn_sm(Btn::Default), Some("page"))
            } else {
                (btn_sm(Btn::Outline), None)
            };
            let href = view.in_mode(m).href(1, None);
            view! { <a class=class href=href aria-current=current>{m.as_str()}</a> }
        })
        .collect_view();
    let q = view.q.clone();
    let tags = view.tags.clone();
    let memory_type = view.memory_type.clone();
    let include_superseded = view.include_superseded;
    view! {
        <div class="flex flex-wrap gap-2 mb-4">{tabs}</div>
        <form method="get" action="/alaya" class="flex flex-wrap items-end gap-3">
            <input type="hidden" name="mode" value=view.mode.as_str() />
            {applies.query.then(|| view! {
                <div class="flex flex-col gap-1.5 grow min-w-56">
                    <label class=LABEL_CLASS for="q">"Query"</label>
                    <input class=INPUT_CLASS id="q" name="q" value=q placeholder="semantic query" />
                </div>
            })}
            {applies.memory_type.then(|| view! {
                <div class="flex flex-col gap-1.5">
                    <label class=LABEL_CLASS for="memory_type">"Type"</label>
                    <select class=SELECT_CLASS id="memory_type" name="memory_type">
                        {["", "note", "decision", "task", "reference"].into_iter().map(|t| {
                            let selected = t == memory_type;
                            let label = if t.is_empty() { "any" } else { t };
                            view! { <option value=t selected=selected>{label}</option> }
                        }).collect_view()}
                    </select>
                </div>
            })}
            {applies.tags.then(|| view! {
                <div class="flex flex-col gap-1.5 grow min-w-56">
                    <label class=LABEL_CLASS for="tags">"Tags (csv)"</label>
                    <input class=INPUT_CLASS id="tags" name="tags" value=tags />
                </div>
            })}
            <label class=format!("{LABEL_CLASS} h-9")>
                <input type="checkbox" name="include_superseded" checked=include_superseded />
                "include superseded"
            </label>
            <button type="submit" class=btn(Btn::Default)>"Search"</button>
        </form>
    }
}

pub async fn browse(
    State(state): State<AppState>,
    session: Session,
    Query(q): Query<BrowseQuery>,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let view = BrowseView::new(q)?;
    let (jar, flash) = take_flash(jar);

    let ignored = (!view.ignored.is_empty()).then(|| {
        notice(
            "not applied",
            format!(
                "{} mode does not apply: {}. The server ignores these, so the results below are not filtered by them.",
                view.mode.as_str(),
                view.ignored.join(", "),
            ),
        )
    });
    // Hybrid hands `memory_type` to the vector search only; its keyword and
    // graph candidates are not type-filtered upstream.
    let partial_type = (view.mode == Mode::Hybrid && !view.memory_type.is_empty()).then(|| {
        notice(
            "partial",
            "In hybrid mode the server applies the type filter to semantic matches only; keyword and graph matches of other types can still appear.".into(),
        )
    });

    let (results, pager) = match view.missing_input() {
        Some(ask) => (
            Vec::new(),
            Pager {
                summary: ask.into(),
                prev: None,
                next: None,
                first: None,
            },
        ),
        None => {
            let res = state.alaya.search(view.upstream()).await?;
            let results = res
                .get("results")
                .and_then(|r| r.as_array())
                .cloned()
                .unwrap_or_default();
            let pager = pager(&view, &res, results.len());
            (results, pager)
        }
    };
    let rows = results.iter().map(result_row).collect_view();
    let form = browse_form(&view);
    let Pager {
        summary,
        prev,
        next,
        first,
    } = pager;

    let content = view! {
        <div class="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>"Memories"</CardTitle>
                    <CardDescription>"Search or browse the corpus. Each mode offers only the filters the server applies in it. Hybrid ranks a bounded pool of candidates for the query, not the whole corpus; browse everything with scan."</CardDescription>
                </CardHeader>
                <CardContent>{form}</CardContent>
            </Card>

            {ignored}
            {partial_type}
            <div class="text-sm text-muted-foreground">{summary}</div>
            <TableWrapper>
                <Table>
                    <TableHeader>
                        <TableRow>
                            <TableHead>"Hash"</TableHead>
                            <TableHead>"Type"</TableHead>
                            <TableHead>"Content"</TableHead>
                            <TableHead>"Tags"</TableHead>
                            <TableHead>"Created"</TableHead>
                        </TableRow>
                    </TableHeader>
                    <TableBody>{rows}</TableBody>
                </Table>
            </TableWrapper>
            <div class="flex gap-3">
                {first.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"↑ First page"</a> })}
                {prev.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"← Prev"</a> })}
                {next.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"Next →"</a> })}
            </div>
        </div>
    };

    Ok((
        jar,
        Html(page("Memories — ops console", &session, flash, content)),
    ))
}

fn urlenc(s: &str) -> String {
    url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
}

// ─── Detail (AC3) ───────────────────────────────────────────────────────────

/// Forward walk of the supersession chain from a memory's
/// `metadata.superseded_by`: `(hash, excerpt)` per hop, bounded to 10 hops,
/// cycle-proof. Fetch failures render as "(unavailable)" — the audit trail is
/// shown even when a link is unreadable, never silently dropped.
async fn supersession_chain(
    alaya: &crate::alaya::AlayaClient,
    mem: &Value,
) -> Vec<(String, String)> {
    let mut chain: Vec<(String, String)> = Vec::new();
    let mut cursor = mem
        .get("metadata")
        .and_then(|m| m.get("superseded_by"))
        .and_then(|s| s.as_str())
        .map(String::from);
    while let Some(next) = cursor.take() {
        if chain.len() >= 10 || validate_hash(&next).is_err() {
            break;
        }
        let label = match alaya.get_memory(&next).await {
            Ok(r) => {
                if let Some(m) = r.get("memory") {
                    cursor = m
                        .get("metadata")
                        .and_then(|md| md.get("superseded_by"))
                        .and_then(|s| s.as_str())
                        .filter(|h| *h != next && !chain.iter().any(|(seen, _)| seen == h))
                        .map(String::from);
                    excerpt(m, 100)
                } else {
                    "(unavailable)".to_string()
                }
            }
            Err(_) => "(unavailable)".to_string(),
        };
        chain.push((next, label));
    }
    chain
}

/// The detail page's relations, grouped by type and direction. A
/// CONTRADICTS edge links to its pair-review page, where the verdict lives.
/// A failed read says so, distinct from an empty list.
fn relations_view(hash: &str, relations: Result<Vec<Value>, AppError>, csrf: &str) -> AnyView {
    let relations = match relations {
        Err(e) => {
            return view! {
                <p class="text-sm mb-4" role="alert">
                    <span class=badge(BadgeKind::Destructive)>"unavailable"</span>
                    " "{format!("Could not load relations: {}", e.detail())}
                </p>
            }
            .into_any();
        }
        Ok(r) if r.is_empty() => {
            return view! {
                <p class="text-sm text-muted-foreground mb-4">"No relations."</p>
            }
            .into_any();
        }
        Ok(r) => r,
    };
    // Outgoing before incoming within a type.
    let mut groups: std::collections::BTreeMap<(String, bool), Vec<Value>> = Default::default();
    for e in relations {
        let incoming = vs(&e, "source") != hash;
        groups
            .entry((vs(&e, "relation_type"), incoming))
            .or_default()
            .push(e);
    }
    let sections = groups
        .into_iter()
        .map(|((rel_type, incoming), edges)| {
            let heading = if incoming {
                format!("{rel_type} · to this memory ({})", edges.len())
            } else {
                format!("{rel_type} · from this memory ({})", edges.len())
            };
            let rows = edges
                .iter()
                .map(|e| relation_row(hash, e, csrf, incoming))
                .collect_view();
            view! {
                <div class="mb-4">
                    <h3 class="text-sm font-medium mb-2">{heading}</h3>
                    <TableWrapper><Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Linked memory"</TableHead>
                                <TableHead>"Created"</TableHead>
                                <TableHead>""</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>{rows}</TableBody>
                    </Table></TableWrapper>
                </div>
            }
        })
        .collect_view();
    sections.into_any()
}

fn relation_row(hash: &str, e: &Value, csrf: &str, incoming: bool) -> impl IntoView + use<> {
    let source = vs(e, "source");
    let target = vs(e, "target");
    let rel_type = vs(e, "relation_type");
    // Link to the far end of the edge, whichever side this memory is.
    let other = if incoming {
        source.clone()
    } else {
        target.clone()
    };
    let created = fmt_epoch(vf(e, "created_at"));
    // The pair page reads the edge in its stored direction: a = source.
    let review = (rel_type == "CONTRADICTS"
        && validate_hash(&source).is_ok()
        && validate_hash(&target).is_ok())
    .then(|| {
        let href = format!("{QUEUE_PATH}/pair?a={source}&b={target}");
        view! { <a class=btn_sm(Btn::Outline) href=href>"Review pair"</a> }
    });
    let csrf = csrf.to_string();
    let back = memory_href(hash);
    view! {
        <TableRow>
            <TableCell><HashLink hash=other /></TableCell>
            <TableCell><span class="text-xs text-muted-foreground">{created}</span></TableCell>
            <TableCell>
                <div class="flex gap-2">
                    {review}
                    <form method="post" action="/alaya/relation/delete">
                        <input type="hidden" name="csrf" value=csrf />
                        <input type="hidden" name="content_hash" value=source />
                        <input type="hidden" name="target_hash" value=target />
                        <input type="hidden" name="relation_type" value=rel_type />
                        <input type="hidden" name="back" value=back />
                        <button type="submit" class=btn_sm(Btn::Outline)>"Delete"</button>
                    </form>
                </div>
            </TableCell>
        </TableRow>
    }
}

pub async fn detail(
    State(state): State<AppState>,
    session: Session,
    Path(hash): Path<String>,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    validate_hash(&hash)?;
    let (jar, flash) = take_flash(jar);

    let res = state.alaya.get_memory(&hash).await?;
    let mem = res
        .get("memory")
        .cloned()
        .ok_or_else(|| AppError::NotFound("memory not found".into()))?;

    // Relations are graph-backed and fail on their own; a failure renders
    // as a note in the Relations card, never as "No relations." and never
    // as a dead page.
    let relations = state
        .alaya
        .relation("get", &hash, None, None)
        .await
        .and_then(|r| {
            r.get("relations")
                .and_then(|x| x.as_array())
                .cloned()
                .ok_or_else(|| AppError::Upstream("alaya-server returned no relations list".into()))
        });

    let chain = supersession_chain(&state.alaya, &mem).await;

    let content_text = vs(&mem, "content");
    let summary = vs(&mem, "summary");
    let mtype = vs(&mem, "memory_type");
    let created = fmt_epoch(vf(&mem, "created_at"));
    let updated = fmt_epoch(vf(&mem, "updated_at"));
    let salience = format!("{:.3}", vf(&mem, "salience_score"));
    let access_count = mem
        .get("access_count")
        .and_then(|a| a.as_u64())
        .unwrap_or(0);
    let trust = mem
        .get("provenance")
        .and_then(|p| p.get("trust_score"))
        .and_then(|t| t.as_f64())
        .map(|t| format!("{t:.2}"))
        .unwrap_or_else(|| "—".into());
    let tags: Vec<String> = mem
        .get("tags")
        .and_then(|t| t.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let metadata_pretty = mem
        .get("metadata")
        .filter(|m| !m.is_null())
        .map(|m| serde_json::to_string_pretty(m).unwrap_or_default())
        .unwrap_or_else(|| "null".into());
    let superseded = is_superseded(&mem);
    let csrf = session.csrf.clone();

    let relations_view = relations_view(&hash, relations, &csrf);

    // view! wraps each expression in a move closure — every string below is
    // a dedicated local used exactly once inside the view.
    let supersede_href = format!("/alaya/supersede?old={hash}");
    let correct_action = format!("/alaya/memory/{hash}/correct");
    let delete_action = format!("/alaya/memory/{hash}/delete");
    let back_href = memory_href(&hash);
    let hash_title = hash.clone();
    let hash_hidden = hash.clone();
    let content_for_edit = content_text.clone();
    let summary_text = summary.clone();
    let (csrf_rel, csrf_correct, csrf_delete) = (csrf.clone(), csrf.clone(), csrf.clone());
    let content = view! {
        <div class="space-y-6">
            <div class="flex items-center gap-3 flex-wrap">
                <h1 class="font-mono text-sm">{hash_title}</h1>
                <span class=badge(BadgeKind::Secondary)>{mtype.clone()}</span>
                {superseded.then(|| view! { <span class=badge(BadgeKind::Warning)>"superseded"</span> })}
            </div>

            <Card>
                <CardHeader><CardTitle>"Content"</CardTitle></CardHeader>
                <CardContent>
                    <pre class="whitespace-pre-wrap text-sm font-sans">{content_text}</pre>
                    {(!summary.is_empty()).then(|| view! {
                        <div class="mt-4 border-t pt-4">
                            <div class="text-xs font-medium text-muted-foreground mb-1">"Summary"</div>
                            <p class="text-sm">{summary_text}</p>
                        </div>
                    })}
                </CardContent>
            </Card>

            <Card>
                <CardHeader><CardTitle>"Stats & metadata"</CardTitle></CardHeader>
                <CardContent>
                    <dl class="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm mb-4">
                        <div><dt class="text-muted-foreground text-xs">"Created"</dt><dd>{created}</dd></div>
                        <div><dt class="text-muted-foreground text-xs">"Updated"</dt><dd>{updated}</dd></div>
                        <div><dt class="text-muted-foreground text-xs">"Salience"</dt><dd>{salience}</dd></div>
                        <div><dt class="text-muted-foreground text-xs">"Accesses"</dt><dd>{access_count}</dd></div>
                        <div><dt class="text-muted-foreground text-xs">"Trust"</dt><dd>{trust}</dd></div>
                    </dl>
                    <div class="flex flex-wrap gap-1 mb-4">
                        {tags.into_iter().map(|t| view! { <span class=badge(BadgeKind::Muted)>{t}</span> }).collect_view()}
                    </div>
                    <pre class="text-xs bg-muted rounded-md p-3 overflow-auto">{metadata_pretty}</pre>
                </CardContent>
            </Card>

            {(!chain.is_empty()).then(|| view! {
                <Card>
                    <CardHeader>
                        <CardTitle>"Supersession chain"</CardTitle>
                        <CardDescription>"This memory was superseded — the audit trail is preserved; nothing is dropped."</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ol class="space-y-2 text-sm">
                            {chain.iter().map(|(h, label)| {
                                let h = h.clone();
                                let label = label.clone();
                                view! {
                                    <li class="flex gap-2 items-baseline">
                                        <span class="text-muted-foreground">"↳"</span>
                                        <HashLink hash=h />
                                        <span class="text-muted-foreground">{label}</span>
                                    </li>
                                }
                            }).collect_view()}
                        </ol>
                    </CardContent>
                </Card>
            })}

            <Card>
                <CardHeader><CardTitle>"Relations"</CardTitle></CardHeader>
                <CardContent>
                    {relations_view}
                    <form method="post" action="/alaya/relation/create" class="flex flex-wrap items-end gap-3">
                        <input type="hidden" name="csrf" value=csrf_rel />
                        <input type="hidden" name="content_hash" value=hash_hidden />
                        <input type="hidden" name="back" value=back_href />
                        <div class="flex flex-col gap-1.5 grow min-w-72">
                            <label class=LABEL_CLASS for="target_hash">"Target hash"</label>
                            <input class=INPUT_CLASS id="target_hash" name="target_hash" placeholder="64-char content hash" required />
                        </div>
                        <div class="flex flex-col gap-1.5">
                            <label class=LABEL_CLASS for="relation_type">"Type"</label>
                            <select class=SELECT_CLASS id="relation_type" name="relation_type">
                                <option value="RELATES_TO">"RELATES_TO"</option>
                                <option value="PRECEDES">"PRECEDES"</option>
                                <option value="CONTRADICTS">"CONTRADICTS"</option>
                            </select>
                        </div>
                        <button type="submit" class=btn(Btn::Secondary)>"Create relation"</button>
                    </form>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>"Curation"</CardTitle>
                    <CardDescription>"Supersede keeps the audit trail; delete is permanent."</CardDescription>
                </CardHeader>
                <CardContent>
                    <div class="flex flex-wrap gap-3 mb-6">
                        <a href=supersede_href class=btn(Btn::Default)>"Supersede with existing…"</a>
                    </div>

                    <details class="mb-6">
                        <summary class="cursor-pointer text-sm font-medium">"Correct & supersede (store fixed copy, then supersede this one)"</summary>
                        <form method="post" action=correct_action class="mt-4 space-y-3">
                            <input type="hidden" name="csrf" value=csrf_correct />
                            <textarea class=TEXTAREA_CLASS name="content" rows="8" required>{content_for_edit}</textarea>
                            <input class=INPUT_CLASS name="reason" placeholder="reason for the correction" required />
                            <button type="submit" class=btn(Btn::Default)>"Store correction & supersede"</button>
                        </form>
                    </details>

                    <details>
                        <summary class="cursor-pointer text-sm font-medium text-destructive">"Delete permanently…"</summary>
                        <form method="post" action=delete_action class="mt-4 flex items-center gap-3">
                            <input type="hidden" name="csrf" value=csrf_delete />
                            <span class="text-sm text-muted-foreground">"This removes the memory and its audit trail. Prefer supersede."</span>
                            <button type="submit" class=btn(Btn::Destructive)>"Delete memory"</button>
                        </form>
                    </details>
                </CardContent>
            </Card>
        </div>
    };

    Ok((
        jar,
        Html(page("Memory — ops console", &session, flash, content)),
    ))
}

// ─── Mutations (AC4–AC6) ────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CsrfOnly {
    #[serde(default)]
    csrf: String,
}

pub async fn delete_memory(
    State(state): State<AppState>,
    session: Session,
    Path(hash): Path<String>,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<CsrfOnly>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&hash)?;
    state.alaya.delete(&hash).await?;
    tracing::info!(sub = ?session.sub, hash = %hash, "memory deleted");
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        "ok",
        format!("Deleted {}.", short_hash(&hash)),
        "/alaya",
    ))
}

#[derive(Deserialize)]
pub struct SupersedeQuery {
    #[serde(default)]
    old: String,
    #[serde(default)]
    new: String,
    /// Where to land after the supersede (the contradictions queue passes
    /// itself). Checked by `safe_next` on submit, not here.
    #[serde(default)]
    back: String,
}

pub async fn supersede_form(
    State(state): State<AppState>,
    session: Session,
    Query(q): Query<SupersedeQuery>,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);

    // Show excerpts for prefilled hashes so the operator sees what they're
    // about to supersede.
    let preview = |hash: String| async {
        if validate_hash(&hash).is_err() {
            return None;
        }
        state
            .alaya
            .get_memory(&hash)
            .await
            .ok()
            .and_then(|r| r.get("memory").map(|m| (hash, excerpt(m, 240))))
    };
    let old_preview = preview(q.old.clone()).await;
    let new_preview = preview(q.new.clone()).await;

    let csrf = session.csrf.clone();
    let preview_card = |title: &'static str, p: Option<(String, String)>| {
        p.map(|(h, text)| {
            view! {
                <div class="rounded-md border p-4">
                    <div class="text-xs font-medium text-muted-foreground mb-1">{title}</div>
                    <HashLink hash=h />
                    <p class="text-sm mt-2">{text}</p>
                </div>
            }
        })
    };

    let content = view! {
        <Card>
            <CardHeader>
                <CardTitle>"Supersede a memory"</CardTitle>
                <CardDescription>
                    "The old memory stays retrievable with a full audit trail (superseded_by, reason). Nothing is silently dropped."
                </CardDescription>
            </CardHeader>
            <CardContent>
                <div class="grid gap-4 sm:grid-cols-2 mb-6">
                    {preview_card("Old (will be superseded)", old_preview)}
                    {preview_card("New (canonical)", new_preview)}
                </div>
                <form method="post" action="/alaya/supersede" class="space-y-3 max-w-2xl">
                    <input type="hidden" name="csrf" value=csrf />
                    <div class="flex flex-col gap-1.5">
                        <label class=LABEL_CLASS for="old_hash">"Old hash (superseded)"</label>
                        <input class=INPUT_CLASS id="old_hash" name="old_hash" value=q.old required />
                    </div>
                    <div class="flex flex-col gap-1.5">
                        <label class=LABEL_CLASS for="new_hash">"New hash (canonical)"</label>
                        <input class=INPUT_CLASS id="new_hash" name="new_hash" value=q.new required />
                    </div>
                    <div class="flex flex-col gap-1.5">
                        <label class=LABEL_CLASS for="reason">"Reason (audit trail)"</label>
                        <input class=INPUT_CLASS id="reason" name="reason" placeholder="why the old memory is superseded" required />
                    </div>
                    <input type="hidden" name="back" value=q.back />
                    <button type="submit" class=btn(Btn::Default)>"Supersede"</button>
                </form>
            </CardContent>
        </Card>
    };

    Ok((
        jar,
        Html(page("Supersede — ops console", &session, flash, content)),
    ))
}

#[derive(Deserialize)]
pub struct SupersedeForm {
    #[serde(default)]
    csrf: String,
    old_hash: String,
    new_hash: String,
    #[serde(default)]
    reason: String,
    #[serde(default)]
    back: String,
}

pub async fn supersede_submit(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<SupersedeForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&form.old_hash)?;
    validate_hash(&form.new_hash)?;
    if form.reason.trim().is_empty() {
        return Err(AppError::BadRequest(
            "a reason is required for the audit trail".into(),
        ));
    }
    state
        .alaya
        .supersede(&form.old_hash, &form.new_hash, form.reason.trim())
        .await?;
    tracing::info!(sub = ?session.sub, old = %form.old_hash, new = %form.new_hash, "memory superseded");
    let to = crate::routes::return_to(&form.back, &memory_href(&form.old_hash));
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        "ok",
        format!(
            "Superseded {} → {}.",
            short_hash(&form.old_hash),
            short_hash(&form.new_hash)
        ),
        &to,
    ))
}

#[derive(Deserialize)]
pub struct CorrectForm {
    #[serde(default)]
    csrf: String,
    content: String,
    reason: String,
}

/// Store a corrected copy (same type + tags), then supersede the original.
pub async fn correct_and_supersede(
    State(state): State<AppState>,
    session: Session,
    Path(hash): Path<String>,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<CorrectForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&hash)?;
    if form.content.trim().is_empty() || form.reason.trim().is_empty() {
        return Err(AppError::BadRequest(
            "content and reason are required".into(),
        ));
    }

    let original = state.alaya.get_memory(&hash).await?;
    let mem = original
        .get("memory")
        .cloned()
        .ok_or_else(|| AppError::NotFound("memory not found".into()))?;

    let store_res = state
        .alaya
        .store(json!({
            "content": form.content,
            "memory_type": mem.get("memory_type"),
            "tags": mem.get("tags"),
        }))
        .await?;
    // alaya-server's answer, echoed verbatim — and until here the one field
    // in this crate recorded with `%` that never met a validator. The
    // plain-text subscriber writes a `Display` field raw, so a `\n` in it
    // forges whole pod-log records in the audit trail for this very write.
    // It has to be a 64-hex hash to survive `memory_href` and `supersede`
    // anyway, and it makes the `short_hash` below honest.
    //
    // One arm for absent, mistyped and malformed alike: the store has
    // already committed in every one of them, so all three owe the operator
    // the same thing — which memory is orphaned and how to find it. Splitting
    // them on JSON type gave the same upstream fault opposite guidance.
    let Some(new_hash) = store_res
        .get("content_hash")
        .and_then(|h| h.as_str())
        .filter(|h| validate_hash(h).is_ok())
        .map(str::to_string)
    else {
        let full = store_res
            .get("content_hash")
            .and_then(|h| h.as_str())
            .unwrap_or("<absent or not a string>");
        let answered = clip(full, MAX_LOGGED_HASH_CHARS);
        tracing::error!(sub = ?session.sub, old = %hash, answered = ?answered, "store returned no usable content_hash");
        return Err(AppError::Upstream(format!(
            "the correction WAS stored but alaya-server returned no usable id for it, \
             so {} could not be superseded — find the correction by searching for its \
             text, then supersede from the original memory's page",
            short_hash(&hash),
        )));
    };
    if new_hash == hash {
        return Err(AppError::BadRequest(
            "corrected content is identical to the original".into(),
        ));
    }

    // Two-phase mutation: the correction is already stored. If the supersede
    // half fails, the error must say exactly what landed — a generic 502
    // would hide a committed write on the memory of record.
    if let Err(e) = state
        .alaya
        .supersede(&hash, &new_hash, form.reason.trim())
        .await
    {
        tracing::error!(sub = ?session.sub, old = %hash, new = ?new_hash, "correction stored but supersede failed");
        let detail = match e {
            AppError::Upstream(d) | AppError::NotFound(d) => d,
            _ => "supersede failed".to_string(),
        };
        return Err(AppError::Upstream(format!(
            "correction WAS stored as {} but superseding {} failed ({detail}) — \
             retry the supersede from the original memory's page",
            short_hash(&new_hash),
            short_hash(&hash),
        )));
    }
    tracing::info!(sub = ?session.sub, old = %hash, new = ?new_hash, "corrected + superseded");
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        "ok",
        format!(
            "Stored correction {} and superseded {}.",
            short_hash(&new_hash),
            short_hash(&hash)
        ),
        &memory_href(&new_hash),
    ))
}

#[derive(Deserialize)]
pub struct RelationForm {
    #[serde(default)]
    csrf: String,
    content_hash: String,
    target_hash: String,
    relation_type: String,
    #[serde(default)]
    back: String,
}

pub async fn relation_create(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<RelationForm>,
) -> Result<Response, AppError> {
    relation_action(state, session, jar, form, "create").await
}

pub async fn relation_delete(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<RelationForm>,
) -> Result<Response, AppError> {
    relation_action(state, session, jar, form, "delete").await
}

async fn relation_action(
    state: AppState,
    session: Session,
    jar: PrivateCookieJar,
    form: RelationForm,
    action: &str,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&form.content_hash)?;
    validate_hash(&form.target_hash)?;
    if !["RELATES_TO", "PRECEDES", "CONTRADICTS"].contains(&form.relation_type.as_str()) {
        return Err(AppError::BadRequest("unknown relation type".into()));
    }
    state
        .alaya
        .relation(
            action,
            &form.content_hash,
            Some(&form.target_hash),
            Some(&form.relation_type),
        )
        .await?;
    tracing::info!(sub = ?session.sub, action = %action, source = %form.content_hash, target = %form.target_hash, rel = %form.relation_type, "relation changed");
    let back = crate::routes::safe_next(&form.back);
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        "ok",
        format!("Relation {}d.", action),
        &back,
    ))
}

// ─── Duplicates (AC5) ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct DuplicatesQuery {
    threshold: Option<f64>,
}

pub async fn duplicates(
    State(state): State<AppState>,
    session: Session,
    Query(q): Query<DuplicatesQuery>,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let threshold = q.threshold.unwrap_or(0.95).clamp(0.5, 1.0);

    let res = state.alaya.find_duplicates(threshold, 200).await?;
    let groups = res
        .get("groups")
        .and_then(|g| g.as_array())
        .cloned()
        .unwrap_or_default();
    let scanned = res
        .get("total_memories_scanned")
        .and_then(|n| n.as_u64())
        .unwrap_or(0);
    let csrf = session.csrf.clone();

    let group_cards = groups
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let csrf = csrf.clone();
            let canonical = vs(g, "canonical_hash");
            let hashes: Vec<String> = g
                .get("hashes")
                .and_then(|h| h.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
                .unwrap_or_default();
            let sim = format!("{:.3}", vf(g, "max_similarity"));
            let members = hashes
                .iter()
                .map(|h| {
                    let is_canonical = *h == canonical;
                    let h_radio = h.clone();
                    let h_check = h.clone();
                    let h_link = h.clone();
                    // The suggested canonical also rides along as a hidden
                    // duplicate_hashes value: if the operator moves the radio
                    // to another member, the old canonical becomes a merge
                    // candidate; merge_submit filters out whichever canonical
                    // was actually chosen, so the whole group always merges.
                    let h_hidden = h.clone();
                    view! {
                        <li class="flex items-center gap-3 text-sm">
                            <label class="flex items-center gap-1 text-xs text-muted-foreground">
                                <input type="radio" name="canonical_hash" value=h_radio checked=is_canonical />
                                "canonical"
                            </label>
                            {is_canonical.then(|| view! {
                                <input type="hidden" name="duplicate_hashes" value=h_hidden />
                            })}
                            <label class="flex items-center gap-1 text-xs text-muted-foreground">
                                <input type="checkbox" name="duplicate_hashes" value=h_check checked=!is_canonical />
                                "merge"
                            </label>
                            <HashLink hash=h_link />
                        </li>
                    }
                })
                .collect_view();
            let title = format!("Group {} — {} memories, similarity ≥ {}", i + 1, hashes.len(), sim);
            view! {
                <Card>
                    <CardHeader><CardTitle>{title}</CardTitle></CardHeader>
                    <CardContent>
                        <form method="post" action="/alaya/duplicates/merge" class="space-y-4">
                            <input type="hidden" name="csrf" value=csrf.clone() />
                            <ul class="space-y-2">{members}</ul>
                            <div class="flex flex-wrap items-end gap-3">
                                <div class="flex flex-col gap-1.5 grow min-w-56">
                                    <label class=LABEL_CLASS>"Reason"</label>
                                    <input class=INPUT_CLASS name="reason" value="Merged by ops-console deduplication" />
                                </div>
                                <label class=format!("{LABEL_CLASS} h-9")>
                                    <input type="checkbox" name="dry_run" checked=true />
                                    "dry run (preview)"
                                </label>
                                <button type="submit" class=btn(Btn::Default)>"Merge"</button>
                            </div>
                        </form>
                    </CardContent>
                </Card>
            }
        })
        .collect_view();

    let summary = format!(
        "{} duplicate groups (scanned {} memories, threshold {threshold})",
        groups.len(),
        scanned
    );
    let content = view! {
        <div class="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>"Duplicates"</CardTitle>
                    <CardDescription>
                        "Merging supersedes each duplicate in favour of the canonical memory — audit trail preserved. Dry-run first."
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form method="get" action="/alaya/duplicates" class="flex items-end gap-3">
                        <div class="flex flex-col gap-1.5">
                            <label class=LABEL_CLASS for="threshold">"Similarity threshold"</label>
                            <input class=INPUT_CLASS id="threshold" name="threshold" value=threshold.to_string() />
                        </div>
                        <button type="submit" class=btn(Btn::Secondary)>"Scan"</button>
                    </form>
                    <p class="text-sm text-muted-foreground mt-3">{summary}</p>
                </CardContent>
            </Card>
            {group_cards}
        </div>
    };

    Ok((
        jar,
        Html(page("Duplicates — ops console", &session, flash, content)),
    ))
}

#[derive(Deserialize)]
pub struct MergeForm {
    #[serde(default)]
    csrf: String,
    canonical_hash: String,
    #[serde(default)]
    duplicate_hashes: Vec<String>,
    #[serde(default)]
    reason: String,
    dry_run: Option<String>,
}

pub async fn merge_submit(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum_extra::extract::Form(mut form): axum_extra::extract::Form<MergeForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&form.canonical_hash)?;
    // The form's checkbox state doesn't re-sync when the operator moves the
    // canonical radio (no JS by design) — the chosen canonical may arrive
    // inside duplicate_hashes. Filter it out instead of half-merging.
    form.duplicate_hashes.retain(|h| *h != form.canonical_hash);
    if form.duplicate_hashes.is_empty() {
        return Err(AppError::BadRequest(
            "select at least one duplicate to merge (other than the canonical)".into(),
        ));
    }
    for h in &form.duplicate_hashes {
        validate_hash(h)?;
    }
    let dry_run = form.dry_run.is_some();

    let res = state
        .alaya
        .merge_duplicates(
            &form.canonical_hash,
            &form.duplicate_hashes,
            form.reason.trim(),
            dry_run,
        )
        .await?;

    if dry_run {
        // Render the preview (AC5): what WOULD be superseded.
        let (jar, _) = take_flash(jar);
        let superseded: Vec<String> = res
            .get("superseded")
            .and_then(|s| s.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let csrf = session.csrf.clone();
        let dup_field = form.duplicate_hashes.clone();
        let canonical_link = form.canonical_hash.clone();
        let canonical_hidden = form.canonical_hash.clone();
        let reason_hidden = form.reason.clone();
        let content = view! {
            <Card>
                <CardHeader>
                    <CardTitle>"Merge preview (dry run)"</CardTitle>
                    <CardDescription>"No changes were made. Review, then commit."</CardDescription>
                </CardHeader>
                <CardContent>
                    <p class="text-sm mb-2">
                        "Canonical: "
                        <HashLink hash=canonical_link />
                    </p>
                    <p class="text-sm mb-2">"Will supersede:"</p>
                    <ul class="space-y-1 mb-6">
                        {superseded.iter().map(|h| {
                            let h = h.clone();
                            view! { <li><HashLink hash=h /></li> }
                        }).collect_view()}
                    </ul>
                    <form method="post" action="/alaya/duplicates/merge" class="flex items-center gap-3">
                        <input type="hidden" name="csrf" value=csrf />
                        <input type="hidden" name="canonical_hash" value=canonical_hidden />
                        {dup_field.into_iter().map(|h| view! {
                            <input type="hidden" name="duplicate_hashes" value=h />
                        }).collect_view()}
                        <input type="hidden" name="reason" value=reason_hidden />
                        <button type="submit" class=btn(Btn::Destructive)>"Commit merge"</button>
                        <a href="/alaya/duplicates" class=btn(Btn::Outline)>"Cancel"</a>
                    </form>
                </CardContent>
            </Card>
        };
        return Ok((
            jar,
            Html(page("Merge preview — ops console", &session, None, content)),
        )
            .into_response());
    }

    let merged = res
        .get("superseded")
        .and_then(|s| s.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    let errors = res
        .get("errors")
        .and_then(|e| e.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    tracing::info!(sub = ?session.sub, canonical = %form.canonical_hash, merged, errors, "duplicates merged");
    let msg = if errors > 0 {
        format!(
            "Merged {merged} duplicates into {} ({errors} errors — see server logs).",
            short_hash(&form.canonical_hash)
        )
    } else {
        format!(
            "Merged {merged} duplicates into {}.",
            short_hash(&form.canonical_hash)
        )
    };
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        if errors > 0 { "error" } else { "ok" },
        msg,
        "/alaya/duplicates",
    ))
}

// ─── Contradictions triage (AC6, LAB-3283 verdicts) ────────────────────────

/// Queue page size, and so the cap on one bulk keep-both submit: the bulk
/// form can only carry what one page rendered.
const QUEUE_PAGE: usize = 50;

/// Every verdict class plus the unjudged sentinel — the values `verdict`
/// may take in the queue URL.
const ALL_VERDICTS: [&str; 5] = [
    "contradiction",
    "supersession",
    "coexist",
    "unrelated",
    "unjudged",
];

const QUEUE_PATH: &str = "/alaya/contradictions";

/// The queue's URL contract — repeatable `verdict`, `resolved=1`, `offset` —
/// shared by the queue, the pair page, and any pane that links into either,
/// so a refresh or a return keeps the operator's place. Parsed by hand
/// because axum's `Query` cannot take a repeated key.
#[derive(Default)]
struct QueueView {
    verdicts: Vec<String>,
    resolved: bool,
    offset: usize,
}

impl QueueView {
    /// Unknown keys are ignored (the pair page adds `a` and `b`); a bad
    /// value for a known key is a 400, before any upstream call.
    fn parse(raw: Option<&str>) -> Result<Self, AppError> {
        let mut view = QueueView::default();
        for (key, value) in url::form_urlencoded::parse(raw.unwrap_or("").as_bytes()) {
            match &*key {
                "verdict" => {
                    if !ALL_VERDICTS.contains(&&*value) {
                        return Err(AppError::BadRequest(format!(
                            "unknown verdict filter (expected one of {})",
                            ALL_VERDICTS.join(", ")
                        )));
                    }
                    view.verdicts.push(value.into_owned());
                }
                "resolved" if value == "1" => view.resolved = true,
                "resolved" => {
                    return Err(AppError::BadRequest("resolved must be 1".into()));
                }
                "offset" => {
                    view.offset = value.parse().map_err(|_| {
                        AppError::BadRequest("offset must be a non-negative integer".into())
                    })?;
                }
                _ => {}
            }
        }
        Ok(view)
    }

    /// The query string for this view at `offset`, without the `?`.
    fn query(&self, offset: usize) -> String {
        let mut qs = url::form_urlencoded::Serializer::new(String::new());
        for v in &self.verdicts {
            qs.append_pair("verdict", v);
        }
        if self.resolved {
            qs.append_pair("resolved", "1");
        }
        if offset > 0 {
            qs.append_pair("offset", &offset.to_string());
        }
        qs.finish()
    }

    fn href(&self, offset: usize) -> String {
        match self.query(offset) {
            qs if qs.is_empty() => QUEUE_PATH.to_string(),
            qs => format!("{QUEUE_PATH}?{qs}"),
        }
    }

    /// The current view at its own offset.
    fn here(&self) -> String {
        self.href(self.offset)
    }

    async fn fetch(&self, alaya: &crate::alaya::AlayaClient) -> Result<Value, AppError> {
        alaya
            .contradictions(QUEUE_PAGE, self.offset, self.resolved, &self.verdicts)
            .await
    }
}

/// A stored judge failure and a never-judged edge both read `unjudged` on
/// the wire. The failure marker carries a reason and `judged_at`; an edge
/// the judge never reached carries neither.
enum JudgeState {
    Judged(String),
    Failed,
    Never,
}

fn judge_state(p: &Value) -> JudgeState {
    let verdict = vs(p, "verdict");
    if !verdict.is_empty() && verdict != "unjudged" {
        return JudgeState::Judged(verdict);
    }
    let judged_at = p.get("judged_at").is_some_and(|x| !x.is_null());
    if judged_at || !vs(p, "verdict_reason").is_empty() {
        JudgeState::Failed
    } else {
        JudgeState::Never
    }
}

fn verdict_badge(verdict: &str) -> BadgeKind {
    match verdict {
        "contradiction" => BadgeKind::Destructive,
        "supersession" => BadgeKind::Warning,
        "coexist" => BadgeKind::Success,
        "unrelated" => BadgeKind::Muted,
        _ => BadgeKind::Secondary,
    }
}

fn is_kept_both(p: &Value) -> bool {
    vs(p, "resolution") == crate::alaya::KEEP_BOTH
}

/// `f64` field as fixed-point, or "—" when absent or null — a missing
/// confidence must never render as a real 0.00.
fn fixed2(p: &Value, key: &str) -> String {
    p.get(key)
        .and_then(Value::as_f64)
        .map(|x| format!("{x:.2}"))
        .unwrap_or_else(|| "—".into())
}

fn or_dash(s: String) -> String {
    if s.is_empty() { "—".into() } else { s }
}

/// The edge's judge and resolution fields, each under its own label.
/// Shared by the queue card and the pair page.
fn edge_facts(p: &Value) -> impl IntoView + use<> {
    let a = vs(p, "memory_a_hash");
    let b = vs(p, "memory_b_hash");
    let state = judge_state(p);
    let failed = matches!(state, JudgeState::Failed);
    let (badge_class, badge_text) = match state {
        JudgeState::Judged(v) => (badge(verdict_badge(&v)), v),
        JudgeState::Failed => (badge(BadgeKind::Destructive), "judge error".into()),
        JudgeState::Never => (badge(BadgeKind::Muted), "not yet judged".into()),
    };
    let reason = vs(p, "verdict_reason");
    let survivor = vs(p, "survivor");
    let survivor_side = match survivor.as_str() {
        "" => None,
        s if s == a => Some("A"),
        s if s == b => Some("B"),
        _ => Some("?"),
    };
    let judge_conf = fixed2(p, "verdict_confidence");
    let detector_conf = fixed2(p, "confidence");
    let model = or_dash(vs(p, "verdict_model"));
    let judged_at = fmt_epoch(vf(p, "judged_at"));
    let detected_at = fmt_epoch(vf(p, "created_at"));
    let resolution = vs(p, "resolution");
    let resolved = !resolution.is_empty() || p.get("resolved_at").is_some_and(|x| !x.is_null());
    let resolved_at = fmt_epoch(vf(p, "resolved_at"));
    let resolved_via = or_dash(vs(p, "resolved_via"));
    let fact = |label: &'static str, value: String| {
        view! {
            <div><dt class="text-muted-foreground text-xs">{label}</dt><dd>{value}</dd></div>
        }
    };
    view! {
        <div class="space-y-3">
            <div class="flex flex-wrap items-center gap-2">
                <span class=badge_class>{badge_text}</span>
                {resolved.then(|| view! {
                    <span class=badge(BadgeKind::Info)>{format!("resolved: {}", or_dash(resolution.clone()))}</span>
                })}
            </div>
            {(!reason.is_empty()).then(|| {
                let class = if failed { "text-sm text-destructive" } else { "text-sm" };
                view! { <p class=class>{reason}</p> }
            })}
            <dl class="grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
                {fact("Judge confidence", judge_conf)}
                {fact("Detector confidence", detector_conf)}
                <div>
                    <dt class="text-muted-foreground text-xs">"Recommended survivor"</dt>
                    <dd class="flex items-center gap-2">
                        {match survivor_side {
                            Some(side) => Either::Left(view! {
                                <span>{side}</span>
                                <HashLink hash=survivor />
                            }),
                            None => Either::Right("—"),
                        }}
                    </dd>
                </div>
                {fact("Judge model", model)}
                {fact("Judged at", judged_at)}
                {fact("Detected at", detected_at)}
                {resolved.then(|| view! {
                    {fact("Resolved at", resolved_at)}
                    {fact("Resolved via", resolved_via)}
                })}
            </dl>
        </div>
    }
}

/// Keep both (or Reopen, for a pair stamped `keep_both`), Keep A, Keep B.
/// Every action carries `back`, so the operator lands where they started.
fn pair_actions(p: &Value, csrf: &str, back: &str) -> impl IntoView + use<> {
    let a = vs(p, "memory_a_hash");
    let b = vs(p, "memory_b_hash");
    let survivor = vs(p, "survivor");
    let recommend_a = !survivor.is_empty() && survivor == a;
    let recommend_b = !survivor.is_empty() && survivor == b;
    let recommend_keep_both = matches!(
        judge_state(p),
        JudgeState::Judged(ref v) if v == "coexist" || v == "unrelated"
    );
    let supersede = |old: &str, new: &str| {
        format!("/alaya/supersede?old={old}&new={new}&back={}", urlenc(back))
    };
    let keep_a = supersede(&b, &a);
    let keep_b = supersede(&a, &b);
    let keep = |href: String, label: &'static str, recommended: bool| {
        let class = btn_sm(if recommended {
            Btn::Default
        } else {
            Btn::Outline
        });
        let text = if recommended {
            format!("{label} — recommended")
        } else {
            label.to_string()
        };
        view! { <a href=href class=class>{text}</a> }
    };
    // Non-destructive first (LAB-3885): one click, no reason — the verdict
    // already carries it. Supersede stays two-step with a reason.
    let (action, label, class) = if is_kept_both(p) {
        (
            "/alaya/contradictions/reopen",
            "Reopen",
            btn_sm(Btn::Outline),
        )
    } else {
        (
            "/alaya/contradictions/keep-both",
            if recommend_keep_both {
                "Keep both — recommended"
            } else {
                "Keep both"
            },
            btn_sm(if recommend_keep_both {
                Btn::Default
            } else {
                Btn::Outline
            }),
        )
    };
    let (csrf, back) = (csrf.to_string(), back.to_string());
    view! {
        <div class="flex flex-wrap items-center gap-3">
            <form method="post" action=action class="inline">
                <input type="hidden" name="csrf" value=csrf />
                <input type="hidden" name="memory_a_hash" value=a />
                <input type="hidden" name="memory_b_hash" value=b />
                <input type="hidden" name="back" value=back />
                <button type="submit" class=class>{label}</button>
            </form>
            {keep(keep_a, "Keep A (supersede B)…", recommend_a)}
            {keep(keep_b, "Keep B (supersede A)…", recommend_b)}
        </div>
    }
}

fn pair_href(a: &str, b: &str, view: &QueueView) -> String {
    let ctx = view.query(view.offset);
    let sep = if ctx.is_empty() { "" } else { "&" };
    format!("{QUEUE_PATH}/pair?a={a}&b={b}{sep}{ctx}")
}

fn queue_card(p: &Value, csrf: &str, view: &QueueView, back: &str) -> impl IntoView + use<> {
    let a = vs(p, "memory_a_hash");
    let b = vs(p, "memory_b_hash");
    let review = pair_href(&a, &b, view);
    let selectable = !is_kept_both(p);
    let pair_value = format!("{a}:{b}");
    let side = |hash: String, text: String, sup: bool, label: &'static str| {
        view! {
            <div class="rounded-md border p-4">
                <div class="flex items-center gap-2 mb-2">
                    <span class="text-xs font-medium text-muted-foreground">{label}</span>
                    <HashLink hash=hash />
                    {sup.then(|| view! { <span class=badge(BadgeKind::Warning)>"superseded"</span> })}
                </div>
                <p class="text-sm">{text}</p>
            </div>
        }
    };
    let a_side = side(
        a.clone(),
        vs(p, "memory_a_content"),
        p.get("memory_a_superseded")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "A",
    );
    let b_side = side(
        b.clone(),
        vs(p, "memory_b_content"),
        p.get("memory_b_superseded")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "B",
    );
    let facts = edge_facts(p);
    let actions = pair_actions(p, csrf, back);
    view! {
        <Card>
            <CardHeader>
                <div class="flex w-full items-center gap-3">
                    {selectable.then(|| view! {
                        // `form=` ties the box to the page's one bulk form
                        // without nesting forms — no JS needed.
                        <input type="checkbox" form="bulk" name="pair" value=pair_value aria-label="select for bulk keep both" />
                    })}
                    <CardTitle>"Contradiction"</CardTitle>
                    <a class="ml-auto text-sm text-primary underline-offset-4 hover:underline" href=review>
                        "Review in full →"
                    </a>
                </div>
            </CardHeader>
            <CardContent>
                <div class="mb-4">{facts}</div>
                <div class="grid gap-4 sm:grid-cols-2 mb-4">{a_side}{b_side}</div>
                {actions}
            </CardContent>
        </Card>
    }
}

pub async fn contradictions(
    State(state): State<AppState>,
    session: Session,
    RawQuery(raw): RawQuery,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let view = QueueView::parse(raw.as_deref())?;
    let (jar, flash) = take_flash(jar);
    let res = view.fetch(&state.alaya).await?;
    let pairs = res
        .get("pairs")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    // Next only on the server's word: a short page can still have more
    // behind it (superseded endpoints are dropped after the graph read).
    let next_href = res
        .get("next_offset")
        .and_then(Value::as_u64)
        .map(|n| view.href(n as usize));
    let prev_href = (view.offset > 0).then(|| view.href(view.offset.saturating_sub(QUEUE_PAGE)));

    let csrf = session.csrf.clone();
    let back = view.here();
    let any_selectable = pairs.iter().any(|p| !is_kept_both(p));
    let cards = pairs
        .iter()
        .map(|p| queue_card(p, &csrf, &view, &back))
        .collect_view();

    let position = if pairs.is_empty() {
        "No pairs match these filters at this offset.".to_string()
    } else {
        // Not "pairs N–M": the server drops pairs with a superseded endpoint
        // after it pages, so a page can hold fewer than it skipped past.
        format!(
            "{} pairs from offset {} (newest first).",
            pairs.len(),
            view.offset
        )
    };
    let verdict_boxes = ALL_VERDICTS
        .iter()
        .map(|v| {
            let checked = view.verdicts.iter().any(|x| x == v);
            view! {
                <label class=LABEL_CLASS>
                    <input type="checkbox" name="verdict" value=*v checked=checked />
                    {*v}
                </label>
            }
        })
        .collect_view();
    let resolved = view.resolved;
    let content = view! {
        <div class="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>"Contradictions"</CardTitle>
                    <CardDescription>
                        "Verdicts are advisory: the judge recommends a survivor, you decide. Keep both when both memories are true or the pair is detector noise — nothing is superseded and Reopen undoes it. Otherwise choose which memory survives — the loser is superseded with a reason, never dropped."
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <form method="get" action=QUEUE_PATH class="flex flex-wrap items-center gap-4">
                        {verdict_boxes}
                        <label class=LABEL_CLASS>
                            <input type="checkbox" name="resolved" value="1" checked=resolved />
                            "include resolved"
                        </label>
                        <button type="submit" class=btn_sm(Btn::Secondary)>"Apply"</button>
                    </form>
                    <p class="text-xs text-muted-foreground mt-3">"No verdict ticked shows the server's default filter."</p>
                    <p class="text-sm text-muted-foreground mt-3">{position}</p>
                </CardContent>
            </Card>
            {any_selectable.then(|| view! {
                <form id="bulk" method="post" action="/alaya/contradictions/keep-both/bulk" class="flex items-center gap-3">
                    <input type="hidden" name="csrf" value=csrf.clone() />
                    <input type="hidden" name="back" value=back.clone() />
                    <button type="submit" class=btn_sm(Btn::Secondary)>"Keep both for selected"</button>
                    <span class="text-xs text-muted-foreground">"Tick pairs below; one submit settles them all."</span>
                </form>
            })}
            {cards}
            <div class="flex gap-3">
                {prev_href.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"← Prev"</a> })}
                {next_href.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"Next →"</a> })}
            </div>
        </div>
    };

    Ok((
        jar,
        Html(page(
            "Contradictions — ops console",
            &session,
            flash,
            content,
        )),
    ))
}

#[derive(Deserialize)]
pub struct PairQuery {
    #[serde(default)]
    a: String,
    #[serde(default)]
    b: String,
}

/// One memory of a pair, in full. A failed fetch says so in its column —
/// never a blank that reads as an empty memory.
fn memory_column(label: &'static str, hash: String, res: Result<Value, AppError>) -> impl IntoView {
    let header = {
        let hash = hash.clone();
        view! {
            <div class="flex items-center gap-2 mb-2">
                <span class="text-xs font-medium text-muted-foreground">{label}</span>
                <HashLink hash=hash />
            </div>
        }
    };
    let body = match res.and_then(|r| {
        r.get("memory")
            .cloned()
            .ok_or_else(|| AppError::NotFound("memory not found".into()))
    }) {
        Err(e) => Either::Left(view! {
            <p class="text-sm text-destructive" role="alert">
                {format!("Could not load this memory: {}", e.detail())}
            </p>
        }),
        Ok(m) => {
            let content = vs(&m, "content");
            let summary = vs(&m, "summary");
            let created = fmt_epoch(vf(&m, "created_at"));
            let superseded_by = m
                .get("metadata")
                .and_then(|md| md.get("superseded_by"))
                .and_then(Value::as_str)
                .map(String::from);
            let tags: Vec<String> = m
                .get("tags")
                .and_then(|t| t.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            Either::Right(view! {
                <div class="space-y-3">
                    <div class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        <span>{format!("created {created}")}</span>
                        {superseded_by.map(|by| view! {
                            <span class=badge(BadgeKind::Warning)>"superseded"</span>
                            {validate_hash(&by).is_ok().then(|| view! { <HashLink hash=by.clone() /> })}
                        })}
                    </div>
                    <pre class="whitespace-pre-wrap text-sm font-sans">{content}</pre>
                    <div class="border-t pt-4">
                        <div class="text-xs font-medium text-muted-foreground mb-1">"Summary"</div>
                        <p class="text-sm">{or_dash(summary)}</p>
                    </div>
                    <div class="flex flex-wrap gap-1">
                        {tags.into_iter().map(|t| view! { <span class=badge(BadgeKind::Muted)>{t}</span> }).collect_view()}
                    </div>
                </div>
            })
        }
    };
    view! { <div class="rounded-md border p-4">{header}{body}</div> }
}

/// Both memories of one pair side by side, in full. The edge's own fields
/// come from the queue page the operator opened it from (its `verdict` /
/// `resolved` / `offset` ride along): the server has no single-pair read,
/// and re-reading that page is the only source that is upstream's word
/// rather than the link's.
pub async fn pair_page(
    State(state): State<AppState>,
    session: Session,
    Query(q): Query<PairQuery>,
    RawQuery(raw): RawQuery,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    validate_hash(&q.a)?;
    validate_hash(&q.b)?;
    let view = QueueView::parse(raw.as_deref())?;
    let (jar, flash) = take_flash(jar);

    let (mem_a, mem_b, queue) = tokio::join!(
        state.alaya.get_memory(&q.a),
        state.alaya.get_memory(&q.b),
        view.fetch(&state.alaya),
    );
    let edge = queue.map(|res| {
        res.get("pairs")
            .and_then(|p| p.as_array())
            .and_then(|pairs| {
                pairs
                    .iter()
                    .find(|p| vs(p, "memory_a_hash") == q.a && vs(p, "memory_b_hash") == q.b)
                    .cloned()
            })
    });

    let csrf = session.csrf.clone();
    let back = view.here();
    let back_link = back.clone();
    let offset = view.offset;
    // Without the edge the actions still work — they name the pair, not
    // the verdict — so Keep both stays offered and the server is the judge
    // of whether the edge exists.
    let fallback_edge = json!({ "memory_a_hash": q.a, "memory_b_hash": q.b });
    let (edge_view, action_edge) = match edge {
        Ok(Some(p)) => (Either::Left(edge_facts(&p)), p),
        Ok(None) => (
            Either::Right(format!(
                "This pair is not on the queue page it was opened from (offset {offset}, same filters): it was settled, or the queue moved, so its verdict and resolution are unknown here. The memories below are live; to reopen a keep-both, find the pair in the queue with resolved pairs included."
            )),
            fallback_edge,
        ),
        Err(e) => (
            Either::Right(format!("Could not load the verdict: {}", e.detail())),
            fallback_edge,
        ),
    };
    let actions = pair_actions(&action_edge, &csrf, &back);
    let col_a = memory_column("A", q.a.clone(), mem_a);
    let col_b = memory_column("B", q.b.clone(), mem_b);
    let content = view! {
        <div class="space-y-6">
            <a class="text-sm text-primary underline-offset-4 hover:underline" href=back_link>"← Back to the queue"</a>
            <Card>
                <CardHeader><CardTitle>"Contradiction pair"</CardTitle></CardHeader>
                <CardContent>
                    <div class="mb-4 text-sm">{edge_view}</div>
                    {actions}
                </CardContent>
            </Card>
            <div class="grid gap-4 sm:grid-cols-2">{col_a}{col_b}</div>
        </div>
    };

    Ok((
        jar,
        Html(page(
            "Contradiction pair — ops console",
            &session,
            flash,
            content,
        )),
    ))
}

#[derive(Deserialize)]
pub struct ResolutionForm {
    #[serde(default)]
    csrf: String,
    memory_a_hash: String,
    memory_b_hash: String,
    #[serde(default)]
    back: String,
}

/// "Keep both" (LAB-3885): stamp the pair resolved without superseding
/// either memory. POST-redirect-GET back to the queue view it came from,
/// which no longer lists the pair.
pub async fn keep_both_submit(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<ResolutionForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&form.memory_a_hash)?;
    validate_hash(&form.memory_b_hash)?;
    state
        .alaya
        .set_resolution(
            &form.memory_a_hash,
            &form.memory_b_hash,
            Some(crate::alaya::KEEP_BOTH),
        )
        .await?;
    tracing::info!(sub = ?session.sub, a = %form.memory_a_hash, b = %form.memory_b_hash, "contradiction kept both");
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        "ok",
        format!(
            "Kept both {} and {} — pair resolved, nothing superseded.",
            short_hash(&form.memory_a_hash),
            short_hash(&form.memory_b_hash)
        ),
        &crate::routes::return_to(&form.back, QUEUE_PATH),
    ))
}

/// Undo a keep-both. The server counts a stamp in EITHER direction as
/// settling the pair, and a re-store can leave one on each edge, so both
/// directions are cleared. Either edge may be missing (the reverse one
/// usually is; the forward one after a stale pair page) and the other is
/// still cleared. The request fails only when neither direction was
/// cleared; a failure on the reverse after the forward cleared is
/// reported, because the pair would stay hidden.
pub async fn reopen_submit(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum::Form(form): axum::Form<ResolutionForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&form.memory_a_hash)?;
    validate_hash(&form.memory_b_hash)?;
    let (a, b) = (&form.memory_a_hash, &form.memory_b_hash);
    let forward_cleared = match state.alaya.set_resolution(a, b, None).await {
        Ok(_) => true,
        Err(AppError::NotFound(_)) => false,
        Err(e) => return Err(e),
    };
    let reverse = match (
        state.alaya.set_resolution(b, a, None).await,
        forward_cleared,
    ) {
        (Ok(_), _) | (Err(AppError::NotFound(_)), true) => None,
        (Err(e), true) => Some(e),
        (Err(e), false) => return Err(e),
    };
    tracing::info!(sub = ?session.sub, a = %a, b = %b, forward_cleared, reverse_failed = reverse.is_some(), "contradiction reopened");
    let (kind, msg) = match reverse {
        None => (
            "ok",
            format!(
                "Reopened {} and {} — the pair is back in the queue.",
                short_hash(a),
                short_hash(b)
            ),
        ),
        Some(e) => (
            "error",
            format!(
                "Cleared the stamp on {} → {}, but clearing the reverse edge failed ({}) — \
                 if that edge is stamped the pair stays out of the queue; retry Reopen.",
                short_hash(a),
                short_hash(b),
                clip(e.detail(), MAX_CAUSE_CHARS)
            ),
        ),
    };
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        kind,
        msg,
        &crate::routes::return_to(&form.back, QUEUE_PATH),
    ))
}

#[derive(Deserialize)]
pub struct BulkKeepBothForm {
    #[serde(default)]
    csrf: String,
    /// `<memory_a_hash>:<memory_b_hash>`, one per ticked card.
    #[serde(default)]
    pair: Vec<String>,
    #[serde(default)]
    back: String,
}

/// Distinct failure causes the bulk report spells out; pairs failing for
/// any other cause are still named, under "other errors". Bounded because
/// the report rides in the flash cookie, which a browser drops silently
/// past 4 KiB — and a dropped report is exactly the silent partial failure
/// this exists to prevent.
const MAX_REPORTED_CAUSES: usize = 3;
/// Calls are sequential (alaya-server's command channel fast-fails when
/// full), so a wedged upstream would hold the request for one client
/// timeout per pair. The whole batch, the call in flight included, gets
/// this long; what is left over is reported as not attempted.
const BULK_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);
const MAX_CAUSE_CHARS: usize = 100;

/// The bulk flash: how many settled, and every failed pair by name,
/// grouped under its cause.
fn bulk_report(total: usize, failed: &[(String, String)]) -> Flash {
    if failed.is_empty() {
        return Flash {
            kind: "ok".into(),
            msg: format!("Kept both for {total} pairs — nothing superseded."),
        };
    }
    let mut groups: Vec<(&str, Vec<&str>)> = Vec::new();
    for (pair, cause) in failed {
        match groups.iter_mut().find(|(c, _)| c == cause) {
            Some((_, pairs)) => pairs.push(pair),
            None => groups.push((cause, vec![pair])),
        }
    }
    let mut parts: Vec<String> = groups
        .iter()
        .take(MAX_REPORTED_CAUSES)
        .map(|(cause, pairs)| format!("{} ({})", pairs.join(", "), clip(cause, MAX_CAUSE_CHARS)))
        .collect();
    let others: Vec<&str> = groups
        .iter()
        .skip(MAX_REPORTED_CAUSES)
        .flat_map(|(_, pairs)| pairs.iter().copied())
        .collect();
    if !others.is_empty() {
        parts.push(format!("{} (other errors)", others.join(", ")));
    }
    Flash {
        kind: "error".into(),
        msg: format!(
            "Kept both for {} of {total} pairs. {} not confirmed — reload the queue to check them: {}.",
            total - failed.len(),
            failed.len(),
            parts.join("; ")
        ),
    }
}

/// Bulk keep-both: one resolution call per ticked pair. The single-pair
/// verb in a loop, not a bulk endpoint — the page size caps it, and every
/// stamp is reversible with Reopen. Every pair is validated before the
/// first call, so a malformed selection writes nothing.
pub async fn keep_both_bulk(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
    axum_extra::extract::Form(form): axum_extra::extract::Form<BulkKeepBothForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    if form.pair.is_empty() {
        return Err(AppError::BadRequest("select at least one pair".into()));
    }
    if form.pair.len() > QUEUE_PAGE {
        return Err(AppError::BadRequest(format!(
            "at most {QUEUE_PAGE} pairs per submit"
        )));
    }
    let pairs = form
        .pair
        .iter()
        .map(|p| {
            let (a, b) = p
                .split_once(':')
                .ok_or_else(|| AppError::BadRequest("malformed pair".into()))?;
            Ok((validate_hash(a)?, validate_hash(b)?))
        })
        .collect::<Result<Vec<_>, AppError>>()?;

    let started = std::time::Instant::now();
    let mut failed: Vec<(String, String)> = Vec::new();
    for (a, b) in &pairs {
        let label = format!("{}↔{}", short_hash(a), short_hash(b));
        let left = BULK_BUDGET.saturating_sub(started.elapsed());
        if left.is_zero() {
            failed.push((label, "not attempted: the batch ran out of time".into()));
            continue;
        }
        let call = state
            .alaya
            .set_resolution(a, b, Some(crate::alaya::KEEP_BOTH));
        // A call cut off after it was sent may still land upstream, so its
        // cause says the outcome is unknown, not that it failed.
        let cause = match tokio::time::timeout(left, call).await {
            Ok(Ok(_)) => continue,
            Ok(Err(e)) => e.detail().to_string(),
            Err(_) => "no answer in time: outcome unknown".to_string(),
        };
        tracing::warn!(sub = ?session.sub, a = %a, b = %b, cause = ?clip(&cause, MAX_CAUSE_CHARS), "bulk keep both: pair not confirmed");
        failed.push((label, cause));
    }
    tracing::info!(sub = ?session.sub, total = pairs.len(), failed = failed.len(), "contradictions kept both in bulk");
    let report = bulk_report(pairs.len(), &failed);
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        &report.kind,
        report.msg,
        &crate::routes::return_to(&form.back, QUEUE_PATH),
    ))
}

// ─── Auth-state view (AC7, read-only) ───────────────────────────────────────

pub async fn auth_view(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let cfg = state.alaya.auth_config().await?;

    let oidc_enabled = cfg
        .get("oidc")
        .and_then(|o| o.get("enabled"))
        .and_then(|e| e.as_bool())
        .unwrap_or(false);
    let issuer = cfg.get("oidc").map(|o| vs(o, "issuer")).unwrap_or_default();
    let audience = cfg
        .get("oidc")
        .map(|o| vs(o, "audience"))
        .unwrap_or_default();
    let static_configured = cfg
        .get("static_bearer_configured")
        .and_then(|b| b.as_bool())
        .unwrap_or(false);
    let ops = cfg
        .get("ops")
        .and_then(|o| o.as_array())
        .cloned()
        .unwrap_or_default();

    let op_rows = ops
        .iter()
        .map(|o| {
            let name = vs(o, "op");
            let oidc_ok = o.get("oidc").and_then(|x| x.as_bool()).unwrap_or(false);
            let mutating = o.get("mutating").and_then(|x| x.as_bool()).unwrap_or(true);
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{name}</span></TableCell>
                    <TableCell>
                        {if mutating {
                            Either::Left(view! { <span class=badge(BadgeKind::Warning)>"mutating"</span> })
                        } else {
                            Either::Right(view! { <span class=badge(BadgeKind::Muted)>"read / additive"</span> })
                        }}
                    </TableCell>
                    <TableCell>
                        {if oidc_ok {
                            Either::Left(view! { <span class=badge(BadgeKind::Success)>"allowed"</span> })
                        } else {
                            Either::Right(view! { <span class=badge(BadgeKind::Destructive)>"denied"</span> })
                        }}
                    </TableCell>
                </TableRow>
            }
        })
        .collect_view();

    let content = view! {
        <div class="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>"Ālaya auth state (read-only)"</CardTitle>
                    <CardDescription>
                        "Live from alaya-server. OIDC principals are read / additive by design (store only, no mutation); changing this is a product decision (LAB-1084), not a console feature."
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <dl class="grid grid-cols-1 sm:grid-cols-3 gap-4 text-sm">
                        <div>
                            <dt class="text-muted-foreground text-xs">"Static bearer"</dt>
                            <dd>{if static_configured { "configured" } else { "NOT configured" }}</dd>
                        </div>
                        <div>
                            <dt class="text-muted-foreground text-xs">"OIDC"</dt>
                            <dd>{if oidc_enabled { "enabled" } else { "disabled" }}</dd>
                        </div>
                        <div>
                            <dt class="text-muted-foreground text-xs">"Issuer / audience"</dt>
                            <dd class="break-all">{issuer}" / "{audience}</dd>
                        </div>
                    </dl>
                </CardContent>
            </Card>
            <Card>
                <CardHeader>
                    <CardTitle>"OIDC principal × operation matrix"</CardTitle>
                    <CardDescription>"The static bearer has full access to every operation. OIDC principals are default-deny: any op not explicitly allowlisted is static-bearer only."</CardDescription>
                </CardHeader>
                <CardContent>
                    <TableWrapper><Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Operation"</TableHead>
                                <TableHead>"Kind"</TableHead>
                                <TableHead>"OIDC principal"</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>{op_rows}</TableBody>
                    </Table></TableWrapper>
                </CardContent>
            </Card>
        </div>
    };

    Ok((
        jar,
        Html(page("Auth state — ops console", &session, flash, content)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// AC-1: the filters each mode offers, pinned to what alaya-core's
    /// per-mode search reads. A change upstream must change this table.
    #[test]
    fn each_mode_applies_exactly_the_filters_its_server_path_reads() {
        for (mode, query, memory_type, tags) in [
            ("hybrid", true, true, false),
            ("scan", false, true, false),
            ("recent", false, true, false),
            ("tag", false, false, true),
        ] {
            let applies = Mode::parse(mode).ok().expect("offered").applies();
            let want = Applies {
                query,
                memory_type,
                tags,
            };
            assert_eq!(applies, want, "{mode}");
        }
        assert!(Mode::parse("similar").is_err(), "not offered");
    }

    fn pair(extra: Value) -> Value {
        let mut p = json!({
            "memory_a_hash": "a".repeat(64),
            "memory_b_hash": "b".repeat(64),
            "confidence": 0.61,
            "created_at": 1788265604.0,
            "memory_a_content": "a summary",
            "memory_b_content": "b summary",
            "verdict": "unjudged",
            "verdict_reason": null,
            "survivor": null,
            "verdict_confidence": null,
            "verdict_model": null,
            "judged_at": null,
            "resolution": null,
            "resolved_at": null,
            "resolved_via": null,
        });
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        p
    }

    fn card(p: &Value) -> String {
        queue_card(p, "tok", &QueueView::default(), QUEUE_PATH).to_html()
    }

    #[test]
    fn queue_view_round_trips_its_url_contract() {
        let raw = "verdict=coexist&verdict=unrelated&resolved=1&offset=100&a=ignored";
        let Ok(view) = QueueView::parse(Some(raw)) else {
            panic!("valid view refused")
        };
        assert_eq!(view.verdicts, ["coexist", "unrelated"]);
        assert!(view.resolved);
        assert_eq!(
            view.href(view.offset),
            "/alaya/contradictions?verdict=coexist&verdict=unrelated&resolved=1&offset=100"
        );
        assert_eq!(QueueView::default().href(0), "/alaya/contradictions");
    }

    #[test]
    fn queue_view_refuses_bad_values() {
        for raw in ["verdict=bogus", "resolved=true", "offset=-1", "offset=x"] {
            assert!(QueueView::parse(Some(raw)).is_err(), "{raw}");
        }
    }

    #[test]
    fn a_judged_card_labels_both_confidences_and_the_survivor() {
        let html = card(&pair(json!({
            "verdict": "supersession",
            "verdict_reason": "B is newer",
            "survivor": "b".repeat(64),
            "verdict_confidence": 0.87,
            "verdict_model": "judge-model-x",
            "judged_at": 1788265700.0,
        })));
        assert!(html.contains(">supersession<"));
        assert!(html.contains("Judge confidence</dt><dd>0.87"), "{html}");
        assert!(html.contains("Detector confidence</dt><dd>0.61"));
        assert!(html.contains("judge-model-x"));
        assert!(html.contains("2026-09-01 12:28"), "judged_at");
        assert!(html.contains("2026-09-01 12:26"), "created_at");
        assert!(html.contains("B is newer"));
        assert!(
            html.contains("Keep B (supersede A)… — recommended"),
            "{html}"
        );
        assert!(html.contains("form=\"bulk\""), "selectable for bulk");
        assert!(!html.contains("judge error") && !html.contains("Resolved via"));
    }

    #[test]
    fn a_stored_judge_failure_renders_as_an_error_with_its_reason() {
        let html = card(&pair(json!({
            "verdict_reason": "unjudged: schema violation",
            "verdict_confidence": 0.0,
            "verdict_model": "judge-model-x",
            "judged_at": 1788265700.0,
        })));
        assert!(html.contains("judge error"));
        assert!(
            html.contains("text-destructive\">unjudged: schema violation"),
            "{html}"
        );
        assert!(!html.contains("not yet judged"));
    }

    #[test]
    fn a_never_judged_card_says_so_and_invents_nothing() {
        let html = card(&pair(json!({})));
        assert!(html.contains("not yet judged"));
        assert!(!html.contains("judge error"));
        assert!(html.contains("Judge confidence</dt><dd>—"), "{html}");
        assert!(html.contains("Judged at</dt><dd>—"));
    }

    #[test]
    fn a_resolved_card_shows_its_stamp_and_offers_reopen_not_bulk() {
        let html = card(&pair(json!({
            "verdict": "coexist",
            "resolution": "keep_both",
            "resolved_at": 1788265800.0,
            "resolved_via": "operator:console",
        })));
        assert!(html.contains("resolved: keep_both"));
        assert!(
            html.contains("Resolved at</dt><dd>2026-09-01 12:30"),
            "{html}"
        );
        assert!(html.contains("Resolved via</dt><dd>operator:console"));
        assert!(html.contains("action=\"/alaya/contradictions/reopen\""));
        assert!(!html.contains("form=\"bulk\""));
        assert!(!html.contains("/alaya/contradictions/keep-both\""));
    }

    #[test]
    fn bulk_report_names_every_failed_pair_under_its_cause() {
        assert_eq!(bulk_report(3, &[]).kind, "ok");
        let failed = [
            ("p1".to_string(), "down".to_string()),
            ("p2".to_string(), "gone".to_string()),
            ("p3".to_string(), "down".to_string()),
        ];
        let f = bulk_report(5, &failed);
        assert_eq!(f.kind, "error");
        assert_eq!(
            f.msg,
            "Kept both for 2 of 5 pairs. 3 not confirmed — reload the queue to check them: p1, p3 (down); p2 (gone)."
        );
    }

    /// Fifty failures, each with its own long cause, still fit the flash
    /// cookie and still name every pair.
    #[test]
    fn bulk_report_stays_bounded_and_names_every_pair() {
        let failed: Vec<(String, String)> = (0..QUEUE_PAGE)
            .map(|i| {
                (
                    format!("{i:012}↔{i:012}"),
                    format!("cause {i} {}", "x".repeat(300)),
                )
            })
            .collect();
        let f = bulk_report(QUEUE_PAGE, &failed);
        assert!(f.msg.len() < 2048, "{} bytes", f.msg.len());
        for (pair, _) in &failed {
            assert!(f.msg.contains(pair.as_str()));
        }
        assert!(f.msg.contains("(other errors)"));
    }
}
