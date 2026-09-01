//! Ālaya module: memory browse/search, detail, and the full curation set —
//! supersede, delete, merge-duplicates, relations, contradiction resolution
//! (AC2–AC7). Every mutation is a POST form carrying the session CSRF token;
//! results follow POST-redirect-GET with a flash banner.

use axum::extract::{Path, Query, State};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum_extra::extract::cookie::PrivateCookieJar;
use leptos::either::Either;
use leptos::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::AppError;
use crate::routes::{fmt_epoch, short_hash, validate_hash};
use crate::session::{Flash, Session, flash_cookie, take_flash};
use crate::state::AppState;
use crate::ui::*;

// ─── Value helpers (defensive rendering over upstream JSON) ────────────────

fn vs(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

fn vf(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0)
}

fn excerpt(v: &Value, max: usize) -> String {
    let text = v
        .get("summary")
        .and_then(|x| x.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| v.get("content").and_then(|x| x.as_str()))
        .unwrap_or("");
    let mut out: String = text.chars().take(max).collect();
    if text.chars().count() > max {
        out.push('…');
    }
    out
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

pub async fn browse(
    State(state): State<AppState>,
    session: Session,
    Query(q): Query<BrowseQuery>,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);

    let mode = q.mode.clone().unwrap_or_else(|| {
        if q.q.trim().is_empty() {
            "scan".into()
        } else {
            "hybrid".into()
        }
    });
    let include_superseded = q.include_superseded.is_some();

    let mut params = json!({
        "mode": mode,
        "query": q.q,
        "page": q.page,
        "page_size": PAGE_SIZE,
        "k": PAGE_SIZE,
        "include_superseded": include_superseded,
        "output": "both",
    });
    if !q.memory_type.is_empty() {
        params["memory_type"] = json!(q.memory_type);
    }
    if !q.tags.trim().is_empty() {
        params["tags"] = json!(q.tags);
    }
    if let Some(c) = q.cursor {
        params["cursor"] = json!(c);
    }

    let res = state.alaya.search(params).await?;
    let results = res
        .get("results")
        .and_then(|r| r.as_array())
        .cloned()
        .unwrap_or_default();
    let total = res.get("total").and_then(|t| t.as_u64());
    let has_more = res
        .get("has_more")
        .and_then(|h| h.as_bool())
        .unwrap_or(false);

    // Pagination targets (scan/tag are page-based; recent is cursor-based).
    let base_qs = |page: usize, cursor: Option<f64>| {
        let mut qs = format!(
            "/alaya?q={}&mode={}&memory_type={}&tags={}&page={page}",
            urlenc(&q.q),
            urlenc(&mode),
            urlenc(&q.memory_type),
            urlenc(&q.tags),
        );
        if include_superseded {
            qs.push_str("&include_superseded=on");
        }
        if let Some(c) = cursor {
            qs.push_str(&format!("&cursor={c}"));
        }
        qs
    };
    let prev_href = (mode != "recent" && q.page > 1).then(|| base_qs(q.page - 1, None));
    let next_href = if mode == "recent" {
        results
            .last()
            .map(|last| base_qs(1, Some(vf(last, "created_at"))))
            .filter(|_| results.len() >= PAGE_SIZE)
    } else if has_more || results.len() >= PAGE_SIZE {
        (mode == "scan" || mode == "tag").then(|| base_qs(q.page + 1, None))
    } else {
        None
    };

    let rows = results
        .iter()
        .map(|m| {
            let hash = vs(m, "content_hash");
            let href = memory_href(&hash);
            let short = short_hash(&hash);
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
                    <TableCell>
                        <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>
                            {short}
                        </a>
                    </TableCell>
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
        })
        .collect_view();

    let count_line = match total {
        Some(t) => format!("{t} memories · page {}", q.page),
        None => format!("{} results", results.len()),
    };

    let content = view! {
        <div class="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>"Memories"</CardTitle>
                    <CardDescription>"Search or browse the corpus. Filters apply in every mode."</CardDescription>
                </CardHeader>
                <CardContent>
                    <form method="get" action="/alaya" class="flex flex-wrap items-end gap-3">
                        <div class="flex flex-col gap-1.5 grow min-w-56">
                            <label class=LABEL_CLASS for="q">"Query"</label>
                            <input class=INPUT_CLASS id="q" name="q" value=q.q.clone() placeholder="semantic query, or empty to browse" />
                        </div>
                        <div class="flex flex-col gap-1.5">
                            <label class=LABEL_CLASS for="mode">"Mode"</label>
                            <select class=SELECT_CLASS id="mode" name="mode">
                                {["hybrid", "scan", "recent", "tag"].into_iter().map(|m| {
                                    let selected = m == mode;
                                    view! { <option value=m selected=selected>{m}</option> }
                                }).collect_view()}
                            </select>
                        </div>
                        <div class="flex flex-col gap-1.5">
                            <label class=LABEL_CLASS for="memory_type">"Type"</label>
                            <select class=SELECT_CLASS id="memory_type" name="memory_type">
                                {["", "note", "decision", "task", "reference"].into_iter().map(|t| {
                                    let selected = t == q.memory_type;
                                    let label = if t.is_empty() { "any" } else { t };
                                    view! { <option value=t selected=selected>{label}</option> }
                                }).collect_view()}
                            </select>
                        </div>
                        <div class="flex flex-col gap-1.5">
                            <label class=LABEL_CLASS for="tags">"Tags (csv)"</label>
                            <input class=INPUT_CLASS id="tags" name="tags" value=q.tags.clone() />
                        </div>
                        <label class=format!("{LABEL_CLASS} h-9")>
                            <input type="checkbox" name="include_superseded" checked=include_superseded />
                            "include superseded"
                        </label>
                        <button type="submit" class=btn(Btn::Default)>"Search"</button>
                    </form>
                </CardContent>
            </Card>

            <div class="text-sm text-muted-foreground">{count_line}</div>
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
                {prev_href.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"← Prev"</a> })}
                {next_href.map(|h| view! { <a class=btn_sm(Btn::Outline) href=h>"Next →"</a> })}
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

    // Relations are graph-backed and non-fatal upstream; treat a failure as
    // an empty list with a note rather than a dead page.
    let relations = state
        .alaya
        .relation("get", &hash, None, None)
        .await
        .ok()
        .and_then(|r| r.get("relations").and_then(|x| x.as_array()).cloned())
        .unwrap_or_default();

    // Supersession chain (forward walk, bounded — cycles can't loop us).
    let mut chain: Vec<(String, String)> = Vec::new();
    let mut cursor = mem
        .get("metadata")
        .and_then(|m| m.get("superseded_by"))
        .and_then(|s| s.as_str())
        .map(String::from);
    let mut hops = 0;
    while let Some(next) = cursor.take() {
        if hops >= 10 || validate_hash(&next).is_err() {
            break;
        }
        hops += 1;
        let label = match state.alaya.get_memory(&next).await {
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

    let relation_rows = relations
        .iter()
        .map(|e| {
            let csrf = csrf.clone();
            let back = memory_href(&hash);
            let source = vs(e, "source");
            let target = vs(e, "target");
            let rel_type = vs(e, "relation_type");
            let rel_badge = rel_type.clone();
            // Link to the far end of the edge, whichever side this memory is.
            let other = if source == hash { target.clone() } else { source.clone() };
            let other_href = memory_href(&other);
            let other_short = short_hash(&other);
            let created = e
                .get("created_at")
                .and_then(|c| c.as_f64())
                .map(fmt_epoch)
                .unwrap_or_else(|| "—".into());
            view! {
                <TableRow>
                    <TableCell><span class=badge(BadgeKind::Info)>{rel_badge}</span></TableCell>
                    <TableCell>
                        <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=other_href>
                            {other_short}
                        </a>
                    </TableCell>
                    <TableCell><span class="text-xs text-muted-foreground">{created}</span></TableCell>
                    <TableCell>
                        <form method="post" action="/alaya/relation/delete">
                            <input type="hidden" name="csrf" value=csrf />
                            <input type="hidden" name="content_hash" value=source />
                            <input type="hidden" name="target_hash" value=target />
                            <input type="hidden" name="relation_type" value=rel_type />
                            <input type="hidden" name="back" value=back />
                            <button type="submit" class=btn_sm(Btn::Outline)>"Delete"</button>
                        </form>
                    </TableCell>
                </TableRow>
            }
        })
        .collect_view();

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
                                let href = memory_href(h);
                                let short = short_hash(h);
                                let label = label.clone();
                                view! {
                                    <li class="flex gap-2 items-baseline">
                                        <span class="text-muted-foreground">"↳"</span>
                                        <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>{short}</a>
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
                    {if relations.is_empty() {
                        Either::Left(view! { <p class="text-sm text-muted-foreground mb-4">"No relations."</p> })
                    } else {
                        Either::Right(view! {
                            <div class="mb-4">
                                <TableWrapper><Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>"Type"</TableHead>
                                            <TableHead>"Linked memory"</TableHead>
                                            <TableHead>"Created"</TableHead>
                                            <TableHead>""</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>{relation_rows}</TableBody>
                                </Table></TableWrapper>
                            </div>
                        })
                    }}
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
    tracing::info!(sub = %session.sub, hash = %hash, "memory deleted");
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
            let href = memory_href(&h);
            let short = short_hash(&h);
            view! {
                <div class="rounded-md border p-4">
                    <div class="text-xs font-medium text-muted-foreground mb-1">{title}</div>
                    <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>{short}</a>
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
    tracing::info!(sub = %session.sub, old = %form.old_hash, new = %form.new_hash, "memory superseded");
    Ok(flash_redirect(
        jar,
        state.secure_cookies(),
        "ok",
        format!(
            "Superseded {} → {}.",
            short_hash(&form.old_hash),
            short_hash(&form.new_hash)
        ),
        &memory_href(&form.old_hash),
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
    let new_hash = store_res
        .get("content_hash")
        .and_then(|h| h.as_str())
        .ok_or_else(|| AppError::Upstream("store returned no content_hash".into()))?
        .to_string();
    if new_hash == hash {
        return Err(AppError::BadRequest(
            "corrected content is identical to the original".into(),
        ));
    }

    state
        .alaya
        .supersede(&hash, &new_hash, form.reason.trim())
        .await?;
    tracing::info!(sub = %session.sub, old = %hash, new = %new_hash, "corrected + superseded");
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
    tracing::info!(sub = %session.sub, action = %action, source = %form.content_hash, target = %form.target_hash, rel = %form.relation_type, "relation changed");
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
                    let href = memory_href(h);
                    let short = short_hash(h);
                    let h_radio = h.clone();
                    let h_check = h.clone();
                    view! {
                        <li class="flex items-center gap-3 text-sm">
                            <label class="flex items-center gap-1 text-xs text-muted-foreground">
                                <input type="radio" name="canonical_hash" value=h_radio checked=is_canonical />
                                "canonical"
                            </label>
                            <label class="flex items-center gap-1 text-xs text-muted-foreground">
                                <input type="checkbox" name="duplicate_hashes" value=h_check checked=!is_canonical />
                                "merge"
                            </label>
                            <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>{short}</a>
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
    axum_extra::extract::Form(form): axum_extra::extract::Form<MergeForm>,
) -> Result<Response, AppError> {
    session.verify_csrf(&form.csrf)?;
    validate_hash(&form.canonical_hash)?;
    if form.duplicate_hashes.is_empty() {
        return Err(AppError::BadRequest(
            "select at least one duplicate to merge".into(),
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
        let canonical_href = memory_href(&form.canonical_hash);
        let canonical_short = short_hash(&form.canonical_hash);
        let dup_field = form.duplicate_hashes.clone();
        let content = view! {
            <Card>
                <CardHeader>
                    <CardTitle>"Merge preview (dry run)"</CardTitle>
                    <CardDescription>"No changes were made. Review, then commit."</CardDescription>
                </CardHeader>
                <CardContent>
                    <p class="text-sm mb-2">
                        "Canonical: "
                        <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=canonical_href>{canonical_short}</a>
                    </p>
                    <p class="text-sm mb-2">"Will supersede:"</p>
                    <ul class="space-y-1 mb-6">
                        {superseded.iter().map(|h| {
                            let href = memory_href(h);
                            let short = short_hash(h);
                            view! { <li><a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>{short}</a></li> }
                        }).collect_view()}
                    </ul>
                    <form method="post" action="/alaya/duplicates/merge" class="flex items-center gap-3">
                        <input type="hidden" name="csrf" value=csrf />
                        <input type="hidden" name="canonical_hash" value=form.canonical_hash.clone() />
                        {dup_field.into_iter().map(|h| view! {
                            <input type="hidden" name="duplicate_hashes" value=h />
                        }).collect_view()}
                        <input type="hidden" name="reason" value=form.reason.clone() />
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
    tracing::info!(sub = %session.sub, canonical = %form.canonical_hash, merged, errors, "duplicates merged");
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

// ─── Contradictions (AC6) ───────────────────────────────────────────────────

pub async fn contradictions(
    State(state): State<AppState>,
    session: Session,
    jar: PrivateCookieJar,
) -> Result<(PrivateCookieJar, Html<String>), AppError> {
    let (jar, flash) = take_flash(jar);
    let res = state.alaya.contradictions(50).await?;
    let pairs = res
        .get("pairs")
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();

    let cards = pairs
        .iter()
        .map(|p| {
            let a = vs(p, "memory_a_hash");
            let b = vs(p, "memory_b_hash");
            let a_text = vs(p, "memory_a_content");
            let b_text = vs(p, "memory_b_content");
            let a_sup = p.get("memory_a_superseded").and_then(|x| x.as_bool()).unwrap_or(false);
            let b_sup = p.get("memory_b_superseded").and_then(|x| x.as_bool()).unwrap_or(false);
            let confidence = format!("{:.2}", vf(p, "confidence"));
            let keep_a = format!("/alaya/supersede?old={b}&new={a}");
            let keep_b = format!("/alaya/supersede?old={a}&new={b}");
            let side = |hash: String, text: String, sup: bool, label: &'static str| {
                let href = memory_href(&hash);
                let short = short_hash(&hash);
                view! {
                    <div class="rounded-md border p-4">
                        <div class="flex items-center gap-2 mb-2">
                            <span class="text-xs font-medium text-muted-foreground">{label}</span>
                            <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>{short}</a>
                            {sup.then(|| view! { <span class=badge(BadgeKind::Warning)>"superseded"</span> })}
                        </div>
                        <p class="text-sm">{text}</p>
                    </div>
                }
            };
            view! {
                <Card>
                    <CardHeader>
                        <CardTitle>{format!("Contradiction — confidence {confidence}")}</CardTitle>
                    </CardHeader>
                    <CardContent>
                        <div class="grid gap-4 sm:grid-cols-2 mb-4">
                            {side(a.clone(), a_text, a_sup, "A")}
                            {side(b.clone(), b_text, b_sup, "B")}
                        </div>
                        <div class="flex gap-3">
                            <a href=keep_a class=btn_sm(Btn::Outline)>"Keep A (supersede B)…"</a>
                            <a href=keep_b class=btn_sm(Btn::Outline)>"Keep B (supersede A)…"</a>
                        </div>
                    </CardContent>
                </Card>
            }
        })
        .collect_view();

    let intro = if pairs.is_empty() {
        "No unresolved contradictions."
    } else {
        "Resolve by choosing which memory survives — the loser is superseded with a reason, never dropped."
    };
    let content = view! {
        <div class="space-y-6">
            <Card>
                <CardHeader>
                    <CardTitle>"Contradictions"</CardTitle>
                    <CardDescription>{intro}</CardDescription>
                </CardHeader>
            </Card>
            {cards}
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
                    <TableCell><span class=badge(BadgeKind::Success)>"allowed"</span></TableCell>
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
                        "Live from alaya-server. OIDC principals are read-only by design; changing this is a product decision (LAB-1084), not a console feature."
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
                    <CardTitle>"Principal × operation matrix"</CardTitle>
                    <CardDescription>"Default-deny: any op not explicitly allowlisted is static-bearer only."</CardDescription>
                </CardHeader>
                <CardContent>
                    <TableWrapper><Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>"Operation"</TableHead>
                                <TableHead>"Kind"</TableHead>
                                <TableHead>"Static principal"</TableHead>
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
