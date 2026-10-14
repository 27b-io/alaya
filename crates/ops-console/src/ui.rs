//! Vendored Rust/UI components + page shell.
//!
//! Components are copy-paste vendored from the Rust/UI Leptos registry
//! (then github.com/rust-ui/ui, now github.com/rust-ui/leptos-ui;
//! `app_crates/registry/src/ui/*.rs`) per the
//! registry's shadcn-style model: the class strings are kept verbatim; the
//! `variants!`/`clx!` macros are expanded to plain functions so the console
//! doesn't take `leptos_ui`/`tw_merge` dependencies for an SSR-only app.
//!
//! Rendering is SSR-only: `view!` output is serialized with `.to_html()`.
//! Leptos escapes all text nodes and attribute values — stored memory
//! content can never inject markup (AC9).

use leptos::prelude::*;

use crate::session::{Flash, Session};
use crate::theme::Theme;

// ─── Buttons (vendored: ui/button.rs) ───────────────────────────────────────

const BTN_BASE: &str = "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium transition-all disabled:pointer-events-none disabled:opacity-50 shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] w-fit hover:cursor-pointer active:scale-[0.98] active:opacity-100 select-none";

pub enum Btn {
    Default,
    Destructive,
    Outline,
    Secondary,
}

/// Button classes: base + variant + default size (`h-9 px-4 py-2`).
pub fn btn(v: Btn) -> String {
    let variant = match v {
        Btn::Default => "bg-primary text-primary-foreground shadow-xs hover:bg-primary/90",
        Btn::Destructive => {
            "bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60"
        }
        Btn::Outline => {
            "border bg-background shadow-xs hover:bg-accent hover:text-accent-foreground"
        }
        Btn::Secondary => "bg-secondary text-secondary-foreground shadow-xs hover:bg-secondary/80",
    };
    format!("{BTN_BASE} {variant} h-9 px-4 py-2")
}

/// Small-size button classes.
pub fn btn_sm(v: Btn) -> String {
    btn(v).replace("h-9 px-4 py-2", "h-8 rounded-md gap-1.5 px-3")
}

// ─── Badge (vendored: ui/badge.rs) ──────────────────────────────────────────

const BADGE_BASE: &str = "inline-flex items-center font-semibold rounded-md border transition-colors w-fit px-2.5 py-0.5 text-xs";

pub enum BadgeKind {
    Secondary,
    Muted,
    Destructive,
    Success,
    Warning,
    Info,
}

pub fn badge(kind: BadgeKind) -> String {
    let variant = match kind {
        BadgeKind::Secondary => "border-transparent bg-secondary text-secondary-foreground",
        BadgeKind::Muted => "border-transparent bg-muted text-muted-foreground",
        BadgeKind::Destructive => {
            "border-transparent shadow bg-destructive text-destructive-foreground dark:bg-destructive/60"
        }
        BadgeKind::Success => "border-transparent bg-success-light text-success-dark",
        BadgeKind::Warning => "border-transparent bg-warning-light text-warning-dark",
        BadgeKind::Info => "border-transparent bg-info-light text-info-dark",
    };
    format!("{BADGE_BASE} {variant}")
}

// ─── Card (vendored: ui/card.rs) ────────────────────────────────────────────

#[component]
pub fn Card(children: Children) -> impl IntoView {
    view! {
        <div class="bg-card text-card-foreground flex flex-col rounded-xl border shadow-sm py-6 gap-4">
            {children()}
        </div>
    }
}

#[component]
pub fn CardHeader(children: Children) -> impl IntoView {
    view! { <div class="flex flex-col items-start gap-1.5 px-6">{children()}</div> }
}

#[component]
pub fn CardTitle(children: Children) -> impl IntoView {
    view! { <h2 class="leading-none font-semibold">{children()}</h2> }
}

#[component]
pub fn CardDescription(children: Children) -> impl IntoView {
    view! { <p class="text-muted-foreground text-sm">{children()}</p> }
}

#[component]
pub fn CardContent(children: Children) -> impl IntoView {
    view! { <div class="px-6">{children()}</div> }
}

// ─── Table (vendored: ui/table.rs) ──────────────────────────────────────────

#[component]
pub fn TableWrapper(children: Children) -> impl IntoView {
    view! { <div class="overflow-auto rounded-md border">{children()}</div> }
}

#[component]
pub fn Table(children: Children) -> impl IntoView {
    view! { <table class="w-full text-sm caption-bottom">{children()}</table> }
}

#[component]
pub fn TableHeader(children: Children) -> impl IntoView {
    view! { <thead class="[&_tr]:border-b sticky top-0 z-10 bg-card">{children()}</thead> }
}

#[component]
pub fn TableBody(children: Children) -> impl IntoView {
    view! { <tbody class="[&_tr:last-child]:border-0">{children()}</tbody> }
}

#[component]
pub fn TableRow(children: Children) -> impl IntoView {
    view! { <tr class="border-b transition-colors hover:bg-muted/50">{children()}</tr> }
}

#[component]
pub fn TableHead(children: Children) -> impl IntoView {
    view! {
        <th class="h-10 px-2 text-left align-middle font-medium text-muted-foreground">
            {children()}
        </th>
    }
}

#[component]
pub fn TableCell(children: Children) -> impl IntoView {
    view! { <td class="p-3 align-middle">{children()}</td> }
}

/// Short, linked content-hash — the one way a hash renders anywhere in the
/// console (browse rows, detail, chains, duplicates, contradictions).
#[component]
pub fn HashLink(hash: String) -> impl IntoView {
    let href = format!("/alaya/memory/{hash}");
    let short: String = hash.chars().take(12).collect();
    view! {
        <a class="font-mono text-xs text-primary underline-offset-4 hover:underline" href=href>
            {short}
        </a>
    }
}

// ─── Form atoms (vendored: ui/input.rs, ui/label.rs, ui/textarea.rs,
//     ui/select_native.rs — class strings only) ─────────────────────────────

pub const INPUT_CLASS: &str = "border-input flex h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] md:text-sm";

pub const LABEL_CLASS: &str =
    "flex items-center gap-2 text-sm leading-none font-medium select-none";

pub const TEXTAREA_CLASS: &str = "border-input placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 flex min-h-16 w-full rounded-md border bg-transparent px-3 py-2 text-base shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] md:text-sm";

pub const SELECT_CLASS: &str = "border-input flex h-9 w-fit items-center justify-between gap-2 rounded-md border bg-transparent px-3 py-2 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]";

// ─── Page shell ─────────────────────────────────────────────────────────────

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Full HTML document around already-rendered body HTML, in `theme`. Pages
/// rendered without a session (errors) pass `Theme::System`.
pub fn document(title: &str, theme: Theme, body_html: String) -> String {
    let class = theme
        .html_class()
        .map(|c| format!(" class=\"{c}\""))
        .unwrap_or_default();
    format!(
        "<!DOCTYPE html><html lang=\"en\"{class}><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta name=\"color-scheme\" content=\"{}\">\
         <meta name=\"referrer\" content=\"same-origin\">\
         <title>{}</title>\
         <link rel=\"stylesheet\" href=\"/static/console.css\"></head>\
         <body class=\"bg-background text-foreground font-sans antialiased\">{}</body></html>",
        theme.color_scheme(),
        escape_html(title),
        body_html
    )
}

// ─── Theme switch ───────────────────────────────────────────────────────────

/// Lucide icons (ISC), inline: the CSP loads nothing but the stylesheet.
fn theme_icon(theme: Theme) -> AnyView {
    match theme {
        Theme::System => view! {
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <rect width="20" height="14" x="2" y="3" rx="2" />
                <line x1="8" x2="16" y1="21" y2="21" />
                <line x1="12" x2="12" y1="17" y2="21" />
            </svg>
        }
        .into_any(),
        Theme::Light => view! {
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2" />
                <path d="M12 20v2" />
                <path d="m4.93 4.93 1.41 1.41" />
                <path d="m17.66 17.66 1.41 1.41" />
                <path d="M2 12h2" />
                <path d="M20 12h2" />
                <path d="m6.34 17.66-1.41 1.41" />
                <path d="m19.07 4.93-1.41 1.41" />
            </svg>
        }
        .into_any(),
        Theme::Dark => view! {
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
            </svg>
        }
        .into_any(),
    }
}

/// One icon button showing the current theme; it opens a menu of the three
/// choices, each a submit button of one POST form carrying the session's CSRF
/// token, like logout. A `<details>` disclosure, so it needs no JavaScript,
/// and one icon wide, so the header row keeps its width.
fn theme_switch(current: Theme, csrf: String) -> impl IntoView {
    let options = Theme::ALL
        .into_iter()
        .map(|t| {
            let (label, name) = match t {
                Theme::System => ("System theme", "System"),
                Theme::Light => ("Light theme", "Light"),
                Theme::Dark => ("Dark theme", "Dark"),
            };
            let class = if t == current {
                "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm bg-accent text-accent-foreground"
            } else {
                "flex w-full items-center gap-2 rounded-sm px-2 py-1.5 text-sm text-foreground hover:bg-accent hover:cursor-pointer"
            };
            view! {
                <button
                    type="submit"
                    name="theme"
                    value=t.as_str()
                    aria-label=label
                    aria-pressed=(t == current).to_string()
                    class=class
                >
                    {theme_icon(t)}
                    {name}
                </button>
            }
        })
        .collect_view();
    view! {
        <details class="relative">
            <summary
                title="Theme"
                aria-label="Theme"
                class="flex h-8 w-8 list-none items-center justify-center rounded-md hover:bg-accent hover:text-foreground hover:cursor-pointer [&::-webkit-details-marker]:hidden"
            >
                {theme_icon(current)}
            </summary>
            <form
                method="post"
                action="/prefs/theme"
                class="absolute right-0 z-20 mt-2 w-36 rounded-md border bg-card p-1 shadow-md"
            >
                <input type="hidden" name="csrf" value=csrf />
                {options}
            </form>
        </details>
    }
}

fn flash_banner(flash: &Flash) -> impl IntoView + use<> {
    let class = match flash.kind.as_str() {
        "ok" => {
            "mb-6 rounded-md border border-transparent bg-success-light text-success-dark px-4 py-3 text-sm"
        }
        _ => {
            "mb-6 rounded-md border border-transparent bg-destructive-light text-destructive-dark px-4 py-3 text-sm"
        }
    };
    let msg = flash.msg.clone();
    view! { <div class=class role="status">{msg}</div> }
}

/// Authenticated page shell for a page reached by GET: top nav (module bar —
/// Ālaya plus the anthropic-lb and selecta read-only panes), the theme switch,
/// flash banner, then the page content.
pub fn page(
    title: &str,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
) -> String {
    shell(title, session, flash, content, true)
}

/// The same shell for a page rendered as the answer to a POST (a preview, a
/// result, a shown-once key). It carries no theme switch: the switch returns to
/// the page it was used on, and this page's URL is a POST-only route that a
/// GET would answer with 405, leaving the page and anything shown once on it.
pub fn result_page(
    title: &str,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
) -> String {
    shell(title, session, flash, content, false)
}

fn shell(
    title: &str,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
    with_theme_switch: bool,
) -> String {
    let who = session.display_name().to_string();
    let csrf = session.csrf.clone();
    let switch = with_theme_switch.then(|| theme_switch(session.theme, session.csrf.clone()));
    let body = view! {
        <div class="min-h-screen">
            <header class="border-b bg-card">
                <div class="mx-auto max-w-6xl px-6 h-14 flex items-center gap-6">
                    <a href="/" class="font-semibold text-sm">"27b ops console"</a>
                    <nav class="flex items-center gap-4 text-sm text-muted-foreground">
                        <a class="hover:text-foreground" href="/alaya">"Memories"</a>
                        <a class="hover:text-foreground" href="/alaya/duplicates">"Duplicates"</a>
                        <a class="hover:text-foreground" href="/alaya/contradictions">"Contradictions"</a>
                        <a class="hover:text-foreground" href="/alaya/health">"Judge health"</a>
                        <a class="hover:text-foreground" href="/alaya/auth">"Auth state"</a>
                        <a class="hover:text-foreground" href="/lb">"anthropic-lb"</a>
                        <a class="hover:text-foreground" href="/lb/clients">"LB clients"</a>
                        <a class="hover:text-foreground" href="/selecta">"selecta"</a>
                    </nav>
                    <div class="ml-auto flex items-center gap-3 text-sm text-muted-foreground">
                        {switch}
                        <span>{who}</span>
                        <form method="post" action="/auth/logout">
                            <input type="hidden" name="csrf" value=csrf />
                            <button type="submit" class="hover:text-foreground underline underline-offset-4">
                                "Log out"
                            </button>
                        </form>
                    </div>
                </div>
            </header>
            <main class="mx-auto max-w-6xl px-6 py-8">
                {flash.as_ref().map(flash_banner)}
                {content}
            </main>
        </div>
    }
    .to_html();
    document(title, session.theme, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_escapes_title() {
        let html = document("<script>alert(1)</script>", Theme::System, String::new());
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    /// AC9: stored memory content rendered through view! must come out
    /// escaped — markup in a memory can never execute in the console.
    #[test]
    fn leptos_escapes_stored_content() {
        let hostile = "<img src=x onerror=alert(1)><script>steal()</script>";
        let html = view! { <div>{hostile.to_string()}</div> }.to_html();
        assert!(!html.contains("<img"));
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;img"));
    }

    /// AC9 (attributes): hostile content placed in an attribute value is
    /// escaped too — it can't break out of the attribute.
    #[test]
    fn leptos_escapes_attribute_values() {
        let hostile = "\"><script>steal()</script>".to_string();
        let html = view! { <input type="hidden" value=hostile /> }.to_html();
        assert!(!html.contains("<script>"));
    }
}
