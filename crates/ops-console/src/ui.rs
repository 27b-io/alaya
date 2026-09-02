//! Vendored Rust/UI components + page shell.
//!
//! Components are copy-paste vendored from the Rust/UI registry
//! (github.com/rust-ui/ui, `app_crates/registry/src/ui/*.rs`) per the
//! registry's shadcn-style model: the class strings are kept verbatim; the
//! `variants!`/`clx!` macros are expanded to plain functions so the console
//! doesn't take `leptos_ui`/`tw_merge` dependencies for an SSR-only app.
//!
//! Rendering is SSR-only: `view!` output is serialized with `.to_html()`.
//! Leptos escapes all text nodes and attribute values — stored memory
//! content can never inject markup (AC9).

use leptos::prelude::*;

use crate::session::{Flash, Session};

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
            "bg-destructive text-white shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20"
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
            "border-transparent shadow bg-destructive text-destructive-foreground"
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

/// Full HTML document around already-rendered body HTML.
pub fn document(title: &str, body_html: String) -> String {
    format!(
        "<!DOCTYPE html><html lang=\"en\"><head><meta charset=\"utf-8\">\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
         <meta name=\"referrer\" content=\"same-origin\">\
         <title>{}</title>\
         <link rel=\"stylesheet\" href=\"/static/console.css\"></head>\
         <body class=\"bg-background text-foreground font-sans antialiased\">{}</body></html>",
        escape_html(title),
        body_html
    )
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

/// Authenticated page shell: top nav (two-tenant module bar — Ālaya now, LB
/// staged behind LAB-1964), flash banner, then the page content.
pub fn page(
    title: &str,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
) -> String {
    let who = session.display_name().to_string();
    let csrf = session.csrf.clone();
    let body = view! {
        <div class="min-h-screen">
            <header class="border-b bg-card">
                <div class="mx-auto max-w-6xl px-6 h-14 flex items-center gap-6">
                    <a href="/" class="font-semibold text-sm">"27b ops console"</a>
                    <nav class="flex items-center gap-4 text-sm text-muted-foreground">
                        <a class="hover:text-foreground" href="/alaya">"Memories"</a>
                        <a class="hover:text-foreground" href="/alaya/duplicates">"Duplicates"</a>
                        <a class="hover:text-foreground" href="/alaya/contradictions">"Contradictions"</a>
                        <a class="hover:text-foreground" href="/alaya/auth">"Auth state"</a>
                    </nav>
                    <div class="ml-auto flex items-center gap-3 text-sm text-muted-foreground">
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
    document(title, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_escapes_title() {
        let html = document("<script>alert(1)</script>", String::new());
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
