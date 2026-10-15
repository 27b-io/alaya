//! The atoms, vendored from the Rust/UI Leptos registry (then
//! github.com/rust-ui/ui, now github.com/rust-ui/leptos-ui;
//! `app_crates/registry/src/ui/*.rs`) per the registry's shadcn-style model:
//! the class strings are kept verbatim; the `variants!`/`clx!` macros are
//! expanded to plain functions so the console doesn't take
//! `leptos_ui`/`tw_merge` dependencies for an SSR-only app.
//!
//! Three deliberate departures, each marked where it is made: the
//! destructive button's ink is a theme token instead of a palette white, the
//! table wrapper has a height cap so its sticky header can stick, and a
//! table column can be marked secondary, to hide on a phone.

use leptos::prelude::*;

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
        // Upstream inks this with the palette's white. `destructive-foreground`
        // is the same near-white in both themes, and a token keeps every
        // colour in the console on the theme (`ui::tests` pins it).
        Btn::Destructive => {
            "bg-destructive text-destructive-foreground shadow-xs hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60"
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

/// The scroll box around a table. A table wider than the screen scrolls
/// sideways in here, never the page. Upstream has no height cap, and without
/// one this box never scrolls down, so the header's `sticky` had nothing to
/// stick to; capped, a long table scrolls inside it under its header.
#[component]
pub fn TableWrapper(children: Children) -> impl IntoView {
    view! { <div class="overflow-auto max-h-[70vh] rounded-md border">{children()}</div> }
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

/// A column's priority, a departure from upstream: a `secondary` column
/// shows from `md` up and is hidden on a phone, so a wide table keeps the
/// columns that say what a row is and how it stands. Mark its head and every
/// one of its cells, or the columns slip.
const SECONDARY: &str = "hidden md:table-cell";

fn column(base: &str, secondary: bool) -> String {
    if secondary {
        format!("{base} {SECONDARY}")
    } else {
        base.to_string()
    }
}

#[component]
pub fn TableHead(#[prop(optional)] secondary: bool, children: Children) -> impl IntoView {
    let class = column(
        "h-10 px-2 text-left align-middle font-medium text-muted-foreground",
        secondary,
    );
    view! { <th class=class>{children()}</th> }
}

#[component]
pub fn TableCell(#[prop(optional)] secondary: bool, children: Children) -> impl IntoView {
    view! { <td class=column("p-3 align-middle", secondary)>{children()}</td> }
}

// ─── Form atoms (vendored: ui/input.rs, ui/label.rs, ui/textarea.rs,
//     ui/select_native.rs — class strings only) ─────────────────────────────

pub const INPUT_CLASS: &str = "border-input flex h-9 w-full min-w-0 rounded-md border bg-transparent px-3 py-1 text-base shadow-xs transition-[color,box-shadow] outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] md:text-sm";

pub const LABEL_CLASS: &str =
    "flex items-center gap-2 text-sm leading-none font-medium select-none";

pub const TEXTAREA_CLASS: &str = "border-input placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-ring/50 flex min-h-16 w-full rounded-md border bg-transparent px-3 py-2 text-base shadow-xs transition-[color,box-shadow] outline-none focus-visible:ring-[3px] md:text-sm";

pub const SELECT_CLASS: &str = "border-input flex h-9 w-fit items-center justify-between gap-2 rounded-md border bg-transparent px-3 py-2 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]";
