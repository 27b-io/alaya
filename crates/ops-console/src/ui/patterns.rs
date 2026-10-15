//! The console's own components, composed from the vendored atoms in
//! `components.rs`: each element that recurs across pages, written once so it
//! looks and reads the same on every page.

use leptos::prelude::*;

use super::components::{BadgeKind, badge};

// ─── Status: the status-to-badge mapping ────────────────────────────────────

/// What a state means to the operator. A page picks the meaning and this
/// picks the look, once, so "failing" on one page and "unreachable" on
/// another wear the same badge. A badge that names a category rather than a
/// state picks its own kind: for example a judge verdict (`verdict_badge` in
/// `routes/alaya.rs`), a memory's type or its tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Working as it should: passing, healthy, reachable.
    Ok,
    /// Worth a look, though nothing is down: degraded, close to a limit.
    Warn,
    /// Down or failing: unavailable, unreachable, failing. A notice with
    /// this status is an alert.
    Error,
    /// No signal either way: not configured, disabled, not run yet.
    Off,
}

impl Status {
    /// The badge classes for this status.
    pub fn badge(self) -> String {
        badge(match self {
            Status::Ok => BadgeKind::Success,
            Status::Warn => BadgeKind::Warning,
            Status::Error => BadgeKind::Destructive,
            Status::Off => BadgeKind::Muted,
        })
    }
}

/// A state's short name, in its status's badge.
#[component]
pub fn StatusBadge(status: Status, #[prop(into)] label: String) -> impl IntoView {
    view! { <span class=status.badge()>{label}</span> }
}

// ─── Notices ────────────────────────────────────────────────────────────────

/// A status badge, then a sentence about it: how a page says a section is
/// unavailable, a check is failing or a module is off. An `Error` notice is
/// an alert, so assistive technology announces it.
#[component]
pub fn Notice(status: Status, #[prop(into)] label: String, children: Children) -> impl IntoView {
    let role = (status == Status::Error).then_some("alert");
    view! {
        <p class="text-sm wrap-break-word" role=role>
            <StatusBadge status=status label=label />
            " "
            {children()}
        </p>
    }
}

/// One dark section, named, in place of that section's content: the rest of
/// the page stays up, and a failure never reads as an empty (all-quiet) list.
pub fn unavailable(what: &str, detail: &str) -> impl IntoView + use<> {
    let msg = format!("{what}: {detail}");
    view! { <Notice status=Status::Error label="unavailable">{msg}</Notice> }
}

/// Where a list or a section has nothing in it: said in words, in a dashed
/// frame, so an empty section never looks broken or still loading.
#[component]
pub fn EmptyState(children: Children) -> impl IntoView {
    view! {
        <div class="flex flex-col items-center gap-2 rounded-lg border border-dashed px-4 py-8 text-center text-sm text-muted-foreground wrap-break-word">
            {children()}
        </div>
    }
}

/// A module this deployment runs without, and what turns it on: never a 404
/// or an empty page.
#[component]
pub fn NotConfigured(children: Children) -> impl IntoView {
    view! {
        <EmptyState>
            <StatusBadge status=Status::Off label="not configured" />
            <p>{children()}</p>
        </EmptyState>
    }
}

// ─── Key values ─────────────────────────────────────────────────────────────

/// Key values as tiles: two across on a phone, four from `md`.
#[component]
pub fn StatGrid(children: Children) -> impl IntoView {
    view! { <dl class="grid grid-cols-2 gap-3 md:grid-cols-4 md:gap-4">{children()}</dl> }
}

/// One key value under its label. The value is the children, so it can be a
/// number, a word or a link. A `text` value is words rather than a figure (a
/// reason, a URL, a list): set at body size, so it wraps without crowding
/// the grid.
#[component]
pub fn Stat(
    #[prop(into)] label: String,
    #[prop(optional)] text: bool,
    children: Children,
) -> impl IntoView {
    let value = if text {
        "mt-1 text-sm wrap-break-word"
    } else {
        "mt-1 text-xl font-semibold tabular-nums wrap-break-word"
    };
    view! {
        <div class="min-w-0 rounded-lg border bg-card p-4">
            <dt class="text-xs text-muted-foreground wrap-break-word">{label}</dt>
            <dd class=value>{children()}</dd>
        </div>
    }
}

// ─── Page header ────────────────────────────────────────────────────────────

/// A page's title, a line on what the page is for, and the page's own
/// actions (the children), which wrap under the title on a narrow screen.
#[component]
pub fn PageHeader(
    #[prop(into)] title: String,
    #[prop(optional, into)] description: Option<String>,
    #[prop(optional)] children: Option<Children>,
) -> impl IntoView {
    view! {
        <div class="flex flex-wrap items-end justify-between gap-x-6 gap-y-3">
            <div class="min-w-0 max-w-3xl space-y-1">
                <h1 class="text-2xl font-semibold tracking-tight wrap-break-word">{title}</h1>
                {description.map(|d| view! { <p class="text-sm text-muted-foreground wrap-break-word">{d}</p> })}
            </div>
            {children.map(|c| view! { <div class="flex flex-wrap items-center gap-2">{c()}</div> })}
        </div>
    }
}

// ─── Links ──────────────────────────────────────────────────────────────────

/// The console's one link style. Underlined, because the neutral theme gives
/// a link no colour of its own: the underline is what marks it, in both
/// themes, and it darkens on hover.
const LINK: &str =
    "underline decoration-muted-foreground/60 underline-offset-4 hover:decoration-foreground";

/// An `external` link leaves the console, so it sends no Referer: a console
/// URL is nobody else's business.
#[component]
pub fn Link(
    #[prop(into)] href: String,
    #[prop(optional)] external: bool,
    children: Children,
) -> impl IntoView {
    let rel = external.then_some("noreferrer");
    view! { <a class=LINK href=href rel=rel>{children()}</a> }
}

/// Short, linked content-hash — the one way a hash renders anywhere in the
/// console (browse rows, detail, chains, duplicates, contradictions).
#[component]
pub fn HashLink(hash: String) -> impl IntoView {
    let href = format!("/alaya/memory/{hash}");
    let short: String = hash.chars().take(12).collect();
    let class = format!("font-mono text-xs {LINK}");
    view! { <a class=class href=href>{short}</a> }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn html(v: impl IntoView) -> String {
        v.into_view().to_html()
    }

    /// Only an error is an alert: a passing check or an unconfigured module
    /// is not news for a screen reader to interrupt with.
    #[test]
    fn a_notice_is_an_alert_only_for_an_error() {
        let failing =
            html(view! { <Notice status=Status::Error label="failing">"step embed"</Notice> });
        assert!(failing.contains(r#"role="alert""#), "{failing}");
        assert!(failing.contains(">failing</span>"), "{failing}");
        assert!(failing.contains("step embed</p>"), "{failing}");
        for status in [Status::Ok, Status::Warn, Status::Off] {
            let quiet = html(view! { <Notice status=status label="x">"y"</Notice> });
            assert!(!quiet.contains("role="), "{status:?}: {quiet}");
        }
    }

    #[test]
    fn unavailable_names_the_section_and_the_cause() {
        let out = html(unavailable("graph stats", "Graph operation failed"));
        assert!(out.contains(">unavailable</span>"), "{out}");
        assert!(out.contains("graph stats: Graph operation failed"), "{out}");
        assert!(out.contains(r#"role="alert""#), "{out}");
    }

    #[test]
    fn not_configured_says_so_and_what_turns_it_on() {
        let out = html(view! { <NotConfigured>"Set LB_URL to enable."</NotConfigured> });
        assert!(out.contains(">not configured</span>"), "{out}");
        assert!(out.contains("<p>Set LB_URL to enable.</p>"), "{out}");
        assert!(out.contains("border-dashed"), "{out}");
    }

    #[test]
    fn an_empty_state_says_so_in_a_dashed_frame() {
        let out = html(view! { <EmptyState>"Nothing is waiting on you."</EmptyState> });
        assert!(out.contains("border-dashed"), "{out}");
        assert!(out.contains(">Nothing is waiting on you.</div>"), "{out}");
    }

    /// A stat is a `dt`/`dd` pair inside the grid's `dl`, so the label still
    /// names the value without the styling.
    #[test]
    fn a_stat_is_a_described_value_in_a_definition_list() {
        let out = html(view! {
            <StatGrid>
                <Stat label="Admitted today">"41"</Stat>
                <Stat label="Memories">"5234"</Stat>
            </StatGrid>
        });
        assert!(out.starts_with("<dl"), "{out}");
        assert!(out.contains(">Admitted today</dt><dd"), "{out}");
        assert!(out.contains(">41</dd></div>"), "{out}");
        assert_eq!(out.matches("<dt").count(), 2, "{out}");
        assert_eq!(out.matches("<dd").count(), 2, "{out}");
        let text = html(view! { <Stat label="Reason" text=true>"stale"</Stat> });
        assert!(
            text.contains(r#"<dd class="mt-1 text-sm wrap-break-word">stale</dd>"#),
            "{text}"
        );
    }

    #[test]
    fn a_page_header_is_the_pages_one_heading() {
        let out = html(view! {
            <PageHeader title="Judge health" description="What the judge did.">
                <a href="/x">"Act"</a>
            </PageHeader>
        });
        assert!(out.contains(">Judge health</h1>"), "{out}");
        assert!(out.contains(">What the judge did.</p>"), "{out}");
        assert!(out.contains(r#"<a href="/x">Act</a>"#), "{out}");
        let bare = html(view! { <PageHeader title="Home" /> });
        assert_eq!(bare.matches("<p").count(), 0, "{bare}");
        assert_eq!(
            bare.matches("<div").count(),
            2,
            "no empty actions box: {bare}"
        );
    }

    /// Every link looks the same, the hash link included, and a hash link
    /// shows the short prefix of the hash it targets.
    #[test]
    fn links_share_one_style() {
        let link = html(view! { <Link href="/alaya">"Memories"</Link> });
        assert_eq!(
            link,
            format!(r#"<a href="/alaya" class="{LINK}">Memories</a>"#)
        );
        let hash = "ab".repeat(32);
        let out = html(view! { <HashLink hash=hash.clone() /> });
        assert!(out.contains(LINK), "{out}");
        assert!(
            out.contains(&format!(r#"href="/alaya/memory/{hash}""#)),
            "{out}"
        );
        assert!(out.contains(">abababababab</a>"), "{out}");
    }

    /// A link off the console sends no Referer; one within it is untouched.
    #[test]
    fn an_external_link_sends_no_referer() {
        let out =
            html(view! { <Link href="https://github.com/x/y/pull/1" external=true>"PR #1"</Link> });
        assert!(out.contains(r#"rel="noreferrer""#), "{out}");
        let own = html(view! { <Link href="/alaya">"Memories"</Link> });
        assert!(!own.contains("rel="), "{own}");
    }
}
