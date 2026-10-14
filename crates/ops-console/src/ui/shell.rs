//! The document and the shell around every page: navigation, theme switch,
//! session controls and the flash banner. Responsive by construction: from
//! `lg` the navigation is a sidebar; below it, a menu in the top bar.

use leptos::prelude::*;

use crate::session::{Flash, Session};
use crate::theme::Theme;

// ─── Navigation ─────────────────────────────────────────────────────────────

/// Where a page sits in the navigation. Each handler names its entry; the
/// shell never guesses from the request path. A page that is not itself an
/// entry names the one it belongs under: a memory's detail page is under
/// Memories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Home,
    Memories,
    Duplicates,
    Contradictions,
    Health,
    Auth,
    Lb,
    LbClients,
    Selecta,
}

/// One module's entries under its heading; `None` heads the console's own.
type NavGroup = (
    Option<&'static str>,
    &'static [(Nav, &'static str, &'static str)],
);

/// The navigation, by module, in display order: the one place a page joins
/// it. Tests check that each `Nav` appears once and that each entry is a page
/// that renders and marks itself (`main.rs`).
const NAV: &[NavGroup] = &[
    (None, &[(Nav::Home, "/", "Home")]),
    (
        Some("Ālaya"),
        &[
            (Nav::Memories, "/alaya", "Memories"),
            (Nav::Duplicates, "/alaya/duplicates", "Duplicates"),
            (
                Nav::Contradictions,
                "/alaya/contradictions",
                "Contradictions",
            ),
            (Nav::Health, "/alaya/health", "Judge health"),
            (Nav::Auth, "/alaya/auth", "Auth state"),
        ],
    ),
    (
        Some("anthropic-lb"),
        &[
            (Nav::Lb, "/lb", "Overview"),
            (Nav::LbClients, "/lb/clients", "Clients"),
        ],
    ),
    (Some("selecta"), &[(Nav::Selecta, "/selecta", "Overview")]),
];

/// The current entry is styled by its `aria-current`, so what assistive
/// technology is told and what is seen cannot disagree.
const NAV_LINK: &str = "flex items-center rounded-md px-2 py-1.5 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground aria-[current=page]:bg-accent aria-[current=page]:font-medium aria-[current=page]:text-accent-foreground";

/// The entries, grouped, with `current` marked. Rendered twice per page, in
/// the sidebar and in the narrow-screen menu; one of the two is always
/// `display: none`, which takes it out of the accessibility tree too.
fn nav_links(current: Nav) -> impl IntoView {
    NAV.iter()
        .map(|(module, entries)| {
            let links = entries
                .iter()
                .map(|(nav, href, label)| {
                    let mark = (*nav == current).then_some("page");
                    view! {
                        <li>
                            <a class=NAV_LINK href=*href aria-current=mark>
                                {*label}
                            </a>
                        </li>
                    }
                })
                .collect_view();
            view! {
                {module.map(|m| {
                    view! { <p class="px-2 pt-4 pb-1 text-xs font-medium text-muted-foreground">{m}</p> }
                })}
                <ul class="space-y-0.5">{links}</ul>
            }
        })
        .collect_view()
}

// ─── Icons ──────────────────────────────────────────────────────────────────
// Lucide icons (ISC), inline: the CSP loads nothing but the stylesheet.

/// The menu button's two faces: bars while the menu is closed, a cross while
/// it is open, switched by the `<details>` element's own `open` state.
fn menu_icons() -> impl IntoView {
    view! {
        <svg class="group-open:hidden" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M4 12h16" />
            <path d="M4 18h16" />
            <path d="M4 6h16" />
        </svg>
        <svg class="hidden group-open:block" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
        </svg>
    }
}

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

// ─── Theme switch ───────────────────────────────────────────────────────────

/// One icon button showing the current theme; it opens a menu of the three
/// choices, each a submit button of one POST form carrying the session's CSRF
/// token, like logout. A `<details>` disclosure, so it needs no JavaScript,
/// and one icon wide, so it fits the top bar at every width.
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

// ─── Document and page ──────────────────────────────────────────────────────

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

/// The authenticated page for a page reached by GET: navigation with `nav`
/// marked current, the top bar (theme switch, operator, log out), the flash
/// banner, then the content.
pub fn page(
    title: &str,
    nav: Nav,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
) -> String {
    shell(title, nav, session, flash, content, true)
}

/// The same shell for a page rendered as the answer to a POST (a preview, a
/// result, a shown-once key). It carries no theme switch: the switch returns to
/// the page it was used on, and this page's URL is a POST-only route that a
/// GET would answer with 405, leaving the page and anything shown once on it.
pub fn result_page(
    title: &str,
    nav: Nav,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
) -> String {
    shell(title, nav, session, flash, content, false)
}

/// The top bar holds one copy of each control at every width, so a form is
/// never on the page twice; only the navigation is drawn twice, once per
/// layout. The content column is `min-w-0`, so a wide table scrolls inside
/// its wrapper instead of widening the page.
fn shell(
    title: &str,
    nav: Nav,
    session: &Session,
    flash: Option<Flash>,
    content: impl IntoView,
    with_theme_switch: bool,
) -> String {
    let who = session.display_name().to_string();
    let csrf = session.csrf.clone();
    let switch = with_theme_switch.then(|| theme_switch(session.theme, session.csrf.clone()));
    let body = view! {
        <a href="#content" class="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-3 focus:z-50 focus:rounded-md focus:bg-card focus:px-3 focus:py-2 focus:text-sm focus:shadow-md">
            "Skip to content"
        </a>
        <div class="min-h-screen lg:flex">
            <aside class="hidden lg:block lg:w-60 lg:shrink-0 lg:border-r lg:bg-card">
                <div class="sticky top-0 max-h-screen overflow-y-auto px-3 pb-6">
                    <a href="/" class="flex h-14 items-center px-2 text-sm font-semibold">
                        "27b ops console"
                    </a>
                    <nav aria-label="Main">{nav_links(nav)}</nav>
                </div>
            </aside>
            <div class="min-w-0 flex-1">
                <header class="sticky top-0 z-30 border-b bg-card">
                    <div class="flex h-14 items-center gap-3 px-4 sm:px-6 lg:px-8">
                        <details class="group lg:hidden">
                            <summary
                                title="Menu"
                                aria-label="Menu"
                                class="-ml-2 flex h-9 w-9 list-none items-center justify-center rounded-md hover:bg-accent hover:cursor-pointer [&::-webkit-details-marker]:hidden"
                            >
                                {menu_icons()}
                            </summary>
                            <nav
                                aria-label="Main"
                                class="absolute inset-x-0 top-full max-h-[calc(100dvh-3.5rem)] overflow-y-auto border-b bg-card px-4 pb-4 shadow-md"
                            >
                                {nav_links(nav)}
                            </nav>
                        </details>
                        <a href="/" class="text-sm font-semibold whitespace-nowrap lg:hidden">
                            "27b ops console"
                        </a>
                        <div class="ml-auto flex min-w-0 items-center gap-3 text-sm text-muted-foreground">
                            {switch}
                            <span class="hidden max-w-48 truncate sm:block">{who}</span>
                            <form method="post" action="/auth/logout">
                                <input type="hidden" name="csrf" value=csrf />
                                <button
                                    type="submit"
                                    class="whitespace-nowrap underline underline-offset-4 hover:text-foreground hover:cursor-pointer"
                                >
                                    "Log out"
                                </button>
                            </form>
                        </div>
                    </div>
                </header>
                <main id="content" class="mx-auto max-w-6xl px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
                    {flash.as_ref().map(flash_banner)}
                    {content}
                </main>
            </div>
        </div>
    }
    .to_html();
    document(title, session.theme, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        crate::session::new_session("admin-sub".into(), None, Some("Ray".into()))
    }

    #[test]
    fn document_escapes_title() {
        let html = document("<script>alert(1)</script>", Theme::System, String::new());
        assert!(!html.contains("<script>alert(1)</script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    /// The page names its entry and the shell marks exactly that one, in
    /// both copies of the navigation, and nothing else.
    #[test]
    fn the_named_entry_and_only_it_is_current() {
        let all = [
            (Nav::Home, "/"),
            (Nav::Memories, "/alaya"),
            (Nav::Duplicates, "/alaya/duplicates"),
            (Nav::Contradictions, "/alaya/contradictions"),
            (Nav::Health, "/alaya/health"),
            (Nav::Auth, "/alaya/auth"),
            (Nav::Lb, "/lb"),
            (Nav::LbClients, "/lb/clients"),
            (Nav::Selecta, "/selecta"),
        ];
        for (nav, href) in all {
            let html = page("t", nav, &session(), None, "");
            let marked = format!(r#"href="{href}" aria-current="page""#);
            assert_eq!(html.matches(&marked).count(), 2, "{nav:?}: {html}");
            assert_eq!(html.matches("aria-current").count(), 2, "{nav:?}");
        }
    }

    /// Every entry, grouped by module in order, each with its own `Nav`.
    #[test]
    fn the_navigation_lists_every_module_once() {
        let headings: Vec<_> = NAV.iter().map(|(m, _)| *m).collect();
        assert_eq!(
            headings,
            [None, Some("Ālaya"), Some("anthropic-lb"), Some("selecta")]
        );
        let mut navs: Vec<Nav> = NAV
            .iter()
            .flat_map(|(_, entries)| entries.iter().map(|(n, _, _)| *n))
            .collect();
        let count = navs.len();
        navs.dedup();
        assert_eq!(navs.len(), count, "an entry is listed twice");
        assert_eq!(count, 9);
    }

    /// Each control is on the page once at every width: one theme switch,
    /// one logout form, one skip link to the one `main`.
    #[test]
    fn the_shell_holds_one_copy_of_each_control() {
        let html = page("t", Nav::Home, &session(), None, "");
        assert_eq!(html.matches(r#"action="/auth/logout""#).count(), 1);
        assert_eq!(html.matches(r#"action="/prefs/theme""#).count(), 1);
        assert_eq!(html.matches(r#"aria-pressed="true""#).count(), 1);
        assert_eq!(html.matches("<main").count(), 1);
        assert!(html.contains(r##"href="#content""##));
        assert!(html.contains(r#"<main id="content""#));
        assert_eq!(html.matches(r#"<nav aria-label="Main""#).count(), 2);
        assert!(html.contains(">Ray</span>"), "the operator is named");
    }

    /// A POST's answer page keeps the navigation and logout but has no theme
    /// switch, which would send the browser to a GET of a POST-only URL.
    #[test]
    fn a_result_page_has_no_theme_switch() {
        let html = result_page("t", Nav::Duplicates, &session(), None, "");
        assert!(!html.contains(r#"action="/prefs/theme""#), "{html}");
        assert_eq!(html.matches(r#"action="/auth/logout""#).count(), 1);
        let marked = r#"href="/alaya/duplicates" aria-current="page""#;
        assert_eq!(html.matches(marked).count(), 2, "{html}");
    }

    /// The menu is a `<details>` that only exists below `lg` and the sidebar
    /// only from `lg`, so exactly one navigation is ever displayed.
    #[test]
    fn one_navigation_per_layout() {
        let html = page("t", Nav::Home, &session(), None, "");
        assert!(html.contains(r#"<aside class="hidden lg:block"#), "{html}");
        assert!(
            html.contains(r#"<details class="group lg:hidden">"#),
            "{html}"
        );
    }
}
