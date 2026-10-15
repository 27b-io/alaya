//! The snapshot world: every GET page the console serves, rendered against
//! loopback fakes of every upstream, so each page can be screenshotted at
//! phone, tablet and desktop widths. The healthy world answers like healthy
//! upstreams with varied data, warning and failing values included; the down
//! world leaves every upstream unreachable, for each page's degraded rendering.
//!
//! `every_get_page_renders_in_the_snapshot_world` keeps both worlds whole and
//! fails when a GET route has no snapshot; `write_snapshots` writes the HTML a
//! static file server and a screenshot tool load.

use std::collections::HashMap;

use axum::Json;
use axum::extract::{Path, Query};
use leptos::prelude::*;
use serde_json::{Value, json};
use sha2::Digest;

use super::*;
use crate::effect::RecordState;
use crate::testkit::{
    FakeMetrics, FakeStore, STORE_TOKEN, T0, V0, World, k3s, k3s_documents, series,
};
use crate::theme::Theme;
use crate::ui::{
    BarSeries, Card, CardContent, CardDescription, CardHeader, CardTitle, Colour, HBarList, Meter,
    PageHeader, Part, Series, Sparkline, StackedBar, Stat, StatGrid, Table, TableBody, TableCell,
    TableHead, TableHeader, TableRow, TableWrapper, fmt_count,
};

const HOUR_US: u64 = 3_600 * 1_000_000;
const DAY_US: u64 = 24 * HOUR_US;
const HOUR_MS: u64 = 3_600 * 1_000;
const DAY_MS: u64 = 24 * HOUR_MS;

/// Nothing listens here, so every call fails at once rather than waiting.
const CLOSED: &str = "http://127.0.0.1:1";

// ─── Pages ──────────────────────────────────────────────────────────────────

/// A page to snapshot.
struct Page {
    /// The route pattern it exercises, as `routes()` names it.
    route: &'static str,
    /// The navigation entry it is under: its own route, or for a page that
    /// is not an entry, the entry it is reached from.
    under: &'static str,
    /// The path to GET.
    path: String,
    /// The stem its files are written under.
    stem: String,
}

/// Every GET page the console serves, as the healthy world renders it.
/// Where a page needs a query to render its full layout, the path carries
/// it: the supersede form prefilled from a contradiction, the pair the
/// queue opens, the write the effect page confirms.
fn pages() -> Vec<Page> {
    let page = |route: &'static str, under: &'static str, path: String, stem: &str| Page {
        route,
        under,
        path,
        stem: stem.to_string(),
    };
    let effect = format!(
        "/lb/clients/effect?op=disabled.ci-review-bot.{T0}&doc={V0}.{}&at=0",
        doc_digest(V0)
    );
    vec![
        page("/", "/", "/".into(), "home"),
        page(
            "/alaya",
            "/alaya",
            "/alaya?include_superseded=on".into(),
            "alaya",
        ),
        page(
            "/alaya/memory/{hash}",
            "/alaya",
            format!("/alaya/memory/{}", hm(1)),
            "alaya-memory",
        ),
        page(
            "/alaya/supersede",
            "/alaya",
            format!(
                "/alaya/supersede?old={}&new={}&back=%2Falaya%2Fcontradictions",
                hm(6),
                hm(7)
            ),
            "alaya-supersede",
        ),
        page(
            "/alaya/duplicates",
            "/alaya/duplicates",
            "/alaya/duplicates".into(),
            "alaya-duplicates",
        ),
        page(
            "/alaya/contradictions",
            "/alaya/contradictions",
            "/alaya/contradictions".into(),
            "alaya-contradictions",
        ),
        page(
            "/alaya/contradictions/pair",
            "/alaya/contradictions",
            format!("/alaya/contradictions/pair?a={}&b={}", hm(6), hm(7)),
            "alaya-contradictions-pair",
        ),
        page(
            "/alaya/auth",
            "/alaya/auth",
            "/alaya/auth".into(),
            "alaya-auth",
        ),
        page(
            "/alaya/health",
            "/alaya/health",
            "/alaya/health".into(),
            "alaya-health",
        ),
        page("/lb", "/lb", "/lb".into(), "lb"),
        page(
            "/lb/clients",
            "/lb/clients",
            "/lb/clients".into(),
            "lb-clients",
        ),
        page(
            "/lb/clients/disable",
            "/lb/clients",
            "/lb/clients/disable?disable=support-triage".into(),
            "lb-clients-disable",
        ),
        page(
            "/lb/clients/effect",
            "/lb/clients",
            effect,
            "lb-clients-effect",
        ),
        page(
            "/lb/clients/history",
            "/lb/clients",
            "/lb/clients/history".into(),
            "lb-clients-history",
        ),
        page("/selecta", "/selecta", "/selecta".into(), "selecta"),
        page(
            "/selecta/task/{id}",
            "/selecta",
            format!("/selecta/task/{FAILED_ID}"),
            "selecta-task",
        ),
    ]
}

/// GET routes that are not pages: the health check, the stylesheet, and the
/// two legs of the login redirect.
const NOT_PAGES: [&str; 4] = [
    "/healthz",
    "/static/console.css",
    "/auth/login",
    "/auth/callback",
];

/// Pages that read no upstream: the down world would render them unchanged.
const READS_NOTHING: [&str; 1] = ["/lb/clients/disable"];

/// Pages whose upstream read is the whole page: with alaya-server down they
/// answer the console's 502 error page rather than a page with a dark
/// section. A page moved to per-section degradation leaves this list.
const FAILS_WHOLE: [&str; 5] = [
    "/alaya",
    "/alaya/memory/{hash}",
    "/alaya/duplicates",
    "/alaya/contradictions",
    "/alaya/auth",
];

/// The down world's pages: every page that reads an upstream.
fn down_pages() -> Vec<Page> {
    pages()
        .into_iter()
        .filter(|p| !READS_NOTHING.contains(&p.route))
        .map(|p| Page {
            stem: format!("{}-down", p.stem),
            ..p
        })
        .collect()
}

/// Text a healthy-world page must not hold: each is a section gone dark, so
/// a fake that stopped answering what a page reads fails here rather than
/// leaving a hollow screenshot.
const DARK: [&str; 4] = [
    ">unavailable<",
    ">unreachable<",
    "Could not load",
    "unreadable",
];

/// Both worlds, each with its pages and whether it is the down world.
async fn worlds() -> [(AppState, Vec<Page>, bool); 2] {
    [
        (healthy_world().await, pages(), false),
        (down_world(), down_pages(), true),
    ]
}

/// The allowlisted operator's cookies, with the theme saved when one is given.
fn operator_cookie(state: &AppState, theme: Option<&str>) -> String {
    let sess = session::new_session("admin-sub".into(), Some("ops@example.test".into()), None);
    let cookie = session_cookie_header(state, &sess);
    match theme {
        Some(t) => format!(
            "{cookie}; {}",
            encrypted_cookie_header(state, crate::theme::THEME_COOKIE, t.into())
        ),
        None => cookie,
    }
}

/// GET every page of `worlds` as the operator, in `theme` when one is given,
/// and hold each to what every snapshot must be: the status its world
/// answers, nothing the CSP blocks, every module configured, one page
/// heading (a `PageHeader`), and in the healthy world no dark section. Returns (stem, HTML) in page order.
async fn render_pages(
    worlds: &[(AppState, Vec<Page>, bool)],
    theme: Option<&str>,
) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (state, pages, down) in worlds {
        let cookie = operator_cookie(state, theme);
        for Page {
            route, path, stem, ..
        } in pages
        {
            let (status, html) = get_page(state, &cookie, path).await;
            let want = if *down && FAILS_WHOLE.contains(route) {
                StatusCode::BAD_GATEWAY
            } else {
                StatusCode::OK
            };
            assert_eq!(status, want, "{stem} (GET {path}): {html}");
            assert!(!html.contains("<script"), "{stem}: a script the CSP blocks");
            assert!(!html.contains(" style="), "{stem}: a style the CSP blocks");
            assert!(!html.contains("not configured"), "{stem}: {html}");
            if status == StatusCode::OK {
                assert_eq!(html.matches("<h1").count(), 1, "{stem}: one page heading");
            }
            if !*down {
                for dark in DARK {
                    assert!(
                        !html.contains(dark),
                        "{stem} renders {dark:?}: a fake does not answer a read the page makes"
                    );
                }
            }
            out.push((stem.clone(), html));
        }
    }
    out
}

#[tokio::test]
async fn every_get_page_renders_in_the_snapshot_world() {
    let worlds = worlds().await;
    let rendered = render_pages(&worlds, None).await;
    let mut stems: Vec<&str> = rendered.iter().map(|(s, _)| s.as_str()).collect();
    stems.sort_unstable();
    stems.dedup();
    assert_eq!(
        stems.len(),
        rendered.len(),
        "two pages share a stem, so one snapshot overwrites the other"
    );

    // A GET page added to `routes()` without a snapshot fails here.
    let (state, snapped) = (&worlds[0].0, &worlds[0].1);
    let cookie = operator_cookie(state, None);
    for (route, _) in routes() {
        if NOT_PAGES.contains(&route) {
            continue;
        }
        let path = route.replace("{hash}", &hm(0)).replace("{id}", FAILED_ID);
        let (status, _) = get_page(state, &cookie, &path).await;
        if status == StatusCode::METHOD_NOT_ALLOWED {
            continue;
        }
        assert!(
            snapped.iter().any(|p| p.route == route),
            "GET {route} answers {status} but has no snapshot: add it to pages()"
        );
    }
}

/// The two copies of the main navigation in `html`: the sidebar's and the
/// narrow-screen menu's.
fn main_navs(html: &str) -> Vec<&str> {
    html.split("<nav aria-label=\"Main\"")
        .skip(1)
        .filter_map(|n| n.split("</nav>").next())
        .collect()
}

/// Each page marks the entry it is under, once in each copy of the
/// navigation, and each entry links to a page that marks itself.
#[tokio::test]
async fn every_page_marks_the_navigation_entry_it_is_under() {
    let state = healthy_world().await;
    let cookie = operator_cookie(&state, None);
    let pages = pages();
    for Page {
        under, path, stem, ..
    } in &pages
    {
        let (_, html) = get_page(&state, &cookie, path).await;
        let navs = main_navs(&html);
        assert_eq!(navs.len(), 2, "{stem}");
        let marked = format!("href=\"{under}\" aria-current=\"page\"");
        for nav in navs {
            assert_eq!(
                nav.matches(&marked).count(),
                1,
                "{stem} is not under {under}"
            );
            assert_eq!(nav.matches("aria-current").count(), 1, "{stem}");
        }
    }
    let (_, home) = get_page(&state, &cookie, "/").await;
    let entries: Vec<&str> = main_navs(&home)[0]
        .split("href=\"")
        .skip(1)
        .filter_map(|h| h.split('"').next())
        .collect();
    for href in &entries {
        assert!(
            pages.iter().any(|p| p.route == *href && p.under == *href),
            "the navigation links {href}, which is no page's own entry"
        );
    }
}

/// Writes `<stem>.<theme>.html` for every page of both worlds in both
/// explicit themes, `pages.txt` with the stems in page order, and the
/// stylesheet where the pages' `/static/console.css` link finds it under a
/// static file server rooted at `SNAPSHOT_DIR`.
#[tokio::test]
#[ignore = "writes HTML snapshots: SNAPSHOT_DIR=<dir> cargo test -p ops-console --bin ops-console write_snapshots -- --ignored"]
async fn write_snapshots() {
    let dir = std::env::var_os("SNAPSHOT_DIR")
        .filter(|d| !d.is_empty())
        .map(std::path::PathBuf::from)
        .expect("SNAPSHOT_DIR is unset: set it to the directory the snapshots are written to");
    let static_dir = dir.join("static");
    std::fs::create_dir_all(&static_dir)
        .unwrap_or_else(|e| panic!("creating {}: {e}", static_dir.display()));
    let css = concat!(env!("CARGO_MANIFEST_DIR"), "/static/console.css");
    std::fs::copy(css, static_dir.join("console.css"))
        .unwrap_or_else(|e| panic!("copying {css}: {e}"));
    let write = |name: String, text: &str| {
        let path = dir.join(name);
        std::fs::write(&path, text).unwrap_or_else(|e| panic!("writing {}: {e}", path.display()));
    };

    let worlds = worlds().await;
    for theme in ["light", "dark"] {
        for (stem, html) in render_pages(&worlds, Some(theme)).await {
            write(format!("{stem}.{theme}.html"), &html);
        }
    }
    for theme in [Theme::Light, Theme::Dark] {
        write(
            format!("{GALLERY}.{}.html", theme.as_str()),
            &gallery_page(theme),
        );
    }
    let stems: Vec<&str> = std::iter::once(GALLERY)
        .chain(
            worlds
                .iter()
                .flat_map(|(_, pages, _)| pages.iter().map(|p| p.stem.as_str())),
        )
        .collect();
    write("pages.txt".into(), &format!("{}\n", stems.join("\n")));
}

// ─── The chart gallery ──────────────────────────────────────────────────────

/// The stem the chart gallery is written under, first among the pages.
const GALLERY: &str = "charts";

/// The gallery in the shell, in `theme`. It is no route: only the snapshot
/// tool renders it, so it is screenshotted with the pages and never served.
fn gallery_page(theme: Theme) -> String {
    let mut sess = session::new_session("admin-sub".into(), Some("ops@example.test".into()), None);
    sess.theme = theme;
    ui::page("Chart gallery", ui::Nav::Home, &sess, None, chart_gallery())
}

/// The judge's verdicts in display order, each on its own series slot in
/// every chart that shows them, so a verdict keeps its colour across charts.
const VERDICTS: [(&str, Colour); 6] = [
    ("contradiction", Colour::SERIES[0]),
    ("supersession", Colour::SERIES[1]),
    ("coexist", Colour::SERIES[2]),
    ("unrelated", Colour::SERIES[3]),
    ("unjudged", Colour::SERIES[4]),
    ("never judged", Colour::SERIES[5]),
];

/// `NOVELTY`'s similarity bands, closest first. They are ordered, so they
/// take steps of one hue; "none" is no band at all.
const BANDS: [(&str, Colour); 5] = [
    ("≥ 0.95", Colour::RAMP[4]),
    ("0.85–0.95", Colour::RAMP[3]),
    ("0.70–0.85", Colour::RAMP[2]),
    ("< 0.70", Colour::RAMP[1]),
    ("none", Colour::OTHER),
];

/// Judge-cap admissions over a fortnight: two days with no sample and a day
/// with none admitted.
const ADMITTED: [Option<f64>; 14] = [
    Some(612.0),
    Some(845.0),
    Some(1000.0),
    Some(1000.0),
    Some(733.0),
    None,
    Some(402.0),
    Some(0.0),
    Some(951.0),
    Some(1000.0),
    Some(688.0),
    None,
    Some(870.0),
    Some(41.0),
];

/// One chart in the gallery, under a caption saying what it shows.
fn example(caption: &'static str, chart: impl IntoView) -> impl IntoView {
    view! {
        <div class="min-w-0 space-y-2">
            <p class="text-xs text-muted-foreground">{caption}</p>
            {chart}
        </div>
    }
}

/// A gallery section: one chart kind, its use and its examples.
fn section(
    title: &'static str,
    about: &'static str,
    children: impl IntoView + 'static,
) -> impl IntoView {
    view! {
        <Card>
            <CardHeader>
                <CardTitle>{title}</CardTitle>
                <CardDescription>{about}</CardDescription>
            </CardHeader>
            <CardContent>{children}</CardContent>
        </Card>
    }
}

/// The chart kit on one page: each chart drawn from the fixtures' data,
/// then its edge cases.
fn chart_gallery() -> impl IntoView {
    let n = |v: u64| Some(v as f64);
    let week: Vec<String> = WEEK.iter().map(|d| d[5..].to_string()).collect();
    let fortnight: Vec<String> = (18..=30)
        .map(|d| format!("09-{d}"))
        .chain(["10-01".to_string()])
        .collect();
    let by_day = |table: &[[u64; 5]; 7], j: usize| table.iter().map(|row| n(row[j])).collect();
    let judged: Vec<Series> = VERDICTS[..5]
        .iter()
        .enumerate()
        .map(|(j, (label, colour))| Series::new(*label, by_day(&JUDGED, j), *colour))
        .collect();
    let novelty: Vec<Series> = BANDS
        .iter()
        .enumerate()
        .map(|(j, (label, colour))| Series::new(*label, by_day(&NOVELTY, j), *colour))
        .collect();
    let verdicts = |counts: [Option<f64>; 6]| {
        VERDICTS
            .iter()
            .zip(counts)
            .map(|((label, colour), v)| Part::new(*label, v, *colour))
            .collect::<Vec<_>>()
    };
    let edges = ["CONTRADICTS", "SUPERSEDES", "HEBBIAN"]
        .into_iter()
        .zip([4_589, 131, 77])
        .enumerate()
        .map(|(i, (kind, count))| Part::new(kind, n(count), Colour::SERIES[i]))
        .collect::<Vec<_>>();
    let ranked = |rows: &[(&str, Option<f64>)]| {
        rows.iter()
            .map(|(label, v)| Part::new(*label, *v, Colour::SERIES[0]))
            .collect::<Vec<_>>()
    };
    let mut failures: Vec<Part> = FAILURES
        .iter()
        .map(|(reason, count)| Part::new(*reason, n(*count), Colour::SERIES[0]))
        .collect();
    failures.push(Part::new("other", n(5), Colour::OTHER));

    let budgets = BUDGETS
        .iter()
        .map(|(client, limit, used)| {
            let (name, label) = (client.to_string(), format!("{client}: live budget"));
            view! {
                <TableRow>
                    <TableCell><span class="font-mono text-xs">{name}</span></TableCell>
                    <TableCell>
                        <Meter label=label used=n(*used) limit=n(*limit) fmt=lb::fmt_tokens />
                    </TableCell>
                </TableRow>
            }
        })
        .collect_view();
    let burn = BURN
        .iter()
        .map(|(client, days)| (client.to_string(), days.to_vec()))
        .chain([("dormant".to_string(), vec![None; 7])])
        .map(|(client, days)| {
            let latest = days[6].map_or_else(|| "—".to_string(), lb::fmt_tokens);
            let label = format!("{client}: daily burn");
            let x = week.clone();
            view! {
                <Stat label=client>
                    {latest}
                    <Sparkline label=label values=days x=x fmt=lb::fmt_tokens />
                </Stat>
            }
        })
        .collect_view();

    let grid = "grid gap-6 md:grid-cols-2";
    view! {
        <div class="space-y-6">
            <PageHeader
                title="Chart gallery"
                description="Every chart in the kit, drawn from the snapshot fixtures, then each one's edge cases. No route serves this page."
            />
            {section(
                "Meter",
                "Used against a limit. The fill turns to warning at 80% and critical at the limit; a tick marks the warning threshold.",
                view! {
                    <div class="space-y-6">
                        <TableWrapper>
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>"Client"</TableHead>
                                        <TableHead>"Live budget"</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>{budgets}</TableBody>
                            </Table>
                        </TableWrapper>
                        <div class=grid>
                            {example("A daily cap, near it", view! { <Meter label="searches" used=n(188) limit=n(200) fmt=fmt_count /> })}
                            {example("A daily cap, reached", view! { <Meter label="downloads" used=n(200) limit=n(200) fmt=fmt_count /> })}
                            {example("Over budget: drawn full, the share as it is", view! { <Meter label="nightly-evals" used=Some(2.43e9) limit=Some(2e9) fmt=lb::fmt_tokens /> })}
                            {example("A utilisation, a share already: 5h window", view! { <Meter label="5h utilisation" used=Some(0.62) limit=Some(1.0) /> })}
                            {example("Disabled names against the store's cap", view! { <Meter label="disabled names" used=n(12) limit=n(1024) fmt=fmt_count /> })}
                            {example("Used unknown: no data, no bar", view! { <Meter label="adds" used=None limit=n(40) fmt=fmt_count /> })}
                            {example("A cap of zero: no share to draw", view! { <Meter label="judge daily cap" used=n(0) limit=n(0) fmt=fmt_count /> })}
                            {example("Half a pair", view! { <Meter label="grabs" used=n(4) limit=None fmt=fmt_count /> })}
                        </div>
                    </div>
                },
            )}
            {section(
                "StackedBar",
                "A part-to-whole mix, with each part's value and share in the legend. A verdict keeps its colour in every chart.",
                view! {
                    <div class=grid>
                        {example("Open pairs by verdict", view! { <StackedBar label="Open pairs by verdict" parts=verdicts([n(7), n(1_613), n(1_172), n(826), n(9), n(1_716)]) /> })}
                        {example("Resolved pairs by verdict", view! { <StackedBar label="Resolved pairs by verdict" parts=verdicts([n(3), n(15), n(36), n(26), n(0), n(97)]) /> })}
                        {example("Corpus edges by type", view! { <StackedBar label="Corpus edges by type" parts=edges /> })}
                        {example("All zero: an empty bar", view! { <StackedBar label="Resolved pairs by verdict" parts=verdicts([n(0); 6]) /> })}
                        {example("A part with no data: no whole", view! { <StackedBar label="Open pairs by verdict" parts=verdicts([n(7), None, n(1_172), n(826), n(9), n(1_716)]) /> })}
                        {example("Nothing known", view! { <StackedBar label="Open pairs by verdict" parts=verdicts([None; 6]) /> })}
                    </div>
                },
            )}
            {section(
                "BarSeries",
                "Bars per day, stacked by category. Ticks at round values; the first and last day labelled; every bar and day has a tooltip.",
                view! {
                    <div class=grid>
                        {example("Pairs judged per day, by verdict", view! { <BarSeries label="Pairs judged per day" x=week.clone() series=judged /> })}
                        {example("Writes per day, by similarity to their nearest neighbour", view! { <BarSeries label="Writes per day by similarity" x=week.clone() series=novelty /> })}
                        {example("Admitted per day: a fortnight, two days with no sample", view! { <BarSeries label="Judge admissions per day" x=fortnight series=vec![Series::new("admitted", ADMITTED.to_vec(), Colour::SERIES[0])] /> })}
                        {example("All zero: bars at the baseline", view! { <BarSeries label="Pairs judged per day" x=week.clone() series=vec![Series::new("judged", vec![n(0); 7], Colour::SERIES[0])] /> })}
                        {example("Nothing known", view! { <BarSeries label="Pairs judged per day" x=week.clone() series=vec![Series::new("judged", vec![None; 7], Colour::SERIES[0])] /> })}
                    </div>
                },
            )}
            {section(
                "Sparkline",
                "A trend under a stat's value: a line from zero, broken where there is no data, the latest day in the accent.",
                view! { <StatGrid>{burn}</StatGrid> },
            )}
            {section(
                "HBarList",
                "A ranking, largest first, each row against the largest. The rest folds into a grey \"other\".",
                view! {
                    <div class=grid>
                        {example("Stored judge failures, by reason", view! { <HBarList label="Stored judge failures" rows=failures /> })}
                        {example("Upstream transport errors, by kind", view! { <HBarList label="Transport errors" rows=ranked(&[("other", n(4_103)), ("timeout", n(706)), ("connect", n(12))]) /> })}
                        {example("A zero and a row with no data", view! { <HBarList label="Transport errors" rows=ranked(&[("timeout", n(706)), ("connect", n(0)), ("reset", None)]) /> })}
                        {example("Nothing known", view! { <HBarList label="Transport errors" rows=ranked(&[("timeout", None)]) /> })}
                    </div>
                },
            )}
        </div>
    }
}

/// The gallery holds to the CSP like every page, draws every kind of chart
/// in both colour families, and shows its no-data cases as such.
#[test]
fn the_chart_gallery_draws_every_chart() {
    let html = gallery_page(Theme::Dark);
    assert!(
        html.contains(r#"<html lang="en" class="scroll-pt-14 dark""#),
        "{html}"
    );
    assert!(!html.contains("<script"), "a script the CSP blocks");
    assert!(!html.contains(" style="), "a style the CSP blocks");
    for class in [
        "fill-chart-1",
        "fill-chart-6",
        "fill-chart-ramp-5",
        "fill-chart-warn",
        "fill-chart-crit",
        "<polyline",
    ] {
        assert!(html.contains(class), "{class}");
    }
    assert!(html.matches(r#"role="img""#).count() >= 30, "{html}");
    assert!(html.contains(">no data</p>"), "{html}");
}

// ─── The worlds ─────────────────────────────────────────────────────────────

/// Every module configured against loopback fakes that answer like healthy
/// upstreams: alaya-server answering every read the alaya pages make, the LB
/// with its metrics store, selecta, and the client store with its readers in
/// that same metrics store.
async fn healthy_world() -> AppState {
    let world = store_world();
    let metrics = fleet_metrics(&world)
        .router()
        .route("/api/v1/query_range", get(burn_history));
    // `fake_selecta` returns a state with selecta alone: its URL is the fake.
    let selecta = fake_selecta(
        &SelectaSeen::default(),
        selecta_world,
        selecta_metrics(),
        (StatusCode::OK, r#"{"status":"ok","seq":9124}"#),
    )
    .await;
    let lb = Router::new().route("/_stats", get(|| async { Json(lb_stats()) }));
    let mut config = test_config();
    config.alaya_url = fake_upstream(alaya_upstream()).await.parse().unwrap();
    config.lb = Some(config::LbConfig {
        url: fake_upstream(lb).await.parse().unwrap(),
        api_key: LB_KEY.into(),
        metrics_url: fake_upstream(metrics).await.parse().unwrap(),
    });
    config.selecta = Some(config::SelectaConfig {
        url: selecta.config.selecta.as_ref().unwrap().url.clone(),
        api_key: SELECTA_KEY.into(),
    });
    config.store = Some(config::StoreConfig {
        url: fake_upstream(client_store(&world).router())
            .await
            .parse()
            .unwrap(),
        token: STORE_TOKEN.into(),
        metrics_addr: "127.0.0.1:0".into(),
    });
    AppState::new(config)
}

/// `test_state_with_lb()` with the client store configured too: built here
/// because the store needs its config before `AppState::new`. Every module
/// configured, every upstream unreachable.
fn down_world() -> AppState {
    let mut config = test_config();
    config.lb = Some(config::LbConfig {
        url: CLOSED.parse().unwrap(),
        api_key: LB_KEY.into(),
        metrics_url: CLOSED.parse().unwrap(),
    });
    config.selecta = Some(config::SelectaConfig {
        url: CLOSED.parse().unwrap(),
        api_key: SELECTA_KEY.into(),
    });
    config.store = Some(config::StoreConfig {
        url: CLOSED.parse().unwrap(),
        token: STORE_TOKEN.into(),
        metrics_addr: "127.0.0.1:0".into(),
    });
    AppState::new(config)
}

// ─── alaya-server ───────────────────────────────────────────────────────────

/// (content, summary, type, tags). Memory 1 is superseded by 2 and 2 by 3:
/// the chain the detail page walks. The pairs and groups draw on the rest.
const CORPUS: [(&str, &str, &str, &[&str]); 12] = [
    (
        "Hybrid search reranks the top 20 RRF candidates with the cross-encoder. Past the 5 s rerank budget it falls back to RRF order instead of failing the search, and logs the fallback once per request. On the 500-question eval set R@5 went from 0.936 to 0.990 with top_n=20.",
        "Hybrid search reranks the top 20 candidates; on a timeout it falls back to RRF order (R@5 0.936 → 0.990).",
        "decision",
        &["search", "rerank", "eval"],
    ),
    (
        "Embedding cache keys are built from the model name and the text only, so a query embedding and a passage embedding of the same text share one cache entry.\n\nSeen while profiling hybrid search: the L2 hit rate jumped from 41% to 97% after the client upgrade, which looked like a win until the rerank scores drifted.\n\nThe lookup that surfaced it served a passage vector to a query: alaya:embed:3f8a2c1e9b7d6054a1c2e3f4d5b6a7980c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6",
        "Embedding cache keys cover the model and the text, so query and passage embeddings share entries: seen as alaya:embed:3f8a2c1e9b7d6054a1c2e3f4d5b6a7980c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5f6.",
        "note",
        &["cache", "embeddings", "profiling"],
    ),
    (
        "Embedding cache keys hash the model, the dimensions and the text with BLAKE2b-256. The prompt name is part of the key too, so a query embedding and a passage embedding of the same text are separate entries.",
        "Embedding cache keys: BLAKE2b-256 over model, dims, prompt name and text.",
        "note",
        &["cache", "embeddings"],
    ),
    (
        "L2 embedding cache keys are alaya:embed:{blake2b256-hex} over [model, dims, prompt_name, text], computed by an un-namespaced client so every SDK derives the same key. A client namespace silently prefixes every key and breaks cross-SDK conformance: the interop vectors at https://docs.example.test/cache/interop/v1/conformance-vectors/embedding-keys-model-dims-prompt-name-text.json fail on their first entry.",
        "Cache keys are cross-SDK interop/v1: alaya:embed:{blake2b256} over model, dims, prompt name and text. Never namespace the client.",
        "reference",
        &["cache", "embeddings", "interop", "conformance-vectors-v1"],
    ),
    (
        "Vector store snapshots run nightly at 02:00 UTC and are kept for 14 days. A restore needs the snapshot and the graph export from the same night.",
        "Snapshots: nightly at 02:00 UTC, kept for 14 days.",
        "reference",
        &["backup", "qdrant"],
    ),
    (
        "Since the retention change, vector store snapshots run every 6 hours and are kept for 7 days. The nightly job was removed.",
        "Snapshots: every 6 hours, kept for 7 days; the nightly job is gone.",
        "note",
        &["backup", "qdrant", "retention"],
    ),
    (
        "The contradiction judge's daily cap defaults to 500 store-path calls per process.",
        "The judge daily cap defaults to 500.",
        "note",
        &["judge", "config"],
    ),
    (
        "JUDGE_DAILY_CAP defaults to 1000 store-path judge calls per UTC day, per process: N replicas spend up to N × cap. Past the cap a pair fails closed to unjudged and the backfill picks it up later; operator backfill is not counted.",
        "JUDGE_DAILY_CAP: 1000 calls per UTC day per process; past it, pairs fail closed to unjudged.",
        "decision",
        &["judge", "config", "budget"],
    ),
    (
        "Never roll alaya-server across a change to the write protocol: stop every old server before a new one starts, or an old server's snapshot write can undo a new one.",
        "No rolling update across a write-protocol change: stop the old servers first.",
        "decision",
        &["deploy", "cas"],
    ),
    (
        "Deploy alaya-server with the Recreate strategy. Scaling the old deployment to one replica is not enough: a rolling update's surge pod still runs beside it.",
        "Use Recreate for alaya-server; a surge pod defeats scaling to one.",
        "note",
        &["deploy", "kubernetes"],
    ),
    (
        "The graph sweep holds a page when more than 3 nodes in a 500-node window are gone from the vector store, or every node in it is, and deletes at most 3 nodes per call.",
        "Graph sweep: holds past 3 gone nodes per window and deletes at most 3 per call.",
        "reference",
        &["graph", "sweep", "safety"],
    ),
    (
        "After any rollout that touched the keyword vector, run POST /backfill/keywords. It writes only the bm25 vector, for points that lack it, and is safe to repeat.",
        "Re-run the keyword backfill after a rollout that touched bm25.",
        "task",
        &["bm25", "backfill"],
    ),
];

/// (superseded, survivor, reason): the chain 1 → 2 → 3.
const SUPERSEDED: [(usize, usize, &str); 2] = [
    (
        1,
        2,
        "Wrong: the prompt name is part of the cache key, so query and passage embeddings of one text never share an entry. The 97% hit rate came from a client that dropped the prompt name, fixed in the same release.",
    ),
    (
        2,
        3,
        "Merged as a duplicate of the interop/v1 reference, which also names the key prefix.",
    ),
];

/// The content hash alaya-server gives `content`: its SHA-256. Every hash in
/// the world is a real one, and every memory a page links is in the corpus.
fn hash_of(content: &str) -> String {
    crate::effect::hex(&sha2::Sha256::digest(content.as_bytes()))
}

/// Memory `i`'s content hash.
fn hm(i: usize) -> String {
    hash_of(CORPUS[i].0)
}

/// The corpus as alaya-server returns memories with `output: both`.
fn memories() -> Vec<Value> {
    CORPUS
        .iter()
        .enumerate()
        .map(|(i, (content, summary, kind, tags))| {
            let created = 1_789_250_000.0 + 187_000.0 * i as f64;
            let mut m = json!({
                "content_hash": hm(i), "content": content, "summary": summary,
                "memory_type": kind, "tags": tags, "created_at": created,
                "updated_at": created + 5_400.0, "salience_score": 0.31 + 0.053 * i as f64,
                "access_count": 3 + 7 * i, "provenance": { "trust_score": 0.62 + 0.03 * i as f64 },
                "metadata": { "source": "mcp", "importance": 0.4 + 0.05 * i as f64 },
            });
            if let Some((_, survivor, reason)) = SUPERSEDED.iter().find(|(old, _, _)| *old == i) {
                m["metadata"]["superseded_by"] = json!(hm(*survivor));
                m["supersession_reason"] = json!(reason);
            }
            m
        })
        .collect()
}

/// An alaya-server answering every read the alaya pages make, from the corpus.
fn alaya_upstream() -> Router {
    Router::new()
        .route(
            "/health/detail",
            get(|| async { Json(json!({ "status": "healthy" })) }),
        )
        .route(
            "/search",
            post(|Json(body): Json<Value>| async move {
                let all = body["include_superseded"] == true;
                let rows: Vec<Value> = memories()
                    .into_iter()
                    .filter(|m| all || m["metadata"].get("superseded_by").is_none())
                    .collect();
                Json(json!({ "results": rows, "has_more": true }))
            }),
        )
        .route(
            "/memories/{hash}",
            get(|Path(hash): Path<String>| async move {
                match memories().into_iter().find(|m| m["content_hash"] == *hash) {
                    Some(m) => (StatusCode::OK, Json(json!({ "found": true, "memory": m }))),
                    None => (
                        StatusCode::NOT_FOUND,
                        Json(json!({ "error": "Resource not found" })),
                    ),
                }
            }),
        )
        .route(
            "/relation",
            post(|Json(body): Json<Value>| async move {
                Json(relations(body["content_hash"].as_str().unwrap_or("")))
            }),
        )
        .route(
            "/duplicates/find",
            post(|| async { Json(duplicate_groups()) }),
        )
        .route(
            "/contradictions",
            post(|| async {
                Json(json!({ "success": true, "pairs": contradiction_pairs(), "next_offset": 50 }))
            }),
        )
        .route("/auth/config", get(|| async { Json(auth_config()) }))
        .route("/stats", get(|| async { Json(health_stats()) }))
}

/// `me`'s relations: each type, both directions, and two CONTRADICTS edges
/// the detail page links to their pair.
fn relations(me: &str) -> Value {
    let edge = |source: &str, target: &str, relation_type: &str, created_at: f64| {
        json!({
            "source": source, "target": target,
            "relation_type": relation_type, "created_at": created_at,
        })
    };
    json!({ "relations": [
        edge(me, &hm(3), "RELATES_TO", 1_790_400_000.0),
        edge(me, &hm(0), "RELATES_TO", 1_790_410_000.0),
        edge(&hm(11), me, "PRECEDES", 1_790_100_000.0),
        edge(me, &hm(5), "CONTRADICTS", 1_790_500_000.0),
        edge(&hm(9), me, "CONTRADICTS", 1_790_600_000.0),
    ]})
}

/// One queue page holding every card state, newest first: a contradiction, a
/// supersession with its survivor, a coexist the judge would keep, a stored
/// judge failure, a pair never judged, and a pair already kept both. The
/// server's filters are ignored: the page renders whatever it is sent.
fn contradiction_pairs() -> Vec<Value> {
    let pair = |a: usize, b: usize, detected: f64, fields: Value| {
        let mut p = queue_pair(&hm(a), &hm(b));
        p["memory_a_content"] = json!(CORPUS[a].1);
        p["memory_b_content"] = json!(CORPUS[b].1);
        p["confidence"] = json!(detected);
        p["verdict_model"] = json!("claude-sonnet-5");
        for (k, v) in fields.as_object().unwrap() {
            p[k] = v.clone();
        }
        p
    };
    let judged_at = 1_790_520_000.0;
    let mut pairs = vec![
        pair(
            4,
            5,
            0.74,
            json!({
                "verdict": "contradiction", "survivor": null, "verdict_confidence": 0.91, "judged_at": judged_at,
                "verdict_reason": "Both describe the current snapshot schedule and cannot both hold: one says nightly at 02:00 UTC with 14-day retention, the other every 6 hours with 7-day retention and the nightly job removed.",
            }),
        ),
        pair(
            6,
            7,
            0.61,
            json!({
                "verdict": "supersession", "survivor": hm(7), "verdict_confidence": 0.97, "judged_at": judged_at + 340.0,
                "verdict_reason": "B states the current default (1000 per UTC day, per process) and the fail-closed behaviour past the cap; A's 500 predates the change, so B supersedes A. Source: https://docs.example.test/alaya/configuration#judge-daily-cap-per-process-budget",
            }),
        ),
        pair(
            8,
            9,
            0.58,
            json!({
                "verdict": "coexist", "survivor": null, "verdict_confidence": 0.88, "judged_at": judged_at + 900.0,
                "verdict_reason": "Both are true and complementary: A forbids rolling across a write-protocol change, B names the deployment strategy that enforces it.",
            }),
        ),
        pair(
            10,
            11,
            0.66,
            json!({
                "verdict": "unjudged", "survivor": null, "verdict_confidence": 0.0, "judged_at": judged_at + 1_500.0,
                "verdict_reason": "unjudged: schema violation: verdict_reason exceeded 2000 characters (got 2417) in response msg_01JX9Q2Z7K4V8M3N5P6R1T0W2Y8B4C6D1E3F5G7H9J",
            }),
        ),
        pair(
            0,
            3,
            0.52,
            json!({
                "verdict": "unjudged", "survivor": null, "verdict_confidence": null,
                "verdict_model": null, "judged_at": null, "verdict_reason": null,
            }),
        ),
        pair(
            11,
            4,
            0.55,
            json!({
                "verdict": "unrelated", "survivor": null, "verdict_confidence": 0.93, "judged_at": judged_at - 86_400.0,
                "verdict_reason": "Unrelated: a backfill task and a backup schedule that share vector-store vocabulary.",
                "resolution": "keep_both", "resolved_at": judged_at - 3_600.0, "resolved_via": "operator:console",
            }),
        ),
    ];
    for (n, p) in pairs.iter_mut().enumerate() {
        p["created_at"] = json!(1_790_400_000.0 - 9_000.0 * n as f64);
    }
    pairs
}

/// Three duplicate groups, one of three memories.
fn duplicate_groups() -> Value {
    let group = |members: &[usize], similarity: f64| {
        let hashes: Vec<String> = members.iter().map(|&i| hm(i)).collect();
        json!({
            "canonical_hash": hashes[0], "hashes": hashes,
            "max_similarity": similarity, "size": members.len(),
        })
    };
    json!({
        "success": true,
        "groups": [group(&[7, 6], 0.991), group(&[3, 0, 11], 0.962), group(&[9, 8], 0.957)],
        "total_memories_scanned": 5_234,
        "total_duplicates_found": 4,
    })
}

/// alaya-server's operations in its own order: (op, mutating, OIDC allowed,
/// read-only bearer allowed).
const OPS: [(&str, bool, bool, bool); 21] = [
    ("search", false, true, true),
    ("get_memory", false, true, true),
    ("check_database_health", false, true, true),
    ("memory_contradictions", false, true, true),
    ("find_duplicates", false, true, true),
    ("store_memory", false, true, false),
    ("corpus_stats", false, false, false),
    ("get_relations", false, false, true),
    ("delete_memory", true, false, false),
    ("memory_supersede", true, false, false),
    ("memory_unsupersede", true, false, false),
    ("memory_confirm", true, false, false),
    ("memory_refute", true, false, false),
    ("resolve_contradiction", true, false, false),
    ("merge_duplicates", true, false, false),
    ("relation", true, false, false),
    ("patch_memory", true, false, false),
    ("backfill_summaries", true, false, false),
    ("backfill_contradictions", true, false, false),
    ("backfill_keywords", true, false, false),
    ("graph_sweep", true, false, false),
];

fn auth_config() -> Value {
    let ops: Vec<Value> = OPS
        .iter()
        .map(|(op, mutating, oidc, readonly)| {
            json!({ "op": op, "mutating": mutating, "oidc": oidc, "readonly": readonly })
        })
        .collect();
    json!({
        "static_bearer_configured": true,
        "readonly_bearer_configured": true,
        "oidc": {
            "enabled": true,
            "issuer": "https://id.example.test/realms/operations",
            "audience": "alaya-server",
        },
        "ops": ops,
    })
}

/// The week the health fixture covers, oldest first.
const WEEK: [&str; 7] = [
    "2026-09-25",
    "2026-09-26",
    "2026-09-27",
    "2026-09-28",
    "2026-09-29",
    "2026-09-30",
    "2026-10-01",
];

/// Pairs judged each day of `WEEK`: contradiction, supersession, coexist,
/// unrelated, unjudged.
const JUDGED: [[u64; 5]; 7] = [
    [2, 41, 133, 12, 1],
    [0, 38, 120, 9, 0],
    [1, 52, 141, 17, 3],
    [0, 12, 31, 4, 0],
    [3, 47, 155, 22, 2],
    [0, 3, 148, 0, 1],
    [1, 3, 26, 0, 1],
];

/// Writes each day of `WEEK` by similarity to their nearest neighbour:
/// 0.95+, 0.85-0.95, 0.70-0.85, below 0.70, none.
const NOVELTY: [[u64; 5]; 7] = [
    [12, 88, 301, 140, 2],
    [9, 102, 287, 133, 0],
    [15, 97, 342, 121, 1],
    [4, 21, 77, 40, 0],
    [18, 131, 405, 162, 5],
    [61, 212, 433, 97, 3],
    [0, 0, 0, 0, 818],
];

/// The judge's most frequent stored failures, whose reasons run long.
const FAILURES: [(&str, u64); 3] = [
    ("unjudged: endpoint missing from vector store", 4),
    (
        "unjudged: schema violation: verdict_reason exceeded 2000 characters (got 2417) for pair 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08↔60303ae22b998861bce3b28f33eec1be758a213c86c93c076dbe9f558c11c752",
        3,
    ),
    (
        "unjudged: empty response from https://llm-gateway.example.test/v1/messages?beta=structured-outputs-2026-08-01 (stop_reason max_tokens)",
        2,
    ),
];

/// `full_stats()` with a week of judging and writes, and stored judge
/// failures whose reasons run long.
fn health_stats() -> Value {
    let mut stats = full_stats();
    let day = |date: &str, keys: [&str; 5], counts: [u64; 5]| {
        let counts: serde_json::Map<String, Value> = keys
            .iter()
            .zip(counts)
            .map(|(k, n)| (k.to_string(), json!(n)))
            .collect();
        json!({ "date": date, "counts": counts })
    };
    stats["contradictions"]["judged_per_day"] = WEEK
        .iter()
        .zip(JUDGED)
        .map(|(d, c)| {
            day(
                d,
                [
                    "contradiction",
                    "supersession",
                    "coexist",
                    "unrelated",
                    "unjudged",
                ],
                c,
            )
        })
        .collect();
    stats["writes"]["novelty"] = WEEK
        .iter()
        .zip(NOVELTY)
        .map(|(d, c)| {
            day(
                d,
                ["0.95+", "0.85-0.95", "0.70-0.85", "below_0.70", "none"],
                c,
            )
        })
        .collect();
    stats["contradictions"]["failures"]["top"] = FAILURES
        .iter()
        .map(|(reason, count)| json!({ "reason": reason, "count": count }))
        .collect();
    stats
}

// ─── anthropic-lb and its metrics store ─────────────────────────────────────

/// (client, daily limit, used today): the fleet-wide budgets, two clients
/// past 80% and one at nothing.
const BUDGETS: [(&str, u64, u64); 6] = [
    ("alaya", 20_000_000, 1_516_943),
    ("ci-review-bot", 50_000_000, 17_913_067),
    ("docs-indexer", 5_000_000, 0),
    ("nightly-evals", 2_000_000_000, 1_312_448_201),
    ("release-notes", 10_000_000, 9_204_118),
    ("support-triage", 5_000_000, 4_466_020),
];

/// `/_stats`, shaped as `routes::lb`'s live sample: one account hard-limited,
/// one warning, one passthrough, one with no window reported yet.
fn lb_stats() -> Value {
    let now = session::now_epoch() as f64;
    let usage: serde_json::Map<String, Value> = BUDGETS
        .iter()
        .map(|(client, limit, used)| (client.to_string(), json!({ "limit": limit, "used": used })))
        .collect();
    json!({
        "strategy": "sticky-weighted-v2",
        // The replica-local mirror, stale on purpose: the fleet aggregate wins.
        "client_budgets": { "ci-review-bot": { "daily_limit": 50_000_000, "used_today": 1, "remaining": 49_999_999 } },
        "cluster": {
            "redis_connected": true,
            "replicas_seen": 3,
            "budget_usage": usage,
            "transport_errors": { "connect": 12, "other": 4_103, "timeout": 706 },
        },
        "endpoints": [
            { "name": "acct-primary-01", "priority": 1, "protocol": "anthropic",
              "utilization_5h": 0.42, "utilization_7d": 0.58, "status_5h": "allowed", "status_7d": "allowed",
              "reset_5h": now + 7_400.0 },
            { "name": "acct-burst-02", "priority": 1, "protocol": "anthropic",
              "utilization_5h": 1.0, "utilization_7d": 0.97, "status_5h": "rejected", "status_7d": "allowed_warning",
              "hard_limited_remaining_secs": 1_260, "reset_5h": now + 1_260.0 },
            { "name": "acct-burst-03", "priority": 2, "protocol": "anthropic",
              "utilization_5h": 0.81, "utilization_7d": 0.88, "status_5h": "allowed_warning", "status_7d": "allowed",
              "reset_5h": now + 3_100.0 },
            { "name": "acct-shared-team-quota-with-a-deliberately-long-name-04", "priority": 2, "protocol": "anthropic",
              "utilization_5h": 0.18, "utilization_7d": 0.33, "status_5h": "allowed", "status_7d": "allowed",
              "reset_5h": now + 15_900.0 },
            { "name": "acct-passthrough", "priority": 3, "protocol": "anthropic", "passthrough": true,
              "utilization_5h": 0.05, "utilization_7d": 0.12, "status_5h": "allowed", "status_7d": "allowed",
              "reset_5h": now + 9_000.0 },
            { "name": "fallback-openai-compatible", "priority": 5, "protocol": "openai",
              "utilization_5h": null, "utilization_7d": 0.02, "status_5h": "allowed", "status_7d": null },
            { "name": "acct-new", "priority": 9, "protocol": "anthropic" },
        ],
    })
}

/// Each client's daily budget peak over the last seven UTC days, oldest
/// first; `None` is a day with no sample. legacy-importer lost its budget
/// mid-week, and release-notes has a budget but no history yet.
const BURN: [(&str, [Option<f64>; 7]); 6] = [
    (
        "alaya",
        [
            Some(1_104_220.0),
            Some(1_310_876.0),
            Some(902_311.0),
            Some(1_655_002.0),
            Some(1_208_934.0),
            Some(1_402_118.0),
            Some(1_516_943.0),
        ],
    ),
    (
        "ci-review-bot",
        [
            Some(22_030_571.0),
            Some(18_792_677.0),
            None,
            Some(25_118_004.0),
            Some(19_903_552.0),
            Some(21_340_871.0),
            Some(17_913_067.0),
        ],
    ),
    (
        "docs-indexer",
        [
            Some(0.0),
            Some(0.0),
            Some(4_912_330.0),
            Some(5_000_000.0),
            Some(4_206_119.0),
            Some(310_442.0),
            Some(0.0),
        ],
    ),
    (
        "legacy-importer",
        [
            Some(880_412.0),
            Some(912_004.0),
            Some(640_118.0),
            None,
            None,
            None,
            None,
        ],
    ),
    (
        "nightly-evals",
        [
            Some(1_620_440_118.0),
            Some(1_712_003_554.0),
            Some(1_801_227_906.0),
            Some(1_954_118_342.0),
            Some(1_403_552_871.0),
            Some(1_331_009_224.0),
            Some(1_312_448_201.0),
        ],
    ),
    (
        "support-triage",
        [
            Some(3_904_118.0),
            Some(4_112_550.0),
            Some(4_401_227.0),
            Some(4_620_014.0),
            Some(4_206_339.0),
            Some(4_512_008.0),
            Some(4_466_020.0),
        ],
    ),
];

/// The metrics store's `query_range` answer for the burn history: `BURN`'s
/// day peaks at the steps the console asks for, oldest first.
async fn burn_history(Query(q): Query<HashMap<String, String>>) -> Json<Value> {
    let num = |k: &str| q.get(k).and_then(|v| v.parse::<i64>().ok()).unwrap_or(0);
    let step = usize::try_from(num("step")).unwrap_or(1).max(1);
    let steps: Vec<i64> = (num("start")..=num("end")).step_by(step).collect();
    let result: Vec<Value> = BURN
        .iter()
        .map(|(client, days)| {
            let values: Vec<Value> = steps
                .iter()
                .zip(days)
                .filter_map(|(t, v)| v.map(|v| json!([t, v.to_string()])))
                .collect();
            json!({ "metric": { "client": client }, "values": values })
        })
        .collect();
    Json(json!({ "status": "success", "data": { "resultType": "matrix", "result": result } }))
}

// ─── The client store and its readers ───────────────────────────────────────

/// The store's restrictions: (name, state, version, actor, reason).
/// relase-notes is a mistyped disable, which the readers report as matching
/// no client.
const RECORDS: [(&str, RecordState, u64, &str, &str); 5] = [
    (
        "ci-review-bot",
        RecordState::Disabled,
        T0,
        "ops@example.test",
        "key leaked in a public CI log at https://ci.example.test/acme/agent/actions/runs/48213907/jobs/112093344#step:7:41 — rotate it before re-enabling",
    ),
    (
        "docs-indexer",
        RecordState::Enabled,
        T0 - 3 * DAY_US,
        "oncall@example.test",
        "re-enabled: the runaway crawl is capped at 5M tokens a day",
    ),
    (
        "nightly-evals",
        RecordState::Disabled,
        T0 - 2 * HOUR_US,
        "ops@example.test",
        "the eval sweep burned 60% of the weekly budget in two hours",
    ),
    (
        "relase-notes",
        RecordState::Disabled,
        T0 - 5 * DAY_US,
        "ops@example.test",
        "pause the release-notes batch until the new prompt ships",
    ),
    (
        "support-triage",
        RecordState::Enabled,
        T0 - 9 * DAY_US,
        "oncall@example.test",
        "false alarm: the spike was the launch",
    ),
];

/// How long before the current document each version in history was
/// written, oldest first. The newest six have rows; the rest are listed by
/// version only.
const HISTORY_AGES_MS: [u64; 10] = [
    40 * DAY_MS,
    31 * DAY_MS,
    20 * DAY_MS,
    12 * DAY_MS,
    9 * DAY_MS,
    6 * DAY_MS,
    3 * DAY_MS,
    DAY_MS,
    2 * HOUR_MS,
    0,
];

/// The digest of the document the store committed at `version`.
fn doc_digest(version: u64) -> String {
    hash_of(&format!("document version {version}"))
}

/// A Unix time in seconds as the store writes its `at` fields.
fn iso(secs: u64) -> String {
    time::OffsetDateTime::from_unix_timestamp(secs as i64)
        .unwrap()
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap()
}

/// The client store's world: CF caught up on both the restrictions and the
/// document, with a history to restore from.
fn store_world() -> World {
    let mut world = World::new(RECORDS.iter().map(|r| (r.0, r.1, r.2)).collect()).cf_caught_up();
    world.doc = (V0, doc_digest(V0));
    world.cf_digest = Some(doc_digest(V0));
    world.history_versions = HISTORY_AGES_MS.iter().map(|age| V0 - age).collect();
    world
}

/// A store serving `world`, with fuller rows than `World` writes: several
/// clients with their keys, a history of every op kind, and each
/// restriction's own actor and reason. An operator has a point-in-time
/// recovery window open: a warning that changes no verdict, so the client
/// pages show their warnings list.
fn client_store(world: &World) -> FakeStore {
    let mut body = world.get_body();
    body["document"]["clients"] = doc_clients();
    body["document"]["history"] = doc_history();
    body["document"]["pitr_window"] = json!({
        "open": true, "usable": true,
        "restore_to": iso((V0 - DAY_MS) / 1_000), "closes": iso((V0 + DAY_MS) / 1_000),
    });
    body["pitr_window"] = json!(true);
    body["restrictions"]["records"] = RECORDS
        .iter()
        .map(|(name, state, version, actor, reason)| {
            json!({
                "name": name, "state": state.as_str(), "version": version,
                "actor": actor, "reason": reason, "at": iso(version / 1_000_000),
            })
        })
        .collect();
    let store = FakeStore::new(world);
    *store.gets.lock().unwrap() = vec![body];
    store
}

/// A key as the document lists it: its id, label and digest, never the key.
fn doc_key(client: &str, label: &str, created_at: &str) -> Value {
    let digest = hash_of(&format!("key {client} {label}"));
    json!({ "id": format!("k{}", &digest[..7]), "label": label, "digest": digest, "created_at": created_at })
}

fn doc_clients() -> Value {
    json!([
        { "name": "alaya", "daily_budget": 20_000_000, "utilization_limit": 0.8,
          "models": ["claude-haiku-4-5", "claude-sonnet-5"], "preferred_endpoints": [],
          "keys": [doc_key("alaya", "2026-09", "2026-09-02T08:14:00Z"),
                   doc_key("alaya", "2026-10 rotation", "2026-10-11T06:00:00Z")] },
        { "name": "ci-review-bot", "daily_budget": 50_000_000, "utilization_limit": 0.9,
          "models": ["claude-*"], "preferred_endpoints": ["acct-primary-01"],
          "keys": [doc_key("ci-review-bot", "2026-10 after the leak", "2026-10-11T06:20:00Z")] },
        { "name": "docs-indexer", "daily_budget": null, "utilization_limit": null,
          "models": [], "preferred_endpoints": [],
          "keys": [doc_key("docs-indexer", "crawler", "2026-07-19T12:00:00Z")] },
        { "name": "nightly-evals", "daily_budget": 2_000_000_000_u64, "utilization_limit": 0.95,
          "models": ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5", "claude-sonnet-4-5-20250929"],
          "preferred_endpoints": ["acct-burst-02", "acct-burst-03", "acct-shared-team-quota-with-a-deliberately-long-name-04"],
          "keys": [doc_key("nightly-evals", "eval-runner-shared-key-for-the-weekly-regression-sweep-2026-q4", "2026-10-01T00:00:00Z")] },
        { "name": "support-triage", "daily_budget": 5_000_000, "utilization_limit": 0.5,
          "models": ["claude-haiku-4-5"], "preferred_endpoints": [], "keys": [] },
    ])
}

/// The newest history rows, newest first, one of each op kind.
fn doc_history() -> Value {
    let row = |age: u64, actor: &str, reason: &str, ops: Value, diff: Value| {
        let version = V0 - age;
        json!({
            "version": version, "digest": doc_digest(version), "at": iso(version / 1_000),
            "role": "writer", "actor": actor, "reason": reason, "ops": ops,
            "recovery_id": null, "diff": diff,
        })
    };
    let rotated = doc_key("alaya", "2026-10 rotation", "");
    let leaked = hash_of("key ci-review-bot 2026-09");
    json!([
        row(
            0,
            "ops@example.test",
            "rotate alaya's key: the 2026-09 key retires at the end of the month",
            json!([{ "op": "add_key", "name": "alaya" }]),
            json!([{ "client": "alaya", "change": "key added", "id": rotated["id"], "label": "2026-10 rotation", "digest": rotated["digest"] }])
        ),
        row(
            2 * HOUR_MS,
            "oncall@example.test",
            "cap nightly-evals after the sweep burned 60% of the weekly budget: https://dashboards.example.test/d/lb-budget/anthropic-lb-budget-burn?var-client=nightly-evals&from=now-7d",
            json!([{ "op": "update", "name": "nightly-evals" }]),
            json!([
                { "client": "nightly-evals", "change": "policy", "field": "daily_budget", "from": 3_000_000_000_u64, "to": 2_000_000_000_u64 },
                { "client": "nightly-evals", "change": "policy", "field": "utilization_limit", "from": 1.0, "to": 0.95 },
            ])
        ),
        row(
            DAY_MS,
            "ops@example.test",
            "revoke the key that leaked in a CI log",
            json!([{ "op": "revoke_key", "digest": leaked }]),
            json!([{ "change": "digest revoked", "digest": leaked }])
        ),
        row(
            3 * DAY_MS,
            "oncall@example.test",
            "onboard the support triage assistant",
            json!([{ "op": "create", "name": "support-triage" }]),
            json!([
                { "client": "support-triage", "change": "created" },
                { "client": "support-triage", "change": "policy", "field": "daily_budget", "from": null, "to": 5_000_000 },
            ])
        ),
        row(
            6 * DAY_MS,
            "ops@example.test",
            "legacy importer retired",
            json!([{ "op": "delete", "name": "legacy-importer" }]),
            json!([{ "client": "legacy-importer", "change": "deleted" }])
        ),
        row(
            9 * DAY_MS,
            "ops@example.test",
            "roll back the docs-indexer budget removal",
            json!([{ "op": "restore", "version": V0 - 12 * DAY_MS }]),
            json!([{ "client": "docs-indexer", "change": "policy", "field": "daily_budget", "from": null, "to": 10_000_000 }])
        ),
    ])
}

/// The readers as the metrics store shows them: three k3s pods, all up and
/// holding the served restrictions, one still on the previous document, and
/// a disabled name that matches no client.
fn fleet_metrics(world: &World) -> FakeMetrics {
    let metrics = FakeMetrics::default();
    let set = world.set_hash();
    k3s(
        &metrics,
        &[
            ("lb-a", "1", &[&set]),
            ("lb-b", "1", &[&set]),
            ("lb-c", "1", &[&set]),
        ],
    );
    let (now, before) = (doc_digest(V0), doc_digest(V0 - 2 * HOUR_MS));
    k3s_documents(
        &metrics,
        &[("lb-a", &[&now]), ("lb-b", &[&now]), ("lb-c", &[&before])],
    );
    metrics.set(
        "anthropic_client_store_unmatched_disabled_name",
        vec![series(
            &[("stack", "k3s"), ("pod", "lb-a"), ("name", "relase-notes")],
            "1",
        )],
    );
    metrics
}

// ─── selecta ────────────────────────────────────────────────────────────────

/// A selecta task id, as `uuid4().hex` renders one.
fn task_id(n: u32) -> String {
    hash_of(&format!("selecta task {n}"))[..32].to_string()
}

/// selecta's MCP answers: fuller lists than `selecta_answer`, which still
/// answers the rest.
fn selecta_world(body: &Value) -> Value {
    let tool = body
        .pointer("/params/name")
        .and_then(Value::as_str)
        .unwrap_or("");
    let args = &body["params"]["arguments"];
    match (tool, args.get("state").and_then(Value::as_str)) {
        ("selecta_list_tasks", Some("failed")) => rpc_ok(json!({ "result": failed_tasks() })),
        // `selecta_answer`'s two rows, one with no PR, behind two newer ones.
        ("selecta_list_tasks", Some("awaiting_approval")) => {
            let mut answer = selecta_answer(body);
            let newer = [
                (
                    3,
                    "selecta_curate_playlist",
                    "https://github.com/acme/selecta-playlists/pull/1048",
                ),
                (
                    4,
                    "selecta_hunt_album",
                    "https://github.com/acme/agent/pull/112",
                ),
            ]
            .map(|(n, verb, pr)| {
                let mut row = selecta_row(&task_id(n), "awaiting_approval", verb, Some(pr), None);
                row["enqueued_at"] = json!(1_791_990_000.0 - 3_600.0 * f64::from(n));
                row
            });
            let rows = &mut answer["result"]["structuredContent"]["result"];
            *rows = newer
                .into_iter()
                .chain(rows.as_array().unwrap().clone())
                .collect();
            answer
        }
        ("selecta_get_task", _) if args["task_id"] == FAILED_ID => rpc_ok(failed_task()),
        ("selecta_download_health", _) => rpc_ok(json!({
            "scope": "all", "window": "1h",
            "signals": { "disk_free_gb": 812.5, "lidarr_queue": 4.0, "qbittorrent_active": 2.0,
                         "qbittorrent_stalled": 1.0, "slskd_queue_count": 17.0, "vpn_up": 1.0 },
            "alerts": [
                "slskd unreachable: connection refused",
                "qbittorrent: 1 torrent stalled for 6h 12m at 43.7% (info hash 9f2c4b1e8a7d6c5b4a3f2e1d0c9b8a7f6e5d4c3b); last tracker error from https://tracker.example.test/announce: timed out",
            ],
        })),
        _ => selecta_answer(body),
    }
}

/// The newest failed tasks, as `selecta_list_tasks` cuts their audits.
fn failed_tasks() -> Vec<Value> {
    let row = |id: &str, verb: &str, audit: Option<&str>, enqueued_at: f64| {
        let mut row = selecta_row(id, "failed", verb, None, audit);
        row["tier"] = json!("auto");
        row["enqueued_at"] = json!(enqueued_at);
        row
    };
    vec![
        row(
            FAILED_ID,
            "selecta_discover",
            Some(
                "lidarr 503 on artist lookup: GET https://lidarr.example.test/api/v1/artist/lookup?term=lidarr%3A0383dadf-2a4e-4d10-a46a-e9e041da8eb3 failed 3 times in 19 s; nothing enqueued, the seed stays in…",
            ),
            1_791_985_400.0,
        ),
        row(
            &task_id(5),
            "selecta_hunt_album",
            Some(
                "slskd search timed out after 45 s for \"Music Has the Right to Children (1998) FLAC\"; 0 peers answered; parked for retry",
            ),
            1_791_970_000.0,
        ),
        row(
            &task_id(6),
            "selecta_curate_playlist",
            None,
            1_791_950_000.0,
        ),
        row(
            &task_id(7),
            "selecta_hunt_artist",
            Some("rate limited by MusicBrainz (HTTP 503); backing off 120 s"),
            1_791_930_000.0,
        ),
    ]
}

/// `selecta_get_task` for the first failure: its whole audit.
fn failed_task() -> Value {
    json!({
        "task_id": FAILED_ID, "state": "failed", "tier": "auto", "verb": "selecta_discover",
        "result_path": "/var/lib/selecta/results/2026/10/13/0123456789abcdef0123456789abcdef/discover-report.partial.json",
        "pr_url": null,
        "audit": "2026-10-13T21:04:11Z discover: seed artist mbid 0383dadf-2a4e-4d10-a46a-e9e041da8eb3, similar-artist depth 2\n\
                  2026-10-13T21:04:12Z lidarr GET https://lidarr.example.test/api/v1/artist/lookup?term=lidarr%3A0383dadf-2a4e-4d10-a46a-e9e041da8eb3 -> 503 Service Unavailable (attempt 1/3)\n\
                  2026-10-13T21:04:18Z lidarr GET https://lidarr.example.test/api/v1/artist/lookup?term=lidarr%3A0383dadf-2a4e-4d10-a46a-e9e041da8eb3 -> 503 Service Unavailable (attempt 2/3)\n\
                  2026-10-13T21:04:30Z lidarr GET https://lidarr.example.test/api/v1/artist/lookup?term=lidarr%3A0383dadf-2a4e-4d10-a46a-e9e041da8eb3 -> 503 Service Unavailable (attempt 3/3)\n\
                  2026-10-13T21:04:31Z giving up: lidarr 503 on artist lookup; nothing enqueued, the seed stays in the discover pool",
    })
}

/// selecta's /metrics: `SELECTA_METRICS` with the four approvals listed, more
/// task states counted and a daily cap nearly spent. Leaked because
/// `fake_selecta` serves a `&'static str`.
fn selecta_metrics() -> &'static str {
    let text = SELECTA_METRICS.replace(
        "queue_depth{state=\"awaiting_approval\"} 2",
        "queue_depth{state=\"awaiting_approval\"} 4",
    ) + "queue_depth{state=\"active\"} 1\nqueue_depth{state=\"done\"} 18342\n\
         queue_depth{state=\"aborted_stale\"} 7\n\
         day_cap_used{cap=\"searches\"} 188\nday_cap_limit{cap=\"searches\"} 200\n\
         day_cap_used{cap=\"downloads\"} 3\nday_cap_limit{cap=\"downloads\"} 25\n";
    Box::leak(text.into_boxed_str())
}
