//! The console's presentation framework: how every page is laid out and how
//! each recurring element looks, written once, so a page composes components
//! instead of hand-writing markup.
//!
//! - `components`: the atoms, vendored from the Rust/UI Leptos registry.
//! - `patterns`: the console's own components, built from the atoms: the
//!   status-to-badge mapping, notices, empty states, stats, page headers and
//!   links.
//! - `charts`: key values as server-rendered SVG: meters, stacked bars, daily
//!   bars, sparklines and rankings.
//! - `shell`: the document and the responsive shell around every page.
//!
//! Rendering is SSR-only: `view!` output is serialized with `.to_html()`.
//! Leptos escapes all text nodes and attribute values — stored memory
//! content can never inject markup (AC9).
//!
//! The CSP admits the console's one stylesheet and nothing else: no script,
//! no style element and no inline style attribute. Colour comes only from
//! the theme tokens in `style/input.css`, never from Tailwind's palette, so
//! it follows the theme. Tests below hold the production code to the first,
//! and every file under `src/` (tests too: Tailwind reads them) and the
//! shipped stylesheet to the second.

mod charts;
mod components;
mod patterns;
mod shell;

#[cfg_attr(
    not(test),
    expect(unused_imports, reason = "no page draws a chart yet")
)]
pub use charts::*;
pub use components::*;
pub use patterns::*;
pub use shell::*;

#[cfg(test)]
mod tests {
    use leptos::prelude::*;

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

    // ─── Source hygiene ──────────────────────────────────────────────────

    /// Utilities that take a colour.
    const COLOURED: [&str; 24] = [
        "text",
        "bg",
        "border",
        "border-x",
        "border-y",
        "border-s",
        "border-e",
        "border-t",
        "border-r",
        "border-b",
        "border-l",
        "outline",
        "ring",
        "ring-offset",
        "inset-ring",
        "divide",
        "fill",
        "stroke",
        "decoration",
        "accent",
        "caret",
        "shadow",
        "inset-shadow",
        "drop-shadow",
    ];

    /// Tailwind v4's palette hues, each taking a shade (`gray-100`).
    const HUES: [&str; 26] = [
        "slate", "gray", "zinc", "neutral", "stone", "mauve", "olive", "mist", "taupe", "red",
        "orange", "amber", "yellow", "lime", "green", "emerald", "teal", "cyan", "sky", "blue",
        "indigo", "violet", "purple", "fuchsia", "pink", "rose",
    ];

    /// `class` is a colour utility naming white, black or a palette hue
    /// and shade, as the test below spells out. Variants, an important mark,
    /// a negative and an opacity are stripped first; theme tokens
    /// (`text-foreground`, `bg-destructive/60`) are not palette colours.
    fn is_palette_class(class: &str) -> bool {
        let utility = class.rsplit(':').next().unwrap_or(class);
        let utility = utility.trim_start_matches(['!', '-']);
        let utility = utility.split('/').next().unwrap_or(utility);
        let is_palette_colour = |colour: &str| {
            colour == "white"
                || colour == "black"
                || HUES.iter().any(|hue| {
                    colour
                        .strip_prefix(hue)
                        .and_then(|rest| rest.strip_prefix('-'))
                        .is_some_and(|shade| {
                            !shade.is_empty() && shade.bytes().all(|b| b.is_ascii_digit())
                        })
                })
        };
        COLOURED.iter().any(|prefix| {
            utility
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix('-'))
                .is_some_and(is_palette_colour)
        })
    }

    #[test]
    fn palette_classes_are_told_from_theme_tokens() {
        // Spelt in halves: whole, they would be palette classes in `src/`,
        // which Tailwind would compile into the stylesheet.
        for bad in [
            concat!("text-", "white"),
            concat!("bg-", "gray-100"),
            concat!("hover:ring-", "red-500/50"),
            concat!("dark:bg-", "zinc-950"),
            concat!("!text-", "black"),
            concat!("border-t-", "slate-200"),
            concat!("fill-", "sky-400"),
            concat!("decoration-", "white/40"),
        ] {
            assert!(is_palette_class(bad), "{bad}");
        }
        for ok in [
            "text-foreground",
            "bg-destructive/60",
            "text-success-dark",
            "border-border",
            "decoration-muted-foreground/60",
            "text-xs",
            "text-2xl",
            "ring-ring/50",
            "shadow-xs",
            "aria-[current=page]:bg-accent",
            "[&::-webkit-details-marker]:hidden",
            "bg-transparent",
            "border-dashed",
        ] {
            assert!(!is_palette_class(ok), "{ok}");
        }
    }

    /// Every Rust source file under `src/` as (path within `src/`, text):
    /// all that Tailwind reads (`@source "../src"`), tests included.
    fn sources() -> Vec<(std::path::PathBuf, String)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut dirs = vec![root.clone()];
        let mut out = Vec::new();
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    dirs.push(path);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    let text = std::fs::read_to_string(&path).unwrap();
                    out.push((path.strip_prefix(&root).unwrap().to_path_buf(), text));
                }
            }
        }
        assert!(out.len() > 20, "found only {} source files", out.len());
        out
    }

    /// The part of a source file that runs outside the tests: a file's
    /// trailing `mod tests` is cut off, and test-only files (`tests/`,
    /// `testkit.rs`, `testlog.rs`) are none of it. Tests may hold hostile
    /// markup, to prove it comes out escaped.
    fn production<'a>(path: &std::path::Path, text: &'a str) -> &'a str {
        let test_only = path.starts_with("tests")
            || path == std::path::Path::new("testkit.rs")
            || path == std::path::Path::new("testlog.rs");
        if test_only {
            return "";
        }
        &text[..text.find("\n#[cfg(test)]\nmod tests").unwrap_or(text.len())]
    }

    /// The CSP forbids scripts and inline style: no code that renders a page
    /// may hold a script or style element or an inline style, attribute or
    /// `style:` directive.
    #[test]
    fn production_code_has_no_script_or_inline_style() {
        let forbidden = [
            concat!("<scr", "ipt"),
            concat!("<sty", "le"),
            concat!("sty", "le="),
            concat!("sty", "le:"),
        ];
        let mut found = Vec::new();
        for (path, text) in sources() {
            for (n, line) in production(&path, &text).lines().enumerate() {
                for f in forbidden.iter().filter(|f| line.contains(**f)) {
                    found.push(format!("src/{}:{}: {f}", path.display(), n + 1));
                }
            }
        }
        assert!(found.is_empty(), "{}", found.join("\n"));
    }

    /// Colour is the theme's alone. Tailwind compiles a palette class from
    /// anywhere it reads, a comment or a test included, so no file under
    /// `src/` may hold one.
    #[test]
    fn src_holds_no_palette_colour() {
        let mut found = Vec::new();
        for (path, text) in sources() {
            for (n, line) in text.lines().enumerate() {
                let classes = line.split(|c: char| {
                    c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ',' | ';' | '{' | '}')
                });
                for class in classes.filter(|c| is_palette_class(c)) {
                    found.push(format!("src/{}:{}: {class}", path.display(), n + 1));
                }
            }
        }
        assert!(found.is_empty(), "{}", found.join("\n"));
    }

    /// What ships: the checked-in stylesheet defines no palette colour. This
    /// catches one compiled from wherever Tailwind read it; the source test
    /// above names the file.
    #[test]
    fn the_stylesheet_defines_no_palette_colour() {
        let css = include_str!("../../static/console.css");
        let leaked: Vec<&str> = HUES
            .iter()
            .copied()
            .chain(["white", "black"])
            .filter(|hue| css.contains(&format!("--color-{hue}")))
            .collect();
        assert!(
            leaked.is_empty(),
            "static/console.css defines palette colours {leaked:?}: a palette class under src/"
        );
    }
}
