//! The console's presentation framework: how every page is laid out and how
//! each recurring element looks, written once, so a page composes components
//! instead of hand-writing markup.
//!
//! - `components`: the atoms, vendored from the Rust/UI Leptos registry.
//! - `patterns`: the console's own components, built from the atoms: the
//!   status-to-badge mapping, notices, empty states, stats, page headers and
//!   links.
//! - `shell`: the document and the responsive shell around every page.
//!
//! Rendering is SSR-only: `view!` output is serialized with `.to_html()`.
//! Leptos escapes all text nodes and attribute values — stored memory
//! content can never inject markup (AC9).
//!
//! The CSP admits the console's one stylesheet and nothing else: no script,
//! no style element and no inline style attribute. Colour comes only from
//! the theme tokens in `style/input.css`, never from Tailwind's palette, so
//! it follows the theme. A test below holds the whole of `src/` to all of it.

mod components;
mod patterns;
mod shell;

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

    /// `class` uses a palette colour: `text-white`, `bg-gray-100`,
    /// `hover:ring-red-500/50`. Variants, an important mark, a negative and
    /// an opacity are stripped first; theme tokens (`text-foreground`,
    /// `bg-destructive/60`) are not palette colours.
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
        for bad in [
            "text-white",
            "bg-gray-100",
            "hover:ring-red-500/50",
            "dark:bg-zinc-950",
            "!text-black",
            "border-t-slate-200",
            "fill-sky-400",
            "decoration-white/40",
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

    /// The production code of every source file under `src/`: a file's
    /// trailing `mod tests` is cut off, and test-only files are skipped.
    /// Tests may hold hostile markup, to prove it comes out escaped.
    fn production_sources() -> Vec<(std::path::PathBuf, String)> {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut dirs = vec![root.clone()];
        let mut out = Vec::new();
        while let Some(dir) = dirs.pop() {
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    // `src/tests/` holds modules of main.rs's own tests.
                    if path != root.join("tests") {
                        dirs.push(path);
                    }
                    continue;
                }
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                if !name.ends_with(".rs") || name == "testkit.rs" || name == "testlog.rs" {
                    continue;
                }
                let src = std::fs::read_to_string(&path).unwrap();
                let cut = src.find("\n#[cfg(test)]\nmod tests").unwrap_or(src.len());
                out.push((path, src[..cut].to_string()));
            }
        }
        assert!(out.len() > 20, "found only {} source files", out.len());
        out
    }

    /// The CSP forbids scripts and inline style, and colour is the theme's
    /// alone: nothing in `src/` may put a script or style element, an inline
    /// style (attribute or `style:` directive) or a Tailwind palette colour
    /// on a page. A palette class even in a comment would be compiled into
    /// the stylesheet, so comments count too.
    #[test]
    fn src_has_no_script_inline_style_or_palette_colour() {
        let forbidden = [
            concat!("<scr", "ipt"),
            concat!("<sty", "le"),
            concat!("sty", "le="),
            concat!("sty", "le:"),
        ];
        let mut found = Vec::new();
        for (path, src) in production_sources() {
            for (n, line) in src.lines().enumerate() {
                let at = format!("{}:{}", path.display(), n + 1);
                for f in forbidden {
                    if line.contains(f) {
                        found.push(format!("{at}: {f}"));
                    }
                }
                let classes = line.split(|c: char| {
                    c.is_whitespace() || matches!(c, '"' | '\'' | '`' | ',' | ';' | '{' | '}')
                });
                for class in classes.filter(|c| is_palette_class(c)) {
                    found.push(format!("{at}: {class}"));
                }
            }
        }
        assert!(found.is_empty(), "{}", found.join("\n"));
    }
}
