//! The operator's colour theme.
//!
//! The choice lives in its own long-lived cookie, not in the session: a
//! session ends after 15 idle minutes, and a theme that reset with it would be
//! chosen again at every login. The `Session` extractor reads the cookie on
//! every request, and `ui::page` renders the choice as a class on `<html>`,
//! which the stylesheet's `dark` variant keys on (`style/input.css`). With no
//! choice saved the OS setting decides. No JavaScript is involved: the CSP
//! allows none, so the switch is a POST form like every other.

use axum_extra::extract::cookie::PrivateCookieJar;

use crate::session::{base_cookie, removal_cookie};

pub const THEME_COOKIE: &str = "console_theme";
const THEME_TTL_SECS: i64 = 365 * 24 * 3600;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Theme {
    /// Follow the OS setting.
    #[default]
    System,
    Light,
    Dark,
}

impl Theme {
    /// In the order the switch shows them.
    pub const ALL: [Theme; 3] = [Theme::System, Theme::Light, Theme::Dark];

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Theme::System),
            "light" => Some(Theme::Light),
            "dark" => Some(Theme::Dark),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Theme::System => "system",
            Theme::Light => "light",
            Theme::Dark => "dark",
        }
    }

    /// The class on `<html>`. None for `System`: the stylesheet then follows
    /// `prefers-color-scheme`.
    pub fn html_class(self) -> Option<&'static str> {
        match self {
            Theme::System => None,
            Theme::Light => Some("light"),
            Theme::Dark => Some("dark"),
        }
    }

    /// The `<meta name="color-scheme">` value. Every link in the console is a
    /// full page load, and this tells the browser which canvas to paint
    /// before the stylesheet arrives, so a dark page does not flash white.
    pub fn color_scheme(self) -> &'static str {
        match self {
            Theme::System => "light dark",
            Theme::Light => "light",
            Theme::Dark => "dark",
        }
    }
}

/// The saved theme. A missing, undecryptable or unknown cookie reads as
/// `System`: the theme is a preference, never a reason to fail a page.
pub fn read(jar: &PrivateCookieJar) -> Theme {
    jar.get(THEME_COOKIE)
        .and_then(|c| Theme::parse(c.value()))
        .unwrap_or_default()
}

/// Save `theme`. `System` removes the cookie rather than storing a value, so
/// the default is the absence of a choice.
pub fn store(jar: PrivateCookieJar, theme: Theme, secure: bool) -> PrivateCookieJar {
    match theme {
        Theme::System => jar.remove(removal_cookie(THEME_COOKIE)),
        t => jar.add(base_cookie(
            THEME_COOKIE,
            t.as_str().to_string(),
            secure,
            THEME_TTL_SECS,
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum_extra::extract::cookie::Key;

    #[test]
    fn every_theme_round_trips_and_nothing_else_parses() {
        for t in Theme::ALL {
            assert_eq!(Theme::parse(t.as_str()), Some(t));
        }
        for bad in ["", "Dark", "auto", "dark ", "solarized"] {
            assert_eq!(Theme::parse(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn a_saved_theme_reads_back_and_system_clears_it() {
        let jar = PrivateCookieJar::new(Key::from(&[7u8; 64]));
        assert_eq!(read(&jar), Theme::System);
        let jar = store(jar, Theme::Dark, true);
        assert_eq!(read(&jar), Theme::Dark);
        let jar = store(jar, Theme::System, true);
        assert_eq!(read(&jar), Theme::System);
    }
}
