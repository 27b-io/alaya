pub mod alaya;
pub mod auth;
pub mod home;
pub mod lb;

use crate::error::AppError;

/// Ceiling on a caller-supplied redirect target — see `safe_next`.
const MAX_NEXT_BYTES: usize = 512;

/// Validate a post-login redirect target: same-site absolute path only —
/// no scheme, no authority, no protocol-relative `//`.
///
/// An allow-list, because the deny-list it replaces was one class short. It
/// refused `\` (browsers normalize it to `/` in a Location, so `/\evil.com`
/// resolves to `//evil.com`) and passed a TAB — a legal header-value byte,
/// admitted by `http`’s own validator alongside the printable range. The
/// WHATWG URL parser removes every ASCII tab and newline from its input
/// before parsing, so `Location: /<TAB>/evil.com` became `//evil.com` and
/// the browser left this origin. CR and LF failed only by accident,
/// rejected by `HeaderValue` as a 500 rather than by this guard.
///
/// Every legitimate target is a fixed route or a 64-char hex hash, so
/// requiring ASCII-graphic bytes refuses the whole class — controls, space
/// and non-ASCII alike — rather than the members someone thought of.
///
/// The length bound is the same failure `MAX_CLAIM_BYTES` closes on the
/// session cookie, on the one field an unauthenticated caller fully
/// controls: `next` rides `GET /auth/login` straight into the login cookie,
/// and a `Set-Cookie` past the 4 KiB every browser allows per RFC 6265 §6.1
/// is dropped silently — the callback then finds no login state and answers
/// 400 "login flow expired" forever, with nothing in the pod log. Bounded
/// here rather than at any call site because this is the chokepoint all three
/// route through. 512 clears the longest real target (`/alaya/memory/` plus a
/// 64-char hash, 78 bytes) about six times over.
pub fn safe_next(next: &str) -> String {
    if next.len() <= MAX_NEXT_BYTES
        && next.starts_with('/')
        && !next.starts_with("//")
        && next.bytes().all(|b| b.is_ascii_graphic() && b != b'\\')
    {
        next.to_string()
    } else {
        "/".to_string()
    }
}

/// 64-char lowercase hex content hash — reject anything else before it
/// reaches a URL or an upstream call.
pub fn validate_hash(hash: &str) -> Result<&str, AppError> {
    if hash.len() == 64
        && hash
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
    {
        Ok(hash)
    } else {
        Err(AppError::BadRequest("invalid content hash".into()))
    }
}

/// Defensive readers over upstream JSON: a missing or mistyped field renders
/// as empty / zero instead of failing the page. Shared by both modules.
pub fn vs(v: &serde_json::Value, key: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string()
}

pub fn vf(v: &serde_json::Value, key: &str) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(0.0)
}

/// Short display prefix for a content hash.
pub fn short_hash(hash: &str) -> String {
    hash.chars().take(12).collect()
}

/// Epoch seconds → `YYYY-MM-DD HH:MM` UTC for display. Missing/zero
/// timestamps render as "—", never as a fictitious 1970 date.
pub fn fmt_epoch(secs: f64) -> String {
    if secs.is_nan() || secs <= 0.0 {
        return "—".into();
    }
    match time::OffsetDateTime::from_unix_timestamp(secs as i64) {
        Ok(t) => {
            let f = time::format_description::well_known::Rfc3339;
            t.format(&f)
                .map(|s| s[..16].replace('T', " "))
                .unwrap_or_else(|_| "-".into())
        }
        Err(_) => "-".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_next_rejects_offsite() {
        assert_eq!(safe_next("/alaya"), "/alaya");
        assert_eq!(safe_next("https://evil.com"), "/");
        assert_eq!(safe_next("//evil.com"), "/");
        // Browsers normalize backslashes in Location — these must not pass.
        assert_eq!(safe_next("/\\evil.com"), "/");
        assert_eq!(safe_next("/\\/evil.com"), "/");
        assert_eq!(safe_next("/alaya\\..\\x"), "/");
        assert_eq!(safe_next(""), "/");
        // A TAB is a legal header-value byte and the URL parser strips it
        // before parsing, so this reached the browser as `//evil.com`.
        assert_eq!(safe_next("/\t/evil.com"), "/");
        assert_eq!(safe_next("/\n/evil.com"), "/");
        assert_eq!(safe_next("/\r/evil.com"), "/");
        assert_eq!(safe_next("/ /evil.com"), "/");
        // Still lets the real targets through.
        assert_eq!(safe_next("/lb"), "/lb");
        let hash = "a".repeat(64);
        assert_eq!(
            safe_next(&format!("/alaya/memory/{hash}")),
            format!("/alaya/memory/{hash}")
        );
    }

    /// `next` is fully caller-controlled on the unauthenticated
    /// `GET /auth/login` and rides into the login cookie. Past 4 KiB the
    /// browser drops the `Set-Cookie` silently, the callback finds no login
    /// state, and every attempt answers 400 "login flow expired" — a crafted
    /// link that stops the recipient logging in, with nothing in the pod log.
    #[test]
    fn safe_next_is_bounded() {
        let at_limit = format!("/{}", "a".repeat(MAX_NEXT_BYTES - 1));
        assert_eq!(safe_next(&at_limit), at_limit);
        let over = format!("/{}", "a".repeat(MAX_NEXT_BYTES));
        assert_eq!(safe_next(&over), "/");
        // Comfortably past what a cookie can hold, which is the real target.
        assert_eq!(safe_next(&format!("/{}", "a".repeat(8192))), "/");
    }

    #[test]
    fn validate_hash_is_strict() {
        let good = "a".repeat(64);
        assert!(validate_hash(&good).is_ok());
        assert!(validate_hash(&"A".repeat(64)).is_err());
        assert!(validate_hash("abc").is_err());
        assert!(validate_hash(&format!("{}/", "a".repeat(63))).is_err());
    }

    #[test]
    fn fmt_epoch_renders_utc_minutes_and_dashes_missing() {
        assert_eq!(fmt_epoch(1788265604.26), "2026-09-01 12:26");
        assert_eq!(fmt_epoch(0.0), "—");
        assert_eq!(fmt_epoch(f64::NAN), "—");
    }
}
