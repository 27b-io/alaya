pub mod alaya;
pub mod auth;
pub mod home;

use crate::error::AppError;

/// Validate a post-login redirect target: same-site absolute path only —
/// no scheme, no authority, no protocol-relative `//`, and no backslashes
/// anywhere (browsers normalize `\` → `/` in Location, so `/\evil.com`
/// would resolve to `//evil.com`).
pub fn safe_next(next: &str) -> String {
    if next.starts_with('/') && !next.starts_with("//") && !next.contains('\\') {
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
