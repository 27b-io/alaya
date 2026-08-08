//! Build identity — crate version, git SHA, build timestamp.
//!
//! Verifying "is build X live?" used to require cluster access or a chain of
//! inference (merge → CI push → Flux bump → host process table). Agent
//! runtimes deliberately hold no kubeconfig, so the fix is a self-describing
//! service, not wider credentials (#70).
//!
//! Populated at compile time from `ALAYA_GIT_SHA` / `ALAYA_BUILT_AT`, set as
//! Docker build args by CI. Absence is never a startup failure: a bare
//! `cargo build` reports the crate version with a null SHA and timestamp, so
//! local dev, tests and third-party builds behave identically.

use serde_json::{Map, Value, json};

/// Crate semver, e.g. `0.1.0`.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Full 40-hex git SHA this binary was built from, or `None`.
pub fn git_sha() -> Option<&'static str> {
    normalize_sha(option_env!("ALAYA_GIT_SHA"))
}

/// RFC3339 build timestamp, or `None`.
pub fn built_at() -> Option<&'static str> {
    normalize_built_at(option_env!("ALAYA_BUILT_AT"))
}

/// Version qualified with the build SHA as semver build metadata, e.g.
/// `0.1.0+2f9c…`. Falls back to the bare version when the SHA is unknown.
///
/// The full 40-hex SHA is used rather than an abbreviation so a consumer can
/// compare it to `git rev-parse HEAD` with no truncation ambiguity.
pub fn version_qualified() -> String {
    qualify(version(), git_sha())
}

/// The build-identity keys as they appear in `GET /health`. Kept here so the
/// key names have one home and the shape is assertable without live backends.
pub fn health_fields() -> Map<String, Value> {
    identity_fields(version(), git_sha(), built_at())
}

fn identity_fields(
    version: &str,
    git_sha: Option<&str>,
    built_at: Option<&str>,
) -> Map<String, Value> {
    let fields = json!({
        "version": version,
        "git_sha": git_sha,
        "built_at": built_at,
    });
    match fields {
        Value::Object(map) => map,
        _ => unreachable!("json! object literal is always an object"),
    }
}

fn qualify(version: &str, git_sha: Option<&str>) -> String {
    match git_sha {
        Some(sha) => format!("{version}+{sha}"),
        None => version.to_string(),
    }
}

/// Accept only a full 40-hex SHA. Rejects the placeholders a build can supply
/// (`""` from an unset `ARG`, `unknown`, `dev`) and short SHAs, so `git_sha`
/// is either verifiable or explicitly absent — never plausible-looking junk.
fn normalize_sha(raw: Option<&str>) -> Option<&str> {
    raw.filter(|s| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit()))
}

/// Shape check, not a full RFC3339 parse — enough to reject the empty/unset
/// case and obvious junk without pulling in a date-time dependency for a
/// string CI generates itself (`date -u +%Y-%m-%dT%H:%M:%SZ`). Ceiling: an
/// impossible-but-well-formed date like `9999-99-99T99:99:99Z` passes.
fn normalize_built_at(raw: Option<&str>) -> Option<&str> {
    raw.filter(|s| {
        let b = s.as_bytes();
        b.len() >= 20 && b[4] == b'-' && b[7] == b'-' && b[10] == b'T' && b[13] == b':'
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "2f9c1a4b6d8e0f2a4c6e8b0d2f4a6c8e0b2d4f6a";
    const TS: &str = "2026-08-09T11:22:33Z";

    #[test]
    fn health_fields_populated_with_build_info() {
        let f = identity_fields("0.1.0", Some(SHA), Some(TS));
        assert_eq!(f["version"], json!("0.1.0"));
        assert_eq!(f["git_sha"], json!(SHA));
        assert_eq!(f["built_at"], json!(TS));
    }

    /// A bare `cargo build` must still report a well-formed body: all three
    /// keys present, the unknown ones explicitly null rather than missing.
    #[test]
    fn health_fields_null_without_build_info() {
        let f = identity_fields("0.1.0", None, None);
        assert_eq!(f.len(), 3);
        assert_eq!(f["version"], json!("0.1.0"));
        assert_eq!(f["git_sha"], Value::Null);
        assert_eq!(f["built_at"], Value::Null);
    }

    #[test]
    fn live_health_fields_always_have_all_three_keys() {
        let f = health_fields();
        assert_eq!(f.len(), 3);
        assert_eq!(f["version"], json!(version()));
        assert!(f.contains_key("git_sha"));
        assert!(f.contains_key("built_at"));
    }

    #[test]
    fn qualify_appends_sha_as_build_metadata() {
        assert_eq!(qualify("0.1.0", Some(SHA)), format!("0.1.0+{SHA}"));
        assert_eq!(qualify("0.1.0", None), "0.1.0");
    }

    #[test]
    fn normalize_sha_accepts_full_hex_only() {
        assert_eq!(normalize_sha(Some(SHA)), Some(SHA));
        // Placeholders a build can legitimately supply.
        assert_eq!(normalize_sha(None), None);
        assert_eq!(normalize_sha(Some("")), None);
        assert_eq!(normalize_sha(Some("dev")), None);
        assert_eq!(normalize_sha(Some("unknown")), None);
        // Short SHA and non-hex of the right length.
        assert_eq!(normalize_sha(Some(&SHA[..7])), None);
        assert_eq!(normalize_sha(Some(&"z".repeat(40))), None);
    }

    #[test]
    fn normalize_built_at_rejects_unset_and_junk() {
        assert_eq!(normalize_built_at(Some(TS)), Some(TS));
        assert_eq!(
            normalize_built_at(Some("2026-08-09T11:22:33+10:00")),
            Some("2026-08-09T11:22:33+10:00")
        );
        assert_eq!(normalize_built_at(None), None);
        assert_eq!(normalize_built_at(Some("")), None);
        assert_eq!(normalize_built_at(Some("unknown")), None);
        assert_eq!(normalize_built_at(Some("2026-08-09")), None);
    }
}
