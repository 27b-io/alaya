//! What a document write needs before the console sends it: the typed
//! confirmations that metrics-store signals call for, and the `restore` row
//! check.
//!
//! Metrics values are untrusted input. Here they can put a typed
//! confirmation in front of a document write and hold up a `restore`, and
//! nothing else. They never refuse any other write, and restriction ops never
//! come here at all.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;

use crate::effect::is_hex64;

/// What the operator types to write while instances report the store
/// behind or edited.
pub const SIGNAL_PHRASE: &str = "write anyway";

/// A typed confirmation a document write needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Needed {
    /// Instances report the store behind or edited, each named.
    Signal(Vec<String>),
    /// A `restore` whose row could not be checked against what instances
    /// installed: why, and the version the operator types.
    Unchecked { version: u64, why: String },
}

impl Needed {
    /// The form field the confirmation is typed into.
    pub fn field(&self) -> &'static str {
        match self {
            Needed::Signal(_) => "confirm_signal",
            Needed::Unchecked { .. } => "confirm_version",
        }
    }

    /// What must be typed, exactly.
    pub fn phrase(&self) -> String {
        match self {
            Needed::Signal(_) => SIGNAL_PHRASE.to_string(),
            Needed::Unchecked { version, .. } => version.to_string(),
        }
    }

    pub fn warning(&self) -> String {
        match self {
            Needed::Signal(who) => format!(
                "Instances report the store behind or edited: {}. The store was rolled back, wiped \
                 or edited outside its write path. A write now publishes its current content at a \
                 newer version, which every reader installs.",
                who.join(", ")
            ),
            Needed::Unchecked { version, why } => {
                format!("Version {version} cannot be checked: {why}.")
            }
        }
    }
}

/// The confirmation a store-alert signal calls for, if any instance raises
/// one. An unreadable signal calls for none: only `import` and `restore`
/// refuse when the metrics store cannot answer.
pub fn for_signals(signals: &Result<BTreeSet<String>, String>) -> Option<Needed> {
    match signals {
        Ok(who) if !who.is_empty() => Some(Needed::Signal(who.iter().cloned().collect())),
        _ => None,
    }
}

/// The first confirmation that was not typed exactly, surrounding spaces
/// aside. `typed` reads a form field.
pub fn missing<'a>(needed: &'a [Needed], typed: impl Fn(&str) -> &'a str) -> Option<&'a Needed> {
    needed
        .iter()
        .find(|n| typed(n.field()).trim() != n.phrase())
}

/// The `restore` row check's verdict on one history row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowCheck {
    /// Instances installed this version with the row's digest.
    Checked,
    /// No install of this version is on record, or the row carries no
    /// digest: a typed confirmation, never a refusal.
    Unchecked(String),
    /// This version was installed with another digest: the row was edited
    /// after the fact.
    Refused(String),
}

/// Check a history row against every (version, digest) pair instances
/// installed in the last 30 days. An empty record is unknown, never "no
/// install": the metrics proxy answers a query outside its pin with an empty
/// result, so the console refuses on it.
pub fn row_check(
    version: u64,
    row_digest: Option<&str>,
    installed: &BTreeMap<u64, BTreeSet<String>>,
) -> Result<RowCheck, String> {
    if installed.is_empty() {
        return Err(
            "the metrics store shows no document installed in the last 30 days, which reads as \
             unknown"
                .into(),
        );
    }
    let Some(digest) = row_digest else {
        return Ok(RowCheck::Unchecked(
            "the store shows a digest only for its 50 newest history rows".into(),
        ));
    };
    match installed.get(&version) {
        None => Ok(RowCheck::Unchecked(
            "no instance installed it in the last 30 days".into(),
        )),
        Some(seen) => match seen.iter().find(|d| *d != digest) {
            Some(other) => Ok(RowCheck::Refused(format!(
                "version {version} was installed with digest {}…, but its history row now holds \
                 {}…: the row was edited after the fact",
                &other[..12.min(other.len())],
                &digest[..12.min(digest.len())]
            ))),
            None => Ok(RowCheck::Checked),
        },
    }
}

/// The digest of the newest history row holding `version`, from a `GET`:
/// the row a `restore` of that version reads. `None` when the row is not
/// among those the store shows a digest for.
pub fn row_digest(store: &Value, version: u64) -> Option<String> {
    store
        .pointer("/document/history")
        .and_then(Value::as_array)?
        .iter()
        .find(|r| r.get("version").and_then(Value::as_u64) == Some(version))
        .and_then(|r| r.get("digest"))
        .and_then(Value::as_str)
        .filter(|d| is_hex64(d))
        .map(str::to_string)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn installed(pairs: &[(u64, &str)]) -> BTreeMap<u64, BTreeSet<String>> {
        let mut m: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
        for (v, d) in pairs {
            m.entry(*v).or_default().insert(d.to_string());
        }
        m
    }

    #[test]
    fn the_row_check_follows_what_instances_installed() {
        let (d, e) = ("d".repeat(64), "e".repeat(64));
        let seen = installed(&[(9, &d), (10, &e)]);
        assert_eq!(row_check(9, Some(&d), &seen), Ok(RowCheck::Checked));
        // Installed with another digest: edited after the fact.
        let RowCheck::Refused(why) = row_check(10, Some(&d), &seen).unwrap() else {
            panic!("an edited row must be refused");
        };
        assert!(why.contains("edited after the fact"), "{why}");
        // Installed with its digest and another one too: still refused.
        let both = installed(&[(9, &d), (9, &e)]);
        assert!(matches!(
            row_check(9, Some(&d), &both),
            Ok(RowCheck::Refused(_))
        ));
        // Too old for the record, or no digest shown: a typed confirmation.
        assert!(matches!(
            row_check(3, Some(&d), &seen),
            Ok(RowCheck::Unchecked(_))
        ));
        assert!(matches!(
            row_check(9, None, &seen),
            Ok(RowCheck::Unchecked(_))
        ));
        // No install on record at all is unknown, whatever the row.
        assert!(row_check(9, Some(&d), &BTreeMap::new()).is_err());
        assert!(row_check(9, None, &BTreeMap::new()).is_err());
    }

    #[test]
    fn row_digest_reads_the_newest_row_holding_the_version() {
        let (d, e) = ("d".repeat(64), "e".repeat(64));
        let get = json!({"document": {"history": [
            {"version": 9, "digest": d}, {"version": 9, "digest": e}, {"version": 8, "digest": "short"}
        ]}});
        assert_eq!(row_digest(&get, 9), Some(d));
        assert_eq!(row_digest(&get, 8), None);
        assert_eq!(row_digest(&get, 7), None);
        assert_eq!(row_digest(&json!({}), 9), None);
    }

    #[test]
    fn a_confirmation_is_the_exact_phrase() {
        let needed = [
            Needed::Signal(vec!["k3s/lb-a (store_edited)".into()]),
            Needed::Unchecked {
                version: 9,
                why: "x".into(),
            },
        ];
        let typed = |pairs: &'static [(&'static str, &'static str)]| {
            move |f: &str| {
                pairs
                    .iter()
                    .find(|(k, _)| *k == f)
                    .map(|(_, v)| *v)
                    .unwrap_or("")
            }
        };
        assert_eq!(
            missing(
                &needed,
                typed(&[
                    ("confirm_signal", " write anyway "),
                    ("confirm_version", "9")
                ])
            ),
            None
        );
        assert_eq!(
            missing(&needed, typed(&[("confirm_signal", "write anyway")])),
            Some(&needed[1])
        );
        assert_eq!(
            missing(
                &needed,
                typed(&[("confirm_signal", "Write Anyway"), ("confirm_version", "9")])
            ),
            Some(&needed[0])
        );
        // An unreadable signal asks for nothing; a raised one does.
        assert_eq!(for_signals(&Err("down".into())), None);
        assert_eq!(for_signals(&Ok(BTreeSet::new())), None);
        assert!(matches!(
            for_signals(&Ok(["k3s/lb-a (store_edited)".to_string()].into())),
            Some(Needed::Signal(_))
        ));
    }
}
