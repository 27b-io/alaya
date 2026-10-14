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
use crate::routes::clip;

/// What the operator types to write while instances report the store
/// behind or edited, or while that cannot be told.
pub const SIGNAL_PHRASE: &str = "write anyway";

/// A typed confirmation a document write needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Needed {
    /// Instances report the store behind or edited, each named.
    Signal(Vec<String>),
    /// Whether any instance reports the store behind or edited cannot be
    /// told, and why.
    Unknown(String),
    /// A `restore` whose row could not be checked against what instances
    /// installed: why, and the version the operator types.
    Unchecked { version: u64, why: String },
}

impl Needed {
    /// The form field the confirmation is typed into.
    pub fn field(&self) -> &'static str {
        match self {
            Needed::Signal(_) | Needed::Unknown(_) => "confirm_signal",
            Needed::Unchecked { .. } => "confirm_version",
        }
    }

    /// What must be typed, exactly.
    pub fn phrase(&self) -> String {
        match self {
            Needed::Signal(_) | Needed::Unknown(_) => SIGNAL_PHRASE.to_string(),
            Needed::Unchecked { version, .. } => version.to_string(),
        }
    }

    pub fn warning(&self) -> String {
        const WHY: &str = "A write on a store that was rolled back, wiped or edited outside its \
            write path publishes that content at a newer version, which every reader installs.";
        match self {
            Needed::Signal(who) => format!(
                "Instances report the store behind or edited: {}. {WHY}",
                clip(&who.join(", "), 600)
            ),
            Needed::Unknown(why) => format!(
                "Whether any instance reports the store behind or edited cannot be told: {}. {WHY}",
                clip(why, 600)
            ),
            Needed::Unchecked { version, why } => {
                format!("Version {version} cannot be checked: {why}.")
            }
        }
    }
}

/// The confirmation the store-alert signal calls for: one when an instance
/// raises it, and one when that cannot be told. A write still goes ahead
/// once it is typed: only `import` and `restore` refuse when the metrics
/// store cannot answer.
pub fn for_signals(signals: &Result<BTreeSet<String>, String>) -> Option<Needed> {
    match signals {
        Ok(who) if who.is_empty() => None,
        Ok(who) => Some(Needed::Signal(who.iter().cloned().collect())),
        Err(why) => Some(Needed::Unknown(why.clone())),
    }
}

/// The first confirmation that was not typed exactly, surrounding spaces
/// aside. `typed` reads a form field.
pub fn missing<'a>(needed: &'a [Needed], typed: impl Fn(&str) -> &'a str) -> Option<&'a Needed> {
    needed
        .iter()
        .find(|n| typed(n.field()).trim() != n.phrase())
}

/// What a `GET` shows of the newest history row holding a version: the row
/// a `restore` of that version reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Row {
    Digest(String),
    /// Shown, with no readable digest: only an edit leaves one.
    Unreadable,
    /// Not among the rows the store shows with a digest, its newest 50.
    NotShown,
}

pub fn history_row(store: &Value, version: u64) -> Result<Row, String> {
    let rows = store
        .get("document")
        .filter(|d| d.get("status").and_then(Value::as_str) == Some("ok"))
        .ok_or("the document side of the store did not answer")?
        .get("history")
        .and_then(Value::as_array)
        .ok_or("the store's history is unreadable")?;
    let Some(row) = rows
        .iter()
        .find(|r| r.get("version").and_then(Value::as_u64) == Some(version))
    else {
        return Ok(Row::NotShown);
    };
    Ok(row
        .get("digest")
        .and_then(Value::as_str)
        .filter(|d| is_hex64(d))
        .map_or(Row::Unreadable, |d| Row::Digest(d.to_string())))
}

/// The `restore` row check's verdict on one history row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowCheck {
    /// Instances installed this version with the row's digest.
    Checked,
    /// No install of this version is on record: a typed confirmation, never
    /// a refusal.
    Unchecked(String),
    /// The row cannot pass: this version was installed with another digest,
    /// or was installed but its row cannot be compared.
    Refused(String),
}

/// Check a history row against every (version, digest) pair instances
/// installed in the last 30 days. A row too old for that record needs a
/// typed confirmation. A row whose version is on record must match it, so
/// one the console cannot compare is refused. An empty record is unknown,
/// never "no install": the metrics proxy answers a query outside its pin
/// with an empty result, so the console refuses on it.
pub fn row_check(
    version: u64,
    row: &Row,
    installed: &BTreeMap<u64, BTreeSet<String>>,
) -> Result<RowCheck, String> {
    if installed.is_empty() {
        return Err(
            "the metrics store shows no document installed in the last 30 days, which reads as \
             unknown"
                .into(),
        );
    }
    let short = |d: &str| d.chars().take(12).collect::<String>();
    Ok(match (row, installed.get(&version)) {
        (Row::Unreadable, _) => RowCheck::Refused(
            "its history row carries no readable digest, which only an edit leaves".into(),
        ),
        (Row::NotShown, Some(_)) => RowCheck::Refused(format!(
            "version {version} was installed in the last 30 days, but the store shows a digest \
             only for its 50 newest history rows, so this row cannot be checked against what was \
             installed: restore a newer version, or recover to it with point-in-time recovery"
        )),
        (Row::NotShown, None) => RowCheck::Unchecked(
            "it is older than the 30-day record of installs, and than the 50 history rows the \
             store shows a digest for"
                .into(),
        ),
        (Row::Digest(_), None) => {
            RowCheck::Unchecked("no instance installed it in the last 30 days".into())
        }
        (Row::Digest(digest), Some(seen)) => match seen.iter().find(|d| *d != digest) {
            Some(other) => RowCheck::Refused(format!(
                "version {version} was installed with digest {}…, but its history row now holds \
                 {}…: the row was edited after the fact",
                short(other),
                short(digest)
            )),
            None => RowCheck::Checked,
        },
    })
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
        let (shown_d, shown_e) = (Row::Digest(d.clone()), Row::Digest(e.clone()));
        let seen = installed(&[(9, &d), (10, &e)]);
        assert_eq!(row_check(9, &shown_d, &seen), Ok(RowCheck::Checked));
        // Installed with another digest: edited after the fact.
        let RowCheck::Refused(why) = row_check(10, &shown_d, &seen).unwrap() else {
            panic!("an edited row must be refused");
        };
        assert!(why.contains("edited after the fact"), "{why}");
        assert_eq!(row_check(10, &shown_e, &seen), Ok(RowCheck::Checked));
        // Installed with its digest and another one too: still refused.
        let both = installed(&[(9, &d), (9, &e)]);
        assert!(matches!(
            row_check(9, &shown_d, &both),
            Ok(RowCheck::Refused(_))
        ));
        // No install on record: a typed confirmation.
        assert!(matches!(
            row_check(3, &shown_d, &seen),
            Ok(RowCheck::Unchecked(_))
        ));
        assert!(matches!(
            row_check(3, &Row::NotShown, &seen),
            Ok(RowCheck::Unchecked(_))
        ));
        // On record, but the row cannot be compared: refused, never typed
        // through.
        let RowCheck::Refused(why) = row_check(9, &Row::NotShown, &seen).unwrap() else {
            panic!("a row on record that cannot be compared must be refused");
        };
        assert!(why.contains("50 newest"), "{why}");
        assert!(matches!(
            row_check(3, &Row::Unreadable, &seen),
            Ok(RowCheck::Refused(_))
        ));
        // No install on record at all is unknown, whatever the row.
        assert!(row_check(9, &shown_d, &BTreeMap::new()).is_err());
        assert!(row_check(9, &Row::NotShown, &BTreeMap::new()).is_err());
    }

    #[test]
    fn history_row_reads_the_newest_row_holding_the_version() {
        let (d, e) = ("d".repeat(64), "e".repeat(64));
        let get = json!({"document": {"status": "ok", "history": [
            {"version": 9, "digest": d}, {"version": 9, "digest": e}, {"version": 8, "digest": "short"}
        ]}});
        assert_eq!(history_row(&get, 9), Ok(Row::Digest(d)));
        assert_eq!(history_row(&get, 8), Ok(Row::Unreadable));
        assert_eq!(history_row(&get, 7), Ok(Row::NotShown));
        // A document side that did not answer is no answer, never "not shown".
        assert!(history_row(&json!({"document": {"status": "unavailable"}}), 9).is_err());
        assert!(history_row(&json!({"document": {"status": "ok"}}), 9).is_err());
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
    }

    /// A raised alert asks for the confirmation, and so does a signal that
    /// cannot be told; a quiet, readable one asks for none.
    #[test]
    fn signals_ask_for_a_confirmation_unless_known_quiet() {
        assert_eq!(for_signals(&Ok(BTreeSet::new())), None);
        assert!(matches!(
            for_signals(&Ok(["k3s/lb-a (store_edited)".to_string()].into())),
            Some(Needed::Signal(_))
        ));
        let unknown = for_signals(&Err("down".into())).unwrap();
        assert_eq!(unknown, Needed::Unknown("down".into()));
        assert_eq!(
            (unknown.field(), unknown.phrase().as_str()),
            ("confirm_signal", SIGNAL_PHRASE)
        );
    }
}
