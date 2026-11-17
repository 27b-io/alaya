//! What a document write needs before the console sends it: the typed
//! confirmations that metrics-store signals call for, the `restore` row
//! check, and the `import` guard.
//!
//! Metrics values are untrusted input. Here they can put a typed
//! confirmation in front of a document write and hold up a `restore` or an
//! `import`, and nothing else. They never refuse any other write, and
//! restriction ops never come here at all.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Bound;

use crate::routes::{clip, short_digest};

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
    /// installed, or an `import` of a document they never installed at or
    /// above the newest version they did: why, and the version the operator
    /// types.
    Unchecked { version: u64, why: String },
    /// An `import` of `version`, below the versions in `newer` that
    /// instances installed in the last 30 days. Overriding the guard takes
    /// typing each of them: each is a document the import discards.
    Discard { version: u64, newer: Vec<u64> },
}

impl Needed {
    /// The form field the confirmation is typed into.
    pub fn field(&self) -> &'static str {
        match self {
            Needed::Signal(_) | Needed::Unknown(_) => "confirm_signal",
            Needed::Unchecked { .. } => "confirm_version",
            Needed::Discard { .. } => "confirm_discard",
        }
    }

    /// What must be typed.
    pub fn phrase(&self) -> String {
        match self {
            Needed::Signal(_) | Needed::Unknown(_) => SIGNAL_PHRASE.to_string(),
            Needed::Unchecked { version, .. } => version.to_string(),
            Needed::Discard { newer, .. } => newer
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", "),
        }
    }

    /// Whether `typed` confirms this. A phrase is typed exactly, surrounding
    /// spaces aside. The versions an import discards are typed each once,
    /// in any order, separated by commas or spaces, and nothing else.
    pub fn satisfied(&self, typed: &str) -> bool {
        match self {
            Needed::Discard { newer, .. } => {
                let mut seen = BTreeSet::new();
                typed
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .filter(|t| !t.is_empty())
                    .all(|t| t.parse::<u64>().is_ok_and(|v| seen.insert(v)))
                    && seen == newer.iter().copied().collect()
            }
            n => typed.trim() == n.phrase(),
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
            Needed::Discard { version, newer } => format!(
                "Version {version} is below {} that instances installed in the last 30 days. \
                 Importing it discards each of them: {}. The store merges every revoked key into \
                 the import, so no revoked key comes back.",
                plural(newer.len(), "version"),
                self.phrase()
            ),
        }
    }
}

fn plural(n: usize, what: &str) -> String {
    if n == 1 {
        format!("1 {what}")
    } else {
        format!("{n} {what}s")
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

/// The first confirmation that was not typed (`Needed::satisfied`). `typed`
/// reads a form field.
pub fn missing<'a>(needed: &'a [Needed], typed: impl Fn(&str) -> &'a str) -> Option<&'a Needed> {
    needed.iter().find(|n| !n.satisfied(typed(n.field())))
}

/// The `restore` row check's verdict on one history row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RowCheck {
    /// Instances installed this version with the row's digest.
    Checked,
    /// No install of this version is on record: a typed confirmation, never
    /// a refusal.
    Unchecked(String),
    /// This version was installed with another digest: the row was edited
    /// after the fact.
    Refused(String),
}

/// Check the digest of the history row a `restore` would restore against
/// every (version, digest) pair instances installed in the last 30 days. A
/// row too old for that record needs a typed confirmation. An empty record
/// is unknown, never "no install": the metrics proxy answers a query outside
/// its pin with an empty result, so the console refuses on it.
pub fn row_check(
    version: u64,
    digest: &str,
    installed: &BTreeMap<u64, BTreeSet<String>>,
) -> Result<RowCheck, String> {
    if installed.is_empty() {
        return Err(
            "the metrics store shows no document installed in the last 30 days, which reads as \
             unknown"
                .into(),
        );
    }
    Ok(match installed.get(&version) {
        None => RowCheck::Unchecked("no instance installed it in the last 30 days".into()),
        Some(seen) => match seen.iter().find(|d| *d != digest) {
            Some(other) => RowCheck::Refused(format!(
                "version {version} was installed with digest {}, but its history row now holds \
                 {}: the row was edited after the fact",
                short_digest(other),
                short_digest(digest)
            )),
            None => RowCheck::Checked,
        },
    })
}

/// The `import` guard's verdict on the document to import.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ImportCheck {
    /// The newest document instances installed in the last 30 days: its
    /// version is the highest they installed, and they installed it with
    /// this digest.
    Newest,
    /// At or above every version instances installed in the last 30 days,
    /// but no instance installed this document there: why. A document
    /// written by hand, or a copy that is not what readers served, comes
    /// here, behind its version typed.
    Unvouched(String),
    /// Below these, which instances installed in the last 30 days.
    Below(Vec<u64>),
}

/// Check a document to import, by its version and the digest of its bytes,
/// against every (version, digest) pair instances installed in the last 30
/// days. One below any installed version needs each newer version typed. One
/// at or above them all goes as the newest only if instances installed it
/// with this digest; any other needs its version typed. An empty record is
/// unknown, never "nothing installed": the metrics proxy answers a query
/// outside its pin with an empty result, so the import is refused on it, a
/// store at version 0 included.
pub fn import_check(
    version: u64,
    digest: &str,
    installed: &BTreeMap<u64, BTreeSet<String>>,
) -> Result<ImportCheck, String> {
    let Some(&high) = installed.keys().next_back() else {
        return Err(
            "the metrics store shows no document installed in the last 30 days, which reads as \
             unknown"
                .into(),
        );
    };
    let newer: Vec<u64> = installed
        .range((Bound::Excluded(version), Bound::Unbounded))
        .map(|(v, _)| *v)
        .collect();
    if !newer.is_empty() {
        return Ok(ImportCheck::Below(newer));
    }
    Ok(match installed.get(&version) {
        Some(seen) if seen.contains(digest) => ImportCheck::Newest,
        Some(seen) => ImportCheck::Unvouched(format!(
            "instances installed version {version} with digest {}, not this document's {}, so it \
             is not the document they served",
            seen.iter()
                .map(|d| short_digest(d))
                .collect::<Vec<_>>()
                .join(", "),
            short_digest(digest)
        )),
        None => ImportCheck::Unvouched(format!(
            "it is above version {high}, the newest instances installed in the last 30 days, so no \
             instance has served it"
        )),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(row_check(9, &d, &seen), Ok(RowCheck::Checked));
        // Installed with another digest: edited after the fact.
        let RowCheck::Refused(why) = row_check(10, &d, &seen).unwrap() else {
            panic!("an edited row must be refused");
        };
        assert!(why.contains("edited after the fact"), "{why}");
        assert_eq!(row_check(10, &e, &seen), Ok(RowCheck::Checked));
        // Installed with its digest and another one too: still refused.
        let both = installed(&[(9, &d), (9, &e)]);
        assert!(matches!(row_check(9, &d, &both), Ok(RowCheck::Refused(_))));
        // No install on record: a typed confirmation.
        assert!(matches!(
            row_check(3, &d, &seen),
            Ok(RowCheck::Unchecked(_))
        ));
        // No install on record at all is unknown.
        assert!(row_check(9, &d, &BTreeMap::new()).is_err());
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

    /// The guard refuses on an empty record and asks for each newer version
    /// below the high-water. At the high-water it goes only for the document
    /// instances installed there; any other, and anything above it, needs its
    /// version typed.
    #[test]
    fn the_import_guard_follows_what_instances_installed() {
        let (d, e, f) = ("d".repeat(64), "e".repeat(64), "f".repeat(64));
        let seen = installed(&[(9, &d), (12, &e), (15, &f)]);
        assert_eq!(import_check(15, &f, &seen), Ok(ImportCheck::Newest));
        // The high-water's version with other bytes: not what readers served.
        let Ok(ImportCheck::Unvouched(why)) = import_check(15, &d, &seen) else {
            panic!("another digest at the high-water must be typed");
        };
        assert!(
            why.contains("installed version 15 with digest ffffffff")
                && why.contains("not this document's dddddddd"),
            "{why}"
        );
        // Above every installed version: no reader served it.
        for v in [16, u64::MAX] {
            let Ok(ImportCheck::Unvouched(why)) = import_check(v, &f, &seen) else {
                panic!("{v}: a document above the high-water must be typed");
            };
            assert!(why.contains("above version 15"), "{why}");
        }
        // Below it, whatever the digest: each newer version typed.
        assert_eq!(
            import_check(12, &e, &seen),
            Ok(ImportCheck::Below(vec![15]))
        );
        assert_eq!(
            import_check(1, &f, &seen),
            Ok(ImportCheck::Below(vec![9, 12, 15]))
        );
        // Installed there with this digest and another one too: the newest.
        let both = installed(&[(15, &e), (15, &f)]);
        assert_eq!(import_check(15, &f, &both), Ok(ImportCheck::Newest));
        assert!(
            import_check(1, &d, &BTreeMap::new())
                .unwrap_err()
                .contains("reads as unknown")
        );
    }

    /// The override names each discarded version once, in any order; a
    /// missing, extra, repeated or stray one does not confirm it.
    #[test]
    fn the_import_override_is_each_newer_version_typed() {
        let n = Needed::Discard {
            version: 1,
            newer: vec![9, 12, 15],
        };
        assert_eq!(n.field(), "confirm_discard");
        assert_eq!(n.phrase(), "9, 12, 15");
        assert!(n.warning().contains("below 3 versions") && n.warning().contains("9, 12, 15"));
        for ok in ["9, 12, 15", "15 9 12", " 12,15,9 ", "9,\n12\t15"] {
            assert!(n.satisfied(ok), "{ok:?}");
        }
        for bad in [
            "",
            "9, 12",
            "9, 12, 15, 16",
            "9, 12, 15, 15",
            "9, 12, 15, x",
            "all",
        ] {
            assert!(!n.satisfied(bad), "{bad:?}");
        }
        assert_eq!(missing(std::slice::from_ref(&n), |_| "15, 12, 9"), None);
        assert_eq!(missing(std::slice::from_ref(&n), |_| "15"), Some(&n));
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
