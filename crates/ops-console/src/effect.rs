//! Whether a client-store write is in effect on each stack. Every page that
//! says "in effect" asks this module, and nothing else decides it.
//!
//! A restriction op (`disable` or `enable`) is in effect on an instance once
//! the instance reports the set hash of the restrictions part the store
//! serves now, and that part carries a record of the op's name in the op's
//! state, at the op's version or newer. Both halves are read from one `GET`,
//! and each op's version is the one the write's own result gave it.
//!
//! No version is ever compared between an instance and the store. A record's
//! version is the store's clock, so after that clock steps back a new op is
//! stamped below versions instances already hold, and "the instance's
//! highest version is at least the op's" would read as delivered before any
//! instance has the op. Equal set hashes cannot be faked that way: they mean
//! the instance holds exactly the records and digests the store serves.
//! Versions are compared only between two records of one name, which is how
//! the console tells that a later op superseded this one.
//!
//! "In effect" means no new admission, so every doubt reads as not in
//! effect, or as unknown when the console cannot see a stack at all.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecordState {
    Disabled,
    Enabled,
}

impl RecordState {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "disabled" => Some(RecordState::Disabled),
            "enabled" => Some(RecordState::Enabled),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RecordState::Disabled => "disabled",
            RecordState::Enabled => "enabled",
        }
    }
}

/// A restriction op as the write's own result reported it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Op {
    pub name: String,
    pub state: RecordState,
    /// The record version the store stamped, in microseconds.
    pub version: u64,
}

/// The restrictions part the store serves to polls now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServedPart {
    /// Changes whenever the served bytes change; the console re-reads it after
    /// reading the metrics store, and confirms nothing if it moved.
    pub tag: String,
    /// The set hash an instance reports once it holds exactly this part's
    /// records and digests. Checked here against the body it summarises.
    pub set_hash: String,
    /// False when the store's integrity record no longer matches its table:
    /// rows were lost, and nothing reads as in effect until a reseed.
    pub full: bool,
    /// name → (state, version): the records in the part's body. The table
    /// can hold rows the part leaves out, so the judgement never reads it.
    pub records: BTreeMap<String, (RecordState, u64)>,
}

#[derive(Deserialize)]
struct PartBody {
    full: bool,
    records: Vec<BodyRecord>,
    revoked_digests: Vec<String>,
}

#[derive(Deserialize)]
struct BodyRecord {
    name: String,
    state: String,
    version: u64,
}

/// A client name: `[a-z0-9]`, then `[a-z0-9_-]`, at most 63 characters.
pub fn is_name(s: &str) -> bool {
    let b = s.as_bytes();
    !b.is_empty()
        && b.len() <= 63
        && (b[0].is_ascii_lowercase() || b[0].is_ascii_digit())
        && b.iter()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_' || *c == b'-')
}

/// 64 lowercase hex characters: every digest, tag and set hash the store sends.
pub fn is_hex64(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}

/// The set hash over `records` and `digests`: one `r <name> <state> <version>`
/// line per record in name order, then one `d <digest>` line per digest in
/// ascending order, each ending in a newline, hashed with SHA-256. Readers
/// and the store compute the same text, so equal hashes mean equal sets.
pub fn set_hash<'a>(
    records: impl IntoIterator<Item = (&'a str, RecordState, u64)>,
    digests: impl IntoIterator<Item = &'a str>,
) -> String {
    let records: BTreeMap<&str, (RecordState, u64)> =
        records.into_iter().map(|(n, s, v)| (n, (s, v))).collect();
    let digests: BTreeSet<&str> = digests.into_iter().collect();
    let mut text = String::new();
    for (name, (state, version)) in &records {
        text.push_str(&format!("r {name} {} {version}\n", state.as_str()));
    }
    for d in &digests {
        text.push_str(&format!("d {d}\n"));
    }
    hex(&Sha256::digest(text.as_bytes()))
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Read `restrictions.part` from a `GET`. Anything the console cannot read
/// exactly is an error, and an unreadable part confirms nothing: the set hash
/// must match the body the console parsed, so a part whose body and hash
/// disagree can never make an op read as delivered.
pub fn parse_part(part: Option<&Value>) -> Result<ServedPart, String> {
    let Some(part) = part else {
        return Err("the store's answer carried no restrictions part".into());
    };
    if part.get("status").and_then(Value::as_str) != Some("ok") {
        return Err("the store could not build the part it serves (unavailable)".into());
    }
    let field = |k: &str| {
        part.get(k)
            .and_then(Value::as_str)
            .filter(|s| is_hex64(s))
            .map(str::to_string)
            .ok_or_else(|| format!("the part's {k} is missing or malformed"))
    };
    let (tag, served_hash) = (field("tag")?, field("set_hash")?);
    let body: PartBody = part
        .get("body")
        .and_then(Value::as_str)
        .and_then(|b| serde_json::from_str(b).ok())
        .ok_or("the part's body could not be read")?;
    let mut records = BTreeMap::new();
    for r in &body.records {
        let state = RecordState::parse(&r.state)
            .filter(|_| is_name(&r.name))
            .ok_or_else(|| {
                format!(
                    "the part holds a malformed record ({:?})",
                    crate::routes::clip(&r.name, 64)
                )
            })?;
        if records.insert(r.name.clone(), (state, r.version)).is_some() {
            return Err(format!("the part holds two records of {}", r.name));
        }
    }
    if !body.revoked_digests.iter().all(|d| is_hex64(d)) {
        return Err("the part holds a malformed digest".into());
    }
    let computed = set_hash(
        records.iter().map(|(n, (s, v))| (n.as_str(), *s, *v)),
        body.revoked_digests.iter().map(String::as_str),
    );
    if computed != served_hash {
        return Err("the part's set hash does not match its body".into());
    }
    Ok(ServedPart {
        tag,
        set_hash: served_hash,
        full: body.full,
        records,
    })
}

/// Why nothing reads as in effect while the part is partial.
pub const PARTIAL: &str = "the restrictions part is partial: the store lost rows or its integrity \
    record, so nothing reads as in effect until restrictions_import reseeds it";

/// Why a confirmation is withheld when the served part moved during a read.
pub const MOVED: &str = "the served part changed, or could not be re-read, while the console was \
    reading; re-checking";

/// CF's state, from the store's record of CF's latest poll.
#[derive(Clone, Debug)]
pub struct Report {
    /// The report was refreshed in the last 15 s.
    pub running: bool,
    /// The hash of the restrictions set CF holds.
    pub set_hash: Option<String>,
}

/// One instance as the metrics store shows it.
#[derive(Clone, Debug, Default)]
pub struct Instance {
    /// The pod, or the scrape instance where there is no pod.
    pub id: String,
    /// Its scrape's `up`.
    pub up: bool,
    /// The `set_hash` label of every restrictions series it exports now,
    /// counted raw. A new set opens a new series while the old one can linger,
    /// so anything but exactly one series leaves its current set unknown.
    pub set_hashes: Vec<String>,
    /// Its restrictions-behind alert: it holds a restriction the store lost.
    pub behind: bool,
}

/// What the console can see of one stack.
#[derive(Clone, Debug)]
pub enum Sight {
    /// CF, through the store's report. `None` when the document side of the
    /// store did not answer, so no report could be read.
    Report(Option<Report>),
    /// A stack seen through the metrics store: its instances, or why they
    /// could not be read.
    Instances(Result<Vec<Instance>, String>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    InEffect,
    NotInEffect(String),
    /// The console cannot see the stack. Never shown as in effect.
    Unknown(String),
}

/// Whether the served part carries an op: the content half of the rule.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Carried {
    Yes,
    /// A newer record of the name in the other state: the op is history.
    Superseded(u64),
    No(String),
}

pub fn carried(part: &ServedPart, op: &Op) -> Carried {
    let Some(&(state, version)) = part.records.get(&op.name) else {
        return Carried::No(format!("the served part holds no record of {}", op.name));
    };
    if state == op.state {
        if version >= op.version {
            Carried::Yes
        } else {
            Carried::No(format!(
                "the served record of {} is {} at {version}, older than this op",
                op.name,
                state.as_str()
            ))
        }
    } else if version > op.version {
        Carried::Superseded(version)
    } else {
        Carried::No(format!(
            "the served part still holds {}'s {} record at {version}",
            op.name,
            state.as_str()
        ))
    }
}

/// Whether a stack holds the served part: the instance half of the rule,
/// which is all there is to say about every record the part carries.
pub fn holds(part: &ServedPart, sight: &Sight) -> Verdict {
    use Verdict::*;
    if !part.full {
        return NotInEffect(PARTIAL.into());
    }
    match sight {
        Sight::Report(None) => Unknown(
            "the document side of the store did not answer, so CF's report could not be read"
                .into(),
        ),
        Sight::Report(Some(r)) if !r.running => NotInEffect(
            "CF has not reported in the last 15 s: asleep, restarting or cut off from the store"
                .into(),
        ),
        Sight::Report(Some(r)) => match &r.set_hash {
            Some(h) if *h == part.set_hash => InEffect,
            Some(_) => NotInEffect("CF holds a different set".into()),
            None => NotInEffect("CF reports no set".into()),
        },
        Sight::Instances(Err(e)) => Unknown(format!("the metrics store could not be read: {e}")),
        Sight::Instances(Ok(list)) if list.is_empty() => {
            Unknown("no instance of this stack is visible in the metrics store".into())
        }
        Sight::Instances(Ok(list)) => {
            let faults: Vec<String> = list
                .iter()
                .filter_map(|i| instance_fault(i, &part.set_hash))
                .collect();
            if faults.is_empty() {
                InEffect
            } else {
                NotInEffect(faults.join("; "))
            }
        }
    }
}

fn instance_fault(i: &Instance, set_hash: &str) -> Option<String> {
    if !i.up {
        return Some(format!(
            "{} is not scraped (up is 0): draining, restarting or unreachable",
            i.id
        ));
    }
    if i.behind {
        return Some(format!(
            "{} holds a restriction the store lost (restrictions behind): confirmed only after a reseed",
            i.id
        ));
    }
    match i.set_hashes.as_slice() {
        [] => Some(format!("{} exports no client-store gauge", i.id)),
        [h] if h == set_hash => None,
        [_] => Some(format!("{} holds a different set", i.id)),
        many => Some(format!(
            "{} exports {} restrictions series, so its current set is unknown",
            i.id,
            many.len()
        )),
    }
}

/// Withhold "in effect" when the served part moved between the two `GET`s
/// that bracket the metrics read: the instances were then compared with a
/// part the store no longer serves.
pub fn steady(v: Verdict, tag_stable: bool) -> Verdict {
    match v {
        Verdict::InEffect if !tag_stable => Verdict::Unknown(MOVED.into()),
        v => v,
    }
}

/// One op's judgement on every stack that counts.
pub struct Judged {
    pub carried: Carried,
    pub stacks: Vec<(String, Verdict)>,
}

pub fn judge_op(
    op: &Op,
    part: &ServedPart,
    sights: &[(String, Sight)],
    tag_stable: bool,
) -> Judged {
    let carried = carried(part, op);
    let stacks = sights
        .iter()
        .map(|(stack, sight)| {
            let v = match &carried {
                Carried::Yes => steady(holds(part, sight), tag_stable),
                Carried::Superseded(v) => Verdict::NotInEffect(format!("superseded at {v}")),
                Carried::No(why) => Verdict::NotInEffect(why.clone()),
            };
            (stack.clone(), v)
        })
        .collect();
    Judged { carried, stacks }
}

/// Records are pruned from the store 31 days after they are written, so an
/// `enable` older than that can never be confirmed: the view drops it rather
/// than show it failing.
pub const PRUNE_AFTER_MICROS: u64 = 31 * 86_400 * 1_000_000;

pub fn aged_out(op: &Op, now_micros: u64) -> bool {
    op.state == RecordState::Enabled && now_micros.saturating_sub(op.version) > PRUNE_AFTER_MICROS
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const T0: u64 = 1_791_698_400_000_000;

    /// The store's own vectors: the empty set, and one disabled record.
    #[test]
    fn set_hash_matches_the_stores_vectors() {
        assert_eq!(
            set_hash([], []),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            set_hash([("acme", RecordState::Disabled, T0)], []),
            "a47a3d8ef66565ff1f83d16a032e3e674ecd6c4168d2f0493ae8e001b785c4da"
        );
        // Order of input never matters: the text is sorted.
        let d1 = "a".repeat(64);
        let d2 = "b".repeat(64);
        assert_eq!(
            set_hash(
                [
                    ("b", RecordState::Enabled, 2),
                    ("a", RecordState::Disabled, 1)
                ],
                [d2.as_str(), d1.as_str()]
            ),
            set_hash(
                [
                    ("a", RecordState::Disabled, 1),
                    ("b", RecordState::Enabled, 2)
                ],
                [d1.as_str(), d2.as_str()]
            )
        );
    }

    /// A part as the store serves it, with a correct set hash and tag.
    fn served(full: bool, records: &[(&str, RecordState, u64)], digests: &[&str]) -> Value {
        let body = json!({
            "full": full,
            "enabled_ahead": 0,
            "records": records.iter().map(|(n, s, v)| json!({"name": n, "state": s.as_str(), "version": v})).collect::<Vec<_>>(),
            "revoked_digests": digests,
        })
        .to_string();
        json!({
            "status": "ok",
            "tag": hex(&Sha256::digest(body.as_bytes())),
            "set_hash": set_hash(records.iter().copied(), digests.iter().copied()),
            "body": body,
        })
    }

    fn part(records: &[(&str, RecordState, u64)]) -> ServedPart {
        parse_part(Some(&served(true, records, &[]))).unwrap()
    }

    use RecordState::{Disabled, Enabled};

    #[test]
    fn parse_part_reads_the_body_and_checks_its_hash() {
        let p = parse_part(Some(&served(true, &[("acme", Disabled, T0)], &[]))).unwrap();
        assert!(p.full);
        assert_eq!(p.records["acme"], (Disabled, T0));
        assert_eq!(
            p.set_hash,
            "a47a3d8ef66565ff1f83d16a032e3e674ecd6c4168d2f0493ae8e001b785c4da"
        );

        // A hash that does not summarise the body confirms nothing.
        let mut lying = served(true, &[("acme", Disabled, T0)], &[]);
        lying["set_hash"] = json!(set_hash([], []));
        assert_eq!(
            parse_part(Some(&lying)).unwrap_err(),
            "the part's set hash does not match its body"
        );
        // Unavailable, missing, malformed: all unreadable.
        assert!(parse_part(Some(&json!({"status": "unavailable"}))).is_err());
        assert!(parse_part(None).is_err());
        let mut bad = served(true, &[], &[]);
        bad["body"] = json!("{not json");
        assert!(parse_part(Some(&bad)).is_err());
        let mut bad = served(true, &[], &[]);
        bad["tag"] = json!("short");
        assert!(parse_part(Some(&bad)).is_err());
        // A record the console cannot place in the set text.
        let mut bad = served(true, &[], &[]);
        bad["body"] = json!(
            json!({"full": true, "enabled_ahead": 0, "records": [{"name": "a b", "state": "disabled", "version": 1}], "revoked_digests": []}).to_string()
        );
        assert!(
            parse_part(Some(&bad))
                .unwrap_err()
                .contains("malformed record")
        );
        // A version that is not a non-negative integer.
        let mut bad = served(true, &[], &[]);
        bad["body"] = json!(
            json!({"full": true, "enabled_ahead": 0, "records": [{"name": "a", "state": "disabled", "version": 1.5}], "revoked_digests": []}).to_string()
        );
        assert!(parse_part(Some(&bad)).is_err());
    }

    fn op(name: &str, state: RecordState, version: u64) -> Op {
        Op {
            name: name.into(),
            state,
            version,
        }
    }

    /// Both halves of the content rule: a record in the op's state at its
    /// version or newer carries it; the other state newer supersedes it; the
    /// op's state older than the op does not carry it.
    #[test]
    fn carried_pins_both_halves_of_the_content_rule() {
        let p = part(&[("acme", Disabled, T0 + 10)]);
        assert_eq!(carried(&p, &op("acme", Disabled, T0 + 10)), Carried::Yes);
        assert_eq!(carried(&p, &op("acme", Disabled, T0 + 5)), Carried::Yes);
        // The served record is in the op's state but older than the op: the
        // part does not carry it (the table lost it, or the store's clock
        // stepped and the record served is the earlier one).
        assert!(matches!(
            carried(&p, &op("acme", Disabled, T0 + 20)),
            Carried::No(_)
        ));
        // A newer record in the other state superseded the op.
        let p = part(&[("acme", Enabled, T0 + 30)]);
        assert_eq!(
            carried(&p, &op("acme", Disabled, T0 + 10)),
            Carried::Superseded(T0 + 30)
        );
        // The other state, but older: the part does not carry the op yet.
        assert!(matches!(
            carried(&p, &op("acme", Disabled, T0 + 40)),
            Carried::No(_)
        ));
        // No record of the name at all.
        assert!(matches!(
            carried(&p, &op("other", Disabled, T0)),
            Carried::No(_)
        ));
    }

    fn running(set_hash: &str) -> Sight {
        Sight::Report(Some(Report {
            running: true,
            set_hash: Some(set_hash.into()),
        }))
    }

    fn pod(id: &str, up: bool, hashes: &[&str]) -> Instance {
        Instance {
            id: id.into(),
            up,
            set_hashes: hashes.iter().map(|h| h.to_string()).collect(),
            behind: false,
        }
    }

    #[test]
    fn cf_is_in_effect_only_running_and_on_the_served_set() {
        let p = part(&[("acme", Disabled, T0)]);
        assert_eq!(holds(&p, &running(&p.set_hash)), Verdict::InEffect);
        assert!(matches!(
            holds(&p, &running(&set_hash([], []))),
            Verdict::NotInEffect(_)
        ));
        let stale = Sight::Report(Some(Report {
            running: false,
            set_hash: Some(p.set_hash.clone()),
        }));
        assert!(matches!(holds(&p, &stale), Verdict::NotInEffect(_)));
        let none = Sight::Report(Some(Report {
            running: true,
            set_hash: None,
        }));
        assert!(matches!(holds(&p, &none), Verdict::NotInEffect(_)));
        assert!(matches!(
            holds(&p, &Sight::Report(None)),
            Verdict::Unknown(_)
        ));
    }

    /// k3s: every listed pod up, with exactly one restrictions series on the
    /// served set; `up` at 0, a missing gauge, a duplicate series or a
    /// different hash each make the stack not in effect; no pod at all, or
    /// an unreadable metrics store, makes it unknown.
    #[test]
    fn instances_are_judged_raw_one_series_per_pod() {
        let p = part(&[("acme", Disabled, T0)]);
        let h = p.set_hash.as_str();
        let other = set_hash([], []);
        let ok = |list: Vec<Instance>| holds(&p, &Sight::Instances(Ok(list)));
        assert_eq!(
            ok(vec![pod("a", true, &[h]), pod("b", true, &[h])]),
            Verdict::InEffect
        );
        let v = ok(vec![pod("a", true, &[h]), pod("b", false, &[h])]);
        assert!(
            matches!(&v, Verdict::NotInEffect(r) if r.contains("b is not scraped")),
            "{v:?}"
        );
        let v = ok(vec![pod("a", true, &[h]), pod("b", true, &[])]);
        assert!(
            matches!(&v, Verdict::NotInEffect(r) if r.contains("no client-store gauge")),
            "{v:?}"
        );
        // A lingering second series: never collapsed through max or
        // last_over_time, even when one of them is the served hash.
        let v = ok(vec![pod("a", true, &[h, &other])]);
        assert!(
            matches!(&v, Verdict::NotInEffect(r) if r.contains("2 restrictions series")),
            "{v:?}"
        );
        let v = ok(vec![pod("a", true, &[&other])]);
        assert!(
            matches!(&v, Verdict::NotInEffect(r) if r.contains("different set")),
            "{v:?}"
        );
        let mut behind = pod("a", true, &[h]);
        behind.behind = true;
        let v = ok(vec![behind]);
        assert!(
            matches!(&v, Verdict::NotInEffect(r) if r.contains("restrictions behind")),
            "{v:?}"
        );
        assert!(matches!(ok(vec![]), Verdict::Unknown(_)));
        assert!(matches!(
            holds(&p, &Sight::Instances(Err("timeout".into()))),
            Verdict::Unknown(_)
        ));
    }

    #[test]
    fn a_partial_part_confirms_nothing_even_on_equal_hashes() {
        let raw = served(false, &[("acme", Disabled, T0)], &[]);
        let p = parse_part(Some(&raw)).unwrap();
        assert!(!p.full);
        assert_eq!(
            holds(&p, &running(&p.set_hash)),
            Verdict::NotInEffect(PARTIAL.into())
        );
        let pods = Sight::Instances(Ok(vec![pod("a", true, &[&p.set_hash])]));
        assert_eq!(holds(&p, &pods), Verdict::NotInEffect(PARTIAL.into()));
    }

    #[test]
    fn a_moved_part_withholds_only_the_confirmation() {
        assert_eq!(steady(Verdict::InEffect, true), Verdict::InEffect);
        assert_eq!(
            steady(Verdict::InEffect, false),
            Verdict::Unknown(MOVED.into())
        );
        let no = Verdict::NotInEffect("x".into());
        assert_eq!(steady(no.clone(), false), no);
    }

    /// The clock step-back the set hash exists for: readers hold a record at
    /// a high version, the store's clock steps back, and a new disable is
    /// stamped below it. A version comparison would call the new op
    /// delivered; the set hash says no until the readers hold the new part.
    #[test]
    fn a_clock_step_back_cannot_fake_delivery() {
        let held = part(&[
            ("acme", Disabled, T0 + 90_000_000),
            ("beta", Enabled, T0 + 90_000_000),
        ]);
        // After the step: the new disable of beta lands below what readers hold.
        let now = part(&[
            ("acme", Disabled, T0 + 90_000_000),
            ("beta", Disabled, T0 + 30_000_000),
        ]);
        let new = op("beta", Disabled, T0 + 30_000_000);
        let readers = vec![
            ("cf".to_string(), running(&held.set_hash)),
            (
                "k3s".to_string(),
                Sight::Instances(Ok(vec![pod("a", true, &[&held.set_hash])])),
            ),
        ];
        let j = judge_op(&new, &now, &readers, true);
        assert_eq!(j.carried, Carried::Yes);
        assert!(
            j.stacks
                .iter()
                .all(|(_, v)| matches!(v, Verdict::NotInEffect(_)))
        );
        // Once every reader holds the new part, it is in effect.
        let caught_up = vec![
            ("cf".to_string(), running(&now.set_hash)),
            (
                "k3s".to_string(),
                Sight::Instances(Ok(vec![pod("a", true, &[&now.set_hash])])),
            ),
        ];
        let j = judge_op(&new, &now, &caught_up, true);
        assert!(j.stacks.iter().all(|(_, v)| *v == Verdict::InEffect));
    }

    #[test]
    fn a_superseded_op_never_reads_as_in_effect() {
        let p = part(&[("acme", Enabled, T0 + 50)]);
        let readers = vec![("cf".to_string(), running(&p.set_hash))];
        let j = judge_op(&op("acme", Disabled, T0), &p, &readers, true);
        assert_eq!(j.carried, Carried::Superseded(T0 + 50));
        assert!(matches!(&j.stacks[0].1, Verdict::NotInEffect(r) if r.contains("superseded")));
    }

    #[test]
    fn enables_age_out_of_the_view_after_the_prune_window() {
        let now = T0 + PRUNE_AFTER_MICROS + 1;
        assert!(aged_out(&op("a", Enabled, T0), now));
        assert!(!aged_out(&op("a", Enabled, T0 + 1), now));
        // A disable is never pruned while it is the name's latest record.
        assert!(!aged_out(&op("a", Disabled, T0), now));
    }

    #[test]
    fn names_and_hex_follow_the_stores_rules() {
        assert!(is_name("acme") && is_name("a") && is_name("0-a_b"));
        assert!(is_name(&"a".repeat(63)) && !is_name(&"a".repeat(64)));
        assert!(
            !is_name("") && !is_name("_x") && !is_name("-x") && !is_name("Acme") && !is_name("a b")
        );
        assert!(is_hex64(&"0f".repeat(32)) && !is_hex64(&"0F".repeat(32)) && !is_hex64("ab"));
    }
}
