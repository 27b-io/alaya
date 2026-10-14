//! What the metrics store says about the LB instances that read the client
//! store: who is up, which restrictions set and which document each holds,
//! which hold a restriction the store lost, which report the store behind or
//! edited, which disabled names match nothing, and which ordinary clients
//! are still in a stack's TOML.
//!
//! Every query goes through its own pinned route (`MetricsClient::instant`),
//! one metric each, and instances are grouped by the `stack` label the
//! scraper puts on them, which names the stack as the store does. Every
//! series the `up` route returns is a reader, so a reader series the console
//! cannot place, with no `stack` label or no pod or instance label, leaves
//! every stack judged here unknown: dropped, it would let its stack read as
//! in effect without it.
//!
//! Metrics values are untrusted input, so nothing here may stop or delay a
//! restriction op: they feed the display and the in-effect judgement, which
//! only ever withholds a confirmation, and the warnings on document writes,
//! which only ever ask for a typed confirmation or refuse a `restore`.

use std::collections::{BTreeMap, BTreeSet};

use crate::effect::{Instance, is_hex64};
use crate::lb::{MetricsClient, Series};
use crate::routes::clip;

const UP: &str = "up";
const RESTRICTIONS: &str = "anthropic_client_store_restrictions";
const DOCUMENT: &str = "anthropic_client_store_document";
const ALERT: &str = "anthropic_client_store_alert";
const UNMATCHED: &str = "anthropic_client_store_unmatched_disabled_name";
const TOML_CLIENT: &str = "anthropic_client_store_toml_client";

/// The alerts a reader raises when the store's document is behind what it
/// holds, or differs from it at its own version: the store was rolled back,
/// wiped or edited outside its write path. A document write on top would
/// publish that content at a newer version, which every reader installs.
const STORE_ALERTS: [&str; 2] = ["store_behind", "store_edited"];

pub struct Fleet {
    /// stack → its instances, as the metrics store lists them.
    pub stacks: Result<BTreeMap<String, Vec<Instance>>, String>,
    /// Instances raising restrictions-behind, as `stack/instance`, read as
    /// `store_alerts` is.
    pub behind: Result<BTreeSet<String>, String>,
    /// Instances reporting the store behind or edited, as
    /// `stack/instance (alert)`. Placed or not: a signal is never dropped.
    /// An error when none is raised but not every instance that is up could
    /// be heard (`heard`), so "none" is never read from a broken read.
    pub store_alerts: Result<BTreeSet<String>, String>,
    /// Disabled names some instance reports matching no client it serves →
    /// the stacks reporting them. Each instance reports at most 5.
    pub unmatched: Result<BTreeMap<String, BTreeSet<String>>, String>,
    /// Ordinary clients still in some instance's TOML → the stacks whose
    /// instances report them.
    pub toml: Result<BTreeMap<String, BTreeSet<String>>, String>,
}

/// The instance a series belongs to: its pod, or its scrape instance.
fn instance_of(s: &Series) -> Option<&str> {
    s.labels
        .get("pod")
        .or_else(|| s.labels.get("instance"))
        .map(String::as_str)
}

/// The stack a series belongs to, by the store's name for it. The CF scrape
/// labels its series `stack="cloudflare"`, and the store calls that stack
/// `cf`.
fn stack_of(s: &Series) -> Option<&str> {
    match s.labels.get("stack").map(String::as_str) {
        Some("cloudflare") => Some("cf"),
        Some(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// `stack/instance` for a series, with a placeholder for what is missing,
/// for the lists of signals that name instances.
fn place(s: &Series) -> String {
    format!(
        "{}/{}",
        stack_of(s).unwrap_or("(no stack)"),
        instance_of(s).unwrap_or("(no instance)")
    )
}

pub async fn read(metrics: &MetricsClient) -> Fleet {
    let (up, sets, docs, alerts, unmatched, toml) = tokio::join!(
        metrics.instant(UP, UP),
        metrics.instant(RESTRICTIONS, RESTRICTIONS),
        metrics.instant(DOCUMENT, DOCUMENT),
        metrics.instant(ALERT, ALERT),
        metrics.instant(UNMATCHED, UNMATCHED),
        metrics.instant(TOML_CLIENT, TOML_CLIENT),
    );
    let err = |e: crate::error::AppError| e.detail().to_string();
    let (up, alerts) = (up.map_err(err), alerts.map_err(err));
    let behind = alerts.as_ref().map_err(Clone::clone).map(|a| {
        raised(a, &["restrictions_behind"])
            .map(place)
            .collect::<BTreeSet<_>>()
    });
    let store_alerts = alerts.as_ref().map_err(Clone::clone).map(|a| {
        raised(a, &STORE_ALERTS)
            .map(|s| format!("{} ({})", place(s), s.labels["alert"]))
            .collect::<BTreeSet<_>>()
    });
    let heard = match (&up, &alerts) {
        (Ok(up), Ok(alerts)) => heard(up, alerts),
        (Err(e), _) | (_, Err(e)) => Err(e.clone()),
    };
    let docs = docs.map_err(err);
    let stacks = match (&up, sets, &behind) {
        (Ok(up), Ok(sets), Ok(behind)) => instances(up, &sets, docs, behind),
        (Err(e), _, _) => Err(e.clone()),
        (_, Err(e), _) => Err(err(e)),
        (_, _, Err(e)) => Err(e.clone()),
    };
    Fleet {
        stacks,
        behind: signal(behind, &heard),
        store_alerts: signal(store_alerts, &heard),
        unmatched: unmatched.map_err(err).map(|s| by_label(&s, "name")),
        toml: toml.map_err(err).map(|s| by_label(&s, "client")),
    }
}

/// Whether the alert read shows every instance's alerts. A reader exports
/// each of its alerts, raised or not, so an instance that is up and exports
/// none is a broken read, never a quiet instance; and with no instance up,
/// there is nothing to hear.
fn heard(up: &[Series], alerts: &[Series]) -> Result<(), String> {
    let live: BTreeSet<String> = up.iter().filter(|s| s.value == 1.0).map(place).collect();
    if live.is_empty() {
        return Err("the metrics store shows no reader that is up".into());
    }
    let exporting: BTreeSet<String> = alerts.iter().map(place).collect();
    let silent: Vec<&str> = live
        .iter()
        .filter(|i| !exporting.contains(*i))
        .map(String::as_str)
        .collect();
    if silent.is_empty() {
        return Ok(());
    }
    Err(format!(
        "readers that are up export no alert series ({}), so their alerts cannot be read",
        clip(&silent.join(", "), 300)
    ))
}

/// A signal as the console may rely on it: an alert raised anywhere always
/// stands, and "none raised" stands only where every instance was heard.
fn signal(
    raised: Result<BTreeSet<String>, String>,
    heard: &Result<(), String>,
) -> Result<BTreeSet<String>, String> {
    match raised {
        Ok(r) if r.is_empty() => heard.clone().map(|()| r),
        other => other,
    }
}

/// The alert series raised now with one of `names`.
fn raised<'a>(alerts: &'a [Series], names: &'a [&str]) -> impl Iterator<Item = &'a Series> {
    alerts.iter().filter(|s| {
        s.value == 1.0
            && s.labels
                .get("alert")
                .is_some_and(|a| names.contains(&a.as_str()))
    })
}

/// `label` value → the stacks of the series carrying it. A series the
/// console cannot place in a stack is listed under its instance, so a
/// warning built on this never loses it.
fn by_label(series: &[Series], label: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for s in series {
        if let Some(value) = s.labels.get(label) {
            let stack = stack_of(s).map(str::to_string).unwrap_or_else(|| place(s));
            out.entry(value.clone()).or_default().insert(stack);
        }
    }
    out
}

/// Instances per stack. The list comes from `up`, one entry per scraped
/// instance: an instance whose gauge lingers after its scrape went away is
/// not listed. Its restrictions and document series are counted raw, every
/// series kept. A reader series the console cannot place is an error naming
/// it. `docs` failing leaves each instance's documents unknown and nothing
/// else.
fn instances(
    up: &[Series],
    sets: &[Series],
    docs: Result<Vec<Series>, String>,
    behind: &BTreeSet<String>,
) -> Result<BTreeMap<String, Vec<Instance>>, String> {
    let docs_ok = docs.as_deref().unwrap_or_default();
    let unplaced: BTreeSet<String> = up
        .iter()
        .chain(sets)
        .chain(docs_ok)
        .filter(|s| stack_of(s).is_none() || instance_of(s).is_none())
        .map(place)
        .collect();
    if !unplaced.is_empty() {
        return Err(format!(
            "reader series the console cannot place in a stack ({}), so their stack cannot be told",
            unplaced.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    let mut by_stack: BTreeMap<String, BTreeMap<String, Instance>> = BTreeMap::new();
    for s in up {
        let (Some(stack), Some(id)) = (stack_of(s), instance_of(s)) else {
            continue;
        };
        let inst = by_stack
            .entry(stack.to_string())
            .or_default()
            .entry(id.to_string())
            .or_insert_with(|| Instance {
                id: id.to_string(),
                up: true,
                set_hashes: Vec::new(),
                behind: behind.contains(&format!("{stack}/{id}")),
                doc_digests: docs.as_ref().map(|_| Vec::new()).map_err(Clone::clone),
            });
        // Two scrapes of one instance: it is up only if both are.
        inst.up &= s.value == 1.0;
    }
    for s in sets {
        let (Some(stack), Some(id)) = (stack_of(s), instance_of(s)) else {
            continue;
        };
        if let Some(inst) = by_stack.get_mut(stack).and_then(|m| m.get_mut(id)) {
            inst.set_hashes
                .push(s.labels.get("set_hash").cloned().unwrap_or_default());
        }
    }
    for s in docs_ok {
        let (Some(stack), Some(id)) = (stack_of(s), instance_of(s)) else {
            continue;
        };
        if let Some(Ok(digests)) = by_stack
            .get_mut(stack)
            .and_then(|m| m.get_mut(id))
            .map(|i| i.doc_digests.as_mut())
        {
            digests.push(s.labels.get("digest").cloned().unwrap_or_default());
        }
    }
    Ok(by_stack
        .into_iter()
        .map(|(stack, m)| (stack, m.into_values().collect()))
        .collect())
}

/// Every (version, digest) pair an instance installed in the last 30 days,
/// for the `restore` row check: version → the digests it was installed
/// with. The route answers instant queries only, so the window is a range
/// selector inside the query. An empty answer is returned empty: the caller
/// reads it as unknown, since a query outside the route's pin is empty too.
pub async fn installed(metrics: &MetricsClient) -> Result<BTreeMap<u64, BTreeSet<String>>, String> {
    let series = metrics
        .instant(
            DOCUMENT,
            &format!("count by (version, digest) (last_over_time({DOCUMENT}[30d]))"),
        )
        .await
        .map_err(|e| e.detail().to_string())?;
    let mut out: BTreeMap<u64, BTreeSet<String>> = BTreeMap::new();
    for s in &series {
        let version = s.labels.get("version").and_then(|v| v.parse::<u64>().ok());
        let digest = s.labels.get("digest").filter(|d| is_hex64(d));
        if let (Some(v), Some(d)) = (version, digest) {
            out.entry(v).or_default().insert(d.clone());
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn series(labels: &[(&str, &str)], value: f64) -> Series {
        Series {
            labels: labels
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            value,
        }
    }

    #[test]
    fn instances_come_from_up_with_every_set_series_kept() {
        let up = vec![
            series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0),
            series(&[("stack", "k3s"), ("pod", "lb-b")], 0.0),
            series(&[("stack", "cloudflare"), ("instance", "edge:443")], 1.0),
        ];
        let sets = vec![
            series(
                &[("stack", "k3s"), ("pod", "lb-a"), ("set_hash", "h1")],
                1.0,
            ),
            series(
                &[("stack", "k3s"), ("pod", "lb-a"), ("set_hash", "h0")],
                1.0,
            ),
            // A gauge whose scrape is gone: not listed.
            series(
                &[("stack", "k3s"), ("pod", "lb-gone"), ("set_hash", "h1")],
                1.0,
            ),
        ];
        let docs = vec![series(
            &[
                ("stack", "k3s"),
                ("pod", "lb-a"),
                ("version", "9"),
                ("digest", "d1"),
            ],
            1.0,
        )];
        let behind: BTreeSet<String> = ["k3s/lb-b".to_string()].into();
        let m = instances(&up, &sets, Ok(docs), &behind).unwrap();
        let k3s = &m["k3s"];
        assert_eq!(k3s.len(), 2);
        let a = k3s.iter().find(|i| i.id == "lb-a").unwrap();
        assert!(a.up && !a.behind);
        assert_eq!(a.set_hashes, ["h1", "h0"]);
        assert_eq!(a.doc_digests.as_deref().unwrap(), ["d1"]);
        let b = k3s.iter().find(|i| i.id == "lb-b").unwrap();
        assert!(!b.up && b.behind && b.set_hashes.is_empty());
        assert!(b.doc_digests.as_ref().unwrap().is_empty());
        // The CF scrape's label is mapped to the store's name for CF.
        assert_eq!(m["cf"][0].id, "edge:443");
        assert!(!m.contains_key("cloudflare"));
    }

    /// A failed document read leaves documents unknown on every instance
    /// and costs the restrictions judgement nothing.
    #[test]
    fn a_failed_document_read_touches_only_documents() {
        let up = vec![series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0)];
        let sets = vec![series(
            &[("stack", "k3s"), ("pod", "lb-a"), ("set_hash", "h1")],
            1.0,
        )];
        let m = instances(&up, &sets, Err("missing route".into()), &BTreeSet::new()).unwrap();
        let a = &m["k3s"][0];
        assert_eq!(a.set_hashes, ["h1"]);
        assert_eq!(a.doc_digests.as_ref().unwrap_err(), "missing route");
    }

    /// A reader the console cannot place in a stack must not drop out of
    /// one: it would let that stack read as in effect without it.
    #[test]
    fn a_reader_the_console_cannot_place_makes_the_stacks_unknown() {
        let up = vec![
            series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0),
            series(&[("pod", "lb-b")], 1.0),
        ];
        let sets = vec![series(
            &[("stack", "k3s"), ("pod", "lb-a"), ("set_hash", "h1")],
            1.0,
        )];
        let err = instances(&up, &sets, Ok(vec![]), &BTreeSet::new()).unwrap_err();
        assert!(err.contains("lb-b"), "{err}");
        // An unlabeled restrictions series counts the same.
        let up = vec![series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0)];
        let sets = vec![series(&[("pod", "lb-c"), ("set_hash", "h1")], 1.0)];
        assert!(
            instances(&up, &sets, Ok(vec![]), &BTreeSet::new())
                .unwrap_err()
                .contains("lb-c")
        );
        // So does a series with a stack but no pod or instance label, and a
        // document series.
        let up = vec![
            series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0),
            series(&[("stack", "k3s")], 1.0),
        ];
        let err = instances(&up, &[], Ok(vec![]), &BTreeSet::new()).unwrap_err();
        assert!(err.contains("k3s/(no instance)"), "{err}");
        let up = vec![series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0)];
        let docs = vec![series(&[("pod", "lb-d"), ("digest", "d1")], 1.0)];
        assert!(
            instances(&up, &[], Ok(docs), &BTreeSet::new())
                .unwrap_err()
                .contains("lb-d")
        );
    }

    /// Every reader exports all its alerts, so "none raised" is believed
    /// only when every instance that is up was heard; a raised alert stands
    /// whatever else is silent.
    #[test]
    fn a_silent_reader_leaves_the_signals_unknown_but_never_hides_a_raised_one() {
        let up = vec![
            series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0),
            series(&[("stack", "k3s"), ("pod", "lb-b")], 1.0),
            series(&[("stack", "k3s"), ("pod", "lb-down")], 0.0),
        ];
        let quiet = |pod: &str| {
            series(
                &[("stack", "k3s"), ("pod", pod), ("alert", "store_edited")],
                0.0,
            )
        };
        assert_eq!(heard(&up, &[quiet("lb-a"), quiet("lb-b")]), Ok(()));
        let err = heard(&up, &[quiet("lb-a")]).unwrap_err();
        assert!(
            err.contains("k3s/lb-b") && !err.contains("lb-down"),
            "{err}"
        );
        assert!(heard(&[], &[quiet("lb-a")]).is_err());
        assert!(heard(&up[2..], &[]).is_err());

        let none = Ok(BTreeSet::new());
        let some: Result<BTreeSet<String>, String> =
            Ok(["k3s/lb-a (store_edited)".to_string()].into());
        assert_eq!(signal(none.clone(), &Ok(())), Ok(BTreeSet::new()));
        assert!(signal(none, &Err("silent".into())).is_err());
        assert_eq!(signal(some.clone(), &Err("silent".into())), some);
        assert!(signal(Err("down".into()), &Ok(())).is_err());
    }

    #[test]
    fn labels_group_by_stack_and_an_unplaced_series_keeps_its_instance() {
        let toml = vec![
            series(
                &[("stack", "k3s"), ("pod", "lb-a"), ("client", "acme")],
                1.0,
            ),
            series(
                &[
                    ("stack", "cloudflare"),
                    ("instance", "edge:443"),
                    ("client", "acme"),
                ],
                1.0,
            ),
            series(&[("pod", "lb-x"), ("client", "beta")], 1.0),
        ];
        let m = by_label(&toml, "client");
        assert_eq!(
            m["acme"].iter().map(String::as_str).collect::<Vec<_>>(),
            ["cf", "k3s"]
        );
        assert_eq!(
            m["beta"].iter().map(String::as_str).collect::<Vec<_>>(),
            ["(no stack)/lb-x"]
        );
    }
}
