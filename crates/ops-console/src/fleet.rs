//! What the metrics store says about the LB instances that read the client
//! store: who is up, which restrictions set each holds, which hold a
//! restriction the store lost, and which disabled names match nothing.
//!
//! Every query goes through its own pinned route (`MetricsClient::instant`),
//! one metric each, and instances are grouped by the `stack` label the
//! scraper puts on them, which names the stack as the store does. Every
//! series the `up` route returns is a reader, so a reader series with no
//! `stack` label leaves every stack judged here unknown: dropped, it would let
//! its stack read as in effect without it.
//!
//! Metrics values are untrusted input, so nothing here may stop or delay a
//! restriction op: they feed the display and the in-effect judgement, which
//! only ever withholds a confirmation.

use std::collections::{BTreeMap, BTreeSet};

use crate::effect::Instance;
use crate::lb::{MetricsClient, Series};

const UP: &str = "up";
const RESTRICTIONS: &str = "anthropic_client_store_restrictions";
const ALERT: &str = "anthropic_client_store_alert";
const UNMATCHED: &str = "anthropic_client_store_unmatched_disabled_name";

pub struct Fleet {
    /// stack → its instances, as the metrics store lists them.
    pub stacks: Result<BTreeMap<String, Vec<Instance>>, String>,
    /// Instances raising restrictions-behind, as `stack/instance`.
    pub behind: Result<BTreeSet<String>, String>,
    /// Disabled names some instance reports matching no client it serves →
    /// the stacks reporting them. Each instance reports at most 5.
    pub unmatched: Result<BTreeMap<String, BTreeSet<String>>, String>,
}

/// The instance a series belongs to: its pod, or its scrape instance.
fn instance_of(s: &Series) -> Option<&str> {
    s.labels
        .get("pod")
        .or_else(|| s.labels.get("instance"))
        .map(String::as_str)
}

fn stack_of(s: &Series) -> Option<&str> {
    s.labels
        .get("stack")
        .map(String::as_str)
        .filter(|v| !v.is_empty())
}

pub async fn read(metrics: &MetricsClient) -> Fleet {
    let (up, sets, behind, unmatched) = tokio::join!(
        metrics.instant(UP, UP),
        metrics.instant(RESTRICTIONS, RESTRICTIONS),
        metrics.instant(
            ALERT,
            r#"anthropic_client_store_alert{alert="restrictions_behind"}"#
        ),
        metrics.instant(UNMATCHED, UNMATCHED),
    );
    let err = |e: crate::error::AppError| e.detail().to_string();
    let behind = behind.map_err(err).map(|series| {
        series
            .iter()
            .filter(|s| s.value == 1.0)
            .filter_map(|s| Some(format!("{}/{}", stack_of(s)?, instance_of(s)?)))
            .collect::<BTreeSet<_>>()
    });
    let stacks = match (up, sets, &behind) {
        (Ok(up), Ok(sets), Ok(behind)) => instances(&up, &sets, behind),
        (Err(e), _, _) | (_, Err(e), _) => Err(err(e)),
        (_, _, Err(e)) => Err(e.clone()),
    };
    let unmatched = unmatched.map_err(err).map(|series| {
        let mut out: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for s in &series {
            if let (Some(name), Some(stack)) = (s.labels.get("name"), stack_of(s)) {
                out.entry(name.clone())
                    .or_default()
                    .insert(stack.to_string());
            }
        }
        out
    });
    Fleet {
        stacks,
        behind,
        unmatched,
    }
}

/// Instances per stack. The list comes from `up`, one entry per scraped
/// instance: an instance whose gauge lingers after its scrape went away is
/// not listed. Its restrictions series are counted raw, every series kept.
/// A reader series with no `stack` label is an error naming its instance.
fn instances(
    up: &[Series],
    sets: &[Series],
    behind: &BTreeSet<String>,
) -> Result<BTreeMap<String, Vec<Instance>>, String> {
    let unlabeled: BTreeSet<&str> = up
        .iter()
        .chain(sets)
        .filter(|s| stack_of(s).is_none())
        .map(|s| instance_of(s).unwrap_or("a series with no pod or instance label"))
        .collect();
    if !unlabeled.is_empty() {
        return Err(format!(
            "reader series without a stack label ({}), so their stack cannot be told",
            unlabeled.into_iter().collect::<Vec<_>>().join(", ")
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
    Ok(by_stack
        .into_iter()
        .map(|(stack, m)| (stack, m.into_values().collect()))
        .collect())
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
        let behind: BTreeSet<String> = ["k3s/lb-b".to_string()].into();
        let m = instances(&up, &sets, &behind).unwrap();
        let k3s = &m["k3s"];
        assert_eq!(k3s.len(), 2);
        let a = k3s.iter().find(|i| i.id == "lb-a").unwrap();
        assert!(a.up && !a.behind);
        assert_eq!(a.set_hashes, ["h1", "h0"]);
        let b = k3s.iter().find(|i| i.id == "lb-b").unwrap();
        assert!(!b.up && b.behind && b.set_hashes.is_empty());
        assert_eq!(m["cloudflare"][0].id, "edge:443");
    }

    /// A reader the console cannot place in a stack must not drop out of
    /// one: it would let that stack read as in effect without it.
    #[test]
    fn a_reader_without_a_stack_label_makes_the_stacks_unknown() {
        let up = vec![
            series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0),
            series(&[("pod", "lb-b")], 1.0),
        ];
        let sets = vec![series(
            &[("stack", "k3s"), ("pod", "lb-a"), ("set_hash", "h1")],
            1.0,
        )];
        let err = instances(&up, &sets, &BTreeSet::new()).unwrap_err();
        assert!(err.contains("lb-b"), "{err}");
        // An unlabeled restrictions series counts the same.
        let up = vec![series(&[("stack", "k3s"), ("pod", "lb-a")], 1.0)];
        let sets = vec![series(&[("pod", "lb-c"), ("set_hash", "h1")], 1.0)];
        assert!(
            instances(&up, &sets, &BTreeSet::new())
                .unwrap_err()
                .contains("lb-c")
        );
    }
}
