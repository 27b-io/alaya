//! The versions the console committed, exported as labels for the metrics
//! scraper.
//!
//! An alert compares every version an LB instance installs against these
//! labels, kept 30 days in the metrics store, and fires on one the console
//! never committed: a write made with the writer token outside the console,
//! or an edit of the store. Versions are labels, not values, because a
//! sample value keeps about 12 significant digits and a version has more.
//! Each series' value is when the console saw the store commit it, in epoch
//! seconds.
//!
//! The lists live in memory. A commit the scraper never saw, because the
//! console restarted inside one scrape interval, reads as an unmatched
//! install and fires the alert: loud, and the console's write log holds the
//! commit to check it against.
//!
//! The document labels are read back too, through the metrics proxy's route
//! pinned to their series: the documents to return to by point-in-time
//! recovery, and the write-log copies an import may diff against. Anything on
//! the host can write the metrics store, so a label read back is a claim, and
//! this process's own list, which nothing outside it can write, is checked
//! first where it can answer.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Mutex;

use crate::effect::is_hex64;
use crate::lb::{MetricsClient, Series};

/// The series each committed document is exported as, and the name of the
/// metrics proxy's route that reads it back.
pub const DOCUMENT_SERIES: &str = "ops_console_committed_document";

/// How far ahead of the console's clock a label's sample may be and still
/// read as a commit time: a day. Anything later is no time the console saw.
const AHEAD_SECS: u64 = 24 * 3600;

/// How many of each kind stay exported. Each was scraped long before it
/// falls off the end; the cap only bounds memory and the page's size.
const KEEP: usize = 1024;

#[derive(Default)]
pub struct Committed {
    records: Mutex<VecDeque<Record>>,
    documents: Mutex<VecDeque<Document>>,
}

struct Record {
    name: String,
    version: u64,
    at: i64,
}

struct Document {
    version: u64,
    digest: String,
    at: i64,
}

fn push<T>(list: &mut VecDeque<T>, item: T) {
    if list.len() == KEEP {
        list.pop_front();
    }
    list.push_back(item);
}

impl Committed {
    /// A restriction record the store wrote for a console request.
    pub fn record(&self, name: &str, version: u64) {
        push(
            &mut self.records.lock().expect("committed lock poisoned"),
            Record {
                name: name.to_string(),
                version,
                at: crate::session::now_epoch(),
            },
        );
    }

    /// A document the store committed for a console request, by its version
    /// and the digest of its exact bytes, as of now.
    pub fn document(&self, version: u64, digest: &str) {
        self.document_at(version, digest, crate::session::now_epoch());
    }

    /// A document the store committed at `at`, in epoch seconds: a recovery
    /// commit is exported when the operator accepts it, as of when the store
    /// made it, since its label is read back as its commit time. A document
    /// already exported keeps its entry, checked and added under one lock: a
    /// recovery accepted twice, even at once, exports one series.
    pub fn document_at(&self, version: u64, digest: &str, at: i64) {
        let mut list = self.documents.lock().expect("committed lock poisoned");
        if list
            .iter()
            .any(|d| d.version == version && d.digest == digest)
        {
            return;
        }
        push(
            &mut list,
            Document {
                version,
                digest: digest.to_string(),
                at,
            },
        );
    }

    /// Whether this process exported this document as committed.
    pub fn holds(&self, version: u64, digest: &str) -> bool {
        self.documents
            .lock()
            .expect("committed lock poisoned")
            .iter()
            .any(|d| d.version == version && d.digest == digest)
    }

    /// The Prometheus text exposition of everything still kept.
    pub fn render(&self) -> String {
        let mut out = String::from(
            "# HELP ops_console_committed_record A restriction record version the console committed; the value is when.\n\
             # TYPE ops_console_committed_record gauge\n",
        );
        for r in self.records.lock().expect("committed lock poisoned").iter() {
            let _ = writeln!(
                out,
                "ops_console_committed_record{{name=\"{}\",version=\"{}\"}} {}",
                escape_label(&r.name),
                r.version,
                r.at
            );
        }
        out.push_str(
            "# HELP ops_console_committed_document A document version and digest the console committed; the value is when.\n\
             # TYPE ops_console_committed_document gauge\n",
        );
        for d in self
            .documents
            .lock()
            .expect("committed lock poisoned")
            .iter()
        {
            let _ = writeln!(
                out,
                "ops_console_committed_document{{version=\"{}\",digest=\"{}\"}} {}",
                d.version,
                escape_label(&d.digest),
                d.at
            );
        }
        out
    }
}

/// A document the console committed, as its label reads back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Label {
    pub version: u64,
    pub digest: String,
    /// When the console saw the commit, in whole epoch seconds.
    pub secs: u64,
}

/// Every document the console committed in the last 30 days, from its labels
/// in the metrics store. An empty answer is returned empty: a query outside
/// the route's pin is empty too, so the caller decides what empty means.
pub async fn read(metrics: &MetricsClient) -> Result<Vec<Label>, String> {
    let series = metrics
        .instant(
            DOCUMENT_SERIES,
            &format!("max by (version, digest) (last_over_time({DOCUMENT_SERIES}[30d]))"),
        )
        .await
        .map_err(|e| e.detail().to_string())?;
    Ok(labels_of(&series, crate::session::now_epoch()))
}

/// The labels that name a document and a time: a version, a 64 hex digest,
/// and a sample from the epoch's first second to a day past `now`. Any other
/// series is no commit the console made, and is left out.
fn labels_of(series: &[Series], now: i64) -> Vec<Label> {
    let latest = u64::try_from(now).unwrap_or(0) + AHEAD_SECS;
    series
        .iter()
        .filter_map(|s| {
            let version = s.labels.get("version")?.parse::<u64>().ok()?;
            let digest = s.labels.get("digest").filter(|d| is_hex64(d))?;
            let plausible = s.value.is_finite() && s.value >= 1.0 && s.value <= latest as f64;
            plausible.then(|| Label {
                version,
                digest: digest.clone(),
                secs: s.value as u64,
            })
        })
        .collect()
}

/// A label value in the text exposition format: backslash, quote and newline
/// escaped. Store names and digests never carry them, but this text is a
/// parser's input.
fn escape_label(v: &str) -> String {
    v.replace('\\', r"\\")
        .replace('"', "\\\"")
        .replace('\n', r"\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_render_as_labels_and_the_oldest_falls_off() {
        let c = Committed::default();
        c.record("acme", 1_791_698_400_000_000);
        let text = c.render();
        assert!(text.contains("# TYPE ops_console_committed_record gauge\n"));
        assert!(
            text.contains(
                "ops_console_committed_record{name=\"acme\",version=\"1791698400000000\"} "
            ),
            "{text}"
        );
        for i in 0..KEEP as u64 {
            c.record("x", i);
        }
        let text = c.render();
        assert!(!text.contains("acme"));
        assert_eq!(text.matches("ops_console_committed_record{").count(), KEEP);
    }

    #[test]
    fn documents_render_as_version_and_digest_labels() {
        let c = Committed::default();
        let d = "d".repeat(64);
        c.document(1_791_700_000_123, &d);
        let text = c.render();
        assert!(text.contains("# TYPE ops_console_committed_document gauge\n"));
        assert!(
            text.contains(&format!(
                "ops_console_committed_document{{version=\"1791700000123\",digest=\"{d}\"}} "
            )),
            "{text}"
        );
        for i in 0..KEEP as u64 {
            c.document(i, "e");
        }
        assert!(!c.render().contains(&d));
    }

    #[test]
    fn a_document_exported_twice_is_one_series() {
        let c = Committed::default();
        let d = "d".repeat(64);
        c.document(9, &d);
        c.document(9, &d);
        c.document(9, &"e".repeat(64));
        assert_eq!(c.render().matches("version=\"9\"").count(), 2);
    }

    /// A recovery accepted later is exported as of its commit, once however
    /// often it is accepted.
    #[test]
    fn a_document_can_be_exported_as_of_its_commit() {
        let c = Committed::default();
        let d = "d".repeat(64);
        c.document_at(9, &d, 1_794_000_000);
        c.document_at(9, &d, 1_794_000_900);
        let text = c.render();
        assert!(
            text.contains(&format!(
                "ops_console_committed_document{{version=\"9\",digest=\"{d}\"}} 1794000000\n"
            )),
            "{text}"
        );
        assert_eq!(text.matches("version=\"9\"").count(), 1);
        assert!(c.holds(9, &d) && !c.holds(9, &"e".repeat(64)) && !c.holds(8, &d));
    }

    /// A label is read back only when it names a document and a time the
    /// console could have seen: no sample can overflow a commit time.
    #[test]
    fn labels_read_back_only_as_plausible_commits() {
        let d = "d".repeat(64);
        let series = |version: &str, digest: &str, value: f64| Series {
            labels: [("version", version), ("digest", digest)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            value,
        };
        let now = 1_794_000_000;
        let read = labels_of(
            &[
                series("9", &d, 1_793_999_000.0),
                series("10", &d, (now + 86_400) as f64),
                series("11", &d, (now + 86_401) as f64),
                series("12", &d, 1.8e19),
                series("13", &d, f64::NAN),
                series("14", &d, 0.0),
                series("x", &d, 1_793_999_000.0),
                series("15", "abc", 1_793_999_000.0),
            ],
            now,
        );
        let kept: Vec<(u64, u64)> = read.iter().map(|l| (l.version, l.secs)).collect();
        assert_eq!(kept, [(9, 1_793_999_000), (10, 1_794_086_400)]);
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape_label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
