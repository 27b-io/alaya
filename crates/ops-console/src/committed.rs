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

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Mutex;

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

fn push<T>(list: &Mutex<VecDeque<T>>, item: T) {
    let mut list = list.lock().expect("committed lock poisoned");
    if list.len() == KEEP {
        list.pop_front();
    }
    list.push_back(item);
}

impl Committed {
    /// A restriction record the store wrote for a console request.
    pub fn record(&self, name: &str, version: u64) {
        push(
            &self.records,
            Record {
                name: name.to_string(),
                version,
                at: crate::session::now_epoch(),
            },
        );
    }

    /// A document the store committed for a console request, by its version
    /// and the digest of its exact bytes.
    pub fn document(&self, version: u64, digest: &str) {
        push(
            &self.documents,
            Document {
                version,
                digest: digest.to_string(),
                at: crate::session::now_epoch(),
            },
        );
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
    fn label_values_are_escaped() {
        assert_eq!(escape_label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
