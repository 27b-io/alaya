//! The versions the console committed, exported as labels for the metrics
//! scraper.
//!
//! An alert compares every version an LB instance installs against these
//! labels, kept 30 days in the metrics store, and fires on one the console
//! never committed: a write made with the writer token outside the console,
//! or an edit of the store. Versions are labels, not values, because a
//! sample value keeps about 12 significant digits and a version has more.
//!
//! The list lives in memory. A commit the scraper never saw, because the
//! console restarted inside one scrape interval, reads as an unmatched
//! install and fires the alert: loud, and the console's write log holds the
//! commit to check it against.

use std::collections::VecDeque;
use std::fmt::Write as _;
use std::sync::Mutex;

/// How many records stay exported. Each was scraped long before it falls off
/// the end; the cap only bounds memory and the page's size.
const KEEP_RECORDS: usize = 1024;

#[derive(Default)]
pub struct Committed {
    records: Mutex<VecDeque<Record>>,
}

struct Record {
    name: String,
    version: u64,
    /// When the console saw the store commit it, epoch seconds.
    at: i64,
}

impl Committed {
    /// A restriction record the store wrote for a console request.
    pub fn record(&self, name: &str, version: u64) {
        let mut records = self.records.lock().expect("committed lock poisoned");
        if records.len() == KEEP_RECORDS {
            records.pop_front();
        }
        records.push_back(Record {
            name: name.to_string(),
            version,
            at: crate::session::now_epoch(),
        });
    }

    /// The Prometheus text exposition of everything still kept.
    pub fn render(&self) -> String {
        let records = self.records.lock().expect("committed lock poisoned");
        let mut out = String::from(
            "# HELP ops_console_committed_record A restriction record version the console committed; the value is when.\n\
             # TYPE ops_console_committed_record gauge\n",
        );
        for r in records.iter() {
            let _ = writeln!(
                out,
                "ops_console_committed_record{{name=\"{}\",version=\"{}\"}} {}",
                escape_label(&r.name),
                r.version,
                r.at
            );
        }
        out
    }
}

/// A label value in the text exposition format: backslash, quote and newline
/// escaped. Store names never carry them, but this text is a parser's input.
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
        for i in 0..KEEP_RECORDS as u64 {
            c.record("x", i);
        }
        let text = c.render();
        assert!(!text.contains("acme"));
        assert_eq!(
            text.matches("ops_console_committed_record{").count(),
            KEEP_RECORDS
        );
    }

    #[test]
    fn label_values_are_escaped() {
        assert_eq!(escape_label("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
