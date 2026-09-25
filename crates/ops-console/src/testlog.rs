//! Test-only log capture.
//!
//! Several defences in this crate are assertions about what reaches the pod
//! log — that no IdP- or upstream-supplied byte can forge a record, and that
//! a warning fires on the event it names and no other. Pinning those needs
//! a subscriber writing somewhere a test can read, which is the fiddly part;
//! this is it, once, instead of per module. `LogBuf::capture` is plain text
//! on purpose (the binary logs JSON): it proves call sites escape untrusted
//! values themselves, independently of the encoder.

use std::io::Write;
use std::sync::{Arc, LazyLock, Mutex};

use tracing::Dispatch;
use tracing::subscriber::{DefaultGuard, NoSubscriber};

/// A dispatcher that lives as long as the test process, so at least two are
/// always registered once any capture starts.
///
/// With exactly one registered, tracing-core takes a fast path that computes
/// a callsite's interest from the default of whichever thread registers it
/// first. A parallel test with no subscriber of its own then caches `never`
/// for every thread, and a capture asserting on that callsite comes back
/// empty. With two, registration asks every registered dispatcher, the
/// capture included. Upstream is tokio-rs/tracing#3611, reported against
/// 0.1.36: retire this once the tracing-core in Cargo.lock carries the fix,
/// not when the issue closes.
static OFF_THE_FAST_PATH: LazyLock<Dispatch> = LazyLock::new(|| Dispatch::new(NoSubscriber::new()));

/// Make `sub` the default subscriber for the current thread until the
/// returned guard drops — the one way a test here installs a subscriber.
/// `set_default`, not `with_default`: the code under test is usually async,
/// and a closure cannot hold an `.await`.
pub fn scoped(sub: impl tracing::Subscriber + Send + Sync + 'static) -> DefaultGuard {
    LazyLock::force(&OFF_THE_FAST_PATH);
    tracing::subscriber::set_default(sub)
}

#[derive(Clone, Default)]
pub struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl LogBuf {
    /// Capture into this buffer on the current thread until the returned
    /// guard drops. Thread-local, so parallel tests cannot cross-contaminate.
    pub fn capture(&self) -> DefaultGuard {
        scoped(
            tracing_subscriber::fmt()
                .with_writer(self.clone())
                .with_ansi(false)
                .finish(),
        )
    }

    pub fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }

    /// The first log record containing `needle`, panicking if none does.
    ///
    /// Selected by line rather than asserting on the whole capture: a
    /// request that logs twice is normal (the console's own record, then
    /// `tower_http`'s), and the injection question is about one record.
    /// A record that a raw `\n` split is no longer one line, so everything
    /// past the split is missing from what this returns — which the caller
    /// catches by asserting the whole payload is present, not by
    /// `separators_in`, whose evidence the split itself removed.
    pub fn record(&self, needle: &str) -> String {
        let logged = self.text();
        logged
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no log record contains {needle:?}: {logged}"))
            .to_string()
    }
}

impl Write for LogBuf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuf {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// Every char in one log record that could end it early for some reader.
///
/// Asserting the record is one `\n`-delimited line would pin the instance
/// and miss the class: a Unicode-aware ingester also breaks on
/// U+2028/U+2029/NEL, and a terminal on ESC. `str`'s `Debug` escapes every
/// one of them, which is what recording an untrusted field with `?` rather
/// than `%` buys — so this is the oracle for "nothing untrusted reached the
/// log raw".
pub fn separators_in(record: &str) -> Vec<char> {
    record
        .chars()
        .filter(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
        .collect()
}
