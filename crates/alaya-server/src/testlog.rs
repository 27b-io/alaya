//! Test-only log capture, copied from `ops-console/src/testlog.rs`.
//!
//! Some defences here are assertions about what reaches the pod log: that a
//! warning fires on the event it names and no other. Pinning those needs a
//! subscriber writing somewhere a test can read, and one that is not fooled
//! by tracing-core's callsite-interest cache (below).

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
/// empty — so a "no WARN" assertion passes on nothing. With two,
/// registration asks every registered dispatcher, the capture included.
/// Upstream is tokio-rs/tracing#3611, reported against 0.1.36: retire this
/// once the tracing-core in Cargo.lock carries the fix, not when the issue
/// closes.
static OFF_THE_FAST_PATH: LazyLock<Dispatch> = LazyLock::new(|| Dispatch::new(NoSubscriber::new()));

/// Make `sub` the default subscriber for the current thread until the
/// returned guard drops — the one way a test here installs a subscriber.
/// `set_default`, not `with_default`: the code under test is usually async,
/// and a closure cannot hold an `.await`.
pub(crate) fn scoped(sub: impl tracing::Subscriber + Send + Sync + 'static) -> DefaultGuard {
    LazyLock::force(&OFF_THE_FAST_PATH);
    tracing::subscriber::set_default(sub)
}

#[derive(Clone, Default)]
pub(crate) struct LogBuf(Arc<Mutex<Vec<u8>>>);

impl LogBuf {
    /// Capture into this buffer on the current thread until the returned
    /// guard drops. Thread-local, so parallel tests cannot cross-contaminate.
    pub(crate) fn capture(&self) -> DefaultGuard {
        scoped(
            tracing_subscriber::fmt()
                .with_writer(self.clone())
                .with_ansi(false)
                .finish(),
        )
    }

    pub(crate) fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
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
