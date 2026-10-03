//! Periodic self-check (LAB-4026): every interval, embed a fixed query past
//! the cache and run it as a read-only hybrid search, expecting a known
//! memory in the top 10. The result lands in the `pod` section of
//! `/health/detail` and `GET /stats`, and a failure logs one WARN on this
//! module's target, `alaya_server::selfcheck`.
//!
//! Never wired to `/health`: the Kubernetes probes stay shallow, because a
//! probe that embeds would restart every pod while the embedder is down, and
//! a restart does not fix the embedder.

use std::time::Duration;

use tokio::sync::oneshot;

use alaya_core::vitals::{SelfCheckOutcome, Step, Vitals};

use crate::{Cmd, CmdInner, REPLY_MARGIN, ServiceHandle, epoch_secs};

/// Budget for one check, embed to last result.
pub(crate) const BUDGET: Duration = Duration::from_secs(10);
const DEFAULT_INTERVAL_SECS: u64 = 300;
/// Floor on the interval: each check is one real embed per pod.
const MIN_INTERVAL_SECS: u64 = 30;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SelfCheckConfig {
    pub(crate) query: String,
    pub(crate) expect_hash: String,
    pub(crate) interval: Duration,
}

/// `SELFCHECK_QUERY`, `SELFCHECK_EXPECT_HASH`, `SELFCHECK_INTERVAL_SECS`.
/// Both of the first two unset disables the check. Only one set is an error,
/// not a quiet disable: a check that silently never runs is the failure this
/// exists to catch.
pub(crate) fn parse(
    query: Option<String>,
    expect_hash: Option<String>,
    interval_secs: Option<String>,
) -> Result<Option<SelfCheckConfig>, String> {
    let (query, expect_hash) = match (query, expect_hash) {
        (None, None) => return Ok(None),
        (Some(q), Some(h)) => (q, h),
        _ => {
            return Err(
                "SELFCHECK_QUERY and SELFCHECK_EXPECT_HASH must be set together, or neither".into(),
            );
        }
    };
    if !alaya_types::memory::validate_content_hash(&expect_hash) {
        return Err("SELFCHECK_EXPECT_HASH must be a 64-char lowercase hex content_hash".into());
    }
    let secs = match interval_secs {
        None => DEFAULT_INTERVAL_SECS,
        Some(s) => s
            .parse::<u64>()
            .ok()
            .filter(|n| *n >= MIN_INTERVAL_SECS)
            .ok_or_else(|| {
                format!("SELFCHECK_INTERVAL_SECS must be a whole number of seconds >= {MIN_INTERVAL_SECS}")
            })?,
    };
    Ok(Some(SelfCheckConfig {
        query,
        expect_hash,
        interval: Duration::from_secs(secs),
    }))
}

/// One check through the worker, recorded in `vitals`. A worker that does
/// not answer is a failure too, named as the `worker` step.
pub(crate) async fn run_once(handle: &ServiceHandle, cfg: &SelfCheckConfig, vitals: &Vitals) {
    let start = std::time::Instant::now();
    let (reply, rx) = oneshot::channel();
    let cmd = Cmd {
        inner: CmdInner::SelfCheck {
            query: cfg.query.clone(),
            expect_hash: cfg.expect_hash.clone(),
            reply,
        },
        span: tracing::Span::none(),
    };
    let outcome = match handle.try_dispatch(cmd) {
        Err((_, msg)) => SelfCheckOutcome::fail(Step::Worker, msg, None),
        Ok(()) => match tokio::time::timeout(BUDGET + REPLY_MARGIN, rx).await {
            Ok(Ok(o)) => o,
            Ok(Err(_)) => {
                SelfCheckOutcome::fail(Step::Worker, "worker dropped the self-check", None)
            }
            Err(_) => SelfCheckOutcome::fail(
                Step::Worker,
                format!(
                    "no answer from the worker within {}s",
                    (BUDGET + REPLY_MARGIN).as_secs()
                ),
                None,
            ),
        },
    };
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let consecutive = vitals.record_selfcheck(epoch_secs(), elapsed_ms, outcome.clone());
    let rerank = outcome.rerank.map(|r| r.as_str());
    match outcome.failing_step {
        Some(step) => tracing::warn!(
            step = step.as_str(),
            error = outcome.error.as_deref().unwrap_or(""),
            rerank,
            consecutive_failures = consecutive,
            elapsed_ms,
            "self-check failed"
        ),
        None => tracing::debug!(rerank, elapsed_ms, "self-check passed"),
    }
}

/// Run the check now and then every `cfg.interval`. The caller aborts the
/// task at shutdown: it holds a `ServiceHandle`, which keeps the worker's
/// channel open.
pub(crate) fn spawn(
    handle: ServiceHandle,
    cfg: SelfCheckConfig,
    vitals: std::sync::Arc<Vitals>,
) -> tokio::task::JoinHandle<()> {
    tracing::info!(interval_s = cfg.interval.as_secs(), "self-check enabled");
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(cfg.interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        loop {
            tick.tick().await;
            run_once(&handle, &cfg, &vitals).await;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn some(s: &str) -> Option<String> {
        Some(s.into())
    }

    fn cfg() -> SelfCheckConfig {
        SelfCheckConfig {
            query: "q".into(),
            expect_hash: "a".repeat(64),
            interval: Duration::from_secs(300),
        }
    }

    /// A worker stand-in that answers one self-check with `outcome`.
    fn answering(outcome: SelfCheckOutcome) -> ServiceHandle {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Cmd>(1);
        tokio::spawn(async move {
            if let Some(Cmd {
                inner: CmdInner::SelfCheck { reply, .. },
                ..
            }) = rx.recv().await
            {
                let _ = reply.send(outcome);
            }
        });
        ServiceHandle { tx }
    }

    /// AC-4: a failed check logs exactly one WARN naming the step, on a
    /// target the deployed filter (`alaya_server=info`) admits; a pass logs
    /// no WARN. A worker that is gone fails as the `worker` step.
    #[tokio::test]
    async fn a_failed_check_warns_once_naming_the_step() {
        let log = crate::testlog::LogBuf::default();
        let _guard = log.capture();
        let vitals = Vitals::new("p".into(), 0, true);

        let pass = SelfCheckOutcome {
            failing_step: None,
            error: None,
            rerank: Some(alaya_core::vitals::Rerank::Ran),
        };
        run_once(&answering(pass), &cfg(), &vitals).await;
        assert!(!log.text().contains("WARN"), "{}", log.text());

        let fail = SelfCheckOutcome::fail(Step::Embed, "connection refused", None);
        run_once(&answering(fail), &cfg(), &vitals).await;
        let (closed, rx) = tokio::sync::mpsc::channel::<Cmd>(1);
        drop(rx);
        run_once(&ServiceHandle { tx: closed }, &cfg(), &vitals).await;

        let text = log.text();
        let warns: Vec<&str> = text.lines().filter(|l| l.contains("WARN")).collect();
        assert_eq!(warns.len(), 2, "{text}");
        assert!(
            warns.iter().all(|l| l.contains("alaya_server::selfcheck")),
            "{text}"
        );
        assert!(warns[0].contains("step=\"embed\""), "{text}");
        assert!(warns[0].contains("connection refused"), "{text}");
        assert!(warns[1].contains("step=\"worker\""), "{text}");
        assert!(warns[1].contains("consecutive_failures=2"), "{text}");
        assert_eq!(
            vitals.snapshot()["selfcheck"]["last"]["failing_step"],
            "worker"
        );
    }

    #[test]
    fn parse_disables_on_neither_and_refuses_half_a_config() {
        let hash = "a".repeat(64);
        assert_eq!(parse(None, None, None), Ok(None));
        assert!(parse(some("q"), None, None).is_err());
        assert!(parse(None, Some(hash.clone()), None).is_err());
        assert!(parse(some("q"), some("ABC"), None).is_err());
        assert!(
            parse(some("q"), Some("A".repeat(64)), None).is_err(),
            "uppercase"
        );
        assert!(
            parse(some("q"), Some(hash.clone()), some("5")).is_err(),
            "below floor"
        );
        assert!(parse(some("q"), Some(hash.clone()), some("x")).is_err());

        let cfg = parse(some("q"), Some(hash.clone()), None).unwrap().unwrap();
        assert_eq!(cfg.interval, Duration::from_secs(300));
        assert_eq!(cfg.expect_hash, hash);
        let cfg = parse(some("q"), Some(hash), some("60")).unwrap().unwrap();
        assert_eq!(cfg.interval, Duration::from_secs(60));
    }
}
