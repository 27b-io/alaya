//! OpenTelemetry OTLP tracing setup.
//!
//! If `OTEL_EXPORTER_OTLP_ENDPOINT` is set, configures OTLP HTTP exporter.
//! If not set, tracing goes to stderr only.
//!
//! Default filter includes `tower_http=info` so HTTP request spans
//! propagate to both the console and the OTLP exporter.

use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{SpanExporter, WithHttpConfig};
use opentelemetry_sdk::{
    Resource,
    trace::{Sampler, SdkTracerProvider},
};
use std::sync::OnceLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Global tracer provider — stored here so `shutdown_tracing()` can flush it.
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// Default log filter — includes tower_http so HTTP request spans reach the
/// OTLP exporter, and opentelemetry for internal diagnostics.
const DEFAULT_FILTER: &str =
    "alaya_server=info,alaya_core=info,alaya_backends=info,tower_http=info";

/// Initialize tracing with optional OTLP export.
///
/// Uses standard OTel env vars:
/// - `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint
/// - `OTEL_EXPORTER_OTLP_HEADERS`: Headers (e.g. "Authorization=Bearer <token>")
/// - `OTEL_SERVICE_NAME`: Service name (default: "alaya-server")
/// - `RUST_LOG`: Log level filter
///
/// Panics when the headers carry a credential and the endpoint would send it in
/// the clear off-cluster — see `check_otlp_endpoint`.
pub fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_FILTER));

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    let otel_endpoint = crate::env_non_empty("OTEL_EXPORTER_OTLP_ENDPOINT");

    if let Some(ref endpoint) = otel_endpoint {
        // Before the exporter is built, so a refusal means the process never
        // starts rather than starting and posting the token. `main` calls
        // `init_tracing` before `Config::from_env`, so this is the earliest
        // guard in the boot, ahead of even the credential-transport table.
        //
        // Both endpoint vars, because `opentelemetry-otlp` prefers the
        // signal-specific one and falls back to the generic
        // (`resolve_http_endpoint`): checking only the generic one leaves
        // `OTEL_EXPORTER_OTLP_ENDPOINT=https://ok` covering an
        // `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://elsewhere` that is what
        // actually gets dialled. Headers resolve by the same preference, so a
        // credential in either var is a credential on this wire.
        let has_credential = crate::env_non_empty("OTEL_EXPORTER_OTLP_HEADERS").is_some()
            || crate::env_non_empty("OTEL_EXPORTER_OTLP_TRACES_HEADERS").is_some();
        for var in [
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        ] {
            if let Some(url) = crate::env_non_empty(var) {
                check_otlp_endpoint(var, &url, has_credential).unwrap_or_else(|e| panic!("{e}"));
            }
        }

        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "alaya-server".to_string());
        // Deliberately the bare SHA, not the qualified version: existing
        // BetterStack queries filter service.version by SHA equality, and this
        // ticket has no mandate to break them. Reads through build_info only to
        // retire the duplicate option_env! — the emitted value is unchanged.
        let git_sha = crate::build_info::git_sha().unwrap_or("dev");

        let resource = Resource::builder()
            .with_service_name(service_name)
            .with_attribute(opentelemetry::KeyValue::new(
                "service.version",
                git_sha.to_string(),
            ))
            .build();

        // The BatchSpanProcessor runs on a dedicated OS thread and calls
        // futures_executor::block_on() — not a tokio runtime. reqwest's async
        // client panics without a tokio reactor. Use reqwest::blocking::Client
        // which works on any thread.
        let blocking_client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .expect("failed to build blocking reqwest client");

        let exporter = match SpanExporter::builder()
            .with_http()
            .with_http_client(blocking_client)
            .build()
        {
            Ok(e) => e,
            Err(e) => {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(fmt_layer)
                    .init();
                tracing::warn!("OTLP exporter failed ({e}), tracing to stderr only");
                return;
            }
        };

        let provider = SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(Sampler::AlwaysOn)
            .with_batch_exporter(exporter)
            .build();

        let tracer = provider.tracer("alaya-server");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .with(otel_layer)
            .init();

        // Store provider so shutdown can flush buffered spans
        let _ = TRACER_PROVIDER.set(provider);

        tracing::info!("OTLP tracing enabled → {endpoint} (version: {git_sha})");
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
}

/// A bearer token in `OTEL_EXPORTER_OTLP_HEADERS` is a credential on the wire,
/// and `http://` is a scheme `reqwest` speaks — so unlike every var in
/// `check_credential_transport`, nothing fails on its own here: an off-cluster
/// plaintext collector just works, and posts the token in the clear. The
/// shipped pattern is off-cluster + bearer (`docs/otlp-rust-betterstack.md`),
/// so this is the deployment, not a hypothetical.
///
/// Same policy as the `Transport::Http` arm of `check_credential_transport`,
/// and the same `is_cluster_local` decides it — `https` anywhere, plain `http`
/// only to a cluster-local collector. It cannot share that function because
/// these vars never pass through `Config`: `opentelemetry-otlp` reads the
/// environment itself, so the check has to sit at the read.
///
/// No credential means nothing to protect, so a plaintext off-cluster endpoint
/// is allowed — this decides whether the request may carry a secret, not
/// whether the URL is tasteful.
///
/// Messages name the host, never the raw value: an endpoint may carry userinfo.
fn check_otlp_endpoint(var: &str, endpoint: &str, has_credential: bool) -> Result<(), String> {
    if !has_credential {
        return Ok(());
    }
    // Unparseable with a credential set is a misconfigured export either way —
    // the exporter would fail its own build and degrade to stderr. Refuse
    // instead of degrading, so the operator learns it from the var name rather
    // than from a silently missing trace stream.
    let parsed =
        reqwest::Url::parse(endpoint).map_err(|e| format!("{var} is not a valid URL ({e})"))?;
    match parsed.scheme() {
        "https" => Ok(()),
        "http" if crate::is_cluster_local(&parsed) => Ok(()),
        scheme => Err(format!(
            "{var}: {scheme}://{} is not https and not a cluster-local http endpoint; \
             OTEL_EXPORTER_OTLP_HEADERS carries a credential that must not travel in \
             the clear",
            parsed.host_str().unwrap_or(""),
        )),
    }
}

/// Flush buffered OTLP spans and shut down the tracer provider.
/// No-op if OTLP was never configured.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get()
        && let Err(e) = provider.shutdown()
    {
        tracing::warn!("OTLP shutdown error: {e}");
    }
}

#[cfg(test)]
mod tests {
    /// `main` installs the subscriber before it builds the tokio runtime, so
    /// that the boot credential-transport guard — which runs in
    /// `Config::from_env`, earlier still — can warn rather than print. This
    /// pins that ordering: installing must not need a reactor.
    ///
    /// It does NOT pin the OTLP arm, and an earlier version of this comment
    /// wrongly claimed it did. `init_tracing` enters that arm only when
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` is set, which `cargo test` does not, so
    /// the blocking client never runs here. Measured twice: forcing the arm
    /// with an unroutable endpoint still returns in 0.00s (the exporter builds
    /// a client, it does not dial), and substituting the async
    /// `reqwest::Client` leaves this test green, because a missing reactor
    /// panics at first export on the batch thread rather than at
    /// construction. Reaching that needs a live span flushed to a real
    /// endpoint, which is not a unit test's job.
    #[test]
    fn installs_without_a_tokio_runtime() {
        assert!(tokio::runtime::Handle::try_current().is_err());
        super::init_tracing();
        tracing::warn!("subscriber is live");
    }

    // ─── OTLP credential transport (LAB-4313) ──────────────────────────────
    //
    // Against `check_otlp_endpoint`, not against a booted process: `set_var`
    // is `unsafe` in edition 2024 and races every other test in this binary,
    // which is why `non_empty_trimmed` and `parse_judge_daily_cap` are split
    // from their reads the same way. `init_tracing` composes this with
    // `unwrap_or_else(|e| panic!(...))`, so an `Err` here IS the refused boot.

    /// A token plus plaintext off-cluster is the one combination that must not
    /// start, and the refusal has to name the var — an operator reading a pod's
    /// crash log gets the variable, not a scheme complaint.
    #[test]
    fn otlp_refuses_a_credential_over_plaintext_off_cluster() {
        for (var, bad) in [
            (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://s2349817.eu-fsn-3.betterstackdata.com",
            ),
            // End-anchored cluster-local matching, same as the sibling guard:
            // a public domain wearing an `svc` label is still public.
            (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "http://evil.svc.attacker.com",
            ),
            (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "HTTP://Collector.Example.Com",
            ),
            // The generic var being fine does not make the signal-specific one
            // fine: it is the one `opentelemetry-otlp` prefers.
            (
                "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                "http://collector.example.com/v1/traces",
            ),
            // Not a scheme reqwest can dial, so it is also not https and not
            // cluster-local http — refused rather than approved and discovered
            // at first export.
            (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                "grpc://collector.example.com",
            ),
        ] {
            let err = super::check_otlp_endpoint(var, bad, true).unwrap_err();
            assert!(err.contains(var), "{bad}: {err}");
        }
        // Unparseable is refused too, and neither message may echo a value
        // that can carry userinfo (CWE-532).
        let err = super::check_otlp_endpoint("OTEL_EXPORTER_OTLP_ENDPOINT", "not a url", true)
            .unwrap_err();
        assert!(err.contains("OTEL_EXPORTER_OTLP_ENDPOINT"), "{err}");
        let err = super::check_otlp_endpoint(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://user:s3cr3t@collector.example.com",
            true,
        )
        .unwrap_err();
        assert!(
            err.contains("collector.example.com") && !err.contains("s3cr3t"),
            "{err}"
        );
    }

    /// TLS anywhere, or plaintext that never leaves the cluster. Both are the
    /// real shipped shapes — BetterStack over https, and the in-cluster
    /// collector `CLAUDE.md` documents.
    #[test]
    fn otlp_allows_https_anywhere_and_cluster_local_plaintext() {
        for ok in [
            "https://s2349817.eu-fsn-3.betterstackdata.com",
            "HTTPS://collector.example.com",
            "http://phoenix-svc.recsys.svc:6006",
            "http://phoenix-svc.recsys.svc.cluster.local:6006",
            // Single-label service DNS, IP literals, loopback — `is_cluster_local`
            // decides all of these, this test only pins that it is consulted.
            "http://collector:4318",
            "http://localhost:4318",
            "http://10.0.0.5:4318",
            "http://[::1]:4318",
        ] {
            assert!(
                super::check_otlp_endpoint("OTEL_EXPORTER_OTLP_ENDPOINT", ok, true).is_ok(),
                "{ok}"
            );
        }
    }

    /// No headers, no secret, nothing to leak — so this must stay a credential
    /// guard and not drift into a URL validator that breaks plaintext
    /// collectors nobody was authenticating to.
    #[test]
    fn otlp_without_a_credential_allows_plaintext_off_cluster() {
        for endpoint in [
            "http://collector.example.com",
            "grpc://collector.example.com",
            "not a url",
        ] {
            assert!(
                super::check_otlp_endpoint("OTEL_EXPORTER_OTLP_ENDPOINT", endpoint, false).is_ok(),
                "{endpoint}"
            );
        }
    }
}
