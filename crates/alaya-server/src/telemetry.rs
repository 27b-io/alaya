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
/// - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` / `_HEADERS`: the signal-specific
///   twins, which take precedence over the pair above.
///
/// Panics when any header is set and the endpoint would send it in the clear
/// off-cluster, via `check_credential_transport` — the same guard, policy and
/// messages as every credential URL in `Config`.
pub fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_FILTER));

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    let otel_endpoint = crate::env_non_empty("OTEL_EXPORTER_OTLP_ENDPOINT");

    if otel_endpoint.is_some() {
        // Checked before the exporter is built, so a refusal means the process
        // never starts rather than starting and posting the token. The policy
        // and the refusal messages are `check_credential_transport`'s: these
        // vars never reach `Config`, but that guard is a free function over
        // (var, url, has_credential, transport), so the OTLP path is one more
        // caller rather than a second copy of the rule.
        let has_credential = OTLP_HEADER_VARS
            .iter()
            .any(|var| crate::env_non_empty(var).is_some());
        for var in OTLP_ENDPOINT_VARS {
            // Raw, not trimmed: `opentelemetry-otlp` reads these with
            // `std::env::var`, so a trimmed copy would certify a string the
            // exporter never dials. A padded value it cannot parse falls back
            // to its default endpoint silently, which is a credential going
            // somewhere nobody configured — refuse instead.
            let Some(raw) = std::env::var(var).ok().filter(|v| !v.trim().is_empty()) else {
                continue;
            };
            if has_credential && raw != raw.trim() {
                panic!(
                    "{var} has leading or trailing whitespace; the exporter reads it raw \
                     and would silently fall back to its own default endpoint"
                );
            }
            crate::check_credential_transport(var, &raw, has_credential, crate::Transport::Http)
                .unwrap_or_else(|e| panic!("{e}"));
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

        // Origin only: an endpoint may carry userinfo or a query credential,
        // and this line goes to the pod log on every boot (CWE-532).
        tracing::info!(
            "OTLP tracing enabled → {} (version: {git_sha})",
            crate::log_safe_origin(otel_endpoint.as_deref().unwrap_or_default())
        );
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
}

/// Endpoint vars in `opentelemetry-otlp`'s own precedence order:
/// `resolve_http_endpoint` tries the signal-specific one first and only then
/// falls back to the generic. Both are guarded, because otherwise a compliant
/// `OTEL_EXPORTER_OTLP_ENDPOINT` certifies a plaintext
/// `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` that is what actually gets dialled.
///
/// The generic var is also the on-switch for the whole arm, so a
/// signal-specific-only config exports nothing at all (and leaks nothing).
const OTLP_ENDPOINT_VARS: [&str; 2] = [
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
];

/// Header vars, same precedence. Any non-empty value counts as a credential:
/// the union is stricter than the exporter's `signal.or(generic)`, and reading
/// header *names* to decide would miss `x-api-key`, `dd-api-key` and friends.
const OTLP_HEADER_VARS: [&str; 2] = [
    "OTEL_EXPORTER_OTLP_HEADERS",
    "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
];

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
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` is set — normally unset under `cargo
    /// test`, but that is a fact about the shell, not about this test, and a
    /// developer who exports the pair `docs/otlp-rust-betterstack.md` shows
    /// will run the arm here (and hit the boot guard). `set_var` is `unsafe`
    /// in edition 2024 and races the rest of this binary, so the test cannot
    /// pin it either way. Measured twice: forcing the arm
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

    /// The policy — https anywhere, plaintext cluster-local only, nothing to
    /// protect means nothing to refuse — is `check_credential_transport`'s and
    /// is pinned in `main.rs`. What is ours is the wiring, and only two things
    /// in it can regress silently.
    ///
    /// First, coverage of the signal-specific twins. Dropping either from these
    /// lists leaves a compliant generic value certifying a plaintext one that
    /// `resolve_http_endpoint` prefers — and no assertion about a refusal
    /// message catches it, because the var is a format argument, never a
    /// branch. Second, the transport: `Redis` here would refuse every endpoint
    /// the exporter can actually speak.
    #[test]
    fn otlp_guards_both_precedence_pairs_as_an_http_transport() {
        assert!(super::OTLP_ENDPOINT_VARS.contains(&"OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"));
        assert!(super::OTLP_HEADER_VARS.contains(&"OTEL_EXPORTER_OTLP_TRACES_HEADERS"));

        let check = |url, has_credential| {
            crate::check_credential_transport(
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                url,
                has_credential,
                crate::Transport::Http,
            )
        };
        // A bearer off-cluster in the clear is the fault this exists for.
        assert!(check("http://collector.example.com", true).is_err());
        assert!(check("https://collector.example.com", true).is_ok());
        // `http://collector:4318` is single-label service DNS — the sanctioned
        // in-cluster path, and the reason `Transport::Http` is the right column.
        assert!(check("http://collector:4318", true).is_ok());
        // No headers, no secret: a plaintext collector nobody authenticates to
        // still boots, so this cannot drift into a URL validator.
        assert!(check("http://collector.example.com", false).is_ok());
    }
}
