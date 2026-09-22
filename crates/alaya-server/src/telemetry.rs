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
/// Panics when the headers the exporter will send are non-empty and the
/// endpoint it will dial would carry them in the clear off-cluster, via
/// `check_credential_transport` — the same guard, policy and messages as every
/// credential URL in `Config`. Which endpoint and which headers is decided by
/// `check_otlp_transport`, by the exporter's own rules.
pub fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_FILTER));

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    // The generic var is the on-switch for the whole arm, so a
    // signal-specific-only config exports nothing at all (and leaks nothing).
    if crate::env_non_empty("OTEL_EXPORTER_OTLP_ENDPOINT").is_some() {
        // Checked before the exporter is built, so a refusal means the process
        // never starts rather than starting and posting the token. Raw, not
        // trimmed: `opentelemetry-otlp` reads these with `std::env::var`, so a
        // trimmed copy would certify a string the exporter never dials.
        let endpoint = check_otlp_transport(
            std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").unwrap_or_default(),
            std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT").ok(),
            std::env::var("OTEL_EXPORTER_OTLP_HEADERS").ok(),
            std::env::var("OTEL_EXPORTER_OTLP_TRACES_HEADERS").ok(),
        )
        .unwrap_or_else(|e| panic!("{e}"));

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
            crate::log_safe_origin(&endpoint)
        );
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
}

/// Guard the endpoint/header pair the exporter will actually use, and return
/// that endpoint. The pair is selected from the same four raw values
/// `opentelemetry-otlp` reads, by its own rules (`resolve_http_endpoint` and
/// `build_client` in `exporter/http/mod.rs`):
///
/// - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` wins when it is set *and parses as
///   an `http::Uri`*; otherwise `OTEL_EXPORTER_OTLP_ENDPOINT` is dialled.
/// - `OTEL_EXPORTER_OTLP_TRACES_HEADERS` wins when it is set *at all*: a
///   set-but-empty value sends no headers and silences the generic ones.
///
/// Only that pair is checked. Guarding a shadowed value refuses a config the
/// exporter never uses — a plaintext legacy generic endpoint behind an `https`
/// signal-specific one took the service down — and a union of the header vars
/// counts a credential the exporter never sends. Guarding too little is the
/// leak: a compliant generic endpoint certifying a plaintext signal-specific
/// one that is what gets dialled.
///
/// Any non-blank header string counts as a credential. The exporter drops
/// entries without a `key=value` shape, so this is stricter, and reading
/// header *names* to decide would miss `x-api-key`, `dd-api-key` and friends.
///
/// Split from the env reads so it can be pinned by a test: `set_var` is
/// `unsafe` in edition 2024 and races every other test in the binary.
fn check_otlp_transport(
    generic_endpoint: String,
    traces_endpoint: Option<String>,
    generic_headers: Option<String>,
    traces_headers: Option<String>,
) -> Result<String, String> {
    let has_credential = crate::non_empty_trimmed(traces_headers.or(generic_headers)).is_some();
    // `axum::http` is the one `http` crate in the lockfile — the exporter's
    // own parser, so the fallback decision is its decision, not an imitation.
    let (var, endpoint) = match traces_endpoint {
        Some(t) if t.parse::<axum::http::Uri>().is_ok() => {
            ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", t)
        }
        _ => ("OTEL_EXPORTER_OTLP_ENDPOINT", generic_endpoint),
    };
    // A padded generic value the exporter cannot parse falls back to its own
    // default endpoint silently, which is a credential going somewhere nobody
    // configured — refuse instead. (A signal-specific value that parsed above
    // cannot be padded; `http::Uri` rejects whitespace.)
    if has_credential && endpoint != endpoint.trim() {
        return Err(format!(
            "{var} has leading or trailing whitespace; the exporter reads it raw \
             and would silently fall back to its own default endpoint"
        ));
    }
    crate::check_credential_transport(var, &endpoint, has_credential, crate::Transport::Http)?;
    Ok(endpoint)
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
    /// is pinned in `main.rs`. What is ours is *which* pair reaches it, and
    /// that must match what the exporter dials, in both directions: guard a
    /// shadowed value and a valid config refuses to boot; miss the winning one
    /// and a bearer goes out in the clear behind a compliant decoy.
    ///
    /// Arguments are the four raw env values in the order `init_tracing`
    /// reads them; `None` is unset, `Some("")` is set-but-empty.
    fn guard(
        generic_endpoint: &str,
        traces_endpoint: Option<&str>,
        generic_headers: Option<&str>,
        traces_headers: Option<&str>,
    ) -> Result<String, String> {
        super::check_otlp_transport(
            generic_endpoint.to_string(),
            traces_endpoint.map(str::to_string),
            generic_headers.map(str::to_string),
            traces_headers.map(str::to_string),
        )
    }

    const BEARER: Option<&str> = Some("Authorization=Bearer token");

    #[test]
    fn signal_specific_endpoint_shadows_the_generic_one() {
        // The exporter dials only the https one, so the plaintext legacy
        // generic behind it must not refuse boot.
        let ok = guard(
            "http://legacy.example.com",
            Some("https://active.example.com"),
            BEARER,
            BEARER,
        );
        assert_eq!(ok.as_deref(), Ok("https://active.example.com"));
        // The mirror image is the leak the guard exists for: the compliant
        // generic is the decoy, the plaintext signal-specific one is dialled.
        let err = guard(
            "https://decoy.example.com",
            Some("http://active.example.com"),
            BEARER,
            None,
        )
        .unwrap_err();
        assert!(err.contains("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"), "{err}");
    }

    #[test]
    fn unparseable_signal_specific_endpoint_falls_back_to_the_generic_one() {
        // `resolve_http_endpoint` skips a signal-specific value `http::Uri`
        // rejects, so the generic one is what gets dialled: refused when it
        // is plaintext off-cluster, booted when it is https.
        let err = guard(
            "http://legacy.example.com",
            Some(" https://active.example.com "),
            BEARER,
            None,
        )
        .unwrap_err();
        assert!(err.contains("OTEL_EXPORTER_OTLP_ENDPOINT:"), "{err}");
        assert!(
            guard(
                "https://legacy.example.com",
                Some(" http://active.example.com "),
                BEARER,
                None,
            )
            .is_ok()
        );
    }

    #[test]
    fn set_but_empty_signal_specific_headers_silence_the_generic_ones() {
        // `env::var(signal).or_else(generic)`: set-but-empty wins and sends
        // nothing, so there is no credential and a plaintext off-cluster
        // collector boots.
        assert!(guard("http://collector.example.com", None, BEARER, Some("")).is_ok());
        assert!(guard("http://collector.example.com", None, BEARER, Some("  ")).is_ok());
        // Unset falls through to the generic headers, which are sent.
        assert!(guard("http://collector.example.com", None, BEARER, None).is_err());
        // Signal-specific headers alone are a credential too.
        assert!(guard("http://collector.example.com", None, None, BEARER).is_err());
    }

    #[test]
    fn padded_generic_endpoint_is_refused_only_when_a_credential_rides_on_it() {
        // The exporter would silently fall back to its default endpoint; with a
        // bearer attached that is refused, without one it is not this guard's
        // question — so it cannot drift into a URL validator.
        let err = guard(" http://collector:4318 ", None, BEARER, None).unwrap_err();
        assert!(err.contains("whitespace"), "{err}");
        assert!(guard(" http://collector:4318 ", None, None, None).is_ok());
    }

    #[test]
    fn otlp_is_an_http_transport() {
        // `http://collector:4318` is single-label service DNS — the sanctioned
        // in-cluster path, and the reason `Transport::Http` is the right
        // column: `Redis` would refuse every endpoint the exporter can speak.
        assert!(guard("http://collector:4318", None, BEARER, None).is_ok());
        assert!(guard("https://collector.example.com", None, BEARER, None).is_ok());
    }
}
