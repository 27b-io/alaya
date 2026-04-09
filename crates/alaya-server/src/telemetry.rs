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
pub fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_FILTER));

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    if let Some(ref endpoint) = otel_endpoint {
        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "alaya-server".to_string());
        let git_sha = option_env!("ALAYA_GIT_SHA").unwrap_or("dev");

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

/// Flush buffered OTLP spans and shut down the tracer provider.
/// No-op if OTLP was never configured.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get()
        && let Err(e) = provider.shutdown()
    {
        tracing::warn!("OTLP shutdown error: {e}");
    }
}
