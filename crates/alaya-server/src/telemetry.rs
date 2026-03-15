//! OpenTelemetry OTLP tracing setup for Phoenix integration.
//!
//! If `OTEL_EXPORTER_OTLP_ENDPOINT` is set, configures OTLP HTTP exporter.
//! If not set, tracing goes to stderr only.
//!
//! Phoenix-specific: set `OTEL_EXPORTER_OTLP_HEADERS="authorization=Bearer <jwt>"`

use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_sdk::{
    Resource,
    trace::{Sampler, SdkTracerProvider},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// Initialize tracing with optional OTLP export.
///
/// Uses standard OTel env vars:
/// - `OTEL_EXPORTER_OTLP_ENDPOINT`: OTLP endpoint (e.g. http://phoenix-svc:6006)
/// - `OTEL_EXPORTER_OTLP_HEADERS`: Headers (e.g. "authorization=Bearer <jwt>")
/// - `OTEL_SERVICE_NAME`: Service name (default: "alaya-server")
/// - `RUST_LOG`: Log level filter
pub fn init_tracing() {
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("alaya_server=info"));

    let fmt_layer = tracing_subscriber::fmt::layer().with_target(true);

    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    if let Some(ref endpoint) = otel_endpoint {
        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "alaya-server".to_string());

        let resource = Resource::builder().with_service_name(service_name).build();

        // OTLP HTTP exporter — reads endpoint + headers from env vars automatically
        let exporter = SpanExporter::builder()
            .with_http()
            .build()
            .expect("failed to create OTLP exporter");

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

        // Keep provider alive for process lifetime
        std::mem::forget(provider);

        tracing::info!("OTLP tracing enabled → {endpoint}");
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt_layer)
            .init();
    }
}
