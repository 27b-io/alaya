# OTLP Tracing for Rust Services → BetterStack

Battle-tested instructions from alaya-server. Covers every pitfall discovered during integration.

## Contents

- [Dependencies](#dependencies)
- [Telemetry Init](#telemetry-init)
- [Environment Variables](#environment-variables)
- [TraceLayer for axum](#tracelayer-for-axum)
- [EnvFilter: The Global Gate](#envfilter-the-global-gate)
- [Cross-Thread Span Propagation](#cross-thread-span-propagation)
- [`#[tracing::instrument]` with `#[async_trait(?Send)]`](#tracinginstrument-with-async_traitsend)
- [Stage Spans for Concurrent Operations](#stage-spans-for-concurrent-operations)
- [Graceful Shutdown](#graceful-shutdown)
- [Git SHA in Binary](#git-sha-in-binary)
- [K8s Deployment](#k8s-deployment)
- [Debugging Checklist](#debugging-checklist)

## Dependencies

The current versions and features are in `crates/alaya-server/Cargo.toml`. The shape:

```toml
opentelemetry = "0.33"
opentelemetry_sdk = "0.33"
opentelemetry-otlp = { version = "0.33", default-features = false, features = ["http-proto", "reqwest-blocking-client", "trace"] }
tracing-opentelemetry = "0.34"
reqwest = { version = "0.13", features = ["blocking", "rustls"] }  # blocking required for OTLP exporter
```

### Critical: the blocking client feature

The batch processor exports from its own OS thread, outside any tokio runtime, so the exporter needs `reqwest::blocking::Client`. Since 0.33 its `HttpClient` impl exists only under `reqwest-blocking-client`; `reqwest-client` alone gives a compile error at `.with_http_client(...)`. Keep the three `opentelemetry*` crates on one minor line, or the SDK types do not match.

### Pitfall: renaming a span after it starts

Since tracing-opentelemetry 0.32 the OTel span starts when the tracing span is first entered, and an `otel.name` recorded after that is silently dropped (status and attributes still land). Rename a running span through the OTel API instead: `span.context().span().update_name(name)` (see `rename_span` in `crates/alaya-server/src/mcp.rs`).

## Telemetry Init

```rust
use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{SpanExporter, WithHttpConfig};
use opentelemetry_sdk::{Resource, trace::{Sampler, SdkTracerProvider}};
use std::sync::OnceLock;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

pub fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(
            "your_crate=info,tower_http=info"
            // Add every crate that has #[tracing::instrument]:
            // "your_crate=info,your_lib=info,tower_http=info"
        ));

    let fmt = tracing_subscriber::fmt::layer().with_target(true);
    let endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();

    if let Some(ref endpoint) = endpoint {
        let service_name = std::env::var("OTEL_SERVICE_NAME")
            .unwrap_or_else(|_| "my-service".into());
        let git_sha = option_env!("GIT_SHA").unwrap_or("dev");

        let resource = Resource::builder()
            .with_service_name(service_name)
            .with_attribute(opentelemetry::KeyValue::new(
                "service.version", git_sha.to_string(),
            ))
            .build();

        // CRITICAL: Use reqwest::blocking::Client.
        // BatchSpanProcessor runs on a dedicated OS thread using
        // futures_executor::block_on(). reqwest's async client panics:
        // "there is no reactor running, must be called from the context
        // of a Tokio 1.x runtime". The panic happens on a background
        // thread and silently dies — no crash, no error, just no traces.
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
                tracing_subscriber::registry().with(filter).with(fmt).init();
                // Never log `e`: it quotes the endpoint value, which may
                // carry userinfo or a query-string token (CWE-532).
                tracing::warn!("OTLP exporter failed to build, tracing to stderr only");
                return;
            }
        };

        let provider = SdkTracerProvider::builder()
            .with_resource(resource)
            .with_sampler(Sampler::AlwaysOn)
            .with_batch_exporter(exporter)
            .build();

        let tracer = provider.tracer("my-service");
        let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt)
            .with(otel_layer)
            .init();

        let _ = TRACER_PROVIDER.set(provider);
        // Origin only, for the same reason: this line is in every boot log.
        let origin = reqwest::Url::parse(endpoint)
            .map(|u| u.origin().ascii_serialization())
            .unwrap_or_else(|_| "<unparseable>".into());
        tracing::info!("OTLP tracing enabled → {origin} (version: {git_sha})");
    } else {
        tracing_subscriber::registry().with(filter).with(fmt).init();
    }
}

pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get()
        && let Err(e) = provider.shutdown()
    {
        tracing::warn!("OTLP shutdown error: {e}");
    }
}
```

## Environment Variables

```bash
OTEL_EXPORTER_OTLP_ENDPOINT=https://<your-ingesting-host>.betterstackdata.com
OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer <token>"
OTEL_SERVICE_NAME=my-service
RUST_LOG=my_crate=info,my_lib=info,tower_http=info
```

- **`https://` is required off-cluster.** Whenever any `OTEL_EXPORTER_OTLP_HEADERS`
  value is set, the server refuses to boot on a plain-`http` endpoint that is not
  cluster-local — the header is assumed to carry a credential, and `http` is a
  scheme reqwest speaks, so nothing else would fail. The
  `OTEL_EXPORTER_OTLP_TRACES_*` twins take precedence exactly as the exporter
  resolves them, and only the pair it will actually use is checked.
- The SDK appends `/v1/traces` to the endpoint automatically
- Headers are parsed on the first `=` sign: `Authorization=Bearer token` → key `Authorization`, value `Bearer token`
- BetterStack per-source endpoint (not `in-otel.logs.betterstack.com`)

## TraceLayer for axum

```rust
use tower_http::trace::TraceLayer;

let app = Router::new()
    .route("/api", post(handler))
    .layer(
        TraceLayer::new_for_http()
            .make_span_with(
                tower_http::trace::DefaultMakeSpan::new()
                    .level(tracing::Level::INFO),  // default is DEBUG — gets filtered out
            ),
    );

// Health routes OUTSIDE the TraceLayer to avoid noise:
let health = Router::new().route("/health", get(health_handler));
let app = health.merge(app);
```

## EnvFilter: The Global Gate

**The EnvFilter filters spans BEFORE they reach any layer, including the OTLP layer.**

If your `RUST_LOG` is `my_crate=info` and you have `#[tracing::instrument]` on methods in `my_lib`, those spans are silently dropped. You must include every crate that has instrumented methods:

```bash
RUST_LOG=my_crate=info,my_lib=info,tower_http=info
```

A failed export is reported only as an event on target `opentelemetry_sdk` (`BatchSpanProcessor.ExportError`), on the batch processor's thread, so a `RUST_LOG` that does not name that target drops every one. alaya-server's console layer adds `opentelemetry_sdk=warn` on top of whatever `RUST_LOG` says, and its OTLP layer does not (`console_layer` in `crates/alaya-server/src/telemetry.rs`), so a 401, a 5xx or an unreachable collector logs one ERROR line per failed batch with no setting. The line carries the status code or "network error", never the URL. For the SDK's debug detail on top:
```bash
RUST_LOG=my_crate=info,opentelemetry_sdk=debug
```

## Cross-Thread Span Propagation

**Contextual span lookup does NOT work across thread boundaries.**

If you have a multi-threaded architecture (e.g., axum on one thread, service worker on another via mpsc channel), spans created via `#[tracing::instrument]` or `.instrument(span)` on the worker thread will NOT become children of spans from the axum thread.

### What doesn't work

```rust
// axum thread creates request span
// sends command to worker thread via mpsc
// worker thread:
let result = svc.do_work()
    .instrument(parent_span_from_axum)  // DOES NOT create parent-child link
    .await;
```

### What works

```rust
// Carry the span in the command struct
struct Cmd {
    inner: CmdInner,
    span: tracing::Span,
}

// Capture on the sender side
let cmd = Cmd {
    inner: CmdInner::Search { ... },
    span: tracing::Span::current(),
};

// On the worker thread: create explicit child span
let child = tracing::info_span!(parent: &cmd.span, "search", mode = "hybrid");
let result = svc.do_work()
    .instrument(child)  // child has explicit parent — works cross-thread
    .await;
```

### Same-thread contextual lookup DOES work

Once you've bridged the thread boundary with an explicit `parent:` span, all `#[tracing::instrument]` and `tracing::info_span!()` calls on the SAME thread will correctly nest as children via contextual lookup:

```rust
// Worker thread:
let bridge = tracing::info_span!(parent: &cmd.span, "search");  // explicit parent
async {
    // These all nest correctly via contextual lookup (same thread):
    svc.search(params).await           // #[instrument] creates child of bridge
        // → search_hybrid()           // #[instrument] creates child of search
        //     → fan_out span          // info_span! creates child of search_hybrid
        //         → embed_batch()     // #[instrument] creates child of fan_out
}.instrument(bridge).await;
```

## `#[tracing::instrument]` with `#[async_trait(?Send)]`

Works fine. The macros are orthogonal:

```rust
#[async_trait(?Send)]
impl MyTrait for MyImpl {
    #[tracing::instrument(skip(self, large_data), fields(n = items.len()))]
    async fn process(&self, items: &[Item], large_data: &[u8]) -> Result<()> {
        // ...
    }
}
```

Rules:
- `skip(self)` always — don't serialize the struct
- `skip` large data (embeddings, byte arrays, request bodies)
- Use `fields()` for small scalars and counts
- `fields(n = items.len())` for collection sizes
- `fields(mode = ?enum_value)` for Debug formatting
- `fields(action = %string_value)` for Display formatting

## Stage Spans for Concurrent Operations

Wrap `futures::join!` groups in spans to show concurrent structure:

```rust
let (embed_result, bm25_results, count) = {
    let _span = tracing::info_span!("fan_out").entered();
    let embed_fut = self.embeddings.embed_batch(&texts, PromptName::Query);
    let bm25_fut = self.vectors.search_by_text(&query, limit, None);
    let count_fut = self.vectors.count();
    futures::join!(embed_fut, bm25_fut, count_fut)
};
```

In the trace waterfall, `embed_batch`, `search_by_text`, and `count` appear as overlapping children of `fan_out` — showing they ran concurrently.

## Graceful Shutdown

```rust
fn main() {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        init_tracing();

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .expect("server error");

        // Flush OTLP spans after axum stops
        shutdown_tracing();
    });
}

async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    let mut sigterm = tokio::signal::unix::signal(
        tokio::signal::unix::SignalKind::terminate()
    ).expect("failed to register SIGTERM handler");

    tokio::select! {
        _ = ctrl_c => tracing::info!("received SIGINT"),
        _ = sigterm.recv() => tracing::info!("received SIGTERM"),
    }
}
```

## Git SHA in Binary

### Dockerfile

```dockerfile
ARG GIT_SHA=unknown
ENV ALAYA_GIT_SHA=${GIT_SHA}
RUN cargo build --release
```

### CI (GitHub Actions)

```yaml
- uses: docker/build-push-action@v6
  with:
    build-args: GIT_SHA=${{ github.sha }}
```

### In code

```rust
let git_sha = crate::build_info::git_sha().unwrap_or("dev");
```

`crates/alaya-server/src/build_info.rs` is the single home for these
compile-time reads — don't re-scatter `option_env!` at call sites. It exposes
`version()` (crate semver), `git_sha()`, `built_at()` and `version_qualified()`
(`<semver>+<sha>`, used by MCP `serverInfo`).

Used in: the `GET /health/detail` body (`version` / `git_sha` / `built_at`), the OTLP
`service.version` resource attribute (bare SHA, so existing SHA-equality
queries keep matching), and the startup log.

## K8s Deployment

```yaml
env:
- name: OTEL_EXPORTER_OTLP_ENDPOINT
  value: "https://<your-ingesting-host>.betterstackdata.com"
- name: OTEL_EXPORTER_OTLP_HEADERS
  valueFrom:
    secretKeyRef:
      name: betterstack-otel
      key: source-token  # value: "Authorization=Bearer <token>"
- name: OTEL_SERVICE_NAME
  value: "my-service"
- name: RUST_LOG
  value: "my_crate=info,my_lib=info,tower_http=info"
```

Network policy: allow egress on port 443 (HTTPS to BetterStack).

## Debugging Checklist

If traces don't show up:

1. **Check `/health/detail` for `git_sha`** (authenticated) — is the right code deployed?
2. **Check startup log** — does it say "OTLP tracing enabled"?
3. **Check for panics** — `kubectl logs | grep panic` — the BatchSpanProcessor panic is silent (background thread)
4. **Check RUST_LOG** — does it include every crate with `#[instrument]`?
5. **Check TraceLayer level** — `DefaultMakeSpan` defaults to DEBUG, which gets filtered by `tower_http=info`
6. **Check cross-thread** — are you using `parent: &span` for cross-thread? Contextual lookup doesn't work
7. **Check for export errors** — `kubectl logs | grep BatchSpanProcessor.ExportError` (logged whatever `RUST_LOG` says); add `opentelemetry_sdk=debug` to RUST_LOG for more
8. **Test from local** — `cargo run --release` with OTEL env vars, query BetterStack directly
