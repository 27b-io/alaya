# Ālaya Prometheus Metrics — Design

**Status:** Draft — pending implementation plan
**Author:** Ray Walker + Claude
**Date:** 2026-04-12

## Purpose

Add Prometheus metrics to `alaya-server` to give VictoriaMetrics on the lab k3s cluster visibility into request rates, latency distributions, error rates, and domain-specific behavior (search result counts, embedding cache effectiveness, service worker backpressure).

Currently the service has distributed tracing (OTLP → BetterStack) and structured logs, but no aggregate metrics — meaning no rate/percentile dashboards, no alerting on error rates, no capacity planning signals.

## Non-Goals

- Bridge instrumentation (`alaya-bridge` is a thin Cypher relay; the interesting behavior lives in `alaya-server`).
- Grafana dashboards (build manually from Grafana UI after metrics ship).
- Per-user / per-tenant metrics (cardinality risk; not needed now).
- Authentication on `/metrics` (matches `/health` — internal-only diagnostic endpoint).
- Backend health gauges (`/health` endpoint already serves this via JSON).

## Architecture

### Library choice

`metrics` (v0.24) + `metrics-exporter-prometheus` (v0.16).

The `metrics` crate is Rust's standard metrics façade — analogous to `tracing` for logs. A globally installed recorder uses atomic operations, so recording from any thread is lock-free. This fits Alaya's split architecture (multi-threaded axum + single-threaded LocalSet) without any coordination primitives: the `counter!` / `histogram!` / `gauge!` macros resolve to atomic increments on a global `Arc<Inner>`.

Rejected alternatives:
- **`opentelemetry-prometheus`** — pulls in the heavyweight OTel metrics SDK and has a verbose `Meter` → `Counter` API. We already have OTel for tracing; coupling metrics to that pipeline adds complexity without benefit.
- **Raw `prometheus` crate** — lower level, manual `Registry` management, more boilerplate for no gain.

### Integration with axum

`PrometheusBuilder::new().install_recorder()` returns a `PrometheusHandle` that is internally `Arc`-wrapped — `Clone + Send + Sync`. It goes directly into axum state alongside the existing `HealthChecker`, on the unauthenticated route group. No HTTP listener from the exporter itself; we serve `/metrics` through the existing axum router so it shares the service port.

A background `tokio::spawn` task on the axum runtime calls `handle.run_upkeep()` every 5 seconds. This is required for histogram aggregation to render correctly — skipping it is a silent data corruption hazard.

### Recording across threads

The `metrics` recorder is global and uses atomics. Recording from the LocalSet worker thread (where `MemoryService` runs) is identical to recording from the axum thread — no channel required, no Send/Sync wrangling. This is the key reason this library works cleanly with Alaya's architecture.

## Metrics Catalog

### Request metrics (instrumented in `service_worker` match arms)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `alaya_request_duration_seconds` | histogram | `op` | Latency distribution per operation |
| `alaya_requests_total` | counter | `op`, `status` | Total requests, tagged with `ok` or `error` |
| `alaya_errors_total` | counter | `op`, `kind` | Error breakdown by AlayaError variant |

`op` label values: `store`, `search`, `delete`, `relation`, `supersede`, `contradictions`, `find_duplicates`, `merge_duplicates`, `patch`, `health`.

`kind` label values (mapped from `AlayaError` variants): `storage`, `embedding`, `graph`, `config`, `validation`, `not_found`, `summary`, `serialization`.

All labels are drawn from fixed enumerations — no cardinality explosion risk.

**Histogram buckets** for `alaya_request_duration_seconds`:
```
[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
```
Chosen based on observed latency range: ~20ms cache hits at the fast end, ~10s cold store at the slow end, with P50 around 200ms for hybrid search.

### Domain metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `alaya_search_results` | histogram | `mode` | Result count distribution per search mode |
| `alaya_embedding_cache_hits_total` | counter | — | Embedding cache hits (CachedEmbedding wrapper) |
| `alaya_embedding_cache_misses_total` | counter | — | Embedding cache misses |
| `alaya_service_channel_depth` | gauge | — | Pending commands in the axum→LocalSet mpsc channel |

**Search results histogram buckets:** `[0, 1, 5, 10, 25, 50, 100]`. Small integers; "0 results" gets its own bucket to track bad-query prevalence.

**Embedding cache metrics:** `CachedEmbedding` in `cached_embedding.rs` already has `Cell<u64>` hit/miss counters and a dead `stats()` method. The Cells get removed; `counter!` macros take their place. Same LocalSet thread; works because the `metrics` recorder is thread-agnostic.

**Channel depth gauge:** `mpsc::Sender` does not expose queue depth, so we track it manually. Increment `alaya_service_channel_depth` in `ServiceHandle::call()` after a successful `tx.send()`; decrement in `service_worker` after `rx.recv()`. Two lines, exposes backpressure directly. Important for detecting long-running operations (`find_duplicates`, `merge_duplicates`) starving the queue.

## Code Changes

### `crates/alaya-server/Cargo.toml`
Add:
```toml
metrics = "0.24"
metrics-exporter-prometheus = "0.16"
```

### `crates/alaya-server/src/telemetry.rs`
Add `init_metrics()` function, with bucket configuration factored out so tests can verify against the same spec:

```rust
const DURATION_BUCKETS: &[f64] = &[
    0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
];
const SEARCH_RESULTS_BUCKETS: &[f64] = &[0.0, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0];

// pub(crate) so the test module can share the same config path
pub(crate) fn configure_builder(builder: PrometheusBuilder) -> PrometheusBuilder {
    builder
        .set_buckets(DURATION_BUCKETS)
        .expect("failed to set default buckets")
        .set_buckets_for_metric(
            Matcher::Full("alaya_search_results".into()),
            SEARCH_RESULTS_BUCKETS,
        )
        .expect("failed to set search_results buckets")
}

pub fn init_metrics() -> PrometheusHandle {
    configure_builder(PrometheusBuilder::new())
        .install_recorder()
        .expect("failed to install metrics recorder")
}
```

The `DURATION_BUCKETS` and `SEARCH_RESULTS_BUCKETS` constants are also used by the test suite so bucket config can be asserted end-to-end without duplicating the values.

### `crates/alaya-server/src/main.rs`

**New helpers** (replace `log_ok` / `log_err` call sites; structured logging remains, metrics are added alongside):
```rust
fn record_ok(op: &str, start: Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    histogram!("alaya_request_duration_seconds", "op" => op.to_string()).record(elapsed);
    counter!("alaya_requests_total", "op" => op.to_string(), "status" => "ok").increment(1);
}

fn record_err(op: &str, err: &AlayaError, start: Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let kind = error_kind(err);
    histogram!("alaya_request_duration_seconds", "op" => op.to_string()).record(elapsed);
    counter!("alaya_requests_total", "op" => op.to_string(), "status" => "error").increment(1);
    counter!("alaya_errors_total", "op" => op.to_string(), "kind" => kind.to_string()).increment(1);
}

fn error_kind(err: &AlayaError) -> &'static str {
    match err {
        AlayaError::Storage(_) => "storage",
        AlayaError::Embedding(_) => "embedding",
        AlayaError::Graph(_) => "graph",
        AlayaError::Config(_) => "config",
        AlayaError::Validation(_) => "validation",
        AlayaError::NotFound(_) => "not_found",
        AlayaError::Summary(_) => "summary",
        AlayaError::Serialization(_) => "serialization",
    }
}
```

Call sites: every `service_worker` match arm, in both the `Ok` and `Err` branches. Existing `tracing::info!` / `log_err` structured logging is preserved — metrics are additive.

**Search results histogram** is recorded in the `Search` arm after extracting `result_count`:
```rust
histogram!("alaya_search_results", "mode" => mode.clone()).record(n as f64);
```

**Channel depth gauge** — two call sites:
```rust
// In ServiceHandle::call(), after tx.send() succeeds:
gauge!("alaya_service_channel_depth").increment(1.0);

// In service_worker, immediately after rx.recv():
gauge!("alaya_service_channel_depth").decrement(1.0);
```

**Routing change** — `/metrics` joins the unauthenticated route group:
```rust
let public = Router::new()
    .route("/health", get(health))
    .with_state(checker)
    .merge(
        Router::new()
            .route("/metrics", get(metrics_handler))
            .with_state(prom_handle)
    );
```

Handler:
```rust
async fn metrics_handler(
    axum::extract::State(handle): axum::extract::State<PrometheusHandle>,
) -> String {
    handle.render()
}
```

**Upkeep task** on the axum runtime:
```rust
let upkeep_handle = prom_handle.clone();
tokio::spawn(async move {
    let mut ticker = tokio::time::interval(Duration::from_secs(5));
    loop {
        ticker.tick().await;
        upkeep_handle.run_upkeep();
    }
});
```

### `crates/alaya-server/src/cached_embedding.rs`

Replace the `Cell<u64> hits` / `Cell<u64> misses` fields and the `#[allow(dead_code)] stats()` method with direct `counter!` calls inside `embed_batch`:
```rust
// On cache hit:
counter!("alaya_embedding_cache_hits_total").increment(1);

// On cache miss:
counter!("alaya_embedding_cache_misses_total").increment(miss_texts.len() as u64);
```

## K8s Exposure

### Service manifest (`lab/k8s/mcp/alaya.yaml`)

**Already correct.** The existing `alaya-server` Service has port 3001 named `http`:

```yaml
spec:
  selector:
    app: alaya-server
  ports:
  - port: 3001
    targetPort: 3001
    name: http
  type: ClusterIP
```

No Service manifest change needed. VMServiceScrape will reference this port by name.

### NetworkPolicy

**Already permissive enough.** The existing `alaya-server-policy` allows ingress on port 3001 with no `from` restriction — any source pod in the cluster can reach it (subject to other policies). vmagent pods in the `monitoring` namespace will be able to scrape without a policy change.

### VMServiceScrape (`lab/k8s/monitoring/`)

New CRD in the monitoring namespace:

```yaml
apiVersion: operator.victoriametrics.com/v1beta1
kind: VMServiceScrape
metadata:
  name: alaya-server
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: alaya-server
  namespaceSelector:
    matchNames: [mcp]
  endpoints:
    - port: http
      interval: 30s
      path: /metrics
```

vmagent auto-discovers via `selectAllByDefault: true`. No operator restart needed. Add to the existing `kustomization.yaml` in `lab/k8s/monitoring/`.

## TraceLayer Exclusion

`/metrics` must be excluded from `tower_http::trace::TraceLayer`, same as `/health` currently is. Every 30-second scrape would otherwise generate an OTel span that propagates to BetterStack, polluting trace data with health-noise. Because the public route group (`/health` + `/metrics`) is merged separately from the protected routes that carry the TraceLayer, this happens naturally with the routing structure above — no explicit exclusion needed.

## Testing

### The global recorder constraint

`PrometheusBuilder::install_recorder()` installs a process-global recorder that can only be installed once per process lifetime. Cargo runs unit tests in parallel by default, so naive tests that each call `install_recorder()` will see only the first succeed and all others fail with "recorder already installed."

**Resolution:** a test-only `OnceLock<PrometheusHandle>` that installs once and shares the handle. The test helper MUST go through the same `configure_builder()` function as production, otherwise the bucket-override test is verifying nothing:

```rust
#[cfg(test)]
static TEST_HANDLE: std::sync::OnceLock<PrometheusHandle> = std::sync::OnceLock::new();

#[cfg(test)]
fn test_handle() -> &'static PrometheusHandle {
    TEST_HANDLE.get_or_init(|| {
        crate::telemetry::configure_builder(PrometheusBuilder::new())
            .install_recorder()
            .expect("failed to install test recorder")
    })
}
```

This requires `configure_builder` to be `pub(crate)` in `telemetry.rs` so the test module can reach it. Tests that need to *assert* specific counter/histogram values use unique metric names so they don't interfere with each other when run in parallel. For bucket tests, which need to verify the real production metric name (`alaya_search_results`), multiple emitters are fine — bucket label presence in render output is the assertion, not specific counts.

Alternative considered: `metrics_util::debugging::Snapshotter` — a non-installing recorder for unit tests. Rejected because it doesn't exercise the `PrometheusHandle::render()` code path we actually care about verifying.

### Unit tests (`crates/alaya-server/src/main.rs` tests module)

1. **`test_metrics_handler_renders`** — use the shared test handle, emit a unique counter via macro, call `handle.render()`, assert the output contains the expected metric name and value.
2. **`test_error_kind_mapping`** — construct each `AlayaError` variant (all 8), pass through `error_kind()`, assert expected label strings. The compiler's exhaustive match check guards against adding a new variant without updating the mapping.
3. **`test_histogram_buckets_applied`** — record a value into `alaya_search_results` with the shared handle, render, assert the bucket labels `le="0"`, `le="1"`, ..., `le="100"` are present. Guards against bucket override misconfiguration.

### Integration verification (manual, post-deploy)

1. Inside cluster: `kubectl exec -n mcp alaya-server-<pod> -- wget -qO- http://localhost:3001/metrics` — check text format parses and contains expected metric names.
2. From vmagent: `kubectl exec -n monitoring vmagent-<pod> -- wget -qO- http://alaya-server.mcp:3001/metrics` — confirms network policy allows monitoring namespace to scrape mcp namespace.
3. In Grafana (via VictoriaMetrics datasource):
   - `sum(rate(alaya_requests_total[5m])) by (op)` — request rate per op
   - `histogram_quantile(0.95, rate(alaya_request_duration_seconds_bucket[5m]))` — P95 latency
   - `rate(alaya_embedding_cache_hits_total[5m]) / (rate(alaya_embedding_cache_hits_total[5m]) + rate(alaya_embedding_cache_misses_total[5m]))` — cache hit ratio
   - `alaya_service_channel_depth` — current backpressure
   Should all return sensible values after 5 minutes of traffic.

### Not tested

- Internal behavior of `metrics` / `metrics-exporter-prometheus` (tested upstream)
- Cardinality limits (all labels are static enumerations)
- Prometheus text format compliance (the exporter handles it)

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Forgetting `run_upkeep()` → histograms render stale | Low | Histogram data silently wrong | Upkeep task on axum runtime; covered in unit test via manual upkeep call |
| Label cardinality explosion via unintended dynamic labels | Low | VictoriaMetrics memory pressure | Code review + fixed label enumerations; no user-supplied labels |
| Metrics recording adds latency to hot path | Very low | P99 latency regression | Atomic counters are ~10ns per op vs ~200ms request baseline; negligible |
| `PrometheusHandle::render()` slow under heavy cardinality | Low | Scrape timeout | 30s scrape interval; cardinality is bounded by fixed label sets (~60 series total) |
| `install_recorder()` called twice in parallel unit tests | High if naive | Test suite flakes | Tests must install the recorder once via `OnceLock` or use `#[serial]` from `serial_test`. See Testing section. |
| Adding a new `AlayaError` variant silently gets a `""` kind label | Medium | Incomplete error categorization in metrics | `error_kind()` uses exhaustive match; compiler forces update |

## Rollback

Rolling back the code is a single commit revert. Rolling back the k8s manifest is a Flux auto-apply after the revert lands. No data migration, no schema change, no stateful dependency — zero-risk rollback.

If metrics cause unexpected behavior (unlikely, but possible) the kill switch is removing the VMServiceScrape CRD — vmagent stops scraping immediately, the `/metrics` endpoint remains harmless on the server.

## Definition of Done

1. `cargo build --workspace`, `cargo clippy -D warnings`, `cargo fmt --check`, `cargo test --workspace` all pass
2. WASM gate passes for `alaya-types` and `alaya-core` (no impact expected)
3. `curl http://localhost:3001/metrics` on a locally-running `alaya-server` returns Prometheus text format with all documented metric names
4. Unit tests for handler rendering, error kind mapping, and bucket application pass
5. Deployed to lab k3s `mcp` namespace via ghcr.io + Flux
6. VMServiceScrape CRD applied to `monitoring` namespace
7. vmagent successfully scraping (verified via VictoriaMetrics query UI)
8. At least one Grafana query returning non-empty data for each metric family
9. Memory saved documenting the final metric catalog and any deployment gotchas
