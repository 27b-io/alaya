//! OpenTelemetry OTLP tracing setup.
//!
//! If `OTEL_EXPORTER_OTLP_ENDPOINT` is set, configures OTLP HTTP exporter.
//! If not set, tracing goes to stderr only.
//!
//! Both layers filter on `RUST_LOG`, so the OTLP exporter sees exactly the
//! spans the console does. The console alone also logs the SDK's own warnings
//! (`console_layer`).

use opentelemetry::trace::TracerProvider;
use opentelemetry_otlp::{SpanExporter, WithHttpConfig};
use opentelemetry_sdk::{
    Resource,
    trace::{Sampler, SdkTracerProvider},
};
use std::sync::OnceLock;
use tracing_subscriber::{
    EnvFilter, Layer,
    filter::{FilterExt, Targets},
    layer::SubscriberExt,
    registry::LookupSpan,
    util::SubscriberInitExt,
};

/// Global tracer provider — stored here so `shutdown_tracing()` can flush it.
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// Default log filter. `alaya_server=info` keeps the request span (it takes
/// this crate's target, see `request_span`) and so every trace's root;
/// `tower_http=info` governs only tower-http's own events.
pub(crate) const DEFAULT_FILTER: &str =
    "alaya_server=info,alaya_core=info,alaya_backends=info,tower_http=info";

/// `RUST_LOG`, or `DEFAULT_FILTER` when it is unset or does not parse.
fn env_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER))
}

/// The console: `RUST_LOG`, plus `opentelemetry_sdk` at warn whatever it says.
/// The SDK reports a failed span export (`BatchSpanProcessor.ExportError`,
/// carrying the status code or "network error", never the URL or a header)
/// and dropped spans only as events on that target, and neither the deployed
/// nor the documented `RUST_LOG` names it, so without this floor a collector
/// that is down or refuses every batch loses them all in silence. A union, so
/// a `RUST_LOG` asking for more of the SDK still gets it. Console only: the
/// OTLP layer filters on `RUST_LOG` alone, so a failure report never reaches
/// the exporter it is about.
fn console_layer<S>() -> impl Layer<S>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    let sdk_warnings = Targets::new().with_target("opentelemetry_sdk", tracing::Level::WARN);
    tracing_subscriber::fmt::layer()
        .with_target(true)
        .with_filter(env_filter().or(sdk_warnings))
}

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
/// Panics when `checked_otlp_endpoint` refuses a parseable endpoint the
/// credential would cross in plaintext — see it for which pair is checked and
/// why. An endpoint that does not parse never stops boot.
pub fn init_tracing() {
    // The generic var is the on-switch for the whole arm, so a
    // signal-specific-only config exports nothing at all (and leaks nothing).
    // Raw, not trimmed, from here on: `opentelemetry-otlp` reads these with
    // `std::env::var`, so a trimmed copy would certify a string it never dials.
    if let Some(generic_endpoint) = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        // Checked before the exporter is built, so a refusal means the process
        // never starts rather than starting and posting the token.
        let endpoint = match checked_otlp_endpoint(
            &generic_endpoint,
            std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
                .ok()
                .as_deref(),
            std::env::var("OTEL_EXPORTER_OTLP_HEADERS").ok().as_deref(),
            std::env::var("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
                .ok()
                .as_deref(),
        ) {
            Ok(OtlpPlan::Export(endpoint)) => endpoint,
            Ok(OtlpPlan::Skip(var)) => {
                tracing_subscriber::registry().with(console_layer()).init();
                tracing::warn!(
                    "{var} is not a URI the OTLP exporter can parse; export skipped, tracing to stderr only"
                );
                return;
            }
            Err(e) => panic!("{e}"),
        };

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

        let exporter = match SpanExporter::builder()
            .with_http()
            .with_http_client(otlp_http_client())
            .build()
        {
            Ok(e) => e,
            Err(e) => {
                tracing_subscriber::registry().with(console_layer()).init();
                tracing::warn!(
                    "OTLP exporter failed ({}), tracing to stderr only",
                    log_safe_build_error(&e)
                );
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
            .with(console_layer())
            .with(otel_layer.with_filter(env_filter()))
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
        tracing_subscriber::registry().with(console_layer()).init();
    }
}

/// What `init_tracing` does with the OTLP config, after guarding the endpoint
/// the exporter will dial together with the headers it will send.
#[derive(Debug, PartialEq)]
enum OtlpPlan {
    /// Build the exporter. Carries the selected var's raw value, without the
    /// signal path, for the origin log line only.
    Export(String),
    /// Build nothing and trace to stderr. Carries the selected var's name.
    Skip(&'static str),
}

/// The plan for the selected endpoint and headers. The pair is selected from the same four raw values
/// `opentelemetry-otlp` 0.33 reads, by its own rules (`resolve_http_endpoint`,
/// `read_env_var` and `build_client` in `exporter/`):
///
/// - `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` wins when it is set and non-empty,
///   parsed verbatim (no trimming); otherwise `OTEL_EXPORTER_OTLP_ENDPOINT` is
///   dialled with the signal path appended. Whichever is selected, a value
///   that does not parse as an `http::Uri` fails the exporter build: nothing
///   is dialled, and there is no fallback to the other var or a default.
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
/// Pure over the raw values, for the reason `non_empty_trimmed` gives.
fn checked_otlp_endpoint(
    generic_endpoint: &str,
    traces_endpoint: Option<&str>,
    generic_headers: Option<&str>,
    traces_headers: Option<&str>,
) -> Result<OtlpPlan, String> {
    let has_credential = traces_headers
        .or(generic_headers)
        .is_some_and(|h| !h.trim().is_empty());
    let (var, endpoint, dialled) = match traces_endpoint.filter(|t| !t.is_empty()) {
        Some(t) => ("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", t, t.to_string()),
        None => {
            // `build_endpoint_uri` strips one slash when both sides have it.
            let path = if generic_endpoint.ends_with('/') {
                "v1/traces"
            } else {
                "/v1/traces"
            };
            (
                "OTEL_EXPORTER_OTLP_ENDPOINT",
                generic_endpoint,
                format!("{generic_endpoint}{path}"),
            )
        }
    };
    // `axum::http` is the one `http` crate in the lockfile — the exporter's
    // own parser. `reqwest::Url` would not do: its WHATWG parser strips
    // whitespace and punycodes hosts that `http::Uri` rejects outright, so it
    // would certify a string the exporter refuses. A refused build dials
    // nothing, so refusing boot over it protected nothing and turned a broken
    // tracing config into an outage. With a credential the exporter is not
    // built at all rather than left to refuse: if this emulation of its
    // parse ever drifts, the credential stays unsent instead of reaching an
    // endpoint the transport check never saw. Without one, the exporter's own
    // build error reports it, through `log_safe_build_error`.
    if dialled.parse::<axum::http::Uri>().is_err() {
        if has_credential {
            return Ok(OtlpPlan::Skip(var));
        }
        return Ok(OtlpPlan::Export(endpoint.to_string()));
    }
    crate::check_credential_transport(var, endpoint, has_credential, crate::Transport::Http)?;
    Ok(OtlpPlan::Export(endpoint.to_string()))
}

/// An exporter build error, minus the value it quotes. `opentelemetry-otlp`
/// formats an invalid endpoint as `<VAR>: invalid endpoint '<value>': ...`, and
/// the value may carry userinfo or a query credential (CWE-532), so only the
/// leading setting name survives, and only when it looks like one.
fn log_safe_build_error(e: &opentelemetry_otlp::ExporterBuildError) -> String {
    use opentelemetry_otlp::ExporterBuildError as E;
    match e {
        E::InvalidConfiguration(msg) => {
            let setting = msg
                .split_once(':')
                .map(|(name, _)| name)
                .filter(|name| name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_'))
                .unwrap_or("unknown setting");
            format!("invalid exporter configuration in {setting}")
        }
        E::InternalFailure(_) => "exporter initialization failed".to_string(),
    }
}

/// The exporter's HTTP client. The BatchSpanProcessor runs on a dedicated OS
/// thread and calls futures_executor::block_on() — not a tokio runtime.
/// reqwest's async client panics without a tokio reactor, so this is
/// reqwest::blocking::Client, which works on any thread. No proxy: it dials
/// the endpoint `checked_otlp_endpoint` certified, never an env proxy.
///
/// No redirects either, for the same reason: a redirect's destination is
/// never certified, and reqwest strips `Authorization` on a cross-origin hop
/// but not the custom headers `OTEL_EXPORTER_OTLP_HEADERS` may carry (an
/// `x-api-key`). A 3xx fails that export instead (pinned by
/// `redirecting_collector_never_forwards_the_credential`).
#[allow(clippy::disallowed_methods, reason = "sets .no_proxy()")]
pub(crate) fn otlp_http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .no_proxy()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .expect("failed to build blocking reqwest client")
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
    /// construction. `redirecting_collector_never_forwards_the_credential`
    /// is the test that flushes a live span through `otlp_http_client` on the
    /// batch thread, off any runtime.
    #[test]
    fn installs_without_a_tokio_runtime() {
        assert!(tokio::runtime::Handle::try_current().is_err());
        super::init_tracing();
        tracing::warn!("subscriber is live");
    }

    // The policy — https anywhere, plaintext cluster-local only, nothing to
    // protect means nothing to refuse — is `check_credential_transport`'s and
    // is pinned in `main.rs`. These pin which pair reaches it. Arguments are
    // the four raw env values in the order `init_tracing` reads them; `None`
    // is unset, `Some("")` is set-but-empty.
    use super::OtlpPlan::{Export, Skip};
    use super::checked_otlp_endpoint as guard;

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
        assert_eq!(ok, Ok(Export("https://active.example.com".into())));
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
    fn a_set_signal_specific_endpoint_is_selected_even_when_unparseable() {
        // 0.33's `resolve_http_endpoint` parses a non-empty signal-specific
        // value verbatim and fails the build on error, with no fallback to the
        // generic one. With a bearer the export is skipped under the var's
        // name; without one, the exporter's build fails closed on its own.
        let secret = " https://collector.example.com/v1/traces?api_key=SECRET ";
        assert_eq!(
            guard("https://legacy.example.com", Some(secret), BEARER, None),
            Ok(Skip("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
        );
        assert_eq!(
            guard("https://legacy.example.com", Some(secret), None, None),
            Ok(Export(secret.into()))
        );
    }

    #[test]
    fn a_set_but_empty_signal_specific_endpoint_selects_the_generic_one() {
        // `read_env_var` treats an empty value as unset.
        let err = guard("http://legacy.example.com", Some(""), BEARER, None).unwrap_err();
        assert!(err.contains("OTEL_EXPORTER_OTLP_ENDPOINT:"), "{err}");
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
    fn unparseable_endpoint_with_a_credential_skips_export_instead_of_refusing_boot() {
        // `build_endpoint_uri` fails on each of these and the exporter refuses
        // to build, so nothing is dialled and refusing boot would protect
        // nothing. `reqwest::Url` accepts all three (it strips the padding
        // and the tab, and tolerates the extra slash), so a guard that parsed
        // with it would certify a string the exporter rejects. With a bearer
        // attached the exporter is never built; without one it is not this
        // guard's question, so it cannot drift into a URL validator.
        for bad in [
            " http://collector:4318 ",
            "https:///collector.example.com",
            "https://collector\t.example.com",
        ] {
            assert_eq!(
                guard(bad, None, BEARER, None),
                Ok(Skip("OTEL_EXPORTER_OTLP_ENDPOINT")),
                "{bad:?}"
            );
            assert_eq!(
                guard(bad, None, None, None),
                Ok(Export(bad.into())),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn exporter_build_errors_are_logged_without_the_endpoint() {
        // The real exporter, not a stand-in: it quotes the endpoint it
        // rejects, so what reaches the log must be the setting name only.
        use opentelemetry_otlp::{SpanExporter, WithExportConfig};

        let bad = " https://user:PASS@collector.example.com/v1/traces?api_key=SECRET ";
        let Err(e) = SpanExporter::builder()
            .with_http()
            .with_endpoint(bad)
            .build()
        else {
            panic!("the exporter accepted an unparseable endpoint");
        };
        assert!(e.to_string().contains("SECRET"), "positive control: {e}");
        let safe = super::log_safe_build_error(&e);
        assert_eq!(safe, "invalid exporter configuration in endpoint");

        // The env-var path formats the same way under the var's name.
        let env = opentelemetry_otlp::ExporterBuildError::InvalidConfiguration(format!(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: invalid endpoint '{bad}': invalid format"
        ));
        assert_eq!(
            super::log_safe_build_error(&env),
            "invalid exporter configuration in OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        );
        // A message of another shape keeps nothing.
        let odd = opentelemetry_otlp::ExporterBuildError::InvalidConfiguration(format!(
            "invalid endpoint '{bad}'"
        ));
        assert_eq!(
            super::log_safe_build_error(&odd),
            "invalid exporter configuration in unknown setting"
        );
    }

    #[test]
    fn otlp_is_an_http_transport() {
        // `http://collector:4318` is single-label service DNS — the sanctioned
        // in-cluster path, and the reason `Transport::Http` is the right
        // column: `Redis` would refuse every endpoint the exporter can speak.
        assert!(guard("http://collector:4318", None, BEARER, None).is_ok());
    }

    use std::sync::{Arc, Mutex};

    /// Lowercase: `http_stub` lowercases the head it logs, so a mixed-case
    /// needle would never match.
    const CANARY: &str = "otlp-redirect-canary";

    /// A loopback HTTP/1.1 server that answers every request with `response`
    /// and logs each request head, lowercased, before answering — so a client
    /// that has its response has already been logged.
    fn http_stub(response: String) -> (String, Arc<Mutex<Vec<String>>>) {
        use std::io::{BufRead, BufReader, Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let log = Arc::new(Mutex::new(Vec::new()));
        let seen = log.clone();
        std::thread::spawn(move || {
            for mut stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut head = String::new();
                while reader.read_line(&mut head).unwrap_or(0) > 2 {}
                let head = head.to_ascii_lowercase();
                let body_len = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .and_then(|v| v.trim().parse().ok())
                    .unwrap_or(0);
                let _ = reader.read_exact(&mut vec![0; body_len]);
                seen.lock().unwrap().push(head);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (base, log)
    }

    fn reply(status: &str, extra: &str) -> String {
        format!("HTTP/1.1 {status}\r\n{extra}content-length: 0\r\nconnection: close\r\n\r\n")
    }

    /// Set only by `a_failed_export_reaches_the_console`, in the child it
    /// spawns; unset, `export_failure_child` does nothing.
    const EXPORT_FAILURE_CHILD: &str = "ALAYA_TEST_EXPORT_FAILURE_CHILD";

    /// Lowercase: `http_stub` lowercases the head it logs.
    const EXPORT_CANARY: &str = "otlp-export-canary";

    /// The production boot path in a process of its own: `init_tracing`
    /// installs the process-global subscriber, and the batch processor reports
    /// a failed export on its worker thread, which sees only that global — a
    /// thread-scoped capture (`testlog`) never receives it.
    #[test]
    fn export_failure_child() {
        if std::env::var_os(EXPORT_FAILURE_CHILD).is_none() {
            return;
        }
        super::init_tracing();
        tracing::info_span!("export_probe").in_scope(|| {});
        let provider = super::TRACER_PROVIDER
            .get()
            .expect("the OTLP arm was not taken");
        assert!(
            provider.force_flush().is_err(),
            "the collector's 401 counted as exported"
        );
        // A failure report that reached the exporter would be a new span to
        // export, and fail again: an export loop.
        let again = provider.force_flush();
        assert!(
            again.is_ok(),
            "the failure report was itself exported: {again:?}"
        );
    }

    #[test]
    fn a_failed_export_reaches_the_console() {
        let (base, log) = http_stub(reply("401 Unauthorized", ""));
        let authority = base.strip_prefix("http://").unwrap();
        let (_crate, module) = module_path!().split_once("::").unwrap();
        // The deployed `RUST_LOG` and the documented one, set as an operator
        // would: neither names `opentelemetry_sdk`.
        for filter in [super::DEFAULT_FILTER, "alaya_server=info"] {
            let mut child = std::process::Command::new(std::env::current_exe().unwrap());
            // The SDK defaults, not the shell's: an inherited
            // `OTEL_BSP_MAX_EXPORT_BATCH_SIZE=1` exports before the flush, a
            // sampler or `OTEL_SDK_DISABLED` exports nothing, and either
            // breaks the child's flush assertions.
            for (key, _) in std::env::vars_os() {
                if key.to_string_lossy().starts_with("OTEL_") {
                    child.env_remove(key);
                }
            }
            let out = child
                .args([
                    &format!("{module}::export_failure_child"),
                    "--exact",
                    "--nocapture",
                ])
                .env(EXPORT_FAILURE_CHILD, "1")
                .env("RUST_LOG", filter)
                .env("NO_COLOR", "1")
                .env("OTEL_EXPORTER_OTLP_ENDPOINT", &base)
                .env(
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
                    format!(
                        "http://user:{EXPORT_CANARY}@{authority}/v1/traces?key={EXPORT_CANARY}"
                    ),
                )
                .env(
                    "OTEL_EXPORTER_OTLP_HEADERS",
                    format!("x-api-key={EXPORT_CANARY}"),
                )
                .output()
                .unwrap();
            let console = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(out.status.success(), "{filter}: {console}");
            let report = console
                .lines()
                .find(|l| l.contains("BatchSpanProcessor.ExportError"))
                .unwrap_or_else(|| {
                    panic!("{filter}: no export failure on the console:\n{console}")
                });
            assert!(report.contains("401"), "{filter}: {report}");
            assert!(
                !console.to_ascii_lowercase().contains(EXPORT_CANARY),
                "{filter}: {console}"
            );
        }
        // Control: the canary was really dialled, in the query and a header,
        // so its absence above is the console withholding it.
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 2, "{log:?}");
        for head in log.iter() {
            assert!(
                head.starts_with(&format!("post /v1/traces?key={EXPORT_CANARY} ")),
                "{head}"
            );
            assert!(
                head.contains(&format!("x-api-key: {EXPORT_CANARY}")),
                "{head}"
            );
        }
    }

    /// The one request `log` holds, which must carry the credential.
    fn only_request(log: &Mutex<Vec<String>>) -> String {
        let log = log.lock().unwrap();
        assert_eq!(log.len(), 1, "{log:?}");
        assert!(log[0].contains(&format!("x-api-key: {CANARY}")), "{log:?}");
        log[0].clone()
    }

    /// One real span through the production exporter and batch processor,
    /// carrying a custom credential header, flushed to `endpoint`. Returns
    /// the export's result; dropping the provider shuts it down.
    fn export_one_span(
        endpoint: &str,
        client: reqwest::blocking::Client,
    ) -> opentelemetry_sdk::error::OTelSdkResult {
        use opentelemetry::trace::{Tracer, TracerProvider};
        use opentelemetry_otlp::{SpanExporter, WithExportConfig, WithHttpConfig};

        let exporter = SpanExporter::builder()
            .with_http()
            .with_endpoint(endpoint)
            .with_headers([("x-api-key".to_string(), CANARY.to_string())].into())
            .with_http_client(client)
            .build()
            .unwrap();
        let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
            .with_batch_exporter(exporter)
            .build();
        provider.tracer("redirect-test").in_span("probe", |_| {});
        provider.force_flush()
    }

    #[test]
    fn redirecting_collector_never_forwards_the_credential() {
        let (sink, sink_log) = http_stub(reply("200 OK", ""));
        let (collector, collector_log) = http_stub(reply(
            "307 Temporary Redirect",
            &format!("location: {sink}/stolen\r\n"),
        ));
        let (direct, direct_log) = http_stub(reply("200 OK", ""));

        // Control: an export that reaches nothing — no client, no span, no
        // flush — would pass the sink assertion below for free.
        let direct_export =
            export_one_span(&format!("{direct}/v1/traces"), super::otlp_http_client());
        assert!(direct_export.is_ok(), "{direct_export:?}");
        assert!(only_request(&direct_log).starts_with("post /v1/traces "));

        // The collector was configured, so it may see the key; the origin
        // its 307 names was not, so it must see nothing, and the export fails.
        let redirected =
            export_one_span(&format!("{collector}/v1/traces"), super::otlp_http_client());
        assert!(redirected.is_err(), "a 3xx counted as exported");
        only_request(&collector_log);
        assert!(sink_log.lock().unwrap().is_empty(), "redirect followed");

        // Control: reqwest's default policy follows the same 307 and forwards
        // the key, so the stub would catch a client that regressed to it.
        #[allow(
            clippy::disallowed_methods,
            reason = "test-only control: reqwest's default redirect policy"
        )]
        let following = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap();
        let _ = export_one_span(&format!("{collector}/v1/traces"), following);
        assert!(only_request(&sink_log).starts_with("post /stolen "));
    }
}
