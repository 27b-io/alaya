//! Fail-closed configuration. Every credential and endpoint is required —
//! a missing or malformed value refuses startup (AC10), never degrades to an
//! open console.

use std::fmt;

/// Origin (scheme://host[:port]) of an absolute URL, for the POST Origin
/// check and the pinned redirect_uri.
pub fn origin_of(url: &url::Url) -> String {
    let mut o = format!("{}://{}", url.scheme(), url.host_str().unwrap_or(""));
    if let Some(port) = url.port() {
        o.push_str(&format!(":{port}"));
    }
    o
}

pub struct Config {
    pub listen_addr: String,
    /// Externally visible base URL (tailnet HTTPS). The OIDC redirect_uri is
    /// pinned to `{public_url}/auth/callback` — never derived from request
    /// headers.
    pub public_url: url::Url,
    pub oidc_issuer: String,
    pub oidc_client_id: String,
    pub oidc_client_secret: String,
    /// Default-deny subject allowlist: only these OIDC `sub` values may hold
    /// a session. Comma-separated, must be non-empty (an empty allowlist is a
    /// misconfiguration, not a policy).
    pub allowed_subjects: Vec<String>,
    /// At least 32 bytes; expanded through SHA-512 into the AES-GCM key for
    /// the private (encrypted) cookie jar (see `AppState::new`).
    pub session_secret: Vec<u8>,
    pub alaya_url: url::Url,
    pub alaya_api_key: String,
    /// anthropic-lb monitoring module; `None` = module disabled.
    pub lb: Option<LbConfig>,
}

/// anthropic-lb read-only module. Optional as a GROUP: a
/// deploy-ordering gap (the image rolls before the three variables are set,
/// or the reverse) leaves the Ālaya module up and the LB card reading "not
/// configured" instead of taking the whole console down. A half-set group
/// is a misconfiguration and refuses startup like any other credential.
pub struct LbConfig {
    /// e.g. `http://anthropic-lb.mcp.svc:8082`
    pub url: url::Url,
    /// Operator client key, sent server-side as `x-api-key` — the LB admin
    /// gate does not accept `Authorization: Bearer`. Never reaches the
    /// browser.
    pub api_key: String,
    /// Prometheus-compatible query API for the 7-day budget-burn history,
    /// read off the LB's own fleet-wide gauges; the console keeps no history
    /// of its own.
    pub metrics_url: url::Url,
}

impl LbConfig {
    /// All three set → enabled; none set → disabled; anything else → error
    /// naming the missing variables.
    pub fn from_parts(
        url: Option<String>,
        api_key: Option<String>,
        metrics_url: Option<String>,
    ) -> Result<Option<Self>, String> {
        // One gate for both URLs, keyless included. Everything it refuses
        // would otherwise boot clean and fail at the first render as a blank
        // history section — which reads as an LB outage, sending the
        // operator after the wrong system. AC10 is refuse-at-startup.
        let parse = |key: &str, v: String| {
            let url: url::Url = v
                .parse()
                .map_err(|e| format!("{key} is not a valid URL: {e}"))?;
            validate_upstream_url(key, &url)?;
            // `join` (lb.rs) appends the endpoint path to `Url::as_str()`,
            // which carries any query and fragment with it — a base of
            // `…/select?token=x` would request `/select?token=x/api/v1/…`
            // and leave the card permanently dark behind a non-2xx. Refuse
            // at boot rather than defer it to the first render.
            if url.query().is_some() || url.fragment().is_some() {
                return Err(format!("{key} must have no query string or fragment"));
            }
            Ok(url)
        };
        match (url, api_key, metrics_url) {
            (None, None, None) => Ok(None),
            (Some(u), Some(k), Some(m)) => Ok(Some(LbConfig {
                url: parse("LB_URL", u)?,
                api_key: k,
                metrics_url: parse("METRICS_URL", m)?,
            })),
            (u, k, m) => {
                let missing: Vec<&str> = [
                    ("LB_URL", u.is_none()),
                    ("LB_API_KEY", k.is_none()),
                    ("METRICS_URL", m.is_none()),
                ]
                .into_iter()
                .filter_map(|(name, absent)| absent.then_some(name))
                .collect();
                Err(format!(
                    "anthropic-lb module is half-configured — set all of LB_URL, LB_API_KEY, METRICS_URL or none (missing: {})",
                    missing.join(", ")
                ))
            }
        }
    }
}

// Never derive Debug for Config — it holds credentials. Upstream URLs render
// as origin only: userinfo and query strings can carry more.
impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("listen_addr", &self.listen_addr)
            .field("public_url", &origin_of(&self.public_url))
            .field("oidc_issuer", &self.oidc_issuer)
            .field("oidc_client_id", &self.oidc_client_id)
            .field("allowed_subjects", &self.allowed_subjects)
            .field("alaya_url", &origin_of(&self.alaya_url))
            .field("lb_url", &self.lb.as_ref().map(|l| origin_of(&l.url)))
            .field(
                "metrics_url",
                &self.lb.as_ref().map(|l| origin_of(&l.metrics_url)),
            )
            .finish_non_exhaustive()
    }
}

/// https everywhere; plaintext http exists only for loopback local dev (a
/// non-loopback http URL would also silently disable the Secure cookie flag
/// — refuse instead).
fn validate_public_url(public_url: &url::Url) -> Result<(), String> {
    let Some(host) = public_url.host_str() else {
        return Err("CONSOLE_PUBLIC_URL must have a host".into());
    };
    // host_str() keeps brackets on IPv6 literals ("[::1]") — strip for parse.
    let bare_host = host.trim_start_matches('[').trim_end_matches(']');
    let is_loopback = bare_host == "localhost"
        || bare_host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false);
    match public_url.scheme() {
        "https" => Ok(()),
        "http" if is_loopback => Ok(()),
        "http" => {
            Err("CONSOLE_PUBLIC_URL must be https (http is allowed only for loopback dev)".into())
        }
        _ => Err("CONSOLE_PUBLIC_URL must be http(s)".into()),
    }
}

/// Hosts a key may reach over plain http. Mirrors alaya-server's rule for
/// SUMMARY_URL / JUDGE_URL: DNS-only cluster names, or a real loopback /
/// private IP literal — `127.0.0.1.evil.com` parses as neither.
fn host_is_private(h: &str) -> bool {
    // Both suffixes are END-anchored: that is what stops
    // `evil.svc.attacker.com` matching, so neither may become a substring
    // test. A non-default cluster domain needs its literal added here.
    if h == "localhost"
        || h.ends_with(".svc")
        || h.ends_with(".svc.cluster.local")
        || h.ends_with(".internal")
    {
        return true;
    }
    match h.parse::<std::net::IpAddr>() {
        Ok(std::net::IpAddr::V4(v4)) => v4.is_loopback() || v4.is_private(),
        // An IPv4-mapped literal (`::ffff:10.0.0.1`) is the v4 address it
        // wraps — `Ipv6Addr::is_loopback` is false for `::ffff:127.0.0.1`,
        // so judge the mapped address or a mapped loopback reads as public.
        // ULA (`fc00::/7`) is v6's private range; without it a v6-native
        // cluster is pushed onto DNS names for no security gain.
        Ok(std::net::IpAddr::V6(v6)) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.is_loopback() || v4.is_private(),
            None => v6.is_loopback() || v6.is_unique_local(),
        },
        Err(_) => false,
    }
}

fn is_cluster_local(url: &url::Url) -> bool {
    let Some(h) = url.host_str() else {
        return false;
    };
    let h = h.trim_start_matches('[').trim_end_matches(']');
    host_is_private(h) || (h.parse::<std::net::IpAddr>().is_err() && !h.contains('.'))
}

/// Transport rule for the console's data upstreams (ALAYA_URL bearer,
/// LB_URL x-api-key, METRICS_URL): https anywhere, plain http only
/// cluster-local — otherwise refuse startup. Deliberately STRICTER than
/// alaya-server's `check_credential_transport`, which still returns early
/// for a keyless URL: the two gates agree on every keyed URL and diverge
/// only here, on purpose.
///
/// METRICS_URL is in scope despite carrying no key. A credential is not the
/// only thing worth a TLS hop: off-cluster plaintext leaves the budget
/// history readable AND rewritable in flight, and a monitoring chart an
/// on-path attacker can rewrite is worse than no chart — the operator acts
///
/// The IdP is NOT on this path — `validate_issuer` holds it to https with
/// no cluster-local exemption at all.
///
/// The refusal names the host, never the value (it may carry userinfo).
fn validate_upstream_url(var: &str, url: &url::Url) -> Result<(), String> {
    match url.scheme() {
        "https" => Ok(()),
        "http" if is_cluster_local(url) => Ok(()),
        // Two different operator mistakes, two different remedies. Telling
        // someone who typed `vmselect.monitoring` to "use TLS" sends them
        // after a certificate when they needed the `.svc` suffix.
        "http" => Err(format!(
            "{var}: http://{} is not cluster-local; use https, or the \
             in-cluster form <service>.<namespace>.svc",
            url.host_str().unwrap_or("")
        )),
        scheme => Err(format!("{var} must be an http(s) URL (got {scheme})")),
    }
}

/// The IdP is stricter than every other upstream: https only, with no
/// cluster-local exemption — and no userinfo.
///
/// The userinfo rule is what keeps a credential out of the logs. The issuer
/// is the one credential-shaped value that reaches a log verbatim: `Config`'s
/// Debug prints it (`?config` at startup, unconditionally) and `OidcRp` logs
/// it on every IdP failure. Refusing the shape at boot is one check;
/// redacting at each log site is a list that grows and will miss one.
///
/// Parsed, not string-matched — '@' is legal in a path, and only the parser
/// decides which bytes are userinfo.
fn validate_issuer(issuer: &str) -> Result<(), String> {
    if !issuer.starts_with("https://") {
        return Err("CONSOLE_OIDC_ISSUER must be https".into());
    }
    let url: url::Url = issuer
        .parse()
        .map_err(|e| format!("CONSOLE_OIDC_ISSUER is not a valid URL: {e}"))?;
    if !url.username().is_empty() || url.password().is_some() {
        return Err("CONSOLE_OIDC_ISSUER must not carry userinfo".into());
    }
    Ok(())
}

fn required(key: &str) -> Result<String, String> {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => Ok(v.trim().to_string()),
        _ => Err(format!("{key} is required and must be non-empty")),
    }
}

/// Unset and empty are the same thing: absent.
fn optional(key: &str) -> Option<String> {
    std::env::var(key)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

impl Config {
    pub fn redirect_uri(&self) -> String {
        format!(
            "{}/auth/callback",
            self.public_url.as_str().trim_end_matches('/')
        )
    }

    pub fn public_origin(&self) -> String {
        origin_of(&self.public_url)
    }

    pub fn subject_allowed(&self, sub: &str) -> bool {
        self.allowed_subjects.iter().any(|s| s == sub)
    }

    /// Read config from the environment. Any error here must abort startup.
    pub fn from_env() -> Result<Self, String> {
        let public_url: url::Url = required("CONSOLE_PUBLIC_URL")?
            .parse()
            .map_err(|e| format!("CONSOLE_PUBLIC_URL is not a valid URL: {e}"))?;
        validate_public_url(&public_url)?;

        let oidc_issuer = required("CONSOLE_OIDC_ISSUER")?;
        validate_issuer(&oidc_issuer)?;

        let allowed_subjects: Vec<String> = required("CONSOLE_ALLOWED_SUBJECTS")?
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if allowed_subjects.is_empty() {
            return Err("CONSOLE_ALLOWED_SUBJECTS must list at least one subject".into());
        }

        let session_secret = required("CONSOLE_SESSION_SECRET")?.into_bytes();
        if session_secret.len() < 32 {
            return Err("CONSOLE_SESSION_SECRET must be at least 32 bytes".into());
        }

        let alaya_url: url::Url = required("ALAYA_URL")?
            .parse()
            .map_err(|e| format!("ALAYA_URL is not a valid URL: {e}"))?;
        validate_upstream_url("ALAYA_URL", &alaya_url)?;

        Ok(Config {
            listen_addr: std::env::var("CONSOLE_LISTEN_ADDR")
                .unwrap_or_else(|_| "0.0.0.0:3002".to_string()),
            public_url,
            oidc_issuer,
            oidc_client_id: required("CONSOLE_OIDC_CLIENT_ID")?,
            oidc_client_secret: required("CONSOLE_OIDC_CLIENT_SECRET")?,
            allowed_subjects,
            session_secret,
            alaya_url,
            alaya_api_key: required("ALAYA_API_KEY")?,
            lb: LbConfig::from_parts(
                optional("LB_URL"),
                optional("LB_API_KEY"),
                optional("METRICS_URL"),
            )?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_strips_path_and_keeps_explicit_port() {
        let u: url::Url = "https://console.example.com/some/path".parse().unwrap();
        assert_eq!(origin_of(&u), "https://console.example.com");
        let u: url::Url = "http://localhost:3002/".parse().unwrap();
        assert_eq!(origin_of(&u), "http://localhost:3002");
    }

    #[test]
    fn public_url_requires_https_except_loopback() {
        let ok = |u: &str| validate_public_url(&u.parse().unwrap()).is_ok();
        assert!(ok("https://console.tail1234.ts.net"));
        assert!(ok("http://localhost:3002"));
        assert!(ok("http://127.0.0.1:3002"));
        assert!(ok("http://[::1]:3002"));
        // Plaintext http on a routable host would ship the session cookie
        // without Secure — must refuse startup.
        assert!(!ok("http://console.tail1234.ts.net"));
        assert!(!ok("http://192.168.1.10:3002"));
        assert!(!ok("ftp://console.example"));
    }

    #[test]
    fn upstream_urls_refuse_plaintext_off_cluster() {
        let ok = |u: &str| validate_upstream_url("LB_URL", &u.parse().unwrap()).is_ok();
        assert!(ok("https://lb.example.com"));
        assert!(ok("http://anthropic-lb.mcp.svc:8082"));
        // The fully-qualified Service name is the form most k8s docs show.
        assert!(ok("http://anthropic-lb.mcp.svc.cluster.local:8082"));
        // Still end-anchored: a public domain wearing an `svc` label is not
        // cluster-local.
        assert!(!ok("http://evil.svc.attacker.com:8082"));
        assert!(!ok(
            "http://anthropic-lb.mcp.svc.cluster.local.evil.com:8082"
        ));
        assert!(ok("http://anthropic-lb:8082"));
        assert!(ok("http://10.0.0.5:8082"));
        assert!(ok("http://localhost:8082"));
        // IPv6 literals: loopback, ULA and IPv4-mapped private addresses are
        // as cluster-local as their v4 spellings. A mapped PUBLIC v4 is not,
        // and a global v6 parses as an IP so it never reaches the
        // single-label fallback.
        assert!(ok("http://[::1]:8082"));
        assert!(ok("http://[fd00::1]:8082"));
        assert!(ok("http://[::ffff:10.0.0.5]:8082"));
        assert!(!ok("http://[::ffff:93.184.216.34]:8082"));
        assert!(!ok("http://[2606:4700::1111]:8082"));
        assert!(!ok("http://lb.example.com:8082"));
        assert!(!ok("http://127.0.0.1.evil.com:8082"));
        assert!(!ok("ftp://anthropic-lb"));
        // The refusal goes to pod logs: name the host, never the userinfo.
        let err =
            validate_upstream_url("LB_URL", &"http://u:s3cret@lb.example.com".parse().unwrap())
                .unwrap_err();
        assert!(
            err.contains("lb.example.com") && !err.contains("s3cret"),
            "{err}"
        );
        // Through the group: BOTH URLs are gated, keyless included. One
        // shared gate in `parse` — the lines differ only in which key
        // reaches it, so each pins a different variable.
        let k = || Some("k".repeat(40));
        let off = || Some("http://metrics.example.com:8428".to_string());
        let on = || Some("http://vm.mcp.svc:8428".to_string());
        assert!(LbConfig::from_parts(Some("http://lb.example.com".into()), k(), on()).is_err());
        assert!(LbConfig::from_parts(Some("http://lb:8082".into()), k(), off()).is_err());
        assert!(LbConfig::from_parts(Some("http://lb:8082".into()), k(), on()).is_ok());
    }

    #[test]
    fn issuer_refuses_userinfo_and_plaintext() {
        assert!(validate_issuer("https://id.test").is_ok());
        assert!(validate_issuer("https://id.test/realms/ops").is_ok());
        assert!(validate_issuer("http://id.test").is_err());
        // The refusal exists so a credential never reaches `?config` at boot
        // or an IdP warning; both print the issuer verbatim.
        let err = validate_issuer("https://console:s3cret@id.test").unwrap_err();
        assert!(err.contains("userinfo") && !err.contains("s3cret"), "{err}");
        assert!(validate_issuer("https://console@id.test").is_err());
        // '@' in a path is not userinfo — only the parser can tell.
        assert!(validate_issuer("https://id.test/a@b").is_ok());
    }

    #[test]
    fn debug_never_prints_credentials() {
        let cfg = Config {
            listen_addr: "0.0.0.0:3002".into(),
            public_url: "https://console.test".parse().unwrap(),
            oidc_issuer: "https://id.test".into(),
            oidc_client_id: "console".into(),
            oidc_client_secret: "SECRET_VALUE".into(),
            allowed_subjects: vec!["sub1".into()],
            session_secret: b"0123456789abcdef0123456789abcdef".to_vec(),
            // Userinfo and query strings are part of `Url::as_str()` —
            // Debug must not print them either.
            alaya_url: "http://svc:URL_USERINFO_VALUE@alaya-server.mcp.svc:3001"
                .parse()
                .unwrap(),
            alaya_api_key: "BEARER_VALUE".into(),
            lb: Some(LbConfig {
                url: "http://anthropic-lb.mcp.svc:8082".parse().unwrap(),
                api_key: "LB_KEY_VALUE".into(),
                metrics_url: "http://metrics.test:8428/select?token=URL_QUERY_VALUE"
                    .parse()
                    .unwrap(),
            }),
        };
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("SECRET_VALUE"));
        assert!(!dbg.contains("BEARER_VALUE"));
        assert!(!dbg.contains("LB_KEY_VALUE"));
        assert!(!dbg.contains("URL_USERINFO_VALUE"), "{dbg}");
        assert!(!dbg.contains("URL_QUERY_VALUE"), "{dbg}");
        assert!(dbg.contains("anthropic-lb.mcp.svc"));
        assert!(dbg.contains("metrics.test:8428"));
    }

    #[test]
    fn lb_config_is_all_or_nothing() {
        let u = || Some("http://lb:8082".to_string());
        let k = || Some("k".repeat(40));
        let m = || Some("http://vm:8428".to_string());
        assert!(LbConfig::from_parts(None, None, None).unwrap().is_none());
        assert!(LbConfig::from_parts(u(), k(), m()).unwrap().is_some());
        // Half-configured refuses startup and names what is missing.
        // (`.err()` rather than `unwrap_err()`: LbConfig deliberately has no
        // Debug impl — it holds the operator key.)
        let err = LbConfig::from_parts(u(), None, m())
            .err()
            .expect("half-configured must refuse");
        assert!(err.ends_with("(missing: LB_API_KEY)"), "{err}");
        let err = LbConfig::from_parts(None, k(), None)
            .err()
            .expect("half-configured must refuse");
        assert!(err.ends_with("(missing: LB_URL, METRICS_URL)"), "{err}");
        // A present-but-garbage URL is an error, not a silent disable.
        assert!(LbConfig::from_parts(Some("not a url".into()), k(), m()).is_err());
        // METRICS_URL carries no key, but it still has to be fetchable —
        // a non-http(s) scheme would otherwise blank the history section.
        let err = LbConfig::from_parts(u(), k(), Some("ftp://metrics.test".into()))
            .err()
            .expect("ftp METRICS_URL must be refused");
        assert!(
            err.starts_with("METRICS_URL must be an http(s) URL"),
            "{err}"
        );
        // A refused URL carrying userinfo names the host and never the
        // credential — the refusal goes to pod logs.
        let err = LbConfig::from_parts(
            u(),
            k(),
            Some("http://ops:hunter2@metrics.example.com:9090".into()),
        )
        .err()
        .expect("plaintext METRICS_URL to a public host must be refused");
        assert!(
            err.contains("metrics.example.com") && !err.contains("hunter2"),
            "{err}"
        );
        // `join` appends the endpoint path to `as_str()`, so a base carrying
        // a query would request `/select?token=x/api/v1/query_range` and
        // leave the card dark behind a non-2xx. Refused at boot instead.
        let err = LbConfig::from_parts(u(), k(), Some("http://vm:8428/select?token=x".into()))
            .err()
            .expect("METRICS_URL with a query must be refused");
        assert!(err.contains("query string or fragment"), "{err}");
    }
}
