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
        let parse = |key: &str, v: String| {
            v.parse::<url::Url>()
                .map_err(|e| format!("{key} is not a valid URL: {e}"))
        };
        match (url, api_key, metrics_url) {
            (None, None, None) => Ok(None),
            (Some(u), Some(k), Some(m)) => {
                let url = parse("LB_URL", u)?;
                validate_keyed_url("LB_URL", &url)?;
                Ok(Some(LbConfig {
                    url,
                    api_key: k,
                    metrics_url: parse("METRICS_URL", m)?,
                }))
            }
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
    if h == "localhost" || h.ends_with(".svc") || h.ends_with(".internal") {
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

/// A URL a credential travels to (ALAYA_URL bearer, LB_URL x-api-key): https
/// anywhere, plain http only cluster-local — otherwise refuse startup.
/// METRICS_URL carries no key and is exempt. The refusal names the host,
/// never the value (it may carry userinfo).
fn validate_keyed_url(var: &str, url: &url::Url) -> Result<(), String> {
    match url.scheme() {
        "https" => Ok(()),
        "http" if is_cluster_local(url) => Ok(()),
        scheme => Err(format!(
            "{var}: {scheme}://{} is neither https nor a cluster-local http endpoint; \
             a key must not travel in the clear",
            url.host_str().unwrap_or("")
        )),
    }
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
        if !oidc_issuer.starts_with("https://") {
            return Err("CONSOLE_OIDC_ISSUER must be https".into());
        }

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
        validate_keyed_url("ALAYA_URL", &alaya_url)?;

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
    fn keyed_urls_refuse_plaintext_off_cluster() {
        let ok = |u: &str| validate_keyed_url("LB_URL", &u.parse().unwrap()).is_ok();
        assert!(ok("https://lb.example.com"));
        assert!(ok("http://anthropic-lb.mcp.svc:8082"));
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
        let err = validate_keyed_url("LB_URL", &"http://u:s3cret@lb.example.com".parse().unwrap())
            .unwrap_err();
        assert!(
            err.contains("lb.example.com") && !err.contains("s3cret"),
            "{err}"
        );
        // Through the group: the keyed URL is checked, the keyless one is not.
        let k = || Some("k".repeat(40));
        let off = || Some("http://metrics.example.com:8428".to_string());
        assert!(LbConfig::from_parts(Some("http://lb.example.com".into()), k(), off()).is_err());
        assert!(LbConfig::from_parts(Some("http://lb:8082".into()), k(), off()).is_ok());
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
    }
}
