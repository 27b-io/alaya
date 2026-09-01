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
}

// Never derive Debug for Config — it holds two credentials.
impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("listen_addr", &self.listen_addr)
            .field("public_url", &self.public_url.as_str())
            .field("oidc_issuer", &self.oidc_issuer)
            .field("oidc_client_id", &self.oidc_client_id)
            .field("allowed_subjects", &self.allowed_subjects)
            .field("alaya_url", &self.alaya_url.as_str())
            .finish_non_exhaustive()
    }
}

fn required(key: &str) -> Result<String, String> {
    match std::env::var(key) {
        Ok(v) if !v.trim().is_empty() => Ok(v.trim().to_string()),
        _ => Err(format!("{key} is required and must be non-empty")),
    }
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
        if public_url.scheme() != "https" && public_url.scheme() != "http" {
            return Err("CONSOLE_PUBLIC_URL must be http(s)".into());
        }
        if public_url.host_str().is_none() {
            return Err("CONSOLE_PUBLIC_URL must have a host".into());
        }

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
    fn debug_never_prints_credentials() {
        let cfg = Config {
            listen_addr: "0.0.0.0:3002".into(),
            public_url: "https://console.test".parse().unwrap(),
            oidc_issuer: "https://id.test".into(),
            oidc_client_id: "console".into(),
            oidc_client_secret: "SECRET_VALUE".into(),
            allowed_subjects: vec!["sub1".into()],
            session_secret: b"0123456789abcdef0123456789abcdef".to_vec(),
            alaya_url: "http://alaya-server.mcp.svc:3001".parse().unwrap(),
            alaya_api_key: "BEARER_VALUE".into(),
        };
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("SECRET_VALUE"));
        assert!(!dbg.contains("BEARER_VALUE"));
    }
}
