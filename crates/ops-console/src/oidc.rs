//! OIDC Relying Party — authorization-code flow with PKCE (S256) against
//! id.27b.io, mirroring the hardening posture of alaya-server's resource-side
//! verifier (`crates/alaya-server/src/oidc.rs`):
//!
//! - redirect-following disabled on discovery / JWKS / token fetches
//! - discovery `issuer` must echo the configured issuer (OIDC Core §4.3)
//! - `jwks_uri` and `token_endpoint` must be HTTPS and same-origin with the
//!   issuer (loopback issuer may use http for local dev)
//! - alg allowlist {RS256, ES256}; `none`/HS* rejected before key lookup
//! - JWKS cooldown so an unknown-`kid` flood can't drive unbounded fetches
//! - `redirect_uri` is pinned from config; never derived from request headers
//!
//! Client authentication at the token endpoint is `client_secret_basic`
//! (RFC 6749 §2.3.1 — the scheme servers MUST support).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use base64::Engine;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, RwLock};

const JWKS_COOLDOWN: Duration = Duration::from_secs(30);
const CLOCK_SKEW_LEEWAY_SECS: u64 = 60;

#[derive(Debug)]
pub struct OidcRpError(pub &'static str);

impl std::fmt::Display for OidcRpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Deserialize, Clone)]
struct Discovery {
    issuer: String,
    authorization_endpoint: String,
    token_endpoint: String,
    jwks_uri: String,
}

#[derive(Deserialize, Clone)]
struct Jwk {
    kty: String,
    kid: Option<String>,
    n: Option<String>,
    e: Option<String>,
    x: Option<String>,
    y: Option<String>,
}

#[derive(Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

#[derive(Deserialize)]
struct TokenResponse {
    id_token: String,
}

/// Verified identity claims from the ID token.
#[derive(Deserialize)]
pub struct IdClaims {
    pub sub: String,
    pub iss: String,
    pub nonce: Option<String>,
    pub email: Option<String>,
    pub name: Option<String>,
    pub preferred_username: Option<String>,
}

fn normalize_issuer(s: &str) -> &str {
    s.strip_suffix('/').unwrap_or(s)
}

/// (scheme, host, port) for RFC 6454 same-origin checks, default ports
/// normalized. Same semantics as alaya-server's `origin_of`.
fn origin_of(u: &str) -> Option<(String, String, u16)> {
    let parsed: url::Url = u.parse().ok()?;
    let scheme = parsed.scheme().to_string();
    let default_port: u16 = match scheme.as_str() {
        "https" => 443,
        "http" => 80,
        _ => return None,
    };
    Some((
        scheme,
        parsed.host_str()?.to_ascii_lowercase(),
        parsed.port().unwrap_or(default_port),
    ))
}

fn is_loopback_origin(origin: &(String, String, u16)) -> bool {
    origin.1 == "localhost"
        || origin
            .1
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// Endpoint must be same-origin with the issuer and https (http allowed only
/// for a loopback issuer, so local dev against a local IdP works).
fn same_origin_https(issuer: &str, endpoint: &str) -> Result<(), OidcRpError> {
    let issuer_origin = origin_of(issuer).ok_or(OidcRpError("issuer form"))?;
    let ep_origin = origin_of(endpoint).ok_or(OidcRpError("endpoint form"))?;
    let scheme_ok =
        ep_origin.0 == "https" || (is_loopback_origin(&issuer_origin) && ep_origin.0 == "http");
    if !scheme_ok || ep_origin != issuer_origin {
        return Err(OidcRpError("endpoint not same-origin with issuer"));
    }
    Ok(())
}

pub fn pkce_challenge_s256(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

pub struct OidcRp {
    issuer: String,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
    http: reqwest::Client,
    discovery: RwLock<Option<Discovery>>,
    keys: RwLock<HashMap<String, Jwk>>,
    fetch_gate: Mutex<Instant>,
}

impl OidcRp {
    pub fn new(
        issuer: String,
        client_id: String,
        client_secret: String,
        redirect_uri: String,
    ) -> Self {
        let http = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(10))
            .build()
            .expect("failed to build OIDC http client");
        let seeded = Instant::now()
            .checked_sub(JWKS_COOLDOWN * 2)
            .unwrap_or_else(Instant::now);
        OidcRp {
            issuer: normalize_issuer(&issuer).to_string(),
            client_id,
            client_secret,
            redirect_uri,
            http,
            discovery: RwLock::new(None),
            keys: RwLock::new(HashMap::new()),
            fetch_gate: Mutex::new(seeded),
        }
    }

    async fn discovery(&self) -> Result<Discovery, OidcRpError> {
        if let Some(d) = self.discovery.read().await.clone() {
            return Ok(d);
        }
        let url = format!("{}/.well-known/openid-configuration", self.issuer);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|_| OidcRpError("discovery failed"))?;
        if !resp.status().is_success() {
            return Err(OidcRpError("discovery status"));
        }
        let disc: Discovery = resp
            .json()
            .await
            .map_err(|_| OidcRpError("discovery parse"))?;
        if normalize_issuer(&disc.issuer) != self.issuer {
            return Err(OidcRpError("discovery issuer mismatch"));
        }
        same_origin_https(&self.issuer, &disc.jwks_uri)?;
        same_origin_https(&self.issuer, &disc.token_endpoint)?;
        same_origin_https(&self.issuer, &disc.authorization_endpoint)?;
        *self.discovery.write().await = Some(disc.clone());
        Ok(disc)
    }

    /// Build the authorization redirect for a fresh login flow.
    pub async fn authorize_url(
        &self,
        state: &str,
        nonce: &str,
        pkce_verifier: &str,
    ) -> Result<String, OidcRpError> {
        let disc = self.discovery().await?;
        let mut u: url::Url = disc
            .authorization_endpoint
            .parse()
            .map_err(|_| OidcRpError("authorization_endpoint form"))?;
        u.query_pairs_mut()
            .append_pair("response_type", "code")
            .append_pair("client_id", &self.client_id)
            .append_pair("redirect_uri", &self.redirect_uri)
            .append_pair("scope", "openid profile email")
            .append_pair("state", state)
            .append_pair("nonce", nonce)
            .append_pair("code_challenge", &pkce_challenge_s256(pkce_verifier))
            .append_pair("code_challenge_method", "S256");
        Ok(u.to_string())
    }

    /// Exchange the authorization code, verify the ID token (signature, iss,
    /// aud, exp, nonce) and return the identity claims.
    pub async fn exchange_and_verify(
        &self,
        code: &str,
        pkce_verifier: &str,
        expected_nonce: &str,
    ) -> Result<IdClaims, OidcRpError> {
        let disc = self.discovery().await?;
        let resp = self
            .http
            .post(&disc.token_endpoint)
            .basic_auth(&self.client_id, Some(&self.client_secret))
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", self.redirect_uri.as_str()),
                ("code_verifier", pkce_verifier),
            ])
            .send()
            .await
            .map_err(|_| OidcRpError("token exchange failed"))?;
        if !resp.status().is_success() {
            return Err(OidcRpError("token exchange rejected"));
        }
        let tokens: TokenResponse = resp
            .json()
            .await
            .map_err(|_| OidcRpError("token response parse"))?;

        let claims = self.verify_id_token(&tokens.id_token).await?;
        // Nonce binds the ID token to this login flow (replay defense).
        if claims.nonce.as_deref() != Some(expected_nonce) {
            return Err(OidcRpError("nonce mismatch"));
        }
        Ok(claims)
    }

    async fn verify_id_token(&self, token: &str) -> Result<IdClaims, OidcRpError> {
        let header = decode_header(token).map_err(|_| OidcRpError("bad id_token header"))?;
        if !matches!(header.alg, Algorithm::RS256 | Algorithm::ES256) {
            return Err(OidcRpError("alg not allowed"));
        }
        let kid = header.kid.ok_or(OidcRpError("missing kid"))?;
        let jwk = self.key_for_kid(&kid).await?;
        let decoding_key = build_decoding_key(&jwk, header.alg)?;

        let mut validation = Validation::new(header.alg);
        // ID token audience is the RP's client_id (OIDC Core §3.1.3.7 #3).
        validation.set_audience(&[&self.client_id]);
        validation.validate_aud = true;
        validation.validate_exp = true;
        validation.leeway = CLOCK_SKEW_LEEWAY_SECS;
        validation.set_required_spec_claims(&["exp", "aud"]);

        let data = decode::<IdClaims>(token, &decoding_key, &validation)
            .map_err(|_| OidcRpError("id_token invalid"))?;
        if normalize_issuer(&data.claims.iss) != self.issuer {
            return Err(OidcRpError("iss mismatch"));
        }
        Ok(data.claims)
    }

    async fn key_for_kid(&self, kid: &str) -> Result<Jwk, OidcRpError> {
        if let Some(jwk) = self.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }
        let mut last_fetch = self.fetch_gate.lock().await;
        if let Some(jwk) = self.keys.read().await.get(kid).cloned() {
            return Ok(jwk);
        }
        if last_fetch.elapsed() < JWKS_COOLDOWN {
            return Err(OidcRpError("unknown kid (cooldown)"));
        }
        // Extend the cooldown before fetching: a down IdP must not turn the
        // console into an outbound-fetch amplifier.
        *last_fetch = Instant::now();

        let disc = self.discovery().await?;
        let resp = self
            .http
            .get(&disc.jwks_uri)
            .send()
            .await
            .map_err(|_| OidcRpError("jwks fetch failed"))?;
        if !resp.status().is_success() {
            return Err(OidcRpError("jwks status"));
        }
        let jwks: Jwks = resp.json().await.map_err(|_| OidcRpError("jwks parse"))?;
        let mut map = HashMap::new();
        for jwk in jwks.keys {
            if let Some(k) = jwk.kid.clone() {
                map.insert(k, jwk);
            }
        }
        *self.keys.write().await = map;

        self.keys
            .read()
            .await
            .get(kid)
            .cloned()
            .ok_or(OidcRpError("unknown kid"))
    }
}

fn build_decoding_key(jwk: &Jwk, alg: Algorithm) -> Result<DecodingKey, OidcRpError> {
    match (jwk.kty.as_str(), alg) {
        ("RSA", Algorithm::RS256) => {
            let n = jwk.n.as_deref().ok_or(OidcRpError("rsa n"))?;
            let e = jwk.e.as_deref().ok_or(OidcRpError("rsa e"))?;
            DecodingKey::from_rsa_components(n, e).map_err(|_| OidcRpError("rsa key"))
        }
        ("EC", Algorithm::ES256) => {
            let x = jwk.x.as_deref().ok_or(OidcRpError("ec x"))?;
            let y = jwk.y.as_deref().ok_or(OidcRpError("ec y"))?;
            DecodingKey::from_ec_components(x, y).map_err(|_| OidcRpError("ec key"))
        }
        _ => Err(OidcRpError("alg/key mismatch")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_matches_rfc7636_appendix_b() {
        // RFC 7636 Appendix B test vector.
        assert_eq!(
            pkce_challenge_s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn same_origin_rejects_cross_origin_and_port_forgery() {
        assert!(same_origin_https("https://id.27b.io", "https://id.27b.io/jwks").is_ok());
        assert!(same_origin_https("https://id.27b.io", "https://evil.io/jwks").is_err());
        assert!(same_origin_https("https://id.27b.io", "https://id.27b.io:8443/jwks").is_err());
        assert!(same_origin_https("https://id.27b.io", "http://id.27b.io/jwks").is_err());
        // Loopback issuer may use http endpoints (local dev).
        assert!(same_origin_https("http://localhost:8787", "http://localhost:8787/jwks").is_ok());
    }

    #[tokio::test]
    async fn authorize_url_pins_redirect_and_carries_pkce() {
        // Discovery is pre-seeded so no network is touched.
        let rp = OidcRp::new(
            "https://id.test".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        *rp.discovery.write().await = Some(Discovery {
            issuer: "https://id.test".into(),
            authorization_endpoint: "https://id.test/authorize".into(),
            token_endpoint: "https://id.test/token".into(),
            jwks_uri: "https://id.test/jwks".into(),
        });
        let u = rp
            .authorize_url("STATE", "NONCE", "VERIFIER")
            .await
            .unwrap();
        assert!(u.contains("redirect_uri=https%3A%2F%2Fconsole.test%2Fauth%2Fcallback"));
        assert!(u.contains("state=STATE"));
        assert!(u.contains("nonce=NONCE"));
        assert!(u.contains("code_challenge_method=S256"));
        assert!(
            !u.contains("secret"),
            "client secret must never be in the authorize URL"
        );
    }
}
