//! OIDC Relying Party — authorization-code flow with PKCE (S256) against
//! id.27b.io.
//!
//! Discovery, same-origin/HTTPS enforcement, the JWKS cache + cooldown and the
//! whole ID-token verify pipeline live in `alaya-oidc` (shared with
//! alaya-server's resource-side verifier). This module keeps what only a
//! relying party decides:
//! - PKCE S256 challenge; `state` and `nonce` are supplied by the login flow
//! - `redirect_uri` is pinned from config; never derived from request headers
//! - `authorization_endpoint` and `token_endpoint` must be present and pass
//!   the same same-origin-https rule as `jwks_uri` (the shared layer checks
//!   only `jwks_uri`, because a resource server never calls the other two)
//! - token exchange with `client_secret_basic` (RFC 6749 §2.3.1 — the scheme
//!   servers MUST support), over the provider's redirect-disabled client
//! - ID-token `aud` is this `client_id` (OIDC Core §3.1.3.7 #3); `nonce` must
//!   match the flow (replay defence)

use alaya_oidc::{Error as OidcError, IssuedClaims, Provider, same_origin_https};
use base64::Engine;
use serde::Deserialize;
use sha2::{Digest, Sha256};

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

impl IssuedClaims for IdClaims {
    fn iss(&self) -> &str {
        &self.iss
    }
}

pub fn pkce_challenge_s256(verifier: &str) -> String {
    let digest = Sha256::digest(verifier.as_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(digest)
}

pub struct OidcRp {
    provider: Provider,
    client_id: String,
    client_secret: String,
    redirect_uri: String,
}

impl OidcRp {
    pub fn new(
        issuer: String,
        client_id: String,
        client_secret: String,
        redirect_uri: String,
    ) -> Self {
        OidcRp {
            provider: Provider::new(&issuer),
            client_id,
            client_secret,
            redirect_uri,
        }
    }

    /// `(authorization_endpoint, token_endpoint)` from discovery, both required
    /// and both same-origin-https with the issuer. Checked together on every
    /// use so a bad `token_endpoint` is refused when the login starts, before
    /// the user is ever redirected.
    async fn endpoints(&self) -> Result<(String, String), OidcError> {
        let disc = self.provider.discovery().await?;
        let authorization = disc.authorization_endpoint.ok_or(OidcError::Invalid(
            "discovery missing authorization_endpoint",
        ))?;
        let token = disc
            .token_endpoint
            .ok_or(OidcError::Invalid("discovery missing token_endpoint"))?;
        same_origin_https(self.provider.issuer(), &authorization)
            .map_err(|_| OidcError::Invalid("authorization_endpoint not same-origin"))?;
        same_origin_https(self.provider.issuer(), &token)
            .map_err(|_| OidcError::Invalid("token_endpoint not same-origin"))?;
        Ok((authorization, token))
    }

    /// Build the authorization redirect for a fresh login flow.
    pub async fn authorize_url(
        &self,
        state: &str,
        nonce: &str,
        pkce_verifier: &str,
    ) -> Result<String, OidcError> {
        let (authorization_endpoint, _) = self.endpoints().await?;
        let mut u: url::Url = authorization_endpoint
            .parse()
            .map_err(|_| OidcError::Invalid("authorization_endpoint form"))?;
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
    ) -> Result<IdClaims, OidcError> {
        let (_, token_endpoint) = self.endpoints().await?;
        let resp = self
            .provider
            .http()
            .post(&token_endpoint)
            .basic_auth(&self.client_id, Some(&self.client_secret))
            .form(&[
                ("grant_type", "authorization_code"),
                ("code", code),
                ("redirect_uri", self.redirect_uri.as_str()),
                ("code_verifier", pkce_verifier),
            ])
            .send()
            .await
            .map_err(|_| OidcError::Invalid("token exchange failed"))?;
        if !resp.status().is_success() {
            return Err(OidcError::Invalid("token exchange rejected"));
        }
        let tokens: TokenResponse = resp
            .json()
            .await
            .map_err(|_| OidcError::Invalid("token response parse"))?;

        // ID token audience is the RP's client_id (OIDC Core §3.1.3.7 #3).
        let claims: IdClaims = self
            .provider
            .verify(&tokens.id_token, &self.client_id)
            .await?;
        // Nonce binds the ID token to this login flow (replay defense).
        if claims.nonce.as_deref() != Some(expected_nonce) {
            return Err(OidcError::Invalid("nonce mismatch"));
        }
        Ok(claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alaya_oidc::Discovery;

    #[test]
    fn pkce_challenge_matches_rfc7636_appendix_b() {
        // RFC 7636 Appendix B test vector.
        assert_eq!(
            pkce_challenge_s256("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
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
        rp.provider.seed_discovery(Discovery {
            issuer: "https://id.test".into(),
            authorization_endpoint: Some("https://id.test/authorize".into()),
            token_endpoint: Some("https://id.test/token".into()),
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

    #[tokio::test]
    async fn cross_origin_rp_endpoint_is_refused_before_redirect() {
        // The shared layer checks only jwks_uri; the RP must apply the same
        // rule to its own endpoints before the user is sent anywhere.
        let rp = OidcRp::new(
            "https://id.test".into(),
            "console".into(),
            "secret".into(),
            "https://console.test/auth/callback".into(),
        );
        rp.provider.seed_discovery(Discovery {
            issuer: "https://id.test".into(),
            authorization_endpoint: Some("https://id.test/authorize".into()),
            token_endpoint: Some("https://evil.test/token".into()),
            jwks_uri: "https://id.test/jwks".into(),
        });
        let err = rp
            .authorize_url("STATE", "NONCE", "VERIFIER")
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "token_endpoint not same-origin");
    }
}
