//! Shared application state.

use std::sync::Arc;

use axum::extract::FromRef;
use axum_extra::extract::cookie::Key;
use sha2::Digest;

use crate::alaya::AlayaClient;
use crate::config::Config;
use crate::oidc::OidcRp;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub alaya: AlayaClient,
    pub oidc: Arc<OidcRp>,
    /// AES-GCM key for the private cookie jar, derived from
    /// CONSOLE_SESSION_SECRET at startup.
    key: Key,
}

impl AppState {
    pub fn new(config: Config) -> Self {
        // cookie::Key::from requires >= 64 bytes; expand the operator-supplied
        // secret (>= 32 bytes, enforced at config load) through SHA-512 so key
        // material length never depends on how long the secret happens to be.
        let expanded = sha2::Sha512::digest(&config.session_secret);
        let key = Key::from(&expanded);
        let config = Arc::new(config);
        let alaya = AlayaClient::new(config.alaya_url.clone(), config.alaya_api_key.clone());
        let oidc = Arc::new(OidcRp::new(
            config.oidc_issuer.clone(),
            config.oidc_client_id.clone(),
            config.oidc_client_secret.clone(),
            config.redirect_uri(),
        ));
        AppState {
            config,
            alaya,
            oidc,
            key,
        }
    }

    /// Cookies are `Secure` whenever the console is served over https —
    /// which is every deployment; plain http exists only for local dev.
    pub fn secure_cookies(&self) -> bool {
        self.config.public_url.scheme() == "https"
    }
}

// Lets PrivateCookieJar::from_request_parts find the encryption key.
impl FromRef<AppState> for Key {
    fn from_ref(state: &AppState) -> Self {
        state.key.clone()
    }
}
