//! Shared application state.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::extract::FromRef;
use axum_extra::extract::cookie::Key;
use sha2::Digest;

use crate::alaya::AlayaClient;
use crate::config::Config;
use crate::lb::{LbClient, MetricsClient};
use crate::oidc::OidcRp;

/// anthropic-lb monitoring module upstreams, present only when the module
/// is configured — see `LbConfig`. One value, not two `Option`s: the
/// all-or-nothing group is encoded in the type.
#[derive(Clone)]
pub struct LbModule {
    pub client: LbClient,
    pub metrics: MetricsClient,
}

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub alaya: AlayaClient,
    pub lb: Option<LbModule>,
    pub oidc: Arc<OidcRp>,
    /// AES-GCM key for the private cookie jar, derived from
    /// CONSOLE_SESSION_SECRET at startup.
    key: Key,
    /// Logout revocation: sid → absolute cookie expiry. Stateless cookies
    /// alone make logout advisory (a captured or in-flight-refreshed cookie
    /// would outlive it); revoked sids are rejected until their absolute
    /// expiry, after which the entry is purged.
    // ponytail: in-memory, per-replica — matches the single-replica deploy
    // (deploy/console: replicas 1). Move to shared storage if replicas > 1.
    revoked: Arc<Mutex<HashMap<String, i64>>>,
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
        let lb = config.lb.as_ref().map(|c| LbModule {
            client: LbClient::new(c.url.clone(), c.api_key.clone()),
            metrics: MetricsClient::new(c.metrics_url.clone()),
        });
        let oidc = Arc::new(OidcRp::new(
            config.oidc_issuer.clone(),
            config.oidc_client_id.clone(),
            config.oidc_client_secret.clone(),
            config.redirect_uri(),
        ));
        AppState {
            config,
            alaya,
            lb,
            oidc,
            key,
            revoked: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Cookies are `Secure` whenever the console is served over https —
    /// which is every deployment; plain http exists only for local dev.
    pub fn secure_cookies(&self) -> bool {
        self.config.public_url.scheme() == "https"
    }

    /// Revoke a session id until its absolute expiry (logout).
    pub fn revoke_session(&self, sid: &str, exp: i64) {
        let mut revoked = self.revoked.lock().expect("revocation lock poisoned");
        let now = crate::session::now_epoch();
        revoked.retain(|_, e| *e > now);
        revoked.insert(sid.to_string(), exp);
    }

    /// True if this session id was logged out.
    pub fn is_revoked(&self, sid: &str) -> bool {
        let revoked = self.revoked.lock().expect("revocation lock poisoned");
        revoked
            .get(sid)
            .is_some_and(|e| *e > crate::session::now_epoch())
    }
}

// Lets PrivateCookieJar::from_request_parts find the encryption key.
impl FromRef<AppState> for Key {
    fn from_ref(state: &AppState) -> Self {
        state.key.clone()
    }
}
