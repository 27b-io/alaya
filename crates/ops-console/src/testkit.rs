//! Test-only OIDC fixtures: a signing key and an `id_token` minter, so the
//! RP tests can drive `verify_id_token` instead of asserting around it.
//!
//! The P-256 keypair is generated once per test process, never committed. A
//! key that only ever exists in memory needs no secret-scanner carve-out —
//! `detect-private-key` and gitleaks scan this file like any other.

use std::sync::LazyLock;

use aws_lc_rs::signature::{ECDSA_P256_SHA256_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde::Serialize;

pub(crate) const ISSUER: &str = "https://id.test";
pub(crate) const CLIENT_ID: &str = "console";
pub(crate) const KID: &str = "test-ec";

pub(crate) struct TestKey {
    signing: jsonwebtoken::EncodingKey,
    /// Affine coordinates, base64url unpadded — the JWK's `x` and `y`.
    pub x: String,
    pub y: String,
}

pub(crate) static KEY: LazyLock<TestKey> = LazyLock::new(|| {
    let pair =
        EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).expect("generate P-256 key");
    // `jsonwebtoken`'s ES256 signer parses the encoding key as PKCS#8 DER.
    let pkcs8 = pair.to_pkcs8v1().expect("encode P-256 key");
    // Uncompressed X9.62 point: 0x04 || X (32 bytes) || Y (32 bytes).
    let point = pair.public_key().as_ref();
    TestKey {
        signing: jsonwebtoken::EncodingKey::from_ec_der(pkcs8.as_ref()),
        x: URL_SAFE_NO_PAD.encode(&point[1..33]),
        y: URL_SAFE_NO_PAD.encode(&point[33..65]),
    }
});

/// The claims the console's verifier reads. `email` and `name` are skipped
/// when absent so a minted token matches what an IdP that omits them sends,
/// rather than carrying explicit nulls the RP would never see.
#[derive(Serialize)]
struct IdTokenClaims {
    iss: String,
    aud: String,
    exp: u64,
    sub: String,
    nonce: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    email: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

/// Sign an ES256 `id_token` that `verify_id_token` accepts: right issuer,
/// right audience, five minutes of life, `kid` matching the seeded JWKS.
pub(crate) fn mint_id_token(sub: &str, email: Option<&str>, name: Option<&str>) -> String {
    mint_id_token_from(ISSUER, sub, email, name)
}

/// `mint_id_token`, as issued by `iss`.
pub(crate) fn mint_id_token_from(
    iss: &str,
    sub: &str,
    email: Option<&str>,
    name: Option<&str>,
) -> String {
    let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::ES256);
    header.kid = Some(KID.into());
    jsonwebtoken::encode(
        &header,
        &IdTokenClaims {
            iss: iss.into(),
            aud: CLIENT_ID.into(),
            exp: crate::session::now_epoch() as u64 + 300,
            sub: sub.into(),
            nonce: "NONCE".into(),
            email: email.map(str::to_string),
            name: name.map(str::to_string),
        },
        &KEY.signing,
    )
    .expect("mint id_token")
}

/// A loopback IdP that serves discovery and a token endpoint which refuses
/// every code (`invalid_grant`), counting the token requests it receives.
/// Returns the issuer to put in `Config::oidc_issuer`; `same_origin_https`
/// admits its `http` endpoints because the issuer is a loopback origin.
pub(crate) async fn mock_idp() -> (String, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use axum::routing::{get, post};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind mock IdP");
    let issuer = format!("http://{}", listener.local_addr().unwrap());
    let discovery = serde_json::json!({
        "issuer": issuer,
        "authorization_endpoint": format!("{issuer}/authorize"),
        "token_endpoint": format!("{issuer}/token"),
        "jwks_uri": format!("{issuer}/jwks"),
    });
    let token_calls = Arc::new(AtomicUsize::new(0));
    let counter = token_calls.clone();
    let idp = axum::Router::new()
        .route(
            "/.well-known/openid-configuration",
            get(move || std::future::ready(axum::Json(discovery.clone()))),
        )
        .route(
            "/token",
            post(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                std::future::ready((
                    axum::http::StatusCode::BAD_REQUEST,
                    r#"{"error":"invalid_grant"}"#,
                ))
            }),
        );
    tokio::spawn(async move { axum::serve(listener, idp).await.unwrap() });
    (issuer, token_calls)
}

// ─── Client-store fakes ─────────────────────────────────────────────────────
//
// A fake store and a fake metrics proxy for the client-store module's route
// tests: restriction ops and document ops drive the same two.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::Request;
use axum::http::{StatusCode, header};
use serde_json::{Value, json};
use sha2::Digest;

use crate::effect::{self, RecordState};

pub(crate) const STORE_TOKEN: &str = "store-writer-token-secret-value-0123456789abc";
pub(crate) const T0: u64 = 1_791_698_400_000_000;
/// The fake store's document version, in Unix milliseconds.
pub(crate) const V0: u64 = 1_791_698_400_123;

/// Every request a fake saw: (method, path and query, body).
pub(crate) type Log = Arc<Mutex<Vec<(String, String, String)>>>;

/// The store's state as a fake `GET /_clients/admin` answers it.
#[derive(Clone)]
pub(crate) struct World {
    pub records: Vec<(&'static str, RecordState, u64)>,
    pub full: bool,
    /// CF's report: (running, set hash). `None` = no report yet.
    pub cf: Option<(bool, String)>,
    /// The digest of the document CF's report holds.
    pub cf_digest: Option<String>,
    pub stacks: Vec<&'static str>,
    /// The document: its version (0 = none) and digest.
    pub doc: (u64, String),
    /// History rows the store shows a digest for, newest first.
    pub history: Vec<(u64, String)>,
    /// Every version in history, oldest first.
    pub history_versions: Vec<u64>,
}

impl World {
    pub(crate) fn new(records: Vec<(&'static str, RecordState, u64)>) -> Self {
        World {
            records,
            full: true,
            cf: None,
            cf_digest: None,
            stacks: vec!["cf", "k3s"],
            doc: (V0, "0".repeat(64)),
            history: vec![],
            history_versions: vec![],
        }
    }

    pub(crate) fn set_hash(&self) -> String {
        effect::set_hash(self.records.iter().copied(), [])
    }

    /// CF running and holding exactly the served set.
    pub(crate) fn cf_caught_up(mut self) -> Self {
        self.cf = Some((true, self.set_hash()));
        self
    }

    pub(crate) fn get_body(&self) -> Value {
        let body = json!({
            "full": self.full,
            "enabled_ahead": 0,
            "records": self.records.iter().map(|(n, s, v)| json!({"name": n, "state": s.as_str(), "version": v})).collect::<Vec<_>>(),
            "revoked_digests": [],
        })
        .to_string();
        let report = self.cf.as_ref().map(|(_, h)| json!({"version": 0, "digest": self.cf_digest, "records_version": 0, "set_hash": h, "set_complete": true, "binary": "lb/1", "at": "2026-10-11T06:00:00.000Z"}));
        let (version, digest) = &self.doc;
        let digest = (*version > 0).then(|| digest.clone());
        json!({
            "document": {
                "status": "ok", "version": version, "digest": digest,
                "clients": [{"name": "acme", "daily_budget": 600_000_000, "utilization_limit": 0.9,
                             "models": ["claude-*"], "preferred_endpoints": [],
                             "keys": [{"id": "k7f3a2c1", "label": "2026-10", "digest": "a".repeat(64), "created_at": "2026-10-11T06:00:00.000Z"}]}],
                "revoked_digests": [], "frozen": [], "recovery_refused": false,
                "pitr_window": {"open": false, "usable": false, "restore_to": null, "closes": null},
                "cf": {"report": report, "complete": null, "running": self.cf.as_ref().is_some_and(|(r, _)| *r)},
                "history": self.history.iter().map(|(v, d)| json!({
                    "version": v, "digest": d, "at": "2026-10-11T06:00:00.000Z", "role": "writer",
                    "actor": "ops@example.test", "reason": "test", "ops": [{"op": "update", "name": "acme"}],
                    "recovery_id": null, "diff": [{"client": "acme", "change": "policy", "field": "daily_budget", "from": 1, "to": 2}]
                })).collect::<Vec<_>>(),
                "history_versions": self.history_versions
            },
            "restrictions": {
                "status": "ok",
                "records": self.records.iter().map(|(n, s, v)| json!({"name": n, "state": s.as_str(), "version": v, "actor": "ops@example.test", "reason": "test", "at": "2026-10-11T06:00:00.000Z"})).collect::<Vec<_>>(),
                "revoked_digests": [],
                "records_version": self.records.iter().map(|r| r.2).max().unwrap_or(0),
                "integrity": {"matches": self.full, "established": null},
                "part": {"status": "ok", "tag": effect::hex(&sha2::Sha256::digest(body.as_bytes())), "body": body, "set_hash": self.set_hash()},
                "markers": []
            },
            "recoveries": [],
            "stacks": self.stacks,
            "pitr_window": false
        })
    }
}

/// A fake store: successive `GET`s are answered from `gets` in order (the
/// last one repeats), and so are successive `POST`s from `posts`; or every
/// `POST` gets `post_page` when set: a body that is not the store's JSON, as
/// an edge block page would be.
#[derive(Clone)]
pub(crate) struct FakeStore {
    pub gets: Arc<Mutex<Vec<Value>>>,
    pub posts: Arc<Mutex<Vec<(StatusCode, Value)>>>,
    pub post_page: Arc<Mutex<Option<StatusCode>>>,
    pub log: Log,
}

impl FakeStore {
    pub(crate) fn new(world: &World) -> Self {
        FakeStore {
            gets: Arc::new(Mutex::new(vec![world.get_body()])),
            post_page: Arc::new(Mutex::new(None)),
            posts: Arc::new(Mutex::new(vec![(
                StatusCode::OK,
                json!({"status": 200, "results": [], "records_version": 0}),
            )])),
            log: Log::default(),
        }
    }

    /// Every `POST` from now on gets this answer.
    pub(crate) fn answer_post(&self, status: StatusCode, body: Value) {
        self.answer_posts(vec![(status, body)]);
    }

    /// Successive `POST`s get these answers, the last one repeating.
    pub(crate) fn answer_posts(&self, answers: Vec<(StatusCode, Value)>) {
        *self.posts.lock().unwrap() = answers;
    }

    /// Successive `GET`s from now on get these worlds, the last repeating.
    pub(crate) fn answer_gets(&self, worlds: &[&World]) {
        *self.gets.lock().unwrap() = worlds.iter().map(|w| w.get_body()).collect();
    }

    pub(crate) fn requests(&self) -> Vec<(String, String, String)> {
        self.log.lock().unwrap().clone()
    }

    pub(crate) fn posts(&self) -> Vec<Value> {
        self.requests()
            .into_iter()
            .filter(|(m, _, _)| m == "POST")
            .map(|(_, _, b)| serde_json::from_str(&b).unwrap())
            .collect()
    }

    pub(crate) fn router(&self) -> Router {
        let me = self.clone();
        Router::new().fallback(move |req: Request| {
            let me = me.clone();
            async move {
                let method = req.method().to_string();
                let path = req.uri().to_string();
                let auth = req.headers().get(header::AUTHORIZATION).cloned();
                let body = axum::body::to_bytes(req.into_body(), 1 << 20)
                    .await
                    .unwrap();
                me.log.lock().unwrap().push((
                    method.clone(),
                    path.clone(),
                    String::from_utf8_lossy(&body).into(),
                ));
                if auth.as_ref().and_then(|a| a.to_str().ok())
                    != Some(&format!("Bearer {STORE_TOKEN}"))
                {
                    return (
                        StatusCode::UNAUTHORIZED,
                        axum::Json(json!({"error": "unauthorized"})),
                    );
                }
                if path != "/_clients/admin" {
                    return (
                        StatusCode::NOT_FOUND,
                        axum::Json(json!({"error": "not found"})),
                    );
                }
                if method == "GET" {
                    return (StatusCode::OK, axum::Json(next(&me.gets)));
                }
                if let Some(status) = *me.post_page.lock().unwrap() {
                    return (status, axum::Json(Value::Null));
                }
                let (status, body) = next(&me.posts);
                (status, axum::Json(body))
            }
        })
    }
}

/// The next answer from a queue whose last answer repeats.
fn next<T: Clone>(queue: &Mutex<Vec<T>>) -> T {
    let mut q = queue.lock().unwrap();
    if q.len() > 1 {
        q.remove(0)
    } else {
        q[0].clone()
    }
}

/// A fake metrics proxy: `/m/<metric>/api/v1/query` answers that metric's
/// series, or its 30-day series when the query holds a `[30d]` range;
/// anything else is a missing route.
#[derive(Clone, Default)]
pub(crate) struct FakeMetrics {
    pub series: Arc<Mutex<BTreeMap<String, Vec<Value>>>>,
    /// Routes that answer 500, as a failed query would.
    pub failing: Arc<Mutex<Vec<String>>>,
    pub log: Log,
}

impl FakeMetrics {
    pub(crate) fn set(&self, metric: &str, series: Vec<Value>) {
        self.series
            .lock()
            .unwrap()
            .insert(metric.to_string(), series);
    }

    /// What a query over the last 30 days of `metric` answers.
    pub(crate) fn set_30d(&self, metric: &str, series: Vec<Value>) {
        self.set(&format!("{metric}[30d]"), series);
    }

    pub(crate) fn router(&self) -> Router {
        let me = self.clone();
        Router::new().fallback(move |req: Request| {
            let me = me.clone();
            async move {
                let path = req.uri().path().to_string();
                let query = url::form_urlencoded::parse(req.uri().query().unwrap_or("").as_bytes())
                    .find(|(k, _)| k == "query")
                    .map(|(_, v)| v.into_owned())
                    .unwrap_or_default();
                me.log.lock().unwrap().push((req.method().to_string(), req.uri().to_string(), String::new()));
                let metric = path.strip_prefix("/m/").and_then(|p| p.strip_suffix("/api/v1/query"));
                let Some(metric) = metric else {
                    return (StatusCode::BAD_REQUEST, axum::Json(json!({"error": "missing route"})));
                };
                if me.failing.lock().unwrap().iter().any(|f| f == metric) {
                    return (StatusCode::INTERNAL_SERVER_ERROR, axum::Json(json!({"error": "query failed"})));
                }
                let key = if query.contains("[30d]") { format!("{metric}[30d]") } else { metric.to_string() };
                let series = me.series.lock().unwrap().get(&key).cloned().unwrap_or_default();
                (StatusCode::OK, axum::Json(json!({"status": "success", "data": {"resultType": "vector", "result": series}})))
            }
        })
    }
}

pub(crate) fn series(labels: &[(&str, &str)], value: &str) -> Value {
    let metric: serde_json::Map<String, Value> = labels
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    json!({"metric": metric, "value": [1_791_698_400.0, value]})
}

/// The alerts every reader exports, raised or not; all quiet here.
const QUIET_ALERTS: [&str; 3] = ["restrictions_behind", "store_behind", "store_edited"];

/// k3s pods, each `(pod, up, [set hashes])`, each exporting its alerts, all
/// quiet, as a healthy reader does.
pub(crate) fn k3s(metrics: &FakeMetrics, pods: &[(&str, &str, &[&str])]) {
    metrics.set(
        "up",
        pods.iter()
            .map(|(p, up, _)| series(&[("stack", "k3s"), ("pod", p)], up))
            .collect(),
    );
    metrics.set(
        "anthropic_client_store_alert",
        pods.iter()
            .flat_map(|(p, _, _)| {
                QUIET_ALERTS
                    .iter()
                    .map(move |a| series(&[("stack", "k3s"), ("pod", p), ("alert", a)], "0"))
            })
            .collect(),
    );
    metrics.set(
        "anthropic_client_store_restrictions",
        pods.iter()
            .flat_map(|(p, _, hashes)| {
                hashes
                    .iter()
                    .map(move |h| series(&[("stack", "k3s"), ("pod", p), ("set_hash", h)], "1"))
            })
            .collect(),
    );
}

/// The document each k3s pod holds, as `(pod, [digests])`.
pub(crate) fn k3s_documents(metrics: &FakeMetrics, pods: &[(&str, &[&str])]) {
    metrics.set(
        "anthropic_client_store_document",
        pods.iter()
            .flat_map(|(p, digests)| {
                digests.iter().map(move |d| {
                    series(
                        &[
                            ("stack", "k3s"),
                            ("pod", p),
                            ("version", "1"),
                            ("digest", d),
                        ],
                        "1",
                    )
                })
            })
            .collect(),
    );
}

/// A metrics store showing one healthy reader with nothing to report: what
/// a document write meets when no instance reports the store behind or
/// edited.
pub(crate) fn quiet_fleet() -> FakeMetrics {
    let metrics = FakeMetrics::default();
    k3s(&metrics, &[("lb-a", "1", &[])]);
    metrics
}
