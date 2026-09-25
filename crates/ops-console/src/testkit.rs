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
    let mut header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::ES256);
    header.kid = Some(KID.into());
    jsonwebtoken::encode(
        &header,
        &IdTokenClaims {
            iss: ISSUER.into(),
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
