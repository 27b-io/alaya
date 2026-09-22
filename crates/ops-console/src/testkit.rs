//! Test-only OIDC fixtures: a signing key and an `id_token` minter, so the
//! RP tests can drive `verify_id_token` instead of asserting around it.
//!
//! Its own file for one reason: `detect-private-key` has no content allowlist,
//! so any file holding a PEM must be excluded from that hook by path. Keeping
//! the key here leaves the hook covering `oidc.rs`, which is where a real key
//! would actually be dangerous. gitleaks still scans this file — its allowlist
//! matches the key's own bytes rather than a path (`.gitleaks.toml`). Same
//! trade `alaya-server`'s `testkit.rs` already made, same key.

use serde::Serialize;

pub(crate) const ISSUER: &str = "https://id.test";
pub(crate) const CLIENT_ID: &str = "console";
pub(crate) const KID: &str = "test-ec";

/// P-256 test keypair: the private PEM and its public point. Generated for
/// tests, used by no deployment, grants access to nothing.
pub(crate) const EC_PRIV_PEM: &str = r#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgYQR4QrEzqrzkofjM
bOF4ADXv20gmQnyEBtfoG+tQ8EqhRANCAATEWqHSurY5EaXckIZjvb3J5PZTyXHS
jwpkxxan/gmmbi+yhKKskU6nPVp3geFMrYyWPEoVuJfihzp2bc9U3mbP
-----END PRIVATE KEY-----"#;
pub(crate) const EC_X: &str = "xFqh0rq2ORGl3JCGY729yeT2U8lx0o8KZMcWp_4Jpm4";
pub(crate) const EC_Y: &str = "L7KEoqyRTqc9WneB4UytjJY8ShW4l-KHOnZtz1TeZs8";

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
        &jsonwebtoken::EncodingKey::from_ec_pem(EC_PRIV_PEM.as_bytes()).expect("ec test key"),
    )
    .expect("mint id_token")
}
