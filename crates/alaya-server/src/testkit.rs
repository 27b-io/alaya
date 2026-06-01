//! Test-only OIDC helpers: fixed RSA-2048 / EC-P256 keypairs + JWT minting.
//!
//! These keys are generated once for tests and MUST NOT be used anywhere real.
//! `mint_*` signs JWTs with the private keys; the matching public JWK components
//! (`RSA_N`/`RSA_E`, `EC_X`/`EC_Y`) are what the verifier validates against —
//! injected directly in `oidc` unit tests and served by the mock IdP in the
//! discovery tests.

use jsonwebtoken::{Algorithm, EncodingKey, Header, encode};
use serde::Serialize;

pub(crate) const ISSUER: &str = "https://issuer.test";
pub(crate) const AUDIENCE: &str = "https://rs.test/mcp";
pub(crate) const KID_RSA: &str = "test-rsa";
pub(crate) const KID_EC: &str = "test-ec";

// Public JWK components (base64url, no padding) matching the private PEMs below.
pub(crate) const RSA_N: &str = "jz7-SSp2pSUCreZDgL4HXbGNXcAaX7297U02uZgHvwZskixCHSBHLZDSo46bf0rDDtHGXi_tah5MKZEV-69_rsk-PABiPHGLLQ4jj-axRkW0MN2-szHyzOrwNU_YGSOOc0BblzoVEe1f4xvr9ILqqDROuZV4cF2r33MMbAY7yCFoFoR9n4k518HyMCGMiRSSEURradcLPdAUJ41YHlqj1-mW0lQg5CIAyVqW0Wb297--XyjBB0vANXrwW-F52bG3HRYTUqOwrg1HH5bEZlFa7ryeI6hBO7YwbViSrtahNozBRD5vvruPnaO_EiyLbu9SxNHrPBguIs3uQg0fb161ZQ";
pub(crate) const RSA_E: &str = "AQAB";
pub(crate) const EC_X: &str = "xFqh0rq2ORGl3JCGY729yeT2U8lx0o8KZMcWp_4Jpm4";
pub(crate) const EC_Y: &str = "L7KEoqyRTqc9WneB4UytjJY8ShW4l-KHOnZtz1TeZs8";

const RSA_PRIV_PEM: &str = r#"-----BEGIN PRIVATE KEY-----
MIIEuwIBADANBgkqhkiG9w0BAQEFAASCBKUwggShAgEAAoIBAQCPPv5JKnalJQKt
5kOAvgddsY1dwBpfvb3tTTa5mAe/BmySLEIdIEctkNKjjpt/SsMO0cZeL+1qHkwp
kRX7r3+uyT48AGI8cYstDiOP5rFGRbQw3b6zMfLM6vA1T9gZI45zQFuXOhUR7V/j
G+v0guqoNE65lXhwXavfcwxsBjvIIWgWhH2fiTnXwfIwIYyJFJIRRGtp1ws90BQn
jVgeWqPX6ZbSVCDkIgDJWpbRZvb3v75fKMEHS8A1evBb4XnZsbcdFhNSo7CuDUcf
lsRmUVruvJ4jqEE7tjBtWJKu1qE2jMFEPm++u4+do78SLItu71LE0es8GC4ize5C
DR9vXrVlAgMBAAECgf8ORGdJ97T8cSZGMouynKgYZLU/CzD5qEjRqMrSd+W0PTV7
bedrhE5SS0rcpUM60Q/6Fm6oISXO1Nt3z7kFc3jDl+6R4ryxEBj5saYTwABWoLf9
K6FLeBFZNKRSpD8X6e2fVKWQEmzGiE/mJkhq+CVRqi5+VEDAGg88tQzLWRzlnxd9
DrOJCZ1jCZYUXL3kU0kj7h4Gvcw1eu+g7qUJeXrHTm7JFhFkqD6iqvmlMH6XQReL
hc6shwAx+kaxBerkYEvSMm5yqa3qzJ1CMGpADyHDhhRcPIAisb4UXKjSdfidXZ8g
NGwWLffumxFdWg47pCGVzCl/T5HtqhRrURzhVvECgYEAyQBIeSEkT69H74XfW7h8
y5z7Pnqp5i6CDGJJA4kOdcOec+JRjc529wJlO9W7fbQTXLHXWSu8zIcw2zvSrzWT
+VLXfmlX0bEQOk01uXIvUaNZy1SmLnk5W9L5/2acm4qGyLEL7ujELFql2DYLKrra
SW0CfQu+84AqTOeM9Py34ocCgYEAtnEWdn4vcCh5Wf5eem0Q/oF3X7RPduWMZedR
xjVF19s4Ez4VbaFRWZerr8RpOuU+sVCiTWFl5mky3LFxaXrdWtvZl3WTISuqfUqq
pQ+STVQbVGpEREdMbgFv1fIN4Mc0KZmhA9Y0fXmOOT88yEzjSw0/ZEjsrWXFKbbO
X6rhZ7MCgYEAjZsYc9XoegcX2+RpvnmT2fLnglXyukrLriPUIpx9RnQhfqzUHd52
K4FRhr0GEQI7ndNgzt6kbUdVIS7dODi73iwBy3o1t3JR53Ebx2FtestlaH1jclxP
D6TsIYXOETqfyGYK7S6pfkICkvdIGLt5K7+TwDr1NSF3K6T5xmMAvaMCgYBYhlcX
9/KcwYbgnATL8tAkLj32Ok+0qX2OlMehHYheTQjQjXdoUrZeerHb/7nv0fyxnSaj
1XbUboc3fwJA5FU0GSljzLEvjziSwwA6R2v+CamZNFcbqlzzo87YSTNitkYhSWJP
skiV+b2BGaYsquI/MJZp2ti86nzY2NMaqJfm8QKBgCKomLWThg+khmEm+/FA2rxt
7p+Hs4F2uU/Em7n1i+pojmwraw+J7a98scazPu+ollLHJjo0yendGCnhyKmMpafE
CiZyd+k1XhszSHMqyWd4gUoHQMCRUMHV+OeWzn4SmL2WHdI/IFFHP64R71OpqxWV
OWF5Sf/MMm2vmjF4Sqxb
-----END PRIVATE KEY-----"#;

const EC_PRIV_PEM: &str = r#"-----BEGIN PRIVATE KEY-----
MIGHAgEAMBMGByqGSM49AgEGCCqGSM49AwEHBG0wawIBAQQgYQR4QrEzqrzkofjM
bOF4ADXv20gmQnyEBtfoG+tQ8EqhRANCAATEWqHSurY5EaXckIZjvb3J5PZTyXHS
jwpkxxan/gmmbi+yhKKskU6nPVp3geFMrYyWPEoVuJfihzp2bc9U3mbP
-----END PRIVATE KEY-----"#;

/// JWT claims for minting. Mirrors the registered claims the verifier checks.
#[derive(Serialize)]
pub(crate) struct TestClaims {
    pub iss: String,
    pub aud: String,
    pub exp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iat: Option<u64>,
}

pub(crate) fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

impl TestClaims {
    /// A claim set that should validate: correct iss/aud, ~5 min lifetime.
    pub(crate) fn valid() -> Self {
        let n = now();
        Self {
            iss: ISSUER.to_string(),
            aud: AUDIENCE.to_string(),
            exp: n + 300,
            iat: Some(n),
        }
    }
}

/// Sign `claims` with the given algorithm and optional `kid`.
pub(crate) fn mint(alg: Algorithm, kid: Option<&str>, claims: &TestClaims) -> String {
    let mut header = Header::new(alg);
    header.kid = kid.map(str::to_string);
    let key = match alg {
        Algorithm::RS256 => {
            EncodingKey::from_rsa_pem(RSA_PRIV_PEM.as_bytes()).expect("rsa test key parses")
        }
        Algorithm::ES256 => {
            EncodingKey::from_ec_pem(EC_PRIV_PEM.as_bytes()).expect("ec test key parses")
        }
        // Models the RS256->HS256 confusion attack: the attacker signs HS256
        // using the (public, known) RSA modulus as the HMAC secret, betting the
        // server reuses key material across algs. The verifier must reject this
        // at the alg allowlist regardless of the secret.
        Algorithm::HS256 => EncodingKey::from_secret(RSA_N.as_bytes()),
        other => panic!("unsupported test alg: {other:?}"),
    };
    encode(&header, claims, &key).expect("mint token")
}

/// Shared `AuthState` builder for the auth/wellknown tests. `oidc_on` attaches
/// a verifier pre-loaded with the RSA test key (validates `testkit`-minted RS256
/// tokens with no network).
pub(crate) fn auth_state(api_key: Option<&str>, oidc_on: bool) -> crate::auth::AuthState {
    crate::auth::AuthState {
        api_key: api_key.map(str::to_string),
        allow_unauthenticated: false,
        oidc: oidc_on.then(crate::oidc::OidcVerifier::test_with_rsa_key),
        public_base_url: "https://rs.test".to_string(),
    }
}
