//! SHA-256 content hashing.

use sha2::{Digest, Sha256};

/// Generate a SHA-256 content hash (64-char lowercase hex).
pub fn generate_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    let result = hasher.finalize();
    hex::encode(result)
}

/// Hex encoding without pulling in the `hex` crate.
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_64_hex_chars() {
        let hash = generate_content_hash("hello world");
        assert_eq!(hash.len(), 64);
        assert!(
            hash.chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
        );
    }

    #[test]
    fn deterministic() {
        let h1 = generate_content_hash("test content");
        let h2 = generate_content_hash("test content");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_content_different_hash() {
        let h1 = generate_content_hash("content A");
        let h2 = generate_content_hash("content B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn known_hash() {
        // SHA-256 of "test" is well-known
        let hash = generate_content_hash("test");
        assert_eq!(
            hash,
            "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
        );
    }
}
