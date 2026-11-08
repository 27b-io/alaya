//! The write log's copy of every document the console commits, and the
//! console's one read of its own log.
//!
//! The write log is the console's record of every client-store write, kept
//! outside the store (`routes/clients.rs`). Each document the store commits
//! goes on it too, as the exact bytes the store committed. After the store is
//! wiped, that copy is what an `import` restores, and what its preview diffs
//! against. The bytes go out as base64, a few KiB per line, because the
//! container runtime splits a line past 16 KiB and the log shipper drops one
//! it cannot rejoin. Every line names the document's version and digest, and
//! its part out of how many.
//!
//! The log is read back through the metrics proxy's write-log route, which
//! pins every query to the console's own log stream. What it reads back is
//! untrusted input, checked against its digest, which catches damage and
//! nothing more. A forged copy can be offered for import, flagged as no
//! document the console committed, but is never what an import is diffed
//! against: that is a copy this process exported or a committed label names.
//! The import's guard judges its bytes against what instances installed, and
//! the store merges every revoked key into an import, so no forgery can bring
//! a revoked key back.

use std::collections::BTreeMap;

use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};
use sha2::Digest;

use crate::lb::MetricsClient;

/// Base64 characters of a document on one line: 9 KiB, as the LB instances
/// log theirs, which keeps the whole line well under 16 KiB. A multiple of
/// 4, so the parts join into one base64 text.
const PART_CHARS: usize = 9 * 1024;

/// The message of every document line, which the read selects on. The
/// `tracing` call in `log_document` must carry the same text.
const DOCUMENT_LINE: &str = "client-store document";

/// Most documents the import page lists.
const LISTED: usize = 50;

/// More lines than any document has: a 2 MiB document is 304 parts. A read
/// takes each distinct line once, so a document logged again adds nothing to
/// it; only a line that differs, a damaged or forged one, can.
const MAX_LINES: usize = 1000;

/// The hex SHA-256 of `text`.
pub fn digest_of(text: &str) -> String {
    crate::effect::hex(&sha2::Sha256::digest(text.as_bytes()))
}

/// Log a committed document's exact bytes. `digest` is the SHA-256 of `text`.
pub fn log_document(version: u64, digest: &str, text: &str) {
    let b64 = STANDARD.encode(text.as_bytes());
    let parts: Vec<&str> = b64
        .as_bytes()
        .chunks(PART_CHARS)
        .map(|c| std::str::from_utf8(c).expect("base64 is ASCII"))
        .collect();
    let total = parts.len();
    for (i, part) in parts.into_iter().enumerate() {
        tracing::info!(
            target: "ops_console::write_log",
            version,
            digest = %digest,
            part = i + 1,
            parts = total,
            b64 = %part,
            "client-store document"
        );
    }
}

/// Log the document a store answer carries as committed, once its bytes hash
/// to the digest the answer reports. Otherwise say why the write log holds no
/// copy of it: a store from before it returned the bytes sends none. True
/// when it logged the document.
pub fn log_committed(version: u64, digest: &str, text: Option<&str>) -> bool {
    match text {
        Some(t) if digest_of(t) == digest => {
            log_document(version, digest, t);
            return true;
        }
        Some(_) => tracing::warn!(
            target: "ops_console::write_log",
            version,
            digest = %digest,
            "client-store document not logged: the store's text does not hash to its digest"
        ),
        None => tracing::warn!(
            target: "ops_console::write_log",
            version,
            digest = %digest,
            "client-store document not logged: the store sent no text"
        ),
    }
    false
}

/// A write's request as the write log keeps it. An import's document becomes
/// its size, because the document the store commits is logged whole from the
/// answer. A restrictions_import's sets become their hashes and sizes,
/// because they come from the readers' own logs, and its override becomes
/// whether one was sent. Every other request is kept as sent: none carries
/// a key.
pub fn logged_request(body: &Value) -> Value {
    let mut out = body.clone();
    if let Some(ops) = out
        .pointer_mut("/document/ops")
        .and_then(Value::as_array_mut)
    {
        for op in ops {
            let size = op.get("text").and_then(Value::as_str).map(str::len);
            if let (Some(size), Some(o)) = (size, op.as_object_mut()) {
                o.remove("text");
                o.insert("bytes".into(), json!(size));
            }
        }
    }
    if let Some(import) = out
        .get_mut("restrictions_import")
        .and_then(Value::as_object_mut)
    {
        let sent = import.remove("override").is_some();
        import.insert("override".into(), json!(sent));
        if let Some(sets) = import.get_mut("sets").and_then(Value::as_array_mut) {
            for set in sets {
                *set = json!({
                    "hash": set.get("hash"),
                    "bytes": set.get("text").and_then(Value::as_str).map(str::len),
                    "complete": set.get("complete"),
                });
            }
        }
    }
    out
}

/// `s`, cut so that the JSON string a log line writes for it holds at most
/// `max` bytes, with `'…'` marking a cut. JSON escapes a quote, a backslash
/// and each control character, so a char count alone does not bound a line.
pub fn bounded(s: &str, max: usize) -> String {
    let cost = |c: char| match c {
        '"' | '\\' | '\n' | '\r' | '\t' | '\u{8}' | '\u{c}' => 2,
        c if (c as u32) < 0x20 => 6,
        c => c.len_utf8(),
    };
    if s.chars().map(cost).sum::<usize>() <= max {
        return s.to_string();
    }
    let mut out = String::new();
    let mut used = '…'.len_utf8();
    for c in s.chars() {
        used += cost(c);
        if used > max {
            break;
        }
        out.push(c);
    }
    out.push('…');
    out
}

/// A document the write log holds, as its first line names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Logged {
    pub version: u64,
    pub digest: String,
    /// When its first line was logged, as the log store reports it.
    pub at: String,
}

/// The documents the write log names, and how many of its first document
/// lines name no version and digest. Only a fault, or a write to the log
/// store from outside the console, makes such a line. It names no document
/// to import, so it is left out rather than allowed to hide the others.
#[derive(Debug, Default)]
pub struct Listing {
    pub docs: Vec<Logged>,
    pub malformed: usize,
}

/// The documents the write log holds from the last 30 days, each once,
/// highest version first: the newest is the last one the console committed.
pub async fn logged(metrics: &MetricsClient) -> Result<Listing, String> {
    let query = format!(
        "_time:30d target:=\"ops_console::write_log\" fields.message:=\"{DOCUMENT_LINE}\" \
         fields.part:=1 | fields _time, fields.version, fields.digest | sort by (_time desc) \
         | limit {LISTED}"
    );
    let lines = metrics
        .write_log(&query, LISTED)
        .await
        .map_err(|e| e.detail().to_string())?;
    let mut out = Listing::default();
    for line in &lines {
        let field = |k: &str| line.get(k).and_then(Value::as_str);
        let version = field("fields.version").and_then(|v| v.parse::<u64>().ok());
        let digest = field("fields.digest").filter(|d| crate::effect::is_hex64(d));
        let (Some(version), Some(digest)) = (version, digest) else {
            out.malformed += 1;
            continue;
        };
        if out
            .docs
            .iter()
            .any(|d| d.version == version && d.digest == digest)
        {
            continue;
        }
        out.docs.push(Logged {
            version,
            digest: digest.to_string(),
            at: field("_time").unwrap_or_default().to_string(),
        });
    }
    out.docs.sort_by_key(|d| std::cmp::Reverse(d.version));
    Ok(out)
}

/// One logged document's exact bytes, joined from its lines and checked.
/// The read takes each distinct line once, however often the document was
/// logged, so two different copies of a part still read as damage.
pub async fn document(
    metrics: &MetricsClient,
    version: u64,
    digest: &str,
) -> Result<String, String> {
    // A u64 and 64 hex characters: nothing in either can change the query.
    if !crate::effect::is_hex64(digest) {
        return Err("a document is named by its 64 hex digest".into());
    }
    let query = format!(
        "_time:30d target:=\"ops_console::write_log\" fields.message:=\"{DOCUMENT_LINE}\" \
         fields.version:={version} fields.digest:=\"{digest}\" \
         | uniq by (fields.part, fields.parts, fields.b64) limit {MAX_LINES}"
    );
    let lines = metrics
        .write_log(&query, MAX_LINES)
        .await
        .map_err(|e| e.detail().to_string())?;
    assemble(version, digest, &lines)
}

/// A document from its lines: every part present, each copy of a part the
/// same, the bytes hashing to the digest, and the document naming the
/// version its lines do.
fn assemble(version: u64, digest: &str, lines: &[Value]) -> Result<String, String> {
    if lines.is_empty() {
        return Err("the write log holds no line of it".into());
    }
    if lines.len() >= MAX_LINES {
        return Err(format!(
            "the write log holds {MAX_LINES} or more distinct lines of it, more than any document has"
        ));
    }
    let mut parts: BTreeMap<u64, &str> = BTreeMap::new();
    let mut total: Option<u64> = None;
    for line in lines {
        let field = |k: &str| line.get(k).and_then(Value::as_str);
        let number = |k: &str| field(k).and_then(|v| v.parse::<u64>().ok());
        let (Some(part), Some(count), Some(b64)) = (
            number("fields.part"),
            number("fields.parts"),
            field("fields.b64"),
        ) else {
            return Err("a line of it names no part, part count or text".into());
        };
        if *total.get_or_insert(count) != count {
            return Err("its lines disagree on how many parts it has".into());
        }
        if parts.insert(part, b64).is_some_and(|seen| seen != b64) {
            return Err(format!(
                "the write log holds two different copies of part {part}"
            ));
        }
    }
    let total = total.unwrap_or_default();
    if parts.keys().any(|p| !(1..=total).contains(p)) {
        return Err(format!("a line of it names a part outside its {total}"));
    }
    if parts.len() as u64 != total {
        return Err(format!(
            "the write log holds {} of its {total} parts",
            parts.len()
        ));
    }
    let joined: String = parts.into_values().collect();
    let bytes = STANDARD
        .decode(joined)
        .map_err(|_| "its parts do not join into base64".to_string())?;
    let text = String::from_utf8(bytes).map_err(|_| "its bytes are not UTF-8".to_string())?;
    if digest_of(&text) != digest {
        return Err("its bytes do not hash to the digest its lines name".into());
    }
    if document_version(&text) != Some(version) {
        return Err("the document names another version than its lines do".into());
    }
    Ok(text)
}

/// The version a document's text names, or `None` when it is no JSON object
/// with a whole-number version.
pub fn document_version(text: &str) -> Option<u64> {
    serde_json::from_str::<Value>(text)
        .ok()?
        .get("version")?
        .as_u64()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lines `log_document` writes, as the log store returns them: one
    /// JSON object per line, every value a string.
    fn lines_of(version: u64, digest: &str, text: &str) -> Vec<Value> {
        let b64 = STANDARD.encode(text.as_bytes());
        let parts: Vec<&[u8]> = b64.as_bytes().chunks(PART_CHARS).collect();
        parts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                json!({
                    "fields.version": version.to_string(), "fields.digest": digest,
                    "fields.part": (i + 1).to_string(), "fields.parts": parts.len().to_string(),
                    "fields.b64": std::str::from_utf8(p).unwrap(),
                })
            })
            .collect()
    }

    fn doc(version: u64, filler: usize) -> String {
        json!({"schema": 1, "version": version, "updated_at": "2026-11-08T00:00:00.000Z",
               "clients": [{"name": "acme", "models": ["x".repeat(filler)]}], "revoked_digests": []})
        .to_string()
    }

    #[test]
    fn a_document_is_joined_from_its_parts_in_any_order_and_checked() {
        let text = doc(9, 40_000);
        let d = digest_of(&text);
        let mut lines = lines_of(9, &d, &text);
        assert!(lines.len() > 3, "{}", lines.len());
        lines.reverse();
        assert_eq!(assemble(9, &d, &lines).unwrap(), text);
        // A document logged twice repeats each part: the same bytes join.
        let twice: Vec<Value> = lines.iter().chain(&lines).cloned().collect();
        assert_eq!(assemble(9, &d, &twice).unwrap(), text);
    }

    #[test]
    fn a_damaged_document_is_refused_naming_the_damage() {
        let text = doc(9, 40_000);
        let d = digest_of(&text);
        let lines = lines_of(9, &d, &text);
        let refused = |lines: &[Value], version: u64, digest: &str| {
            assemble(version, digest, lines).unwrap_err()
        };
        assert!(refused(&lines[1..], 9, &d).contains("parts"));
        let mut other = lines.clone();
        other.push(json!({"fields.part": "1", "fields.parts": lines.len().to_string(), "fields.b64": "QUJD"}));
        assert!(refused(&other, 9, &d).contains("two different copies of part 1"));
        let mut miscounted = lines.clone();
        miscounted[0]["fields.parts"] = json!("99");
        assert!(refused(&miscounted, 9, &d).contains("disagree"));
        assert!(refused(&lines, 9, &"e".repeat(64)).contains("do not hash"));
        assert!(refused(&lines, 8, &d).contains("another version"));
        assert!(refused(&[], 9, &d).contains("no line"));
        let mut bare = lines.clone();
        bare[0].as_object_mut().unwrap().remove("fields.b64");
        assert!(refused(&bare, 9, &d).contains("names no part"));
        let mut stray = lines.clone();
        stray[0]["fields.part"] = json!("99");
        assert!(refused(&stray, 9, &d).contains("outside"));
        let many = vec![lines[0].clone(); MAX_LINES];
        assert!(refused(&many, 9, &d).contains("distinct lines"));
    }

    #[test]
    fn an_import_and_a_reseed_are_logged_without_their_bodies() {
        let text = doc(9, 10);
        let body = json!({"document": {"expected_version": 0, "actor": "a", "reason": "r", "dry_run": false,
            "ops": [{"op": "import", "text": text, "expected_digest": "d".repeat(64)}]}});
        let logged = logged_request(&body);
        assert_eq!(
            logged["document"]["ops"][0],
            json!({"op": "import", "expected_digest": "d".repeat(64), "bytes": text.len()})
        );
        let set = "r acme disabled 1791700000123000\n";
        let reseed = json!({"restrictions_import": {"actor": "a", "reason": "r",
            "sets": [{"text": set, "hash": "h".repeat(64), "complete": true}],
            "override": "the typed phrase"}});
        let logged = logged_request(&reseed);
        assert_eq!(
            logged["restrictions_import"],
            json!({"actor": "a", "reason": "r", "override": true,
                   "sets": [{"hash": "h".repeat(64), "bytes": set.len(), "complete": true}]})
        );
        assert!(!logged.to_string().contains("acme"));
        let mut plain = reseed.clone();
        plain["restrictions_import"]
            .as_object_mut()
            .unwrap()
            .remove("override");
        assert_eq!(
            logged_request(&plain)["restrictions_import"]["override"],
            json!(false)
        );
        // Any other request is kept whole.
        let delete = json!({"document": {"ops": [{"op": "delete", "name": "acme"}]}});
        assert_eq!(logged_request(&delete), delete);
    }

    #[test]
    fn bounded_counts_what_json_escaping_writes() {
        assert_eq!(bounded("plain", 5), "plain");
        // Each quote costs two bytes: three fit in 9 beside the mark's three.
        assert_eq!(bounded("\"\"\"\"\"", 9), "\"\"\"…");
        // A control character costs six.
        assert_eq!(bounded("\u{1}\u{1}", 9), "\u{1}…");
        let long = "é".repeat(1000);
        let out = bounded(&long, 101);
        assert!(
            serde_json::to_string(&out).unwrap().len() <= 103,
            "{}",
            out.len()
        );
    }
}
