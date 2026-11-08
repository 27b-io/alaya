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
//! pins every query to the console's own log stream. The console does not
//! guard the log store, so what it reads back is untrusted input: a document
//! read back is checked against its digest, which catches damage and nothing
//! more. A forged copy can be
//! offered for import, flagged as no document the console committed, but is
//! never what an import is diffed against: that is a copy this process
//! exported or a committed label names. The import's guard judges its bytes
//! against what instances installed, and the store merges every revoked key
//! into an import, so no forgery can bring a revoked key back.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

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

/// More lines than any document has: a 2 MiB document is 304 parts, and a
/// document logged twice repeats each. Two copies of the largest one fit the
/// response cap of a read; a third does not, so a recovery's document is
/// logged again only when the write log holds no whole copy (`held`).
const MAX_LINES: usize = 1000;

/// How long a copy this process logged may take to read back whole. The log
/// shipper and the log store index a line within seconds; inside this, a
/// copy that reads as absent or partial is still on its way.
const SETTLING_SECS: i64 = 600;

/// Most claims `Settling` keeps.
const SETTLING_KEPT: usize = 64;

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

/// The recovery documents this process is logging, each with when it claimed
/// it (epoch seconds). A copy the log store has not finished indexing reads
/// as absent or partial, so a document claimed in the last `SETTLING_SECS` is
/// not logged again: every copy is read back with the rest.
#[derive(Default)]
pub struct Settling(Mutex<VecDeque<(u64, String, i64)>>);

impl Settling {
    /// Claim the logging of a document as of `now`: true unless this process
    /// claimed it in the last `SETTLING_SECS`. Tested and recorded under one
    /// lock, so two acceptances at once log one copy.
    pub fn claim(&self, version: u64, digest: &str, now: i64) -> bool {
        let mut claims = self.0.lock().expect("settling lock poisoned");
        claims.retain(|(_, _, at)| now - *at < SETTLING_SECS);
        if claims.iter().any(|(v, d, _)| *v == version && d == digest) {
            return false;
        }
        if claims.len() == SETTLING_KEPT {
            claims.pop_front();
        }
        claims.push_back((version, digest.to_string(), now));
        true
    }

    /// Give back a claim under which nothing was logged.
    pub fn release(&self, version: u64, digest: &str) {
        self.0
            .lock()
            .expect("settling lock poisoned")
            .retain(|(v, d, _)| !(*v == version && d == digest));
    }
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
pub async fn document(
    metrics: &MetricsClient,
    version: u64,
    digest: &str,
) -> Result<String, String> {
    read(metrics, version, digest)
        .await
        .map_err(Unreadable::into_text)
}

/// What the write log holds of one document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Held {
    /// A whole copy that reads back to its exact bytes.
    Readable,
    /// No line of it.
    Absent,
    /// Lines that agree, with parts missing: a whole copy logged again
    /// completes it.
    Partial,
}

/// What the write log holds of one document, or why that cannot be told:
/// a read that failed, or lines no further copy can mend. Then another copy
/// could only add to what a read takes, so none is logged.
pub async fn held(metrics: &MetricsClient, version: u64, digest: &str) -> Result<Held, String> {
    held_of(read(metrics, version, digest).await)
}

fn held_of(read: Result<String, Unreadable>) -> Result<Held, String> {
    match read {
        Ok(_) => Ok(Held::Readable),
        Err(Unreadable::Absent) => Ok(Held::Absent),
        Err(Unreadable::Missing(_)) => Ok(Held::Partial),
        Err(Unreadable::Damaged(why)) => Err(why),
    }
}

/// Why the write log yields no document.
#[derive(Debug, PartialEq, Eq)]
enum Unreadable {
    /// It holds no line of it.
    Absent,
    /// Its lines agree, but some parts are missing.
    Missing(String),
    /// Anything else, a read that failed included.
    Damaged(String),
}

impl Unreadable {
    fn into_text(self) -> String {
        match self {
            Unreadable::Absent => "the write log holds no line of it".into(),
            Unreadable::Missing(why) | Unreadable::Damaged(why) => why,
        }
    }
}

async fn read(metrics: &MetricsClient, version: u64, digest: &str) -> Result<String, Unreadable> {
    // A u64 and 64 hex characters: nothing in either can change the query.
    if !crate::effect::is_hex64(digest) {
        return Err(Unreadable::Damaged(
            "a document is named by its 64 hex digest".into(),
        ));
    }
    let query = format!(
        "_time:30d target:=\"ops_console::write_log\" fields.message:=\"{DOCUMENT_LINE}\" \
         fields.version:={version} fields.digest:=\"{digest}\" \
         | fields fields.part, fields.parts, fields.b64 | limit {MAX_LINES}"
    );
    let lines = metrics
        .write_log(&query, MAX_LINES)
        .await
        .map_err(|e| Unreadable::Damaged(e.detail().to_string()))?;
    assemble(version, digest, &lines)
}

/// A document from its lines: every part present, each copy of a part the
/// same, the bytes hashing to the digest, and the document naming the
/// version its lines do.
fn assemble(version: u64, digest: &str, lines: &[Value]) -> Result<String, Unreadable> {
    use Unreadable::{Absent, Damaged, Missing};
    if lines.is_empty() {
        return Err(Absent);
    }
    if lines.len() >= MAX_LINES {
        return Err(Damaged(format!(
            "the write log holds {MAX_LINES} or more lines of it, more than any document has"
        )));
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
            return Err(Damaged(
                "a line of it names no part, part count or text".into(),
            ));
        };
        if *total.get_or_insert(count) != count {
            return Err(Damaged(
                "its lines disagree on how many parts it has".into(),
            ));
        }
        if parts.insert(part, b64).is_some_and(|seen| seen != b64) {
            return Err(Damaged(format!(
                "the write log holds two different copies of part {part}"
            )));
        }
    }
    let total = total.unwrap_or_default();
    if parts.keys().any(|p| !(1..=total).contains(p)) {
        return Err(Damaged(format!(
            "a line of it names a part outside its {total}"
        )));
    }
    if parts.len() as u64 != total {
        return Err(Missing(format!(
            "the write log holds {} of its {total} parts",
            parts.len()
        )));
    }
    let joined: String = parts.into_values().collect();
    let bytes = STANDARD
        .decode(joined)
        .map_err(|_| Damaged("its parts do not join into base64".into()))?;
    let text = String::from_utf8(bytes).map_err(|_| Damaged("its bytes are not UTF-8".into()))?;
    if digest_of(&text) != digest {
        return Err(Damaged(
            "its bytes do not hash to the digest its lines name".into(),
        ));
    }
    let named = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|d| d.get("version")?.as_u64());
    if named != Some(version) {
        return Err(Damaged(
            "the document names another version than its lines do".into(),
        ));
    }
    Ok(text)
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
            assemble(version, digest, lines).unwrap_err().into_text()
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
    }

    /// Only a copy with parts missing, or none at all, is one another copy
    /// can mend; a copy damaged any other way, or a read that failed, is not.
    #[test]
    fn what_the_write_log_holds_is_told_apart() {
        let text = doc(9, 40_000);
        let d = digest_of(&text);
        let lines = lines_of(9, &d, &text);
        let held = |lines: &[Value]| held_of(assemble(9, &d, lines));
        assert_eq!(held(&lines), Ok(Held::Readable));
        assert_eq!(held(&[]), Ok(Held::Absent));
        assert_eq!(held(&lines[1..]), Ok(Held::Partial));
        let mut stray = lines.clone();
        stray[0]["fields.part"] = json!("99");
        assert!(held(&stray).unwrap_err().contains("outside"));
        let mut other = lines.clone();
        other.push(json!({"fields.part": "1", "fields.parts": lines.len().to_string(), "fields.b64": "QUJD"}));
        assert!(held(&other).is_err());
        assert_eq!(
            held_of(Err(Unreadable::Damaged("metrics 500".into()))),
            Err("metrics 500".into())
        );
    }

    /// A claim holds for the settling time, once however often it is asked
    /// for, and a released one can be claimed again at once.
    #[test]
    fn a_document_is_claimed_once_while_it_settles() {
        let (s, d) = (Settling::default(), "d".repeat(64));
        assert!(s.claim(9, &d, 1_000));
        assert!(!s.claim(9, &d, 1_000 + SETTLING_SECS - 1));
        assert!(s.claim(9, &"e".repeat(64), 1_000) && s.claim(10, &d, 1_000));
        assert!(s.claim(9, &d, 1_000 + SETTLING_SECS));
        s.release(9, &d);
        assert!(s.claim(9, &d, 1_000 + SETTLING_SECS));
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
