//! QdrantClient — VectorStorage implementation over the Qdrant REST API.
//!
//! All calls use raw `reqwest` HTTP to stay WASM-compatible (no qdrant-client crate).

use std::collections::HashMap;
use std::hash::BuildHasher;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use alaya_types::{
    AlayaError, Result,
    memory::{
        HealthStatus, Memory, MetadataUpdate, PatchMemoryRequest, ScoredMemory, ScrollResult,
    },
    search::PayloadFilter,
};

use crate::{StoreMode, VectorStorage};

// ─── Client ──────────────────────────────────────────────────────────────────

pub struct QdrantClient {
    client: reqwest::Client,
    base_url: String,
    collection: String,
    tag_collection: String,
    /// Serialises this client's writes to the memories collection. Every
    /// write is already conditional on the revision it read (see
    /// `update_points`), so this is no longer what keeps writes from undoing
    /// one another; another process never sees it. What it still buys: the
    /// service's spawned tasks (search-time access increments, enrichment,
    /// duplicate merge) do not race each other, so the bounded CAS retries
    /// are spent only on cross-process races — a burst of in-process
    /// increments on one hot memory would otherwise exhaust them and drop
    /// counts. Held by every writer; reads never take it (alaya#86, #130).
    write_lock: futures::lock::Mutex<()>,
    /// Random per client. With `writes`, makes every revision token unique
    /// across processes (see `stamp`).
    instance: u64,
    writes: AtomicU64,
}

impl QdrantClient {
    /// Fails with `Config` on bearer material that is not a valid header
    /// value (e.g. a trailing newline, #97) or on a client build error. The
    /// error must never echo `api_key`: `InvalidHeaderValue` carries no
    /// payload and neither message below interpolates the key.
    pub fn new(base_url: String, collection: String, api_key: Option<String>) -> Result<Self> {
        let mut headers = HeaderMap::new();
        if let Some(key) = api_key {
            let val = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|e| {
                AlayaError::Config(format!(
                    "qdrant api key is not valid HTTP header material: {e}"
                ))
            })?;
            headers.insert(AUTHORIZATION, val);
        }

        let builder = reqwest::Client::builder().default_headers(headers);

        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30));

        let client = builder.build().map_err(|e| {
            AlayaError::Config(format!(
                "qdrant HTTP client: {}",
                crate::redact_reqwest_error(e)
            ))
        })?;

        let tag_collection = format!("{collection}_tags");
        Ok(Self {
            client,
            base_url,
            collection,
            tag_collection,
            write_lock: futures::lock::Mutex::new(()),
            // OS-seeded per process on native targets. wasm32 (deferred, no
            // deployment) has no seed source in std and gets fixed keys.
            instance: std::collections::hash_map::RandomState::new().hash_one(0u8),
            writes: AtomicU64::new(0),
        })
    }

    /// Refuse a Qdrant that cannot make writes conditional. `update_mode`
    /// (insert-only / update-only upserts) arrived in 1.17, and older servers
    /// silently ignore it: an insert-only write would overwrite an existing
    /// memory and an update-only write would resurrect a deleted one
    /// (alaya#130). Returns the server version. `Config` when the version is
    /// too old or unreadable — retrying will not help; `Storage` when Qdrant
    /// could not be asked.
    pub async fn check_server_version(&self) -> Result<String> {
        let resp = self
            .client
            .get(format!("{}/", self.base_url))
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;
        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }
        let body: Value = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;
        let version = body
            .get("version")
            .and_then(Value::as_str)
            .ok_or_else(|| AlayaError::Config("Qdrant did not report its version".into()))?;
        match meets_min_version(version) {
            Some(true) => Ok(version.to_string()),
            Some(false) => Err(AlayaError::Config(format!(
                "Qdrant {version} is too old: conditional writes need Qdrant >= {}.{} \
                 (update_mode); an older server ignores it and would overwrite",
                MIN_QDRANT_VERSION.0, MIN_QDRANT_VERSION.1
            ))),
            None => Err(AlayaError::Config(format!(
                "unrecognised Qdrant version {version:?}"
            ))),
        }
    }

    /// Stamp `write` (a payload object) with a fresh revision: `rev` becomes
    /// a token no other write in any process will ever use, and it is
    /// appended to the `rev_log` read in `prev`, oldest entries dropped past
    /// `REV_LOG_LEN`. Returns the token.
    fn stamp(&self, write: &mut Value, prev: Option<&Value>) -> String {
        let rev = format!(
            "{:016x}-{:x}",
            self.instance,
            self.writes.fetch_add(1, Ordering::Relaxed)
        );
        let mut log = prev
            .and_then(|p| p.get(REV_LOG))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        log.push(json!(rev));
        let excess = log.len().saturating_sub(REV_LOG_LEN);
        log.drain(..excess);
        write[REV] = json!(rev);
        write[REV_LOG] = Value::Array(log);
        rev
    }

    /// Write `payload` to one point only while its revision is still
    /// `read_rev`. A point whose revision moved, or that is gone, is left
    /// untouched and Qdrant still answers 200 — only a read-back tells.
    async fn write_payload_if(
        &self,
        point_id: &str,
        read_rev: Option<&str>,
        payload: &Value,
        how: PayloadWrite,
    ) -> Result<()> {
        let url = format!(
            "{}/collections/{}/points/payload?wait=true",
            self.base_url, self.collection
        );
        let request = match how {
            PayloadWrite::Merge => self.client.post(url),
            PayloadWrite::Replace => self.client.put(url),
        };
        let body = json!({
            "payload": payload,
            "filter": { "must": [{ "has_id": [point_id] }, rev_condition(read_rev)] },
        });
        let resp = request
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;
        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }
        Ok(())
    }

    /// Read-modify-write points under compare-and-set on `rev` (alaya#130).
    ///
    /// Reads each point (payload restricted to `keys` plus the revision keys,
    /// or all of it when `keys` is `None`), lets `modify` build the write from
    /// what it read — `None` leaves the point alone — commits it conditionally
    /// on the revision read, and reads back to see whether it landed. A write
    /// that lost a race is rebuilt from a fresh read, up to
    /// `MAX_WRITE_ATTEMPTS` rounds; there is never an unconditional fallback.
    /// Every write of a round is built before any is sent, so an error from
    /// `modify` in the first round aborts with nothing written; a read error
    /// aborts too. A failed write is that point's outcome alone: the rest of
    /// the batch still goes out, and the caller learns per point what landed.
    /// `batch` decides what a point absent from the first read does.
    async fn update_points(
        &self,
        ids: &[String],
        keys: Option<&[&str]>,
        how: PayloadWrite,
        batch: Batch,
        modify: impl Fn(&Value) -> Result<Option<Value>>,
    ) -> Result<HashMap<String, Update>> {
        let keys: Option<Vec<&str>> =
            keys.map(|k| k.iter().copied().chain([REV, REV_LOG]).collect());
        let mut pending: Vec<String> = ids.to_vec();
        pending.sort();
        pending.dedup();
        let mut outcome = HashMap::new();

        for round in 0..MAX_WRITE_ATTEMPTS {
            if pending.is_empty() {
                break;
            }
            let read = by_id(self.retrieve(&pending, keys.as_deref()).await?);
            if round == 0
                && batch == Batch::Strict
                && pending.iter().any(|id| !read.contains_key(id))
            {
                for id in pending {
                    let update = if read.contains_key(&id) {
                        Update::Skipped
                    } else {
                        Update::Absent
                    };
                    outcome.insert(id, update);
                }
                return Ok(outcome);
            }

            let mut writes = Vec::new();
            for id in pending.drain(..) {
                let Some(prev) = read.get(&id) else {
                    let gone = if round == 0 {
                        Update::Absent
                    } else {
                        Update::Deleted
                    };
                    outcome.insert(id, gone);
                    continue;
                };
                let Some(mut write) = modify(prev)? else {
                    outcome.insert(id, Update::Skipped);
                    continue;
                };
                let read_rev = rev_of(prev).map(str::to_owned);
                let mine = self.stamp(&mut write, Some(prev));
                writes.push((id, read_rev, mine, write));
            }

            let mut written = Vec::new();
            for (id, read_rev, mine, write) in writes {
                match self
                    .write_payload_if(&id, read_rev.as_deref(), &write, how)
                    .await
                {
                    Ok(()) => written.push((id, read_rev, mine)),
                    Err(e) => {
                        outcome.insert(id, Update::Unconfirmed(e.to_string()));
                    }
                }
            }
            if written.is_empty() {
                break;
            }

            let ids: Vec<String> = written.iter().map(|(id, _, _)| id.clone()).collect();
            let mut back = by_id(self.retrieve(&ids, keys.as_deref()).await?);
            for (id, read_rev, mine) in written {
                let now = back.remove(&id);
                match landed(read_rev.as_deref(), now.as_ref(), &mine) {
                    Landed::Yes => {
                        outcome.insert(id, Update::Landed(now.unwrap_or_default()));
                    }
                    Landed::No => pending.push(id),
                    Landed::Gone => {
                        outcome.insert(id, Update::Deleted);
                    }
                    Landed::Unknown => {
                        outcome.insert(id, Update::Unconfirmed(unconfirmed(&mine)));
                    }
                }
            }
        }

        for id in pending {
            outcome.insert(
                id,
                Update::Unconfirmed(format!(
                    "write lost {MAX_WRITE_ATTEMPTS} races in a row; not applied"
                )),
            );
        }
        Ok(outcome)
    }

    /// Record one access on each point: count + 1 and a capped timestamp
    /// history, both computed from the copy the write is conditional on, so
    /// two processes counting the same hit record two accesses, not one.
    /// The caller holds `write_lock`.
    async fn bump_access(&self, point_ids: &[String]) -> Result<HashMap<String, Update>> {
        let now = now_secs();
        self.update_points(
            point_ids,
            Some(&["access_count", "access_timestamps"]),
            PayloadWrite::Merge,
            Batch::BestEffort,
            |prev| {
                let count = prev
                    .get("access_count")
                    .and_then(Value::as_u64)
                    .unwrap_or(0);
                let mut timestamps: Vec<f64> = prev
                    .get("access_timestamps")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_f64).collect())
                    .unwrap_or_default();
                timestamps.push(now);
                cap_timestamps(&mut timestamps, MAX_ACCESS_TIMESTAMPS);
                Ok(Some(json!({
                    "access_count": count + 1,
                    "access_timestamps": timestamps,
                })))
            },
        )
        .await
    }

    /// PUT one point, `update_mode` `insert_only` (no `condition`) or
    /// `update_only` conditional on `condition`.
    async fn upsert_point(
        &self,
        point_id: &str,
        vector: &[f32],
        payload: &Value,
        condition: Option<Value>,
    ) -> Result<()> {
        let mut body = json!({
            "points": [{ "id": point_id, "vector": vector, "payload": payload }],
            "update_mode": if condition.is_some() { "update_only" } else { "insert_only" },
        });
        // `update_filter` takes a whole Filter; a bare condition is a 400.
        if let Some(c) = condition {
            body["update_filter"] = json!({ "must": [c] });
        }
        let resp = self
            .client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;
        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }
        Ok(())
    }

    /// Ensure the memory collections exist, creating any that are absent with
    /// the configured vector size and Cosine distance. Idempotent — an existing
    /// collection is left untouched (never recreated, so no data is dropped).
    /// Run once at startup so a fresh Qdrant volume accepts writes with no
    /// manual `curl -X PUT` bootstrap (#31).
    ///
    /// The main collection is required: an error propagates so the caller can
    /// retry (Qdrant may not be ready yet at boot). The `{collection}_tags`
    /// sidecar is best-effort — tag upserts and semantic-tag search are already
    /// non-fatal in the service layer, so its absence must not block startup.
    pub async fn ensure_collection(&self, dimensions: usize) -> Result<()> {
        self.ensure_one(&self.collection, dimensions).await?;
        if let Err(e) = self.ensure_one(&self.tag_collection, dimensions).await {
            tracing::warn!(
                collection = %self.tag_collection,
                error = %e,
                "tag collection ensure failed (non-fatal)"
            );
        }
        Ok(())
    }

    /// Create `collection` with `dimensions`-wide Cosine vectors if it does not
    /// already exist; a present collection is a no-op.
    async fn ensure_one(&self, collection: &str, dimensions: usize) -> Result<()> {
        // Existence probe: a 2xx means present (no-op), a 404 means absent
        // (create). Any other status or a transport error is a real fault —
        // propagate it so the startup retry backs off on the true cause rather
        // than misreading it as "missing" and firing a doomed create.
        let resp = self
            .client
            .get(format!("{}/collections/{}", self.base_url, collection))
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if resp.status().is_success() {
            tracing::debug!(collection = %collection, "Qdrant collection present");
            return Ok(());
        }
        if resp.status() != reqwest::StatusCode::NOT_FOUND {
            return Err(qdrant_error(resp).await);
        }

        // Absent (404) → create with the configured vector size and Cosine distance.
        let body = json!({ "vectors": { "size": dimensions, "distance": "Cosine" } });
        let resp = self
            .client
            .put(format!("{}/collections/{}", self.base_url, collection))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        tracing::info!(
            collection = %collection,
            dimensions,
            "created Qdrant collection (distance=Cosine)"
        );
        Ok(())
    }
}

// ─── Conditional writes (alaya#130) ─────────────────────────────────────────
//
// More than one process may write the memories collection, so no in-process
// lock can order their writes. Every write to an existing point instead
// commits only while the point still carries the revision it read, and every
// write stamps a fresh one. Qdrant answers 200 whether or not a conditional
// write applied, so the writer reads back: its token in `rev_log` means it
// landed; the token it conditioned on still in the log without its own means
// it lost the race and rebuilds from a fresh read.

/// Oldest Qdrant that honours `update_mode`; older servers ignore it. The
/// protocol also assumes one copy of each point (single node,
/// `replication_factor` 1): under Qdrant's default weak write ordering each
/// replica evaluates a condition on its own.
const MIN_QDRANT_VERSION: (u64, u64) = (1, 17);
/// Payload key: the revision token of the last write. Absent on points last
/// written before conditional writes, which count as the zero revision.
const REV: &str = "rev";
/// Payload key: the last `REV_LOG_LEN` revision tokens, oldest first.
const REV_LOG: &str = "rev_log";
/// Writes that may land between a write and its read-back before the
/// read-back can no longer say whether that write landed.
const REV_LOG_LEN: usize = 8;
/// Rounds a writer retries a lost race before it returns an error.
const MAX_WRITE_ATTEMPTS: usize = 8;

/// How a conditional payload write lands: `Merge` sets the given keys and
/// keeps the rest (Qdrant set payload); `Replace` writes the whole payload
/// (overwrite payload), for writers that must remove a key.
#[derive(Clone, Copy)]
enum PayloadWrite {
    Merge,
    Replace,
}

/// How `update_points` treats a point absent from its first read.
#[derive(Clone, Copy, PartialEq)]
enum Batch {
    /// Every point must exist, or nothing is written (the present ones come
    /// back `Skipped`). For writes that name the memories they change — a
    /// supersession, a patch.
    Strict,
    /// An absent point is skipped. For fire-and-forget writes (access counts).
    BestEffort,
}

/// What `update_points` did to one point.
enum Update {
    /// The write landed; the payload as read back.
    Landed(Value),
    /// `modify` chose not to write.
    Skipped,
    /// No such point when first read.
    Absent,
    /// Present when first read, deleted before the write was confirmed.
    Deleted,
    /// Not applied, or applied but unconfirmable; the reason.
    Unconfirmed(String),
}

/// Whether a conditional write landed, judged from its read-back.
#[derive(Debug, PartialEq)]
enum Landed {
    Yes,
    /// Lost the race: another write took the revision first.
    No,
    /// More than `REV_LOG_LEN` writes, or a delete and re-create, since the
    /// read: the log no longer reaches back far enough to tell.
    Unknown,
    /// The point is gone.
    Gone,
}

/// Whether `version` (e.g. `1.17.1`) is at least `MIN_QDRANT_VERSION`;
/// `None` when it does not parse.
fn meets_min_version(version: &str) -> Option<bool> {
    let mut parts = version.trim().split('.');
    let major: u64 = parts.next()?.parse().ok()?;
    let minor: String = parts
        .next()?
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    Some((major, minor.parse::<u64>().ok()?) >= MIN_QDRANT_VERSION)
}

fn rev_of(payload: &Value) -> Option<&str> {
    payload.get(REV).and_then(Value::as_str)
}

/// Filter matching a point whose revision is `rev`; `None` matches a point
/// with no revision yet.
fn rev_condition(rev: Option<&str>) -> Value {
    match rev {
        Some(r) => json!({ "key": REV, "match": { "value": r } }),
        None => json!({ "is_empty": { "key": REV } }),
    }
}

/// Judge from the payload read back after a write stamped `mine`, made
/// conditional on `read_rev`, whether that write landed. Every write appends
/// its token to the log it read, so the log is a suffix of the point's write
/// history: had ours landed it would sit right after `read_rev` (or first,
/// on a point with no revision), and it is absent only if the log no longer
/// reaches back that far.
fn landed(read_rev: Option<&str>, readback: Option<&Value>, mine: &str) -> Landed {
    let Some(payload) = readback else {
        return Landed::Gone;
    };
    let log: Vec<&str> = payload
        .get(REV_LOG)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    if log.contains(&mine) {
        return Landed::Yes;
    }
    let reaches_back = match read_rev {
        Some(r) => log.contains(&r),
        None => log.len() < REV_LOG_LEN,
    };
    if reaches_back {
        Landed::No
    } else {
        Landed::Unknown
    }
}

fn unconfirmed(rev: &str) -> String {
    format!(
        "could not confirm write {rev}: more than {REV_LOG_LEN} writes or a delete landed before \
         its read-back; it may or may not have applied"
    )
}

/// Retrieved points keyed by id, payload only.
fn by_id(points: Vec<Value>) -> HashMap<String, Value> {
    points
        .into_iter()
        .filter_map(|mut p| {
            let id = p.get("id")?.as_str()?.to_string();
            Some((
                id,
                p.get_mut("payload").map(Value::take).unwrap_or_default(),
            ))
        })
        .collect()
}

/// The whole payload `patch` leaves behind on a point that held `prev`.
fn apply_patch(prev: &Value, patch: &PatchMemoryRequest, now: f64) -> Value {
    let mut next = prev.clone();

    // summary_embedding is derived from the summary text. A patch that
    // changes the text without a replacement vector must remove the old one,
    // or the record carries an embedding of text it no longer holds and every
    // later re-store faithfully preserves that stale pair (see `store`). The
    // removal lands in the same write as the new summary.
    if let Some(new_summary) = &patch.summary
        && patch.summary_embedding.is_none()
        && prev.get("summary").and_then(Value::as_str) != Some(new_summary.as_str())
        && let Some(obj) = next.as_object_mut()
    {
        obj.remove("summary_embedding");
    }

    if let Some(ref tags) = patch.tags {
        next["tags"] = json!(tags);
    }
    if let Some(ref summary) = patch.summary {
        next["summary"] = json!(summary);
    }
    if let Some(ref memory_type) = patch.memory_type {
        next["memory_type"] = json!(memory_type);
    }
    if let Some(ref se) = patch.summary_embedding {
        next["summary_embedding"] = json!(se);
    }

    // Metadata merge: apply incoming keys, delete null keys
    if let Some(ref incoming) = patch.metadata {
        let mut merged = prev
            .get("metadata")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        for (k, v) in incoming {
            if v.is_null() {
                merged.remove(k);
            } else {
                merged.insert(k.clone(), v.clone());
            }
        }
        next["metadata"] = Value::Object(merged);
    }

    next["updated_at"] = json!(now);
    next
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ─── Access timestamp capping ────────────────────────────────────────────────

/// Maximum number of access timestamps to retain per memory.
/// The spaced_repetition module only needs recent inter-access intervals,
/// so 100 entries is more than sufficient.
const MAX_ACCESS_TIMESTAMPS: usize = 100;

/// Trim `timestamps` to keep only the most recent `max` entries.
fn cap_timestamps(timestamps: &mut Vec<f64>, max: usize) {
    if timestamps.len() > max {
        let drain_count = timestamps.len() - max;
        timestamps.drain(..drain_count);
    }
}

// ─── UUID generation (must match Python _hash_to_uuid) ──────────────────────

/// Convert a content_hash (64-char hex SHA-256) to a UUID string.
///
/// Python takes the first 32 hex chars and formats as UUID-4 style.
/// Must match exactly for data compatibility.
fn hash_to_uuid(content_hash: &str) -> Result<String> {
    // Byte length alone is not enough: a 64-byte value holding a multibyte
    // character would make the slice below panic mid-character. All-hex
    // guarantees ASCII, so every byte index is a char boundary.
    if content_hash.len() != 64 || !content_hash.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(AlayaError::Validation(format!(
            "content_hash must be 64-char SHA-256 hex, got {} chars. \
             Pass the full content_hash from search/store_memory results, \
             not a truncated display or log prefix.",
            content_hash.len()
        )));
    }
    let hex = &content_hash[..32];
    uuid::Uuid::parse_str(hex)
        .map(|u| u.to_string())
        .map_err(|e| AlayaError::Validation(format!("invalid content_hash hex: {e}")))
}

// ─── Filter construction ────────────────────────────────────────────────────

fn build_filter(filter: &PayloadFilter) -> Value {
    let mut must = Vec::new();
    let mut should = Vec::new();

    if let Some(ref mt) = filter.memory_type {
        must.push(json!({
            "key": "memory_type",
            "match": { "value": mt }
        }));
    }

    if let Some(ref tags) = filter.tags {
        if filter.tags_match_all {
            // AND — each tag must be present
            for tag in tags {
                must.push(json!({
                    "key": "tags",
                    "match": { "value": tag }
                }));
            }
        } else {
            // OR — any tag matches
            for tag in tags {
                should.push(json!({
                    "key": "tags",
                    "match": { "value": tag }
                }));
            }
        }
    }

    // Note: exclude_superseded is handled at the application layer (MemoryService)
    // after retrieval, not at the Qdrant filter level. Qdrant's nested payload
    // filtering for "field does not exist" is unreliable without explicit indexes.

    if let Some(min_trust) = filter.min_trust_score {
        must.push(json!({
            "key": "metadata.provenance.trust_score",
            "range": { "gte": min_trust }
        }));
    }

    let mut f = json!({});
    if !must.is_empty() {
        f["must"] = json!(must);
    }
    if !should.is_empty() {
        f["should"] = json!(should);
    }
    f
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Parse a Qdrant point into a Memory struct.
fn point_to_memory(point: &Value) -> Option<Memory> {
    parse_payload(point.get("payload")?)
}

/// Parse a Qdrant point payload into a Memory struct.
fn parse_payload(payload: &Value) -> Option<Memory> {
    Some(Memory {
        content: payload.get("content")?.as_str()?.to_string(),
        content_hash: payload.get("content_hash")?.as_str()?.to_string(),
        tags: payload
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default(),
        memory_type: payload
            .get("memory_type")
            .and_then(|v| v.as_str())
            .unwrap_or("note")
            .to_string(),
        metadata: payload
            .get("metadata")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        created_at: payload
            .get("created_at")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        updated_at: payload
            .get("updated_at")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        embedding: None, // Never return embeddings from Qdrant queries
        summary: payload
            .get("summary")
            .and_then(|v| v.as_str())
            .map(String::from),
        salience_score: payload
            .get("salience_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0),
        access_count: payload
            .get("access_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        access_timestamps: payload
            .get("access_timestamps")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
            .unwrap_or_default(),
        emotional_valence: payload
            .get("emotional_valence")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        encoding_context: payload
            .get("encoding_context")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        provenance: payload
            .get("provenance")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        summary_embedding: payload
            .get("summary_embedding")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            }),
    })
}

fn point_to_scored(point: &Value) -> Option<ScoredMemory> {
    let score = point.get("score")?.as_f64()?;
    let memory = point_to_memory(point)?;
    Some(ScoredMemory { memory, score })
}

/// Top-level fields of a metadata update (`access_count`, `extra`) — these live at the
/// payload root, matching where they are stored and read. Never contains `superseded_by`.
fn top_level_payload(updates: &MetadataUpdate) -> serde_json::Map<String, Value> {
    let mut payload = serde_json::Map::new();
    if let Some(ac) = updates.access_count {
        payload.insert("access_count".into(), json!(ac));
    }
    if let Some(ref extra) = updates.extra {
        for (k, v) in extra {
            payload.insert(k.clone(), v.clone());
        }
    }
    payload
}

/// Build the payload JSON for upsert from a Memory struct.
fn memory_to_payload(memory: &Memory) -> Value {
    let mut payload = json!({
        "content": memory.content,
        "content_hash": memory.content_hash,
        "tags": memory.tags,
        "memory_type": memory.memory_type,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
        "salience_score": memory.salience_score,
        "access_count": memory.access_count,
        "access_timestamps": memory.access_timestamps,
    });

    if let Some(ref m) = memory.metadata {
        payload["metadata"] = json!(m);
    }
    if let Some(ref s) = memory.summary {
        payload["summary"] = json!(s);
    }
    if let Some(ref ev) = memory.emotional_valence {
        payload["emotional_valence"] = json!(ev);
    }
    if let Some(ref ec) = memory.encoding_context {
        payload["encoding_context"] = json!(ec);
    }
    if let Some(ref p) = memory.provenance {
        payload["provenance"] = json!(p);
    }
    if let Some(ref se) = memory.summary_embedding {
        payload["summary_embedding"] = json!(se);
    }

    payload
}

/// Carry the server-maintained fields of `prev`, the stored payload, over the
/// caller's `payload` on a re-store (see `VectorStorage::store`).
fn carry_over(payload: &mut Value, prev: &Value) {
    for key in [
        "created_at",
        "access_count",
        "access_timestamps",
        "supersession_reason",
    ] {
        if let Some(v) = prev.get(key) {
            payload[key] = v.clone();
        }
    }
    // The supersession marker is written by mark_superseded, never by a store
    // caller, so it is server-maintained too: a re-store must not resurrect a
    // superseded memory (alaya-core's is_superseded reads exactly this key).
    if let Some(sb) = prev.pointer("/metadata/superseded_by") {
        payload["metadata"]["superseded_by"] = sb.clone();
    }
    // summary_embedding is derived server-side from the summary text. Keep it
    // only while that text is unchanged, so a re-store neither drops it
    // silently nor keeps a vector for a summary it no longer describes.
    if payload.get("summary_embedding").is_none()
        && prev.get("summary").is_some()
        && prev.get("summary") == payload.get("summary")
        && let Some(se) = prev.get("summary_embedding")
    {
        payload["summary_embedding"] = se.clone();
    }
}

async fn qdrant_error(resp: reqwest::Response) -> AlayaError {
    let status = resp.status();
    let body = resp.text().await.unwrap_or_default();
    AlayaError::Storage(format!("Qdrant {status}: {body}"))
}

impl QdrantClient {
    /// Retrieve one raw point (payload, no vector) by id; `None` when absent.
    async fn retrieve_point(&self, point_id: &str) -> Result<Option<Value>> {
        Ok(self
            .retrieve(&[point_id.to_string()], None)
            .await?
            .into_iter()
            .next())
    }

    /// Retrieve raw points by id, no vectors; the payload restricted to
    /// `keys`, or whole when `None`. Absent ids are simply missing.
    async fn retrieve(&self, point_ids: &[String], keys: Option<&[&str]>) -> Result<Vec<Value>> {
        let body = json!({
            "ids": point_ids,
            "with_payload": keys.map_or(json!(true), |k| json!(k)),
            "with_vector": false,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        // `result: []` is the only legitimate "absent". A 2xx whose `result`
        // is missing or null is a protocol violation; treating it as absent
        // would send `store` down the insert path for a present point and make
        // a read-back report a landed write as a delete. Fail closed.
        data.result
            .ok_or_else(|| AlayaError::Storage("Qdrant retrieve returned no result".into()))
    }
}

// ─── VectorStorage ──────────────────────────────────────────────────────────

#[async_trait(?Send)]
impl VectorStorage for QdrantClient {
    #[tracing::instrument(skip(self, memory), fields(hash = %memory.content_hash))]
    async fn store(&self, memory: &Memory, mode: StoreMode) -> Result<(bool, String)> {
        let hash = &memory.content_hash;
        let point_id = hash_to_uuid(hash)?;
        let embedding = memory
            .embedding
            .as_ref()
            .ok_or_else(|| AlayaError::Validation("memory has no embedding".into()))?;

        let _write = self.write_lock.lock().await;

        for _ in 0..MAX_WRITE_ATTEMPTS {
            // Qdrant upsert replaces the payload wholesale and the point id is
            // derived from content_hash, so the existing point's
            // server-maintained fields must be carried over or a re-store
            // silently zeroes them (alaya#86). Existence is judged on the raw
            // point: a payload that no longer parses as a Memory still exists
            // and must not be overwritten as new. Fail closed: a retrieve error
            // propagates rather than falling through to a write.
            let existing = self.retrieve_point(&point_id).await?;
            if existing.is_some() && mode == StoreMode::InsertOnly {
                return Ok((false, hash.clone()));
            }
            let prev = existing
                .as_ref()
                .map(|p| p.get("payload").unwrap_or(&Value::Null));
            let mut payload = memory_to_payload(memory);
            if let Some(prev) = prev {
                carry_over(&mut payload, prev);
            }
            let read_rev = prev.and_then(rev_of).map(str::to_owned);
            let mine = self.stamp(&mut payload, prev);

            // New content is insert-only: losing the insert race to another
            // process leaves its point untouched and the next round re-stores
            // over it. A re-store is update-only and conditional on the
            // revision its carry-over read, so a supersession, access
            // increment or delete that landed in between rejects it instead
            // of being rolled back, and it rebuilds from a fresh read.
            let condition = existing
                .as_ref()
                .map(|_| rev_condition(read_rev.as_deref()));
            self.upsert_point(&point_id, embedding, &payload, condition)
                .await?;

            let back = self
                .retrieve(std::slice::from_ref(&point_id), Some(&[REV, REV_LOG]))
                .await?;
            let now = back
                .first()
                .map(|p| p.get("payload").unwrap_or(&Value::Null));
            match landed(read_rev.as_deref(), now, &mine) {
                Landed::Yes => return Ok((existing.is_none(), hash.clone())),
                Landed::No => continue,
                // Deleted after our write or before it: either way nothing of
                // this store survives, and retrying could resurrect a memory
                // deleted after our write had landed. Say so.
                Landed::Gone => {
                    return Err(AlayaError::Conflict(format!(
                        "memory {hash} was deleted while it was being stored; store it again \
                         to keep it"
                    )));
                }
                Landed::Unknown => return Err(AlayaError::Storage(unconfirmed(&mine))),
            }
        }
        Err(AlayaError::Storage(format!(
            "store of {hash} lost {MAX_WRITE_ATTEMPTS} write races in a row; not stored"
        )))
    }

    async fn get_by_hash(&self, content_hash: &str) -> Result<Option<Memory>> {
        let point_id = hash_to_uuid(content_hash)?;
        Ok(self
            .retrieve_point(&point_id)
            .await?
            .as_ref()
            .and_then(point_to_memory))
    }

    async fn get_batch(&self, hashes: &[&str]) -> Result<Vec<Memory>> {
        if hashes.is_empty() {
            return Ok(Vec::new());
        }

        let ids: Vec<String> = hashes
            .iter()
            .map(|h| hash_to_uuid(h))
            .collect::<Result<Vec<String>>>()?;

        Ok(self
            .retrieve(&ids, None)
            .await?
            .iter()
            .filter_map(point_to_memory)
            .collect())
    }

    async fn delete(&self, content_hash: &str) -> Result<bool> {
        let point_id = hash_to_uuid(content_hash)?;
        // The one unconditional write, and safe as one: it reads nothing, and
        // a conditional write that read the point before it is rejected after
        // it (update-only upsert, `has_id` filter), so no snapshot can bring
        // the point back (see `update_points`).
        let _write = self.write_lock.lock().await;

        let body = json!({
            "points": [point_id]
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/delete?wait=true",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok(true)
    }

    async fn update_metadata(&self, content_hash: &str, updates: MetadataUpdate) -> Result<()> {
        self.update_metadata_batch(&[content_hash], updates).await
    }

    async fn update_metadata_batch(
        &self,
        content_hashes: &[&str],
        updates: MetadataUpdate,
    ) -> Result<()> {
        if content_hashes.is_empty() {
            return Ok(());
        }

        let _write = self.write_lock.lock().await;

        let point_ids: Vec<String> = content_hashes
            .iter()
            .map(|h| hash_to_uuid(h))
            .collect::<Result<Vec<String>>>()?;

        // One conditional write per memory carries the top-level fields
        // (supersession_reason, access_count) and the supersession marker
        // together, so the marker never lands without its reason. The marker
        // lives INSIDE the nested metadata object (issue #54: dotted map keys
        // create flat literal fields), which is rewritten whole from the copy
        // just read — safe only because the write is conditional on it.
        let top = top_level_payload(&updates);
        let outcome = self
            .update_points(
                &point_ids,
                Some(&["metadata"]),
                PayloadWrite::Merge,
                Batch::Strict,
                |prev| {
                    let mut write = Value::Object(top.clone());
                    if let Some(ref sb) = updates.superseded_by {
                        let mut metadata = match prev.get("metadata") {
                            None | Some(Value::Null) => serde_json::Map::new(),
                            Some(Value::Object(m)) => m.clone(),
                            Some(_) => {
                                return Err(AlayaError::Storage(
                                    "stored metadata is not an object; refusing to replace it"
                                        .into(),
                                ));
                            }
                        };
                        metadata.insert("superseded_by".into(), json!(sb));
                        write["metadata"] = Value::Object(metadata);
                    }
                    Ok(write
                        .as_object()
                        .is_some_and(|w| !w.is_empty())
                        .then_some(write))
                },
            )
            .await?;

        for (hash, id) in content_hashes.iter().zip(&point_ids) {
            match outcome.get(id) {
                Some(Update::Landed(_) | Update::Skipped) => {}
                // Gone mid-update: a deleted memory needs no marker, and the
                // rest of the batch has committed, so its caller must go on.
                Some(Update::Deleted) => {
                    tracing::warn!(%hash, "memory deleted during a metadata update; skipped");
                }
                Some(Update::Absent) | None => {
                    return Err(AlayaError::NotFound(format!("memory {hash} not found")));
                }
                Some(Update::Unconfirmed(why)) => {
                    return Err(AlayaError::Storage(format!(
                        "metadata update of {hash}: {why}"
                    )));
                }
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self, patch), fields(hash = %content_hash))]
    async fn patch_memory(&self, content_hash: &str, patch: &PatchMemoryRequest) -> Result<Memory> {
        if patch.is_empty() {
            return Err(AlayaError::Validation("patch is empty".into()));
        }

        let point_id = hash_to_uuid(content_hash)?;
        let not_found = || AlayaError::NotFound(format!("memory {content_hash} not found"));
        let now = now_secs();

        let _write = self.write_lock.lock().await;

        // The whole payload is rewritten from the copy just read, which is
        // what lets the summary and the removal of its stale embedding land
        // as one write; the write is conditional on that copy.
        let mut outcome = self
            .update_points(
                std::slice::from_ref(&point_id),
                None,
                PayloadWrite::Replace,
                Batch::Strict,
                |prev| {
                    if parse_payload(prev).is_none() {
                        return Err(not_found());
                    }
                    Ok(Some(apply_patch(prev, patch, now)))
                },
            )
            .await?;
        match outcome.remove(&point_id) {
            Some(Update::Landed(payload)) => parse_payload(&payload).ok_or_else(|| {
                AlayaError::Storage(format!("patched memory {content_hash} no longer parses"))
            }),
            Some(Update::Unconfirmed(why)) => Err(AlayaError::Storage(format!(
                "patch of {content_hash}: {why}"
            ))),
            Some(Update::Absent | Update::Deleted | Update::Skipped) | None => Err(not_found()),
        }
    }

    async fn set_generated_summary(
        &self,
        content_hash: &str,
        summary: &str,
        summary_embedding: Option<Vec<f32>>,
    ) -> Result<bool> {
        let point_id = hash_to_uuid(content_hash)?;
        let not_found = || AlayaError::NotFound(format!("memory {content_hash} not found"));
        let patch = PatchMemoryRequest {
            summary: Some(summary.to_string()),
            summary_embedding,
            ..Default::default()
        };
        let now = now_secs();

        let _write = self.write_lock.lock().await;

        // Decided on the copy the write is conditional on: a caller summary
        // that landed after the background job started wins, because the job
        // either sees it here or loses the race and sees it on the re-read.
        let mut outcome = self
            .update_points(
                std::slice::from_ref(&point_id),
                None,
                PayloadWrite::Replace,
                Batch::Strict,
                |prev| {
                    let Some(existing) = parse_payload(prev) else {
                        return Err(not_found());
                    };
                    if existing.summary.is_some() {
                        return Ok(None);
                    }
                    Ok(Some(apply_patch(prev, &patch, now)))
                },
            )
            .await?;
        match outcome.remove(&point_id) {
            Some(Update::Landed(_)) => Ok(true),
            Some(Update::Skipped) => Ok(false),
            Some(Update::Unconfirmed(why)) => Err(AlayaError::Storage(format!(
                "generated summary for {content_hash}: {why}"
            ))),
            Some(Update::Absent | Update::Deleted) | None => Err(not_found()),
        }
    }

    #[tracing::instrument(skip(self, embedding, filters), fields(limit))]
    async fn search_by_vector(
        &self,
        embedding: &[f32],
        limit: usize,
        filters: Option<PayloadFilter>,
    ) -> Result<Vec<ScoredMemory>> {
        let mut body = json!({
            "vector": embedding,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
        });

        if let Some(ref f) = filters {
            let filter = build_filter(f);
            if filter != json!({}) {
                body["filter"] = filter;
            }
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        Ok(data
            .result
            .unwrap_or_default()
            .iter()
            .filter_map(point_to_scored)
            .collect())
    }

    #[tracing::instrument(skip(self), fields(n_tags = tags.len(), match_all, limit))]
    async fn search_by_tags(
        &self,
        tags: &[&str],
        match_all: bool,
        limit: usize,
    ) -> Result<Vec<ScoredMemory>> {
        let filter = if match_all {
            let must: Vec<Value> = tags
                .iter()
                .map(|t| json!({"key": "tags", "match": {"value": t}}))
                .collect();
            json!({"must": must})
        } else {
            let should: Vec<Value> = tags
                .iter()
                .map(|t| json!({"key": "tags", "match": {"value": t}}))
                .collect();
            json!({"should": should})
        };

        let body = json!({
            "filter": filter,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/scroll",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<ScrollResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        let points = data.result.map(|r| r.points).unwrap_or_default();

        // Scroll doesn't have scores — assign 1.0 (tag-matched)
        Ok(points
            .iter()
            .filter_map(|p| {
                let memory = point_to_memory(p)?;
                Some(ScoredMemory { memory, score: 1.0 })
            })
            .collect())
    }

    #[tracing::instrument(skip(self, tag_embedding), fields(limit))]
    async fn search_similar_tags(
        &self,
        tag_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<String>> {
        let body = json!({
            "vector": tag_embedding,
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "score_threshold": 0.5,
        });

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/search",
                self.base_url, self.tag_collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<Vec<Value>> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        Ok(data
            .result
            .unwrap_or_default()
            .iter()
            .filter_map(|p| {
                p.get("payload")
                    .and_then(|pl| pl.get("tag"))
                    .and_then(|t| t.as_str())
                    .map(String::from)
            })
            .collect())
    }

    async fn upsert_tags(&self, tags: &[(&str, Vec<f32>)]) -> Result<()> {
        if tags.is_empty() {
            return Ok(());
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let points: Vec<Value> = tags
            .iter()
            .map(|(tag, embedding)| {
                let hex = format!("{:x}", Sha256::digest(tag.as_bytes()));
                let uuid = uuid::Uuid::parse_str(&hex[..32])
                    .expect("first 32 hex chars are always valid UUID input");
                json!({
                    "id": uuid.to_string(),
                    "vector": embedding,
                    "payload": { "tag": *tag, "created_at": now }
                })
            })
            .collect();

        let body = json!({ "points": points });
        let resp = self
            .client
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.base_url, self.tag_collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        Ok(())
    }

    async fn get_all(&self, limit: usize, offset: Option<&str>) -> Result<ScrollResult> {
        let mut body = json!({
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "order_by": {
                "key": "created_at",
                "direction": "desc"
            },
        });

        if let Some(off) = offset {
            body["offset"] = json!(off);
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/scroll",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<ScrollResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        let result = data.result.unwrap_or_default();
        let memories = result.points.iter().filter_map(point_to_memory).collect();

        let next_offset = result.next_page_offset.map(|v| v.to_string());

        Ok(ScrollResult {
            memories,
            next_offset,
        })
    }

    async fn get_recent(
        &self,
        limit: usize,
        start_from: Option<f64>,
        memory_type: Option<&str>,
    ) -> Result<Vec<Memory>> {
        let mut order_by = json!({
            "key": "created_at",
            "direction": "desc"
        });
        if let Some(ts) = start_from {
            order_by["start_from"] = json!(ts);
        }

        let mut body = json!({
            "limit": limit,
            "with_payload": true,
            "with_vector": false,
            "order_by": order_by,
        });

        if let Some(mt) = memory_type {
            body["filter"] = json!({
                "must": [{
                    "key": "memory_type",
                    "match": {"value": mt}
                }]
            });
        }

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/scroll",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<ScrollResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        let points = data.result.map(|r| r.points).unwrap_or_default();

        Ok(points.iter().filter_map(point_to_memory).collect())
    }

    #[tracing::instrument(skip(self))]
    async fn count(&self) -> Result<usize> {
        let body = json!({"exact": true});

        let resp = self
            .client
            .post(format!(
                "{}/collections/{}/points/count",
                self.base_url, self.collection
            ))
            .json(&body)
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Err(qdrant_error(resp).await);
        }

        let data: QdrantResponse<CountResponse> = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        Ok(data.result.map(|r| r.count).unwrap_or(0))
    }

    #[tracing::instrument(skip(self))]
    async fn get_all_tags(&self) -> Result<Vec<String>> {
        // Scroll the tag collection to get all tags
        let mut all_tags = Vec::new();
        let mut offset: Option<Value> = None;

        loop {
            let mut body = json!({
                "limit": 100,
                "with_payload": true,
                "with_vector": false,
            });

            if let Some(ref off) = offset {
                body["offset"] = off.clone();
            }

            let resp = self
                .client
                .post(format!(
                    "{}/collections/{}/points/scroll",
                    self.base_url, self.tag_collection
                ))
                .json(&body)
                .send()
                .await
                .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

            if !resp.status().is_success() {
                return Err(qdrant_error(resp).await);
            }

            let data: QdrantResponse<ScrollResponse> = resp
                .json()
                .await
                .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

            let result = data.result.unwrap_or_default();
            for point in &result.points {
                if let Some(tag) = point
                    .get("payload")
                    .and_then(|p| p.get("tag"))
                    .and_then(|t| t.as_str())
                {
                    all_tags.push(tag.to_string());
                }
            }

            match result.next_page_offset {
                Some(next) => offset = Some(next),
                None => break,
            }
        }

        Ok(all_tags)
    }

    async fn increment_access_count(&self, content_hash: &str) -> Result<()> {
        let point_id = hash_to_uuid(content_hash)?;
        let _write = self.write_lock.lock().await;
        match self
            .bump_access(std::slice::from_ref(&point_id))
            .await?
            .remove(&point_id)
        {
            Some(Update::Unconfirmed(why)) => Err(AlayaError::Storage(format!(
                "access increment of {content_hash}: {why}"
            ))),
            // Absent: nothing to count, as before.
            _ => Ok(()),
        }
    }

    #[tracing::instrument(skip(self), fields(n = content_hashes.len()))]
    async fn increment_access_count_batch(&self, content_hashes: &[&str]) -> Result<()> {
        // Non-fatal throughout: an access count is ranking input, and this
        // runs fire-and-forget after a search that has already answered.
        let point_ids: Vec<String> = content_hashes
            .iter()
            .filter_map(|h| hash_to_uuid(h).ok())
            .collect();
        if point_ids.is_empty() {
            return Ok(());
        }

        let _write = self.write_lock.lock().await;
        let outcome = match self.bump_access(&point_ids).await {
            Ok(o) => o,
            Err(e) => {
                tracing::warn!(
                    op = "increment_access_count_batch",
                    n = point_ids.len(),
                    error = %e,
                    "batch increment_access_count failed"
                );
                return Ok(());
            }
        };
        for (id, update) in outcome {
            if let Update::Unconfirmed(why) = update {
                tracing::warn!(
                    op = "increment_access_count_batch",
                    point = %id,
                    error = %why,
                    "access increment not applied"
                );
            }
        }
        Ok(())
    }

    #[tracing::instrument(skip(self))]
    async fn health(&self) -> Result<HealthStatus> {
        let resp = self
            .client
            .get(format!("{}/collections/{}", self.base_url, self.collection))
            .send()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        if !resp.status().is_success() {
            return Ok(HealthStatus {
                status: "unhealthy".into(),
                backend: "qdrant".into(),
                details: None,
            });
        }

        let data: Value = resp
            .json()
            .await
            .map_err(|e| AlayaError::Storage(crate::redact_reqwest_error(e)))?;

        let status = data
            .pointer("/result/status")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let points_count = data
            .pointer("/result/points_count")
            .and_then(|v| v.as_u64());

        let mut details = HashMap::new();
        if let Some(pc) = points_count {
            details.insert("points_count".into(), json!(pc));
        }

        Ok(HealthStatus {
            status: status.into(),
            backend: "qdrant".into(),
            details: if details.is_empty() {
                None
            } else {
                Some(details)
            },
        })
    }
}

// ─── Qdrant response wrapper types ──────────────────────────────────────────

#[derive(Deserialize)]
struct QdrantResponse<T> {
    #[allow(dead_code)]
    status: Option<String>,
    result: Option<T>,
}

#[derive(Deserialize, Default)]
struct ScrollResponse {
    #[serde(default)]
    points: Vec<Value>,
    next_page_offset: Option<Value>,
}

#[derive(Deserialize)]
struct CountResponse {
    count: usize,
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// #97 repro shape: a control character in the bearer. Must be `Config`,
    /// and the message must not echo the key it is rejecting.
    #[test]
    fn new_rejects_control_chars_without_echoing_the_key() {
        let Err(err) = QdrantClient::new(
            "http://qdrant".into(),
            "memories".into(),
            Some("abc\n".into()),
        ) else {
            panic!("a control character in the bearer must be rejected");
        };
        let msg = err.to_string();
        assert!(matches!(err, AlayaError::Config(_)), "{msg}");
        assert!(!msg.contains("abc"), "error echoed the key: {msg}");
    }

    #[test]
    fn hash_to_uuid_matches_python() {
        // Python: uuid.UUID("a" * 32) = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        let hash = "a".repeat(64);
        let uuid = hash_to_uuid(&hash).unwrap();
        assert_eq!(uuid, "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa");
    }

    #[test]
    fn hash_to_uuid_real_hash() {
        // SHA-256 of "test" starts with "9f86d081..."
        let hash = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
        let uuid = hash_to_uuid(hash).unwrap();
        assert_eq!(uuid, "9f86d081-884c-7d65-9a2f-eaa0c55ad015");
    }

    #[test]
    fn hash_to_uuid_short_input_returns_error() {
        let result = hash_to_uuid("abc");
        assert!(result.is_err());
    }

    #[test]
    fn hash_to_uuid_short_input_error_message_guides_caller() {
        // Truncated 8-char prefix from log display — the most common bad input
        let err = hash_to_uuid("ffa51984").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("8 chars"),
            "should report actual length: {msg}"
        );
        assert!(
            msg.contains("64"),
            "should reference full hash length: {msg}"
        );
        assert!(
            msg.contains("search") || msg.contains("store_memory"),
            "should point to source of full hash: {msg}"
        );
    }

    #[test]
    fn hash_to_uuid_non_hex_returns_error() {
        let result = hash_to_uuid(&"zz".repeat(32));
        assert!(result.is_err());
    }

    #[test]
    fn hash_to_uuid_multibyte_64_bytes_returns_error_not_panic() {
        // 31 ASCII + 'é' (2 bytes) + 31 ASCII = 64 bytes, byte 32 mid-character.
        let hash = format!("{}é{}", "a".repeat(31), "a".repeat(31));
        assert_eq!(hash.len(), 64);
        assert!(hash_to_uuid(&hash).is_err());
    }

    #[test]
    fn build_filter_empty() {
        let f = PayloadFilter::default();
        let filter = build_filter(&f);
        assert_eq!(filter, json!({}));
    }

    #[test]
    fn build_filter_memory_type() {
        let f = PayloadFilter {
            memory_type: Some("note".into()),
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert!(filter["must"].is_array());
        assert_eq!(filter["must"][0]["key"], "memory_type");
    }

    #[test]
    fn build_filter_tags_or() {
        let f = PayloadFilter {
            tags: Some(vec!["rust".into(), "wasm".into()]),
            tags_match_all: false,
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert!(filter["should"].is_array());
        assert_eq!(filter["should"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn build_filter_tags_and() {
        let f = PayloadFilter {
            tags: Some(vec!["rust".into(), "wasm".into()]),
            tags_match_all: true,
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert!(filter["must"].is_array());
        assert_eq!(filter["must"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn build_filter_superseded_is_noop() {
        // Superseded filtering happens at application layer, not Qdrant
        let f = PayloadFilter {
            exclude_superseded: true,
            ..Default::default()
        };
        let filter = build_filter(&f);
        assert_eq!(filter, json!({}));
    }

    #[test]
    fn build_filter_trust_score() {
        let f = PayloadFilter {
            min_trust_score: Some(0.5),
            ..Default::default()
        };
        let filter = build_filter(&f);
        let must = filter["must"].as_array().unwrap();
        assert!(must.iter().any(|c| c.get("range").is_some()));
    }

    #[test]
    fn point_to_memory_parses_full() {
        let point = json!({
            "id": "some-uuid",
            "payload": {
                "content": "hello world",
                "content_hash": "a".repeat(64),
                "tags": ["tag1", "tag2"],
                "memory_type": "note",
                "created_at": 1710432000.0,
                "updated_at": 1710432001.0,
                "salience_score": 0.5,
                "access_count": 3,
                "access_timestamps": [1.0, 2.0, 3.0],
                "summary": "a summary",
            }
        });
        let mem = point_to_memory(&point).unwrap();
        assert_eq!(mem.content, "hello world");
        assert_eq!(mem.tags.len(), 2);
        assert_eq!(mem.salience_score, 0.5);
        assert_eq!(mem.access_count, 3);
    }

    #[test]
    fn point_to_scored_includes_score() {
        let point = json!({
            "id": "uuid",
            "score": 0.95,
            "payload": {
                "content": "test",
                "content_hash": "b".repeat(64),
                "tags": [],
                "memory_type": "note",
                "created_at": 0.0,
                "updated_at": 0.0,
            }
        });
        let scored = point_to_scored(&point).unwrap();
        assert_eq!(scored.score, 0.95);
    }

    #[test]
    fn memory_to_payload_roundtrip() {
        let mem = Memory {
            content: "test content".into(),
            content_hash: "c".repeat(64),
            tags: vec!["t1".into()],
            memory_type: "note".into(),
            metadata: None,
            created_at: 100.0,
            updated_at: 200.0,
            embedding: None,
            summary: Some("summary".into()),
            salience_score: 0.7,
            access_count: 5,
            access_timestamps: vec![1.0, 2.0],
            emotional_valence: None,
            encoding_context: None,
            provenance: None,
            summary_embedding: None,
        };
        let payload = memory_to_payload(&mem);
        assert_eq!(payload["content"], "test content");
        assert_eq!(payload["salience_score"], 0.7);
        assert_eq!(payload["access_count"], 5);
        assert!(payload.get("metadata").is_none());
        assert_eq!(payload["summary"], "summary");
    }

    #[test]
    fn cap_timestamps_noop_below_limit() {
        let mut ts: Vec<f64> = (0..50).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 50);
        assert_eq!(ts[0], 0.0);
    }

    #[test]
    fn cap_timestamps_noop_at_limit() {
        let mut ts: Vec<f64> = (0..100).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 100);
        assert_eq!(ts[0], 0.0);
        assert_eq!(ts[99], 99.0);
    }

    #[test]
    fn cap_timestamps_trims_oldest_when_over_limit() {
        let mut ts: Vec<f64> = (0..150).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 100);
        // Oldest 50 removed, first remaining is 50.0
        assert_eq!(ts[0], 50.0);
        assert_eq!(ts[99], 149.0);
    }

    #[test]
    fn cap_timestamps_trims_one_over() {
        let mut ts: Vec<f64> = (0..101).map(|i| i as f64).collect();
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert_eq!(ts.len(), 100);
        assert_eq!(ts[0], 1.0);
    }

    #[test]
    fn cap_timestamps_empty_vec() {
        let mut ts: Vec<f64> = vec![];
        cap_timestamps(&mut ts, MAX_ACCESS_TIMESTAMPS);
        assert!(ts.is_empty());
    }

    #[test]
    fn top_level_payload_never_contains_supersession() {
        let mut extra = HashMap::new();
        extra.insert("supersession_reason".to_string(), json!("corrected"));
        let updates = MetadataUpdate {
            superseded_by: Some("x".to_string()),
            access_count: Some(42),
            extra: Some(extra),
        };
        let p = top_level_payload(&updates);
        assert_eq!(p["access_count"], json!(42));
        assert_eq!(p["supersession_reason"], json!("corrected"));
        assert!(p.get("superseded_by").is_none());
        assert!(p.get("metadata").is_none());
    }

    #[test]
    fn top_level_payload_empty_when_only_supersession_set() {
        let updates = MetadataUpdate {
            superseded_by: Some("x".to_string()),
            ..Default::default()
        };
        assert!(top_level_payload(&updates).is_empty());
    }

    // ─── Conditional writes (alaya#130) ─────────────────────────────────

    fn client() -> QdrantClient {
        QdrantClient::new("http://qdrant".into(), "memories".into(), None).unwrap()
    }

    /// `n` revision tokens none of which is the reader's or the writer's.
    fn others(n: usize) -> Value {
        json!((0..n).map(|i| format!("other-{i}")).collect::<Vec<_>>())
    }

    #[test]
    fn landed_yes_when_the_writers_token_is_in_the_log() {
        let back = json!({"rev": "mine", "rev_log": ["r1", "mine"]});
        assert_eq!(landed(Some("r1"), Some(&back), "mine"), Landed::Yes);
        let back = json!({"rev_log": ["mine", "later"]});
        assert_eq!(landed(None, Some(&back), "mine"), Landed::Yes);
    }

    #[test]
    fn landed_no_when_the_read_revision_is_still_in_the_log() {
        let back = json!({"rev_log": ["r0", "r1", "other"]});
        assert_eq!(landed(Some("r1"), Some(&back), "mine"), Landed::No);
    }

    /// A write conditioned on "no revision yet" that lost: its token would sit
    /// first in the log, and a log shorter than `REV_LOG_LEN` still reaches
    /// back to the point's first stamped write.
    #[test]
    fn landed_no_for_an_unrevisioned_read_while_the_log_is_short() {
        for n in [0, 1, REV_LOG_LEN - 1] {
            let back = json!({ "rev_log": others(n) });
            assert_eq!(landed(None, Some(&back), "mine"), Landed::No, "{n} entries");
        }
    }

    /// More than `REV_LOG_LEN` writes, or a delete and re-create, since the
    /// read: the log no longer says whether ours was among them.
    #[test]
    fn landed_unknown_when_the_log_rotated_past_the_read_revision() {
        let rotated = json!({ "rev_log": others(REV_LOG_LEN) });
        assert_eq!(landed(Some("r1"), Some(&rotated), "mine"), Landed::Unknown);
        let recreated = json!({"rev": "fresh", "rev_log": ["fresh"]});
        assert_eq!(
            landed(Some("r1"), Some(&recreated), "mine"),
            Landed::Unknown
        );
    }

    #[test]
    fn landed_unknown_for_an_unrevisioned_read_once_the_log_is_full() {
        let full = json!({ "rev_log": others(REV_LOG_LEN) });
        assert_eq!(landed(None, Some(&full), "mine"), Landed::Unknown);
    }

    #[test]
    fn landed_gone_when_the_point_is_absent() {
        assert_eq!(landed(Some("r1"), None, "mine"), Landed::Gone);
        assert_eq!(landed(None, None, "mine"), Landed::Gone);
    }

    #[test]
    fn meets_min_version_requires_1_17() {
        for (version, want) in [
            ("1.16.3", Some(false)),
            ("0.99.0", Some(false)),
            ("1.17.0", Some(true)),
            ("1.17.1", Some(true)),
            ("1.18.0-dev", Some(true)),
            ("2.0.0", Some(true)),
            ("garbage", None),
            ("1", None),
            ("", None),
        ] {
            assert_eq!(meets_min_version(version), want, "{version:?}");
        }
    }

    #[test]
    fn rev_condition_matches_the_read_revision_or_none_yet() {
        assert_eq!(
            rev_condition(Some("r1")),
            json!({"key": "rev", "match": {"value": "r1"}})
        );
        assert_eq!(rev_condition(None), json!({"is_empty": {"key": "rev"}}));
    }

    #[test]
    fn stamp_appends_its_token_to_the_log_it_read() {
        let prev = json!({"rev": "r1", "rev_log": ["r0", "r1"]});
        let mut write = json!({"summary": "s"});
        let token = client().stamp(&mut write, Some(&prev));
        assert_eq!(write["rev"], json!(token));
        assert_eq!(write["rev_log"], json!(["r0", "r1", token]));
        assert_eq!(
            write["summary"],
            json!("s"),
            "the rest of the write is untouched"
        );
    }

    #[test]
    fn stamp_starts_the_log_on_a_point_without_one() {
        for prev in [None, Some(json!({"content": "legacy"}))] {
            let mut write = json!({});
            let token = client().stamp(&mut write, prev.as_ref());
            assert_eq!(write["rev_log"], json!([token]), "{prev:?}");
        }
    }

    #[test]
    fn stamp_keeps_only_the_newest_rev_log_len_tokens() {
        let prev = json!({ "rev_log": others(REV_LOG_LEN) });
        let mut write = json!({});
        let token = client().stamp(&mut write, Some(&prev));
        let log = write["rev_log"].as_array().unwrap();
        assert_eq!(log.len(), REV_LOG_LEN);
        assert_eq!(log[0], json!("other-1"), "the oldest token is dropped");
        assert_eq!(log[REV_LOG_LEN - 1], json!(token));
    }

    #[test]
    fn stamp_never_reuses_a_token() {
        let (a, b) = (client(), client());
        let mut write = json!({});
        let tokens = [
            a.stamp(&mut write, None),
            a.stamp(&mut write, None),
            b.stamp(&mut write, None),
        ];
        assert_ne!(tokens[0], tokens[1], "same client");
        assert_ne!(tokens[0], tokens[2], "another client (process)");
        assert_ne!(tokens[1], tokens[2]);
    }

    fn stored() -> Value {
        json!({
            "content": "c",
            "summary": "A",
            "summary_embedding": [0.5, 0.25],
            "tags": ["t"],
            "metadata": {"keep": 1, "drop": 2},
            "rev": "r1",
            "rev_log": ["r1"],
            "custom": "kept",
            "updated_at": 1.0,
        })
    }

    #[test]
    fn apply_patch_changed_summary_without_embedding_removes_it() {
        let patch = PatchMemoryRequest {
            summary: Some("B".into()),
            ..Default::default()
        };
        let next = apply_patch(&stored(), &patch, 9.0);
        assert_eq!(next["summary"], json!("B"));
        assert!(next.get("summary_embedding").is_none(), "{next}");
        assert_eq!(next["updated_at"], json!(9.0));
    }

    #[test]
    fn apply_patch_keeps_embedding_for_an_unchanged_summary_or_replaces_it() {
        let same = PatchMemoryRequest {
            summary: Some("A".into()),
            ..Default::default()
        };
        assert_eq!(
            apply_patch(&stored(), &same, 9.0)["summary_embedding"],
            json!([0.5, 0.25])
        );

        let replaced = PatchMemoryRequest {
            summary: Some("B".into()),
            summary_embedding: Some(vec![0.75]),
            ..Default::default()
        };
        assert_eq!(
            apply_patch(&stored(), &replaced, 9.0)["summary_embedding"],
            json!([0.75])
        );
    }

    #[test]
    fn apply_patch_merges_metadata_and_deletes_null_keys() {
        let patch = PatchMemoryRequest {
            metadata: Some(HashMap::from([
                ("drop".to_string(), Value::Null),
                ("new".to_string(), json!(3)),
            ])),
            ..Default::default()
        };
        let next = apply_patch(&stored(), &patch, 9.0);
        assert_eq!(next["metadata"], json!({"keep": 1, "new": 3}));
    }

    /// The patch replaces the whole payload, so everything it does not name
    /// must come through from the copy it read.
    #[test]
    fn apply_patch_preserves_payload_keys_it_does_not_name() {
        let patch = PatchMemoryRequest {
            tags: Some(vec!["u".into()]),
            ..Default::default()
        };
        let mut want = stored();
        want["tags"] = json!(["u"]);
        want["updated_at"] = json!(9.0);
        assert_eq!(apply_patch(&stored(), &patch, 9.0), want);
    }

    #[cfg(not(target_arch = "wasm32"))]
    mod server_version {
        use super::*;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        async fn check(response: ResponseTemplate) -> Result<String> {
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/"))
                .respond_with(response)
                .mount(&server)
                .await;
            QdrantClient::new(server.uri(), "memories".into(), None)
                .unwrap()
                .check_server_version()
                .await
        }

        fn root(version: Option<&str>) -> ResponseTemplate {
            let mut body = json!({"title": "qdrant - vector search engine"});
            if let Some(v) = version {
                body["version"] = json!(v);
            }
            ResponseTemplate::new(200).set_body_json(body)
        }

        #[tokio::test]
        async fn accepts_a_qdrant_that_honours_update_mode() {
            assert_eq!(check(root(Some("1.17.1"))).await.unwrap(), "1.17.1");
        }

        #[tokio::test]
        async fn refuses_an_older_qdrant_naming_its_version() {
            let err = check(root(Some("1.16.3"))).await.unwrap_err();
            assert!(
                matches!(err, AlayaError::Config(ref m) if m.contains("1.16.3")),
                "{err:?}"
            );
        }

        #[tokio::test]
        async fn refuses_a_qdrant_that_does_not_report_its_version() {
            let err = check(root(None)).await.unwrap_err();
            assert!(matches!(err, AlayaError::Config(_)), "{err:?}");
        }

        /// Qdrant could not be asked: retryable, unlike a version refusal.
        #[tokio::test]
        async fn a_server_error_is_a_storage_error() {
            let err = check(ResponseTemplate::new(500)).await.unwrap_err();
            assert!(matches!(err, AlayaError::Storage(_)), "{err:?}");
        }
    }
}
