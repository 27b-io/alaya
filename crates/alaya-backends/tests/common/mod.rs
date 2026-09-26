//! A stateful fake of one Qdrant collection (`memories`) for wiremock.
//!
//! `QdrantClient` writes by compare-and-set on a payload `rev` (alaya#130): it
//! reads, writes conditionally on the revision it read, and reads back to see
//! whether the write applied — Qdrant answers 200 either way. Static mock
//! responses cannot model that, so this fake holds the points and applies each
//! request to them the way Qdrant 1.17 does:
//!
//! - retrieve (`POST /points`): `ids`, `with_payload` `true` or a key list;
//! - upsert (`PUT /points`): `update_mode` `upsert` (default), `insert_only`
//!   or `update_only`; `update_filter` limits which EXISTING points update;
//! - set payload (`POST /points/payload`, merge), overwrite payload (`PUT
//!   /points/payload`), delete payload keys (`POST /points/payload/delete`)
//!   and delete points (`POST /points/delete`), selected by `points` or
//!   `filter`;
//! - filters: `must` / `should` / `must_not` over `has_id`, `key` +
//!   `match.value`, `is_empty`, `is_null` and nested filters. A condition
//!   where a Filter belongs (a bare `update_filter` condition) is a 400, as in
//!   Qdrant — never a filter that matches everything.
//!
//! Anything outside that subset is a 400 naming what the fake lacks, never a
//! guess, and every write must carry `wait=true`.
//!
//! A request is applied the moment it arrives; `delay_next` only holds the
//! RESPONSE back. A delayed retrieve hands the client a snapshot that goes
//! stale while it waits; a delayed write has applied before the client learns
//! so. `before_writes` runs a hook just before a write applies: another
//! process's write landing between the client's read and its conditional write.

#![allow(dead_code)] // each test crate uses a different subset

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use serde_json::{Map, Value, json};
use wiremock::matchers::{method, path, path_regex};
use wiremock::{Mock, MockServer, Request, Respond, ResponseTemplate};

pub const POINTS_PATH: &str = "/collections/memories/points";
pub const PAYLOAD_PATH: &str = "/collections/memories/points/payload";
pub const PAYLOAD_DELETE_PATH: &str = "/collections/memories/points/payload/delete";
pub const POINTS_DELETE_PATH: &str = "/collections/memories/points/delete";

/// Revision tokens a `rev_log` keeps, as every alaya writer caps it.
pub const REV_LOG_LEN: usize = 8;

/// Point id → payload. Vectors are not kept: no test reads one back.
pub type Points = HashMap<String, Value>;

type Hook = Box<dyn FnMut(&mut Points) + Send>;
type Failure = (u16, String);

#[derive(Clone, Default)]
pub struct FakeQdrant(Arc<Mutex<State>>);

#[derive(Default)]
struct State {
    points: Points,
    /// Runs before each of the next `.0` writes applies.
    hook: Option<(usize, Hook)>,
}

impl FakeQdrant {
    pub fn insert(&self, id: &str, payload: Value) {
        self.lock().points.insert(id.to_string(), payload);
    }

    /// The payload the collection holds for `id`.
    pub fn point(&self, id: &str) -> Option<Value> {
        self.lock().points.get(id).cloned()
    }

    /// Run `hook` on the points just before each of the next `times` writes
    /// applies (another process writing inside the client's read→write
    /// window). A write the fake rejects as malformed does not consume it.
    pub fn before_writes(&self, times: usize, hook: impl FnMut(&mut Points) + Send + 'static) {
        self.lock().hook = Some((times, Box::new(hook)));
    }

    /// Serve every request to the collection's points endpoints.
    pub async fn mount(&self, server: &MockServer) {
        Mock::given(path_regex(r"^/collections/memories/points"))
            .respond_with(self.clone())
            .mount(server)
            .await;
    }

    /// Hold back the response to the next `http_method at` request by
    /// `delay`. The request itself applies on arrival (see the module docs).
    pub async fn delay_next(
        &self,
        server: &MockServer,
        http_method: &str,
        at: &str,
        delay: Duration,
    ) {
        Mock::given(method(http_method))
            .and(path(at))
            .respond_with(Delayed(self.clone(), delay))
            .up_to_n_times(1)
            .with_priority(1)
            .mount(server)
            .await;
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.0.lock().expect("fake state lock poisoned")
    }
}

impl Respond for FakeQdrant {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        match self.lock().handle(request) {
            Ok(result) => ResponseTemplate::new(200)
                .set_body_json(json!({"status": "ok", "time": 0.0, "result": result})),
            Err((status, error)) => ResponseTemplate::new(status)
                .set_body_json(json!({"status": {"error": error}, "time": 0.0})),
        }
    }
}

struct Delayed(FakeQdrant, Duration);

impl Respond for Delayed {
    fn respond(&self, request: &Request) -> ResponseTemplate {
        self.0.respond(request).set_delay(self.1)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Upsert,
    InsertOnly,
    UpdateOnly,
}

enum Select {
    Ids(Vec<String>),
    Filter(Value),
}

enum Op {
    Upsert {
        points: Vec<(String, Value)>,
        mode: Mode,
        filter: Option<Value>,
    },
    SetPayload(Map<String, Value>, Select),
    OverwritePayload(Map<String, Value>, Select),
    DeletePayload(Vec<String>, Select),
    DeletePoints(Select),
}

impl State {
    fn handle(&mut self, request: &Request) -> Result<Value, Failure> {
        let body: Value = request
            .body_json()
            .map_err(|e| bad(format!("body is not JSON: {e}")))?;
        let route = (request.method.as_str(), request.url.path());
        if route == ("POST", POINTS_PATH) {
            return self.retrieve(&body);
        }
        let op = match route {
            ("PUT", POINTS_PATH) => parse_upsert(&body)?,
            ("POST", PAYLOAD_PATH) => Op::SetPayload(payload_of(&body)?, select_of(&body)?),
            ("PUT", PAYLOAD_PATH) => Op::OverwritePayload(payload_of(&body)?, select_of(&body)?),
            ("POST", PAYLOAD_DELETE_PATH) => Op::DeletePayload(keys_of(&body)?, select_of(&body)?),
            ("POST", POINTS_DELETE_PATH) => Op::DeletePoints(select_of(&body)?),
            (m, p) => return Err((404, format!("the fake does not serve {m} {p}"))),
        };
        if !request
            .url
            .query_pairs()
            .any(|(k, v)| k == "wait" && v == "true")
        {
            return Err(bad(
                "writes must pass wait=true: without it Qdrant acknowledges before applying",
            ));
        }
        self.run_hook();
        self.apply(op)?;
        Ok(json!({"operation_id": 0, "status": "completed"}))
    }

    fn run_hook(&mut self) {
        let Some((left, hook)) = self.hook.as_mut() else {
            return;
        };
        hook(&mut self.points);
        *left -= 1;
        if *left == 0 {
            self.hook = None;
        }
    }

    fn retrieve(&self, body: &Value) -> Result<Value, Failure> {
        let ids = body
            .get("ids")
            .and_then(Value::as_array)
            .ok_or_else(|| bad("retrieve needs `ids`"))?;
        if body.get("with_vector").is_some_and(|v| v != &json!(false)) {
            return Err(bad("the fake keeps no vectors"));
        }
        let keys: Option<Vec<&str>> = match body.get("with_payload") {
            None | Some(Value::Bool(true)) => None,
            Some(Value::Array(keys)) => Some(
                keys.iter()
                    .map(|k| {
                        k.as_str()
                            .ok_or_else(|| bad("with_payload keys must be strings"))
                    })
                    .collect::<Result<_, _>>()?,
            ),
            Some(other) => {
                return Err(bad(format!("the fake does not model with_payload {other}")));
            }
        };
        let mut seen = HashSet::new();
        let mut found = Vec::new();
        for id in ids {
            let id = id_of(id)?;
            let Some(payload) = self.points.get(&id) else {
                continue;
            };
            if !seen.insert(id.clone()) {
                continue;
            }
            let payload = match &keys {
                None => payload.clone(),
                Some(keys) => Value::Object(
                    keys.iter()
                        .filter_map(|k| Some((k.to_string(), payload.get(*k)?.clone())))
                        .collect(),
                ),
            };
            found.push(json!({"id": id, "payload": payload}));
        }
        Ok(Value::Array(found))
    }

    fn apply(&mut self, op: Op) -> Result<(), Failure> {
        match op {
            Op::Upsert {
                points,
                mode,
                filter,
            } => {
                for (id, payload) in points {
                    let write = match self.points.get(&id) {
                        None => mode != Mode::UpdateOnly,
                        Some(_) if mode == Mode::InsertOnly => false,
                        Some(existing) => filter.as_ref().is_none_or(|f| matches(f, &id, existing)),
                    };
                    if write {
                        self.points.insert(id, payload);
                    }
                }
            }
            Op::SetPayload(payload, select) => {
                for id in self.selected(&select, true)? {
                    let point = self.points.get_mut(&id).expect("selected points exist");
                    for (k, v) in &payload {
                        point[k.as_str()] = v.clone();
                    }
                }
            }
            Op::OverwritePayload(payload, select) => {
                for id in self.selected(&select, true)? {
                    self.points.insert(id, Value::Object(payload.clone()));
                }
            }
            Op::DeletePayload(keys, select) => {
                for id in self.selected(&select, true)? {
                    let point = self.points.get_mut(&id).expect("selected points exist");
                    if let Some(obj) = point.as_object_mut() {
                        for k in &keys {
                            obj.remove(k);
                        }
                    }
                }
            }
            Op::DeletePoints(select) => {
                for id in self.selected(&select, false)? {
                    self.points.remove(&id);
                }
            }
        }
        Ok(())
    }

    /// The existing points `select` names. Qdrant fails a payload write that
    /// names an absent id (`must_exist`); deleting one is a no-op.
    fn selected(&self, select: &Select, must_exist: bool) -> Result<Vec<String>, Failure> {
        Ok(match select {
            Select::Ids(ids) => {
                if must_exist && let Some(id) = ids.iter().find(|id| !self.points.contains_key(*id))
                {
                    return Err((404, format!("No point with id {id} found")));
                }
                ids.iter()
                    .filter(|id| self.points.contains_key(*id))
                    .cloned()
                    .collect()
            }
            Select::Filter(f) => self
                .points
                .iter()
                .filter(|(id, payload)| matches(f, id, payload))
                .map(|(id, _)| id.clone())
                .collect(),
        })
    }
}

fn bad(msg: impl Into<String>) -> Failure {
    (400, msg.into())
}

fn id_of(v: &Value) -> Result<String, Failure> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        other => Err(bad(format!("invalid point id {other}"))),
    }
}

fn parse_upsert(body: &Value) -> Result<Op, Failure> {
    let points = body
        .get("points")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("the fake takes the `points` list form of upsert only"))?
        .iter()
        .map(|p| {
            let id = id_of(p.get("id").unwrap_or(&Value::Null))?;
            let payload = match p.get("payload") {
                None | Some(Value::Null) => json!({}),
                Some(v @ Value::Object(_)) => v.clone(),
                Some(other) => return Err(bad(format!("payload must be an object, got {other}"))),
            };
            Ok((id, payload))
        })
        .collect::<Result<_, _>>()?;
    let mode = match body.get("update_mode") {
        None => Mode::Upsert,
        Some(m) if m == "upsert" => Mode::Upsert,
        Some(m) if m == "insert_only" => Mode::InsertOnly,
        Some(m) if m == "update_only" => Mode::UpdateOnly,
        Some(other) => return Err(bad(format!("unknown update_mode {other}"))),
    };
    let filter = match body.get("update_filter") {
        None => None,
        Some(f) => {
            check_filter(f)?;
            Some(f.clone())
        }
    };
    Ok(Op::Upsert {
        points,
        mode,
        filter,
    })
}

fn payload_of(body: &Value) -> Result<Map<String, Value>, Failure> {
    if body.get("key").is_some() {
        // Nested writes are how `metadata.superseded_by` used to land; the
        // client no longer sends them, so the fake does not model them.
        return Err(bad("the fake does not model `key` (nested payload writes)"));
    }
    body.get("payload")
        .and_then(Value::as_object)
        .cloned()
        .ok_or_else(|| bad("a payload write needs a `payload` object"))
}

fn keys_of(body: &Value) -> Result<Vec<String>, Failure> {
    body.get("keys")
        .and_then(Value::as_array)
        .and_then(|keys| keys.iter().map(|k| k.as_str().map(str::to_owned)).collect())
        .ok_or_else(|| bad("delete payload needs a `keys` list of strings"))
}

fn select_of(body: &Value) -> Result<Select, Failure> {
    match (body.get("points"), body.get("filter")) {
        (Some(Value::Array(ids)), None) => Ok(Select::Ids(
            ids.iter().map(id_of).collect::<Result<_, _>>()?,
        )),
        (None, Some(f)) => {
            check_filter(f)?;
            Ok(Select::Filter(f.clone()))
        }
        _ => Err(bad(
            "exactly one of a `points` list and a `filter` selects the points",
        )),
    }
}

const CLAUSES: [&str; 3] = ["must", "should", "must_not"];

/// A clause's conditions: Qdrant takes one condition or a list.
fn conditions(clause: &Value) -> Vec<&Value> {
    match clause {
        Value::Array(list) => list.iter().collect(),
        Value::Object(_) => vec![clause],
        _ => Vec::new(),
    }
}

/// 400 unless `f` is a Filter the fake can evaluate.
fn check_filter(f: &Value) -> Result<(), Failure> {
    let obj = f
        .as_object()
        .ok_or_else(|| bad(format!("a filter must be an object, got {f}")))?;
    for (clause, conds) in obj {
        if !CLAUSES.contains(&clause.as_str()) {
            return Err(bad(format!(
                "unknown filter field {clause:?} in {f}: a condition is not a Filter"
            )));
        }
        if !(conds.is_array() || conds.is_object()) {
            return Err(bad(format!(
                "filter clause {clause:?} must be a condition or a list"
            )));
        }
        for c in conditions(conds) {
            check_condition(c)?;
        }
    }
    Ok(())
}

fn check_condition(c: &Value) -> Result<(), Failure> {
    let Some(obj) = c.as_object() else {
        return Err(bad(format!("a condition must be an object, got {c}")));
    };
    let has = |keys: &[&str]| obj.len() == keys.len() && keys.iter().all(|k| obj.contains_key(*k));
    let ok = if has(&["has_id"]) {
        obj["has_id"].is_array()
    } else if has(&["is_empty"]) || has(&["is_null"]) {
        obj.values()
            .all(|v| v.get("key").is_some_and(Value::is_string))
    } else if has(&["key", "match"]) {
        obj["key"].is_string()
            && obj["match"]
                .as_object()
                .is_some_and(|m| m.len() == 1 && m.contains_key("value"))
    } else {
        return check_filter(c);
    };
    if ok {
        Ok(())
    } else {
        Err(bad(format!("the fake does not model condition {c}")))
    }
}

/// Whether the point `id` holding `payload` matches the (checked) filter `f`.
fn matches(f: &Value, id: &str, payload: &Value) -> bool {
    let clause = |name: &str| f.get(name).map(conditions).unwrap_or_default();
    let should = clause("should");
    clause("must").iter().all(|c| holds(c, id, payload))
        && (should.is_empty() || should.iter().any(|c| holds(c, id, payload)))
        && !clause("must_not").iter().any(|c| holds(c, id, payload))
}

fn holds(c: &Value, id: &str, payload: &Value) -> bool {
    if let Some(ids) = c.get("has_id").and_then(Value::as_array) {
        return ids.iter().any(|v| id_of(v).is_ok_and(|v| v == id));
    }
    if let Some(key) = c.pointer("/is_empty/key").and_then(Value::as_str) {
        return match field(payload, key) {
            None | Some(Value::Null) => true,
            Some(Value::Array(a)) => a.is_empty(),
            Some(_) => false,
        };
    }
    if let Some(key) = c.pointer("/is_null/key").and_then(Value::as_str) {
        return matches!(field(payload, key), Some(Value::Null));
    }
    if let (Some(key), Some(want)) = (
        c.get("key").and_then(Value::as_str),
        c.pointer("/match/value"),
    ) {
        return match field(payload, key) {
            Some(Value::Array(values)) => values.contains(want),
            Some(v) => v == want,
            None => false,
        };
    }
    matches(c, id, payload)
}

/// The payload value at a dotted `key`.
fn field<'a>(payload: &'a Value, key: &str) -> Option<&'a Value> {
    key.split('.').try_fold(payload, |v, k| v.get(k))
}

// ─── Other writers ──────────────────────────────────────────────────────────

static EXTERNAL_WRITES: AtomicU64 = AtomicU64::new(0);

/// Another alaya process's write to `id` landing: merges `set` into the
/// payload (creating the point when absent — another process inserting the
/// same content first) and stamps a fresh revision the way every alaya writer
/// does. Returns the revision token.
pub fn external_write(points: &mut Points, id: &str, set: Value) -> String {
    let rev = format!(
        "external-{}",
        EXTERNAL_WRITES.fetch_add(1, Ordering::Relaxed)
    );
    let point = points.entry(id.to_string()).or_insert_with(|| json!({}));
    for (k, v) in set.as_object().expect("`set` is an object") {
        point[k.as_str()] = v.clone();
    }
    let mut log = point
        .get("rev_log")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    log.push(json!(rev));
    let excess = log.len().saturating_sub(REV_LOG_LEN);
    log.drain(..excess);
    point["rev"] = json!(rev);
    point["rev_log"] = Value::Array(log);
    rev
}

// ─── Request log ────────────────────────────────────────────────────────────

/// Every request except retrieves, in arrival order, as ("METHOD path?query",
/// body). The query is part of the contract: a write without `wait=true`
/// returns on Qdrant's acknowledgement, before it is visible.
pub async fn writes(server: &MockServer) -> Vec<(String, Value)> {
    server
        .received_requests()
        .await
        .expect("request recording is on")
        .into_iter()
        .filter(|r| !(r.method.as_str() == "POST" && r.url.path() == POINTS_PATH))
        .map(|r| {
            let line = match r.url.query() {
                Some(q) => format!("{} {}?{q}", r.method, r.url.path()),
                None => format!("{} {}", r.method, r.url.path()),
            };
            (line, serde_json::from_slice(&r.body).unwrap_or(Value::Null))
        })
        .collect()
}

/// `writes` without the bodies.
pub async fn write_order(server: &MockServer) -> Vec<String> {
    writes(server)
        .await
        .into_iter()
        .map(|(line, _)| line)
        .collect()
}

/// Bodies of the retrieves, in arrival order.
pub async fn retrieves(server: &MockServer) -> Vec<Value> {
    server
        .received_requests()
        .await
        .expect("request recording is on")
        .into_iter()
        .filter(|r| r.method.as_str() == "POST" && r.url.path() == POINTS_PATH)
        .map(|r| serde_json::from_slice(&r.body).unwrap_or(Value::Null))
        .collect()
}

/// The filter a write conditional on `rev` (none yet: `None`) carries on its
/// own: `{"must": [<rev condition>]}`.
pub fn rev_filter(rev: Option<&str>) -> Value {
    json!({ "must": [rev_cond(rev)] })
}

/// The condition matching a point whose revision is `rev` (none yet: `None`).
pub fn rev_cond(rev: Option<&str>) -> Value {
    match rev {
        Some(r) => json!({"key": "rev", "match": {"value": r}}),
        None => json!({"is_empty": {"key": "rev"}}),
    }
}

fn must_of(filter: &Value) -> Vec<&Value> {
    filter.get("must").map(conditions).unwrap_or_default()
}

fn is_rev_cond(c: &Value) -> bool {
    c.pointer("/match/value").is_some_and(Value::is_string) && c["key"] == "rev"
        || c == &rev_cond(None)
}

/// Whether a write (from `writes`) can only insert, or only apply while the
/// point still holds the revision it read: an upsert that is `insert_only`, or
/// `update_only` with a revision condition in `update_filter`; a payload write
/// whose filter pins one id AND a revision.
pub fn is_conditional(line: &str, body: &Value) -> bool {
    if line.starts_with(&format!("PUT {POINTS_PATH}?")) {
        return match body["update_mode"].as_str() {
            Some("insert_only") => true,
            Some("update_only") => must_of(&body["update_filter"]).into_iter().any(is_rev_cond),
            _ => false,
        };
    }
    let payload_write = [
        format!("POST {PAYLOAD_PATH}?"),
        format!("PUT {PAYLOAD_PATH}?"),
    ];
    if payload_write.iter().any(|p| line.starts_with(p.as_str())) {
        let must = must_of(&body["filter"]);
        let one_id = must.iter().any(|c| {
            c.get("has_id")
                .and_then(Value::as_array)
                .is_some_and(|ids| ids.len() == 1)
        });
        return body.get("points").is_none() && one_id && must.into_iter().any(is_rev_cond);
    }
    false
}
