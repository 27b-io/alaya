//! `update_metadata{,_batch}` against a stateful fake Qdrant (`common`).
//!
//! The supersession marker (`metadata.superseded_by`) is the COMMIT POINT of a
//! metadata update: searches hide a memory as soon as it lands, so it must
//! never land without its reason (#55, PR #56). It used to be the second of
//! two writes, top-level fields first. Now each memory gets ONE merge carrying
//! the top-level fields and the whole `metadata` object with the marker set,
//! conditional on the revision the metadata was read at (alaya#130): the pair
//! lands together or not at all, and sibling metadata keys written by anyone
//! in between are never dropped by the rewrite.
//!
//! `reverse_supersession` is the reverse (LAB-6876): marker and reason leave in
//! one conditional overwrite that also appends the audit entry, and only while
//! the marker still names the survivor the caller read.

mod common;

use std::collections::HashMap;

use alaya_backends::{ReversalOutcome, ReversalRecord, VectorStorage, qdrant::QdrantClient};
use alaya_types::{
    AlayaError,
    memory::{Memory, MetadataUpdate},
};
use common::{FakeQdrant, PAYLOAD_PATH, external_write, is_conditional, retrieves, writes};
use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

fn client_for(server: &MockServer) -> QdrantClient {
    QdrantClient::new(server.uri(), "memories".into(), None).unwrap()
}

fn hash(c: char) -> String {
    c.to_string().repeat(64)
}

/// The point id `QdrantClient` derives from `hash(c)`.
fn id(c: char) -> String {
    let h = c.to_string().repeat(32);
    format!(
        "{}-{}-{}-{}-{}",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

fn memory_payload(c: char, rev: Option<&str>) -> Value {
    let mut payload = json!({
        "content": format!("memory {c}"),
        "content_hash": hash(c),
        "tags": [],
        "memory_type": "note",
        "created_at": 1000.0,
        "updated_at": 1000.0,
        "access_count": 2,
        "metadata": {"source": "import", "provenance": {"trust": 0.9}},
    });
    if let Some(r) = rev {
        payload["rev"] = json!(r);
        payload["rev_log"] = json!([r]);
    }
    payload
}

fn supersede(access_count: Option<u64>) -> MetadataUpdate {
    let mut extra = HashMap::new();
    extra.insert("supersession_reason".to_string(), json!("merged"));
    MetadataUpdate {
        superseded_by: Some(hash('b')),
        access_count,
        extra: Some(extra),
    }
}

async fn fake_with(points: &[(char, Option<&str>)]) -> (MockServer, FakeQdrant) {
    let server = MockServer::start().await;
    let fake = FakeQdrant::default();
    for (c, rev) in points {
        fake.insert(&id(*c), memory_payload(*c, *rev));
    }
    fake.mount(&server).await;
    (server, fake)
}

#[tokio::test]
async fn supersession_lands_with_its_reason_in_one_conditional_write() {
    let (server, fake) = fake_with(&[('a', Some("r1"))]).await;

    client_for(&server)
        .update_metadata(&hash('a'), supersede(Some(7)))
        .await
        .expect("update succeeds");

    let w = writes(&server).await;
    assert_eq!(w.len(), 1, "one write carries marker and reason: {w:?}");
    let (line, body) = &w[0];
    assert_eq!(line, &format!("POST {PAYLOAD_PATH}?wait=true"), "a merge");
    assert_eq!(
        body["filter"],
        json!({"must": [{"has_id": [id('a')]}, {"key": "rev", "match": {"value": "r1"}}]}),
        "conditional on the point AND the revision the metadata was read at"
    );
    assert!(
        body.get("points").is_none() && body.get("key").is_none(),
        "{body}"
    );
    let payload = &body["payload"];
    assert_eq!(payload["access_count"], json!(7));
    assert_eq!(payload["supersession_reason"], json!("merged"));
    assert_eq!(
        payload["metadata"],
        json!({"source": "import", "provenance": {"trust": 0.9}, "superseded_by": hash('b')}),
        "the marker lives INSIDE the metadata object, siblings kept"
    );
    assert!(
        payload.get("metadata.superseded_by").is_none(),
        "a dotted key would create a flat literal field (#54): {payload}"
    );

    let stored = fake.point(&id('a')).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!(hash('b')));
    assert_eq!(stored["metadata"]["source"], json!("import"));
    assert_eq!(stored["supersession_reason"], json!("merged"));
    assert_eq!(stored["access_count"], json!(7));
    assert_eq!(
        stored["content"],
        json!("memory a"),
        "a merge keeps the rest"
    );
    assert_eq!(stored["rev_log"][0], json!("r1"));
    assert_eq!(stored["rev_log"][1], stored["rev"]);
}

/// Atomicity replaces the old aux-first ordering: when the one write fails,
/// neither the marker nor its reason lands, and nothing is retried blindly.
#[tokio::test]
async fn failed_write_lands_neither_marker_nor_reason() {
    let (server, fake) = fake_with(&[('a', None)]).await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_PATH))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
        )
        .with_priority(1)
        .mount(&server)
        .await;

    let result = client_for(&server)
        .update_metadata(&hash('a'), supersede(Some(7)))
        .await;
    assert!(matches!(result, Err(AlayaError::Storage(_))), "{result:?}");

    assert_eq!(
        writes(&server).await.len(),
        1,
        "a failed write is not retried"
    );
    assert_eq!(fake.point(&id('a')).unwrap(), memory_payload('a', None));
}

#[tokio::test]
async fn supersession_only_update_writes_only_the_marker() {
    let (server, fake) = fake_with(&[('a', None)]).await;

    let updates = MetadataUpdate {
        superseded_by: Some(hash('b')),
        ..Default::default()
    };
    client_for(&server)
        .update_metadata(&hash('a'), updates)
        .await
        .expect("update succeeds");

    let w = writes(&server).await;
    assert_eq!(w.len(), 1, "{w:?}");
    let mut keys: Vec<&String> = w[0].1["payload"].as_object().unwrap().keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        ["metadata", "rev", "rev_log"],
        "no empty top-level fields"
    );
    assert_eq!(
        w[0].1["filter"]["must"][1],
        json!({"is_empty": {"key": "rev"}}),
        "a point with no revision yet is the zero revision"
    );
    let stored = fake.point(&id('a')).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!(hash('b')));
    assert_eq!(stored["metadata"]["provenance"], json!({"trust": 0.9}));
    assert_eq!(stored["access_count"], json!(2));
}

/// Batch supersede (alaya#6): the reads stay batched — one retrieve before
/// and one after, whatever the batch size — while each point gets its own
/// write, because each is conditional on its own revision.
#[tokio::test]
async fn batch_update_reads_in_bulk_and_writes_each_point_conditionally() {
    let (server, fake) = fake_with(&[('1', Some("r1")), ('2', None), ('3', Some("r3"))]).await;

    client_for(&server)
        .update_metadata_batch(&[&hash('1'), &hash('2'), &hash('3')], supersede(None))
        .await
        .expect("batch update succeeds");

    let reads = retrieves(&server).await;
    assert_eq!(
        reads.len(),
        2,
        "one read and one read-back for 3 points: {reads:?}"
    );
    assert_eq!(reads[0]["ids"].as_array().unwrap().len(), 3);

    let w = writes(&server).await;
    assert_eq!(w.len(), 3, "{w:?}");
    for (line, body) in &w {
        assert!(is_conditional(line, body), "{line} {body}");
    }
    for (c, rev) in [('1', Some("r1")), ('2', None), ('3', Some("r3"))] {
        let write = w
            .iter()
            .find(|(_, b)| b["filter"]["must"][0]["has_id"] == json!([id(c)]))
            .unwrap_or_else(|| panic!("no write for {c}"));
        assert_eq!(write.1["filter"]["must"][1], common::rev_cond(rev));
        let stored = fake.point(&id(c)).unwrap();
        assert_eq!(stored["metadata"]["superseded_by"], json!(hash('b')));
        assert_eq!(stored["metadata"]["source"], json!("import"));
        assert_eq!(stored["supersession_reason"], json!("merged"));
    }
}

#[tokio::test]
async fn batch_update_empty_hashes_sends_nothing() {
    let (server, _fake) = fake_with(&[]).await;

    client_for(&server)
        .update_metadata_batch(&[], supersede(Some(7)))
        .await
        .expect("empty batch is a no-op");

    let requests = server.received_requests().await.unwrap();
    assert!(requests.is_empty(), "{requests:?}");
}

#[tokio::test]
async fn update_of_an_absent_memory_is_not_found_and_writes_nothing() {
    let (server, fake) = fake_with(&[]).await;

    let result = client_for(&server)
        .update_metadata(&hash('a'), supersede(Some(7)))
        .await;
    assert!(matches!(result, Err(AlayaError::NotFound(_))), "{result:?}");
    assert!(writes(&server).await.is_empty());
    assert!(
        fake.point(&id('a')).is_none(),
        "an update never creates a point"
    );
}

/// Stored metadata that is not an object cannot take a marker without being
/// replaced; the update refuses rather than overwrite it.
#[tokio::test]
async fn non_object_metadata_is_refused_not_replaced() {
    let (server, fake) = fake_with(&[]).await;
    let mut payload = memory_payload('a', None);
    payload["metadata"] = json!("legacy string");
    fake.insert(&id('a'), payload.clone());

    let result = client_for(&server)
        .update_metadata(&hash('a'), supersede(None))
        .await;
    assert!(matches!(result, Err(AlayaError::Storage(_))), "{result:?}");
    assert!(writes(&server).await.is_empty());
    assert_eq!(fake.point(&id('a')).unwrap(), payload);
}

/// Another process writes the memory — a sibling metadata key and an access
/// count — between this update's read and its write. The write is rejected
/// (its revision moved), re-read and re-sent, so the marker lands WITHOUT
/// rolling back either of the other writer's fields.
#[tokio::test]
async fn lost_race_is_retried_and_keeps_the_other_writers_fields() {
    let (server, fake) = fake_with(&[('a', Some("r1"))]).await;
    let a = id('a');
    fake.before_writes(1, move |points| {
        external_write(
            points,
            &a,
            json!({
                "access_count": 11,
                "metadata": {"source": "import", "provenance": {"trust": 0.9}, "note": "pinned"},
            }),
        );
    });

    client_for(&server)
        .update_metadata(&hash('a'), supersede(None))
        .await
        .expect("the update lands on its second round");

    let stored = fake.point(&id('a')).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!(hash('b')));
    assert_eq!(
        stored["metadata"]["note"],
        json!("pinned"),
        "sibling key kept"
    );
    assert_eq!(
        stored["access_count"],
        json!(11),
        "other writer's count kept"
    );
    assert_eq!(stored["supersession_reason"], json!("merged"));

    let log = stored["rev_log"].as_array().unwrap();
    assert_eq!(log.len(), 3, "r1, the other writer, this update: {log:?}");
    let w = writes(&server).await;
    assert_eq!(w.len(), 2, "{w:?}");
    assert_eq!(w[0].1["filter"]["must"][1], common::rev_cond(Some("r1")));
    assert_eq!(
        w[1].1["filter"]["must"][1],
        common::rev_cond(log[1].as_str()),
        "the retry is conditional on the revision it re-read"
    );
}

// ─── Batch semantics ────────────────────────────────────────────────────────

/// Matches a conditional payload write aimed at one point.
struct WriteTo(String);

impl wiremock::Match for WriteTo {
    fn matches(&self, request: &wiremock::Request) -> bool {
        let Ok(body) = serde_json::from_slice::<Value>(&request.body) else {
            return false;
        };
        body.pointer("/filter/must/0/has_id/0") == Some(&json!(self.0))
    }
}

/// A supersession batch is checked whole before anything is written: one
/// absent memory is `NotFound` and the present ones are left unmarked, so the
/// caller never ends up with a marked memory it will not write a graph edge for.
#[tokio::test]
async fn batch_update_with_an_absent_memory_writes_nothing() {
    let (server, fake) = fake_with(&[('a', Some("r1")), ('c', None)]).await;

    let result = client_for(&server)
        .update_metadata_batch(
            &[hash('a').as_str(), hash('d').as_str(), hash('c').as_str()],
            supersede(None),
        )
        .await;
    assert!(
        matches!(result, Err(AlayaError::NotFound(ref m)) if m.contains(&hash('d'))),
        "{result:?}"
    );
    assert!(writes(&server).await.is_empty(), "nothing may be marked");
    for c in ['a', 'c'] {
        assert!(
            fake.point(&id(c)).unwrap()["metadata"]
                .get("superseded_by")
                .is_none()
        );
    }
}

/// Access counts are fire-and-forget: one point's failed write is that
/// point's loss alone, and the rest of the batch is still counted.
#[tokio::test]
async fn batch_access_increment_survives_one_failed_write() {
    let (server, fake) = fake_with(&[('a', Some("r1")), ('b', None), ('c', Some("r3"))]).await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_PATH))
        .and(WriteTo(id('b')))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
        )
        .with_priority(1)
        .mount(&server)
        .await;

    client_for(&server)
        .increment_access_count_batch(&[hash('a').as_str(), hash('b').as_str(), hash('c').as_str()])
        .await
        .expect("fire-and-forget: a failed point is logged, not returned");
    assert_eq!(fake.point(&id('a')).unwrap()["access_count"], json!(3));
    assert_eq!(
        fake.point(&id('b')).unwrap()["access_count"],
        json!(2),
        "its write failed"
    );
    assert_eq!(fake.point(&id('c')).unwrap()["access_count"], json!(3));
}

/// Writes are atomic per memory, not per batch: one point's failed write does
/// not stop the rest, and the call reports the failure. The caller re-reads
/// to learn which landed (alaya-core's `mark_superseded` does).
#[tokio::test]
async fn strict_batch_keeps_going_past_one_failed_write() {
    let (server, fake) = fake_with(&[('a', Some("r1")), ('b', None), ('c', Some("r3"))]).await;
    Mock::given(method("POST"))
        .and(path(PAYLOAD_PATH))
        .and(WriteTo(id('b')))
        .respond_with(
            ResponseTemplate::new(500).set_body_json(json!({"status": {"error": "boom"}})),
        )
        .with_priority(1)
        .mount(&server)
        .await;

    let result = client_for(&server)
        .update_metadata_batch(
            &[hash('a').as_str(), hash('b').as_str(), hash('c').as_str()],
            supersede(None),
        )
        .await;
    assert!(matches!(result, Err(AlayaError::Storage(_))), "{result:?}");
    let marker = |c| {
        fake.point(&id(c)).unwrap()["metadata"]
            .get("superseded_by")
            .cloned()
    };
    assert_eq!(marker('a'), Some(json!(hash('b'))));
    assert_eq!(marker('b'), None, "its write failed");
    assert_eq!(marker('c'), Some(json!(hash('b'))));
}

/// Every write of a round is built before any is sent: a memory that cannot
/// take the marker fails the call before a sibling is marked.
#[tokio::test]
async fn batch_update_that_cannot_build_one_write_sends_none() {
    let (server, fake) = fake_with(&[('a', Some("r1"))]).await;
    let mut payload = memory_payload('c', None);
    payload["metadata"] = json!("legacy string");
    fake.insert(&id('c'), payload);

    let result = client_for(&server)
        .update_metadata_batch(&[hash('a').as_str(), hash('c').as_str()], supersede(None))
        .await;
    assert!(matches!(result, Err(AlayaError::Storage(_))), "{result:?}");
    assert!(writes(&server).await.is_empty(), "nothing may be marked");
}

// ─── Reversal: reverse_supersession (LAB-6876) ───────────────────────────────

fn reversal() -> ReversalRecord {
    ReversalRecord {
        at: 2000.0,
        via: "operator:test".into(),
        reason: "wrong merge".into(),
    }
}

/// `a` superseded by `by`, with a reason, at revision `rev`.
fn superseded_payload(c: char, by: char, rev: Option<&str>) -> Value {
    let mut payload = memory_payload(c, rev);
    payload["metadata"]["superseded_by"] = json!(hash(by));
    payload["supersession_reason"] = json!("merged");
    payload
}

/// The reverse of a supersession is ONE whole-payload overwrite, conditional
/// on the revision read: marker and reason leave together, the audit entry
/// arrives in the same write, and every other field is the copy just read.
#[tokio::test]
async fn unsupersede_clears_marker_and_reason_and_logs_them_in_one_conditional_write() {
    let (server, fake) = fake_with(&[]).await;
    fake.insert(&id('a'), superseded_payload('a', 'b', Some("r1")));

    let outcome = client_for(&server)
        .reverse_supersession(&hash('a'), &json!(hash('b')), &reversal())
        .await
        .expect("clear succeeds");
    assert_eq!(
        outcome,
        ReversalOutcome::Cleared {
            supersession_reason: Some(json!("merged"))
        }
    );

    let w = writes(&server).await;
    assert_eq!(w.len(), 1, "one write: {w:?}");
    let (line, body) = &w[0];
    assert_eq!(
        line,
        &format!("PUT {PAYLOAD_PATH}?wait=true"),
        "an overwrite: a merge cannot remove a key"
    );
    assert!(is_conditional(line, body), "{line} {body}");
    assert_eq!(body["filter"]["must"][1], common::rev_cond(Some("r1")));

    let stored = fake.point(&id('a')).unwrap();
    assert_eq!(
        stored["metadata"],
        json!({"source": "import", "provenance": {"trust": 0.9}}),
        "marker gone, siblings kept"
    );
    assert!(stored.get("supersession_reason").is_none(), "{stored}");
    assert_eq!(
        stored["supersession_log"],
        json!([{
            "superseded_by": hash('b'),
            "supersession_reason": "merged",
            "unsuperseded_at": 2000.0,
            "unsuperseded_via": "operator:test",
            "reason": "wrong merge",
        }])
    );
    assert_eq!(stored["content"], json!("memory a"));
    assert_eq!(stored["access_count"], json!(2));
    assert_eq!(stored["rev_log"][0], json!("r1"));
    assert_eq!(stored["rev_log"][1], stored["rev"]);
}

/// A second reversal appends: the audit trail only grows.
#[tokio::test]
async fn unsupersede_appends_to_an_existing_log() {
    let (server, fake) = fake_with(&[]).await;
    let mut payload = superseded_payload('a', 'c', None);
    payload["supersession_log"] = json!([{"superseded_by": hash('b'), "reason": "first"}]);
    fake.insert(&id('a'), payload);

    client_for(&server)
        .reverse_supersession(&hash('a'), &json!(hash('c')), &reversal())
        .await
        .expect("clear succeeds");

    let log = fake.point(&id('a')).unwrap()["supersession_log"].clone();
    assert_eq!(log.as_array().unwrap().len(), 2, "{log}");
    assert_eq!(
        log[0],
        json!({"superseded_by": hash('b'), "reason": "first"})
    );
    assert_eq!(log[1]["superseded_by"], json!(hash('c')));
}

/// A live memory, or one superseded by something other than what the caller
/// read, is reported as found and never written.
#[tokio::test]
async fn unsupersede_writes_nothing_unless_the_marker_is_the_expected_one() {
    let (server, fake) = fake_with(&[('a', Some("r1"))]).await;
    let superseded = superseded_payload('c', 'd', None);
    fake.insert(&id('c'), superseded.clone());
    let client = client_for(&server);

    let live = client
        .reverse_supersession(&hash('a'), &json!(hash('b')), &reversal())
        .await
        .expect("a live memory is an outcome, not an error");
    assert_eq!(live, ReversalOutcome::NotSuperseded);

    let other = client
        .reverse_supersession(&hash('c'), &json!(hash('b')), &reversal())
        .await
        .expect("another survivor is an outcome, not an error");
    assert_eq!(other, ReversalOutcome::SupersededByOther(json!(hash('d'))));

    assert!(writes(&server).await.is_empty());
    assert_eq!(fake.point(&id('c')).unwrap(), superseded);
}

#[tokio::test]
async fn unsupersede_of_an_absent_memory_is_not_found() {
    let (server, _fake) = fake_with(&[]).await;

    let result = client_for(&server)
        .reverse_supersession(&hash('a'), &json!(hash('b')), &reversal())
        .await;
    assert!(matches!(result, Err(AlayaError::NotFound(_))), "{result:?}");
    assert!(writes(&server).await.is_empty());
}

/// Another process counts an access inside the read→write window: the
/// overwrite built from the stale copy is rejected and rebuilt, so the
/// reversal lands without rolling the count back.
#[tokio::test]
async fn unsupersede_lost_race_is_rebuilt_and_keeps_the_other_writers_fields() {
    let (server, fake) = fake_with(&[]).await;
    fake.insert(&id('a'), superseded_payload('a', 'b', Some("r1")));
    let a = id('a');
    fake.before_writes(1, move |points| {
        external_write(points, &a, json!({"access_count": 11}));
    });

    client_for(&server)
        .reverse_supersession(&hash('a'), &json!(hash('b')), &reversal())
        .await
        .expect("lands on its second round");

    let stored = fake.point(&id('a')).unwrap();
    assert_eq!(
        stored["access_count"],
        json!(11),
        "other writer's count kept"
    );
    assert!(
        stored["metadata"].get("superseded_by").is_none(),
        "{stored}"
    );
    assert_eq!(writes(&server).await.len(), 2);
}

/// A re-supersede to another survivor lands inside the window: the rebuilt
/// write sees the new marker and stands down, leaving that supersession whole.
#[tokio::test]
async fn unsupersede_racing_a_re_supersede_stands_down() {
    let (server, fake) = fake_with(&[]).await;
    fake.insert(&id('a'), superseded_payload('a', 'b', Some("r1")));
    let a = id('a');
    fake.before_writes(1, move |points| {
        let mut metadata = points[&a]["metadata"].clone();
        metadata["superseded_by"] = json!(hash('c'));
        external_write(
            points,
            &a,
            json!({"metadata": metadata, "supersession_reason": "newer"}),
        );
    });

    let outcome = client_for(&server)
        .reverse_supersession(&hash('a'), &json!(hash('b')), &reversal())
        .await
        .expect("an outcome, not an error");
    assert_eq!(
        outcome,
        ReversalOutcome::SupersededByOther(json!(hash('c')))
    );

    let stored = fake.point(&id('a')).unwrap();
    assert_eq!(stored["metadata"]["superseded_by"], json!(hash('c')));
    assert_eq!(stored["supersession_reason"], json!("newer"));
    assert!(stored.get("supersession_log").is_none(), "{stored}");
}

// ─── Reading the log back: get_by_hash (LAB-6929) ────────────────────────────

/// `payload`, stored as memory `c`, read back through `get_by_hash`.
async fn read_back(c: char, payload: Value) -> Memory {
    let (server, fake) = fake_with(&[]).await;
    fake.insert(&id(c), payload);
    client_for(&server)
        .get_by_hash(&hash(c))
        .await
        .expect("read succeeds")
        .expect("memory found")
}

/// The log a reversal writes is the log a read returns: one entry, with the
/// caller's reason and every key the reversal stored. The supersession
/// reason reads back while the memory is superseded, and not after.
#[tokio::test]
async fn reversal_log_is_read_back_by_get_by_hash() {
    let (server, fake) = fake_with(&[('c', None)]).await;
    fake.insert(&id('a'), superseded_payload('a', 'b', Some("r1")));
    let client = client_for(&server);
    let superseded = client.get_by_hash(&hash('a')).await.unwrap().unwrap();
    assert_eq!(superseded.supersession_reason, Some(json!("merged")));

    client
        .reverse_supersession(&hash('a'), &json!(hash('b')), &reversal())
        .await
        .expect("clear succeeds");
    let mem = client.get_by_hash(&hash('a')).await.unwrap().unwrap();
    assert_eq!(
        mem.supersession_reason, None,
        "the reason moved into the log"
    );
    assert_eq!(
        mem.supersession_log,
        Some(vec![json!({
            "superseded_by": hash('b'),
            "supersession_reason": "merged",
            "unsuperseded_at": 2000.0,
            "unsuperseded_via": "operator:test",
            "reason": "wrong merge",
        })])
    );

    let never_reversed = client.get_by_hash(&hash('c')).await.unwrap().unwrap();
    assert_eq!(never_reversed.supersession_log, None, "no log, no field");
}

/// A log in a shape no writer makes still reads back whole and never hides
/// the memory: a value that is not an array is one entry, as the next
/// reversal treats it, and an entry missing keys comes back as stored.
#[tokio::test]
async fn malformed_log_reads_back_and_never_hides_the_memory() {
    let mut payload = memory_payload('a', None);
    payload["supersession_log"] = json!({"reason": "not an array"});
    assert_eq!(
        read_back('a', payload).await.supersession_log,
        Some(vec![json!({"reason": "not an array"})])
    );

    let mut payload = memory_payload('a', None);
    payload["supersession_log"] = json!([{"superseded_by": hash('b'), "reason": "first"}, 7]);
    assert_eq!(
        read_back('a', payload).await.supersession_log,
        Some(vec![
            json!({"superseded_by": hash('b'), "reason": "first"}),
            json!(7),
        ])
    );
}
