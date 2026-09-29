//! Two-process write races against a real Qdrant (alaya#130).
//!
//! Ignored by default: they need `QDRANT_TEST_URL`, a disposable Qdrant >= 1.17
//! (every test makes and drops its own collection). CI runs them against a
//! Qdrant service container; a run without the URL fails rather than passing
//! silently:
//!
//!   docker run -d -p 16417:6333 qdrant/qdrant:v1.17.1
//!   QDRANT_TEST_URL=http://localhost:16417 \
//!     cargo test -p alaya-backends --test qdrant_cas_real -- --include-ignored --test-threads=1
//!
//! `QDRANT_TEST_URL_OLD`, if set to a Qdrant < 1.17, also checks the startup
//! refusal against it.
//!
//! Each race is forced, not hoped for. Writer A (this process) talks to Qdrant
//! through a proxy that holds A's first write request once A has read the
//! point. Writer B runs in a CHILD PROCESS — this test binary re-executed
//! into `child_process_writer` — with its own client, straight to Qdrant, and
//! finishes its write inside A's read→write window. Then A's write is
//! released. Without compare-and-set, A's write would roll B's back.

use std::sync::{Arc, Mutex};

use alaya_backends::{StoreMode, VectorStorage, qdrant::QdrantClient};
use alaya_types::{AlayaError, memory::Memory, memory::MetadataUpdate};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;

const CHILD_ROLE: &str = "ALAYA_CAS_CHILD_ROLE";
const CHILD_RESULT: &str = "CAS_CHILD_RESULT ";

fn qdrant_url() -> String {
    std::env::var("QDRANT_TEST_URL")
        .expect("QDRANT_TEST_URL must point at a disposable Qdrant >= 1.17")
}

fn memory(content: &str, tags: &[&str]) -> Memory {
    Memory {
        content: content.into(),
        content_hash: format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(content)),
        tags: tags.iter().map(|t| t.to_string()).collect(),
        memory_type: "note".into(),
        metadata: None,
        created_at: 1000.0,
        updated_at: 1000.0,
        embedding: Some(vec![0.5, 0.75]),
        summary: None,
        salience_score: 0.5,
        access_count: 0,
        access_timestamps: vec![],
        emotional_valence: None,
        encoding_context: None,
        provenance: None,
        summary_embedding: None,
    }
}

/// A fresh collection, dropped on `cleanup`.
struct Collection {
    url: String,
    name: String,
}

impl Collection {
    async fn new(url: &str, test: &str) -> Self {
        let name = format!("cas_{test}_{}", std::process::id());
        let c = Self {
            url: url.into(),
            name,
        };
        c.cleanup().await;
        QdrantClient::new(c.url.clone(), c.name.clone(), None)
            .unwrap()
            .ensure_collection(2)
            .await
            .expect("create test collection");
        c
    }

    fn client(&self, url: &str) -> QdrantClient {
        QdrantClient::new(url.into(), self.name.clone(), None).unwrap()
    }

    /// The raw payload of `hash`'s point, or `None` when it is gone.
    async fn payload(&self, hash: &str) -> Option<Value> {
        let id = uuid::Uuid::parse_str(&hash[..32]).unwrap().to_string();
        let body: Value = reqwest::Client::new()
            .post(format!("{}/collections/{}/points", self.url, self.name))
            .json(&json!({"ids": [id], "with_payload": true}))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        body["result"]
            .as_array()
            .unwrap()
            .first()
            .map(|p| p["payload"].clone())
    }

    /// Write a point the way a pre-alaya#130 writer did: no `rev` at all.
    async fn put_legacy(&self, m: &Memory) {
        let id = uuid::Uuid::parse_str(&m.content_hash[..32])
            .unwrap()
            .to_string();
        let payload = json!({
            "content": m.content, "content_hash": m.content_hash, "tags": m.tags,
            "memory_type": m.memory_type, "created_at": m.created_at,
            "updated_at": m.updated_at, "salience_score": m.salience_score,
            "access_count": 4, "access_timestamps": [1.0, 2.0, 3.0, 4.0],
        });
        let resp = reqwest::Client::new()
            .put(format!(
                "{}/collections/{}/points?wait=true",
                self.url, self.name
            ))
            .json(&json!({"points": [{"id": id, "vector": [0.5, 0.75], "payload": payload}]}))
            .send()
            .await
            .unwrap();
        assert!(resp.status().is_success());
    }

    async fn cleanup(&self) {
        let _ = reqwest::Client::new()
            .delete(format!("{}/collections/{}", self.url, self.name))
            .send()
            .await;
    }
}

// ─── Gating proxy ──────────────────────────────────────────────────────────

/// Holds the first request whose request line starts with `prefix` until the
/// test releases it; forwards everything else untouched.
struct Gate {
    prefix: String,
    held: Mutex<Option<oneshot::Sender<()>>>,
    release: Mutex<Option<oneshot::Receiver<()>>>,
    /// Request lines seen, in arrival order.
    seen: Mutex<Vec<String>>,
}

struct Proxy {
    url: String,
    gate: Arc<Gate>,
    held: oneshot::Receiver<()>,
    release: oneshot::Sender<()>,
}

impl Proxy {
    async fn start(upstream: &str, prefix: String) -> Self {
        let upstream = upstream
            .trim_start_matches("http://")
            .trim_end_matches('/')
            .to_string();
        let (held_tx, held_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let gate = Arc::new(Gate {
            prefix,
            held: Mutex::new(Some(held_tx)),
            release: Mutex::new(Some(release_rx)),
            seen: Mutex::default(),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let g = gate.clone();
        tokio::spawn(async move {
            while let Ok((client, _)) = listener.accept().await {
                let server = TcpStream::connect(&upstream).await.unwrap();
                tokio::spawn(pump(client, server, g.clone()));
            }
        });
        Self {
            url,
            gate,
            held: held_rx,
            release: release_tx,
        }
    }

    fn seen(&self, prefix: &str) -> usize {
        let seen = self.gate.seen.lock().unwrap();
        seen.iter().filter(|l| l.starts_with(prefix)).count()
    }
}

/// Forward one connection. Requests are parsed just far enough (request line,
/// Content-Length) to be held whole; responses stream back untouched.
async fn pump(client: TcpStream, server: TcpStream, gate: Arc<Gate>) {
    let (mut c_read, mut c_write) = client.into_split();
    let (mut s_read, mut s_write) = server.into_split();
    tokio::spawn(async move {
        let _ = tokio::io::copy(&mut s_read, &mut c_write).await;
    });
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let Some(head_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            match c_read.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
            continue;
        };
        let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
        let len: usize = head
            .lines()
            .find_map(|l| {
                let (k, v) = l.split_once(':')?;
                k.eq_ignore_ascii_case("content-length")
                    .then(|| v.trim().parse().ok())?
            })
            .unwrap_or(0);
        let total = head_end + 4 + len;
        while buf.len() < total {
            match c_read.read(&mut chunk).await {
                Ok(0) | Err(_) => return,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
        let request: Vec<u8> = buf.drain(..total).collect();
        let line = head.lines().next().unwrap_or_default().to_string();
        gate.seen.lock().unwrap().push(line.clone());
        if line.starts_with(&gate.prefix) {
            let held = gate.held.lock().unwrap().take();
            if let Some(tx) = held {
                let _ = tx.send(());
                let rx = gate.release.lock().unwrap().take().unwrap();
                let _ = rx.await;
            }
        }
        if s_write.write_all(&request).await.is_err() {
            return;
        }
    }
}

// ─── Child process ─────────────────────────────────────────────────────────

/// Writer B. Does nothing unless re-executed by `run_child` with
/// `ALAYA_CAS_CHILD_ROLE` set; then performs one write with its own client,
/// straight to Qdrant, and prints its result for the parent.
#[tokio::test]
#[ignore = "needs QDRANT_TEST_URL"]
async fn child_process_writer() {
    let Ok(role) = std::env::var(CHILD_ROLE) else {
        return;
    };
    let env = |k: &str| std::env::var(k).unwrap();
    let client =
        QdrantClient::new(env("ALAYA_CAS_URL"), env("ALAYA_CAS_COLLECTION"), None).unwrap();
    let hash = env("ALAYA_CAS_HASH");
    let result = match role.as_str() {
        "supersede" => {
            let mut extra = std::collections::HashMap::new();
            extra.insert("supersession_reason".to_string(), json!("merged"));
            client
                .update_metadata(
                    &hash,
                    MetadataUpdate {
                        superseded_by: Some("b".repeat(64)),
                        extra: Some(extra),
                        ..Default::default()
                    },
                )
                .await
                .map(|()| json!("ok"))
        }
        "increment" => {
            let mut r = Ok(json!("ok"));
            for _ in 0..3 {
                r = r.and(
                    client
                        .increment_access_count(&hash)
                        .await
                        .map(|()| json!("ok")),
                );
            }
            r
        }
        "delete" => client.delete(&hash).await.map(|d| json!(d)),
        "store" => client
            .store(
                &memory(&env("ALAYA_CAS_CONTENT"), &["from-b"]),
                StoreMode::Upsert,
            )
            .await
            .map(|(created, _)| json!({ "created": created })),
        other => panic!("unknown child role {other}"),
    };
    let result = result.unwrap_or_else(|e| panic!("child {role} failed: {e}"));
    println!("{CHILD_RESULT}{result}");
}

/// Run writer B to completion in a child process and return its result.
async fn run_child(role: &str, url: &str, collection: &str, hash: &str, content: &str) -> Value {
    let out = tokio::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "child_process_writer",
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ROLE, role)
        .env("ALAYA_CAS_URL", url)
        .env("ALAYA_CAS_COLLECTION", collection)
        .env("ALAYA_CAS_HASH", hash)
        .env("ALAYA_CAS_CONTENT", content)
        .output()
        .await
        .expect("spawn child writer");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "child {role} failed: {stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let line = stdout
        .lines()
        .find_map(|l| l.split_once(CHILD_RESULT).map(|(_, r)| r))
        .unwrap_or_else(|| panic!("child {role} printed no result: {stdout}"));
    serde_json::from_str(line).unwrap()
}

/// Start writer A's `op` through a proxy that holds A's first PUT to the
/// points endpoint (the store upsert), run writer B in a child process while
/// A is held, release A, and return A's result and how many upserts A sent.
async fn race<T>(
    coll: &Collection,
    url: &str,
    child: (&str, &str, &str),
    op: impl AsyncFnOnce(QdrantClient) -> T,
) -> (T, Value, usize) {
    let prefix = format!("PUT /collections/{}/points?", coll.name);
    let mut proxy = Proxy::start(url, prefix.clone()).await;
    let a = coll.client(&proxy.url);
    let (role, hash, content) = child;

    let child_run = async {
        (&mut proxy.held).await.expect("writer A reached its write");
        let out = run_child(role, url, &coll.name, hash, content).await;
        let release = std::mem::replace(&mut proxy.release, oneshot::channel().0);
        release.send(()).unwrap();
        out
    };
    let (a_result, b_result) = tokio::join!(op(a), child_run);
    let upserts = proxy.seen(&prefix);
    (a_result, b_result, upserts)
}

// ─── Races ─────────────────────────────────────────────────────────────────

/// Re-store vs supersede, on a legacy point with no `rev` (the zero
/// revision): B's marker and reason survive A's re-store, and A's own
/// caller-owned fields still land.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn restore_racing_a_supersede_keeps_the_marker() {
    let url = qdrant_url();
    let coll = Collection::new(&url, "supersede").await;
    let seed = memory("supersede probe", &["seed"]);
    coll.put_legacy(&seed).await;

    let hash = seed.content_hash.clone();
    let (a, b, upserts) = race(&coll, &url, ("supersede", &hash, ""), async |a| {
        a.store(&memory("supersede probe", &["from-a"]), StoreMode::Upsert)
            .await
    })
    .await;

    assert_eq!(b, json!("ok"));
    let (created, _) = a.expect("A's re-store lands after a retry");
    assert!(!created, "a re-store never reports created");
    assert_eq!(upserts, 2, "A's first write lost the race and was retried");
    let p = coll.payload(&hash).await.expect("point still exists");
    assert_eq!(p["metadata"]["superseded_by"], json!("b".repeat(64)), "{p}");
    assert_eq!(p["supersession_reason"], json!("merged"), "{p}");
    assert_eq!(p["tags"], json!(["from-a"]), "A's re-store applied: {p}");
    assert_eq!(p["access_count"], json!(4), "legacy history carried: {p}");
    coll.cleanup().await;
}

/// Re-store vs access increments: three accesses counted by another process
/// inside A's window are all kept.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn restore_racing_access_increments_keeps_every_count() {
    let url = qdrant_url();
    let coll = Collection::new(&url, "increment").await;
    let seed = memory("increment probe", &["seed"]);
    let (created, hash) = coll
        .client(&url)
        .store(&seed, StoreMode::Upsert)
        .await
        .unwrap();
    assert!(created);

    let (a, b, upserts) = race(&coll, &url, ("increment", &hash, ""), async |a| {
        a.store(&memory("increment probe", &["from-a"]), StoreMode::Upsert)
            .await
    })
    .await;

    assert_eq!(b, json!("ok"));
    a.expect("A's re-store lands after a retry");
    assert_eq!(upserts, 2, "A's first write lost the race and was retried");
    let p = coll.payload(&hash).await.expect("point still exists");
    assert_eq!(p["access_count"], json!(3), "{p}");
    assert_eq!(p["access_timestamps"].as_array().unwrap().len(), 3, "{p}");
    assert_eq!(p["tags"], json!(["from-a"]), "{p}");
    coll.cleanup().await;
}

/// Re-store vs delete: the delete stands. A's re-store must not bring the
/// point back from its pre-delete copy, and says the memory is gone.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn restore_racing_a_delete_does_not_resurrect() {
    let url = qdrant_url();
    let coll = Collection::new(&url, "delete").await;
    let seed = memory("delete probe", &["seed"]);
    let (_, hash) = coll
        .client(&url)
        .store(&seed, StoreMode::Upsert)
        .await
        .unwrap();

    let (a, b, _) = race(&coll, &url, ("delete", &hash, ""), async |a| {
        a.store(&memory("delete probe", &["from-a"]), StoreMode::Upsert)
            .await
    })
    .await;

    assert_eq!(b, json!(true));
    let err = a.expect_err("A must not report a store that the delete removed");
    assert!(matches!(err, AlayaError::Conflict(_)), "got {err:?}");
    assert!(
        coll.payload(&hash).await.is_none(),
        "the deleted point stayed deleted"
    );
    coll.cleanup().await;
}

/// Two processes storing the same new content: exactly one reports
/// `created`. A loses the insert race and takes the re-store path.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn lost_insert_race_reports_created_false() {
    let url = qdrant_url();
    let coll = Collection::new(&url, "insert").await;
    let content = "insert probe";
    let hash = memory(content, &[]).content_hash;

    let (a, b, upserts) = race(&coll, &url, ("store", &hash, content), async |a| {
        a.store(&memory(content, &["from-a"]), StoreMode::Upsert)
            .await
    })
    .await;

    assert_eq!(b, json!({ "created": true }), "B inserted first");
    let (created, _) = a.expect("A re-stores over B's point");
    assert!(
        !created,
        "the loser of an insert race must not report created"
    );
    assert_eq!(upserts, 2);
    let p = coll.payload(&hash).await.unwrap();
    assert_eq!(p["tags"], json!(["from-a"]), "{p}");
    coll.cleanup().await;
}

/// Two processes storing the same new content, one of them insert-only:
/// the insert-only store that loses writes nothing and reports
/// `created: false`.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn insert_only_store_that_loses_the_insert_race_writes_nothing() {
    let url = qdrant_url();
    let coll = Collection::new(&url, "insertonly").await;
    let content = "insert-only probe";
    let hash = memory(content, &[]).content_hash;

    let (a, b, upserts) = race(&coll, &url, ("store", &hash, content), async |a| {
        a.store(&memory(content, &["from-a"]), StoreMode::InsertOnly)
            .await
    })
    .await;

    assert_eq!(b, json!({ "created": true }));
    let (created, _) = a.expect("an existing record is an answer, not an error");
    assert!(!created);
    assert_eq!(upserts, 1, "no second write after the lost insert");
    let p = coll.payload(&hash).await.unwrap();
    assert_eq!(p["tags"], json!(["from-b"]), "{p}");
    coll.cleanup().await;
}

/// The startup gate reads the real server's version.
#[tokio::test(flavor = "current_thread")]
#[ignore = "needs QDRANT_TEST_URL"]
async fn server_version_gate_against_real_servers() {
    let url = qdrant_url();
    let version = QdrantClient::new(url, "unused".into(), None)
        .unwrap()
        .check_server_version()
        .await
        .expect("QDRANT_TEST_URL must be Qdrant >= 1.17");
    assert!(!version.is_empty());

    let Ok(old) = std::env::var("QDRANT_TEST_URL_OLD") else {
        return;
    };
    let err = QdrantClient::new(old, "unused".into(), None)
        .unwrap()
        .check_server_version()
        .await
        .expect_err("a Qdrant < 1.17 must be refused");
    assert!(matches!(err, AlayaError::Config(_)), "got {err:?}");
}
