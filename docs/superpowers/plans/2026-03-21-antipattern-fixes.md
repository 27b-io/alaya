# Ālaya Antipattern Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 critical, 7 major, and 5 minor antipatterns identified by the Winston audit before the Workers migration refactor.

**Architecture:** Surgical fixes to existing files. No new crates, no structural changes. Each task is independently committable and testable. Ordered by dependency: C3 (panic fix) → C2 (panic fix) → M4 (bug fix) → M1 (DRY) → M7/m7 (ergonomics) → M2 (observability) → M6 (performance) → m2/m5 (cleanup).

**Tech Stack:** Rust, serde_json, tracing. No new dependencies.

**Quality gates:** Every commit step implies running the full gate per CLAUDE.md: `cargo fmt --all -- --check && cargo clippy --workspace -- -D warnings && cargo test --workspace` plus WASM gate for types/backends/core.

**Note:** C1 (typed response structs) is deferred — it's a larger refactor that should be done alongside the HttpClient abstraction in the Workers spec. The remaining fixes are all independent and safe.

---

### Task 1: Fix `hash_to_uuid` panic (C3)

**Files:**
- Modify: `crates/alaya-backends/src/qdrant.rs:61-66`
- Modify: `crates/alaya-backends/src/qdrant.rs` (test section)

- [ ] **Step 1: Write failing test for short hash**

```rust
#[test]
fn hash_to_uuid_short_input_returns_error() {
    let result = hash_to_uuid("abc");
    assert!(result.is_err());
}

#[test]
fn hash_to_uuid_non_hex_returns_error() {
    let result = hash_to_uuid(&"zz".repeat(32));
    assert!(result.is_err());
}
```

- [ ] **Step 2: Run tests — confirm they panic (proving the bug)**

Run: `cargo test -p alaya-backends hash_to_uuid`
Expected: PANIC on both new tests

- [ ] **Step 3: Fix hash_to_uuid to return Result**

Replace lines 61-66:
```rust
fn hash_to_uuid(content_hash: &str) -> Result<String> {
    if content_hash.len() < 32 {
        return Err(AlayaError::Validation(format!(
            "content_hash too short: {} chars (need 64)",
            content_hash.len()
        )));
    }
    let hex = &content_hash[..32];
    uuid::Uuid::parse_str(hex)
        .map(|u| u.to_string())
        .map_err(|e| AlayaError::Validation(format!("invalid content_hash hex: {e}")))
}
```

- [ ] **Step 4: Fix all call sites — add `?` propagation**

Every `hash_to_uuid(x)` becomes `hash_to_uuid(x)?`. There are 6 call sites in qdrant.rs (lines ~240, ~273, ~351, ~376, ~766). The methods already return `Result`, so `?` propagates cleanly.

**Special case** — `get_batch` (line ~314) uses `.map()` inside a collect:
```rust
// Before:
let ids: Vec<String> = hashes.iter().map(|h| hash_to_uuid(h)).collect();
// After — collect into Result:
let ids: Vec<String> = hashes
    .iter()
    .map(|h| hash_to_uuid(h))
    .collect::<Result<Vec<String>>>()?;
```

- [ ] **Step 5: Update existing tests that call hash_to_uuid**

The existing tests (`hash_to_uuid_matches_python`, `hash_to_uuid_real_hash`) need to unwrap:
```rust
let uuid = hash_to_uuid(&hash).unwrap();
```

- [ ] **Step 6: Run all tests**

Run: `cargo test --workspace`
Expected: All pass including 2 new tests

- [ ] **Step 7: Commit**

```bash
git add crates/alaya-backends/src/qdrant.rs
git commit -m "fix(backends): hash_to_uuid returns Result instead of panicking on invalid input"
```

---

### Task 2: Fix `truncate` UTF-8 panic (C2)

**Files:**
- Modify: `crates/alaya-core/src/service.rs:1036-1042`

- [ ] **Step 1: Write failing test**

Add to service.rs tests (or a new test module):
```rust
#[test]
fn truncate_handles_multibyte_utf8() {
    // Chinese characters are 3 bytes each
    let chinese = "你好世界测试内容";
    let result = truncate(chinese, 4);
    assert!(result.len() <= 15); // 4 chars * 3 bytes + "..."
    assert!(result.ends_with("..."));
}

#[test]
fn truncate_handles_emoji() {
    let emoji = "🎉🎊🎈🎁🎂";
    let result = truncate(emoji, 3);
    assert!(result.ends_with("..."));
}

#[test]
fn truncate_short_string_unchanged() {
    assert_eq!(truncate("hello", 10), "hello");
}
```

- [ ] **Step 2: Run tests — confirm panic on multibyte**

Run: `cargo test -p alaya-core truncate`
Expected: PANIC on `truncate_handles_multibyte_utf8`

- [ ] **Step 3: Fix truncate to use char boundary**

**Semantic change:** Parameter becomes `max_chars` (character count) not `max_len` (byte count). For ASCII text, no behavior change. For CJK/emoji, 200 chars may be up to 800 bytes — this is the correct behavior (truncate by character, not by byte).

```rust
fn truncate(s: &str, max_chars: usize) -> String {
    let char_count = s.chars().count();
    if char_count <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{truncated}...")
    }
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test --workspace`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add crates/alaya-core/src/service.rs
git commit -m "fix(core): truncate uses char boundaries, not byte boundaries (UTF-8 safe)"
```

---

### Task 3: Fix `count_negation_words` substring matching (M4)

**Files:**
- Modify: `crates/alaya-core/src/interference.rs:221-226`
- Modify: test section of same file

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn negation_count_does_not_match_substrings() {
    // "notation" contains "not" as substring but is not a negation
    let count = count_negation_words("the notation was clear");
    assert_eq!(count, 0, "should not match 'not' inside 'notation'");
}

#[test]
fn negation_count_matches_whole_words() {
    let count = count_negation_words("this is not valid and cannot work");
    assert_eq!(count, 2); // "not" and "cannot"
}
```

- [ ] **Step 2: Run — confirm substring bug**

Run: `cargo test -p alaya-core negation_count`
Expected: First test FAILS (counts 1 instead of 0)

- [ ] **Step 3: Fix to use contains_word**

Replace lines 221-226:
```rust
fn count_negation_words(text: &str) -> usize {
    let lower = text.to_lowercase();
    NEGATION_WORDS
        .iter()
        .filter(|&&word| contains_word(&lower, word))
        .count()
}
```

- [ ] **Step 4: Run all tests**

Run: `cargo test --workspace`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add crates/alaya-core/src/interference.rs
git commit -m "fix(core): count_negation_words uses word boundary matching, not substring"
```

---

### Task 4: DRY the GraphRef boilerplate (M1)

**Files:**
- Create: `crates/alaya-backends/src/graph_ref.rs`
- Modify: `crates/alaya-backends/src/lib.rs`
- Modify: `crates/alaya-server/src/main.rs:271-392`
- Modify: `crates/alaya-core/tests/integration.rs:326-443`

- [ ] **Step 1: Create shared graph_ref module in alaya-backends**

Move the wrapper types and all three trait impls into `crates/alaya-backends/src/graph_ref.rs`. Use macros for all three traits (not just GraphService). Expand the existing `delegate_graph!` macro pattern from `main.rs:275-363`. Add analogous macros for `HebbianService` (1 method: `enqueue_strengthen`) and `ConsolidationService` (4 methods: `decay_all_edges`, `decay_stale_edges`, `prune_weak_edges`, `get_orphan_nodes`):

```rust
//! Rc-based wrappers for sharing a single GraphHttpClient across three trait bounds.

use std::rc::Rc;
use crate::graph::GraphHttpClient;

pub struct GraphRef(pub Rc<GraphHttpClient>);
pub struct HebbianRef(pub Rc<GraphHttpClient>);
pub struct ConsolidationRef(pub Rc<GraphHttpClient>);

macro_rules! delegate_graph_service { ... }
macro_rules! delegate_hebbian_service { ... }
macro_rules! delegate_consolidation_service { ... }

delegate_graph_service!(GraphRef);
delegate_hebbian_service!(HebbianRef);
delegate_consolidation_service!(ConsolidationRef);
```

- [ ] **Step 2: Export from lib.rs**

Add to `crates/alaya-backends/src/lib.rs`:
```rust
pub mod graph_ref;
```

- [ ] **Step 3: Replace boilerplate in main.rs**

Remove lines 271-392 (all wrapper structs + trait impls). Replace with:
```rust
use alaya_backends::graph_ref::{GraphRef, HebbianRef, ConsolidationRef};
```

- [ ] **Step 4: Replace boilerplate in integration.rs**

Remove lines 326-443. Replace with same import.

- [ ] **Step 5: Run all tests**

Run: `cargo test --workspace`
Expected: All pass. Net deletion ~100 lines.

- [ ] **Step 6: Commit**

```bash
git add crates/alaya-backends/src/graph_ref.rs crates/alaya-backends/src/lib.rs crates/alaya-server/src/main.rs crates/alaya-core/tests/integration.rs
git commit -m "refactor(backends): extract GraphRef/HebbianRef/ConsolidationRef to shared module"
```

---

### Task 5: Ergonomics — Config Clone + Cmd named fields (M7 + m7)

**Files:**
- Modify: `crates/alaya-server/src/main.rs`

- [ ] **Step 1: Add `#[derive(Clone)]` to Config**

Line 30: `struct Config {` → `#[derive(Clone)] struct Config {`

Replace lines 195-205 (`let cfg_clone = Config { ... }`) with:
```rust
let cfg_clone = config.clone();
```

- [ ] **Step 2: Convert Cmd to named fields**

Replace lines 68-81:
```rust
pub(crate) enum Cmd {
    Health { reply: oneshot::Sender<Value> },
    Store { params: StoreParams, reply: oneshot::Sender<Value> },
    Search { params: SearchParams, reply: oneshot::Sender<Value> },
    Delete { hash: String, reply: oneshot::Sender<Value> },
    Relation { params: RelationParams, reply: oneshot::Sender<Value> },
    Supersede { old_hash: String, new_hash: String, reason: String, reply: oneshot::Sender<Value> },
    Contradictions { limit: usize, reply: oneshot::Sender<Value> },
    FindDuplicates { threshold: f64, limit: usize, strategy: CanonicalStrategy, reply: oneshot::Sender<Value> },
    MergeDuplicates { canonical: String, duplicates: Vec<String>, reason: String, dry_run: bool, reply: oneshot::Sender<Value> },
}
```

- [ ] **Step 3: Update all Cmd construction sites**

Update handler functions (health, store, search, etc.) and service_worker match arms to use named field syntax.

- [ ] **Step 4: Update mcp.rs dispatch**

The mcp.rs `dispatch_tool` function constructs Cmd variants — update to named fields.

- [ ] **Step 5: Run all tests**

Run: `cargo test --workspace`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add crates/alaya-server/src/main.rs crates/alaya-server/src/mcp.rs
git commit -m "refactor(server): derive Clone on Config, named fields on Cmd enum"
```

---

### Task 6: Log errors before safe_message (M2)

**Files:**
- Modify: `crates/alaya-server/src/main.rs:104-176` (service_worker)

- [ ] **Step 1: Add tracing::error before every safe_message call**

In the `service_worker` match arms, add error logging before the safe response:
```rust
Cmd::Store { params, reply } => {
    let result = match svc.store_memory(params).await {
        Ok(r) => json!(r),
        Err(e) => {
            tracing::error!("store_memory failed: {e:?}");
            json!({"success": false, "error": e.safe_message()})
        }
    };
    let _ = reply.send(result);
}
```

Repeat for all 9 command variants.

- [ ] **Step 2: Run all tests**

Run: `cargo test --workspace`
Expected: All pass (logging doesn't affect behavior)

- [ ] **Step 3: Commit**

```bash
git add crates/alaya-server/src/main.rs
git commit -m "fix(server): log full error details before returning safe_message to clients"
```

---

### Task 7: Fix N+1 in memory_contradictions (M6)

**Files:**
- Modify: `crates/alaya-core/src/service.rs:839-878`

- [ ] **Step 1: Replace N+1 loop with batch fetch**

Replace the sequential loop:
```rust
pub async fn memory_contradictions(&self, limit: usize) -> Result<Value> {
    let pairs = self.graph.get_all_contradictions(limit).await?;

    // Batch fetch all referenced memories
    let all_hashes: Vec<&str> = pairs
        .iter()
        .flat_map(|p| [p.memory_a_hash.as_str(), p.memory_b_hash.as_str()])
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let memories = self.vectors.get_batch(&all_hashes).await.unwrap_or_default();
    let lookup: std::collections::HashMap<&str, &Memory> = memories
        .iter()
        .map(|m| (m.content_hash.as_str(), m))
        .collect();

    let mut enriched: Vec<Value> = Vec::new();
    for pair in &pairs {
        let a = lookup.get(pair.memory_a_hash.as_str());
        let b = lookup.get(pair.memory_b_hash.as_str());

        enriched.push(serde_json::json!({
            "memory_a_hash": pair.memory_a_hash,
            "memory_b_hash": pair.memory_b_hash,
            "confidence": pair.confidence,
            "memory_a_content": a.map(|m| truncate(&m.content, 200)),
            "memory_b_content": b.map(|m| truncate(&m.content, 200)),
            "memory_a_superseded": a.and_then(|m| {
                m.metadata.as_ref()?.get("superseded_by")
            }).is_some(),
            "memory_b_superseded": b.and_then(|m| {
                m.metadata.as_ref()?.get("superseded_by")
            }).is_some(),
        }));
    }

    Ok(serde_json::json!({
        "success": true,
        "pairs": enriched,
        "total": enriched.len(),
    }))
}
```

- [ ] **Step 2: Run all tests**

Run: `cargo test --workspace`
Expected: All pass

- [ ] **Step 3: Run integration test to verify against real backends**

Run: `QDRANT_URL=http://10.43.119.230:6333 EMBEDDING_URL=http://10.43.242.167 cargo test -p alaya-core --test integration -- --test-threads=1`
Expected: All 5 pass

- [ ] **Step 4: Commit**

```bash
git add crates/alaya-core/src/service.rs
git commit -m "perf(core): batch fetch in memory_contradictions (was N+1 sequential queries)"
```

---

### Task 8: Cleanup — dead code + inconsistent dispatch (m2 + m5)

**Files:**
- Modify: `crates/alaya-backends/src/qdrant.rs:72, 117-119`
- Modify: `crates/alaya-server/src/mcp.rs:262-383`

- [ ] **Step 1: Remove dead must_not Vec**

Remove line 72 (`let must_not: Vec<Value> = Vec::new();`) and lines 117-119 (the `if !must_not.is_empty()` check).

- [ ] **Step 2: Create typed param structs for remaining tools**

In mcp.rs, add structs for the manually-extracted tools:
```rust
#[derive(Deserialize)]
struct DeleteParams { content_hash: String }

#[derive(Deserialize)]
struct SupersedeParams { old_id: String, new_id: String, #[serde(default)] reason: String }

#[derive(Deserialize)]
struct ContradictionsParams { #[serde(default = "default_contradictions_limit")] limit: usize }
fn default_contradictions_limit() -> usize { 20 }

#[derive(Deserialize)]
struct FindDuplicatesParams {
    #[serde(default = "default_dup_threshold")] similarity_threshold: f64,
    #[serde(default = "default_dup_limit")] limit: usize,
    #[serde(default)] strategy: CanonicalStrategy,
}
fn default_dup_threshold() -> f64 { 0.95 }
fn default_dup_limit() -> usize { 100 }

#[derive(Deserialize)]
struct MergeDuplicatesParams {
    canonical_hash: String,
    duplicate_hashes: Vec<String>,
    #[serde(default = "default_merge_reason")] reason: String,
    #[serde(default)] dry_run: bool,
}
fn default_merge_reason() -> String { "Merged by deduplication".into() }
```

- [ ] **Step 3: Replace manual extraction with from_value**

Replace all manual `.get().and_then().ok_or()` chains with `serde_json::from_value::<T>(args)?`.

- [ ] **Step 4: Run all tests**

Run: `cargo test --workspace`
Expected: All pass (7 MCP unit tests still green)

- [ ] **Step 5: Commit**

```bash
git add crates/alaya-backends/src/qdrant.rs crates/alaya-server/src/mcp.rs
git commit -m "refactor: remove dead must_not Vec, standardize MCP dispatch on typed params"
```

---

## Summary

| Task | Fixes | Risk | Lines changed (est.) |
|------|-------|------|---------------------|
| 1 | C3 (hash_to_uuid panic) | Critical panic fix | ~20 |
| 2 | C2 (truncate UTF-8 panic) | Critical panic fix | ~10 |
| 3 | M4 (negation substring bug) | Bug fix | ~5 |
| 4 | M1 (GraphRef DRY) | -100 lines of duplication | ~120 net deletion |
| 5 | M7 + m7 (Config Clone, Cmd named) | Ergonomics | ~40 |
| 6 | M2 (error logging) | Observability | ~20 |
| 7 | M6 (N+1 batch) | Performance | ~20 |
| 8 | m2 + m5 (dead code, dispatch) | Cleanup | ~50 |

**Deferred:**
- C1 (typed response structs) — do alongside HttpClient refactor
- M3 (increment_access_count TOCTOU) — document, fix when adding cachekit
- M5 (tag caching) — fix when adding cachekit
- m3 (get_recent over-fetch) — minor, fix opportunistically
- m4 (MemoryService unit tests) — add alongside typed responses
- m6 (tool_schemas validation) — add alongside MCP extraction to alaya-core
