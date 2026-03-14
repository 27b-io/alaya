//! Typed Cypher query construction for all FalkorDB graph operations.
//!
//! Each function returns `(query_string, params_map, readonly)`.
//! Relationship labels come from compile-time enums — never from raw user input.

use std::collections::HashMap;

use alaya_types::graph::{Direction, SystemRelationType, UserRelationType};
use serde_json::{Value, json};

/// A fully-constructed Cypher query ready to dispatch.
/// `(cypher, params, readonly)`
pub type CypherQuery = (String, HashMap<String, Value>, bool);

// ─── helpers ─────────────────────────────────────────────────────────────────

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

// ─── node operations ─────────────────────────────────────────────────────────

/// MERGE a Memory node, setting `created_at` only on creation.
pub fn ensure_node(hash: &str, ts: f64) -> CypherQuery {
    let q = "MERGE (m:Memory {content_hash: $hash}) \
             ON CREATE SET m.created_at = $ts"
        .to_string();
    (
        q,
        params(&[("hash", json!(hash)), ("ts", json!(ts))]),
        false,
    )
}

/// DETACH DELETE a Memory node by hash.
pub fn delete_node(hash: &str) -> CypherQuery {
    let q = "MATCH (m:Memory {content_hash: $hash}) DETACH DELETE m".to_string();
    (q, params(&[("hash", json!(hash))]), false)
}

// ─── typed edge operations ────────────────────────────────────────────────────

/// MERGE a typed edge between two memories.
///
/// `confidence` is optional — when `None` the `e.confidence` property is omitted.
pub fn create_typed_edge(
    src: &str,
    dst: &str,
    rel: UserRelationType,
    ts: f64,
    confidence: Option<f64>,
) -> CypherQuery {
    let label = rel.cypher_label();
    let q = if let Some(_conf) = confidence {
        format!(
            "MATCH (a:Memory {{content_hash: $src}}), (b:Memory {{content_hash: $dst}}) \
             MERGE (a)-[e:{label}]->(b) \
             ON CREATE SET e.created_at = $ts, e.confidence = $conf \
             RETURN count(e)"
        )
    } else {
        format!(
            "MATCH (a:Memory {{content_hash: $src}}), (b:Memory {{content_hash: $dst}}) \
             MERGE (a)-[e:{label}]->(b) \
             ON CREATE SET e.created_at = $ts \
             RETURN count(e)"
        )
    };

    let mut p = params(&[("src", json!(src)), ("dst", json!(dst)), ("ts", json!(ts))]);
    if let Some(conf) = confidence {
        p.insert("conf".to_string(), json!(conf));
    }
    (q, p, false)
}

/// Query typed edges for a hash in the given direction.
///
/// Returns one query per `UserRelationType` variant.  Caller unions the results.
pub fn get_typed_edges(
    hash: &str,
    rel: UserRelationType,
    direction: Direction,
    limit: u32,
) -> CypherQuery {
    let label = rel.cypher_label();
    let q = match direction {
        Direction::Outgoing => format!(
            "MATCH (a:Memory {{content_hash: $hash}})-[e:{label}]->(b:Memory) \
             RETURN a.content_hash, b.content_hash, e.created_at \
             LIMIT $lim"
        ),
        Direction::Incoming => format!(
            "MATCH (a:Memory)-[e:{label}]->(b:Memory {{content_hash: $hash}}) \
             RETURN a.content_hash, b.content_hash, e.created_at \
             LIMIT $lim"
        ),
        Direction::Both => format!(
            "MATCH (a:Memory {{content_hash: $hash}})-[e:{label}]-(b:Memory) \
             RETURN a.content_hash, b.content_hash, e.created_at \
             LIMIT $lim"
        ),
    };
    (
        q,
        params(&[("hash", json!(hash)), ("lim", json!(limit))]),
        true,
    )
}

/// DELETE a single typed edge between two memories.
pub fn delete_typed_edge(src: &str, dst: &str, rel: UserRelationType) -> CypherQuery {
    let label = rel.cypher_label();
    let q = format!(
        "MATCH (a:Memory {{content_hash: $src}})-[e:{label}]->(b:Memory {{content_hash: $dst}}) \
         DELETE e RETURN count(e)"
    );
    (
        q,
        params(&[("src", json!(src)), ("dst", json!(dst))]),
        false,
    )
}

// ─── system edge operations ───────────────────────────────────────────────────

/// MERGE a system-managed edge (e.g. SUPERSEDES).
pub fn create_system_edge(src: &str, dst: &str, rel: SystemRelationType, ts: f64) -> CypherQuery {
    let label = rel.cypher_label();
    let q = format!(
        "MATCH (a:Memory {{content_hash: $src}}), (b:Memory {{content_hash: $dst}}) \
         MERGE (a)-[e:{label}]->(b) \
         ON CREATE SET e.created_at = $ts \
         RETURN count(e)"
    );
    (
        q,
        params(&[("src", json!(src)), ("dst", json!(dst)), ("ts", json!(ts))]),
        false,
    )
}

// ─── contradiction operations ─────────────────────────────────────────────────

/// Fetch all CONTRADICTS pairs ordered by `created_at DESC`.
pub fn get_all_contradictions(limit: u32) -> CypherQuery {
    let q = "MATCH (a:Memory)-[e:CONTRADICTS]->(b:Memory) \
             RETURN a.content_hash, b.content_hash, e.confidence, e.created_at \
             ORDER BY e.created_at DESC \
             LIMIT $lim"
        .to_string();
    (q, params(&[("lim", json!(limit))]), true)
}

/// Fetch CONTRADICTS pairs touching any of the supplied hashes.
pub fn get_contradictions_for_hashes(hashes: &[&str]) -> CypherQuery {
    let q = "MATCH (a:Memory)-[e:CONTRADICTS]->(b:Memory) \
             WHERE a.content_hash IN $hashes OR b.content_hash IN $hashes \
             RETURN a.content_hash, b.content_hash, e.confidence"
        .to_string();
    let hash_list: Vec<Value> = hashes.iter().map(|h| json!(h)).collect();
    (q, params(&[("hashes", Value::Array(hash_list))]), true)
}

// ─── Hebbian read operations ──────────────────────────────────────────────────

/// Walk HEBBIAN edges up to `max_hops` (capped at 3) from a source node.
pub fn get_neighbors(hash: &str, max_hops: u8, min_weight: f64, limit: u32) -> CypherQuery {
    let hops = max_hops.min(3);
    let q = format!(
        "MATCH (src:Memory {{content_hash: $hash}})-[e:HEBBIAN*1..{hops}]->(dst:Memory) \
         WHERE ALL(r IN e WHERE r.weight >= $min_w) \
         WITH dst, e, length(e) AS hops \
         RETURN DISTINCT dst.content_hash AS hash, \
                reduce(w = 1.0, r IN e | w * r.weight) AS path_weight, \
                hops \
         ORDER BY path_weight DESC \
         LIMIT $lim"
    );
    (
        q,
        params(&[
            ("hash", json!(hash)),
            ("min_w", json!(min_weight)),
            ("lim", json!(limit)),
        ]),
        true,
    )
}

/// Spreading activation from a set of seed hashes.
pub fn spreading_activation(seeds: &[&str], max_hops: u8) -> CypherQuery {
    let hops = max_hops.min(3);
    let q = format!(
        "MATCH (src:Memory)-[e:HEBBIAN*1..{hops}]->(dst:Memory) \
         WHERE src.content_hash IN $seeds AND NOT dst.content_hash IN $seeds \
         WITH dst.content_hash AS hash, \
              reduce(w = 1.0, r IN e | w * r.weight) AS path_weight, \
              length(e) AS hops \
         RETURN hash, path_weight, hops"
    );
    let seed_list: Vec<Value> = seeds.iter().map(|s| json!(s)).collect();
    (q, params(&[("seeds", Value::Array(seed_list))]), true)
}

/// Maximum HEBBIAN weight for all edges within a set of hashes.
pub fn hebbian_boosts_within(hashes: &[&str]) -> CypherQuery {
    let q = "MATCH (a:Memory)-[e:HEBBIAN]->(b:Memory) \
             WHERE a.content_hash IN $hashes AND b.content_hash IN $hashes \
             RETURN a.content_hash AS hash, max(e.weight) AS max_weight"
        .to_string();
    let hash_list: Vec<Value> = hashes.iter().map(|h| json!(h)).collect();
    (q, params(&[("hashes", Value::Array(hash_list))]), true)
}

// ─── Hebbian write operations ─────────────────────────────────────────────────

/// MERGE a HEBBIAN edge and apply LTP update on match.
#[allow(clippy::too_many_arguments)]
pub fn strengthen_edge(
    src: &str,
    dst: &str,
    init_weight: f64,
    rate: f64,
    max_weight: f64,
    spacing_modifier: f64,
    ts: f64,
) -> CypherQuery {
    let q = "MATCH (a:Memory {content_hash: $src}), (b:Memory {content_hash: $dst}) \
             MERGE (a)-[e:HEBBIAN]->(b) \
             ON CREATE SET e.weight = $init_w, e.co_access_count = 1, \
                           e.created_at = $ts, e.last_co_access = $ts \
             ON MATCH SET e.weight = toFloat(CASE \
               WHEN e.weight + $rate * (1.0 - e.weight / $max_w) * $sp_mod > $max_w THEN $max_w \
               ELSE e.weight + $rate * (1.0 - e.weight / $max_w) * $sp_mod END), \
               e.co_access_count = e.co_access_count + 1, \
               e.last_co_access = $ts"
        .to_string();
    (
        q,
        params(&[
            ("src", json!(src)),
            ("dst", json!(dst)),
            ("init_w", json!(init_weight)),
            ("rate", json!(rate)),
            ("max_w", json!(max_weight)),
            ("sp_mod", json!(spacing_modifier)),
            ("ts", json!(ts)),
        ]),
        false,
    )
}

// ─── consolidation operations ─────────────────────────────────────────────────

/// Decay all HEBBIAN edge weights by a factor (batch, limited).
pub fn decay_all_edges(decay: f64, limit: u32) -> CypherQuery {
    let q = "MATCH ()-[e:HEBBIAN]->() WITH e LIMIT $lim \
             SET e.weight = toFloat(e.weight * $decay) RETURN count(e)"
        .to_string();
    (
        q,
        params(&[("decay", json!(decay)), ("lim", json!(limit))]),
        false,
    )
}

/// Decay HEBBIAN edges that have not been co-accessed since `before_ts`.
pub fn decay_stale_edges(before_ts: f64, decay: f64, limit: u32) -> CypherQuery {
    let q = "MATCH ()-[e:HEBBIAN]->() WHERE e.last_co_access < $ts WITH e LIMIT $lim \
             SET e.weight = toFloat(e.weight * $decay) RETURN count(e)"
        .to_string();
    (
        q,
        params(&[
            ("ts", json!(before_ts)),
            ("decay", json!(decay)),
            ("lim", json!(limit)),
        ]),
        false,
    )
}

/// DELETE HEBBIAN edges whose weight has dropped below `threshold`.
pub fn prune_weak_edges(threshold: f64, limit: u32) -> CypherQuery {
    let q = "MATCH ()-[e:HEBBIAN]->() WHERE e.weight < $thresh WITH e LIMIT $lim \
             DELETE e RETURN count(e)"
        .to_string();
    (
        q,
        params(&[("thresh", json!(threshold)), ("lim", json!(limit))]),
        false,
    )
}

/// Find Memory nodes with no edges of any tracked type.
pub fn get_orphan_nodes(limit: u32) -> CypherQuery {
    let q = "MATCH (m:Memory) \
             WHERE NOT (m)-[:HEBBIAN]-() \
               AND NOT (m)-[:RELATES_TO]-() \
               AND NOT (m)-[:PRECEDES]-() \
               AND NOT (m)-[:CONTRADICTS]-() \
               AND NOT (m)-[:SUPERSEDES]-() \
             RETURN m.content_hash LIMIT $lim"
        .to_string();
    (q, params(&[("lim", json!(limit))]), true)
}

/// Return multiple queries for a full graph statistics snapshot.
///
/// Each tuple is independent; results must be collected by the caller.
pub fn get_graph_stats() -> Vec<CypherQuery> {
    let empty: HashMap<String, Value> = HashMap::new();

    let node_count = (
        "MATCH (m:Memory) RETURN count(m)".to_string(),
        empty.clone(),
        true,
    );
    let hebbian_count = (
        "MATCH ()-[e:HEBBIAN]->() RETURN count(e)".to_string(),
        empty.clone(),
        true,
    );

    let mut queries = vec![node_count, hebbian_count];

    // one query per user relation type
    for rel in [
        UserRelationType::RelatesTo,
        UserRelationType::Precedes,
        UserRelationType::Contradicts,
    ] {
        let label = rel.cypher_label();
        queries.push((
            format!("MATCH ()-[e:{label}]->() RETURN count(e)"),
            empty.clone(),
            true,
        ));
    }

    // system relation types
    let label = SystemRelationType::Supersedes.cypher_label();
    queries.push((
        format!("MATCH ()-[e:{label}]->() RETURN count(e)"),
        empty.clone(),
        true,
    ));

    queries
}

// ─── schema ───────────────────────────────────────────────────────────────────

/// DDL statements to ensure required indexes exist.
pub fn schema_statements() -> Vec<CypherQuery> {
    let empty: HashMap<String, Value> = HashMap::new();
    vec![
        (
            "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.content_hash)".to_string(),
            empty.clone(),
            false,
        ),
        (
            "CREATE INDEX IF NOT EXISTS FOR (m:Memory) ON (m.created_at)".to_string(),
            empty,
            false,
        ),
    ]
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // node operations

    #[test]
    fn ensure_node_shape() {
        let (q, p, ro) = ensure_node("abc123", 1_710_000_000.0);
        assert!(q.contains("MERGE"), "must MERGE");
        assert!(q.contains("Memory"), "must target Memory label");
        assert!(q.contains("ON CREATE SET"), "must set on create");
        assert!(p.contains_key("hash"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    #[test]
    fn delete_node_shape() {
        let (q, p, ro) = delete_node("abc123");
        assert!(q.contains("DETACH DELETE"));
        assert!(p.contains_key("hash"));
        assert!(!ro);
    }

    // typed edge operations

    #[test]
    fn create_typed_edge_with_confidence() {
        let (q, p, ro) = create_typed_edge("a", "b", UserRelationType::RelatesTo, 1.0, Some(0.9));
        assert!(q.contains("RELATES_TO"));
        assert!(q.contains("e.confidence"));
        assert!(p.contains_key("conf"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    #[test]
    fn create_typed_edge_without_confidence() {
        let (q, p, ro) = create_typed_edge("a", "b", UserRelationType::Contradicts, 1.0, None);
        assert!(q.contains("CONTRADICTS"));
        assert!(
            !q.contains("e.confidence"),
            "must omit confidence when None"
        );
        assert!(!p.contains_key("conf"));
        assert!(!ro);
    }

    #[test]
    fn get_typed_edges_outgoing() {
        let (q, p, ro) = get_typed_edges("h", UserRelationType::Precedes, Direction::Outgoing, 100);
        assert!(q.contains("PRECEDES"));
        assert!(q.contains("LIMIT"));
        assert!(p.contains_key("hash"));
        assert!(p.contains_key("lim"));
        assert!(ro);
        // outgoing: hash node on left
        let hash_pos = q.find("content_hash: $hash").unwrap();
        let arrow_pos = q.find("->").unwrap();
        assert!(hash_pos < arrow_pos, "source should appear before arrow");
    }

    #[test]
    fn get_typed_edges_incoming() {
        let (q, p, ro) = get_typed_edges("h", UserRelationType::Precedes, Direction::Incoming, 50);
        assert!(ro);
        assert!(p.contains_key("hash"));
        // incoming: hash node on right
        let hash_pos = q.find("content_hash: $hash").unwrap();
        let arrow_pos = q.find("->").unwrap();
        assert!(hash_pos > arrow_pos, "target should appear after arrow");
    }

    #[test]
    fn delete_typed_edge_shape() {
        let (q, p, ro) = delete_typed_edge("s", "d", UserRelationType::RelatesTo);
        assert!(q.contains("DELETE e"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(!ro);
    }

    // system edge

    #[test]
    fn create_system_edge_shape() {
        let (q, p, ro) = create_system_edge("s", "d", SystemRelationType::Supersedes, 0.0);
        assert!(q.contains("SUPERSEDES"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    // contradiction operations

    #[test]
    fn get_all_contradictions_shape() {
        let (q, p, ro) = get_all_contradictions(20);
        assert!(q.contains("CONTRADICTS"));
        assert!(q.contains("ORDER BY e.created_at DESC"));
        assert!(p.contains_key("lim"));
        assert!(ro);
    }

    #[test]
    fn get_contradictions_for_hashes_shape() {
        let (q, p, ro) = get_contradictions_for_hashes(&["a", "b"]);
        assert!(q.contains("CONTRADICTS"));
        assert!(q.contains("IN $hashes"));
        assert!(p.contains_key("hashes"));
        let arr = p["hashes"].as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(ro);
    }

    // Hebbian reads

    #[test]
    fn get_neighbors_caps_hops() {
        let (q, p, ro) = get_neighbors("h", 10, 0.1, 25);
        // 10 capped to 3
        assert!(q.contains("HEBBIAN*1..3"), "hops must be capped at 3");
        assert!(p.contains_key("hash"));
        assert!(p.contains_key("min_w"));
        assert!(p.contains_key("lim"));
        assert!(ro);
    }

    #[test]
    fn get_neighbors_hops_not_capped_when_small() {
        let (q, _, _) = get_neighbors("h", 2, 0.1, 10);
        assert!(q.contains("HEBBIAN*1..2"));
    }

    #[test]
    fn spreading_activation_shape() {
        let (q, p, ro) = spreading_activation(&["s1", "s2"], 2);
        assert!(q.contains("HEBBIAN*1..2"));
        assert!(q.contains("NOT dst.content_hash IN $seeds"));
        assert!(p.contains_key("seeds"));
        assert!(ro);
    }

    #[test]
    fn hebbian_boosts_within_shape() {
        let (q, p, ro) = hebbian_boosts_within(&["a", "b", "c"]);
        assert!(q.contains("HEBBIAN"));
        assert!(q.contains("max(e.weight)"));
        assert!(p.contains_key("hashes"));
        assert!(ro);
    }

    // Hebbian writes

    #[test]
    fn strengthen_edge_shape() {
        let (q, p, ro) = strengthen_edge("s", "d", 0.3, 0.1, 1.0, 1.0, 1_710_000_000.0);
        assert!(q.contains("HEBBIAN"));
        assert!(q.contains("ON CREATE SET"));
        assert!(q.contains("ON MATCH SET"));
        assert!(q.contains("co_access_count"));
        assert!(p.contains_key("src"));
        assert!(p.contains_key("dst"));
        assert!(p.contains_key("init_w"));
        assert!(p.contains_key("rate"));
        assert!(p.contains_key("max_w"));
        assert!(p.contains_key("sp_mod"));
        assert!(p.contains_key("ts"));
        assert!(!ro);
    }

    // consolidation

    #[test]
    fn decay_all_edges_shape() {
        let (q, p, ro) = decay_all_edges(0.95, 1000);
        assert!(q.contains("HEBBIAN"));
        assert!(q.contains("e.weight * $decay"));
        assert!(p.contains_key("decay"));
        assert!(p.contains_key("lim"));
        assert!(!ro);
    }

    #[test]
    fn decay_stale_edges_shape() {
        let (q, p, ro) = decay_stale_edges(1_710_000_000.0, 0.9, 500);
        assert!(q.contains("last_co_access < $ts"));
        assert!(p.contains_key("ts"));
        assert!(p.contains_key("decay"));
        assert!(!ro);
    }

    #[test]
    fn prune_weak_edges_shape() {
        let (q, p, ro) = prune_weak_edges(0.05, 500);
        assert!(q.contains("e.weight < $thresh"));
        assert!(q.contains("DELETE e"));
        assert!(p.contains_key("thresh"));
        assert!(p.contains_key("lim"));
        assert!(!ro);
    }

    #[test]
    fn get_orphan_nodes_shape() {
        let (q, p, ro) = get_orphan_nodes(100);
        assert!(q.contains("NOT (m)-[:HEBBIAN]-()"));
        assert!(q.contains("NOT (m)-[:SUPERSEDES]-()"));
        assert!(p.contains_key("lim"));
        assert!(ro);
    }

    // graph stats

    #[test]
    fn get_graph_stats_returns_multiple_queries() {
        let qs = get_graph_stats();
        // node_count + hebbian + 3 user rels + 1 system rel = 6
        assert_eq!(qs.len(), 6);
        // all must be readonly
        assert!(qs.iter().all(|(_, _, ro)| *ro));
        let cypher_joined: String = qs
            .iter()
            .map(|(q, _, _)| q.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        assert!(cypher_joined.contains("HEBBIAN"));
        assert!(cypher_joined.contains("RELATES_TO"));
        assert!(cypher_joined.contains("PRECEDES"));
        assert!(cypher_joined.contains("CONTRADICTS"));
        assert!(cypher_joined.contains("SUPERSEDES"));
    }

    // schema

    #[test]
    fn schema_statements_shape() {
        let stmts = schema_statements();
        assert_eq!(stmts.len(), 2);
        assert!(stmts.iter().all(|(_, _, ro)| !ro), "DDL is not readonly");
        assert!(stmts[0].0.contains("content_hash"));
        assert!(stmts[1].0.contains("created_at"));
    }
}
