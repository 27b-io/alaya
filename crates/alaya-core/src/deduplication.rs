//! Duplicate detection via pairwise cosine similarity and Union-Find clustering.
//!
//! Pure computation — no async, no HTTP. Given embeddings and metadata,
//! finds near-duplicate memories and groups them transitively.

use serde::{Deserialize, Serialize};

/// Cosine similarity between two vectors, computed in f64 for precision.
///
/// Returns 0.0 if either vector has zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");

    let mut dot = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;

    for (ai, bi) in a.iter().zip(b.iter()) {
        let ai = *ai as f64;
        let bi = *bi as f64;
        dot += ai * bi;
        norm_a += ai * ai;
        norm_b += bi * bi;
    }

    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 { 0.0 } else { dot / denom }
}

/// A pair of memories detected as near-duplicates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicatePair {
    pub hash_a: String,
    pub hash_b: String,
    pub similarity: f64,
}

/// A group of transitively similar memories with a selected canonical entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuplicateGroup {
    pub hashes: Vec<String>,
    pub canonical_hash: String,
    pub max_similarity: f64,
    pub size: usize,
}

/// Strategy for selecting which memory in a duplicate group to keep.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanonicalStrategy {
    #[default]
    KeepNewest,
    KeepOldest,
    KeepMostAccessed,
}

/// Disjoint-set (Union-Find) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }
}

/// Find all pairs of memories whose cosine similarity meets or exceeds `threshold`.
///
/// O(n^2) pairwise comparison. Results sorted descending by similarity.
pub fn find_duplicate_pairs(
    hashes: &[&str],
    embeddings: &[Vec<f32>],
    threshold: f64,
) -> Vec<DuplicatePair> {
    debug_assert_eq!(hashes.len(), embeddings.len());

    let n = hashes.len();
    let mut pairs = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            if sim >= threshold {
                pairs.push(DuplicatePair {
                    hash_a: hashes[i].to_string(),
                    hash_b: hashes[j].to_string(),
                    similarity: sim,
                });
            }
        }
    }

    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    pairs
}

/// Cluster duplicate pairs into transitive groups using Union-Find.
///
/// Returns only groups with 2+ members. Each group is a `Vec<String>` of content hashes.
pub fn group_duplicates(pairs: &[DuplicatePair], all_hashes: &[&str]) -> Vec<Vec<String>> {
    if pairs.is_empty() {
        return Vec::new();
    }

    let index: std::collections::HashMap<&str, usize> = all_hashes
        .iter()
        .enumerate()
        .map(|(i, h)| (*h, i))
        .collect();

    let mut uf = UnionFind::new(all_hashes.len());

    for pair in pairs {
        if let (Some(&a), Some(&b)) = (
            index.get(pair.hash_a.as_str()),
            index.get(pair.hash_b.as_str()),
        ) {
            uf.union(a, b);
        }
    }

    // Group indices by root.
    let mut groups: std::collections::HashMap<usize, Vec<String>> =
        std::collections::HashMap::new();
    for (i, hash) in all_hashes.iter().enumerate() {
        let root = uf.find(i);
        groups.entry(root).or_default().push(hash.to_string());
    }

    groups.into_values().filter(|g| g.len() >= 2).collect()
}

/// Full deduplication pipeline: find pairs, cluster, select canonical per group.
///
/// `created_ats` and `access_counts` must be parallel arrays to `hashes`/`embeddings`.
pub fn build_duplicate_groups(
    hashes: &[&str],
    embeddings: &[Vec<f32>],
    created_ats: &[f64],
    access_counts: &[u64],
    threshold: f64,
    strategy: CanonicalStrategy,
) -> Vec<DuplicateGroup> {
    let pairs = find_duplicate_pairs(hashes, embeddings, threshold);
    let groups = group_duplicates(&pairs, hashes);

    let index: std::collections::HashMap<&str, usize> =
        hashes.iter().enumerate().map(|(i, h)| (*h, i)).collect();

    groups
        .into_iter()
        .map(|group_hashes| {
            let canonical_hash =
                select_canonical(&group_hashes, &index, created_ats, access_counts, strategy);

            // Max pairwise similarity within this group.
            let max_similarity = pairs
                .iter()
                .filter(|p| group_hashes.contains(&p.hash_a) && group_hashes.contains(&p.hash_b))
                .map(|p| p.similarity)
                .fold(0.0_f64, f64::max);

            DuplicateGroup {
                size: group_hashes.len(),
                hashes: group_hashes,
                canonical_hash,
                max_similarity,
            }
        })
        .collect()
}

/// Select the canonical memory from a group based on the chosen strategy.
fn select_canonical(
    group_hashes: &[String],
    index: &std::collections::HashMap<&str, usize>,
    created_ats: &[f64],
    access_counts: &[u64],
    strategy: CanonicalStrategy,
) -> String {
    group_hashes
        .iter()
        .max_by(|a, b| {
            let ia = index[a.as_str()];
            let ib = index[b.as_str()];
            match strategy {
                CanonicalStrategy::KeepNewest => created_ats[ia]
                    .partial_cmp(&created_ats[ib])
                    .unwrap_or(std::cmp::Ordering::Equal),
                CanonicalStrategy::KeepOldest => created_ats[ib]
                    .partial_cmp(&created_ats[ia])
                    .unwrap_or(std::cmp::Ordering::Equal),
                CanonicalStrategy::KeepMostAccessed => access_counts[ia].cmp(&access_counts[ib]),
            }
        })
        .cloned()
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- cosine_similarity ---

    #[test]
    fn cosine_identical_vectors() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "identical vectors should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "orthogonal vectors should have similarity 0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_opposite_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-10,
            "opposite vectors should have similarity -1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let z = vec![0.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &z), 0.0);
        assert_eq!(cosine_similarity(&z, &a), 0.0);
        assert_eq!(cosine_similarity(&z, &z), 0.0);
    }

    #[test]
    fn cosine_known_value() {
        // cos([1,0], [1,1]) = 1/sqrt(2)
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }

    // --- UnionFind ---

    #[test]
    fn union_find_basic() {
        let mut uf = UnionFind::new(5);
        uf.union(0, 1);
        uf.union(2, 3);
        assert_eq!(uf.find(0), uf.find(1));
        assert_eq!(uf.find(2), uf.find(3));
        assert_ne!(uf.find(0), uf.find(2));
    }

    #[test]
    fn union_find_transitivity() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(1, 2);
        uf.union(2, 3);
        let root = uf.find(0);
        assert_eq!(uf.find(1), root);
        assert_eq!(uf.find(2), root);
        assert_eq!(uf.find(3), root);
    }

    #[test]
    fn union_find_self() {
        let mut uf = UnionFind::new(3);
        assert_eq!(uf.find(0), 0);
        assert_eq!(uf.find(1), 1);
        uf.union(1, 1); // no-op
        assert_eq!(uf.find(1), 1);
    }

    // --- find_duplicate_pairs ---

    #[test]
    fn find_pairs_above_threshold() {
        let hashes = ["a", "b", "c"];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0], // very similar to a
            vec![0.0, 0.0, 1.0],  // orthogonal to both
        ];
        let pairs = find_duplicate_pairs(&hashes, &embeddings, 0.9);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].hash_a, "a");
        assert_eq!(pairs[0].hash_b, "b");
        assert!(pairs[0].similarity >= 0.9);
    }

    #[test]
    fn find_pairs_sorted_descending() {
        let hashes = ["a", "b", "c"];
        let embeddings = vec![vec![1.0, 0.0], vec![0.99, 0.14], vec![0.8, 0.6]];
        let pairs = find_duplicate_pairs(&hashes, &embeddings, 0.7);
        assert!(pairs.len() >= 2);
        for w in pairs.windows(2) {
            assert!(w[0].similarity >= w[1].similarity);
        }
    }

    #[test]
    fn find_pairs_none_above_threshold() {
        let hashes = ["a", "b"];
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let pairs = find_duplicate_pairs(&hashes, &embeddings, 0.5);
        assert!(pairs.is_empty());
    }

    // --- group_duplicates ---

    #[test]
    fn group_transitive_clustering() {
        let pairs = vec![
            DuplicatePair {
                hash_a: "a".into(),
                hash_b: "b".into(),
                similarity: 0.95,
            },
            DuplicatePair {
                hash_a: "b".into(),
                hash_b: "c".into(),
                similarity: 0.92,
            },
        ];
        let all = ["a", "b", "c", "d"];
        let groups = group_duplicates(&pairs, &all);
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.len(), 3);
        assert!(g.contains(&"a".to_string()));
        assert!(g.contains(&"b".to_string()));
        assert!(g.contains(&"c".to_string()));
    }

    #[test]
    fn group_separate_clusters() {
        let pairs = vec![
            DuplicatePair {
                hash_a: "a".into(),
                hash_b: "b".into(),
                similarity: 0.95,
            },
            DuplicatePair {
                hash_a: "c".into(),
                hash_b: "d".into(),
                similarity: 0.93,
            },
        ];
        let all = ["a", "b", "c", "d"];
        let groups = group_duplicates(&pairs, &all);
        assert_eq!(groups.len(), 2);
    }

    #[test]
    fn group_empty_pairs() {
        let groups = group_duplicates(&[], &["a", "b"]);
        assert!(groups.is_empty());
    }

    // --- build_duplicate_groups ---

    type TestData = (Vec<&'static str>, Vec<Vec<f32>>, Vec<f64>, Vec<u64>);

    fn test_data() -> TestData {
        let hashes = vec!["h1", "h2", "h3"];
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0], // near-duplicate of h1
            vec![0.0, 0.0, 1.0],  // different
        ];
        let created_ats = vec![100.0, 200.0, 300.0];
        let access_counts = vec![10, 5, 20];
        (hashes, embeddings, created_ats, access_counts)
    }

    #[test]
    fn build_groups_keep_newest() {
        let (hashes, embeddings, created_ats, access_counts) = test_data();
        let groups = build_duplicate_groups(
            &hashes,
            &embeddings,
            &created_ats,
            &access_counts,
            0.9,
            CanonicalStrategy::KeepNewest,
        );
        assert_eq!(groups.len(), 1);
        let g = &groups[0];
        assert_eq!(g.size, 2);
        assert_eq!(g.canonical_hash, "h2"); // created_at 200 > 100
        assert!(g.hashes.contains(&"h1".to_string()));
        assert!(g.hashes.contains(&"h2".to_string()));
        assert!(g.max_similarity >= 0.9);
    }

    #[test]
    fn build_groups_keep_oldest() {
        let (hashes, embeddings, created_ats, access_counts) = test_data();
        let groups = build_duplicate_groups(
            &hashes,
            &embeddings,
            &created_ats,
            &access_counts,
            0.9,
            CanonicalStrategy::KeepOldest,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].canonical_hash, "h1"); // created_at 100 < 200
    }

    #[test]
    fn build_groups_keep_most_accessed() {
        let (hashes, embeddings, created_ats, access_counts) = test_data();
        let groups = build_duplicate_groups(
            &hashes,
            &embeddings,
            &created_ats,
            &access_counts,
            0.9,
            CanonicalStrategy::KeepMostAccessed,
        );
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].canonical_hash, "h1"); // access_count 10 > 5
    }

    #[test]
    fn build_groups_high_threshold_no_matches() {
        let (hashes, embeddings, created_ats, access_counts) = test_data();
        let groups = build_duplicate_groups(
            &hashes,
            &embeddings,
            &created_ats,
            &access_counts,
            0.9999,
            CanonicalStrategy::default(),
        );
        assert!(groups.is_empty());
    }
}
