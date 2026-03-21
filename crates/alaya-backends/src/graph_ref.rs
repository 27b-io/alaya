//! Newtype wrappers that split a single `Rc<GraphHttpClient>` across the
//! three separate trait bounds that `MemoryService` requires.
//!
//! `GraphHttpClient` implements `GraphService`, `HebbianService`, and
//! `ConsolidationService`, but Rust does not allow constructing three
//! `Box<dyn Trait>` from a single value without cloning or wrapping.
//! These newtypes wrap an `Rc` so the same client can be shared cheaply
//! across all three boxes in a single-threaded (`?Send`) context.

use std::rc::Rc;

use crate::graph::GraphHttpClient;
use crate::{ConsolidationService, GraphService, HebbianService};

/// Delegates `GraphService` to a shared `GraphHttpClient`.
pub struct GraphRef(pub Rc<GraphHttpClient>);

/// Delegates `HebbianService` to a shared `GraphHttpClient`.
pub struct HebbianRef(pub Rc<GraphHttpClient>);

/// Delegates `ConsolidationService` to a shared `GraphHttpClient`.
pub struct ConsolidationRef(pub Rc<GraphHttpClient>);

// ─── GraphService ────────────────────────────────────────────────────────────

macro_rules! impl_graph_service {
    ($wrapper:ty) => {
        #[async_trait::async_trait(?Send)]
        impl GraphService for $wrapper {
            async fn ensure_node(&self, h: &str, t: f64) -> alaya_types::Result<()> {
                self.0.ensure_node(h, t).await
            }
            async fn delete_node(&self, h: &str) -> alaya_types::Result<()> {
                self.0.delete_node(h).await
            }
            async fn create_typed_edge(
                &self,
                s: &str,
                d: &str,
                r: alaya_types::graph::UserRelationType,
                m: alaya_types::graph::EdgeMeta,
            ) -> alaya_types::Result<bool> {
                self.0.create_typed_edge(s, d, r, m).await
            }
            async fn get_typed_edges(
                &self,
                h: &str,
                r: Option<alaya_types::graph::UserRelationType>,
                d: alaya_types::graph::Direction,
                l: usize,
            ) -> alaya_types::Result<Vec<alaya_types::graph::Edge>> {
                self.0.get_typed_edges(h, r, d, l).await
            }
            async fn delete_typed_edge(
                &self,
                s: &str,
                d: &str,
                r: alaya_types::graph::UserRelationType,
            ) -> alaya_types::Result<bool> {
                self.0.delete_typed_edge(s, d, r).await
            }
            async fn create_system_edge(
                &self,
                s: &str,
                d: &str,
                r: alaya_types::graph::SystemRelationType,
                t: f64,
            ) -> alaya_types::Result<bool> {
                self.0.create_system_edge(s, d, r, t).await
            }
            async fn get_all_contradictions(
                &self,
                l: usize,
            ) -> alaya_types::Result<Vec<alaya_types::graph::Contradiction>> {
                self.0.get_all_contradictions(l).await
            }
            async fn get_contradictions_for_hashes(
                &self,
                h: &[&str],
            ) -> alaya_types::Result<
                std::collections::HashMap<String, Vec<alaya_types::graph::ContradictionRef>>,
            > {
                self.0.get_contradictions_for_hashes(h).await
            }
            async fn get_neighbors(
                &self,
                h: &str,
                hops: u8,
                w: f64,
                l: usize,
            ) -> alaya_types::Result<Vec<alaya_types::graph::Neighbor>> {
                self.0.get_neighbors(h, hops, w, l).await
            }
            async fn spreading_activation(
                &self,
                s: &[&str],
                hops: u8,
                d: f64,
                min: f64,
                l: usize,
            ) -> alaya_types::Result<std::collections::HashMap<String, f64>> {
                self.0.spreading_activation(s, hops, d, min, l).await
            }
            async fn hebbian_boosts_within(
                &self,
                h: &[&str],
            ) -> alaya_types::Result<std::collections::HashMap<String, f64>> {
                self.0.hebbian_boosts_within(h).await
            }
            async fn get_stats(&self) -> alaya_types::Result<alaya_types::graph::GraphStats> {
                self.0.get_stats().await
            }
        }
    };
}

impl_graph_service!(GraphRef);

// ─── HebbianService ──────────────────────────────────────────────────────────

#[async_trait::async_trait(?Send)]
impl HebbianService for HebbianRef {
    async fn enqueue_strengthen(
        &self,
        p: &[alaya_types::graph::CoAccessPair],
    ) -> alaya_types::Result<()> {
        self.0.enqueue_strengthen(p).await
    }
}

// ─── ConsolidationService ────────────────────────────────────────────────────

#[async_trait::async_trait(?Send)]
impl ConsolidationService for ConsolidationRef {
    async fn decay_all_edges(&self, f: f64, l: usize) -> alaya_types::Result<usize> {
        self.0.decay_all_edges(f, l).await
    }
    async fn decay_stale_edges(&self, b: f64, f: f64, l: usize) -> alaya_types::Result<usize> {
        self.0.decay_stale_edges(b, f, l).await
    }
    async fn prune_weak_edges(&self, t: f64, l: usize) -> alaya_types::Result<usize> {
        self.0.prune_weak_edges(t, l).await
    }
    async fn get_orphan_nodes(&self, l: usize) -> alaya_types::Result<Vec<String>> {
        self.0.get_orphan_nodes(l).await
    }
}
