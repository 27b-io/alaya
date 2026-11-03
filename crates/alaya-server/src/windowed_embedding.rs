//! Embeds a passage longer than the embedder's input window as the
//! L2-normalised mean of overlapping windows, so its tail reaches the vector
//! instead of being truncated away. Two passages that share a long prefix
//! but differ after it get distinct vectors.
//!
//! Wraps the cache: each window is cached under the ordinary key for its own
//! text, and the whole-text key of a long passage, whose cached vector is the
//! truncated one, is never read. Queries and passages that fit pass through
//! untouched, so their vectors and cache keys stay as they were.
//!
//! Windows are cut on the embedder's own token boundaries (its `/tokenize`),
//! so each one fits exactly. When the budget or a token layout cannot be
//! read, the text is sent whole, as before (the embedder truncates it), and
//! a warning is logged.

use std::cell::Cell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::StreamExt;

use alaya_backends::traits::{EmbeddingProvider, Tokenizer};
use alaya_core::vitals::{CountingEmbedding, Vitals};
use alaya_types::memory::HealthStatus;
use alaya_types::search::PromptName;
use alaya_types::{AlayaError, Result};

use crate::cached_embedding::CachedEmbedding;

/// L1 embedding-cache entries (~40 MB at 1024 dims).
const L1_CAPACITY: u64 = 10_000;

/// The embedding stack every store and search goes through, outermost
/// first: windowing, the cache, the failure counter, the endpoint. The
/// counter sits under the cache because a cache hit is not an embed call
/// that could fail. Windowing sits above the cache because below it, a long
/// passage's truncated whole-text entry would keep answering for it.
pub fn embedding_stack(
    endpoint: Box<dyn EmbeddingProvider>,
    tokenizer: Box<dyn Tokenizer>,
    l2: Option<cachekit::CacheKit>,
    vitals: Arc<Vitals>,
) -> WindowedEmbedding {
    let counted = CountingEmbedding::new(endpoint, vitals);
    let cached = CachedEmbedding::new(Box::new(counted), L1_CAPACITY, l2);
    WindowedEmbedding::new(Box::new(cached), tokenizer)
}

/// Most windows one passage is embedded as, however long it is. Past this
/// the windows spread evenly from the head to the tail, leaving gaps between
/// them, so the work per store stays bounded. A store runs inline on the
/// worker loop, so every search sent to the same process waits for all of
/// it. On a CPU embedder three `WINDOW_TOKENS` windows cost about what one
/// 2048-token input does, so behind a 2048-token input window the longest
/// store holds searches no longer than it did before windowing.
const MAX_WINDOWS: usize = 3;
/// Tokens per window, at most. Smaller windows cost less per token on CPU
/// (attention grows with input length), and each one holds the embedder for
/// less time, so a query from another process queued there behind a long
/// store waits less.
const WINDOW_TOKENS: usize = 1024;
/// Neighbouring windows share `1 / OVERLAP_DIVISOR` of a window, so a
/// sentence cut at one window's edge appears whole in the next.
const OVERLAP_DIVISOR: usize = 8;
/// Tokens kept free below the embedder's passage budget: a window's text,
/// tokenized on its own, can split a few tokens differently at its cut edges.
const DRIFT_TOKENS: usize = 8;
/// Most UTF-8 bytes one byte of text can grow into under Unicode
/// normalization: NFKC turns U+FDFA, 3 bytes, into 33 (Unicode's
/// normalization FAQ lists this as the worst case).
const MAX_NORMALIZED_GROWTH: usize = 11;
/// `/tokenize` calls one batch keeps in flight. A dedup scan embeds up to
/// a few hundred passages in one batch; this bounds the requests, and the
/// replies held in memory, at once.
const TOKENIZE_CONCURRENCY: usize = 4;
/// How long a read of the embedder's budget, or a failure to read it,
/// stands. Bounded so a redeployed embedder with a different window is
/// picked up without a restart; a failure is retried, and warned about, at
/// most this often.
const BUDGET_TTL: Duration = Duration::from_secs(60);

/// One read of the embedder's passage budget.
#[derive(Clone, Copy)]
struct BudgetRead {
    /// Tokens of text a window may hold; `None`: the read failed, or left
    /// no room for a window.
    tokens: Option<usize>,
    at: Instant,
}

pub struct WindowedEmbedding {
    inner: Box<dyn EmbeddingProvider>,
    tokenizer: Box<dyn Tokenizer>,
    budget: Cell<Option<BudgetRead>>,
}

impl WindowedEmbedding {
    pub fn new(inner: Box<dyn EmbeddingProvider>, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            inner,
            tokenizer,
            budget: Cell::new(None),
        }
    }

    /// Tokens of passage text one input may hold, or `None` when that cannot
    /// be read or leaves no room (logged): passages are then sent whole.
    async fn budget(&self) -> Option<usize> {
        if let Some(read) = self.budget.get()
            && read.at.elapsed() < BUDGET_TTL
        {
            return read.tokens;
        }
        let tokens = match self.tokenizer.passage_token_budget().await {
            Ok(n) if n > DRIFT_TOKENS => Some(n - DRIFT_TOKENS),
            Ok(n) => {
                tracing::warn!(
                    budget = n,
                    "embedder input window too small to split, long passages embed truncated"
                );
                None
            }
            Err(e) => {
                tracing::warn!("embedder input window unknown, long passages embed truncated: {e}");
                None
            }
        };
        self.budget.set(Some(BudgetRead {
            tokens,
            at: Instant::now(),
        }));
        tokens
    }

    /// The windows `text` embeds as, or `None` when it is sent whole: it
    /// fits `budget`, or its token layout could not be read (logged).
    async fn windows<'t>(&self, text: &'t str, budget: usize) -> Option<Vec<&'t str>> {
        // Per the `Tokenizer` contract, a token holds at least one byte of
        // normalized text, plus one leading word marker. Normalization leaves
        // ASCII alone and grows any other byte to at most
        // MAX_NORMALIZED_GROWTH bytes, so a text whose bound is under the
        // budget fits unread, and a few em dashes cost no round trip, even on
        // a cache hit. Past the bound only the tokenizer can say (NFKC turns
        // U+FDFA, 3 bytes, into four tokens).
        let ascii = text.bytes().filter(u8::is_ascii).count();
        let bound = (text.len() - ascii)
            .saturating_mul(MAX_NORMALIZED_GROWTH)
            .saturating_add(ascii);
        if bound < budget {
            return None;
        }
        let spans = match self.tokenizer.token_spans(text).await {
            Ok(spans) => spans,
            Err(e) => {
                tracing::warn!(
                    bytes = text.len(),
                    "long passage embeds truncated, tokenize failed: {e}"
                );
                return None;
            }
        };
        if spans.len() <= budget {
            return None;
        }
        let size = WINDOW_TOKENS.min(budget);
        let windows: Option<Vec<&str>> = window_starts(spans.len(), size)
            .into_iter()
            .map(|s| text.get(spans[s].0..spans[s + size - 1].1))
            .collect();
        let Some(windows) = windows else {
            tracing::warn!(
                bytes = text.len(),
                "tokenize spans off char boundaries, passage embeds truncated"
            );
            return None;
        };
        tracing::info!(
            tokens = spans.len(),
            windows = windows.len(),
            "embedding long passage as windows"
        );
        Some(windows)
    }

    /// The L2-normalised mean of the windows' vectors. One window per
    /// request, in order: the embedder serves requests first come, first
    /// served, so a query another process sends mid-passage waits there for
    /// one window, not the whole passage.
    async fn pooled(&self, windows: &[&str]) -> Result<Vec<f32>> {
        let dims = self.inner.dimensions();
        let mut sum = vec![0.0_f32; dims];
        for window in windows {
            let vector = self
                .inner
                .embed_batch(&[window], PromptName::Passage)
                .await?
                .into_iter()
                .next()
                .filter(|v| v.len() == dims)
                .ok_or_else(|| {
                    AlayaError::Embedding(format!("window embed returned no {dims}-dim vector"))
                })?;
            for (s, x) in sum.iter_mut().zip(vector) {
                *s += x;
            }
        }
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if !norm.is_normal() {
            return Err(AlayaError::Embedding(format!(
                "pooled window vector has norm {norm}"
            )));
        }
        Ok(sum.into_iter().map(|x| x / norm).collect())
    }
}

/// First-token index of each window over `n` tokens, `size` tokens a window,
/// for `n > size > 0`. Consecutive windows overlap by at least
/// `size / OVERLAP_DIVISOR` tokens unless that takes more than
/// [`MAX_WINDOWS`]; then `MAX_WINDOWS` windows spread evenly. The first
/// window starts at the head and the last ends at the tail either way.
fn window_starts(n: usize, size: usize) -> Vec<usize> {
    let stride = size - size / OVERLAP_DIVISOR;
    let count = (1 + (n - size).div_ceil(stride)).min(MAX_WINDOWS);
    (0..count).map(|i| i * (n - size) / (count - 1)).collect()
}

#[async_trait(?Send)]
impl EmbeddingProvider for WindowedEmbedding {
    async fn embed_batch(&self, texts: &[&str], prompt_name: PromptName) -> Result<Vec<Vec<f32>>> {
        // Queries stay single-pass.
        if prompt_name == PromptName::Query {
            return self.inner.embed_batch(texts, prompt_name).await;
        }
        let Some(budget) = self.budget().await else {
            return self.inner.embed_batch(texts, prompt_name).await;
        };

        let plans: Vec<Option<Vec<&str>>> = futures::stream::iter(texts)
            .map(|t| self.windows(t, budget))
            .buffered(TOKENIZE_CONCURRENCY)
            .collect()
            .await;

        let single: Vec<&str> = texts
            .iter()
            .zip(&plans)
            .filter(|(_, p)| p.is_none())
            .map(|(t, _)| *t)
            .collect();
        let single_vectors = if single.is_empty() {
            Vec::new()
        } else {
            self.inner.embed_batch(&single, prompt_name).await?
        };
        if single_vectors.len() != single.len() {
            return Err(AlayaError::Embedding(format!(
                "embedding service returned {} vectors for {} texts",
                single_vectors.len(),
                single.len()
            )));
        }

        let mut out = Vec::with_capacity(texts.len());
        let mut single_vectors = single_vectors.into_iter();
        for plan in plans {
            match plan {
                Some(windows) => out.push(self.pooled(&windows).await?),
                None => out.extend(single_vectors.next()),
            }
        }
        Ok(out)
    }

    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn health(&self) -> Result<HealthStatus> {
        self.inner.health().await
    }

    /// Forwarded: a probe must reach past the cache below.
    async fn probe(&self, text: &str) -> Result<()> {
        self.inner.probe(text).await
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;

    const DIMS: usize = 16;

    /// Expands under the stub tokenizer as U+FDFA does under the real one:
    /// one 3-byte character, four tokens, all over the same bytes.
    const EXPANDS: char = '\u{FDFA}';

    /// Byte spans of the stub's tokens: one per space-separated word, and
    /// four per [`EXPANDS`].
    fn spans(text: &str) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        let mut start = None;
        for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
            if c == ' ' || c == EXPANDS {
                if let Some(s) = start.take() {
                    spans.push((s, i));
                }
                if c == EXPANDS {
                    spans.extend([(i, i + c.len_utf8()); 4]);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        spans
    }

    /// Tokenizes with [`spans`] and truncates input past `max_input` tokens
    /// when embedding, as the real embedder does.
    struct Words {
        max_input: usize,
        calls: RefCell<Vec<Vec<String>>>,
        /// How many of the next `/tokenize` calls fail.
        tokenize_failures: Cell<u32>,
    }

    impl Words {
        fn new(max_input: usize) -> std::rc::Rc<Self> {
            std::rc::Rc::new(Self {
                max_input,
                calls: RefCell::default(),
                tokenize_failures: Cell::new(0),
            })
        }

        /// Each token adds 1 to a dimension picked by its hash.
        fn vector(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0_f32; DIMS];
            for (start, stop) in spans(text).into_iter().take(self.max_input) {
                let h = text[start..stop]
                    .bytes()
                    .fold(7_usize, |h, b| h.wrapping_mul(31).wrapping_add(b as usize));
                v[h % DIMS] += 1.0;
            }
            let norm = v
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt()
                .max(f32::MIN_POSITIVE);
            v.into_iter().map(|x| x / norm).collect()
        }
    }

    struct Shared(std::rc::Rc<Words>);

    #[async_trait(?Send)]
    impl EmbeddingProvider for Shared {
        async fn embed_batch(&self, texts: &[&str], _p: PromptName) -> Result<Vec<Vec<f32>>> {
            self.0
                .calls
                .borrow_mut()
                .push(texts.iter().map(|t| t.to_string()).collect());
            Ok(texts.iter().map(|t| self.0.vector(t)).collect())
        }
        fn dimensions(&self) -> usize {
            DIMS
        }
        fn model_name(&self) -> &str {
            "words"
        }
        async fn health(&self) -> Result<HealthStatus> {
            unreachable!()
        }
    }

    #[async_trait(?Send)]
    impl Tokenizer for Shared {
        async fn passage_token_budget(&self) -> Result<usize> {
            Ok(self.0.max_input)
        }
        async fn token_spans(&self, text: &str) -> Result<Vec<(usize, usize)>> {
            let failures = self.0.tokenize_failures.get();
            if failures > 0 {
                self.0.tokenize_failures.set(failures - 1);
                return Err(AlayaError::Embedding("down".into()));
            }
            Ok(spans(text))
        }
    }

    fn windowed(words: &std::rc::Rc<Words>) -> WindowedEmbedding {
        WindowedEmbedding::new(
            Box::new(Shared(words.clone())),
            Box::new(Shared(words.clone())),
        )
    }

    /// `n` distinct words, `w0 w1 …`.
    fn text(n: usize) -> String {
        (0..n)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn token_count(s: &str) -> usize {
        spans(s).len()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    #[tokio::test]
    async fn a_passage_that_fits_passes_through_in_one_call() {
        let words = Words::new(2048);
        let e = windowed(&words);
        let long = text(2000);
        let out = e
            .embed_batch(&["short", &long], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(*words.calls.borrow(), vec![vec!["short".to_string(), long]]);
    }

    #[tokio::test]
    async fn a_query_is_never_windowed() {
        let words = Words::new(100);
        let e = windowed(&words);
        let long = text(5000);
        e.embed_batch(&[&long], PromptName::Query).await.unwrap();
        assert_eq!(*words.calls.borrow(), vec![vec![long]]);
    }

    /// A long shared prefix, then two different tails.
    /// Single-pass, the embedder truncates both to the same prefix; windowed,
    /// the tail window tells them apart.
    #[tokio::test]
    async fn a_shared_prefix_with_different_tails_gives_distinct_vectors() {
        let prefix = "calm ".repeat(2200);
        let a = format!("{prefix}the deploy failed on the staging cluster");
        let b = format!("{prefix}grandma's lemon cake needs more zest");

        let words = Words::new(2048);
        let truncated = Shared(words.clone());
        let ta = truncated
            .embed_batch(&[&a], PromptName::Passage)
            .await
            .unwrap();
        let tb = truncated
            .embed_batch(&[&b], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(ta, tb, "the stub must truncate like the real embedder");

        let e = windowed(&words);
        let out = e.embed_batch(&[&a, &b], PromptName::Passage).await.unwrap();
        assert_ne!(out[0], out[1], "tails must reach the vectors");
    }

    #[tokio::test]
    async fn windows_cover_head_to_tail_one_request_each_and_fit() {
        let words = Words::new(1040);
        let e = windowed(&words);
        let long = text(3000);
        e.embed_batch(&[&long], PromptName::Passage).await.unwrap();

        let calls = words.calls.borrow();
        assert!(calls.len() > 1);
        assert!(calls.iter().all(|c| c.len() == 1), "one window per request");
        let windows: Vec<&String> = calls.iter().map(|c| &c[0]).collect();
        assert!(windows[0].starts_with("w0 "));
        assert!(windows.last().unwrap().ends_with(" w2999"));
        for w in &windows {
            assert_eq!(token_count(w), WINDOW_TOKENS.min(1040 - DRIFT_TOKENS));
        }
    }

    /// AC: a passage far past the cap sends the embedder no more than
    /// MAX_WINDOWS windows, and still reaches its tail.
    #[tokio::test]
    async fn a_passage_past_the_cap_sends_at_most_max_windows() {
        let words = Words::new(2048);
        let e = windowed(&words);
        let huge = text(100_000);
        e.embed_batch(&[&huge], PromptName::Passage).await.unwrap();

        let calls = words.calls.borrow();
        assert_eq!(calls.len(), MAX_WINDOWS);
        assert!(calls.last().unwrap()[0].ends_with(" w99999"));
    }

    #[tokio::test]
    async fn the_pooled_vector_is_the_normalised_mean_of_the_windows() {
        let words = Words::new(1040);
        let e = windowed(&words);
        let long = text(3000);
        let pooled = e.embed_batch(&[&long], PromptName::Passage).await.unwrap();

        let mut sum = [0.0_f32; DIMS];
        for call in words.calls.borrow().iter() {
            for (s, x) in sum.iter_mut().zip(words.vector(&call[0])) {
                *s += x;
            }
        }
        let norm = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        let expected: Vec<f32> = sum.iter().map(|x| x / norm).collect();
        assert!(cosine(&pooled[0], &expected) > 0.99999);
        assert!((cosine(&pooled[0], &pooled[0]) - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn a_mixed_batch_keeps_its_order() {
        let words = Words::new(1040);
        let e = windowed(&words);
        let long = text(3000);
        let out = e
            .embed_batch(&["alpha", &long, "beta"], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], words.vector("alpha"));
        assert_eq!(out[2], words.vector("beta"));
        assert_ne!(out[1], words.vector(&long));
    }

    #[tokio::test]
    async fn a_failed_tokenize_falls_back_to_single_pass() {
        let words = Words::new(1040);
        words.tokenize_failures.set(u32::MAX);
        let e = windowed(&words);
        let long = text(3000);
        e.embed_batch(&[&long], PromptName::Passage).await.unwrap();
        assert_eq!(*words.calls.borrow(), vec![vec![long]]);
    }

    /// A non-TEI endpoint has no `/info`: passages embed single-pass, and
    /// the failed read is not retried (or warned about) on every embed.
    #[tokio::test]
    async fn an_unreadable_window_passes_through_and_is_not_reread_each_embed() {
        struct NoInfo(std::rc::Rc<Cell<u32>>);
        #[async_trait(?Send)]
        impl Tokenizer for NoInfo {
            async fn passage_token_budget(&self) -> Result<usize> {
                self.0.set(self.0.get() + 1);
                Err(AlayaError::Embedding("404".into()))
            }
            async fn token_spans(&self, _: &str) -> Result<Vec<(usize, usize)>> {
                unreachable!("no window, no tokenize")
            }
        }
        let words = Words::new(1040);
        let reads = std::rc::Rc::new(Cell::new(0));
        let e = WindowedEmbedding::new(
            Box::new(Shared(words.clone())),
            Box::new(NoInfo(reads.clone())),
        );
        let long = text(3000);
        for _ in 0..3 {
            e.embed_batch(&[&long], PromptName::Passage).await.unwrap();
        }
        assert_eq!(reads.get(), 1);
        assert_eq!(*words.calls.borrow(), vec![vec![long]; 3]);
    }

    /// A passage shorter in bytes than the budget can still hold more
    /// tokens than it: Unicode normalization expands U+FDFA, 3 bytes, into
    /// four tokens. Such a pair, sent whole, truncates to the same vector;
    /// it must be windowed instead, every window must fit, and the tails
    /// must reach the vectors.
    #[tokio::test]
    async fn a_short_passage_that_normalization_expands_is_windowed() {
        let prefix = EXPANDS.to_string().repeat(600);
        let a = format!("{prefix} the deploy failed on staging");
        let b = format!("{prefix} grandma lemon cake needs zest");
        let budget = 2048 - DRIFT_TOKENS;
        assert!(a.len() < budget && b.len() < budget);
        assert!(token_count(&a) > budget);

        let words = Words::new(2048);
        let truncated = Shared(words.clone());
        let ta = truncated
            .embed_batch(&[&a], PromptName::Passage)
            .await
            .unwrap();
        let tb = truncated
            .embed_batch(&[&b], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(ta, tb, "sent whole, the pair must truncate alike");
        words.calls.borrow_mut().clear();

        let e = windowed(&words);
        let out = e.embed_batch(&[&a, &b], PromptName::Passage).await.unwrap();
        assert_ne!(out[0], out[1], "tails must reach the vectors");
        let calls = words.calls.borrow();
        assert!(
            calls.iter().all(|c| c.len() == 1),
            "windows only, nothing whole"
        );
        assert!(calls.iter().all(|c| token_count(&c[0]) <= budget));
        assert!(calls.iter().any(|c| c[0].ends_with(" staging")));
        assert!(calls.iter().any(|c| c[0].ends_with(" zest")));
    }

    /// ASCII under the budget is never sent to the tokenizer: its behaviour,
    /// and its cost, are what they were before windowing.
    #[tokio::test]
    async fn a_short_ascii_passage_skips_the_tokenizer() {
        let words = Words::new(2048);
        words.tokenize_failures.set(u32::MAX);
        let e = windowed(&words);
        let short = "x".repeat(2048 - DRIFT_TOKENS - 1);
        e.embed_batch(&[&short], PromptName::Passage).await.unwrap();
        assert_eq!(
            words.tokenize_failures.get(),
            u32::MAX,
            "tokenizer was asked"
        );
        assert_eq!(*words.calls.borrow(), vec![vec![short]]);
    }

    /// An em dash does not send a short passage to the tokenizer: each
    /// non-ASCII byte counts as `MAX_NORMALIZED_GROWTH` toward the budget,
    /// and only a passage whose count reaches it is tokenized.
    #[tokio::test]
    async fn a_non_ascii_passage_skips_the_tokenizer_while_its_bound_fits() {
        let budget = 2048 - DRIFT_TOKENS;
        let dash = "\u{2014}";
        let fits = format!(
            "{dash}{}",
            "x".repeat(budget - 1 - dash.len() * MAX_NORMALIZED_GROWTH)
        );
        let reaches = format!("{fits}x");
        let words = Words::new(2048);
        words.tokenize_failures.set(u32::MAX);
        let e = windowed(&words);

        e.embed_batch(&[&fits], PromptName::Passage).await.unwrap();
        assert_eq!(
            words.tokenize_failures.get(),
            u32::MAX,
            "tokenizer was asked"
        );
        e.embed_batch(&[&reaches], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(
            words.tokenize_failures.get(),
            u32::MAX - 1,
            "tokenizer was not asked"
        );
    }

    /// Windowing must sit above the cache. A tokenizer failure sends a long
    /// passage whole once, which caches its truncated vector under the
    /// whole-text key; once the tokenizer is back, the same passage must be
    /// windowed again, not answered from that entry.
    #[tokio::test]
    async fn the_stack_windows_past_a_cached_truncated_vector() {
        let words = Words::new(1040);
        words.tokenize_failures.set(1);
        let stack = embedding_stack(
            Box::new(Shared(words.clone())),
            Box::new(Shared(words.clone())),
            None,
            Arc::new(Vitals::default()),
        );
        let long = text(3000);
        let whole = stack
            .embed_batch(&[&long], PromptName::Passage)
            .await
            .unwrap();
        assert_eq!(whole[0], words.vector(&long), "first pass is sent whole");

        let pooled = stack
            .embed_batch(&[&long], PromptName::Passage)
            .await
            .unwrap();
        assert_ne!(
            pooled[0], whole[0],
            "the truncated whole-text entry answered"
        );
        assert_eq!(words.calls.borrow().len(), 1 + MAX_WINDOWS);
    }

    #[test]
    fn windows_overlap_until_the_cap_then_spread() {
        let size = 1024;
        let overlap = size / OVERLAP_DIVISOR;
        // Just past one window: two, head and tail.
        assert_eq!(window_starts(1025, size), vec![0, 1]);
        // Within the cap: contiguous, every neighbour overlapping.
        let starts = window_starts(2500, size);
        assert_eq!(starts.first(), Some(&0));
        assert_eq!(starts.last().map(|s| s + size), Some(2500));
        assert!(starts.windows(2).all(|p| p[0] + size >= p[1] + overlap));
        // Past the cap: exactly MAX_WINDOWS, still head to tail.
        let starts = window_starts(1_000_000, size);
        assert_eq!(starts.len(), MAX_WINDOWS);
        assert_eq!(starts.first(), Some(&0));
        assert_eq!(starts.last().map(|s| s + size), Some(1_000_000));
    }
}
