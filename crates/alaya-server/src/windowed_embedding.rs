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
//! so each one fits exactly. When the window size or a token layout cannot
//! be read, the text is embedded single-pass, truncated as before, and a
//! warning is logged.

use std::cell::Cell;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use alaya_backends::traits::{EmbeddingProvider, Tokenizer};
use alaya_types::memory::HealthStatus;
use alaya_types::search::PromptName;
use alaya_types::{AlayaError, Result};

/// Most windows one passage is embedded as, however long it is. Past this
/// the windows spread evenly from the head to the tail, leaving gaps between
/// them, so the work per store stays bounded. A store runs inline on the
/// worker loop, so every search sent to the same process waits for all of
/// it. On a CPU embedder three `WINDOW_TOKENS` windows cost about what one
/// 2048-token input did (~4.5 s), so the longest store holds searches no
/// longer than the longest store did before windowing.
pub const MAX_WINDOWS: usize = 3;
/// Tokens per window, at most. Smaller windows cost less per token on CPU
/// (attention grows with input length), and each one holds the embedder for
/// less time, so a query queued there behind a long store waits less.
const WINDOW_TOKENS: usize = 1024;
/// Neighbouring windows share `1 / OVERLAP_DIVISOR` of a window, so a
/// sentence cut at one window's edge appears whole in the next.
const OVERLAP_DIVISOR: usize = 8;
/// Tokens kept free in every input for what the embedder adds around the
/// text: the passage prompt prefix and the special tokens (7 in all for the
/// default model), plus re-tokenisation drift at a window's cut edges.
const RESERVED_TOKENS: usize = 16;
/// How long a read of the embedder's input window is trusted. Bounded so a
/// redeployed embedder with a different window is picked up without a
/// restart.
const MAX_INPUT_TTL: Duration = Duration::from_secs(60);

pub struct WindowedEmbedding {
    inner: Box<dyn EmbeddingProvider>,
    tokenizer: Box<dyn Tokenizer>,
    max_input: Cell<Option<(usize, Instant)>>,
}

impl WindowedEmbedding {
    pub fn new(inner: Box<dyn EmbeddingProvider>, tokenizer: Box<dyn Tokenizer>) -> Self {
        Self {
            inner,
            tokenizer,
            max_input: Cell::new(None),
        }
    }

    /// The embedder's input window in tokens, or `None` when it cannot be
    /// read (logged): passages then embed single-pass.
    async fn max_input_tokens(&self) -> Option<usize> {
        if let Some((n, at)) = self.max_input.get()
            && at.elapsed() < MAX_INPUT_TTL
        {
            return Some(n);
        }
        match self.tokenizer.max_input_tokens().await {
            Ok(n) => {
                self.max_input.set(Some((n, Instant::now())));
                Some(n)
            }
            Err(e) => {
                tracing::warn!("embedder input window unknown, long passages embed truncated: {e}");
                None
            }
        }
    }

    /// The windows `text` embeds as, or `None` when it embeds single-pass:
    /// it fits `max_input`, or its token layout could not be read (logged).
    async fn windows<'t>(&self, text: &'t str, max_input: usize) -> Option<Vec<&'t str>> {
        let budget = max_input.saturating_sub(RESERVED_TOKENS);
        // Every token spans at least one byte, so this text fits unread.
        if text.len() <= budget {
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
        if size == 0 {
            tracing::warn!(
                max_input,
                "embedder input window too small to split into, passage embeds truncated"
            );
            return None;
        }
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
    /// served, so a query that arrives mid-passage waits for one window, not
    /// the whole passage.
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
/// for `n > size`. Consecutive windows overlap by at least
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
        let Some(max_input) = self.max_input_tokens().await else {
            return self.inner.embed_batch(texts, prompt_name).await;
        };

        let plans =
            futures::future::join_all(texts.iter().map(|t| self.windows(t, max_input))).await;
        if plans.iter().all(Option::is_none) {
            return self.inner.embed_batch(texts, prompt_name).await;
        }

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

    /// Splits on spaces, one token per word, and truncates input past
    /// `max_input` words when embedding, as the real embedder does.
    struct Words {
        max_input: usize,
        calls: RefCell<Vec<Vec<String>>>,
        tokenize_fails: bool,
    }

    impl Words {
        fn new(max_input: usize) -> std::rc::Rc<Self> {
            std::rc::Rc::new(Self {
                max_input,
                calls: RefCell::default(),
                tokenize_fails: false,
            })
        }

        /// Each word adds 1 to a dimension picked by its hash.
        fn vector(&self, text: &str) -> Vec<f32> {
            let mut v = vec![0.0_f32; DIMS];
            for word in text
                .split(' ')
                .filter(|w| !w.is_empty())
                .take(self.max_input)
            {
                let h = word
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
        async fn max_input_tokens(&self) -> Result<usize> {
            Ok(self.0.max_input)
        }
        async fn token_spans(&self, text: &str) -> Result<Vec<(usize, usize)>> {
            if self.0.tokenize_fails {
                return Err(AlayaError::Embedding("down".into()));
            }
            let mut spans = Vec::new();
            let mut start = None;
            for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
                match (c == ' ', start) {
                    (false, None) => start = Some(i),
                    (true, Some(s)) => {
                        spans.push((s, i));
                        start = None;
                    }
                    _ => {}
                }
            }
            Ok(spans)
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
        s.split(' ').filter(|w| !w.is_empty()).count()
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

    /// The LAB-12564 pair: a long shared prefix, then two different tails.
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
            assert_eq!(token_count(w), WINDOW_TOKENS.min(1040 - RESERVED_TOKENS));
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
        let words = std::rc::Rc::new(Words {
            max_input: 1040,
            calls: RefCell::default(),
            tokenize_fails: true,
        });
        let e = windowed(&words);
        let long = text(3000);
        e.embed_batch(&[&long], PromptName::Passage).await.unwrap();
        assert_eq!(*words.calls.borrow(), vec![vec![long]]);
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
