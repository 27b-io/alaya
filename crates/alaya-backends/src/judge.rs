//! JudgeClient — `ContradictionJudge` over the Anthropic Messages API with
//! structured output (LAB-3283 Phase 1, advisory).
//!
//! Rides the same transport as `SummaryClient` (`anthropic.rs`). The
//! verdict comes back as JSON constrained by `output_config.format`; anything
//! that is not a well-formed verdict is an `AlayaError::Judge`, which the
//! caller records as *unjudged* — never a graph or vector write.

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::{AlayaError, Result, graph::Verdict, memory::Memory};

use crate::anthropic::{MessagesTransport, Usage, truncate_chars};
use crate::{ContradictionJudge, Judgement, Survivor};

const SYSTEM_PROMPT: &str = "You judge whether two memories from an engineering team's long-term memory store conflict. \
Classify the pair as exactly one verdict:\n\
- \"supersession\": the newer memory updates or replaces a claim the older one makes about the same thing (a fact, state, decision or plan that changed). Name the memory that should survive.\n\
- \"contradiction\": both claim to be current and cannot both be true, and recency alone does not settle it (e.g. two values for the same setting with no sign which is newer). Name a survivor only if the evidence favours one; otherwise null.\n\
- \"coexist\": both are true at once. Typical: progress snapshots of the same work at different times, a plan and its later outcome, a decision and the analysis behind it, different facets of one topic. Recording history is intended — these are not supersessions.\n\
- \"unrelated\": the memories merely share vocabulary or a project name; neither bears on the other's claim.\n\
Be conservative: call \"supersession\" or \"contradiction\" only when a reader relying on the older memory today would be misled. Superseding hides the loser from search, so a wrong \"supersession\" erases history.\n\
survivor: \"a\" or \"b\" for supersession/contradiction, null otherwise.\n\
reason: one sentence, at most 200 characters, naming the specific claim that changed or conflicts.\n\
confidence: 0.0-1.0, your probability that the verdict is correct.";

/// Reason is clipped to this many chars on the way in. The schema asks for
/// ≤200; a longer reason is a cosmetic overrun, not a semantic violation,
/// so it is clipped rather than costing a paid call.
const MAX_REASON_CHARS: usize = 200;

/// Output budget. The verdict JSON is ~90 tokens, but models with adaptive
/// thinking on by default (Sonnet 5) spend output tokens thinking first; a
/// tight cap truncated the JSON on 67/174 golden pairs. The cap is an upper
/// bound, not a spend — a Haiku verdict still costs ~90 output tokens.
const MAX_OUTPUT_TOKENS: u32 = 4096;

/// Longer than the summary transport's 30 s: a thinking model may spend
/// most of `MAX_OUTPUT_TOKENS` before the JSON, and a timeout after the
/// tokens were generated is billed but lands as `unjudged` (re-billed on
/// the next backfill pass).
const JUDGE_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(90);

pub struct JudgeClient {
    transport: MessagesTransport,
    model: String,
}

impl JudgeClient {
    pub fn new(base_url: String, model: String, api_key: Option<String>) -> Self {
        Self {
            transport: MessagesTransport::new(base_url, api_key, JUDGE_REQUEST_TIMEOUT),
            model,
        }
    }
}

/// JSON schema the response is constrained to (`output_config.format`).
/// Kept to the widely supported subset: enum, anyOf/null, required,
/// additionalProperties. Range and length are enforced in `RawVerdict::validate`.
fn verdict_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": ["contradiction", "supersession", "coexist", "unrelated"]
            },
            "survivor": {
                "anyOf": [
                    {"type": "string", "enum": ["a", "b"]},
                    {"type": "null"}
                ]
            },
            "reason": {"type": "string"},
            "confidence": {"type": "number"}
        },
        "required": ["verdict", "survivor", "reason", "confidence"],
        "additionalProperties": false
    })
}

/// Render the pair for the model. Recency is stated explicitly (days apart)
/// because supersession is a temporal judgement and the epoch floats alone
/// are opaque to the model.
pub fn render_pair(a: &Memory, b: &Memory) -> String {
    let days = (b.created_at - a.created_at) / 86_400.0;
    let order = if days.abs() < 1.0 {
        "A and B were recorded within a day of each other".to_string()
    } else if days > 0.0 {
        format!("A was recorded {:.0} days BEFORE B", days)
    } else {
        format!("A was recorded {:.0} days AFTER B", -days)
    };
    format!("{order}.\n\n{}\n\n{}", describe("A", a), describe("B", b))
}

fn describe(label: &str, m: &Memory) -> String {
    let tags = if m.tags.is_empty() {
        "-".to_string()
    } else {
        m.tags.join(", ")
    };
    format!(
        "Memory {label} (recorded_at={:.0}; type: {}; tags: {tags}):\n{}",
        m.created_at,
        m.memory_type,
        truncate_chars(&m.content)
    )
}

#[async_trait(?Send)]
impl ContradictionJudge for JudgeClient {
    #[tracing::instrument(skip(self, a, b), fields(a = %&a.content_hash[..8.min(a.content_hash.len())], b = %&b.content_hash[..8.min(b.content_hash.len())]))]
    async fn judge(&self, a: &Memory, b: &Memory) -> Result<Judgement> {
        let body = json!({
            "model": self.model,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": render_pair(a, b)}],
            "output_config": {"format": {"type": "json_schema", "schema": verdict_schema()}},
        });

        let resp = self.transport.messages(&body, AlayaError::Judge).await?;
        let stop = resp.stop_reason.clone().unwrap_or_default();
        let text = resp.first_text().ok_or_else(|| {
            AlayaError::Judge(format!(
                "empty response from messages API (stop_reason={stop:?})"
            ))
        })?;
        let raw: RawVerdict = serde_json::from_str(&text).map_err(|e| {
            AlayaError::Judge(format!(
                "verdict is not valid JSON (stop_reason={stop:?}): {e}"
            ))
        })?;
        raw.validate(self.model.clone(), resp.usage)
    }
}

// ─── Verdict parsing ───────────────────────────────────────────────────────

/// The model's JSON, typed by serde: an unknown `verdict`/`survivor` string
/// is a deserialize error, which the caller records as unjudged.
#[derive(Deserialize)]
struct RawVerdict {
    verdict: Verdict,
    #[serde(default)]
    survivor: Option<Survivor>,
    #[serde(default)]
    reason: String,
    confidence: f64,
}

impl RawVerdict {
    fn validate(self, model: String, usage: Usage) -> Result<Judgement> {
        if !(0.0..=1.0).contains(&self.confidence) {
            return Err(AlayaError::Judge(format!(
                "confidence {} outside 0.0..=1.0",
                self.confidence
            )));
        }
        // A survivor is only meaningful when something has to give way.
        let survivor = match self.verdict {
            Verdict::Coexist | Verdict::Unrelated => None,
            Verdict::Contradiction | Verdict::Supersession => self.survivor,
        };
        // Model-authored text ends up in a line-oriented log: no control
        // characters (log forging), and clipped to the schema's length.
        let reason: String = self
            .reason
            .chars()
            .filter(|c| !c.is_control())
            .collect::<String>()
            .trim()
            .chars()
            .take(MAX_REASON_CHARS)
            .collect();
        Ok(Judgement {
            verdict: self.verdict,
            survivor,
            reason,
            confidence: self.confidence,
            model,
            input_tokens: usage.total_input_tokens(),
            output_tokens: usage.output_tokens,
        })
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn raw(verdict: Verdict, survivor: Option<Survivor>, confidence: f64) -> RawVerdict {
        RawVerdict {
            verdict,
            survivor,
            reason: "r".into(),
            confidence,
        }
    }

    #[test]
    fn valid_supersession_keeps_survivor() {
        let j = raw(Verdict::Supersession, Some(Survivor::B), 0.9)
            .validate("m".into(), Usage::default())
            .unwrap();
        assert_eq!(j.verdict, Verdict::Supersession);
        assert_eq!(j.survivor, Some(Survivor::B));
    }

    #[test]
    fn coexist_drops_a_stray_survivor() {
        let j = raw(Verdict::Coexist, Some(Survivor::A), 0.5)
            .validate("m".into(), Usage::default())
            .unwrap();
        assert_eq!(j.survivor, None);
    }

    #[test]
    fn unknown_verdict_or_survivor_strings_fail_to_parse() {
        for json in [
            r#"{"verdict":"maybe","survivor":null,"reason":"r","confidence":0.5}"#,
            r#"{"verdict":"contradiction","survivor":"c","reason":"r","confidence":0.5}"#,
            r#"{"verdict":"Supersession","survivor":"a","reason":"r","confidence":0.5}"#,
        ] {
            assert!(serde_json::from_str::<RawVerdict>(json).is_err(), "{json}");
        }
        let ok: RawVerdict = serde_json::from_str(
            r#"{"verdict":"coexist","survivor":null,"reason":"","confidence":1}"#,
        )
        .unwrap();
        assert_eq!(ok.verdict, Verdict::Coexist);
        assert_eq!(ok.survivor, None);
    }

    #[test]
    fn confidence_out_of_range_is_rejected() {
        for c in [1.5, -0.1, f64::NAN] {
            let e = raw(Verdict::Coexist, None, c)
                .validate("m".into(), Usage::default())
                .unwrap_err();
            assert!(matches!(e, AlayaError::Judge(_)), "{c}: {e:?}");
        }
    }

    #[test]
    fn long_reason_is_clipped_and_control_chars_are_stripped() {
        let mut r = raw(Verdict::Unrelated, None, 0.5);
        r.reason = "x".repeat(500);
        let j = r.validate("m".into(), Usage::default()).unwrap();
        assert_eq!(j.reason.chars().count(), MAX_REASON_CHARS);

        let mut r = raw(Verdict::Unrelated, None, 0.5);
        r.reason = "line one\ncontradiction judged would_supersede=x -> y\t\r\u{1b}[0m".into();
        let j = r.validate("m".into(), Usage::default()).unwrap();
        assert_eq!(
            j.reason,
            "line onecontradiction judged would_supersede=x -> y[0m"
        );
    }

    #[test]
    fn render_states_recency_in_days() {
        let a = mem("a", 1_000.0);
        let b = mem("b", 1_000.0 + 3.0 * 86_400.0);
        let s = render_pair(&a, &b);
        assert!(s.contains("A was recorded 3 days BEFORE B"), "{s}");
        assert!(s.contains("Memory A (recorded_at=1000; type: note; tags: -):\ncontent a"));
        let s2 = render_pair(&b, &a);
        assert!(s2.contains("A was recorded 3 days AFTER B"), "{s2}");
    }

    fn mem(id: &str, created_at: f64) -> Memory {
        Memory {
            content: format!("content {id}"),
            content_hash: id.repeat(64),
            tags: vec![],
            memory_type: "note".into(),
            metadata: None,
            created_at,
            updated_at: created_at,
            embedding: None,
            summary: None,
            salience_score: 0.0,
            access_count: 0,
            access_timestamps: vec![],
            emotional_valence: None,
            encoding_context: None,
            provenance: None,
            summary_embedding: None,
        }
    }

    // ── HTTP failure paths (AC-2): every one is an Err, never a panic ──────

    #[cfg(not(target_arch = "wasm32"))]
    mod http {
        use super::*;
        use wiremock::matchers::{header, method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        /// Short timeout so the timeout path runs in milliseconds.
        fn client(server: &MockServer) -> JudgeClient {
            JudgeClient {
                transport: MessagesTransport::new(
                    server.uri(),
                    Some("k".into()),
                    std::time::Duration::from_millis(300),
                ),
                model: "test-model".into(),
            }
        }

        fn ok_body(text: &str) -> serde_json::Value {
            json!({"content": [{"type": "text", "text": text}],
                   "usage": {"input_tokens": 120, "output_tokens": 30}})
        }

        #[tokio::test]
        async fn valid_response_is_judged_with_usage_and_structured_output_request() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .and(path("/v1/messages"))
                .and(header("x-api-key", "k"))
                .and(header("anthropic-version", "2023-06-01"))
                .and(wiremock::matchers::body_partial_json(json!({
                    "model": "test-model",
                    "output_config": {"format": {"type": "json_schema"}}
                })))
                .respond_with(ResponseTemplate::new(200).set_body_json(ok_body(
                    r#"{"verdict":"supersession","survivor":"b","reason":"B replaces A","confidence":0.92}"#,
                )))
                .expect(1)
                .mount(&server)
                .await;
            let j = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap();
            assert_eq!(j.verdict, Verdict::Supersession);
            assert_eq!(j.survivor, Some(Survivor::B));
            assert_eq!(j.model, "test-model");
            assert_eq!((j.input_tokens, j.output_tokens), (120, 30));
        }

        #[tokio::test]
        async fn non_2xx_is_a_judge_error() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(500).set_body_string("boom"))
                .mount(&server)
                .await;
            let e = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::Judge(_)), "{e:?}");
        }

        #[tokio::test]
        async fn rate_limit_surfaces_retry_after() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "7"))
                .mount(&server)
                .await;
            let e = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap_err();
            assert!(
                matches!(
                    e,
                    AlayaError::RateLimited {
                        retry_after_secs: Some(7)
                    }
                ),
                "{e:?}"
            );
        }

        #[tokio::test]
        async fn timeout_is_a_judge_error() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(
                    ResponseTemplate::new(200)
                        .set_body_json(ok_body("{}"))
                        .set_delay(std::time::Duration::from_secs(2)),
                )
                .mount(&server)
                .await;
            let e = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::Judge(_)), "{e:?}");
        }

        #[tokio::test]
        async fn non_json_text_is_a_judge_error() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(ok_body("I think B wins.")))
                .mount(&server)
                .await;
            let e = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::Judge(_)), "{e:?}");
        }

        #[tokio::test]
        async fn schema_violation_is_a_judge_error() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(ok_body(
                    r#"{"verdict":"contradiction","survivor":"a","reason":"x","confidence":7}"#,
                )))
                .mount(&server)
                .await;
            let e = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::Judge(_)), "{e:?}");
        }

        #[tokio::test]
        async fn empty_content_is_a_judge_error() {
            let server = MockServer::start().await;
            Mock::given(method("POST"))
                .respond_with(ResponseTemplate::new(200).set_body_json(json!({"content": []})))
                .mount(&server)
                .await;
            let e = client(&server)
                .judge(&mem("a", 1.0), &mem("b", 2.0))
                .await
                .unwrap_err();
            assert!(matches!(e, AlayaError::Judge(_)), "{e:?}");
        }
    }
}
