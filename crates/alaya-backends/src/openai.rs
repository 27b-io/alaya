//! OpenAI-compatible chat-completions wire for `MessagesTransport`
//! (`POST /v1/chat/completions`): any endpoint that speaks it, e.g. a
//! LiteLLM proxy or vLLM. The HTTP client, timeouts and failure
//! classification are shared with the Anthropic wire in `anthropic.rs`.

use serde::Deserialize;
use serde_json::{Value, json};

use alaya_types::{AlayaError, Result};

use crate::anthropic::{Completion, Prompt, Usage};

/// `Authorization: Bearer <key>`, marked sensitive so it never reaches a
/// `Debug` print. The key must be a single line of visible ASCII: a space or
/// tab would still make a valid header value, but never a valid bearer.
pub(crate) fn headers(api_key: Option<String>) -> Result<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();
    if let Some(key) = api_key {
        if key.is_empty() || !key.bytes().all(|b| b.is_ascii_graphic()) {
            return Err(AlayaError::Config(
                "openai api key is not a single line of visible ASCII".into(),
            ));
        }
        let mut val =
            reqwest::header::HeaderValue::from_str(&format!("Bearer {key}")).map_err(|e| {
                AlayaError::Config(format!(
                    "openai api key is not valid HTTP header material: {e}"
                ))
            })?;
        val.set_sensitive(true);
        headers.insert(reqwest::header::AUTHORIZATION, val);
    }
    Ok(headers)
}

/// Chat-completions body. `max_completion_tokens`, never `max_tokens`, and
/// no `temperature`: reasoning models reject both. Structured output rides
/// `response_format` json_schema in strict mode, built from the same schema
/// the Anthropic wire sends.
pub(crate) fn body(p: &Prompt<'_>) -> Value {
    let mut body = json!({
        "model": p.model,
        "max_completion_tokens": p.max_tokens,
        "messages": [
            {"role": "system", "content": p.system},
            {"role": "user", "content": p.user},
        ],
    });
    if let Some((name, schema)) = &p.schema {
        body["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {"name": name, "strict": true, "schema": schema},
        });
    }
    body
}

#[derive(Deserialize)]
pub(crate) struct ChatResponse {
    #[serde(default)]
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<ChatUsage>,
}

#[derive(Deserialize)]
struct Choice {
    #[serde(default)]
    message: Option<ChatMessage>,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
}

/// `prompt_tokens` already includes any cached prompt tokens, so it maps to
/// the total input count on its own.
#[derive(Deserialize)]
struct ChatUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

impl ChatResponse {
    /// Read `choices[0]`. A content-filter stop or a refusal is `err(..)`:
    /// the same pair would be filtered or refused again, so retrying it only
    /// re-bills. Text is trimmed; an empty reply is `text: None`, for the
    /// role client to report as it does on the Anthropic wire.
    pub(crate) fn into_completion(self, err: fn(String) -> AlayaError) -> Result<Completion> {
        let usage = self
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                ..Usage::default()
            })
            .unwrap_or_default();
        let Some(choice) = self.choices.into_iter().next() else {
            return Ok(Completion {
                text: None,
                usage,
                stop_reason: None,
            });
        };
        let message = choice.message.unwrap_or(ChatMessage {
            content: None,
            refusal: None,
        });
        if choice.finish_reason.as_deref() == Some("content_filter") {
            return Err(err("chat completion stopped by the content filter".into()));
        }
        // The refusal text is the model's prose; it is not echoed into an
        // error that lands on an edge and in the log.
        if message.refusal.is_some() {
            return Err(err(format!(
                "model refused the request (finish_reason={:?})",
                choice.finish_reason.unwrap_or_default()
            )));
        }
        Ok(Completion {
            text: message
                .content
                .map(|c| c.trim().to_string())
                .filter(|c| !c.is_empty()),
            usage,
            stop_reason: choice.finish_reason,
        })
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> Result<Completion> {
        serde_json::from_str::<ChatResponse>(json)
            .unwrap()
            .into_completion(AlayaError::Judge)
    }

    #[test]
    fn reads_first_choice_and_maps_usage() {
        let c = parse(
            r#"{"id":"x","choices":[{"index":0,"message":{"role":"assistant","content":" hi ","refusal":null},"finish_reason":"stop"}],"usage":{"prompt_tokens":120,"completion_tokens":30,"total_tokens":150}}"#,
        )
        .unwrap();
        assert_eq!(c.text.as_deref(), Some("hi"));
        assert_eq!(c.usage.total_input_tokens(), 120);
        assert_eq!(c.usage.output_tokens, 30);
        assert_eq!(c.stop_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn missing_usage_reads_as_zero() {
        for json in [
            r#"{"choices":[{"message":{"content":"x"}}]}"#,
            r#"{"choices":[{"message":{"content":"x"}}],"usage":null}"#,
            r#"{"choices":[{"message":{"content":"x"}}],"usage":{}}"#,
        ] {
            let c = parse(json).unwrap();
            assert_eq!(
                (c.usage.total_input_tokens(), c.usage.output_tokens),
                (0, 0),
                "{json}"
            );
        }
    }

    #[test]
    fn empty_or_null_content_and_no_choices_are_no_text() {
        for json in [
            r#"{"choices":[]}"#,
            r#"{}"#,
            r#"{"choices":[{"message":{"content":null},"finish_reason":"length"}]}"#,
            r#"{"choices":[{"message":{"content":"  "}}]}"#,
            r#"{"choices":[{"finish_reason":"stop"}]}"#,
        ] {
            assert!(parse(json).unwrap().text.is_none(), "{json}");
        }
    }

    #[test]
    fn content_filter_and_refusal_are_deterministic() {
        for json in [
            r#"{"choices":[{"message":{"content":null},"finish_reason":"content_filter"}]}"#,
            r#"{"choices":[{"message":{"content":null,"refusal":"I can't help with that."},"finish_reason":"stop"}]}"#,
        ] {
            let e = parse(json).err().expect(json);
            assert!(matches!(e, AlayaError::Judge(_)), "{json}: {e:?}");
            assert!(!e.to_string().contains("can't help"), "echoed refusal: {e}");
        }
    }

    #[test]
    fn body_uses_completion_budget_and_strict_schema() {
        let p = Prompt {
            model: "m",
            system: "sys",
            user: "usr",
            max_tokens: 42,
            schema: Some(("verdict", json!({"type": "object"}))),
        };
        assert_eq!(
            body(&p),
            json!({
                "model": "m",
                "max_completion_tokens": 42,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "usr"},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "verdict", "strict": true, "schema": {"type": "object"}},
                },
            })
        );
        let plain = body(&Prompt { schema: None, ..p });
        assert!(plain.get("response_format").is_none());
        assert!(plain.get("max_tokens").is_none());
        assert!(plain.get("temperature").is_none());
    }

    #[test]
    fn key_must_be_one_line_of_visible_ascii_and_is_never_echoed() {
        for bad in ["sk-abc\n", "sk abc", "sk-\tabc", "sk-é", ""] {
            let e = headers(Some(bad.into())).err().expect(bad);
            assert!(matches!(e, AlayaError::Config(_)), "{bad:?}: {e:?}");
            assert!(!e.to_string().contains("abc"), "echoed key: {e}");
        }
        let h = headers(Some("sk-abc".into())).unwrap();
        let auth = h.get(reqwest::header::AUTHORIZATION).unwrap();
        assert!(auth.is_sensitive());
        assert_eq!(auth.to_str().unwrap(), "Bearer sk-abc");
        assert!(headers(None).unwrap().is_empty());
    }
}
