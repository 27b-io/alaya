//! selecta read-only module — upstream client (LAB-6674).
//!
//! Read-only by design: the console RENDERS what selecta is doing and has no
//! path that changes it. The credential is a selecta read-only bearer, which
//! selecta accepts only on its `/readonly/mcp` mount, and that mount
//! registers the four read verbs and no verb that enqueues work — so the
//! guarantee holds at selecta, not just in this file. Approvals are GitHub PR
//! merges; the console links to the PR and never offers a button.
//!
//! Three reads, all server-side, all on selecta's one HTTP port:
//! - MCP `tools/call` on `/readonly/mcp`: one stateless JSON-RPC POST per
//!   read, no `initialize` handshake. Results are read from
//!   `structuredContent` — list tools wrap their rows as `{"result": [...]}`;
//!   `content[]` (one text block per row) is ignored;
//! - `GET /metrics`: selecta's control-plane gauges (queue depth per state,
//!   run budget, daily caps, lease);
//! - `GET /healthz`: selecta's own verdict on whether its heartbeat advances.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use serde::Deserialize;
use serde_json::{Value, json};

use crate::error::{AppError, MAX_SHOWN_ERROR_CHARS};
use crate::http::{self, join, json_body};
use crate::routes::clip;

pub const LIST_TASKS: &str = "selecta_list_tasks";
pub const GET_TASK: &str = "selecta_get_task";
pub const DOWNLOAD_HEALTH: &str = "selecta_download_health";

/// MCP-facing task states, in lifecycle order. `queue_depth` labels its
/// samples with the queue's internal names; `mcp_state` maps them onto these.
pub const STATES: [&str; 8] = [
    "queued",
    "running",
    "awaiting_approval",
    "approved",
    "executing",
    "done",
    "aborted_stale",
    "failed",
];

fn mcp_state(internal: &str) -> &str {
    match internal {
        "ready" => "queued",
        "active" => "running",
        other => other,
    }
}

/// selecta task ids are `uuid4().hex`. Checked before an id reaches a URL
/// this console renders or an upstream call it makes.
pub fn valid_task_id(id: &str) -> bool {
    id.len() == 32
        && id
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// One `selecta_list_tasks` row. Unknown fields are ignored; a missing
/// required one fails the whole list, loudly.
#[derive(Debug, Deserialize)]
pub struct TaskRow {
    pub task_id: String,
    pub state: String,
    pub tier: String,
    pub verb: String,
    /// Epoch seconds.
    pub enqueued_at: f64,
    pub pr_url: Option<String>,
    /// The first 200 chars of the scrubbed audit, `…`-terminated when cut.
    pub audit: Option<String>,
}

/// `selecta_get_task`.
#[derive(Debug, Deserialize)]
pub struct TaskView {
    pub task_id: String,
    pub state: String,
    pub tier: String,
    pub verb: String,
    pub result_path: Option<String>,
    /// The whole scrubbed audit.
    pub audit: Option<String>,
    pub pr_url: Option<String>,
}

/// `selecta_download_health`. A client selecta could not reach has its
/// signals absent and an alert naming it, so `signals` is a map, not a
/// struct.
#[derive(Debug, Deserialize)]
pub struct HealthSnapshot {
    pub signals: BTreeMap<String, f64>,
    pub alerts: Vec<String>,
}

/// The `/healthz` verdict.
pub struct Liveness {
    /// selecta's own judgement: the watchdog heartbeat advanced inside its
    /// deadline.
    pub advancing: bool,
    pub seq: Option<u64>,
    /// selecta's reason when it is not advancing.
    pub detail: Option<String>,
}

/// The control-plane gauges the pane renders.
pub struct Gauges {
    /// MCP state → rows. All time: selecta never deletes a queue row.
    pub queue_depth: BTreeMap<String, f64>,
    pub budget: Budget,
    /// cap → (used today, limit). Empty until selecta seeds its caps.
    pub day_caps: BTreeMap<String, (Option<f64>, Option<f64>)>,
    /// 0 when this replica does not hold the lease.
    pub lease_remaining_s: f64,
}

/// The per-run budget. selecta exports what it consumed, not its caps; a cap
/// appears only inside `reason`, once it is hit.
pub struct Budget {
    pub tokens: f64,
    pub wallclock_s: f64,
    pub toolcalls: f64,
    pub parked: bool,
    pub reason: String,
}

/// selecta's ledger reports a torn budget read as consumed = 2^62, which
/// parks it. A value this large is that sentinel, not a count.
pub const TORN_SENTINEL: f64 = 4_611_686_018_427_387_904.0;

#[derive(Clone)]
pub struct SelectaClient {
    base: url::Url,
    api_key: String,
    http: reqwest::Client,
}

impl SelectaClient {
    pub fn new(base: url::Url, api_key: String) -> Self {
        SelectaClient {
            base,
            api_key,
            http: http::client(Duration::from_secs(20)),
        }
    }

    /// One MCP `tools/call` against the read-only mount, returning the
    /// result's `structuredContent`. No trailing slash: Starlette answers
    /// `/readonly/mcp/` with a 307, which this client refuses to follow. No
    /// `Origin` header either — selecta 403s any Origin on this route.
    async fn call(&self, tool: &str, arguments: Value) -> Result<Value, AppError> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": { "name": tool, "arguments": arguments },
        });
        let req = self
            .http
            .post(join(&self.base, "/readonly/mcp"))
            .bearer_auth(&self.api_key)
            .header(reqwest::header::ACCEPT, "application/json")
            .json(&body);
        structured(tool, json_body(tool, req).await?)
    }

    /// Newest-first rows in `state`, at most `limit` (selecta takes 1..=500
    /// and refuses anything else; without a limit it returns every row ever
    /// queued).
    pub async fn list_tasks(&self, state: &str, limit: u32) -> Result<Vec<TaskRow>, AppError> {
        let sc = self
            .call(LIST_TASKS, json!({ "state": state, "limit": limit }))
            .await?;
        let rows: Vec<TaskRow> = rows_of(LIST_TASKS, sc)?;
        if let Some(bad) = rows.iter().find(|r| !valid_task_id(&r.task_id)) {
            return Err(unparseable(
                LIST_TASKS,
                &format!("malformed task_id {:?}", bad.task_id),
            ));
        }
        Ok(rows)
    }

    pub async fn get_task(&self, task_id: &str) -> Result<TaskView, AppError> {
        let sc = self.call(GET_TASK, json!({ "task_id": task_id })).await?;
        serde_json::from_value(sc).map_err(|e| unparseable(GET_TASK, &e.to_string()))
    }

    pub async fn download_health(&self) -> Result<HealthSnapshot, AppError> {
        let sc = self.call(DOWNLOAD_HEALTH, json!({})).await?;
        serde_json::from_value(sc).map_err(|e| unparseable(DOWNLOAD_HEALTH, &e.to_string()))
    }

    pub async fn gauges(&self) -> Result<Gauges, AppError> {
        const WHAT: &str = "selecta /metrics";
        let resp = self
            .http
            .get(join(&self.base, "/metrics"))
            .send()
            .await
            .map_err(|e| AppError::transport(WHAT, &e))?;
        let status = resp.status();
        let text = http::body_text(WHAT, resp)
            .await
            .map_err(|e| AppError::body(WHAT, e))?;
        if !status.is_success() {
            return Err(AppError::Upstream(format!("{WHAT} {status}")));
        }
        parse_gauges(&text).map_err(|e| AppError::Upstream(format!("{WHAT}: {e}")))
    }

    /// `/healthz` answers 200 when the heartbeat advances and 503 with a
    /// reason when it does not; both are answers, not failures.
    pub async fn liveness(&self) -> Result<Liveness, AppError> {
        const WHAT: &str = "selecta /healthz";
        let resp = self
            .http
            .get(join(&self.base, "/healthz"))
            .send()
            .await
            .map_err(|e| AppError::transport(WHAT, &e))?;
        let status = resp.status();
        let text = http::body_text(WHAT, resp)
            .await
            .map_err(|e| AppError::body(WHAT, e))?;
        parse_liveness(status.as_u16(), &text)
            .map_err(|e| AppError::Upstream(format!("{WHAT}: {e}")))
    }
}

fn unparseable(tool: &str, why: &str) -> AppError {
    AppError::Upstream(format!(
        "{tool}: unparseable payload ({})",
        clip(why, MAX_SHOWN_ERROR_CHARS)
    ))
}

/// The `structuredContent` of a JSON-RPC `tools/call` response. A tool that
/// raises (an unknown task, a refused argument, an unregistered tool) comes
/// back as HTTP 200 with `isError: true` and the message in `content[0]`; a
/// malformed request as a JSON-RPC `error`. Both name the tool.
fn structured(tool: &str, v: Value) -> Result<Value, AppError> {
    if let Some(err) = v.get("error") {
        let code = err.get("code").and_then(Value::as_i64).unwrap_or(0);
        let msg = err.get("message").and_then(Value::as_str).unwrap_or("");
        return Err(AppError::Upstream(format!(
            "{tool}: JSON-RPC error {code}: {}",
            clip(msg, MAX_SHOWN_ERROR_CHARS)
        )));
    }
    let Some(result) = v.get("result") else {
        return Err(unparseable(tool, "no result"));
    };
    if result.get("isError").and_then(Value::as_bool) == Some(true) {
        let msg = result
            .pointer("/content/0/text")
            .and_then(Value::as_str)
            .unwrap_or("no message");
        return Err(AppError::Upstream(format!(
            "{tool} failed: {}",
            clip(msg, MAX_SHOWN_ERROR_CHARS)
        )));
    }
    match result.get("structuredContent") {
        Some(sc @ Value::Object(_)) => Ok(sc.clone()),
        _ => Err(unparseable(tool, "no structuredContent object")),
    }
}

/// The rows of a list tool's `{"result": [...]}`. A missing or non-list
/// `result` is an error, never an empty list: an empty list reads as "all
/// quiet".
fn rows_of<T: serde::de::DeserializeOwned>(tool: &str, mut sc: Value) -> Result<Vec<T>, AppError> {
    match sc.get_mut("result").map(Value::take) {
        Some(rows @ Value::Array(_)) => {
            serde_json::from_value(rows).map_err(|e| unparseable(tool, &e.to_string()))
        }
        _ => Err(unparseable(tool, "structuredContent has no result list")),
    }
}

fn parse_liveness(status: u16, body: &str) -> Result<Liveness, String> {
    if status != 200 && status != 503 {
        return Err(format!("unexpected status {status}"));
    }
    let v: Value = serde_json::from_str(body).map_err(|_| "non-JSON body".to_string())?;
    let detail = v
        .get("detail")
        .and_then(Value::as_str)
        .map(|d| clip(d, MAX_SHOWN_ERROR_CHARS));
    let seq = v.get("seq").and_then(Value::as_u64);
    match (status, v.get("status").and_then(Value::as_str)) {
        (200, Some("ok")) => Ok(Liveness {
            advancing: true,
            seq,
            detail: None,
        }),
        (503, Some("stale")) => Ok(Liveness {
            advancing: false,
            seq,
            detail,
        }),
        _ => Err(format!("unexpected answer for status {status}")),
    }
}

/// One exposition sample: `name{k="v",...} value`.
struct Sample {
    name: String,
    labels: Vec<(String, String)>,
    value: f64,
}

impl Sample {
    fn label(&self, key: &str) -> Option<&str> {
        self.labels
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Parse one exposition line. Label values unescape `\\`, `\"` and `\n`,
/// the three escapes the format defines.
fn parse_sample(line: &str) -> Option<Sample> {
    let (name, mut rest) = line.split_at(line.find(['{', ' '])?);
    let mut labels = Vec::new();
    if let Some(body) = rest.strip_prefix('{') {
        rest = body;
        loop {
            if let Some(after) = rest.strip_prefix('}') {
                rest = after;
                break;
            }
            let (key, quoted) = rest.split_once("=\"")?;
            let mut value = String::new();
            let mut chars = quoted.char_indices();
            let close = loop {
                match chars.next()? {
                    (_, '\\') => match chars.next()?.1 {
                        'n' => value.push('\n'),
                        c => value.push(c),
                    },
                    (i, '"') => break i,
                    (_, c) => value.push(c),
                }
            };
            labels.push((key.to_string(), value));
            rest = &quoted[close + 1..];
            rest = rest.strip_prefix(',').unwrap_or(rest);
        }
    }
    let value = rest.split_whitespace().next()?.parse::<f64>().ok()?;
    Some(Sample {
        name: name.to_string(),
        labels,
        value,
    })
}

fn parse_gauges(text: &str) -> Result<Gauges, String> {
    let mut declared: BTreeSet<&str> = BTreeSet::new();
    let mut samples: Vec<Sample> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(comment) = line.strip_prefix('#') {
            let mut words = comment.split_whitespace();
            if words.next() == Some("TYPE")
                && let Some(name) = words.next()
            {
                declared.insert(name);
            }
            continue;
        }
        samples.push(parse_sample(line).ok_or_else(|| format!("unparseable line {}", n + 1))?);
    }
    let scalar = |name: &str| -> Result<f64, String> {
        samples
            .iter()
            .find(|s| s.name == name)
            .map(|s| s.value)
            .ok_or_else(|| format!("missing {name}"))
    };
    let parked = samples
        .iter()
        .find(|s| s.name == "budget_parked")
        .ok_or("missing budget_parked")?;
    // Families whose zero rows are simply omitted must at least be declared:
    // a renamed metric must not read as an empty queue.
    for family in ["queue_depth", "day_cap_used", "day_cap_limit"] {
        if !declared.contains(family) {
            return Err(format!("missing {family}"));
        }
    }
    let mut queue_depth = BTreeMap::new();
    let mut day_caps: BTreeMap<String, (Option<f64>, Option<f64>)> = BTreeMap::new();
    for s in &samples {
        match (s.name.as_str(), s.label("state"), s.label("cap")) {
            ("queue_depth", Some(state), _) => {
                *queue_depth
                    .entry(mcp_state(state).to_string())
                    .or_insert(0.0) += s.value;
            }
            ("day_cap_used", _, Some(cap)) => {
                day_caps.entry(cap.to_string()).or_default().0 = Some(s.value)
            }
            ("day_cap_limit", _, Some(cap)) => {
                day_caps.entry(cap.to_string()).or_default().1 = Some(s.value)
            }
            _ => {}
        }
    }
    Ok(Gauges {
        queue_depth,
        budget: Budget {
            tokens: scalar("budget_consumed_tokens")?,
            wallclock_s: scalar("budget_consumed_wallclock")?,
            toolcalls: scalar("budget_consumed_toolcalls")?,
            parked: parked.value != 0.0,
            reason: parked.label("reason").unwrap_or("").to_string(),
        },
        day_caps,
        lease_remaining_s: scalar("lease_remaining_s")?,
    })
}

/// A GitHub pull-request URL, or `None`. Only a link that parses as
/// `https://github.com/<owner>/<repo>/pull/<n>` is ever rendered as a link:
/// `pr_url` is upstream text, and an `href` is the one place escaping alone
/// does not make it safe.
pub fn github_pr(url: &str) -> Option<String> {
    let u: url::Url = url.parse().ok()?;
    if u.scheme() != "https"
        || u.host_str() != Some("github.com")
        || u.port().is_some()
        || !u.username().is_empty()
        || u.password().is_some()
        || u.query().is_some()
        || u.fragment().is_some()
    {
        return None;
    }
    let segs: Vec<&str> = u.path_segments()?.collect();
    let name_ok = |s: &str| {
        !s.is_empty()
            && s.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
    };
    match segs.as_slice() {
        [owner, repo, "pull", n]
            if name_ok(owner)
                && name_ok(repo)
                && !n.is_empty()
                && n.bytes().all(|b| b.is_ascii_digit()) =>
        {
            Some(u.to_string())
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A live-shaped `/metrics` body: selecta's exact renderer output, with a
    /// parked reason that exercises every label escape.
    const METRICS: &str = r#"# TYPE budget_consumed_tokens gauge
budget_consumed_tokens 1834201
# TYPE budget_consumed_wallclock gauge
budget_consumed_wallclock 3600.0
# TYPE budget_consumed_toolcalls gauge
budget_consumed_toolcalls 212
# TYPE budget_parked gauge
budget_parked{reason="wallclock cap reached (3600.0/3600.0) \"q\" a\\b\nnext"} 1
# TYPE watchdog_heartbeat_seq gauge
watchdog_heartbeat_seq 9124
# TYPE queue_depth gauge
queue_depth{state="active"} 1
queue_depth{state="awaiting_approval"} 2
queue_depth{state="done"} 1301
queue_depth{state="failed"} 412
queue_depth{state="ready"} 3
# TYPE velocity_tier gauge
velocity_tier{tier="full"} 1
# TYPE day_cap_used gauge
day_cap_used{cap="adds"} 12
day_cap_used{cap="grabs"} 40
# TYPE day_cap_limit gauge
day_cap_limit{cap="adds"} 40
day_cap_limit{cap="grabs"} 120
# TYPE lease_remaining_s gauge
lease_remaining_s 41.5
"#;

    #[test]
    fn gauges_parse_selectas_exposition() {
        let g = parse_gauges(METRICS).unwrap();
        // Internal queue names map onto the MCP state names.
        assert_eq!(g.queue_depth["queued"], 3.0);
        assert_eq!(g.queue_depth["running"], 1.0);
        assert_eq!(g.queue_depth["failed"], 412.0);
        assert!(!g.queue_depth.contains_key("ready"));
        assert_eq!(g.budget.tokens, 1_834_201.0);
        assert_eq!(g.budget.wallclock_s, 3600.0);
        assert_eq!(g.budget.toolcalls, 212.0);
        assert!(g.budget.parked);
        assert_eq!(
            g.budget.reason,
            "wallclock cap reached (3600.0/3600.0) \"q\" a\\b\nnext"
        );
        assert_eq!(g.day_caps["adds"], (Some(12.0), Some(40.0)));
        assert_eq!(g.day_caps["grabs"], (Some(40.0), Some(120.0)));
        assert_eq!(g.lease_remaining_s, 41.5);
    }

    #[test]
    fn gauges_unparked_budget_has_an_empty_reason() {
        let text = METRICS.replace(
            r#"budget_parked{reason="wallclock cap reached (3600.0/3600.0) \"q\" a\\b\nnext"} 1"#,
            r#"budget_parked{reason=""} 0"#,
        );
        let g = parse_gauges(&text).unwrap();
        assert!(!g.budget.parked);
        assert_eq!(g.budget.reason, "");
    }

    /// Fail loud: a gauge that disappears (a rename upstream) is an error,
    /// never a zero or an empty table. A family whose zero rows selecta
    /// omits must still be declared.
    #[test]
    fn gauges_refuse_a_missing_metric() {
        for (gone, needle) in [
            ("lease_remaining_s 41.5\n", "missing lease_remaining_s"),
            (
                "budget_consumed_tokens 1834201\n",
                "missing budget_consumed_tokens",
            ),
            ("# TYPE queue_depth gauge\n", "missing queue_depth"),
            ("# TYPE day_cap_limit gauge\n", "missing day_cap_limit"),
        ] {
            let err = parse_gauges(&METRICS.replace(gone, "")).err().unwrap();
            assert_eq!(err, needle);
        }
        let parked_line = METRICS
            .lines()
            .find(|l| l.starts_with("budget_parked"))
            .unwrap();
        let err = parse_gauges(&METRICS.replace(parked_line, ""))
            .err()
            .unwrap();
        assert_eq!(err, "missing budget_parked");
        let err = parse_gauges(&format!("{METRICS}garbage line\n"))
            .err()
            .unwrap();
        assert!(err.starts_with("unparseable line"), "{err}");
    }

    /// An empty queue is declared with no samples: every state is zero, and
    /// that is a real answer, not a parse failure.
    #[test]
    fn gauges_accept_an_empty_queue_and_unseeded_caps() {
        let text: String = METRICS
            .lines()
            .filter(|l| !l.starts_with("queue_depth{") && !l.starts_with("day_cap_"))
            .map(|l| format!("{l}\n"))
            .collect();
        let g = parse_gauges(&text).unwrap();
        assert!(g.queue_depth.is_empty());
        assert!(g.day_caps.is_empty());
    }

    #[test]
    fn sample_parser_handles_multiple_labels_and_rejects_junk() {
        let s = parse_sample(r#"m{a="1",b="x,y}z"} 2.5"#).unwrap();
        assert_eq!(s.name, "m");
        assert_eq!(s.label("a"), Some("1"));
        assert_eq!(s.label("b"), Some("x,y}z"));
        assert_eq!(s.value, 2.5);
        assert!(parse_sample(r#"m{a="unterminated} 1"#).is_none());
        assert!(parse_sample("m notanumber").is_none());
        assert!(parse_sample("lonely").is_none());
    }

    fn rpc_result(result: Value) -> Value {
        json!({"jsonrpc": "2.0", "id": 1, "result": result})
    }

    #[test]
    fn structured_returns_the_structured_content() {
        let sc = json!({"result": []});
        let v = rpc_result(json!({"content": [], "structuredContent": sc, "isError": false}));
        assert_eq!(structured(LIST_TASKS, v).ok(), Some(json!({"result": []})));
    }

    /// Every failure names the tool. A tool error arrives as HTTP 200 with
    /// `isError: true` — the shape selecta returns for an unknown task and
    /// for a verb the read-only mount does not register.
    #[test]
    fn structured_errors_name_the_tool() {
        let detail = |r: Result<Value, AppError>| r.err().unwrap().detail().to_string();
        let tool_err = rpc_result(json!({
            "content": [{"type": "text", "text": "Unknown tool: selecta_hunt_artist"}],
            "isError": true
        }));
        assert_eq!(
            detail(structured("selecta_hunt_artist", tool_err)),
            "selecta_hunt_artist failed: Unknown tool: selecta_hunt_artist"
        );
        let rpc_err = json!({"jsonrpc": "2.0", "id": 1,
            "error": {"code": -32602, "message": "Invalid request parameters", "data": ""}});
        assert_eq!(
            detail(structured(GET_TASK, rpc_err)),
            "selecta_get_task: JSON-RPC error -32602: Invalid request parameters"
        );
        assert_eq!(
            detail(structured(GET_TASK, json!({"jsonrpc": "2.0", "id": 1}))),
            "selecta_get_task: unparseable payload (no result)"
        );
        // A result with only `content[]` is not parsed from the text blocks.
        let no_sc =
            rpc_result(json!({"content": [{"type": "text", "text": "{}"}], "isError": false}));
        assert_eq!(
            detail(structured(GET_TASK, no_sc)),
            "selecta_get_task: unparseable payload (no structuredContent object)"
        );
        // Upstream text is clipped before it reaches a page.
        let long = rpc_result(
            json!({"content": [{"type": "text", "text": "x".repeat(500)}], "isError": true}),
        );
        assert!(detail(structured(GET_TASK, long)).ends_with('…'));
    }

    fn row(id: &str) -> Value {
        json!({"task_id": id, "state": "failed", "tier": "auto", "verb": "selecta_discover",
               "enqueued_at": 1_790_000_000.25, "pr_url": null, "audit": "boom"})
    }

    #[test]
    fn rows_parse_and_never_degrade_to_an_empty_list() {
        let id = "a".repeat(32);
        let rows: Vec<TaskRow> = rows_of(LIST_TASKS, json!({"result": [row(&id)]}))
            .ok()
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].enqueued_at, 1_790_000_000.25);
        assert_eq!(rows[0].audit.as_deref(), Some("boom"));
        let detail = |r: Result<Vec<TaskRow>, AppError>| r.err().unwrap().detail().to_string();
        assert_eq!(
            detail(rows_of(LIST_TASKS, json!({"rows": []}))),
            "selecta_list_tasks: unparseable payload (structuredContent has no result list)"
        );
        assert_eq!(
            detail(rows_of(LIST_TASKS, json!({"result": {}}))),
            "selecta_list_tasks: unparseable payload (structuredContent has no result list)"
        );
        let mut bad = row(&id);
        bad.as_object_mut().unwrap().remove("verb");
        let err = detail(rows_of(LIST_TASKS, json!({"result": [bad]})));
        assert!(err.contains("missing field `verb`"), "{err}");
    }

    #[test]
    fn task_ids_are_uuid4_hex() {
        assert!(valid_task_id("0123456789abcdef0123456789abcdef"));
        assert!(!valid_task_id("0123456789ABCDEF0123456789abcdef"));
        assert!(!valid_task_id("0123456789abcdef0123456789abcde"));
        assert!(!valid_task_id("0123456789abcdef-123456789abcdef"));
        assert!(!valid_task_id("../../0123456789abcdef0123456789"));
    }

    #[test]
    fn liveness_reads_both_answers_and_refuses_others() {
        let ok = parse_liveness(200, r#"{"status":"ok","seq":42}"#).unwrap();
        assert!(ok.advancing && ok.seq == Some(42) && ok.detail.is_none());
        let stale = parse_liveness(
            503,
            r#"{"status":"stale","detail":"heartbeat not advancing","seq":7}"#,
        )
        .unwrap();
        assert!(!stale.advancing);
        assert_eq!(stale.detail.as_deref(), Some("heartbeat not advancing"));
        let torn =
            parse_liveness(503, r#"{"status":"stale","detail":"heartbeat unreadable"}"#).unwrap();
        assert!(!torn.advancing && torn.seq.is_none());
        assert!(parse_liveness(500, "oops").is_err());
        assert!(parse_liveness(200, "<html>").is_err());
        // A 503 that does not say "stale" is not a heartbeat verdict.
        assert!(parse_liveness(503, r#"{"status":"ok"}"#).is_err());
    }

    #[test]
    fn only_github_pull_request_urls_become_links() {
        let ok = "https://github.com/acme/agent/pull/7";
        assert_eq!(github_pr(ok).as_deref(), Some(ok));
        for bad in [
            "javascript:alert(1)",
            "http://github.com/o/r/pull/7",
            "https://github.com.evil.test/o/r/pull/7",
            "https://evil.test/o/r/pull/7",
            "https://user@github.com/o/r/pull/7",
            "https://github.com:8443/o/r/pull/7",
            "https://github.com/o/r/pull/7?x=1",
            "https://github.com/o/r/pull/7#x",
            "https://github.com/o/r/issues/7",
            "https://github.com/o/r/pull/7x",
            "https://github.com/o/r/pull/",
            "https://github.com/o/r/pull/7/files",
            "not a url",
        ] {
            assert_eq!(github_pr(bad), None, "{bad}");
        }
    }
}
