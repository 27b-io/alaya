//! FalkorDB GRAPH.QUERY RESP result-set parser.
//!
//! `GRAPH.QUERY` returns a 3-element RESP array:
//!   [0] column headers  — `[[col_type_int, col_name_str], …]`
//!   [1] result set      — `[[cell, …], …]`
//!   [2] stats           — `["Nodes created: 1", …]`
//!
//! Write-only queries may return only 1 or 2 elements (no result rows).

use std::collections::HashMap;

use redis::Value;

/// Parsed result from a FalkorDB `GRAPH.QUERY` command.
#[derive(Debug, Clone)]
pub struct FalkorResult {
    /// Column names extracted from the header row.
    pub columns: Vec<String>,
    /// Result rows — each cell is a JSON-compatible value.
    pub result_set: Vec<Vec<serde_json::Value>>,
    /// Execution stats keyed by stat name (e.g. `"Nodes created"` → `"1"`).
    pub stats: HashMap<String, String>,
}

impl FalkorResult {
    /// Parse a `redis::Value` returned by `GRAPH.QUERY` into a `FalkorResult`.
    pub fn parse(value: &Value) -> Result<Self, String> {
        let top = as_array(value).ok_or_else(|| format!("expected top-level array, got {value:?}"))?;

        match top.len() {
            // ── write-only: just stats ──────────────────────────────────────
            1 => Ok(Self {
                columns: vec![],
                result_set: vec![],
                stats: parse_stats(&top[0])?,
            }),

            // ── two-element response ────────────────────────────────────────
            // Either [headers, stats] (empty result set) or [result_set, stats].
            // Distinguish by whether element 0 looks like a header list.
            2 => {
                let stats = parse_stats(&top[1])?;
                if looks_like_headers(&top[0]) {
                    let columns = parse_columns(&top[0])?;
                    Ok(Self { columns, result_set: vec![], stats })
                } else {
                    // No headers supplied — treat as anonymous result set.
                    let result_set = parse_result_set(&top[0], 0)?;
                    let col_count = result_set.first().map_or(0, |r| r.len());
                    let columns = (0..col_count).map(|i| format!("col{i}")).collect();
                    Ok(Self { columns, result_set, stats })
                }
            }

            // ── full response: [headers, result_set, stats] ─────────────────
            3 => {
                let columns = parse_columns(&top[0])?;
                let result_set = parse_result_set(&top[1], columns.len())?;
                let stats = parse_stats(&top[2])?;
                Ok(Self { columns, result_set, stats })
            }

            n => Err(format!("unexpected top-level array length {n}")),
        }
    }

    /// Returns `true` when there are no result rows.
    pub fn is_empty(&self) -> bool {
        self.result_set.is_empty()
    }

    /// Returns the first cell of the first row as `i64` — useful for `COUNT` queries.
    pub fn count(&self) -> Option<i64> {
        self.result_set.first()?.first()?.as_i64()
    }
}

// ─── helpers ──────────────────────────────────────────────────────────────────

fn as_array(v: &Value) -> Option<&Vec<Value>> {
    match v {
        Value::Array(a) => Some(a),
        _ => None,
    }
}

/// Heuristic: element 0 is a headers list when it is an array whose first item
/// is itself a 2-element array `[int, name_bytes]`.
fn looks_like_headers(v: &Value) -> bool {
    let Some(outer) = as_array(v) else { return false };
    let Some(first) = outer.first() else {
        // Empty array is ambiguous; treat as headers (empty column list).
        return true;
    };
    let Some(inner) = as_array(first) else { return false };
    if inner.len() != 2 {
        return false;
    }
    // First element should be a column-type integer.
    matches!(inner[0], Value::Int(_))
}

/// Extract column names from the headers element.
///
/// Each header is `[col_type_int, col_name_bulk_string]`.
fn parse_columns(v: &Value) -> Result<Vec<String>, String> {
    let headers = as_array(v)
        .ok_or_else(|| format!("headers must be an array, got {v:?}"))?;

    headers
        .iter()
        .map(|h| {
            let inner = as_array(h)
                .ok_or_else(|| format!("each header must be an array, got {h:?}"))?;
            if inner.len() < 2 {
                return Err(format!("header must have at least 2 elements, got {inner:?}"));
            }
            bulk_string_to_string(&inner[1])
                .ok_or_else(|| format!("column name must be a string, got {:?}", inner[1]))
        })
        .collect()
}

/// Convert the result-set element into rows of `serde_json::Value` cells.
fn parse_result_set(v: &Value, col_count: usize) -> Result<Vec<Vec<serde_json::Value>>, String> {
    let rows = as_array(v)
        .ok_or_else(|| format!("result set must be an array, got {v:?}"))?;

    rows.iter()
        .enumerate()
        .map(|(row_idx, row)| {
            let cells = as_array(row)
                .ok_or_else(|| format!("row {row_idx} must be an array, got {row:?}"))?;
            if col_count > 0 && cells.len() != col_count {
                return Err(format!(
                    "row {row_idx} has {} cells, expected {col_count}",
                    cells.len()
                ));
            }
            cells.iter().map(resp_cell_to_json).collect()
        })
        .collect()
}

/// Convert a single RESP cell value to a `serde_json::Value`.
///
/// FalkorDB quirk: floats are returned as bulk strings (e.g. `b"3.14"`).  We
/// try a float parse first; only fall back to string when that fails.
fn resp_cell_to_json(v: &Value) -> Result<serde_json::Value, String> {
    match v {
        Value::Nil => Ok(serde_json::Value::Null),

        Value::Int(i) => Ok(serde_json::json!(*i)),

        Value::Boolean(b) => Ok(serde_json::Value::Bool(*b)),

        Value::Double(f) => {
            let n = serde_json::Number::from_f64(*f)
                .ok_or_else(|| format!("non-finite f64: {f}"))?;
            Ok(serde_json::Value::Number(n))
        }

        Value::BulkString(bytes) => {
            let s = std::str::from_utf8(bytes)
                .map_err(|e| format!("non-UTF-8 bulk string: {e}"))?;
            // Try float first (FalkorDB sends floats as bulk strings).
            if let Ok(f) = s.parse::<f64>()
                && let Some(n) = serde_json::Number::from_f64(f)
            {
                return Ok(serde_json::Value::Number(n));
            }
            Ok(serde_json::Value::String(s.to_owned()))
        }

        Value::SimpleString(s) => Ok(serde_json::Value::String(s.clone())),

        Value::VerbatimString { text, .. } => Ok(serde_json::Value::String(text.clone())),

        other => Err(format!("unsupported RESP value in result cell: {other:?}")),
    }
}

/// Parse a stats element — an array of `"Key: value"` strings — into a map.
fn parse_stats(v: &Value) -> Result<HashMap<String, String>, String> {
    let items = as_array(v)
        .ok_or_else(|| format!("stats must be an array, got {v:?}"))?;

    let mut map = HashMap::new();
    for item in items {
        if let Some(s) = bulk_string_to_string(item)
            && let Some((k, val)) = s.split_once(": ")
        {
            map.insert(k.to_owned(), val.to_owned());
            // Lines without ": " are silently ignored (e.g. empty entries).
        }
    }
    Ok(map)
}

/// Try to decode a `BulkString` or `SimpleString` as UTF-8.
fn bulk_string_to_string(v: &Value) -> Option<String> {
    match v {
        Value::BulkString(b) => std::str::from_utf8(b).ok().map(str::to_owned),
        Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    fn header(col_type: i64, name: &str) -> Value {
        Value::Array(vec![Value::Int(col_type), bulk(name)])
    }

    fn stat(s: &str) -> Value {
        bulk(s)
    }

    // ── 1. Full result with data ──────────────────────────────────────────────

    #[test]
    fn full_result_with_data() {
        let response = Value::Array(vec![
            // Headers
            Value::Array(vec![
                header(1, "hash"),
                header(1, "weight"),
            ]),
            // Result set
            Value::Array(vec![Value::Array(vec![
                bulk("abc"),
                bulk("0.5"),
            ])]),
            // Stats
            Value::Array(vec![
                stat("Cached execution: 1"),
                stat("Query internal execution time: 0.5 milliseconds"),
            ]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.columns, vec!["hash", "weight"]);
        assert_eq!(r.result_set.len(), 1);
        assert_eq!(r.result_set[0][0], serde_json::json!("abc"));
        assert_eq!(r.result_set[0][1], serde_json::json!(0.5));
        assert!(!r.is_empty());
    }

    // ── 2. Write-only / empty result ─────────────────────────────────────────

    #[test]
    fn write_only_single_element() {
        let response = Value::Array(vec![Value::Array(vec![
            stat("Nodes created: 2"),
            stat("Labels added: 1"),
        ])]);

        let r = FalkorResult::parse(&response).unwrap();
        assert!(r.columns.is_empty());
        assert!(r.is_empty());
        assert_eq!(r.stats.get("Nodes created").map(String::as_str), Some("2"));
        assert_eq!(r.stats.get("Labels added").map(String::as_str), Some("1"));
    }

    #[test]
    fn two_element_headers_and_stats() {
        let response = Value::Array(vec![
            // Headers (no rows)
            Value::Array(vec![header(1, "content_hash")]),
            // Stats
            Value::Array(vec![stat("Nodes deleted: 3")]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.columns, vec!["content_hash"]);
        assert!(r.is_empty());
        assert_eq!(r.stats.get("Nodes deleted").map(String::as_str), Some("3"));
    }

    // ── 3. Count query ────────────────────────────────────────────────────────

    #[test]
    fn count_query() {
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "n")]),
            Value::Array(vec![Value::Array(vec![Value::Int(42)])]),
            Value::Array(vec![stat("Cached execution: 1")]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.count(), Some(42));
    }

    // ── 4. Float values as bulk strings ──────────────────────────────────────

    #[test]
    fn float_bulk_string() {
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "w")]),
            Value::Array(vec![Value::Array(vec![bulk("3.14")])]),
            Value::Array(vec![stat("Cached execution: 1")]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        // JSON numbers from f64 lose precision; compare as f64
        let v = r.result_set[0][0].as_f64().unwrap();
        assert!((v - 3.14).abs() < 1e-10);
    }

    // ── 5. Null values ────────────────────────────────────────────────────────

    #[test]
    fn null_cells() {
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "a"), header(1, "b")]),
            Value::Array(vec![Value::Array(vec![Value::Nil, bulk("hello")])]),
            Value::Array(vec![stat("Cached execution: 0")]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.result_set[0][0], serde_json::Value::Null);
        assert_eq!(r.result_set[0][1], serde_json::json!("hello"));
    }

    // ── 6. Stats parsing ─────────────────────────────────────────────────────

    #[test]
    fn stats_key_value_extraction() {
        let response = Value::Array(vec![
            Value::Array(vec![]),
            Value::Array(vec![]),
            Value::Array(vec![
                stat("Nodes created: 0"),
                stat("Relationships created: 5"),
                stat("Cached execution: 1"),
                stat("Query internal execution time: 1.234 milliseconds"),
            ]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.stats["Nodes created"], "0");
        assert_eq!(r.stats["Relationships created"], "5");
        assert_eq!(r.stats["Cached execution"], "1");
        assert_eq!(r.stats["Query internal execution time"], "1.234 milliseconds");
    }

    // ── 7. Integer cell ───────────────────────────────────────────────────────

    #[test]
    fn integer_cell() {
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "id")]),
            Value::Array(vec![Value::Array(vec![Value::Int(-7)])]),
            Value::Array(vec![]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.result_set[0][0], serde_json::json!(-7_i64));
    }

    // ── 8. Boolean cell ───────────────────────────────────────────────────────

    #[test]
    fn boolean_cell() {
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "flag")]),
            Value::Array(vec![Value::Array(vec![Value::Boolean(true)])]),
            Value::Array(vec![]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.result_set[0][0], serde_json::json!(true));
    }

    // ── 9. Multiple rows ──────────────────────────────────────────────────────

    #[test]
    fn multiple_rows() {
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "x")]),
            Value::Array(vec![
                Value::Array(vec![Value::Int(1)]),
                Value::Array(vec![Value::Int(2)]),
                Value::Array(vec![Value::Int(3)]),
            ]),
            Value::Array(vec![stat("Cached execution: 1")]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.result_set.len(), 3);
        assert_eq!(r.count(), Some(1));
    }

    // ── 10. Error: non-array top-level ────────────────────────────────────────

    #[test]
    fn error_on_non_array() {
        let result = FalkorResult::parse(&Value::Int(42));
        assert!(result.is_err());
    }

    // ── 11. String that looks like a number (integer string) ─────────────────

    #[test]
    fn integer_string_parses_as_number() {
        // "42" as a bulk string should parse as the number 42, not the string "42".
        let response = Value::Array(vec![
            Value::Array(vec![header(1, "n")]),
            Value::Array(vec![Value::Array(vec![bulk("42")])]),
            Value::Array(vec![]),
        ]);

        let r = FalkorResult::parse(&response).unwrap();
        assert_eq!(r.result_set[0][0].as_f64(), Some(42.0));
    }
}
