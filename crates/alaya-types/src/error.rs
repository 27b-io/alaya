use thiserror::Error;

/// Top-level error type for Alaya.
/// Maps to JSON-RPC error codes in the MCP transport layer.
#[derive(Debug, Error)]
pub enum AlayaError {
    #[error("vector storage error: {0}")]
    Storage(String),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("graph error: {0}")]
    Graph(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("summary generation error: {0}")]
    Summary(String),

    #[error("rerank error: {0}")]
    Rerank(String),

    #[error("contradiction judge error: {0}")]
    Judge(String),

    /// Upstream returned 429. `retry_after_secs` is the server's hint, when
    /// it sent one; callers that retry own the backoff.
    #[error("upstream rate limited (retry-after: {retry_after_secs:?}s)")]
    RateLimited { retry_after_secs: Option<u64> },

    /// Transient upstream failure — connect/timeout, 5xx, or an auth/model
    /// misconfiguration (401/403/404) that is not the request's fault.
    /// Nothing about the input caused it, so callers must not record a
    /// per-item failure; retry later.
    #[error("upstream unavailable: {0}")]
    Unavailable(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl AlayaError {
    /// JSON-RPC error code per MCP spec.
    pub fn jsonrpc_code(&self) -> i32 {
        match self {
            Self::Storage(_) => -32000,
            Self::Embedding(_) => -32001,
            Self::Graph(_) => -32002,
            Self::Config(_) => -32003,
            Self::Validation(_) => -32602,
            Self::NotFound(_) => -32004,
            Self::Summary(_) => -32005,
            Self::Rerank(_) => -32006,
            Self::Judge(_) => -32007,
            Self::RateLimited { .. } => -32008,
            Self::Unavailable(_) => -32009,
            Self::Serialization(_) => -32600,
        }
    }

    /// Sanitized message safe for external consumers.
    /// Never includes hostnames, connection strings, or stack traces.
    pub fn safe_message(&self) -> &'static str {
        match self {
            Self::Storage(_) => "Vector storage operation failed",
            Self::Embedding(_) => "Embedding generation failed",
            Self::Graph(_) => "Graph operation failed",
            Self::Config(_) => "Service configuration error",
            Self::Validation(_) => "Invalid request parameters",
            Self::NotFound(_) => "Resource not found",
            Self::Summary(_) => "Summary generation failed",
            Self::Rerank(_) => "Rerank operation failed",
            Self::Judge(_) => "Contradiction judge failed",
            Self::RateLimited { .. } => "Upstream rate limited",
            Self::Unavailable(_) => "Upstream temporarily unavailable",
            Self::Serialization(_) => "Invalid request format",
        }
    }
}

pub type Result<T> = std::result::Result<T, AlayaError>;
