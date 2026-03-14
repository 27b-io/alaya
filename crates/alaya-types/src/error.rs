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
            Self::Serialization(_) => "Invalid request format",
        }
    }
}

pub type Result<T> = std::result::Result<T, AlayaError>;
