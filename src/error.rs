use thiserror::Error;

#[derive(Error, Debug)]
pub enum CortexError {
    #[error("Database error: {0}")]
    Db(#[from] rusqlite::Error),

    #[error("Database task error: {0}")]
    DbTask(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("HNSW error: {0}")]
    Hnsw(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("LLM error: {0}")]
    Llm(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Tool error: {0}")]
    Tool(String),

    #[error("Config error: {0}")]
    Config(String),

    #[error("Max iterations reached: {0}")]
    MaxIterations(usize),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Channel error: {0}")]
    Channel(String),

    #[error("Unsupported: {0}")]
    Unsupported(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),
}

pub type Result<T> = std::result::Result<T, CortexError>;
