//! pesti-embeddings: Text-to-vector embedding inference using PESTI's transformer stack.
//!
//! Supports encoder-only models (BERT-style) for semantic search, clustering,
//! and similarity tasks. Uses existing pesti-runner transformer implementation
//! with bidirectional attention support.

pub mod embedder;
pub mod pooling;
pub mod similarity;

pub use embedder::{Embedder, EmbeddingConfig};
pub use pooling::PoolingStrategy;
pub use similarity::{cosine_similarity, dot_product};