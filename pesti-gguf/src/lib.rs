pub mod error;
pub mod types;
pub mod parser;
pub mod writer;

#[cfg(test)]
pub mod tests {
    pub mod defensive_tests;
    pub mod gguf_v3_conformance;
    pub mod large_model_conformance;
    pub mod r#mod;
    
    // Re-export the helper function for use in test modules
    pub use r#mod::conformance_corpus_path;
}

pub use error::GgufError;
pub use types::*;
pub use parser::{compute_data_section_start, extract_tensor_bytes, extract_tensor_bytes_from, extract_tensor_bytes_from_path, parse_gguf, parse_gguf_reader, tensor_bytes_for_dtype};
pub use writer::GgufWriter;


