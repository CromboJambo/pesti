//! Session save/load for llama.cpp context state.

use std::path::Path;
use std::sync::Arc;

use llama_cpp_2::context::{LlamaContext, session};

/// Manages session save/load for a [`LlamaContext`](LlamaContext).
pub struct SessionManager {
    context: Arc<LlamaContext>,
}

impl SessionManager {
    pub fn new(context: Arc<LlamaContext>) -> Self {
        Self { context }
    }

    /// Save the current context state to a file.
    pub fn save(&self, path: &str) -> Result<(), String> {
        self.context.save_session_file(path)?;
        Ok(())
    }

    /// Load context state from a file.
    pub fn load(&self, path: &str) -> Result<(), String> {
        self.context.load_session_file(path)?;
        Ok(())
    }

    /// Check if a session file exists.
    pub fn session_exists(path: &str) -> bool {
        std::fs::metadata(path).is_ok()
    }

    /// Save a specific sequence's state to a file.
    pub fn save_seq_state(&self, seq_id: i32, path: &str) -> Result<(), String> {
        self.context.state_seq_save_file(seq_id, path)?;
        Ok(())
    }

    /// Load a specific sequence's state from a file.
    pub fn load_seq_state(&self, seq_id: i32, path: &str) -> Result<(), String> {
        self.context.state_seq_load_file(seq_id, path)?;
        Ok(())
    }

    /// Get the size needed to save the full state.
    pub fn state_size(&self) -> Result<usize, String> {
        let size = self.context.get_state_size()?;
        Ok(size)
    }

    /// Get the size needed to save a specific sequence's state.
    pub fn state_seq_size(&self, seq_id: i32) -> Result<usize, String> {
        let size = self.context.state_seq_get_size_ext(seq_id)?;
        Ok(size)
    }
}
