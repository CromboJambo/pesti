//! Weight-space programming API for PESTI.
//!
//! Provides runtime read/analyze/edit capabilities on model weights without
//! requiring a training loop. Built on pesti-runner's GGUF weight loading
//! infrastructure with atomic edit application and snapshot/rollback support.
//!
//! ## Core workflow
//!
//! ```no_run
//! use std::path::Path;
//! use pesti_weightspace::{WeightSpaceSession, WeightEdit};
//!
//! let mut session = WeightSpaceSession::open("model.gguf")?;
//!
//! // Read a weight value
//! let val = session.read_value("tok_embeddings.weight", &[0, 0])?;
//!
//! // Apply surgical edits atomically
//! session.apply_edits(vec![
//!     WeightEdit { param: "tok_embeddings.weight".into(), index: vec![0, 0], value: 0.5 },
//!     WeightEdit { param: "output.weight".into(), index: vec![3, 7], value: -0.2 },
//! ])?;
//!
//! // Save edited model
//! session.save_edited_model("model_patched.gguf")?;
//! # Ok::<(), pesti_weightspace::WeightSpaceError>(())
//! ```

pub mod error;
pub mod peft;
pub mod session;
pub mod stats;

pub use error::{Result, WeightSpaceError};
pub use peft::{apply_lora_adapter, lora_delta_to_edits, unmerge_lora_adapter};
pub use session::{Snapshot, WeightEdit, WeightOp, WeightSpaceSession};
pub use stats::WeightStatOp;
