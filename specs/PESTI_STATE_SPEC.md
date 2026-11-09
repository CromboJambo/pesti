# PESTI State: Persistence and Provenance for Weight-Space Operations

## §0 Goal + Scope

**Goal**: Add a lightweight persistence layer to PESTI that makes weight-space operations auditable, adapters traceable, and inheritance revalidatable. Every model pesti touches is registered; every weight edit is logged; every saved adapter carries provenance metadata and lifecycle state.

**NOT in scope (yet)**:
- ❌ Distributed state or multi-node replication (that's crabjar/mirror-lab's domain)
- ❌ Full knowledge graph or semantic indexing of what adapters encode
- ❌ Replacing `pesti-durable-objects` — this spec builds on it, not around it
- ❌ Automatic validation of adapter correctness (revalidation *triggers* yes, running the validation no)

**Scope**: One SQLite database (`pesti-state.db`) with three tables, integrated into `pesti-weightspace` so that sessions are tracked, operations are logged, and saved adapters are registered with provenance. The revalidation trigger mechanism: when a base model's hash changes, all adapters built on it are flagged for re-examination.

**Design principle**: "No silent inheritance." Any knowledge (weight pattern, adapter, trained behavior) that enters the system must carry enough provenance to be questioned later. If the context it was built in changes, the system must surface that fact rather than silently continuing to use stale knowledge.

## §1 Context & Why Now

**Principle**: PESTI is moving from a stateless inference engine toward a weight-space programming substrate. Stateless execution works for one-shot inference; it fails when weights become a memory medium.

**Why now**:
- `pesti-weightspace` (Week 30) provides in-memory weight manipulation with snapshot/rollback — but sessions are ephemeral. No record survives process exit of what was done or why.
- `pesti-durable-objects` exists as a generic JSON snapshot layer (extracted from herdr) but is domain-agnostic and not wired into weightspace.
- LoRA adapters via the PEFT bridge can encode behavioral knowledge directly into weight space — but there's no persistent record of what an adapter encodes, what assumptions it was built under, or when it should be revalidated.
- The "Memory Bottleneck Computing" analysis identified **State (Layer 4)** as the load-bearing layer that must be reinforced before Representation, Execution, or Orchestration can scale safely. State failures are silent; other layers have tight feedback loops (parse/run/tok/s).

**Failure modes this prevents**:
1. A biased LoRA adapter is trained and saved. Six months later it's loaded into a new session with no record of its training data, assumptions, or known limitations. The bias is inherited silently.
2. A base model is updated (new GGUF release). Adapters built on the old version are still loaded without anyone noticing the mismatch — the weight patterns may no longer be valid.
3. Multiple weight-space sessions edit the same model over time. There's no audit trail of what changed, when, or which session made which edits.

**Existing infrastructure to build on**:
- `pesti-durable-objects` — snapshot persistence, versioning, migration (`save_snapshot`, `load_snapshot`, `SNAPSHOT_VERSION`)
- `pesti-durable-objects::object_id::ObjectId` — UUID-based identity (reuse for adapter IDs)
- `pesti-weightspace::session::WeightSpaceSession` — the session being instrumented
- `pesti-gguf` — GGUF header parsing (for model identification/hashing)

## §2 Architecture

### New Crate: `pesti-state`

```toml
[package]
name = "pesti-state"
version.workspace = true
edition.workspace = true
description = "Persistence and provenance layer for PESTI weight-space operations"

[dependencies]
pesti-durable-objects = { path = "../pesti-durable-objects" }
pesti-gguf = { path = "../pesti-gguf" }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rusqlite = { version = "0.31", features = ["bundled"] }
uuid = { version = "1.0", features = ["v4"] }
chrono = { version = "0.4", features = ["serde"] }
thiserror = "1.0"
```

### Database Schema

Single SQLite file, default path `~/.pesti/state.db` (overridable via constructor).

```sql
-- Models pesti has encountered
CREATE TABLE IF NOT EXISTS models (
    id              TEXT PRIMARY KEY,       -- SHA-256 of GGUF header bytes
    path            TEXT NOT NULL,          -- filesystem path at time of registration
    architecture    TEXT NOT NULL,          -- e.g. "llama", "tinyllama"
    quantization    TEXT NOT NULL,          -- e.g. "Q4_K_M", "F16"
    param_count     INTEGER,                -- approximate parameter count
    tensor_count    INTEGER,                -- number of tensors
    first_seen      TEXT NOT NULL,          -- ISO 8601 timestamp
    last_verified   TEXT                    -- last time hash was re-confirmed
);

-- Append-only log of weight-space operations
CREATE TABLE IF NOT EXISTS operations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id        TEXT REFERENCES models(id),
    session_id      TEXT NOT NULL,          -- UUID identifying the WeightSpaceSession
    op_type         TEXT NOT NULL,          -- 'open' | 'edit' | 'snapshot' | 'rollback' | 'save' | 'close'
    details         JSON NOT NULL,          -- operation-specific payload (see below)
    timestamp       TEXT NOT NULL           -- ISO 8601
);

-- Saved adapters / edited models with provenance
CREATE TABLE IF NOT EXISTS adapters (
    id              TEXT PRIMARY KEY,       -- UUID (ObjectId)
    base_model_id   TEXT REFERENCES models(id),
    output_path     TEXT NOT NULL,          -- where the adapter/edited model was saved
    description     TEXT,                   -- human-readable: what this is supposed to encode
    assumptions     JSON,                   -- conditions under which it's valid
    created_at      TEXT NOT NULL,          -- ISO 8601
    verified_at     TEXT,                   -- last validation timestamp
    state           TEXT NOT NULL DEFAULT 'unverified',
    operation_ids   JSON                    -- array of operation IDs that produced this adapter
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_operations_model ON operations(model_id);
CREATE INDEX IF NOT EXISTS idx_operations_session ON operations(session_id);
CREATE INDEX IF NOT EXISTS idx_adapters_base ON adapters(base_model_id);
CREATE INDEX IF NOT EXISTS idx_adapters_state ON adapters(state);
```

### Adapter Lifecycle States

```rust
/// Lifecycle state for a registered adapter.
/// Represents where an adapter is in the knowledge validation pipeline:
/// Observed → Recorded → Examined → Tested → Accepted → Inherited → Revalidated
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterState {
    /// Adapter exists but has not been examined or tested.
    Unverified,
    /// Adapter has been tested and produces expected behavior.
    Validated,
    /// Adapter is actively in use by inference sessions.
    Active,
    /// Base model changed; adapter needs re-examination before reuse.
    NeedsRevalidation,
    /// Adapter has been superseded or determined invalid.
    Deprecated,
}
```

### Core API: `PestiState`

```rust
use std::path::{Path, PathBuf};
use rusqlite::Connection;
use crate::adapter::{AdapterRecord, AdapterState};
use crate::model::ModelRecord;
use crate::operation::{OperationLog, OperationType};

/// PESTI's persistent state layer.
///
/// Owns the SQLite connection and provides typed access to models,
/// operations, and adapters. Thread-safe via interior mutability
/// (Connection is Send + Sync when opened in multi-threaded mode).
pub struct PestiState {
    conn: Connection,
    db_path: PathBuf,
}

impl PestiState {
    /// Open or create the state database at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StateError>;

    /// Default location: ~/.pesti/state.db
    pub fn default() -> Result<Self, StateError>;

    // ─── Model Registry ────────────────────────────────────────────

    /// Register a model (by GGUF header hash) if not already known.
    /// Returns the model ID. If the model exists but with a different
    /// path, updates the path and last_verified timestamp.
    pub fn register_model(&self, record: &ModelRecord) -> Result<String, StateError>;

    /// Look up a model by its content hash.
    pub fn get_model(&self, id: &str) -> Result<Option<ModelRecord>, StateError>;

    /// Get all models currently registered.
    pub fn list_models(&self) -> Result<Vec<ModelRecord>, StateError>;

    // ─── Operation Log ─────────────────────────────────────────────

    /// Append an operation to the audit log.
    pub fn log_operation(&self, op: &OperationLog) -> Result<i64, StateError>;

    /// Get all operations for a given session.
    pub fn session_operations(&self, session_id: &str) -> Result<Vec<OperationLog>, StateError>;

    /// Get all operations for a given model (across all sessions).
    pub fn model_operations(&self, model_id: &str) -> Result<Vec<OperationLog>, StateError>;

    // ─── Adapter Registry ──────────────────────────────────────────

    /// Register a new adapter with provenance metadata.
    pub fn register_adapter(&self, record: AdapterRecord) -> Result<String, StateError>;

    /// Update an adapter's lifecycle state.
    pub fn update_adapter_state(&self, adapter_id: &str, state: AdapterState) -> Result<(), StateError>;

    /// Get all adapters built on a specific base model.
    pub fn adapters_for_model(&self, model_id: &str) -> Result<Vec<AdapterRecord>, StateError>;

    /// List all adapters in a given lifecycle state.
    pub fn list_adapters(&self, state: Option<AdapterState>) -> Result<Vec<AdapterRecord>, StateError>;

    // ─── Revalidation ──────────────────────────────────────────────

    /// When a base model's hash changes (or is re-verified as different),
    /// flag all adapters built on the old version for revalidation.
    /// Returns the count of adapters flagged.
    pub fn invalidate_adapters_for_model(&self, old_model_id: &str) -> Result<usize, StateError>;

    /// Check if any registered adapters need revalidation.
    pub fn needs_revalidation(&self) -> Result<Vec<AdapterRecord>, StateError>;
}
```

### Operation Details Payloads

The `details` JSON field in the `operations` table varies by `op_type`:

```rust
/// Operation-specific payloads stored as JSON in the operations table.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum OperationDetails {
    /// Session opened on a model
    Open {
        session_id: String,
        model_path: String,
        tensor_count: usize,
    },
    /// Weight edits applied
    Edit {
        edit_count: usize,
        tensors_modified: Vec<String>,  // just names, not values (privacy/size)
        snapshot_taken: bool,           // was a snapshot taken before this edit?
    },
    /// Snapshot created
    Snapshot {
        snapshot_index: usize,          // position in the session's snapshot stack
    },
    /// Rollback performed
    Rollback {
        from_snapshot: usize,
    },
    /// Model saved to disk
    Save {
        output_path: String,
        format: String,                 // "safetensors" | "gguf"
        adapter_id: Option<String>,     // if registered as an adapter
    },
    /// Session closed
    Close {
        total_edits: usize,
        snapshots_created: usize,
        rollbacks_performed: usize,
        duration_secs: f64,
    },
}
```

### Integration with `pesti-weightspace`

Minimal changes to `WeightSpaceSession`. The state layer is optional — sessions can run without it (backward compatible), but when provided, every significant operation is logged.

```rust
// pesti-weightspace/src/session.rs (additions)

use pesti_state::{PestiState, OperationLog, OperationDetails, AdapterRecord};

pub struct WeightSpaceSession {
    model_path: PathBuf,
    weights: GgufWeights,
    snapshots: VecDeque<Snapshot>,
    
    // New: optional state tracking
    session_id: String,                  // always generated (UUID)
    opened_at: std::time::Instant,       // for duration tracking
    state: Option<PestiState>,           // None = untracked session
    model_id: Option<String>,            // set when state is present
    edit_count: usize,                   // running total for close log
}

impl WeightSpaceSession {
    /// Open a tracked weight-space session.
    /// Registers the model, logs the open operation.
    pub fn open_tracked(model_path: impl AsRef<Path>, state: &PestiState) -> Result<Self> {
        let mut session = Self::open(model_path)?;
        session.state = Some(state.clone());
        
        // Register model and get its ID
        let header_bytes = compute_gguf_header_hash(&session.weights.header);
        let model_id = state.register_model(&ModelRecord {
            id: header_bytes,
            path: session.model_path.to_string_lossy().to_string(),
            architecture: extract_architecture(&session.weights.header),
            quantization: extract_quantization(&session.weights.header),
            param_count: estimate_param_count(&session.weights),
            tensor_count: session.weights.tensors.len(),
        })?;
        session.model_id = Some(model_id.clone());
        
        // Log the open
        state.log_operation(&OperationLog {
            model_id: Some(model_id),
            session_id: session.session_id.clone(),
            op_type: OperationType::Open,
            details: OperationDetails::Open {
                session_id: session.session_id.clone(),
                model_path: session.model_path.to_string_lossy().to_string(),
                tensor_count: session.weights.tensors.len(),
            },
        })?;
        
        Ok(session)
    }

    /// Save as a registered adapter with provenance.
    pub fn save_as_adapter(
        &self,
        output_path: impl AsRef<Path>,
        description: &str,
        assumptions: Vec<String>,
    ) -> Result<String> {
        let path = output_path.as_ref();
        self.save_edited_model(path)?;
        
        let state = self.state.as_ref()
            .ok_or(WeightSpaceError::NoStateAttached)?;
        let model_id = self.model_id.clone()
            .ok_or(WeightSpaceError::NoModelId)?;
        
        let adapter_id = state.register_adapter(AdapterRecord {
            id: ObjectId::generate().to_string(),
            base_model_id: model_id,
            output_path: path.to_string_lossy().to_string(),
            description: Some(description.to_string()),
            assumptions: if assumptions.is_empty() { None } else { Some(assumptions) },
            state: AdapterState::Unverified,
            operation_ids: vec![/* relevant op IDs from this session */],
        })?;
        
        Ok(adapter_id)
    }

    // Drop impl logs the close operation if state is attached
}
```

### Revalidation Trigger Flow

```
1. User opens model.gguf via WeightSpaceSession::open_tracked()
2. pesti-state hashes GGUF header → gets model_id "abc123..."
3. Query: SELECT * FROM adapters WHERE base_model_id = 'abc123...' AND state != 'deprecated'
4. If any adapters exist with state = 'needs_revalidation':
   → Warn user: "3 adapters for this model are flagged for revalidation"
5. Separately, when a NEW version of a model appears (different hash):
   → Query old adapters by their base_model_id
   → UPDATE adapters SET state = 'needs_revalidation' WHERE base_model_id = old_hash
   → This is the "no silent inheritance" mechanism
```

### Relationship to `pesti-durable-objects`

`pesti-state` uses SQLite for its primary storage (relational, queryable). `pesti-durable-objects` remains available for:
- Large binary snapshots (full weight dumps that don't fit in metadata)
- Versioned format migration logic (`SNAPSHOT_VERSION`)
- The `ObjectId` type is reused directly

They're complementary: durable-objects handles "save this big blob and be able to restore it," pesti-state handles "track what this blob is, who made it, and whether it's still valid."

## §3 Acceptance Criteria

### Build Gates
- [ ] `cargo build -p pesti-state` succeeds
- [ ] `cargo test -p pesti-state` all tests pass
- [ ] No new clippy warnings in pesti-state or pesti-weightspace
- [ ] `pesti-weightspace` builds and tests pass with state integration (backward compatible — untracked sessions still work)

### Unit Tests
- [ ] `tests/schema.rs` — database creation, table existence, index verification
- [ ] `tests/models.rs` — register_model idempotency, get_model, list_models
- [ ] `tests/operations.rs` — log_operation ordering, session_operations query, model_operations query
- [ ] `tests/adapters.rs` — register_adapter, update_adapter_state, adapters_for_model, list_adapters with state filter
- [ ] `tests/revalidation.rs` — invalidate_adapters_for_model flags correctly, needs_revalidation returns correct set
- [ ] `tests/integration.rs` — full flow: open_tracked → edit → save_as_adapter → close → reopen → verify log and adapter record exist

### Integration with pesti-weightspace
- [ ] `WeightSpaceSession::open_tracked()` logs open operation
- [ ] `apply_edits()` logs edit operations (when state attached)
- [ ] `snapshot()` / `rollback()` logged
- [ ] Drop of session logs close operation with duration and totals
- [ ] Untracked sessions (`open()`) work identically to before — zero behavior change

### CLI/Query Interface (minimal)
- [x] `pesti-state list-models` — print registered models
- [x] `pesti-state list-adapters [--state <state>]` — print adapters, filterable by state
- [x] `pesti-state show-adapter <id>` — full provenance record
- [x] `pesti-state session-log <session_id>` — replay operations for a session

## §4 Done-When / Handoff

When complete:
1. Update `ROADMAP.md` — add PESTI State section under Week 30 or Week 31
2. Add EDR to `CHANGELOG.md` documenting the design decision (SQLite vs JSON, three-table schema, revalidation trigger)
3. Update `docs/PROJECT_STATUS.md` with new crate info
4. Update workspace `Cargo.toml` to include `pesti-state` as a member
5. Document the state file location and backup considerations in `docs/README.md`

Stop at next gate: Integration with crabjar for cross-repo provenance (shared `Provenance` type), and automatic validation pipelines that consume the `needs_revalidation` state.

## §5 Constraints & Cautions

- **SQLite is single-writer**: PESTI sessions are typically single-process, but if multiple pesti processes access the same state.db concurrently, use WAL mode (`PRAGMA journal_mode=WAL`) to allow concurrent readers with serialized writers.
- **Don't log weight values in operations**: The `details` JSON for edits should record tensor names and counts, not actual f32 values. Weight values are in the model files; the operation log is metadata. Logging millions of floats into SQLite defeats the purpose.
- **Model hashing must be stable**: Use SHA-256 of the GGUF header bytes (not the full file — too slow for large models). The header contains architecture, tensor names/shapes/dtypes which uniquely identify the model structure. If two files have different headers but same path, they're different models.
- **Backward compatibility is non-negotiable**: `WeightSpaceSession::open()` must continue to work without any state dependency. State tracking is opt-in via `open_tracked()`. Existing code that doesn't know about pesti-state should not break.
- **Database file location**: Default to `~/.pesti/state.db` but allow override. Never hardcode a path that requires root or writes outside the user's home.
- **Migration strategy**: Follow `pesti-durable-objects` pattern — include a schema version in the database (`PRAGMA user_version`) and provide migration functions for future schema changes.

## §6 Suggested TDD Order

1. Create `pesti-state` crate with Cargo.toml, add to workspace members
2. Implement database initialization (schema creation, WAL mode, PRAGMA user_version)
3. Define data types: `ModelRecord`, `OperationLog`, `OperationDetails`, `AdapterRecord`, `AdapterState` — with serialization tests
4. Implement `PestiState::open()` / `default()` with migration support
5. Implement model registry: `register_model`, `get_model`, `list_models` — test idempotency
6. Implement operation log: `log_operation`, `session_operations`, `model_operations` — test ordering and queries
7. Implement adapter registry: `register_adapter`, `update_adapter_state`, `adapters_for_model`, `list_adapters`
8. Implement revalidation: `invalidate_adapters_for_model`, `needs_revalidation`
9. Integrate with `pesti-weightspace`: add `session_id`, `state`, `model_id` fields to `WeightSpaceSession`
10. Implement `open_tracked()` and operation logging hooks in existing methods
11. Implement `save_as_adapter()` 
12. Implement Drop-based close logging
13. Write integration test: full lifecycle with real model file (tinyllama-q4.gguf)
14. Add minimal CLI subcommands

## §7 Open Questions

1. **Should the state database be per-project or global?**
   - Default: Global (`~/.pesti/state.db`). A model is a model regardless of which project directory you're in. If isolation is needed later, add a `project` column and filter queries.

2. **How granular should operation logging be?**
   - Default: Log open/edit/snapshot/rollback/save/close. Don't log individual tensor reads (too noisy). Edit operations record count + tensor names, not values.

3. **Should adapters support versioning (v1, v2 of the same adapter)?**
   - Default: No. Each save creates a new adapter record with a new UUID. If you want to track "improvements" to an adapter, that's a relationship between records, not a version field. Keep it simple initially.

4. **Integration with mirror-log (hermes-memory feature)?**
   - Default: Defer. `pesti-durable-objects` already has optional mirror-log integration. If pesti-state events should also flow to mirror-log, that's a later wiring task, not part of the core schema.

5. **Should there be a retention policy for the operations log?**
   - Default: No automatic deletion. The log is append-only and small (metadata only). If it grows too large, add a `pesti-state prune --before <date>` command later. Don't auto-delete audit data.
