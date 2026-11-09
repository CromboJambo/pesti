//! pesti-state CLI — query commands against the PESTI state database.
//!
//! Subcommands:
//!   list-models       Print registered models
//!   list-adapters     Print adapters (filterable by state)
//!   show-adapter      Full provenance record for an adapter
//!   session-log       Replay operations for a weight-space session

use clap::{Parser, Subcommand};
use pesti_state::{AdapterState, OperationDetails, PestiState};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "pesti-state",
    about = "Query the PESTI state database (models, adapters, operations)"
)]
struct Cli {
    /// Path to the state database (default: ~/.pesti/state.db)
    #[arg(short, long, global = true)]
    db: Option<PathBuf>,

    /// Output as JSON instead of formatted text
    #[arg(short, long, global = true)]
    json: bool,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// List all registered models
    ListModels,

    /// List adapters (optionally filtered by state)
    ListAdapters {
        /// Filter by lifecycle state (unverified, validated, active, needs_revalidation, deprecated)
        #[arg(short, long)]
        state: Option<String>,
    },

    /// Show full provenance record for an adapter
    ShowAdapter {
        /// Adapter ID
        id: String,
    },

    /// Replay operations for a weight-space session
    SessionLog {
        /// Session UUID
        session_id: String,
    },
}

fn main() {
    let cli = Cli::parse();

    // Open the state database (read-only path doesn't exist in PestiState API,
    // so we open normally — all our operations are SELECTs)
    let state = match cli.db {
        Some(path) => PestiState::open(&path),
        None => PestiState::open_default(),
    }
    .unwrap_or_else(|e| {
        eprintln!("error: failed to open state database: {e}");
        std::process::exit(1);
    });

    let exit_code = match &cli.command {
        Command::ListModels => cmd_list_models(&state, cli.json),
        Command::ListAdapters { state: filter } => {
            cmd_list_adapters(&state, filter.as_deref(), cli.json)
        }
        Command::ShowAdapter { id } => cmd_show_adapter(&state, id, cli.json),
        Command::SessionLog { session_id } => cmd_session_log(&state, session_id, cli.json),
    };

    std::process::exit(exit_code);
}

fn cmd_list_models(state: &PestiState, json: bool) -> i32 {
    let models = match state.list_models() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: failed to list models: {e}");
            return 1;
        }
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&models).unwrap_or_default()
        );
        return 0;
    }

    if models.is_empty() {
        println!("No models registered.");
        return 0;
    }

    // Header
    println!(
        "{:<12} {:<14} {:<10} {:>6} {}",
        "ID", "ARCHITECTURE", "QUANT", "TENSORS", "PATH"
    );
    println!("{}", "-".repeat(96));

    for m in &models {
        let id_short = &m.id[..m.id.len().min(12)];
        println!(
            "{:<12} {:<14} {:<10} {:>6} {}",
            id_short, m.architecture, m.quantization, m.tensor_count, m.path
        );
    }

    println!("\n{} model(s) registered.", models.len());
    0
}

fn cmd_list_adapters(state: &PestiState, state_filter: Option<&str>, json: bool) -> i32 {
    let adapter_state = match state_filter {
        Some(s) => match s.parse::<AdapterState>() {
            Ok(st) => st,
            Err(e) => {
                eprintln!("error: invalid state filter '{s}': {e}");
                return 1;
            }
        },
        None => AdapterState::Active, // default: show active (matches spec example)
    };

    let adapters = match state.list_adapters(Some(adapter_state)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: failed to list adapters: {e}");
            return 1;
        }
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&adapters).unwrap_or_default()
        );
        return 0;
    }

    if adapters.is_empty() {
        println!("No {} adapters found.", adapter_state);
        return 0;
    }

    // Header
    println!(
        "{:<12} {:<12} {:<20} {:<36} {}",
        "ID", "STATE", "BASE MODEL", "CREATED", "OUTPUT PATH"
    );
    println!("{}", "-".repeat(108));

    for a in &adapters {
        let id_short = &a.id[..a.id.len().min(12)];
        let base_short = &a.base_model_id[..a.base_model_id.len().min(20)];
        // Trim timestamp to date+time without fractional seconds
        let created = a.created_at.split('.').next().unwrap_or(&a.created_at);
        println!(
            "{:<12} {:<12} {:<20} {:<36} {}",
            id_short, a.state, base_short, created, a.output_path
        );
    }

    println!(
        "\n{} adapter(s) in state '{}'.",
        adapters.len(),
        adapter_state
    );
    0
}

fn cmd_show_adapter(state: &PestiState, id: &str, json: bool) -> i32 {
    let record = match state.get_adapter(id) {
        Ok(Some(r)) => r,
        Ok(None) => {
            eprintln!("error: adapter '{}' not found", id);
            return 1;
        }
        Err(e) => {
            eprintln!("error: failed to look up adapter: {e}");
            return 1;
        }
    };

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&record).unwrap_or_default()
        );
        return 0;
    }

    println!("Adapter: {}", record.id);
    println!("State:   {}", record.state);
    println!("Created: {}", record.created_at);
    if let Some(v) = &record.verified_at {
        println!("Verified: {}", v);
    }
    println!();
    println!("Base model: {}", record.base_model_id);
    println!("Output:     {}", record.output_path);
    if let Some(desc) = &record.description {
        println!("Description: {}", desc);
    }
    if let Some(assumptions) = &record.assumptions {
        if !assumptions.is_empty() {
            println!();
            println!("Assumptions:");
            for a in assumptions {
                println!("  - {}", a);
            }
        }
    }
    if !record.operation_ids.is_empty() {
        println!();
        println!(
            "Operation IDs: {}",
            record
                .operation_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // Fetch and display linked operations
    let ops = match state.model_operations(&record.base_model_id) {
        Ok(ops) => ops,
        Err(_) => Vec::new(),
    };
    let linked_ops: Vec<_> = ops
        .iter()
        .filter(|op| record.operation_ids.iter().any(|&oid| op.id == Some(oid)))
        .collect();

    if !linked_ops.is_empty() {
        println!();
        println!("Provenance operations:");
        for op in &linked_ops {
            println!(
                "  [{}] {} — {}",
                op.timestamp,
                op.op_type,
                describe_details(&op.details)
            );
        }
    }

    0
}

fn cmd_session_log(state: &PestiState, session_id: &str, json: bool) -> i32 {
    let ops = match state.session_operations(session_id) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("error: failed to read session log: {e}");
            return 1;
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&ops).unwrap_or_default());
        return 0;
    }

    if ops.is_empty() {
        println!("No operations found for session '{}'.", session_id);
        return 0;
    }

    println!("Session: {}", session_id);
    println!("{}", "-".repeat(80));

    for op in &ops {
        let ts = op.timestamp.split('.').next().unwrap_or(&op.timestamp);
        let id_str = op.id.map(|i| format!("#{}", i)).unwrap_or_default();
        println!(
            "[{}] {} {:<10} {}",
            ts,
            id_str,
            op.op_type,
            describe_details(&op.details)
        );
    }

    println!("\n{} operation(s).", ops.len());
    0
}

/// Human-readable summary of operation details.
fn describe_details(details: &OperationDetails) -> String {
    match details {
        OperationDetails::Open {
            model_path,
            tensor_count,
            ..
        } => {
            format!("opened {} ({} tensors)", model_path, tensor_count)
        }
        OperationDetails::Edit {
            edit_count,
            tensors_modified,
            snapshot_taken,
        } => {
            let snap = if *snapshot_taken {
                " [auto-snapshot]"
            } else {
                ""
            };
            format!(
                "{} edit(s): {}{}",
                edit_count,
                tensors_modified.join(", "),
                snap
            )
        }
        OperationDetails::Snapshot { snapshot_index } => {
            format!("snapshot #{}", snapshot_index)
        }
        OperationDetails::Rollback { from_snapshot } => {
            format!("rolled back to snapshot #{}", from_snapshot)
        }
        OperationDetails::Save {
            output_path,
            format,
            adapter_id,
        } => {
            let adapter = adapter_id
                .as_ref()
                .map(|id| format!(" (adapter: {})", &id[..id.len().min(12)]))
                .unwrap_or_default();
            format!("saved to {} as {}{}", output_path, format, adapter)
        }
        OperationDetails::Close {
            total_edits,
            snapshots_created,
            rollbacks_performed,
            duration_secs,
        } => {
            format!(
                "closed ({} edits, {} snapshots, {} rollbacks, {:.1}s)",
                total_edits, snapshots_created, rollbacks_performed, duration_secs
            )
        }
    }
}
