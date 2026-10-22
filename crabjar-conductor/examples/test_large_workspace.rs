//! Test Phase 4 workspace analysis against a larger Rust repo.

use crabjar_conductor::decision::workspace::{analyze_workspace, AnalysisBudget};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let workspace_path = if args.len() > 1 {
        Path::new(&args[1])
    } else {
        Path::new(".")
    };
    let criteria_path = "specs/criteria/error-handling-adequacy.json";

    println!("Analyzing workspace: {}", workspace_path.display());
    println!("Criteria: {}", criteria_path);
    println!();

    let budget = AnalysisBudget {
        max_crates: 0, // Analyze all crates
        max_time_seconds: 0,
        collapse_large_regions: true,
    };

    let result = analyze_workspace(workspace_path, criteria_path, budget);

    println!("=== Workspace Analysis Results ===");
    println!(
        "Total crates discovered: {}",
        result.architectural_properties.total_crates
    );
    println!(
        "Max dependency depth: {}",
        result.architectural_properties.max_dependency_depth
    );
    println!(
        "Circular dependencies: {}",
        result.architectural_properties.circular_dependencies.len()
    );
    if !result.architectural_properties.circular_dependencies.is_empty() {
        for cycle in &result.architectural_properties.circular_dependencies {
            println!("  Cycle: {}", cycle.join(" -> "));
        }
    }
    println!(
        "Orphaned crates: {}",
        result.architectural_properties.orphaned_crates.join(", ")
    );
    println!(
        "Coupling score: {:.4}",
        result.architectural_properties.coupling_score
    );
    println!(
        "Cohesion score: {:.4}",
        result.architectural_properties.cohesion_score
    );
    println!();

    println!("=== Evaluation Summary ===");
    println!(
        "Total evaluations: {}",
        result.evaluation_summary.total_evaluations
    );
    println!(
        "Structural condition matches: {}",
        result.evaluation_summary.structural_condition_matches
    );
    println!(
        "Embedding fallbacks: {}",
        result.evaluation_summary.embedding_fallbacks
    );
    println!(
        "Average confidence: {:.4}",
        result.evaluation_summary.average_confidence
    );
    println!();

    println!("=== Budget Usage ===");
    println!("Crates evaluated: {}", result.budget_used.crates_evaluated);
    println!(
        "Time spent: {}s",
        result.budget_used.time_spent_seconds
    );
    println!("Budget exhausted: {}", result.budget_used.budget_exhausted);

    // Print per-crate results
    println!();
    println!("=== Per-Crate Results ===");
    for (name, r) in &result.crate_evaluations {
        println!(
            "  {}: structural={}, embedding={}, confidence={:.2}",
            name,
            if r.structural_match { "✓" } else { "✗" },
            if r.embedding_used { "used" } else { "not used" },
            r.confidence
        );
    }
}
