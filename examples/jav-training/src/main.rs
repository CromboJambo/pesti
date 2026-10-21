//! Example: JEV training data selection on synthetic coding examples.
//!
//! Demonstrates how JEV identifies redundant vs novel constraint contributions
//! in a training dataset using PESTI embeddings.

use pesti_jav_training::selection::{select_minimal_dataset, SelectionConfig, TrainingExample};
use pesti_jav_training::metrics::evaluate_selection;

fn main() {
    // Create synthetic coding examples with intentional redundancy
    let examples = vec![
        TrainingExample {
            id: "1".to_string(),
            text: "fn add(a: i32, b: i32) -> i32 { return a + b; }".to_string(),
            labels: vec!["arithmetic", "addition"],
        },
        TrainingExample {
            id: "2".to_string(),
            text: "fn subtract(a: i32, b: i32) -> i32 { return a - b; }".to_string(),
            labels: vec!["arithmetic", "subtraction"],
        },
        TrainingExample {
            id: "3".to_string(),
            text: "fn multiply(a: i32, b: i32) -> i32 { return a * b; }".to_string(),
            labels: vec!["arithmetic", "multiplication"],
        },
        TrainingExample {
            id: "4".to_string(),
            text: "fn increment(x: i32) -> i32 { return x + 1; }".to_string(),
            labels: vec!["arithmetic", "addition"],
        },
        TrainingExample {
            id: "5".to_string(),
            text: "fn decrement(x: i32) -> i32 { return x - 1; }".to_string(),
            labels: vec!["arithmetic", "subtraction"],
        },
        TrainingExample {
            id: "6".to_string(),
            text: "fn double(x: i32) -> i32 { return x * 2; }".to_string(),
            labels: vec!["arithmetic", "multiplication"],
        },
        TrainingExample {
            id: "7".to_string(),
            text: "fn factorial(n: i32) -> i32 { if n <= 1 { return 1; } return n * factorial(n - 1); }".to_string(),
            labels: vec!["recursion", "math"],
        },
        TrainingExample {
            id: "8".to_string(),
            text: "fn fibonacci(n: i32) -> i32 { if n <= 1 { return n; } return fibonacci(n-1) + fibonacci(n-2); }".to_string(),
            labels: vec!["recursion", "math"],
        },
    ];

    println!("JEV Training Data Selection Example");
    println!("====================================\n");
    println!("Total examples: {}", examples.len());
    println!("(Some are intentionally redundant)\n");

    // Run JEV selection (requires embedding model path)
    let config = SelectionConfig {
        embedding_model: "/path/to/embedding-model.gguf".to_string(),
        constraint_threshold: 0.5,
        max_examples: None,
    };

    match select_minimal_dataset(&examples, &config) {
        Ok(result) => {
            let metrics = evaluate_selection(&result.selected, &result.deferred);
            println!("{}", metrics);

            println!("\nSelected examples (novel constraints):");
            for example in &result.selected {
                println!("  {}: {}", example.id, example.text);
            }

            println!("\nDeferred examples (redundant constraints):");
            for deferred in &result.deferred {
                println!("  {} - Reason: {}", deferred.example.id, deferred.reason);
            }
        }
        Err(e) => {
            println!("Error running JEV selection: {}", e);
            println!("\nNote: Requires embedding model at specified path.");
            println!("Usage: cargo run --example jav_training_data_selection -- <embedding-model-path>");
        }
    }
}