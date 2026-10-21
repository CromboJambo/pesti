//! JEV training data selection tool.
//!
//! Finds minimal sufficient subset of training examples using constraint propagation.

use anyhow::Result;
use clap::{Arg, Command};
use pesti_jav_training::selection::{select_minimal_dataset, SelectionConfig};
use pesti_jav_training::metrics::evaluate_selection;

fn main() -> Result<()> {
    let matches = Command::new("pesti-jav-training")
        .version("0.1.0")
        .about("JEV training data selection via constraint propagation")
        .arg(
            Arg::new("input")
                .required(true)
                .help("Path to training examples file (JSON lines)")
        )
        .arg(
            Arg::new("embedding-model")
                .short('m')
                .long("embedding-model")
                .default_value("/path/to/embedding-model.gguf")
                .help("Path to embedding model GGUF file")
        )
        .arg(
            Arg::new("threshold")
                .short('t')
                .long("threshold")
                .value_name("THRESHOLD")
                .default_value("0.5")
                .help("Constraint novelty threshold (higher = stricter)")
        )
        .arg(
            Arg::new("max-examples")
                .short('n')
                .long("max-examples")
                .value_name("N")
                .help("Maximum number of examples to consider")
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .default_value("selected.json")
                .help("Output file for selected subset")
        )
        .get_matches();

    let input_path = matches.get_one::<String>("input").unwrap();
    let embedding_model = matches.get_one::<String>("embedding-model").unwrap().clone();
    let threshold = matches.get_one::<f32>("threshold").unwrap().clone();
    let max_examples = matches.get_one::<usize>("max-examples");
    let output_path = matches.get_one::<String>("output").unwrap().clone();

    // Load training examples from JSON lines
    let examples = load_examples(input_path)?;

    println!("Loaded {} training examples", examples.len());

    // Configure selection
    let config = SelectionConfig {
        embedding_model,
        constraint_threshold: threshold,
        max_examples: max_examples.map(|n| *n),
    };

    // Run JEV selection
    let result = select_minimal_dataset(&examples, &config)?;

    // Evaluate and report
    let metrics = evaluate_selection(&result.selected, &result.deferred);
    println!("{}", metrics);

    // Save selected subset
    save_examples(&output_path, &result.selected)?;
    println!("Selected {} examples saved to {}", result.selected.len(), output_path);

    Ok(())
}

/// Load training examples from JSON lines file.
fn load_examples(path: &str) -> Result<Vec<pesti_jav_training::TrainingExample>> {
    let content = std::fs::read_to_string(path)?;
    let mut examples = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let obj: serde_json::Value = serde_json::from_str(line)?;

        examples.push(pesti_jav_training::TrainingExample {
            id: obj["id"].as_str().unwrap_or("unknown").to_string(),
            text: obj["text"].as_str().unwrap_or("").to_string(),
            labels: obj["labels"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_str()).map(String::from).collect())
                .unwrap_or_default(),
        });
    }

    Ok(examples)
}

/// Save training examples to JSON lines file.
fn save_examples(path: &str, examples: &[pesti_jav_training::TrainingExample]) -> Result<()> {
    let mut content = String::new();

    for example in examples {
        let obj = serde_json::json!({
            "id": example.id,
            "text": example.text,
            "labels": example.labels,
        });
        content.push_str(&serde_json::to_string(&obj)?);
        content.push('\n');
    }

    std::fs::write(path, content)?;
    Ok(())
}