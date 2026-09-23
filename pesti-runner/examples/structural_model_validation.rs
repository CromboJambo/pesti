//! Model Validation Experiment
//! 
//! Tests whether an actual LLM can answer code structure questions better from
//! structural tokens than from raw BPE tokens. Uses pesti-runner with a real GGUF model.
//!
//! Design: Give same question, two different representations, compare answers.

use std::time::Instant;
use pesti_runner::llama::{LlamaRunner, SamplingConfig};

fn main() {
    let model_path = "/home/crombo/projects/pesti/test_models/tinyllama-q8.gguf";
    
    // Small Rust program with clear structure
    let source = r#"fn factorial(n: u64) -> u64 {
    if n <= 1 { return 1; }
    n * factorial(n - 1)
}

fn main() {
    println!("{}", factorial(5));
}"#;

    // Structural representation (what our tokenizer produces)
    let structural = r#"FUNCTION_DECL: fn factorial(n: u64) -> u64
├─ IF condition: n <= 1 → return 1
└─ expression: n * recursive_call(factorial, n-1)

FUNCTION_DECL: fn main()
└─ print call with factorial(5)"#;

    let question = "How many functions are defined? What are their names and return types?";

    println!("=== MODEL VALIDATION EXPERIMENT ===");
    println!("Model: {}", model_path);
    println!();
    println!("Question: {}", question);
    println!();

    // Build runner
    let runner = LlamaRunner::builder(model_path)
        .n_ctx(2048)
        .build()
        .expect("Failed to build runner");

    let config = SamplingConfig {
        max_tokens: 100,
        temperature: 0.7,
        ..Default::default()
    };

    // CONDITION A: Raw source (BPE tokens)
    println!("--- CONDITION A: BPE Tokens (raw source) ---");
    let prompt_a = format!(
        "Analyze this Rust program and answer: {}\n\n{}",
        question, source
    );
    let t0 = Instant::now();
    let result_a = runner.generate(&prompt_a, &config).expect("generation failed");
    let elapsed_a = t0.elapsed();
    println!("Answer ({}ms):\n{}\n", elapsed_a.as_millis(), result_a.text);

    // Reset KV cache between generations
    runner.clear_kv_cache().expect("failed to clear kv cache");

    // CONDITION B: Structural tokens
    println!("--- CONDITION B: Structural Tokens ---");
    let prompt_b = format!(
        "Analyze this structural representation of a Rust program and answer: {}\n\n{}",
        question, structural
    );
    let t1 = Instant::now();
    let result_b = runner.generate(&prompt_b, &config).expect("generation failed");
    let elapsed_b = t1.elapsed();
    println!("Answer ({}ms):\n{}\n", elapsed_b.as_millis(), result_b.text);

    // Manual evaluation
    println!("=== EVALUATION ===");
    println!("Expected: 2 functions - factorial(u64)->u64, main()->()");
    println!();
    
    let a_correct = result_a.text.contains("factorial") && result_a.text.contains("main");
    let b_correct = result_b.text.contains("factorial") && result_b.text.contains("main");
    
    println!("BPE answer correct: {}", if a_correct { "YES" } else { "NO" });
    println!("Structural answer correct: {}", if b_correct { "YES" } else { "NO" });
}
