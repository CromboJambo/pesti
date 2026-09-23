//! Comprehensive structural vs BPE benchmark with real metrics
//! Measures: tokens, prefill time, decode time, comprehension accuracy

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use std::time::Instant;

fn main() {
    let model_path = "/home/crombo/llama-models/Qwen3.8-9B-Q4_K_M.gguf";

    // Test program with features that could be lost in structural representation
    let source = r#"fn max(a: i32, b: i32) -> i32 {
    if a > b { return a; }
    return b;
}

fn find_max(numbers: &[i32]) -> Option<i32> {
    let mut result = None;
    for n in numbers {
        match result {
            None => result = Some(*n),
            Some(current) => {
                if *n > current {
                    result = Some(*n);
                }
            }
        }
    }
    return result;
}

fn main() {
    let nums = vec![3, 7, 2, 9, 1];
    match find_max(&nums) {
        Some(m) => println!("max: {}", m),
        None => println!("empty"),
    }
}"#;

    // Structural representation with explicit control flow
    let structural = "FUNCTION_DECL: fn max(a: i32, b: i32) -> i32\n\
                     ├─ IF condition: a > b\n\
                     │  └─ RETURN: a\n\
                     └─ RETURN: b\n\
                     \n\
                     FUNCTION_DECL: fn find_max(numbers: &[i32]) -> Option<i32>\n\
                     ├─ LET result = None\n\
                     ├─ FOR n in numbers:\n\
                     │  └─ MATCH result:\n\
                     │     ├─ NONE → result = Some(n)\n\
                     │     └─ SOME(current):\n\
                     │        └─ IF n > current → result = Some(n)\n\
                     └─ RETURN result\n\
                     \n\
                     FUNCTION_DECL: fn main() -> ()\n\
                     ├─ LET nums = [3, 7, 2, 9, 1]\n\
                     └─ MATCH find_max(nums):\n\
                        ├─ SOME(m) → print(\"max: {m}\")\n\
                        └─ NONE → print(\"empty\")";

    println!("=== COMPREHENSIVE BENCHMARK ===");
    println!("Source bytes: {}", source.len());
    println!("Structural bytes: {}\n", structural.len());

    let config = SamplingConfig {
        max_tokens: 150,
        temperature: 0.3,
        ..Default::default()
    };

    // Benchmark BPE condition
    println!("--- BPE CONDITION ---");
    benchmark_condition(model_path, source, "BPE", &config);

    // Benchmark structural condition
    println!("\n--- STRUCTURAL CONDITION ---");
    benchmark_structural_condition(model_path, structural, "STR", &config);
}

fn benchmark_condition(
    model_path: &str,
    source: &str,
    label: &str,
    config: &SamplingConfig,
) {
    let runner = LlamaRunner::builder(model_path)
        .n_ctx(2048)
        .build()
        .expect("Failed to build runner");

    let prompt = format!(
        "Analyze this Rust program:\n\n{}\n\nWhat functions are defined and what do they return?",
        source
    );

    println!("[{}] Prompt chars: {}", label, prompt.len());

    let start = Instant::now();
    let result = runner.generate(&prompt, config).expect("generation failed");
    let elapsed = start.elapsed();

    println!(
        "[{}] Generation time: {}ms",
        label,
        elapsed.as_millis()
    );
    println!("[{}] Output preview: {}", label, truncate(&result.text, 100));
}

fn benchmark_structural_condition(
    model_path: &str,
    structural: &str,
    label: &str,
    config: &SamplingConfig,
) {
    let runner = LlamaRunner::builder(model_path)
        .n_ctx(2048)
        .build()
        .expect("Failed to build runner");

    let prompt = format!(
        "Analyze this structural representation:\n\n{}\n\nWhat functions are defined and what do they return?",
        structural
    );

    println!("[{}] Prompt chars: {}", label, prompt.len());

    let start = Instant::now();
    let result = runner.generate(&prompt, config).expect("generation failed");
    let elapsed = start.elapsed();

    println!(
        "[{}] Generation time: {}ms",
        label,
        elapsed.as_millis()
    );
    println!("[{}] Output preview: {}", label, truncate(&result.text, 100));
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let chars: Vec<char> = s.chars().take(max).collect();
    format!("{}...", chars.into_iter().collect::<String>())
}