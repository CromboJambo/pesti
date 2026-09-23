//! Harder model validation: control flow and relationships
//! Tests whether structural tokens help with harder reasoning tasks

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use std::time::Instant;

fn main() {
    let model_path = "/home/crombo/llama-models/Qwen3.8-9B-Q4_K_M.gguf";
    
    // Program with control flow and relationships
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
    let structural = r#"FUNCTION_DECL: fn max(a: i32, b: i32) -> i32
├─ IF condition: a > b
│  └─ RETURN: a
└─ RETURN: b

FUNCTION_DECL: fn find_max(numbers: &[i32]) -> Option<i32>
├─ LET result = None
├─ FOR n in numbers:
│  └─ MATCH result:
│     ├─ NONE → result = Some(n)
│     └─ SOME(current):
│        └─ IF n > current → result = Some(n)
└─ RETURN result

FUNCTION_DECL: fn main() -> ()
├─ LET nums = [3, 7, 2, 9, 1]
└─ MATCH find_max(nums):
   ├─ SOME(m) → print("max: {m}")
   └─ NONE → print("empty")"#;

    let questions = vec![
        ("function_count", "How many functions are defined?"),
        ("control_flow", "Does this program use loops? What kind?"),
        ("branching", "How many distinct code paths exist in find_max?"),
        ("return_types", "What does find_max return and why?"),
        ("data_flow", "How does data flow from main to the output?"),
    ];

    println!("=== HARDER MODEL VALIDATION ===\n");

    let config = SamplingConfig {
        max_tokens: 150,
        temperature: 0.3,
        ..Default::default()
    };

    for (name, question) in &questions {
        println!("--- {} ---", name);
        println!("Q: {}\n", question);

        // Fresh runner per question to avoid KV cache position issues
        let runner = LlamaRunner::builder(model_path)
            .n_ctx(2048)
            .build()
            .expect("Failed to build runner");

        // Condition A: BPE
        let prompt_a = format!("Analyze this Rust program:\n\n{}\n\n{}", source, question);
        let t0 = Instant::now();
        let result_a = runner.generate(&prompt_a, &config).expect("generation failed");
        let elapsed_a = t0.elapsed();
        println!("[BPE] ({}ms) {}", elapsed_a.as_millis(), truncate(&result_a.text, 120));

        drop(runner);

        // Fresh runner for structural condition
        let runner2 = LlamaRunner::builder(model_path)
            .n_ctx(2048)
            .build()
            .expect("Failed to build runner");

        // Condition B: Structural
        let prompt_b = format!("Analyze this structural representation:\n\n{}\n\n{}", structural, question);
        let t1 = Instant::now();
        let result_b = runner2.generate(&prompt_b, &config).expect("generation failed");
        let elapsed_b = t1.elapsed();
        println!("[STR] ({}ms) {}", elapsed_b.as_millis(), truncate(&result_b.text, 120));
        
        drop(runner2);
        println!();
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max { return s.to_string(); }
    let chars: Vec<char> = s.chars().take(max).collect();
    format!("{}...", chars.into_iter().collect::<String>())
}
