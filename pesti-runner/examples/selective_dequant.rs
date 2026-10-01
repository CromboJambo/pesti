//! Selective Dequantization Analysis
//!
//! Measures quantization sensitivity across different Rust code generation tasks.
//! Identifies which task types benefit most from higher precision layers.
//!
//! Usage: cargo run --example selective_dequant [model_q4] [model_q5] [model_q8]

use std::path::PathBuf;
use std::time::Instant;

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: {} <model_q4> <model_q5> <model_q8>", args[0]);
        std::process::exit(1);
    }

    let models = vec![
        ("Q4_K_M", PathBuf::from(&args[1])),
        ("Q5_K_M", PathBuf::from(&args[2])),
        ("Q8_0", PathBuf::from(&args[3])),
    ];

    // Diverse set of Rust code generation tasks
    let tasks = vec![
        ("prime_check", "fn is_prime(n: u32) -> bool {\n"),
        ("fibonacci", "fn fib(n: u32) -> u32 {\n"),
        ("string_reverse", "fn reverse(s: &str) -> String {\n"),
        ("vector_sort", "fn sort(v: &mut Vec<i32>) {\n"),
        ("linked_list_push", "impl LinkedList {\n    fn push(&mut self, val: i32) {\n"),
        ("hashmap_insert", "fn insert(map: &mut HashMap<String, i32>, k: String, v: i32) {\n"),
        ("error_handling", "fn read_file(path: &str) -> Result<String, io::Error> {\n"),
        ("async_task", "async fn process(data: Vec<u8>) -> usize {\n"),
    ];

    println!("=== Selective Dequantization Sensitivity Analysis ===");
    println!("Tasks: {}\n", tasks.len());

    // Evaluate each quantization level
    let mut results: Vec<(String, Vec<(String, f64)>)> = Vec::new();
    for (name, path) in &models {
        println!("Evaluating {}...", name);
        let task_results = evaluate_tasks(name, path, &tasks)?;
        results.push((name.to_string(), task_results));
    }

    // Analyze sensitivity
    let q4_scores = &results[0].1;
    let q5_scores = &results[1].1;
    let q8_scores = &results[2].1;

    analyze_sensitivity(&tasks, q4_scores, q5_scores, q8_scores);

    Ok(())
}

fn evaluate_tasks(
    model_name: &str,
    model_path: &PathBuf,
    tasks: &[(&str, &str)],
) -> anyhow::Result<Vec<(String, f64)>> {
    let tokenizer = StructuralTokenizer::new();
    let mut results = Vec::new();

    for (task_name, prompt) in tasks {
        // Fresh runner per task to reset KV cache
        let runner = LlamaRunner::builder(model_path).n_ctx(2048).build()?;

        let start = Instant::now();
        let result = runner.generate(prompt, &SamplingConfig::precise())?;
        let elapsed = start.elapsed().as_secs_f64();

        // Score structural quality
        let code = extract_first_item(&result.text);
        let wrapped = format!("fn main() {{\n{}\n}}", code);

        let score = match tokenizer.tokenize(&wrapped) {
            Ok(tokens) => {
                let mut s = 0.0;
                if tokens.iter().any(|t| matches!(t.kind, TokenKind::FnDecl)) {
                    s += 0.25;
                }
                if tokens.iter().any(|t| matches!(t.kind, TokenKind::ReturnExpr)) {
                    s += 0.25;
                }
                if tokens.iter().any(|t| matches!(t.kind, TokenKind::BlockStart)) {
                    s += 0.25;
                }
                if tokens.iter().any(|t| matches!(t.kind, TokenKind::IfElse)) {
                    s += 0.25;
                }
                s
            }
            Err(_) => 0.0,
        };

        results.push((task_name.to_string(), score));

        print!(
            "  {}: {:.0}% ({} tok/s)\n",
            task_name,
            score * 100.0,
            result.generated_tokens as f64 / elapsed
        );
    }

    Ok(results)
}

fn analyze_sensitivity(
    tasks: &[(&str, &str)],
    q4_scores: &[(String, f64)],
    q5_scores: &[(String, f64)],
    q8_scores: &[(String, f64)],
) {
    println!("\n=== Quantization Sensitivity by Task ===");
    println!("Task\tQ4\tQ5\tQ8\tQ4→Q8 Δ");
    println!("{}", "-".repeat(60));

    for (i, (task_name, _)) in tasks.iter().enumerate() {
        let s4 = q4_scores[i].1;
        let s5 = q5_scores[i].1;
        let s8 = q8_scores[i].1;
        let delta = s8 - s4;

        println!(
            "{}\t{:.0}%\t{:.0}%\t{:.0}%\t{:+.0}%",
            task_name, s4 * 100.0, s5 * 100.0, s8 * 100.0, delta * 100.0
        );

        // Identify high-sensitivity tasks that benefit from dequantization
        if delta > 0.25 {
            println!("  → HIGH SENSITIVITY: benefits significantly from higher precision");
        }
    }

    // Summary recommendation
    println!("\n=== Selective Dequantization Recommendations ===");

    let mut high_sensitivity = Vec::new();
    for (i, (task_name, _)) in tasks.iter().enumerate() {
        if q8_scores[i].1 - q4_scores[i].1 > 0.25 {
            high_sensitivity.push(task_name.to_string());
        }
    }

    if high_sensitivity.is_empty() {
        println!("All tasks show similar quantization robustness.");
        println!("Full Q4_K_M may be sufficient for this model/task set.");
    } else {
        println!("Tasks most affected by quantization (consider higher precision):");
        for task in &high_sensitivity {
            println!("  - {}", task);
        }
        println!("\nStrategy: Keep attention layers at Q5/Q8, quantize FFN layers to Q4.");
        println!("Focus dequantization on early layers that establish code structure.");
    }
}

fn extract_first_item(text: &str) -> String {
    let mut depth = 0;
    for (i, ch) in text.chars().enumerate() {
        if ch == '{' {
            depth += 1;
        } else if ch == '}' {
            depth -= 1;
            if depth == 0 {
                return text[..=i].to_string();
            }
        }
    }
    text.to_string()
}
