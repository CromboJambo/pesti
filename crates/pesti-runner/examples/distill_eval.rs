//! Teacher-Student Distillation Evaluation
//!
//! Measures reconstruction gap between teacher and student models on Rust code generation.
//! Uses pesti-runner for inference and pesti-structural-tokenizer for structural comparison.
//!
//! Usage: cargo run --example distill_eval [teacher_model] [student_model] [corpus_file]

use std::path::PathBuf;
use std::time::Instant;

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: {} <teacher_model> <student_model> [corpus_file]",
            args[0]
        );
        eprintln!("If no corpus file, uses built-in test prompts.");
        std::process::exit(1);
    }

    let teacher_path = PathBuf::from(&args[1]);
    let student_path = PathBuf::from(&args[2]);

    if !teacher_path.exists() {
        eprintln!("Teacher model not found: {}", teacher_path.display());
        std::process::exit(1);
    }
    if !student_path.exists() {
        eprintln!("Student model not found: {}", student_path.display());
        std::process::exit(1);
    }

    // Load corpus or use default prompts
    let prompts: Vec<String> = if args.len() >= 4 && PathBuf::from(&args[3]).exists() {
        let content = std::fs::read_to_string(&args[3])?;
        content
            .lines()
            .filter(|l| !l.is_empty())
            .map(String::from)
            .collect::<Vec<_>>()
    } else {
        default_prompts()
    };

    println!("=== Teacher-Student Distillation Evaluation ===");
    println!("Teacher: {}", teacher_path.display());
    println!("Student: {}", student_path.display());
    println!("Prompts: {}\n", prompts.len());

    // Evaluate both models
    let teacher_results = evaluate_model("teacher", &teacher_path, &prompts)?;
    let student_results = evaluate_model("student", &student_path, &prompts)?;

    // Compute reconstruction gap
    println!("\n=== Reconstruction Gap Analysis ===");
    compute_reconstruction_gap(&teacher_results, &student_results);

    Ok(())
}

fn default_prompts() -> Vec<String> {
    vec![
        "fn is_prime(n: u32) -> bool {\n".to_string(),
        "fn fibonacci(n: u32) -> u32 {\n".to_string(),
        "fn reverse_string(s: &str) -> String {\n".to_string(),
    ]
}

struct ModelResult {
    name: String,
    outputs: Vec<String>,
    throughputs: Vec<f64>,
    structural_scores: Vec<f64>,
}

fn evaluate_model(
    name: &str,
    model_path: &PathBuf,
    prompts: &[String],
) -> anyhow::Result<ModelResult> {
    println!("Evaluating {}...", name);

    let tokenizer = StructuralTokenizer::new();
    let mut outputs = Vec::new();
    let mut throughputs = Vec::new();
    let mut structural_scores = Vec::new();

    for (i, prompt) in prompts.iter().enumerate() {
        print!("  [{}] ", i + 1);

        // Fresh runner per prompt to reset KV cache
        let runner = LlamaRunner::builder(model_path).n_ctx(2048).build()?;

        let start = Instant::now();
        let result = runner.generate(prompt, &SamplingConfig::precise())?;
        let elapsed = start.elapsed().as_secs_f64();

        let throughput = result.generated_tokens as f64 / elapsed;
        throughputs.push(throughput);

        // Evaluate structural quality
        let code_to_eval = extract_first_item(&result.text);
        let wrapped_code = format!("fn main() {{\n{}\n}}", code_to_eval);

        let score = match tokenizer.tokenize(&wrapped_code) {
            Ok(tokens) => {
                let has_fn = tokens.iter().any(|t| matches!(t.kind, TokenKind::FnDecl));
                let has_return = tokens
                    .iter()
                    .any(|t| matches!(t.kind, TokenKind::ReturnExpr));
                let has_block = tokens
                    .iter()
                    .any(|t| matches!(t.kind, TokenKind::BlockStart));
                let has_if = tokens.iter().any(|t| matches!(t.kind, TokenKind::IfElse));

                let mut score = 0.0;
                if has_fn {
                    score += 0.25;
                }
                if has_return {
                    score += 0.25;
                }
                if has_block {
                    score += 0.25;
                }
                if has_if {
                    score += 0.25;
                }
                score
            }
            Err(_) => 0.0,
        };

        structural_scores.push(score);
        outputs.push(result.text.clone());

        println!("{} tok/s (struct: {:.0}%)", throughput, score * 100.0);
    }

    let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    let avg_structural = structural_scores.iter().sum::<f64>() / structural_scores.len() as f64;

    println!(
        "{}: avg {:.2} tok/s, avg structural {:.0}%",
        name,
        avg_throughput,
        avg_structural * 100.0
    );

    Ok(ModelResult {
        name: name.to_string(),
        outputs,
        throughputs,
        structural_scores,
    })
}

fn compute_reconstruction_gap(teacher: &ModelResult, student: &ModelResult) {
    println!("Prompt\tTeacher Struct\tStudent Struct\tGap");
    println!("{}", "-".repeat(60));

    let mut total_gap = 0.0;

    for (i, ts) in teacher.structural_scores.iter().enumerate() {
        let ss = student.structural_scores[i];
        let gap = ts - ss;
        total_gap += gap.abs();

        println!(
            " {}\t{:.0}%\t\t{:.0}%\t\t{:+.0}%",
            i + 1,
            ts * 100.0,
            ss * 100.0,
            gap * 100.0
        );
    }

    let avg_gap = total_gap / teacher.structural_scores.len() as f64;
    println!("\nAverage reconstruction gap: {:.0}%", avg_gap * 100.0);

    if avg_gap < 0.1 {
        println!("Student matches teacher quality.");
    } else if avg_gap < 0.25 {
        println!("Student has moderate structural degradation vs teacher.");
    } else {
        println!("Student shows significant structural degradation vs teacher.");
    }
}

/// Extract just the first top-level Rust item from generated text.
fn extract_first_item(text: &str) -> String {
    let mut depth = 0;
    let mut in_string = false;
    let mut in_char = false;
    let mut in_comment = false;
    let mut in_block_comment = false;

    for (i, ch) in text.chars().enumerate() {
        if in_block_comment {
            if ch == '*' && i + 1 < text.len() && &text[i + 1..i + 2] == "/" {
                in_block_comment = false;
            }
            continue;
        }
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            continue;
        }
        if in_string {
            if ch == '"' && text.chars().nth(i - 1) != Some('\\') {
                in_string = false;
            }
            continue;
        }
        if in_char {
            if ch == '\'' && text.chars().nth(i - 1) != Some('\\') {
                in_char = false;
            }
            continue;
        }

        if ch == '/' && i + 1 < text.len() {
            let next = &text[i + 1..i + 2];
            if next == "/" {
                in_comment = true;
                continue;
            } else if next == "*" {
                in_block_comment = true;
                continue;
            }
        }

        if ch == '"' {
            in_string = true;
            continue;
        }
        if ch == '\'' {
            in_char = true;
            continue;
        }

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
