//! Compression Cycle Benchmark
//!
//! Measures the capability/compute tradeoff across quantization levels for Rust code generation.
//! Uses pesti-runner's llama.cpp FFI path with structural evaluation via pesti-structural-tokenizer.
//!
//! Usage: cargo run --example compression_cycle [model_path] [output_dir]

use std::path::PathBuf;
use std::time::Instant;

use pesti_runner::llama::{LlamaRunner, SamplingConfig};
use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

fn main() -> anyhow::Result<()> {
    let output_dir = std::env::args().nth(2)
        .unwrap_or_else(|| "compression_results".to_string());

    // Run benchmark across multiple quantization levels
    let models = vec![
        ("Q4_K_M", "/home/crombo/projects/active/pesti/test_models/tinyllama-q4.gguf"),
        ("Q5_K_M", "/home/crombo/projects/active/pesti/test_models/tinyllama-q5.gguf"),
        ("Q8_0",   "/home/crombo/projects/active/pesti/test_models/tinyllama-q8.gguf"),
    ];

    let output_dir = PathBuf::from(output_dir);
    std::fs::create_dir_all(&output_dir)?;

    println!("=== PESTI Compression Cycle Benchmark ===");
    println!("Testing across quantization levels\n");

    let mut results = Vec::new();

    for (quant_name, model_path) in models {
        let path = PathBuf::from(model_path);
        if !path.exists() {
            eprintln!("Model not found: {}", path.display());
            continue;
        }

        println!("\n--- {} ---", quant_name);
        match run_cycle(path, output_dir.clone()) {
            Ok(r) => {
                if !r.is_empty() {
                    results.push((quant_name.to_string(), r.into_iter().next().unwrap()));
                }
            }
            Err(e) => eprintln!("Error running {}: {}", quant_name, e),
        }
    }

    // Print summary table
    println!("\n=== Summary: Quantization vs. Performance ===");
    println!("{:<8} {:>10} {:>12} {:>12}", "Quant", "tok/s", "structural%", "tokens");
    println!("{}", "-".repeat(48));
    for (quant, r) in &results {
        let best = results.iter().find(|(_, x)| x.throughput == r.throughput);
        println!(
            "{:<8} {:>10.2} {:>11.0}% {:>12}",
            quant,
            r.throughput,
            r.structural_score * 100.0,
            r.tokens_generated
        );
    }

    Ok(())
}

#[derive(Debug)]
struct CycleResult {
    quantization: String,
    throughput: f64,
    structural_score: f64,
    compile_success: bool,
    tokens_generated: usize,
}

fn run_cycle(
    model_path: PathBuf,
    output_dir: PathBuf,
) -> anyhow::Result<Vec<CycleResult>> {
    let mut results = Vec::new();

    println!("Loading model...");
    let load_start = Instant::now();

    let runner = LlamaRunner::builder(&model_path)
        .n_ctx(2048)
        .build()?;

    println!("Loaded in {:.1}s", load_start.elapsed().as_secs_f64());

    // Test prompt: generate complete Rust function
    let prompt = "fn is_prime(n: u32) -> bool {\n";

    println!("\nGenerating Rust code...");
    let gen_start = Instant::now();

    let result = runner.generate(prompt, &SamplingConfig::precise())?;

    let elapsed = gen_start.elapsed().as_secs_f64();
    let throughput = result.generated_tokens as f64 / elapsed;

    println!(
        "Generated {} tokens in {:.3}s ({:.2} tok/s)",
        result.generated_tokens, elapsed, throughput
    );
    println!("Output: {}", &result.text[..std::cmp::min(200, result.text.len())]);

    // Evaluate structural quality using pesti-structural-tokenizer
    let tokenizer = StructuralTokenizer::new();
    let mut structural_score = 0.0;
    let compile_success = false; // Would require actual rustc invocation

    // Model may generate past our target function (into fn main() etc).
    // Extract just the first top-level item by finding the boundary.
    let code_to_eval = extract_first_item(&result.text);

    // Wrap in fn main() so it parses as a complete Rust item
    let wrapped_code = format!("fn main() {{\n{}\n}}", code_to_eval);

    match tokenizer.tokenize(&wrapped_code) {
        Ok(tokens) => {
            println!("Tokenized into {} tokens", tokens.len());
            // Score based on presence of expected Rust structures
            let has_fn = tokens.iter().any(|t| matches!(t.kind, TokenKind::FnDecl));
            let has_return = tokens.iter().any(|t| matches!(t.kind, TokenKind::ReturnExpr));
            let has_block = tokens.iter().any(|t| matches!(t.kind, TokenKind::BlockStart));
            let has_if = tokens.iter().any(|t| matches!(t.kind, TokenKind::IfElse));

            if has_fn {
                structural_score += 0.25;
            }
            if has_return {
                structural_score += 0.25;
            }
            if has_block {
                structural_score += 0.25;
            }
            if has_if {
                structural_score += 0.25;
            }
        }
        Err(e) => {
            println!("Tokenization error: {}", e);
        }
    }

    println!("Structural score: {:.0}%", structural_score * 100.0);

    let result_entry = CycleResult {
        quantization: "Q4_K_M".to_string(),
        throughput,
        structural_score,
        tokens_generated: result.generated_tokens,
        compile_success,
    };

    results.push(result_entry);

    Ok(results)
}

/// Extract just the first top-level Rust item from generated text.
/// The model often continues past our target function (into fn main() etc).
/// We find the closing brace that matches our opening brace depth and truncate there.
fn extract_first_item(text: &str) -> String {
    let mut depth = 0;
    let mut in_string = false;
    let mut in_char = false;
    let mut in_comment = false;
    let mut in_block_comment = false;

    for (i, ch) in text.chars().enumerate() {
        if in_block_comment {
            if ch == '*' && i + 1 < text.len() && &text[i+1..i+2] == "/" {
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
            let next = &text[i+1..i+2];
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
                // Found the end of the first top-level item
                return text[..=i].to_string();
            }
        }
    }

    // If we never hit depth 0, return everything (malformed)
    text.to_string()
}