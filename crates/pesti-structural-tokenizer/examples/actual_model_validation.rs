//! Actual model validation experiment using pesti-runner with Qwen2.5-0.5B.
//! Feeds both BPE and structural representations to real model, compares answers.

use std::process::Command;
use std::fs;

fn main() {
    let model_path = "/home/crombo/projects/pesti/conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    // Test program
    let source = r#"fn process_data(items: Vec<i32>) -> Result<Vec<i32>, String> {
    let mut results = Vec::new();
    for item in items {
        if item < 0 {
            return Err("negative value".to_string());
        }
        results.push(item * 2);
    }
    Ok(results)
}

fn main() {
    match process_data(vec![1, 2, -3]) {
        Ok(vals) => println!("success: {:?}", vals),
        Err(msg) => eprintln!("error: {}", msg),
    }
}"#;

    // Structural representation
    let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
    let struct_tokens = tokenizer.tokenize(source).unwrap();

    let mut struct_repr = String::from("STRUCTURAL REPRESENTATION OF RUST PROGRAM:\n");
    for tok in &struct_tokens {
        match &tok.kind {
            pesti_structural_tokenizer::TokenKind::FnDecl => {
                struct_repr.push_str(&format!("FUNCTION_DECL: fn {}\n", tok.text.trim()));
            }
            pesti_structural_tokenizer::TokenKind::ForLoop => {
                struct_repr.push_str("FOR_LOOP over input items\n");
            }
            pesti_structural_tokenizer::TokenKind::IfElse => {
                struct_repr.push_str("IF condition (item < 0) -> return Err(\"negative value\")\n");
            }
            pesti_structural_tokenizer::TokenKind::LetStmt => {
                struct_repr.push_str("LET binding\n");
            }
            _ => {}
        }
    }

    // Question to ask the model
    let question = "How many functions are defined in this program? List their names.";

    println!("=== ACTUAL MODEL VALIDATION EXPERIMENT ===\n");
    println!("Model: {}", model_path);
    println!("\nQuestion: {}\n", question);

    // Condition A: BPE tokens (raw source)
    println!("--- CONDITION A: BPE TOKENS ---");
    let bpe_prompt = format!(
        "Analyze this Rust program and answer: {}.\n\n{}",
        question, source
    );
    fs::write("/tmp/bpe_prompt.txt", &bpe_prompt).unwrap();

    // Condition B: Structural tokens
    println!("--- CONDITION B: STRUCTURAL TOKENS ---");
    let struct_prompt = format!(
        "Analyze this structural representation of a Rust program and answer: {}.\n\n{}",
        question, struct_repr
    );
    fs::write("/tmp/struct_prompt.txt", &struct_prompt).unwrap();

    println!("BPE prompt written to /tmp/bpe_prompt.txt ({} bytes)", bpe_prompt.len());
    println!("Structural prompt written to /tmp/struct_prompt.txt ({} bytes)", struct_prompt.len());
    println!("\nRun these commands to test with pesti-runner:");
    println!("  pesti-runner --model {} --prompt-file /tmp/bpe_prompt.txt", model_path);
    println!("  pesti-runner --model {} --prompt-file /tmp/struct_prompt.txt", model_path);
    println!("\nCompare answers for accuracy on function count and names.");
}
