//! Model validation experiment: Can an LLM answer questions about code
//! structure better from structural tokens than from BPE tokens?
//!
//! Design: Take source code, create two prompts (one with BPE token stream,
//! one with structural tokens), ask the same questions to both. Compare answers.

use std::process::Command;

fn main() {
    // Test program with known structure
    let source = r#"
fn process_data(items: Vec<i32>) -> Result<Vec<i32>, String> {
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
}
"#;

    // Structural representation
    let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
    let struct_tokens = tokenizer.tokenize(source).unwrap();

    let mut struct_repr = String::from("STRUCTURAL REPRESENTATION:\n");
    for tok in &struct_tokens {
        match &tok.kind {
            pesti_structural_tokenizer::TokenKind::FnDecl => {
                struct_repr.push_str(&format!("FN_DECL: fn {} -> {}\n", 
                    tok.text.trim(), "Result<Vec<i32>, String>"));
            }
            pesti_structural_tokenizer::TokenKind::ForLoop => {
                struct_repr.push_str("FOR_LOOP over input items\n");
            }
            pesti_structural_tokenizer::TokenKind::IfElse => {
                struct_repr.push_str("IF condition (item < 0) -> return Err\n");
            }
            pesti_structural_tokenizer::TokenKind::LetStmt => {
                struct_repr.push_str("LET binding\n");
            }
            _ => {}
        }
    }

    // Questions to ask the model about this code
    let questions = [
        "What function names exist in this program?",
        "Does this program handle errors? How?",
        "What control flow structures are used?",
        "How many functions are defined?",
        "Is there a loop? What does it iterate over?",
    ];

    println!("=== MODEL VALIDATION EXPERIMENT ===\n");
    println!("Source program structure:");
    println!("{}", struct_repr);
    println!("\nQuestions to evaluate model understanding:\n");
    for (i, q) in questions.iter().enumerate() {
        println!("{}. {}", i + 1, q);
    }

    println!("\n--- EXPERIMENT DESIGN ---\n");
    println!("Condition A: Feed BPE token stream ({} tokens)", source.len());
    println!("Condition B: Feed structural representation ({} tokens)\n", struct_tokens.len());
    println!("Measure: Model accuracy in answering structural questions.");
    println!("Success criterion: Condition B >= Condition A accuracy.");

    // Note: To actually run this, you'd need to call pesti-runner or another
    // LLM backend with these two conditions and score the answers.
    // This experiment demonstrates the design; execution requires a model.
}
