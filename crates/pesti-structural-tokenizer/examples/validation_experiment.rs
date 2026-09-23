//! Validation experiment: Can a model use structural tokens to answer
//! questions about code structure? Compares against BPE baseline.

use pesti_structural_tokenizer::{StructuralToken, StructuralTokenizer};
use std::collections::HashMap;
use tiktoken_rs::cl100k_base;

fn main() {
    let src = include_str!("fibonacci.rs");

    // Tokenize both ways
    let bpe = cl100k_base().unwrap();
    let bpe_tokens = bpe.encode_ordinary(src);
    drop(bpe);

    let tokenizer = StructuralTokenizer::new();
    let struct_tokens = tokenizer.tokenize(src).unwrap();

    println!("=== VALIDATION EXPERIMENT ===\n");
    println!(
        "Source: fibonacci.rs ({} bytes, {} lines)",
        src.len(),
        src.lines().count()
    );
    println!("BPE tokens: {}", bpe_tokens.len());
    println!("Structural tokens: {}\n", struct_tokens.len());

    // Test 1: Can we identify function names?
    println!("TEST 1: Function name identification");
    test_function_names(&struct_tokens, src);

    // Test 2: Can we identify control flow structure?
    println!("\nTEST 2: Control flow structure");
    test_control_flow(&struct_tokens);

    // Test 3: Can we reconstruct program shape from structural tokens alone?
    println!("\nTEST 3: Program shape reconstruction");
    test_program_shape(&struct_tokens);

    // Test 4: Information density per token
    println!("\nTEST 4: Information density");
    test_information_density(&bpe_tokens, &struct_tokens, src);
}

fn test_function_names(tokens: &[StructuralToken], _src: &str) {
    let mut found_fns = Vec::new();
    for tok in tokens {
        match tok.kind {
            pesti_structural_tokenizer::TokenKind::FnDecl => {
                println!("  DEBUG FnDecl text: {:?}", tok.text);
                // FnDecl text is "name(args) -> ret " — extract name before '('
                if let Some(end) = tok.text.find('(') {
                    found_fns.push(tok.text[..end].trim().to_string());
                }
            }
            _ => {}
        }
    }

    println!(
        "Functions identified from structural tokens: {:?}",
        found_fns
    );

    let expected = ["fib", "main"];
    for name in &expected {
        if found_fns.iter().any(|f| f.contains(name)) {
            println!("  ✓ Found '{}' as expected", name);
        } else {
            println!(
                "  ✗ Missing '{}' (ground truth says it should be there)",
                name
            );
        }
    }
}

fn test_control_flow(tokens: &[StructuralToken]) {
    let mut has_if = false;
    let mut has_while = false;
    let mut has_for = false;

    for tok in tokens {
        match tok.kind {
            pesti_structural_tokenizer::TokenKind::IfElse => has_if = true,
            pesti_structural_tokenizer::TokenKind::WhileLoop => has_while = true,
            pesti_structural_tokenizer::TokenKind::ForLoop => has_for = true,
            _ => {}
        }
    }

    println!("Control flow detected:");
    println!("  if/else: {}", if has_if { "✓" } else { "✗" });
    println!("  while:   {}", if has_while { "✓" } else { "✗" });
    println!("  for:     {}", if has_for { "✓" } else { "✗" });
}

fn test_program_shape(tokens: &[StructuralToken]) {
    let mut shape = Vec::new();
    for tok in tokens.iter().take(20) {
        shape.push(format!("{:?}", tok.kind));
    }

    println!("Program shape (first 20 token kinds):");
    println!("{}", shape.join(" → "));
}

fn test_information_density(bpe_tokens: &[u32], struct_tokens: &[StructuralToken], src: &str) {
    let mut bpe_vocab_used = HashMap::new();
    for t in bpe_tokens {
        *bpe_vocab_used.entry(*t).or_insert(0usize) += 1;
    }

    let mut struct_vocab_used = HashMap::new();
    for tok in struct_tokens {
        *struct_vocab_used
            .entry(format!("{:?}", tok.kind))
            .or_insert(0usize) += 1;
    }

    println!(
        "BPE vocabulary used: {} unique tokens",
        bpe_vocab_used.len()
    );
    println!(
        "Structural vocabulary used: {} unique kinds",
        struct_vocab_used.len()
    );
    println!();
    println!("Information per token (bytes):");
    println!("  BPE: {:.2}", src.len() as f64 / bpe_tokens.len() as f64);
    println!(
        "  Structural: {:.2}",
        src.len() as f64 / struct_tokens.len() as f64
    );
}
