use std::fs;
use pesti_structural_tokenizer::{Budget, StructuralTokenizer};

fn main() {
    let src = fs::read_to_string("/home/crombo/projects/active/pesti/pesti-runner/src/llama.rs").unwrap();
    let tokenizer = StructuralTokenizer::new();
    let budget = Budget { max_tokens: 100, min_body_stmts: 3 };

    // Structural only
    let structural = tokenizer.tokenize_with_budget(&src, budget.clone()).unwrap();
    let struct_tokens = structural.iter()
        .filter(|e| matches!(e, pesti_structural_tokenizer::Emission::Node(_)))
        .count();
    println!("Structural tokens: {}", struct_tokens);

    // Semantic annotations
    let semantic = tokenizer.tokenize_with_semantics(&src, budget).unwrap();
    let sem_tokens = semantic.iter()
        .filter(|e| matches!(&e.emission, pesti_structural_tokenizer::Emission::Node(_)))
        .count();
    let annotated = semantic.iter().filter(|e| e.tag.is_some()).count();

    println!("Semantic tokens: {}", sem_tokens);
    println!("Annotated positions: {}", annotated);
    println!("Compression ratio: {:.1}x", src.split_whitespace().count() as f64 / struct_tokens as f64);
}
