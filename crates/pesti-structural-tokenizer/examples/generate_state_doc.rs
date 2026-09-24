use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: {} <input.rs> <output.md>", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let output_path = &args[2];

    let content = fs::read_to_string(input_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", input_path, e));

    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&content).expect("tokenize failed");

    // Build a simple text representation of the token stream
    let mut doc = String::new();
    for tok in &tokens {
        doc.push_str(&format!("{}\n", tok.kind));
    }

    // Write markdown state doc
    let filename = PathBuf::from(input_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let mut md = format!("# Structural Analysis: {}\n\n", filename);
    md.push_str(&format!("**Lines:** {}\n\n", content.lines().count()));
    md.push_str("## Structural Tokens\n\n");
    md.push_str("```\n");
    md.push_str(&doc);
    md.push_str("```\n");

    fs::write(output_path, &md)
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", output_path, e));

    println!("State doc written to {}", output_path);
}
