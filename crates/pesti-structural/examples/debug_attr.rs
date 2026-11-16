use std::path::Path;
use pesti_structural::parser::parse_peg_file;
use pesti_structural::lexer::tokenize_rust;
use pesti_structural::evaluator::PegEvaluator;

fn main() {
    let grammar = parse_peg_file(Path::new("crates/pesti-structural/grammars/rust.peg")).unwrap();
    
    // Test first part of config.rs (up to the failing impl block)
    let source = std::fs::read_to_string("/tmp/config_part1.rs").unwrap();
    println!("Part 1 length: {} bytes", source.len());
    let (tokens, lex_errors) = tokenize_rust(&source);
    println!("Tokens: {}, Lex errors: {}", tokens.len(), lex_errors.len());
    
    let evaluator = PegEvaluator::new(&grammar, &tokens);
    match evaluator.parse() {
        pesti_structural::evaluator::ParseResult { tree: Some(_tree), errors, .. } => {
            if !errors.is_empty() {
                for e in &errors {
                    println!("Error: {:?}", e);
                }
            } else {
                println!("Part 1 OK");
            }
        }
        pesti_structural::evaluator::ParseResult { tree: None, errors, .. } => {
            for e in &errors {
                println!("Part 1 FAIL: {:?}", e);
            }
            if let Some(e) = errors.first() {
                let pos = e.span.start;
                let context = source.get(pos.saturating_sub(30)..pos+30).unwrap_or("");
                println!("Context: {:?}", context);
            }
        }
    }
}
