use std::env;
use std::fs;
use std::process;

use pesti_structural_tokenizer::{Budget, StructuralTokenizer};

fn main() {
    let args = env::args().collect::<Vec<_>>();

    if args.len() < 3 {
        eprintln!("Usage: struct-tok <command> [args...]");
        eprintln!(
            "Commands: tokenize FILE, detokenize FILE, stats FILE, encode TEXT, decode TOKENS"
        );
        process::exit(1);
    }

    let cmd = &args[1];
    match cmd.as_str() {
        "tokenize" => {
            if args.len() < 3 {
                eprintln!("Usage: struct-tok tokenize FILE");
                process::exit(1);
            }
            let src = fs::read_to_string(&args[2]).expect("Failed to read file");
            let tokenizer = StructuralTokenizer::new();
            match tokenizer.tokenize(&src) {
                Ok(tokens) => {
                    for (i, tok) in tokens.iter().enumerate() {
                        println!("{}: {:?} \"{}\"", i, tok.kind, tok.text);
                    }
                }
                Err(e) => {
                    eprintln!("Tokenize error: {}", e);
                    process::exit(1);
                }
            }
        }
        "detokenize" => {
            if args.len() < 3 {
                eprintln!("Usage: struct-tok detokenize FILE");
                process::exit(1);
            }
            let src = fs::read_to_string(&args[2]).expect("Failed to read file");
            let tokenizer = StructuralTokenizer::new();
            match tokenizer.tokenize(&src) {
                Ok(tokens) => {
                    for tok in &tokens {
                        print!("{}", tok.text);
                    }
                    println!();
                }
                Err(e) => {
                    eprintln!("Tokenize error: {}", e);
                    process::exit(1);
                }
            }
        }
        "stats" => {
            if args.len() < 3 {
                eprintln!("Usage: struct-tok stats FILE");
                process::exit(1);
            }
            let src = fs::read_to_string(&args[2]).expect("Failed to read file");
            let tokenizer = StructuralTokenizer::new();
            match tokenizer.tokenize(&src) {
                Ok(tokens) => {
                    println!("Tokens: {}", tokens.len());
                    // Count by kind
                    let mut counts = std::collections::HashMap::new();
                    for tok in &tokens {
                        let key = format!("{:?}", tok.kind);
                        *counts.entry(key).or_insert(0usize) += 1;
                    }
                    for (kind, count) in counts.iter() {
                        println!("  {}: {}", kind, count);
                    }
                }
                Err(e) => {
                    eprintln!("Tokenize error: {}", e);
                    process::exit(1);
                }
            }
        }
        "encode" => {
            if args.len() < 3 {
                eprintln!("Usage: struct-tok encode TEXT");
                process::exit(1);
            }
            let src = &args[2];
            let tokenizer = StructuralTokenizer::new();
            match tokenizer.tokenize(src) {
                Ok(tokens) => {
                    let ids: Vec<String> = tokens
                        .iter()
                        .map(|t| {
                            // Simple encoding: kind name + hash of text
                            format!("{:?}:{}", t.kind, t.text.len())
                        })
                        .collect();
                    println!("{}", ids.join(" "));
                }
                Err(e) => {
                    eprintln!("Tokenize error: {}", e);
                    process::exit(1);
                }
            }
        }
        "decode" => {
            if args.len() < 3 {
                eprintln!("Usage: struct-tok decode TOKENS");
                process::exit(1);
            }
            let tokens_str = &args[2];
            // Simple decoding: just print back
            for tok in tokens_str.split(" ") {
                if !tok.is_empty() {
                    println!("{}", tok);
                }
            }
        }
        _ => {
            eprintln!("Unknown command: {}", cmd);
            process::exit(1);
        }
    }
}
