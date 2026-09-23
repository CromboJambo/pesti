//! Inspect suspicious high-compression file - show actual token content

use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};
use std::fs;

fn main() {
    let path = "/home/crombo/projects/turso/cli/opcodes_dictionary.rs";
    let src = fs::read_to_string(path).expect("read");

    println!("Source: {} bytes, {} lines", src.len(), src.lines().count());

    let tokenizer = StructuralTokenizer::new();
    match tokenizer.tokenize(&src) {
        Ok(tokens) => {
            println!("\nTokens: {}", tokens.len());
            for (i, tok) in tokens.iter().enumerate() {
                println!("\n--- Token {} ---", i);
                match &tok.kind {
                    TokenKind::StructDecl => {
                        println!("StructDecl ({} chars):", tok.text.len());
                        let preview = if tok.text.len() > 200 {
                            format!("{}...", &tok.text[..200])
                        } else {
                            tok.text.clone()
                        };
                        println!("{}", preview);
                    }
                    TokenKind::ImplBlock => {
                        println!("ImplBlock ({} chars):", tok.text.len());
                        let preview = if tok.text.len() > 200 {
                            format!("{}...", &tok.text[..200])
                        } else {
                            tok.text.clone()
                        };
                        println!("{}", preview);
                    }
                    TokenKind::ExprStmt => {
                        println!("ExprStmt ({} chars):", tok.text.len());
                        let preview = if tok.text.len() > 200 {
                            format!("{}...", &tok.text[..200])
                        } else {
                            tok.text.clone()
                        };
                        println!("{}", preview);
                    }
                    _ => {
                        println!("{:?} -> {}", tok.kind, tok.text);
                    }
                }
            }
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}