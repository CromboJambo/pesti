use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;

fn main() {
    let content = fs::read_to_string("pesti-runner/src/model.rs").unwrap();
    let tokenizer = StructuralTokenizer::new();
    
    match tokenizer.tokenize(&content) {
        Ok(tokens) => {
            println!("Total tokens: {}", tokens.len());
            
            // Count by kind
            let mut fn_count = 0;
            let mut struct_count = 0;
            for tok in &tokens {
                if let pesti_structural_tokenizer::TokenKind::FnDecl = tok.kind {
                    fn_count += 1;
                }
                if let pesti_structural_tokenizer::TokenKind::StructDecl = tok.kind {
                    struct_count += 1;
                }
            }
            
            println!("FnDecl tokens: {}", fn_count);
            println!("StructDecl tokens: {}", struct_count);
            
            // Print first 20 token kinds
            for (i, tok) in tokens.iter().take(20).enumerate() {
                println!("Token {}: {:?}", i, tok.kind);
            }
        }
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }
}
