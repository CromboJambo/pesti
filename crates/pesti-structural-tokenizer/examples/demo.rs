//! Demo: Structural tokenization of real Rust code
//!
//! Shows how the structural tokenizer encodes syntactic boundaries
//! vs what BPE would produce.

use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

fn main() {
    let rust_code = r#"
fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    }
    let a = 0;
    let b = 1;
    for _ in 0..n - 2 {
        let temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

fn main() {
    let result = fibonacci(10);
    println!("{}", result);
}
"#;

    println!("=== Structural Tokenization Demo ===\n");
    println!("Input: {} bytes of Rust source\n", rust_code.len());

    let tokenizer = StructuralTokenizer::new();
    match tokenizer.tokenize(rust_code) {
        Ok(tokens) => {
            println!("Output: {} structural tokens\n", tokens.len());
            println!("{:<6} {:<15} {:<20} {}", "Idx", "Kind", "Text", "Span");
            println!("{}", "-".repeat(70));

            for (i, token) in tokens.iter().enumerate() {
                let kind_name = match &token.kind {
                    TokenKind::FnDecl => "fn_decl".to_string(),
                    TokenKind::LetStmt => "let_stmt".to_string(),
                    TokenKind::ReturnExpr => "return_expr".to_string(),
                    TokenKind::IfElse => "if_else".to_string(),
                    TokenKind::ForLoop => "for_loop".to_string(),
                    TokenKind::WhileLoop => "while_loop".to_string(),
                    TokenKind::MatchExpr => "match_expr".to_string(),
                    TokenKind::CallExpr => "call_expr".to_string(),
                    TokenKind::BinaryOp => "binary_op".to_string(),
                    TokenKind::UnaryOp => "unary_op".to_string(),
                    TokenKind::IntLit => "int_lit".to_string(),
                    TokenKind::FloatLit => "float_lit".to_string(),
                    TokenKind::StrLit => "str_lit".to_string(),
                    TokenKind::Ident(s) => {
                        if s.len() > 15 {
                            format!("ident({}...)", &s[..12])
                        } else {
                            format!("ident({})", s)
                        }
                    }
                    TokenKind::BlockStart => "{".to_string(),
                    TokenKind::BlockEnd => "}".to_string(),
                    other => format!("{:?}", other),
                };

                let text_str = if token.text.len() > 18 {
                    format!("{}...", &token.text[..15])
                } else {
                    token.text.clone()
                };

                println!(
                    "{:<6} {:<15} {:<20} {}-{}",
                    i, kind_name, text_str, token.span.0, token.span.1
                );
            }

            // Compare to BPE (rough estimate)
            let bpe_estimate = rust_code.len() / 4; // ~4 bytes per BPE token avg
            println!("\n=== Compression Comparison ===");
            println!("Structural tokens: {}", tokens.len());
            println!("Estimated BPE tokens: ~{}", bpe_estimate);
            println!(
                "Compression ratio: {:.1}x fewer tokens",
                bpe_estimate as f64 / tokens.len() as f64
            );
        }
        Err(e) => {
            eprintln!("Tokenization error: {}", e);
        }
    }
}
