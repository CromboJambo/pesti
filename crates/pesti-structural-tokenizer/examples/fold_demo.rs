use pesti_structural_tokenizer::{fold, Span, SpanItem, StructuralToken, TokenKind};

fn make_token(kind: TokenKind, text: &str) -> StructuralToken {
    StructuralToken {
        kind,
        text: text.to_string(),
        span: (0, 0),
    }
}

fn print_span_tree(span: &Span, indent: usize) {
    let prefix = "  ".repeat(indent);
    
    if span.open_token.is_some() {
        println!("{}[BLOCK OPEN]", prefix);
    } else {
        println!("{}[ROOT]", prefix);
    }
    
    for item in &span.body {
        match item {
            SpanItem::Token(tok) => {
                if !tok.text.is_empty() {
                    println!("{}  {} ({})", prefix, tok.kind, tok.text);
                }
            }
            SpanItem::Span(child) => {
                print_span_tree(child, indent + 1);
            }
        }
    }
    
    if span.close_token.is_some() {
        println!("{}[BLOCK CLOSE]", prefix);
    }
}

fn main() {
    // Demo 1: Manual fold with raw tokens (shows the fold mechanics)
    println!("=== Demo 1: Raw Fold Mechanics ===\n");
    
    let raw_tokens = vec![
        make_token(TokenKind::Keyword("if".into()), "if"),
        make_token(TokenKind::ParenOpen, "("),
        make_token(TokenKind::Ident("x".into()), "x"),
        make_token(TokenKind::ParenClose, ")"),
        make_token(TokenKind::BraceOpen, "{"),
        make_token(TokenKind::Keyword("let".into()), "let"),
        make_token(TokenKind::Ident("y".into()), "y"),
        make_token(TokenKind::BinaryOp, "="),
        make_token(TokenKind::IntLit, "1"),
        make_token(TokenKind::Semicolon, ";"),
        make_token(TokenKind::BraceOpen, "{"),
        make_token(TokenKind::Keyword("return".into()), "return"),
        make_token(TokenKind::Ident("y".into()), "y"),
        make_token(TokenKind::BinaryOp, "+"),
        make_token(TokenKind::IntLit, "1"),
        make_token(TokenKind::Semicolon, ";"),
        make_token(TokenKind::BraceClose, "}"),
        make_token(TokenKind::BraceClose, "}"),
    ];
    
    println!("Input: {} raw tokens with nested braces", raw_tokens.len());
    if let Some(folded) = fold(raw_tokens) {
        print_span_tree(&folded, 0);
        
        fn count_spans(span: &Span) -> usize {
            let mut count = 1;
            for item in &span.body {
                if let SpanItem::Span(child) = item {
                    count += count_spans(child);
                }
            }
            count
        }
        
        println!("\nFolded into {} nested span nodes", count_spans(&folded));
    }
    
    // Demo 2: Real Rust through syn tokenizer (shows structural items)
    println!("\n=== Demo 2: Syn-Based Structural Items ===\n");
    
    let tokenizer = pesti_structural_tokenizer::StructuralTokenizer::new();
    let code = "fn add(a: i32, b: i32) -> i32 { return a + b; }";
    let tokens = tokenizer.tokenize(code).unwrap();
    
    println!("Input: {} bytes of Rust", code.len());
    println!("Structural items: {}", tokens.len());
    for tok in &tokens {
        println!("  {} ({})", tok.kind, tok.text);
    }
}
