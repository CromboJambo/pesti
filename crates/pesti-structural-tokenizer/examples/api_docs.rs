use pesti_structural_tokenizer::{StructuralTokenizer, fold};

fn extract_function_docs(code: &str) -> Vec<String> {
    let tokenizer = StructuralTokenizer::new();
    let tokens = match tokenizer.tokenize(code) {
        Ok(t) => t,
        Err(_) => return vec![],
    };

    let folded = match fold(tokens) {
        Some(f) => f,
        None => return vec![],
    };

    let mut docs = Vec::new();
    collect_functions(&folded, &mut docs);
    docs
}

fn collect_functions(span: &pesti_structural_tokenizer::Span, docs: &mut Vec<String>) {
    for item in &span.body {
        match item {
            pesti_structural_tokenizer::SpanItem::Token(tok) => {
                // Look for function declarations
                if matches!(&tok.kind, pesti_structural_tokenizer::TokenKind::Keyword(k) if k == "fn") {
                    // Collect until we hit the opening brace
                    let mut sig = String::new();
                    collect_signature(span, &mut sig);
                    docs.push(sig);
                }
            }
            pesti_structural_tokenizer::SpanItem::Span(child) => {
                collect_functions(child, docs);
            }
        }
    }
}

fn collect_signature(span: &pesti_structural_tokenizer::Span, sig: &mut String) {
    for item in &span.body {
        match item {
            pesti_structural_tokenizer::SpanItem::Token(tok) => {
                if tok.kind == pesti_structural_tokenizer::TokenKind::BraceOpen {
                    break;
                }
                sig.push_str(&tok.text);
            }
            _ => {}
        }
    }
}

fn main() {
    let code = r#"
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn multiply(a: i32, b: i32) -> i32 {
    return a * b;
}

fn main() {
    let result = add(1, 2);
}
"#;

    let docs = extract_function_docs(code);

    println!("## API Documentation\n");
    for doc in docs {
        println!("- `{}`", doc.trim());
    }
}
