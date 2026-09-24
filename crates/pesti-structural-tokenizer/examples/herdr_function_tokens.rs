use pesti_structural_tokenizer::StructuralTokenizer;
use std::fs;

fn main() {
    let path = "/tmp/herdr/src/workspace.rs";
    let content = fs::read_to_string(path).expect("failed to read");

    let tokenizer = StructuralTokenizer::new();
    let tokens = tokenizer.tokenize(&content).expect("tokenization failed");

    // Find the first substantial function (skip tiny accessors)
    for i in 0..tokens.len() {
        if matches!(tokens[i].kind, pesti_structural_tokenizer::TokenKind::FnDecl) {
            let preview = tokens[i].text.chars().take(80).collect::<String>();
            if preview.contains("fn ") && !preview.contains("-> ()") {
                // Found a non-trivial function — print it and the next 15 tokens
                let end = std::cmp::min(i + 20, tokens.len());
                for j in i..end {
                    let tok = &tokens[j];
                    let preview = if tok.text.chars().count() > 80 {
                        format!("{}...", tok.text.chars().take(77).collect::<String>())
                    } else {
                        tok.text.clone()
                    };
                    println!("[{:>3}] {:?}: {}", j, tok.kind, preview);
                }
                break;
            }
        }
    }
}
