use pesti_structural::lexer::tokenize_rust;

fn main() {
    let source = r#"
#[test]
fn test_config() {
    let config = Config {
        name: "test".to_string(),
        verbose: true,
    };
    assert!(config.verbose);
}
"#;
    
    let (tokens, errors) = tokenize_rust(source);
    println!("Tokens: {}, Errors: {}", tokens.len(), errors.len());
    for (i, t) in tokens.iter().enumerate() {
        println!("{:3}: {:?} '{}' [{}..{}]", i, t.kind, t.text, t.span.start, t.span.end);
    }
}
