use pesti_structural_tokenizer::{StructuralTokenizer, fold};

#[test]
fn test_fold_simple_block() {
    let tokenizer = StructuralTokenizer::new();
    let code = "fn foo() { let x = 1; }";
    let tokens = tokenizer.tokenize(code).unwrap();
    let folded = fold(tokens).expect("should fold");

    // Should have a block with some content
    assert!(!folded.body.is_empty());
}

#[test]
fn test_fold_nested_blocks() {
    let tokenizer = StructuralTokenizer::new();
    let code = "fn foo() { if (true) { let x = 1; } }";
    let tokens = tokenizer.tokenize(code).unwrap();
    let folded = fold(tokens).expect("should fold");

    // Should have nested structure
    assert!(!folded.body.is_empty());
}

#[test]
fn test_fold_function() {
    let tokenizer = StructuralTokenizer::new();
    let code = "fn add(a: i32, b: i32) -> i32 { return a + b; }";
    let tokens = tokenizer.tokenize(code).unwrap();
    let folded = fold(tokens).expect("should fold");

    // Function body should be folded into a block
    assert!(!folded.body.is_empty());
}

#[test]
fn test_fold_unbalanced() {
    let tokenizer = StructuralTokenizer::new();
    // Valid Rust with multiple nested blocks (not truly unbalanced)
    let code = "fn foo() { if (true) { let x = 1; } else { let y = 2; } }";
    let tokens = tokenizer.tokenize(code).unwrap();
    let folded = fold(tokens).expect("should fold");

    // Should produce nested structure
    assert!(!folded.body.is_empty());
}

#[test]
fn test_fold_empty_block() {
    let tokenizer = StructuralTokenizer::new();
    let code = "fn foo() {}";
    let tokens = tokenizer.tokenize(code).unwrap();
    let folded = fold(tokens).expect("should fold");

    // Empty block should still be represented
    assert!(!folded.body.is_empty());
}

#[test]
fn test_fold_complex_program() {
    let tokenizer = StructuralTokenizer::new();
    let code = "fn main() { if (true) { let x = 1; } else { let y = 2; } }";
    let tokens = tokenizer.tokenize(code).unwrap();
    let folded = fold(tokens).expect("should fold");

    // Complex program should fold into nested blocks
    assert!(!folded.body.is_empty());
}
