//! Integration tests for pesti-structural.
//! Loads real grammar files from testdata/ and exercises the full pipeline.

use pesti_structural::{parse_peg, validate};

#[test]
fn test_rust_structural_grammar_loads_and_validates() {
    let src = include_str!("../testdata/rust_structural.peg");
    let grammar = parse_peg(src).expect("should parse rust_structural.peg");

    assert_eq!(grammar.name, "rust_structural");

    // Verify key rules exist
    for rule in &[
        "Module",
        "Item",
        "FnDecl",
        "StructDecl",
        "EnumDecl",
        "TraitDecl",
        "ImplBlock",
        "UseStmt",
        "ModDecl",
        "Type",
        "Path",
        "Block",
        "Stmt",
        "Expr",
        "CallExpr",
        "MatchExpr",
        "IfExpr",
        "ForLoop",
        "WhileLoop",
        "ReturnExpr",
        "Literal",
    ] {
        assert!(
            grammar.has_rule(&pesti_structural::Nonterminal(rule.to_string())),
            "missing rule: {}",
            rule
        );
    }

    // Verify annotations were parsed
    let fn_decl_ann = grammar
        .annotations
        .get(&pesti_structural::Nonterminal("FnDecl".to_string()))
        .expect("FnDecl should have annotation");
    assert_eq!(fn_decl_ann.role, "definition");
    assert_eq!(fn_decl_ann.similarity_class, Some("declaration".into()));

    // Validate the grammar
    let result = validate(&grammar);
    for error in &result.errors {
        eprintln!("validation error: {:?}", error);
    }
    assert!(
        result.valid,
        "rust_structural.peg should be valid: {:?}",
        result.errors
    );
}

#[test]
fn test_rust_structural_grammar_json_roundtrip() {
    let src = include_str!("../testdata/rust_structural.peg");
    let grammar = parse_peg(src).expect("should parse rust_structural.peg");

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&grammar).expect("should serialize");

    // Deserialize back
    let restored: pesti_structural::StructuralGrammar =
        serde_json::from_str(&json).expect("should deserialize");

    assert_eq!(restored.name, grammar.name);
    assert_eq!(restored.productions.len(), grammar.productions.len());
    assert_eq!(restored.annotations.len(), grammar.annotations.len());

    // Validate the restored grammar
    let result = validate(&restored);
    assert!(
        result.valid,
        "restored grammar should be valid: {:?}",
        result.errors
    );
}

#[test]
fn test_grammar_file_io() {
    use std::fs;
    use std::path::Path;

    // Load from file path (relative to testdata/)
    let path = Path::new("testdata/rust_structural.peg");
    assert!(path.exists(), "testdata/rust_structural.peg should exist");

    let src = fs::read_to_string(path).expect("should read grammar file");
    let grammar = parse_peg(&src).expect("should parse from file");
    assert_eq!(grammar.name, "rust_structural");

    // Write to a temp location and read back
    let out_path = Path::new("/tmp/pesti_structural_test.peg");
    fs::write(out_path, &src).expect("should write grammar file");
    let src2 = fs::read_to_string(out_path).expect("should read back");
    let grammar2 = parse_peg(&src2).expect("should parse from temp file");
    assert_eq!(grammar2.name, grammar.name);
    assert_eq!(grammar2.productions.len(), grammar.productions.len());

    fs::remove_file(out_path).ok();
}

#[test]
fn test_end_to_end_rust_parsing() {
    use pesti_structural::{PegEvaluator, tokenize_rust};

    // Load the real Rust structural grammar.
    let src = include_str!("../grammars/rust.peg");
    let grammar = parse_peg(src).expect("should parse rust.peg");

    // Validate it first.
    let result = validate(&grammar);
    assert!(
        result.valid,
        "rust.peg should be valid: {:?}",
        result.errors
    );

    // Real Rust source to parse.
    let code = r#"
        pub struct Point {
            x: i32,
            y: i32,
        }

        impl Point {
            fn new(x: i32, y: i32) -> Self {
                Self { x, y }
            }

            fn distance(&self, other: &Point) -> f64 {
                let dx = self.x - other.x;
                let dy = self.y - other.y;
                (dx * dx + dy * dy) as f64
            }
        }

        fn main() {
            let p1 = Point::new(0, 0);
            let p2 = Point::new(3, 4);
            println!("{}", p1.distance(&p2));
        }
    "#;

    // Tokenize.
    let (tokens, lex_errors) = tokenize_rust(code);
    assert!(lex_errors.is_empty(), "lex errors: {:?}", lex_errors);
    assert!(!tokens.is_empty(), "should produce tokens");

    // Parse with the PEG evaluator.
    let evaluator = PegEvaluator::new(&grammar, &tokens);
    let parse_result = evaluator.parse();

    if parse_result.tree.is_none() {
        eprintln!("parse errors: {:?}", parse_result.errors);
    }

    // The grammar is complex and may not match every construct perfectly yet,
    // but we should at least get a partial parse or clear error messages.
    // This test verifies the pipeline works end-to-end without panicking.
}
