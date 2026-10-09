//! Tests for the semantic annotation layer.

use pesti_structural_tokenizer::semantic::{
    analyze_errors, analyze_mutability, analyze_ownership, annotate_all, project_annotations,
    ErrorHandling, MutKind, OwnershipMode, SemanticTag,
};

#[test]
fn test_ownership_owned_binding() {
    let src = "let x = String::new();";
    let anns = analyze_ownership(src);
    // Should detect owned binding
    assert!(!anns.is_empty(), "Should annotate owned binding");
    
    let found_owned = anns.values().any(|a| a.mode == OwnershipMode::Owned);
    assert!(found_owned, "Should have at least one Owned annotation");
}

#[test]
fn test_ownership_borrowed_binding() {
    let src = "let x: &str = \"hello\";";
    let anns = analyze_ownership(src);
    
    let found_borrowed = anns.values().any(|a| a.mode == OwnershipMode::Borrowed);
    assert!(found_borrowed, "Should detect borrowed binding");
}

#[test]
fn test_ownership_mutably_borrowed_binding() {
    let src = "let x: &mut i32 = &mut 42;";
    let anns = analyze_ownership(src);
    
    let found_mut = anns.values().any(|a| a.mode == OwnershipMode::MutablyBorrowed);
    assert!(found_mut, "Should detect mutably borrowed binding");
}

#[test]
fn test_error_handling_propagated() {
    let src = "fn foo() -> Result<(), Error> { bar()?; Ok(()) }";
    let anns = analyze_errors(src);
    
    let found_propagated = anns.values().any(|a| a.handling == ErrorHandling::Propagated);
    assert!(found_propagated, "Should detect ? propagation");
}

#[test]
fn test_error_handling_unwrap() {
    let src = "let x = result.unwrap();";
    let anns = analyze_errors(src);
    
    let found_unwrapped = anns.values().any(|a| a.handling == ErrorHandling::Unwrapped);
    assert!(found_unwrapped, "Should detect .unwrap()");
}

#[test]
fn test_error_handling_expect() {
    let src = "let x = result.expect(\"msg\");";
    let anns = analyze_errors(src);
    
    let found_expected = anns.values().any(|a| a.handling == ErrorHandling::Expected);
    assert!(found_expected, "Should detect .expect()");
}

#[test]
fn test_mutability_let_binding() {
    let src = "let x = 42;";
    let anns = analyze_mutability(src);
    
    let found_immutable = anns.values().any(|a| !a.mutable && a.kind == MutKind::Binding);
    assert!(found_immutable, "Should detect immutable binding");
}

#[test]
fn test_mutability_let_mut_binding() {
    let src = "let mut x = 42;";
    let anns = analyze_mutability(src);
    
    let found_mutable = anns.values().any(|a| a.mutable && a.kind == MutKind::Binding);
    assert!(found_mutable, "Should detect mutable binding");
}

#[test]
fn test_semantic_tag_encode_decode_ownership() {
    let tag = SemanticTag::encode(Some(OwnershipMode::Owned), None, false, 2);
    assert_eq!(tag.ownership(), Some(OwnershipMode::Owned));
    
    let tag2 = SemanticTag::encode(Some(OwnershipMode::Borrowed), None, false, 1);
    assert_eq!(tag2.ownership(), Some(OwnershipMode::Borrowed));
    
    let tag3 = SemanticTag::encode(Some(OwnershipMode::MutablyBorrowed), None, false, 0);
    assert_eq!(tag3.ownership(), Some(OwnershipMode::MutablyBorrowed));
}

#[test]
fn test_semantic_tag_encode_decode_error_handling() {
    let tag = SemanticTag::encode(None, Some(ErrorHandling::Propagated), false, 2);
    assert_eq!(tag.error_handling(), Some(ErrorHandling::Propagated));
    
    let tag2 = SemanticTag::encode(None, Some(ErrorHandling::Unwrapped), false, 1);
    assert_eq!(tag2.error_handling(), Some(ErrorHandling::Unwrapped));
}

#[test]
fn test_semantic_tag_confidence_bit() {
    let low_conf = SemanticTag::encode(Some(OwnershipMode::Owned), None, false, 0);
    assert!(!low_conf.is_proven());
    
    let high_conf = SemanticTag::encode(Some(OwnershipMode::Owned), None, false, 2);
    assert!(high_conf.is_proven());
}

#[test]
fn test_semantic_tag_mutability_bit() {
    let immutable = SemanticTag::encode(Some(OwnershipMode::Owned), None, false, 2);
    assert!(!immutable.is_mutable());
    
    let mutable = SemanticTag::encode(Some(OwnershipMode::Owned), None, true, 2);
    assert!(mutable.is_mutable());
}

#[test]
fn test_annotate_all_integration() {
    let src = r#"
        fn process(data: Vec<u8>) -> Result<String, Error> {
            let result = transform(&data)?;
            Ok(result)
        }
    "#;
    
    let anns = annotate_all(src, 10);
    
    assert_eq!(anns.tags.len(), 10, "Should produce tag for each token position");
    
    // Should have detected some ownership annotations
    assert!(!anns.ownership.is_empty() || !anns.errors.is_empty(), 
            "Should detect at least one semantic property");
}

#[test]
fn test_project_annotations() {
    let src = "let x = 42;";
    let anns = annotate_all(src, 3);
    
    let tokens = vec![1, 2, 3];
    let projected = project_annotations(tokens, &anns);
    
    assert_eq!(projected.len(), 3);
    for (tok, tag) in &projected {
        assert!(*tok > 0);
        // Tags should be valid u8 values
        assert!(tag.value <= 255);
    }
}

#[test]
fn test_complex_program_analysis() {
    let src = r#"
        use std::fs::File;
        use std::io::{self, Read};

        fn read_config(path: &str) -> Result<Config, io::Error> {
            let mut file = File::open(path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            Ok(Config::parse(contents))
        }

        struct Config {
            value: i32,
        }

        impl Config {
            fn parse(s: String) -> Self {
                Config { value: 0 }
            }
        }
    "#;
    
    let ownership = analyze_ownership(src);
    let errors = analyze_errors(src);
    let mutability = analyze_mutability(src);
    
    // Should detect File::open's ? propagation
    assert!(!errors.is_empty(), "Should detect error handling patterns");
    
    // Should detect mutable bindings
    assert!(mutability.values().any(|a| a.mutable), 
            "Should detect mutable bindings (file, contents)");
    
    // Print summary for debugging
    eprintln!("Ownership annotations: {}", ownership.len());
    eprintln!("Error handling annotations: {}", errors.len());
    eprintln!("Mutability annotations: {}", mutability.len());
}
