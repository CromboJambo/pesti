//! # pesti-structural
//!
//! PEG-based structural tokenization with embedding features.
//!
//! This crate implements the formal PEG grammar format for pesti's structural
//! tokenizers, grounded in Kripke–Platek set theory. The grammar is a first-class
//! finite data object `(N, Σ, P, e_S)` — inspectable, serializable, and verifiable.
//!
//! ## Core Concepts
//!
//! - **StructuralGrammar**: The 4-tuple of nonterminals, terminals, productions, and start rule
//! - **Expr**: PEG expression forms (sequence, ordered choice, optional, repeat, capture)
//! - **ParseTree**: Result of applying a grammar to a token stream, with provenance
//! - **Validation**: Well-foundedness, reachability, and determinism checks
//!
//! ## Usage
//!
//! ```no_run
//! use pesti_structural::{parse_peg, validate};
//!
//! let source = r#"grammar rust;
//! Start = { 'fn' name: Ident }
//! "#;
//!
//! let grammar = parse_peg(source).unwrap();
//! let result = validate(&grammar);
//! assert!(result.valid);
//! ```

pub mod expr;
pub mod grammar;
pub mod parser;
pub mod tree;
pub mod validate;

// Re-export key types at crate root for ergonomic use.
pub use expr::Expr;
pub use grammar::{Nonterminal, RuleAnnotation, StructuralGrammar, TerminalKind};
pub use parser::{parse_peg, parse_peg_file, ParseError};
pub use tree::{Span, TreeNode};
pub use validate::{validate, ValidationError, ValidationResult};

/// The Lexer trait: converts raw source text into a stream of terminal tokens.
///
/// Each language provides its own lexer implementation. The structural grammar
/// operates on terminal *kinds*, never on raw characters — this separation is
/// what makes grammars portable across lexers.
pub trait Lexer {
    /// Tokenize source text into a sequence of (terminal kind, span) pairs.
    fn tokenize(&self, source: &str) -> Vec<(TerminalKind, Span)>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_end_to_end_parse_and_validate() {
        let src = r#"grammar test;
Start = { 'fn' name: Ident params: '(' ')' }
"#;
        let grammar = parse_peg(src).unwrap();
        let result = validate(&grammar);
        assert!(result.valid, "errors: {:?}", result.errors);
    }

    #[test]
    fn test_json_serialization_roundtrip() {
        let src = r#"grammar test;
Start = { 'fn' name: Ident }
"#;
        let grammar = parse_peg(src).unwrap();
        let json = grammar.to_json().unwrap();
        let restored = StructuralGrammar::from_json(&json).unwrap();
        assert_eq!(grammar.name, restored.name);
        assert_eq!(grammar.start, restored.start);
    }
}
