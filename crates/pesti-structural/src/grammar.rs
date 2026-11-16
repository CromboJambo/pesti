//! Core structural grammar types: the finite 4-tuple `(N, Σ, P, e_S)`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::expr::Expr;

/// A nonterminal symbol (rule name). Newtype to prevent confusion with terminals.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Nonterminal(pub String);

impl fmt::Display for Nonterminal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Terminal token kinds recognized by the lexer.
///
/// This is a closed set defined by the lexer implementation. The grammar
/// references these by name — it never sees raw characters.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TerminalKind {
    /// Identifier (variable names, function names, types).
    Ident,
    /// Integer literal.
    IntLit,
    /// Floating point literal.
    FloatLit,
    /// String literal.
    StrLit,
    /// Character literal.
    CharLit,
    /// Boolean literal (`true` / `false`).
    BoolLit,
    /// Keyword (language-specific: `fn`, `let`, `if`, `return`, etc.).
    Keyword(String),
    /// Punctuation/symbol token (`(`, `)`, `{`, `}`, `;`, `->`, etc.).
    Punct(String),
    /// Comment (line or block). Typically filtered out by the lexer.
    Comment,
    /// Whitespace. Typically filtered out by the lexer.
    Whitespace,
    /// End of input marker.
    Eof,
}

impl fmt::Display for TerminalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TerminalKind::Ident => write!(f, "Ident"),
            TerminalKind::IntLit => write!(f, "IntLit"),
            TerminalKind::FloatLit => write!(f, "FloatLit"),
            TerminalKind::StrLit => write!(f, "StrLit"),
            TerminalKind::CharLit => write!(f, "CharLit"),
            TerminalKind::BoolLit => write!(f, "BoolLit"),
            TerminalKind::Keyword(k) => write!(f, "\'{}\'", k),
            TerminalKind::Punct(p) => write!(f, "\'{}\'", p),
            TerminalKind::Comment => write!(f, "Comment"),
            TerminalKind::Whitespace => write!(f, "Whitespace"),
            TerminalKind::Eof => write!(f, "Eof"),
        }
    }
}

/// Embedding annotations for a rule. These tell the embedding extractor
/// which parts of the parse tree are semantically meaningful.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuleAnnotation {
    /// Semantic role of this construct: "definition", "declaration",
    /// "expression", "statement", "type", "module", etc.
    pub role: String,

    /// Named captures within this rule that should be embedded.
    /// References the `name:` labels in the production expression.
    #[serde(default)]
    pub embed_fields: Vec<String>,

    /// Similarity class for cross-language matching.
    /// Rules with the same class are considered structurally similar
    /// across languages (e.g., "declaration" matches function declarations
    /// in Rust, Python, and TypeScript).
    #[serde(default)]
    pub similarity_class: Option<String>,
}

/// A structural grammar: the finite 4-tuple `(N, Σ, P, e_S)`.
///
/// - `N`: set of nonterminals (rule names)
/// - `Σ`: terminal alphabet (implicitly defined by the lexer)
/// - `P`: productions mapping each nonterminal to a PEG expression
/// - `e_S`: start expression (the rule to begin parsing from)
///
/// All objects are finite and constructible — no impredicative circularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralGrammar {
    /// Grammar name (e.g., "rust_structural").
    pub name: String,

    /// Productions: nonterminal → expression.
    /// Uses BTreeMap for deterministic iteration order.
    pub productions: BTreeMap<Nonterminal, Expr>,

    /// Start rule: which nonterminal to begin parsing from.
    pub start: Nonterminal,

    /// Embedding annotations per rule.
    #[serde(default)]
    pub annotations: BTreeMap<Nonterminal, RuleAnnotation>,
}

impl StructuralGrammar {
    /// Create a new empty grammar with the given name and start rule.
    pub fn new(name: impl Into<String>, start: impl Into<String>) -> Self {
        let start = Nonterminal(start.into());
        Self {
            name: name.into(),
            productions: BTreeMap::new(),
            start,
            annotations: BTreeMap::new(),
        }
    }

    /// Add a production rule.
    pub fn add_rule(&mut self, nt: impl Into<String>, expr: Expr) {
        self.productions.insert(Nonterminal(nt.into()), expr);
    }

    /// Add an annotation for a rule.
    pub fn add_annotation(&mut self, nt: impl Into<String>, annotation: RuleAnnotation) {
        self.annotations.insert(Nonterminal(nt.into()), annotation);
    }

    /// Get the set of all nonterminals defined in this grammar.
    pub fn nonterminals(&self) -> BTreeSet<&Nonterminal> {
        self.productions.keys().collect()
    }

    /// Check if a nonterminal is defined.
    pub fn has_rule(&self, nt: &Nonterminal) -> bool {
        self.productions.contains_key(nt)
    }

    /// Get the expression for a rule, if it exists.
    pub fn rule(&self, nt: &Nonterminal) -> Option<&Expr> {
        self.productions.get(nt)
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_new() {
        let g = StructuralGrammar::new("test", "Start");
        assert_eq!(g.name, "test");
        assert_eq!(g.start.0, "Start");
        assert!(g.productions.is_empty());
    }

    #[test]
    fn test_add_rule() {
        let mut g = StructuralGrammar::new("test", "Start");
        g.add_rule("Start", Expr::Terminal(TerminalKind::Ident));
        assert!(g.has_rule(&Nonterminal("Start".into())));
        assert_eq!(g.nonterminals().len(), 1);
    }

    #[test]
    fn test_terminal_kind_display() {
        assert_eq!(TerminalKind::Ident.to_string(), "Ident");
        assert_eq!(TerminalKind::Keyword("fn".into()).to_string(), "'fn'");
        assert_eq!(TerminalKind::Punct("->".into()).to_string(), "'->'");
    }

    #[test]
    fn test_json_roundtrip() {
        let mut g = StructuralGrammar::new("test", "Start");
        g.add_rule(
            "Start",
            Expr::Sequence(vec![
                Expr::Terminal(TerminalKind::Keyword("fn".into())),
                Expr::Capture("name".into(), Box::new(Expr::Terminal(TerminalKind::Ident))),
            ]),
        );
        g.add_annotation(
            "Start",
            RuleAnnotation {
                role: "definition".into(),
                embed_fields: vec!["name".into()],
                similarity_class: Some("declaration".into()),
            },
        );

        let json = g.to_json().unwrap();
        let g2 = StructuralGrammar::from_json(&json).unwrap();
        assert_eq!(g.name, g2.name);
        assert_eq!(g.start, g2.start);
        assert_eq!(g.productions.len(), g2.productions.len());
        assert_eq!(g.annotations.len(), g2.annotations.len());
    }
}
