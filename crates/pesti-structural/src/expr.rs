//! PEG expression forms. Finite inductive type — every value has finite depth.

use serde::{Deserialize, Serialize};

use crate::grammar::{Nonterminal, TerminalKind};

/// PEG expression forms.
///
/// This is the core of the grammar: every production rule maps a nonterminal
/// to an `Expr`. The forms are deliberately minimal — they cover all practical
/// programming language structures without CFG-level nondeterminism.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    /// Match a terminal token of this kind.
    ///
    /// Examples: `Ident`, `IntLit`, `'fn'`, `'->'`.
    Terminal(TerminalKind),

    /// Sequence: all expressions must match in order (left to right).
    ///
    /// Example: `'fn' Ident '(' ')'` matches a function signature with no params.
    Sequence(Vec<Expr>),

    /// Ordered choice: first matching alternative wins (deterministic).
    ///
    /// Example: `FuncDef / StructDef / EnumDef` tries each in order.
    OrderedChoice(Vec<Expr>),

    /// Optional: matches zero or one times.
    ///
    /// Example: `visibility?` — a function may or may not have visibility.
    Optional(Box<Expr>),

    /// Repeat: matches zero or more times.
    ///
    /// Must consume at least one terminal per iteration (enforced by validation).
    /// Example: `Item*` — a module contains zero or more items.
    Repeat(Box<Expr>),

    /// Reference to another nonterminal rule.
    ///
    /// Example: `params: ParamList` references the `ParamList` rule.
    Nonterminal(Nonterminal),

    /// Named capture: label a subexpression for embedding features.
    ///
    /// Example: `name: Ident` captures the function name span.
    Capture(String, Box<Expr>),

    /// Character range: matches any single character in [start..=end].
    ///
    /// Example: `'a'..'z'` matches any lowercase ASCII letter.
    CharRange(char, char),
}

impl Expr {
    /// Returns true if this expression can match the empty string (ε).
    ///
    /// Used by validation to detect empty loops. This is a conservative
    /// approximation — it returns `true` only when we can prove ε-matching,
    /// and `false` otherwise (which may include cases where ε-matching
    /// is actually possible but hard to determine statically).
    pub fn can_match_empty(&self) -> bool {
        match self {
            Expr::Terminal(_) => false,
            Expr::Sequence(exprs) => exprs.iter().all(|e| e.can_match_empty()),
            Expr::OrderedChoice(alts) => alts.iter().any(|e| e.can_match_empty()),
            Expr::Optional(_) => true,
            Expr::Repeat(_) => true,       // Zero repetitions = empty match
            Expr::Nonterminal(_) => false, // Conservative: assume non-empty
            Expr::Capture(_, inner) => inner.can_match_empty(),
            Expr::CharRange(_, _) => false,
        }
    }

    /// Collect all nonterminal references in this expression (including nested).
    pub fn collect_nonterminals(&self, out: &mut Vec<Nonterminal>) {
        match self {
            Expr::Terminal(_) | Expr::CharRange(_, _) => {}
            Expr::Sequence(exprs) => {
                for e in exprs {
                    e.collect_nonterminals(out);
                }
            }
            Expr::OrderedChoice(alts) => {
                for e in alts {
                    e.collect_nonterminals(out);
                }
            }
            Expr::Optional(inner) | Expr::Repeat(inner) | Expr::Capture(_, inner) => {
                inner.collect_nonterminals(out);
            }
            Expr::Nonterminal(nt) => out.push(nt.clone()),
        }
    }

    /// Collect all named captures in this expression.
    pub fn collect_captures(&self, out: &mut Vec<String>) {
        match self {
            Expr::Capture(name, inner) => {
                out.push(name.clone());
                inner.collect_captures(out);
            }
            Expr::Sequence(exprs) | Expr::OrderedChoice(exprs) => {
                for e in exprs {
                    e.collect_captures(out);
                }
            }
            Expr::Optional(inner) | Expr::Repeat(inner) => inner.collect_captures(out),
            Expr::Terminal(_) | Expr::Nonterminal(_) | Expr::CharRange(_, _) => {}
        }
    }

    /// Returns the terminal kinds this expression can directly match.
    /// Used for determinism checking of ordered choices.
    pub fn first_terminals(&self, out: &mut Vec<TerminalKind>) {
        match self {
            Expr::Terminal(tk) => out.push(tk.clone()),
            Expr::Sequence(exprs) => {
                if let Some(first) = exprs.first() {
                    first.first_terminals(out);
                }
            }
            Expr::OrderedChoice(alts) => {
                for e in alts {
                    e.first_terminals(out);
                }
            }
            Expr::Optional(inner) | Expr::Repeat(inner) | Expr::Capture(_, inner) => {
                inner.first_terminals(out);
            }
            Expr::Nonterminal(_) | Expr::CharRange(_, _) => {} // Would need grammar context to resolve
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_cannot_match_empty() {
        assert!(!Expr::Terminal(TerminalKind::Ident).can_match_empty());
    }

    #[test]
    fn test_optional_can_match_empty() {
        assert!(Expr::Optional(Box::new(Expr::Terminal(TerminalKind::Ident))).can_match_empty());
    }

    #[test]
    fn test_repeat_can_match_empty() {
        assert!(Expr::Repeat(Box::new(Expr::Terminal(TerminalKind::Ident))).can_match_empty());
    }

    #[test]
    fn test_sequence_all_empty() {
        let seq = Expr::Sequence(vec![
            Expr::Optional(Box::new(Expr::Terminal(TerminalKind::Ident))),
            Expr::Optional(Box::new(Expr::Terminal(TerminalKind::IntLit))),
        ]);
        assert!(seq.can_match_empty());
    }

    #[test]
    fn test_sequence_one_nonempty() {
        let seq = Expr::Sequence(vec![
            Expr::Optional(Box::new(Expr::Terminal(TerminalKind::Ident))),
            Expr::Terminal(TerminalKind::IntLit),
        ]);
        assert!(!seq.can_match_empty());
    }

    #[test]
    fn test_ordered_choice_any_empty() {
        let choice = Expr::OrderedChoice(vec![
            Expr::Terminal(TerminalKind::Ident),
            Expr::Optional(Box::new(Expr::Terminal(TerminalKind::IntLit))),
        ]);
        assert!(choice.can_match_empty());
    }

    #[test]
    fn test_collect_nonterminals() {
        let expr = Expr::Sequence(vec![
            Expr::Nonterminal(Nonterminal("FuncDef".into())),
            Expr::Capture(
                "name".into(),
                Box::new(Expr::Nonterminal(Nonterminal("Ident".into()))),
            ),
        ]);
        let mut nts = Vec::new();
        expr.collect_nonterminals(&mut nts);
        assert_eq!(nts.len(), 2);
    }

    #[test]
    fn test_collect_captures() {
        let expr = Expr::Sequence(vec![
            Expr::Capture("a".into(), Box::new(Expr::Terminal(TerminalKind::Ident))),
            Expr::Capture(
                "b".into(),
                Box::new(Expr::Repeat(Box::new(Expr::Terminal(TerminalKind::IntLit)))),
            ),
        ]);
        let mut caps = Vec::new();
        expr.collect_captures(&mut caps);
        assert_eq!(caps, vec!["a", "b"]);
    }
}
