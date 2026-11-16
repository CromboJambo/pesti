//! PEG evaluator: applies a structural grammar to a token stream.
//!
//! This module implements the recursive descent matching engine that takes
//! a validated `StructuralGrammar` and a sequence of lexical tokens, then
//! produces an annotated parse tree (`TreeNode`).
//!
//! The evaluator follows strict PEG semantics:
//! - Ordered choice: first alternative that matches wins
//! - No backtracking ambiguity: deterministic by construction
//! - Left recursion is rejected at validation time (guaranteed termination)

use std::fmt;

use crate::expr::Expr;
use crate::grammar::{Nonterminal, StructuralGrammar, TerminalKind};
use crate::tree::{Span, TreeNode};

/// A lexical token produced by a language-specific lexer.
#[derive(Debug, Clone)]
pub struct Token {
    /// The terminal kind (e.g., Ident, Keyword("fn"), Punct("{")).
    pub kind: TerminalKind,
    /// The raw text of the token.
    pub text: String,
    /// Source span covered by this token.
    pub span: Span,
}

impl Token {
    /// Create a new token.
    pub fn new(kind: TerminalKind, text: impl Into<String>, span: Span) -> Self {
        Self {
            kind,
            text: text.into(),
            span,
        }
    }
}

/// A parse error with location and context information.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Human-readable description of what went wrong.
    pub message: String,
    /// Source position where the error occurred.
    pub span: Span,
    /// What was expected at this position (for error messages).
    pub expected: Vec<String>,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)?;
        if !self.expected.is_empty() {
            write!(f, " (expected: {})", self.expected.join(" | "))?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// Result of parsing a token stream with a grammar.
#[derive(Debug)]
pub struct ParseResult {
    /// The parse tree, if the entire input was consumed successfully.
    pub tree: Option<TreeNode>,
    /// Any errors encountered during parsing.
    pub errors: Vec<ParseError>,
}

/// Internal match result: either a successful match with the resulting node
/// and new position, or a failure (which is normal in PEG backtracking).
type MatchResult = Result<(TreeNode, usize), Vec<String>>;

/// PEG evaluator that applies a grammar to a token stream.
pub struct PegEvaluator<'a> {
    grammar: &'a StructuralGrammar,
    tokens: &'a [Token],
}

impl<'a> PegEvaluator<'a> {
    /// Create a new evaluator for the given grammar and token stream.
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(grammar: &'a StructuralGrammar, tokens: &'a [Token]) -> Self {
        Self { grammar, tokens }
    }

    /// Parse the entire token stream starting from the grammar's start rule.
    ///
    /// Returns a `ParseResult` containing the parse tree (if successful) and
    /// any errors encountered.
    pub fn parse(&self) -> ParseResult {
        let start_name = &self.grammar.start.0;
        match self.match_nonterminal(start_name, 0) {
            Ok((tree, end_pos)) => {
                if end_pos == self.tokens.len() {
                    ParseResult {
                        tree: Some(tree),
                        errors: Vec::new(),
                    }
                } else {
                    // Consumed part of the input but not all — report remaining tokens.
                    let remaining = &self.tokens[end_pos];
                    ParseResult {
                        tree: None,
                        errors: vec![ParseError {
                            message: format!(
                                "trailing tokens after complete parse ({} tokens remaining)",
                                self.tokens.len() - end_pos
                            ),
                            span: remaining.span,
                            expected: vec!["end of input".to_string()],
                        }],
                    }
                }
            }
            Err(expected) => {
                let pos = if self.tokens.is_empty() {
                    Span::new(0, 0)
                } else {
                    self.tokens[0].span
                };
                ParseResult {
                    tree: None,
                    errors: vec![ParseError {
                        message: format!("failed to parse with start rule '{}'", start_name),
                        span: pos,
                        expected,
                    }],
                }
            }
        }
    }

    /// Try to match an expression starting at the given token position.
    fn match_expr(&self, expr: &Expr, pos: usize) -> MatchResult {
        match expr {
            Expr::Terminal(kind) => self.match_terminal(kind, pos),
            Expr::Sequence(items) => self.match_sequence(items, pos),
            Expr::OrderedChoice(alts) => self.match_choice(alts, pos),
            Expr::Optional(inner) => self.match_optional(inner, pos),
            Expr::Repeat(inner) => self.match_repeat(inner, pos),
            Expr::Nonterminal(name) => self.match_nonterminal(&name.0, pos),
            Expr::Capture(name, inner) => self.match_capture(name, inner, pos),
            // CharRange is a lexer-level construct; at the PEG level it's not
            // directly matchable against tokens. Fail gracefully.
            Expr::CharRange(_, _) => Err(vec!["char range (lexer-level)".to_string()]),
        }
    }

    /// Match a terminal token of the given kind.
    fn match_terminal(&self, kind: &TerminalKind, pos: usize) -> MatchResult {
        if pos >= self.tokens.len() {
            return Err(vec![format!("terminal {:?}", kind)]);
        }

        let token = &self.tokens[pos];

        // Check if the token matches. Terminal literals ('fn', '{') match by
        // text regardless of how the lexer classified the token, following
        // pest semantics. Named kinds (Ident, IntLit) match by kind.
        let matches = match (kind, &token.kind) {
            // Exact kind match (Ident, IntLit, FloatLit, etc.)
            (a, b) if a == b => true,
            // Keyword literal: 'fn' matches any token with text "fn"
            (TerminalKind::Keyword(k), _) => token.text == *k,
            // Punct literal: '{' matches any token with text "{"
            (TerminalKind::Punct(p), _) => token.text == *p,
            _ => false,
        };

        if matches {
            let node = TreeNode::terminal(format!("{:?}", kind), token.span);
            Ok((node, pos + 1))
        } else {
            Err(vec![format!("terminal {:?}", kind)])
        }
    }

    /// Match a sequence of expressions (all must succeed in order).
    fn match_sequence(&self, items: &[Expr], pos: usize) -> MatchResult {
        let mut current_pos = pos;
        let mut children = Vec::new();
        let mut expected = Vec::new();

        for item in items {
            match self.match_expr(item, current_pos) {
                Ok((node, new_pos)) => {
                    children.push(node);
                    current_pos = new_pos;
                }
                Err(mut e) => {
                    expected.append(&mut e);
                    // In a sequence, if any item fails, the whole sequence fails.
                    // But we continue to collect what was expected for better errors.
                    break;
                }
            }
        }

        // If we matched all items, return success.
        if current_pos >= pos || children.len() == items.len() {
            // Check if we actually completed the sequence.
            if children.len() == items.len() {
                let span = Span::new(
                    self.tokens[pos].span.start,
                    self.tokens[current_pos.saturating_sub(1)].span.end,
                );
                let node = TreeNode::internal("seq", span, children);
                Ok((node, current_pos))
            } else {
                Err(expected)
            }
        } else {
            Err(expected)
        }
    }

    /// Match an ordered choice (first alternative that succeeds wins).
    fn match_choice(&self, alts: &[Expr], pos: usize) -> MatchResult {
        let mut all_expected = Vec::new();

        for alt in alts {
            match self.match_expr(alt, pos) {
                Ok(result) => return Ok(result),
                Err(mut e) => {
                    all_expected.append(&mut e);
                    // Try the next alternative.
                }
            }
        }

        Err(all_expected)
    }

    /// Match an optional expression (succeeds even if the inner fails).
    fn match_optional(&self, inner: &Expr, pos: usize) -> MatchResult {
        self.match_expr(inner, pos).map_or_else(
            |_| {
                Ok((
                    TreeNode::terminal("optional(empty)", Span::new(pos, pos)),
                    pos,
                ))
            },
            Ok,
        )
    }

    /// Match a repeat expression (zero or more times).
    ///
    /// PEG semantics: keep matching as long as the inner expression succeeds
    /// AND consumes at least one token per iteration (no empty loops).
    fn match_repeat(&self, inner: &Expr, pos: usize) -> MatchResult {
        let mut current_pos = pos;
        let mut children = Vec::new();

        loop {
            if current_pos >= self.tokens.len() {
                break;
            }

            match self.match_expr(inner, current_pos) {
                Ok((node, new_pos)) => {
                    // Must consume at least one token per iteration.
                    if new_pos <= current_pos {
                        // Empty match — stop to avoid infinite loop.
                        break;
                    }
                    children.push(node);
                    current_pos = new_pos;
                }
                Err(_) => {
                    // Failed to match — repeat is done.
                    break;
                }
            }
        }

        let span = if children.is_empty() {
            Span::new(pos, pos)
        } else {
            Span::new(
                self.tokens[pos].span.start,
                self.tokens[current_pos.saturating_sub(1)].span.end,
            )
        };
        let node = TreeNode::internal("repeat", span, children);
        Ok((node, current_pos))
    }

    /// Match a nonterminal reference (look up the rule and match its expression).
    fn match_nonterminal(&self, name: &str, pos: usize) -> MatchResult {
        let expr = self.grammar.productions.get(&Nonterminal(name.to_string()));
        match expr {
            Some(e) => {
                let (child, new_pos) = self.match_expr(e, pos)?;
                // Wrap in a node labeled with the nonterminal name for provenance.
                let span = child.span;
                let node = TreeNode::internal(name.to_string(), span, vec![child]);
                Ok((node, new_pos))
            }
            None => Err(vec![format!("undefined rule '{}'", name)]),
        }
    }

    /// Match a named capture (match the inner expression and record it).
    fn match_capture(&self, name: &str, inner: &Expr, pos: usize) -> MatchResult {
        let (child, new_pos) = self.match_expr(inner, pos)?;
        // Create a parent node that holds the capture.
        let span = child.span;
        let node = TreeNode::internal(format!("capture:{}", name), span, vec![])
            .with_capture(name.to_string(), child);
        Ok((node, new_pos))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_peg;

    fn tokenize_simple(source: &str) -> Vec<Token> {
        // Very simple tokenizer for testing: splits on whitespace.
        source
            .split_whitespace()
            .enumerate()
            .map(|(i, word)| {
                let start = i * (word.len() + 1);
                let end = start + word.len();
                let kind = if word.chars().all(|c| c.is_alphanumeric() || c == '_') {
                    TerminalKind::Ident
                } else if word.chars().all(|c| c.is_ascii_digit()) {
                    TerminalKind::IntLit
                } else {
                    TerminalKind::Punct(word.to_string())
                };
                Token::new(kind, word, Span::new(start, end))
            })
            .collect()
    }

    #[test]
    fn test_match_simple_sequence() {
        let grammar_src = r#"grammar test;
Start = { 'fn' name: Ident '(' ')' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        let tokens = tokenize_simple("fn foo ( )");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();

        assert!(result.tree.is_some(), "errors: {:?}", result.errors);
        let tree = result.tree.unwrap();
        assert_eq!(tree.label, "Start");
    }

    #[test]
    fn test_match_failure() {
        let grammar_src = r#"grammar test;
Start = { 'fn' name: Ident '(' ')' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        // Missing the function name.
        let tokens = tokenize_simple("fn ( )");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();

        assert!(result.tree.is_none());
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_match_optional() {
        let grammar_src = r#"grammar test;
Start = { 'fn' name: Ident ('pub')? }
"#;
        let grammar = parse_peg(grammar_src).unwrap();

        // With optional present.
        let tokens = tokenize_simple("fn foo pub");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();
        assert!(result.tree.is_some(), "errors: {:?}", result.errors);

        // Without optional.
        let tokens = tokenize_simple("fn foo");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();
        assert!(result.tree.is_some(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_match_repeat() {
        let grammar_src = r#"grammar test;
Start = { item: Ident* }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        let tokens = tokenize_simple("a b c d");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();

        assert!(result.tree.is_some(), "errors: {:?}", result.errors);
    }

    #[test]
    fn test_match_ordered_choice() {
        let grammar_src = r#"grammar test;
Start = { kind: 'foo' / 'bar' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();

        // First alternative matches.
        let tokens = tokenize_simple("foo");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();
        assert!(result.tree.is_some(), "errors: {:?}", result.errors);

        // Second alternative matches.
        let tokens = tokenize_simple("bar");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();
        assert!(result.tree.is_some(), "errors: {:?}", result.errors);

        // Neither matches.
        let tokens = tokenize_simple("baz");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();
        assert!(result.tree.is_none());
    }

    #[test]
    fn test_match_capture() {
        let grammar_src = r#"grammar test;
Start = { 'fn' name: Ident '(' ')' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        let tokens = tokenize_simple("fn foo ( )");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();

        assert!(result.tree.is_some());
        let tree = result.tree.unwrap();
        // The capture should be somewhere in the tree.
        let formatted = tree.pretty("fn foo ( )");
        assert!(formatted.contains("capture:name"));
    }

    #[test]
    fn test_trailing_tokens_error() {
        let grammar_src = r#"grammar test;
Start = { 'fn' name: Ident '(' ')' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        // Extra token at the end.
        let tokens = tokenize_simple("fn foo ( ) extra");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();

        assert!(result.tree.is_none());
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].message.contains("trailing tokens"));
    }
}
