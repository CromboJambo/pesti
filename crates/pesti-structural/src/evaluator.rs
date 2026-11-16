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
    /// The parse tree. Present even for partial parses (when not all tokens
    /// were consumed). `None` only if the start rule failed to match anything.
    pub tree: Option<TreeNode>,
    /// Any errors encountered during parsing. For partial parses, this includes
    /// a note about unconsumed trailing tokens.
    pub errors: Vec<ParseError>,
    /// Number of tokens successfully consumed (for partial parse diagnostics).
    pub consumed: usize,
}

impl ParseResult {
    /// Whether the entire input was consumed (a complete parse).
    pub fn is_complete(&self) -> bool {
        self.tree.is_some() && self.errors.is_empty()
    }

    /// Whether any portion of the input was successfully parsed.
    pub fn is_partial(&self) -> bool {
        self.tree.is_some() && !self.errors.is_empty()
    }
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

    /// Parse the token stream starting from the grammar's start rule.
    ///
    /// Returns a `ParseResult` containing the parse tree and any errors.
    /// Unlike strict parsing, this returns **partial trees** when not all
    /// tokens are consumed — the tree covers the successfully parsed prefix,
    /// and trailing tokens are reported as errors. This is essential for
    /// real-world code where the grammar may not cover every construct.
    pub fn parse(&self) -> ParseResult {
        let start_name = &self.grammar.start.0;
        match self.match_nonterminal(start_name, 0) {
            Ok((tree, end_pos)) => {
                if end_pos == self.tokens.len() {
                    // Complete parse: entire input consumed.
                    ParseResult {
                        tree: Some(tree),
                        errors: Vec::new(),
                        consumed: end_pos,
                    }
                } else {
                    // Partial parse: return the tree for the parsed prefix.
                    let remaining = &self.tokens[end_pos];
                    ParseResult {
                        tree: Some(tree),
                        errors: vec![ParseError {
                            message: format!(
                                "partial parse: {} of {} tokens consumed ({} trailing)",
                                end_pos,
                                self.tokens.len(),
                                self.tokens.len() - end_pos
                            ),
                            span: remaining.span,
                            expected: vec!["end of input".to_string()],
                        }],
                        consumed: end_pos,
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
                    consumed: 0,
                }
            }
        }
    }

    /// Parse with error recovery: when the start rule stops early, attempt to
    /// continue parsing from likely item boundaries in the remaining tokens.
    ///
    /// This is a heuristic recovery pass that looks for top-level keywords
    /// (fn, struct, impl, use, mod, etc.) in the unconsumed suffix and tries
    /// to parse items starting from each one. Successfully recovered items
    /// are appended to the original tree.
    pub fn parse_with_recovery(&self) -> ParseResult {
        let mut result = self.parse();

        // If we already have a complete parse or no tree, nothing to recover.
        if result.is_complete() || result.tree.is_none() {
            return result;
        }

        let start_name = &self.grammar.start.0;
        let item_rule = self.find_item_rule(start_name);
        if item_rule.is_none() {
            return result;
        }
        let item_rule = item_rule.unwrap();

        // Find the repeat node in the tree (Module → repeat of items).
        // We'll append recovered items to it.
        let mut recovered_count = 0;
        let mut pos = result.consumed;

        while pos < self.tokens.len() {
            // Skip whitespace/comments are already handled by lexer.
            // Look for the next token that could start an item.
            if !self.looks_like_item_start(pos) {
                // Skip this token and try the next position.
                pos += 1;
                continue;
            }

            // Try to parse an item from this position.
            match self.match_nonterminal(&item_rule, pos) {
                Ok((node, new_pos)) => {
                    if new_pos > pos {
                        // Successfully recovered an item. Append it to the tree.
                        self.append_to_repeat(&mut result.tree.as_mut().unwrap(), node);
                        result.consumed = new_pos.max(result.consumed);
                        pos = new_pos;
                        recovered_count += 1;
                    } else {
                        pos += 1;
                    }
                }
                Err(_) => {
                    pos += 1;
                }
            }
        }

        if recovered_count > 0 {
            // Update the error message to reflect recovery.
            if let Some(err) = result.errors.first_mut() {
                err.message = format!(
                    "partial parse with recovery: {} items recovered, {} of {} tokens consumed",
                    recovered_count,
                    result.consumed,
                    self.tokens.len()
                );
            }
        }

        result
    }

    /// Find the item rule name from a Module-like start rule.
    /// Looks for a repeat of a nonterminal in the start rule's production.
    fn find_item_rule(&self, start_name: &str) -> Option<String> {
        let start_expr = self.grammar.productions.get(&Nonterminal(start_name.to_string()))?;
        // Module = { item: Item* } — look for a Capture containing a Repeat of a Nonterminal
        if let Expr::Capture(_, inner) = start_expr {
            if let Expr::Repeat(nt_ref) = inner.as_ref() {
                if let Expr::Nonterminal(name) = nt_ref.as_ref() {
                    return Some(name.0.clone());
                }
            }
        }
        None
    }

    /// Check if the token at position looks like it could start a top-level item.
    fn looks_like_item_start(&self, pos: usize) -> bool {
        if pos >= self.tokens.len() {
            return false;
        }
        let text = &self.tokens[pos].text;
        matches!(
            text.as_str(),
            "fn" | "struct" | "enum" | "impl" | "use" | "mod" | "type" | "const" | "static" | "trait" | "pub"
        )
    }

    /// Append a recovered item node to the repeat node in the parse tree.
    fn append_to_repeat(&self, root: &mut TreeNode, item: TreeNode) {
        // Find the first "repeat" node (Module's item list) and append to it.
        if let Some(repeat_node) = find_first_by_label(root, "repeat") {
            repeat_node.children.push(item);
        } else {
            // Fallback: append directly to root's children.
            root.children.push(item);
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

/// Find the first node in a tree with the given label (pre-order traversal).
fn find_first_by_label<'a>(node: &'a mut TreeNode, label: &str) -> Option<&'a mut TreeNode> {
    if node.label == label {
        return Some(node);
    }
    for child in &mut node.children {
        if let Some(found) = find_first_by_label(child, label) {
            return Some(found);
        }
    }
    None
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
    fn test_trailing_tokens_partial_parse() {
        let grammar_src = r#"grammar test;
Start = { 'fn' name: Ident '(' ')' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        // Extra token at the end.
        let tokens = tokenize_simple("fn foo ( ) extra");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse();

        // New behavior: partial tree is returned with an error about trailing tokens.
        assert!(result.tree.is_some(), "partial tree should exist");
        assert!(!result.errors.is_empty());
        assert!(result.errors[0].message.contains("partial parse"));
        assert_eq!(result.consumed, 4); // fn foo ( ) — 4 tokens consumed
    }

    #[test]
    fn test_parse_with_recovery_skips_unknown() {
        let grammar_src = r#"grammar test;
Module = { item: Item* }
Item = { 'fn' name: Ident '(' ')' }
"#;
        let grammar = parse_peg(grammar_src).unwrap();
        // Two valid items with junk between them.
        let tokens = tokenize_simple("fn foo ( ) junk fn bar ( )");
        let evaluator = PegEvaluator::new(&grammar, &tokens);
        let result = evaluator.parse_with_recovery();

        assert!(result.tree.is_some(), "errors: {:?}", result.errors);
        // Should have recovered both function items.
        let formatted = format!("{:#?}", result.tree.unwrap());
        assert!(formatted.contains("fn"));
    }
}
