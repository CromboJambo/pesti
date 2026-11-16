//! Basic Rust lexer for structural tokenization.
//!
//! This module provides a simple tokenizer that converts Rust source text
//! into a sequence of `Token`s with appropriate `TerminalKind` values.
//! It's designed to be "good enough" for structural parsing — it doesn't
//! handle every edge case of the Rust grammar, but covers the common cases
//! needed for the `rust.peg` structural grammar.

use std::fmt;

use crate::evaluator::Token;
use crate::grammar::TerminalKind;
use crate::tree::Span;

/// A lexical error with position information.
#[derive(Debug, Clone)]
pub struct LexError {
    /// Description of the error.
    pub message: String,
    /// Source position where the error occurred.
    pub span: Span,
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}", self.message, self.span)
    }
}

impl std::error::Error for LexError {}

/// Common Rust keywords that map to specific `TerminalKind::Keyword` values.
const KEYWORDS: &[&str] = &[
    "fn", "struct", "enum", "impl", "trait", "mod", "use", "pub", "let", "mut", "const", "static",
    "type", "where", "match", "if", "else", "while", "for", "loop", "return", "break", "continue",
    "move", "ref", "self", "Self", "super", "crate", "async", "await", "dyn", "box", "unsafe",
    "extern", "macro", "in", "as",
];

/// A simple Rust lexer that produces structural tokens.
pub struct RustLexer<'a> {
    source: &'a str,
    pos: usize,
}

impl<'a> RustLexer<'a> {
    /// Create a new lexer for the given source.
    fn new(source: &'a str) -> Self {
        Self { source, pos: 0 }
    }

    /// Tokenize the entire source, returning all tokens and any errors.
    pub fn tokenize(&mut self) -> (Vec<Token>, Vec<LexError>) {
        let mut tokens = Vec::new();
        let mut errors = Vec::new();

        loop {
            // Skip whitespace and comments.
            self.skip_whitespace_and_comments();

            if self.peek().is_none() {
                break;
            }

            let start = self.pos;
            match self.next_token(&mut errors) {
                Some(token) => tokens.push(token),
                None => {
                    // Error was already recorded.
                    self.advance();
                }
            }
            let _ = start; // Used for error spans if needed.
        }

        (tokens, errors)
    }

    /// Get the next token without consuming it.
    fn peek(&self) -> Option<char> {
        // Get the character at the current byte offset.
        self.source.get(self.pos..)?.chars().next()
    }

    /// Peek at the character N bytes ahead (for multi-char lookahead).
    fn peek_at(&self, byte_offset: usize) -> Option<char> {
        let target = self.pos + byte_offset;
        self.source.get(target..)?.chars().next()
    }

    /// Advance the position by one character.
    fn advance(&mut self) {
        if let Some(c) = self.peek() {
            self.pos += c.len_utf8();
        }
    }

    /// Skip whitespace, comments, and attributes.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                None => break,
                Some(c) if c.is_whitespace() => self.advance(),
                Some('/') => {
                    // Check for comment.
                    let next = self.peek_at(1);
                    match next {
                        Some('/') => {
                            // Line comment: skip to end of line.
                            while let Some(c) = self.peek() {
                                if c == '\n' {
                                    break;
                                }
                                self.advance();
                            }
                        }
                        Some('*') => {
                            // Block comment: skip to closing */.
                            self.advance(); // Skip /
                            self.advance(); // Skip *
                            while let Some(c) = self.peek() {
                                if c == '*' && self.peek_at(1) == Some('/') {
                                    self.advance(); // Skip *
                                    self.advance(); // Skip /
                                    break;
                                }
                                self.advance();
                            }
                        }
                        _ => break,
                    }
                }
                Some('#') => {
                    // Check for attribute: #[...] or #![...]
                    let next = self.peek_at(1);
                    if next == Some('[') {
                        self.skip_attribute();
                    } else {
                        break;
                    }
                }
                Some(_) => break,
            }
        }
    }

    /// Skip an attribute like `#[derive(Debug)]` or `#![no_std]`.
    fn skip_attribute(&mut self) {
        self.advance(); // Skip #
        if self.peek() == Some('!') {
            self.advance(); // Skip ! (inner attribute)
        }
        if self.peek() == Some('[') {
            self.advance(); // Skip [
            let mut depth = 1;
            while let Some(c) = self.peek() {
                match c {
                    '[' | '(' | '{' => {
                        depth += 1;
                        self.advance();
                    }
                    ']' | ')' | '}' => {
                        depth -= 1;
                        if depth == 0 {
                            self.advance(); // Skip closing bracket
                            return;
                        }
                        self.advance();
                    }
                    _ => self.advance(),
                }
            }
        }
    }

    /// Lex the next token from the source.
    fn next_token(&mut self, _errors: &mut Vec<LexError>) -> Option<Token> {
        let start = self.pos;
        let c = self.peek()?;

        // Identifiers and keywords.
        if c.is_alphabetic() || c == '_' {
            return self.lex_identifier(start);
        }

        // Numbers.
        if c.is_ascii_digit() {
            return self.lex_number(start);
        }

        // String literals.
        if c == '"' {
            return self.lex_string(start);
        }

        // Character literals.
        if c == '\'' {
            return self.lex_char(start);
        }

        // Punctuation and operators.
        let punct = self.lex_punct();
        Some(Token::new(
            TerminalKind::Punct(punct.clone()),
            punct,
            Span::new(start, self.pos),
        ))
    }

    /// Lex an identifier or keyword.
    fn lex_identifier(&mut self, start: usize) -> Option<Token> {
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }

        let text: String = self.source[start..self.pos].to_string();
        let kind = if KEYWORDS.contains(&text.as_str()) {
            TerminalKind::Keyword(text.clone())
        } else {
            TerminalKind::Ident
        };

        Some(Token::new(kind, text, Span::new(start, self.pos)))
    }

    /// Lex a numeric literal.
    fn lex_number(&mut self, start: usize) -> Option<Token> {
        let mut is_float = false;

        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || c == '_' {
                self.advance();
            } else if c == '.' && !is_float {
                is_float = true;
                self.advance();
            } else if c == 'e' || c == 'E' {
                self.advance();
                if let Some(sign) = self.peek() {
                    if sign == '+' || sign == '-' {
                        self.advance();
                    }
                }
            } else {
                break;
            }
        }

        let text: String = self.source[start..self.pos].to_string();
        let kind = if is_float {
            TerminalKind::FloatLit
        } else {
            TerminalKind::IntLit
        };

        Some(Token::new(kind, text, Span::new(start, self.pos)))
    }

    /// Lex a string literal.
    fn lex_string(&mut self, start: usize) -> Option<Token> {
        self.advance(); // Skip opening quote.

        while let Some(c) = self.peek() {
            if c == '\\' {
                // Escape sequence: skip the next character.
                self.advance();
                self.advance();
            } else if c == '"' {
                self.advance(); // Skip closing quote.
                break;
            } else {
                self.advance();
            }
        }

        let text: String = self.source[start..self.pos].to_string();
        Some(Token::new(
            TerminalKind::StrLit,
            text,
            Span::new(start, self.pos),
        ))
    }

    /// Lex a character literal.
    fn lex_char(&mut self, start: usize) -> Option<Token> {
        self.advance(); // Skip opening quote.

        while let Some(c) = self.peek() {
            if c == '\\' {
                // Escape sequence: skip the next character.
                self.advance();
                self.advance();
            } else if c == '\'' {
                self.advance(); // Skip closing quote.
                break;
            } else {
                self.advance();
            }
        }

        let text: String = self.source[start..self.pos].to_string();
        Some(Token::new(
            TerminalKind::CharLit,
            text,
            Span::new(start, self.pos),
        ))
    }

    /// Lex a punctuation token (single or multi-character).
    fn lex_punct(&mut self) -> String {
        let start = self.pos;

        // Try two-character operators first.
        let two_char_ops = [
            "==", "!=", "<=", ">=", "->", "=>", "..", "::", "<<", ">>", "+=", "-=", "*=", "/=",
            "%=", "&=", "|=", "^=",
        ];

        for op in &two_char_ops {
            let end = start + op.len();
            if self.source.len() >= end && &self.source[start..end] == *op {
                self.pos = end;
                return op.to_string();
            }
        }

        // Single-character punctuation.
        let c = self.peek().unwrap_or(' ');
        self.advance();
        c.to_string()
    }
}

/// Tokenize Rust source text into a sequence of structural tokens.
///
/// This is the main entry point for lexing Rust code. It returns both the
/// tokens and any lexical errors encountered (errors don't stop tokenization).
pub fn tokenize_rust(source: &str) -> (Vec<Token>, Vec<LexError>) {
    let mut lexer = RustLexer::new(source);
    lexer.tokenize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_identifier() {
        let (tokens, errors) = tokenize_rust("foo_bar");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::Ident);
        assert_eq!(tokens[0].text, "foo_bar");
    }

    #[test]
    fn test_lex_keyword() {
        let (tokens, errors) = tokenize_rust("fn");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::Keyword("fn".to_string()));
    }

    #[test]
    fn test_lex_number() {
        let (tokens, errors) = tokenize_rust("42");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::IntLit);
        assert_eq!(tokens[0].text, "42");
    }

    #[test]
    fn test_lex_float() {
        let (tokens, errors) = tokenize_rust("3.14");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::FloatLit);
    }

    #[test]
    fn test_lex_string() {
        let (tokens, errors) = tokenize_rust(r#""hello world""#);
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::StrLit);
    }

    #[test]
    fn test_lex_punct() {
        let (tokens, errors) = tokenize_rust("=>");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::Punct("=>".to_string()));
    }

    #[test]
    fn test_lex_function_signature() {
        let source = "fn main() -> i32 { 42 }";
        let (tokens, errors) = tokenize_rust(source);
        assert!(errors.is_empty());

        // fn main ( ) -> i32 { 42 }
        assert_eq!(tokens.len(), 9);
        assert_eq!(tokens[0].text, "fn");
        assert_eq!(tokens[1].text, "main");
        assert_eq!(tokens[2].text, "(");
        assert_eq!(tokens[3].text, ")");
        assert_eq!(tokens[4].text, "->");
        assert_eq!(tokens[5].text, "i32");
        assert_eq!(tokens[6].text, "{");
        assert_eq!(tokens[7].text, "42");
        assert_eq!(tokens[8].text, "}");
    }

    #[test]
    fn test_lex_with_comments() {
        let source = "// comment\nfn main() {}";
        let (tokens, errors) = tokenize_rust(source);
        assert!(errors.is_empty());
        // Comments are skipped. Tokens: fn main ( ) { }
        assert_eq!(tokens.len(), 6);
    }

    #[test]
    fn test_lex_struct() {
        let source = "struct Point { x: i32, y: i32 }";
        let (tokens, errors) = tokenize_rust(source);
        assert!(errors.is_empty());
        // struct Point { x : i32 , y : i32 }
        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].text, "struct");
        assert_eq!(tokens[1].text, "Point");
    }
}
