//! Language-aware lexer for structural tokenization.
//!
//! This module provides simple tokenizers that convert source text into a
//! sequence of `Token`s with appropriate `TerminalKind` values. Each language
//! (Rust, Python) has its own lexer tailored to its syntax — comments, strings,
//! keywords, and operators differ significantly between languages.
//!
//! Design: "good enough" for structural parsing. We don't need perfect lexical
//! analysis; we need reliable token streams that the PEG grammars can work with.

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

// ============================================================================
// Python Lexer
// ============================================================================

/// Python keywords that map to `TerminalKind::Keyword` values.
const PYTHON_KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue",
    "def", "del", "elif", "else", "except", "finally", "for", "from", "global", "if", "import",
    "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return", "try", "while",
    "with", "yield", "match", "case",
];

/// A simple Python lexer that produces structural tokens.
pub struct PythonLexer<'a> {
    source: &'a str,
    pos: usize,
}

impl<'a> PythonLexer<'a> {
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

            match self.next_token(&mut errors) {
                Some(token) => tokens.push(token),
                None => {
                    // Error was already recorded.
                    self.advance();
                }
            }
        }

        (tokens, errors)
    }

    /// Get the character at the current position without consuming it.
    fn peek(&self) -> Option<char> {
        self.source.get(self.pos..)?.chars().next()
    }

    /// Peek at the character N bytes ahead.
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

    /// Check if the source at `pos` starts with the given string.
    fn starts_with(&self, s: &str) -> bool {
        self.source
            .get(self.pos..)
            .map_or(false, |rest| rest.starts_with(s))
    }

    /// Skip whitespace and comments.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                None => break,
                Some(c) if c.is_whitespace() => self.advance(),
                Some('#') => {
                    // Line comment: skip to end of line.
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                Some(_) => break,
            }
        }
    }

    /// Lex the next token from the source.
    fn next_token(&mut self, _errors: &mut Vec<LexError>) -> Option<Token> {
        let start = self.pos;
        let c = self.peek()?;

        // String literals (with optional prefixes: f, r, b, u, and combinations).
        if c == '"' || c == '\'' {
            return self.lex_string(start);
        }

        // String prefixes followed by a quote.
        if (c.is_ascii_alphabetic() || c == '_') && self.is_string_prefix(start) {
            return self.lex_prefixed_string(start);
        }

        // Identifiers and keywords.
        if c.is_alphabetic() || c == '_' {
            return self.lex_identifier(start);
        }

        // Numbers (including complex literals like 3j).
        if c.is_ascii_digit() {
            return self.lex_number(start);
        }

        // Punctuation and operators.
        let punct = self.lex_punct();
        Some(Token::new(
            TerminalKind::Punct(punct.clone()),
            punct,
            Span::new(start, self.pos),
        ))
    }

    /// Check if the current position starts a string prefix followed by a quote.
    fn is_string_prefix(&self, start: usize) -> bool {
        let rest = &self.source[start..];
        // Check for common prefixes: f, r, b, u, fr, rf, br, rb (case-insensitive first char)
        for prefix in ["fr", "rf", "br", "rb", "f", "r", "b", "u"] {
            if rest.len() > prefix.len() && rest[..prefix.len()].eq_ignore_ascii_case(prefix) {
                let after_prefix = &rest[prefix.len()..];
                if let Some(q) = after_prefix.chars().next() {
                    if q == '"' || q == '\'' {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Lex a string with an optional prefix (f, r, b, u, fr, etc.).
    fn lex_prefixed_string(&mut self, start: usize) -> Option<Token> {
        // Skip the prefix characters.
        let rest = &self.source[start..];
        for prefix in ["fr", "rf", "br", "rb", "f", "r", "b", "u"] {
            if rest.len() > prefix.len() && rest[..prefix.len()].eq_ignore_ascii_case(prefix) {
                let after_prefix = &rest[prefix.len()..];
                if let Some(q) = after_prefix.chars().next() {
                    if q == '"' || q == '\'' {
                        self.pos += prefix.len();
                        break;
                    }
                }
            }
        }

        // Now lex the string itself (same logic as lex_string but we're past the prefix).
        self.lex_string_body(start)
    }

    /// Lex a string literal (handles single, double, and triple quotes).
    fn lex_string(&mut self, start: usize) -> Option<Token> {
        self.lex_string_body(start)
    }

    /// Core string lexing logic: handles '...', "...", '''...''', """...""".
    fn lex_string_body(&mut self, start: usize) -> Option<Token> {
        let quote = self.peek()?;
        // Check for triple quotes.
        if self.starts_with(&format!("{}{}", quote, quote)) {
            // Triple-quoted string.
            let triple = format!("{}{}{}", quote, quote, quote);
            self.advance(); // Skip first quote
            self.advance(); // Skip second quote
            self.advance(); // Skip third quote

            while let Some(c) = self.peek() {
                if c == '\\' {
                    // Escape sequence.
                    self.advance();
                    self.advance();
                } else if self.starts_with(&triple) {
                    // Closing triple quote.
                    self.advance();
                    self.advance();
                    self.advance();
                    break;
                } else {
                    self.advance();
                }
            }
        } else {
            // Single-quoted string.
            self.advance(); // Skip opening quote.

            while let Some(c) = self.peek() {
                if c == '\\' {
                    // Escape sequence.
                    self.advance();
                    self.advance();
                } else if c == quote || c == '\n' {
                    // Closing quote or end of line (unterminated string).
                    if c == quote {
                        self.advance();
                    }
                    break;
                } else {
                    self.advance();
                }
            }
        }

        let text: String = self.source[start..self.pos].to_string();
        Some(Token::new(
            TerminalKind::StrLit,
            text,
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
        let kind = if PYTHON_KEYWORDS.contains(&text.as_str()) {
            TerminalKind::Keyword(text.clone())
        } else {
            TerminalKind::Ident
        };

        Some(Token::new(kind, text, Span::new(start, self.pos)))
    }

    /// Lex a numeric literal (int, float, complex).
    fn lex_number(&mut self, start: usize) -> Option<Token> {
        let mut is_float = false;
        let mut is_complex = false;

        // Handle hex, octal, binary prefixes.
        if self.starts_with("0x") || self.starts_with("0X") {
            self.advance();
            self.advance();
            while let Some(c) = self.peek() {
                if c.is_ascii_hexdigit() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
        } else if self.starts_with("0o") || self.starts_with("0O") {
            self.advance();
            self.advance();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
        } else if self.starts_with("0b") || self.starts_with("0B") {
            self.advance();
            self.advance();
            while let Some(c) = self.peek() {
                if c == '0' || c == '1' || c == '_' {
                    self.advance();
                } else {
                    break;
                }
            }
        } else {
            // Decimal number.
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() || c == '_' {
                    self.advance();
                } else if c == '.' && !is_float {
                    is_float = true;
                    self.advance();
                } else if (c == 'e' || c == 'E') && !is_complex {
                    self.advance();
                    if let Some(sign) = self.peek() {
                        if sign == '+' || sign == '-' {
                            self.advance();
                        }
                    }
                } else if c == 'j' || c == 'J' {
                    // Complex literal (e.g., 3j, 4.5j).
                    is_complex = true;
                    is_float = true;
                    self.advance();
                } else {
                    break;
                }
            }
        }

        let text: String = self.source[start..self.pos].to_string();
        let kind = if is_float || is_complex {
            TerminalKind::FloatLit
        } else {
            TerminalKind::IntLit
        };

        Some(Token::new(kind, text, Span::new(start, self.pos)))
    }

    /// Lex a punctuation token (single or multi-character).
    fn lex_punct(&mut self) -> String {
        let start = self.pos;

        // Three-character operators first.
        let three_char_ops = ["**=", "//=", ">>=", "<<="];
        for op in &three_char_ops {
            if self.starts_with(op) {
                self.pos = start + op.len();
                return op.to_string();
            }
        }

        // Two-character operators.
        let two_char_ops = [
            "==", "!=", "<=", ">=", "->", ":=", "**", "//", "<<", ">>", "+=", "-=", "*=", "/=",
            "%=", "&=", "|=", "^=", "@@",
        ];
        for op in &two_char_ops {
            if self.starts_with(op) {
                self.pos = start + op.len();
                return op.to_string();
            }
        }

        // Single-character punctuation.
        let c = self.peek().unwrap_or(' ');
        self.advance();
        c.to_string()
    }
}

/// Tokenize Python source text into a sequence of structural tokens.
///
/// This is the main entry point for lexing Python code. It returns both the
/// tokens and any lexical errors encountered (errors don't stop tokenization).
pub fn tokenize_python(source: &str) -> (Vec<Token>, Vec<LexError>) {
    let mut lexer = PythonLexer::new(source);
    lexer.tokenize()
}

/// Tokenize source text using the appropriate language-specific lexer.
///
/// `language` should be "rust", "python", or "typescript" (case-insensitive).
/// Unknown languages fall back to the Rust lexer for backwards compatibility.
pub fn tokenize_for_language(source: &str, language: &str) -> (Vec<Token>, Vec<LexError>) {
    match language.to_lowercase().as_str() {
        "python" | "py" => tokenize_python(source),
        "typescript" | "ts" => tokenize_rust(source), // Placeholder: TS uses Rust-style tokens for now.
        _ => tokenize_rust(source),
    }
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

    // Python lexer tests.

    #[test]
    fn test_python_lex_identifier() {
        let (tokens, errors) = tokenize_python("foo_bar");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::Ident);
        assert_eq!(tokens[0].text, "foo_bar");
    }

    #[test]
    fn test_python_lex_keyword() {
        let (tokens, errors) = tokenize_python("def");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::Keyword("def".to_string()));
    }

    #[test]
    fn test_python_lex_number() {
        let (tokens, errors) = tokenize_python("42");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::IntLit);
        assert_eq!(tokens[0].text, "42");
    }

    #[test]
    fn test_python_lex_float() {
        let (tokens, errors) = tokenize_python("3.14");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::FloatLit);
    }

    #[test]
    fn test_python_lex_string() {
        let (tokens, errors) = tokenize_python(r#""hello world""#);
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::StrLit);
    }

    #[test]
    fn test_python_lex_triple_string() {
        let (tokens, errors) = tokenize_python(r#""""docstring\ncontent""""#);
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::StrLit);
    }

    #[test]
    fn test_python_lex_fstring() {
        let (tokens, errors) = tokenize_python(r#"f"value is {x}""#);
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::StrLit);
    }

    #[test]
    fn test_python_lex_comment() {
        let source = "# this is a comment\ndef main(): pass";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // def main ( ) : pass
        assert_eq!(tokens.len(), 6);
    }

    #[test]
    fn test_python_lex_function_def() {
        let source = "def main(x: int, y: str) -> bool:\n    return True";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // def main ( x : int , y : str ) -> bool : return True
        assert_eq!(tokens[0].text, "def");
        assert_eq!(tokens[1].text, "main");
    }

    #[test]
    fn test_python_lex_class_def() {
        let source = "class MyClass(Base):\n    pass";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // class MyClass ( Base ) : pass
        assert_eq!(tokens[0].text, "class");
        assert_eq!(tokens[1].text, "MyClass");
    }

    #[test]
    fn test_python_lex_walrus() {
        let (tokens, errors) = tokenize_python("if x := 5: pass");
        assert!(errors.is_empty());
        // if x := 5 : pass
        assert_eq!(tokens.len(), 6);
        assert_eq!(tokens[2].text, ":=");
    }

    #[test]
    fn test_python_lex_decorators() {
        let source = "@staticmethod\ndef foo(): pass";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // @ staticmethod def foo ( ) : pass
        assert_eq!(tokens[0].text, "@");
        assert_eq!(tokens[1].text, "staticmethod");
        assert_eq!(tokens[2].text, "def");
    }

    #[test]
    fn test_python_lex_imports() {
        let source = "from typing import List, Dict\nimport os";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // from typing import List , Dict import os
        assert_eq!(tokens[0].text, "from");
        assert_eq!(tokens[1].text, "typing");
        assert_eq!(tokens[2].text, "import");
    }

    #[test]
    fn test_python_lex_star_args() {
        let source = "def foo(*args, **kwargs): pass";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // def foo ( * args , ** kwargs ) : pass
        assert_eq!(tokens[3].text, "*");
        assert_eq!(tokens[6].text, "**");
    }

    #[test]
    fn test_python_lex_async() {
        let source = "async def fetch(url): return await get(url)";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // async def fetch ( url ) : return await get ( url )
        assert_eq!(tokens[0].text, "async");
        assert_eq!(tokens[1].text, "def");
    }

    #[test]
    fn test_python_lex_comprehension() {
        let source = "x = [i * 2 for i in range(10) if i % 2 == 0]";
        let (tokens, errors) = tokenize_python(source);
        assert!(errors.is_empty());
        // x = [ i * 2 for i in range ( 10 ) if i % 2 == 0 ]
        assert_eq!(tokens[0].text, "x");
        assert_eq!(tokens[1].text, "=");
    }

    #[test]
    fn test_python_lex_complex_number() {
        let (tokens, errors) = tokenize_python("3.5j");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::FloatLit);
    }

    #[test]
    fn test_python_lex_hex_number() {
        let (tokens, errors) = tokenize_python("0xFF");
        assert!(errors.is_empty());
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, TerminalKind::IntLit);
    }
}
