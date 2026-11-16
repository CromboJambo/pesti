//! Meta-parser: parses `.peg` grammar files into `StructuralGrammar`.
//!
//! This is a hand-rolled recursive descent parser. The grammar of the `.peg`
//! format itself is simple enough that we don't need a full PEG engine —
//! just careful tokenization and recursive parsing.

use std::collections::BTreeSet;

use thiserror::Error;

use crate::expr::Expr;
use crate::grammar::{Nonterminal, RuleAnnotation, StructuralGrammar, TerminalKind};

/// Errors that can occur while parsing a `.peg` grammar file.
#[derive(Debug, Error)]
pub enum ParseError {
    #[error("unexpected token at line {line}, col {col}: expected {expected}, found {found}")]
    UnexpectedToken {
        line: usize,
        col: usize,
        expected: String,
        found: String,
    },

    #[error("unexpected end of input at line {line}, col {col}: expected {expected}")]
    UnexpectedEof {
        line: usize,
        col: usize,
        expected: String,
    },

    #[error("duplicate rule definition: {rule}")]
    DuplicateRule { rule: String },

    #[error("unknown terminal kind: {kind}")]
    UnknownTerminalKind { kind: String },

    #[error("start rule not defined: {rule}")]
    StartRuleNotDefined { rule: String },
}

/// A token in the `.peg` grammar format.
#[derive(Debug, Clone, PartialEq)]
enum Token {
    /// Identifier (rule name, capture name, terminal kind).
    Ident(String),
    /// Single-quoted string (terminal literal like `'fn'`).
    Quoted(String),
    /// Double-quoted string (annotation values like `"definition"`).
    StringLit(String),
    /// Symbol token.
    Symbol(char),
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "{s}"),
            Token::Quoted(s) => write!(f, "'{s}'"),
            Token::StringLit(s) => write!(f, "\"{s}\""),
            Token::Symbol(c) => write!(f, "{c}"),
        }
    }
}

/// Tokenizer for `.peg` grammar files.
struct Tokenizer {
    input: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Tokenizer {
    fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.input.get(self.pos + offset).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.input.get(self.pos).copied();
        if let Some(c) = ch {
            self.pos += 1;
            if c == '\n' {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
        }
        ch
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(' ') | Some('\t') | Some('\r') | Some('\n') => {
                    self.advance();
                }
                Some('#') => {
                    // Line comment: skip to end of line
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                Some('/') if self.peek_at(1) == Some('/') => {
                    // C-style line comment: skip to end of line
                    while let Some(c) = self.peek() {
                        if c == '\n' {
                            break;
                        }
                        self.advance();
                    }
                }
                _ => break,
            }
        }
    }

    fn next_token(&mut self) -> Result<Option<(Token, usize, usize)>, ParseError> {
        self.skip_whitespace_and_comments();

        let (line, col) = (self.line, self.col);

        match self.peek() {
            None => Ok(None),
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        self.advance();
                    } else {
                        break;
                    }
                }
                let ident: String = self.input[start..self.pos].iter().collect();
                Ok(Some((Token::Ident(ident), line, col)))
            }
            Some('\'') => {
                self.advance(); // consume opening quote
                let mut s = String::new();
                while let Some(c) = self.peek() {
                    if c == '\\' {
                        self.advance(); // consume backslash
                        if let Some(escaped) = self.peek() {
                            s.push(match escaped {
                                'n' => '\n',
                                't' => '\t',
                                '\\' => '\\',
                                '\'' => '\'',
                                _ => escaped,
                            });
                            self.advance();
                        }
                    } else if c == '\'' {
                        break;
                    } else {
                        s.push(c);
                        self.advance();
                    }
                }
                self.advance(); // consume closing quote
                Ok(Some((Token::Quoted(s), line, col)))
            }
            Some('"') => {
                self.advance(); // consume opening quote
                let start = self.pos;
                while let Some(c) = self.peek() {
                    if c == '"' {
                        break;
                    }
                    self.advance();
                }
                let s: String = self.input[start..self.pos].iter().collect();
                self.advance(); // consume closing quote
                Ok(Some((Token::StringLit(s), line, col)))
            }
            Some(c) if "=(){}/*?:,[];~+.@!&".contains(c) => {
                self.advance();
                Ok(Some((Token::Symbol(c), line, col)))
            }
            Some(c) => Err(ParseError::UnexpectedToken {
                line,
                col,
                expected: "identifier, string, or symbol".into(),
                found: format!("{c}"),
            }),
        }
    }

    fn tokenize_all(&mut self) -> Result<Vec<(Token, usize, usize)>, ParseError> {
        let mut tokens = Vec::new();
        while let Some(tok) = self.next_token()? {
            tokens.push(tok);
        }
        Ok(tokens)
    }
}

/// Parse a terminal kind name into its enum variant.
fn parse_terminal_kind(name: &str) -> Option<TerminalKind> {
    match name {
        "Ident" => Some(TerminalKind::Ident),
        "IntLit" => Some(TerminalKind::IntLit),
        "FloatLit" => Some(TerminalKind::FloatLit),
        "StrLit" => Some(TerminalKind::StrLit),
        "CharLit" => Some(TerminalKind::CharLit),
        "BoolLit" => Some(TerminalKind::BoolLit),
        "Comment" => Some(TerminalKind::Comment),
        "Whitespace" => Some(TerminalKind::Whitespace),
        "Eof" => Some(TerminalKind::Eof),
        _ => None,
    }
}

/// Parser state: token stream + position.
struct Parser<'a> {
    tokens: &'a [(Token, usize, usize)],
    pos: usize,
    /// Nonterminals defined so far (for distinguishing terminals from nonterminals).
    defined_rules: BTreeSet<String>,
}

impl<'a> Parser<'a> {
    fn new(tokens: &'a [(Token, usize, usize)]) -> Self {
        Self {
            tokens,
            pos: 0,
            defined_rules: BTreeSet::new(),
        }
    }

    fn peek(&self) -> Option<&(Token, usize, usize)> {
        self.tokens.get(self.pos)
    }

    fn peek_token(&self) -> Option<&Token> {
        self.peek().map(|t| &t.0)
    }

    /// Look ahead n tokens from current position (without consuming).
    fn peek_at_offset(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|t| &t.0)
    }

    fn advance(&mut self) -> Option<(Token, usize, usize)> {
        let tok = self.tokens.get(self.pos).cloned();
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn expect_symbol(&mut self, expected: char) -> Result<(), ParseError> {
        match self.peek_token() {
            Some(Token::Symbol(c)) if *c == expected => {
                self.advance();
                Ok(())
            }
            Some(tok) => Err(ParseError::UnexpectedToken {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: expected.to_string(),
                found: tok.to_string(),
            }),
            None => Err(ParseError::UnexpectedEof {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: expected.to_string(),
            }),
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.peek_token() {
            Some(Token::Ident(s)) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            Some(tok) => Err(ParseError::UnexpectedToken {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "identifier".into(),
                found: tok.to_string(),
            }),
            None => Err(ParseError::UnexpectedEof {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "identifier".into(),
            }),
        }
    }

    fn pos_line(&self) -> usize {
        self.tokens.get(self.pos).map(|t| t.1).unwrap_or(0)
    }

    fn pos_col(&self) -> usize {
        self.tokens.get(self.pos).map(|t| t.2).unwrap_or(0)
    }

    /// Parse the full grammar file.
    fn parse_grammar_file(&mut self) -> Result<StructuralGrammar, ParseError> {
        // "grammar" name
        let keyword = self.expect_ident()?;
        if keyword != "grammar" {
            return Err(ParseError::UnexpectedToken {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "\"grammar\"".into(),
                found: keyword,
            });
        }

        let name = self.expect_ident()?;

        // Pest format: grammar name { ... }
        // Legacy format: grammar name ; ...
        if self.peek_token() == Some(&Token::Symbol('{')) {
            self.advance(); // consume opening brace of grammar body
        } else {
            self.expect_symbol(';')?; // legacy semicolon form
        }

        // First pass: collect all rule names (for distinguishing terminals from nonterminals)
        let save_pos = self.pos;
        let mut rule_names = BTreeSet::new();
        while let Some(Token::Ident(_)) = self.peek_token() {
            if self.peek_token() == Some(&Token::Ident("annotations".into())) {
                break;
            }
            let _rule_name = self.expect_ident()?;
            rule_names.insert(_rule_name.clone());
            self.expect_symbol('=')?;
            self.expect_symbol('{')?; // consume opening brace
            // Skip the expression body: find matching "}"
            self.skip_expr_body()?;
        }
        self.pos = save_pos;

        self.defined_rules = rule_names;

        let mut grammar = StructuralGrammar::new(name, "");
        let mut first_rule = None;

        // Parse rules
        while let Some(Token::Ident(_)) = self.peek_token() {
            if self.peek_token() == Some(&Token::Ident("annotations".into())) {
                break;
            }
            let rule_name = self.expect_ident()?;
            if grammar
                .productions
                .contains_key(&Nonterminal(rule_name.clone()))
            {
                return Err(ParseError::DuplicateRule { rule: rule_name });
            }
            if first_rule.is_none() {
                first_rule = Some(rule_name.clone());
            }
            self.expect_symbol('=')?;
            self.expect_symbol('{')?;
            let expr = self.parse_expr()?;
            self.expect_symbol('}')?;
            grammar.add_rule(rule_name.clone(), expr);
        }

        // Parse annotations block (optional)
        if self.peek_token() == Some(&Token::Ident("annotations".into())) {
            self.advance(); // consume "annotations"
            self.expect_symbol('{')?;
            while self.peek_token() != Some(&Token::Symbol('}')) {
                let rule_name = self.expect_ident()?;
                self.expect_symbol('{')?;
                let annotation = self.parse_annotation()?;
                self.expect_symbol('}')?;
                grammar.add_annotation(rule_name, annotation);
            }
            self.expect_symbol('}')?;
        }

        // Consume closing brace of grammar body (pest format)
        if self.peek_token() == Some(&Token::Symbol('}')) {
            self.advance();
        }

        // Set start rule to the first defined rule in source order (convention)
        if let Some(first) = first_rule {
            grammar.start = Nonterminal(first);
        } else {
            return Err(ParseError::StartRuleNotDefined {
                rule: "(no rules defined)".into(),
            });
        }

        Ok(grammar)
    }

    /// Skip over an expression body (from after "{" to matching "}").
    fn skip_expr_body(&mut self) -> Result<(), ParseError> {
        let mut depth = 1;
        while depth > 0 {
            match self.peek_token() {
                Some(Token::Symbol('{')) => {
                    depth += 1;
                    self.advance();
                }
                Some(Token::Symbol('}')) => {
                    depth -= 1;
                    self.advance();
                }
                Some(_) => {
                    self.advance();
                }
                None => {
                    return Err(ParseError::UnexpectedEof {
                        line: self.pos_line(),
                        col: self.pos_col(),
                        expected: "matching '}'".into(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Parse an expression: sequence of atoms, with ordered choice.
    /// expr = seq ( "/" seq )*
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let first_seq = self.parse_sequence()?;

        if self.peek_token() == Some(&Token::Symbol('/')) {
            let mut alts = vec![first_seq];
            while self.peek_token() == Some(&Token::Symbol('/')) {
                self.advance(); // consume "/"
                alts.push(self.parse_sequence()?);
            }
            Ok(Expr::OrderedChoice(alts))
        } else {
            Ok(first_seq)
        }
    }

    /// Parse a sequence: one or more atoms.
    /// seq = atom+
    fn parse_sequence(&mut self) -> Result<Expr, ParseError> {
        let mut items = Vec::new();
        loop {
            // Skip optional separators (comma or tilde)
            if matches!(
                self.peek_token(),
                Some(Token::Symbol(',')) | Some(Token::Symbol('~'))
            ) {
                self.advance();
            }

            // Check for end of sequence
            match self.peek_token() {
                Some(Token::Symbol('}'))
                | Some(Token::Symbol(')'))
                | Some(Token::Symbol('/'))
                | None => break,
                _ => {}
            }

            items.push(self.parse_atom()?);
        }

        if items.is_empty() {
            return Err(ParseError::UnexpectedToken {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "expression".into(),
                found: "end of sequence".into(),
            });
        }

        match items.len() {
            0 => unreachable!(),
            1 => Ok(items.into_iter().next().unwrap()),
            _ => Ok(Expr::Sequence(items)),
        }
    }

    /// Parse an atom: the smallest expression unit.
    /// atom = capture / quantified / terminal_ref / nonterminal_ref / "(" expr ")"
    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        // Grouped expression
        if self.peek_token() == Some(&Token::Symbol('(')) {
            self.advance(); // consume "("
            let inner = self.parse_expr()?;
            self.expect_symbol(')')?;
            return self.apply_quantifier(inner);
        }

        // Clone the token to avoid borrow conflicts with advance()
        let current = match self.peek_token() {
            Some(tok) => tok.clone(),
            None => {
                return Err(ParseError::UnexpectedEof {
                    line: self.pos_line(),
                    col: self.pos_col(),
                    expected: "expression atom".into(),
                });
            }
        };

        match current {
            Token::Symbol('@') => {
                // Capture syntax: @name expr
                self.advance(); // consume "@"
                let name = self.expect_ident()?;
                let inner = self.parse_atom()?;
                Ok(Expr::Capture(name, Box::new(inner)))
            }
            Token::Ident(name) => {
                // Check for capture: "name:" followed by an expression
                if self.peek_ahead_is_colon() {
                    self.advance(); // consume name
                    self.expect_symbol(':')?;
                    let inner = self.parse_atom()?;
                    return Ok(Expr::Capture(name, Box::new(inner)));
                }

                // Consume the identifier
                self.advance();

                // If this name is defined as a rule in the grammar, it's a nonterminal reference.
                // Otherwise, check if it's a built-in terminal kind.
                if self.defined_rules.contains(&name) {
                    let nt = Nonterminal(name);
                    return self.apply_quantifier(Expr::Nonterminal(nt));
                }

                if let Some(tk) = parse_terminal_kind(&name) {
                    return self.apply_quantifier(Expr::Terminal(tk));
                }

                // Fallback: undefined nonterminal reference (will be caught by validation)
                let nt = Nonterminal(name);
                self.apply_quantifier(Expr::Nonterminal(nt))
            }
            Token::Quoted(s) => {
                // Check for character range: 'a'..'z' (single char, .., single char)
                if s.len() == 1
                    && self.peek_at_offset(1) == Some(&Token::Symbol('.'))
                    && self.peek_at_offset(2) == Some(&Token::Symbol('.'))
                    && matches!(self.peek_at_offset(3), Some(Token::Quoted(e)) if e.len() == 1)
                {
                    let start_c = s.chars().next().unwrap();
                    // Consume 'a', '.', '.', 'z'
                    self.advance(); // start char
                    self.advance(); // first dot
                    self.advance(); // second dot
                    let end_tok = self.advance().unwrap().0;
                    let end_c = match end_tok {
                        Token::Quoted(e) => e.chars().next().unwrap(),
                        _ => unreachable!(),
                    };
                    return self.apply_quantifier(Expr::CharRange(start_c, end_c));
                }

                // Quoted string: keyword or punctuation terminal
                self.advance();
                let tk = if s.chars().all(|c| c.is_ascii_alphabetic() || c == '_') {
                    TerminalKind::Keyword(s)
                } else {
                    TerminalKind::Punct(s)
                };
                self.apply_quantifier(Expr::Terminal(tk))
            }
            tok => Err(ParseError::UnexpectedToken {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "expression atom".into(),
                found: tok.to_string(),
            }),
        }
    }

    /// Check if the token after the current one is a colon (for capture detection).
    fn peek_ahead_is_colon(&self) -> bool {
        self.tokens
            .get(self.pos + 1)
            .map(|t| matches!(t.0, Token::Symbol(':')))
            .unwrap_or(false)
    }

    /// Apply quantifier (?, *, +) if present after an atom.
    fn apply_quantifier(&mut self, expr: Expr) -> Result<Expr, ParseError> {
        match self.peek_token() {
            Some(Token::Symbol('?')) => {
                self.advance();
                Ok(Expr::Optional(Box::new(expr)))
            }
            Some(Token::Symbol('*')) => {
                self.advance();
                Ok(Expr::Repeat(Box::new(expr)))
            }
            Some(Token::Symbol('+')) => {
                // One-or-more: e+ ≡ e e*
                self.advance();
                let repeated = Expr::Repeat(Box::new(expr.clone()));
                Ok(Expr::Sequence(vec![expr, repeated]))
            }
            _ => Ok(expr),
        }
    }

    /// Parse an annotation body: key-value pairs.
    fn parse_annotation(&mut self) -> Result<RuleAnnotation, ParseError> {
        let mut role = String::new();
        let mut embed_fields = Vec::new();
        let mut similarity_class = None;

        while self.peek_token() != Some(&Token::Symbol('}')) {
            let key = self.expect_ident()?;
            self.expect_symbol(':')?;

            match key.as_str() {
                "role" => {
                    role = self.parse_string_value()?;
                }
                "embed_fields" => {
                    embed_fields = self.parse_string_list()?;
                }
                "class" => {
                    similarity_class = Some(self.parse_string_value()?);
                }
                other => {
                    return Err(ParseError::UnexpectedToken {
                        line: self.pos_line(),
                        col: self.pos_col(),
                        expected: "\"role\", \"embed_fields\", or \"class\"".into(),
                        found: other.into(),
                    });
                }
            }

            // Optional comma separator
            if self.peek_token() == Some(&Token::Symbol(',')) {
                self.advance();
            }
        }

        Ok(RuleAnnotation {
            role,
            embed_fields,
            similarity_class,
        })
    }

    fn parse_string_value(&mut self) -> Result<String, ParseError> {
        match self.peek_token() {
            Some(Token::StringLit(s)) => {
                let s = s.clone();
                self.advance();
                Ok(s)
            }
            Some(tok) => Err(ParseError::UnexpectedToken {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "string value".into(),
                found: tok.to_string(),
            }),
            None => Err(ParseError::UnexpectedEof {
                line: self.pos_line(),
                col: self.pos_col(),
                expected: "string value".into(),
            }),
        }
    }

    fn parse_string_list(&mut self) -> Result<Vec<String>, ParseError> {
        self.expect_symbol('[')?;
        let mut items = Vec::new();
        while self.peek_token() != Some(&Token::Symbol(']')) {
            items.push(self.parse_string_value()?);
            if self.peek_token() == Some(&Token::Symbol(',')) {
                self.advance();
            }
        }
        self.expect_symbol(']')?;
        Ok(items)
    }
}

/// Parse a `.peg` grammar file from source text.
pub fn parse_peg(source: &str) -> Result<StructuralGrammar, ParseError> {
    let mut tokenizer = Tokenizer::new(source);
    let tokens = tokenizer.tokenize_all()?;
    let mut parser = Parser::new(&tokens);
    parser.parse_grammar_file()
}

/// Parse a `.peg` grammar file from a file path.
pub fn parse_peg_file(path: &std::path::Path) -> Result<StructuralGrammar, ParseError> {
    let source = std::fs::read_to_string(path).map_err(|e| ParseError::UnexpectedToken {
        line: 0,
        col: 0,
        expected: "readable file".into(),
        found: e.to_string(),
    })?;
    parse_peg(&source)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_grammar() {
        let src = r#"grammar test;
Start = { 'fn' name: Ident }
"#;
        let g = parse_peg(src).unwrap();
        assert_eq!(g.name, "test");
        assert_eq!(g.start.0, "Start");
        assert!(g.has_rule(&Nonterminal("Start".into())));
    }

    #[test]
    fn test_parse_sequence() {
        let src = r#"grammar test;
Start = { 'fn' name: Ident params: '(' ')' }
"#;
        let g = parse_peg(src).unwrap();
        let expr = g.rule(&Nonterminal("Start".into())).unwrap();
        match expr {
            Expr::Sequence(items) => assert_eq!(items.len(), 4),
            _ => panic!("expected sequence"),
        }
    }

    #[test]
    fn test_parse_ordered_choice() {
        let src = r#"grammar test;
Start = { A / B }
A = { 'a' }
B = { 'b' }
"#;
        let g = parse_peg(src).unwrap();
        let expr = g.rule(&Nonterminal("Start".into())).unwrap();
        match expr {
            Expr::OrderedChoice(alts) => assert_eq!(alts.len(), 2),
            _ => panic!("expected ordered choice"),
        }
    }

    #[test]
    fn test_parse_quantifiers() {
        let src = r#"grammar test;
Start = { 'a'? 'b'* }
"#;
        let g = parse_peg(src).unwrap();
        let expr = g.rule(&Nonterminal("Start".into())).unwrap();
        match expr {
            Expr::Sequence(items) => {
                assert!(matches!(items[0], Expr::Optional(_)));
                assert!(matches!(items[1], Expr::Repeat(_)));
            }
            _ => panic!("expected sequence"),
        }
    }

    #[test]
    fn test_parse_annotations() {
        let src = r#"grammar test;
FuncDef = { 'fn' name: Ident }

annotations {
    FuncDef { role: "definition", embed_fields: ["name"], class: "declaration" }
}
"#;
        let g = parse_peg(src).unwrap();
        let ann = g.annotations.get(&Nonterminal("FuncDef".into())).unwrap();
        assert_eq!(ann.role, "definition");
        assert_eq!(ann.embed_fields, vec!["name"]);
        assert_eq!(ann.similarity_class, Some("declaration".into()));
    }

    #[test]
    fn test_parse_terminal_kinds() {
        let src = r#"grammar test;
Start = { Ident IntLit FloatLit StrLit BoolLit }
"#;
        let g = parse_peg(src).unwrap();
        let expr = g.rule(&Nonterminal("Start".into())).unwrap();
        match expr {
            Expr::Sequence(items) => assert_eq!(items.len(), 5),
            _ => panic!("expected sequence"),
        }
    }

    #[test]
    fn test_duplicate_rule_error() {
        let src = r#"grammar test;
A = { 'a' }
A = { 'b' }
"#;
        assert!(matches!(
            parse_peg(src),
            Err(ParseError::DuplicateRule { .. })
        ));
    }

    #[test]
    fn test_terminal_kind_display_in_grammar() {
        let src = r#"grammar test;
Start = { Ident }
"#;
        let g = parse_peg(src).unwrap();
        let expr = g.rule(&Nonterminal("Start".into())).unwrap();
        match expr {
            Expr::Terminal(TerminalKind::Ident) => {}
            _ => panic!("expected Terminal(Ident)"),
        }
    }

    #[test]
    fn test_nested_captures() {
        let src = r#"grammar test;
Start = { outer: ( 'fn' inner: Ident ) }
"#;
        let g = parse_peg(src).unwrap();
        let expr = g.rule(&Nonterminal("Start".into())).unwrap();
        match expr {
            Expr::Capture(name, _) => assert_eq!(name, "outer"),
            _ => panic!("expected capture"),
        }
    }

    #[test]
    fn test_complex_grammar() {
        let src = r#"grammar rust_structural;

Module       = { item: Item* }
Item         = { kind: FuncDef / StructDef }
FuncDef      = { Visibility?, 'fn', name: Ident, params: ParamList, body: Block }
StructDef    = { 'struct', name: Ident, fields: '{' Field* '}' }
ParamList    = { '(' param: Param* ')' }
Param        = { name: Ident, ':', Type }
Type         = { Ident / Path }
Path         = { seg: Ident ('::' seg: Ident)* }
Block        = { '{' stmt: Stmt* '}' }
Stmt         = { Expr }
Expr         = { Call / Literal }
Call         = { name: Ident, '(' arg: Expr* ')' }
Literal      = { IntLit / FloatLit / StrLit }
Visibility   = { 'pub' }
Field        = { name: Ident, ':', Type }

annotations {
    FuncDef   { role: "definition", embed_fields: ["name", "params"], class: "declaration" }
    StructDef { role: "definition", embed_fields: ["name", "fields"], class: "declaration" }
    Call      { role: "expression", embed_fields: ["name", "arg"], class: "invocation" }
}
"#;
        let g = parse_peg(src).unwrap();
        assert_eq!(g.name, "rust_structural");
        assert_eq!(g.productions.len(), 15);
        assert_eq!(g.annotations.len(), 3);
        assert!(g.has_rule(&Nonterminal("FuncDef".into())));
        assert!(g.has_rule(&Nonterminal("StructDef".into())));
    }
}
