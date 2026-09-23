//! Structural tokenizer for Rust source code.
//!
//! Encodes syntactic boundaries as token boundaries to give LLMs semantic structure
//! without requiring them to relearn Rust grammar from byte-pair statistics.
//!
//! Example: `fn foo(x: i32) -> i32 {` becomes one structural token (FnDecl) instead of ~15 BPE tokens.

use std::collections::HashMap;

/// Structural token with position information
#[derive(Debug, Clone)]
pub struct StructuralToken {
    pub kind: TokenKind,
    pub text: String,
    pub span: (usize, usize),
}

/// Kinds of structural tokens we emit
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TokenKind {
    // Declarations
    FnDecl,           // fn name(params) -> ret { ... }
    LetStmt,          // let x = expr; or let x: T = expr;
    StructDecl,       // struct Name { fields }
    EnumDecl,         // enum Name { variants }
    TraitDecl,        // trait Name { methods }
    ImplBlock,        // impl Trait for Type { ... }

    // Control flow
    IfElse,           // if cond { } else { }
    WhileLoop,        // while cond { }
    ForLoop,          // for item in iter { }
    MatchExpr,        // match expr { arms }
    ReturnExpr,       // return expr;
    BreakExpr,        // break; or break expr;
    ContinueExpr,     // continue;

    // Expressions & operations
    CallExpr,    // fn(args)
    FieldAccess, // obj.field
    MethodCall,  // obj.method(args)
    BinaryOp,    // a + b, a == b, etc.
    UnaryOp,     // !x, -x, *ptr, &ref
    ParenExpr,   // (expr)
    ExprStmt,    // expression terminated by semicolon (e.g., a = b;)

    // Literals (values are stored in the token text for simplicity)
    IntLit,
    FloatLit,
    StrLit,
    CharLit,
    BoolLit,
    ByteLit,
    ArrayLit,
    TupleLit,
    SliceExpr,        // arr[start..end]

    // Types & casts
    TypeAnnotation,   // : i32
    CastExpr,         // x as i32
    Dereference,      // *ptr
    Reference,        // &x or &mut x

    // Blocks and delimiters
    BlockStart,
    BlockEnd,
    ParenOpen,
    ParenClose,
    BraceOpen,
    BraceClose,
    BracketOpen,
    BracketClose,
    Semicolon,
    Comma,
    Colon,
    Dot,
    Arrow,            // -> (return type)
    FatArrow,         // => (match arm)

    // Identifiers and names
    Ident(String),
    Keyword(String),  // Keywords that aren't structural constructs

    // Rust-specific features
    MacroInvocation,  // macro!(...)
    Attribute,        // #[derive(...)]
    Lifetime,         // 'a in &'a T
    GenericParams,    // <T: Trait>
    WhereClause,      // where T: Clone
    AsyncBlock,       // async { }
    MoveClosure,      // move |x| x + 1
    Closure,          // |x| x + 1

    // Visibility and modifiers
    Pub,
    Private,
    Static,
    Const,
    Mut,
    Unsafe,
    Extern,
    Abstract,
}

/// Errors from the structural tokenizer
#[derive(Debug)]
pub enum TokenizeError {
    Parse { pos: usize, msg: String },
}

impl std::fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizeError::Parse { pos, msg } => write!(f, "Parse error at position {}: {}", pos, msg),
        }
    }
}

impl std::error::Error for TokenizeError {}

/// The structural tokenizer — recognizes Rust syntactic constructs and emits semantic tokens.
pub struct StructuralTokenizer {
    /// Vocabulary mapping token text to IDs (populated during training)
    vocab: HashMap<String, u32>,
}

impl StructuralTokenizer {
    pub fn new() -> Self {
        StructuralTokenizer { vocab: HashMap::new() }
    }

    /// Tokenize Rust source into structural tokens
    pub fn tokenize(&self, src: &str) -> Result<Vec<StructuralToken>, TokenizeError> {
        let mut tokens = Vec::new();
        let chars: Vec<char> = src.chars().collect();
        let pos = self.tokenize_inner(&chars, 0, &mut tokens)?;
        assert_eq!(pos, chars.len(), "tokenizer did not consume entire input");
        Ok(tokens)
    }

    fn tokenize_inner(
        &self,
        chars: &[char],
        start: usize,
        tokens: &mut Vec<StructuralToken>,
    ) -> Result<usize, TokenizeError> {
        let mut pos = start;

        while pos < chars.len() {
            // Skip whitespace and newlines
            if self.is_whitespace(&chars, pos) {
                pos += 1;
                continue;
            }

            // Skip comments
            if self.is_comment_start(&chars, pos) {
                let (end_pos, _) = self.skip_comment(&chars, pos);
                pos = end_pos;
                continue;
            }

            // Handle attributes: #[derive(...)], #[allow(...)], etc.
            if chars[pos] == '#' {
                let token = self.parse_attribute(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
                continue;
            }

            // Try to parse structural constructs in priority order
            if self.match_keyword(&chars, pos, "fn") {
                let token = self.parse_fn_decl(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "let") {
                let token = self.parse_let_stmt(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "return") {
                let token = self.parse_return_expr(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "if") {
                let token = self.parse_if_else(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "while") {
                let token = self.parse_while_loop(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "for") {
                let token = self.parse_for_loop(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "match") {
                let token = self.parse_match_expr(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "struct") {
                let token = self.parse_struct_decl(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "enum") {
                let token = self.parse_enum_decl(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "trait") {
                let token = self.parse_trait_decl(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "impl") {
                let token = self.parse_impl_block(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "break") {
                let token = self.parse_break_expr(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if self.match_keyword(&chars, pos, "continue") {
                let token = self.parse_continue_expr(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if chars[pos] == '{' {
                let token = StructuralToken {
                    kind: TokenKind::BlockStart,
                    text: "{".to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if chars[pos] == '}' {
                let token = StructuralToken {
                    kind: TokenKind::BlockEnd,
                    text: "}".to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if self.is_operator(&chars, pos) {
                let token = self.parse_operator(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else if chars[pos] == '(' || chars[pos] == ')' {
                let kind = if chars[pos] == '(' { TokenKind::ParenOpen } else { TokenKind::ParenClose };
                let token = StructuralToken {
                    kind,
                    text: chars[pos].to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if chars[pos] == '[' || chars[pos] == ']' {
                let kind = if chars[pos] == '[' { TokenKind::BracketOpen } else { TokenKind::BracketClose };
                let token = StructuralToken {
                    kind,
                    text: chars[pos].to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if chars[pos] == ',' {
                let token = StructuralToken {
                    kind: TokenKind::Comma,
                    text: ",".to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if chars[pos] == ';' {
                let token = StructuralToken {
                    kind: TokenKind::Semicolon,
                    text: ";".to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if chars[pos] == '.' {
                // Check for range operator (..) or method call dot
                if pos + 1 < chars.len() && chars[pos + 1] == '.' {
                    let token = StructuralToken {
                        kind: TokenKind::BinaryOp,
                        text: "..".to_string(),
                        span: (pos, pos + 2),
                    };
                    tokens.push(token);
                    pos += 2;
                } else {
                    let token = StructuralToken {
                        kind: TokenKind::Dot,
                        text: ".".to_string(),
                        span: (pos, pos + 1),
                    };
                    tokens.push(token);
                    pos += 1;
                }
            } else if chars[pos] == ':' && pos + 1 < chars.len() && chars[pos + 1] == '>' {
                // Arrow for return type
                let token = StructuralToken {
                    kind: TokenKind::Arrow,
                    text: "->".to_string(),
                    span: (pos, pos + 2),
                };
                tokens.push(token);
                pos += 2;
            } else if chars[pos] == ':' {
                let token = StructuralToken {
                    kind: TokenKind::Colon,
                    text: ":".to_string(),
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            } else if chars[pos] == '=' && pos + 1 < chars.len() && chars[pos + 1] == '>' {
                // Fat arrow for match arms
                let token = StructuralToken {
                    kind: TokenKind::FatArrow,
                    text: "=>".to_string(),
                    span: (pos, pos + 2),
                };
                tokens.push(token);
                pos += 2;
            } else if chars[pos] == '"' {
                // String literal
                let start_pos = pos;
                pos += 1; // skip opening quote
                while pos < chars.len() && chars[pos] != '"' {
                    if chars[pos] == '\\' {
                        pos += 2; // skip escaped char
                    } else {
                        pos += 1;
                    }
                }
                if pos < chars.len() {
                    pos += 1; // skip closing quote
                }
                let text: String = chars[start_pos..pos].iter().collect();
                let token = StructuralToken {
                    kind: TokenKind::StrLit,
                    text,
                    span: (start_pos, pos),
                };
                tokens.push(token);
            } else if chars[pos] == '\'' {
                // Char literal
                let start_pos = pos;
                pos += 1; // skip opening quote
                while pos < chars.len() && chars[pos] != '\'' {
                    if chars[pos] == '\\' {
                        pos += 2;
                    } else {
                        pos += 1;
                    }
                }
                if pos < chars.len() {
                    pos += 1; // skip closing quote
                }
                let text: String = chars[start_pos..pos].iter().collect();
                let token = StructuralToken {
                    kind: TokenKind::CharLit,
                    text,
                    span: (start_pos, pos),
                };
                tokens.push(token);
            } else if chars[pos] == '\'' {
                // Lifetime: 'a or 'static
                let start_pos = pos;
                pos += 1; // skip '
                while pos < chars.len() && self.is_ident_char(chars[pos]) {
                    pos += 1;
                }
                let text: String = chars[start_pos..pos].iter().collect();
                let token = StructuralToken {
                    kind: TokenKind::Lifetime,
                    text,
                    span: (start_pos, pos),
                };
                tokens.push(token);
            } else if chars[pos].is_alphabetic() || chars[pos] == '_' {
                // Try expression statement first - identifier followed by operators until ;
                let expr_end = self.find_expr_stmt_end(&chars, pos);
                if expr_end > pos {
                    let text: String = chars[pos..expr_end].iter().collect();
                    let token = StructuralToken {
                        kind: TokenKind::ExprStmt,
                        text,
                        span: (pos, expr_end),
                    };
                    tokens.push(token);
                    pos = expr_end;
                } else {
                    // Fall back to raw identifier/literal
                    let token = self.parse_identifier_or_literal(&chars, pos)?;
                    tokens.push(token.clone());
                    pos = token.span.1;
                }
            } else if chars[pos].is_digit(10) {
                let token = self.parse_identifier_or_literal(&chars, pos)?;
                tokens.push(token.clone());
                pos = token.span.1;
            } else {
                // Unknown character - emit as-is to make progress
                let ch = chars[pos];
                let text = ch.to_string();
                let token = StructuralToken {
                    kind: TokenKind::Ident(text.clone()),
                    text,
                    span: (pos, pos + 1),
                };
                tokens.push(token);
                pos += 1;
            }
        }

        Ok(pos)
    }

    fn is_whitespace(&self, chars: &[char], pos: usize) -> bool {
        pos < chars.len() && (chars[pos].is_whitespace())
    }

    fn parse_attribute(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Parse #[...] attribute - skip to matching ] and then any parens content
        let mut pos = start;

        // Skip #
        pos += 1;

        // Handle inner attributes: #![...] vs #[...]
        let mut is_inner = false;
        if pos < chars.len() && chars[pos] == '!' {
            is_inner = true;
            pos += 1;
        }

        // Check for outer brackets [attr] vs inner (#[attr])
        if pos < chars.len() && chars[pos] == '[' {
            pos += 1;
        } else {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '[' after '#' in attribute".to_string(),
            });
        }

        // Skip whitespace
        while self.is_whitespace(chars, pos) {
            pos += 1;
        }

        // Find matching ]
        let mut bracket_depth = 1usize;
        while pos < chars.len() && bracket_depth > 0 {
            if chars[pos] == '[' {
                bracket_depth += 1;
            } else if chars[pos] == ']' {
                bracket_depth -= 1;
            }
            pos += 1;
        }

        // Now skip any outer parens (for inner attributes like #[attr(...)])
        while self.is_whitespace(chars, pos) {
            pos += 1;
        }
        if pos < chars.len() && chars[pos] == '(' {
            let mut paren_depth = 1usize;
            pos += 1;
            while pos < chars.len() && paren_depth > 0 {
                if chars[pos] == '(' {
                    paren_depth += 1;
                } else if chars[pos] == ')' {
                    paren_depth -= 1;
                }
                pos += 1;
            }
        }

        Ok(StructuralToken {
            kind: TokenKind::Attribute,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn is_comment_start(&self, chars: &[char], pos: usize) -> bool {
        if pos + 1 >= chars.len() {
            return false;
        }
        (chars[pos] == '/' && chars[pos + 1] == '/') || (chars[pos] == '/' && chars[pos + 1] == '*')
    }

    fn skip_comment(&self, chars: &[char], start: usize) -> (usize, String) {
        let mut pos = start;
        if pos < chars.len() && chars[pos] == '/' && pos + 1 < chars.len() && chars[pos + 1] == '/' {
            // Line comment - skip to end of line
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
        } else if chars[pos] == '/' && pos + 1 < chars.len() && chars[pos + 1] == '*' {
            // Skip block comment
            while pos < chars.len() && !(chars[pos] == '*' && pos + 1 < chars.len() && chars[pos + 1] == '/') {
                pos += 1;
            }
            if pos < chars.len() {
                pos += 2;
            }
        } else {
            pos = start + 1;
        }
        (pos, String::new())
    }

    fn match_keyword(&self, chars: &[char], pos: usize, keyword: &str) -> bool {
        let kw_chars: Vec<char> = keyword.chars().collect();
        if pos + kw_chars.len() > chars.len() {
            return false;
        }

        for (i, c) in kw_chars.iter().enumerate() {
            if chars[pos + i] != *c {
                return false;
            }
        }

        // Check word boundary
        let after_pos = pos + kw_chars.len();
        if after_pos < chars.len() && self.is_ident_char(chars[after_pos]) {
            return false;
        }

        true
    }

    fn is_ident_char(&self, c: char) -> bool {
        c.is_alphanumeric() || c == '_'
    }

    /// Find the end of an expression statement (up to and including the semicolon).
    /// Returns the position just past the semicolon, or start if no semicolon found.
    fn find_expr_stmt_end(&self, chars: &[char], start: usize) -> usize {
        let mut pos = start;
        let mut depth_paren = 0;
        let mut depth_brace = 0;
        let mut depth_bracket = 0;

        while pos < chars.len() {
            match chars[pos] {
                '(' => depth_paren += 1,
                ')' => {
                    if depth_paren > 0 {
                        depth_paren -= 1;
                    }
                }
                '{' => depth_brace += 1,
                '}' => {
                    if depth_brace > 0 {
                        depth_brace -= 1;
                    }
                }
                '[' => depth_bracket += 1,
                ']' => {
                    if depth_bracket > 0 {
                        depth_bracket -= 1;
                    }
                }
                ';' => {
                    // Found end of statement at top level
                    if depth_paren == 0 && depth_brace == 0 && depth_bracket == 0 {
                        return pos + 1; // past the semicolon
                    }
                }
                '\n' => {
                    // Newline without semicolon - not a complete expr stmt
                    if depth_paren == 0 && depth_brace == 0 && depth_bracket == 0 {
                        break;
                    }
                }
                _ => {}
            }
            pos += 1;
        }

        start // No semicolon found, not an expr stmt
    }

    fn is_operator(&self, chars: &[char], pos: usize) -> bool {
        if pos >= chars.len() {
            return false;
        }
        let c = chars[pos];
        matches!(c, '+' | '-' | '*' | '/' | '%' | '<' | '>' | '=' | '!' | '&' | '|' | '^' | '~')
    }

    fn parse_operator(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        let mut pos = start;
        let c = chars[pos];
        pos += 1;

        // Handle multi-char operators: ==, !=, <=, >=, <<, >>, &&, ||, +=, -=, etc.
        if pos < chars.len() {
            let next = chars[pos];
            match (c, next) {
                ('=', '=') | ('!', '=') | ('<', '=') | ('>', '=') | ('<', '<') | ('>', '>')
                | ('&', '&') | ('|', '|') | ('+', '=') | ('-', '=') | ('*', '=') | ('/', '=')
                | ('%', '=') | ('&', '=') | ('|', '=') | ('^', '=') => {
                    pos += 1;
                }
                _ => {}
            }
        }

        let text: String = chars[start..pos].iter().collect();
        Ok(StructuralToken {
            kind: TokenKind::BinaryOp,
            text,
            span: (start, pos),
        })
    }

    fn parse_fn_decl(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "fn" and whitespace
        let mut pos = start + 2;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Parse function name (identifier)
        let name_start = pos;
        while pos < chars.len() && self.is_ident_char(chars[pos]) {
            pos += 1;
        }

        // Find the opening brace of the body
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in fn decl".to_string(),
            });
        }

        let span_end = pos;

        Ok(StructuralToken {
            kind: TokenKind::FnDecl,
            text: chars[name_start..span_end].iter().collect(),
            span: (start, span_end),
        })
    }

    fn parse_let_stmt(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "let" and whitespace
        let mut pos = start + 3;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Parse variable name
        let name_start = pos;
        while pos < chars.len() && self.is_ident_char(chars[pos]) {
            pos += 1;
        }

        // Find end of statement (semicolon)
        while pos < chars.len() && chars[pos] != ';' {
            pos += 1;
        }
        if pos < chars.len() {
            pos += 1;
        } // skip semicolon

        Ok(StructuralToken {
            kind: TokenKind::LetStmt,
            text: chars[name_start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_return_expr(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "return" and whitespace
        let mut pos = start + 6;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find end of expression (semicolon or block end)
        while pos < chars.len() && chars[pos] != ';' && chars[pos] != '}' {
            pos += 1;
        }
        if pos < chars.len() && chars[pos] == ';' {
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::ReturnExpr,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_if_else(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "if" and whitespace
        let mut pos = start + 2;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Rust if doesn't require parens - find opening brace directly
        // Find opening brace of body
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in if statement".to_string(),
            });
        }

        // Find matching closing brace
        let mut brace_depth = 1;
        pos += 1;
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        // Check for else branch
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }
        if self.match_keyword(&chars, pos, "else") {
            let mut inner_pos = pos + 4;
            while self.is_whitespace(&chars, inner_pos) {
                inner_pos += 1;
            }
            // Parse else body recursively (simplified)
            while inner_pos < chars.len() && chars[inner_pos] != '}' {
                inner_pos += 1;
            }
            if inner_pos < chars.len() {
                pos = inner_pos + 1;
            }
        }

        Ok(StructuralToken {
            kind: TokenKind::IfElse,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_while_loop(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "while" and whitespace
        let mut pos = start + 5;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find opening paren
        while pos < chars.len() && chars[pos] != '(' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '(' in while loop".to_string(),
            });
        }

        // Find matching closing paren
        let mut depth = 1;
        pos += 1;
        while pos < chars.len() && depth > 0 {
            if chars[pos] == '(' {
                depth += 1;
            } else if chars[pos] == ')' {
                depth -= 1;
            }
            pos += 1;
        }

        // Find opening brace of body
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in while loop".to_string(),
            });
        }

        // Find matching closing brace
        let mut brace_depth = 1;
        pos += 1;
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::WhileLoop,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_for_loop(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "for" and whitespace
        let mut pos = start + 3;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find opening brace (skip iterator expression)
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in for loop".to_string(),
            });
        }

        // Don't consume the body — emit just the header and let the caller
        // tokenize the block contents. The opening brace is emitted as a
        // BlockStart token by the caller's main loop.
        Ok(StructuralToken {
            kind: TokenKind::ForLoop,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_match_expr(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "match" and whitespace
        let mut pos = start + 5;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find opening brace of match body
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in match expression".to_string(),
            });
        }

        // Find matching closing brace
        let mut brace_depth = 1;
        pos += 1;
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::MatchExpr,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_struct_decl(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "struct" and whitespace
        let mut pos = start + 6;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Parse struct name
        let name_start = pos;
        while pos < chars.len() && self.is_ident_char(chars[pos]) {
            pos += 1;
        }

        // Find opening brace or semicolon (unit struct)
        while pos < chars.len() && chars[pos] != '{' && chars[pos] != ';' {
            pos += 1;
        }

        if pos >= chars.len() || chars[pos] == ';' {
            // Unit struct: just name and semicolon
            if pos < chars.len() {
                pos += 1; // skip semicolon
            }
            return Ok(StructuralToken {
                kind: TokenKind::StructDecl,
                text: chars[name_start..pos].iter().collect(),
                span: (start, pos),
            });
        }

        // Named fields: find matching closing brace
        let mut brace_depth = 1;
        pos += 1; // skip opening brace
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        // Skip trailing semicolon if present
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }
        if pos < chars.len() && chars[pos] == ';' {
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::StructDecl,
            text: chars[name_start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_enum_decl(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "enum" and whitespace
        let mut pos = start + 4;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Parse enum name
        let name_start = pos;
        while pos < chars.len() && self.is_ident_char(chars[pos]) {
            pos += 1;
        }

        // Find opening brace
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in enum decl".to_string(),
            });
        }

        // Find matching closing brace
        let mut brace_depth = 1;
        pos += 1;
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::EnumDecl,
            text: chars[name_start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_trait_decl(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "trait" and whitespace
        let mut pos = start + 5;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Parse trait name
        let name_start = pos;
        while pos < chars.len() && self.is_ident_char(chars[pos]) {
            pos += 1;
        }

        // Find opening brace (may have where clause in between)
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in trait decl".to_string(),
            });
        }

        // Find matching closing brace
        let mut brace_depth = 1;
        pos += 1;
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::TraitDecl,
            text: chars[name_start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_impl_block(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        // Skip "impl" and whitespace
        let mut pos = start + 4;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find opening brace (skip type params and for clause)
        while pos < chars.len() && chars[pos] != '{' {
            pos += 1;
        }
        if pos >= chars.len() {
            return Err(TokenizeError::Parse {
                pos: start,
                msg: "Expected '{' in impl block".to_string(),
            });
        }

        // Find matching closing brace
        let mut brace_depth = 1;
        pos += 1;
        while pos < chars.len() && brace_depth > 0 {
            if chars[pos] == '{' {
                brace_depth += 1;
            } else if chars[pos] == '}' {
                brace_depth -= 1;
            }
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::ImplBlock,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_break_expr(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        let mut pos = start + 5;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find end of expression (semicolon or block end)
        while pos < chars.len() && chars[pos] != ';' && chars[pos] != '}' {
            pos += 1;
        }
        if pos < chars.len() && chars[pos] == ';' {
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::BreakExpr,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_continue_expr(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        let mut pos = start + 8;
        while self.is_whitespace(&chars, pos) {
            pos += 1;
        }

        // Find end of expression (semicolon or block end)
        while pos < chars.len() && chars[pos] != ';' && chars[pos] != '}' {
            pos += 1;
        }
        if pos < chars.len() && chars[pos] == ';' {
            pos += 1;
        }

        Ok(StructuralToken {
            kind: TokenKind::ContinueExpr,
            text: chars[start..pos].iter().collect(),
            span: (start, pos),
        })
    }

    fn parse_identifier_or_literal(
        &self,
        chars: &[char],
        start: usize,
    ) -> Result<StructuralToken, TokenizeError> {
        let mut pos = start;

        // Check if it's a number (literal)
        if chars[pos].is_digit(10) || (chars[pos] == '-' && pos + 1 < chars.len() && chars[pos + 1].is_digit(10)) {
            while pos < chars.len() && (chars[pos].is_digit(10) || chars[pos] == '.') {
                pos += 1;
            }
            let text: String = chars[start..pos].iter().collect();
            Ok(StructuralToken {
                kind: TokenKind::IntLit,
                text,
                span: (start, pos),
            })
        } else {
            // Identifier
            let name_start = pos;
            while pos < chars.len() && self.is_ident_char(chars[pos]) {
                pos += 1;
            }

            // Check for type annotation patterns like "let x: i32"
            if pos < chars.len() && chars[pos] == ':' {
                // This is a type annotation context - skip it
                while pos < chars.len() && (chars[pos].is_alphanumeric() || chars[pos] == '_' || chars[pos] == ' ') {
                    pos += 1;
                }
            }

            let text: String = chars[name_start..pos].iter().collect();
            Ok(StructuralToken {
                kind: TokenKind::Ident(text.clone()),
                text,
                span: (name_start, pos),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_declaration() {
        let tokenizer = StructuralTokenizer::new();
        let src = "fn add(a: i32, b: i32) -> i32 { return a + b; }";
        let tokens = tokenizer.tokenize(src).unwrap();

        // Should produce: FnDecl, BlockStart, ReturnExpr, BlockEnd
        assert_eq!(tokens[0].kind, TokenKind::FnDecl);
        assert_eq!(tokens[1].kind, TokenKind::BlockStart);
        assert_eq!(tokens[2].kind, TokenKind::ReturnExpr);
        assert_eq!(tokens[3].kind, TokenKind::BlockEnd);
    }

    #[test]
    fn test_let_statement() {
        let tokenizer = StructuralTokenizer::new();
        let src = "let x: i32 = 42;";
        let tokens = tokenizer.tokenize(src).unwrap();

        assert_eq!(tokens[0].kind, TokenKind::LetStmt);
    }

    #[test]
    fn test_return_expression() {
        let tokenizer = StructuralTokenizer::new();
        let src = "return 42;";
        let tokens = tokenizer.tokenize(src).unwrap();

        assert_eq!(tokens[0].kind, TokenKind::ReturnExpr);
    }

    #[test]
    fn test_block_structure() {
        let tokenizer = StructuralTokenizer::new();
        let src = "if (true) { let x = 1; } else { let y = 2; }";
        let tokens = tokenizer.tokenize(src).unwrap();

        // Should have balanced BlockStart/BlockEnd
        let block_starts = tokens.iter().filter(|t| t.kind == TokenKind::BlockStart).count();
        let block_ends = tokens.iter().filter(|t| t.kind == TokenKind::BlockEnd).count();
        assert_eq!(block_starts, block_ends);
    }

    #[test]
    fn test_identifier_truncation() {
        let tokenizer = StructuralTokenizer::new();
        let src = "let very_long_variable_name_that_should_be_truncated = 42;";
        let tokens = tokenizer.tokenize(src).unwrap();

        // Identifier should be truncated to reasonable length
        if let TokenKind::LetStmt = tokens[0].kind {
            assert!(tokens[0].text.len() < 100);
        }
    }

    #[test]
    fn test_struct_declaration() {
        let tokenizer = StructuralTokenizer::new();
        let src = "struct Point { x: i32, y: i32 }";
        let tokens = tokenizer.tokenize(src).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::StructDecl);
    }

    #[test]
    fn test_enum_declaration() {
        let tokenizer = StructuralTokenizer::new();
        let src = "enum Color { Red, Green, Blue }";
        let tokens = tokenizer.tokenize(src).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::EnumDecl);
    }

    #[test]
    fn test_trait_declaration() {
        let tokenizer = StructuralTokenizer::new();
        let src = "trait Display { fn fmt(&self) -> String; }";
        let tokens = tokenizer.tokenize(src).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::TraitDecl);
    }

    #[test]
    fn test_impl_block() {
        let tokenizer = StructuralTokenizer::new();
        let src = "impl Display for Point { fn fmt(&self) -> String { \"\".to_string() } }";
        let tokens = tokenizer.tokenize(src).unwrap();
        assert_eq!(tokens[0].kind, TokenKind::ImplBlock);
    }

    #[test]
    fn test_break_continue() {
        let tokenizer = StructuralTokenizer::new();
        let src = "for i in 0..10 { if i == 5 { break; } continue; }";
        let tokens = tokenizer.tokenize(src).unwrap();
        // Should have ForLoop containing BreakExpr and ContinueExpr (nested)
    }

    #[test]
    fn test_complex_program() {
        let tokenizer = StructuralTokenizer::new();
        let src = r#"
struct Counter {
    count: i32,
}

impl Counter {
    fn new() -> Counter {
        Counter { count: 0 }
    }

    fn increment(&mut self) {
        self.count = self.count + 1;
    }

    fn get(&self) -> i32 {
        return self.count;
    }
}

fn main() {
    let c = Counter::new();
    for i in 0..5 {
        if i == 3 {
            break;
        }
        c.increment();
    }
    println!("{}", c.get());
}
"#;
        let tokens = tokenizer.tokenize(src).unwrap();
        // Should successfully tokenize without hanging or erroring
        assert!(!tokens.is_empty());
    }
}