//! Structural tokenizer for Rust source code using `syn`.
//! Parses into AST via syn, then manually walks emitting semantic tokens.

use std::collections::HashMap;
use syn::{Item, parse_file};

/// Structural token with position information
#[derive(Debug, Clone)]
pub struct StructuralToken {
    pub kind: TokenKind,
    pub text: String,
    pub span: (usize, usize),
}

/// Kinds of structural tokens we emit
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Declarations
    FnDecl,
    LetStmt,
    StructDecl,
    EnumDecl,
    TraitDecl,
    ImplBlock,
    UseStmt,
    ModDecl,

    // Control flow
    IfElse,
    WhileLoop,
    ForLoop,
    MatchExpr,
    ReturnExpr,
    BreakExpr,
    ContinueExpr,

    // Expressions & operations
    CallExpr,
    FieldAccess,
    MethodCall,
    BinaryOp,
    UnaryOp,
    ParenExpr,
    ExprStmt,

    // Literals
    IntLit,
    FloatLit,
    StrLit,
    CharLit,
    BoolLit,
    ByteLit,
    ArrayLit,
    TupleLit,
    SliceExpr,

    // Types & casts
    TypeAnnotation,
    CastExpr,
    Dereference,
    Reference,

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
    Arrow,
    FatArrow,

    // Identifiers and names
    Ident(String),
    Keyword(String),

    // Rust-specific features
    MacroInvocation,
    Attribute,
    Lifetime,
    GenericParams,
    WhereClause,
    AsyncBlock,
    MoveClosure,
    Closure,
    Punctuator,

    // Visibility and modifiers
    Pub,
    Private,
    Static,
}

/// Errors from the structural tokenizer
#[derive(Debug)]
pub enum TokenizeError {
    Parse { pos: usize, msg: String },
}

impl std::fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizeError::Parse { pos, msg } => {
                write!(f, "Parse error at position {}: {}", pos, msg)
            }
        }
    }
}

impl std::error::Error for TokenizeError {}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenKind::FnDecl => write!(f, "FN_DECL"),
            TokenKind::LetStmt => write!(f, "LET_STMT"),
            TokenKind::StructDecl => write!(f, "STRUCT_DECL"),
            TokenKind::EnumDecl => write!(f, "ENUM_DECL"),
            TokenKind::TraitDecl => write!(f, "TRAIT_DECL"),
            TokenKind::ImplBlock => write!(f, "IMPL_BLOCK"),
            TokenKind::UseStmt => write!(f, "USE_STMT"),
            TokenKind::ModDecl => write!(f, "MOD_DECL"),
            TokenKind::IfElse => write!(f, "IF_ELSE"),
            TokenKind::WhileLoop => write!(f, "WHILE_LOOP"),
            TokenKind::ForLoop => write!(f, "FOR_LOOP"),
            TokenKind::MatchExpr => write!(f, "MATCH_EXPR"),
            TokenKind::ReturnExpr => write!(f, "RETURN_EXPR"),
            TokenKind::BreakExpr => write!(f, "BREAK_EXPR"),
            TokenKind::ContinueExpr => write!(f, "CONTINUE_EXPR"),
            TokenKind::CallExpr => write!(f, "CALL_EXPR"),
            TokenKind::FieldAccess => write!(f, "FIELD_ACCESS"),
            TokenKind::MethodCall => write!(f, "METHOD_CALL"),
            TokenKind::BinaryOp => write!(f, "BINARY_OP"),
            TokenKind::UnaryOp => write!(f, "UNARY_OP"),
            TokenKind::ParenExpr => write!(f, "PAREN_EXPR"),
            TokenKind::ExprStmt => write!(f, "EXPR_STMT"),
            TokenKind::IntLit => write!(f, "INT_LIT"),
            TokenKind::FloatLit => write!(f, "FLOAT_LIT"),
            TokenKind::StrLit => write!(f, "STR_LIT"),
            TokenKind::CharLit => write!(f, "CHAR_LIT"),
            TokenKind::BoolLit => write!(f, "BOOL_LIT"),
            TokenKind::ByteLit => write!(f, "BYTE_LIT"),
            TokenKind::ArrayLit => write!(f, "ARRAY_LIT"),
            TokenKind::TupleLit => write!(f, "TUPLE_LIT"),
            TokenKind::SliceExpr => write!(f, "SLICE_EXPR"),
            TokenKind::TypeAnnotation => write!(f, "TYPE_ANNOT"),
            TokenKind::CastExpr => write!(f, "CAST_EXPR"),
            TokenKind::Dereference => write!(f, "DEREF"),
            TokenKind::Reference => write!(f, "REF"),
            TokenKind::BlockStart => write!(f, "BLOCK_START"),
            TokenKind::BlockEnd => write!(f, "BLOCK_END"),
            TokenKind::ParenOpen => write!(f, "("),
            TokenKind::ParenClose => write!(f, ")"),
            TokenKind::BraceOpen => write!(f, "{{"),
            TokenKind::BraceClose => write!(f, "}}"),
            TokenKind::BracketOpen => write!(f, "["),
            TokenKind::BracketClose => write!(f, "]"),
            TokenKind::Semicolon => write!(f, ";"),
            TokenKind::Comma => write!(f, ","),
            TokenKind::Colon => write!(f, ":"),
            TokenKind::Dot => write!(f, "."),
            TokenKind::Arrow => write!(f, "->"),
            TokenKind::FatArrow => write!(f, "=>"),
            TokenKind::Ident(name) => write!(f, "IDENT({})", name),
            TokenKind::Keyword(kw) => write!(f, "KW({})", kw),
            TokenKind::MacroInvocation => write!(f, "MACRO"),
            TokenKind::Attribute => write!(f, "ATTR"),
            TokenKind::Lifetime => write!(f, "LIFETIME"),
            TokenKind::GenericParams => write!(f, "GENERIC_PARAMS"),
            TokenKind::WhereClause => write!(f, "WHERE"),
            TokenKind::AsyncBlock => write!(f, "ASYNC"),
            TokenKind::MoveClosure => write!(f, "MOVE_CLOSURE"),
            TokenKind::Closure => write!(f, "CLOSURE"),
            TokenKind::Punctuator => write!(f, "PUNCT"),
            TokenKind::Pub => write!(f, "pub"),
            TokenKind::Private => write!(f, "private"),
            TokenKind::Static => write!(f, "static"),
        }
    }
}

/// The structural tokenizer — uses syn to parse, then manually walks AST.
pub struct StructuralTokenizer {
    #[allow(dead_code)]
    vocab: HashMap<String, u32>,
}

impl StructuralTokenizer {
    pub fn new() -> Self {
        StructuralTokenizer {
            vocab: HashMap::new(),
        }
    }

    /// Tokenize Rust source into structural tokens using syn's parser.
    pub fn tokenize(&self, src: &str) -> Result<Vec<StructuralToken>, TokenizeError> {
        let ast = parse_file(src).map_err(|e| TokenizeError::Parse {
            pos: 0,
            msg: format!("syn parse error: {}", e),
        })?;

        let mut tokens = Vec::new();
        self.walk_items(&ast.items, &mut tokens);
        Ok(tokens)
    }

    fn walk_items(&self, items: &[Item], tokens: &mut Vec<StructuralToken>) {
        for item in items {
            match item {
                Item::Fn(f) => {
                    let sig = format!("fn {}", f.sig.ident);
                    self.emit(tokens, TokenKind::FnDecl, sig);

                    // Walk function body statements
                    for stmt in &f.block.stmts {
                        self.walk_stmt(stmt, tokens);
                    }
                }
                Item::Struct(s) => {
                    let name = s.ident.to_string();
                    self.emit(tokens, TokenKind::StructDecl, name);
                    if let syn::Fields::Named(fields) = &s.fields {
                        for field in &fields.named {
                            if let Some(ident) = &field.ident {
                                self.emit(
                                    tokens,
                                    TokenKind::Ident(ident.to_string()),
                                    ident.to_string(),
                                );
                            }
                        }
                    }
                }
                Item::Enum(e) => {
                    let name = e.ident.to_string();
                    self.emit(tokens, TokenKind::EnumDecl, name);
                    for variant in &e.variants {
                        self.emit(
                            tokens,
                            TokenKind::Ident(variant.ident.to_string()),
                            variant.ident.to_string(),
                        );
                    }
                }
                Item::Trait(t) => {
                    let name = t.ident.to_string();
                    self.emit(tokens, TokenKind::TraitDecl, name);
                    for item in &t.items {
                        if let syn::TraitItem::Fn(m) = item {
                            self.emit(tokens, TokenKind::FnDecl, format!("fn {}", m.sig.ident));
                        }
                    }
                }
                Item::Impl(i) => {
                    self.emit(tokens, TokenKind::ImplBlock, "impl".to_string());
                    for item in &i.items {
                        if let syn::ImplItem::Fn(m) = item {
                            self.emit(tokens, TokenKind::FnDecl, format!("fn {}", m.sig.ident));
                        }
                    }
                }
                Item::Use(_u) => {
                    self.emit(tokens, TokenKind::UseStmt, "use".to_string());
                }
                Item::Mod(m) => {
                    let name = m.ident.to_string();
                    self.emit(tokens, TokenKind::ModDecl, name);
                    if let Some((_brace, items)) = &m.content {
                        self.walk_items(items, tokens);
                    }
                }
                _ => {}
            }
        }
    }

    fn walk_stmt(&self, stmt: &syn::Stmt, tokens: &mut Vec<StructuralToken>) {
        match stmt {
            syn::Stmt::Local(_let_stmt) => {
                self.emit_kind(tokens, TokenKind::LetStmt);
            }
            syn::Stmt::Expr(expr, semi) => {
                self.walk_expr(expr, tokens);
                if semi.is_some() {
                    self.emit_kind(tokens, TokenKind::Semicolon);
                }
            }
            syn::Stmt::Item(item) => match item {
                Item::Fn(inner) => {
                    let sig = format!("fn {}", inner.sig.ident);
                    self.emit(tokens, TokenKind::FnDecl, sig);
                }
                _ => {}
            },
            _ => {}
        }
    }

    fn walk_expr(&self, expr: &syn::Expr, tokens: &mut Vec<StructuralToken>) {
        match expr {
            syn::Expr::If(eif) => {
                self.emit_kind(tokens, TokenKind::IfElse);
                // Walk into the then branch to find break/continue/etc
                for stmt in &eif.then_branch.stmts {
                    self.walk_stmt(stmt, tokens);
                }
            }
            syn::Expr::While(_ewhile) => {
                self.emit_kind(tokens, TokenKind::WhileLoop);
            }
            syn::Expr::ForLoop(efor) => {
                self.emit_kind(tokens, TokenKind::ForLoop);
                // Walk into loop body to find break/continue
                for stmt in &efor.body.stmts {
                    self.walk_stmt(stmt, tokens);
                }
            }
            syn::Expr::Match(_ematch) => {
                self.emit_kind(tokens, TokenKind::MatchExpr);
            }
            syn::Expr::Return(_eret) => {
                self.emit_kind(tokens, TokenKind::ReturnExpr);
            }
            syn::Expr::Break(_) => {
                self.emit_kind(tokens, TokenKind::BreakExpr);
            }
            syn::Expr::Continue(_) => {
                self.emit_kind(tokens, TokenKind::ContinueExpr);
            }
            syn::Expr::Call(_ecall) => {
                self.emit_kind(tokens, TokenKind::CallExpr);
            }
            syn::Expr::MethodCall(emethod) => {
                let method_name = emethod.method.to_string();
                self.emit(tokens, TokenKind::MethodCall, method_name);
            }
            syn::Expr::Field(efield) => {
                let field_name = match &efield.member {
                    syn::Member::Named(ident) => ident.to_string(),
                    syn::Member::Unnamed(index) => index.index.to_string(),
                };
                self.emit(tokens, TokenKind::FieldAccess, field_name);
            }
            syn::Expr::Binary(ebin) => {
                // BinOp doesn't impl Debug/Display — map to string manually
                let op_str = match &ebin.op {
                    syn::BinOp::Add(_) => "+",
                    syn::BinOp::Sub(_) => "-",
                    syn::BinOp::Mul(_) => "*",
                    syn::BinOp::Div(_) => "/",
                    syn::BinOp::Rem(_) => "%",
                    syn::BinOp::And(_) => "&&",
                    syn::BinOp::Or(_) => "||",
                    syn::BinOp::Eq(_) => "==",
                    syn::BinOp::Ne(_) => "!=",
                    syn::BinOp::Lt(_) => "<",
                    syn::BinOp::Le(_) => "<=",
                    syn::BinOp::Gt(_) => ">",
                    syn::BinOp::Ge(_) => ">=",
                    syn::BinOp::Shl(_) => "<<",
                    syn::BinOp::Shr(_) => ">>",
                    syn::BinOp::BitAnd(_) => "&",
                    syn::BinOp::BitOr(_) => "|",
                    syn::BinOp::BitXor(_) => "^",
                    _ => "?",
                };
                self.emit(tokens, TokenKind::BinaryOp, op_str.to_string());
            }
            syn::Expr::Unary(eunary) => {
                let op_str = match &eunary.op {
                    syn::UnOp::Deref(_) => "*",
                    syn::UnOp::Not(_) => "!",
                    syn::UnOp::Neg(_) => "-",
                    _ => "?",
                };
                self.emit(tokens, TokenKind::UnaryOp, op_str.to_string());
            }
            syn::Expr::Paren(_eparen) => {
                self.emit_kind(tokens, TokenKind::ParenExpr);
            }
            syn::Expr::Lit(elit) => match &elit.lit {
                syn::Lit::Int(l) => {
                    self.emit(tokens, TokenKind::IntLit, l.to_string());
                }
                syn::Lit::Float(f) => {
                    self.emit(tokens, TokenKind::FloatLit, f.to_string());
                }
                syn::Lit::Str(s) => {
                    let val = s.value();
                    let truncated = if val.len() > 50 {
                        format!("{}...", &val[..50])
                    } else {
                        val
                    };
                    self.emit(tokens, TokenKind::StrLit, truncated);
                }
                syn::Lit::Char(c) => {
                    let ch = c.value();
                    self.emit(tokens, TokenKind::CharLit, format!("{}", ch));
                }
                syn::Lit::Bool(b) => {
                    self.emit(tokens, TokenKind::BoolLit, b.value().to_string());
                }
                _ => {}
            },
            syn::Expr::Array(_earray) => {
                self.emit_kind(tokens, TokenKind::ArrayLit);
            }
            syn::Expr::Tuple(_etuple) => {
                self.emit_kind(tokens, TokenKind::TupleLit);
            }
            syn::Expr::Index(_eindex) => {
                self.emit_kind(tokens, TokenKind::SliceExpr);
            }
            syn::Expr::Cast(_ecast) => {
                self.emit_kind(tokens, TokenKind::CastExpr);
            }
            // Deref is a Unary op in syn 2.x, not its own variant
            syn::Expr::Reference(eref) => {
                self.emit_kind(tokens, TokenKind::Reference);
                if eref.mutability.is_some() {
                    self.emit(
                        tokens,
                        TokenKind::Ident("mut".to_string()),
                        "mut".to_string(),
                    );
                }
            }
            syn::Expr::Async(_easync) => {
                self.emit_kind(tokens, TokenKind::AsyncBlock);
            }
            syn::Expr::Closure(_eclosure) => {
                // Detect move by looking for "move" in source (syn doesn't expose this directly)
                self.emit_kind(tokens, TokenKind::Closure);
            }
            syn::Expr::Macro(emacro) => {
                let name = emacro
                    .mac
                    .path
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .unwrap_or_default();
                self.emit(tokens, TokenKind::MacroInvocation, name);
            }
            syn::Expr::Block(eblock) => {
                // ExprBlock has a `block` field of type Block
                for stmt in &eblock.block.stmts {
                    self.walk_stmt(stmt, tokens);
                }
            }
            syn::Expr::Path(epath) => {
                let segments = epath
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>();
                if !segments.is_empty() {
                    self.emit(
                        tokens,
                        TokenKind::Ident(segments.join("::")),
                        segments.join("::"),
                    );
                }
            }
            _ => {}
        }
    }

    fn emit(&self, tokens: &mut Vec<StructuralToken>, kind: TokenKind, text: String) {
        tokens.push(StructuralToken {
            kind,
            text,
            span: (0, 0), // No source spans in syn-based approach
        });
    }

    fn emit_kind(&self, tokens: &mut Vec<StructuralToken>, kind: TokenKind) {
        self.emit(tokens, kind, String::new());
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

        // Should produce: FnDecl, ReturnExpr
        assert_eq!(tokens[0].kind, TokenKind::FnDecl);
        assert_eq!(tokens[1].kind, TokenKind::ReturnExpr);
    }

    #[test]
    fn test_let_statement() {
        let tokenizer = StructuralTokenizer::new();
        let src = "fn main() { let x: i32 = 42; }";
        let tokens = tokenizer.tokenize(src).unwrap();

        assert_eq!(tokens[0].kind, TokenKind::FnDecl);
        assert_eq!(tokens[1].kind, TokenKind::LetStmt);
    }

    #[test]
    fn test_return_expression() {
        let tokenizer = StructuralTokenizer::new();
        let src = "fn main() { return 42; }";
        let tokens = tokenizer.tokenize(src).unwrap();

        assert_eq!(tokens[0].kind, TokenKind::FnDecl);
        assert_eq!(tokens[1].kind, TokenKind::ReturnExpr);
    }

    #[test]
    fn test_block_structure() {
        let tokenizer = StructuralTokenizer::new();
        let src = "fn main() { if true { let x = 1; } else { let y = 2; } }";
        let tokens = tokenizer.tokenize(src).unwrap();

        // Should have IfElse token
        assert!(tokens.iter().any(|t| t.kind == TokenKind::IfElse));
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
        let src = "fn main() { for i in 0..10 { if i == 5 { break; } continue; } }";
        let tokens = tokenizer.tokenize(src).unwrap();

        assert!(tokens.iter().any(|t| t.kind == TokenKind::ForLoop));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::BreakExpr));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::ContinueExpr));
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
}
"#;
        let tokens = tokenizer.tokenize(src).unwrap();

        // Should have struct, impl, and function declarations
        assert!(tokens.iter().any(|t| t.kind == TokenKind::StructDecl));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::ImplBlock));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::FnDecl));
    }
}
