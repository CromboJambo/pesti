//! Structural tokenizer for Rust source code using `syn`.
//! Parses into AST via syn, then manually walks emitting semantic tokens.
//! Supports budget-aware collapsing for LLM context size management.

use std::collections::HashMap;
use syn::{Item, parse_file};

mod folder;
pub use folder::{Span, SpanItem, SpanKind, fold};

pub mod errors;
pub use errors::{
    legacy_to_structured, structured_to_legacy, AnalysisResult, ErrorContext, ErrorKind,
    LegacyTokenizeError, RecoveryConfig, Severity, TokenizerError, TokenizerResult,
};

pub mod semantic;
pub use semantic::*;

/// Precise location info for a token within source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpanInfo {
    /// 1-indexed line number.
    pub line: usize,
    /// 1-indexed column number.
    pub column: usize,
    /// Byte offset from file start.
    pub start_byte: usize,
    /// Byte offset after token end.
    pub end_byte: usize,
}

/// Structural token with position information
#[derive(Debug, Clone)]
pub struct StructuralToken {
    pub kind: TokenKind,
    pub text: String,
    /// Legacy span field (line_start, line_end) — kept for backward compat.
    pub span: (usize, usize),
    /// Precise location info (populated when source is available).
    pub location: Option<SpanInfo>,
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

    // Rust keywords
    Const,
    Mut,

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
            TokenKind::Const => write!(f, "CONST"),
            TokenKind::Mut => write!(f, "MUT"),
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

/// Budget configuration for budget-aware tokenization.
#[derive(Debug, Clone)]
pub struct Budget {
    /// Maximum number of structural tokens to emit.
    pub max_tokens: usize,
    /// Minimum statement count in a body worth eliding — smaller bodies are always emitted in full.
    pub min_body_stmts: usize,
}

impl Default for Budget {
    fn default() -> Self {
        Budget {
            max_tokens: 1000,
            min_body_stmts: 5,
        }
    }
}

/// Tag for node kind used in elision summaries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeKindTag {
    FnDecl,
    MatchExpr,
    Closure,
    IfElse,
    ForLoop,
    WhileLoop,
    Block,
    Other,
}

impl std::fmt::Display for NodeKindTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NodeKindTag::FnDecl => write!(f, "fn"),
            NodeKindTag::MatchExpr => write!(f, "match"),
            NodeKindTag::Closure => write!(f, "closure"),
            NodeKindTag::IfElse => write!(f, "if"),
            NodeKindTag::ForLoop => write!(f, "for"),
            NodeKindTag::WhileLoop => write!(f, "while"),
            NodeKindTag::Block => write!(f, "block"),
            NodeKindTag::Other => write!(f, "other"),
        }
    }
}

/// Marker for a subtree that was elided due to budget constraints.
#[derive(Debug, Clone)]
pub struct ElidedSpan {
    /// Byte range of the elided source.
    pub range: (usize, usize),
    /// Number of nodes collapsed into this marker.
    pub node_count: usize,
    /// Breakdown by node kind for consumer to assess what was skipped.
    pub kind_summary: Vec<(NodeKindTag, usize)>,
}

/// Emission result — either a token or an elision marker.
#[derive(Debug, Clone)]
pub enum Emission {
    Node(StructuralToken),
    Elided(ElidedSpan),
}

/// Tokenization output with semantic annotations.
#[derive(Debug, Clone)]
pub struct SemanticEmission {
    pub emission: Emission,
    /// Semantic tag at this position (if any). None for elision markers.
    pub tag: Option<SemanticTag>,
}

impl std::fmt::Display for Emission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Emission::Node(tok) => write!(f, "{}", tok.kind),
            Emission::Elided(span) => {
                let summary_parts: Vec<String> = span
                    .kind_summary
                    .iter()
                    .map(|(k, n)| format!("{}x{}", k, n))
                    .collect();
                write!(
                    f,
                    "ELIDED({} nodes: {})",
                    span.node_count,
                    summary_parts.join(", ")
                )
            }
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

    /// Tokenize Rust source into structural tokens (no budget).
    pub fn tokenize(&self, src: &str) -> Result<Vec<StructuralToken>, TokenizeError> {
        let ast = parse_file(src).map_err(|e| TokenizeError::Parse {
            pos: 0,
            msg: format!("syn parse error: {}", e),
        })?;

        let mut tokens = Vec::new();
        self.walk_items(&ast.items, src, &mut tokens);
        Ok(tokens)
    }

    /// Tokenize with budget-aware collapsing. Returns emissions including Elided markers.
    pub fn tokenize_with_budget(
        &self,
        src: &str,
        budget: Budget,
    ) -> Result<Vec<Emission>, TokenizeError> {
        let ast = parse_file(src).map_err(|e| TokenizeError::Parse {
            pos: 0,
            msg: format!("syn parse error: {}", e),
        })?;

        let mut emissions = Vec::new();
        let mut budget_state = BudgetState {
            max_tokens: budget.max_tokens,
            emitted: 0,
            min_body_stmts: budget.min_body_stmts,
        };

        self.walk_items_budgeted(&ast.items, &mut emissions, &mut budget_state, src);
        Ok(emissions)
    }

    /// Tokenize with budget and project semantic annotations onto each token position.
    pub fn tokenize_with_semantics(
        &self,
        src: &str,
        budget: Budget,
    ) -> Result<Vec<SemanticEmission>, TokenizeError> {
        // Structural pass first
        let emissions = self.tokenize_with_budget(src, budget)?;

        // Semantic analysis (lightweight string scanning)
        let ownership = analyze_ownership(src);
        let errors = analyze_errors(src);
        let mutability = analyze_mutability(src);

        // Project onto token positions
        let semantic_emissions: Vec<SemanticEmission> = emissions
            .into_iter()
            .enumerate()
            .map(|(i, emission)| {
                let own_ann = ownership.get(&i).cloned();
                let err_ann = errors.get(&i).cloned();
                let mut_ann = mutability.get(&i).cloned();

                let tag = if own_ann.is_some() || err_ann.is_some() || mut_ann.is_some() {
                    Some(SemanticTag::encode(
                        own_ann.as_ref().map(|a| a.mode),
                        err_ann.as_ref().map(|a| a.handling),
                        mut_ann.as_ref().is_some_and(|a| a.mutable),
                        own_ann
                            .as_ref()
                            .map(|a| a.confidence)
                            .or_else(|| err_ann.as_ref().map(|a| a.confidence))
                            .or_else(|| mut_ann.as_ref().map(|a| a.confidence))
                            .unwrap_or(0),
                    ))
                } else {
                    None
                };

                SemanticEmission { emission, tag }
            })
            .collect();

        Ok(semantic_emissions)
    }

    fn walk_items(&self, items: &[Item], src: &str, tokens: &mut Vec<StructuralToken>) {
        for item in items {
            match item {
                Item::Fn(f) => {
                    let sig = format!("fn {}", f.sig.ident);
                    self.emit_from_span(tokens, TokenKind::FnDecl, sig, src, &f.sig.ident);

                    // Walk function body statements
                    for stmt in &f.block.stmts {
                        self.walk_stmt(stmt, src, tokens);
                    }
                }
                Item::Struct(s) => {
                    let name = s.ident.to_string();
                    self.emit_from_span(tokens, TokenKind::StructDecl, name, src, &s.ident);
                    if let syn::Fields::Named(fields) = &s.fields {
                        for field in &fields.named {
                            if let Some(ident) = &field.ident {
                                self.emit_from_span(
                                    tokens,
                                    TokenKind::Ident(ident.to_string()),
                                    ident.to_string(),
                                    src,
                                    ident,
                                );
                            }
                        }
                    }
                }
                Item::Enum(e) => {
                    let name = e.ident.to_string();
                    self.emit_from_span(tokens, TokenKind::EnumDecl, name, src, &e.ident);
                    for variant in &e.variants {
                        self.emit_from_span(
                            tokens,
                            TokenKind::Ident(variant.ident.to_string()),
                            variant.ident.to_string(),
                            src,
                            &variant.ident,
                        );
                    }
                }
                Item::Trait(t) => {
                    let name = t.ident.to_string();
                    self.emit_from_span(tokens, TokenKind::TraitDecl, name, src, &t.ident);
                    for item in &t.items {
                        if let syn::TraitItem::Fn(m) = item {
                            self.emit_from_span(
                                tokens,
                                TokenKind::FnDecl,
                                format!("fn {}", m.sig.ident),
                                src,
                                &m.sig.ident,
                            );
                        }
                    }
                }
                Item::Impl(i) => {
                    self.emit_kind(tokens, TokenKind::ImplBlock);
                    for item in &i.items {
                        if let syn::ImplItem::Fn(m) = item {
                            self.emit_from_span(
                                tokens,
                                TokenKind::FnDecl,
                                format!("fn {}", m.sig.ident),
                                src,
                                &m.sig.ident,
                            );
                        }
                    }
                }
                Item::Use(_u) => {
                    self.emit_kind(tokens, TokenKind::UseStmt);
                }
                Item::Mod(m) => {
                    let name = m.ident.to_string();
                    self.emit_from_span(tokens, TokenKind::ModDecl, name, src, &m.ident);
                    if let Some((_brace, items)) = &m.content {
                        self.walk_items(items, src, tokens);
                    }
                }
                _ => {}
            }
        }
    }

    fn walk_stmt(&self, stmt: &syn::Stmt, src: &str, tokens: &mut Vec<StructuralToken>) {
        match stmt {
            syn::Stmt::Local(_let_stmt) => {
                self.emit_kind(tokens, TokenKind::LetStmt);
            }
            syn::Stmt::Expr(expr, semi) => {
                self.walk_expr(expr, src, tokens);
                if semi.is_some() {
                    self.emit_kind(tokens, TokenKind::Semicolon);
                }
            }
            syn::Stmt::Item(item) => {
                if let Item::Fn(inner) = item {
                    let sig = format!("fn {}", inner.sig.ident);
                    self.emit(tokens, TokenKind::FnDecl, sig);
                }
            }
            _ => {}
        }
    }

    fn walk_expr(&self, expr: &syn::Expr, src: &str, tokens: &mut Vec<StructuralToken>) {
        match expr {
            syn::Expr::If(eif) => {
                self.emit_kind(tokens, TokenKind::IfElse);
                // Walk into the then branch to find break/continue/etc
                for stmt in &eif.then_branch.stmts {
                    self.walk_stmt(stmt, src, tokens);
                }
            }
            syn::Expr::While(_ewhile) => {
                self.emit_kind(tokens, TokenKind::WhileLoop);
            }
            syn::Expr::ForLoop(efor) => {
                self.emit_kind(tokens, TokenKind::ForLoop);
                // Walk into loop body to find break/continue
                for stmt in &efor.body.stmts {
                    self.walk_stmt(stmt, src, tokens);
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
                    self.walk_stmt(stmt, src, tokens);
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
        self.emit_with_location(tokens, kind, text, None);
    }

    fn emit_kind(&self, tokens: &mut Vec<StructuralToken>, kind: TokenKind) {
        self.emit_with_location(tokens, kind, String::new(), None);
    }

    /// Emit a token with precise location info computed from syn span.
    fn emit_from_span<S>(&self, tokens: &mut Vec<StructuralToken>, kind: TokenKind, text: String, src: &str, node: &S)
    where
        S: syn::spanned::Spanned,
    {
        let span = node.span();

        // Try to get source text for this span directly
        let span_text = match span.source_text() {
            Some(t) => t,
            None => text.clone(),
        };

        // Find the location by searching for the span text in the source
        if let Some(byte_offset) = src.find(&span_text) {
            let loc = compute_span_info(src, byte_offset, byte_offset + span_text.len());
            self.emit_with_location(tokens, kind, span_text, Some(loc));
        } else {
            // Fallback: no location info
            self.emit_with_location(tokens, kind, span_text, None);
        }
    }

    /// Emit a token with explicit location info.
    fn emit_with_location(
        &self,
        tokens: &mut Vec<StructuralToken>,
        kind: TokenKind,
        text: String,
        location: Option<SpanInfo>,
    ) {
        let line = location.map(|l| l.line).unwrap_or(0);
        tokens.push(StructuralToken {
            kind,
            text,
            span: (line, line),
            location,
        });
    }

    // Budget-aware walkers

    fn walk_items_budgeted(
        &self,
        items: &[Item],
        emissions: &mut Vec<Emission>,
        budget: &mut BudgetState,
        src: &str,
    ) {
        for item in items {
            if budget.emitted >= budget.max_tokens {
                break;
            }

            match item {
                Item::Fn(f) => {
                    let sig = format!("fn {}", f.sig.ident);
                    self.emit_emission_from_span(emissions, TokenKind::FnDecl, sig, src, &f.sig.ident);
                    budget.emitted += 1;

                    // Check if body is worth eliding (many statements)
                    let body_size = self.get_body_size(&f.block);
                    if body_size >= budget.min_body_stmts {
                        // Count what's in the body without emitting
                        let (node_count, kind_summary) = self.count_body_nodes(&f.block);
                        if node_count > 0 && budget.emitted < budget.max_tokens {
                            // Emit elision marker instead of walking
                            emissions.push(Emission::Elided(ElidedSpan {
                                range: (0, 0), // span tracking not implemented yet
                                node_count,
                                kind_summary,
                            }));
                            budget.emitted += 1;
                        }
                    } else {
                        // Small body — walk normally
                        for stmt in &f.block.stmts {
                            if budget.emitted >= budget.max_tokens {
                                break;
                            }
                            self.walk_stmt_budgeted(stmt, emissions, budget);
                        }
                    }
                }
                Item::Struct(s) => {
                    let name = s.ident.to_string();
                    self.emit_emission_from_span(emissions, TokenKind::StructDecl, name, src, &s.ident);
                    budget.emitted += 1;
                }
                Item::Enum(e) => {
                    let name = e.ident.to_string();
                    self.emit_emission_from_span(emissions, TokenKind::EnumDecl, name, src, &e.ident);
                    budget.emitted += 1;
                }
                Item::Trait(t) => {
                    let name = t.ident.to_string();
                    self.emit_emission_from_span(emissions, TokenKind::TraitDecl, name, src, &t.ident);
                    budget.emitted += 1;
                }
                Item::Impl(_i) => {
                    self.emit_emission(emissions, TokenKind::ImplBlock, "impl".to_string());
                    budget.emitted += 1;
                }
                Item::Use(_u) => {
                    self.emit_emission(emissions, TokenKind::UseStmt, "use".to_string());
                    budget.emitted += 1;
                }
                Item::Mod(m) => {
                    let name = m.ident.to_string();
                    self.emit_emission_from_span(emissions, TokenKind::ModDecl, name, src, &m.ident);
                    budget.emitted += 1;
                    if let Some((_brace, items)) = &m.content {
                        self.walk_items_budgeted(items, emissions, budget, src);
                    }
                }
                _ => {}
            }
        }
    }

    fn walk_stmt_budgeted(
        &self,
        stmt: &syn::Stmt,
        emissions: &mut Vec<Emission>,
        budget: &mut BudgetState,
    ) {
        if budget.emitted >= budget.max_tokens {
            return;
        }

        match stmt {
            syn::Stmt::Local(_let_stmt) => {
                self.emit_emission(emissions, TokenKind::LetStmt, String::new());
                budget.emitted += 1;
            }
            syn::Stmt::Expr(expr, semi) => {
                self.walk_expr_budgeted(expr, emissions, budget);
                if semi.is_some() {
                    self.emit_emission(emissions, TokenKind::Semicolon, String::new());
                    budget.emitted += 1;
                }
            }
            _ => {}
        }
    }

    fn walk_expr_budgeted(
        &self,
        expr: &syn::Expr,
        emissions: &mut Vec<Emission>,
        budget: &mut BudgetState,
    ) {
        if budget.emitted >= budget.max_tokens {
            return;
        }

        match expr {
            syn::Expr::If(eif) => {
                self.emit_emission(emissions, TokenKind::IfElse, String::new());
                budget.emitted += 1;
                for stmt in &eif.then_branch.stmts {
                    self.walk_stmt_budgeted(stmt, emissions, budget);
                }
            }
            syn::Expr::While(_ewhile) => {
                self.emit_emission(emissions, TokenKind::WhileLoop, String::new());
                budget.emitted += 1;
            }
            syn::Expr::ForLoop(efor) => {
                self.emit_emission(emissions, TokenKind::ForLoop, String::new());
                budget.emitted += 1;
                for stmt in &efor.body.stmts {
                    self.walk_stmt_budgeted(stmt, emissions, budget);
                }
            }
            syn::Expr::Match(_ematch) => {
                self.emit_emission(emissions, TokenKind::MatchExpr, String::new());
                budget.emitted += 1;
            }
            syn::Expr::Return(_eret) => {
                self.emit_emission(emissions, TokenKind::ReturnExpr, String::new());
                budget.emitted += 1;
            }
            syn::Expr::Break(_) => {
                self.emit_emission(emissions, TokenKind::BreakExpr, String::new());
                budget.emitted += 1;
            }
            syn::Expr::Continue(_) => {
                self.emit_emission(emissions, TokenKind::ContinueExpr, String::new());
                budget.emitted += 1;
            }
            syn::Expr::Call(_ecall) => {
                self.emit_emission(emissions, TokenKind::CallExpr, String::new());
                budget.emitted += 1;
            }
            syn::Expr::MethodCall(emethod) => {
                let method_name = emethod.method.to_string();
                self.emit_emission(emissions, TokenKind::MethodCall, method_name);
                budget.emitted += 1;
            }
            _ => {}
        }
    }

    fn emit_emission(&self, emissions: &mut Vec<Emission>, kind: TokenKind, text: String) {
        self.emit_emission_with_location(emissions, kind, text, None);
    }

    fn emit_emission_from_span<T: syn::spanned::Spanned>(
        &self,
        emissions: &mut Vec<Emission>,
        kind: TokenKind,
        text: String,
        src: &str,
        node: &T,
    ) {
        let span = node.span();

        // Try to get source text for this span directly
        let span_text = match span.source_text() {
            Some(t) => t,
            None => text.clone(),
        };

        // Find the location by searching for the span text in the source
        if let Some(byte_offset) = src.find(&span_text) {
            let loc = compute_span_info(src, byte_offset, byte_offset + span_text.len());
            self.emit_emission_with_location(emissions, kind, span_text, Some(loc));
        } else {
            // Fallback: no location info
            self.emit_emission_with_location(emissions, kind, span_text, None);
        }
    }

    fn emit_emission_with_location(
        &self,
        emissions: &mut Vec<Emission>,
        kind: TokenKind,
        text: String,
        location: Option<SpanInfo>,
    ) {
        let line = location.map(|l| l.line).unwrap_or(0);
        emissions.push(Emission::Node(StructuralToken {
            kind,
            text,
            span: (line, line),
            location,
        }));
    }

    fn get_body_size(&self, block: &syn::Block) -> usize {
        // Count statements as proxy for body complexity/size
        block.stmts.len()
    }

    fn count_body_nodes(&self, block: &syn::Block) -> (usize, Vec<(NodeKindTag, usize)>) {
        let mut counts: HashMap<NodeKindTag, usize> = HashMap::new();
        let mut total = 0;

        for stmt in &block.stmts {
            match stmt {
                syn::Stmt::Local(_) => {
                    *counts.entry(NodeKindTag::Other).or_insert(0) += 1;
                    total += 1;
                }
                syn::Stmt::Expr(expr, _) => {
                    self.count_expr_nodes(expr, &mut counts, &mut total);
                }
                _ => {}
            }
        }

        let mut summary: Vec<(NodeKindTag, usize)> = counts.into_iter().collect();
        summary.sort_by(|a, b| b.1.cmp(&a.1));
        (total, summary)
    }

    fn count_expr_nodes(
        &self,
        expr: &syn::Expr,
        counts: &mut HashMap<NodeKindTag, usize>,
        total: &mut usize,
    ) {
        match expr {
            syn::Expr::If(eif) => {
                *counts.entry(NodeKindTag::IfElse).or_insert(0) += 1;
                *total += 1;
                for stmt in &eif.then_branch.stmts {
                    self.count_stmt_nodes(stmt, counts, total);
                }
            }
            syn::Expr::Match(_ematch) => {
                *counts.entry(NodeKindTag::MatchExpr).or_insert(0) += 1;
                *total += 1;
            }
            syn::Expr::ForLoop(efor) => {
                *counts.entry(NodeKindTag::ForLoop).or_insert(0) += 1;
                *total += 1;
                for stmt in &efor.body.stmts {
                    self.count_stmt_nodes(stmt, counts, total);
                }
            }
            syn::Expr::While(_ewhile) => {
                *counts.entry(NodeKindTag::WhileLoop).or_insert(0) += 1;
                *total += 1;
            }
            syn::Expr::Closure(_eclosure) => {
                *counts.entry(NodeKindTag::Closure).or_insert(0) += 1;
                *total += 1;
            }
            _ => {}
        }
    }

    fn count_stmt_nodes(
        &self,
        stmt: &syn::Stmt,
        counts: &mut HashMap<NodeKindTag, usize>,
        total: &mut usize,
    ) {
        match stmt {
            syn::Stmt::Local(_) => {
                *counts.entry(NodeKindTag::Other).or_insert(0) += 1;
                *total += 1;
            }
            syn::Stmt::Expr(expr, _) => {
                self.count_expr_nodes(expr, counts, total);
            }
            _ => {}
        }
    }
}

impl Default for StructuralTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal budget state tracker.
struct BudgetState {
    max_tokens: usize,
    emitted: usize,
    min_body_stmts: usize,
}

/// Compute SpanInfo from byte offsets within source.
fn compute_span_info(src: &str, start_byte: usize, end_byte: usize) -> SpanInfo {
    let bytes = src.as_bytes();
    
    // Clamp to valid range
    let start = start_byte.min(bytes.len());
    let end = end_byte.min(bytes.len());
    
    // Compute line and column from byte offset
    let mut line = 1;
    let mut col = 1;
    for (i, &b) in bytes.iter().enumerate() {
        if i >= start { break; }
        if b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    
    SpanInfo {
        line,
        column: col,
        start_byte,
        end_byte,
    }
}

/// Extract source text for a span.
fn extract_span_text(src: &str, start_byte: usize, end_byte: usize) -> String {
    let bytes = src.as_bytes();
    let start = start_byte.min(bytes.len());
    let end = end_byte.min(bytes.len());
    String::from_utf8_lossy(&bytes[start..end]).to_string()
}

/// Count bytes from start of source to the given line/column.
fn count_bytes_before_line_col(src: &str, target_line: u32, target_col: u32) -> usize {
    let mut line = 1u32;
    let mut col = 1u32;
    
    for (i, b) in src.bytes().enumerate() {
        if line == target_line && col == target_col {
            return i;
        }
        if b == b'\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    
    src.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fn_declaration() {
        let tokenizer = StructuralTokenizer::new();
        let src = "fn add(a: i32, b: i32) -> i32 { return a + b; }";
        let tokens = tokenizer.tokenize(src).unwrap();

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

        assert!(tokens.iter().any(|t| t.kind == TokenKind::StructDecl));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::ImplBlock));
        assert!(tokens.iter().any(|t| t.kind == TokenKind::FnDecl));
    }

    #[test]
    fn test_budget_basic() {
        let tokenizer = StructuralTokenizer::new();
        let src = "fn foo() { let x = 1; let y = 2; return x + y; }";
        let budget = Budget {
            max_tokens: 5,
            min_body_stmts: 1,
        };
        let emissions = tokenizer.tokenize_with_budget(src, budget).unwrap();

        // Should emit fewer than unlimited mode
        assert!(emissions.len() <= 5);
    }

    #[test]
    fn test_budget_elides_large_bodies() {
        // Build a function with a large body (many statements)
        let mut src = "fn big() {".to_string();
        for i in 0..20 {
            src.push_str(&format!("let x{} = {};\n", i, i));
        }
        src.push_str("}");

        let tokenizer = StructuralTokenizer::new();
        let budget = Budget {
            max_tokens: 10,
            min_body_stmts: 1,
        };
        let emissions = tokenizer.tokenize_with_budget(&src, budget).unwrap();

        // Should have FnDecl + Elided marker, not all the LetStmts
        assert!(
            emissions
                .iter()
                .any(|e| matches!(e, Emission::Node(t) if t.kind == TokenKind::FnDecl))
        );
        assert!(emissions.iter().any(|e| matches!(e, Emission::Elided(_))));
    }

    #[test]
    fn test_elided_span_has_summary() {
        let src =
            "fn big() { for i in 0..10 { if i == 5 { break; } } match x { 1 => {}, _ => {} } }";
        let tokenizer = StructuralTokenizer::new();
        let budget = Budget {
            max_tokens: 5,
            min_body_stmts: 1,
        };
        let emissions = tokenizer.tokenize_with_budget(src, budget).unwrap();

        // Find elided span and verify it has kind summary
        for e in &emissions {
            if let Emission::Elided(span) = e {
                assert!(!span.kind_summary.is_empty());
                assert!(span.node_count > 0);
            }
        }
    }
}
