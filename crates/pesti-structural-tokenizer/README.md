# pesti-structural-tokenizer

Rust structural tokenizer for LLM code understanding — parses Rust source into AST via `syn`, then emits a compact stream of semantic tokens that encode syntactic structure while discarding implementation details.

## What This Does

Traditional byte-pair encoding (BPE) tokenizers treat source code as opaque text. This crate takes a different approach: it uses `syn` to parse Rust source into a complete AST, then walks the tree emitting **structural tokens** that capture declarations, control flow, and expressions while eliding bodies of non-declaration functions.

The result is that LLMs get explicit signals about code structure without having to infer it from whitespace and punctuation alone — and the token count drops dramatically (57KB → 2KB structural digests). This improves code understanding, generation accuracy, and reduces hallucination of syntactically invalid code.

## Why Structural Tokenization?

LLMs trained on raw BPE tokens must learn Rust's syntax implicitly. Structural tokenization makes this knowledge **explicit**:

- `FN_DECL` — unambiguous function definition boundary
- `BRACE_OPEN` / `BRACE_CLOSE` — explicit block delimiters
- `IF_ELSE` — control flow structure is first-class
- `STRUCT_FIELD` — data layout boundaries are clear
- `MATCH_EXPR` — pattern matching expressions are distinct units

This matters for code generation, refactoring tools, and any agentic workflow that needs to understand Rust source.

## Installation

```toml
[dependencies]
pesti-structural-tokenizer = "0.1"
```

## Quick Start

```rust
use pesti_structural_tokenizer::{StructuralTokenizer, TokenKind};

let code = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;

let tokenizer = StructuralTokenizer::new();
let tokens = tokenizer.tokenize(code)?;

// Count structural vs content tokens
let structural = tokens.iter()
    .filter(|t| matches!(t.kind, TokenKind::FnDecl | TokenKind::ReturnExpr))
    .count();

println!("Total tokens: {}", tokens.len());
println!("Structural tokens: {}", structural);
```

## API Overview

### Core Types

- `StructuralTokenizer` — the tokenizer instance (use `new()` or `Default::default()`)
- `StructuralToken` — a single token with kind, text, and source span
- `TokenKind` — enum of ~90 distinct structural token kinds
- `TokenizeError` — parse errors from syn

### Main Functions

```rust
let tokenizer = StructuralTokenizer::new();

// Tokenize Rust source into structural tokens
let tokens = tokenizer.tokenize(source)?;
```

### Token Kinds

The tokenizer produces ~90 distinct token kinds covering:

- **Definitions**: `FN_DECL`, `STRUCT_DECL`, `ENUM_DECL`, `TRAIT_DECL`, `IMPL_BLOCK`
- **Control flow**: `IF_ELSE`, `FOR_LOOP`, `WHILE_LOOP`, `MATCH_EXPR`, `BREAK_EXPR`, `CONTINUE_EXPR`
- **Expressions**: `CALL_EXPR`, `METHOD_CALL`, `FIELD_ACCESS`, `BINARY_OP`, `UNARY_OP`
- **Literals**: `INT_LIT`, `FLOAT_LIT`, `STR_LIT`, `CHAR_LIT`, `BOOL_LIT`
- **Delimiters**: `BRACE_OPEN/CLOSE`, `PAREN_OPEN/CLOSE`, `BRACKET_OPEN/CLOSE`
- **Attributes**: `MACRO_INVOCATION`, `ATTRIBUTE`, `LIFETIME`, `GENERIC_PARAMS`

See the [TokenKind enum](https://docs.rs/pesti-structural-tokenizer/latest/pesti_structural_tokenizer/enum.TokenKind.html) for the complete list.

## Use Cases

### LLM Code Understanding (PESTI Runner)

Primary use case: feed structurally tokenized code to LLMs via PESTI's inference engine, giving models explicit syntactic context without requiring them to parse Rust grammar themselves.

### Code Analysis Tools

Build static analysis tools that understand Rust structure:

```rust
// Find all function declarations
let fn_decls = tokens.iter()
    .filter(|t| matches!(t.kind, TokenKind::FnDecl))
    .collect::<Vec<_>>();
```

### Code Generation Pipelines

Generate Rust code with guaranteed syntactic correctness by emitting structural tokens first, then filling in content.

### Educational Tools

Visualize Rust code structure for teaching/learning purposes.

## Relationship to pesti-runner

This crate is a dependency of [pesti-runner](https://github.com/CromboJambo/pesti), PESTI's inference engine. However, it's designed to be useful independently for any tool that needs to analyze or generate Rust code with structural awareness.

## Architecture

- **Parsing**: `syn` crate parses source into complete AST (replaces hand-written recursive descent parser from v0.1.0)
- **Walking**: Manual AST walker traverses items, statements, and expressions
- **Emission**: Structural tokens emitted based on node kind; function bodies elided except for declaration signatures

## License

AGPL-3.0-or-later