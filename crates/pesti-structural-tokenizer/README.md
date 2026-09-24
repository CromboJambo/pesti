# pesti-structural-tokenizer

Rust structural tokenizer for LLM code understanding — encodes syntactic boundaries as token boundaries.

## What This Does

Traditional byte-pair encoding (BPE) tokenizers treat source code as opaque text. This crate takes a different approach: it identifies **syntactic and semantic boundaries** in Rust source code and inserts explicit structural tokens at those boundaries, while using BPE only for the content within each region.

The result is that LLMs get explicit signals about code structure without having to infer it from whitespace and punctuation alone. This improves code understanding, generation accuracy, and reduces hallucination of syntactically invalid code.

## Why Structural Tokenization?

LLMs trained on raw BPE tokens must learn Rust's syntax implicitly. Structural tokenization makes this knowledge **explicit**:

- `FUNC_DEF` — unambiguous function definition boundary
- `BRACE_OPEN` / `BRACE_CLOSE` — explicit block delimiters
- `IF_ELSE` — control flow structure is first-class
- `STRUCT_FIELD` — data layout boundaries are clear
- `MATCH_ARM` — pattern matching arms are distinct units

This matters for code generation, refactoring tools, and any agentic workflow that needs to understand Rust source.

## Installation

```toml
[dependencies]
pesti-structural-tokenizer = "0.1"
```

## Quick Start

```rust
use pesti_structural_tokenizer::{tokenize, TokenKind};

let code = r#"
fn add(a: i32, b: i32) -> i32 {
    a + b
}
"#;

let tokens = tokenize(code)?;

// Count structural vs content tokens
let structural = tokens.iter()
    .filter(|t| matches!(t.kind, TokenKind::FUNC_DEF | TokenKind::BRACE_OPEN | ...))
    .count();

println!("Total tokens: {}", tokens.len());
println!("Structural tokens: {}", structural);
```

## API Overview

### Core Types

- `StructuralToken` — a single token with kind, text, and source span
- `Spanned<T>` — value with source location (start/end line/column)
- `SourceText` — wrapper for source code being tokenized

### Main Functions

```rust
// Tokenize Rust source into structural tokens
fn tokenize(source: &str) -> Result<StructuralTokens, TokenizeError>

// Serialize to JSON for LLM consumption
fn to_json(tokens: &[StructuralToken]) -> String
fn to_compact_json(tokens: &[StructuralToken]) -> String

// Debug representation
fn to_string(tokens: &[StructuralToken]) -> String
```

### Token Kinds

The tokenizer produces ~100 distinct token kinds covering:

- **Definitions**: `FUNC_DEF`, `STRUCT_DEF`, `ENUM_DEF`, `TRAIT_DEF`, `TYPEDEF`
- **Control flow**: `IF_ELSE`, `FOR_LOOP`, `WHILE_LOOP`, `MATCH_ARM`, `BREAK`, `CONTINUE`
- **Expressions**: `METHOD_CALL`, `FIELD_ACCESS`, `INDEX_EXPR`, `CLOSURE_EXPR`
- **Literals**: `INT_LITERAL`, `STRING_LITERAL`, `BOOL_LITERAL`, etc.
- **Delimiters**: `BRACE_OPEN/CLOSE`, `PAREN_OPEN/CLOSE`, `BRACKET_OPEN/CLOSE`
- **Attributes**: `TEST_ATTR`, `DERIVE_ATTR`, `DOC_COMMENT`, etc.

See the [TokenKind enum](https://docs.rs/pesti-structural-tokenizer/latest/pesti_structural_tokenizer/enum.TokenKind.html) for the complete list.

## Use Cases

### LLM Code Understanding (PESTI Runner)

Primary use case: feed structurally tokenized code to LLMs via PESTI's inference engine, giving models explicit syntactic context without requiring them to parse Rust grammar themselves.

### Code Analysis Tools

Build static analysis tools that understand Rust structure:

```rust
// Find all public functions
let public_fns = tokens.iter()
    .filter(|t| matches!(t.kind, TokenKind::FUNC_DEF))
    .collect::<Vec<_>>();
```

### Code Generation Pipelines

Generate Rust code with guaranteed syntactic correctness by emitting structural tokens first, then filling in content.

### Educational Tools

Visualize Rust code structure for teaching/learning purposes.

## Relationship to pesti-runner

This crate is a dependency of [pesti-runner](https://github.com/CromboJambo/pesti), PESTI's inference engine. However, it's designed to be useful independently for any tool that needs to analyze or generate Rust code with structural awareness.

## License

AGPL-3.0-or-later
