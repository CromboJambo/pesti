# Syn-Based Structural Tokenizer Migration

**Status:** Planning / Design  
**Branch:** `feat/syn-structural-tokenizer`  
**Created:** 2026-09-24

## Context

The current `pesti-structural-tokenizer` is a hand-written recursive descent parser with ~30 `parse_*` methods. Each method independently implements its own brace/paren balancing logic, leading to:

1. **Fragility:** No error recovery — one malformed construct halts the entire parse
2. **Inconsistency:** Different parsing strategies across methods (some use `find_matching_brace`, others manual counting)
3. **Coverage gaps:** Missing async/await blocks, const generics, turbofish (`::<T>`), lifetime bounds, macro_rules!, and more
4. **Redundancy:** ~1500 lines of hand-written parsing vs what could be ~200 lines with syn

## Goal

Replace the hand-written parser with `syn`'s battle-tested Rust parser, then build pesti's unique value proposition on top: **LLM-context-budget compression** — deciding what an agent needs to see given a token budget.

The crate is NOT about parsing Rust (syn does that). It's about producing a compressed structural representation tuned for LLM context windows.

## Architecture

```
Source code → syn::parse_file() → syn AST → pesti Structural Tokenizer → StructuralTokens
```

### Layer 1: Parsing (syn)
- `syn::parse_file()` produces a complete, correct Rust AST
- Handles all edge cases, error recovery, and syntax variants
- Zero custom parsing logic needed

### Layer 2: Structural Token Emission (pesti's value-add)
- Walk the syn AST with a visitor pattern
- Emit pesti `StructuralToken` stream based on budget constraints
- **This is where pesti adds value over raw syn output:**
  - Collapse function bodies but keep signatures
  - Prioritize state-machine-relevant nodes (fn decls, impl blocks, trait bounds)
  - Elide anything under N bytes of source
  - Produce 57KB→2KB structural digests for agent context windows

## Implementation Plan

### Phase 1: Add syn dependency and basic parsing
```toml
# Cargo.toml
syn = { version = "2", features = ["full", "visit"] }
```

Replace `parse_file()` call with syn's, verify it parses all existing test cases.

### Phase 2: AST visitor for structural tokens
Implement a `Visitor` that walks the syn AST and emits pesti tokens:

```rust
use syn::visit::{self, Visit};

struct StructuralVisitor<'a> {
    tokenizer: &'a mut StructuralTokenizer,
}

impl<'ast> Visit<'ast> for StructuralVisitor<'_> {
    fn visit_item_fn(&mut self, f: &'ast syn::ItemFn) {
        // Emit FnDecl token with signature info
        // Walk body based on budget
    }
    
    fn visit_item_struct(&mut self, s: &'ast syn::ItemStruct) {
        // Emit StructDecl
    }
    
    // ... other item types
}
```

### Phase 3: Budget-aware traversal
Add the LLM-context-budget layer that makes pesti unique:

```rust
pub struct TokenizeOptions {
    pub budget_bytes: usize,      // Max source bytes to represent
    pub collapse_bodies: bool,    // Replace fn bodies with [...]
    pub prioritize_signatures: bool,
}
```

### Phase 4: Remove legacy parsing code
Delete all `parse_*` methods, helper functions for brace matching, etc. Keep only the token emission logic and tests.

## API Changes

**Before:**
```rust
let tokenizer = StructuralTokenizer::new();
let tokens = tokenizer.tokenize(src)?;
```

**After (same interface):**
```rust
let tokenizer = StructuralTokenizer::new();
let tokens = tokenizer.tokenize(src)?;  // Now uses syn internally
```

The public API stays identical — this is an implementation detail change.

## Testing Strategy

1. **Regression:** All existing tests must pass (fn decl, let stmt, struct/enum/trait/impl, control flow)
2. **New coverage:** Add tests for previously-missing constructs:
   - async fn / await
   - const generics
   - turbofish operator
   - lifetime bounds in generics
   - macro_rules! definitions
3. **Edge cases:** Malformed input should produce syn parse errors, not panic

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| syn 2.x API changes | Pin exact version; use established patterns from ecosystem |
| Performance regression | Benchmark tokenize() before/after; syn is well-optimized |
| Feature creep (trying to parse everything) | Focus on structural tokens only, not full AST traversal |

## Success Criteria

1. All existing tests pass
2. New tests for async/await, const generics, turbofish pass
3. Code size reduction: ~1500 lines → ~300-400 lines (75%+ reduction)
4. No regression in tokenization accuracy on test corpus

## References

- syn crate docs: https://docs.rs/syn/latest/syn/
- syn visitor pattern: https://docs.rs/syn/latest/syn/visit/index.html
- Original pesti tokenizer: `crates/pesti-structural-tokenizer/src/lib.rs` (current)