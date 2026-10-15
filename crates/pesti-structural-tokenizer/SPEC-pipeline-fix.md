# Spec: Fix pesti-pipeline-tool Availability

**Crate:** `pesti-structural-tokenizer`  
**Priority:** Medium  
**Status:** Proposed  

## Problem Statement

The structural tokenizer pipeline tool (`hermes-pipeline`) exists in the codebase but is not functional due to a mismatch between how the pipeline invokes `struct-tok` and how `struct-tok` actually parses arguments. This prevents using pesti's structural analysis tools for codebase comparison workflows.

## Current State

The `pesti-structural-tokenizer` crate contains:
- Library (`lib.rs`): Functional `StructuralTokenizer` with budget-aware tokenization, semantic annotation, and elision support
- Binary `struct-tok`: CLI tool with commands: `tokenize FILE`, `detokenize FILE`, `stats FILE`, `encode TEXT`, `decode TOKENS`
- Binary `hermes-pipeline`: Watch/process directory pipeline that wraps struct-tok

## Root Cause

**Argument parsing mismatch:** The pipeline invokes `struct-tok encode --file <path>` but the `encode` command expects text directly as its argument, not a file path. There is no `--file` flag support in the encode command.

From `hermes-pipeline.rs`:
```rust
let output = std::process::Command::new("struct-tok")
    .args(["encode", "--file"])
    .arg(file)
    .output();
```

From `struct-tok.rs` encode handler:
```rust
"encode" => {
    let src = &args[2];  // Expects TEXT, not --file PATH
    ...
}
```

## Proposed Fix

### Option A: Add --file flag support to struct-tok (Recommended)

Modify the `encode` command in `struct-tok.rs` to accept an optional `--file` flag:

```rust
"encode" => {
    // Check for --file flag
    let file_idx = args.iter().position(|a| a == "--file");
    if let Some(idx) = file_idx {
        if idx + 1 >= args.len() {
            eprintln!("--file requires a path");
            process::exit(1);
        }
        let src = fs::read_to_string(&args[idx + 1]).expect("Failed to read file");
        // ... tokenize src
    } else if args.len() >= 3 {
        let src = &args[2];
        // ... tokenize src (existing behavior)
    }
}
```

**Pros:** Maintains backward compatibility, minimal change, matches pipeline's expectation  
**Cons:** Adds slight complexity to argument parsing

### Option B: Fix hermes-pipeline to use tokenize command

Change `hermes-pipeline.rs` to use the `tokenize` command (which already supports files) instead of `encode`:

```rust
let output = std::process::Command::new("struct-tok")
    .args(["tokenize"])
    .arg(file)
    .output();
```

**Pros:** Uses existing tested code path  
**Cons:** Output format differs (tokenize prints detailed token info, encode prints compact IDs) — may break downstream consumers expecting the encode format

### Option C: Both A and B

Implement both fixes for maximum flexibility and robustness.

## Additional Issues to Address

1. **Binary discoverability:** The binaries are built as targets within `pesti-structural-tokenizer` but not exposed as separate packages. Users must build with explicit `--bin` flags. Consider either:
   - Documenting the build/install commands clearly
   - Creating a separate crate for CLI tools that depends on the library

2. **Error handling in pipeline:** The pipeline silently fails when struct-tok errors. Add stderr capture and reporting.

3. **Missing --file support in other commands:** `detokenize` and `decode` also lack file support. Consider adding consistently across all commands.

## Acceptance Criteria

- [ ] `hermes-pipeline process <file>` successfully tokenizes a Rust source file
- [ ] `struct-tok encode --file <path>` works (if Option A chosen)
- [ ] Existing `struct-tok` commands continue to work (backward compatibility)
- [ ] Pipeline errors are reported clearly, not silently swallowed
- [ ] Tests added for the fixed argument parsing

## Files to Modify

- `crates/pesti-structural-tokenizer/src/bin/struct-tok.rs` — add --file support to encode command
- `crates/pesti-structural-tokenizer/src/bin/hermes-pipeline.rs` — improve error handling
- `crates/pesti-structural-tokenizer/tests/` — add integration tests for CLI tools

## Doubt Block

```json
{
  "doubt": {
    "assumptions": [
      "The encode command's output format is what downstream consumers expect",
      "No other callers depend on the current broken --file behavior"
    ],
    "blind_spots": [
      "May be additional argument parsing issues in other commands not yet discovered",
      "The semantic annotation pipeline may have separate issues not covered here"
    ],
    "last_validation": "2026-10-14",
    "stale_after": "2026-10-21"
  }
}
```
