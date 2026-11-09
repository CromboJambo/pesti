# Structure-Aware Code Embeddings

This prototype demonstrates combining pesti-structural-tokenizer with pesti-embeddings to create embeddings that capture both semantic meaning and syntactic structure.

## How It Works

1. **Tokenize code** with pesti-structural-tokenizer to extract structural tokens (FnDecl, ForLoop, IfElse, etc.)
2. **Embed separately**: Embed the original source code AND the structural representation independently
3. **Combine in vector space**: Weighted average of text and structure embeddings creates a joint representation

## Results

### Similarity Comparison
- **Baseline (text only)**: add vs multiply = 0.9710, add vs loop = 0.9365
- **Structure-aware (50/50)**: add vs multiply = 0.9920 (+0.0210), add vs loop = 0.9400 (+0.0035)

The structural representation successfully captures syntactic similarity — structurally similar code becomes much more similar to each other, while structurally different code shows less improvement.

### Code Search
Query: Find functions with structure `FnDecl ReturnExpr Semicolon` (simple binary operation)

Results ranked by similarity:
1. multiply - 0.9984 (identical structure)
2. simple_add - 0.9954 (identical structure)
3. conditional - 0.9820 (has if/else)
4. nested_loop - 0.9737 (has nested loops)
5. loop_process - 0.9381 (just a for loop)

## Applications

- **Code search**: Find structurally similar code patterns across a codebase
- **Duplicate detection**: Identify functionally equivalent but syntactically different code
- **Refactoring suggestions**: Find code that could be consolidated or refactored
- **Testing**: Generate test cases based on structural similarity

## Running the Examples

```bash
# Structure-aware embeddings comparison
cargo run --example structure_aware_embeddings -p pesti-embeddings

# Structure-aware code search
cargo run --example structure_aware_search -p pesti-embeddings
```

## Next Steps

1. Integrate into pesti-embeddings CLI with `--structure-aware` flag
2. Add weight tuning parameters (text vs structure balance)
3. Explore applications in pesti-analyze for code similarity analysis
4. Consider GGUF model configuration analysis use cases