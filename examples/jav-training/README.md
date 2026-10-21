# JEV Training Data Selection

Finds the minimal sufficient subset of training examples using constraint propagation via PESTI embeddings.

## Concept

JEV identifies which training examples add novel behavioral constraints vs redundant ones. Instead of training on all N examples, you train on a smaller subset that captures all unique constraints.

This is like finding the smallest subset of invoices that adds up to the reconciliation total — you don't need every invoice, just enough to cover all unique obligations.

## Usage

```bash
# Build the selection tool
cargo build -p pesti-jav-training --release

# Run JEV selection on training data
./target/release/pesti-jav-training \
    examples/jav-training/synthetic_examples.jsonl \
    --embedding-model /path/to/embedding-model.gguf \
    --threshold 0.5 \
    --output selected_subset.jsonl

# Evaluate results
echo "Selected subset saved to selected_subset.jsonl"
```

## Parameters

- `input`: Path to training examples file (JSON lines format)
- `--embedding-model (-m)`: Path to embedding model GGUF file
- `--threshold (-t)`: Constraint novelty threshold (higher = stricter, default 0.5)
- `--max-examples (-n)`: Maximum number of examples to consider
- `--output (-o)`: Output file for selected subset

## Training Example Format

JSON lines with fields:
- `id`: Unique identifier
- `text`: The training example text
- `labels`: Array of labels/tags

Example:
```json
{"id": "1", "text": "fn add(a: i32, b: i32) -> i32 { return a + b; }", "labels": ["arithmetic"]}
```

## Metrics

The tool reports:
- Compression ratio (% of examples selected)
- Examples selected vs deferred
- Reasons for deferral (redundant constraint, below threshold)

## Experimentation

To test JEV training data selection quality:

1. Train on full dataset → baseline accuracy
2. Run JEV selection → get minimal subset
3. Train on JEV-selected subset → measure compression ratio vs accuracy loss
4. Analyze deferred examples to understand what JEV identified as redundant

Higher thresholds = stricter selection = more compression but potentially lower coverage.