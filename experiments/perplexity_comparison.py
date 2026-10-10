#!/usr/bin/env python3
"""
Perplexity comparison: BPE vs Structural+Semantic tokenization for Rust code.

Uses a simple n-gram language model to measure how predictable each
tokenization scheme is. Lower perplexity = more efficient representation.
"""

import math
from collections import defaultdict
from pathlib import Path


def collect_rust_files(root, max_files=50):
    """Collect .rs files from the project."""
    files = []
    for f in Path(root).rglob("*.rs"):
        if ".git" not in str(f) and "target/" not in str(f):
            files.append(f)
        if len(files) >= max_files:
            break
    return files


def bpe_tokenize(text):
    """Simple word-level BPE approximation."""
    # Split on non-alphanumeric
    import re
    tokens = re.findall(r'\w+|[^\w\s]', text)
    return tokens


def structural_tokenize_approx(text):
    """Approximate structural tokenization using regex patterns.

    This is a proxy for the real pesti tokenizer - maps surface forms
    to structural categories.
    """
    import re
    
    # Map of pattern -> structural type ID
    patterns = [
        (r'fn\s+\w+', 'FN_DEF'),
        (r'struct\s+\w+', 'STRUCT_DEF'),
        (r'enum\s+\w+', 'ENUM_DEF'),
        (r'let\s+\w+', 'LET_BIND'),
        (r'\bif\b', 'IF'),
        (r'\belse\b', 'ELSE'),
        (r'\bwhile\b', 'WHILE'),
        (r'\bfor\b', 'FOR'),
        (r'\breturn\b', 'RETURN'),
        (r'\bmatch\b', 'MATCH'),
        (r'\bimpl\b', 'IMPL'),
        (r'\btrait\b', 'TRAIT'),
        (r'\bpub\b', 'PUB'),
        (r'\bfn\b', 'FN_KW'),
    ]
    
    tokens = []
    i = 0
    while i < len(text):
        matched = False
        for pattern, token_type in patterns:
            m = re.match(pattern, text[i:])
            if m:
                tokens.append(token_type)
                i += m.end()
                matched = True
                break
        
        if not matched:
            # Skip non-structural content (literals, identifiers, punctuation)
            c = text[i]
            if c.isalnum() or c == '_':
                # Identifier - classify by context
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                ident = text[i:j]
                
                # Check if it's a type annotation
                if j < len(text) and text[j] == ':':
                    tokens.append('TYPE_ANNOT')
                elif j < len(text) and text[j] == '(':
                    tokens.append('FN_CALL')
                else:
                    tokens.append('IDENT')
                
                i = j
            else:
                # Punctuation/operator - classify broadly
                if c in '+-*/%':
                    tokens.append('OP_ARITH')
                elif c in '<>=':
                    tokens.append('OP_CMP')
                elif c in '{}[]()':
                    tokens.append('DELIM')
                else:
                    tokens.append('PUNCT')
                i += 1
    
    return tokens


def build_ngram_model(sequences, n=2):
    """Build an n-gram language model from token sequences."""
    counts = defaultdict(lambda: defaultdict(int))
    
    for seq in sequences:
        # Add start/end markers
        seq = ['<S>'] + seq + ['</S>']
        
        for i in range(len(seq) - n + 1):
            context = tuple(seq[i:i+n-1])
            next_token = seq[i+n-1]
            counts[context][next_token] += 1
    
    return counts


def perplexity_ngram(model, sequences, n=2):
    """Calculate perplexity of sequences under the n-gram model."""
    total_log_prob = 0.0
    total_tokens = 0
    
    for seq in sequences:
        seq = ['<S>'] + seq + ['</S>']
        
        for i in range(len(seq) - n + 1):
            context = tuple(seq[i:i+n-1])
            next_token = seq[i+n-1]
            
            if context in model:
                total_count = sum(model[context].values())
                count = model[context].get(next_token, 0)
                
                # Laplace smoothing
                vocab_size = len(set(t for ctx in model.values() for t in ctx.keys()))
                prob = (count + 1) / (total_count + vocab_size)
            else:
                # OOV context - uniform distribution
                vocab_size = len(set(t for ctx in model.values() for t in ctx.keys()))
                prob = 1.0 / (vocab_size + 1)
            
            total_log_prob += math.log(prob)
            total_tokens += 1
    
    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.exp(-avg_log_prob)
    
    return perplexity


def main():
    print("=== Perplexity Comparison: BPE vs Structural Tokenization ===")
    print()
    
    # Collect Rust source files
    rust_files = collect_rust_files("/home/crombo/projects/active/pesti", max_files=50)
    print(f"Found {len(rust_files)} Rust files")
    
    # Read and tokenize with both schemes
    bpe_sequences = []
    structural_sequences = []
    
    for filepath in rust_files:
        try:
            content = Path(filepath).read_text()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
        
        # BPE tokenization (word-level approximation)
        bpe_tokens = bpe_tokenize(content)
        if len(bpe_tokens) > 10:  # Skip very short files
            bpe_sequences.append(bpe_tokens)
        
        # Structural tokenization (approximation)
        structural_tokens = structural_tokenize_approx(content)
        if len(structural_tokens) > 10:
            structural_sequences.append(structural_tokens)
    
    print(f"BPE sequences: {len(bpe_sequences)}")
    print(f"Structural sequences: {len(structural_sequences)}")
    
    # Split into train/val (80/20)
    split_bpe = int(0.8 * len(bpe_sequences))
    bpe_train = bpe_sequences[:split_bpe]
    bpe_val = bpe_sequences[split_bpe:]
    
    split_struct = int(0.8 * len(structural_sequences))
    struct_train = structural_sequences[:split_struct]
    struct_val = structural_sequences[split_struct:]
    
    print(f"Train/val split: BPE ({len(bpe_train)}/{len(bpe_val)}), "
          f"Structural ({len(struct_train)}/{len(struct_val)})")
    
    # Build and evaluate n-gram models
    print("\nBuilding BPE bigram model...")
    bpe_model = build_ngram_model(bpe_train, n=2)
    bpe_ppl = perplexity_ngram(bpe_model, bpe_val, n=2)
    print(f"BPE validation perplexity: {bpe_ppl:.4f}")
    
    print("\nBuilding structural bigram model...")
    struct_model = build_ngram_model(struct_train, n=2)
    struct_ppl = perplexity_ngram(struct_model, struct_val, n=2)
    print(f"Structural validation perplexity: {struct_ppl:.4f}")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"BPE perplexity:           {bpe_ppl:.4f}")
    print(f"Structural perplexity:    {struct_ppl:.4f}")
    
    if struct_ppl < bpe_ppl:
        improvement = (1 - struct_ppl / bpe_ppl) * 100
        print(f"Improvement:             {improvement:.1f}% lower perplexity")
        print("Structural tokenization is MORE efficient!")
    else:
        degradation = (struct_ppl / bpe_ppl - 1) * 100
        print(f"Degradation:             {degradation:.1f}% higher perplexity")
        print("BPE tokenization is more efficient.")


if __name__ == "__main__":
    main()
