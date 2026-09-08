#!/usr/bin/env python3
"""Test different prompts by encoding with PESTI's tokenizer, then comparing."""
import subprocess
import sys

def pesti_tokenize(model_path, text):
    """Use PESTI's tokenize example to encode text."""
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "pesti-runner", 
         "--example", "tokenize", model_path, text],
        capture_output=True, text=True
    )
    # Parse output: last line should be the token IDs
    lines = result.stdout.strip().split('\n')
    for line in reversed(lines):
        if '[' in line and ']' in line:
            ids_str = line.split('[')[1].split(']')[0]
            return [int(x) for x in ids_str.split(',') if x.strip()]
    raise ValueError(f"Could not parse token IDs from output: {result.stdout}")

def main():
    model = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    
    prompts = [
        ("fox", "The quick brown fox jumps over the lazy dog."),
        ("france", "The capital of France is"),
        ("math", "What is 2+2?"),
        ("poem", "Write a short poem about"),
        ("code", "Write a Python function that"),
    ]
    
    for name, prompt in prompts:
        print(f"\n=== {name}: {prompt} ===")
        
        # Tokenize with PESTI
        ids = pesti_tokenize(model, prompt)
        ids_str = ",".join(str(i) for i in ids)
        print(f"  tokens: {ids}")
        
        # Run numpy oracle
        result_ref = subprocess.run(
            ["python3", "ref_autoregressive.py", model, ids_str, "--steps=5"],
            capture_output=True, text=True
        )
        for line in result_ref.stdout.split('\n'):
            if "Generated tokens:" in line:
                print(f"  numpy: {line}")
        
        # Run Rust
        result_rust = subprocess.run(
            ["./target/release/examples/cpu_e2e_generate", model, prompt, "5"],
            capture_output=True, text=True
        )
        for line in result_rust.stdout.split('\n'):
            if line.startswith("ids:"):
                print(f"  rust:  {line}")

if __name__ == "__main__":
    main()
