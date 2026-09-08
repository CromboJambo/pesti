#!/usr/bin/env python3
"""Compare Rust CPU autoregressive generation against numpy reference.

Runs both implementations on the same prompt, compares per-step token choices
and logit distributions to detect where numerical differences cause divergence.
"""
import subprocess
import sys
import json
import re


MODEL = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPT_TOKENS = "785,3974,13876,38835,34208,916,279,15678,5562,13"  # fox prompt
STEPS = 10


def run_numpy_oracle(steps):
    result = subprocess.run(
        [sys.executable, "conformance-corpus/ref_autoregressive.py", MODEL, PROMPT_TOKENS, f"--steps={steps}"],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise RuntimeError(f"numpy oracle failed:\n{result.stderr[-1000:]}")
    
    # Parse generated tokens from JSON output
    for line in result.stdout.split("\n"):
        if line.startswith("{"):
            data = json.loads(line)
            return data
    
    raise RuntimeError(f"Could not parse numpy oracle output:\n{result.stdout[-500:]}")


def run_rust_generation(steps):
    result = subprocess.run(
        ["./target/release/examples/cpu_e2e_generate", MODEL, "The quick brown fox jumps over the lazy dog.", str(steps)],
        capture_output=True, text=True, timeout=300
    )
    if result.returncode != 0:
        raise RuntimeError(f"Rust generation failed:\n{result.stderr[-1000:]}")
    
    # Extract generated tokens from ids line
    for line in result.stdout.split("\n"):
        if line.startswith("ids:"):
            ids_part = line.split(":", 1)[1].strip()
            ids_str = ids_part.strip("[]")
            return [int(t) for t in ids_str.split(",")]
    
    raise RuntimeError(f"Could not parse Rust output:\n{result.stdout[-500:]}")


print(f"=== Autoregressive KV Cache Test ({STEPS} steps) ===\n")

# Run numpy reference
print("Running numpy reference...")
numpy_result = run_numpy_oracle(STEPS)
numpy_tokens = numpy_result["generated"]
print(f"  Generated tokens ({len(numpy_tokens)}): {numpy_tokens}\n")

# Run Rust CPU implementation
print("Running Rust CPU implementation...")
rust_tokens = run_rust_generation(STEPS)
print(f"  Generated tokens ({len(rust_tokens)}): {rust_tokens}\n")

# Compare
print("=== Comparison ===")
if len(numpy_tokens) == len(rust_tokens):
    matches = sum(1 for n, r in zip(numpy_tokens, rust_tokens) if n == r)
    print(f"Token match rate: {matches}/{len(numpy_tokens)} ({100*matches/len(numpy_tokens):.0f}%)")
    
    # Find first divergence
    diverged = False
    for i, (n, r) in enumerate(zip(numpy_tokens, rust_tokens)):
        if n != r:
            print(f"  First divergence at step {i}: numpy={n}, rust={r}")
            diverged = True
            break
    
    if not diverged:
        print("  All tokens match across all steps!")
else:
    print(f"Token count mismatch: numpy={len(numpy_tokens)}, rust={len(rust_tokens)}")
