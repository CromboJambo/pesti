#!/usr/bin/env python3
"""Comprehensive autoregressive KV cache test with per-step comparison.

Runs numpy oracle vs Rust CPU implementation, compares token-by-token choices,
and reports first divergence point if any. Validates that KV cache updates
produce identical results across both implementations.
"""
import subprocess
import sys
import json


MODEL = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPT = "The quick brown fox jumps over the lazy dog."
STEPS = 10


def run_numpy_oracle(steps, prompt_text):
    result = subprocess.run(
        [sys.executable, "conformance-corpus/ref_autoregressive.py", MODEL, "--steps", str(steps), f"--prompt={prompt_text}"],
        capture_output=True, text=True, timeout=120
    )
    # Find JSON object in output (it's the last large block)
    lines = result.stdout.strip().split("\n")
    json_start = None
    for i, line in enumerate(lines):
        if line.strip() == "{" or line.strip().startswith("{"):
            json_start = i
            break
    if json_start is None:
        raise RuntimeError(f"Could not find JSON in numpy oracle output:\n{result.stdout[-500:]}")

    # Join from the start of JSON to end
    json_text = "\n".join(lines[json_start:])
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Could not parse numpy oracle JSON: {e}\nJSON text:\n{json_text[-500:]}")


def run_rust_generation(steps, prompt_text):
    result = subprocess.run(
        ["./target/release/examples/cpu_e2e_generate", MODEL, prompt_text, str(steps)],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip()


print(f"=== Autoregressive KV Cache Test ({STEPS} steps) ===")
print()

# Run numpy reference
print("Running numpy reference...")
numpy_result = run_numpy_oracle(STEPS, PROMPT)
numpy_tokens = numpy_result["generated"]
print(f"  Generated tokens ({len(numpy_tokens)}): {numpy_tokens}")

# Run Rust CPU implementation
print("\nRunning Rust CPU implementation...")
rust_output = run_rust_generation(STEPS, PROMPT)
for line in rust_output.split("\n"):
    if "text:" in line or "tokens:" in line or "ids:" in line:
        print(f"  {line.strip()}")

# Extract Rust token IDs from ids: line
rust_tokens = []
for line in rust_output.split("\n"):
    if line.startswith("ids:"):
        # Parse [220, 3555, ...] format
        ids_part = line.split(":", 1)[1].strip()
        ids_str = ids_part.strip("[]")
        rust_tokens = [int(t) for t in ids_str.split(",")]
        break

# Extract Rust text for comparison
rust_text = ""
for line in rust_output.split("\n"):
    if line.startswith("text:"):
        rust_text = line.split(":", 1)[1].strip()
        break

print("\n=== Comparison ===")
print(f"Numpy tokens: {numpy_tokens}")
print(f"Rust tokens:  {rust_tokens}")

# Per-step comparison
if len(numpy_tokens) == len(rust_tokens):
    matches = sum(1 for n, r in zip(numpy_tokens, rust_tokens) if n == r)
    print(f"\nToken match rate: {matches}/{len(numpy_tokens)} ({100*matches/len(numpy_tokens):.0f}%)")

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
    print(f"\nToken count mismatch: numpy={len(numpy_tokens)}, rust={len(rust_tokens)}")

print(f"\nRust generated text: '{rust_text}'")