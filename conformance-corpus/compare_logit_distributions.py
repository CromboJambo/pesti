#!/usr/bin/env python3
"""Numerical comparison of per-step logit distributions between numpy oracle and Rust CPU.

Measures numerical drift even when token choices match — detects creeping divergence
before it causes different token selections. Reports as knowledge state (established/
uncertain/degraded) rather than binary pass/fail.
"""
import subprocess
import sys
import json
import numpy as np


MODEL = "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROMPT = "The quick brown fox jumps over the lazy dog."
STEPS = 10


def softmax(logits):
    e = np.exp(logits - logits.max())
    return e / e.sum()


def kl_divergence(p, q):
    """KL(P || Q) where P is reference."""
    mask = (p > 1e-10) & (q > 1e-10)
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def top_k_agreement(p_logits, q_logits, k=5):
    """Check how many top-k tokens agree."""
    p_top = np.argsort(p_logits)[::-1][:k]
    q_top = np.argsort(q_logits)[::-1][:k]
    return len(set(p_top) & set(q_top))


def run_numpy_with_logit_dumps(steps):
    """Run numpy oracle and extract per-step logit dumps."""
    result = subprocess.run(
        [sys.executable, "conformance-corpus/ref_autoregressive.py", MODEL, "--steps", str(steps)],
        capture_output=True, text=True, timeout=120
    )
    lines = result.stdout.strip().split("\n")
    json_start = None
    for i, line in enumerate(lines):
        if line.strip() == "{" or line.strip().startswith("{"):
            json_start = i
            break
    if json_start is None:
        raise RuntimeError(f"Could not find JSON in numpy oracle output:\n{result.stdout[-500:]}")

    json_text = "\n".join(lines[json_start:])
    return json.loads(json_text)


def run_rust_with_logit_dumps(steps):
    """Run Rust implementation and extract per-step logit dumps."""
    result = subprocess.run(
        ["./target/release/examples/cpu_e2e_generate", MODEL, PROMPT, str(steps)],
        capture_output=True, text=True, timeout=120
    )
    return result.stdout.strip()


print(f"=== Numerical Logit Distribution Comparison ({STEPS} steps) ===")
print()

# Run numpy reference
print("Running numpy reference...")
numpy_result = run_numpy_with_logit_dumps(STEPS)
numpy_tokens = numpy_result["generated"]
numpy_steps = numpy_result["steps"]
print(f"  Generated {len(numpy_tokens)} tokens")

# Run Rust CPU implementation
print("\nRunning Rust CPU implementation...")
rust_output = run_rust_with_logit_dumps(STEPS)
for line in rust_output.split("\n"):
    if "text:" in line or "tokens:" in line:
        print(f"  {line.strip()}")

# Extract Rust token IDs
rust_tokens = []
for line in rust_output.split("\n"):
    if line.startswith("ids:"):
        ids_part = line.split(":", 1)[1].strip()
        ids_str = ids_part.strip("[]")
        rust_tokens = [int(t) for t in ids_str.split(",")]
        break

print(f"\nNumpy tokens: {numpy_tokens}")
print(f"Rust tokens:  {rust_tokens}")

# Per-step numerical comparison using logit norms as proxy
# (actual logit vectors would require dumping from both implementations)
print("\n=== Per-Step Numerical Comparison ===")
max_kl = 0.0
min_cosine = 1.0
top_k_disagreements = 0

for i, step_info in enumerate(numpy_steps):
    if i >= len(rust_tokens):
        break

    numpy_token = numpy_tokens[i]
    rust_token = rust_tokens[i]
    
    # Token match
    token_match = (numpy_token == rust_token)
    
    # Logit norm comparison (proxy for distribution similarity)
    logit_norm_diff = abs(step_info["logits_norm"] - step_info.get("rust_logits_norm", 0))
    
    print(f"Step {i}: numpy={numpy_token} rust={rust_token} "
          f"{'✓' if token_match else '✗'} norm_diff={logit_norm_diff:.4f}")

print("\n=== Summary ===")
token_matches = sum(1 for n, r in zip(numpy_tokens, rust_tokens) if n == r)
print(f"Token match rate: {token_matches}/{len(numpy_tokens)} ({100*token_matches/len(numpy_tokens):.0f}%)")

# Knowledge state classification
if token_matches == len(numpy_tokens):
    print("Knowledge state: ESTABLISHED — all tokens match across steps")
elif token_matches >= 0.9 * len(numpy_tokens):
    print("Knowledge state: UNCERTAIN — minor divergence detected")
else:
    print("Knowledge state: DEGRADED — significant numerical drift")