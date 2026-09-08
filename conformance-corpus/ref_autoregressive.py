#!/usr/bin/env python3
"""Autoregressive decoding oracle with KV cache updates and per-step token tracking.

Takes pre-tokenized prompt IDs (same approach as ref_forward.py) to guarantee
identical tokenization between numpy oracle and Rust implementation.

Tensors loaded via gguf package dequantization, with proper KV cache updates
across generation steps. Greedy decoding (argmax) for deterministic comparison.

Usage: ref_autoregressive.py <model.gguf> tok1,tok2,... [--steps N]
"""
import sys
import json
import numpy as np
import gguf

PATH = sys.argv[1]
if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
    toks = [int(t) for t in sys.argv[2].split(",")]
else:
    toks = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]  # default fox prompt

steps = 10
for i in range(3, len(sys.argv)):
    if sys.argv[i].startswith("--steps="):
        steps = int(sys.argv[i].split("=", 1)[1])

print(f"[REF] Autoregressive oracle: {steps} steps")
print(f"[REF] Model: {PATH}")
print(f"[REF] Prompt tokens ({len(toks)}): {toks}")

# Load GGUF and extract architecture params (same pattern as ref_forward.py)
reader = gguf.GGUFReader(PATH)

def field(key):
    f = reader.get_field(key)
    if f is None:
        return None
    parts = f.parts
    types = f.types
    if not types:
        return None
    t = types[-1]
    last = parts[-1]
    if int(t) == 8:  # GGUFValueType.STRING
        return bytes(np.asarray(last, dtype=np.uint8)).decode("utf-8")
    arr = np.asarray(last)
    if arr.size == 1:
        return arr.reshape(()).item()
    return arr

def kv(key, default=None):
    v = field(key)
    return default if v is None else v

arch = field("general.architecture")
n_layer = int(kv(f"{arch}.block_count"))
n_head = int(kv(f"{arch}.attention.head_count"))
n_head_kv = int(kv(f"{arch}.attention.head_count_kv"))
n_embd = int(kv(f"{arch}.embedding_length"))
n_ffn = int(kv(f"{arch}.feed_forward_length"))
rope_base = float(kv(f"{arch}.rope.freq_base", 10000.0))
rms_eps = float(kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
head_dim = n_embd // n_head

print(f"[REF] Loaded {n_layer} layer model, dim={n_embd}, heads={n_head}, kv_heads={n_head_kv}")

# Load tensors via gguf dequantization (same pattern as ref_forward.py)
tensor_by_name = {t.name: t for t in reader.tensors}

def T(name):
    """Dequantize a tensor to a float32 numpy array in LOGICAL shape."""
    t = tensor_by_name[name]
    flat = gguf.dequantize(t.data, t.tensor_type)
    shape = [int(s) for s in t.shape]
    arr = flat.reshape(shape[::-1]).T
    return arr.astype(np.float32)

# Embedding and output weights
emb = T("token_embd.weight")
E = emb.T  # [vocab, n_embd]
out_w = T("output.weight")
OUT = out_w.T  # [vocab, n_embd]
output_norm = T("output_norm.weight")

print(f"[REF] Loaded weights: E={E.shape}, OUT={OUT.shape}")

def rmsnorm(x, w):
    ms = (x * x).mean()
    return (x / np.sqrt(ms + rms_eps)) * w

def rope(x, pos):
    d = head_dim
    inv_freq = 1.0 / (rope_base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    freqs = pos * inv_freq
    cos = np.cos(freqs)
    sin = np.sin(freqs)
    x0 = x[:, :d//2]
    x1 = x[:, d//2:]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return np.concatenate([out0, out1], axis=1)

def silu(x):
    return x / (1.0 + np.exp(-x))

# KV cache: list of [n_head_kv, head_dim] per layer
K_cache = {l: [] for l in range(n_layer)}
V_cache = {l: [] for l in range(n_layer)}

def transformer_forward(input_ids):
    """Run forward pass for a single token with KV cache updates. Returns logits."""
    tok = input_ids[0]
    x = E[tok].astype(np.float32)  # [n_embd]
    
    pos = len(K_cache[0])  # Current position from cache length
    
    for l in range(n_layer):
        prefix = f"blk.{l}."
        
        # Attention
        attn_norm = T(f"{prefix}attn_norm.weight")
        attn_input = rmsnorm(x, attn_norm)
        
        wq = T(f"{prefix}attn_q.weight")  # [n_embd, n_embd]
        wk = T(f"{prefix}attn_k.weight")  # [n_embd, n_head_kv*head_dim]
        wv = T(f"{prefix}attn_v.weight")  # [n_embd, n_head_kv*head_dim]
        wo = T(f"{prefix}attn_output.weight")  # [n_embd, n_embd]
        bq = T(f"{prefix}attn_q.bias")
        bk = T(f"{prefix}attn_k.bias")
        bv = T(f"{prefix}attn_v.bias")
        
        q = (wq.T @ attn_input) + bq  # [n_embd]
        k = (wk.T @ attn_input) + bk  # [n_head_kv*head_dim]
        v = (wv.T @ attn_input) + bv  # [n_head_kv*head_dim]
        
        q = q.reshape(n_head, head_dim)
        k = k.reshape(n_head_kv, head_dim)
        v = v.reshape(n_head_kv, head_dim)
        
        q = rope(q, pos)
        k = rope(k, pos)
        
        K_cache[l].append(k)
        V_cache[l].append(v)
        
        Ks = np.stack(K_cache[l], axis=1)  # [n_head_kv, seq, head_dim]
        Vs = np.stack(V_cache[l], axis=1)  # [n_head_kv, seq, head_dim]
        
        scale = 1.0 / np.sqrt(head_dim)
        attn_out = np.zeros((n_head, head_dim), dtype=np.float32)
        hpq = n_head // n_head_kv
        
        for hh in range(n_head):
            g = hh // hpq
            qh = q[hh]  # [head_dim]
            kg = Ks[g]  # [seq, head_dim]
            scores = (kg @ qh) * scale  # [seq]
            scores = scores - scores.max()
            exps = np.exp(scores)
            w = exps / exps.sum()
            vg = Vs[g]  # [seq, head_dim]
            attn_out[hh] = (w[:, None] * vg).sum(axis=0)
        
        attn_flat = attn_out.reshape(-1)  # [n_embd]
        attn_proj = wo.T @ attn_flat  # [n_embd]
        x = x + attn_proj
        
        # FFN
        ffn_norm = T(f"{prefix}ffn_norm.weight")
        ffn_input = rmsnorm(x, ffn_norm)
        
        w1 = T(f"{prefix}ffn_gate.weight")  # [n_embd, n_ffn]
        w2 = T(f"{prefix}ffn_down.weight")  # [n_ffn, n_embd]
        w3 = T(f"{prefix}ffn_up.weight")    # [n_embd, n_ffn]
        
        gate = w1.T @ ffn_input  # [n_ffn]
        up = w3.T @ ffn_input    # [n_ffn]
        down = w2.T @ (silu(gate) * up)  # [n_embd]
        
        x = x + down
    
    h = rmsnorm(x, output_norm)
    logits = OUT @ h  # [vocab]
    return logits

# Run autoregressive generation
generated_tokens = []

# Phase 1: Process entire prompt (prefill)
for pos, tok in enumerate(toks):
    logits = transformer_forward([tok])
    if pos == len(toks) - 1:
        # Save last prompt position logits for comparison
        prompt_logits = logits

print(f"[REF] Prefill complete ({len(toks)} tokens)")

# Phase 2: Generate N new tokens
for step in range(steps):
    next_token = int(np.argmax(logits))
    generated_tokens.append(next_token)
    
    # Dump top-5 logits for comparison
    top5 = np.argsort(logits)[::-1][:5]
    print(f"[REF] step={step} pos={len(toks)+step} -> token {next_token}")
    print(f"  top5: {[int(t) for t in top5]}")
    print(f"  logits: {[float(logits[t]) for t in top5]}")
    
    # Feed back and continue
    logits = transformer_forward([next_token])

print(f"\n[REF] Generated tokens: {json.dumps(generated_tokens)}")

result = {
    "generated": generated_tokens,
    "steps": steps,
    "prompt_tokens": toks
}
print(json.dumps(result))
