#!/usr/bin/env python3
"""Diagnose the ffn_down (w2) layout mismatch.

pesti's Linear.forward computes:  down[o] = sum_i swiglu[i] * W[o*in + i]
  with in=4864 (intermediate), out=896 (hidden).
The numpy reference computes:      down = w2_ref.T @ swiglu,  w2_ref = T(...)
  => down[o] = sum_i flat[o*shape[0] + i] * swiglu[i]

So pesti is correct only if GGUF shape[0] of ffn_down == 4864.
This script computes down under several candidate layouts and reports which
matches the reference (norm 7.9331).
"""
import sys
import numpy as np
import gguf

PATH = sys.argv[1]
tok = 785
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
    if int(t) == 8:
        return bytes(np.asarray(last, dtype=np.uint8)).decode("utf-8")
    arr = np.asarray(last)
    if arr.size == 1:
        return arr.reshape(()).item()
    return arr

def kv(key, default=None):
    v = field(key)
    return default if v is None else v

arch = field("general.architecture")
n_head = int(kv(f"{arch}.attention.head_count"))
n_head_kv = int(kv(f"{arch}.attention.head_count_kv"))
n_embd = int(kv(f"{arch}.embedding_length"))
n_ffn = int(kv(f"{arch}.feed_forward_length"))
rope_base = float(kv(f"{arch}.rope.freq_base", 10000.0))
rms_eps = float(kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
head_dim = n_embd // n_head
print(f"arch={arch} n_embd={n_embd} n_ffn={n_ffn} n_head={n_head} n_kv={n_head_kv}")

tensor_by_name = {t.name: t for t in reader.tensors}
def T(name):
    t = tensor_by_name[name]
    flat = gguf.dequantize(t.data, t.tensor_type)
    shape = [int(s) for s in t.shape]
    arr = flat.reshape(shape[::-1]).T
    return arr.astype(np.float32)

# Report the actual GGUF shape of ffn_down
t2 = tensor_by_name["blk.0.ffn_down.weight"]
print(f"ffn_down GGUF shape = {list(t2.shape)}  (flat len={gguf.dequantize(t2.data, t2.tensor_type).size})")

E = T("token_embd.weight").T
x = E[tok].astype(np.float32)

l = 0
attn_norm = T(f"blk.{l}.attn_norm.weight")
attn_input = (x / np.sqrt((x*x).mean() + rms_eps)) * attn_norm
wq = T(f"blk.{l}.attn_q.weight"); wk = T(f"blk.{l}.attn_k.weight")
wv = T(f"blk.{l}.attn_v.weight"); wo = T(f"blk.{l}.attn_output.weight")
bq = T(f"blk.{l}.attn_q.bias"); bk = T(f"blk.{l}.attn_k.bias"); bv = T(f"blk.{l}.attn_v.bias")
q = (wq.T @ attn_input) + bq
k = (wk.T @ attn_input) + bk
v = (wv.T @ attn_input) + bv
n_rep = n_head // n_head_kv
attn_out = np.repeat(v.reshape(n_head_kv, head_dim), n_rep, axis=0).flatten()
attn_proj = wo.T @ attn_out
x2 = x + attn_proj

ffn_norm = T(f"blk.{l}.ffn_norm.weight")
ffn_input = (x2 / np.sqrt((x2*x2).mean() + rms_eps)) * ffn_norm
w1 = T(f"blk.{l}.ffn_gate.weight"); w2 = T(f"blk.{l}.ffn_down.weight"); w3 = T(f"blk.{l}.ffn_up.weight")
gate = w1.T @ ffn_input
up = w3.T @ ffn_input
silu = gate / (1.0 + np.exp(-gate))
swiglu = silu * up

flat = gguf.dequantize(t2.data, t2.tensor_type).astype(np.float32).reshape(-1)
print(f"flat len={flat.size}  (n_embd*n_ffn={n_embd*n_ffn})")

# Reference down (correct)
down_ref = w2.T @ swiglu
print(f"\n[REF]  down norm={float(np.sqrt((down_ref*down_ref).sum())):.4f} head={down_ref[:8].round(4).tolist()}")

# Candidate layouts: down[o] = sum_i swiglu[i] * flat[o*stride + i]
for stride in (n_ffn, n_embd):
    n_out = flat.size // stride
    rows = flat[:n_out*stride].reshape(n_out, stride)
    down = rows @ swiglu[:stride]
    print(f"[L{stride}] down norm={float(np.sqrt((down*down).sum())):.4f} (out={n_out}) head={down[:8].round(4).tolist()}")
