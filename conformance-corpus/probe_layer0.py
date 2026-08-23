#!/usr/bin/env python3
"""Probe: embedding norms + layer-0 intermediates for a single token, to
diff against pesti's PESTI_DEBUG_HIDDEN dump and localize the first
diverging sub-op."""
import sys
import numpy as np
import gguf

PATH = sys.argv[1]
tok = int(sys.argv[2]) if len(sys.argv) > 2 else 785

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
n_layer = int(kv(f"{arch}.block_count"))
n_head = int(kv(f"{arch}.attention.head_count"))
n_head_kv = int(kv(f"{arch}.attention.head_count_kv"))
n_embd = int(kv(f"{arch}.embedding_length"))
n_ffn = int(kv(f"{arch}.feed_forward_length"))
rope_base = float(kv(f"{arch}.rope.freq_base", 10000.0))
rms_eps = float(kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
head_dim = n_embd // n_head

tensor_by_name = {t.name: t for t in reader.tensors}
def T(name):
    t = tensor_by_name[name]
    flat = gguf.dequantize(t.data, t.tensor_type)
    shape = [int(s) for s in t.shape]
    arr = flat.reshape(shape[::-1]).T
    return arr.astype(np.float32)

E = T("token_embd.weight").T  # [vocab, n_embd]

x = E[tok].astype(np.float32)
print(f"[REF] tok={tok} embed norm={float(np.sqrt((x*x).sum())):.4f} head={x[:8].round(4).tolist()}")

# Layer 0 intermediates at pos 0 (single token, no cache history)
l = 0
attn_norm = T(f"blk.{l}.attn_norm.weight")
attn_input = (x / np.sqrt((x*x).mean() + rms_eps)) * attn_norm
wq = T(f"blk.{l}.attn_q.weight"); wk = T(f"blk.{l}.attn_k.weight")
wv = T(f"blk.{l}.attn_v.weight"); wo = T(f"blk.{l}.attn_output.weight")
bq = T(f"blk.{l}.attn_q.bias"); bk = T(f"blk.{l}.attn_k.bias"); bv = T(f"blk.{l}.attn_v.bias")
q = (wq.T @ attn_input) + bq
k = (wk.T @ attn_input) + bk
v = (wv.T @ attn_input) + bv
print(f"[REF] L0 attn_input norm={float(np.sqrt((attn_input*attn_input).sum())):.4f} head={attn_input[:8].round(4).tolist()}")
print(f"[REF] L0 q norm={float(np.sqrt((q*q).sum())):.4f} head={q[:8].round(4).tolist()}")
print(f"[REF] L0 k norm={float(np.sqrt((k*k).sum())):.4f} head={k[:8].round(4).tolist()}")
print(f"[REF] L0 v norm={float(np.sqrt((v*v).sum())):.4f} head={v[:8].round(4).tolist()}")

# RoPE on q,k at pos 0 -> cos=1, sin=0 => unchanged
# single-token attention: softmax over 1 position = 1.0, attn_out = v
# GQA: query head h uses kv head (h // n_rep). The expansion is BLOCK-wise:
# heads 0..n_rep-1 -> kv0, heads n_rep..2*n_rep-1 -> kv1, etc.
# np.repeat along axis=0 of the [n_kv, head_dim] matrix gives exactly that.
n_rep = n_head // n_head_kv
attn_out = np.repeat(v.reshape(n_head_kv, head_dim), n_rep, axis=0).flatten()  # [n_embd]
attn_proj = wo.T @ attn_out
x2 = x + attn_proj
print(f"[REF] L0 attn_proj norm={float(np.sqrt((attn_proj*attn_proj).sum())):.4f} head={attn_proj[:8].round(4).tolist()}")
print(f"[REF] L0 after-attn x norm={float(np.sqrt((x2*x2).sum())):.4f} head={x2[:8].round(4).tolist()}")

# FFN
ffn_norm = T(f"blk.{l}.ffn_norm.weight")
ffn_input = (x2 / np.sqrt((x2*x2).mean() + rms_eps)) * ffn_norm
w1 = T(f"blk.{l}.ffn_gate.weight"); w2 = T(f"blk.{l}.ffn_down.weight"); w3 = T(f"blk.{l}.ffn_up.weight")
gate = w1.T @ ffn_input
up = w3.T @ ffn_input
silu = gate / (1.0 + np.exp(-gate))
down = w2.T @ (silu * up)
x3 = x2 + down
print(f"[REF] L0 ffn_input norm={float(np.sqrt((ffn_input*ffn_input).sum())):.4f} head={ffn_input[:8].round(4).tolist()}")
print(f"[REF] L0 gate norm={float(np.sqrt((gate*gate).sum())):.4f} head={gate[:8].round(4).tolist()}")
print(f"[REF] L0 up   norm={float(np.sqrt((up*up).sum())):.4f} head={up[:8].round(4).tolist()}")
print(f"[REF] L0 down norm={float(np.sqrt((down*down).sum())):.4f} head={down[:8].round(4).tolist()}")
print(f"[REF] L0 after-ffn x norm={float(np.sqrt((x3*x3).sum())):.4f} head={x3[:8].round(4).tolist()}")
