#!/usr/bin/env python3
"""Reference forward pass for Qwen2.5-0.5B in pure numpy float32.

Loads tensors directly from the GGUF file via the `gguf` package (which
dequantizes Q8_0 etc. to float32 numpy), implements the Qwen2 forward, and
prints the argmax + top-5 logits for the last prompt position. This is the
independent oracle to diff against pesti's output and localize the bug.

Usage: ref_forward.py <model.gguf> [tok1,tok2,...]
"""
import sys
import numpy as np
import gguf

PATH = sys.argv[1]
if len(sys.argv) > 2:
    toks = [int(t) for t in sys.argv[2].split(",")]
else:
    toks = [785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13]  # fox prompt

reader = gguf.GGUFReader(PATH)

def field(key):
    """Decode a GGUF scalar/string field to a Python value.

    A ReaderField stores its value as the LAST entry in `.parts` (a memmap);
    `.types[-1]` is the GGUFValueType. STRING -> bytes; else numpy scalar/array.
    """
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
print(f"arch={arch}")

n_layer = int(kv(f"{arch}.block_count"))
n_head = int(kv(f"{arch}.attention.head_count"))
n_head_kv = int(kv(f"{arch}.attention.head_count_kv"))
n_embd = int(kv(f"{arch}.embedding_length"))
n_ffn = int(kv(f"{arch}.feed_forward_length"))
rope_base = float(kv(f"{arch}.rope.freq_base", 10000.0))
rms_eps = float(kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
head_dim = n_embd // n_head
print(f"n_layer={n_layer} n_head={n_head} n_head_kv={n_head_kv} n_embd={n_embd} "
      f"n_ffn={n_ffn} rope_base={rope_base} rms_eps={rms_eps} head_dim={head_dim}")

# Build a name -> tensor map.
tensor_by_name = {t.name: t for t in reader.tensors}

def T(name):
    """Dequantize a tensor to a float32 numpy array in LOGICAL shape.

    GGUF stores tensors with dim0 innermost (contiguous). dequantize() returns
    a flat array in that same order, so reshape to the REVERSED logical shape
    then transpose to recover the logical [dim0, dim1, ...] layout.
    """
    t = tensor_by_name[name]
    flat = gguf.dequantize(t.data, t.tensor_type)
    shape = [int(s) for s in t.shape]
    arr = flat.reshape(shape[::-1]).T
    return arr.astype(np.float32)

# token_embd: shape [n_embd, vocab] (dim0=n_embd fastest) -> we want [vocab, n_embd]
emb = T("token_embd.weight")
print("token_embd shape (gguf order):", emb.shape)
# gguf get_tensor returns array with the SAME axis order as the file:
# file ne=[n_embd, vocab] means axis0=n_embd, axis1=vocab.
# So emb[i, j] = embedding of token j, component i. We want E[token, comp] = emb.T
E = emb.T  # [vocab, n_embd]
print("E (vocab, n_embd):", E.shape)

out_w = T("output.weight")  # [n_embd, vocab]
OUT = out_w.T  # [vocab, n_embd]
print("OUT (vocab, n_embd):", OUT.shape)

output_norm = T("output_norm.weight")  # [n_embd]

def rmsnorm(x, w, eps=rms_eps):
    # x: [n_embd]
    ms = (x * x).mean()
    return (x / np.sqrt(ms + eps)) * w

def rope(x, pos, n_heads, base=rope_base):
    # x: [n_heads, head_dim]
    d = head_dim
    inv_freq = 1.0 / (base ** (np.arange(0, d, 2, dtype=np.float32) / d))
    freqs = (pos * inv_freq)  # [d/2]
    cos = np.cos(freqs)  # [d/2]
    sin = np.sin(freqs)
    # half-split: first half and second half
    x0 = x[:, :d//2]
    x1 = x[:, d//2:]
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return np.concatenate([out0, out1], axis=1)

def silu(x):
    return x / (1.0 + np.exp(-x))

# Forward pass over the prompt, building KV caches.
K_cache = {l: [] for l in range(n_layer)}  # list of [n_head_kv, head_dim]
V_cache = {l: [] for l in range(n_layer)}

h = None
for pos, tok in enumerate(toks):
    x = E[tok].astype(np.float32)  # [n_embd]
    if pos == len(toks) - 1:
        print(f"\n[REF] pos={pos} tok={tok} embed[:8]={x[:8].round(3).tolist()}")
    for l in range(n_layer):
        # attention
        attn_norm = T(f"blk.{l}.attn_norm.weight")
        attn_input = rmsnorm(x, attn_norm)
        # projections: attn_q.weight shape [n_embd, n_head*head_dim] -> W^T @ x
        wq = T(f"blk.{l}.attn_q.weight")  # [n_embd, n_embd]
        wk = T(f"blk.{l}.attn_k.weight")  # [n_embd, n_head_kv*head_dim]
        wv = T(f"blk.{l}.attn_v.weight")  # [n_embd, n_head_kv*head_dim]
        wo = T(f"blk.{l}.attn_output.weight")  # [n_embd, n_embd]
        bq = T(f"blk.{l}.attn_q.bias")
        bk = T(f"blk.{l}.attn_k.bias")
        bv = T(f"blk.{l}.attn_v.bias")
        # q = (attn_input @ wq) + bq ; wq is [in, out] so q = wq.T @ x
        q = (wq.T @ attn_input) + bq  # [n_embd]
        k = (wk.T @ attn_input) + bk  # [n_head_kv*head_dim]
        v = (wv.T @ attn_input) + bv  # [n_head_kv*head_dim]
        q = q.reshape(n_head, head_dim)
        k = k.reshape(n_head_kv, head_dim)
        v = v.reshape(n_head_kv, head_dim)
        q = rope(q, pos, n_head)
        k = rope(k, pos, n_head_kv)
        K_cache[l].append(k)
        V_cache[l].append(v)
        Ks = np.stack(K_cache[l], axis=1)  # [n_head_kv, seq, head_dim]
        Vs = np.stack(V_cache[l], axis=1)  # [n_head_kv, seq, head_dim]
        # GQA: each q head h uses kv head h // (n_head//n_head_kv)
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
        attn_proj = (wo.T @ attn_flat)  # [n_embd]
        x = x + attn_proj
        # ffn
        ffn_norm = T(f"blk.{l}.ffn_norm.weight")
        ffn_input = rmsnorm(x, ffn_norm)
        w1 = T(f"blk.{l}.ffn_gate.weight")  # [n_embd, n_ffn]
        w2 = T(f"blk.{l}.ffn_down.weight")  # [n_ffn, n_embd]
        w3 = T(f"blk.{l}.ffn_up.weight")    # [n_embd, n_ffn]
        gate = (w1.T @ ffn_input)  # [n_ffn]
        up = (w3.T @ ffn_input)    # [n_ffn]
        down = (w2.T @ (silu(gate) * up))  # [n_embd]
        x = x + down
        if pos == len(toks) - 1:
            norm = float(np.sqrt((x * x).sum()))
            print(f"[REF] layer={l} pos={pos} norm={norm:.4f} head={x[:8].round(3).tolist()}")
    h = x
    if pos == len(toks) - 1:
        last_hidden = h

h = rmsnorm(last_hidden, output_norm)
logits = OUT @ h  # [vocab]
print(f"[REF] pre-head norm={float(np.sqrt((h*h).sum())):.4f} head={h[:8].round(3).tolist()}")
top = np.argsort(logits)[::-1][:8]
print("\n=== REFERENCE (numpy float32) ===")
print("top-8 tokens:", top.tolist())
print("top-8 logits:", [round(float(logits[i]), 3) for i in top])
print("argmax:", int(top[0]))
# Save full logits for diffing
np.save("/tmp/ref_logits.npy", logits)
print("saved /tmp/ref_logits.npy")
