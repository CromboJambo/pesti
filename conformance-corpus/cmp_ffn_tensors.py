#!/usr/bin/env python3
"""Compare pesti's dumped FFN intermediates (gate/up/swiglu/down) against the
numpy reference at full precision to localize the `down` divergence."""
import sys
import numpy as np
import gguf

PATH = sys.argv[1]
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "/tmp/ffncmp"
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
rms_eps = float(kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
head_dim = n_embd // n_head

tensor_by_name = {t.name: t for t in reader.tensors}
def T(name):
    t = tensor_by_name[name]
    flat = gguf.dequantize(t.data, t.tensor_type)
    shape = [int(s) for s in t.shape]
    arr = flat.reshape(shape[::-1]).T
    return arr.astype(np.float32)

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
down = w2.T @ swiglu

def load(name):
    return np.fromfile(f"{PREFIX}.{name}.f32", dtype=np.float32)

def cmp(name, got, ref):
    d = np.abs(got - ref)
    print(f"{name:7s} len={got.size}/{ref.size} maxdiff={d.max():.3e} "
          f"mean={d.mean():.3e} normratio={np.linalg.norm(got)/np.linalg.norm(ref):.5f} "
          f"corr={np.corrcoef(got, ref)[0,1]:.6f}")
    return d

cmp("gate", load("gate"), gate)
cmp("up", load("up"), up)
cmp("swiglu", load("swiglu"), swiglu)
cmp("down", load("down"), down)

# If swiglu matches but down doesn't, the bug is in w2.forward.
# In that case, recompute down from pesti's OWN swiglu using the reference w2
# to confirm the GEMM is the culprit.
got_swiglu = load("swiglu")
down_from_pesti_swiglu = w2.T @ got_swiglu
cmp("down(pesti_swiglu, ref_w2)", down_from_pesti_swiglu, down)

# And recompute down from reference swiglu using pesti's w2 layout
got_w2 = np.fromfile("/tmp/deqcmp5/w2.f32", dtype=np.float32).reshape(896, 4864)
down_ref_swiglu_pesti_w2 = got_w2 @ swiglu
cmp("down(ref_swiglu, pesti_w2)", down_ref_swiglu_pesti_w2, down)
