#!/usr/bin/env python3
"""Decisive dequant test.

Takes the RAW Q8_0 bytes of token_embd.weight (via the validated `gguf`
package) and dequantizes them three ways:
  A) the gguf package's own dequantize (validated: gives correct argmax 2585)
  B) pesti's EXACT Q8_0 algorithm ported to Python (34-byte blocks, f16 scale
     at [0:2], 32 int8 at [2:34], x = scale * q)
  C) the token-785 embedding pesti actually produced at runtime

If A == B, pesti's dequant ALGORITHM is correct -> the bug is in byte
extraction / offset / element_count in load_gguf_weights.
If A != B, pesti's dequant algorithm is wrong.
"""
import sys
import numpy as np
import gguf

PATH = sys.argv[1]
tok = int(sys.argv[2]) if len(sys.argv) > 2 else 785
EMBD = 896

r = gguf.GGUFReader(PATH)
t = {x.name: x for x in r.tensors}["token_embd.weight"]
raw = np.frombuffer(t.data, dtype=np.uint8)  # raw Q8_0 bytes
n_elem = 896 * 151936
print(f"raw bytes={len(raw)}  n_elem={n_elem}  blocks={n_elem//32}  "
      f"expected_bytes={(n_elem//32)*34}")

# A) package dequantize (validated)
A = gguf.dequantize(t.data, t.tensor_type).astype(np.float32)
print(f"A (package) len={len(A)}  elem[0:4]={A[0:4].round(4).tolist()}")

# B) pesti's exact algorithm (fully vectorized, same math)
def pesti_q8_0(data, element_count):
    num_blocks = (element_count + 31) // 32
    blocks = data[:num_blocks * 34].reshape(num_blocks, 34)
    scales = np.frombuffer(blocks[:, 0:2].tobytes(), dtype="<f2").astype(np.float32)
    qs = blocks[:, 2:34].astype(np.int8).astype(np.float32)
    out = (scales[:, None] * qs).reshape(-1)
    return out[:element_count]

B_full = pesti_q8_0(raw, n_elem)
print(f"B (pesti algo) len={len(B_full)}  elem[0:4]={B_full[0:4].round(4).tolist()}")

# Compare A and B on a sample
diff = np.abs(A - B_full)
print(f"A vs B: max_abs_diff={diff.max():.6f}  mean_abs_diff={diff.mean():.6f}")
print(f"A vs B: equal within 1e-3? {bool((diff < 1e-3).all())}")

# Token 785 embedding (flat index tok*EMBD .. +EMBD)
s = tok * EMBD
embA = A[s:s+EMBD]
embB = B_full[s:s+EMBD]
print(f"\ntoken {tok} embed[:8]:")
print(f"  A (package) = {embA[:8].round(4).tolist()}  norm={np.sqrt((embA**2).sum()):.4f}")
print(f"  B (pesti)   = {embB[:8].round(4).tolist()}  norm={np.sqrt((embB**2).sum()):.4f}")
print(f"  PESTI runtime = [0.096, -0.073, 0.007, -0.504, -0.249, 0.318, 0.041, 0.232]  norm=13.6885")
