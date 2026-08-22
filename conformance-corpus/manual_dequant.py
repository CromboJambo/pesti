#!/usr/bin/env python3
"""Ground-truth GGUF header parser + manual Q8_0 dequant of a specific
token's embedding, straight from the raw file bytes. This bypasses both the
`gguf` package and pesti, so it tells us what the file ACTUALLY contains.
"""
import sys
import struct
import numpy as np

PATH = sys.argv[1]
tok = int(sys.argv[2]) if len(sys.argv) > 2 else 785
EMBD = 896

with open(PATH, "rb") as f:
    data = f.read()

# Header
magic = data[0:4]
version = struct.unpack_from("<I", data, 4)[0]
n_tensors = struct.unpack_from("<Q", data, 8)[0]
n_kv = struct.unpack_from("<Q", data, 16)[0]
print(f"magic={magic} version={version} n_tensors={n_tensors} n_kv={n_kv}")

GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_UINT16, GGUF_TYPE_INT16 = 0, 1, 2, 3
GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32, GGUF_TYPE_BOOL = 4, 5, 6, 7
GGUF_TYPE_STRING, GGUF_TYPE_ARRAY, GGUF_TYPE_UINT64, GGUF_TYPE_INT64 = 8, 9, 10, 11
GGUF_TYPE_FLOAT64 = 12

def read_string(off):
    ln = struct.unpack_from("<Q", data, off)[0]
    s = data[off + 8 : off + 8 + ln].decode("utf-8")
    return s, off + 8 + ln

def read_value(off, vtype):
    """Read a GGUF value, return (python_value, new_offset)."""
    if vtype == GGUF_TYPE_UINT8:
        return data[off], off + 1
    if vtype == GGUF_TYPE_INT8:
        return struct.unpack_from("<b", data, off)[0], off + 1
    if vtype == GGUF_TYPE_UINT16:
        return struct.unpack_from("<H", data, off)[0], off + 2
    if vtype == GGUF_TYPE_INT16:
        return struct.unpack_from("<h", data, off)[0], off + 2
    if vtype == GGUF_TYPE_UINT32:
        return struct.unpack_from("<I", data, off)[0], off + 4
    if vtype == GGUF_TYPE_INT32:
        return struct.unpack_from("<i", data, off)[0], off + 4
    if vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack_from("<f", data, off)[0], off + 4
    if vtype == GGUF_TYPE_BOOL:
        return data[off] != 0, off + 1
    if vtype == GGUF_TYPE_STRING:
        return read_string(off)
    if vtype == GGUF_TYPE_UINT64:
        return struct.unpack_from("<Q", data, off)[0], off + 8
    if vtype == GGUF_TYPE_INT64:
        return struct.unpack_from("<q", data, off)[0], off + 8
    if vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack_from("<d", data, off)[0], off + 8
    if vtype == GGUF_TYPE_ARRAY:
        atype = data[off]
        alen = struct.unpack_from("<Q", data, off + 1)[0]
        off += 9
        vals = []
        for _ in range(alen):
            v, off = read_value(off, atype)
            vals.append(v)
        return vals, off
    raise ValueError(f"unknown vtype {vtype} at {off}")

# Skip metadata kv
off = 24
for _ in range(n_kv):
    _, off = read_string(off)  # key
    vtype = data[off]
    off += 1
    _, off = read_value(off, vtype)  # value

# Tensor info
tensors = {}
for _ in range(n_tensors):
    name, off = read_string(off)
    ndims = struct.unpack_from("<I", data, off)[0]
    off += 4
    dims = list(struct.unpack_from(f"<{ndims}Q", data, off))
    off += ndims * 8
    dtype = struct.unpack_from("<I", data, off)[0]
    off += 4
    toff = struct.unpack_from("<Q", data, off)[0]
    off += 8
    tensors[name] = (dims, dtype, toff)

# Data section starts at 32-byte alignment after tensor info
data_start = (off + 31) // 32 * 32
print(f"data_start={data_start}")

DTYPE_NAMES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 4: "Q5_0", 5: "Q5_1",
               6: "Q8_0", 7: "Q2_K", 8: "Q3_K", 9: "Q4_K", 10: "Q5_K",
               11: "Q6_K", 12: "Q8_K", 13: "Q2_K_S", 14: "Q3_K_S", 15: "BF16"}

for nm in ["token_embd.weight", "output.weight", "blk.0.attn_q.weight", "blk.0.attn_norm.weight"]:
    dims, dtype, toff = tensors[nm]
    print(f"{nm}: dims={dims} dtype={DTYPE_NAMES.get(dtype, dtype)} file_offset={toff}")

# Now manually dequantize token `tok`'s embedding from token_embd.weight raw bytes.
dims, dtype, toff = tensors["token_embd.weight"]
print(f"\n=== Manual Q8_0 dequant of token {tok} embedding ===")
print(f"dtype={DTYPE_NAMES.get(dtype, dtype)} (expect Q8_0=6)")
base_file = data_start + toff

if dtype != 6:
    print(f"WARNING: token_embd is NOT Q8_0 (dtype={dtype}); manual Q8_0 path may be wrong")

# flat element index for token tok, component i = tok*EMBD + i
elem_start = tok * EMBD
# Q8_0: 32 elems / 34 bytes
block0 = elem_start // 32
elem_in_block = elem_start % 32
block_bytes = 34
raw = data[base_file + block0 * block_bytes : base_file + (block0 + 2) * block_bytes]
# scale = f16 at raw[0:2]
scale = np.frombuffer(raw[0:2], dtype="<f2").astype(np.float32)[0]
qs = np.frombuffer(raw[2:2 + 32], dtype=np.int8).astype(np.float32)
deq = scale * qs
# We want components 0..8 of the token => elements elem_start..elem_start+8
# which are deq[elem_in_block : elem_in_block+8]
vals = deq[elem_in_block : elem_in_block + 8]
print(f"scale={scale:.6f}")
print(f"token {tok} embed[:8] (manual Q8_0) = {vals.round(4).tolist()}")
print(f"norm of full 896 (approx, first block only) = {np.sqrt((deq**2).sum()):.4f}")

# Reference values (from validated numpy forward):
ref = [-0.032, -0.0018, 0.0121, -0.0156, 0.0156, -0.0295, 0.0025, -0.0156]
print(f"REF embed[:8] = {ref}")
print(f"PESTI embed[:8] = [0.096, -0.073, 0.007, -0.504, -0.249, 0.318, 0.041, 0.232]")
