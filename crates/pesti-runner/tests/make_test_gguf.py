#!/usr/bin/env python3
"""Generate a minimal synthetic GGUF file for testing."""
import struct
import sys
import os

vocab_size = 10

# Tensor specs: (name, shape)
tensors = [
    ("tok_embeddings.weight", [64]),
    ("output.weight", [vocab_size, 64]),
    ("layers.0.attention.wq.weight", [64, 64]),
    ("layers.0.attention.wk.weight", [64, 64]),
    ("layers.0.attention.wv.weight", [64, 64]),
    ("layers.0.attention.wo.weight", [64, 64]),
]

# Compute offsets
offsets = []
cum = 0
for name, shape in tensors:
    offsets.append(cum)
    elems = 1
    for d in shape:
        elems *= d
    cum += elems * 2

# KV pairs: (key, value)
kv_pairs = [
    ("general.architecture", "llama"),
    ("general.file_type", "F16"),
    ("general.alignment", 32),
    ("llama.context_length", 4096),
    ("llama.embedding_length", 64),
    ("llama.block_count", 1),
    ("llama.attention.head_count", 4),
    ("llama.attention.head_count_kv", 2),
    ("llama.feed_forward_length", 128),
    ("llama.rope.dimension_count", 64),
    ("llama.attention.layer_norm_rms_epsilon", 1e-5),
    ("tokenizer.ggml.model", 2),
    ("tokenizer.ggml.tokens", ["tok" + str(i) for i in range(vocab_size)]),
]

# Build file
buf = bytearray()

# Header
buf.extend(b"GGUF")
buf.extend(struct.pack("<I", 3))  # version
buf.extend(struct.pack("<Q", len(tensors)))  # tensor count
buf.extend(struct.pack("<Q", len(kv_pairs)))  # kv count

# KV pairs
for kv in kv_pairs:
    key, val = kv
    key_bytes = key.encode()
    buf.extend(struct.pack("<I", len(key_bytes)))
    buf.extend(key_bytes)
    
    if isinstance(val, str):
        buf.extend(struct.pack("<I", 10))  # String
        v_bytes = val.encode()
        buf.extend(struct.pack("<Q", len(v_bytes)))
        buf.extend(v_bytes)
    elif isinstance(val, int):
        buf.extend(struct.pack("<I", 4))  # Uint32
        buf.extend(struct.pack("<I", val))
    elif isinstance(val, float):
        buf.extend(struct.pack("<I", 8))  # Float32
        buf.extend(struct.pack("<f", val))
    elif isinstance(val, list):
        buf.extend(struct.pack("<B", 11))  # Array
        buf.extend(struct.pack("<Q", len(val)))  # count
        buf.extend(struct.pack("<B", 10))  # element_type = String
        for item in val:
            item_bytes = item.encode()
            buf.extend(struct.pack("<Q", len(item_bytes)))
            buf.extend(item_bytes)

# Tensor info
for i, (name, shape) in enumerate(tensors):
    name_bytes = name.encode()
    buf.extend(struct.pack("<Q", len(name_bytes)))
    buf.extend(name_bytes)
    buf.extend(struct.pack("<I", len(shape)))
    for dim in shape:
        buf.extend(struct.pack("<Q", dim))
    buf.extend(struct.pack("<I", 1))  # dtype F16
    buf.extend(struct.pack("<Q", offsets[i]))

# Compute data section start (same as Rust compute_data_section_start)
# header_base = 24 (magic+version+counts)
# + kv_size + tensor_size
# then align to 32
kv_size = 0
for kv in kv_pairs:
    key, val = kv
    key_bytes = key.encode()
    kv_size += 4 + len(key_bytes) + 4
    if isinstance(val, str):
        kv_size += 8 + len(val.encode())
    elif isinstance(val, int):
        kv_size += 4
    elif isinstance(val, float):
        kv_size += 4
    elif isinstance(val, list):
        kv_size += 1 + 8
        for item in val:
            kv_size += 8 + len(item.encode())

tensor_size = 0
for name, shape in tensors:
    tensor_size += 8 + len(name) + 4 + len(shape) * 8 + 4 + 8

data_section_start = 24 + kv_size + tensor_size
data_section_start = (data_section_start + 31) & ~31

# Pad to data section start
buf.extend(b"\x00" * (data_section_start - len(buf)))

# Tensor data (all F16 1.0 = 0x3F00)
total = 0
for _, shape in tensors:
    elems = 1
    for d in shape:
        elems *= d
    total += elems * 2
buf.extend(b"\x00\x3F" * total)

output_path = sys.argv[1]
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
with open(output_path, "wb") as f:
    f.write(buf)

print(f"Written {len(buf)} bytes to {output_path}")
print(f"Data section start: {data_section_start}")
print(f"Total tensor bytes: {total}")
for i, (name, shape) in enumerate(tensors):
    elems = 1
    for d in shape:
        elems *= d
    print(f"  {i}: {name} offset={offsets[i]} shape={shape} bytes={elems*2}")
