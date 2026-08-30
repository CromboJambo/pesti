#!/usr/bin/env python3
"""GGUF v3 tensor shape + KV dump. Usage: gguf_shapes.py <file> [tensor_substr]

Parses via the `gguf` crate's GGUFReader (memmap-backed) instead of a
hand-rolled struct parser — the old parser crashed on value types it
didn't know (e.g. 0x21000000 array encodings in Qwen2.5 files).
"""
import struct
import sys

from gguf import GGUFReader

# KV fields worth showing: architecture dims, heads, layers, vocab, template.
# Substrings span both legacy names (n_embd, n_layer, n_head, n_vocab) and
# modern ones (embedding_length, block_count, attention.head_count, ...).
KV_FILTER = [
    'template', 'vocab', 'embd', 'embedding', 'layer', 'head',
    'block_count', 'context_length', 'feed_forward', 'architecture',
]


def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else None

    with open(path, 'rb') as f:
        header = f.read(24)
    assert header[:4] == b'GGUF', header[:4]
    ver = struct.unpack_from('<I', header, 4)[0]

    r = GGUFReader(path)
    print(f"version={ver} n_tensors={len(r.tensors)} n_kv={len(r.fields)} data_section@{r.data_offset}")
    for k in r.fields:
        field = r.get_field(k)
        if field is None:
            continue
        if any(s in k for s in KV_FILTER):
            print(f"  {k} = {str(field.contents())[:200]}")
    print("---- tensors ----")
    for t in r.tensors:
        if filt is None or filt in t.name:
            print(f"  {t.name}: dims={[int(d) for d in t.shape]} type={int(t.tensor_type)} off={t.data_offset}")


if __name__ == '__main__':
    main()
