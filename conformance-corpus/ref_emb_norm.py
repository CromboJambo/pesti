#!/usr/bin/env python3
"""Reference embedding norms for PESTI conformance.

Dequantizes token_embd.weight from the GGUF and prints L2 norms for a few
token ids. PESTI's CPU path should produce norms within ~1e-3 relative
(quantization error) of these.
"""
import sys
import numpy as np
import gguf

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        'conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf'
    tokens = [int(x) for x in sys.argv[2:]] or [1, 2585, 1657, 14201, 99999]

    g = gguf.GGUFReader(path)
    t = [x for x in g.tensors if x.name == 'token_embd.weight'][0]
    shape = list(t.shape)  # [embed_dim, vocab_size]
    embed_dim, vocab = shape[0], shape[1]
    print(f'tensor {t.name} shape={shape} dtype={t.tensor_type}')

    # t.data is the raw quantized bytes. Use the gguf package dequantizer.
    raw = t.data
    # gguf.ReaderTensor has .data (bytes). Dequantize per dtype.
    from gguf.constants import GGMLQuantizationType
    qtype = GGMLQuantizationType(t.tensor_type)
    print('qtype:', qtype)
    # Use gguf's dequant helpers if available
    try:
        deq = t.data_dequantized  # may exist
    except Exception as e:
        deq = None
    if deq is None:
        # manual: use gguf package's dequantize function if present
        import gguf.dequantize as dq
        print('dequantize module:', [x for x in dir(dq) if not x.startswith('_')][:20])
        return

    arr = np.asarray(deq, dtype=np.float32).reshape(embed_dim, vocab)
    print('dequantized shape:', arr.shape)
    for tok in tokens:
        col = arr[:, tok]
        print(f'token {tok}: norm={np.linalg.norm(col):.6f} mean={col.mean():.6f}')

if __name__ == '__main__':
    main()
