#!/usr/bin/env python3
"""GGUF v3 tensor shape + KV dump. Usage: gguf_shapes.py <file> [tensor_substr]"""
import struct, sys

def rdstr(data, o):
    n, = struct.unpack_from('<Q', data, o); o += 8
    return data[o:o+n].decode('utf-8', 'replace'), o + n

def rdval(data, o):
    t, = struct.unpack_from('<I', data, o); o += 4
    if t == 8:
        return rdstr(data, o)
    if t == 9:
        et, = struct.unpack_from('<I', data, o); o += 4
        cnt, = struct.unpack_from('<Q', data, o); o += 8
        arr = []
        for _ in range(cnt):
            v, o = rdval(data, o)
            arr.append(v)
        return arr, o
    fmt = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}[t]
    v, = struct.unpack_from(fmt, data, o); o += struct.calcsize(fmt)
    return v, o

def main():
    path = sys.argv[1]
    filt = sys.argv[2] if len(sys.argv) > 2 else None
    with open(path, 'rb') as f:
        data = f.read()
    assert data[:4] == b'GGUF', data[:4]
    ver, = struct.unpack_from('<I', data, 4)
    n_tensors, n_kv = struct.unpack_from('<QQ', data, 8)
    off = 24
    kv = {}
    for _ in range(n_kv):
        k, off = rdstr(data, off)
        v, off = rdval(data, off)
        kv[k] = v
    print(f"version={ver} n_tensors={n_tensors} n_kv={n_kv} data_section@{off}")
    for k, v in kv.items():
        if 'template' in k or 'vocab' in k or 'n_embd' in k or 'n_layer' in k \
           or 'n_head' in k or 'n_vocab' in k or k == 'general.architecture':
            s = str(v)
            print(f"  {k} = {s[:200]}")
    print("---- tensors ----")
    for i in range(n_tensors):
        name, off = rdstr(data, off)
        ndims, = struct.unpack_from('<I', data, off); off += 4
        dims = struct.unpack_from(f'<{ndims}Q', data, off); off += ndims * 8
        ttype, = struct.unpack_from('<I', data, off); off += 4
        toff, = struct.unpack_from('<Q', data, off); off += 8
        if filt is None or filt in name:
            print(f"  {name}: dims={list(dims)} type={ttype} off={toff}")

if __name__ == '__main__':
    main()
