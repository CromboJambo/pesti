#!/usr/bin/env python3
"""All-layer numpy probe for Qwen2.5-0.5B conformance.

Runs the pure-numpy float32 Qwen2 forward over a prompt (same math as
`ref_forward.py`) and captures the FULL per-layer hidden-state vector at the
last prompt position, saving each to disk so a diff tool can compare full
vectors (not just the first 8 dims) against pesti's dumper.

This is the all-layer generalization of `probe_layer0.py`. It replaces the
old broken stub that imported a non-existent `pesti_runner.utils` module
(pesti has no Python bindings; the Rust dumper `dump_all_layers.rs` is the
Rust-side counterpart).

Outputs (under --out, default /tmp/probe_all_layers):
    layer_<l>.f32    full [n_embd] hidden state after layer l (last prompt pos)
    prehead.f32      full [n_embd] hidden state after output_norm
    embed.f32        full [n_embd] input embedding (last prompt pos)
    logits.f32       full [vocab] logits
    manifest.json    per-layer {norm, head8, path} + top-8 + argmax

Usage:
    python3 probe_all_layers.py <model.gguf> [tok1,tok2,...] [--out DIR]

Default prompt is the same 10-token "fox" prompt ref_forward.py uses.
"""
import json
import os
import sys

import numpy as np
import gguf


def parse_args(argv):
    out = "/tmp/probe_all_layers"
    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = argv[i + 1]
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    if not positional:
        print(__doc__)
        sys.exit(2)
    path = positional[0]
    toks = [int(t) for t in positional[1].split(",")] if len(positional) > 1 else [
        785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13,
    ]
    return path, toks, out


def make_reader(path):
    reader = gguf.GGUFReader(path)

    def field(key):
        f = reader.get_field(key)
        if f is None:
            return None
        parts, types = f.parts, f.types
        if not types:
            return None
        t, last = types[-1], parts[-1]
        if int(t) == 8:  # STRING
            return bytes(np.asarray(last, dtype=np.uint8)).decode("utf-8")
        arr = np.asarray(last)
        return arr.reshape(()).item() if arr.size == 1 else arr

    def kv(key, default=None):
        v = field(key)
        return default if v is None else v

    tensor_by_name = {t.name: t for t in reader.tensors}

    def T(name):
        t = tensor_by_name[name]
        flat = gguf.dequantize(t.data, t.tensor_type)
        shape = [int(s) for s in t.shape]
        return flat.reshape(shape[::-1]).T.astype(np.float32)

    return field, kv, T


def main():
    path, toks, out = parse_args(sys.argv[1:])
    os.makedirs(out, exist_ok=True)

    field, kv, T = make_reader(path)
    arch = field("general.architecture")
    n_layer = int(kv(f"{arch}.block_count"))
    n_head = int(kv(f"{arch}.attention.head_count"))
    n_head_kv = int(kv(f"{arch}.attention.head_count_kv"))
    n_embd = int(kv(f"{arch}.embedding_length"))
    rope_base = float(kv(f"{arch}.rope.freq_base", 10000.0))
    rms_eps = float(kv(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6))
    head_dim = n_embd // n_head

    print(
        f"arch={arch} n_layer={n_layer} n_head={n_head} n_head_kv={n_head_kv} "
        f"n_embd={n_embd} rope_base={rope_base} rms_eps={rms_eps} head_dim={head_dim}"
    )

    E = T("token_embd.weight").T  # [vocab, n_embd]
    OUT = T("output.weight").T  # [vocab, n_embd]
    output_norm = T("output_norm.weight")  # [n_embd]

    def rmsnorm(x, w):
        ms = (x * x).mean()
        return (x / np.sqrt(ms + rms_eps)) * w

    def rope(x, pos):
        d = head_dim
        inv_freq = 1.0 / (rope_base ** (np.arange(0, d, 2, dtype=np.float32) / d))
        freqs = pos * inv_freq
        cos, sin = np.cos(freqs), np.sin(freqs)
        x0, x1 = x[:, : d // 2], x[:, d // 2:]
        out0 = x0 * cos - x1 * sin
        out1 = x0 * sin + x1 * cos
        return np.concatenate([out0, out1], axis=1)

    def silu(x):
        return x / (1.0 + np.exp(-x))

    K_cache = {l: [] for l in range(n_layer)}
    V_cache = {l: [] for l in range(n_layer)}

    last = len(toks) - 1
    layer_vecs = {}
    embed_vec = None
    prehead_vec = None

    for pos, tok in enumerate(toks):
        x = E[tok].astype(np.float32)
        if pos == last:
            embed_vec = x.copy()

        for l in range(n_layer):
            attn_input = rmsnorm(x, T(f"blk.{l}.attn_norm.weight"))
            wq = T(f"blk.{l}.attn_q.weight")
            wk = T(f"blk.{l}.attn_k.weight")
            wv = T(f"blk.{l}.attn_v.weight")
            wo = T(f"blk.{l}.attn_output.weight")
            bq = T(f"blk.{l}.attn_q.bias")
            bk = T(f"blk.{l}.attn_k.bias")
            bv = T(f"blk.{l}.attn_v.bias")
            q = (wq.T @ attn_input) + bq
            k = (wk.T @ attn_input) + bk
            v = (wv.T @ attn_input) + bv
            q = rope(q.reshape(n_head, head_dim), pos)
            k = rope(k.reshape(n_head_kv, head_dim), pos)
            v = v.reshape(n_head_kv, head_dim)
            K_cache[l].append(k)
            V_cache[l].append(v)
            Ks = np.stack(K_cache[l], axis=1)  # [n_kv, seq, hd]
            Vs = np.stack(V_cache[l], axis=1)
            scale = 1.0 / np.sqrt(head_dim)
            attn_out = np.zeros((n_head, head_dim), dtype=np.float32)
            hpq = n_head // n_head_kv
            for hh in range(n_head):
                g = hh // hpq
                scores = (Ks[g] @ q[hh]) * scale
                scores = scores - scores.max()
                exps = np.exp(scores)
                w = exps / exps.sum()
                attn_out[hh] = (w[:, None] * Vs[g]).sum(axis=0)
            x = x + wo.T @ attn_out.reshape(-1)

            ffn_input = rmsnorm(x, T(f"blk.{l}.ffn_norm.weight"))
            w1 = T(f"blk.{l}.ffn_gate.weight")
            w2 = T(f"blk.{l}.ffn_down.weight")
            w3 = T(f"blk.{l}.ffn_up.weight")
            gate = w1.T @ ffn_input
            up = w3.T @ ffn_input
            x = x + w2.T @ (silu(gate) * up)

            if pos == last:
                layer_vecs[l] = x.copy()

    # NOTE: `last` is the last PROMPT POSITION (len(toks)-1); the layer dict is
    # keyed by LAYER index, so the final hidden state is layer n_layer-1, not
    # layer `last`. (Earlier version indexed layer_vecs[last] by position, which
    # silently applied output_norm to the wrong layer's output.)
    h = rmsnorm(layer_vecs[n_layer - 1], output_norm)
    prehead_vec = h.copy()
    logits = OUT @ h  # [vocab]

    # Save full vectors + manifest.
    np.save(f"{out}/embed.f32", embed_vec)
    for l, vec in layer_vecs.items():
        np.save(f"{out}/layer_{l}.f32", vec)
    np.save(f"{out}/prehead.f32", prehead_vec)
    np.save(f"{out}/logits.f32", logits)

    top = np.argsort(logits)[::-1][:8]
    manifest = {
        "arch": arch,
        "n_layer": n_layer,
        "n_embd": n_embd,
        "prompt": toks,
        "last_pos": last,
        "embed": {
            "norm": float(np.sqrt((embed_vec * embed_vec).sum())),
            "head8": embed_vec[:8].round(4).tolist(),
            "path": f"{out}/embed.f32",
        },
        "layers": [
            {
                "layer": l,
                "norm": float(np.sqrt((layer_vecs[l] * layer_vecs[l]).sum())),
                "head8": layer_vecs[l][:8].round(4).tolist(),
                "path": f"{out}/layer_{l}.f32",
            }
            for l in range(n_layer)
        ],
        "prehead": {
            "norm": float(np.sqrt((prehead_vec * prehead_vec).sum())),
            "head8": prehead_vec[:8].round(4).tolist(),
            "path": f"{out}/prehead.f32",
        },
        "top8_tokens": top.tolist(),
        "top8_logits": [round(float(logits[i]), 4) for i in top],
        "argmax": int(top[0]),
        "logits_path": f"{out}/logits.f32",
    }
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Print summary table.
    print(f"{'layer':>6} | {'norm':>10} | head8")
    print("-" * 60)
    for l in range(n_layer):
        m = manifest["layers"][l]
        print(f"{l:>6} | {m['norm']:>10.4f} | {m['head8']}")
    print("-" * 60)
    print(f"{'prehead':>6} | {manifest['prehead']['norm']:>10.4f} | {manifest['prehead']['head8']}")
    print()
    print("top-8 tokens:", manifest["top8_tokens"])
    print("top-8 logits:", manifest["top8_logits"])
    print("argmax:", manifest["argmax"])
    print(f"saved full vectors + manifest to {out}/")


if __name__ == "__main__":
    main()
