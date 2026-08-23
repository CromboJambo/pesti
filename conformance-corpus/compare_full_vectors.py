#!/usr/bin/env python3
"""Full-vector conformance diff: compare pesti's dumped full per-layer vectors
(raw f32 files from `dump_all_layers --dump`) against the numpy probe's saved
arrays (`probe_all_layers.py`), comparing ALL 896 dims per layer (not just
head-8).

This is the gold-standard check: head-8 + norm + top-8 + argmax conformance is
necessary but not sufficient; a bug could hide in dims 8..895. This tool
catches that.

Usage:
    python3 compare_full_vectors.py <rust_dump_dir> <numpy_probe_dir> [--tol 1e-3]

  rust_dump_dir  : dir with embed.f32, layer_<l>.f32, prehead.f32, logits.f32
                   (raw little-endian f32, no header)
  numpy_probe_dir: dir with embed.f32.npy, layer_<l>.f32.npy, prehead.f32.npy,
                   logits.f32.npy (from probe_all_layers.py)

Exit code 0 = all layers within tolerance, 1 = divergence beyond tolerance.
"""
import json
import sys

import numpy as np


def load_raw(path):
    """Load a raw little-endian f32 file (no header)."""
    return np.fromfile(path, dtype="<f4")


def load_npy(path):
    # probe_all_layers.py uses np.save(), which appends .npy and writes a header.
    return np.asarray(np.load(path), dtype=np.float32)


def cmp_vec(name, rust, ref, tol):
    d = np.abs(rust - ref)
    maxd = float(d.max())
    meand = float(d.mean())
    nr = float(np.linalg.norm(rust) / np.linalg.norm(ref))
    corr = float(np.corrcoef(rust, ref)[0, 1]) if rust.size > 1 else 1.0
    ok = maxd <= tol
    flag = "OK " if ok else "FAIL"
    print(f"{flag} {name:8s} n={rust.size:<6} max|d|={maxd:.3e} mean|d|={meand:.3e} "
          f"normratio={nr:.6f} corr={corr:.6f}")
    return ok


def parse_args(argv):
    """Parse [rust_dir numpy_dir --tol X] (tol also accepted as --tol=X)."""
    tol = 1e-3
    positional = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--tol":
            i += 1
            tol = float(argv[i])
        elif a.startswith("--tol="):
            tol = float(a.split("=", 1)[1])
        else:
            positional.append(a)
        i += 1
    return positional, tol


def main():
    args, tol = parse_args(sys.argv[1:])

    if len(args) != 2:
        print(__doc__)
        sys.exit(2)
    rust_dir, np_dir = args[0], args[1]

    # Read numpy manifest to learn n_layer.
    with open(f"{np_dir}/manifest.json") as f:
        man = json.load(f)
    n_layer = man["n_layer"]

    print(f"Full-vector conformance: {n_layer} layers, tol={tol:.1e}")
    print(f"  rust: {rust_dir}")
    print(f"  numpy: {np_dir}")
    print("-" * 78)

    all_ok = True

    # embed
    all_ok &= cmp_vec("embed", load_raw(f"{rust_dir}/embed.f32"),
                      load_npy(f"{np_dir}/embed.f32.npy"), tol)

    # per-layer full vectors
    for l in range(n_layer):
        all_ok &= cmp_vec(f"layer_{l}", load_raw(f"{rust_dir}/layer_{l}.f32"),
                          load_npy(f"{np_dir}/layer_{l}.f32.npy"), tol)

    # pre-head
    all_ok &= cmp_vec("prehead", load_raw(f"{rust_dir}/prehead.f32"),
                      load_npy(f"{np_dir}/prehead.f32.npy"), tol)

    # logits (full vocab)
    all_ok &= cmp_vec("logits", load_raw(f"{rust_dir}/logits.f32"),
                      load_npy(f"{np_dir}/logits.f32.npy"), max(tol, 0.05))

    print("-" * 78)
    if all_ok:
        print(f"=== VERDICT: PASS — all {n_layer} layers + prehead + logits within tol ===")
        sys.exit(0)
    else:
        print("=== VERDICT: FAIL — at least one vector exceeds tolerance ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
