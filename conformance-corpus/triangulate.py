#!/usr/bin/env python3
"""Triangulation: pesti-gguf vs an independent reference parse of the same file.

For a GGUF file, compares pesti-gguf's parsed output (via gguf-inspect --json)
field-by-field against a raw struct-based reference parse that follows the
python `gguf` package's algorithm (same GGML_QUANT_SIZES table, same layout
walk). Also validates physical geometry: data_section_start + sum of padded
tensor sizes must equal file size.

Reference parse covers:
  - version, counts, KV pairs (all value types incl. arrays)
  - tensor info: name, dims, dtype, offset
  - data section start (= end of tensor info)
  - per-tensor n_bytes via the ggml quant block-size table

Usage: triangulate.py <file.gguf> [--cli PATH]
Exit 0 = full agreement, 1 = discrepancies found.
"""
import json
import math
import os
import struct
import subprocess
import sys

import gguf as pgguf

CLI = "/home/crombo/projects/pesti/target/debug/gguf-inspect"
if "--cli" in sys.argv:
    i = sys.argv.index("--cli")
    CLI = sys.argv[i + 1]
PATH = [a for a in sys.argv[1:] if a != "--cli" and a != CLI][0]


# ---------------------------------------------------------------- pesti side
def pesti_parse(path: str) -> dict:
    r = subprocess.run([CLI, path, "--json"], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def norm_pesti_kv(val: dict):
    """pesti JSON KV value: {"String": "x"} / {"Uint32": 5} / {"Array": [...]}."""
    (tag, v), = val.items()
    if tag == "Array":
        return [norm_pesti_kv(x) for x in v]
    return v


# ---------------------------------------------------------------- reference side
# Raw parse following the python gguf package algorithm (v1/v2/v3 aware).
def _fmt(width: int, n: int = 1) -> str:
    base = "I" if width == 4 else "Q"
    return f"<{base * n}"


# GGUF scalar value types -> struct format (matches python gguf package).
SCALAR_FMT = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
    10: "<Q", 11: "<q", 12: "<d",
}


def rd_str(data: bytes, off: int, width: int):
    (n,) = struct.unpack_from(_fmt(width), data, off)
    off += width
    return data[off:off + n].decode("utf-8", "replace"), off + n


def rd_val(data: bytes, off: int, width: int):
    (t,) = struct.unpack_from("<I", data, off)
    off += 4
    if t == 8:  # string
        return rd_str(data, off, width)
    if t == 9:  # array: element type in header, elements follow directly
        (et,) = struct.unpack_from("<I", data, off)
        off += 4
        (cnt,) = struct.unpack_from("<Q", data, off)
        off += 8
        arr = []
        for _ in range(cnt):
            if et == 8:  # string element: no per-element type tag
                v, off = rd_str(data, off, width)
            elif et == 9:  # nested array
                v, off = rd_val(data, off, width)
            else:
                fmt = SCALAR_FMT[et]
                (v,) = struct.unpack_from(fmt, data, off)
                off += struct.calcsize(fmt)
            arr.append(v)
        return arr, off
    fmt = SCALAR_FMT[t]
    (v,) = struct.unpack_from(fmt, data, off)
    off += struct.calcsize(fmt)
    return v, off


def ref_parse(path: str):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:4] == b"GGUF", "bad magic"
    (ver,) = struct.unpack_from("<I", data, 4)
    w = 4 if ver == 1 else 8  # v1: u32 widths; v2/v3: u64
    (n_tensors, n_kv) = struct.unpack_from(_fmt(w, 2), data, 8)
    off = 8 + 2 * w

    kv = {}
    for _ in range(n_kv):
        k, off = rd_str(data, off, w)
        v, off = rd_val(data, off, w)
        kv[k] = v
    tensor_info_start = off

    tensors = []
    for _ in range(n_tensors):
        name, off = rd_str(data, off, w)
        (ndims,) = struct.unpack_from("<I", data, off)
        off += 4
        dims = list(struct.unpack_from(f"<{ndims}Q", data, off))
        off += ndims * 8
        (dtype,) = struct.unpack_from("<I", data, off)
        off += 4
        (toff,) = struct.unpack_from(_fmt(w), data, off)
        off += w
        tensors.append({"name": name, "dims": dims, "dtype": dtype, "offset": toff})
    raw_data_start = off

    # The writer aligns the data section start to general.alignment
    # (gguf-py gguf_reader.py: padding = offs % alignment; offs += alignment - padding).
    # Read the alignment from the KV we just parsed (default 32 if absent).
    align = kv.get("general.alignment", 32)
    padding = raw_data_start % align
    if padding:
        raw_data_start += align - padding
    data_section_start = raw_data_start

    # per-tensor n_bytes via the python gguf package's quant size table
    n_bytes = {}
    for t in tensors:
        qt = pgguf.GGMLQuantizationType(t["dtype"])
        n_elems = math.prod(t["dims"])
        block_size, type_size = pgguf.GGML_QUANT_SIZES[qt]
        n_bytes[t["name"]] = n_elems * type_size // block_size

    return {
        "version": ver,
        "kv": kv,
        "tensors": tensors,
        "data_section_start": data_section_start,
        "n_bytes": n_bytes,
    }


def norm_py_kv(v):
    if hasattr(v, "item"):
        return v.item()
    if isinstance(v, list):
        return [norm_py_kv(x) for x in v]
    return v


# ------------------------------------------------------------------- compare
def main() -> int:
    p = pesti_parse(PATH)
    r = ref_parse(PATH)
    fsize = os.path.getsize(PATH)

    fails, ok = [], []

    if p["version"] != r["version"]:
        fails.append(f"version: pesti={p['version']} ref={r['version']}")
    else:
        ok.append(f"version={p['version']}")

    if p["data_section_start"] != r["data_section_start"]:
        fails.append(
            f"data_section_start: pesti={p['data_section_start']} ref={r['data_section_start']}"
        )
    else:
        ok.append(f"data_section_start={r['data_section_start']}")

    # alignment: read general.alignment from reference KV
    ref_align = r["kv"].get("general.alignment", 32)
    if p["data_alignment"] != ref_align:
        fails.append(f"alignment: pesti={p['data_alignment']} ref={ref_align}")
    else:
        ok.append(f"alignment={ref_align}")

    # KV keys
    pesti_kv = {kv["key"]: norm_pesti_kv(kv["value"]) for kv in p["kv_pairs"]}
    only_pesti = set(pesti_kv) - set(r["kv"])
    only_ref = set(r["kv"]) - set(pesti_kv)
    if only_pesti:
        fails.append(f"KV only in pesti: {sorted(only_pesti)}")
    if only_ref:
        fails.append(f"KV only in reference: {sorted(only_ref)}")
    if not only_pesti and not only_ref:
        ok.append(f"kv keys: {len(r['kv'])} match")

    # KV values
    kv_mismatch = 0
    for k in sorted(set(pesti_kv) & set(r["kv"])):
        a, b = pesti_kv[k], norm_py_kv(r["kv"][k])
        if isinstance(a, float) and isinstance(b, float):
            same = math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
        else:
            same = a == b
        if not same:
            kv_mismatch += 1
            if kv_mismatch <= 5:
                fails.append(f"KV {k}: pesti={str(a)[:80]} ref={str(b)[:80]}")
    if kv_mismatch:
        fails.append(f"KV value mismatches: {kv_mismatch} total")
    else:
        ok.append(f"kv values: all {len(set(pesti_kv) & set(r['kv']))} match")

    # Tensors
    if len(p["tensors"]) != len(r["tensors"]):
        fails.append(f"tensor count: pesti={len(p['tensors'])} ref={len(r['tensors'])}")
    stored = {t["name"]: t["stored_size"] for t in p.get("tensors_stored_size", [])}
    t_mismatch = 0
    for i, (pt, rt) in enumerate(zip(p["tensors"], r["tensors"])):
        if pt["name"] != rt["name"]:
            fails.append(f"tensor[{i}] name: pesti={pt['name']} ref={rt['name']}")
            t_mismatch += 1
            continue
        if list(pt["shape"]) != rt["dims"]:
            fails.append(f"tensor {pt['name']} shape: pesti={pt['shape']} ref={rt['dims']}")
            t_mismatch += 1
        if pt["dtype"] != rt["dtype"]:
            fails.append(f"tensor {pt['name']} dtype: pesti={pt['dtype']} ref={rt['dtype']}")
            t_mismatch += 1
        if pt["offset"] != rt["offset"]:
            fails.append(f"tensor {pt['name']} offset: pesti={pt['offset']} ref={rt['offset']}")
            t_mismatch += 1
        ss = stored.get(pt["name"])
        if ss is None:
            fails.append(f"tensor {pt['name']}: pesti stored_size() errored")
            t_mismatch += 1
        elif ss != r["n_bytes"][pt["name"]]:
            fails.append(
                f"tensor {pt['name']} stored_size: pesti={ss} ref={r['n_bytes'][pt['name']]}"
            )
            t_mismatch += 1
    if t_mismatch:
        fails.append(f"tensor mismatches: {t_mismatch} total")
    else:
        ok.append(f"tensors: {len(r['tensors'])} name/shape/dtype/offset/stored_size match")

    # Geometry: pesti's padded sum vs actual file size
    align = p["data_alignment"] or 1
    dss = r["data_section_start"]
    padded = 0
    for t in r["tensors"]:
        ss = r["n_bytes"][t["name"]]
        padded += (ss + align - 1) // align * align
    delta = padded - (fsize - dss)
    if delta != 0:
        fails.append(f"geometry: data-section delta {delta} (padded={padded}, actual={fsize - dss})")
    else:
        ok.append(f"geometry: zero delta over {fsize - dss} data bytes")

    print(f"== {os.path.basename(PATH)}")
    for line in ok:
        print(f"  ok   {line}")
    if fails:
        for line in fails:
            print(f"  FAIL {line}")
        return 1
    print("  PASS: pesti-gguf and reference agree on every field")
    return 0


if __name__ == "__main__":
    sys.exit(main())
