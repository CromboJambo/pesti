#!/usr/bin/env python3
"""Compare pesti's all-layer dumper output against the numpy reference
(`ref_forward.py`) to verify 24-layer conformance.

Both the Rust dumper (`dump_all_layers.rs`) and the numpy oracle emit the same
line grammar (norm + first-8 head per layer, pre-head, top-8 tokens/logits,
argmax). This tool parses both, aligns them by layer, and reports:
  - per-layer norm delta (Rust - ref)
  - per-layer head max-abs-delta (first 8 dims)
  - pre-head norm + head delta
  - top-8 token-id match and logit deltas
  - argmax match
  - a PASS/FAIL verdict against a tolerance

Usage:
    python3 compare_all_layers.py <rust_output.txt> <ref_output.txt> [--tol 1e-3]

Exit code 0 = conformance within tolerance, 1 = divergence beyond tolerance.

Tolerance note: the two paths dequantize Q4_K independently (pesti's Rust
dequant vs gguf's numpy dequant) and accumulate in different orders, so a
sub-1e-3 absolute delta on f32 hidden states is expected and acceptable. The
meaningful conformance signal is (a) all 24 layer norms tracking to ~1e-3,
(b) top-8 token ids identical, and (c) argmax identical.
"""
import re
import sys


def parse(path):
    """Extract per-layer norms/heads + summary from a dumper/oracle output file."""
    layers = {}
    prehead = {}
    summary = {}

    # Matches: [P] layer=0 pos=9 norm=3.8732 head=[-0.0277,...]
    #          [REF] layer=0 pos=9 norm=3.8731 head=[-0.0280,...]
    layer_re = re.compile(
        r"\[(?:P|REF)\]\s+layer=(\d+)\s+pos=\d+\s+norm=([-\d.eE+]+)\s+head=\[([^\]]*)\]"
    )
    prehead_re = re.compile(
        r"\[(?:P|REF)\]\s+pre-head\s+norm=([-\d.eE+]+)\s+head=\[([^\]]*)\]"
    )
    tokens_re = re.compile(r"top-8 tokens:\s*\[([^\]]*)\]")
    logits_re = re.compile(r"top-8 logits:\s*\[([^\]]*)\]")
    argmax_re = re.compile(r"argmax:\s*(\d+)")

    with open(path) as f:
        for line in f:
            m = layer_re.search(line)
            if m:
                layers[int(m.group(1))] = (
                    float(m.group(2)),
                    [float(x) for x in m.group(3).split(",")],
                )
                continue
            m = prehead_re.search(line)
            if m:
                prehead = (float(m.group(1)), [float(x) for x in m.group(2).split(",")])
                continue
            m = tokens_re.search(line)
            if m:
                summary["tokens"] = [int(x) for x in m.group(1).split(",")]
                continue
            m = logits_re.search(line)
            if m:
                summary["logits"] = [float(x) for x in m.group(1).split(",")]
                continue
            m = argmax_re.search(line)
            if m:
                summary["argmax"] = int(m.group(1))
                continue

    return {"layers": layers, "prehead": prehead, "summary": summary}


def head_delta(a, b):
    n = min(len(a), len(b))
    return max(abs(a[i] - b[i]) for i in range(n)) if n else float("nan")


def parse_args(argv):
    """Parse [rust.txt ref.txt --tol X] (tol also accepted as --tol=X)."""
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

    rust = parse(args[0])
    ref = parse(args[1])

    rl = rust["layers"]
    fl = ref["layers"]
    n_layers = max(len(rl), len(fl))

    print(f"{'layer':>6} | {'rust_norm':>10} {'ref_norm':>10} {'d_norm':>9} | {'max|dhead|':>11}")
    print("-" * 62)

    max_dnorm = 0.0
    max_dhead = 0.0
    missing = []
    for i in range(n_layers):
        if i not in rl or i not in fl:
            missing.append(i)
            continue
        rn, rh = rl[i]
        fn, fh = fl[i]
        dnorm = abs(rn - fn)
        dh = head_delta(rh, fh)
        max_dnorm = max(max_dnorm, dnorm)
        max_dhead = max(max_dhead, dh)
        print(f"{i:>6} | {rn:>10.4f} {fn:>10.4f} {dnorm:>9.2e} | {dh:>11.2e}")

    # pre-head
    print("-" * 62)
    verdict = "PASS"
    problems = []

    if missing:
        problems.append(f"missing layers in one output: {missing}")

    if max_dnorm > tol:
        verdict = "FAIL"
        problems.append(f"max layer-norm delta {max_dnorm:.2e} > tol {tol:.1e}")
    if max_dhead > tol:
        verdict = "FAIL"
        problems.append(f"max layer-head delta {max_dhead:.2e} > tol {tol:.1e}")

    # pre-head comparison
    if rust["prehead"] and ref["prehead"]:
        rn, rh = rust["prehead"]
        fn, fh = ref["prehead"]
        pdn = abs(rn - fn)
        pdh = head_delta(rh, fh)
        print(f"{'prehead':>6} | {rn:>10.4f} {fn:>10.4f} {pdn:>9.2e} | {pdh:>11.2e}")
        if pdn > tol or pdh > tol:
            verdict = "FAIL"
            problems.append(f"pre-head delta norm={pdn:.2e} head={pdh:.2e} > tol")
    else:
        problems.append("pre-head missing in one output")

    # summary: tokens, logits, argmax
    rs, fs = rust["summary"], ref["summary"]
    print()
    rt, ft = rs.get("tokens"), fs.get("tokens")
    if rt is not None and ft is not None:
        tok_match = rt == ft
        print(f"top-8 tokens rust: {rt}")
        print(f"top-8 tokens ref : {ft}")
        print(f"top-8 tokens match: {tok_match}")
        if not tok_match:
            verdict = "FAIL"
            problems.append("top-8 token ids differ")
    else:
        problems.append("top-8 tokens missing in one output")

    rlg, flg = rs.get("logits"), fs.get("logits")
    if rlg is not None and flg is not None:
        max_dlogit = max(abs(rlg[i] - flg[i]) for i in range(min(len(rlg), len(flg))))
        print(f"top-8 logits rust: {['%.3f' % x for x in rlg]}")
        print(f"top-8 logits ref : {['%.3f' % x for x in flg]}")
        print(f"max top-8 logit delta: {max_dlogit:.4f}")
        if max_dlogit > 0.05:
            verdict = "FAIL"
            problems.append(f"max top-8 logit delta {max_dlogit:.4f} > 0.05")
    else:
        problems.append("top-8 logits missing in one output")

    ra, fa = rs.get("argmax"), fs.get("argmax")
    if ra is not None and fa is not None:
        am_match = ra == fa
        print(f"argmax rust={ra} ref={fa} match={am_match}")
        if not am_match:
            verdict = "FAIL"
            problems.append(f"argmax differs: rust={ra} ref={fa}")
    else:
        problems.append("argmax missing in one output")

    print()
    print(f"=== VERDICT: {verdict} (tol={tol:.1e}) ===")
    if problems:
        for p in problems:
            print(f"  - {p}")
    else:
        print(
            f"  all {n_layers} layers + pre-head + top-8 + argmax within tolerance."
        )

    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
