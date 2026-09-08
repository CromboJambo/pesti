#!/usr/bin/env python3
"""Knowledge-state oracle: classify each layer by epistemic status rather than binary pass/fail."""
import re
import sys
import json


def parse(path):
    layers = {}
    prehead = {}
    summary = {}

    layer_re = re.compile(r"\[(?:P|REF)\]\s+layer=(\d+)\s+pos=\d+\s+norm=([-\d.eE+]+)\s+head=\[([^\]]*)\]")
    prehead_re = re.compile(r"\[(?:P|REF)\]\s+pre-head\s+norm=([-\d.eE+]+)\s+head=\[([^\]]*)\]")
    tokens_re = re.compile(r"top-8 tokens:\s*\[([^\]]*)\]")
    logits_re = re.compile(r"top-8 logits:\s*\[([^\]]*)\]")
    argmax_re = re.compile(r"argmax:\s*(\d+)")

    with open(path) as f:
        content = f.read()
    
    print(f"DEBUG: parsed {len(content)} chars from {path}", file=sys.stderr)
    
    for line in content.split('\n'):
            m = layer_re.search(line)
            if m:
                layers[int(m.group(1))] = (float(m.group(2)), [float(x) for x in m.group(3).split(",")])
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


def classify_layer(layer, rust_norm, ref_norm, rust_head, ref_head, tol):
    dnorm = abs(rust_norm - ref_norm)
    dh = head_delta(rust_head, ref_head)

    margin_norm = 1.0 - (dnorm / tol) if tol > 0 else 1.0
    margin_head = 1.0 - (dh / tol) if tol > 0 else 1.0
    confidence = max(0.0, min(1.0, (margin_norm + margin_head) / 2.0))

    if dnorm / tol >= dh / tol:
        binding = f"layer_norm_delta ({dnorm:.2e})"
    else:
        binding = f"head_max_abs_delta ({dh:.2e})"

    if confidence > 0.7:
        status = "established"
        rec = ""
    elif confidence > 0.3:
        status = "uncertain"
        rec = f"watch {binding} on next change"
    else:
        status = "degraded"
        if dnorm > tol:
            rec = f"investigate layer norm divergence at layer {layer}"
        else:
            rec = f"investigate head divergence at layer {layer}"

    return {
        "layer": layer, "status": status, "norm_delta": dnorm, "head_delta": dh,
        "binding_constraint": binding, "confidence": confidence, "recommendation": rec
    }


def main():
    print(f"DEBUG: sys.argv = {sys.argv}", file=sys.stderr)
    
    if len(sys.argv) < 3:
        print("Usage: oracle_knowledge.py <rust_output.txt> <ref_output.txt> [--tol X]")
        sys.exit(2)

    tol = 1e-3
    i = 0
    positional = []
    while i < len(sys.argv[1:]):
        arg = sys.argv[i+1]  # Fix: access sys.argv[i+1], not sys.argv[i]
        if arg == "--tol":
            i += 1
            tol = float(sys.argv[i+1])
        elif arg.startswith("--tol="):
            tol = float(arg.split("=", 1)[1])
        else:
            positional.append(arg)
        i += 1

    if len(positional) != 2:
        print("Need exactly two output files")
        sys.exit(2)

    rust_path, ref_path = positional[0], positional[1]
    
    # Verify we're reading actual output files, not the script itself
    for p in [rust_path, ref_path]:
        if not p.startswith('/tmp/') and p != 'oracle_knowledge.py':
            print(f"DEBUG: checking path {p}", file=sys.stderr)
    
    rust = parse(rust_path)
    ref = parse(ref_path)

    rl = rust["layers"]
    fl = ref["layers"]
    n_layers = max(len(rl), len(fl))

    print(f"Knowledge-state oracle (tol={tol:.1e})")
    print("=" * 80)

    established = []
    uncertain = []
    degraded = []

    for i in range(n_layers):
        if i not in rl or i not in fl:
            degraded.append({
                "layer": i, "status": "degraded", "norm_delta": float("nan"),
                "head_delta": float("nan"), "binding_constraint": "missing from one output",
                "confidence": 0.0, "recommendation": ""
            })
            continue

        rn, rh = rl[i]
        fn, fh = fl[i]
        k = classify_layer(i, rn, fn, rh, fh, tol)

        if k["status"] == "established":
            established.append(k)
        elif k["status"] == "uncertain":
            uncertain.append(k)
        else:
            degraded.append(k)

    print(f"\nKnowledge state across {n_layers} layers:")
    print(f"  Established: {len(established)} ({100*len(established)/n_layers:.0f}%)")
    print(f"  Uncertain:   {len(uncertain)} ({100*len(uncertain)/n_layers:.0f}%)")
    print(f"  Degraded:    {len(degraded)} ({100*len(degraded)/n_layers:.0f}%)")

    total_conf = sum(k["confidence"] for k in established + uncertain + degraded) / n_layers
    print(f"\nOverall confidence: {total_conf:.3f}")

    if uncertain or degraded:
        tightest = min(uncertain + degraded, key=lambda k: k["confidence"])
        print(f"Tightest constraint: layer {tightest['layer']} - {tightest['binding_constraint']}")
        if tightest["recommendation"]:
            print(f"  -> {tightest['recommendation']}")

    if uncertain:
        print("\nUncertain layers (watch on next change):")
        for k in uncertain:
            print(f"  L{k['layer']}: normDelta={k['norm_delta']:.2e} headDelta={k['head_delta']:.2e} [{k['binding_constraint']}]")

    if degraded:
        print("\nDegraded layers (need attention):")
        for k in degraded:
            print(f"  L{k['layer']}: normDelta={k['norm_delta']:.2e} headDelta={k['head_delta']:.2e} [{k['binding_constraint']}]")
            if k["recommendation"]:
                print(f"    -> {k['recommendation']}")

    print("\nNext experiments to increase knowledge:")
    if uncertain or degraded:
        worst = min(uncertain + degraded, key=lambda k: k["confidence"])
        print(f"  - Stress test layer {worst['layer']} with longer sequences")
        print(f"  - Check accumulation order at layer {worst['layer']}")
    else:
        print("  - Increase sequence length to stress rope/attention")
        print("  - Test edge-case inputs (very long context)")

    result = {
        "n_layers": n_layers,
        "established": len(established),
        "uncertain": len(uncertain),
        "degraded": len(degraded),
        "confidence": total_conf,
        "layers": [
            {"layer": k["layer"], "status": k["status"], "confidence": k["confidence"],
             "binding": k["binding_constraint"], "recommendation": k["recommendation"]}
            for k in (established + uncertain + degraded)
        ]
    }

    print(f"\n--- JSON ---")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
