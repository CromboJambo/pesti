#!/usr/bin/env python3
"""Compare PESTI dequantized weights against llama.cpp reference."""

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def load_peasti_weights(gguf_path: str) -> Dict[str, np.ndarray]:
    """Load dequantized weights from PESTI (placeholder - implement later)."""
    # For now, just return empty dict
    # This will be filled in when we have actual PESTI dequantization output
    print(f"Loading PESTI weights from: {gguf_path}")
    return {}

def load_reference_weights(ref_dir: str) -> Dict[str, np.ndarray]:
    """Load reference dequantized weights from llama.cpp."""
    ref_path = Path(ref_dir)
    weights = {}
    
    for npy_file in ref_path.glob("*.npy"):
        weights[npy_file.stem] = np.load(npy_file)
        print(f"  Loaded: {npy_file.name} ({weights[npy_file.stem].nbytes:,} bytes)")
    
    return weights

def compare_weights(peasti: Dict[str, np.ndarray], reference: Dict[str, np.ndarray], 
                   tolerance: float = 1e-6) -> Tuple[int, int, list]:
    """Compare PESTI weights against reference."""
    passed = 0
    failed = 0
    failures = []
    
    for name, ref_data in reference.items():
        if name not in pesti:
            print(f"⚠️  Missing in PESTI: {name}")
            continue
        
        pesti_data = pesti[name]
        
        if pesti_data.shape != ref_data.shape:
            print(f"❌ Shape mismatch: {name} (PESTI={pesti_data.shape}, Ref={ref_data.shape})")
            failures.append((name, "shape", pesti_data.shape, ref_data.shape))
            failed += 1
            continue
        
        diff = np.abs(peasti_data - ref_data).max()
        if diff <= tolerance:
            passed += 1
            print(f"✅ {name}: max_diff={diff:.2e}")
        else:
            failures.append((name, "value", diff, tolerance))
            failed += 1
            print(f"❌ {name}: max_diff={diff:.6f} (tolerance={tolerance})")
    
    return passed, failed, failures

def main():
    parser = argparse.ArgumentParser(description="Compare PESTI vs llama.cpp dequantization")
    parser.add_argument("--peasti-weights", "-p", help="PESTI weight output (JSON/npz)")
    parser.add_argument("--reference-dir", "-r", required=True, 
                       help="Directory with llama.cpp reference weights")
    parser.add_argument("--tolerance", "-t", type=float, default=1e-6,
                       help="Tolerance for comparison (default: 1e-6)")
    
    args = parser.parse_args()
    
    print(f"🔍 Loading reference weights from: {args.reference_dir}")
    reference = load_reference_weights(args.reference_dir)
    print(f"\n✅ Loaded {len(reference)} reference tensors\n")
    
    if args.peasti_weights:
        print(f"📦 Loading PESTI weights from: {args.peasti_weights}")
        pesti = load_peasti_weights(args.peasti_weights)
    else:
        print("⚠️  No PESTI weights provided - skipping comparison")
        pesti = {}
    
    if pesti:
        print(f"\n🧪 Comparing {len(pesti)} tensors...")
        passed, failed, failures = compare_weights(peasti, reference, args.tolerance)
        
        print(f"\n{'='*60}")
        print(f"Results: {passed} passed, {failed} failed")
        
        if failed > 0:
            print(f"\nFailures:")
            for name, issue, val, tol in failures[:5]:
                print(f"  - {name}: {issue} ({val:.6f} vs {tol})")
    else:
        print("\n⚠️  No comparison performed (no PESTI weights)")

if __name__ == "__main__":
    main()
