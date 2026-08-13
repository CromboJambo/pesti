#!/usr/bin/env python3
"""Generate reference dequantized weights from GGUF files using llama-cpp-python."""

import argparse
import numpy as np
from pathlib import Path
try:
    from llama_cpp import Llama
except ImportError:
    print("Installing llama-cpp-python...")
    import subprocess
    subprocess.check_call(["pip", "install", "llama-cpp-python"])
    from llama_cpp import Llama

def extract_weights(model_path: str, output_dir: str):
    """Extract dequantized weights from a GGUF model."""
    
    # Load model (CPU only for now)
    print(f"Loading model: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=1,  # Minimal context
        n_threads=1,  # Single thread for deterministic output
        verbose=False
    )
    
    # Get all weights
    weights = llm.llama_model._model.weights
    
    # Save each weight as a separate .npy file
    output_path = Path(output_dir) / Path(model_path).stem
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, data in weights.items():
        npy_file = output_path / f"{name}.npy"
        np.save(npy_file, data)
        print(f"  Saved: {npy_file.name} ({data.nbytes:,} bytes)")
    
    print(f"\n✅ All weights saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate reference dequantized weights")
    parser.add_argument("model_path", help="Path to GGUF model file")
    parser.add_argument("--output-dir", "-o", default="./conformance-corpus/references",
                       help="Output directory for reference weights")
    
    args = parser.parse_args()
    
    extract_weights(args.model_path, args.output_dir)

if __name__ == "__main__":
    main()
