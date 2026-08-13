#!/bin/bash
# Compile Flash Attention forward kernel to PTX for sm_8.9
set -e

echo "=== Compiling Flash Attention PTX ==="

cd /home/crombo/flash-attention/csrc/flash_attn/src

# Choose a head dimension (64 is common for small models)
HDIM=128
PRECISION="fp16"
CAUSAL=""

echo "Compiling flash_fwd_hdim${HDIM}_${PRECISION}${CAUSAL}_sm80.cu..."
echo "Target: sm_89 (RTX 4070 Ti SUPER)"
echo ""

# Compile to PTX
nvcc \
    -arch=sm_89 \
    -ptx \
    -O3 \
    -std=c++17 \
    flash_fwd_hdim${HDIM}_${PRECISION}${CAUSAL}_sm80.cu \
    -o /home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/flash_attention_kernel.ptx

echo ""
echo "✅ PTX compiled successfully!"
echo "Output: /home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/flash_attention_kernel.ptx"
echo ""
echo "Next steps:"
echo "1. Verify the new PTX loads correctly"
echo "2. Run conformance test to check numerical output"
echo "3. Benchmark speedup vs baseline"
