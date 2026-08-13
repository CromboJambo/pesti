#!/bin/bash
# Build flash-attention with Python setup
set -e

echo "=== Building Flash Attention with pip ==="

cd /home/crombo/flash-attention

# Install dependencies if needed (Arch Linux workaround)
pip install --break-system-packages torch ninja packaging

# Build and install the CUDA extension
echo ""
echo "Building CUDA extension..."
python setup.py install

echo ""
echo "✅ Flash Attention built!"
echo ""
echo "PTX files should be in:"
echo "  flash_attn/csrc/flash_attn/build/*/lib/*.ptx"
echo ""
echo "Next: Find and copy the PTX to pesti-runner"
