#!/bin/bash
# Clone and prepare flash-attention for compilation
set -e

echo "=== Cloning Flash Attention Repository ==="

cd /home/crombo

# Clone the official flash-attention repo
if [ ! -d "flash-attention" ]; then
    echo "Cloning https://github.com/Dao-AILab/flash-attention.git..."
    git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
else
    echo "✓ flash-attention already exists, pulling latest..."
    cd flash-attention && git pull && cd ..
fi

echo ""
echo "=== Repository Ready ==="
echo "Location: /home/crombo/flash-attention"
echo ""
echo "Next steps:"
echo "1. Compile PTX for sm_8.9 (RTX 4070 Ti SUPER)"
echo "   cd flash-attention && python setup.py install --cpp_ext --cuda_ext"
echo ""
echo "2. Or compile just the PTX:"
echo "   cd flash-attention/csrc/flash_attn"
echo "   nvcc -arch=sm_89 -ptx flash_attention.cu -o flash_attention_kernel.ptx"
