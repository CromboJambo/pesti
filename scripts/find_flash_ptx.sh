#!/bin/bash
# Find and copy Flash Attention PTX after build
set -e

echo "=== Finding Flash Attention PTX ==="

cd /home/crombo/flash-attention

# Wait for build to complete (check if .so exists)
while [ ! -f "build/lib.linux-x86_64-cpython-314/flash_attn/*.so" ]; do
    echo "Waiting for build to complete..."
    sleep 5
done

echo "Build complete!"

# Find PTX files
PTX_FILES=$(find build -name "*.ptx" 2>/dev/null)

if [ -z "$PTX_FILES" ]; then
    echo "⚠️  No PTX files found in build directory"
    echo "Checking installed package..."
    
    # Try to find in site-packages
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    PTX_FILES=$(find $SITE_PACKAGES -name "*flash*attn*.ptx" 2>/dev/null)
fi

if [ -n "$PTX_FILES" ]; then
    echo ""
    echo "✅ Found PTX files:"
    echo "$PTX_FILES"
    echo ""
    echo "Best candidate (largest/most complete):"
    BEST_PTX=$(echo "$PTX_FILES" | xargs ls -S | head -1)
    echo "$BEST_PTX ($(ls -lh $BEST_PTX | awk '{print $5}'))"
    echo ""
    echo "Copy command:"
    echo "cp $BEST_PTX /home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/flash_attention_kernel.ptx"
else
    echo "❌ No PTX files found!"
    echo "Build may have completed without generating PTX."
fi
