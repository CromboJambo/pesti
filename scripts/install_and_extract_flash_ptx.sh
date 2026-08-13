#!/bin/bash
# Install flash-attn via pip and extract PTX
set -e

echo "=== Installing flash-attn via pip ==="

# First install torch (required dependency)
pip install --break-system-packages torch psutil ninja packaging

# Then install flash-attn (with build isolation disabled to use system torch)
pip install --break-system-packages --no-build-isolation flash-attn

echo ""
echo "=== Finding compiled PTX ==="

# Find the installed package location
PACKAGE_PATH=$(python3 -c "import flash_attn; import os; print(os.path.dirname(flash_attn.__file__))")
echo "Package location: $PACKAGE_PATH"

# Look for .so files (may contain embedded PTX)
SO_FILES=$(find $PACKAGE_PATH -name "*.so" 2>/dev/null)

if [ -n "$SO_FILES" ]; then
    echo ""
    echo "✅ Found compiled libraries:"
    echo "$SO_FILES"
    echo ""
    
    # Try to extract PTX from the .so
    for SO in $SO_FILES; do
        echo "Extracting PTX from: $SO"
        OUTPUT="/tmp/flash_attn_$(basename $SO .so).ptx"
        cuobjdump -ptx "$SO" > "$OUTPUT" 2>/dev/null || echo "  ⚠️  No PTX found in this file"
        
        if [ -s "$OUTPUT" ]; then
            echo "  ✅ Extracted $(wc -l < $OUTPUT) lines of PTX to $OUTPUT"
        fi
    done
    
    # Find the largest PTX file (likely the main kernel)
    LARGEST_PTX=$(find /tmp -name "*.ptx" -type f -exec ls -S {} + 2>/dev/null | head -1)
    
    if [ -n "$LARGEST_PTX" ]; then
        echo ""
        echo "=== Best PTX Found ==="
        echo "File: $LARGEST_PTX"
        echo "Size: $(ls -lh $LARGEST_PTX | awk '{print $5}')"
        echo "Lines: $(wc -l < $LARGEST_PTX)"
        echo ""
        echo "Copy command:"
        echo "cp $LARGEST_PTX /home/crombo/projects/pesti/pesti-runner/src/kernel/ptx/flash_attention_kernel.ptx"
    else
        echo "⚠️  No usable PTX found in .so files"
        echo "The flash-attn package may use Triton JIT compilation instead of pre-compiled PTX"
    fi
else
    echo "❌ No .so files found!"
fi

echo ""
echo "=== Done ==="
