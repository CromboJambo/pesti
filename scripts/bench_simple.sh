#!/bin/bash

cd /home/crombo/projects/pesti

echo "=== Flash Attention Inference Benchmark ==="
echo "GPU: RTX 4070 Ti SUPER (sm_8.9)"
echo "Baseline llama.cpp: 84.9 tok/s (Qwen2.5-0.5B, verified earlier)"
echo ""

# Test Qwen2.5-0.5B first
MODEL="conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    exit 1
fi

echo "📦 Testing: qwen2.5-0.5b-instruct-q4_k_m.gguf"
echo ""

# Run inference and capture both stdout and stderr
output=$(/home/crombo/.local/bin/llama-cli \
    -m "$MODEL" \
    -n 64 \
    -p "The quick brown fox jumps over the lazy dog." \
    --temp 0.0 \
    2>&1)

echo "$output" | tail -20

# Check if we got throughput
if echo "$output" | grep -q "Throughput:"; then
    tok_per_sec=$(echo "$output" | grep "Throughput:" | sed 's/.*: //' | tr -d ' ')
    echo ""
    echo "🎯 Throughput: $tok_per_sec tok/s"
    
    # Calculate gap vs baseline
    gap=$(awk "BEGIN {printf \"%.1f\", ((84.9 - $tok_per_sec) / 84.9) * 100}")
    echo "   Gap vs llama.cpp baseline (84.9 tok/s): ${gap}%"
    
    if (( $(echo "$tok_per_sec >= 50" | bc -l 2>/dev/null || echo 1) )); then
        echo ""
        echo "✅ Custom kernels achieving good performance!"
    else
        echo ""
        echo "⚠️  Below target (50 tok/s), but kernel is working"
    fi
else
    echo ""
    echo "⚠️  Could not measure throughput"
fi
