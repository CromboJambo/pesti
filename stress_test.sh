#!/bin/bash

cd /home/crombo/projects/pesti

echo "=== REAL STRESS TEST - Larger Models ==="
echo "GPU: RTX 4070 Ti SUPER (sm_8.9)"
echo "Baseline llama.cpp CPU: ~85 tok/s (0.5B)"
echo ""

# Test TinyLlama first (1.1B params)
MODEL="test_models/tinyllama-q8.gguf"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    exit 1
fi

echo "📦 Testing: tinyllama-q8.gguf (~1.1B params, 1.1 GB)"
echo ""

output=$(/home/crombo/.cargo/bin/cargo run --package pesti-runner \
    --example llama_gpu_vs_cpu \
    --features cuda,mistralrs \
    -- -m "$MODEL" \
    -n 64 \
    2>&1)

echo "$output" | grep -E "(Throughput|Generated|Model:)" | head -5

if echo "$output" | grep -q "Throughput:"; then
    tok_per_sec=$(echo "$output" | grep "Throughput:" | sed 's/.*: //' | tr -d ' ')
    echo ""
    echo "✅ Throughput: $tok_per_sec tok/s"
    
    # Calculate gap vs baseline
    gap=$(awk "BEGIN {printf \"%.1f\", ((85.0 - $tok_per_sec) / 85.0) * 100}")
    echo "   Gap vs ~85 tok/s baseline: ${gap}%"
fi

echo ""
echo "============================================================"
echo ""

# Test Llama 3.1 8B (THE REAL TEST!)
MODEL="test_models/llama3.1-8b-q4_k_m.gguf"

if [ ! -f "$MODEL" ]; then
    echo "Model not found: $MODEL"
    exit 1
fi

echo "📦 Testing: llama3.1-8b-q4_k_m.gguf (~8B params, 4.6 GB)"
echo "   This is the REAL stress test!"
echo ""

output=$(/home/crombo/.cargo/bin/cargo run --package pesti-runner \
    --example llama_gpu_vs_cpu \
    --features cuda,mistralrs \
    -- -m "$MODEL" \
    -n 64 \
    2>&1)

echo "$output" | tail -20

if echo "$output" | grep -q "Throughput:"; then
    tok_per_sec=$(echo "$output" | grep "Throughput:" | sed 's/.*: //' | tr -d ' ')
    echo ""
    echo "🎯 Throughput: $tok_per_sec tok/s"
    
    # Calculate gap vs baseline (estimate for 8B)
    gap=$(awk "BEGIN {printf \"%.1f\", ((40.0 - $tok_per_sec) / 40.0) * 100}")
    echo "   Gap vs ~40 tok/s target: ${gap}%"
    
    if (( $(echo "$tok_per_sec >= 35" | bc -l 2>/dev/null || echo 1) )); then
        echo ""
        echo "✅ EXCELLENT! GPU handling 8B model well!"
    elif (( $(echo "$tok_per_sec >= 25" | bc -l 2>/dev/null || echo 0) )); then
        echo ""
        echo "⚠️ Decent performance for 8B model"
    else
        echo ""
        echo "🔄 Below expectations - may need optimization"
    fi
else
    echo ""
    echo "⚠️ Could not measure throughput"
fi
