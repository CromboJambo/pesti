#!/bin/bash
set -e

cd /home/crombo/projects/pesti

echo "=== Flash Attention Inference Benchmark ==="
echo "GPU: RTX 4070 Ti SUPER (sm_8.9)"
echo "Baseline llama.cpp: 84.9 tok/s (Qwen2.5-0.5B, verified earlier)"
echo ""

MODELS=(
    "conformance-corpus/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    "conformance-corpus/qwen2.5-0.5b-instruct-q8_0.gguf"
    "conformance-corpus/qwen2.5-3b-instruct-q4_k_m.gguf"
)

PROMPT="The quick brown fox jumps over the lazy dog."
NUM_TOKENS=64

for model in "${MODELS[@]}"; do
    if [ ! -f "$model" ]; then
        echo "⊘ Skipping $model (not found)"
        continue
    fi
    
    file_size=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model")
    size_gb=$(echo "scale=2; $file_size / 1073741824" | bc)
    
    echo ""
    echo "📦 Testing: $(basename $model) (${size_gb} GB)"
    echo "   Running: llama-cli -m $model -n $NUM_TOKENS -p '$PROMPT' --temp 0.0"
    
    start_time=$(date +%s.%N)
    
    # Run inference and capture output
    output=$(/home/crombo/.local/bin/llama-cli         -m "$model"         -n $NUM_TOKENS         -p "$PROMPT"         --temp 0.0         2>&1)
    
    end_time=$(date +%s.%N)
    elapsed=$(echo "$end_time - $start_time" | bc)
    
    # Extract tokens/sec from output
    if echo "$output" | grep -q "Throughput:"; then
        tok_per_sec=$(echo "$output" | grep "Throughput:" | sed 's/.*: //' | tr -d ' ')
        echo "✅ Throughput: $tok_per_sec tok/s"
        
        # Calculate gap vs baseline (84.9 for 0.5B)
        if [[ $model == *"0.5b"* ]]; then
            gap=$(echo "scale=1; (($baseline - $tok_per_sec) / $baseline) * 100" | bc)
            echo "   Gap vs llama.cpp baseline (84.9 tok/s): ${gap}%"
        fi
        
        echo "$model|$tok_per_sec|$size_gb|$elapsed" >> /tmp/benchmark_results.csv
    else
        echo "⚠️  Could not measure throughput"
        # Print last few lines for debugging
        echo "$output" | tail -5 | while read line; do
            echo "   $line"
        done
    fi
    
    echo ""
done

echo "============================================================"
echo "SUMMARY"
echo "============================================================"

if [ -f /tmp/benchmark_results.csv ]; then
    cat /tmp/benchmark_results.csv | while IFS='|' read model tok size elapsed; do
        status="✅"
        if (( $(echo "$tok < 50" | bc -l) )); then
            status="⚠️"
        fi
        echo "${status} $(basename $model): $tok tok/s ($size GB)"
    done
    
    # Calculate average
    avg=$(cat /tmp/benchmark_results.csv | cut -d'|' -f2 | awk '{sum+=$1; count++} END {if(count>0) print sum/count; else print 0}')
    echo ""
    echo "Average throughput: $avg tok/s"
    
    # Decision recommendation
    if (( $(echo "$avg >= 50" | bc -l) )); then
        echo ""
        echo "🎯 Recommendation: Continue Option C grind!"
        echo "   Custom kernels are approaching target performance."
    elif (( $(echo "$avg >= 30" | bc -l) )); then
        echo ""
        echo "⚖️  Recommendation: Hybrid approach (Option B+C)"
        echo "   Use mistral.rs for production, continue learning custom kernels."
    else
        echo ""
        echo "🔄 Recommendation: Pivot to Option B (mistral.rs hybrid)"
        echo "   Custom kernels need more tuning before parity."
    fi
    
    rm /tmp/benchmark_results.csv
fi
