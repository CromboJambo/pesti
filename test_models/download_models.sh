#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "📥 Downloading TinyLlama benchmark models..."
echo ""

MODELS=(
    "Q3_K_M:tinyllama-q3.gguf:https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-V1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf"
    "Q4_K_M:tinyllama-q4.gguf:https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-V1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    "Q5_K_M:tinyllama-q5.gguf:https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-V1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
    "Q8_0:tinyllama-q8.gguf:https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-V1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
)

for model in "${MODELS[@]}"; do
    IFS=':' read -r quant name url <<< "$model"
    
    if [ -f "$name" ]; then
        echo "✅ $quant: Already downloaded ($name)"
        continue
    fi
    
    echo "⬇️  Downloading $quant..."
    curl -L -o "$name" "$url" 2>&1 | tail -3
    ls -lh "$name"
    echo ""
done

echo "🎉 All models downloaded!"
echo ""
echo "To run benchmarks:"
echo "  cargo run --package pesti-runner --example q4_stress_test 500 test"
echo ""
echo "To benchmark all quantizations:"
echo "  ./benchmark_all_quant.sh"
