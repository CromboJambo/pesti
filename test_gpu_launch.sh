#!/bin/bash
set -euo pipefail

cd /home/crombo/projects/pesti

echo "=== Testing GPU Kernel Launch ==="
echo ""

# Use Qwen2.5-0.5b-Q2_K model (323 MiB) - fits easily on GPU1 with 2 GiB free
MODEL_PATH="conformance-corpus/qwen2.5-0.5b-instruct-q2_k.gguf"
NUM_THREADS=4

echo "Running test_gpu_attention.rs example..."
cargo run --package pesti-runner --example test_gpu_attention \
    --features gpu \
    -- \
    --model "$MODEL_PATH" \
    --num-threads "$NUM_THREADS" \
    --device cuda:1 \
    --seq-len 256 \
    --batch-size 1
