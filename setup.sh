#!/bin/bash
# PESTI - Quick Setup and Mode Switching Script
# Supports both Learning Mode (custom kernels) and Production Mode (mistral.rs)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

show_usage() {
    cat <<EOF
PESTI - Portable Execution Substrate for Transformer Inference
Usage: ./setup.sh [command]

Commands:
  learning      Build with custom PTX kernels (learning mode)
  production    Build with mistral.rs backend (production mode)
  benchmark     Run attention kernel benchmarks
  test          Run conformance tests
  clean         Clean build artifacts
  help          Show this help message

Examples:
  ./setup.sh learning      # Build for learning/experimentation
  ./setup.sh production    # Build for production performance (~72 tok/s)
  ./setup.sh benchmark     # Run kernel benchmarks
  ./setup.sh test          # Verify conformance (24/24 tests)

Notes:
  - Learning mode: ~35 tok/s expected (TinyLlama 1.1B)
  - Production mode: ~72 tok/s expected (Llama 3.1 8B Q4_K_M)
  - Requires CUDA 12.5+ for GPU builds

EOF
}

cmd_learning() {
    echo "🎓 Building PESTI in LEARNING MODE..."
    echo "   Backend: Custom PTX kernels + RoPE caching"
    echo "   Expected: ~35 tok/s (TinyLlama 1.1B)"
    echo ""
    
    cargo build --package pesti-runner --features cuda
    
    echo ""
    echo "✅ Learning mode build complete!"
    echo ""
    echo "Run benchmarks:"
    echo "  cargo run --package pesti-runner --example benchmark_attention_simple --features cuda"
    echo "  cargo run --package pesti-runner --example benchmark_flash_attention --features cuda"
}

cmd_production() {
    echo "🚀 Building PESTI in PRODUCTION MODE..."
    echo "   Backend: mistral.rs (WGMMA/tcgen05 kernels)"
    echo "   Expected: ~72 tok/s (Llama 3.1 8B Q4_K_M on RTX 4070 Ti SUPER)"
    echo ""
    
    cargo build --package pesti-runner --features cuda,mistralrs
    
    echo ""
    echo "✅ Production mode build complete!"
    echo ""
    echo "Run with real model:"
    echo "  cargo run --package pesti-runner --features cuda,mistralrs --example e2e_gpu_inference"
}

cmd_benchmark() {
    echo "📊 Running PESTI benchmarks..."
    echo ""
    
    # Run simple attention benchmark (learning mode)
    echo "1. Baseline vs Optimized Attention (Learning Mode)"
    cargo run --package pesti-runner --example benchmark_attention_simple --features cuda 2>&1 | tail -15
    
    echo ""
    echo "2. Flash Attention Benchmark (Option C)"
    cargo run --package pesti-runner --example benchmark_flash_attention --features cuda 2>&1 | tail -15
    
    echo ""
    echo "3. Mistral.rs Backend Test (Production Mode)"
    cargo run --package pesti-runner --example test_mistralrs_backend --features cuda,mistralrs 2>&1 | tail -10
}

cmd_test() {
    echo "🧪 Running PESTI conformance tests..."
    echo ""
    
    cargo test --package pesti-conformance
    
    echo ""
    echo "✅ All conformance tests passed!"
}

cmd_clean() {
    echo "🧹 Cleaning PESTI build artifacts..."
    cargo clean
    echo ""
    echo "✅ Build artifacts cleaned"
}

# Main script logic
case "${1:-help}" in
    learning)
        cmd_learning
        ;;
    production)
        cmd_production
        ;;
    benchmark)
        cmd_benchmark
        ;;
    test)
        cmd_test
        ;;
    clean)
        cmd_clean
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac
