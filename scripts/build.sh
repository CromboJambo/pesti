#!/usr/bin/env bash
# pesti-runner build targets
set -euo pipefail

cd "$(dirname "$0")/.."

build_mistralrs() {
    echo "Building with mistral.rs backend (cuda + mistralrs features)..."
    cargo build --package pesti-runner --features cuda,mistralrs
}

build_cuda_only() {
    echo "Building with CUDA only (no mistral.rs)..."
    cargo build --package pesti-runner --features cuda
}

build_cpu_only() {
    echo "Building CPU-only (no CUDA, no mistral.rs)..."
    cargo build --package pesti-runner
}

case "${1:-}" in
    mistralrs) build_mistralrs ;;
    cuda)      build_cuda_only ;;
    cpu)       build_cpu_only ;;
    *)
        echo "Usage: $0 {mistralrs|cuda|cpu}"
        echo "  mistralrs  - Build with CUDA + mistral.rs backend"
        echo "  cuda       - Build with CUDA only (no mistral.rs)"
        echo "  cpu        - Build CPU-only (no CUDA, no mistral.rs)"
        ;;
esac