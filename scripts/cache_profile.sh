#!/bin/bash
# Cache profiling script for PESTI ndarray benchmarks

set -e

cd /home/crombo/projects/pesti

echo "=== Cache Profiling Suite ==="
echo ""

# Check if perf is available
if ! command -v perf &> /dev/null; then
    echo "⚠️  perf not found, installing..."
    sudo apt-get install -y linux-tools-generic perf
fi

echo "Running cache profiling for scaling benchmarks..."
echo ""

# Profile the scaling benchmark with cache events
echo "1. Profiling L1/L2/L3 cache misses..."
perf stat -e l1-dcache-loads,l1-dcache-load-misses,l2-cache-loads,l2-cache-load-misses,llc-loads,llc-load-misses \
    cargo run --package pesti-runner --example scaling_benchmark 2>&1 | tee /tmp/hermes-perf-cache.txt

echo ""
echo "2. Profiling branch misses and cycles..."
perf stat -e branches,branch-misses,cycles,instructions,cache-references,cache-misses \
    cargo run --package pesti-runner --example scaling_benchmark 2>&1 | tee /tmp/hermes-perf-cycles.txt

echo ""
echo "3. Generating flame graph (if perf-record available)..."
perf record -g cargo run --package pesti-runner --example ndarray_benchmark 2>&1
perf script > /tmp/hermes-flame.txt 2>/dev/null || echo "⚠️  perf script failed, skipping flame graph"

echo ""
echo "=== Results ==="
echo ""
echo "L1 Cache Performance:"
grep -A 5 "l1-dcache" /tmp/hermes-perf-cache.txt | tail -n +2 || echo "No L1 data available"

echo ""
echo "L2 Cache Performance:"
grep -A 5 "l2-cache" /tmp/hermes-perf-cache.txt | tail -n +2 || echo "No L2 data available"

echo ""
echo "L3 (LLC) Performance:"
grep -A 5 "llc-loads" /tmp/hermes-perf-cache.txt | tail -n +2 || echo "No L3 data available"

echo ""
echo "Branch Efficiency:"
grep -E "(branches|branch-misses)" /tmp/hermes-perf-cycles.txt | tail -n +2 || echo "No branch data available"

echo ""
echo "Instructions per Cycle (IPC):"
if grep -q "cycles" /tmp/hermes-perf-cycles.txt && grep -q "instructions" /tmp/hermes-perf-cycles.txt; then
    cycles=$(grep "cycles" /tmp/hermes-perf-cycles.txt | awk '{print $1}')
    instructions=$(grep "instructions" /tmp/hermes-perf-cycles.txt | awk '{print $1}')
    echo "IPC = $instructions / $cycles = $(echo "scale=3; $instructions / $cycles" | bc)"
else
    echo "Could not calculate IPC"
fi

# Cleanup temp files
rm -f /tmp/hermes-perf-cache.txt /tmp/hermes-perf-cycles.txt /tmp/hermes-flame.txt

echo ""
echo "=== Analysis ==="
echo "Key metrics to analyze:"
echo "  • L1 miss rate: Should be < 5% for good cache utilization"
echo "  • L2 miss rate: Should be < 10% for efficient memory access"
echo "  • L3 miss rate: Should be < 20% for optimal bandwidth usage"
echo "  • IPC: Higher is better (target > 2.0 for AVX-512)"
echo "  • Branch efficiency: Lower branch-miss % indicates better prediction"
