#!/bin/bash
# Test WGMMA kernel launch after killing llama-server

echo "=== WGMMA Kernel Launch Test ==="
echo ""

# Kill llama-server if running
if pgrep -x "llama-server" > /dev/null; then
    echo "⚠️  Found llama-server (PID $(pgrep -x llama-server)), killing..."
    kill $(pgrep -x llama-server)
    sleep 3
fi

# Wait a moment for VRAM to free up
echo "Waiting 5 seconds for VRAM to clear..."
sleep 5

# Check GPU status
echo ""
echo "=== GPU Status ==="
nvidia-smi pmon -c 1 | grep llama-server || echo "✅ No llama-server running"

# Run the test
echo ""
echo "=== Running WGMMA Test ==="
cd /home/crombo/projects/pesti
cargo run --package pesti-runner --example test_attention_kernel 2>&1 | tail -20

# Check if successful
if [ ${PIPESTATUS[1]} -eq 0 ]; then
    echo ""
    echo "✅ Test completed successfully!"
else
    echo ""
    echo "❌ Test failed - check output above"
fi
