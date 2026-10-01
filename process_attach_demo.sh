#!/bin/bash
# Process Discovery & Attachment Protocol - Demo
# Demonstrates middleware layer for agent process management

set -euo pipefail

echo "=== Process Discovery & Attachment Protocol Demo ==="
echo ""

# Step 1: Generate command for user to run
echo "Step 1: Generating command..."
CMD="sleep 30 && echo done"
echo "Command: $CMD"
echo "(In real use, agent would provide this to user)"
echo ""

# For demo, we start the process ourselves
echo "(Demo: starting test process ourselves...)"
bash -c "$CMD" &
TARGET_PID=$!
echo "Started test process with PID: $TARGET_PID"
echo ""

# Step 2: Discover the process by PID
echo "Step 2: Discovering process by PID..."
discover_by_pid() {
    local pid=$1
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "Found process:"
        ps -p "$pid" -o pid,stat,etime,cmd --no-headers
        return 0
    else
        echo "Process not found or exited"
        return 1
    fi
}

if discover_by_pid $TARGET_PID; then
    # Step 3: Monitor the process
    echo ""
    echo "Step 3: Monitoring process..."
    
    ITERATIONS=0
    while [ $ITERATIONS -lt 5 ]; do
        sleep 2
        ITERATIONS=$((ITERATIONS + 1))
        
        if ! ps -p "$TARGET_PID" > /dev/null 2>&1; then
            echo "Process exited after $((ITERATIONS * 2)) seconds"
            break
        fi
        
        # Show elapsed time from process start
        ELAPSED=$(ps -p "$TARGET_PID" -o etime --no-headers | tr -d ' ')
        echo "[$ITERATIONS] Process still running (elapsed: $ELAPSED)..."
    done
    
    # Step 4: Get final status
    echo ""
    echo "Step 4: Final status check..."
    if ps -p "$TARGET_PID" > /dev/null 2>&1; then
        echo "Process still running:"
        discover_by_pid $TARGET_PID
        
        # Clean up
        echo "Cleaning up..."
        kill $TARGET_PID 2>/dev/null || true
    else
        echo "Process completed naturally"
    fi
else
    echo "Failed to discover process"
fi

echo ""
echo "=== Demo Complete ==="
