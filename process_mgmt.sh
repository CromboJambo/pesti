// Process Management Middleware for Hermes Agent
// Spawns, discovers, monitors, and attaches to running processes.
//
// Usage: ./process_mgmt.sh <command> [args...]
// Commands:
//   spawn <cmd>         - Generate command with PID capture wrapper
//   discover <pid>      - Check if process exists and get status
//   monitor <pid> <timeout_s> [interval_s] - Monitor with heartbeat feedback
//   attach <pid>        - Attach to running process for interaction

#!/bin/bash
set -euo pipefail

spawn_command() {
    local cmd="$1"
    echo "(${cmd}) & echo \"PID: \$!\""
}

discover_by_pid() {
    local pid="$1"
    ps -p "$pid" -o pid,stat,etime,cmd --no-headers 2>/dev/null || return 1
}

monitor_process() {
    local pid="$1"
    local timeout_s="${2:-300}"
    local interval_s="${3:-5}"
    
    local start=$(date +%s)
    
    while true; do
        local now=$(date +%s)
        local elapsed=$((now - start))
        
        if [ "$elapsed" -ge "$timeout_s" ]; then
            echo "Monitor result: timeout after ${elapsed}s"
            return 1
        fi
        
        if ! discover_by_pid "$pid" > /dev/null; then
            echo "Monitor result: exited after ${elapsed}s"
            return 0
        fi
        
        echo "[${elapsed}s] Process still running (PID ${pid})..."
        sleep "$interval_s"
    done
}

attach_to_process() {
    local pid="$1"
    
    if ! discover_by_pid "$pid" > /dev/null; then
        echo "Process not found: $pid"
        return 1
    fi
    
    echo "Process found. To debug interactively:"
    echo "  gdb -p $pid"
    echo "Or to inspect memory:"
    echo "  cat /proc/$pid/status"
}

case "$1" in
    spawn)
        shift
        spawn_command "$*"
        ;;
    discover)
        shift
        if result=$(discover_by_pid "$1"); then
            echo "Found process:"
            echo "$result"
        else
            echo "Process not found or exited: $1"
            exit 1
        fi
        ;;
    monitor)
        shift
        monitor_process "$@"
        ;;
    attach)
        shift
        attach_to_process "$1"
        ;;
    *)
        echo "Usage: process_mgmt.sh <command> [args...]"
        echo "Commands: spawn|discover|monitor|attach"
        exit 1
        ;;
esac