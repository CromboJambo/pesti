#!/bin/bash
# diagnostic_gpu_test.sh

echo "============================================="
echo "== Diagnostic Test: GPU Hardware Visibility =="
echo "============================================="

# Test 1: General system query (should show all devices)
echo -e "\n--- Running 'nvidia-smi' for comprehensive view ---"
nvidia-smi

# Test 2: Query specific details for the RTX 4070 (GPU 0)
echo -e "\n--- Detailed Check: GPU 0 (RTX 4070) ---"
# Using grep and awk to parse a few key fields from the output of nvidia-smi.
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,{| Name | Total Memory | Free Memory | Driver Version |}

# Test 3: Query specific details for the RTX 5060 Ti (GPU 1)
echo -e "\n--- Detailed Check: GPU 1 (RTX 5060 Ti) ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,{| Name | Total Memory | Free Memory | Driver Version |}

echo -e "\n============================================="
echo "Diagnostic check complete."