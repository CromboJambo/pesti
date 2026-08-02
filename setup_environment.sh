#!/bin/bash
# setup_environment.sh
# ---------------------------------------------------
# Source this file to set up critical environment variables for PESTI and local LLM inference tasks.
# Usage: source $(realpath $0)
#
# Note: This script is silent when sourced. Run directly to see output.

# Define the canonical directory for local model weights (as per memory).
export HF_HOME="/home/crombo/projects/pesti/lms-models"

# Exporting this variable makes it available to subprocesses and other scripts that source this file.
export HF_HOME

if [ -d "$HF_HOME" ]; then
    mkdir -p "$HF_HOME" 2>/dev/null
fi
