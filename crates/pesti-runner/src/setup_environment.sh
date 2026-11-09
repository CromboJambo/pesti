#!/bin/bash
# PESTI Environment Setup Script
# This script sets up essential environment variables for local LLM development.

echo "--- Setting up PESTI required environment paths ---"

# 1. Set the Hugging Face Cache Home to include the symlinked models
export HF_HOME="/home/crombo/projects/llm-workspace/llmstudio_models"

# 2. Ensure other local inference tools use this root cache if applicable (e.g., llama.cpp)
export LMSTUDIO_MODELS=/home/crombo/projects/llm-workspace/llmstudio_models

echo "✅ Environment variables set for local model paths."
echo "Run 'hermes chat' or other PESTI tools now that the cache is correctly pointed."