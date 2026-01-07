#!/bin/bash
# Setup script for megatron-memory-estimator skill
#
# This script installs all required dependencies for the skill to work.
# Run this once before using the skill.

set -e

echo "Setting up Megatron Memory Estimator environment..."
echo

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found. Please install Python and pip first."
    exit 1
fi

echo "Installing required packages..."
echo

# Install core dependencies
pip install mbridge transformers torch megatron-core einops termcolor tabulate pyyaml

echo
echo "✓ Setup complete!"
echo
echo "You can now use the skill with:"
echo "  python scripts/estimate_from_hf.py deepseek-ai/DeepSeek-V3 --tp 4 --pp 4 --ep 8"
echo
