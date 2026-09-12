#!/bin/bash
# Generate conda-lock.yml for reproducible environment across machines
#
# Run this after updating environment.yml:
#   bash generate-conda-lock.sh
#
# Then commit conda-lock.yml to track exact versions and builds.

set -euo pipefail

echo "Installing conda-lock (if not already installed)..."
~/miniconda3/bin/conda install -y -c conda-forge conda-lock 2>&1 | tail -3

echo "Generating conda-lock.yml from environment.yml (linux-64 only)..."
# Only lock for linux-64 to avoid cross-platform resolution issues
conda-lock lock --file environment.yml --lockfile conda-lock.yml --platform linux-64

echo "✓ Generated conda-lock.yml"
echo ""
echo "To reproduce this exact environment on Linux:"
echo "  conda-lock install --name ebsd conda-lock.yml"
