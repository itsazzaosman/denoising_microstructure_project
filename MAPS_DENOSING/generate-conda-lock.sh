#!/bin/bash
# Generate conda-lock.yml for reproducible environment across machines
#
# Run this after updating environment.yml:
#   bash generate-conda-lock.sh
#
# Then commit conda-lock.yml to track exact versions and builds.

set -euo pipefail

echo "Installing conda-lock (if not already installed)..."
conda install -y -c conda-forge conda-lock

echo "Generating conda-lock.yml from environment.yml..."
conda-lock lock --file environment.yml --lockfile conda-lock.yml

echo "✓ Generated conda-lock.yml"
echo ""
echo "To reproduce this exact environment on another machine:"
echo "  conda-lock install --name ebsd conda-lock.yml"
