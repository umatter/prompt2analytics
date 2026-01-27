#!/bin/bash
# run_all.sh - Run complete validation and benchmark suite
# Usage: ./scripts/run_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "Complete Validation and Benchmark Suite"
echo "========================================"
echo ""

# 1. Generate synthetic data
echo "Step 1: Generating synthetic datasets..."
cd "$SCRIPT_DIR/.."
Rscript datasets/generate_synthetic.R datasets/

# 2. Run validation
echo ""
echo "Step 2: Running validation suite..."
"$SCRIPT_DIR/run_validation.sh" || true

# 3. Run benchmarks
echo ""
echo "Step 3: Running benchmark suite..."
"$SCRIPT_DIR/run_benchmark.sh"

# 4. Generate final report
echo ""
echo "Step 4: Generating final reports..."
"$SCRIPT_DIR/generate_report.sh"

echo ""
echo "========================================"
echo "Complete! Results in results/summaries/"
echo "========================================"
