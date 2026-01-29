#!/bin/bash
# generate_outputs.sh
#
# Generates .tex output files for paper examples by running actual CLI commands.
# These files are then included in the paper via \VerbatimInput{}.
#
# Usage: ./paper/code/generate_outputs.sh
# Called by: make generate (before make jss/arxiv)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PAPER_DIR")"

P2A="${P2A:-$PROJECT_ROOT/target/release/p2a}"
DATA_DIR="$PROJECT_ROOT/validation/datasets"
OUTPUT_DIR="$PAPER_DIR/generated"
SESSION_FILE="/tmp/p2a_paper_examples_$$.json"

# Check if binary exists
if [ ! -f "$P2A" ]; then
    echo "Error: p2a binary not found at $P2A"
    echo "Build with: cargo build --release -p p2a-cli"
    exit 1
fi

# Check if Grunfeld data exists
if [ ! -f "$DATA_DIR/grunfeld.csv" ]; then
    echo "Error: grunfeld.csv not found at $DATA_DIR/grunfeld.csv"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean any existing session
rm -f "$SESSION_FILE"

echo "Generating example outputs for paper..."
echo "Binary: $P2A"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Helper function to run command and save output
generate_output() {
    local name="$1"
    local output_file="$OUTPUT_DIR/${name}.tex"
    shift

    echo "  Generating: $name"
    "$P2A" --session "$SESSION_FILE" "$@" > "$output_file" 2>&1 || true
}

# ============================================================
# Load dataset first (needed for all subsequent commands)
# ============================================================
echo "Loading dataset..."
"$P2A" --session "$SESSION_FILE" data load "$DATA_DIR/grunfeld.csv" --name grunfeld > /dev/null 2>&1

# ============================================================
# CLI Examples
# ============================================================
echo "Generating CLI examples..."

# Data description
generate_output "cli_data_describe" data describe grunfeld

# Data head (first 5 rows)
generate_output "cli_data_head" data head grunfeld --n 5

# OLS with robust standard errors
generate_output "cli_ols" reg ols grunfeld -y inv -x value capital --robust hc1

# Regression diagnostics
generate_output "cli_diagnostics" reg diagnostics grunfeld -y inv -x value capital

# Fixed effects
generate_output "cli_fe" panel fe grunfeld -y inv -x value capital --entity firm

# Random effects
generate_output "cli_re" panel re grunfeld -y inv -x value capital --entity firm

# Hausman test
generate_output "cli_hausman" panel hausman grunfeld -y inv -x value capital --entity firm

# Two-way HDFE
generate_output "cli_hdfe" panel hdfe grunfeld -y inv -x value capital --fe firm year

# JSON output
generate_output "cli_ols_json" -F json reg ols grunfeld -y inv -x value capital --robust hc1

# ============================================================
# Generate timestamp file for reproducibility
# ============================================================
cat > "$OUTPUT_DIR/generated_info.tex" << EOF
% Auto-generated output files for prompt2analytics paper
% Generated: $(date -Iseconds)
% Binary: $P2A
% Binary version: $($P2A --version 2>/dev/null || echo "unknown")
% Data: $DATA_DIR/grunfeld.csv
EOF

# Cleanup
rm -f "$SESSION_FILE"

echo ""
echo "Done! Generated files:"
ls -la "$OUTPUT_DIR"/*.tex
echo ""
echo "To rebuild paper with fresh outputs: make clean-generated jss"
