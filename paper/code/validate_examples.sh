#!/bin/bash
# validate_examples.sh
#
# This script captures authentic CLI outputs for the paper examples.
# Run after building the CLI: cargo build --release -p p2a-cli
#
# Usage: ./paper/code/validate_examples.sh
# Output: paper/code/example_outputs.txt

set -e

P2A="${P2A:-./target/release/p2a}"
DATA_DIR="./validation/datasets"
OUTPUT_FILE="./paper/code/example_outputs.txt"
SESSION_FILE="/tmp/p2a_paper_session.json"

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

# Clean any existing session
rm -f "$SESSION_FILE"

echo "Capturing CLI outputs for paper examples..."
echo "Binary: $P2A"
echo "Output: $OUTPUT_FILE"
echo ""

{
    echo "============================================================"
    echo "EXAMPLE OUTPUTS FOR JSS PAPER"
    echo "Generated: $(date)"
    echo "Binary: $P2A"
    echo "============================================================"
    echo ""

    echo "============================================================"
    echo "1. DATA LOADING"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json data load grunfeld.csv --name grunfeld"
    echo "---"
    $P2A --session "$SESSION_FILE" data load "$DATA_DIR/grunfeld.csv" --name grunfeld
    echo ""

    echo "============================================================"
    echo "2. DATA DESCRIBE"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json data describe grunfeld"
    echo "---"
    $P2A --session "$SESSION_FILE" data describe grunfeld
    echo ""

    echo "============================================================"
    echo "3. DATA HEAD"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json data head grunfeld --n 5"
    echo "---"
    $P2A --session "$SESSION_FILE" data head grunfeld --n 5
    echo ""

    echo "============================================================"
    echo "4. OLS REGRESSION WITH ROBUST SEs"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json reg ols grunfeld -y inv -x value capital --robust hc1"
    echo "---"
    $P2A --session "$SESSION_FILE" reg ols grunfeld -y inv -x value capital --robust hc1
    echo ""

    echo "============================================================"
    echo "5. REGRESSION DIAGNOSTICS"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json reg diagnostics grunfeld -y inv -x value capital"
    echo "---"
    $P2A --session "$SESSION_FILE" reg diagnostics grunfeld -y inv -x value capital
    echo ""

    echo "============================================================"
    echo "6. FIXED EFFECTS"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json panel fe grunfeld -y inv -x value capital --entity firm"
    echo "---"
    $P2A --session "$SESSION_FILE" panel fe grunfeld -y inv -x value capital --entity firm
    echo ""

    echo "============================================================"
    echo "7. RANDOM EFFECTS"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json panel re grunfeld -y inv -x value capital --entity firm"
    echo "---"
    $P2A --session "$SESSION_FILE" panel re grunfeld -y inv -x value capital --entity firm
    echo ""

    echo "============================================================"
    echo "8. HAUSMAN TEST"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json panel hausman grunfeld -y inv -x value capital --entity firm"
    echo "---"
    $P2A --session "$SESSION_FILE" panel hausman grunfeld -y inv -x value capital --entity firm
    echo ""

    echo "============================================================"
    echo "9. HIGH-DIMENSIONAL FIXED EFFECTS (TWO-WAY)"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json panel hdfe grunfeld -y inv -x value capital --fe firm year"
    echo "---"
    $P2A --session "$SESSION_FILE" panel hdfe grunfeld -y inv -x value capital --fe firm year
    echo ""

    echo "============================================================"
    echo "10. JSON OUTPUT"
    echo "============================================================"
    echo ""
    echo "Command: p2a --session analysis.json -F json reg ols grunfeld -y inv -x value capital --robust hc1"
    echo "---"
    $P2A --session "$SESSION_FILE" -F json reg ols grunfeld -y inv -x value capital --robust hc1
    echo ""

    echo "============================================================"
    echo "END OF EXAMPLES"
    echo "============================================================"

} > "$OUTPUT_FILE" 2>&1

echo "Done! Output saved to: $OUTPUT_FILE"
echo ""
echo "Next steps:"
echo "1. Review the output file"
echo "2. Update paper/examples_draft.tex with actual values"
echo "3. Run R comparison to verify numerical accuracy"

# Cleanup
rm -f "$SESSION_FILE"
