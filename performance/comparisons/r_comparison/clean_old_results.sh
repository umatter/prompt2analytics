#!/usr/bin/env bash
# Remove timestamped benchmark result files older than 30 days
# These files are gitignored; this script is for local hygiene

RESULTS_DIR="$(dirname "$0")/results"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory not found: $RESULTS_DIR"
    exit 1
fi

echo "Cleaning up result files older than 30 days in $RESULTS_DIR..."
find "$RESULTS_DIR" -name "r_*_20*.csv" -mtime +30 -print -delete
find "$RESULTS_DIR" -name "rust_*_20*.json" -mtime +30 -print -delete
echo "Done."
