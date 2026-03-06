#!/bin/bash
# Unified Validation + Benchmarking Pipeline
#
# Phase 1: R generates data CSVs + runs methods + captures timing + outputs
# Phase 2: Rust loads same CSVs + runs methods + captures timing + outputs
# Phase 3: Merge compares timing AND correctness
#
# Usage:
#   ./run_unified.sh              # Full pipeline
#   ./run_unified.sh --r-only     # Only R phase
#   ./run_unified.sh --rust-only  # Only Rust phase
#   ./run_unified.sh --merge-only # Only merge phase

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
R_BENCH_DIR="$SCRIPT_DIR/r_comparison"
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$R_BENCH_DIR/results"

# Parse arguments
RUN_R=true
RUN_RUST=true
RUN_MERGE=true

for arg in "$@"; do
  case $arg in
    --r-only) RUN_RUST=false; RUN_MERGE=false ;;
    --rust-only) RUN_R=false; RUN_MERGE=false ;;
    --merge-only) RUN_R=false; RUN_RUST=false ;;
    --help)
      echo "Usage: $0 [--r-only|--rust-only|--merge-only]"
      echo ""
      echo "Options:"
      echo "  --r-only     Only run R phase (generate data + benchmark)"
      echo "  --rust-only  Only run Rust phase (load data + benchmark)"
      echo "  --merge-only Only merge existing R + Rust results"
      exit 0
      ;;
  esac
done

echo "============================================"
echo "  prompt2analytics - Unified Pipeline"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Data dir:     $DATA_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Date:         $(date)"
echo ""

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"
mkdir -p "$PROJECT_ROOT/performance/results"

OVERALL_START=$(date +%s)

# ============================================
# Phase 1: R generates data + benchmarks
# ============================================

if $RUN_R; then
  echo "=== Phase 1: R Unified Benchmarks ==="
  echo ""

  if ! command -v Rscript &> /dev/null; then
    echo "ERROR: Rscript not found. Install R to run Phase 1."
    exit 1
  fi

  echo "R version: $(Rscript --vanilla -e 'cat(R.version.string)')"
  echo ""

  cd "$R_BENCH_DIR"

  R_START=$(date +%s)
  Rscript --vanilla unified_benchmark.R --data-dir "$DATA_DIR" 2>&1 | tee "$RESULTS_DIR/unified_r.log"
  R_END=$(date +%s)

  echo ""
  echo "Phase 1 complete ($(( R_END - R_START ))s)"
  echo ""
fi

# ============================================
# Phase 2: Rust loads data + benchmarks
# ============================================

if $RUN_RUST; then
  echo "=== Phase 2: Rust Unified Benchmarks ==="
  echo ""

  cd "$PROJECT_ROOT"

  RUST_START=$(date +%s)
  cargo bench -p p2a-core --bench unified_benchmarks -- --data-dir "$DATA_DIR" 2>&1 | tee "$RESULTS_DIR/unified_rust.log"
  RUST_END=$(date +%s)

  echo ""
  echo "Phase 2 complete ($(( RUST_END - RUST_START ))s)"
  echo ""
fi

# ============================================
# Phase 3: Merge results
# ============================================

if $RUN_MERGE; then
  echo "=== Phase 3: Merge Results ==="
  echo ""

  if ! command -v Rscript &> /dev/null; then
    echo "ERROR: Rscript not found. Install R to run merge."
    exit 1
  fi

  cd "$R_BENCH_DIR"
  Rscript --vanilla unified_merge.R 2>&1 | tee "$RESULTS_DIR/unified_merge.log"

  echo ""
  echo "Phase 3 complete."

  # Print quick summary from output CSV
  if [ -f "$RESULTS_DIR/comparison_unified.csv" ]; then
    echo ""
    echo "=== Quick Summary ==="
    TOTAL=$(tail -n +2 "$RESULTS_DIR/comparison_unified.csv" | wc -l)
    AGREE=$(tail -n +2 "$RESULTS_DIR/comparison_unified.csv" | awk -F',' '{print $8}' | grep -c "TRUE" || true)
    DISAGREE=$(tail -n +2 "$RESULTS_DIR/comparison_unified.csv" | awk -F',' '{print $8}' | grep -c "FALSE" || true)
    echo "Total comparisons: $TOTAL"
    echo "Outputs agree:     $AGREE"
    echo "Outputs disagree:  $DISAGREE"
    echo ""
    echo "Full results: $RESULTS_DIR/comparison_unified.csv"
  fi
fi

# ============================================
# Summary
# ============================================

OVERALL_END=$(date +%s)
TOTAL_TIME=$(( OVERALL_END - OVERALL_START ))

echo ""
echo "============================================"
echo "Total time: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"
echo "============================================"
