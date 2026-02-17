#!/bin/bash
# Master Benchmark & Validation Runner
#
# Runs all Rust benchmarks, validation tests, R benchmarks, and merges results.
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --rust-only  # Only Rust benchmarks + validation
#   ./run_all.sh --r-only     # Only R benchmarks
#   ./run_all.sh --merge-only # Only merge existing results
#   ./run_all.sh --quick      # Skip comprehensive benchmarks, run validation only
#
# Prerequisites:
#   - Rust toolchain with cargo
#   - R with packages: bench, sandwich, plm, lfe, forecast, changepoint, survival,
#     MatchIt, WeightIt, randomForest, e1071, dbscan, Rtsne, spdep, rugarch, dlm
#   - p2a-core crate builds successfully

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
R_BENCH_DIR="$SCRIPT_DIR/r_comparison"
RESULTS_DIR="$R_BENCH_DIR/results"

# Parse arguments
RUN_RUST=true
RUN_R=true
RUN_MERGE=true
QUICK=false

for arg in "$@"; do
  case $arg in
    --rust-only) RUN_R=false ;;
    --r-only) RUN_RUST=false ;;
    --merge-only) RUN_RUST=false; RUN_R=false ;;
    --quick) QUICK=true ;;
    --help) echo "Usage: $0 [--rust-only|--r-only|--merge-only|--quick]"; exit 0 ;;
  esac
done

echo "============================================"
echo "  prompt2analytics — Full Benchmark Suite"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "R benchmarks: $R_BENCH_DIR"
echo "Results dir:  $RESULTS_DIR"
echo "Date:         $(date)"
echo ""

mkdir -p "$RESULTS_DIR"
mkdir -p "$PROJECT_ROOT/performance/results"

# Track overall timing
OVERALL_START=$(date +%s)

# ============================================
# Phase 1: Rust Validation Tests
# ============================================

if $RUN_RUST; then
  echo "=== Phase 1: Rust Validation Tests ==="
  echo ""

  cd "$PROJECT_ROOT"

  echo "Running: cargo test -p p2a-core -- test_validate"
  VALIDATION_START=$(date +%s)

  if cargo test -p p2a-core -- test_validate 2>&1 | tee "$RESULTS_DIR/validation.log"; then
    VALIDATION_END=$(date +%s)
    echo ""
    echo "Validation tests PASSED ($(( VALIDATION_END - VALIDATION_START ))s)"

    # Count tests
    PASS_COUNT=$(grep -c "test result: ok" "$RESULTS_DIR/validation.log" || echo "0")
    TOTAL_TESTS=$(grep "test result:" "$RESULTS_DIR/validation.log" | tail -1 || echo "unknown")
    echo "Results: $TOTAL_TESTS"
  else
    echo ""
    echo "WARNING: Some validation tests FAILED"
    echo "Check $RESULTS_DIR/validation.log for details"
  fi

  echo ""

  # ============================================
  # Phase 2: Rust Benchmarks
  # ============================================

  if ! $QUICK; then
    echo "=== Phase 2: Rust Comprehensive Benchmarks ==="
    echo ""

    echo "Running: cargo bench -p p2a-core --bench comprehensive_benchmarks"
    BENCH_START=$(date +%s)

    if cargo bench -p p2a-core --bench comprehensive_benchmarks 2>&1 | tee "$RESULTS_DIR/rust_bench.log"; then
      BENCH_END=$(date +%s)
      echo ""
      echo "Rust benchmarks COMPLETED ($(( BENCH_END - BENCH_START ))s)"
    else
      echo ""
      echo "WARNING: Rust benchmarks had errors"
      echo "Check $RESULTS_DIR/rust_bench.log for details"
    fi

    echo ""
  else
    echo "=== Phase 2: Skipped (--quick mode) ==="
    echo ""
  fi
fi

# ============================================
# Phase 3: R Benchmarks
# ============================================

if $RUN_R; then
  echo "=== Phase 3: R Benchmarks ==="
  echo ""

  # Check R is available
  if ! command -v Rscript &> /dev/null; then
    echo "WARNING: Rscript not found. Skipping R benchmarks."
    echo "Install R and required packages to run R benchmarks."
  else
    echo "R version: $(Rscript --vanilla -e 'cat(R.version.string)')"
    echo ""

    cd "$R_BENCH_DIR"

    # Check required packages
    echo "Checking required R packages..."
    Rscript --vanilla -e '
      required <- c("bench", "sandwich", "plm", "lfe")
      missing <- required[!sapply(required, requireNamespace, quietly = TRUE)]
      if (length(missing) > 0) {
        cat(sprintf("WARNING: Missing packages: %s\n", paste(missing, collapse = ", ")))
        cat("Install with: install.packages(c(\"", paste(missing, collapse = "\", \""), "\"))\n")
      } else {
        cat("All core packages available.\n")
      }
    '
    echo ""

    R_BENCH_START=$(date +%s)

    if $QUICK; then
      echo "Quick mode: Running comprehensive benchmark only"
      Rscript --vanilla benchmark_comprehensive.R 2>&1 | tee "$RESULTS_DIR/r_comprehensive.log"
    else
      echo "Running all R benchmarks via r_benchmark_runner.R"
      Rscript --vanilla r_benchmark_runner.R 2>&1 | tee "$RESULTS_DIR/r_runner.log"
    fi

    R_BENCH_END=$(date +%s)
    echo ""
    echo "R benchmarks COMPLETED ($(( R_BENCH_END - R_BENCH_START ))s)"
    echo ""
  fi
fi

# ============================================
# Phase 4: Merge Results
# ============================================

if $RUN_MERGE; then
  echo "=== Phase 4: Merging Results ==="
  echo ""

  cd "$R_BENCH_DIR"

  if command -v Rscript &> /dev/null; then
    Rscript --vanilla merge_results.R 2>&1 | tee "$RESULTS_DIR/merge.log"
    echo ""
  else
    echo "WARNING: Rscript not available for merging."
    echo "Run manually: cd $R_BENCH_DIR && Rscript merge_results.R"
  fi
fi

# ============================================
# Summary
# ============================================

OVERALL_END=$(date +%s)
TOTAL_TIME=$(( OVERALL_END - OVERALL_START ))

echo "============================================"
echo "  Summary"
echo "============================================"
echo ""
echo "Total time: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"
echo ""

# List output files
echo "Output files:"
if [ -d "$RESULTS_DIR" ]; then
  echo "  R benchmarks:"
  ls -lh "$RESULTS_DIR"/r_benchmarks_all.csv 2>/dev/null || echo "    (not yet generated)"
  echo "  Comparisons:"
  ls -lh "$RESULTS_DIR"/comparison_speed.csv 2>/dev/null || echo "    (not yet generated)"
  ls -lh "$RESULTS_DIR"/comparison_memory.csv 2>/dev/null || echo "    (not yet generated)"
  echo "  Coverage:"
  ls -lh "$RESULTS_DIR"/validation_coverage.csv 2>/dev/null || echo "    (not yet generated)"
  echo "  Logs:"
  ls -lh "$RESULTS_DIR"/*.log 2>/dev/null || echo "    (none)"
fi

echo ""
echo "Next steps:"
echo "  - Review results in $RESULTS_DIR/"
echo "  - Generate figures: Rscript $R_BENCH_DIR/analyze_results.R"
echo "  - Generate LaTeX tables: Rscript $R_BENCH_DIR/generate_latex_tables.R"
echo ""
echo "Done."
