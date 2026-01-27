#!/bin/bash
# run_validation.sh - Run R vs Rust validation for all methods
# Usage: ./scripts/run_validation.sh [--dry-run] [method]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$BASE_DIR/results/validation"
DATASETS_DIR="$BASE_DIR/datasets"
R_SCRIPTS_DIR="$BASE_DIR/r_scripts"
RUST_RUNNER="$BASE_DIR/rust_runner/target/release/run-validation"

# Parse options
DRY_RUN=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Track results
PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)) || true; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)) || true; }
log_skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; ((SKIP_COUNT++)) || true; }
log_info() { echo -e "[INFO] $1"; }
log_dry() { echo -e "${CYAN}[DRY-RUN]${NC} $1"; }

# Run single validation
run_validation() {
    local method=$1
    local dataset=$2
    local r_args=$3
    local rust_args=$4

    local dataset_name=$(basename "$dataset" .csv)
    local r_output="$RESULTS_DIR/r_${method}_${dataset_name}.json"
    local rust_output="$RESULTS_DIR/rust_${method}_${dataset_name}.json"

    if [ "$DRY_RUN" -eq 1 ]; then
        log_dry "Would validate $method on $dataset_name"
        log_dry "  R: Rscript run_method.R --method $method --data $dataset $r_args"
        log_dry "  Rust: run-validation --method $method --data $dataset $rust_args"
        return 0
    fi

    log_info "Validating $method on $dataset_name..."

    # Run R
    if Rscript "$R_SCRIPTS_DIR/run_method.R" --method "$method" --data "$dataset" \
        $r_args --output "$r_output" 2>/dev/null; then
        log_info "  R completed"
    else
        log_fail "$method on $dataset_name (R failed)"
        return 1
    fi

    # Run Rust (validation runner)
    if $RUST_RUNNER --method "$method" --data "$dataset" $rust_args > "$rust_output" 2>/dev/null; then
        log_info "  Rust completed"
    else
        log_fail "$method on $dataset_name (Rust failed)"
        return 1
    fi

    # Compare results
    if "$SCRIPT_DIR/compare_results.sh" "$r_output" "$rust_output" > /dev/null 2>&1; then
        log_pass "$method on $dataset_name"
        return 0
    else
        log_fail "$method on $dataset_name (comparison failed)"
        return 1
    fi
}

# Check for Rust runner
if [ ! -f "$RUST_RUNNER" ]; then
    echo "Building Rust validation runner..."
    (cd "$BASE_DIR/rust_runner" && cargo build --release) || {
        echo "Error: Failed to build Rust runner"
        exit 1
    }
fi

# Specific method or all
TARGET_METHOD="${1:-all}"

echo "========================================"
echo "Rust vs R Validation Suite"
echo "========================================"
echo ""

# Copy existing datasets if not present
if [ ! -f "$DATASETS_DIR/longley.csv" ]; then
    cp "$BASE_DIR/../../../validation/datasets/longley.csv" "$DATASETS_DIR/" 2>/dev/null || true
fi
if [ ! -f "$DATASETS_DIR/grunfeld.csv" ]; then
    cp "$BASE_DIR/../../../validation/datasets/grunfeld.csv" "$DATASETS_DIR/" 2>/dev/null || true
fi

# Generate synthetic data if needed
if [ ! -f "$DATASETS_DIR/synthetic_ols_n1000.csv" ]; then
    log_info "Generating synthetic datasets..."
    Rscript "$DATASETS_DIR/generate_synthetic.R" "$DATASETS_DIR"
fi

# ============= OLS Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "ols" ]; then
    echo ""
    echo "--- OLS Validation ---"

    # Longley dataset
    LONGLEY="$DATASETS_DIR/longley.csv"
    if [ -f "$LONGLEY" ]; then
        run_validation "ols" "$LONGLEY" \
            "-y Employed -x GNP,Unemployed,Armed.Forces,Population,Year" \
            "-y Employed -x GNP Unemployed Armed.Forces Population Year"
    else
        log_skip "OLS on longley (dataset not found)"
    fi

    # Synthetic dataset
    SYNTH_OLS="$DATASETS_DIR/synthetic_ols_n1000.csv"
    if [ -f "$SYNTH_OLS" ]; then
        run_validation "ols" "$SYNTH_OLS" \
            "-y y -x x1,x2,x3" \
            "-y y -x x1 x2 x3"
    fi
fi

# ============= OLS HC1 Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "ols_hc1" ]; then
    echo ""
    echo "--- OLS HC1 Validation ---"

    SYNTH_OLS="$DATASETS_DIR/synthetic_ols_n1000.csv"
    if [ -f "$SYNTH_OLS" ]; then
        run_validation "ols_hc1" "$SYNTH_OLS" \
            "-y y -x x1,x2,x3" \
            "-y y -x x1 x2 x3 -r hc1"
    fi
fi

# ============= Panel FE Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "panel_fe" ]; then
    echo ""
    echo "--- Panel FE Validation ---"

    GRUNFELD="$DATASETS_DIR/grunfeld.csv"
    if [ -f "$GRUNFELD" ]; then
        run_validation "panel_fe" "$GRUNFELD" \
            "-y invest -x value,capital -e firm -t year" \
            "-y invest -x value capital -e firm -t year"
    else
        log_skip "Panel FE on grunfeld (dataset not found)"
    fi

    SYNTH_PANEL="$DATASETS_DIR/synthetic_panel_n2000.csv"
    if [ -f "$SYNTH_PANEL" ]; then
        run_validation "panel_fe" "$SYNTH_PANEL" \
            "-y y -x x1,x2 -e entity -t time" \
            "-y y -x x1 x2 -e entity -t time"
    fi
fi

# ============= Panel RE Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "panel_re" ]; then
    echo ""
    echo "--- Panel RE Validation ---"

    GRUNFELD="$DATASETS_DIR/grunfeld.csv"
    if [ -f "$GRUNFELD" ]; then
        run_validation "panel_re" "$GRUNFELD" \
            "-y invest -x value,capital -e firm -t year" \
            "-y invest -x value capital -e firm -t year"
    else
        log_skip "Panel RE on grunfeld (dataset not found)"
    fi
fi

# ============= Logit Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "logit" ]; then
    echo ""
    echo "--- Logit Validation ---"

    SYNTH_BIN="$DATASETS_DIR/synthetic_binary_n1000.csv"
    if [ -f "$SYNTH_BIN" ]; then
        run_validation "logit" "$SYNTH_BIN" \
            "-y y -x x1,x2" \
            "-y y -x x1 x2"
    fi
fi

# ============= Probit Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "probit" ]; then
    echo ""
    echo "--- Probit Validation ---"

    SYNTH_BIN="$DATASETS_DIR/synthetic_binary_n1000.csv"
    if [ -f "$SYNTH_BIN" ]; then
        run_validation "probit" "$SYNTH_BIN" \
            "-y y -x x1,x2" \
            "-y y -x x1 x2"
    fi
fi

# ============= K-Means Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "kmeans" ]; then
    echo ""
    echo "--- K-Means Validation ---"

    SYNTH_CLUST="$DATASETS_DIR/synthetic_cluster_n1000.csv"
    if [ -f "$SYNTH_CLUST" ]; then
        run_validation "kmeans" "$SYNTH_CLUST" \
            "-x x1,x2 -k 3 -s 42" \
            "-x x1 x2 -k 3 -s 42"
    fi
fi

# ============= PCA Validation =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "pca" ]; then
    echo ""
    echo "--- PCA Validation ---"

    SYNTH_PCA="$DATASETS_DIR/synthetic_pca_n1000.csv"
    if [ -f "$SYNTH_PCA" ]; then
        run_validation "pca" "$SYNTH_PCA" \
            "-x x1,x2,x3,x4" \
            "-x x1 x2 x3 x4"
    fi
fi

# ============= Summary =============
echo ""
echo "========================================"
echo "Validation Summary"
echo "========================================"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}"
echo -e "${RED}Failed: $FAIL_COUNT${NC}"
echo -e "${YELLOW}Skipped: $SKIP_COUNT${NC}"
echo ""

# Save summary
SUMMARY_FILE="$BASE_DIR/results/summaries/validation_summary.json"
mkdir -p "$(dirname "$SUMMARY_FILE")"
cat > "$SUMMARY_FILE" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "passed": $PASS_COUNT,
  "failed": $FAIL_COUNT,
  "skipped": $SKIP_COUNT,
  "total": $((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
}
EOF

exit $FAIL_COUNT
