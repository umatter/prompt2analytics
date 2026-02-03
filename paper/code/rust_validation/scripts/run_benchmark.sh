#!/bin/bash
# run_benchmark.sh - Run performance benchmarks for R vs Rust
# Usage: ./scripts/run_benchmark.sh [method] [sample_size]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$BASE_DIR/results/benchmarks"
DATASETS_DIR="$BASE_DIR/datasets"
R_SCRIPTS_DIR="$BASE_DIR/r_scripts"
RUST_RUNNER="$BASE_DIR/rust_runner/target/release/run-validation"

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

# Configuration
ITERATIONS=${ITERATIONS:-100}
WARMUP=${WARMUP:-5}
SAMPLE_SIZES=(100 1000 10000 100000)

# Colors
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${CYAN}[BENCH]${NC} $1"; }

# Check for hyperfine
if ! command -v hyperfine &> /dev/null; then
    echo "Warning: hyperfine not found. Install for better Rust benchmarks."
    echo "  cargo install hyperfine"
    USE_HYPERFINE=0
else
    USE_HYPERFINE=1
fi

# Check for Rust runner
if [ ! -f "$RUST_RUNNER" ]; then
    echo "Building Rust validation runner..."
    (cd "$BASE_DIR/rust_runner" && cargo build --release) || {
        echo "Error: Failed to build Rust runner"
        exit 1
    }
fi

# Benchmark single method
benchmark_method() {
    local method=$1
    local n=$2
    local r_args=$3
    local rust_args=$4
    local dataset_type=$5

    # Convert method name to kebab-case for Rust CLI
    local rust_method="${method//_/-}"

    local dataset="$DATASETS_DIR/synthetic_${dataset_type}_n${n}.csv"

    if [ ! -f "$dataset" ]; then
        log_info "Skipping $method n=$n (dataset not found)"
        return
    fi

    log_info "Benchmarking $method (n=$n)..."

    # R benchmark
    local r_output="$RESULTS_DIR/r_${method}_n${n}.json"
    Rscript "$R_SCRIPTS_DIR/benchmark_method.R" \
        --method "$method" --data "$dataset" \
        $r_args \
        --iterations "$ITERATIONS" --warmup "$WARMUP" \
        --output "$r_output" 2>/dev/null || {
            log_info "  R benchmark failed for $method n=$n"
            return
        }

    # Rust benchmark
    local rust_output="$RESULTS_DIR/rust_${method}_n${n}.json"

    if [ "$USE_HYPERFINE" -eq 1 ]; then
        hyperfine --warmup "$WARMUP" --runs "$ITERATIONS" \
            --export-json "$rust_output" \
            "$RUST_RUNNER --method $rust_method --data $dataset $rust_args" 2>/dev/null
    else
        # Fallback: simple timing loop
        local times=()

        for i in $(seq 1 $WARMUP); do
            $RUST_RUNNER --method "$rust_method" --data "$dataset" $rust_args > /dev/null 2>&1
        done

        for i in $(seq 1 $ITERATIONS); do
            local start=$(date +%s%N)
            $RUST_RUNNER --method "$rust_method" --data "$dataset" $rust_args > /dev/null 2>&1
            local end=$(date +%s%N)
            local elapsed=$(( (end - start) / 1000 ))  # microseconds
            times+=($elapsed)
        done

        # Calculate stats
        IFS=$'\n' sorted=($(sort -n <<<"${times[*]}")); unset IFS
        local min=${sorted[0]}
        local max=${sorted[-1]}
        local median_idx=$((ITERATIONS / 2))
        local median=${sorted[$median_idx]}
        local sum=0
        for t in "${times[@]}"; do sum=$((sum + t)); done
        local mean=$((sum / ITERATIONS))

        cat > "$rust_output" << EOF
{
  "method": "$method",
  "n": $n,
  "iterations": $ITERATIONS,
  "timing": {
    "min_us": $min,
    "median_us": $median,
    "mean_us": $mean,
    "max_us": $max
  }
}
EOF
    fi

    log_info "  Completed $method n=$n"
}

# Target method
TARGET_METHOD="${1:-all}"
TARGET_SIZE="${2:-all}"

echo "========================================"
echo "Rust vs R Performance Benchmark"
echo "========================================"
echo "Iterations: $ITERATIONS"
echo "Warmup: $WARMUP"
echo ""

# Generate synthetic data if needed
if [ ! -f "$DATASETS_DIR/synthetic_ols_n100.csv" ]; then
    log_info "Generating synthetic datasets..."
    Rscript "$DATASETS_DIR/generate_synthetic.R" "$DATASETS_DIR"
fi

# Determine sample sizes to run
if [ "$TARGET_SIZE" = "all" ]; then
    SIZES=("${SAMPLE_SIZES[@]}")
else
    SIZES=("$TARGET_SIZE")
fi

# ============= OLS Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "ols" ]; then
    echo ""
    echo "--- OLS Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "ols" "$n" "-y y -x x1,x2,x3" \
            "-y y -x x1 x2 x3" "ols"
    done
fi

# ============= Panel FE Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "panel_fe" ]; then
    echo ""
    echo "--- Panel FE Benchmarks ---"
    # Panel datasets: n=200, 1000, 10000, 100000
    PANEL_SIZES=(200 1000 10000 100000)
    for n in "${PANEL_SIZES[@]}"; do
        benchmark_method "panel_fe" "$n" "-y y -x x1,x2 -e entity -t time" \
            "-y y -x x1 x2 -e entity -t time" "panel"
    done
fi

# ============= Logit Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "logit" ]; then
    echo ""
    echo "--- Logit Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "logit" "$n" "-y y -x x1,x2" \
            "-y y -x x1 x2" "binary"
    done
fi

# ============= K-Means Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "kmeans" ]; then
    echo ""
    echo "--- K-Means Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "kmeans" "$n" "-x x1,x2 -k 3 -s 42" \
            "-x x1 x2 -k 3 -s 42" "cluster"
    done
fi

# ============= PCA Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "pca" ]; then
    echo ""
    echo "--- PCA Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "pca" "$n" "-x x1,x2,x3,x4" \
            "-x x1 x2 x3 x4" "pca"
    done
fi

# ============= Sort Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "sort" ]; then
    echo ""
    echo "--- Sort Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "sort" "$n" "-x x1" \
            "-x x1" "ols"
    done
fi

# ============= Filter Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "filter" ]; then
    echo ""
    echo "--- Filter Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "filter" "$n" "-x x1" \
            "-x x1" "ols"
    done
fi

# ============= Group By Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "group_by" ]; then
    echo ""
    echo "--- Group By Benchmarks ---"
    # Use clustered data which has a cluster column
    for n in "${SIZES[@]}"; do
        benchmark_method "group_by" "$n" "-y y -c cluster" \
            "-y y -g cluster" "clustered"
    done
fi

# ============= OLS HC0 Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "ols_hc0" ]; then
    echo ""
    echo "--- OLS HC0 Robust SE Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "ols_robust" "$n" "-y y -x x1,x2,x3 -r HC0" \
            "-y y -x x1 x2 x3 --robust hc0" "ols"
    done
fi

# ============= OLS HC2 Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "ols_hc2" ]; then
    echo ""
    echo "--- OLS HC2 Robust SE Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "ols_robust" "$n" "-y y -x x1,x2,x3 -r HC2" \
            "-y y -x x1 x2 x3 --robust hc2" "ols"
    done
fi

# ============= OLS HC3 Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "ols_hc3" ]; then
    echo ""
    echo "--- OLS HC3 Robust SE Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "ols_robust" "$n" "-y y -x x1,x2,x3 -r HC3" \
            "-y y -x x1 x2 x3 --robust hc3" "ols"
    done
fi

# ============= Panel RE Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "panel_re" ]; then
    echo ""
    echo "--- Panel RE Benchmarks ---"
    PANEL_SIZES=(200 1000 10000 100000)
    for n in "${PANEL_SIZES[@]}"; do
        benchmark_method "panel_re" "$n" "-y y -x x1,x2 -e entity -t time" \
            "-y y -x x1 x2 -e entity -t time" "panel"
    done
fi

# ============= Probit Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "probit" ]; then
    echo ""
    echo "--- Probit Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "probit" "$n" "-y y -x x1,x2" \
            "-y y -x x1 x2" "binary"
    done
fi

# ============= DBSCAN Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "dbscan" ]; then
    echo ""
    echo "--- DBSCAN Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "dbscan" "$n" "-x x1,x2" \
            "-x x1 x2" "cluster"
    done
fi

# ============= Hierarchical Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "hierarchical" ]; then
    echo ""
    echo "--- Hierarchical Clustering Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "hierarchical" "$n" "-x x1,x2 -k 3" \
            "-x x1 x2 -k 3" "cluster"
    done
fi

# ============= Select Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "select" ]; then
    echo ""
    echo "--- Select Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "select" "$n" "-x x1,x2" \
            "-x x1 x2" "ols"
    done
fi

# ============= Standardize Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "standardize" ]; then
    echo ""
    echo "--- Standardize Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "standardize" "$n" "-x x1,x2,x3" \
            "-x x1 x2 x3" "ols"
    done
fi

# ============= Lag Benchmarks =============
if [ "$TARGET_METHOD" = "all" ] || [ "$TARGET_METHOD" = "lag" ]; then
    echo ""
    echo "--- Lag Benchmarks ---"
    for n in "${SIZES[@]}"; do
        benchmark_method "lag" "$n" "-x x1 -k 1" \
            "-x x1 -k 1" "ols"
    done
fi

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"

# Generate summary
"$SCRIPT_DIR/generate_report.sh"
