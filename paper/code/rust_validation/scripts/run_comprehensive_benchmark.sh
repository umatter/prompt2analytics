#!/bin/bash
# =============================================================================
# run_comprehensive_benchmark.sh
# Comprehensive benchmark: R vs Rust across ~50 methods
# Outputs benchmark_summary.json for paper figures
# =============================================================================

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$BASE_DIR"
RUST_RUNNER="$BASE_DIR/rust_runner/target/release/run-validation"
R_METHODS_DIR="$BASE_DIR/r_scripts/methods"
R_BENCHMARK="$BASE_DIR/r_scripts/benchmark_method.R"
DATASETS_DIR="$BASE_DIR/datasets/synthetic"
RESULTS_DIR="$BASE_DIR/results"
SUMMARY_DIR="$RESULTS_DIR/summaries"

# Configuration
ITERATIONS=${ITERATIONS:-20}
N=${N:-100000}
DATE=$(which date 2>/dev/null || echo "/usr/bin/date")

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p "$RESULTS_DIR/benchmarks" "$SUMMARY_DIR"

# =============================================================================
# Timing helpers
# =============================================================================

# Time a Rust method using --benchmark mode (loads data once, times computation only)
# Usage: time_rust <method> <data_file> <extra_args...>
time_rust() {
    local method="$1"
    local data="$2"
    shift 2
    local extra_args="$@"

    local output
    output=$("$RUST_RUNNER" --method "$method" --data "$data" \
        --benchmark --iterations "$ITERATIONS" --warmup 3 \
        $extra_args 2>/dev/null)

    if [ $? -ne 0 ]; then
        echo "-1"
        return
    fi

    # Output the JSON timing block for Python processing
    echo "$output"
}

# Time an R method via benchmark_method.R, return JSON with timing
# Usage: time_r <method> <data_file> <extra_r_args>
time_r() {
    local method="$1"
    local data="$2"
    local iterations="$3"
    local output_file="$4"
    shift 4
    local extra_args="$@"

    if [ ! -f "$R_METHODS_DIR/${method}.R" ]; then
        echo "SKIP"
        return
    fi

    Rscript "$R_BENCHMARK" \
        --method "$method" \
        --data "$data" \
        --iterations "$iterations" \
        --warmup 3 \
        --output "$output_file" \
        $extra_args > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        echo "FAIL"
        return
    fi
    echo "OK"
}

# =============================================================================
# Benchmark runner: processes one method
# =============================================================================

# benchmark_method <label> <r_method> <rust_method> <dataset_file> <r_args> <rust_args>
benchmark_method() {
    local label="$1"
    local r_method="$2"
    local rust_method="$3"
    local dataset="$4"
    local r_args="$5"
    local rust_args="$6"
    local r_output="$RESULTS_DIR/benchmarks/r_${label}_n${N}.json"

    echo -e "${BLUE}[$label]${NC} R=$r_method, Rust=$rust_method, data=$(basename $dataset)"

    # R benchmark
    local r_status
    r_status=$(time_r "$r_method" "$dataset" "$ITERATIONS" "$r_output" $r_args)

    local r_median_us=0 r_min_us=0 r_max_us=0
    if [ "$r_status" = "OK" ] && [ -f "$r_output" ]; then
        r_median_us=$(python3 -c "import json; d=json.load(open('$r_output')); print(int(d.get('timing',{}).get('median_us', 0)))" 2>/dev/null || echo 0)
        r_min_us=$(python3 -c "import json; d=json.load(open('$r_output')); print(int(d.get('timing',{}).get('min_us', 0)))" 2>/dev/null || echo 0)
        r_max_us=$(python3 -c "import json; d=json.load(open('$r_output')); print(int(d.get('timing',{}).get('max_us', 0)))" 2>/dev/null || echo 0)
        echo -e "  R: ${GREEN}${r_median_us} us${NC}"
    elif [ "$r_status" = "SKIP" ]; then
        echo -e "  R: ${YELLOW}SKIP (no R script)${NC}"
    else
        echo -e "  R: ${RED}FAIL${NC}"
    fi

    # Rust benchmark (using --benchmark mode: loads data once, times computation only)
    local rust_output
    rust_output=$(time_rust "$rust_method" "$dataset" $rust_args)

    local rust_median_us=0 rust_min_us=0 rust_max_us=0 speedup=0
    if [ "$rust_output" = "-1" ]; then
        echo -e "  Rust: ${RED}FAIL${NC}"
    else
        # Parse JSON timing output from --benchmark mode
        read rust_median_us rust_min_us rust_max_us <<< $(python3 -c "
import json, sys
try:
    d = json.loads('''$rust_output''')
    t = d.get('timing', {})
    print(int(t.get('median_us', 0)), int(t.get('min_us', 0)), int(t.get('max_us', 0)))
except:
    print('0 0 0')
" 2>/dev/null || echo "0 0 0")
        echo -e "  Rust: ${GREEN}${rust_median_us} us${NC}"
    fi

    # Calculate speedup
    if [ "$r_median_us" -gt 0 ] && [ "$rust_median_us" -gt 0 ]; then
        speedup=$(python3 -c "print(round($r_median_us / $rust_median_us, 2))")
        echo -e "  Speedup: ${GREEN}${speedup}x${NC}"
    fi

    # Write result JSON fragment
    python3 -c "
import json
result = {
    'n': $N,
    'r_median_us': $r_median_us,
    'r_min_us': $r_min_us,
    'r_max_us': $r_max_us,
    'rust_median_us': $rust_median_us,
    'rust_min_us': $rust_min_us,
    'rust_max_us': $rust_max_us,
    'speedup': round($r_median_us / $rust_median_us, 2) if $rust_median_us > 0 and $r_median_us > 0 else 0
}
with open('$RESULTS_DIR/benchmarks/${label}_n${N}.json', 'w') as f:
    json.dump(result, f, indent=2)
"
    echo ""
}

# =============================================================================
# MAIN: Define and run all benchmarks
# =============================================================================

echo "============================================"
echo "  Comprehensive R vs Rust Benchmark Suite"
echo "============================================"
echo "Iterations: $ITERATIONS"
echo "Sample size: $N"
echo "Runner: $RUST_RUNNER"
echo ""

# Check prerequisites
if [ ! -f "$RUST_RUNNER" ]; then
    echo -e "${RED}Error: Rust runner not found at $RUST_RUNNER${NC}"
    echo "Build with: cd $BASE_DIR/rust_runner && cargo build --release"
    exit 1
fi

# ---- REGRESSION ----
echo -e "\n${BLUE}=== REGRESSION ===${NC}\n"

OLS_DATA="$DATASETS_DIR/synthetic_ols_n${N}.csv"

benchmark_method "ols" "ols" "ols" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

benchmark_method "ols_hc0" "ols_hc0" "ols-hc0" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

benchmark_method "ols_hc1" "ols_hc1" "ols-hc1" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

benchmark_method "ols_hc3" "ols_hc3" "ols-hc3" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

benchmark_method "ols_hc2" "ols_hc2" "ols-hc2" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

benchmark_method "diagnostics" "diagnostics" "diagnostics" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

benchmark_method "nls" "nls" "nls" "$OLS_DATA" \
    "-y y -x x1" \
    "-y y -x x1"

CLUST_DATA="$DATASETS_DIR/synthetic_clustered_n${N}.csv"
benchmark_method "ols_clustered" "ols_clustered" "ols-clustered" "$CLUST_DATA" \
    "-y y -x x1,x2 -c cluster" \
    "-y y -x x1 x2 --cluster-var cluster"

benchmark_method "quantreg" "quantreg" "quantile-reg" "$OLS_DATA" \
    "-y y -x x1,x2,x3" \
    "-y y -x x1 x2 x3"

# GLS and LOESS skipped at n=100K (O(n^2)/O(n^3) methods)
if [ "$N" -le 10000 ]; then
    benchmark_method "gls" "gls_regression" "gls" "$OLS_DATA" \
        "-y y -x x1,x2,x3" \
        "-y y -x x1 x2 x3"

    benchmark_method "loess" "loess" "loess" "$OLS_DATA" \
        "-y y -x x1" \
        "-y y -x x1"
fi

# ---- PANEL ----
echo -e "\n${BLUE}=== PANEL ===${NC}\n"

PANEL_DATA="$DATASETS_DIR/synthetic_panel_n${N}.csv"

benchmark_method "panel_fe" "panel_fe" "panel-fe" "$PANEL_DATA" \
    "-y y -x x1,x2 -e entity -t time" \
    "-y y -x x1 x2 -e entity -t time"

benchmark_method "panel_re" "panel_re" "panel-re" "$PANEL_DATA" \
    "-y y -x x1,x2 -e entity -t time" \
    "-y y -x x1 x2 -e entity -t time"

benchmark_method "hausman" "hausman" "hausman" "$PANEL_DATA" \
    "-y y -x x1,x2 -e entity -t time" \
    "-y y -x x1 x2 -e entity -t time"

benchmark_method "hdfe" "panel_hdfe" "hdfe" "$PANEL_DATA" \
    "-y y -x x1,x2 -e entity -t time" \
    "-y y -x x1 x2 --fe-cols entity time"

# ---- DISCRETE CHOICE ----
echo -e "\n${BLUE}=== DISCRETE CHOICE ===${NC}\n"

BINARY_DATA="$DATASETS_DIR/synthetic_binary_n${N}.csv"

benchmark_method "logit" "logit" "logit" "$BINARY_DATA" \
    "-y y -x x1,x2" \
    "-y y -x x1 x2"

benchmark_method "probit" "probit" "probit" "$BINARY_DATA" \
    "-y y -x x1,x2" \
    "-y y -x x1 x2"

COUNT_DATA="$DATASETS_DIR/synthetic_count_n${N}.csv"
benchmark_method "negbin" "negbin" "neg-bin" "$COUNT_DATA" \
    "-y y -x x1,x2" \
    "-y y -x x1 x2"

# ---- CAUSAL INFERENCE ----
echo -e "\n${BLUE}=== CAUSAL INFERENCE ===${NC}\n"

IV_DATA="$DATASETS_DIR/synthetic_iv_n${N}.csv"
benchmark_method "iv2sls" "iv_2sls" "iv2sls" "$IV_DATA" \
    "-y y -x x_exog --endog_vars x_endog -i z1,z2" \
    "-y y -x x_exog --endog-vars x_endog --instruments z1 z2"

# ---- SURVIVAL ----
echo -e "\n${BLUE}=== SURVIVAL ===${NC}\n"

SURV_DATA="$DATASETS_DIR/synthetic_survival_n${N}.csv"

benchmark_method "kaplan_meier" "kaplan_meier" "kaplan-meier" "$SURV_DATA" \
    "-y time --event_var event" \
    "-t time --event-var event"

benchmark_method "cox_ph" "cox_ph" "cox-ph" "$SURV_DATA" \
    "-y time -x x1,x2 --event_var event" \
    "-t time -x x1 x2 --event-var event"

# ---- FORECASTING / TIME SERIES ----
echo -e "\n${BLUE}=== FORECASTING ===${NC}\n"

TS_DATA="$DATASETS_DIR/synthetic_timeseries_n${N}.csv"

benchmark_method "arima" "arima" "arima" "$TS_DATA" \
    "-y y -t t" \
    "-y y -t t"

benchmark_method "holt_winters" "holt_winters" "holt-winters" "$TS_DATA" \
    "-y y -t t" \
    "-y y -t t"

benchmark_method "stl" "stl" "stl" "$TS_DATA" \
    "-y y -t t" \
    "-y y -t t"

benchmark_method "ar" "ar" "ar" "$TS_DATA" \
    "-y y -t t" \
    "-y y -t t"

benchmark_method "mstl" "mstl" "mstl" "$TS_DATA" \
    "-y y -t t" \
    "-y y -t t"

benchmark_method "changepoint" "changepoint" "changepoint" "$TS_DATA" \
    "-y y" \
    "-y y"

MVAR_DATA="$DATASETS_DIR/synthetic_multivar_ts_n${N}.csv"

benchmark_method "var_model" "var_model" "var" "$MVAR_DATA" \
    "-y y1 -x y2 -t t" \
    "-y y1 -x y2 -t t"

benchmark_method "granger" "granger" "granger-causality" "$MVAR_DATA" \
    "-y y1 -x y2 -t t" \
    "-y y1 -x y2 -t t"

# ---- STATS / HYPOTHESIS TESTS ----
echo -e "\n${BLUE}=== STATISTICS ===${NC}\n"

FACTOR_DATA="$DATASETS_DIR/synthetic_factor_n${N}.csv"

benchmark_method "anova_oneway" "anova_oneway" "anova-oneway" "$FACTOR_DATA" \
    "-y y -x group --factor_var group" \
    "-y y --factor-var group"

benchmark_method "anova_twoway" "anova_twoway" "anova-twoway" "$FACTOR_DATA" \
    "-y y -x group,factor2 --factor_var group --factor2_var factor2" \
    "-y y --factor-var group --factor2-var factor2"

benchmark_method "ttest_one_sample" "ttest_one_sample" "t-test-one-sample" "$OLS_DATA" \
    "-y y" \
    "-y y"

benchmark_method "ttest_two_sample" "ttest_two_sample" "t-test-two-sample" "$OLS_DATA" \
    "-y x1 -x x2" \
    "-x x1 x2"

benchmark_method "shapiro_wilk" "shapiro_wilk" "shapiro-wilk" "$OLS_DATA" \
    "-y y" \
    "-y y"

# Fisher exact is O(n^2) with simulation — skip at large n
if [ "$N" -le 10000 ]; then
    benchmark_method "fisher_exact" "fisher_exact" "fisher-exact" "$BINARY_DATA" \
        "-y y -x x1" \
        "-x y x1"
fi

benchmark_method "wilcoxon" "wilcoxon" "wilcoxon" "$OLS_DATA" \
    "-y x1 -x x2" \
    "-x x1 x2"

benchmark_method "kruskal_wallis" "kruskal_wallis" "kruskal-wallis" "$FACTOR_DATA" \
    "-y y -x group --factor_var group" \
    "-y y --factor-var group"

benchmark_method "chisq_gof" "chisq_gof" "chisq-gof" "$FACTOR_DATA" \
    "-y y -x group --factor_var group" \
    "-y y --factor-var group"

benchmark_method "acf" "acf_pacf" "acf-pacf" "$TS_DATA" \
    "-y y" \
    "-y y"

# ---- ML ----
echo -e "\n${BLUE}=== MACHINE LEARNING ===${NC}\n"

CLUSTER_DATA="$DATASETS_DIR/synthetic_cluster_n${N}.csv"

benchmark_method "kmeans" "kmeans" "kmeans" "$CLUSTER_DATA" \
    "-y x1 -x x2 -k 3" \
    "-x x1 x2 -k 3"

benchmark_method "dbscan" "dbscan" "dbscan" "$CLUSTER_DATA" \
    "-y x1 -x x2" \
    "-x x1 x2"

PCA_DATA="$DATASETS_DIR/synthetic_pca_n${N}.csv"
benchmark_method "pca" "pca" "pca" "$PCA_DATA" \
    "-y x1 -x x2,x3,x4" \
    "-x x1 x2 x3 x4"

# Hierarchical clustering is O(n^2) memory — skip at large n
if [ "$N" -le 10000 ]; then
    benchmark_method "hierarchical" "hierarchical" "hierarchical" "$CLUSTER_DATA" \
        "-y x1 -x x2" \
        "-x x1 x2"
fi

# ---- DATA MUNGING ----
echo -e "\n${BLUE}=== DATA MUNGING ===${NC}\n"

benchmark_method "sort" "sort" "sort" "$OLS_DATA" \
    "-y y --sort_col y" \
    "--sort-col y"

benchmark_method "filter" "filter" "filter" "$OLS_DATA" \
    "-y x1 -x x1" \
    "-x x1"

benchmark_method "group_by" "group_by" "group-by" "$CLUST_DATA" \
    "-y y -g cluster" \
    "-y y -g cluster"

benchmark_method "standardize" "standardize" "standardize" "$OLS_DATA" \
    "-y y -x x1,x2" \
    "-x x1 x2"

benchmark_method "select" "select" "select" "$OLS_DATA" \
    "-y y -x x1" \
    "-x y x1"

# lag/lead/diff omitted — trivial operations dominated by CLI overhead

# =============================================================================
# AGGREGATE: Build benchmark_summary.json
# =============================================================================

echo ""
echo -e "${BLUE}=== Building Summary ===${NC}"

python3 << 'PYEOF'
import json, glob, os

results_dir = os.environ.get('RESULTS_DIR', 'results')
summary_dir = os.environ.get('SUMMARY_DIR', 'results/summaries')
n = int(os.environ.get('N', '100000'))

methods = {}
for f in sorted(glob.glob(f"{results_dir}/benchmarks/*_n{n}.json")):
    basename = os.path.basename(f)
    # Remove _nXXXXXX.json suffix and r_/rust_ prefix
    label = basename.replace(f"_n{n}.json", "")
    if label.startswith("r_"):
        continue  # Skip R-only output files

    with open(f) as fh:
        data = json.load(fh)

    if data.get('r_median_us', 0) == 0 and data.get('rust_median_us', 0) == 0:
        continue

    if label not in methods:
        methods[label] = []
    methods[label].append(data)

# Calculate overall stats
all_speedups = []
for method_results in methods.values():
    for r in method_results:
        if r.get('speedup', 0) > 0:
            all_speedups.append(r['speedup'])

summary = {
    "generated": __import__('datetime').datetime.now().isoformat(),
    "overall_stats": {
        "mean_speedup": round(sum(all_speedups) / len(all_speedups), 2) if all_speedups else 0,
        "median_speedup": round(sorted(all_speedups)[len(all_speedups)//2], 2) if all_speedups else 0,
        "min_speedup": round(min(all_speedups), 2) if all_speedups else 0,
        "max_speedup": round(max(all_speedups), 2) if all_speedups else 0,
        "n_benchmarks": len(all_speedups)
    },
    "methods": methods
}

output_path = f"{summary_dir}/benchmark_summary.json"
with open(output_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary written to: {output_path}")
print(f"Methods benchmarked: {len(methods)}")
print(f"Speedup range: {summary['overall_stats']['min_speedup']}x - {summary['overall_stats']['max_speedup']}x")
print(f"Median speedup: {summary['overall_stats']['median_speedup']}x")

# Print table
print(f"\n{'Method':<20} {'R (ms)':>10} {'Rust (ms)':>10} {'Speedup':>10}")
print("-" * 55)
for method, results in sorted(methods.items()):
    for r in results:
        r_ms = r.get('r_median_us', 0) / 1000
        rust_ms = r.get('rust_median_us', 0) / 1000
        speedup = r.get('speedup', 0)
        status = f"{speedup:.1f}x" if speedup > 0 else "N/A"
        print(f"{method:<20} {r_ms:>10.1f} {rust_ms:>10.1f} {status:>10}")
PYEOF

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Benchmark Complete${NC}"
echo -e "${GREEN}============================================${NC}"
echo "Results: $SUMMARY_DIR/benchmark_summary.json"
echo ""
echo "Next: Generate figures with:"
echo "  cd ../.. && Rscript code/fig_benchmark_histogram.R"
echo "  cd ../.. && Rscript code/fig_benchmark_speedup.R"
echo "  cd ../.. && Rscript code/fig_benchmark_execution.R"
