#!/bin/bash
# quick_benchmark.sh - Quick R vs Rust benchmark for paper figures
# Focus on n=100,000 with 30 iterations for each method

# Continue on individual method errors
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$BASE_DIR/results/benchmarks"
DATASETS_DIR="$BASE_DIR/datasets"
R_SCRIPTS_DIR="$BASE_DIR/r_scripts"
RUST_RUNNER="$BASE_DIR/rust_runner/target/release/run-validation"

mkdir -p "$RESULTS_DIR"

ITERATIONS=${ITERATIONS:-30}
WARMUP=${WARMUP:-3}

echo "========================================"
echo "Quick Benchmark: R vs Rust at n=100,000"
echo "Iterations: $ITERATIONS, Warmup: $WARMUP"
echo "========================================"

# Benchmark a single method: R and Rust
# Args: label n r_method r_args rust_method rust_args dataset_type
benchmark() {
    local label=$1
    local n=$2
    local r_method=$3
    local r_args=$4
    local rust_method=$5
    local rust_args=$6
    local dataset_type=$7

    local dataset="$DATASETS_DIR/synthetic_${dataset_type}_n${n}.csv"
    if [ ! -f "$dataset" ]; then
        echo "SKIP $label n=$n (dataset not found: $dataset)"
        return
    fi

    echo ""
    echo "--- $label (n=$n) ---"

    # R benchmark
    local r_output="$RESULTS_DIR/r_${label}_n${n}.json"
    echo "  R benchmark..."
    if ! Rscript "$R_SCRIPTS_DIR/benchmark_method.R" \
        --method "$r_method" --data "$dataset" \
        $r_args \
        --iterations "$ITERATIONS" --warmup "$WARMUP" \
        --output "$r_output" 2>/dev/null; then
        echo "  R benchmark FAILED, skipping"
        return
    fi
    local r_median=$(python3 -c "import json; d=json.load(open('$r_output')); print(f\"{d['timing']['median_us']:.1f}\")")
    echo "  R median: ${r_median} us"

    # Rust benchmark (simple timing loop)
    local rust_output="$RESULTS_DIR/rust_${label}_n${n}.json"
    echo "  Rust benchmark..."

    # Test if Rust method works first
    if ! "$RUST_RUNNER" --method "$rust_method" --data "$dataset" $rust_args > /dev/null 2>&1; then
        echo "  Rust method FAILED, skipping"
        return
    fi

    # Warmup
    for i in $(seq 1 $WARMUP); do
        "$RUST_RUNNER" --method "$rust_method" --data "$dataset" $rust_args > /dev/null 2>&1
    done

    # Measure
    local times=()
    for i in $(seq 1 $ITERATIONS); do
        local start_ns=$(date +%s%N)
        "$RUST_RUNNER" --method "$rust_method" --data "$dataset" $rust_args > /dev/null 2>&1
        local end_ns=$(date +%s%N)
        local elapsed_us=$(( (end_ns - start_ns) / 1000 ))
        times+=($elapsed_us)
    done

    # Calculate stats with python (join times with commas)
    local times_csv=$(IFS=,; echo "${times[*]}")
    python3 -c "
import json, sys
times = [$times_csv]
times.sort()
n = len(times)
result = {
    'method': '$label',
    'n': $n,
    'iterations': $ITERATIONS,
    'timing': {
        'min_us': times[0],
        'median_us': times[n//2],
        'mean_us': sum(times)/n,
        'max_us': times[-1]
    }
}
json.dump(result, open('$rust_output', 'w'), indent=2)
print(f\"  Rust median: {times[n//2]} us\")
speedup = float('$r_median') / times[n//2] if times[n//2] > 0 else 0
print(f\"  Speedup: {speedup:.1f}x\")
"
}

# Run all methods at n=100,000
# Format: benchmark label n r_method "r_args" rust_method "rust_args" dataset_type

# OLS
benchmark "ols" 100000 \
    "ols" "-y y -x x1,x2,x3" \
    "ols" "-y y -x x1 x2 x3" \
    "ols"

# OLS HC1 (robust SE)
benchmark "ols_hc1" 100000 \
    "ols_hc1" "-y y -x x1,x2,x3" \
    "ols-hc1" "-y y -x x1 x2 x3" \
    "ols"

# OLS HC3 (robust SE, computationally intensive)
benchmark "ols_hc3" 100000 \
    "ols_hc3" "-y y -x x1,x2,x3" \
    "ols-hc3" "-y y -x x1 x2 x3" \
    "ols"

# Panel FE
benchmark "panel_fe" 100000 \
    "panel_fe" "-y y -x x1,x2 -e entity -t time" \
    "panel-fe" "-y y -x x1 x2 -e entity -t time" \
    "panel"

# Panel RE
benchmark "panel_re" 100000 \
    "panel_re" "-y y -x x1,x2 -e entity -t time" \
    "panel-re" "-y y -x x1 x2 -e entity -t time" \
    "panel"

# Logit
benchmark "logit" 100000 \
    "logit" "-y y -x x1,x2" \
    "logit" "-y y -x x1 x2" \
    "binary"

# Probit
benchmark "probit" 100000 \
    "probit" "-y y -x x1,x2" \
    "probit" "-y y -x x1 x2" \
    "binary"

# K-Means
benchmark "kmeans" 100000 \
    "kmeans" "-x x1,x2 -k 3 -s 42" \
    "kmeans" "-x x1 x2 -k 3 -s 42" \
    "cluster"

# PCA
benchmark "pca" 100000 \
    "pca" "-x x1,x2,x3,x4" \
    "pca" "-x x1 x2 x3 x4" \
    "pca"

# DBSCAN
benchmark "dbscan" 100000 \
    "dbscan" "-x x1,x2" \
    "dbscan" "-x x1 x2" \
    "cluster"

# Hierarchical clustering
benchmark "hierarchical" 100000 \
    "hierarchical" "-x x1,x2 -k 3" \
    "hierarchical" "-x x1 x2 -k 3" \
    "cluster"

# Sort
benchmark "sort" 100000 \
    "sort" "-x x1" \
    "sort" "-x x1" \
    "ols"

# Filter
benchmark "filter" 100000 \
    "filter" "-x x1" \
    "filter" "-x x1" \
    "ols"

# Group By
benchmark "group_by" 100000 \
    "group_by" "-y y -c cluster" \
    "group-by" "-y y -g cluster" \
    "clustered"

# Select
benchmark "select" 100000 \
    "select" "-x x1,x2" \
    "select" "-x x1 x2" \
    "ols"

# Standardize
benchmark "standardize" 100000 \
    "standardize" "-x x1,x2,x3" \
    "standardize" "-x x1 x2 x3" \
    "ols"

# Lag
benchmark "lag" 100000 \
    "lag" "-x x1 -k 1" \
    "lag" "-x x1 -k 1" \
    "ols"

echo ""
echo "========================================"
echo "Benchmark Complete"
echo "========================================"

# Generate summary JSON
echo "Generating summary..."
python3 << 'PYEOF'
import json, glob, os

results_dir = os.environ.get('RESULTS_DIR', 'results/benchmarks')
summary = {"generated": "unknown", "methods": {}, "overall_stats": {}}

# Find all paired results (r_* and rust_*)
r_files = sorted(glob.glob(os.path.join(results_dir, "r_*_n*.json")))

for r_file in r_files:
    base = os.path.basename(r_file)
    # Extract method and n: r_ols_n100000.json -> ols, 100000
    parts = base[2:-5].split("_n")
    if len(parts) != 2:
        continue
    method = parts[0]
    n = int(parts[1])

    rust_file = os.path.join(results_dir, f"rust_{method}_n{n}.json")
    if not os.path.exists(rust_file):
        continue

    with open(r_file) as f:
        r_data = json.load(f)
    with open(rust_file) as f:
        rust_data = json.load(f)

    r_timing = r_data.get("timing", {})
    rust_timing = rust_data.get("timing", {})

    r_median = r_timing.get("median_us", 0)
    rust_median = rust_timing.get("median_us", 0)
    speedup = round(r_median / rust_median, 2) if rust_median > 0 else 0

    entry = {
        "n": n,
        "r_median_us": r_median,
        "r_min_us": r_timing.get("min_us", 0),
        "r_max_us": r_timing.get("max_us", 0),
        "rust_median_us": rust_median,
        "rust_min_us": rust_timing.get("min_us", 0),
        "rust_max_us": rust_timing.get("max_us", 0),
        "speedup": speedup
    }

    if method not in summary["methods"]:
        summary["methods"][method] = []
    summary["methods"][method].append(entry)

# Calculate overall stats
all_speedups = []
for method_entries in summary["methods"].values():
    for entry in method_entries:
        if entry["speedup"] > 0:
            all_speedups.append(entry["speedup"])

if all_speedups:
    summary["overall_stats"] = {
        "mean_speedup": round(sum(all_speedups)/len(all_speedups), 2),
        "median_speedup": round(sorted(all_speedups)[len(all_speedups)//2], 2),
        "min_speedup": round(min(all_speedups), 2),
        "max_speedup": round(max(all_speedups), 2),
        "n_benchmarks": len(all_speedups)
    }

summary_path = os.path.join(os.path.dirname(results_dir), "summaries", "benchmark_summary.json")
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary written to: {summary_path}")
print(f"Methods benchmarked: {len(summary['methods'])}")
if all_speedups:
    print(f"Mean speedup: {summary['overall_stats']['mean_speedup']}x")
    print(f"Median speedup: {summary['overall_stats']['median_speedup']}x")
    print(f"Range: {summary['overall_stats']['min_speedup']}x - {summary['overall_stats']['max_speedup']}x")
PYEOF

echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Summary at: $BASE_DIR/results/summaries/benchmark_summary.json"
