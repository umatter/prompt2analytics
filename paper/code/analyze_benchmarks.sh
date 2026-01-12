#!/bin/bash
# analyze_benchmarks.sh
#
# Analyzes benchmark comparison data between R and Rust implementations
# and generates visualizations for the JSS paper.
#
# Prerequisites:
#   - p2a CLI binary in PATH (or set P2A_BIN variable)
#   - jq for JSON parsing
#
# Usage:
#   ./analyze_benchmarks.sh
#
# Output:
#   - paper/figures/benchmark_*.png - Visualization files
#   - paper/code/comparison_data.csv - Combined comparison dataset

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPER_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$PAPER_DIR")"
FIGURES_DIR="$PAPER_DIR/figures"
OUTPUT_DIR="$SCRIPT_DIR"

# Benchmark data paths
R_RESULTS_DIR="$PROJECT_ROOT/performance/comparisons/r_comparison/results"
RUST_RESULTS_DIR="$PROJECT_ROOT/performance/results"

# p2a binary (adjust if needed)
P2A_BIN="${P2A_BIN:-p2a}"

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."

    if ! command -v jq &> /dev/null; then
        echo "Error: jq is required but not installed."
        echo "Install with: sudo apt-get install jq (Debian/Ubuntu)"
        exit 1
    fi

    if ! command -v "$P2A_BIN" &> /dev/null; then
        echo "Warning: p2a CLI not found in PATH."
        echo "Attempting to use local build..."
        P2A_BIN="$PROJECT_ROOT/target/release/p2a"
        if [ ! -f "$P2A_BIN" ]; then
            echo "Error: p2a binary not found. Build with: cargo build --release -p p2a-cli"
            exit 1
        fi
    fi

    echo "Using p2a: $P2A_BIN"
}

# Create output directories
create_directories() {
    mkdir -p "$FIGURES_DIR"
    mkdir -p "$OUTPUT_DIR"
}

# Find the most recent Rust benchmark file
find_rust_benchmark() {
    local latest
    latest=$(ls -t "$RUST_RESULTS_DIR"/rust_comprehensive_*.json 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "Error: No Rust benchmark files found in $RUST_RESULTS_DIR"
        exit 1
    fi
    echo "$latest"
}

# Find the most recent R comprehensive benchmark file
find_r_benchmark() {
    local latest
    latest=$(ls -t "$R_RESULTS_DIR"/r_comprehensive_*.csv 2>/dev/null | head -1)
    if [ -z "$latest" ]; then
        echo "Warning: No comprehensive R benchmark file found, using combined_results.csv"
        echo "$R_RESULTS_DIR/combined_results.csv"
    else
        echo "$latest"
    fi
}

# Create comparison dataset directly from source files
create_comparison_dataset() {
    local rust_file="$1"
    local r_file="$2"
    local output_file="$3"

    echo "Creating comparison dataset..."

    # Write header
    echo "category,method,n,r_median_us,rust_median_us,speedup" > "$output_file"

    # Helper function to get R timing
    get_r_time() {
        local method="$1"
        local n="$2"
        grep "^\"$method\",$n," "$r_file" 2>/dev/null | head -1 | cut -d',' -f6 | tr -d '"'
    }

    # Helper function to get Rust timing
    get_rust_time() {
        local method="$1"
        local variant="$2"
        local n="$3"
        jq -r ".[] | select(.method == \"$method\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1
    }

    # Process OLS benchmarks
    for n in 100 1000 10000; do
        local r_time=$(get_r_time "OLS" "$n")
        local rust_time=$(get_rust_time "OLS" "standard" "$n")

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "Regression,OLS_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process OLS+HC1 benchmarks
    for n in 100 1000 10000; do
        local r_time=$(get_r_time "OLS+HC1" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"OLS\" and .variant == \"HC1\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "Regression,OLS_HC1_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process Fixed Effects benchmarks (using FE_plm from R)
    for n in 100 1000 5000; do
        local r_time=$(get_r_time "FE_plm" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"Panel_FE\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "Panel,FE_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process Logit benchmarks
    for n in 100 500 1000; do
        local r_time=$(get_r_time "Logit" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"Logit\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "Discrete,Logit_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process K-Means benchmarks
    for n in 100 1000 5000; do
        local r_time=$(get_r_time "K-Means" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"KMeans\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "ML,KMeans_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process PCA benchmarks
    for n in 100 1000 5000; do
        local r_time=$(get_r_time "PCA" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"PCA\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "ML,PCA_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process ARIMA benchmarks
    for n in 100 200 500; do
        local r_time=$(get_r_time "ARIMA" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"ARIMA\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "TimeSeries,ARIMA_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    # Process MSTL benchmarks
    for n in 100 200 500; do
        local r_time=$(get_r_time "MSTL" "$n")
        local rust_time=$(jq -r ".[] | select(.method == \"MSTL\" and .n == $n) | .time_median_us" "$rust_file" 2>/dev/null | head -1)

        if [ -n "$r_time" ] && [ -n "$rust_time" ] && [ "$rust_time" != "null" ]; then
            local speedup=$(echo "scale=2; $r_time / $rust_time" | bc 2>/dev/null || echo "NA")
            echo "TimeSeries,MSTL_n${n},$n,$r_time,$rust_time,$speedup" >> "$output_file"
        fi
    done

    echo "Comparison dataset created: $output_file"
    echo ""
    echo "Dataset preview:"
    cat "$output_file"
}

# Generate visualizations using p2a CLI
generate_visualizations() {
    local comparison_csv="$1"
    local session_file="$OUTPUT_DIR/benchmark_session.json"

    # Clean up old session
    rm -f "$session_file"

    echo ""
    echo "Generating visualizations..."

    # Load comparison data
    echo "Loading dataset..."
    "$P2A_BIN" --session "$session_file" data load "$comparison_csv" --name benchmarks

    # Data summary
    echo ""
    echo "Benchmark data summary:"
    "$P2A_BIN" --session "$session_file" data describe benchmarks || true

    # Generate scatter plot: R vs Rust timing
    echo ""
    echo "Creating R vs Rust timing scatter plot..."
    "$P2A_BIN" --session "$session_file" viz scatter benchmarks \
        --x-col r_median_us \
        --y-col rust_median_us \
        --title "R vs Rust Execution Time (microseconds)" \
        --file "$FIGURES_DIR/benchmark_scatter.png" || echo "Warning: Scatter plot generation failed"

    # Generate histogram of speedup values
    echo ""
    echo "Creating speedup histogram..."
    "$P2A_BIN" --session "$session_file" viz histogram benchmarks \
        --col speedup \
        --bins 10 \
        --title "Distribution of Speedup Factors (R/Rust)" \
        --file "$FIGURES_DIR/benchmark_speedup_hist.png" || echo "Warning: Histogram generation failed"

    echo ""
    echo "Visualizations saved to: $FIGURES_DIR/"
}

# Generate summary statistics
generate_summary() {
    local comparison_csv="$1"

    echo ""
    echo "========================================"
    echo "BENCHMARK SUMMARY"
    echo "========================================"
    echo ""

    # Count total comparisons
    local total
    total=$(tail -n +2 "$comparison_csv" | wc -l)
    echo "Total benchmarks compared: $total"

    # Calculate average speedup
    local avg_speedup
    avg_speedup=$(tail -n +2 "$comparison_csv" | \
        grep -v "NA" | \
        cut -d',' -f6 | \
        awk '{sum+=$1; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    echo "Average speedup (R/Rust): ${avg_speedup}x"

    # Find max speedup
    local max_line
    max_line=$(tail -n +2 "$comparison_csv" | \
        grep -v "NA" | \
        sort -t',' -k6 -rn | \
        head -1)
    if [ -n "$max_line" ]; then
        echo "Highest speedup: $(echo "$max_line" | cut -d',' -f2) at $(echo "$max_line" | cut -d',' -f6)x"
    fi

    # Find min speedup (where Rust is slowest relative to R)
    local min_line
    min_line=$(tail -n +2 "$comparison_csv" | \
        grep -v "NA" | \
        sort -t',' -k6 -n | \
        head -1)
    if [ -n "$min_line" ]; then
        echo "Lowest speedup: $(echo "$min_line" | cut -d',' -f2) at $(echo "$min_line" | cut -d',' -f6)x"
    fi

    echo ""
    echo "========================================"
    echo "BENCHMARK DETAILS BY CATEGORY"
    echo "========================================"
    echo ""

    for category in Regression Panel Discrete ML TimeSeries; do
        local cat_data
        cat_data=$(grep "^$category," "$comparison_csv" 2>/dev/null)
        if [ -n "$cat_data" ]; then
            echo "--- $category ---"
            echo "$cat_data" | while IFS=',' read -r cat method n r_time rust_time speedup; do
                printf "  %-20s n=%-5s R: %10.2f us  Rust: %10.2f us  Speedup: %sx\n" \
                    "$method" "$n" "$r_time" "$rust_time" "$speedup"
            done
            echo ""
        fi
    done
}

# Main execution
main() {
    echo "========================================="
    echo "Benchmark Analysis Script"
    echo "========================================="
    echo ""

    check_prerequisites
    create_directories

    # Find benchmark files
    RUST_FILE=$(find_rust_benchmark)
    R_FILE=$(find_r_benchmark)

    echo "Using Rust benchmark: $RUST_FILE"
    echo "Using R benchmark: $R_FILE"
    echo ""

    # Create comparison dataset
    COMPARISON_CSV="$OUTPUT_DIR/comparison_data.csv"
    create_comparison_dataset "$RUST_FILE" "$R_FILE" "$COMPARISON_CSV"

    # Generate visualizations
    generate_visualizations "$COMPARISON_CSV"

    # Generate summary
    generate_summary "$COMPARISON_CSV"

    echo ""
    echo "========================================="
    echo "Analysis complete!"
    echo "========================================="
    echo ""
    echo "Output files:"
    echo "  - Comparison data: $COMPARISON_CSV"
    ls -la "$FIGURES_DIR"/*.png 2>/dev/null && echo "" || echo "  - No figures generated (check p2a CLI)"
    echo ""
    echo "Use these figures in the paper with:"
    echo "  \\includegraphics{figures/benchmark_scatter}"
    echo "  \\includegraphics{figures/benchmark_speedup_hist}"
}

# Run main
main "$@"
