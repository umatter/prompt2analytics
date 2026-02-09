#!/bin/bash
# Compare multi-turn evaluation results between V1 and V2
#
# Usage: ./scripts/compare_eval_versions.sh [model_pattern]
#
# Example:
#   ./scripts/compare_eval_versions.sh           # Compare all models
#   ./scripts/compare_eval_versions.sh gpt-4o    # Compare gpt-4o results

set -e
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

V1_DIR="$PROJECT_DIR/results/multi_turn"
V2_DIR="$PROJECT_DIR/results/multi_turn_v2"

MODEL_PATTERN="${1:-*}"

echo "========================================"
echo "Multi-Turn Evaluation: V1 vs V2 Comparison"
echo "========================================"
echo ""
echo "V1: Plain text history (tool names only)"
echo "V2: Structured history (tool args + results)"
echo ""

# Function to calculate stats from JSONL file
calc_stats() {
    local file="$1"
    if [[ ! -f "$file" ]]; then
        echo "0 0 0 0 0"
        return
    fi

    local total=$(jq -s 'length' "$file")
    local correct=$(jq -s '[.[] | select(.match_type == "exact" or .match_type == "acceptable")] | length' "$file")

    # Accuracy by turn
    local t1_total=$(jq -s '[.[] | select(.turn == 1)] | length' "$file")
    local t1_correct=$(jq -s '[.[] | select(.turn == 1 and (.match_type == "exact" or .match_type == "acceptable"))] | length' "$file")

    local t2_total=$(jq -s '[.[] | select(.turn == 2)] | length' "$file")
    local t2_correct=$(jq -s '[.[] | select(.turn == 2 and (.match_type == "exact" or .match_type == "acceptable"))] | length' "$file")

    local t3_total=$(jq -s '[.[] | select(.turn == 3)] | length' "$file")
    local t3_correct=$(jq -s '[.[] | select(.turn == 3 and (.match_type == "exact" or .match_type == "acceptable"))] | length' "$file")

    local t4_total=$(jq -s '[.[] | select(.turn == 4)] | length' "$file")
    local t4_correct=$(jq -s '[.[] | select(.turn == 4 and (.match_type == "exact" or .match_type == "acceptable"))] | length' "$file")

    echo "$total $correct $t1_total $t1_correct $t2_total $t2_correct $t3_total $t3_correct $t4_total $t4_correct"
}

# Header
printf "%-40s | %-15s | %-15s | %-10s\n" "Model" "V1 Accuracy" "V2 Accuracy" "Δ"
printf "%s\n" "$(printf '%.0s-' {1..90})"

# Find matching models
shopt -s nullglob
for v1_file in "$V1_DIR"/${MODEL_PATTERN}*.jsonl; do
    [[ -f "$v1_file" ]] || continue

    model_name=$(basename "$v1_file" | sed 's/_all_.*//')

    # Find corresponding V2 file (most recent)
    v2_file=$(ls -t "$V2_DIR"/${model_name}*.jsonl 2>/dev/null | head -1)

    # Get V1 stats
    read v1_total v1_correct v1_t1_total v1_t1_correct v1_t2_total v1_t2_correct v1_t3_total v1_t3_correct v1_t4_total v1_t4_correct <<< $(calc_stats "$v1_file")

    if [[ "$v1_total" -gt 0 ]]; then
        v1_acc=$(echo "scale=1; $v1_correct * 100 / $v1_total" | bc)
    else
        v1_acc="N/A"
    fi

    # Get V2 stats (if available)
    if [[ -n "$v2_file" && -f "$v2_file" ]]; then
        read v2_total v2_correct v2_t1_total v2_t1_correct v2_t2_total v2_t2_correct v2_t3_total v2_t3_correct v2_t4_total v2_t4_correct <<< $(calc_stats "$v2_file")

        if [[ "$v2_total" -gt 0 ]]; then
            v2_acc=$(echo "scale=1; $v2_correct * 100 / $v2_total" | bc)
            delta=$(echo "scale=1; $v2_acc - $v1_acc" | bc)
            if [[ $(echo "$delta > 0" | bc) -eq 1 ]]; then
                delta="+$delta"
            fi
        else
            v2_acc="N/A"
            delta="N/A"
        fi
    else
        v2_acc="Not run"
        delta="N/A"
    fi

    printf "%-40s | %12s%% | %12s%% | %10s%%\n" "$model_name" "$v1_acc" "$v2_acc" "$delta"
done

echo ""
echo "========================================"
echo "Detailed Turn-by-Turn Comparison"
echo "========================================"

for v1_file in "$V1_DIR"/${MODEL_PATTERN}*.jsonl; do
    [[ -f "$v1_file" ]] || continue

    model_name=$(basename "$v1_file" | sed 's/_all_.*//')
    v2_file=$(ls -t "$V2_DIR"/${model_name}*.jsonl 2>/dev/null | head -1)

    [[ -z "$v2_file" || ! -f "$v2_file" ]] && continue

    echo ""
    echo "Model: $model_name"
    printf "%-8s | %-12s | %-12s | %-10s\n" "Turn" "V1" "V2" "Δ"
    printf "%s\n" "$(printf '%.0s-' {1..50})"

    for turn in 1 2 3 4 5; do
        v1_t=$(jq -s "[.[] | select(.turn == $turn)] | length" "$v1_file")
        v1_c=$(jq -s "[.[] | select(.turn == $turn and (.match_type == \"exact\" or .match_type == \"acceptable\"))] | length" "$v1_file")

        v2_t=$(jq -s "[.[] | select(.turn == $turn)] | length" "$v2_file")
        v2_c=$(jq -s "[.[] | select(.turn == $turn and (.match_type == \"exact\" or .match_type == \"acceptable\"))] | length" "$v2_file")

        if [[ "$v1_t" -gt 0 ]]; then
            v1_acc=$(echo "scale=0; $v1_c * 100 / $v1_t" | bc)
            v1_str="$v1_c/$v1_t (${v1_acc}%)"
        else
            v1_str="N/A"
            v1_acc=0
        fi

        if [[ "$v2_t" -gt 0 ]]; then
            v2_acc=$(echo "scale=0; $v2_c * 100 / $v2_t" | bc)
            v2_str="$v2_c/$v2_t (${v2_acc}%)"
            delta=$(echo "$v2_acc - $v1_acc" | bc)
            if [[ "$delta" -gt 0 ]]; then
                delta_str="+${delta}%"
            else
                delta_str="${delta}%"
            fi
        else
            v2_str="N/A"
            delta_str="N/A"
        fi

        if [[ "$v1_t" -gt 0 || "$v2_t" -gt 0 ]]; then
            printf "Turn %-3d | %-12s | %-12s | %-10s\n" "$turn" "$v1_str" "$v2_str" "$delta_str"
        fi
    done
done

echo ""
echo "========================================"
echo "To run V2 evaluation for a model:"
echo "  ./scripts/run_multi_turn_eval_v2.sh <model> <provider> all"
echo ""
echo "Examples:"
echo "  ./scripts/run_multi_turn_eval_v2.sh gpt-4o-mini openai all"
echo "  ./scripts/run_multi_turn_eval_v2.sh claude-3-5-haiku-20241022 anthropic all"
echo "========================================"
