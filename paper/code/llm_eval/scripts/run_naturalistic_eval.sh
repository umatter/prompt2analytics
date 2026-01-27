#!/bin/bash
# Naturalistic prompt evaluation script
# Usage: ./scripts/run_naturalistic_eval.sh <model> <provider> <category|all> [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="${1:-gpt-4o}"
PROVIDER="${2:-openai}"
CATEGORY="${3:-all}"
DRY_RUN="${4:-}"

# Validate inputs
if [[ ! "$PROVIDER" =~ ^(openai|anthropic|openrouter|ollama)$ ]]; then
    echo "Error: Provider must be one of: openai, anthropic, openrouter, ollama"
    exit 1
fi

# Check API keys
case "$PROVIDER" in
    openai)
        [[ -z "$OPENAI_API_KEY" ]] && echo "Error: OPENAI_API_KEY not set" && exit 1
        ;;
    anthropic)
        [[ -z "$ANTHROPIC_API_KEY" ]] && echo "Error: ANTHROPIC_API_KEY not set" && exit 1
        ;;
    openrouter)
        [[ -z "$OPENROUTER_API_KEY" ]] && echo "Error: OPENROUTER_API_KEY not set" && exit 1
        ;;
esac

# Create output directory
RESULTS_DIR="$PROJECT_DIR/results/naturalistic"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${CATEGORY}_${TIMESTAMP}.jsonl"

TOOLS_FILE="$PROJECT_DIR/config/tools.json"

# Get test files
if [[ "$CATEGORY" == "all" ]]; then
    TEST_FILES=$(ls "$PROJECT_DIR/test_cases/naturalistic/"*.json 2>/dev/null)
else
    TEST_FILES="$PROJECT_DIR/test_cases/naturalistic/${CATEGORY}_natural.json"
    if [[ ! -f "$TEST_FILES" ]]; then
        echo "Error: Test file not found: $TEST_FILES"
        exit 1
    fi
fi

echo "Naturalistic Evaluation: $MODEL ($PROVIDER)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

# Track metrics by prompt type
declare -A TYPE_TOTAL
declare -A TYPE_CORRECT

TOTAL=0
EXACT=0
ACCEPTABLE=0
CATEGORY_MATCH=0
FAILED=0

for TEST_FILE in $TEST_FILES; do
    CATEGORY_NAME=$(basename "$TEST_FILE" .json | sed 's/_natural//')
    echo "Category: $CATEGORY_NAME"

    TEST_COUNT=$(jq '.tests | length' "$TEST_FILE")

    for ((i=0; i<TEST_COUNT; i++)); do
        ((++TOTAL))

        TEST_ID=$(jq -r ".tests[$i].id" "$TEST_FILE")
        PROMPT=$(jq -r ".tests[$i].prompt" "$TEST_FILE")
        PROMPT_TYPE=$(jq -r ".tests[$i].prompt_type" "$TEST_FILE")
        DATASET_CTX=$(jq -r ".tests[$i].dataset_context" "$TEST_FILE")
        EXPECTED=$(jq -c ".tests[$i].expected_tools" "$TEST_FILE")
        ACCEPTABLE_TOOLS=$(jq -c ".tests[$i].acceptable_tools" "$TEST_FILE")
        LINGUISTIC_FEATURES=$(jq -c ".tests[$i].linguistic_features" "$TEST_FILE")

        # Track by prompt type
        TYPE_TOTAL[$PROMPT_TYPE]=$((${TYPE_TOTAL[$PROMPT_TYPE]:-0} + 1))

        if [[ "$DRY_RUN" == "--dry-run" ]]; then
            echo "  $TEST_ID [$PROMPT_TYPE]: $PROMPT"
            continue
        fi

        # Build and send prompt
        FULL_PROMPT=$("$SCRIPT_DIR/build_prompt.sh" "$PROMPT" "$DATASET_CTX")

        START_TIME=$(date +%s%3N)
        RESPONSE=$("$SCRIPT_DIR/call_${PROVIDER}.sh" "$MODEL" "$FULL_PROMPT" "$TOOLS_FILE" 2>/dev/null || echo '{"tool": null}')
        END_TIME=$(date +%s%3N)
        LATENCY=$((END_TIME - START_TIME))

        SELECTED=$(echo "$RESPONSE" | jq -r '.tool // "null"')

        # Score response
        SCORE_RESULT=$("$SCRIPT_DIR/score_response.sh" \
            "$SELECTED" \
            "$EXPECTED" \
            "$ACCEPTABLE_TOOLS" \
            "$CATEGORY_NAME" 2>/dev/null || echo '{"match_type": "none", "score": 0}')

        MATCH_TYPE=$(echo "$SCORE_RESULT" | jq -r '.match_type')
        SCORE=$(echo "$SCORE_RESULT" | jq -r '.score')

        case "$MATCH_TYPE" in
            exact) ((++EXACT)); TYPE_CORRECT[$PROMPT_TYPE]=$((${TYPE_CORRECT[$PROMPT_TYPE]:-0} + 1)) ;;
            acceptable) ((++ACCEPTABLE)); TYPE_CORRECT[$PROMPT_TYPE]=$((${TYPE_CORRECT[$PROMPT_TYPE]:-0} + 1)) ;;
            category) ((++CATEGORY_MATCH)) ;;
            *) ((++FAILED)) ;;
        esac

        # Write result
        RESULT=$(jq -n \
            --arg timestamp "$(date -Iseconds)" \
            --arg model "$MODEL" \
            --arg test_id "$TEST_ID" \
            --arg category "$CATEGORY_NAME" \
            --arg prompt_type "$PROMPT_TYPE" \
            --arg prompt "$PROMPT" \
            --arg selected "$SELECTED" \
            --argjson expected "$EXPECTED" \
            --arg match_type "$MATCH_TYPE" \
            --argjson score "$SCORE" \
            --argjson latency_ms "$LATENCY" \
            --argjson linguistic_features "$LINGUISTIC_FEATURES" \
            '{
                timestamp: $timestamp,
                model: $model,
                test_id: $test_id,
                category: $category,
                prompt_type: $prompt_type,
                prompt: $prompt,
                selected: $selected,
                expected: $expected,
                match_type: $match_type,
                score: $score,
                latency_ms: $latency_ms,
                linguistic_features: $linguistic_features
            }')

        echo "$RESULT" >> "$OUTPUT_FILE"

        echo "  $TEST_ID [$PROMPT_TYPE]: $MATCH_TYPE"

        sleep 0.5
    done
done

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo "Total: $TOTAL"
    echo "Exact: $EXACT ($(echo "scale=1; $EXACT * 100 / $TOTAL" | bc)%)"
    echo "Acceptable: $ACCEPTABLE ($(echo "scale=1; $ACCEPTABLE * 100 / $TOTAL" | bc)%)"
    echo "Category: $CATEGORY_MATCH ($(echo "scale=1; $CATEGORY_MATCH * 100 / $TOTAL" | bc)%)"
    echo "Failed: $FAILED ($(echo "scale=1; $FAILED * 100 / $TOTAL" | bc)%)"

    ACCURACY=$(echo "scale=1; ($EXACT + $ACCEPTABLE) * 100 / $TOTAL" | bc)
    echo ""
    echo "Overall Accuracy: ${ACCURACY}%"

    echo ""
    echo "By Prompt Type:"
    for TYPE in "${!TYPE_TOTAL[@]}"; do
        T_TOTAL=${TYPE_TOTAL[$TYPE]}
        T_CORRECT=${TYPE_CORRECT[$TYPE]:-0}
        T_ACC=$(echo "scale=1; $T_CORRECT * 100 / $T_TOTAL" | bc)
        echo "  $TYPE: $T_CORRECT/$T_TOTAL (${T_ACC}%)"
    done

    echo ""
    echo "Results saved to: $OUTPUT_FILE"
fi
