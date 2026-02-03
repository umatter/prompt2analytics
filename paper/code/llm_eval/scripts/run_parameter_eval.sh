#!/bin/bash
# Parameter extraction evaluation script
# Usage: ./scripts/run_parameter_eval.sh <model> <provider> [--dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODEL="${1:-gpt-4o}"
PROVIDER="${2:-openai}"
DRY_RUN="${3:-}"

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
RESULTS_DIR="$PROJECT_DIR/results/parameter_extraction"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${TIMESTAMP}.jsonl"

TOOLS_FILE="$PROJECT_DIR/config/tools.json"
TEST_FILE="$PROJECT_DIR/test_cases/parameter_extraction/parameter_tests.json"

if [[ ! -f "$TEST_FILE" ]]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

echo "Parameter Extraction Evaluation: $MODEL ($PROVIDER)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

TOTAL=0
TOOL_CORRECT=0
TOTAL_PARAMS=0
CORRECT_PARAMS=0

TEST_COUNT=$(jq '.tests | length' "$TEST_FILE")

for ((i=0; i<TEST_COUNT; i++)); do
    ((++TOTAL))

    TEST_ID=$(jq -r ".tests[$i].id" "$TEST_FILE")
    PROMPT=$(jq -r ".tests[$i].prompt" "$TEST_FILE")
    EXPECTED_TOOL=$(jq -r ".tests[$i].expected_tool" "$TEST_FILE")
    DATASET_CTX=$(jq -r ".tests[$i].dataset_context" "$TEST_FILE")
    EXPECTED_PARAMS=$(jq -c ".tests[$i].expected_parameters" "$TEST_FILE")
    PARAM_DIFFICULTY=$(jq -c ".tests[$i].parameter_difficulty" "$TEST_FILE")

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "$TEST_ID: $PROMPT"
        echo "  Expected tool: $EXPECTED_TOOL"
        echo "  Expected params: $EXPECTED_PARAMS"
        continue
    fi

    # Build and send prompt
    FULL_PROMPT=$("$SCRIPT_DIR/build_prompt.sh" "$PROMPT" "$DATASET_CTX")

    START_TIME=$(date +%s%3N)
    RESPONSE=$("$SCRIPT_DIR/call_${PROVIDER}.sh" "$MODEL" "$FULL_PROMPT" "$TOOLS_FILE" 2>/dev/null || echo '{"tool": null, "arguments": {}}')
    END_TIME=$(date +%s%3N)
    LATENCY=$((END_TIME - START_TIME))

    SELECTED_TOOL=$(echo "$RESPONSE" | jq -r '.tool // "null"')
    # Handle arguments - may be a JSON object or a JSON string
    RAW_ARGS=$(echo "$RESPONSE" | jq -c '.arguments // {}')
    # If it's a string, parse it as JSON; if already an object, use as-is
    if echo "$RAW_ARGS" | jq -e 'type == "string"' > /dev/null 2>&1; then
        SELECTED_ARGS=$(echo "$RAW_ARGS" | jq -r '.' | jq -c '.')
    else
        SELECTED_ARGS="$RAW_ARGS"
    fi

    # Check tool selection
    TOOL_MATCH="false"
    if [[ "$SELECTED_TOOL" == "$EXPECTED_TOOL" ]]; then
        TOOL_MATCH="true"
        ((++TOOL_CORRECT))
    fi

    # Score parameters
    PARAM_RESULT=$("$SCRIPT_DIR/score_parameters.sh" "$EXPECTED_PARAMS" "$SELECTED_ARGS")

    PARAM_PRECISION=$(echo "$PARAM_RESULT" | jq -r '.precision')
    PARAM_RECALL=$(echo "$PARAM_RESULT" | jq -r '.recall')
    PARAM_F1=$(echo "$PARAM_RESULT" | jq -r '.f1')
    PARAM_DETAILS=$(echo "$PARAM_RESULT" | jq -c '.details')

    # Accumulate parameter counts
    EXPECTED_COUNT=$(echo "$PARAM_RESULT" | jq -r '.expected_count')
    CORRECT_COUNT=$(echo "$PARAM_RESULT" | jq -r '.correct_count')
    ((TOTAL_PARAMS += EXPECTED_COUNT))
    ((CORRECT_PARAMS += CORRECT_COUNT))

    # Write result
    RESULT=$(jq -n \
        --arg timestamp "$(date -Iseconds)" \
        --arg model "$MODEL" \
        --arg test_id "$TEST_ID" \
        --arg prompt "$PROMPT" \
        --arg expected_tool "$EXPECTED_TOOL" \
        --arg selected_tool "$SELECTED_TOOL" \
        --argjson tool_match "$TOOL_MATCH" \
        --argjson expected_params "$EXPECTED_PARAMS" \
        --argjson selected_args "$SELECTED_ARGS" \
        --argjson param_precision "$PARAM_PRECISION" \
        --argjson param_recall "$PARAM_RECALL" \
        --argjson param_f1 "$PARAM_F1" \
        --argjson param_details "$PARAM_DETAILS" \
        --argjson param_difficulty "$PARAM_DIFFICULTY" \
        --argjson latency_ms "$LATENCY" \
        '{
            timestamp: $timestamp,
            model: $model,
            test_id: $test_id,
            prompt: $prompt,
            expected_tool: $expected_tool,
            selected_tool: $selected_tool,
            tool_match: $tool_match,
            expected_params: $expected_params,
            selected_args: $selected_args,
            param_precision: $param_precision,
            param_recall: $param_recall,
            param_f1: $param_f1,
            param_details: $param_details,
            param_difficulty: $param_difficulty,
            latency_ms: $latency_ms
        }')

    echo "$RESULT" >> "$OUTPUT_FILE"

    echo "$TEST_ID: tool=$TOOL_MATCH, precision=$PARAM_PRECISION, recall=$PARAM_RECALL, f1=$PARAM_F1"

    sleep 0.5
done

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo "Total Tests: $TOTAL"
    TOOL_ACC=$(echo "scale=1; $TOOL_CORRECT * 100 / $TOTAL" | bc)
    echo "Tool Selection Accuracy: $TOOL_CORRECT/$TOTAL (${TOOL_ACC}%)"

    if [[ $TOTAL_PARAMS -gt 0 ]]; then
        PARAM_ACC=$(echo "scale=1; $CORRECT_PARAMS * 100 / $TOTAL_PARAMS" | bc)
        echo "Parameter Accuracy: $CORRECT_PARAMS/$TOTAL_PARAMS (${PARAM_ACC}%)"
    fi

    # Calculate average metrics from results
    AVG_PRECISION=$(jq -s '[.[].param_precision] | add / length' "$OUTPUT_FILE")
    AVG_RECALL=$(jq -s '[.[].param_recall] | add / length' "$OUTPUT_FILE")
    AVG_F1=$(jq -s '[.[].param_f1] | add / length' "$OUTPUT_FILE")

    echo ""
    echo "Parameter Extraction Metrics:"
    echo "  Avg Precision: $(printf "%.3f" $AVG_PRECISION)"
    echo "  Avg Recall: $(printf "%.3f" $AVG_RECALL)"
    echo "  Avg F1: $(printf "%.3f" $AVG_F1)"

    echo ""
    echo "Results saved to: $OUTPUT_FILE"
fi
