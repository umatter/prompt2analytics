#!/bin/bash
# Out-of-scope detection evaluation script
# Usage: ./scripts/run_oos_eval.sh <model> <provider> [--dry-run]

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
RESULTS_DIR="$PROJECT_DIR/results/out_of_scope"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${TIMESTAMP}.jsonl"

TOOLS_FILE="$PROJECT_DIR/config/tools.json"
TEST_FILE="$PROJECT_DIR/test_cases/out_of_scope.json"

if [[ ! -f "$TEST_FILE" ]]; then
    echo "Error: Test file not found: $TEST_FILE"
    exit 1
fi

echo "Out-of-Scope Detection Evaluation: $MODEL ($PROVIDER)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

TOTAL=0
OOS_DETECTED=0
FALSE_TOOL=0
GRACEFUL=0

TEST_COUNT=$(jq '.tests | length' "$TEST_FILE")

for ((i=0; i<TEST_COUNT; i++)); do
    ((++TOTAL))

    TEST_ID=$(jq -r ".tests[$i].id" "$TEST_FILE")
    PROMPT=$(jq -r ".tests[$i].prompt" "$TEST_FILE")
    OOS_REASON=$(jq -r ".tests[$i].oos_reason" "$TEST_FILE")
    CLOSEST=$(jq -c ".tests[$i].closest_available" "$TEST_FILE")
    EXPLANATION=$(jq -r ".tests[$i].explanation" "$TEST_FILE")

    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "$TEST_ID [$OOS_REASON]: $PROMPT"
        echo "  Closest: $CLOSEST"
        continue
    fi

    # Build prompt - explicitly ask for tool selection
    FULL_PROMPT="I have a dataset and need to perform the following analysis:

$PROMPT

If this analysis is supported, select the appropriate tool. If not supported, indicate that this is out of scope for the available tools.

Available tool categories: regression, panel, causal inference (IV, DiD, RD), discrete choice (logit, probit), time series (ARIMA, VAR), hypothesis testing, machine learning (clustering, PCA, random forest), visualization."

    START_TIME=$(date +%s%3N)
    RESPONSE=$("$SCRIPT_DIR/call_${PROVIDER}.sh" "$MODEL" "$FULL_PROMPT" "$TOOLS_FILE" 2>/dev/null || echo '{"tool": null}')
    END_TIME=$(date +%s%3N)
    LATENCY=$((END_TIME - START_TIME))

    SELECTED_TOOL=$(echo "$RESPONSE" | jq -r '.tool // "null"')
    RESPONSE_CONTENT=$(echo "$RESPONSE" | jq -r '.content // ""')

    # Score the response
    SCORE_RESULT=$("$SCRIPT_DIR/score_oos_response.sh" "$SELECTED_TOOL" "$RESPONSE_CONTENT" "$CLOSEST")

    DETECTED_OOS=$(echo "$SCORE_RESULT" | jq -r '.detected_oos')
    SELECTED_FALSE_TOOL=$(echo "$SCORE_RESULT" | jq -r '.selected_false_tool')
    SUGGESTED_ALTERNATIVE=$(echo "$SCORE_RESULT" | jq -r '.suggested_alternative')
    SCORE_TYPE=$(echo "$SCORE_RESULT" | jq -r '.score_type')

    case "$SCORE_TYPE" in
        "oos_detected")
            ((++OOS_DETECTED))
            ;;
        "false_tool")
            ((++FALSE_TOOL))
            ;;
        "graceful")
            ((++OOS_DETECTED))
            ((++GRACEFUL))
            ;;
    esac

    # Write result
    RESULT=$(jq -n \
        --arg timestamp "$(date -Iseconds)" \
        --arg model "$MODEL" \
        --arg test_id "$TEST_ID" \
        --arg prompt "$PROMPT" \
        --arg oos_reason "$OOS_REASON" \
        --argjson closest_available "$CLOSEST" \
        --arg explanation "$EXPLANATION" \
        --arg selected_tool "$SELECTED_TOOL" \
        --arg response_content "$RESPONSE_CONTENT" \
        --argjson detected_oos "$DETECTED_OOS" \
        --argjson selected_false_tool "$SELECTED_FALSE_TOOL" \
        --argjson suggested_alternative "$SUGGESTED_ALTERNATIVE" \
        --arg score_type "$SCORE_TYPE" \
        --argjson latency_ms "$LATENCY" \
        '{
            timestamp: $timestamp,
            model: $model,
            test_id: $test_id,
            prompt: $prompt,
            oos_reason: $oos_reason,
            closest_available: $closest_available,
            explanation: $explanation,
            selected_tool: $selected_tool,
            response_content: $response_content,
            detected_oos: $detected_oos,
            selected_false_tool: $selected_false_tool,
            suggested_alternative: $suggested_alternative,
            score_type: $score_type,
            latency_ms: $latency_ms
        }')

    echo "$RESULT" >> "$OUTPUT_FILE"

    echo "$TEST_ID [$OOS_REASON]: $SCORE_TYPE (selected: $SELECTED_TOOL)"

    sleep 0.5
done

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo "Total Tests: $TOTAL"

    OOS_RATE=$(echo "scale=1; $OOS_DETECTED * 100 / $TOTAL" | bc)
    echo "OOS Detection Rate: $OOS_DETECTED/$TOTAL (${OOS_RATE}%)"

    FALSE_RATE=$(echo "scale=1; $FALSE_TOOL * 100 / $TOTAL" | bc)
    echo "False Tool Selection Rate: $FALSE_TOOL/$TOTAL (${FALSE_RATE}%)"

    if [[ $OOS_DETECTED -gt 0 ]]; then
        GRACEFUL_RATE=$(echo "scale=1; $GRACEFUL * 100 / $OOS_DETECTED" | bc)
        echo "Graceful Degradation (of detected): $GRACEFUL/$OOS_DETECTED (${GRACEFUL_RATE}%)"
    fi

    echo ""
    echo "Results saved to: $OUTPUT_FILE"
fi
