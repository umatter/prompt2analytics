#!/bin/bash
# Multi-turn conversation evaluation script V2
# Uses enhanced conversation history with tool arguments and results
#
# Usage: ./scripts/run_multi_turn_eval_v2.sh <model> <provider> <category|all> [--dry-run]
#
# Key improvements over V1:
# - Includes tool arguments in history (not just tool names)
# - Simulates tool results for context retention
# - Uses enhanced prompt format

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
        if [[ -z "$OPENAI_API_KEY" ]]; then
            echo "Error: OPENAI_API_KEY not set"
            exit 1
        fi
        ;;
    anthropic)
        if [[ -z "$ANTHROPIC_API_KEY" ]]; then
            echo "Error: ANTHROPIC_API_KEY not set"
            exit 1
        fi
        ;;
    openrouter)
        if [[ -z "$OPENROUTER_API_KEY" ]]; then
            echo "Error: OPENROUTER_API_KEY not set"
            exit 1
        fi
        ;;
esac

# Create output directory (v2 results separate from v1)
RESULTS_DIR="$PROJECT_DIR/results/multi_turn_v2"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${CATEGORY}_${TIMESTAMP}.jsonl"

TOOLS_FILE="$PROJECT_DIR/config/tools_filtered.json"

# Get test files
if [[ "$CATEGORY" == "all" ]]; then
    TEST_FILES=$(ls "$PROJECT_DIR/test_cases/multi_turn/"*.json 2>/dev/null)
else
    TEST_FILES="$PROJECT_DIR/test_cases/multi_turn/${CATEGORY}_conversations.json"
    if [[ ! -f "$TEST_FILES" ]]; then
        echo "Error: Test file not found: $TEST_FILES"
        exit 1
    fi
fi

echo "Multi-Turn Evaluation V2: $MODEL ($PROVIDER)"
echo "Output: $OUTPUT_FILE"
echo "========================================"
echo "Features: Tool arguments + simulated results in history"
echo "========================================"

TOTAL_TURNS=0
CORRECT_TURNS=0
COMPLETED_CONVERSATIONS=0
TOTAL_CONVERSATIONS=0

# Simulated tool results for common tools (helps LLM understand context)
get_simulated_result() {
    local tool="$1"
    local args="$2"

    case "$tool" in
        regression_ols|regression_clustered)
            echo "OLS Results: R²=0.847, Adj.R²=0.832, F=45.2(p<0.001). Coefficients: intercept=2.34(0.45), x1=0.89(0.12)**, x2=-0.23(0.08)*"
            ;;
        regression_vif)
            echo "VIF Results: x1=1.23, x2=1.45, x3=2.12. No multicollinearity detected (all VIF<5)."
            ;;
        hypothesis_jarque_bera)
            echo "Jarque-Bera Test: JB=3.45, p=0.178. Residuals appear normally distributed."
            ;;
        hypothesis_breusch_pagan)
            echo "Breusch-Pagan Test: BP=8.23, p=0.016. Evidence of heteroskedasticity."
            ;;
        viz_*)
            echo "Visualization generated successfully."
            ;;
        panel_fe)
            echo "Fixed Effects: Within R²=0.72, Between R²=0.58. Entity FE significant (F=12.3, p<0.001)."
            ;;
        iv_2sls)
            echo "2SLS Results: First-stage F=28.4 (>10). Sargan test p=0.34 (instruments valid)."
            ;;
        causal_did)
            echo "DiD Estimate: ATT=0.234 (SE=0.089), p=0.009. Pre-trends test: p=0.67 (parallel trends supported)."
            ;;
        *)
            echo "Tool executed successfully."
            ;;
    esac
}

for TEST_FILE in $TEST_FILES; do
    CATEGORY_NAME=$(basename "$TEST_FILE" .json | sed 's/_conversations//')
    echo "Category: $CATEGORY_NAME"

    # Process each conversation
    CONV_COUNT=$(jq '.conversations | length' "$TEST_FILE")

    for ((c=0; c<CONV_COUNT; c++)); do
        CONV_ID=$(jq -r ".conversations[$c].id" "$TEST_FILE")
        CONV_DESC=$(jq -r ".conversations[$c].description" "$TEST_FILE")
        DATASET_CTX=$(jq -r ".conversations[$c].dataset_context" "$TEST_FILE")
        TURN_COUNT=$(jq ".conversations[$c].turns | length" "$TEST_FILE")

        ((++TOTAL_CONVERSATIONS))

        echo "  Conversation: $CONV_ID - $CONV_DESC"

        # Build conversation history with proper format
        # Format: [{"role": "user/assistant/tool", "content": "...", "tool_calls": [...]}]
        CONVERSATION_HISTORY="[]"
        CONV_CORRECT=0
        LAST_TOOL=""
        LAST_ARGS="{}"

        for ((t=0; t<TURN_COUNT; t++)); do
            ((++TOTAL_TURNS))

            TURN_NUM=$(jq -r ".conversations[$c].turns[$t].turn" "$TEST_FILE")
            USER_PROMPT=$(jq -r ".conversations[$c].turns[$t].user_prompt" "$TEST_FILE")
            EXPECTED_TOOL=$(jq -r ".conversations[$c].turns[$t].expected_tool" "$TEST_FILE")
            ACCEPTABLE_TOOLS=$(jq -c ".conversations[$c].turns[$t].acceptable_tools" "$TEST_FILE")
            CONTEXT_FROM_PREV=$(jq -r ".conversations[$c].turns[$t].context_from_previous" "$TEST_FILE")
            EXPECTED_PARAMS=$(jq -c ".conversations[$c].turns[$t].expected_parameter_change // {}" "$TEST_FILE")

            if [[ "$DRY_RUN" == "--dry-run" ]]; then
                echo "    Turn $TURN_NUM: $USER_PROMPT"
                echo "      Expected: $EXPECTED_TOOL"
                if [[ "$EXPECTED_PARAMS" != "{}" ]]; then
                    echo "      Expected params: $EXPECTED_PARAMS"
                fi
                continue
            fi

            # Build prompt with enhanced conversation history
            FULL_PROMPT=$("$SCRIPT_DIR/build_multi_turn_prompt_v2.sh" \
                "$USER_PROMPT" \
                "$DATASET_CTX" \
                "$CONVERSATION_HISTORY" \
                "$CONTEXT_FROM_PREV")

            # Call the model
            START_TIME=$(date +%s%3N)
            RESPONSE=$("$SCRIPT_DIR/call_${PROVIDER}.sh" "$MODEL" "$FULL_PROMPT" "$TOOLS_FILE" 2>/dev/null || echo '{"tool": null}')
            END_TIME=$(date +%s%3N)
            LATENCY=$((END_TIME - START_TIME))

            # Extract selected tool and arguments
            SELECTED=$(echo "$RESPONSE" | jq -r '.tool // "null"')
            SELECTED_ARGS=$(echo "$RESPONSE" | jq -c '.arguments // {}')

            # Score the response
            SCORE_RESULT=$("$SCRIPT_DIR/score_response.sh" \
                "$SELECTED" \
                "[\"$EXPECTED_TOOL\"]" \
                "$ACCEPTABLE_TOOLS" \
                "$CATEGORY_NAME" 2>/dev/null || echo '{"match_type": "none", "score": 0}')

            MATCH_TYPE=$(echo "$SCORE_RESULT" | jq -r '.match_type')
            SCORE=$(echo "$SCORE_RESULT" | jq -r '.score')

            if [[ "$MATCH_TYPE" == "exact" || "$MATCH_TYPE" == "acceptable" ]]; then
                ((++CORRECT_TURNS))
                ((++CONV_CORRECT))
            fi

            # Update conversation history with proper format
            # 1. Add user message
            CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq \
                --arg content "$USER_PROMPT" \
                '. + [{"role": "user", "content": $content}]')

            # 2. Add assistant message with tool call
            if [[ "$SELECTED" != "null" && -n "$SELECTED" ]]; then
                TOOL_CALL_ID="call_${TURN_NUM}"

                # Create tool call object
                TOOL_CALL=$(jq -n \
                    --arg id "$TOOL_CALL_ID" \
                    --arg name "$SELECTED" \
                    --argjson args "$SELECTED_ARGS" \
                    '{"id": $id, "name": $name, "arguments": $args}')

                CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq \
                    --argjson tool_call "$TOOL_CALL" \
                    '. + [{"role": "assistant", "content": "", "tool_calls": [$tool_call]}]')

                # 3. Add simulated tool result
                SIMULATED_RESULT=$(get_simulated_result "$SELECTED" "$SELECTED_ARGS")
                CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq \
                    --arg id "$TOOL_CALL_ID" \
                    --arg result "$SIMULATED_RESULT" \
                    '. + [{"role": "tool", "tool_call_id": $id, "content": $result}]')

                LAST_TOOL="$SELECTED"
                LAST_ARGS="$SELECTED_ARGS"
            else
                CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq \
                    '. + [{"role": "assistant", "content": "I could not determine the appropriate tool."}]')
            fi

            # Calculate history length for logging
            HISTORY_LEN=$(echo "$CONVERSATION_HISTORY" | jq 'length')

            # Write result
            RESULT=$(jq -n \
                --arg timestamp "$(date -Iseconds)" \
                --arg model "$MODEL" \
                --arg conv_id "$CONV_ID" \
                --argjson turn "$TURN_NUM" \
                --arg category "$CATEGORY_NAME" \
                --arg prompt "$USER_PROMPT" \
                --arg selected "$SELECTED" \
                --argjson selected_args "$SELECTED_ARGS" \
                --arg expected "$EXPECTED_TOOL" \
                --argjson expected_params "$EXPECTED_PARAMS" \
                --arg match_type "$MATCH_TYPE" \
                --argjson score "$SCORE" \
                --argjson latency_ms "$LATENCY" \
                --argjson context_from_prev "$CONTEXT_FROM_PREV" \
                --argjson history_length "$HISTORY_LEN" \
                '{
                    timestamp: $timestamp,
                    model: $model,
                    conversation_id: $conv_id,
                    turn: $turn,
                    category: $category,
                    prompt: $prompt,
                    selected: $selected,
                    selected_args: $selected_args,
                    expected: $expected,
                    expected_params: $expected_params,
                    match_type: $match_type,
                    score: $score,
                    latency_ms: $latency_ms,
                    context_from_previous: $context_from_prev,
                    history_length: $history_length,
                    eval_version: "v2"
                }')

            echo "$RESULT" >> "$OUTPUT_FILE"

            STATUS="✗"
            [[ "$MATCH_TYPE" == "exact" || "$MATCH_TYPE" == "acceptable" ]] && STATUS="✓"
            echo "    Turn $TURN_NUM: $STATUS $MATCH_TYPE ($SELECTED vs $EXPECTED_TOOL) [history: $HISTORY_LEN msgs]"

            # Rate limiting
            sleep 0.5
        done

        if [[ "$CONV_CORRECT" -eq "$TURN_COUNT" ]]; then
            ((++COMPLETED_CONVERSATIONS))
            echo "    → Conversation completed successfully!"
        fi
    done
done

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo "Total Turns: $TOTAL_TURNS"
    echo "Correct Turns: $CORRECT_TURNS"
    if [[ "$TOTAL_TURNS" -gt 0 ]]; then
        TURN_ACC=$(echo "scale=1; $CORRECT_TURNS * 100 / $TOTAL_TURNS" | bc)
        echo "Turn Accuracy: ${TURN_ACC}%"
    fi
    echo ""
    echo "Total Conversations: $TOTAL_CONVERSATIONS"
    echo "Fully Completed: $COMPLETED_CONVERSATIONS"
    if [[ "$TOTAL_CONVERSATIONS" -gt 0 ]]; then
        CONV_COMP=$(echo "scale=1; $COMPLETED_CONVERSATIONS * 100 / $TOTAL_CONVERSATIONS" | bc)
        echo "Conversation Completion: ${CONV_COMP}%"
    fi
    echo ""
    echo "Results saved to: $OUTPUT_FILE"

    # Generate accuracy by turn breakdown
    echo ""
    echo "Accuracy by Turn:"
    for turn_num in 1 2 3 4 5; do
        TURN_TOTAL=$(jq -s "[.[] | select(.turn == $turn_num)] | length" "$OUTPUT_FILE" 2>/dev/null || echo 0)
        TURN_CORRECT=$(jq -s "[.[] | select(.turn == $turn_num and (.match_type == \"exact\" or .match_type == \"acceptable\"))] | length" "$OUTPUT_FILE" 2>/dev/null || echo 0)
        if [[ "$TURN_TOTAL" -gt 0 ]]; then
            TURN_ACC=$(echo "scale=0; $TURN_CORRECT * 100 / $TURN_TOTAL" | bc)
            echo "  Turn $turn_num: $TURN_CORRECT/$TURN_TOTAL (${TURN_ACC}%)"
        fi
    done
fi
