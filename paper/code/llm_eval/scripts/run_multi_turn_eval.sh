#!/bin/bash
# Multi-turn conversation evaluation script
# Usage: ./scripts/run_multi_turn_eval.sh <model> <provider> <category|all> [--dry-run]

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

# Create output directory
RESULTS_DIR="$PROJECT_DIR/results/multi_turn"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${CATEGORY}_${TIMESTAMP}.jsonl"

TOOLS_FILE="$PROJECT_DIR/config/tools.json"

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

echo "Multi-Turn Evaluation: $MODEL ($PROVIDER)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

TOTAL_TURNS=0
CORRECT_TURNS=0
COMPLETED_CONVERSATIONS=0
TOTAL_CONVERSATIONS=0

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

        # Build conversation history
        CONVERSATION_HISTORY="[]"
        CONV_CORRECT=0

        for ((t=0; t<TURN_COUNT; t++)); do
            ((++TOTAL_TURNS))

            TURN_NUM=$(jq -r ".conversations[$c].turns[$t].turn" "$TEST_FILE")
            USER_PROMPT=$(jq -r ".conversations[$c].turns[$t].user_prompt" "$TEST_FILE")
            EXPECTED_TOOL=$(jq -r ".conversations[$c].turns[$t].expected_tool" "$TEST_FILE")
            ACCEPTABLE_TOOLS=$(jq -c ".conversations[$c].turns[$t].acceptable_tools" "$TEST_FILE")
            CONTEXT_FROM_PREV=$(jq -r ".conversations[$c].turns[$t].context_from_previous" "$TEST_FILE")

            if [[ "$DRY_RUN" == "--dry-run" ]]; then
                echo "    Turn $TURN_NUM: $USER_PROMPT"
                echo "      Expected: $EXPECTED_TOOL"
                continue
            fi

            # Build prompt with conversation history
            FULL_PROMPT=$("$SCRIPT_DIR/build_multi_turn_prompt.sh" \
                "$USER_PROMPT" \
                "$DATASET_CTX" \
                "$CONVERSATION_HISTORY" \
                "$CONTEXT_FROM_PREV")

            # Call the model
            START_TIME=$(date +%s%3N)
            RESPONSE=$("$SCRIPT_DIR/call_${PROVIDER}.sh" "$MODEL" "$FULL_PROMPT" "$TOOLS_FILE" 2>/dev/null || echo '{"tool": null}')
            END_TIME=$(date +%s%3N)
            LATENCY=$((END_TIME - START_TIME))

            # Extract selected tool
            SELECTED=$(echo "$RESPONSE" | jq -r '.tool // "null"')

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

            # Update conversation history
            CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq \
                --arg role "user" \
                --arg content "$USER_PROMPT" \
                '. + [{"role": $role, "content": $content}]')

            CONVERSATION_HISTORY=$(echo "$CONVERSATION_HISTORY" | jq \
                --arg role "assistant" \
                --arg tool "$SELECTED" \
                '. + [{"role": $role, "tool_selected": $tool}]')

            # Write result
            RESULT=$(jq -n \
                --arg timestamp "$(date -Iseconds)" \
                --arg model "$MODEL" \
                --arg conv_id "$CONV_ID" \
                --argjson turn "$TURN_NUM" \
                --arg category "$CATEGORY_NAME" \
                --arg prompt "$USER_PROMPT" \
                --arg selected "$SELECTED" \
                --arg expected "$EXPECTED_TOOL" \
                --arg match_type "$MATCH_TYPE" \
                --argjson score "$SCORE" \
                --argjson latency_ms "$LATENCY" \
                --argjson context_from_prev "$CONTEXT_FROM_PREV" \
                '{
                    timestamp: $timestamp,
                    model: $model,
                    conversation_id: $conv_id,
                    turn: $turn,
                    category: $category,
                    prompt: $prompt,
                    selected: $selected,
                    expected: $expected,
                    match_type: $match_type,
                    score: $score,
                    latency_ms: $latency_ms,
                    context_from_previous: $context_from_prev
                }')

            echo "$RESULT" >> "$OUTPUT_FILE"

            echo "    Turn $TURN_NUM: $MATCH_TYPE ($SELECTED vs $EXPECTED_TOOL)"

            # Rate limiting
            sleep 0.5
        done

        if [[ "$CONV_CORRECT" -eq "$TURN_COUNT" ]]; then
            ((++COMPLETED_CONVERSATIONS))
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
    TURN_ACC=$(echo "scale=1; $CORRECT_TURNS * 100 / $TOTAL_TURNS" | bc)
    echo "Turn Accuracy: ${TURN_ACC}%"
    echo ""
    echo "Total Conversations: $TOTAL_CONVERSATIONS"
    echo "Fully Completed: $COMPLETED_CONVERSATIONS"
    CONV_COMP=$(echo "scale=1; $COMPLETED_CONVERSATIONS * 100 / $TOTAL_CONVERSATIONS" | bc)
    echo "Conversation Completion: ${CONV_COMP}%"
    echo ""
    echo "Results saved to: $OUTPUT_FILE"
fi
