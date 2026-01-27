#!/bin/bash
# Interpretation evaluation script
# Usage: ./scripts/run_interpretation_eval.sh <model> <provider> <category|all> [--dry-run]

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
RESULTS_DIR="$PROJECT_DIR/results/interpretation"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_' | tr ':' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${CATEGORY}_${TIMESTAMP}.jsonl"

# Get test files
if [[ "$CATEGORY" == "all" ]]; then
    TEST_FILES=$(ls "$PROJECT_DIR/test_cases/interpretation/"*.json 2>/dev/null)
else
    TEST_FILES="$PROJECT_DIR/test_cases/interpretation/${CATEGORY}_interpretation.json"
    if [[ ! -f "$TEST_FILES" ]]; then
        echo "Error: Test file not found: $TEST_FILES"
        exit 1
    fi
fi

echo "Interpretation Evaluation: $MODEL ($PROVIDER)"
echo "Output: $OUTPUT_FILE"
echo "========================================"

TOTAL=0
TOTAL_ELEMENTS=0
COVERED_ELEMENTS=0
TOTAL_ERRORS=0
ERROR_COUNT=0

for TEST_FILE in $TEST_FILES; do
    CATEGORY_NAME=$(basename "$TEST_FILE" .json | sed 's/_interpretation//')
    echo "Category: $CATEGORY_NAME"

    TEST_COUNT=$(jq '.tests | length' "$TEST_FILE")

    for ((i=0; i<TEST_COUNT; i++)); do
        ((++TOTAL))

        TEST_ID=$(jq -r ".tests[$i].id" "$TEST_FILE")
        TASK_TYPE=$(jq -r ".tests[$i].task_type" "$TEST_FILE")
        SIMULATED_OUTPUT=$(jq -c ".tests[$i].simulated_output" "$TEST_FILE")
        INTERP_PROMPT=$(jq -r ".tests[$i].interpretation_prompt" "$TEST_FILE")
        EXPECTED_ELEMENTS=$(jq -c ".tests[$i].expected_elements" "$TEST_FILE")
        INCORRECT_INTERPS=$(jq -c ".tests[$i].incorrect_interpretations" "$TEST_FILE")

        if [[ "$DRY_RUN" == "--dry-run" ]]; then
            echo "  $TEST_ID [$TASK_TYPE]: $INTERP_PROMPT"
            continue
        fi

        # Build the prompt with simulated output
        FULL_PROMPT="Here are the results from an econometric analysis:

$SIMULATED_OUTPUT

$INTERP_PROMPT

Provide a clear and accurate interpretation of these results."

        START_TIME=$(date +%s%3N)

        # Call the model (without tool calling - we want free-form response)
        case "$PROVIDER" in
            openai)
                RESPONSE=$(curl -s "https://api.openai.com/v1/chat/completions" \
                    -H "Content-Type: application/json" \
                    -H "Authorization: Bearer $OPENAI_API_KEY" \
                    -d "$(jq -n \
                        --arg model "$MODEL" \
                        --arg prompt "$FULL_PROMPT" \
                        '{
                            model: $model,
                            messages: [
                                {role: "system", content: "You are an expert econometrician. Provide accurate statistical interpretations."},
                                {role: "user", content: $prompt}
                            ],
                            temperature: 0,
                            max_tokens: 1024
                        }')" | jq -r '.choices[0].message.content // ""')
                ;;
            anthropic)
                RESPONSE=$(curl -s "https://api.anthropic.com/v1/messages" \
                    -H "Content-Type: application/json" \
                    -H "x-api-key: $ANTHROPIC_API_KEY" \
                    -H "anthropic-version: 2023-06-01" \
                    -d "$(jq -n \
                        --arg model "$MODEL" \
                        --arg prompt "$FULL_PROMPT" \
                        '{
                            model: $model,
                            max_tokens: 1024,
                            system: "You are an expert econometrician. Provide accurate statistical interpretations.",
                            messages: [{role: "user", content: $prompt}]
                        }')" | jq -r '.content[0].text // ""')
                ;;
            openrouter)
                RESPONSE=$(curl -s "https://openrouter.ai/api/v1/chat/completions" \
                    -H "Content-Type: application/json" \
                    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
                    -d "$(jq -n \
                        --arg model "$MODEL" \
                        --arg prompt "$FULL_PROMPT" \
                        '{
                            model: $model,
                            messages: [
                                {role: "system", content: "You are an expert econometrician. Provide accurate statistical interpretations."},
                                {role: "user", content: $prompt}
                            ],
                            temperature: 0,
                            max_tokens: 1024
                        }')" | jq -r '.choices[0].message.content // ""')
                ;;
            ollama)
                RESPONSE=$(curl -s "http://localhost:11434/api/chat" \
                    -d "$(jq -n \
                        --arg model "$MODEL" \
                        --arg prompt "$FULL_PROMPT" \
                        '{
                            model: $model,
                            messages: [
                                {role: "system", content: "You are an expert econometrician. Provide accurate statistical interpretations."},
                                {role: "user", content: $prompt}
                            ],
                            stream: false,
                            options: {temperature: 0}
                        }')" | jq -r '.message.content // ""')
                ;;
        esac

        END_TIME=$(date +%s%3N)
        LATENCY=$((END_TIME - START_TIME))

        # Score the interpretation
        SCORE_RESULT=$("$SCRIPT_DIR/score_interpretation.sh" "$RESPONSE" "$EXPECTED_ELEMENTS" "$INCORRECT_INTERPS")

        ELEMENT_COVERAGE=$(echo "$SCORE_RESULT" | jq -r '.element_coverage')
        ERROR_DETECTED=$(echo "$SCORE_RESULT" | jq -r '.errors_detected')
        ACCURACY=$(echo "$SCORE_RESULT" | jq -r '.accuracy')
        ELEMENT_DETAILS=$(echo "$SCORE_RESULT" | jq -c '.element_details')
        ERROR_DETAILS=$(echo "$SCORE_RESULT" | jq -c '.error_details')

        # Accumulate totals
        N_ELEMENTS=$(echo "$EXPECTED_ELEMENTS" | jq 'length')
        N_COVERED=$(echo "$SCORE_RESULT" | jq -r '.elements_covered')
        N_POSSIBLE_ERRORS=$(echo "$INCORRECT_INTERPS" | jq 'length')
        N_ERRORS=$(echo "$SCORE_RESULT" | jq -r '.errors_count')

        TOTAL_ELEMENTS=$((TOTAL_ELEMENTS + N_ELEMENTS))
        COVERED_ELEMENTS=$((COVERED_ELEMENTS + N_COVERED))
        TOTAL_ERRORS=$((TOTAL_ERRORS + N_POSSIBLE_ERRORS))
        ERROR_COUNT=$((ERROR_COUNT + N_ERRORS))

        # Write result
        RESULT=$(jq -n \
            --arg timestamp "$(date -Iseconds)" \
            --arg model "$MODEL" \
            --arg test_id "$TEST_ID" \
            --arg category "$CATEGORY_NAME" \
            --arg task_type "$TASK_TYPE" \
            --arg prompt "$INTERP_PROMPT" \
            --arg response "$RESPONSE" \
            --argjson element_coverage "$ELEMENT_COVERAGE" \
            --argjson errors_detected "$ERROR_DETECTED" \
            --argjson accuracy "$ACCURACY" \
            --argjson element_details "$ELEMENT_DETAILS" \
            --argjson error_details "$ERROR_DETAILS" \
            --argjson latency_ms "$LATENCY" \
            '{
                timestamp: $timestamp,
                model: $model,
                test_id: $test_id,
                category: $category,
                task_type: $task_type,
                prompt: $prompt,
                response: $response,
                element_coverage: $element_coverage,
                errors_detected: $errors_detected,
                accuracy: $accuracy,
                element_details: $element_details,
                error_details: $error_details,
                latency_ms: $latency_ms
            }')

        echo "$RESULT" >> "$OUTPUT_FILE"

        echo "  $TEST_ID: coverage=$ELEMENT_COVERAGE, errors=$ERROR_DETECTED, accuracy=$ACCURACY"

        sleep 0.5
    done
done

if [[ "$DRY_RUN" != "--dry-run" ]]; then
    echo ""
    echo "========================================"
    echo "Summary"
    echo "========================================"
    echo "Total Tests: $TOTAL"

    if [[ $TOTAL_ELEMENTS -gt 0 ]]; then
        ELEM_RATE=$(echo "scale=1; $COVERED_ELEMENTS * 100 / $TOTAL_ELEMENTS" | bc)
        echo "Element Coverage: $COVERED_ELEMENTS/$TOTAL_ELEMENTS (${ELEM_RATE}%)"
    fi

    if [[ $TOTAL_ERRORS -gt 0 ]]; then
        ERR_RATE=$(echo "scale=1; $ERROR_COUNT * 100 / $TOTAL_ERRORS" | bc)
        echo "Error Rate: $ERROR_COUNT/$TOTAL_ERRORS (${ERR_RATE}%)"
    fi

    AVG_ACC=$(jq -s '[.[].accuracy] | add / length' "$OUTPUT_FILE")
    echo "Average Accuracy: $(printf "%.3f" $AVG_ACC)"

    echo ""
    echo "Results saved to: $OUTPUT_FILE"
fi
