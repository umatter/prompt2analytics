#!/bin/bash
#
# run_eval.sh - Main evaluation runner for LLM tool selection
#
# Usage: ./run_eval.sh <model> <provider> <category|all> [--dry-run]
#
# Examples:
#   ./run_eval.sh gpt-4o openai all
#   ./run_eval.sh llama3.2 ollama regression
#   ./run_eval.sh gpt-4o-mini openai panel --dry-run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$BASE_DIR/config"
TEST_DIR="$BASE_DIR/test_cases"
RESULTS_DIR="$BASE_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
MODEL="${1:-gpt-4o}"
PROVIDER="${2:-openai}"
CATEGORY="${3:-all}"
DRY_RUN=false

if [[ "$4" == "--dry-run" ]] || [[ "$3" == "--dry-run" ]]; then
    DRY_RUN=true
    if [[ "$3" == "--dry-run" ]]; then
        CATEGORY="all"
    fi
fi

# Validate provider
if [[ "$PROVIDER" != "openai" && "$PROVIDER" != "ollama" && "$PROVIDER" != "anthropic" && "$PROVIDER" != "openrouter" ]]; then
    echo -e "${RED}Error: Provider must be 'openai', 'anthropic', 'openrouter', or 'ollama'${NC}"
    exit 1
fi

# Check API key for OpenAI
if [[ "$PROVIDER" == "openai" && -z "$OPENAI_API_KEY" ]]; then
    echo -e "${RED}Error: OPENAI_API_KEY environment variable not set${NC}"
    exit 1
fi

# Check API key for Anthropic
if [[ "$PROVIDER" == "anthropic" && -z "$ANTHROPIC_API_KEY" ]]; then
    echo -e "${RED}Error: ANTHROPIC_API_KEY environment variable not set${NC}"
    exit 1
fi

# Check API key for OpenRouter
if [[ "$PROVIDER" == "openrouter" && -z "$OPENROUTER_API_KEY" ]]; then
    echo -e "${RED}Error: OPENROUTER_API_KEY environment variable not set${NC}"
    exit 1
fi

# Check Ollama is running for local models
if [[ "$PROVIDER" == "ollama" ]]; then
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${RED}Error: Ollama not running. Start with 'ollama serve'${NC}"
        exit 1
    fi
fi

# Create results directory
mkdir -p "$RESULTS_DIR"

# Generate output filename (sanitize model name for filesystem)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SAFE=$(echo "$MODEL" | tr '/' '_')
OUTPUT_FILE="$RESULTS_DIR/${MODEL_SAFE}_${CATEGORY}_${TIMESTAMP}.jsonl"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}LLM Tool Selection Evaluation${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Model:    ${GREEN}$MODEL${NC}"
echo -e "Provider: ${GREEN}$PROVIDER${NC}"
echo -e "Category: ${GREEN}$CATEGORY${NC}"
echo -e "Output:   ${GREEN}$OUTPUT_FILE${NC}"
if $DRY_RUN; then
    echo -e "${YELLOW}DRY RUN - No API calls will be made${NC}"
fi
echo ""

# Get list of test files
if [[ "$CATEGORY" == "all" ]]; then
    TEST_FILES=$(ls "$TEST_DIR"/*.json 2>/dev/null || true)
else
    TEST_FILES="$TEST_DIR/${CATEGORY}.json"
    if [[ ! -f "$TEST_FILES" ]]; then
        echo -e "${RED}Error: Test file not found: $TEST_FILES${NC}"
        echo "Available categories:"
        ls "$TEST_DIR"/*.json | xargs -n1 basename | sed 's/.json//'
        exit 1
    fi
fi

# Counters
TOTAL=0
EXACT=0
ACCEPTABLE=0
CATEGORY_MATCH=0
FAILED=0

# Process each test file
for TEST_FILE in $TEST_FILES; do
    CURRENT_CATEGORY=$(basename "$TEST_FILE" .json)
    echo -e "${BLUE}Processing category: ${CURRENT_CATEGORY}${NC}"

    # Get number of tests in file
    NUM_TESTS=$(jq '.tests | length' "$TEST_FILE")

    for i in $(seq 0 $((NUM_TESTS - 1))); do
        # Extract test data
        TEST_ID=$(jq -r ".tests[$i].id" "$TEST_FILE")
        PROMPT=$(jq -r ".tests[$i].prompt" "$TEST_FILE")
        DATASET_CTX=$(jq -r ".tests[$i].dataset_context" "$TEST_FILE")
        EXPECTED=$(jq -r ".tests[$i].expected_tools | @json" "$TEST_FILE")
        ACCEPTABLE_TOOLS=$(jq -r ".tests[$i].acceptable_tools | @json" "$TEST_FILE")
        DIFFICULTY=$(jq -r ".tests[$i].difficulty" "$TEST_FILE")

        TOTAL=$((TOTAL + 1))

        echo -n "  [$TEST_ID] "

        if $DRY_RUN; then
            echo -e "${YELLOW}(dry run)${NC} $PROMPT"
            continue
        fi

        # Build the prompt and call the LLM
        START_TIME=$(date +%s%3N)

        FULL_PROMPT=$(bash "$SCRIPT_DIR/build_prompt.sh" "$PROMPT" "$DATASET_CTX")

        if [[ "$PROVIDER" == "openai" ]]; then
            RESPONSE=$(bash "$SCRIPT_DIR/call_openai.sh" "$MODEL" "$FULL_PROMPT" "$CONFIG_DIR/tools.json")
        elif [[ "$PROVIDER" == "anthropic" ]]; then
            RESPONSE=$(bash "$SCRIPT_DIR/call_anthropic.sh" "$MODEL" "$FULL_PROMPT" "$CONFIG_DIR/tools.json")
        elif [[ "$PROVIDER" == "openrouter" ]]; then
            RESPONSE=$(bash "$SCRIPT_DIR/call_openrouter.sh" "$MODEL" "$FULL_PROMPT" "$CONFIG_DIR/tools.json")
        else
            RESPONSE=$(bash "$SCRIPT_DIR/call_ollama.sh" "$MODEL" "$FULL_PROMPT" "$CONFIG_DIR/tools.json")
        fi

        END_TIME=$(date +%s%3N)
        LATENCY=$((END_TIME - START_TIME))

        # Extract selected tool from response
        SELECTED=$(echo "$RESPONSE" | jq -r '.tool // empty')

        if [[ -z "$SELECTED" ]]; then
            SELECTED="none"
        fi

        # Score the response
        SCORE_RESULT=$(bash "$SCRIPT_DIR/score_response.sh" "$SELECTED" "$EXPECTED" "$ACCEPTABLE_TOOLS" "$CURRENT_CATEGORY")
        MATCH_TYPE=$(echo "$SCORE_RESULT" | jq -r '.match_type')
        SCORE=$(echo "$SCORE_RESULT" | jq -r '.score')

        # Update counters
        case "$MATCH_TYPE" in
            exact)
                EXACT=$((EXACT + 1))
                echo -e "${GREEN}✓${NC} exact ($SELECTED) [${LATENCY}ms]"
                ;;
            acceptable)
                ACCEPTABLE=$((ACCEPTABLE + 1))
                echo -e "${YELLOW}~${NC} acceptable ($SELECTED) [${LATENCY}ms]"
                ;;
            category)
                CATEGORY_MATCH=$((CATEGORY_MATCH + 1))
                echo -e "${YELLOW}○${NC} category ($SELECTED) [${LATENCY}ms]"
                ;;
            *)
                FAILED=$((FAILED + 1))
                echo -e "${RED}✗${NC} wrong ($SELECTED) [${LATENCY}ms]"
                ;;
        esac

        # Write result to JSONL (compact format with -c)
        RESULT=$(jq -nc \
            --arg ts "$(date -Iseconds)" \
            --arg model "$MODEL" \
            --arg test_id "$TEST_ID" \
            --arg category "$CURRENT_CATEGORY" \
            --arg prompt "$PROMPT" \
            --arg selected "$SELECTED" \
            --argjson expected "$EXPECTED" \
            --arg match_type "$MATCH_TYPE" \
            --arg score "$SCORE" \
            --arg latency_ms "$LATENCY" \
            --arg difficulty "$DIFFICULTY" \
            '{
                timestamp: $ts,
                model: $model,
                test_id: $test_id,
                category: $category,
                prompt: $prompt,
                selected: $selected,
                expected: $expected,
                match_type: $match_type,
                score: ($score | tonumber),
                latency_ms: ($latency_ms | tonumber),
                difficulty: $difficulty
            }')

        echo "$RESULT" >> "$OUTPUT_FILE"

        # Small delay to avoid rate limiting
        sleep 0.5
    done

    echo ""
done

if $DRY_RUN; then
    echo -e "${YELLOW}Dry run complete. $TOTAL tests would be executed.${NC}"
    exit 0
fi

# Print summary
echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}================================${NC}"
echo -e "Total tests:     $TOTAL"
echo -e "Exact matches:   ${GREEN}$EXACT${NC} ($(echo "scale=1; $EXACT * 100 / $TOTAL" | bc)%)"
echo -e "Acceptable:      ${YELLOW}$ACCEPTABLE${NC} ($(echo "scale=1; $ACCEPTABLE * 100 / $TOTAL" | bc)%)"
echo -e "Category match:  ${YELLOW}$CATEGORY_MATCH${NC} ($(echo "scale=1; $CATEGORY_MATCH * 100 / $TOTAL" | bc)%)"
echo -e "Failed:          ${RED}$FAILED${NC} ($(echo "scale=1; $FAILED * 100 / $TOTAL" | bc)%)"
echo ""
echo -e "Accuracy (exact + acceptable): $(echo "scale=1; ($EXACT + $ACCEPTABLE) * 100 / $TOTAL" | bc)%"
echo ""
echo -e "Results saved to: ${GREEN}$OUTPUT_FILE${NC}"

# Generate report
bash "$SCRIPT_DIR/generate_report.sh" "$OUTPUT_FILE"
