#!/bin/bash
# Score out-of-scope response
# Usage: ./scripts/score_oos_response.sh <selected_tool> <response_content> <closest_available_json>

SELECTED_TOOL="$1"
RESPONSE_CONTENT="$2"
CLOSEST_AVAILABLE="$3"

RESPONSE_LOWER=$(echo "$RESPONSE_CONTENT" | tr '[:upper:]' '[:lower:]')

DETECTED_OOS="false"
SELECTED_FALSE_TOOL="false"
SUGGESTED_ALTERNATIVE="false"
SCORE_TYPE="false_tool"

# Check if model indicated out of scope
if [[ "$SELECTED_TOOL" == "null" || "$SELECTED_TOOL" == "" ]]; then
    # No tool selected - check if response indicates OOS
    if echo "$RESPONSE_LOWER" | grep -qE '(not supported|out of scope|cannot|not available|not implement|don'\''t have|doesn'\''t support|unable to|not capable)'; then
        DETECTED_OOS="true"
        SCORE_TYPE="oos_detected"
    fi
else
    # A tool was selected - this is a false positive (incorrectly selected a tool)
    SELECTED_FALSE_TOOL="true"
    SCORE_TYPE="false_tool"

    # Check if selected tool is one of the closest available (partial credit)
    IS_CLOSEST=$(echo "$CLOSEST_AVAILABLE" | jq --arg tool "$SELECTED_TOOL" 'any(. == $tool)')
    if [[ "$IS_CLOSEST" == "true" ]]; then
        # Selected a related tool - some graceful degradation
        SUGGESTED_ALTERNATIVE="true"
        SCORE_TYPE="graceful"
        DETECTED_OOS="false"
    fi
fi

# Check if response suggested alternatives even if tool selected
if [[ "$DETECTED_OOS" == "true" ]]; then
    # Check for alternative suggestions
    for TOOL in $(echo "$CLOSEST_AVAILABLE" | jq -r '.[]'); do
        TOOL_PATTERN=$(echo "$TOOL" | sed 's/_/ /g')
        if echo "$RESPONSE_LOWER" | grep -qE "(${TOOL}|${TOOL_PATTERN}|alternative|instead|similar)"; then
            SUGGESTED_ALTERNATIVE="true"
            SCORE_TYPE="graceful"
            break
        fi
    done
fi

# Output result
jq -n \
    --argjson detected_oos "$DETECTED_OOS" \
    --argjson selected_false_tool "$SELECTED_FALSE_TOOL" \
    --argjson suggested_alternative "$SUGGESTED_ALTERNATIVE" \
    --arg score_type "$SCORE_TYPE" \
    '{
        detected_oos: $detected_oos,
        selected_false_tool: $selected_false_tool,
        suggested_alternative: $suggested_alternative,
        score_type: $score_type
    }'
