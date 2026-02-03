#!/bin/bash
#
# score_response.sh - Score LLM tool selection response
#
# Usage: ./score_response.sh <selected> <expected_json> <acceptable_json> <category>
#
# Returns JSON with match_type and score

SELECTED="$1"
EXPECTED_JSON="$2"
ACCEPTABLE_JSON="$3"
CATEGORY="$4"

# Handle null/empty selection
if [[ -z "$SELECTED" || "$SELECTED" == "null" || "$SELECTED" == "none" ]]; then
    echo '{"match_type": "none", "score": 0.0}'
    exit 0
fi

# Check for exact match
EXACT_MATCH=$(echo "$EXPECTED_JSON" | jq -r --arg sel "$SELECTED" 'if . | type == "array" then (. | index($sel) != null) else . == $sel end')

if [[ "$EXACT_MATCH" == "true" ]]; then
    echo '{"match_type": "exact", "score": 1.0}'
    exit 0
fi

# Check for acceptable match
ACCEPTABLE_MATCH=$(echo "$ACCEPTABLE_JSON" | jq -r --arg sel "$SELECTED" 'if . | type == "array" then (. | index($sel) != null) else . == $sel end')

if [[ "$ACCEPTABLE_MATCH" == "true" ]]; then
    echo '{"match_type": "acceptable", "score": 0.7}'
    exit 0
fi

# Check for category match (same prefix)
# Map categories to tool prefixes
case "$CATEGORY" in
    regression)
        PREFIX="regression_"
        ;;
    panel)
        PREFIX="panel_|hausman_|feglm"
        ;;
    causal)
        PREFIX="iv_|diff_|treatment_|rd_|mediation_|synthetic_"
        ;;
    discrete)
        PREFIX="logit|probit|feglm"
        ;;
    timeseries)
        PREFIX="ts_|timeseries_"
        ;;
    hypothesis)
        PREFIX="hypothesis_|anova_|power_|compute_correlation"
        ;;
    ml)
        PREFIX="ml_|multivariate_"
        ;;
    viz)
        PREFIX="viz_"
        ;;
    *)
        PREFIX=""
        ;;
esac

if [[ -n "$PREFIX" ]]; then
    if echo "$SELECTED" | grep -qE "^($PREFIX)"; then
        echo '{"match_type": "category", "score": 0.3}'
        exit 0
    fi
fi

# No match
echo '{"match_type": "none", "score": 0.0}'
