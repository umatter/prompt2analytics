#!/bin/bash
# Score parameter extraction accuracy
# Usage: ./scripts/score_parameters.sh <expected_params_json> <selected_args_json>

EXPECTED_PARAMS="$1"
SELECTED_ARGS="$2"

# Initialize counters
EXPECTED_COUNT=0
EXTRACTED_COUNT=0
CORRECT_COUNT=0
DETAILS="[]"

# Get list of expected parameters
PARAM_NAMES=$(echo "$EXPECTED_PARAMS" | jq -r 'keys[]')

for PARAM in $PARAM_NAMES; do
    ((++EXPECTED_COUNT))

    # Get expected value and alternatives
    EXPECTED_VAL=$(echo "$EXPECTED_PARAMS" | jq -r ".\"$PARAM\".expected")
    ALTERNATIVES=$(echo "$EXPECTED_PARAMS" | jq -c ".\"$PARAM\".alternatives // []")
    IS_IMPLICIT=$(echo "$EXPECTED_PARAMS" | jq -r ".\"$PARAM\".implicit // false")
    ORDER_MATTERS=$(echo "$EXPECTED_PARAMS" | jq -r ".\"$PARAM\".order_matters // true")

    # Get selected value (try different possible key names)
    SELECTED_VAL=$(echo "$SELECTED_ARGS" | jq -r ".\"$PARAM\" // .\"${PARAM}_col\" // .\"${PARAM}s\" // \"__NOT_FOUND__\"")

    # Check if parameter was extracted
    EXTRACTED="false"
    CORRECT="false"

    if [[ "$SELECTED_VAL" != "__NOT_FOUND__" && "$SELECTED_VAL" != "null" ]]; then
        EXTRACTED="true"
        ((++EXTRACTED_COUNT))

        # Check for exact match
        if [[ "$SELECTED_VAL" == "$EXPECTED_VAL" ]]; then
            CORRECT="true"
            ((++CORRECT_COUNT))
        else
            # Check alternatives
            ALT_MATCH=$(echo "$ALTERNATIVES" | jq --arg sel "$SELECTED_VAL" 'any(. == $sel)')
            if [[ "$ALT_MATCH" == "true" ]]; then
                CORRECT="true"
                ((++CORRECT_COUNT))
            else
                # Handle array comparisons (for x_cols, instruments, etc.)
                if echo "$EXPECTED_VAL" | jq -e 'type == "array"' > /dev/null 2>&1; then
                    if echo "$SELECTED_VAL" | jq -e 'type == "array"' > /dev/null 2>&1; then
                        if [[ "$ORDER_MATTERS" == "false" ]]; then
                            # Sort and compare
                            EXPECTED_SORTED=$(echo "$EXPECTED_VAL" | jq -c 'sort')
                            SELECTED_SORTED=$(echo "$SELECTED_VAL" | jq -c 'sort')
                            if [[ "$EXPECTED_SORTED" == "$SELECTED_SORTED" ]]; then
                                CORRECT="true"
                                ((++CORRECT_COUNT))
                            fi
                        else
                            if [[ "$EXPECTED_VAL" == "$SELECTED_VAL" ]]; then
                                CORRECT="true"
                                ((++CORRECT_COUNT))
                            fi
                        fi
                    fi
                fi

                # Handle numeric tolerance
                if echo "$EXPECTED_VAL" | grep -qE '^-?[0-9]+\.?[0-9]*$'; then
                    if echo "$SELECTED_VAL" | grep -qE '^-?[0-9]+\.?[0-9]*$'; then
                        DIFF=$(echo "$EXPECTED_VAL - $SELECTED_VAL" | bc -l | tr -d '-')
                        if (( $(echo "$DIFF < 0.001" | bc -l) )); then
                            CORRECT="true"
                            ((++CORRECT_COUNT))
                        fi
                    fi
                fi
            fi
        fi
    elif [[ "$IS_IMPLICIT" == "true" ]]; then
        # Implicit parameters that use defaults are considered correct if not specified
        CORRECT="true"
        ((++CORRECT_COUNT))
    fi

    # Add to details
    DETAIL=$(jq -n \
        --arg param "$PARAM" \
        --arg expected "$EXPECTED_VAL" \
        --arg selected "$SELECTED_VAL" \
        --argjson extracted "$EXTRACTED" \
        --argjson correct "$CORRECT" \
        --argjson implicit "$IS_IMPLICIT" \
        '{
            parameter: $param,
            expected: $expected,
            selected: $selected,
            extracted: $extracted,
            correct: $correct,
            implicit: $implicit
        }')

    DETAILS=$(echo "$DETAILS" | jq --argjson detail "$DETAIL" '. + [$detail]')
done

# Calculate metrics
if [[ $EXTRACTED_COUNT -gt 0 ]]; then
    PRECISION=$(echo "scale=3; $CORRECT_COUNT / $EXTRACTED_COUNT" | bc)
else
    PRECISION="0"
fi

if [[ $EXPECTED_COUNT -gt 0 ]]; then
    RECALL=$(echo "scale=3; $CORRECT_COUNT / $EXPECTED_COUNT" | bc)
else
    RECALL="0"
fi

if (( $(echo "$PRECISION + $RECALL > 0" | bc -l) )); then
    F1=$(echo "scale=3; 2 * $PRECISION * $RECALL / ($PRECISION + $RECALL)" | bc)
else
    F1="0"
fi

# Output result
jq -n \
    --argjson precision "$PRECISION" \
    --argjson recall "$RECALL" \
    --argjson f1 "$F1" \
    --argjson expected_count "$EXPECTED_COUNT" \
    --argjson extracted_count "$EXTRACTED_COUNT" \
    --argjson correct_count "$CORRECT_COUNT" \
    --argjson details "$DETAILS" \
    '{
        precision: $precision,
        recall: $recall,
        f1: $f1,
        expected_count: $expected_count,
        extracted_count: $extracted_count,
        correct_count: $correct_count,
        details: $details
    }'
