#!/bin/bash
# Score interpretation response for element coverage and errors
# Usage: ./scripts/score_interpretation.sh <response> <expected_elements_json> <incorrect_interps_json>

RESPONSE="$1"
EXPECTED_ELEMENTS="$2"
INCORRECT_INTERPS="$3"

# Convert response to lowercase for matching
RESPONSE_LOWER=$(echo "$RESPONSE" | tr '[:upper:]' '[:lower:]')

# Initialize results
ELEMENTS_COVERED=0
ELEMENTS_REQUIRED=0
ELEMENT_DETAILS="[]"

ERRORS_FOUND=0
ERROR_DETAILS="[]"

# Check for expected elements
N_ELEMENTS=$(echo "$EXPECTED_ELEMENTS" | jq 'length')

for ((i=0; i<N_ELEMENTS; i++)); do
    ELEMENT=$(echo "$EXPECTED_ELEMENTS" | jq -r ".[$i].element")
    REQUIRED=$(echo "$EXPECTED_ELEMENTS" | jq -r ".[$i].required")
    DESCRIPTION=$(echo "$EXPECTED_ELEMENTS" | jq -r ".[$i].description")

    if [[ "$REQUIRED" == "true" ]]; then
        ((++ELEMENTS_REQUIRED))
    fi

    # Check if element is covered in response (keyword-based heuristic)
    FOUND="false"

    case "$ELEMENT" in
        "percentage_interpretation"|"percentage_interpret")
            if echo "$RESPONSE_LOWER" | grep -qE '(percent|%|8\.3|8\.32|proportional)'; then
                FOUND="true"
            fi
            ;;
        "log_scale_awareness")
            if echo "$RESPONSE_LOWER" | grep -qE '(log|logarithm|semi-elastic|percentage change)'; then
                FOUND="true"
            fi
            ;;
        "statistical_significance"|"significance")
            if echo "$RESPONSE_LOWER" | grep -qE '(significant|p-value|p value|p<|statistically)'; then
                FOUND="true"
            fi
            ;;
        "ceteris_paribus"|"controlled_comparison")
            if echo "$RESPONSE_LOWER" | grep -qE '(holding.*constant|ceteris paribus|controlling|other things equal|all else equal)'; then
                FOUND="true"
            fi
            ;;
        "marginal_effect_formula"|"marginal_effect")
            if echo "$RESPONSE_LOWER" | grep -qE '(marginal effect|derivative|dy/dx|partial)'; then
                FOUND="true"
            fi
            ;;
        "turning_point")
            if echo "$RESPONSE_LOWER" | grep -qE '(turning point|maximum|minimum|peak|34|35)'; then
                FOUND="true"
            fi
            ;;
        "diminishing_returns")
            if echo "$RESPONSE_LOWER" | grep -qE '(diminishing|decreasing|negative squared|concave|inverted)'; then
                FOUND="true"
            fi
            ;;
        "reference_category"|"gender_gap")
            if echo "$RESPONSE_LOWER" | grep -qE '(reference|baseline|compared to|relative to|gap|difference|male|men)'; then
                FOUND="true"
            fi
            ;;
        "differential_effect"|"interaction")
            if echo "$RESPONSE_LOWER" | grep -qE '(interaction|differential|varies|depends on|moderat|differ)'; then
                FOUND="true"
            fi
            ;;
        "context_matters"|"r_squared")
            if echo "$RESPONSE_LOWER" | grep -qE '(cross-section|individual|context|common|typical|expected)'; then
                FOUND="true"
            fi
            ;;
        "coefficients_valid"|"coefficients_unbiased")
            if echo "$RESPONSE_LOWER" | grep -qE '(unbiased|coefficient.*valid|point estimate|still consistent)'; then
                FOUND="true"
            fi
            ;;
        "reject_null")
            if echo "$RESPONSE_LOWER" | grep -qE '(reject|significant|p.*<|less than 0\.05)'; then
                FOUND="true"
            fi
            ;;
        "probability_interpretation")
            if echo "$RESPONSE_LOWER" | grep -qE '(probability|chance|likelihood|if.*null.*true)'; then
                FOUND="true"
            fi
            ;;
        "evidence_language")
            if echo "$RESPONSE_LOWER" | grep -qE '(evidence|suggests|indicates|support)'; then
                FOUND="true"
            fi
            ;;
        "parallel_trends"|"parallel_trend")
            if echo "$RESPONSE_LOWER" | grep -qE '(parallel|common trend|pre-trend|similar trend)'; then
                FOUND="true"
            fi
            ;;
        "late_interpretation"|"local_effect")
            if echo "$RESPONSE_LOWER" | grep -qE '(late|local|complier|near.*cutoff|margin)'; then
                FOUND="true"
            fi
            ;;
        "selection_on_observables"|"unobserved_confounders")
            if echo "$RESPONSE_LOWER" | grep -qE '(unobserved|unmeasured|confound|selection|observ)'; then
                FOUND="true"
            fi
            ;;
        "heteroskedasticity_presence"|"se_affected")
            if echo "$RESPONSE_LOWER" | grep -qE '(heteroskedastic|variance|standard error|se.*bias|inconsistent)'; then
                FOUND="true"
            fi
            ;;
        "robust_se_solution")
            if echo "$RESPONSE_LOWER" | grep -qE '(robust|hc[0-3]|white|huber|sandwich)'; then
                FOUND="true"
            fi
            ;;
        "cannot_prove_null"|"power_considerations")
            if echo "$RESPONSE_LOWER" | grep -qE '(cannot prove|fail.*reject|power|sample size|type ii)'; then
                FOUND="true"
            fi
            ;;
        "at_least_one_differs"|"post_hoc_needed")
            if echo "$RESPONSE_LOWER" | grep -qE '(at least one|some.*differ|post.*hoc|pairwise|multiple comparison)'; then
                FOUND="true"
            fi
            ;;
        "multiple_testing_problem"|"adjustment_needed")
            if echo "$RESPONSE_LOWER" | grep -qE '(multiple.*test|bonferroni|false.*positive|fdr|adjustment|correction)'; then
                FOUND="true"
            fi
            ;;
        *)
            # Generic check: look for keywords from description
            KEYWORDS=$(echo "$DESCRIPTION" | tr '[:upper:]' '[:lower:]' | grep -oE '[a-z]{4,}' | head -5)
            for KW in $KEYWORDS; do
                if echo "$RESPONSE_LOWER" | grep -q "$KW"; then
                    FOUND="true"
                    break
                fi
            done
            ;;
    esac

    if [[ "$FOUND" == "true" ]]; then
        ((++ELEMENTS_COVERED))
    fi

    DETAIL=$(jq -n \
        --arg element "$ELEMENT" \
        --argjson required "$REQUIRED" \
        --argjson found "$FOUND" \
        --arg description "$DESCRIPTION" \
        '{element: $element, required: $required, found: $found, description: $description}')

    ELEMENT_DETAILS=$(echo "$ELEMENT_DETAILS" | jq --argjson detail "$DETAIL" '. + [$detail]')
done

# Check for incorrect interpretations (errors)
N_ERRORS=$(echo "$INCORRECT_INTERPS" | jq 'length')

for ((i=0; i<N_ERRORS; i++)); do
    ERROR=$(echo "$INCORRECT_INTERPS" | jq -r ".[$i].error")
    ERROR_DESC=$(echo "$INCORRECT_INTERPS" | jq -r ".[$i].description")

    ERROR_FOUND="false"

    case "$ERROR" in
        "absolute_interpretation")
            if echo "$RESPONSE_LOWER" | grep -qE '\$0\.08|\$0\.0832|dollar.*increase.*0\.08'; then
                ERROR_FOUND="true"
            fi
            ;;
        "probability_of_null"|"proves_alternative")
            if echo "$RESPONSE_LOWER" | grep -qE '(probability.*null.*true|proves|definitely|certainly)'; then
                ERROR_FOUND="true"
            fi
            ;;
        "causal_language"|"causal_claim")
            if echo "$RESPONSE_LOWER" | grep -qE '(causes|caused by|causal effect)' && ! echo "$RESPONSE_LOWER" | grep -qE '(assumption|if|caution|caveat)'; then
                ERROR_FOUND="true"
            fi
            ;;
        "proves_no_effect"|"accept_null")
            if echo "$RESPONSE_LOWER" | grep -qE '(no effect|accept.*null|proves.*no|definitely no)'; then
                ERROR_FOUND="true"
            fi
            ;;
        "coefficients_biased"|"coefficient_bias")
            if echo "$RESPONSE_LOWER" | grep -qE '(coefficient.*biased|biased.*coefficient|ols.*biased)' && echo "$RESPONSE_LOWER" | grep -qE '(heteroskedastic)'; then
                ERROR_FOUND="true"
            fi
            ;;
        "ate_claim")
            if echo "$RESPONSE_LOWER" | grep -qE '(average treatment effect|ate)' && ! echo "$RESPONSE_LOWER" | grep -qE '(late|local|complier)'; then
                ERROR_FOUND="true"
            fi
            ;;
        *)
            # Heuristic check based on description keywords
            ;;
    esac

    if [[ "$ERROR_FOUND" == "true" ]]; then
        ((++ERRORS_FOUND))
    fi

    E_DETAIL=$(jq -n \
        --arg error "$ERROR" \
        --argjson found "$ERROR_FOUND" \
        --arg description "$ERROR_DESC" \
        '{error: $error, found: $found, description: $description}')

    ERROR_DETAILS=$(echo "$ERROR_DETAILS" | jq --argjson detail "$E_DETAIL" '. + [$detail]')
done

# Calculate metrics
if [[ $N_ELEMENTS -gt 0 ]]; then
    ELEMENT_COVERAGE=$(echo "scale=3; $ELEMENTS_COVERED / $N_ELEMENTS" | bc)
else
    ELEMENT_COVERAGE="1"
fi

if [[ $N_ERRORS -gt 0 ]]; then
    ERROR_RATE=$(echo "scale=3; $ERRORS_FOUND / $N_ERRORS" | bc)
else
    ERROR_RATE="0"
fi

# Accuracy = element coverage * (1 - error_rate)
# Higher is better: more elements covered, fewer errors
ACCURACY=$(echo "scale=3; $ELEMENT_COVERAGE * (1 - $ERROR_RATE)" | bc)

# Output result
jq -n \
    --argjson element_coverage "$ELEMENT_COVERAGE" \
    --argjson elements_covered "$ELEMENTS_COVERED" \
    --argjson elements_total "$N_ELEMENTS" \
    --argjson errors_detected "$ERRORS_FOUND" \
    --argjson errors_count "$ERRORS_FOUND" \
    --argjson errors_total "$N_ERRORS" \
    --argjson accuracy "$ACCURACY" \
    --argjson element_details "$ELEMENT_DETAILS" \
    --argjson error_details "$ERROR_DETAILS" \
    '{
        element_coverage: $element_coverage,
        elements_covered: $elements_covered,
        elements_total: $elements_total,
        errors_detected: $errors_detected,
        errors_count: $errors_count,
        errors_total: $errors_total,
        accuracy: $accuracy,
        element_details: $element_details,
        error_details: $error_details
    }'
