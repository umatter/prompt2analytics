#!/bin/bash
#
# call_openrouter.sh - OpenRouter API wrapper with tool calling
#
# Usage: ./call_openrouter.sh <model> <prompt> <tools_file>
#
# OpenRouter provides access to many models via OpenAI-compatible API
# Models are specified as provider/model-name (e.g., meta-llama/llama-3.2-3b-instruct)
#
# Returns JSON with selected tool

set -e

MODEL="$1"
PROMPT="$2"
TOOLS_FILE="$3"

if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo '{"error": "OPENROUTER_API_KEY not set"}' >&2
    exit 1
fi

# Load tools from config (OpenAI format works for OpenRouter)
TOOLS=$(cat "$TOOLS_FILE" | jq '.tools')

# Build the request (OpenAI-compatible format)
REQUEST=$(jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --argjson tools "$TOOLS" \
    '{
        model: $model,
        messages: [
            {
                role: "system",
                content: "You are a helpful econometrics assistant. When the user describes an analysis task, select the single most appropriate tool from the available tools. Only call one tool."
            },
            {
                role: "user",
                content: $prompt
            }
        ],
        tools: $tools,
        tool_choice: "required",
        temperature: 0,
        max_tokens: 1024
    }')

# Make the API call
RESPONSE=$(curl -s -X POST "https://openrouter.ai/api/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENROUTER_API_KEY" \
    -H "HTTP-Referer: https://github.com/p2a-paper" \
    -H "X-Title: p2a-llm-eval" \
    -d "$REQUEST")

# Check for errors
ERROR=$(echo "$RESPONSE" | jq -r '.error.message // empty')
if [[ -n "$ERROR" ]]; then
    echo "{\"error\": \"$ERROR\"}" >&2
    exit 1
fi

# Extract the tool call (same format as OpenAI)
TOOL_CALL=$(echo "$RESPONSE" | jq -r '.choices[0].message.tool_calls[0].function.name // empty')

if [[ -z "$TOOL_CALL" ]]; then
    # Some models may not support tool calling - try to extract from content
    CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty')

    # Try to extract tool name from content if model doesn't support tool calling natively
    EXTRACTED_TOOL=$(echo "$CONTENT" | grep -oE '\b(regression_|panel_|iv_|diff_|treatment_|rd_|logit|probit|ts_|timeseries_|hypothesis_|anova_|ml_|viz_|multivariate_|kaplan_|cox_|compute_|describe_|load_|feglm|hausman_|mediation_|synthetic_|power_)[a-z_]+' | head -1 || true)

    if [[ -n "$EXTRACTED_TOOL" ]]; then
        echo "{\"tool\": \"$EXTRACTED_TOOL\", \"content\": \"$CONTENT\", \"extracted\": true}"
    else
        echo "{\"tool\": null, \"content\": \"$CONTENT\"}"
    fi
else
    ARGUMENTS=$(echo "$RESPONSE" | jq -c '.choices[0].message.tool_calls[0].function.arguments // {}')
    echo "{\"tool\": \"$TOOL_CALL\", \"arguments\": $ARGUMENTS}"
fi
