#!/bin/bash
#
# call_ollama.sh - Ollama API wrapper with tool calling
#
# Usage: ./call_ollama.sh <model> <prompt> <tools_file>
#
# Returns JSON with selected tool

set -e

MODEL="$1"
PROMPT="$2"
TOOLS_FILE="$3"

OLLAMA_ENDPOINT="${OLLAMA_ENDPOINT:-http://localhost:11434}"

# Load tools and convert to Ollama format
# Ollama uses a similar but slightly different tool format
TOOLS=$(cat "$TOOLS_FILE" | jq '[.tools[] | {
    type: "function",
    function: {
        name: .function.name,
        description: .function.description,
        parameters: .function.parameters
    }
}]')

# Build the request
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
        stream: false,
        options: {
            temperature: 0
        }
    }')

# Make the API call
RESPONSE=$(curl -s -X POST "$OLLAMA_ENDPOINT/api/chat" \
    -H "Content-Type: application/json" \
    -d "$REQUEST")

# Check for errors
ERROR=$(echo "$RESPONSE" | jq -r '.error // empty')
if [[ -n "$ERROR" ]]; then
    echo "{\"error\": \"$ERROR\"}" >&2
    exit 1
fi

# Extract the tool call from Ollama response
# Ollama returns tool_calls in the message
TOOL_CALL=$(echo "$RESPONSE" | jq -r '.message.tool_calls[0].function.name // empty')

if [[ -z "$TOOL_CALL" ]]; then
    # If no tool call, check the content for tool mentions
    CONTENT=$(echo "$RESPONSE" | jq -r '.message.content // empty')

    # Try to extract tool name from content if model doesn't support tool calling natively
    # Look for patterns like "I would use regression_ols" or "tool: regression_ols"
    EXTRACTED_TOOL=$(echo "$CONTENT" | grep -oE '\b(regression_|panel_|iv_|diff_|treatment_|rd_|logit|probit|ts_|timeseries_|hypothesis_|anova_|ml_|viz_|multivariate_|kaplan_|cox_|compute_|describe_|load_)[a-z_]+' | head -1 || true)

    if [[ -n "$EXTRACTED_TOOL" ]]; then
        echo "{\"tool\": \"$EXTRACTED_TOOL\", \"content\": \"$CONTENT\", \"extracted\": true}"
    else
        echo "{\"tool\": null, \"content\": \"$CONTENT\"}"
    fi
else
    ARGUMENTS=$(echo "$RESPONSE" | jq '.message.tool_calls[0].function.arguments // {}')
    echo "{\"tool\": \"$TOOL_CALL\", \"arguments\": $ARGUMENTS}"
fi
