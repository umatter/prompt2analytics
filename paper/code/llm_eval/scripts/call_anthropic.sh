#!/bin/bash
#
# call_anthropic.sh - Anthropic API wrapper with tool calling
#
# Usage: ./call_anthropic.sh <model> <prompt> <tools_file>
#
# Returns JSON with selected tool

set -e

MODEL="$1"
PROMPT="$2"
TOOLS_FILE="$3"

if [[ -z "$ANTHROPIC_API_KEY" ]]; then
    echo '{"error": "ANTHROPIC_API_KEY not set"}' >&2
    exit 1
fi

# Load tools and convert to Anthropic format
# Anthropic uses a slightly different tool schema
TOOLS=$(cat "$TOOLS_FILE" | jq '[.tools[] | {
    name: .function.name,
    description: .function.description,
    input_schema: .function.parameters
}]')

# Build the request
REQUEST=$(jq -n \
    --arg model "$MODEL" \
    --arg prompt "$PROMPT" \
    --argjson tools "$TOOLS" \
    '{
        model: $model,
        max_tokens: 1024,
        system: "You are a helpful econometrics assistant. When the user describes an analysis task, select the single most appropriate tool from the available tools. Only call one tool.",
        tools: $tools,
        tool_choice: {"type": "any"},
        messages: [
            {
                role: "user",
                content: $prompt
            }
        ]
    }')

# Make the API call
RESPONSE=$(curl -s -X POST "https://api.anthropic.com/v1/messages" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -d "$REQUEST")

# Check for errors
ERROR=$(echo "$RESPONSE" | jq -r '.error.message // empty')
if [[ -n "$ERROR" ]]; then
    echo "{\"error\": \"$ERROR\"}" >&2
    exit 1
fi

# Extract the tool call from Anthropic response
# Anthropic returns tool_use blocks in the content array
TOOL_CALL=$(echo "$RESPONSE" | jq -r '.content[] | select(.type == "tool_use") | .name' | head -1)

if [[ -z "$TOOL_CALL" ]]; then
    # Try to get text content if no tool was called
    CONTENT=$(echo "$RESPONSE" | jq -r '.content[] | select(.type == "text") | .text' | head -1)
    echo "{\"tool\": null, \"content\": \"$CONTENT\"}"
else
    # Use -c for compact output to avoid multiline JSON
    ARGUMENTS=$(echo "$RESPONSE" | jq -c '[.content[] | select(.type == "tool_use") | .input][0]')
    echo "{\"tool\": \"$TOOL_CALL\", \"arguments\": $ARGUMENTS}"
fi
