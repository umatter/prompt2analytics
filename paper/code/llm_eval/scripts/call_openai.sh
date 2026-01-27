#!/bin/bash
#
# call_openai.sh - OpenAI API wrapper with tool calling
#
# Usage: ./call_openai.sh <model> <prompt> <tools_file>
#
# Returns JSON with selected tool

set -e

MODEL="$1"
PROMPT="$2"
TOOLS_FILE="$3"

if [[ -z "$OPENAI_API_KEY" ]]; then
    echo '{"error": "OPENAI_API_KEY not set"}' >&2
    exit 1
fi

# Load tools from config
TOOLS=$(cat "$TOOLS_FILE" | jq '.tools')

# Determine parameters based on model family
# gpt-5 and o1/o3 models use max_completion_tokens and don't support temperature=0
if [[ "$MODEL" == gpt-5* ]] || [[ "$MODEL" == o1* ]] || [[ "$MODEL" == o3* ]]; then
    TOKEN_PARAM="max_completion_tokens"
    USE_TEMP=false
else
    TOKEN_PARAM="max_tokens"
    USE_TEMP=true
fi

# Build the request
if [[ "$USE_TEMP" == "true" ]]; then
    REQUEST=$(jq -n \
        --arg model "$MODEL" \
        --arg prompt "$PROMPT" \
        --arg token_param "$TOKEN_PARAM" \
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
            temperature: 0
        } + {($token_param): 256}')
else
    # Reasoning models need more tokens for chain-of-thought
    REQUEST=$(jq -n \
        --arg model "$MODEL" \
        --arg prompt "$PROMPT" \
        --arg token_param "$TOKEN_PARAM" \
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
            tool_choice: "required"
        } + {($token_param): 2048}')
fi

# Make the API call
RESPONSE=$(curl -s -X POST "https://api.openai.com/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $OPENAI_API_KEY" \
    -d "$REQUEST")

# Check for errors
ERROR=$(echo "$RESPONSE" | jq -r '.error.message // empty')
if [[ -n "$ERROR" ]]; then
    echo "{\"error\": \"$ERROR\"}" >&2
    exit 1
fi

# Extract the tool call
TOOL_CALL=$(echo "$RESPONSE" | jq -r '.choices[0].message.tool_calls[0].function.name // empty')

if [[ -z "$TOOL_CALL" ]]; then
    # Try to get from content if no tool was called
    CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // empty')
    echo "{\"tool\": null, \"content\": \"$CONTENT\"}"
else
    ARGUMENTS=$(echo "$RESPONSE" | jq '.choices[0].message.tool_calls[0].function.arguments // {}')
    echo "{\"tool\": \"$TOOL_CALL\", \"arguments\": $ARGUMENTS}"
fi
