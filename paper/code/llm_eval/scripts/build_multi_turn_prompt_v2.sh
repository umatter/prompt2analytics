#!/bin/bash
# Build prompt with conversation history for multi-turn evaluation (V2)
# This version includes tool arguments and results in the history
#
# Usage: ./scripts/build_multi_turn_prompt_v2.sh <user_prompt> <dataset_context> <conversation_history_json> <use_context>
#
# History format (JSON array):
# [
#   {"role": "user", "content": "..."},
#   {"role": "assistant", "content": "...", "tool_calls": [{"name": "...", "arguments": {...}}]},
#   {"role": "tool", "tool_call_id": "...", "content": "..."}
# ]

USER_PROMPT="$1"
DATASET_CONTEXT="$2"
CONV_HISTORY="$3"
USE_CONTEXT="$4"

# Build the system context with enhanced dataset info
SYSTEM_CONTEXT="You are an analytics assistant with access to statistical tools.

Dataset loaded: $DATASET_CONTEXT

"

# Add conversation history if applicable
if [[ "$USE_CONTEXT" == "true" && "$CONV_HISTORY" != "[]" && -n "$CONV_HISTORY" ]]; then
    HISTORY_TEXT=""

    # Parse conversation history
    HISTORY_LEN=$(echo "$CONV_HISTORY" | jq 'length')

    for ((i=0; i<HISTORY_LEN; i++)); do
        ROLE=$(echo "$CONV_HISTORY" | jq -r ".[$i].role")

        if [[ "$ROLE" == "user" ]]; then
            CONTENT=$(echo "$CONV_HISTORY" | jq -r ".[$i].content")
            HISTORY_TEXT+="User: $CONTENT
"
        elif [[ "$ROLE" == "assistant" ]]; then
            CONTENT=$(echo "$CONV_HISTORY" | jq -r ".[$i].content // \"\"")
            TOOL_CALLS=$(echo "$CONV_HISTORY" | jq -c ".[$i].tool_calls // []")

            if [[ "$TOOL_CALLS" != "[]" ]]; then
                # Extract first tool call
                TOOL_NAME=$(echo "$TOOL_CALLS" | jq -r ".[0].name // \"unknown\"")
                TOOL_ARGS=$(echo "$TOOL_CALLS" | jq -c ".[0].arguments // {}")

                # Format arguments nicely (key=value pairs)
                ARGS_FORMATTED=$(echo "$TOOL_ARGS" | jq -r 'to_entries | map("\(.key)=\(.value)") | join(", ")' 2>/dev/null || echo "$TOOL_ARGS")

                HISTORY_TEXT+="Assistant: Called $TOOL_NAME($ARGS_FORMATTED)
"
            elif [[ -n "$CONTENT" ]]; then
                HISTORY_TEXT+="Assistant: $CONTENT
"
            fi
        elif [[ "$ROLE" == "tool" ]]; then
            TOOL_RESULT=$(echo "$CONV_HISTORY" | jq -r ".[$i].content // \"\"")

            # Truncate long results
            if [[ ${#TOOL_RESULT} -gt 500 ]]; then
                TOOL_RESULT="${TOOL_RESULT:0:500}... [truncated]"
            fi

            if [[ -n "$TOOL_RESULT" ]]; then
                HISTORY_TEXT+="Tool result: $TOOL_RESULT
"
            fi
        fi
    done

    if [[ -n "$HISTORY_TEXT" ]]; then
        SYSTEM_CONTEXT+="Previous conversation:
$HISTORY_TEXT
"
    fi
fi

# Add current task
FULL_PROMPT="${SYSTEM_CONTEXT}Current task: $USER_PROMPT

Select the most appropriate analysis tool for this task. Consider the conversation history when choosing tools and parameters."

echo "$FULL_PROMPT"
