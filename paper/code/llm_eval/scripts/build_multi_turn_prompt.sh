#!/bin/bash
# Build prompt with conversation history for multi-turn evaluation
# Usage: ./scripts/build_multi_turn_prompt.sh <user_prompt> <dataset_context> <conversation_history_json> <use_context>

USER_PROMPT="$1"
DATASET_CONTEXT="$2"
CONV_HISTORY="$3"
USE_CONTEXT="$4"

# Build the system context
SYSTEM_CONTEXT="I have a dataset loaded with the following structure:
$DATASET_CONTEXT

"

# Add conversation history if applicable
if [[ "$USE_CONTEXT" == "true" && "$CONV_HISTORY" != "[]" ]]; then
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
            TOOL=$(echo "$CONV_HISTORY" | jq -r ".[$i].tool_selected")
            HISTORY_TEXT+="Assistant: [Selected tool: $TOOL]
"
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

Select the most appropriate analysis tool for this task."

echo "$FULL_PROMPT"
