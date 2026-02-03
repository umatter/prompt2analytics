#!/bin/bash
#
# build_prompt.sh - Construct LLM prompt with dataset context
#
# Usage: ./build_prompt.sh <user_prompt> <dataset_context>
#
# Returns the full prompt string

USER_PROMPT="$1"
DATASET_CONTEXT="$2"

# Build a contextual prompt that helps the LLM understand the task
cat << EOF
I have a dataset loaded with the following structure:
$DATASET_CONTEXT

Task: $USER_PROMPT

Select the most appropriate analysis tool for this task.
EOF
