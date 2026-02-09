#!/usr/bin/env python3
"""
Multi-turn evaluation with actual MCP server execution.

This evaluator:
1. Connects to the MCP server's LLM chat endpoint
2. Actually executes tools and captures results
3. Builds proper conversation history with tool calls and results
4. Tests real context retention across turns

Usage:
    python run_multi_turn_eval_v2.py --model gpt-4o-mini --provider openai
    python run_multi_turn_eval_v2.py --model claude-3-5-haiku-20241022 --provider anthropic
    python run_multi_turn_eval_v2.py --dry-run  # Test without API calls
"""

import argparse
import json
import os
import sys
import time
import requests
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configuration
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8080")
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
TEST_CASES_DIR = PROJECT_DIR / "test_cases" / "multi_turn"
RESULTS_DIR = PROJECT_DIR / "results" / "multi_turn_v2"


def load_test_files(category: str) -> list[dict]:
    """Load test conversations from JSON files."""
    conversations = []

    if category == "all":
        files = list(TEST_CASES_DIR.glob("*.json"))
    else:
        files = [TEST_CASES_DIR / f"{category}_conversations.json"]

    for file_path in files:
        if not file_path.exists():
            print(f"Warning: Test file not found: {file_path}")
            continue

        with open(file_path) as f:
            data = json.load(f)
            cat_name = file_path.stem.replace("_conversations", "")
            for conv in data.get("conversations", []):
                conv["category"] = cat_name
                conversations.append(conv)

    return conversations


def check_mcp_server() -> bool:
    """Check if MCP server is running."""
    try:
        resp = requests.get(f"{MCP_SERVER_URL}/health", timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def create_session() -> Optional[str]:
    """Create a new session on the MCP server."""
    try:
        resp = requests.post(
            f"{MCP_SERVER_URL}/api/sessions",
            headers={"Content-Type": "application/json"},
            json={},
            timeout=10
        )
        if resp.status_code in (200, 201):
            data = resp.json()
            return data.get("data", {}).get("session_id")
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Failed to create session: {e}")
    return None


def call_tool(session_id: str, tool_name: str, arguments: dict) -> Optional[dict]:
    """Call a tool on the MCP server."""
    try:
        resp = requests.post(
            f"{MCP_SERVER_URL}/api/tools/{tool_name}",
            headers={"Content-Type": "application/json"},
            json={
                "session_id": session_id,
                "arguments": arguments
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Failed to call tool {tool_name}: {e}")
    return None


def load_test_dataset(session_id: str, dataset_context: str) -> Optional[str]:
    """Load a test dataset based on the context description."""
    if not dataset_context:
        return None

    # Parse dataset name from context like "grunfeld dataset: firm, year, inv, value, capital"
    # or "gdp_quarterly: gdp, date (quarterly from 1960-2023)"
    if " dataset:" in dataset_context:
        dataset_name = dataset_context.split(" dataset:")[0].strip().lower()
        columns_part = dataset_context.split(" dataset:")[1] if " dataset:" in dataset_context else ""
    elif ":" in dataset_context:
        dataset_name = dataset_context.split(":")[0].strip().lower()
        columns_part = dataset_context.split(":")[1] if ":" in dataset_context else ""
    else:
        dataset_name = dataset_context.strip().lower()
        columns_part = ""

    # Map to actual test data files
    dataset_mappings = {
        "grunfeld": "grunfeld.csv",
        "wages": "wages.csv",
        "growth": "growth.csv",
        "cps": "cps.csv",
        "sales": "sales.csv",
    }

    # Try to load from test_data directory
    test_data_dir = PROJECT_DIR / "test_data"
    if dataset_name in dataset_mappings:
        data_file = test_data_dir / dataset_mappings[dataset_name]
        if data_file.exists():
            result = call_tool(session_id, "load_dataset", {
                "path": str(data_file),
                "name": dataset_name
            })
            if result and result.get("success"):
                return dataset_name

    # Parse column names from context (remove parenthetical notes)
    if columns_part:
        # Remove parenthetical content like "(quarterly from 1960-2023)"
        import re
        columns_part = re.sub(r'\([^)]*\)', '', columns_part)
        col_list = [c.strip() for c in columns_part.split(",") if c.strip()]
    else:
        col_list = ["x", "y"]

    # Generate synthetic data using create_dataset with CSV content
    # Build a simple CSV with random-ish data
    import random
    n_rows = 100
    csv_lines = [",".join(col_list)]  # header
    for i in range(n_rows):
        row = []
        for col in col_list:
            if col in ["date", "year", "time", "quarter"]:
                row.append(str(2000 + i % 24))
            elif col in ["firm", "group", "id", "unit"]:
                row.append(str(i % 10))
            else:
                row.append(str(round(random.uniform(0, 100), 2)))
        csv_lines.append(",".join(row))
    csv_content = "\n".join(csv_lines)

    result = call_tool(session_id, "create_dataset", {
        "name": dataset_name,
        "csv_content": csv_content
    })
    if result and result.get("success"):
        return dataset_name

    return None


def call_llm_chat(
    session_id: str,
    conversation_id: str,
    message: str,
    model: str,
    provider: str,
    history: list[dict],
    interpret: bool = False
) -> dict:
    """Call the MCP server's LLM chat endpoint.

    Args:
        interpret: If False, returns raw tool_calls for visibility.
                   If True, LLM interprets results for natural conversation.
    """
    # Map provider to ProviderConfig format expected by the API
    # provider_type must be lowercase: "openai", "anthropic", "ollama"
    provider_type_map = {
        "openai": "openai",
        "anthropic": "anthropic",
        "openrouter": "openai",  # OpenRouter uses OpenAI-compatible API
        "ollama": "ollama",
    }

    provider_type = provider_type_map.get(provider, "openai")

    # Build provider config
    provider_config = {
        "provider_type": provider_type,
        "model": model,
    }

    # Handle OpenRouter: set base_url and API key
    if provider == "openrouter":
        provider_config["base_url"] = "https://openrouter.ai/api/v1"
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if api_key:
            provider_config["api_key"] = api_key
        else:
            print("  Warning: OPENROUTER_API_KEY not set")

    payload = {
        "session_id": session_id,
        "conversation_id": conversation_id,
        "message": message,
        "provider": provider_config,
        "history": history,
        "retrieve_history": False,  # We manage history ourselves for testing
        "interpret": interpret,
    }

    try:
        resp = requests.post(
            f"{MCP_SERVER_URL}/api/llm/chat",
            json=payload,
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "tool_calls": [], "response": ""}


def extract_tool_selection(response: dict) -> tuple[Optional[str], Optional[dict], Optional[str], Optional[str]]:
    """Extract tool name, arguments, result, and ID from LLM response.

    Returns: (tool_name, tool_args, tool_result, tool_call_id)
    """
    # Handle error responses
    if response.get("error"):
        return None, None, None, None

    # Navigate the response structure: {success, data: {message: {tool_calls, tool_results}}}
    data = response.get("data", {})
    message = data.get("message", {})
    tool_calls = message.get("tool_calls", [])
    tool_results = message.get("tool_results", [])

    if not tool_calls:
        return None, None, None, None

    # Get the first tool call
    first_call = tool_calls[0]
    tool_name = first_call.get("name")
    tool_args = first_call.get("arguments", {})
    tool_call_id = first_call.get("id", f"call_{uuid.uuid4().hex[:8]}")

    # Parse arguments if string
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            tool_args = {}

    # Get the corresponding tool result if available
    tool_result = ""
    if tool_results:
        first_result = tool_results[0]
        tool_result = first_result.get("content", "")

    # Truncate result for storage
    if isinstance(tool_result, str) and len(tool_result) > 2000:
        tool_result = tool_result[:2000] + "... [truncated]"
    elif isinstance(tool_result, dict):
        tool_result = json.dumps(tool_result)[:2000]

    return tool_name, tool_args, tool_result, tool_call_id


def build_history_entry(role: str, content: str, tool_calls: list = None, tool_results: list = None) -> dict:
    """Build a history entry in OpenAI-compatible format."""
    entry = {"role": role, "content": content}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    if tool_results:
        entry["tool_results"] = tool_results
    return entry


def score_tool_selection(selected: str, expected: str, acceptable: list[str]) -> tuple[str, float]:
    """Score the tool selection."""
    if not selected:
        return "none", 0.0

    if selected == expected:
        return "exact", 1.0

    if selected in acceptable:
        return "acceptable", 0.8

    # Check for category match (e.g., regression_ols matches regression_*)
    expected_prefix = expected.split("_")[0] if "_" in expected else expected
    selected_prefix = selected.split("_")[0] if "_" in selected else selected

    if expected_prefix == selected_prefix:
        return "category", 0.5

    return "wrong", 0.0


def run_conversation(
    conv: dict,
    model: str,
    provider: str,
    dry_run: bool = False
) -> list[dict]:
    """Run a single multi-turn conversation and return results."""
    results = []
    conv_id = conv["id"]
    dataset_context = conv.get("dataset_context", "")
    turns = conv.get("turns", [])

    # Create a real session on the MCP server
    if not dry_run:
        session_id = create_session()
        if not session_id:
            print(f"    Error: Failed to create session for {conv_id}")
            return results
    else:
        session_id = f"eval_{conv_id}_{uuid.uuid4().hex[:8]}"

    mcp_conv_id = f"conv_{conv_id}_{uuid.uuid4().hex[:8]}"

    # Load or generate test dataset
    if not dry_run:
        dataset_name = load_test_dataset(session_id, dataset_context)
        if dataset_name:
            print(f"    Loaded dataset: {dataset_name}")

    # Track conversation history in proper format
    history = []

    for turn in turns:
        turn_num = turn["turn"]
        user_prompt = turn["user_prompt"]
        expected_tool = turn["expected_tool"]
        acceptable_tools = turn.get("acceptable_tools", [expected_tool])
        context_from_prev = turn.get("context_from_previous", False)

        if dry_run:
            print(f"    Turn {turn_num}: {user_prompt[:60]}...")
            print(f"      Expected: {expected_tool}")
            results.append({
                "turn": turn_num,
                "prompt": user_prompt,
                "expected": expected_tool,
                "selected": None,
                "match_type": "dry_run",
                "score": 0.0,
            })
            continue

        # Call LLM with full history
        # Use interpret=False for all turns to capture tool selection
        # But include natural language context in history for better multi-turn
        start_time = time.time()
        response = call_llm_chat(
            session_id=session_id,
            conversation_id=mcp_conv_id,
            message=user_prompt,
            model=model,
            provider=provider,
            history=history if context_from_prev else [],
            interpret=False  # Need to see tool_calls for scoring
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract tool selection
        selected_tool, tool_args, tool_result, tool_call_id = extract_tool_selection(response)

        # Score
        match_type, score = score_tool_selection(selected_tool, expected_tool, acceptable_tools)

        # Extract message content from response structure
        message_data = response.get("data", {}).get("message", {})
        assistant_content = message_data.get("content", "")

        # Update history with proper format (user message + assistant with tool calls)
        history.append(build_history_entry("user", user_prompt))

        if selected_tool:
            # Add assistant message with tool call (use actual ID from response)
            tool_call_entry = {
                "id": tool_call_id,
                "name": selected_tool,
                "arguments": tool_args or {}
            }
            history.append(build_history_entry(
                "assistant",
                assistant_content,
                tool_calls=[tool_call_entry]
            ))

            # Add tool result as separate message
            # MCP server expects tool_results as a list inside the message
            # with is_error field (required by ToolResult struct)
            if tool_result:
                history.append({
                    "role": "tool",
                    "content": "",  # Content is inside tool_results
                    "tool_results": [{
                        "tool_call_id": tool_call_id,
                        "content": tool_result,
                        "is_error": False
                    }]
                })
        else:
            # Just add assistant response
            history.append(build_history_entry("assistant", assistant_content))

        result = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "conversation_id": conv_id,
            "turn": turn_num,
            "category": conv.get("category", "unknown"),
            "prompt": user_prompt,
            "selected": selected_tool,
            "selected_args": tool_args,
            "expected": expected_tool,
            "match_type": match_type,
            "score": score,
            "latency_ms": latency_ms,
            "context_from_previous": context_from_prev,
            "history_length": len(history),
            "tool_result_included": bool(tool_result),
        }
        results.append(result)

        status = "✓" if match_type in ("exact", "acceptable") else "✗"
        print(f"    Turn {turn_num}: {status} {match_type} ({selected_tool} vs {expected_tool})")

        # Rate limiting
        time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-turn evaluation with MCP server")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model to evaluate")
    parser.add_argument("--provider", default="openai",
                       choices=["openai", "anthropic", "openrouter", "ollama"])
    parser.add_argument("--category", default="all", help="Test category or 'all'")
    parser.add_argument("--dry-run", action="store_true", help="Print test cases without running")
    parser.add_argument("--server-url", default=None, help="MCP server URL")
    args = parser.parse_args()

    global MCP_SERVER_URL
    if args.server_url:
        MCP_SERVER_URL = args.server_url

    # Check server
    if not args.dry_run:
        if not check_mcp_server():
            print(f"Error: MCP server not running at {MCP_SERVER_URL}")
            print("Start it with: cargo run -p p2a-mcp --features full -- --transport http --port 8080")
            sys.exit(1)
        print(f"Connected to MCP server at {MCP_SERVER_URL}")

    # Load test cases
    conversations = load_test_files(args.category)
    if not conversations:
        print(f"No test cases found for category: {args.category}")
        sys.exit(1)

    print(f"\nMulti-Turn Evaluation V2: {args.model} ({args.provider})")
    print(f"Test cases: {len(conversations)} conversations")
    print("=" * 60)

    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_").replace(":", "_")
    output_file = RESULTS_DIR / f"{model_safe}_{args.category}_{timestamp}.jsonl"

    all_results = []
    total_turns = 0
    correct_turns = 0
    completed_conversations = 0

    for conv in conversations:
        conv_id = conv["id"]
        description = conv.get("description", "")
        turn_count = len(conv.get("turns", []))

        print(f"\n  [{conv['category']}] {conv_id}: {description}")

        results = run_conversation(conv, args.model, args.provider, args.dry_run)
        all_results.extend(results)

        # Count stats
        conv_correct = sum(1 for r in results if r["match_type"] in ("exact", "acceptable"))
        total_turns += len(results)
        correct_turns += conv_correct

        if conv_correct == turn_count:
            completed_conversations += 1

    # Write results
    if not args.dry_run:
        with open(output_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total Turns: {total_turns}")
    print(f"Correct Turns: {correct_turns}")
    if total_turns > 0:
        print(f"Turn Accuracy: {correct_turns * 100 / total_turns:.1f}%")
    print()
    print(f"Total Conversations: {len(conversations)}")
    print(f"Fully Completed: {completed_conversations}")
    if len(conversations) > 0:
        print(f"Conversation Completion: {completed_conversations * 100 / len(conversations):.1f}%")
    print()

    if not args.dry_run:
        print(f"Results saved to: {output_file}")

    # Detailed breakdown by turn number
    print("\nAccuracy by Turn:")
    turn_stats = {}
    for r in all_results:
        t = r["turn"]
        if t not in turn_stats:
            turn_stats[t] = {"total": 0, "correct": 0}
        turn_stats[t]["total"] += 1
        if r["match_type"] in ("exact", "acceptable"):
            turn_stats[t]["correct"] += 1

    for turn_num in sorted(turn_stats.keys()):
        stats = turn_stats[turn_num]
        acc = stats["correct"] * 100 / stats["total"] if stats["total"] > 0 else 0
        print(f"  Turn {turn_num}: {stats['correct']}/{stats['total']} ({acc:.0f}%)")


if __name__ == "__main__":
    main()
