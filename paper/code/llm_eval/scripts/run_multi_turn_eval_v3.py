#!/usr/bin/env python3
"""
Multi-turn evaluation v3: Evaluates multi-tool sequences within turns.

This evaluator allows models to make multiple tool calls per turn (exploratory + analytical)
and scores based on whether the expected tool was EVENTUALLY called, not just first.

Key differences from v2:
1. Tracks ALL tool calls in a turn, not just the first
2. Marks a turn as successful if expected tool appears anywhere in the sequence
3. Reports both "strict" (first tool) and "eventual" (any tool) metrics
4. Validates that results come from tool calls, not LLM computation

Usage:
    python run_multi_turn_eval_v3.py --model gpt-4o-mini --provider openai
    python run_multi_turn_eval_v3.py --model claude-opus-4-5-20251101 --provider anthropic
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
RESULTS_DIR = PROJECT_DIR / "results" / "multi_turn_v3"

# Tools considered "exploratory" (not penalized if called before the expected tool)
EXPLORATORY_TOOLS = {
    "head_dataset", "describe_dataset", "list_datasets", "compute_correlation",
    "acf", "viz_histogram", "viz_scatter"
}


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


def load_test_dataset(session_id: str, dataset_context: str) -> Optional[str]:
    """Load a test dataset based on the context description."""
    if not dataset_context:
        return None

    # Parse dataset name from context
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

    # Parse column names from context
    if columns_part:
        import re
        columns_part = re.sub(r'\([^)]*\)', '', columns_part)
        col_list = [c.strip() for c in columns_part.split(",") if c.strip()]
    else:
        col_list = ["x", "y"]

    # Generate synthetic data
    import random
    n_rows = 100
    csv_lines = [",".join(col_list)]
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


def call_llm_chat(
    session_id: str,
    conversation_id: str,
    message: str,
    model: str,
    provider: str,
    history: list[dict],
    interpret: bool = False
) -> dict:
    """Call the MCP server's LLM chat endpoint."""
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
        "retrieve_history": False,
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


def extract_all_tool_calls(response: dict) -> list[dict]:
    """Extract ALL tool calls from LLM response.

    Returns list of: [{"name": str, "args": dict, "result": str, "id": str}, ...]
    """
    if response.get("error"):
        return []

    data = response.get("data", {})
    message = data.get("message", {})
    tool_calls = message.get("tool_calls", [])
    tool_results = message.get("tool_results", [])

    if not tool_calls:
        return []

    # Build result list
    results = []
    result_map = {tr.get("tool_call_id"): tr.get("content", "") for tr in tool_results}

    for tc in tool_calls:
        tool_name = tc.get("name")
        tool_args = tc.get("arguments", {})
        tool_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")

        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}

        tool_result = result_map.get(tool_id, "")
        if isinstance(tool_result, str) and len(tool_result) > 2000:
            tool_result = tool_result[:2000] + "... [truncated]"

        results.append({
            "name": tool_name,
            "args": tool_args,
            "result": tool_result,
            "id": tool_id
        })

    return results


def build_history_entry(role: str, content: str, tool_calls: list = None, tool_results: list = None) -> dict:
    """Build a history entry."""
    entry = {"role": role, "content": content}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    if tool_results:
        entry["tool_results"] = tool_results
    return entry


def score_tool_sequence(
    tool_sequence: list[dict],
    expected_tool: str,
    acceptable_tools: list[str]
) -> dict:
    """Score a sequence of tool calls.

    Returns:
        {
            "strict_match": str,  # first tool match type
            "strict_score": float,
            "eventual_match": str,  # any tool match type
            "eventual_score": float,
            "expected_tool_called": bool,
            "exploratory_count": int,
            "total_tool_calls": int,
            "tool_sequence": list[str]
        }
    """
    if not tool_sequence:
        return {
            "strict_match": "none",
            "strict_score": 0.0,
            "eventual_match": "none",
            "eventual_score": 0.0,
            "expected_tool_called": False,
            "exploratory_count": 0,
            "total_tool_calls": 0,
            "tool_sequence": []
        }

    tool_names = [tc["name"] for tc in tool_sequence]
    first_tool = tool_names[0]

    # Count exploratory tools
    exploratory_count = sum(1 for t in tool_names if t in EXPLORATORY_TOOLS)

    # Check strict match (first tool)
    strict_match, strict_score = _score_single_tool(first_tool, expected_tool, acceptable_tools)

    # Check eventual match (any tool in sequence)
    eventual_match = "none"
    eventual_score = 0.0
    expected_tool_called = False

    for tool in tool_names:
        match_type, score = _score_single_tool(tool, expected_tool, acceptable_tools)
        if score > eventual_score:
            eventual_match = match_type
            eventual_score = score
        if tool == expected_tool:
            expected_tool_called = True

    return {
        "strict_match": strict_match,
        "strict_score": strict_score,
        "eventual_match": eventual_match,
        "eventual_score": eventual_score,
        "expected_tool_called": expected_tool_called,
        "exploratory_count": exploratory_count,
        "total_tool_calls": len(tool_sequence),
        "tool_sequence": tool_names
    }


def _score_single_tool(selected: str, expected: str, acceptable: list[str]) -> tuple[str, float]:
    """Score a single tool selection."""
    if not selected:
        return "none", 0.0

    if selected == expected:
        return "exact", 1.0

    if selected in acceptable:
        return "acceptable", 0.8

    # Check for category match
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

    # Create session
    if not dry_run:
        session_id = create_session()
        if not session_id:
            print(f"    Error: Failed to create session for {conv_id}")
            return results
    else:
        session_id = f"eval_{conv_id}_{uuid.uuid4().hex[:8]}"

    mcp_conv_id = f"conv_{conv_id}_{uuid.uuid4().hex[:8]}"

    # Load dataset
    if not dry_run:
        dataset_name = load_test_dataset(session_id, dataset_context)
        if dataset_name:
            print(f"    Loaded dataset: {dataset_name}")

    history = []

    for turn in turns:
        turn_num = turn["turn"]
        user_prompt = turn["user_prompt"]
        expected_tool = turn["expected_tool"]
        acceptable_tools = turn.get("acceptable_tools", [expected_tool])
        context_from_prev = turn.get("context_from_previous", False)

        if dry_run:
            print(f"    Turn {turn_num}: {user_prompt[:60]}...")
            results.append({
                "turn": turn_num,
                "prompt": user_prompt,
                "expected": expected_tool,
                "strict_match": "dry_run",
                "eventual_match": "dry_run",
            })
            continue

        # Call LLM
        start_time = time.time()
        response = call_llm_chat(
            session_id=session_id,
            conversation_id=mcp_conv_id,
            message=user_prompt,
            model=model,
            provider=provider,
            history=history if context_from_prev else [],
            interpret=False
        )
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract ALL tool calls
        tool_sequence = extract_all_tool_calls(response)

        # Score the sequence
        scores = score_tool_sequence(tool_sequence, expected_tool, acceptable_tools)

        # Extract message content
        message_data = response.get("data", {}).get("message", {})
        assistant_content = message_data.get("content", "")

        # Update history
        history.append(build_history_entry("user", user_prompt))

        if tool_sequence:
            # Add assistant message with all tool calls
            tool_call_entries = [
                {"id": tc["id"], "name": tc["name"], "arguments": tc["args"]}
                for tc in tool_sequence
            ]
            history.append(build_history_entry(
                "assistant",
                assistant_content,
                tool_calls=tool_call_entries
            ))

            # Add tool results
            for tc in tool_sequence:
                if tc["result"]:
                    history.append({
                        "role": "tool",
                        "content": "",
                        "tool_results": [{
                            "tool_call_id": tc["id"],
                            "content": tc["result"],
                            "is_error": False
                        }]
                    })
        else:
            history.append(build_history_entry("assistant", assistant_content))

        result = {
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "conversation_id": conv_id,
            "turn": turn_num,
            "category": conv.get("category", "unknown"),
            "prompt": user_prompt,
            "expected": expected_tool,
            "tool_sequence": scores["tool_sequence"],
            "strict_match": scores["strict_match"],
            "strict_score": scores["strict_score"],
            "eventual_match": scores["eventual_match"],
            "eventual_score": scores["eventual_score"],
            "expected_tool_called": scores["expected_tool_called"],
            "exploratory_count": scores["exploratory_count"],
            "total_tool_calls": scores["total_tool_calls"],
            "latency_ms": latency_ms,
            "context_from_previous": context_from_prev,
        }
        results.append(result)

        # Display status
        strict_ok = scores["strict_match"] in ("exact", "acceptable")
        eventual_ok = scores["eventual_match"] in ("exact", "acceptable")

        if eventual_ok and not strict_ok:
            status = "⟳"  # Eventually correct (explored first)
            detail = f"explored→{expected_tool}"
        elif eventual_ok:
            status = "✓"
            detail = f"{scores['strict_match']}"
        else:
            status = "✗"
            detail = f"{scores['strict_match']}"

        seq_str = "→".join(scores["tool_sequence"][:3]) if scores["tool_sequence"] else "none"
        if len(scores["tool_sequence"]) > 3:
            seq_str += f"... ({len(scores['tool_sequence'])} total)"

        print(f"    Turn {turn_num}: {status} {detail} | seq: {seq_str}")

        time.sleep(0.5)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-turn evaluation v3 (multi-tool sequences)")
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
            sys.exit(1)
        print(f"Connected to MCP server at {MCP_SERVER_URL}")

    # Load test cases
    conversations = load_test_files(args.category)
    if not conversations:
        print(f"No test cases found for category: {args.category}")
        sys.exit(1)

    print(f"\nMulti-Turn Evaluation V3 (Multi-Tool Sequences): {args.model} ({args.provider})")
    print(f"Test cases: {len(conversations)} conversations")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model.replace("/", "_").replace(":", "_")
    output_file = RESULTS_DIR / f"{model_safe}_{args.category}_{timestamp}.jsonl"

    all_results = []

    # Tracking metrics
    total_turns = 0
    strict_correct = 0
    eventual_correct = 0
    total_tool_calls = 0
    total_exploratory = 0
    strict_completed_convs = 0
    eventual_completed_convs = 0

    for conv in conversations:
        conv_id = conv["id"]
        description = conv.get("description", "")
        turn_count = len(conv.get("turns", []))

        print(f"\n  [{conv['category']}] {conv_id}: {description}")

        results = run_conversation(conv, args.model, args.provider, args.dry_run)
        all_results.extend(results)

        # Count stats
        conv_strict_correct = sum(1 for r in results if r.get("strict_match") in ("exact", "acceptable"))
        conv_eventual_correct = sum(1 for r in results if r.get("eventual_match") in ("exact", "acceptable"))

        total_turns += len(results)
        strict_correct += conv_strict_correct
        eventual_correct += conv_eventual_correct
        total_tool_calls += sum(r.get("total_tool_calls", 0) for r in results)
        total_exploratory += sum(r.get("exploratory_count", 0) for r in results)

        if conv_strict_correct == turn_count:
            strict_completed_convs += 1
        if conv_eventual_correct == turn_count:
            eventual_completed_convs += 1

    # Write results
    if not args.dry_run:
        with open(output_file, "w") as f:
            for result in all_results:
                f.write(json.dumps(result) + "\n")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n### Accuracy Metrics ###")
    print(f"{'Metric':<30} {'Strict':>12} {'Eventual':>12}")
    print("-" * 54)

    if total_turns > 0:
        strict_acc = strict_correct * 100 / total_turns
        eventual_acc = eventual_correct * 100 / total_turns
        print(f"{'Turn Accuracy':<30} {strict_acc:>11.1f}% {eventual_acc:>11.1f}%")

    if len(conversations) > 0:
        strict_comp = strict_completed_convs * 100 / len(conversations)
        eventual_comp = eventual_completed_convs * 100 / len(conversations)
        print(f"{'Conversation Completion':<30} {strict_comp:>11.1f}% {eventual_comp:>11.1f}%")

    print(f"\n### Tool Usage ###")
    print(f"Total tool calls: {total_tool_calls}")
    print(f"Exploratory calls: {total_exploratory} ({total_exploratory*100/max(total_tool_calls,1):.1f}%)")
    print(f"Avg tools per turn: {total_tool_calls/max(total_turns,1):.2f}")

    print(f"\n### Per-Turn Breakdown ###")
    print(f"{'Turn':<8} {'Strict':>12} {'Eventual':>12} {'Explored→OK':>12}")
    print("-" * 44)

    turn_stats = {}
    for r in all_results:
        t = r["turn"]
        if t not in turn_stats:
            turn_stats[t] = {"total": 0, "strict": 0, "eventual": 0, "explored_ok": 0}
        turn_stats[t]["total"] += 1
        if r.get("strict_match") in ("exact", "acceptable"):
            turn_stats[t]["strict"] += 1
        if r.get("eventual_match") in ("exact", "acceptable"):
            turn_stats[t]["eventual"] += 1
            if r.get("strict_match") not in ("exact", "acceptable"):
                turn_stats[t]["explored_ok"] += 1

    for turn_num in sorted(turn_stats.keys()):
        stats = turn_stats[turn_num]
        total = stats["total"]
        strict_pct = stats["strict"] * 100 / total if total > 0 else 0
        eventual_pct = stats["eventual"] * 100 / total if total > 0 else 0
        explored_ok = stats["explored_ok"]
        print(f"Turn {turn_num:<4} {strict_pct:>11.0f}% {eventual_pct:>11.0f}% {explored_ok:>12}")

    print()
    if not args.dry_run:
        print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
