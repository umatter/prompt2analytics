#!/usr/bin/env python3
"""
End-to-end evaluation script for prompt2analytics.
Drives the p2a-mcp backend API directly with LLM chat/stream endpoint.
Records tool selection, parameters, numerical results, and interpretation.
"""

import json
import os
import sys
import time
import re
import requests
import argparse
from datetime import datetime
from pathlib import Path

BASE_URL = "http://localhost:8081"
DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_DIR = Path(__file__).parent / "results"
TEST_CASES_FILE = Path(__file__).parent / "test_cases.json"

# Model configurations
MODELS = {
    "claude-opus-4.6": {
        "provider_type": "anthropic",
        "model": "claude-opus-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "claude-sonnet-4.6": {
        "provider_type": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "claude-sonnet-4": {
        "provider_type": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "claude-haiku-4.5": {
        "provider_type": "anthropic",
        "model": "claude-haiku-4-5-20251001",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "gpt-4.1-mini": {
        "provider_type": "openai",
        "model": "openai/gpt-4.1-mini",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "gemini-2.5-flash": {
        "provider_type": "openai",
        "model": "google/gemini-2.5-flash",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "llama-4-scout": {
        "provider_type": "openai",
        "model": "meta-llama/llama-4-scout",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    "ministral-3b": {
        "provider_type": "openai",
        "model": "mistralai/ministral-3b-2512",
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
}

# Dataset -> file mapping
DATASET_FILES = {
    "eval_cross_section": "eval_cross_section.csv",
    "eval_panel": "eval_panel.csv",
    "eval_timeseries": "eval_timeseries.csv",
    "eval_treatment": "eval_treatment.csv",
    "eval_messy": "eval_messy.csv",
    "eval_survey": "eval_survey.csv",
}


def create_session():
    """Create a new backend session."""
    r = requests.post(f"{BASE_URL}/api/sessions", json={})
    r.raise_for_status()
    return r.json()["data"]["session_id"]


def load_dataset(session_id, name, path):
    """Load a dataset into the session."""
    r = requests.post(
        f"{BASE_URL}/api/tools/load_dataset",
        json={
            "session_id": session_id,
            "arguments": {"path": str(path), "name": name},
        },
    )
    r.raise_for_status()
    data = r.json()
    if data.get("success"):
        print(f"  Loaded {name}: {data['data']['content'][0]['text'][:80]}...")
    else:
        print(f"  FAILED to load {name}: {data}")
    return data.get("success", False)


def get_provider_config(model_key):
    """Build provider config for a model."""
    cfg = MODELS[model_key]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        print(f"  WARNING: {cfg['api_key_env']} not set, skipping {model_key}")
        return None
    config = {
        "provider_type": cfg["provider_type"],
        "model": cfg["model"],
        "api_key": api_key,
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }
    if "base_url" in cfg:
        config["base_url"] = cfg["base_url"]
    return config


def send_chat(session_id, message, provider_config, history=None):
    """Send a chat message and collect the full SSE response."""
    payload = {
        "session_id": session_id,
        "message": message,
        "provider": provider_config,
        "history": history or [],
        "interpret": True,
        "retrieve_history": False,
    }

    result = {
        "tool_calls": [],
        "content_parts": [],
        "full_content": "",
        "error": None,
        "raw_events": [],
    }

    max_retries = 5
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{BASE_URL}/api/llm/chat/stream",
                json=payload,
                stream=True,
                timeout=180,
            )
            if r.status_code == 429 or r.status_code == 503:
                wait = min(2 ** attempt * 3, 60)
                print(f"        Rate limited ({r.status_code}), waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            r.raise_for_status()
            break
        except requests.exceptions.HTTPError as e:
            if "429" in str(e) or "503" in str(e):
                wait = min(2 ** attempt * 3, 60)
                print(f"        Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
                continue
            result["error"] = str(e)
            return result
    else:
        result["error"] = "Rate limited after max retries"
        return result

    try:
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]  # strip "data: "
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            result["raw_events"].append(event)
            evt_type = event.get("type")

            if evt_type == "tool_start":
                result["tool_calls"].append({
                    "tool": event.get("tool"),
                    "arguments": event.get("arguments", {}),
                    "result": None,
                    "elapsed_ms": None,
                })
            elif evt_type == "tool_end":
                # Match to last tool_start
                for tc in reversed(result["tool_calls"]):
                    if tc["tool"] == event.get("tool") and tc["result"] is None:
                        tc["result"] = event.get("result", "")
                        tc["elapsed_ms"] = event.get("elapsed_ms")
                        break
            elif evt_type == "content":
                result["content_parts"].append(event.get("text", ""))
            elif evt_type == "done":
                msg = event.get("message", {})
                result["full_content"] = msg.get("content", "")
            elif evt_type == "error":
                err_msg = event.get("error", "Unknown error")
                # Check if rate limited - trigger retry
                if "429" in err_msg or "rate" in err_msg.lower():
                    result["error"] = err_msg
                    # Return with error, let caller decide to retry
                    return result
                result["error"] = err_msg

    except requests.exceptions.Timeout:
        result["error"] = "Request timed out after 120s"
    except Exception as e:
        result["error"] = str(e)

    return result


def score_tool_selection(expected_tool, actual_tools):
    """Score tool selection: 2=exact, 1=acceptable, 0=wrong."""
    if not actual_tools:
        return 0
    raw_tool = actual_tools[0]["tool"]
    # Clean tool name (may include args for some models)
    primary_tool = raw_tool.split(",")[0].split("{")[0].strip() if raw_tool else ""

    # Exact match
    if primary_tool == expected_tool:
        return 2

    # Acceptable alternatives (map of expected -> acceptable alternatives)
    acceptable = {
        "regression_ols": ["regression_clustered", "regression_diagnostics"],
        "regression_diagnostics": ["regression_ols"],
        "compute_correlation": ["viz_heatmap", "cor_test"],
        "viz_heatmap": ["compute_correlation", "cor_test"],
        "describe_dataset": ["head_dataset", "data_quality_profile"],
        "str_replace_value": ["munge_filter", "munge_mutate", "munge_replace"],
        "munge_replace": ["str_replace_value", "munge_filter", "munge_mutate"],
        "munge_rename": ["munge_mutate"],
        "munge_filter": ["str_replace_value", "munge_replace", "munge_mutate", "munge_drop_na"],
        "t_test": ["regression_ols", "wilcoxon_test", "hypothesis_t_test"],
        "anova": ["regression_ols", "t_test", "anova_one_way"],
        "anova_one_way": ["anova", "regression_ols"],
        "shapiro_wilk": ["regression_diagnostics", "hypothesis_shapiro_wilk"],
        "hypothesis_shapiro_wilk": ["shapiro_wilk", "regression_diagnostics"],
        "chi_squared_test": ["cor_test", "hypothesis_chisq_independence"],
        "hypothesis_chisq_independence": ["chi_squared_test", "cor_test"],
        "regression_clustered": ["regression_ols", "regression_hac", "anova_one_way"],
        "iv_2sls": ["regression_ols", "iv_first_stage"],
        "negbin": ["feglm", "poisson", "zeroinfl", "logit"],
        "ordered_model": ["logit", "probit", "regression_ols"],
        "ts_var": ["ts_arima_fit", "regression_ols"],
        "timeseries_pp_test": ["ts_arima_fit"],
        "ts_arima_fit": ["ts_var", "timeseries_decompose", "timeseries_acf", "ts_arima_forecast"],
        "hypothesis_t_test": ["munge_filter", "regression_ols", "wilcoxon_test"],
        "logit": ["probit", "regression_ols"],
        "ml_kmeans": ["ml_hierarchical", "ml_dbscan"],
        "ml_pca": ["compute_correlation", "describe_dataset"],
        "diff_in_diff": ["regression_ols", "treatment_ipw"],
        "treatment_ipw": ["treatment_doubly_robust", "treatment_weightit", "propensity_matching", "treatment_entropy_balance", "treatment_cbps"],
        "propensity_matching": ["treatment_ipw", "treatment_cbps"],
        "rd_estimate": ["regression_ols"],
        "panel_fixed_effects": ["panel_hdfe", "regression_ols"],
        "panel_random_effects": ["panel_fixed_effects"],
        "hausman_test": ["panel_fixed_effects"],
    }

    alts = acceptable.get(expected_tool, [])
    if primary_tool in alts:
        return 1

    # Multi-step: if expected tool contains "+", check if all tools present
    if "+" in (expected_tool or ""):
        expected_parts = [t.strip() for t in expected_tool.split("+")]
        actual_names = [tc["tool"] for tc in actual_tools]
        if all(any(ep in an for an in actual_names) for ep in expected_parts):
            return 2

    return 0


def score_parameters(expected_params, actual_tools):
    """Score parameter extraction: 2=all correct, 1=core correct, 0=wrong."""
    if not actual_tools or not expected_params:
        return 0 if expected_params else 2

    actual_args = actual_tools[0].get("arguments", {})

    # Check core parameters (y_col/y, x_cols/x, dataset)
    core_correct = True
    optional_correct = True

    for key, expected_val in expected_params.items():
        actual_val = actual_args.get(key)
        # Try alternate key names
        alt_keys = {
            "y_col": ["y"],
            "x_cols": ["x"],
            "entity_col": ["entity_var"],
            "time_col": ["time_var"],
            "cluster_var": ["cluster1"],
            "column": ["columns"],
            "columns": ["column"],
        }
        if actual_val is None and key in alt_keys:
            for alt in alt_keys[key]:
                actual_val = actual_args.get(alt)
                if actual_val is not None:
                    break

        if actual_val is None:
            # Core params: y, x, dataset, treatment, outcome
            if key in ("y_col", "y", "x_cols", "x", "endogenous", "instruments",
                       "treatment_col", "outcome_col", "entity_col", "column"):
                core_correct = False
            else:
                optional_correct = False
            continue

        # Compare values
        if isinstance(expected_val, list):
            if isinstance(actual_val, list):
                if set(str(v) for v in expected_val) != set(str(v) for v in actual_val):
                    if key in ("x_cols", "x", "endogenous", "instruments", "covariates"):
                        core_correct = False
                    else:
                        optional_correct = False
            else:
                optional_correct = False
        elif str(expected_val).lower() != str(actual_val).lower():
            if key in ("y_col", "y", "treatment_col", "outcome_col", "entity_col"):
                core_correct = False
            else:
                optional_correct = False

    if core_correct and optional_correct:
        return 2
    elif core_correct:
        return 1
    return 0


def score_numerical(expected_numerical, tool_result_text):
    """Score numerical correctness: 2=within tolerance, 1=reasonable, 0=wrong."""
    if not expected_numerical or not tool_result_text:
        return None  # N/A

    score = 2
    for key, expected in expected_numerical.items():
        if isinstance(expected, str) and expected.startswith("≈"):
            target = float(expected[1:])
            # Try to find the value in the result text
            patterns = [
                rf"{key}[:\s]+([+-]?\d+\.?\d*)",
                rf"([+-]?\d+\.?\d*)\s*.*{key}",
            ]
            found = False
            for pat in patterns:
                m = re.search(pat, tool_result_text, re.IGNORECASE)
                if m:
                    actual = float(m.group(1))
                    rel_error = abs(actual - target) / max(abs(target), 1e-10)
                    if rel_error > 0.5:  # More than 50% off
                        score = min(score, 0)
                    elif rel_error > 0.1:  # More than 10% off
                        score = min(score, 1)
                    found = True
                    break
            if not found:
                score = min(score, 1)  # Can't verify = reasonable

    return score


def score_interpretation(full_content, tool_calls):
    """Score interpretation: 2=accurate, 1=superficial, 0=misleading."""
    if not full_content:
        return 0

    # Basic heuristics - check for common quality indicators
    content_lower = full_content.lower()

    # Check it references actual results
    has_numbers = bool(re.search(r'\d+\.?\d*', full_content))
    has_context = any(w in content_lower for w in [
        "significant", "coefficient", "p-value", "statistic",
        "result", "model", "regression", "test", "effect",
    ])
    has_interpretation = len(full_content) > 100

    if has_numbers and has_context and has_interpretation:
        return 2
    elif has_context or has_numbers:
        return 1
    return 0


def run_single_test(session_id, test_case, clarity, provider_config):
    """Run a single test and return scored result."""
    prompt = test_case["prompts"][clarity]
    expected_tool = test_case["expected_tool"]
    expected_params = test_case.get("expected_params", {})
    expected_numerical = test_case.get("expected_numerical", {})

    print(f"    [{test_case['id']}/{clarity}] {prompt[:60]}...")

    # Retry logic for rate limits
    response = None
    for retry in range(4):
        response = send_chat(session_id, prompt, provider_config)
        if response["error"] and ("429" in str(response["error"]) or "rate" in str(response["error"]).lower()):
            wait = min(2 ** retry * 5, 60)
            print(f"        Rate limited, retrying in {wait}s...")
            time.sleep(wait)
            response = {"tool_calls": [], "content_parts": [], "full_content": "", "error": None, "raw_events": []}
            continue
        break

    if response["error"]:
        print(f"      ERROR: {response['error'][:80]}")
        return {
            "test_id": test_case["id"],
            "clarity": clarity,
            "prompt": prompt,
            "error": response["error"],
            "tool_selected": None,
            "tool_expected": expected_tool,
            "parameters_extracted": {},
            "parameters_expected": expected_params,
            "scores": {
                "tool_selection": 0,
                "parameter_extraction": 0,
                "numerical_correctness": None,
                "interpretation": 0,
            },
            "total_score": 0,
            "interpretation_summary": "",
        }

    # Extract tool info (clean tool name - may include args for some models)
    raw_tool = response["tool_calls"][0]["tool"] if response["tool_calls"] else None
    tool_selected = raw_tool.split(",")[0].split("{")[0].strip() if raw_tool else None
    params_extracted = response["tool_calls"][0]["arguments"] if response["tool_calls"] else {}
    tool_result = response["tool_calls"][0].get("result", "") if response["tool_calls"] else ""

    # Score
    s_tool = score_tool_selection(expected_tool, response["tool_calls"])
    s_params = score_parameters(expected_params, response["tool_calls"])
    s_numerical = score_numerical(expected_numerical, tool_result)
    s_interp = score_interpretation(response["full_content"], response["tool_calls"])

    total = s_tool + s_params + (s_numerical if s_numerical is not None else 0) + s_interp
    max_possible = 6 + (2 if s_numerical is not None else 0)

    print(f"      Tool: {tool_selected} (expected: {expected_tool}) -> {s_tool}/2")
    print(f"      Params: {s_params}/2, Numerical: {s_numerical}, Interp: {s_interp}/2")

    return {
        "test_id": test_case["id"],
        "clarity": clarity,
        "prompt": prompt,
        "tool_selected": tool_selected,
        "tool_expected": expected_tool,
        "parameters_extracted": params_extracted,
        "parameters_expected": expected_params,
        "numerical_output": tool_result[:500] if tool_result else "",
        "interpretation_summary": response["full_content"][:500] if response["full_content"] else "",
        "scores": {
            "tool_selection": s_tool,
            "parameter_extraction": s_params,
            "numerical_correctness": s_numerical,
            "interpretation": s_interp,
        },
        "total_score": total,
        "max_score": max_possible,
    }


def run_multi_turn(session_id, mt_case, provider_config):
    """Run a multi-turn conversation and return results."""
    mt_id = mt_case["id"]
    turns = mt_case["turns"]
    history = []
    results = []

    print(f"    [MT: {mt_id}] {len(turns)} turns")

    for i, turn in enumerate(turns):
        prompt = turn["prompt"]
        print(f"      Turn {i+1}: {prompt[:60]}...")

        response = send_chat(session_id, prompt, provider_config, history)

        tool_selected = response["tool_calls"][0]["tool"] if response["tool_calls"] else None
        s_tool = 2 if tool_selected else 0  # Simplified scoring for MT

        turn_result = {
            "turn": i + 1,
            "prompt": prompt,
            "tool_selected": tool_selected,
            "error": response["error"],
            "scores": {"tool_selection": s_tool},
        }
        results.append(turn_result)

        # Update history for next turn
        history.append({"role": "user", "content": prompt})
        if response["full_content"]:
            history.append({"role": "assistant", "content": response["full_content"][:2000]})

        if response["error"]:
            print(f"        ERROR: {response['error'][:60]}")
            break

        # Brief pause to avoid rate limiting
        time.sleep(1)

    completed = len(results) == len(turns) and all(r["error"] is None for r in results)
    return {
        "conversation_id": mt_id,
        "dataset": mt_case["dataset"],
        "turns": results,
        "completed": completed,
    }


def run_evaluation_for_model(model_key, test_cases, clarities=None):
    """Run the full evaluation for one model."""
    if clarities is None:
        clarities = ["precise", "moderate", "vague"]

    provider_config = get_provider_config(model_key)
    if provider_config is None:
        return None

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_key} ({MODELS[model_key]['model']})")
    print(f"{'='*60}")

    all_results = {
        "model": model_key,
        "model_id": MODELS[model_key]["model"],
        "provider": MODELS[model_key]["provider_type"],
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "multi_turn": [],
    }

    # Group tests by dataset
    datasets_needed = set()
    for tc in test_cases["single_turn"]:
        datasets_needed.add(tc["dataset"])
    for mt in test_cases.get("multi_turn", []):
        datasets_needed.add(mt["dataset"])

    # Process by dataset to minimize session creation
    for dataset_name in sorted(datasets_needed):
        print(f"\n--- Dataset: {dataset_name} ---")

        # Create session and load dataset
        session_id = create_session()
        csv_path = DATASETS_DIR / DATASET_FILES.get(dataset_name, f"{dataset_name}.csv")
        if not csv_path.exists():
            print(f"  SKIP: {csv_path} not found")
            continue
        if not load_dataset(session_id, dataset_name, csv_path):
            print(f"  SKIP: Failed to load {dataset_name}")
            continue

        # Run single-turn tests for this dataset
        dataset_tests = [tc for tc in test_cases["single_turn"] if tc["dataset"] == dataset_name]
        for tc in dataset_tests:
            for clarity in clarities:
                # New session per clarity level to avoid context contamination
                if clarity != "precise":
                    session_id = create_session()
                    load_dataset(session_id, dataset_name, csv_path)

                result = run_single_test(session_id, tc, clarity, provider_config)
                all_results["tests"].append(result)
                # Rate limit delay - longer for OpenRouter models
                delay = 8.0 if provider_config.get("base_url", "").startswith("https://openrouter") else 0.5
                time.sleep(delay)

        # Run multi-turn tests for this dataset
        dataset_mts = [mt for mt in test_cases.get("multi_turn", [])
                       if mt["dataset"] == dataset_name]
        for mt in dataset_mts:
            session_id = create_session()
            load_dataset(session_id, dataset_name, csv_path)
            mt_result = run_multi_turn(session_id, mt, provider_config)
            all_results["multi_turn"].append(mt_result)
            time.sleep(1)

    return all_results


def compute_summary(all_model_results):
    """Compute summary statistics across all models."""
    summary = []
    for model_results in all_model_results:
        if model_results is None:
            continue

        tests = model_results["tests"]
        total = len(tests)
        if total == 0:
            continue

        # Adequate rate (>= 5/8 or proportional for tests without numerical)
        adequate = sum(1 for t in tests
                       if (t["total_score"] / max(t.get("max_score", 8), 1)) >= 0.625)

        # By clarity
        by_clarity = {}
        for clarity in ["precise", "moderate", "vague"]:
            ct = [t for t in tests if t["clarity"] == clarity]
            if ct:
                by_clarity[clarity] = {
                    "total": len(ct),
                    "adequate": sum(1 for t in ct
                                    if (t["total_score"] / max(t.get("max_score", 8), 1)) >= 0.625),
                    "avg_tool_score": sum(t["scores"]["tool_selection"] for t in ct) / len(ct),
                    "avg_param_score": sum(t["scores"]["parameter_extraction"] for t in ct) / len(ct),
                }

        # Tool selection accuracy
        tool_exact = sum(1 for t in tests if t["scores"]["tool_selection"] == 2)
        tool_acceptable = sum(1 for t in tests if t["scores"]["tool_selection"] >= 1)

        # Multi-turn
        mt = model_results.get("multi_turn", [])
        mt_completed = sum(1 for m in mt if m.get("completed", False))

        summary.append({
            "model": model_results["model"],
            "model_id": model_results["model_id"],
            "total_tests": total,
            "adequate_count": adequate,
            "adequate_rate": adequate / total if total else 0,
            "tool_exact_rate": tool_exact / total if total else 0,
            "tool_acceptable_rate": tool_acceptable / total if total else 0,
            "by_clarity": by_clarity,
            "multi_turn_total": len(mt),
            "multi_turn_completed": mt_completed,
        })

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run e2e evaluation")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Models to evaluate")
    parser.add_argument("--clarity", nargs="+", default=["precise", "moderate", "vague"],
                        help="Clarity levels to test")
    parser.add_argument("--skip-mt", action="store_true",
                        help="Skip multi-turn tests")
    parser.add_argument("--test-ids", nargs="+", default=None,
                        help="Only run specific test IDs (e.g., R1 P1)")
    args = parser.parse_args()

    # Load test cases
    with open(TEST_CASES_FILE) as f:
        test_cases = json.load(f)

    # Filter test IDs if specified
    if args.test_ids:
        test_cases["single_turn"] = [
            tc for tc in test_cases["single_turn"]
            if tc["id"] in args.test_ids
        ]

    if args.skip_mt:
        test_cases["multi_turn"] = []

    # Verify backend is running
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        r.raise_for_status()
        print(f"Backend OK: {r.json()}")
    except Exception as e:
        print(f"ERROR: Backend not reachable at {BASE_URL}: {e}")
        sys.exit(1)

    # Run evaluation
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_key in args.models:
        if model_key not in MODELS:
            print(f"Unknown model: {model_key}")
            continue

        results = run_evaluation_for_model(model_key, test_cases, args.clarity)
        if results:
            all_results.append(results)

            # Save per-model results
            outfile = RESULTS_DIR / f"results_{model_key}.json"
            with open(outfile, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nSaved results to {outfile}")

    # Compute and save summary
    if all_results:
        summary = compute_summary(all_results)
        summary_file = RESULTS_DIR / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {summary_file}")

        # Print summary table
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        for s in summary:
            print(f"\n{s['model']} ({s['model_id']}):")
            print(f"  Adequate rate: {s['adequate_count']}/{s['total_tests']} "
                  f"({s['adequate_rate']:.1%})")
            print(f"  Tool exact: {s['tool_exact_rate']:.1%}, "
                  f"acceptable: {s['tool_acceptable_rate']:.1%}")
            for cl, cd in s.get("by_clarity", {}).items():
                print(f"  {cl}: {cd['adequate']}/{cd['total']} adequate, "
                      f"tool={cd['avg_tool_score']:.2f}/2")
            if s["multi_turn_total"] > 0:
                print(f"  Multi-turn: {s['multi_turn_completed']}/{s['multi_turn_total']} completed")


if __name__ == "__main__":
    main()
