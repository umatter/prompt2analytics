#!/usr/bin/env python3
"""
Generate a filtered tools.json containing only tools used in multi-turn test cases,
plus commonly needed supporting tools. OpenAI API limits to 128 tools.
"""

import json
from pathlib import Path

CONFIG_DIR = Path("/home/umatter/tools/prompt2analytics/paper/code/llm_eval/config")
TEST_CASES_DIR = Path("/home/umatter/tools/prompt2analytics/paper/code/llm_eval/test_cases/multi_turn")
FULL_TOOLS_FILE = CONFIG_DIR / "tools.json"
FILTERED_TOOLS_FILE = CONFIG_DIR / "tools_filtered.json"

def extract_tools_from_test_cases() -> set[str]:
    """Extract all tool names referenced in test case files."""
    tools = set()

    for json_file in TEST_CASES_DIR.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        for conv in data.get("conversations", []):
            for turn in conv.get("turns", []):
                # Get expected and acceptable tools
                if "expected_tool" in turn:
                    tools.add(turn["expected_tool"])
                for tool in turn.get("acceptable_tools", []):
                    tools.add(tool)

    return tools

def get_supporting_tools() -> set[str]:
    """Tools that support test case tools (e.g., diagnostics, visualization)."""
    return {
        # Tool discovery (meta-tools for scaling beyond 128 tools)
        "search_tools",
        "list_tool_categories",
        "tool_info",

        # Data management
        "load_dataset",
        "describe_dataset",
        "head_dataset",
        "list_datasets",
        "create_dataset",

        # Commonly used diagnostics
        "regression_diagnostics",
        "regression_ols",
        "regression_clustered",

        # Commonly needed visualization
        "viz_scatter",
        "viz_histogram",
        "viz_line",
        "viz_residual_diagnostics",
        "viz_coefficient",

        # Hypothesis tests
        "hypothesis_shapiro_wilk",
        "hypothesis_t_test_one",
        "hypothesis_t_test_two",
        "hypothesis_f_test",

        # Panel tools
        "panel_fixed_effects",
        "panel_random_effects",
        "panel_hdfe",
        "panel_gmm",
        "panel_gls",
        "hausman_test",

        # Causal tools
        "iv_2sls",
        "iv_first_stage",
        "iv_sargan_test",
        "diff_in_diff",
        "staggered_did",
        "rd_estimate",
        "rd_bw",
        "synthetic_control",
        "scpi",
        "gsynth",
        "propensity_matching",
        "treatment_ipw",
        "treatment_cbps",
        "treatment_doubly_robust",

        # Time series
        "ts_arima_fit",
        "ts_arima_forecast",
        "ts_var",
        "ts_var_irf",
        "ts_vecm",
        "ts_mstl",
        "timeseries_decompose",
        "timeseries_pp_test",
        "timeseries_box_test",

        # Regression variants
        "regression_nls",
        "regression_loess",
        "regression_quantile",
        "regression_bgtest",
        "regression_driscoll_kraay",
    }

def main():
    # Load all tools
    with open(FULL_TOOLS_FILE) as f:
        all_tools = json.load(f)["tools"]

    tool_dict = {t["function"]["name"]: t for t in all_tools}

    # Get required tools
    test_case_tools = extract_tools_from_test_cases()
    supporting_tools = get_supporting_tools()
    required_tools = test_case_tools | supporting_tools

    print(f"Tools from test cases: {len(test_case_tools)}")
    print(f"Supporting tools: {len(supporting_tools)}")
    print(f"Total unique required: {len(required_tools)}")

    # Filter to existing tools
    filtered_tools = []
    missing_tools = []

    for tool_name in sorted(required_tools):
        if tool_name in tool_dict:
            filtered_tools.append(tool_dict[tool_name])
        else:
            missing_tools.append(tool_name)

    print(f"\nFiltered tools (exist): {len(filtered_tools)}")
    if missing_tools:
        print(f"Missing tools ({len(missing_tools)}): {missing_tools}")

    # Check we're under the limit
    if len(filtered_tools) > 128:
        print(f"\nWARNING: Still over 128 tool limit ({len(filtered_tools)} tools)")
        print("Need to reduce further")
    else:
        print(f"\n✓ Under 128 tool limit")

    # Write filtered tools
    output = {"tools": filtered_tools}
    with open(FILTERED_TOOLS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nOutput: {FILTERED_TOOLS_FILE}")

    # List the tools
    print("\nIncluded tools:")
    for t in filtered_tools:
        print(f"  - {t['function']['name']}")

if __name__ == "__main__":
    main()
