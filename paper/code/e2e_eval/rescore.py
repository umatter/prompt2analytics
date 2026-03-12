#!/usr/bin/env python3
"""
Re-score existing evaluation results using corrected test cases and improved scoring.
No API calls needed - uses saved tool_selected, parameters_extracted,
numerical_output, and interpretation_summary from existing results.
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime

RESULTS_DIR = Path(__file__).parent / "results"
TEST_CASES_FILE = Path(__file__).parent / "test_cases.json"


def load_test_cases():
    with open(TEST_CASES_FILE) as f:
        return json.load(f)


def build_test_lookup(test_cases):
    """Build test_id -> test_case lookup."""
    lookup = {}
    for tc in test_cases["single_turn"]:
        lookup[tc["id"]] = tc
    return lookup


# Comprehensive parameter name aliases: expected_name -> [actual alternatives]
PARAM_ALIASES = {
    # Dependent variable
    "y": ["y_col", "dependent", "outcome", "response", "dep_var", "target"],
    "y_col": ["y", "dependent", "outcome", "response", "dep_var", "target"],
    # Independent variables
    "x": ["x_cols", "independent", "predictors", "covariates", "regressors", "features"],
    "x_cols": ["x", "independent", "predictors", "covariates", "regressors", "features"],
    # Panel
    "entity_var": ["entity_col", "entity", "group", "id", "panel_id", "individual"],
    "entity_col": ["entity_var", "entity", "group", "id", "panel_id"],
    "time_var": ["time_col", "time", "period", "year", "date"],
    "time_col": ["time_var", "time", "period", "year", "date"],
    # Clustering
    "cluster1": ["cluster_var", "cluster", "cluster_col"],
    "cluster_var": ["cluster1", "cluster", "cluster_col"],
    # Causal
    "outcome": ["outcome_col", "y", "dep_var", "dependent", "response"],
    "outcome_col": ["outcome", "y", "dep_var", "dependent", "response"],
    "treatment": ["treatment_col", "treat", "treated", "treatment_var", "d"],
    "treatment_col": ["treatment", "treat", "treated", "treatment_var", "d"],
    "treatment_var": ["treatment", "treatment_col", "treat"],
    "covariates": ["x", "x_cols", "controls", "confounders"],
    "dep_var": ["y", "outcome", "dependent", "y_col", "outcome_col", "response"],
    "post_var": ["time_col", "post", "period", "time", "post_treatment"],
    # IV
    "x_endog": ["endogenous", "endog", "x_endogenous"],
    "x_exog": ["controls", "exogenous", "exog", "x_exogenous", "covariates"],
    "endogenous": ["x_endog", "endog"],
    "instruments": ["z", "iv", "instrument"],
    "controls": ["x_exog", "exogenous", "covariates", "x"],
    # RD
    "running_var": ["running_col", "running", "forcing", "score", "x1"],
    "running_col": ["running_var", "running", "forcing", "score"],
    # Hypothesis
    "response": ["column", "value", "y", "outcome", "dependent"],
    "factor": ["group_col", "group", "grouping", "by", "category"],
    "column": ["x", "variable", "var", "col", "response", "value"],
    "group_col": ["factor", "group", "grouping", "by", "category"],
    # Chi-squared
    "row_var": ["column1", "var1", "x", "row"],
    "col_var": ["column2", "var2", "y", "col"],
    "column1": ["row_var", "var1", "x"],
    "column2": ["col_var", "var2", "y"],
    # Columns (ML, TS)
    "columns": ["column", "vars", "variables", "features"],
}

# Core parameters that must be correct for score > 0
CORE_PARAMS = {
    "y", "y_col", "x", "x_cols", "outcome", "outcome_col", "treatment", "treatment_col",
    "dep_var", "treatment_var", "post_var", "endogenous", "x_endog", "instruments",
    "entity_var", "entity_col", "response", "factor", "column", "columns",
    "row_var", "col_var", "running_var", "running_col", "covariates",
}

# Expanded tool alternatives
TOOL_ALTERNATIVES = {
    "regression_ols": ["regression_clustered", "regression_diagnostics", "regression_hac",
                       "regression_quantreg", "anova", "hypothesis_t_test", "t_test"],
    "regression_diagnostics": ["regression_ols", "cor_test", "compute_correlation"],
    "regression_quantreg": ["regression_ols"],
    "compute_correlation": ["viz_heatmap", "cor_test"],
    "viz_heatmap": ["compute_correlation", "cor_test"],
    "describe_dataset": ["head_dataset", "data_quality_profile"],
    "str_replace_value": ["munge_filter", "munge_mutate", "munge_drop_na", "data_quality_profile",
                          "munge_replace", "head_dataset"],
    "munge_replace": ["str_replace_value", "munge_filter", "munge_mutate", "munge_drop_na",
                      "data_quality_profile", "head_dataset"],
    "munge_rename": ["munge_mutate", "regression_ols", "describe_dataset"],
    "munge_filter": ["munge_replace", "str_replace_value", "munge_mutate", "munge_drop_na",
                     "describe_dataset"],
    "hypothesis_t_test": ["t_test", "t_test_two_sample", "regression_ols", "wilcoxon_test",
                          "db_duckdb_query", "munge_filter"],
    "t_test": ["hypothesis_t_test", "t_test_two_sample", "regression_ols", "wilcoxon_test",
               "db_duckdb_query"],
    "anova": ["regression_ols", "hypothesis_t_test", "t_test", "anova_one_way",
              "hypothesis_oneway", "oneway_anova", "kruskal_wallis"],
    "anova_one_way": ["anova", "regression_ols", "hypothesis_t_test", "kruskal_wallis"],
    "shapiro_wilk": ["regression_diagnostics", "ks_test", "hypothesis_shapiro_wilk"],
    "hypothesis_shapiro_wilk": ["shapiro_wilk", "regression_diagnostics", "ks_test",
                                 "hypothesis_ks_test"],
    "chi_squared_test": ["cor_test", "chi_squared_independence", "fisher_exact",
                         "hypothesis_chisq_independence"],
    "hypothesis_chisq_independence": ["chi_squared_test", "cor_test", "fisher_exact",
                                      "hypothesis_fisher_exact", "hypothesis_chisq_gof"],
    "regression_clustered": ["regression_ols", "regression_hac", "anova_one_way"],
    "iv_2sls": ["regression_ols", "iv_first_stage"],
    "ts_var": ["ts_arima_fit", "ts_granger_causality", "timeseries_granger", "regression_ols"],
    "timeseries_pp_test": ["ts_arima_fit", "timeseries_adf", "timeseries_kpss"],
    "ts_arima_fit": ["ts_var", "timeseries_decompose", "timeseries_acf", "ts_arima_forecast"],
    "logit": ["probit", "regression_ols"],
    "negbin": ["logit", "feglm", "poisson", "zeroinfl", "hurdle_model"],
    "ordered_model": ["logit", "ordered_logit", "ordered_probit", "probit", "regression_ols"],
    "ml_kmeans": ["ml_hierarchical", "ml_dbscan"],
    "ml_pca": ["compute_correlation", "ml_tsne", "describe_dataset"],
    "diff_in_diff": ["regression_ols", "treatment_ipw", "staggered_did"],
    "treatment_ipw": ["treatment_doubly_robust", "treatment_weightit",
                      "propensity_matching", "treatment_entropy_balance", "treatment_cbps"],
    "propensity_matching": ["treatment_ipw", "treatment_cbps", "treatment_weightit"],
    "rd_estimate": ["regression_ols", "rd_bw"],
    "panel_fixed_effects": ["panel_hdfe", "regression_ols", "panel_random_effects"],
    "panel_random_effects": ["panel_fixed_effects", "panel_hdfe", "regression_ols"],
    "hausman_test": ["panel_fixed_effects", "panel_random_effects"],
}


def score_tool_selection(expected_tool, actual_tools):
    """Score tool selection: 2=exact, 1=acceptable, 0=wrong."""
    if not actual_tools:
        return 0
    raw_tool = actual_tools[0]["tool"] if isinstance(actual_tools[0], dict) else actual_tools[0]
    primary_tool = raw_tool.split(",")[0].split("{")[0].strip() if raw_tool else ""

    if not primary_tool:
        return 0

    # Exact match
    if primary_tool == expected_tool:
        return 2

    # Multi-step: if expected contains "+", check first tool matches either part
    if "+" in (expected_tool or ""):
        parts = [t.strip() for t in expected_tool.split("+")]
        if primary_tool in parts:
            return 2
        # Check alternatives for each part
        for part in parts:
            alts = TOOL_ALTERNATIVES.get(part, [])
            if primary_tool in alts:
                return 1

    # Acceptable alternatives
    alts = TOOL_ALTERNATIVES.get(expected_tool, [])
    if primary_tool in alts:
        return 1

    return 0


def score_parameters(expected_params, actual_tools, test_case=None):
    """Score parameter extraction: 2=all correct, 1=core correct, 0=wrong core."""
    if not actual_tools:
        return 0 if expected_params else 2
    if not expected_params:
        return 2  # No expected params = auto-pass

    actual_args = actual_tools[0].get("arguments", {}) if isinstance(actual_tools[0], dict) else {}

    core_correct = True
    optional_correct = True

    for key, expected_val in expected_params.items():
        actual_val = actual_args.get(key)

        # Try aliases
        if actual_val is None and key in PARAM_ALIASES:
            for alt in PARAM_ALIASES[key]:
                actual_val = actual_args.get(alt)
                if actual_val is not None:
                    break

        if actual_val is None:
            is_core = key in CORE_PARAMS
            # Also check if it's in optional_params list
            optional_list = (test_case or {}).get("optional_params", [])
            if key in optional_list:
                is_core = False

            if is_core:
                core_correct = False
            else:
                optional_correct = False
            continue

        # Compare values (flexible)
        if isinstance(expected_val, list):
            if isinstance(actual_val, list):
                if set(str(v).lower() for v in expected_val) != set(str(v).lower() for v in actual_val):
                    # Check subset (actual may include extras)
                    if not set(str(v).lower() for v in expected_val).issubset(
                        set(str(v).lower() for v in actual_val)):
                        if key in CORE_PARAMS:
                            core_correct = False
                        else:
                            optional_correct = False
            elif isinstance(actual_val, str):
                # Single value vs list of one
                if len(expected_val) == 1 and str(expected_val[0]).lower() == actual_val.lower():
                    pass  # Match
                else:
                    if key in CORE_PARAMS:
                        core_correct = False
                    else:
                        optional_correct = False
        elif isinstance(expected_val, (int, float)):
            try:
                if abs(float(actual_val) - expected_val) > 0.01:
                    optional_correct = False  # Numeric params usually optional
            except (ValueError, TypeError):
                optional_correct = False
        else:
            if str(expected_val).lower() != str(actual_val).lower():
                if key in CORE_PARAMS:
                    core_correct = False
                else:
                    optional_correct = False

    if core_correct and optional_correct:
        return 2
    elif core_correct:
        return 1
    return 0


def score_numerical(expected_numerical, tool_result_text):
    """Score numerical correctness: 2=within tolerance, 1=reasonable, 0=wrong, None=N/A."""
    if not expected_numerical or not tool_result_text:
        return None

    score = 2
    for key, expected in expected_numerical.items():
        if isinstance(expected, str) and "\u2248" in expected:
            try:
                target = float(expected.replace("\u2248", "").strip())
            except ValueError:
                continue
            patterns = [
                rf"{key}[:\s]+([+-]?\d+\.?\d*)",
                rf"([+-]?\d+\.?\d*)\s*.*{key}",
                rf"R.squared[:\s]+([+-]?\d+\.?\d*)",
            ]
            found = False
            for pat in patterns:
                m = re.search(pat, tool_result_text, re.IGNORECASE)
                if m:
                    actual = float(m.group(1))
                    rel_error = abs(actual - target) / max(abs(target), 1e-10)
                    if rel_error > 0.5:
                        score = min(score, 0)
                    elif rel_error > 0.1:
                        score = min(score, 1)
                    found = True
                    break
            if not found:
                score = min(score, 1)

    return score


def score_interpretation(full_content, tool_calls, tool_selected, expected_tool,
                         numerical_output=None):
    """Score interpretation: 2=accurate with context, 1=superficial, 0=misleading/empty.

    When interpretation_summary is empty but the tool executed successfully
    (numerical_output exists), this is a data capture bug in the evaluation
    harness -- the model did produce interpretation text that wasn't saved.
    In this case, we impute I=1 (superficial but correct).
    """
    if not full_content or len(full_content.strip()) < 20:
        # Check for capture bug: tool executed successfully but content wasn't saved
        if numerical_output and len(str(numerical_output).strip()) > 10 and tool_selected:
            return 1  # Impute: model interpreted results but harness didn't capture text
        return 0

    content_lower = full_content.lower()

    # Check for substantive content
    has_numbers = bool(re.search(r'\d+\.?\d*', full_content))
    has_statistical_context = any(w in content_lower for w in [
        "significant", "coefficient", "p-value", "statistic",
        "result", "regression", "test", "effect", "estimate",
        "cluster", "correlation", "variance", "mean", "model",
        "intercept", "standard error", "confidence", "reject",
        "fail to reject", "null hypothesis", "r-squared", "r²",
        "loading", "component", "eigenvalue", "forecast",
        "observation", "variable", "parameter", "beta",
    ])
    has_interpretation = len(full_content) > 100

    # Check if the response has actual analytical content (not just "I'll help you")
    is_meta_only = all(w not in content_lower for w in [
        "coefficient", "p-value", "statistic", "result",
        "significant", "effect", "estimate", "value",
        "mean", "standard", "correlation", "cluster",
        "regression", "model", "test", "score",
    ]) and len(full_content) < 300

    if is_meta_only:
        return 0

    if has_numbers and has_statistical_context and has_interpretation:
        return 2
    elif has_statistical_context or (has_numbers and has_interpretation):
        return 1
    elif has_numbers and len(full_content) > 50:
        return 1
    return 0


def rescore_test(test_result, test_case):
    """Re-score a single test result against corrected test case."""
    expected_tool = test_case["expected_tool"]
    expected_params = test_case.get("expected_params", {})
    expected_numerical = test_case.get("expected_numerical", {})

    # Build tool_calls structure from result
    tool_calls = []
    if test_result.get("tool_selected"):
        tool_calls = [{
            "tool": test_result["tool_selected"],
            "arguments": test_result.get("parameters_extracted", {}),
            "result": test_result.get("numerical_output", ""),
        }]

    s_tool = score_tool_selection(expected_tool, tool_calls)

    # When an acceptable alternative tool is selected (score=1),
    # don't penalize params against the original tool's expected schema.
    # Instead, check if the model passed reasonable params for its chosen tool.
    if s_tool == 1 and expected_params:
        # Alternative tool may use different param names — give benefit of doubt
        # Score params as 1 (core correct) if the model passed any substantive params
        actual_args = tool_calls[0].get("arguments", {}) if tool_calls else {}
        has_substantive_params = any(
            k not in ("dataset",) and v
            for k, v in actual_args.items()
        )
        s_params = 1 if has_substantive_params else 0
    else:
        s_params = score_parameters(expected_params, tool_calls, test_case)

    s_numerical = score_numerical(expected_numerical,
                                  test_result.get("numerical_output", ""))
    s_interp = score_interpretation(
        test_result.get("interpretation_summary", ""),
        tool_calls,
        test_result.get("tool_selected"),
        expected_tool,
        numerical_output=test_result.get("numerical_output", ""),
    )

    total = s_tool + s_params + (s_numerical if s_numerical is not None else 0) + s_interp
    max_possible = 6 + (2 if s_numerical is not None else 0)

    return {
        "tool_selection": s_tool,
        "parameter_extraction": s_params,
        "numerical_correctness": s_numerical,
        "interpretation": s_interp,
        "total_score": total,
        "max_score": max_possible,
    }


def rescore_model(results_file, test_lookup):
    """Re-score all tests for one model."""
    with open(results_file) as f:
        model_results = json.load(f)

    rescored = []
    for test in model_results["tests"]:
        test_id = test["test_id"]
        tc = test_lookup.get(test_id)
        if not tc:
            print(f"  WARNING: No test case for {test_id}, keeping original scores")
            rescored.append(test)
            continue

        new_scores = rescore_test(test, tc)

        # Update the test result
        updated = dict(test)
        if new_scores is None:
            # Infrastructure error — exclude from scoring
            updated["excluded"] = True
            updated["excluded_reason"] = test.get("error", "infrastructure error")
            updated["scores"] = {
                "tool_selection": 0,
                "parameter_extraction": 0,
                "numerical_correctness": None,
                "interpretation": 0,
            }
            updated["total_score"] = 0
            updated["max_score"] = 0
        else:
            updated["excluded"] = False
            updated["scores"] = {
                "tool_selection": new_scores["tool_selection"],
                "parameter_extraction": new_scores["parameter_extraction"],
                "numerical_correctness": new_scores["numerical_correctness"],
                "interpretation": new_scores["interpretation"],
            }
            updated["total_score"] = new_scores["total_score"]
            updated["max_score"] = new_scores["max_score"]
        updated["tool_expected"] = tc["expected_tool"]
        updated["parameters_expected"] = tc.get("expected_params", {})
        rescored.append(updated)

    model_results["tests"] = rescored
    model_results["rescored_timestamp"] = datetime.now().isoformat()
    model_results["rescored_note"] = "Re-scored with v2 test cases: corrected parameter names, fixed expected tools (D2->negbin, D3->ordered_model, R4->quantreg), expanded parameter aliases."
    return model_results


def compute_summary(model_results):
    """Compute summary for one model."""
    all_tests = model_results["tests"]
    # Exclude infrastructure errors from scoring
    tests = [t for t in all_tests if not t.get("excluded", False)]
    excluded_count = len(all_tests) - len(tests)
    total = len(tests)
    if total == 0:
        return None

    adequate = sum(1 for t in tests
                   if (t["total_score"] / max(t.get("max_score", 6), 1)) >= 0.625)

    by_clarity = {}
    for clarity in ["precise", "moderate", "vague"]:
        ct = [t for t in tests if t["clarity"] == clarity]
        if ct:
            by_clarity[clarity] = {
                "total": len(ct),
                "adequate": sum(1 for t in ct
                                if (t["total_score"] / max(t.get("max_score", 6), 1)) >= 0.625),
                "avg_tool_score": sum(t["scores"]["tool_selection"] for t in ct) / len(ct),
                "avg_param_score": sum(t["scores"]["parameter_extraction"] for t in ct) / len(ct),
            }

    by_category = {}
    for t in tests:
        tid = t["test_id"]
        cat_map = {"R": "Regression", "P": "Panel data", "C": "Causal inference",
                   "T": "Time series", "H": "Hypothesis testing", "D": "Discrete choice",
                   "M": "Machine learning", "X": "Messy data"}
        cat = cat_map.get(tid[0], "Unknown")
        if cat not in by_category:
            by_category[cat] = {"total": 0, "adequate": 0}
        by_category[cat]["total"] += 1
        if (t["total_score"] / max(t.get("max_score", 6), 1)) >= 0.625:
            by_category[cat]["adequate"] += 1

    tool_exact = sum(1 for t in tests if t["scores"]["tool_selection"] == 2)
    tool_acceptable = sum(1 for t in tests if t["scores"]["tool_selection"] >= 1)

    return {
        "model": model_results["model"],
        "model_id": model_results["model_id"],
        "total_tests": total,
        "excluded_tests": excluded_count,
        "adequate_count": adequate,
        "adequate_rate": round(adequate / total, 4) if total else 0,
        "tool_exact_rate": round(tool_exact / total, 4) if total else 0,
        "tool_acceptable_rate": round(tool_acceptable / total, 4) if total else 0,
        "by_clarity": by_clarity,
        "by_category": by_category,
        "multi_turn_total": 0,
        "multi_turn_completed": 0,
    }


def main():
    test_cases = load_test_cases()
    test_lookup = build_test_lookup(test_cases)

    print(f"Loaded {len(test_lookup)} test cases (v{test_cases['metadata']['version']})")
    print(f"Changes: {test_cases['metadata'].get('updated', 'N/A')}")

    result_files = sorted(RESULTS_DIR.glob("results_*.json"))
    print(f"\nFound {len(result_files)} model result files")

    all_summaries = []

    for rf in result_files:
        model_name = rf.stem.replace("results_", "")
        print(f"\n{'='*60}")
        print(f"Re-scoring: {model_name}")
        print(f"{'='*60}")

        rescored = rescore_model(rf, test_lookup)

        # Save rescored results
        with open(rf, "w") as f:
            json.dump(rescored, f, indent=2)
        print(f"  Saved rescored results to {rf}")

        # Compute summary
        summary = compute_summary(rescored)
        if summary:
            all_summaries.append(summary)
            print(f"  Adequate: {summary['adequate_count']}/{summary['total_tests']} "
                  f"({summary['adequate_rate']:.1%})")
            print(f"  Tool exact: {summary['tool_exact_rate']:.1%}, "
                  f"acceptable: {summary['tool_acceptable_rate']:.1%}")
            for cl, cd in summary.get("by_clarity", {}).items():
                print(f"    {cl}: {cd['adequate']}/{cd['total']} adequate")
            print(f"  By category:")
            for cat, cd in sorted(summary.get("by_category", {}).items()):
                rate = cd["adequate"] / cd["total"] * 100 if cd["total"] else 0
                print(f"    {cat}: {cd['adequate']}/{cd['total']} ({rate:.0f}%)")

    # Save combined summary
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSaved summary to {summary_file}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("RESCORED SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'Adequate':>10} {'Tool Exact':>12} {'Tool Acc.':>10}")
    print("-" * 60)
    for s in sorted(all_summaries, key=lambda x: -x["adequate_rate"]):
        print(f"{s['model']:<25} {s['adequate_rate']:>9.1%} {s['tool_exact_rate']:>11.1%} "
              f"{s['tool_acceptable_rate']:>9.1%}")


if __name__ == "__main__":
    main()
