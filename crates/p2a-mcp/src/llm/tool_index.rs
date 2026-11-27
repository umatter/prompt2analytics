//! The chat's tool index: every tool the assistant may call stays reachable while only a core
//! list goes to the model with full schemas.
//!
//! In [`LlmToolMode::Index`] (the default) the model gets [`CORE_TOOLS`] with their schemas, two
//! chat-only tools, `p2a_help` (look a tool up) and `p2a_run` (run any tool by name), and a
//! one-line index of every other callable tool in the system prompt ([`index_text`]). The tool
//! loop unwraps a `p2a_run` call into a call of the named tool, so everything after admission
//! sees the real name. [`LlmToolMode::Tier1`] is the earlier behaviour (the `TIER1_TOOLS` list),
//! kept for reproducing the paper's evaluation.

use super::ToolDefinition;
use super::tools::{INTERNAL_TOOLS, check_llm_tool_call};
use crate::tools::registry::{ToolCategory, category_name, registry, search_tools_scored};

pub use crate::config::LlmToolMode;

/// The chat-only tool that looks a tool up.
pub const P2A_HELP: &str = "p2a_help";
/// The chat-only tool that runs any tool by name.
pub const P2A_RUN: &str = "p2a_run";

/// Tools sent with their full schemas in index mode: the tools the paper's evaluation prompts
/// expect most, data management and the common plots.
pub const CORE_TOOLS: &[&str] = &[
    "load_dataset",
    "create_dataset",
    "list_datasets",
    "describe_dataset",
    "head_dataset",
    "export_dataset",
    "regression_ols",
    "regression_clustered",
    "regression_diagnostics",
    "panel_fixed_effects",
    "panel_random_effects",
    "hausman_test",
    "logit",
    "iv_2sls",
    "diff_in_diff",
    "treatment_ipw",
    "propensity_matching",
    "hypothesis_t_test",
    "anova_one_way",
    "compute_correlation",
    "marginal_effects",
    "ts_arima_fit",
    "ts_arima_forecast",
    "munge_filter",
    "munge_mutate",
    "munge_rename",
    "munge_one_hot_encode",
    "viz_scatter",
    "viz_histogram",
    "viz_line",
    "viz_boxplot",
];

/// Most bytes the index may take in the system prompt (about 6,000 tokens).
pub const INDEX_BYTE_BUDGET: usize = 24_000;

/// Longest index phrase, in characters.
const PHRASE_CHARS: usize = 70;

/// Most matches a `p2a_help` query lists.
const HELP_MATCHES: usize = 8;

/// Whether `name` is one of the chat-only tools (no router tool by that name).
pub fn is_chat_only(name: &str) -> bool {
    name == P2A_HELP || name == P2A_RUN
}

/// The definitions of `p2a_help` and `p2a_run`.
pub fn meta_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: P2A_HELP.to_string(),
            description: "Look up a tool from the tool index: with `tool`, its description and \
                          parameters; with `query`, the best-matching tools."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "A tool name from the index."},
                    "query": {"type": "string", "description": "Words describing the method."}
                },
                "additionalProperties": false
            }),
        },
        ToolDefinition {
            name: P2A_RUN.to_string(),
            description: "Run any tool from the tool index: `tool` is its name, `args` its \
                          parameters as p2a_help lists them."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tool": {"type": "string", "description": "A tool name from the index."},
                    "args": {"type": "object", "description": "The tool's parameters."}
                },
                "required": ["tool"],
                "additionalProperties": false
            }),
        },
    ]
}

/// Words whose period does not end a sentence.
const ABBREVIATIONS: &[&str] = &["e.g", "i.e", "etc", "vs", "cf", "approx", "incl", "resp"];

/// A tool's index phrase: its description's first sentence on one line (whitespace runs, line
/// breaks included, become one space; a period after an abbreviation such as "e.g." does not end
/// the sentence), cut to [`PHRASE_CHARS`].
fn phrase(description: &str) -> String {
    let flat = description.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut end = None;
    let mut from = 0;
    while let Some(i) = flat[from..].find(". ") {
        let at = from + i;
        let word = flat[..at]
            .rsplit(' ')
            .next()
            .unwrap_or("")
            .trim_start_matches(|c: char| !c.is_alphanumeric())
            .to_ascii_lowercase();
        if !ABBREVIATIONS.contains(&word.as_str()) {
            end = Some(at);
            break;
        }
        from = at + 2;
    }
    let first = match end {
        Some(at) => &flat[..at],
        None => flat.trim_end_matches('.'),
    };
    p2a_text::ellipsize(first.trim(), PHRASE_CHARS).into_owned()
}

/// A text field of a `p2a_help` call, trimmed; an empty one is absent.
pub fn help_field<'a>(arguments: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    arguments
        .get(key)
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// The index line of one tool.
fn index_line(name: &str, description: &str) -> String {
    format!("{name}: {}", phrase(description))
}

/// The index of every callable tool outside [`CORE_TOOLS`], grouped by category in
/// [`ToolCategory::all`] order.
pub fn index_text(callable: &dyn Fn(&str) -> bool) -> String {
    let mut out = String::from("## Tool index\n");
    for &category in ToolCategory::all() {
        let lines: Vec<String> = registry()
            .iter()
            .filter(|t| t.category == category)
            .filter(|t| {
                let n = t.name.as_str();
                callable(n) && !INTERNAL_TOOLS.contains(&n) && !CORE_TOOLS.contains(&n)
            })
            .map(|t| index_line(&t.name, &t.description))
            .collect();
        if lines.is_empty() {
            continue;
        }
        out.push_str(&format!("\n### {}\n", category_name(category)));
        for line in lines {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out
}

/// A JSON schema type as one word (`string`, `integer or null`, ...).
fn schema_type(prop: &serde_json::Value) -> String {
    match prop.get("type") {
        Some(serde_json::Value::String(t)) => t.clone(),
        Some(serde_json::Value::Array(ts)) => ts
            .iter()
            .filter_map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join(" or "),
        _ => match prop.get("anyOf").and_then(|a| a.as_array()) {
            Some(alts) => alts
                .iter()
                .map(schema_type)
                .collect::<Vec<_>>()
                .join(" or "),
            None => "value".to_string(),
        },
    }
}

/// The answer of a `p2a_help` call: with `tool`, the tool's description and one line per
/// parameter; with `query` (and no `tool`), the best-matching callable tools as index lines.
pub fn help(
    arguments: &serde_json::Value,
    callable: &dyn Fn(&str) -> bool,
) -> Result<String, String> {
    let field = |k: &str| help_field(arguments, k);
    if let Some(name) = field("tool") {
        check_llm_tool_call(name, callable)?;
        let info = registry()
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("Unknown tool `{name}`. Use a tool from the index."))?;
        let mut out = format!("{name}: {}\n\nParameters:\n", info.description);
        let required: Vec<&str> = info
            .input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|r| r.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        match info
            .input_schema
            .get("properties")
            .and_then(|p| p.as_object())
        {
            Some(props) if !props.is_empty() => {
                for (pname, prop) in props {
                    let need = if required.contains(&pname.as_str()) {
                        "required".to_string()
                    } else if let Some(d) = prop.get("default") {
                        format!("default {d}")
                    } else {
                        "optional".to_string()
                    };
                    let desc = prop
                        .get("description")
                        .and_then(|d| d.as_str())
                        .unwrap_or("");
                    out.push_str(&format!(
                        "- {pname} ({}, {need}): {desc}\n",
                        schema_type(prop)
                    ));
                }
            }
            _ => out.push_str("(none)\n"),
        }
        out.push_str(&format!(
            "\nRun it with p2a_run: {{\"tool\": \"{name}\", \"args\": {{...}}}}."
        ));
        return Ok(out);
    }
    if let Some(query) = field("query") {
        let matches: Vec<String> = search_tools_scored(query, None, HELP_MATCHES * 4)
            .into_iter()
            .filter(|m| callable(&m.tool.name))
            .take(HELP_MATCHES)
            .map(|m| index_line(&m.tool.name, &m.tool.description))
            .collect();
        if matches.is_empty() {
            return Ok(format!(
                "No tool matches {query:?}. Read the tool index in your instructions, or try \
                 other words."
            ));
        }
        return Ok(format!(
            "Tools matching {query:?} (call p2a_help with `tool` for parameters):\n{}",
            matches.join("\n")
        ));
    }
    Err(
        "p2a_help needs `tool` or `query`: a tool name from the index, or words describing \
         the method."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn callable(name: &str) -> bool {
        !INTERNAL_TOOLS.contains(&name) && registry().iter().any(|t| t.name == name)
    }

    #[test]
    fn index_lists_every_callable_non_core_tool_once() {
        let idx = index_text(&callable);
        for t in registry() {
            let n = t.name.as_str();
            let lines = idx
                .lines()
                .filter(|l| l.starts_with(&format!("{n}: ")))
                .count();
            let want = usize::from(callable(n) && !CORE_TOOLS.contains(&n));
            assert_eq!(lines, want, "{n}");
        }
        assert!(idx.contains("kaplan_meier: ") && idx.contains("sar_model: "));
    }

    #[test]
    fn index_stays_under_budget() {
        let n = index_text(&callable).len();
        assert!(n <= INDEX_BYTE_BUDGET, "{n} bytes");
    }

    #[test]
    fn core_tools_are_registered_and_not_internal() {
        for t in CORE_TOOLS {
            assert!(callable(t), "{t}");
        }
    }

    #[test]
    fn meta_tools_are_not_router_tools() {
        let defs = meta_tool_definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, [P2A_HELP, P2A_RUN]);
        for n in names {
            assert!(is_chat_only(n) && !registry().iter().any(|t| t.name == n));
        }
        assert_eq!(defs[1].parameters["required"], json!(["tool"]));
    }

    #[test]
    fn phrase_keeps_abbreviations_and_one_line() {
        assert_eq!(
            phrase("Fit a model, e.g. a probit. More text."),
            "Fit a model, e.g. a probit"
        );
        assert_eq!(
            phrase("Line one\ncontinues here. Next."),
            "Line one continues here"
        );
    }

    /// k-means uses the columns as given, so its description says how to put them on one scale.
    #[test]
    fn kmeans_description_points_to_standardizing() {
        let t = registry().iter().find(|t| t.name == "ml_kmeans").unwrap();
        assert!(
            t.description.contains("munge_standardize"),
            "{}",
            t.description
        );
    }

    #[test]
    fn help_with_tool_lists_parameters() {
        let h = help(&json!({"tool": "sar_model"}), &callable).unwrap();
        assert!(h.contains("dataset") && h.contains("required"), "{h}");
    }

    #[test]
    fn help_with_query_lists_matches() {
        let h = help(&json!({"query": "kaplan meier survival"}), &callable).unwrap();
        assert!(h.contains("kaplan_meier: "), "{h}");
    }

    #[test]
    fn help_refuses_internal_and_unknown() {
        let e = help(&json!({"tool": "cleaning_session_start"}), &callable).unwrap_err();
        assert!(e.contains("not available"), "{e}");
        let e = help(&json!({"tool": "no_such_tool"}), &callable).unwrap_err();
        assert!(e.contains("Unknown tool"), "{e}");
    }

    #[test]
    fn help_needs_tool_or_query_and_prefers_tool() {
        let e = help(&json!({}), &callable).unwrap_err();
        assert!(e.contains("`tool` or `query`"), "{e}");
        let both = help(&json!({"tool": "cox_ph", "query": "kmeans"}), &callable).unwrap();
        assert!(
            both.contains("cox_ph") && !both.contains("ml_kmeans: "),
            "{both}"
        );
    }
}
