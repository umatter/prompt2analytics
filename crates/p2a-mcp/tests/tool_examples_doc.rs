//! The MCP documentation is checked against the tool router (INF-5, INF-12).
//!
//! - Every `/tool key:value ...` chat example inside a fence of docs/guides/MCP_TOOL_EXAMPLES.md and
//!   README.md names a registered tool, only fields of its input schema and every required field, with
//!   values of the field's JSON type. The guide's `### tool` headings and `**Parameters:**` bullets name
//!   real tools and fields (INF-5). Request structs refuse an unknown field, so a drifted
//!   example fails for its reader.
//! - README.md's tool table is generated from the category table and its MCP sections name real tools;
//!   docs/mcp/tools-reference.md is what `generate_markdown_docs()` renders; every "N tools" in the docs
//!   is the registered count (INF-12).
//!
//! The `database` tools (`OPTIONAL_TOOLS`) exist only in builds with that feature (the default build).
//! Without it (CI's `http,llm,auth,websocket`) their examples are skipped and the reference is compared row by
//! row. `P2A_UPDATE_DOCS=1` rewrites the generated parts (the reference only in a `database` build).

#[path = "../examples/generate_tool_docs.rs"]
mod generator;

use p2a_mcp::AnalyticsServer;
use p2a_mcp::tools::generate_markdown_docs;
use p2a_mcp::tools::registry::{CATEGORY_TABLE, OPTIONAL_TOOLS};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

const EXAMPLE_DOCS: [&str; 2] = ["docs/guides/MCP_TOOL_EXAMPLES.md", "README.md"];
const COUNT_DOCS: [&str; 6] = [
    "README.md",
    "CLAUDE.md",
    "API_ENDPOINTS.md",
    "docs/guides/MCP_TOOL_EXAMPLES.md",
    "docs/guides/TESTING.md",
    "docs/cookbook.md",
];

/// This build's input schemas by tool name.
fn schemas() -> &'static BTreeMap<String, Value> {
    static SCHEMAS: OnceLock<BTreeMap<String, Value>> = OnceLock::new();
    SCHEMAS.get_or_init(|| {
        AnalyticsServer::new()
            .router_tools()
            .into_iter()
            .map(|t| (t.name.to_string(), Value::Object((*t.input_schema).clone())))
            .collect()
    })
}

fn read(rel: &str) -> String {
    std::fs::read_to_string(generator::repo_root().join(rel))
        .unwrap_or_else(|e| panic!("{rel}: {e}"))
}

fn update_requested() -> bool {
    std::env::var_os("P2A_UPDATE_DOCS").is_some_and(|v| v == "1")
}

struct ToolExample {
    at: String,
    tool: String,
    args: Vec<(String, String)>,
}

/// `/tool key:value ...` lines inside fences.
fn tool_examples(file: &str, md: &str) -> Vec<ToolExample> {
    let mut out = Vec::new();
    let mut in_fence = false;
    for (i, raw) in md.lines().enumerate() {
        let t = raw.trim();
        if t.starts_with("```") || t.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        let Some(rest) = t.strip_prefix('/').filter(|_| in_fence) else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == '_')
            .collect();
        let after = &rest[name.len()..];
        if name.is_empty() || !(after.is_empty() || after.starts_with(char::is_whitespace)) {
            continue; // a path such as /path/to/data.csv
        }
        out.push(ToolExample {
            at: format!("{file}:{} /{name}", i + 1),
            tool: name,
            args: chat_args(after),
        });
    }
    out
}

/// The chat shorthand `key:value key2:"quoted value" list:a,b json:[1, 2]`: a word that starts with
/// `identifier:` opens an argument; any other word continues the previous value.
fn chat_args(s: &str) -> Vec<(String, String)> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let (mut quote, mut depth) = (false, 0i32);
    for c in s.chars() {
        match c {
            '"' => {
                quote = !quote;
                cur.push(c);
            }
            '[' | '{' if !quote => {
                depth += 1;
                cur.push(c);
            }
            ']' | '}' if !quote => {
                depth -= 1;
                cur.push(c);
            }
            c if c.is_whitespace() && !quote && depth <= 0 => {
                if !cur.is_empty() {
                    words.push(std::mem::take(&mut cur));
                }
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        words.push(cur);
    }
    let mut args: Vec<(String, String)> = Vec::new();
    for w in words {
        let key: String = w
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        let is_key = key
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            && w[key.len()..].starts_with(':');
        if is_key {
            let value = w[key.len() + 1..].to_string();
            args.push((key, value));
        } else if let Some(last) = args.last_mut() {
            last.1.push(' ');
            last.1.push_str(&w);
        }
    }
    args
}

/// A `$ref` to `#/$defs/X` (or any `#/...` path) resolved against the schema root, one level.
fn resolve_ref<'a>(root: &'a Value, v: &'a Value) -> &'a Value {
    match v
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|r| r.strip_prefix("#/"))
    {
        Some(path) => path.split('/').fold(root, |acc, k| &acc[k]),
        None => v,
    }
}

/// The JSON types a property accepts (`type`, type arrays, `anyOf`/`oneOf`/`allOf`, refs).
fn prop_types(root: &Value, prop: &Value, depth: usize) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if depth > 8 {
        return out;
    }
    let prop = resolve_ref(root, prop);
    match prop.get("type") {
        Some(Value::String(t)) => {
            out.insert(t.clone());
        }
        Some(Value::Array(ts)) => {
            out.extend(ts.iter().filter_map(Value::as_str).map(str::to_string))
        }
        _ => {}
    }
    for k in ["anyOf", "oneOf", "allOf"] {
        if let Some(Value::Array(subs)) = prop.get(k) {
            for s in subs {
                out.extend(prop_types(root, s, depth + 1));
            }
        }
    }
    if out.is_empty() && prop.get("enum").is_some() {
        out.insert("string".to_string());
    }
    out
}

/// Whether a shorthand value fits the property (strings, arrays and objects accept any text).
fn value_fits(value: &str, types: &BTreeSet<String>, allowed: Option<&Vec<Value>>) -> bool {
    let v = value.trim();
    let v = v
        .strip_prefix('"')
        .and_then(|x| x.strip_suffix('"'))
        .unwrap_or(v);
    if let Some(vals) = allowed
        && !vals.iter().any(|e| e.as_str() == Some(v))
    {
        return false;
    }
    let concrete: Vec<&str> = types
        .iter()
        .map(String::as_str)
        .filter(|t| *t != "null")
        .collect();
    concrete.is_empty()
        || concrete.iter().any(|t| match *t {
            "integer" => v.parse::<i64>().is_ok(),
            "number" => v.parse::<f64>().is_ok_and(f64::is_finite),
            "boolean" => v == "true" || v == "false",
            _ => true,
        })
}

/// Option values the tools refuse: option strings are free text in the input schemas, so the type check
/// cannot see them. (tool, field, value), compared case-insensitively.
const REFUSED: &[(&str, &str, &str)] = &[("regression_quantreg", "se", "rank")];

/// The refused `field:value` of an example, if it uses one.
fn refused_value(ex: &ToolExample) -> Option<String> {
    ex.args.iter().find_map(|(k, v)| {
        let v = v.trim().trim_matches('"');
        REFUSED
            .iter()
            .find(|(tool, field, value)| {
                *tool == ex.tool && field == k && value.eq_ignore_ascii_case(v)
            })
            .map(|(_, field, value)| format!("{field}:{value}"))
    })
}

fn check(ex: &ToolExample, errors: &mut Vec<String>, skipped: &mut usize) {
    if let Some(r) = refused_value(ex) {
        errors.push(format!("{}: `{r}` is refused by the tool", ex.at));
    }
    let Some(schema) = schemas().get(&ex.tool) else {
        if OPTIONAL_TOOLS.contains(&ex.tool.as_str()) {
            *skipped += 1;
        } else {
            errors.push(format!("{}: no tool `{}` is registered", ex.at, ex.tool));
        }
        return;
    };
    let props = schema.get("properties").and_then(Value::as_object);
    let names: Vec<&str> = props
        .map(|p| p.keys().map(String::as_str).collect())
        .unwrap_or_default();
    for (k, v) in &ex.args {
        match props.and_then(|p| p.get(k)) {
            None => errors.push(format!(
                "{}: `{k}` is not a field (fields: {})",
                ex.at,
                names.join(", ")
            )),
            Some(p) => {
                let types = prop_types(schema, p, 0);
                let allowed = resolve_ref(schema, p).get("enum").and_then(Value::as_array);
                if !value_fits(v, &types, allowed) {
                    errors.push(format!(
                        "{}: `{k}:{v}` does not fit the field's type {types:?}",
                        ex.at
                    ));
                }
            }
        }
    }
    for r in schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        if !ex.args.iter().any(|(k, _)| k == r) {
            errors.push(format!("{}: required field `{r}` is missing", ex.at));
        }
    }
}

#[test]
fn test_chat_args_parse_the_shorthand() {
    let a = chat_args(
        r#" dataset:sales x:sqft,bedrooms title:"Sales Analysis" start:[200, 0.1] query:SELECT * FROM t path::memory:"#,
    );
    let keys: Vec<&str> = a.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["dataset", "x", "title", "start", "query", "path"]);
    assert_eq!(a[2].1, "\"Sales Analysis\"");
    assert_eq!(a[3].1, "[200, 0.1]");
    assert_eq!(a[4].1, "SELECT * FROM t");
    assert_eq!(a[5].1, ":memory:");
    // Plan sync: the list and map values Tier 2 added to the guide (load_dataset na_strings, chisq_gof expected).
    let b = chat_args(
        r#" path:/d/x.csv na_strings:[".", "NA", ""] expected:{"O":0.44,"A":0.42} column:blood_type"#,
    );
    let keys: Vec<&str> = b.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["path", "na_strings", "expected", "column"]);
    assert_eq!(b[1].1, r#"[".", "NA", ""]"#);
    assert_eq!(b[2].1, r#"{"O":0.44,"A":0.42}"#);
}

#[test]
fn test_json_types_resolve_refs_and_nullable_unions() {
    let schema = json!({
        "type": "object",
        "properties": {
            "k": {"type": ["integer", "null"]},
            "level": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "flag": {"type": "boolean"},
            "spec": {"$ref": "#/$defs/Spec"},
            "method": {"enum": ["a", "b"]}
        },
        "$defs": {"Spec": {"type": "object"}}
    });
    let t = |p: &str| prop_types(&schema, &schema["properties"][p], 0);
    assert!(value_fits("3", &t("k"), None) && !value_fits("three", &t("k"), None));
    assert!(value_fits("0.95", &t("level"), None) && !value_fits("high", &t("level"), None));
    assert!(value_fits("true", &t("flag"), None) && !value_fits("yes", &t("flag"), None));
    assert!(value_fits("{\"a\":1}", &t("spec"), None));
    let allowed = schema["properties"]["method"]["enum"].as_array();
    assert!(value_fits("a", &t("method"), allowed) && !value_fits("c", &t("method"), allowed));
}

/// A refused option value fails the check even though its type fits.
#[test]
fn test_refused_option_values_are_caught() {
    let md = "```\n/regression_quantreg dataset:d y:y x:x se:rank\n/regression_quantreg dataset:d y:y x:x se:nid\n```\n";
    let found: Vec<Option<String>> = tool_examples("fixture.md", md)
        .iter()
        .map(refused_value)
        .collect();
    assert_eq!(found, [Some("se:rank".to_string()), None]);
}

/// INF-5: every chat-shorthand tool example names a registered tool and only its fields, gives every
/// required field, and each value fits the field's type. Failed on the older schemas: `log_rank_test`,
/// 84 unknown-field uses in 63 tool/field pairs (`kaplan_meier strata:`, `cox_ph x:`, `rd_estimate
/// bandwidth:`, ...) and 57 missing required fields.
#[test]
fn test_documented_tool_examples_match_the_input_schemas() {
    let (mut errors, mut skipped, mut n) = (Vec::new(), 0usize, 0usize);
    for file in EXAMPLE_DOCS {
        for ex in tool_examples(file, &read(file)) {
            n += 1;
            check(&ex, &mut errors, &mut skipped);
        }
    }
    assert!(
        n >= 100,
        "only {n} tool examples found: did the extraction break?"
    );
    assert!(
        errors.is_empty(),
        "{} problems in the tool examples ({skipped} examples of tools this build lacks were skipped):\n{}",
        errors.len(),
        errors.join("\n")
    );
}

/// `### name` or `### \`name\`` where name is one snake_case token.
fn heading_tool(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("### ")?.trim().trim_matches('`');
    let tool_like = rest.chars().next().is_some_and(|c| c.is_ascii_lowercase())
        && rest
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    tool_like.then_some(rest)
}

/// INF-5: the guide's `### tool` headings name tools and its `**Parameters:**` bullets name fields of the
/// tool above them. Fails before: heading `log_rank_test` and 23 bullets (`rd_estimate` `y`, `running`,
/// `bandwidth`, `polynomial`; `cox_ph` `x`, `ties`, `robust_se`; `ml_svm` `C`; ...).
#[test]
fn test_documented_tool_headings_and_parameter_lists_are_real() {
    let md = read("docs/guides/MCP_TOOL_EXAMPLES.md");
    let (mut problems, mut tool, mut params, mut in_fence) =
        (Vec::new(), None::<String>, false, false);
    for (i, line) in md.lines().enumerate() {
        let at = format!("MCP_TOOL_EXAMPLES.md:{}", i + 1);
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if line.starts_with('#') {
            params = false;
            tool = heading_tool(line).map(str::to_string);
            if let Some(t) = &tool
                && !schemas().contains_key(t)
                && !OPTIONAL_TOOLS.contains(&t.as_str())
            {
                problems.push(format!("{at}: heading `{t}` is not a tool"));
            }
            continue;
        }
        if line.starts_with("**Parameters") {
            params = true;
            continue;
        }
        if params {
            if let Some(rest) = line.strip_prefix("- `") {
                let field = rest.split('`').next().unwrap_or("");
                if let Some(t) = &tool
                    && let Some(schema) = schemas().get(t)
                    && schema["properties"].get(field).is_none()
                {
                    problems.push(format!("{at}: `{field}` is not a field of `{t}`"));
                }
            } else if !line.trim().is_empty() && !line.starts_with(' ') {
                params = false;
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} headings or parameter bullets name no tool or field:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// INF-12: README.md's tool table is generated from the category table. Fails before: the markers are
/// missing (the hand-written table named 12 nonexistent tools and three empty wildcards).
#[test]
fn test_readme_tool_table_is_generated() {
    let readme = read("README.md");
    let want = generator::wrap_body(&generator::readme_tool_table());
    let have = generator::section_body(
        &readme,
        generator::README_TABLE_BEGIN,
        generator::README_TABLE_END,
    )
    .unwrap_or_else(|e| panic!("README.md: {e}"));
    if have != want && update_requested() {
        let new = generator::splice(
            &readme,
            generator::README_TABLE_BEGIN,
            generator::README_TABLE_END,
            &generator::readme_tool_table(),
        )
        .unwrap_or_else(|e| panic!("README.md: {e}"));
        generator::write_atomically(&generator::repo_root().join("README.md"), &new)
            .unwrap_or_else(|e| panic!("{e}"));
        return;
    }
    assert!(
        have == want,
        "README.md's tool table is stale: run `cargo run -p p2a-mcp --example generate_tool_docs`"
    );
}

fn glob(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !name.starts_with(first) || !name.ends_with(last) || name.len() < first.len() + last.len() {
        return false;
    }
    let mut rest = &name[first.len()..name.len() - last.len()];
    for p in &parts[1..parts.len() - 1] {
        match rest.find(p) {
            Some(k) => rest = &rest[k + p.len()..],
            None => return false,
        }
    }
    true
}

/// INF-12: in README.md's sections whose heading names MCP, every backticked snake_case or wildcard
/// token, and every lowercase identifier in a table row, is a tool or matches one. Fails before:
/// regression_quantile, iv_diagnostics, synth_control, matching_*, ipw_*, multinomial_logit,
/// ordered_probit, poisson, zip, zinb, ts_garch, ts_kalman_*, spatial_sar, spatial_sem, spatial_sac.
#[test]
fn test_readme_mcp_sections_name_real_tools() {
    let readme = read("README.md");
    let names: Vec<&str> = CATEGORY_TABLE.iter().map(|row| row.0).collect();
    let (mut level, mut in_fence, mut problems) = (None::<usize>, false, Vec::new());
    for (i, line) in readme.lines().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        if line.starts_with('#') {
            let depth = line.chars().take_while(|c| *c == '#').count();
            if line.contains("MCP") {
                level = Some(depth);
            } else if level.is_some_and(|l| depth <= l) {
                level = None;
            }
            continue;
        }
        if level.is_none() {
            continue;
        }
        let row = line.trim_start().starts_with('|');
        for span in line.split('`').skip(1).step_by(2) {
            let ident = span.chars().next().is_some_and(|c| c.is_ascii_lowercase())
                && span
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '*');
            if !ident || !(span.contains('_') || row) {
                continue;
            }
            let known = if span.contains('*') {
                names.iter().any(|n| glob(span, n))
            } else {
                names.contains(&span)
            };
            if !known {
                problems.push(format!("README.md:{}: `{span}` is not a tool", i + 1));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} README names are not tools:\n{}",
        problems.len(),
        problems.join("\n")
    );
}

/// The committed docs/mcp/tools-reference.md is what `generate_markdown_docs()` renders: equal with the
/// `database` feature (default); without it (CI's set), this build's rows plus exactly the optional ones.
#[test]
fn test_tools_reference_is_generated() {
    let path = "docs/mcp/tools-reference.md";
    let want = generate_markdown_docs();
    let have = read(path);
    if cfg!(feature = "database") {
        if have != want && update_requested() {
            generator::write_atomically(&generator::repo_root().join(path), &want)
                .unwrap_or_else(|e| panic!("{e}"));
            return;
        }
        assert!(
            have == want,
            "{path} is stale: run `cargo run -p p2a-mcp --example generate_tool_docs` (default features)"
        );
    } else {
        let rows = |md: &str| -> BTreeSet<String> {
            md.lines()
                .filter(|l| l.starts_with("| `"))
                .map(str::to_string)
                .collect()
        };
        let is_optional = |l: &String| {
            OPTIONAL_TOOLS
                .iter()
                .any(|t| l.starts_with(&format!("| `{t}` |")))
        };
        let committed: BTreeSet<String> = rows(&have)
            .into_iter()
            .filter(|l| !is_optional(l))
            .collect();
        let built = rows(&want);
        assert!(
            committed == built,
            "{path} is stale (regenerate with the default features): missing {:?}; not in this build {:?}",
            built.difference(&committed).collect::<Vec<_>>(),
            committed.difference(&built).collect::<Vec<_>>()
        );
        let absent: Vec<&str> = OPTIONAL_TOOLS
            .iter()
            .copied()
            .filter(|t| !have.contains(&format!("| `{t}` |")))
            .collect();
        assert!(
            absent.is_empty(),
            "{path} lacks {absent:?}: regenerate it with the default features"
        );
        // The rest of the page: the total and every category heading's count are those of the
        // default build (the category table), and the categories are this build's.
        let total = format!("Total tools: {}\n", CATEGORY_TABLE.len());
        assert!(
            have.contains(&total),
            "{path} does not say `{}`: regenerate it with the default features",
            total.trim_end()
        );
        let committed_sections = reference_sections(&have);
        let miscounted: Vec<&(String, usize, usize)> = committed_sections
            .iter()
            .filter(|(_, declared, rows)| declared != rows)
            .collect();
        assert!(
            miscounted.is_empty(),
            "{path}: category headings whose count is not their number of rows (name, heading, rows): {miscounted:?}"
        );
        let rows_total: usize = committed_sections.iter().map(|s| s.2).sum();
        assert_eq!(rows_total, CATEGORY_TABLE.len(), "{path}: tool rows");
        let names =
            |v: &[(String, usize, usize)]| v.iter().map(|s| s.0.clone()).collect::<Vec<_>>();
        assert_eq!(
            names(&committed_sections),
            names(&reference_sections(&want)),
            "{path}: categories"
        );
    }
}

/// `(category, count in its heading, tool rows under it)` of every `## Category (N tools)` section
/// of the tools reference.
fn reference_sections(md: &str) -> Vec<(String, usize, usize)> {
    let mut out: Vec<(String, usize, usize)> = Vec::new();
    for line in md.lines() {
        if let Some(head) = line.strip_prefix("## ") {
            let (name, count) = head
                .strip_suffix(" tools)")
                .and_then(|h| h.rsplit_once(" ("))
                .unwrap_or_else(|| panic!("unexpected heading in the tools reference: {line}"));
            let declared = count
                .parse()
                .unwrap_or_else(|_| panic!("unexpected heading in the tools reference: {line}"));
            out.push((name.to_string(), declared, 0));
        } else if line.starts_with("| `")
            && let Some(last) = out.last_mut()
        {
            last.2 += 1;
        }
    }
    out
}

/// `(line, number)` of every "N tools", "N MCP tools", "N analytics tools", "N more tools", "N handlers"
/// and "(N total)"; N may carry a `+`.
fn tool_counts(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let words: Vec<String> = line
            .split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| matches!(c, '(' | ')' | ',' | '.' | ';' | ':' | '*' | '`'))
                    .to_string()
            })
            .collect();
        for (k, w) in words.iter().enumerate() {
            let digits = w.trim_end_matches('+');
            if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            let mut j = k + 1;
            if words
                .get(j)
                .is_some_and(|x| matches!(x.as_str(), "MCP" | "analytics" | "more"))
            {
                j += 1;
            }
            if words
                .get(j)
                .is_some_and(|x| matches!(x.as_str(), "tools" | "handlers" | "total"))
            {
                out.push((i + 1, w.clone()));
            }
        }
    }
    out
}

#[test]
fn test_tool_counts_are_found() {
    let found = tool_counts(
        "exposes 270 tools.\n## MCP Tools (270 total)\nabout (60+ tools)\n... 268 more tools\n17 handlers\n7 database tools",
    );
    let numbers: Vec<&str> = found.iter().map(|(_, n)| n.as_str()).collect();
    assert_eq!(numbers, ["270", "270", "60+", "268", "17"]);
}

/// INF-12: every tool count in the docs is the default build's (270, the category table) or the count
/// without the database tools; the router agrees with the table. Fails before: MCP_TOOL_EXAMPLES.md's
/// "60+ tools" (and API_ENDPOINTS.md's "268 more tools", since removed).
#[test]
fn test_documented_tool_counts_match_the_router() {
    let full = CATEGORY_TABLE.len();
    let without_db = full - OPTIONAL_TOOLS.len();
    let lacking = OPTIONAL_TOOLS
        .iter()
        .filter(|t| !schemas().contains_key(**t))
        .count();
    assert_eq!(
        schemas().len() + lacking,
        full,
        "the router and the category table disagree (the registry tests in tools/registry.rs say where)"
    );
    let mut problems = Vec::new();
    for file in COUNT_DOCS {
        for (line, n) in tool_counts(&read(file)) {
            if !n
                .parse::<usize>()
                .is_ok_and(|v| v == full || v == without_db)
            {
                problems.push(format!("{file}:{line}: \"{n}\" tools; the server has {full} ({without_db} without the database feature)"));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} tool counts are wrong:\n{}",
        problems.len(),
        problems.join("\n")
    );
}
