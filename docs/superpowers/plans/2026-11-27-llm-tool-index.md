# Chat Tool Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every tool the chat may call is reachable from the chat model, at a fraction of today's per-call tool tokens, with today's behaviour kept as a server setting.

**Architecture:** A new `llm/tool_index.rs` holds the mode, the native core list, the generated index and the two chat-only tools' definitions and `p2a_help` answers. The tool loop unwraps `p2a_run` calls into real calls before loop detection and re-wraps non-offered calls when it sends messages to a provider. The chat routes admit the two chat-only names, answer `p2a_help` without the governor, and pick the tool list and prompt by the server's mode.

**Tech Stack:** Rust 2024, p2a-mcp (rmcp router, axum routes, tokio), clap config.

**Spec:** `docs/superpowers/specs/2026-11-27-llm-tool-index-design.md` (approved; its "Decisions from the builder check" section binds).

## Global Constraints

- Work in `/home/umatter/tools/prompt2analytics-fixes` (branch `fix/forecast-names`), absolute paths only; never touch `/home/umatter/tools/prompt2analytics`.
- Build and test on gazelle: rsync changed files to `umatter@gazelle:bench/prompt2analytics/`, then `ssh umatter@gazelle 'cd ~/bench/prompt2analytics && source ~/.cargo/env && TMPDIR=/dev/shm <cargo command>'`; run `cargo fmt` there and rsync formatted files back. Gazelle's git is meaningless; check diffs locally.
- Test command (CI feature set): `cargo test -q -p p2a-mcp --lib --no-default-features --features http,llm,auth,websocket --target-dir target/ci -- <filter>`.
- MCP tool names, request structs and handlers unchanged. `p2a_help` and `p2a_run` are not router tools: no `CATEGORY_TABLE` row, no limits rule, no dispatch arm, no change to the tool count.
- `INTERNAL_TOOLS` unchanged; internal tools appear nowhere in the index or `p2a_help` output.
- `tier1` mode: tool list and system prompt byte-identical to today's.
- No tool runs outside `ToolGovernor`; nothing executes a program.
- Commits: explicit `git add` paths, never `performance/`; end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No push without the user's go-ahead.
- New code clippy-clean: `cargo clippy -q -p p2a-mcp --no-default-features --features http,llm,auth,websocket --target-dir target/ci`.

## Review Focus

1. A model in `index` mode calls a long-tail tool **directly by name** (skipping `p2a_run`): it must still run, as today (registered tools stay callable). Test in Task 4.
2. `p2a_run` naming an internal or unknown tool: the same refusal text and audit outcome as a direct call. Test in Task 4.
3. `p2a_run` with `args` missing: treated as `{}` (tools with all-optional parameters, e.g. `list_datasets`, must work); `args` present but not an object, or `tool` missing: tool error naming the shape. Test in Task 2.
4. A conversation recorded in `tier1` mode continued in `index` mode, and the reverse (stored `p2a_help` calls become text in `tier1`). Test in Task 2.
5. `p2a_help` with both `tool` and `query`, or neither: `tool` wins; neither is an error naming the two fields. Test in Task 1.

---

### Task 1: `llm/tool_index.rs` — mode, core list, index, help

**Files:**
- Create: `crates/p2a-mcp/src/llm/tool_index.rs`
- Modify: `crates/p2a-mcp/src/llm/mod.rs` (declare module, re-export the Produces items)

**Interfaces:**
- Consumes: `tools::registry::{registry, search_tools_scored, category_name, ToolInfo}`; `llm::{INTERNAL_TOOLS, TIER1_TOOLS, ToolDefinition}`; `p2a_text::ellipsize`.
- Produces:
  - `pub enum LlmToolMode { Index, Tier1 }` — `Default` = `Index`; derives `Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum` (values `index`, `tier1`).
  - `pub const P2A_HELP: &str = "p2a_help"; pub const P2A_RUN: &str = "p2a_run";`
  - `pub const CORE_TOOLS: &[&str]` — exactly: `load_dataset, create_dataset, list_datasets, describe_dataset, head_dataset, export_dataset, regression_ols, regression_clustered, regression_diagnostics, panel_fixed_effects, panel_random_effects, hausman_test, logit, iv_2sls, diff_in_diff, treatment_ipw, propensity_matching, hypothesis_t_test, anova_one_way, compute_correlation, marginal_effects, ts_arima_fit, ts_arima_forecast, munge_filter, munge_mutate, munge_rename, munge_one_hot_encode, viz_scatter, viz_histogram, viz_line, viz_boxplot` (the paper's most expected tools plus data management and common plots).
  - `pub fn is_chat_only(name: &str) -> bool` — `P2A_HELP` or `P2A_RUN`.
  - `pub fn meta_tool_definitions() -> Vec<ToolDefinition>` — the two definitions. `p2a_help` params: `tool` (string), `query` (string), neither required. `p2a_run` params: `tool` (string, required), `args` (object). Descriptions: `p2a_help` "Look up a tool from the tool index: with `tool`, its description and parameters; with `query`, the best-matching tools." `p2a_run` "Run any tool from the tool index: `tool` is its name, `args` its parameters as p2a_help lists them."
  - `pub fn index_text(callable: &dyn Fn(&str) -> bool) -> String` — the markdown index (heading `## Tool index`, then `### <category>` groups in `ToolCategory::all()` order, one `name: phrase` line per tool that is callable, not internal, not in `CORE_TOOLS`); phrase = first sentence of the description (up to the first `". "`), `ellipsize`d to 70 chars (about 230 lines at about 95 bytes stay under the budget).
  - `pub const INDEX_BYTE_BUDGET: usize = 24_000;`
  - `pub fn help(arguments: &serde_json::Value, callable: &dyn Fn(&str) -> bool) -> Result<String, String>`.

- [ ] **Step 1: Write the failing tests** (in the module's `#[cfg(test)] mod tests`; `callable` = registered in `registry()` and not in `INTERNAL_TOOLS`)

```rust
#[test] fn index_lists_every_callable_non_core_tool_once() {
    let idx = index_text(&callable);
    for t in registry() { let n = t.tool.name.as_ref();
        let lines = idx.lines().filter(|l| l.starts_with(&format!("{n}: "))).count();
        let want = usize::from(callable(n) && !CORE_TOOLS.contains(&n));
        assert_eq!(lines, want, "{n}");
    }
    assert!(idx.contains("kaplan_meier: ") && idx.contains("sar_model: "));
}
#[test] fn index_stays_under_budget() { assert!(index_text(&callable).len() <= INDEX_BYTE_BUDGET); }
#[test] fn core_tools_are_registered_and_not_internal() { for t in CORE_TOOLS { assert!(callable(t), "{t}"); } }
#[test] fn help_with_tool_lists_parameters() {
    let h = help(&json!({"tool": "sar_model"}), &callable).unwrap();
    assert!(h.contains("dataset") && h.contains("required"), "{h}");
}
#[test] fn help_with_query_lists_matches() {
    let h = help(&json!({"query": "kaplan meier survival"}), &callable).unwrap();
    assert!(h.contains("kaplan_meier: "), "{h}");
}
#[test] fn help_refuses_internal_and_unknown() {
    assert!(help(&json!({"tool": "cleaning_session_start"}), &callable).unwrap_err().contains("not available"));
    assert!(help(&json!({"tool": "no_such_tool"}), &callable).unwrap_err().contains("Unknown tool"));
}
#[test] fn help_needs_tool_or_query_and_prefers_tool() {
    assert!(help(&json!({}), &callable).unwrap_err().contains("`tool` or `query`"));
    let both = help(&json!({"tool": "cox_ph", "query": "kmeans"}), &callable).unwrap();
    assert!(both.contains("cox_ph") && !both.contains("ml_kmeans: "));
}
```

- [ ] **Step 2: Run, expect FAIL** — filter `llm::tool_index`; compile error (module missing).

- [ ] **Step 3: Implement** the Produces items. `help` with `tool`: refuse via `check_llm_tool_call(name, callable)` (same texts as direct calls), then the full description plus one line per input-schema property `- name (type, required|default <v>|optional): description`. With `query`: `search_tools_scored(query, None, 8)` filtered by `callable`, rendered as index lines; no match → `Ok` text saying none matched and to read the index.

- [ ] **Step 4: Run, expect PASS** — same filter.

- [ ] **Step 5: Commit** — `feat(llm): tool index, core list and p2a_help answers for the chat`.

### Task 2: Tool loop — unwrap `p2a_run`, key `p2a_help`, re-wrap for the provider

**Files:**
- Modify: `crates/p2a-mcp/src/llm/tool_loop.rs` (`LoopGuard::check`, `run_tool_loop`, tests)

**Interfaces:**
- Consumes: `P2A_HELP`, `P2A_RUN` (Task 1).
- Produces:
  - `fn unwrap_run(call: ToolCall) -> ToolCall` — `p2a_run{tool, args}` → `ToolCall{ name: tool, arguments: args (missing → {}), id }`; malformed → unchanged (the executor answers it, Task 4).
  - `pub fn for_provider(messages: &[Message], tools: &[ToolDefinition]) -> Vec<Message>` — each assistant tool call whose name is not in `tools`: if `p2a_run` is offered, rendered as `p2a_run{tool: name, args}`; else if it is `p2a_help`, the call and its result become plain text (an assistant line `Looked up <args> with p2a_help.` and the result content moved into a user-role text message, the tool call and tool result removed); else unchanged.

- [ ] **Step 1: Write the failing tests** (with the existing `Scripted` provider and a recording executor; `offered` = definitions of `p2a_run`, `p2a_help`, `regression_ols`)

```rust
#[tokio::test] async fn p2a_run_calls_run_the_named_tool()        // executor sees ("sar_model", args); the returned Message's tool_calls name "sar_model"
#[tokio::test] async fn p2a_run_without_args_runs_with_an_empty_object() // executor sees ("list_datasets", {})
#[tokio::test] async fn malformed_p2a_run_reaches_the_executor_unchanged() // {"args": 3} → executor sees ("p2a_run", ...)
#[tokio::test] async fn p2a_run_is_unwrapped_only_when_offered()  // tools without p2a_run → executor sees "p2a_run"
#[tokio::test] async fn loop_limits_count_the_real_tool() // four rounds of p2a_run{sar_model} with different args → stops with repeated_tool_note("sar_model")
#[tokio::test] async fn different_help_lookups_do_not_stop_the_loop() // p2a_help{tool:a}, {tool:b}, {tool:c}, {tool:d} in four rounds → all four run
#[test] fn for_provider_wraps_calls_of_tools_not_offered()  // a stored "sar_model" call → p2a_run{tool:"sar_model",args}; "regression_ols" unchanged; ids and results untouched
#[test] fn for_provider_turns_help_into_text_when_help_is_not_offered() // tools = tier1 list: no tool call or tool result named p2a_help remains; its result text is present
#[tokio::test] async fn provider_sees_wrapped_history_and_clients_see_real_names() // Scripted.seen second call contains p2a_run{tool:"sar_model"}; returned tool_calls name "sar_model"
```

- [ ] **Step 2: Run, expect FAIL** — filter `llm::tool_loop`.

- [ ] **Step 3: Implement.** In `run_tool_loop`: map `turn.tool_calls` through `unwrap_run` only when `tools` offers `p2a_run`, before `guard.check`; send `provider.call(&for_provider(&conversation, tools), tools, stream)`. In `LoopGuard::check`, the per-round key is `p2a_help:<tool or query>` for `p2a_help` calls, the name otherwise (both the round count and the repeat hash already include arguments; only the round-count key changes).

- [ ] **Step 4: Run, expect PASS** — filter `llm::tool_loop`, plus the whole `llm::` filter for regressions.

- [ ] **Step 5: Commit** — `feat(llm): the tool loop unwraps p2a_run and shows providers their own call shape`.

### Task 3: Mode setting, tool list and system prompt

**Files:**
- Modify: `crates/p2a-mcp/src/config.rs` (new arg), `crates/p2a-mcp/src/server.rs` (`AnalyticsServer` field, builder, `list_tools_for_llm`), `crates/p2a-mcp/src/transport/http.rs` (`build_app` passes the mode), `crates/p2a-mcp/src/llm/tools.rs` (prompt), `docs/guides/DEPLOYMENT.md` (configuration reference row)

**Interfaces:**
- Consumes: Task 1's `LlmToolMode`, `CORE_TOOLS`, `meta_tool_definitions`, `index_text`.
- Produces:
  - Config: `#[arg(long, value_enum, env = "P2A_LLM_TOOLS", default_value = "index")] pub llm_tools: LlmToolMode`.
  - `AnalyticsServer::with_llm_tool_mode(self, mode: LlmToolMode) -> Self`, `pub fn llm_tool_mode(&self) -> LlmToolMode` (default `Index`).
  - `list_tools_for_llm(&self)`: `Tier1` → today's filter; `Index` → router tools in `CORE_TOOLS` (not internal) followed by `meta_tool_definitions()`.
  - `pub fn get_system_prompt_for(mode: LlmToolMode, dataset_context: Option<&str>) -> String` in `llm/tools.rs`; `get_system_prompt_with_context` stays as the `Tier1` form. `Index`: base prompt, then the instruction block below, then `index_text` (generated once in a `OnceLock`), then the dataset block, then `LANGUAGE_RULE`.
  - Instruction block (exact copy): `## Tools beyond your tool list\n\nYour tool list holds the most used tools. Every other tool is in the index below: run it with \`p2a_run\` (its name and its parameters as \`args\`), and call \`p2a_help\` with its name first when its parameters are not obvious. Never say a method is unavailable before checking the index or \`p2a_help\`.\n`

- [ ] **Step 1: Write the failing tests**

```rust
// llm/tools.rs
#[test] fn tier1_prompt_is_unchanged() { assert_eq!(get_system_prompt_for(LlmToolMode::Tier1, Some("d")), get_system_prompt_with_context(Some("d"))); }
#[test] fn index_prompt_puts_the_index_before_the_datasets() {
    let p = get_system_prompt_for(LlmToolMode::Index, Some("### Dataset \"d\""));
    let (i, d) = (p.find("## Tool index").unwrap(), p.find("<<<DATASETS").unwrap());
    assert!(i < d && p.ends_with(LANGUAGE_RULE) && p.contains("p2a_run"));
}
#[test] fn every_tool_named_in_the_prompt_is_reachable() // generalise the "is named in the prompt" check: each backticked tool name in the base prompt is in TIER1_TOOLS and (CORE_TOOLS or the index)
// server.rs
#[test] fn index_mode_offers_core_and_meta_tools() // names == CORE_TOOLS ∪ {p2a_help, p2a_run}
#[test] fn tier1_mode_offers_todays_list() // names == TIER1_TOOLS ∩ registered
#[test] fn index_mode_sends_far_fewer_tool_bytes() // serde_json size of Index tools + index_text < 40% of Tier1 tools' size
```

- [ ] **Step 2: Run, expect FAIL** — filters `llm::tools`, `server::tests`.

- [ ] **Step 3: Implement** the Produces items; `build_app` calls `with_llm_tool_mode(config.llm_tools)`; `build_chat_messages` (`transport/llm_routes.rs:1073`) calls `get_system_prompt_for(server.llm_tool_mode(), …)` (pass the mode in as a parameter). DEPLOYMENT.md: one row for `P2A_LLM_TOOLS` (`index` default; `tier1` reproduces the paper's evaluation setup).

- [ ] **Step 4: Run, expect PASS** — same filters, then `--test tool_examples_doc` (counts unchanged).

- [ ] **Step 5: Commit** — `feat(mcp): P2A_LLM_TOOLS selects the chat's tool index or today's list`.

### Task 4: Chat routes — admit the chat-only tools, answer `p2a_help`

**Files:**
- Modify: `crates/p2a-mcp/src/transport/llm_routes.rs` (`admit_tool_call`/`refuse_unless_callable`, both `ToolExecutor` impls, tests)

**Interfaces:**
- Consumes: `is_chat_only`, `help`, `P2A_HELP`, `P2A_RUN` (Task 1); `llm_callable_tools` (existing).
- Produces: in both executors, after admission: `P2A_HELP` → `help(&arguments, &|n| llm_callable_tools(server).contains(n) && !INTERNAL_TOOLS.contains(&n))` instead of `governor.call`, with an audit entry (origin `llm_tool`, sanitised arguments, outcome `ok` or `error`) and, in the streaming executor, the usual `ToolStart`/`ToolEnd` events and DB row; `P2A_RUN` (only malformed ones reach here) → `Err("p2a_run needs {\"tool\": \"<name>\", \"args\": {...}}: …")` naming what was wrong.

- [ ] **Step 1: Write the failing tests** (existing `governed_parts()` gives a governor that refuses every call as low on memory, which proves `p2a_help` bypasses it)

```rust
#[tokio::test] async fn p2a_help_answers_without_the_governor() // SessionToolExecutor.execute("p2a_help", {"tool":"cox_ph"}) is Ok and lists parameters
#[tokio::test] async fn streaming_p2a_help_emits_tool_events()  // ToolStart{tool:"p2a_help"} then ToolEnd with the help text
#[tokio::test] async fn malformed_p2a_run_is_a_shape_error()    // execute("p2a_run", {"args": 3}) → Err containing "p2a_run needs"
#[tokio::test] async fn internal_tool_through_p2a_run_is_refused_like_a_direct_call() // execute("cleaning_session_start", {}) error text equals the direct-call refusal (the loop has unwrapped it)
#[tokio::test] async fn long_tail_tool_called_directly_still_runs() // execute("sar_model", ...) reaches the governor (error "low on memory", not "Unknown tool")
```

- [ ] **Step 2: Run, expect FAIL** — filter `transport::llm_routes`.

- [ ] **Step 3: Implement**: `refuse_unless_callable` returns `Ok` for `is_chat_only` names; the executors branch on the name after `admit_tool_call`.

- [ ] **Step 4: Run, expect PASS** — `transport::llm_routes` and `llm::`; then `cargo test -q -p p2a-mcp --features full --target-dir target/plain --lib -- transport::llm_routes` (db-gated paths).

- [ ] **Step 5: Commit** — `feat(mcp): the chat routes answer p2a_help and refuse malformed p2a_run calls`.

### Task 5: Documentation, token comparison, browser check

**Files:**
- Modify: `CLAUDE.md` (Router Composition paragraph: the chat reaches every non-internal tool through the index; the paper-dependency row for the system prompt names `P2A_LLM_TOOLS`), `API_ENDPOINTS.md` only if an event or field changed (expected: none).

- [ ] **Step 1:** Measure on the gazelle test backend after a rebuild/restart (backend log `backend24.log`): characters / 4 of the tool definitions plus system prompt per call in both modes; record both numbers for the PR body.
- [ ] **Step 2:** Browser check against `http://127.0.0.1:8099` (index mode): a Kaplan-Meier + Cox request on the uploaded `lung` dataset runs `kaplan_meier`/`cox_ph` (tool cards show real names); a SAR request on a dataset with coordinates finds `sar_model` via the index; a plain OLS request still uses native `regression_ols` in one round.
- [ ] **Step 3:** Run the CI-feature p2a-mcp test suite and clippy (Global Constraints); expect all pass, no new warnings.
- [ ] **Step 4: Commit** — `docs: the chat's tool index and P2A_LLM_TOOLS`.
