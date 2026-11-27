# Every tool reachable from the chat: a tool index, `p2a_help` and `p2a_run`

Date: 2026-11-27. Status: approved design.

## Why

The chat model sees only the 128 tools of `TIER1_TOOLS` (`crates/p2a-mcp/src/llm/tools.rs`),
the request cap of most LLM APIs. A tool outside the list still runs when the model names it,
but the model cannot know it exists, so whole categories are invisible: spatial (10 tools) and
mediation (2) today, survival (5) until commit `318a4133`. Asked for a Kaplan-Meier curve, the
model answered that no tool does survival analysis.

The list is also expensive. Measured on the local test backend from `/api/tools` (characters / 4):

| What is sent per model call | ≈ tokens |
|---|---|
| today: 128 tools with full schemas | 55,000 |
| all 272 tools with full schemas | 115,000 |
| names and descriptions of all 272, no schemas | 29,000 |

It is sent on every round of the tool loop. The goal is **full coverage, efficiently**: every
tool the chat may call is reachable, and a model call costs less than today, not more.

## Goals

- Every tool the assistant may call (all registered tools except `INTERNAL_TOOLS`) is
  discoverable and callable from the chat.
- A model call carries far fewer tool tokens than today's ≈ 55k.
- Tool names stay as they are everywhere outside the model's own messages: the SSE `tool_start` /
  `tool_end` events, the tool cards in the UI, the conversation DB rows, the audit log, the
  loop limits. The paper's evaluation (`run_evaluation.py`, scoring the `tool_start` name) keeps
  working unchanged.
- Today's behaviour stays available as a server setting, so the paper's evaluation can be rerun
  under the configuration it measured.

## Non-goals

- No change to `INTERNAL_TOOLS`. The cleaning, session and search tools stay out of the chat;
  exposing them is a separate decision.
- No change to MCP tool names, request structs or handlers, and nothing for stdio MCP clients,
  which already have `search_tools` / `tool_info`.
- No provider-specific tool search (Anthropic's deferred tools); the design must work the same
  with OpenRouter, OpenAI, Anthropic and Ollama.

## Anti-goals

- Any path that executes a program or shell command from model output. Tools run in-process,
  through `ToolGovernor`, as today.
- A second execution path that skips the governor, the request limits, the session data or the
  audit entry.
- Silent behaviour change for the paper: the evaluation must be able to pin the old mode.

## Approaches considered

- **A. Command tool plus index (chosen).** About 25 frequent tools stay native; two generic
  tools, `p2a_help` and `p2a_run`, reach the rest; the system prompt carries a one-line index of
  every callable tool. ≈ 10-15k tool and index tokens per call. A long-tail tool costs one extra
  round (look up, then run) unless the index line suffices. The provider no longer validates a
  long-tail call's arguments against its schema; the server does, as it already does for every
  call (`deny_unknown_fields`, typed deserialisation, errors returned to the model).
- **B. Load on demand.** A search tool adds the matching tools' full definitions to the tool list
  from the next round on. Keeps provider-side argument checking, but the list grows over a
  conversation, some providers handle a changing tool list poorly, and the extra round is the
  same as A's.
- **C. The `p2a` CLI behind a shell tool.** Rejected: the CLI covers about 145 analyses of 272
  tools, keeps its state in session files rather than the server's datasets (uploads and
  tool-made datasets would be invisible), runs outside the governor's deadline, memory admission,
  quotas and audit, and makes model output start processes on the hosted server.

The user chose A, having asked for full coverage "in an efficient way", and confirmed that
commands must run in the server's own environment.

## Design

### 1. Modes

A server setting selects how tools are offered to the chat model, decided in `config.rs` with the
other knobs (CLI flag plus `P2A_*` environment variable, documented in DEPLOYMENT.md's
configuration reference):

- `index` (new default): native core tools + `p2a_help` + `p2a_run` + the index in the prompt.
- `tier1`: exactly today's behaviour (the `TIER1_TOOLS` list, today's system prompt). The
  paper's evaluation sets this to reproduce its numbers.

### 2. Native core tools

A new constant list, about 25 tools, sent with full schemas as today: data management
(`load_dataset`, `create_dataset`, `list_datasets`, `describe_dataset`, `head_dataset`,
`export_dataset`), the most used estimators and tests (`regression_ols`, `panel_fixed_effects`,
`logit`, `hypothesis_t_test`, `compute_correlation`, `ts_arima_fit`, ...), the common plots and
munging steps. Chosen by how often tools are expected in the paper's single-turn prompts
(`paper/code/e2e_eval/test_cases.json`) and the system prompt's examples; every tool the system
prompt names must be native or in the index (the existing test at `tools.rs` that checks prompt
names against `TIER1_TOOLS` is generalised).

### 3. The index

Appended to the system prompt in `index` mode: every callable non-native tool, grouped by its
registry category (`tools/registry.rs`, `CATEGORY_TABLE`), one line each:

```
## Survival
kaplan_meier: Kaplan-Meier curves and median survival, optional strata
cox_ph: Cox proportional hazards, hazard ratios, robust or clustered SEs
```

The phrase is generated, not hand-written: the description's first sentence, cut on a char
boundary to about 80 characters (`p2a_text::ellipsize`). Target size: under 6k tokens for about
230 lines; a test fails above a set byte budget. Generated once (`OnceLock`), so the prompt
prefix is stable and providers' prompt caching applies.

### 4. `p2a_help`

`p2a_help({ "tool": "<name>" })` or `p2a_help({ "query": "<words>" })`.

- With `tool`: the tool's full description and its parameters in a compact form, one line per
  property (name, type, required or default, description), not the raw JSON schema. An internal
  or unknown name gets the same refusal text as a direct call.
- With `query`: the best matches from `search_tools_scored` (at most 8), restricted to callable
  tools, as index lines.

Answered in-process from the registry; it touches no dataset. It still produces `tool_start` /
`tool_end` events and a DB row under its own name, so the UI and history show the lookup.

### 5. `p2a_run`

`p2a_run({ "tool": "<name>", "args": { ... } })`. Its schema says that `args` are the tool's
parameters as `p2a_help` lists them.

The call is **unwrapped at admission**: before `admit_tool_call` in `llm_routes.rs`, a `p2a_run`
call becomes a call of `tool` with `args`, keeping the provider's tool-call id. Everything after
admission (the `check_llm_tool_call` refusals, the audit entry, `ToolStart`, the DB row, the
governor, `MAX_ROUNDS_PER_TOOL` and `MAX_CALLS_PER_ROUND`, the `tool_end` status) sees the real
tool name. A malformed `p2a_run` call (no `tool`, `args` not an object) is a tool error naming the
expected shape. Calling a native tool or `p2a_help` through `p2a_run` works.

### 6. History sent back to the provider

The provider must see its own call shape again: when the conversation history is converted for
a provider, a stored tool call whose name is not in the tool list of the current request is
rendered as a `p2a_run` call carrying that name and its arguments. This also keeps old
conversations, recorded in `tier1` mode, valid in `index` mode.

### 7. System prompt

In `index` mode, a short block before the index: call native tools directly; for any other tool,
call `p2a_run` with the name from the index; call `p2a_help` first when the parameters are not
obvious; never say a method is unavailable without checking the index or `p2a_help`. The
technical-vocabulary block stays. In `tier1` mode the prompt is byte-identical to today's.

### 8. Tests

- The index lists every callable non-native tool exactly once and no internal tool; its size is
  under the budget.
- `p2a_help` returns parameters for a long-tail tool (e.g. `sar_model`) and refuses internal names.
- A `p2a_run` call runs the named tool through the governor: the `tool_start` event, the DB row
  and the audit entry carry the real name; loop limits count the real name.
- History conversion renders a non-offered stored call as `p2a_run`.
- `tier1` mode produces today's tool list and system prompt unchanged.
- Token comparison recorded in the PR: tool-list plus prompt tokens per call in both modes.

### 9. Documentation and paper

DEPLOYMENT.md (the new knob), API_ENDPOINTS.md if an event changes (none should), CLAUDE.md's
Router Composition paragraph (the chat now reaches every callable tool), and the paper-dependency
row for the system prompt. The paper's evaluation runner should pass the `tier1` setting until
the evaluation is rerun on `index`; that is a paper-repo change made separately.

## Decisions from the builder check

Decided by the user:

- **The mode is a server setting only**, not a per-request field: no API change, and hosted
  clients cannot pick it. The paper's evaluation runner starts its own backend with `tier1` to
  reproduce its numbers.
- **Anthropic prompt caching is a later, separate change.** This work only keeps the prompt
  prefix stable (index before the dataset block) so caching can apply once it is turned on.

Calls made by the author (marked so the user can check them):

- **Chat-only tools.** `p2a_help` and `p2a_run` are not router (MCP) tools: no `CATEGORY_TABLE`
  row, no limits rules, no dispatch arm, no change to the tool count, nothing for stdio clients.
  `admit_tool_call` gets a branch for them before `refuse_unless_callable`. `p2a_run` is unwrapped
  there and the real tool then goes through the normal path. `p2a_help` is answered from the
  registry without the governor (it reads no dataset and runs no method) but gets an audit entry
  (origin `llm_tool`, its arguments, outcome `ok` or the refusal), `tool_start` / `tool_end` events
  and a DB row.
- **Loop limits for `p2a_help`.** It counts toward `MAX_CALLS_PER_ROUND` and
  `max_tool_iterations` like any call, but `MAX_ROUNDS_PER_TOOL` and the exact-repeat check key it
  by its target (`p2a_help:<tool>` or `p2a_help:<query>`), so looking up four different tools in
  four rounds does not stop the answer, while asking for the same help repeatedly still does.
- **Real names to every client.** The non-streaming response's `tool_calls`, the SSE events, the DB
  rows and the history the server returns all carry the real tool name. The rewrite to `p2a_run`
  (section 6) happens in one place: where messages are converted for a provider call, against the
  tool list of that call. It therefore covers calls inside the running loop, DB history and
  client-supplied history alike. The loop's `call_<request>_<n>` ids stay as they are; an unwrapped
  call keeps the id the loop gave the `p2a_run` call.
- **`tier1` reading an `index`-mode conversation.** Stored calls are sent as today: under their
  real names, with their results. `tier1` already lets a model call a non-offered registered tool,
  so such history is what today's server can produce too. Stored `p2a_help` calls are sent as
  plain text (the request and its answer) instead of a tool call, as `p2a_help` is not a tool in
  that mode.
- **Index placement.** The index goes before the per-request dataset block, so the prompt prefix
  up to the end of the index is the same on every request.

## Left to the builder

The exact native list within the rule of section 2; the names of the setting and its env var;
where the unwrapping lives (a function beside `admit_tool_call`, or in `tool_loop.rs` before
loop detection, as long as loop limits see the real name); the compact parameter format.
