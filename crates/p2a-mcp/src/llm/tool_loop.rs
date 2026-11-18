//! The one tool-calling loop of the three LLM providers (LLM-20).
//!
//! A provider implements [`ToolLoopProvider`]: [`ToolLoopProvider::call`] makes one model call (its
//! own conversion of the conversation into wire messages included), streaming text deltas when a
//! sink is given. [`run_tool_loop`] does the rest: the iteration cap, loop detection, tool
//! execution, `interpret`, and tool-call ids unique per request (`call_<request>_<n>`). A call is
//! announced on the stream ([`StreamChunk::ToolCall`]) only when it is about to run.

use std::collections::{HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};

use async_trait::async_trait;

use super::context::{push_tool_results_for_model, truncate_str};
use super::provider::effective_max_tool_iterations;
use super::{
    LlmError, Message, MessageRole, ProviderConfig, StreamChunk, ToolCall, ToolDefinition,
    ToolExecutor, ToolResult,
};

/// Where streamed chunks go (the chat routes' callback).
pub type StreamSink = dyn Fn(StreamChunk) + Send + Sync;

/// One model turn as a provider returned it. Tool-call ids are the provider's (Ollama: empty); the
/// loop replaces them.
#[derive(Debug, Clone, Default)]
pub struct AssistantTurn {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// A provider's part of the tool loop: message conversion and one model call.
// The attribute async_trait expands to, on a boxed future.
#[allow(clippy::double_must_use)]
#[async_trait]
pub trait ToolLoopProvider: Send + Sync {
    /// One model turn. With `stream`, each text delta goes to the sink as [`StreamChunk::Text`] as it
    /// arrives; nothing else is sent to it.
    async fn call(
        &self,
        messages: &[Message],
        tools: &[ToolDefinition],
        stream: Option<&StreamSink>,
    ) -> Result<AssistantTurn, LlmError>;
}

/// The settings of one run of the loop.
#[derive(Debug, Clone)]
pub struct ToolLoopConfig {
    /// Model calls allowed (at least 1 is made).
    pub max_iterations: usize,
    /// `false`: return the first round's calls and results without calling the model again.
    pub interpret: bool,
    /// Cap on the tool-result text re-sent to the model (`llm_history_bytes`).
    pub loop_tool_bytes: usize,
    /// The `<request>` part of the tool-call ids.
    pub request_id: String,
}

impl ToolLoopConfig {
    /// The loop of one chat request of a provider: its `max_tool_iterations` clamped to
    /// `iterations_cap` (`llm_max_tool_iterations`) and a fresh request id.
    pub fn for_provider(
        config: &ProviderConfig,
        iterations_cap: usize,
        loop_tool_bytes: usize,
        interpret: bool,
    ) -> Self {
        Self {
            max_iterations: effective_max_tool_iterations(
                None,
                config.max_tool_iterations,
                iterations_cap,
            ),
            interpret,
            loop_tool_bytes,
            request_id: new_request_id(),
        }
    }
}

/// A request id: 12 lowercase hex digits (the top 48 bits of a v4 UUID, all random). With the call
/// number, a tool-call id stays within OpenAI's 40-character limit and Anthropic's `[A-Za-z0-9_-]`.
pub fn new_request_id() -> String {
    format!("{:012x}", uuid::Uuid::new_v4().as_u128() >> 80)
}

/// The id of the request's `n`-th tool call (from 1).
pub fn tool_call_id(request_id: &str, n: usize) -> String {
    format!("call_{request_id}_{n}")
}

/// Rounds of one request in which one tool name may appear: a round that would be the tool's
/// fourth stops the loop before it runs. Calls of one tool in the same round (parallel calls)
/// count once.
pub const MAX_ROUNDS_PER_TOOL: usize = 3;
/// Tool calls run in one round. The calls past it get an error result without running, so the
/// model can make them in the next round.
pub const MAX_CALLS_PER_ROUND: usize = 16;
/// The answer when the loop reaches the iteration cap and the model wrote no text.
pub const MAX_ITERATIONS_NOTE: &str = "Analysis stopped at the limit of model calls for one \
     request before the model wrote an answer. Ask again with a narrower request.";
/// The answer when a round repeats the previous one exactly and the model wrote no text.
pub const REPEATED_CALLS_NOTE: &str = "Analysis detected a repeated tool call pattern and stopped. Please try rephrasing your request.";

/// The result of a call past [`MAX_CALLS_PER_ROUND`] in its round, which does not run.
fn over_round_cap_result() -> String {
    format!(
        "not run: a round may make at most {MAX_CALLS_PER_ROUND} tool calls; make the rest in \
         the next round"
    )
}

/// The answer when one tool appears in more than [`MAX_ROUNDS_PER_TOOL`] rounds and the model
/// wrote no text; the name is cut to 64 bytes on a char boundary.
pub fn repeated_tool_note(name: &str) -> String {
    format!(
        "Analysis stopped: tool '{}' was called in more than {MAX_ROUNDS_PER_TOOL} rounds. Ask \
         again with a narrower request.",
        truncate_str(name, 64)
    )
}

/// Why a round of tool calls is refused before it runs.
enum Stop {
    Repeat,
    Tool(String),
}

/// Loop detection: an exact repeat of the previous round (names and arguments), or a tool name
/// in more than [`MAX_ROUNDS_PER_TOOL`] rounds of the request. Each name counts once per round,
/// however often the round calls it, so parallel calls of one tool run.
#[derive(Default)]
struct LoopGuard {
    previous: Option<u64>,
    rounds: HashMap<String, usize>,
}

impl LoopGuard {
    fn check(&mut self, calls: &[ToolCall]) -> Option<Stop> {
        let mut hasher = DefaultHasher::new();
        for tc in calls {
            tc.name.hash(&mut hasher);
            tc.arguments.to_string().hash(&mut hasher);
        }
        let hash = hasher.finish();
        if self.previous == Some(hash) {
            return Some(Stop::Repeat);
        }
        self.previous = Some(hash);
        let mut seen: HashSet<&str> = HashSet::new();
        for tc in calls {
            if !seen.insert(tc.name.as_str()) {
                continue;
            }
            let n = self.rounds.entry(tc.name.clone()).or_insert(0);
            *n += 1;
            if *n > MAX_ROUNDS_PER_TOOL {
                return Some(Stop::Tool(tc.name.clone()));
            }
        }
        None
    }
}

/// Run the tool loop: call the model; while it asks for tools, run them (in order, announcing each
/// on the stream just before it runs; at most [`MAX_CALLS_PER_ROUND`] per round) and call it again
/// with the results, until it answers without a tool call, `interpret` is false (the first round's
/// calls and results are returned), the loop guard refuses a round, or `max_iterations` model
/// calls were made.
pub async fn run_tool_loop(
    provider: &dyn ToolLoopProvider,
    messages: &[Message],
    tools: &[ToolDefinition],
    executor: &dyn ToolExecutor,
    config: &ToolLoopConfig,
    stream: Option<&StreamSink>,
) -> Result<Message, LlmError> {
    let mut conversation = messages.to_vec();
    let mut guard = LoopGuard::default();
    let mut last_text = String::new();
    let mut issued = 0usize;
    for _ in 0..config.max_iterations.max(1) {
        let turn = provider.call(&conversation, tools, stream).await?;
        if !turn.content.is_empty() {
            last_text.clone_from(&turn.content);
        }
        if turn.tool_calls.is_empty() {
            return Ok(Message::assistant(turn.content));
        }
        if let Some(stop) = guard.check(&turn.tool_calls) {
            let note = match stop {
                Stop::Repeat => {
                    tracing::warn!(
                        "Loop detected: exact repeat of the previous tool calls, stopping"
                    );
                    REPEATED_CALLS_NOTE.to_string()
                }
                Stop::Tool(name) => {
                    tracing::warn!(tool = %truncate_str(&name, 64), "Loop detected: tool called too often, stopping");
                    repeated_tool_note(&name)
                }
            };
            return Ok(Message::assistant(if last_text.is_empty() {
                note
            } else {
                last_text
            }));
        }
        let calls: Vec<ToolCall> = turn
            .tool_calls
            .into_iter()
            .map(|tc| {
                issued += 1;
                ToolCall {
                    id: tool_call_id(&config.request_id, issued),
                    ..tc
                }
            })
            .collect();
        conversation.push(Message::assistant_with_tools(turn.content, calls.clone()));
        let mut results = Vec::with_capacity(calls.len());
        let mut outputs = Vec::with_capacity(calls.len());
        for (i, tc) in calls.iter().enumerate() {
            if i >= MAX_CALLS_PER_ROUND {
                // Not run and not announced; every call keeps a result, which the providers
                // require.
                let content = over_round_cap_result();
                outputs.push(format!("**{}**:\n{}", tc.name, content));
                results.push(ToolResult {
                    tool_call_id: tc.id.clone(),
                    content,
                    is_error: true,
                });
                continue;
            }
            if let Some(sink) = stream {
                sink(StreamChunk::ToolCall {
                    tool_call: tc.clone(),
                });
            }
            let outcome = executor.execute(&tc.name, tc.arguments.clone()).await;
            let is_error = outcome.is_err();
            let content = outcome.unwrap_or_else(|e| format!("Error: {e}"));
            outputs.push(format!("**{}**:\n{}", tc.name, content));
            let result = ToolResult {
                tool_call_id: tc.id.clone(),
                content,
                is_error,
            };
            if let Some(sink) = stream {
                sink(StreamChunk::ToolResult {
                    tool_result: result.clone(),
                });
            }
            results.push(result);
        }
        push_tool_results_for_model(&mut conversation, &results, config.loop_tool_bytes);
        if !config.interpret {
            tracing::info!(
                "interpret=false: returning the tool results without another model call"
            );
            return Ok(Message {
                role: MessageRole::Assistant,
                content: outputs.join("\n\n"),
                tool_calls: Some(calls),
                tool_results: Some(results),
            });
        }
    }
    tracing::warn!(
        max = config.max_iterations,
        "Tool loop reached its iteration cap"
    );
    Ok(Message::assistant(if last_text.is_empty() {
        MAX_ITERATIONS_NOTE.to_string()
    } else {
        last_text
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::sync::{Arc, Mutex};

    /// Answers scripted turns in order (the last repeats), streams each turn's text and records every
    /// conversation it is sent.
    struct Scripted {
        turns: Vec<AssistantTurn>,
        seen: Mutex<Vec<Vec<Message>>>,
    }
    impl Scripted {
        fn new(turns: Vec<AssistantTurn>) -> Self {
            Self {
                turns,
                seen: Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> usize {
            self.seen.lock().unwrap().len()
        }
    }
    #[async_trait]
    impl ToolLoopProvider for Scripted {
        async fn call(
            &self,
            messages: &[Message],
            _tools: &[ToolDefinition],
            stream: Option<&StreamSink>,
        ) -> Result<AssistantTurn, LlmError> {
            let n = {
                let mut seen = self.seen.lock().unwrap();
                seen.push(messages.to_vec());
                seen.len() - 1
            };
            let turn = self.turns[n.min(self.turns.len() - 1)].clone();
            if let Some(sink) = stream
                && !turn.content.is_empty()
            {
                sink(StreamChunk::Text {
                    content: turn.content.clone(),
                });
            }
            Ok(turn)
        }
    }
    fn turn(content: &str, calls: &[(&str, Value)]) -> AssistantTurn {
        AssistantTurn {
            content: content.into(),
            tool_calls: calls
                .iter()
                .map(|(n, a)| ToolCall {
                    id: "provider-id".into(),
                    name: (*n).into(),
                    arguments: a.clone(),
                })
                .collect(),
        }
    }
    /// Runs every tool and records its name; `boom` fails with "it broke".
    #[derive(Default)]
    struct Tools(Mutex<Vec<String>>);
    #[async_trait]
    impl ToolExecutor for Tools {
        async fn execute(&self, name: &str, _arguments: Value) -> Result<String, String> {
            self.0.lock().unwrap().push(name.to_string());
            if name == "boom" {
                Err("it broke".into())
            } else {
                Ok(format!("ok-{name}"))
            }
        }
    }
    fn config(max_iterations: usize, interpret: bool) -> ToolLoopConfig {
        ToolLoopConfig {
            max_iterations,
            interpret,
            loop_tool_bytes: 65_536,
            request_id: "req".into(),
        }
    }
    async fn run(p: &Scripted, tools: &Tools, cfg: &ToolLoopConfig) -> (Message, Vec<String>) {
        let chunks = Arc::new(Mutex::new(Vec::new()));
        let c = chunks.clone();
        let sink: Box<StreamSink> = Box::new(move |ch| {
            let label = match ch {
                StreamChunk::Text { content } => format!("text:{content}"),
                StreamChunk::ToolCall { tool_call } => {
                    format!("call:{}:{}", tool_call.id, tool_call.name)
                }
                StreamChunk::ToolResult { tool_result } => {
                    format!("result:{}", tool_result.tool_call_id)
                }
                StreamChunk::Done => "done".into(),
                StreamChunk::Error { message } => format!("error:{message}"),
            };
            c.lock().unwrap().push(label);
        });
        let m = run_tool_loop(
            p,
            &[Message::user("go")],
            &[],
            tools,
            cfg,
            Some(sink.as_ref()),
        )
        .await
        .unwrap();
        let labels = chunks.lock().unwrap().clone();
        (m, labels)
    }

    #[tokio::test]
    async fn ids_are_unique_within_the_request_and_replace_the_provider_s() {
        let p = Scripted::new(vec![
            turn("", &[("a", json!({})), ("b", json!({}))]),
            turn("", &[("c", json!({}))]),
            turn("done", &[]),
        ]);
        let tools = Tools::default();
        let (m, chunks) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(
            (
                m.content.as_str(),
                m.tool_calls.is_none(),
                m.tool_results.is_none()
            ),
            ("done", true, true)
        );
        assert_eq!(*tools.0.lock().unwrap(), ["a", "b", "c"]);
        assert_eq!(
            chunks,
            [
                "call:call_req_1:a",
                "result:call_req_1",
                "call:call_req_2:b",
                "result:call_req_2",
                "call:call_req_3:c",
                "result:call_req_3",
                "text:done"
            ]
        );
        let third: Vec<String> = p.seen.lock().unwrap()[2]
            .iter()
            .map(|m| match m.role {
                MessageRole::Assistant => format!(
                    "assistant:{}",
                    m.tool_calls
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|c| c.id.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                MessageRole::Tool => format!(
                    "tool:{}",
                    m.tool_results
                        .as_ref()
                        .unwrap()
                        .iter()
                        .map(|r| r.tool_call_id.as_str())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                other => other.to_string(),
            })
            .collect();
        assert_eq!(
            third,
            [
                "user",
                "assistant:call_req_1,call_req_2",
                "tool:call_req_1,call_req_2",
                "assistant:call_req_3",
                "tool:call_req_3"
            ]
        );
    }

    #[tokio::test]
    async fn the_iteration_cap_returns_the_note_or_the_last_text() {
        let script = |content: &str| {
            (1..=5)
                .map(|i| {
                    turn(
                        if i == 2 { content } else { "" },
                        &[(format!("t{i}").as_str(), json!({}))],
                    )
                })
                .collect::<Vec<_>>()
        };
        let p = Scripted::new(script(""));
        let tools = Tools::default();
        let (m, _) = run(&p, &tools, &config(2, true)).await;
        assert_eq!((p.calls(), tools.0.lock().unwrap().len()), (2, 2));
        assert_eq!(
            m.content,
            "Analysis stopped at the limit of model calls for one request before the model wrote \
             an answer. Ask again with a narrower request."
        );
        let p = Scripted::new(script("working"));
        let (m, _) = run(&p, &Tools::default(), &config(2, true)).await;
        assert_eq!(m.content, "working");
    }

    #[tokio::test]
    async fn an_exact_repeat_stops_before_it_runs_or_is_announced() {
        let p = Scripted::new(vec![
            turn("", &[("t", json!({"a": 1}))]),
            turn("", &[("t", json!({"a": 1}))]),
        ]);
        let tools = Tools::default();
        let (m, chunks) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(*tools.0.lock().unwrap(), ["t"]);
        assert_eq!(chunks.iter().filter(|c| c.starts_with("call:")).count(), 1);
        assert_eq!(m.content, REPEATED_CALLS_NOTE);
    }

    #[tokio::test]
    async fn the_fourth_round_of_one_tool_stops_the_loop() {
        let p = Scripted::new(
            (1..=4)
                .map(|i| turn("", &[("t", json!({"i": i}))]))
                .collect(),
        );
        let tools = Tools::default();
        let (m, _) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(tools.0.lock().unwrap().len(), MAX_ROUNDS_PER_TOOL);
        assert_eq!(
            m.content,
            "Analysis stopped: tool 't' was called in more than 3 rounds. Ask again with a \
             narrower request."
        );
    }

    /// LLM-10: four parallel calls of one tool in one round run, and the model's answer follows.
    #[tokio::test]
    async fn parallel_calls_of_one_tool_in_one_round_run() {
        let calls: Vec<(&str, Value)> = (0..4)
            .map(|i| ("describe_dataset", json!({"name": format!("d{i}")})))
            .collect();
        let p = Scripted::new(vec![turn("", &calls), turn("four datasets", &[])]);
        let tools = Tools::default();
        let (m, chunks) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(tools.0.lock().unwrap().len(), 4);
        assert_eq!(chunks.iter().filter(|c| c.starts_with("call:")).count(), 4);
        assert_eq!(m.content, "four datasets");
    }

    /// A round of 17 calls runs 16; the 17th gets an error result without running or being
    /// announced, and the loop goes on.
    #[tokio::test]
    async fn calls_past_the_round_cap_get_an_error_result_and_the_loop_goes_on() {
        let calls: Vec<(&str, Value)> = (0..17).map(|i| ("t", json!({"i": i}))).collect();
        let p = Scripted::new(vec![turn("", &calls), turn("done", &[])]);
        let tools = Tools::default();
        let (m, chunks) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(tools.0.lock().unwrap().len(), MAX_CALLS_PER_ROUND);
        assert_eq!(chunks.iter().filter(|c| c.starts_with("call:")).count(), 16);
        assert_eq!(m.content, "done");
        let second = &p.seen.lock().unwrap()[1];
        let results = second.last().unwrap().tool_results.as_ref().unwrap();
        assert_eq!(results.len(), 17);
        assert!(results[..16].iter().all(|r| !r.is_error));
        assert_eq!(
            (results[16].tool_call_id.as_str(), results[16].is_error),
            ("call_req_17", true)
        );
        assert_eq!(
            results[16].content,
            "not run: a round may make at most 16 tool calls; make the rest in the next round"
        );
    }

    #[tokio::test]
    async fn interpret_false_returns_the_calls_and_their_results() {
        let p = Scripted::new(vec![
            turn("", &[("a", json!({})), ("boom", json!({}))]),
            turn("never", &[]),
        ]);
        let (m, _) = run(&p, &Tools::default(), &config(25, false)).await;
        assert_eq!(p.calls(), 1);
        assert_eq!(m.content, "**a**:\nok-a\n\n**boom**:\nError: it broke");
        let ids: Vec<&str> = m
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(|c| c.id.as_str())
            .collect();
        assert_eq!(ids, ["call_req_1", "call_req_2"]);
        let results: Vec<(&str, &str, bool)> = m
            .tool_results
            .as_ref()
            .unwrap()
            .iter()
            .map(|r| (r.tool_call_id.as_str(), r.content.as_str(), r.is_error))
            .collect();
        assert_eq!(
            results,
            [
                ("call_req_1", "ok-a", false),
                ("call_req_2", "Error: it broke", true)
            ]
        );
    }

    #[tokio::test]
    async fn without_a_stream_the_loop_still_runs() {
        let p = Scripted::new(vec![turn("", &[("a", json!({}))]), turn("done", &[])]);
        let tools = Tools::default();
        let m = run_tool_loop(
            &p,
            &[Message::user("go")],
            &[],
            &tools,
            &config(25, true),
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            (m.content.as_str(), tools.0.lock().unwrap().len()),
            ("done", 1)
        );
    }

    #[test]
    fn request_ids_are_12_hex_digits_and_differ() {
        let (a, b) = (new_request_id(), new_request_id());
        assert!(
            a.len() == 12
                && a.bytes()
                    .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)),
            "{a}"
        );
        assert_ne!(a, b); // 48 random bits each
        assert_eq!(tool_call_id(&a, 7), format!("call_{a}_7"));
    }

    #[test]
    fn the_stop_note_cuts_a_long_tool_name_on_a_char_boundary() {
        // 'é' is 2 bytes: 64 bytes are 32 of them.
        let note = repeated_tool_note(&"é".repeat(100));
        assert!(
            note.contains(&"é".repeat(32)) && !note.contains(&"é".repeat(33)),
            "{note}"
        );
    }
}
