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

use super::context::{json_str, push_tool_results_for_model, truncate_str};
use super::provider::effective_max_tool_iterations;
use super::tool_index::{P2A_HELP, P2A_RUN, help_field};
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
/// however often the round calls it, so parallel calls of one tool run; a name that appears only
/// past [`MAX_CALLS_PER_ROUND`] in its round, where its calls do not run, does not count.
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
        let mut seen: HashSet<String> = HashSet::new();
        for tc in calls.iter().take(MAX_CALLS_PER_ROUND) {
            let key = round_key(tc);
            if !seen.insert(key.clone()) {
                continue;
            }
            let n = self.rounds.entry(key).or_insert(0);
            *n += 1;
            if *n > MAX_ROUNDS_PER_TOOL {
                return Some(Stop::Tool(tc.name.clone()));
            }
        }
        None
    }
}

/// The name a call counts under for [`MAX_ROUNDS_PER_TOOL`]: its tool, except that a `p2a_help`
/// lookup counts per target, so looking up several tools in turn does not stop the answer.
fn round_key(tc: &ToolCall) -> String {
    if tc.name != P2A_HELP {
        return tc.name.clone();
    }
    // What `p2a_help` looks up: its `tool`, else its `query`, trimmed, an empty one absent.
    let target = help_field(&tc.arguments, "tool")
        .or_else(|| help_field(&tc.arguments, "query"))
        .unwrap_or("");
    format!("{P2A_HELP}:{target}")
}

/// A `p2a_run` call as a call of the tool it names (`args` missing or null: no arguments; `args`
/// as JSON text: parsed). A malformed one (no tool name, `args` not an object, parameters beside
/// `tool` instead of inside `args`) is returned unchanged; the executor answers
/// it with the expected shape. A `p2a_run` naming `p2a_run` is unwrapped until it names
/// another tool or is malformed.
fn unwrap_run(mut call: ToolCall) -> ToolCall {
    while call.name == P2A_RUN {
        let next = unwrap_once(call.clone());
        if next.name == P2A_RUN && next.arguments == call.arguments {
            return next;
        }
        call = next;
    }
    call
}

/// One level of [`unwrap_run`].
fn unwrap_once(call: ToolCall) -> ToolCall {
    if call.name != P2A_RUN {
        return call;
    }
    let tool = call
        .arguments
        .get("tool")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|t| !t.is_empty());
    // Parameters beside `tool` instead of inside `args` are not dropped: the call stays
    // malformed and the executor says where they go. `args` sent as JSON text is parsed.
    let flat = call
        .arguments
        .as_object()
        .is_some_and(|o| o.keys().any(|k| k != "tool" && k != "args"));
    let args = match call.arguments.get("args") {
        _ if flat => None,
        None | Some(serde_json::Value::Null) => Some(serde_json::json!({})),
        Some(v @ serde_json::Value::Object(_)) => Some(v.clone()),
        Some(serde_json::Value::String(text)) => serde_json::from_str::<serde_json::Value>(text)
            .ok()
            .filter(serde_json::Value::is_object),
        Some(_) => None,
    };
    match (tool, args) {
        (Some(tool), Some(arguments)) => ToolCall {
            id: call.id,
            name: tool.to_string(),
            arguments,
        },
        _ => call,
    }
}

/// Longest part of the question the language note quotes, in characters.
const NOTE_QUESTION_CHARS: usize = 200;

/// The note that goes with each round's tool results: the question the results answer, quoted,
/// and that the reply is in its language. Tool results reach Anthropic models as a user turn, so
/// after a tool call the latest "user" text had no language and an English question after German
/// turns was answered in German; a rule in the system prompt alone did not change that.
fn language_note(messages: &[Message]) -> Option<String> {
    let question = messages
        .iter()
        .rev()
        .find(|m| m.role == MessageRole::User && m.tool_results.is_none())?;
    let text = question.content.trim();
    if text.is_empty() {
        return None;
    }
    let quoted = json_str(&p2a_text::ellipsize(text, NOTE_QUESTION_CHARS));
    // English unless the question is plainly in another language. A model asked for "the
    // language of the question" guesses it from names and content words: gpt-4o answered the
    // English "In lalonde, estimate the effect of treat on re78 ..." and "summarize the dataset"
    // in Spanish.
    Some(if is_plainly_other_language(text) {
        format!(
            "Tool results for the question {quoted}. That question is not in English: reply in \
             its language."
        )
    } else {
        format!(
            "Tool results for the question {quoted}. Reply in English unless the user asked for \
             another language."
        )
    })
}

/// English function words that are not also common words of German, Dutch or French ("in",
/// "an", "is" and "on" are); one of them keeps a question English.
const ENGLISH_MARKERS: &[&str] = &[
    "the", "of", "with", "and", "what", "how", "does", "which", "between", "this", "that",
    "please", "can", "you", "my", "did", "were", "there", "than", "whether", "using", "from", "by",
    "should", "would", "could", "has", "have", "its", "into", "show", "each", "for", "are", "to",
    "it",
];

/// Common words of the other languages the app is used in (German, Spanish, French, Italian,
/// Portuguese, Dutch) that are not English words.
const OTHER_MARKERS: &[&str] = &[
    // German
    "der",
    "das",
    "und",
    "ist",
    "nicht",
    "wie",
    "viele",
    "mit",
    "von",
    "ein",
    "eine",
    "den",
    "dem",
    "des",
    "bitte",
    "zeige",
    "berechne",
    "zeilen",
    "spalten",
    "werte",
    "fehlende",
    "mittelwert",
    "auf",
    "aus",
    "bei",
    "nach",
    "welche",
    "gibt",
    "es",
    "mir",
    "ich",
    "oder",
    "auch",
    "noch",
    "einmal",
    "nochmal",
    "zeichne",
    "zeichnen",
    "erstelle",
    "zwischen",
    "gegen",
    // Spanish, Portuguese
    "el",
    "los",
    "las",
    "del",
    "por",
    "con",
    "una",
    "que",
    "sobre",
    "para",
    "filas",
    "columnas",
    "calcula",
    "muestra",
    "cual",
    "como",
    "os",
    "uma",
    "com",
    "da",
    "do",
    "otra",
    "vez",
    "dibuja",
    // French, Italian
    "le",
    "les",
    "est",
    "une",
    "sur",
    "dans",
    "avec",
    "quel",
    "quelle",
    "pour",
    "lignes",
    "colonnes",
    "moyenne",
    "calcule",
    "montre",
    "il",
    "gli",
    "della",
    "che",
    "sono",
    "encore",
    "dessine",
    // Dutch
    "het",
    "een",
    "van",
    "wat",
    "hoe",
    "niet",
    "zijn",
];

/// Whether a question is plainly not English: no English function word, and at least two
/// common words of another language, or one with a letter outside ASCII (ä, é, ñ). Names of
/// datasets and columns carry neither, so a question made of them stays English.
fn is_plainly_other_language(text: &str) -> bool {
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
        .collect();
    if words.iter().any(|w| ENGLISH_MARKERS.contains(w)) {
        return false;
    }
    let other: std::collections::HashSet<&str> = words
        .iter()
        .copied()
        .filter(|w| OTHER_MARKERS.contains(w))
        .collect();
    let accented = words.iter().any(|w| w.chars().any(|c| !c.is_ascii()));
    other.len() >= 2 || (other.len() == 1 && accented)
}

/// Puts the note on the newest tool-results message and takes it off older ones, so the model
/// reads it once, next to the results it is about to interpret.
fn attach_note(conversation: &mut [Message], note: Option<&str>) {
    let Some(note) = note else { return };
    let newest = conversation.len().saturating_sub(1);
    for (i, m) in conversation.iter_mut().enumerate() {
        if m.role == MessageRole::Tool {
            m.content = if i == newest {
                note.to_string()
            } else {
                String::new()
            };
        }
    }
}

/// The messages as a provider must see them.
///
/// Every message carries real tool names (the loop unwraps `p2a_run`, and history from the
/// database or the client is stored that way). For a request whose tool list does not offer a
/// stored call's tool: with `p2a_run` offered, the call is shown as `p2a_run` naming it; without,
/// a `p2a_help` call and its result become plain text (that tool does not exist in this request),
/// and any other call is left as it is, as a model may call a registered tool it was not offered.
pub fn for_provider(messages: &[Message], tools: &[ToolDefinition]) -> Vec<Message> {
    let offers = |n: &str| tools.iter().any(|t| t.name == n);
    let run_offered = offers(P2A_RUN);
    let help_as_text = !offers(P2A_HELP);
    let mut out = Vec::with_capacity(messages.len());
    let mut texted: HashSet<String> = HashSet::new();
    // Texted results already sent: a result can be repeated (the attached results of an
    // `interpret: false` turn and a tool message after it), and goes out once.
    let mut answered: HashSet<String> = HashSet::new();
    for m in messages {
        let mut m = m.clone();
        if let Some(calls) = m.tool_calls.take() {
            let mut kept = Vec::with_capacity(calls.len());
            for c in calls {
                if offers(&c.name) {
                    kept.push(c);
                } else if c.name == P2A_HELP && help_as_text {
                    let line = format!("(Looked up {} with p2a_help.)", c.arguments);
                    m.content = if m.content.is_empty() {
                        line
                    } else {
                        format!("{}\n{line}", m.content)
                    };
                    texted.insert(c.id);
                } else if run_offered {
                    kept.push(ToolCall {
                        arguments: serde_json::json!({"tool": c.name, "args": c.arguments}),
                        name: P2A_RUN.to_string(),
                        id: c.id,
                    });
                } else {
                    kept.push(c);
                }
            }
            m.tool_calls = (!kept.is_empty()).then_some(kept);
        }
        if !texted.is_empty()
            && let Some(results) = m.tool_results.take()
        {
            let (moved, kept): (Vec<ToolResult>, Vec<ToolResult>) = results
                .into_iter()
                .partition(|r| texted.contains(&r.tool_call_id));
            let mut text: Vec<String> = moved
                .iter()
                .filter(|r| answered.insert(r.tool_call_id.clone()))
                .map(|r| format!("p2a_help answered:\n{}", r.content))
                .collect();
            if kept.is_empty() {
                if m.role == MessageRole::Assistant {
                    // An attached turn: its text and other calls stay before the answers.
                    if m.tool_calls.is_some() || !m.content.is_empty() {
                        out.push(m);
                    }
                } else if !m.content.is_empty() {
                    // A results message's content is the round's language note.
                    text.push(m.content);
                }
                if !text.is_empty() {
                    out.push(Message::user(text.join("\n\n")));
                }
                continue;
            }
            m.tool_results = Some(kept);
            out.push(m);
            if !text.is_empty() {
                out.push(Message::user(text.join("\n\n")));
            }
            continue;
        }
        out.push(m);
    }
    out
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
    let note = language_note(messages);
    let mut guard = LoopGuard::default();
    let mut last_text = String::new();
    let mut issued = 0usize;
    for _ in 0..config.max_iterations.max(1) {
        let mut turn = provider
            .call(&for_provider(&conversation, tools), tools, stream)
            .await?;
        if tools.iter().any(|t| t.name == P2A_RUN) {
            turn.tool_calls = turn.tool_calls.into_iter().map(unwrap_run).collect();
        }
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
        attach_note(&mut conversation, note.as_deref());
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

    /// Records every call's name and arguments.
    #[derive(Default)]
    struct Recorded(Mutex<Vec<(String, Value)>>);
    #[async_trait]
    impl ToolExecutor for Recorded {
        async fn execute(&self, name: &str, arguments: Value) -> Result<String, String> {
            self.0.lock().unwrap().push((name.to_string(), arguments));
            Ok(format!("ok-{name}"))
        }
    }
    fn def(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.into(),
            description: String::new(),
            parameters: json!({"type": "object"}),
        }
    }
    /// The index-mode tool list of these tests.
    fn offered() -> Vec<ToolDefinition> {
        vec![def(P2A_RUN), def(P2A_HELP), def("regression_ols")]
    }
    async fn run_offered(p: &Scripted, defs: &[ToolDefinition]) -> (Message, Vec<(String, Value)>) {
        let exec = Recorded::default();
        let m = run_tool_loop(
            p,
            &[Message::user("go")],
            defs,
            &exec,
            &config(25, false),
            None,
        )
        .await
        .unwrap();
        let calls = exec.0.lock().unwrap().clone();
        (m, calls)
    }
    fn run_call(tool: &str, args: Value) -> (&'static str, Value) {
        (P2A_RUN, json!({"tool": tool, "args": args}))
    }

    #[tokio::test]
    async fn p2a_run_calls_run_the_named_tool() {
        let p = Scripted::new(vec![turn(
            "",
            &[run_call("sar_model", json!({"dataset": "d"}))],
        )]);
        let (m, calls) = run_offered(&p, &offered()).await;
        assert_eq!(calls, [("sar_model".to_string(), json!({"dataset": "d"}))]);
        assert_eq!(m.tool_calls.unwrap()[0].name, "sar_model");
    }

    #[tokio::test]
    async fn p2a_run_without_args_runs_with_an_empty_object() {
        let p = Scripted::new(vec![turn(
            "",
            &[(P2A_RUN, json!({"tool": "list_datasets"}))],
        )]);
        let (_, calls) = run_offered(&p, &offered()).await;
        assert_eq!(calls, [("list_datasets".to_string(), json!({}))]);
    }

    #[tokio::test]
    async fn malformed_p2a_run_reaches_the_executor_unchanged() {
        for args in [
            json!({"args": 3}),
            json!({"tool": "x", "args": 3}),
            json!({"tool": ""}),
        ] {
            let p = Scripted::new(vec![turn("", &[(P2A_RUN, args.clone())])]);
            let (_, calls) = run_offered(&p, &offered()).await;
            assert_eq!(calls, [(P2A_RUN.to_string(), args)]);
        }
    }

    /// Parameters given beside `tool` instead of inside `args` are not dropped: the call stays a
    /// malformed `p2a_run`, which the executor answers with the expected shape.
    #[tokio::test]
    async fn flat_p2a_run_is_not_unwrapped() {
        let args = json!({"tool": "sar_model", "dataset": "d"});
        let p = Scripted::new(vec![turn("", &[(P2A_RUN, args.clone())])]);
        let (_, calls) = run_offered(&p, &offered()).await;
        assert_eq!(calls, [(P2A_RUN.to_string(), args)]);
    }

    #[tokio::test]
    async fn p2a_run_args_as_json_text_are_parsed() {
        let p = Scripted::new(vec![turn(
            "",
            &[(
                P2A_RUN,
                json!({"tool": "sar_model", "args": "{\"dataset\": \"d\"}"}),
            )],
        )]);
        let (_, calls) = run_offered(&p, &offered()).await;
        assert_eq!(calls, [("sar_model".to_string(), json!({"dataset": "d"}))]);
    }

    #[tokio::test]
    async fn p2a_run_is_unwrapped_only_when_offered() {
        let p = Scripted::new(vec![turn("", &[run_call("sar_model", json!({}))])]);
        let (_, calls) = run_offered(&p, &[def("regression_ols")]).await;
        assert_eq!(calls[0].0, P2A_RUN);
    }

    #[tokio::test]
    async fn loop_limits_count_the_real_tool() {
        let p = Scripted::new(
            (1..=4)
                .map(|i| turn("", &[run_call("sar_model", json!({"i": i}))]))
                .collect(),
        );
        let exec = Recorded::default();
        let m = run_tool_loop(
            &p,
            &[Message::user("go")],
            &offered(),
            &exec,
            &config(25, true),
            None,
        )
        .await
        .unwrap();
        assert_eq!(exec.0.lock().unwrap().len(), MAX_ROUNDS_PER_TOOL);
        assert_eq!(m.content, repeated_tool_note("sar_model"));
    }

    #[tokio::test]
    async fn different_help_lookups_do_not_stop_the_loop() {
        let mut turns: Vec<AssistantTurn> = ["a", "b", "c", "d"]
            .iter()
            .map(|t| turn("", &[(P2A_HELP, json!({"tool": t}))]))
            .collect();
        turns.push(turn("done", &[]));
        let p = Scripted::new(turns);
        let exec = Recorded::default();
        let m = run_tool_loop(
            &p,
            &[Message::user("go")],
            &offered(),
            &exec,
            &config(25, true),
            None,
        )
        .await
        .unwrap();
        assert_eq!(exec.0.lock().unwrap().len(), 4);
        assert_eq!(m.content, "done");
    }

    /// A stored conversation: the user's question, one assistant turn calling `calls`, and the
    /// tool message with their results.
    fn stored(calls: &[(&str, Value)]) -> Vec<Message> {
        let tc: Vec<ToolCall> = calls
            .iter()
            .enumerate()
            .map(|(i, (n, a))| ToolCall {
                id: format!("c{i}"),
                name: (*n).into(),
                arguments: a.clone(),
            })
            .collect();
        let results: Vec<ToolResult> = tc
            .iter()
            .map(|c| ToolResult {
                tool_call_id: c.id.clone(),
                content: format!("result of {}", c.name),
                is_error: false,
            })
            .collect();
        let mut tool_msg = Message::assistant("");
        tool_msg.role = MessageRole::Tool;
        tool_msg.tool_results = Some(results);
        vec![
            Message::user("go"),
            Message::assistant_with_tools("", tc),
            tool_msg,
        ]
    }

    #[test]
    fn for_provider_wraps_calls_of_tools_not_offered() {
        let msgs = stored(&[
            ("sar_model", json!({"dataset": "d"})),
            ("regression_ols", json!({"y": "y"})),
        ]);
        let out = for_provider(&msgs, &offered());
        let calls = out[1].tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].name, P2A_RUN);
        assert_eq!(
            calls[0].arguments,
            json!({"tool": "sar_model", "args": {"dataset": "d"}})
        );
        assert_eq!(calls[0].id, "c0");
        assert_eq!(calls[1].name, "regression_ols");
        assert_eq!(
            serde_json::to_value(&out[2].tool_results).unwrap(),
            serde_json::to_value(&msgs[2].tool_results).unwrap()
        );
    }

    #[test]
    fn for_provider_turns_help_into_text_when_help_is_not_offered() {
        let msgs = stored(&[
            (P2A_HELP, json!({"tool": "cox_ph"})),
            ("regression_ols", json!({})),
        ]);
        let out = for_provider(&msgs, &[def("regression_ols")]);
        let names: Vec<&str> = out
            .iter()
            .flat_map(|m| m.tool_calls.iter().flatten().map(|c| c.name.as_str()))
            .collect();
        assert_eq!(names, ["regression_ols"]);
        let result_ids: Vec<&str> = out
            .iter()
            .flat_map(|m| {
                m.tool_results
                    .iter()
                    .flatten()
                    .map(|r| r.tool_call_id.as_str())
            })
            .collect();
        assert_eq!(result_ids, ["c1"]);
        let all_text: String = out.iter().map(|m| m.content.as_str()).collect();
        assert!(all_text.contains("result of p2a_help"), "{all_text}");
        assert!(all_text.contains("cox_ph"), "{all_text}");
    }

    fn help_call(id: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: P2A_HELP.into(),
            arguments: json!({"tool": "cox_ph"}),
        }
    }
    fn help_result(id: &str) -> ToolResult {
        ToolResult {
            tool_call_id: id.into(),
            content: "cox_ph: Cox model".into(),
            is_error: false,
        }
    }

    /// An `interpret: false` turn carries its calls and results on one assistant message; when
    /// all its calls were `p2a_help` and help is not offered, the turn stays, as text.
    #[test]
    fn for_provider_keeps_an_attached_turn_whose_calls_were_all_help() {
        let mut turn = Message::assistant_with_tools("Looking.", vec![help_call("h1")]);
        turn.tool_results = Some(vec![help_result("h1")]);
        let out = for_provider(&[Message::user("go"), turn], &[def("regression_ols")]);
        assert_eq!(out.len(), 3, "{out:?}");
        assert_eq!(out[1].role, MessageRole::Assistant);
        assert!(out[1].content.starts_with("Looking.") && out[1].content.contains("Looked up"));
        assert!(out[1].tool_calls.is_none() && out[1].tool_results.is_none());
        assert_eq!(out[2].role, MessageRole::User);
        assert!(out[2].content.contains("p2a_help answered:"));
    }

    #[test]
    fn for_provider_texts_a_help_answer_once() {
        let mut turn = Message::assistant_with_tools("", vec![help_call("h1")]);
        turn.tool_results = Some(vec![help_result("h1")]);
        let mut repeat = Message::assistant("");
        repeat.role = MessageRole::Tool;
        repeat.tool_results = Some(vec![help_result("h1")]);
        let out = for_provider(
            &[Message::user("go"), turn, repeat],
            &[def("regression_ols")],
        );
        let answers = out
            .iter()
            .filter(|m| m.content.contains("p2a_help answered:"))
            .count();
        assert_eq!(answers, 1, "{out:?}");
    }

    #[test]
    fn for_provider_keeps_the_language_note_of_texted_results() {
        let mut results = Message::assistant("Answer in English.");
        results.role = MessageRole::Tool;
        results.tool_results = Some(vec![help_result("h1")]);
        let msgs = [
            Message::user("go"),
            Message::assistant_with_tools("", vec![help_call("h1")]),
            results,
        ];
        let out = for_provider(&msgs, &[def("regression_ols")]);
        let last = out.last().unwrap();
        assert_eq!(last.role, MessageRole::User);
        assert!(
            last.content.contains("p2a_help answered:")
                && last.content.contains("Answer in English."),
            "{}",
            last.content
        );
    }

    /// The loop guard keys a lookup by what `p2a_help` looks up: trimmed, an empty `tool` absent.
    #[test]
    fn help_round_key_matches_what_help_looks_up() {
        let key = |a: Value| {
            round_key(&ToolCall {
                id: "x".into(),
                name: P2A_HELP.into(),
                arguments: a,
            })
        };
        assert_eq!(
            key(json!({"tool": "", "query": "kaplan"})),
            "p2a_help:kaplan"
        );
        assert_eq!(
            key(json!({"tool": "cox_ph "})),
            key(json!({"tool": "cox_ph"}))
        );
    }

    #[tokio::test]
    async fn nested_p2a_run_is_unwrapped_to_the_innermost_tool() {
        let nested =
            json!({"tool": P2A_RUN, "args": {"tool": "sar_model", "args": {"dataset": "d"}}});
        let p = Scripted::new(vec![turn("", &[(P2A_RUN, nested)])]);
        let (_, calls) = run_offered(&p, &offered()).await;
        assert_eq!(calls, [("sar_model".to_string(), json!({"dataset": "d"}))]);
    }

    #[test]
    fn for_provider_leaves_history_alone_in_tier1_with_no_help() {
        let msgs = stored(&[("sar_model", json!({}))]);
        let out = for_provider(&msgs, &[def("regression_ols")]);
        assert_eq!(
            serde_json::to_value(&out[1].tool_calls).unwrap(),
            serde_json::to_value(&msgs[1].tool_calls).unwrap()
        );
    }

    #[tokio::test]
    async fn provider_sees_wrapped_history_and_clients_see_real_names() {
        let p = Scripted::new(vec![
            turn("", &[run_call("sar_model", json!({"dataset": "d"}))]),
            turn("done", &[]),
        ]);
        let exec = Recorded::default();
        let m = run_tool_loop(
            &p,
            &[Message::user("go")],
            &offered(),
            &exec,
            &config(25, true),
            None,
        )
        .await
        .unwrap();
        assert_eq!(m.content, "done");
        let seen = p.seen.lock().unwrap();
        let call = &seen[1][1].tool_calls.as_ref().unwrap()[0];
        assert_eq!(call.name, P2A_RUN);
        assert_eq!(call.arguments["tool"], "sar_model");
        assert_eq!(exec.0.lock().unwrap()[0].0, "sar_model");
    }

    /// Each round's results carry the note quoting the question; an older round's note is
    /// taken off, and the user's own messages are left as they were.
    #[tokio::test]
    async fn the_newest_tool_results_carry_the_language_note() {
        let p = Scripted::new(vec![
            turn("", &[("a", json!({}))]),
            turn("", &[("b", json!({}))]),
            turn("done", &[]),
        ]);
        let tools = Tools::default();
        let (m, _) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(m.content, "done");
        let seen = p.seen.lock().unwrap();
        let note = "Tool results for the question \"go\". Reply in English unless the user asked \
                    for another language.";
        assert_eq!(seen[1].last().unwrap().role, MessageRole::Tool);
        assert_eq!(seen[1].last().unwrap().content, note);
        let tool_notes: Vec<&str> = seen[2]
            .iter()
            .filter(|m| m.role == MessageRole::Tool)
            .map(|m| m.content.as_str())
            .collect();
        assert_eq!(tool_notes, ["", note]);
        assert_eq!(seen[2][0].content, "go");
    }

    /// The note quotes at most 200 characters of the question, JSON-quoted, and is left out
    /// without a question.
    #[test]
    fn the_language_note_is_bounded_and_quoted() {
        let long = format!("{}\"\n{}", "é".repeat(150), "x".repeat(150));
        let note = language_note(&[Message::user(long)]).unwrap();
        assert!(note.contains("...\". "), "{note}");
        assert!(!note.contains('\n'));
        assert!(note.chars().count() < 300, "{}", note.chars().count());
        assert_eq!(language_note(&[Message::assistant("hi")]), None);
        assert_eq!(language_note(&[Message::user("  ")]), None);
    }

    /// English is the default: plain English questions, short ones and questions made of names
    /// keep it; only a question plainly in another language gets the note to answer in it.
    #[test]
    fn the_note_keeps_english_unless_the_question_is_plainly_another_language() {
        let english = [
            "In lalonde, estimate the effect of treat on re78 with propensity score matching on \
             age, educ, race, married, nodegree, re74 and re75.",
            "summarize the dataset",
            "Fit a logit of d on x and z.",
            "Add grp to that model and compare the two models by AIC.",
            "go",
            "lalonde re78 treat",
            "plot Größe against Alter",
            "describe grunfeld",
        ];
        for q in english {
            assert!(!is_plainly_other_language(q), "{q}");
            let note = language_note(&[Message::user(q)]).unwrap();
            assert!(
                note.contains("Reply in English unless the user asked"),
                "{q}: {note}"
            );
        }
        let other = [
            "Wie viele Zeilen hat der Datensatz?",
            "Anzahl Zeilen und fehlende Werte",
            "Estima el efecto de treat sobre re78 con emparejamiento por puntaje de propensión.",
            "Quel est l'effet de treat sur re78 dans lalonde ?",
            "Wat is het effect van treat op re78?",
            "Zeige mir die Größe",
            "Bitte die Heatmap noch einmal zeichnen.",
            "Dibuja otra vez el histograma",
            "Dessine encore le graphique",
        ];
        for q in other {
            assert!(is_plainly_other_language(q), "{q}");
            let note = language_note(&[Message::user(q)]).unwrap();
            assert!(
                note.contains("not in English: reply in its language"),
                "{q}: {note}"
            );
        }
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

    /// A tool named only past the round cap does not run in that round, so the round does not
    /// count toward its [`MAX_ROUNDS_PER_TOOL`]: after three such rounds it still runs.
    #[tokio::test]
    async fn a_call_past_the_round_cap_does_not_count_a_round_of_its_tool() {
        let capped_round = |r: usize| {
            let mut calls: Vec<(&str, Value)> = (0..MAX_CALLS_PER_ROUND)
                .map(|i| (["a", "b", "c"][r], json!({"i": i})))
                .collect();
            calls.push(("t", json!({"r": r})));
            turn("", &calls)
        };
        let mut turns: Vec<AssistantTurn> = (0..3).map(capped_round).collect();
        turns.push(turn("", &[("t", json!({"last": true}))]));
        turns.push(turn("done", &[]));
        let p = Scripted::new(turns);
        let tools = Tools::default();
        let (m, _) = run(&p, &tools, &config(25, true)).await;
        assert_eq!(m.content, "done");
        let ran = tools.0.lock().unwrap();
        assert_eq!(ran.len(), 3 * MAX_CALLS_PER_ROUND + 1);
        assert_eq!(ran.last().map(String::as_str), Some("t"));
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
