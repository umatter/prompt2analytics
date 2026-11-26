//! Message component for displaying chat messages

use dioxus::prelude::*;

use crate::exhibit::ExhibitViewer;
use crate::state::chat::ChatMessage;
use crate::utils::render_answer;

/// Props for the Message component
#[derive(Props, Clone, PartialEq)]
pub struct MessageProps {
    /// The message to display
    pub message: ChatMessage,
}

/// Whether a message must be redrawn: its text, streaming and interrupted
/// state, its plot count, and its tool calls (a call added, ended or
/// replaced gives `tool_rev` a revision unique in the process).
impl PartialEq for ChatMessage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.is_streaming == other.is_streaming
            && self.interrupted == other.interrupted
            && self.tool_rev == other.tool_rev
            && self.images.len() == other.images.len()
            && self.tool_calls.len() == other.tool_calls.len()
            && self.content == other.content
    }
}

/// Render message content based on role
fn render_content(message: &ChatMessage, viewer: Option<ExhibitViewer>) -> Element {
    if message.role == "user" {
        rsx! {
            p { class: "k-q whitespace-pre-wrap",
                "{message.content}"
            }
        }
    } else {
        let content = &message.content;
        if message.is_streaming {
            rsx! {
                div { class: "k-answer",
                    {render_answer(content, viewer)}
                    span { class: "streaming-cursor", aria_hidden: "true" }
                }
            }
        } else {
            rsx! {
                div { class: "k-answer", {render_answer(content, viewer)} }
            }
        }
    }
}

/// Message component: one turn of the transcript, a narrow label column
/// ("you" or "p2a", with the time) beside the content
#[component]
pub fn Message(props: MessageProps) -> Element {
    let message = &props.message;
    let is_user = message.role == "user";
    let viewer = try_use_context::<ExhibitViewer>();

    let timestamp = crate::utils::local_clock(&message.timestamp);

    if is_user {
        // User message: the question, set large
        rsx! {
            div { class: "k-turn message-enter",
                div { class: "k-who me",
                    "you"
                    span { class: "k-when", "{timestamp}" }
                }
                div { class: "min-w-0",
                    {render_content(message, viewer)}
                }
            }
        }
    } else {
        let n_calls = message.tool_calls.len();
        // Assistant message: the tool-call tree, then the answer
        rsx! {
            div { class: "k-turn message-enter",
                div { class: "k-who",
                    "p2a"
                    span { class: "k-when", "{timestamp}" }
                }
                div { class: "k-body",
                    // Tool calls as a process tree, each line opening to its
                    // arguments and result
                    if n_calls > 0 {
                        div { class: "k-tree k-calls", role: "list", aria_label: "Tool calls",
                            for (i , tool_call) in message.tool_calls.iter().enumerate() {
                                div { key: "{tool_call.id}", role: "listitem",
                                    crate::components::ToolCallDisplay {
                                        id: tool_call.id.clone(),
                                        name: tool_call.name.clone(),
                                        arguments: tool_call.arguments.clone(),
                                        result: tool_call.result.clone(),
                                        success: tool_call.success,
                                        // A Done call without a result has ended too.
                                        finished: tool_call.is_finished() || !message.is_streaming,
                                        last: i + 1 == n_calls
                                    }
                                }
                            }
                        }
                    }

                    // Plots, made by the calls above and placed before the answer that
                    // refers to them; each under a rule with its caption and actions;
                    // a click opens the viewer
                    for (idx , image) in message.images.iter().enumerate() {
                        crate::components::FigureExhibit {
                            key: "{idx}",
                            image: image.clone(),
                            svg: message.svgs.get(idx).cloned().unwrap_or_default(),
                            label: format!("Figure {}", idx + 1),
                        }
                    }

                    // Message content
                    {render_content(message, viewer)}

                    // An answer that ended before it was finished
                    if message.interrupted {
                        p { class: "k-note", "Answer interrupted" }
                    }

                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::state::{ChatMessage, ChatState};

    /// The card of a call that ended or failed is redrawn although the text
    /// and the number of calls are unchanged.
    #[test]
    fn a_tool_call_status_change_redraws_the_message() {
        let mut chat = ChatState::new();
        chat.add_user_message("fit");
        chat.add_streaming_message();
        chat.append_content("Fitting.");
        chat.set_active_tool(
            "regression_ols".into(),
            serde_json::json!({"dataset": "sales"}),
        );
        let running = chat.messages[1].clone();
        assert_eq!(running, chat.messages[1].clone());
        chat.clear_active_tool(Some("Error: no column".into()), Some("error"));
        assert_ne!(running, chat.messages[1], "failed");

        // Calls loaded from the store replace the streamed ones.
        let before = chat.messages[1].clone();
        let calls = chat.messages[1].tool_calls.clone();
        chat.messages[1].set_tool_calls(calls);
        assert_ne!(before, chat.messages[1], "replaced");

        let mut interrupted = chat.messages[1].clone();
        let shown = interrupted.clone();
        interrupted.interrupted = true;
        assert_ne!(shown, interrupted);
    }

    /// A message rebuilt from the store keeps its id, text and number of
    /// calls; a call whose outcome changed between the two loads still
    /// redraws it.
    #[test]
    fn a_rebuilt_message_with_another_outcome_redraws() {
        use crate::api::{ConversationMessage, PersistedToolCall};
        let stored = ConversationMessage {
            id: "m-1".into(),
            conversation_id: "c-1".into(),
            role: "assistant".into(),
            content: "Fitted.".into(),
            created_at: "2026-11-18T10:00:00Z".into(),
            token_count: None,
            model: None,
        };
        let call = |status: &str| PersistedToolCall {
            id: "call_1".into(),
            message_id: "m-1".into(),
            conversation_id: "c-1".into(),
            tool_name: "regression_ols".into(),
            arguments: r#"{"dataset": "sales"}"#.into(),
            result: Some("OLS results".into()),
            error: None,
            status: status.into(),
            started_at: "2026-11-18T10:00:01Z".into(),
            completed_at: None,
            duration_ms: None,
            images: Vec::new(),
            image_svgs: Vec::new(),
        };
        let first = ChatMessage::from_conversation_message_with_tools(&stored, &[call("success")]);
        let second = ChatMessage::from_conversation_message_with_tools(&stored, &[call("error")]);
        assert_ne!(first, second);
        // A clone is the same message and is not redrawn.
        assert_eq!(first, first.clone());
    }

    /// A message reopened from the store shows the plots its calls stored,
    /// in the order of the calls.
    #[test]
    fn a_reopened_message_shows_its_stored_plots() {
        use crate::api::{ConversationMessage, PersistedToolCall};
        let stored = ConversationMessage {
            id: "m-1".into(),
            conversation_id: "c-1".into(),
            role: "assistant".into(),
            content: "Plotted.".into(),
            created_at: "2026-11-24T10:00:00Z".into(),
            token_count: None,
            model: None,
        };
        let call = |id: &str, images: &[&str]| PersistedToolCall {
            id: id.into(),
            message_id: "m-1".into(),
            conversation_id: "c-1".into(),
            tool_name: "viz_scatter".into(),
            arguments: "{}".into(),
            result: Some("ok".into()),
            error: None,
            status: "success".into(),
            started_at: "2026-11-24T10:00:01Z".into(),
            completed_at: None,
            duration_ms: None,
            images: images.iter().map(|s| s.to_string()).collect(),
            image_svgs: if id == "a" {
                vec!["<svg/>".into()]
            } else {
                Vec::new()
            },
        };
        let msg = ChatMessage::from_conversation_message_with_tools(
            &stored,
            &[
                call("a", &["AAA"]),
                call("b", &[]),
                call("c", &["CCC", "DDD"]),
            ],
        );
        assert_eq!(msg.images, ["AAA", "CCC", "DDD"]);
        assert_eq!(msg.svgs, ["<svg/>", "", ""]);
    }
}

#[cfg(test)]
mod timing {
    use super::*;
    use crate::state::ToolCallInfo;
    use std::time::Instant;

    /// The props comparison before the per-call fields.
    fn base_eq(a: &ChatMessage, b: &ChatMessage) -> bool {
        a.id == b.id
            && a.content == b.content
            && a.is_streaming == b.is_streaming
            && a.images.len() == b.images.len()
            && a.tool_calls.len() == b.tool_calls.len()
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// 3 warm-ups, then 31 interleaved runs; medians in ms.
    fn time_pair(mut base: impl FnMut() -> usize, mut head: impl FnMut() -> usize) -> (f64, f64) {
        for _ in 0..3 {
            std::hint::black_box(base());
            std::hint::black_box(head());
        }
        let (mut b, mut h) = (Vec::new(), Vec::new());
        for _ in 0..31 {
            let t = Instant::now();
            std::hint::black_box(base());
            b.push(t.elapsed().as_secs_f64() * 1e3);
            let t = Instant::now();
            std::hint::black_box(head());
            h.push(t.elapsed().as_secs_f64() * 1e3);
        }
        (median(b), median(h))
    }

    fn transcript(n: usize) -> Vec<ChatMessage> {
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    ChatMessage::user(&format!("question {i}: regress price on sqft"))
                } else {
                    let mut m = ChatMessage::assistant_streaming();
                    m.is_streaming = false;
                    m.content = format!("Answer {i}: ").repeat(40);
                    for k in 0..2 {
                        m.tool_calls.push(ToolCallInfo {
                            id: format!("call_{i}_{k}"),
                            name: "regression_ols".into(),
                            arguments: serde_json::json!({"dataset": "sales"}),
                            result: Some("OLS results".repeat(20)),
                            success: Some(true),
                            is_expanded: false,
                        });
                    }
                    m
                }
            })
            .collect()
    }

    #[test]
    #[ignore = "benchmark; times the message props comparison against the base"]
    fn time_props_comparison() {
        for n in [100, 1_000, 10_000] {
            let a = transcript(n);
            let b = a.clone();
            let (mb, mh) = time_pair(
                || a.iter().zip(&b).filter(|(x, y)| base_eq(x, y)).count(),
                || a.iter().zip(&b).filter(|(x, y)| x == y).count(),
            );
            println!(
                "TIMING props comparison {n} messages: base {mb:.4} ms, head {mh:.4} ms, head/base {:.3}",
                mh / mb
            );
        }
    }
}
