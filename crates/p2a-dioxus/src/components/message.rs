//! Message component for displaying chat messages

use dioxus::prelude::*;

use crate::state::chat::ChatMessage;
use crate::utils::render_markdown;

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
fn render_content(message: &ChatMessage) -> Element {
    if message.role == "user" {
        rsx! {
            p { class: "text-white whitespace-pre-wrap",
                "{message.content}"
            }
        }
    } else {
        let content = &message.content;
        if message.is_streaming {
            rsx! {
                div { class: "text-gray-800 dark:text-gray-200",
                    {render_markdown(content)}
                    span { class: "inline-block w-2 h-4 ml-0.5 bg-teal-500 animate-pulse" }
                }
            }
        } else {
            render_markdown(content)
        }
    }
}

/// Message component
#[component]
pub fn Message(props: MessageProps) -> Element {
    let message = &props.message;
    let is_user = message.role == "user";

    let timestamp = message.timestamp.format("%H:%M").to_string();

    if is_user {
        // User message - right aligned, teal background
        rsx! {
            div { class: "flex justify-end mb-4 message-enter",
                div { class: "max-w-[80%] rounded-2xl rounded-tr-sm px-4 py-3 bg-teal-600 text-white shadow-sm",
                    // Message content
                    {render_content(message)}

                    // Timestamp
                    div { class: "text-xs text-teal-200 mt-2 text-right",
                        "{timestamp}"
                    }
                }
            }
        }
    } else {
        // Assistant message - left aligned, gray background
        rsx! {
            div { class: "flex justify-start mb-4 message-enter",
                div { class: "max-w-[85%]",
                    // Tools Used indicator (orange for Rust/p2a-core branding)
                    if !message.tool_calls.is_empty() {
                        div { class: "mb-2 px-3 py-2 rounded-lg bg-orange-50 dark:bg-orange-900/30 border border-orange-200 dark:border-orange-700",
                            div { class: "flex items-center gap-2 flex-wrap",
                                // Rust gear icon
                                svg {
                                    class: "w-4 h-4 text-orange-600 dark:text-orange-400 flex-shrink-0",
                                    fill: "none",
                                    stroke: "currentColor",
                                    view_box: "0 0 24 24",
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                                    }
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                    }
                                }
                                span { class: "text-xs font-semibold text-orange-700 dark:text-orange-300",
                                    "Rust Analytics:"
                                }
                                // Tool chips
                                for tool_call in message.tool_calls.iter() {
                                    span {
                                        key: "{tool_call.id}",
                                        class: "inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-orange-100 dark:bg-orange-800 text-orange-800 dark:text-orange-200",
                                        "{tool_call.name}"
                                    }
                                }
                            }
                        }
                    }

                    // Main message bubble
                    div { class: "rounded-2xl rounded-tl-sm px-4 py-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700",
                        // Message content
                        {render_content(message)}

                        // An answer that ended before it was finished
                        if message.interrupted {
                            p { class: "mt-2 text-xs italic text-gray-500 dark:text-gray-400",
                                "Answer interrupted"
                            }
                        }

                        // Display images if any
                        if !message.images.is_empty() {
                            div { class: "mt-3 space-y-2",
                                for (idx , image) in message.images.iter().enumerate() {
                                    img {
                                        key: "{idx}",
                                        src: "data:image/png;base64,{image}",
                                        class: "max-w-full rounded-lg shadow-md",
                                        alt: "Visualization"
                                    }
                                }
                            }
                        }

                        // Timestamp
                        div { class: "text-xs text-gray-500 dark:text-gray-400 mt-2",
                            "{timestamp}"
                        }
                    }

                    // Tool call details (expandable cards) displayed below
                    if !message.tool_calls.is_empty() {
                        div { class: "mt-2 space-y-2",
                            for tool_call in message.tool_calls.iter() {
                                crate::components::ToolCallDisplay {
                                    key: "{tool_call.id}",
                                    id: tool_call.id.clone(),
                                    name: tool_call.name.clone(),
                                    arguments: tool_call.arguments.clone(),
                                    result: tool_call.result.clone(),
                                    success: tool_call.success,
                                    // A Done call without a result has ended too.
                                    finished: tool_call.is_finished() || !message.is_streaming
                                }
                            }
                        }
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
        };
        let first = ChatMessage::from_conversation_message_with_tools(&stored, &[call("success")]);
        let second = ChatMessage::from_conversation_message_with_tools(&stored, &[call("error")]);
        assert_ne!(first, second);
        // A clone is the same message and is not redrawn.
        assert_eq!(first, first.clone());
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
