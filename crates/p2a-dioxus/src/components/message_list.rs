//! MessageList component for displaying all chat messages

use dioxus::prelude::*;

use crate::components::{Message, P2aWordmark};
use crate::state::{ChatMessage, ChatState};

/// What the auto-scroll watches: the message count and the last message's
/// size (text, plots, tool calls and finished calls).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScrollKey {
    pub len: usize,
    pub last_len: usize,
    pub last_images: usize,
    pub last_tools: usize,
    pub last_finished: usize,
}

impl ScrollKey {
    /// The key of a transcript (constant time).
    pub fn of(messages: &[ChatMessage]) -> Self {
        let last = messages.last();
        Self {
            len: messages.len(),
            last_len: last.map_or(0, |m| m.content.len()),
            last_images: last.map_or(0, |m| m.images.len()),
            last_tools: last.map_or(0, |m| m.tool_calls.len()),
            last_finished: last.map_or(0, |m| {
                m.tool_calls.iter().filter(|tc| tc.is_finished()).count()
            }),
        }
    }
}

/// How to scroll after the transcript changed from `prev` to `next`: to a
/// new message smoothly, whatever the position (the user sent, or a reply
/// started); along a growing last message at once, and only while the view
/// is at the bottom; otherwise not.
pub fn autoscroll(
    prev: Option<ScrollKey>,
    next: ScrollKey,
    at_bottom: bool,
) -> Option<ScrollBehavior> {
    match prev {
        None => Some(ScrollBehavior::Smooth),
        Some(p) if p.len != next.len => Some(ScrollBehavior::Smooth),
        Some(p) if p != next && at_bottom => Some(ScrollBehavior::Instant),
        Some(_) => None,
    }
}

/// Whether a scroll position is within 48 px of the bottom.
pub fn is_near_bottom(scroll_top: f64, scroll_height: i32, client_height: i32) -> bool {
    f64::from(scroll_height) - f64::from(client_height) - scroll_top <= 48.0
}

/// Whether the list follows new text after a scroll from `last_top` to
/// `top`: near the bottom it does; a scroll up stops it; a scroll down that
/// has not reached the bottom (a smooth scroll still running) keeps `was`.
pub fn next_at_bottom(
    was: bool,
    last_top: f64,
    top: f64,
    scroll_height: i32,
    client_height: i32,
) -> bool {
    if is_near_bottom(top, scroll_height, client_height) {
        true
    } else if top < last_top {
        false
    } else {
        was
    }
}

/// Props for MessageList
#[derive(Props, Clone, PartialEq)]
pub struct MessageListProps {
    /// Callback when a quick suggestion is clicked
    #[props(default)]
    pub on_suggestion: Option<EventHandler<String>>,
    /// Whether the suggestion chips are disabled (a send runs)
    #[props(default)]
    pub suggestions_disabled: bool,
    /// Whether the scroll container's view is at the bottom (`None`: follow
    /// always)
    #[props(default)]
    pub at_bottom: Option<Signal<bool>>,
}

/// MessageList component - displays all messages with auto-scroll
#[component]
pub fn MessageList(props: MessageListProps) -> Element {
    // Get chat state from context
    let chat_state = use_context::<Signal<ChatState>>();

    // Reference for scrolling using Dioxus's platform-agnostic MountedData
    let mut scroll_element: Signal<Option<std::rc::Rc<MountedData>>> = use_signal(|| None);

    // Follow the transcript (autoscroll): a new message, or a growing last
    // message while the view is at the bottom. Runs once per render that
    // changed the transcript; the key is constant-time.
    let mut prev_key = use_signal(|| None::<ScrollKey>);
    let at_bottom = props.at_bottom;
    use_effect(move || {
        let key = ScrollKey::of(&chat_state.read().messages);
        let Some(mounted) = scroll_element.read().clone() else {
            return;
        };
        let follow = at_bottom.is_none_or(|s| *s.peek());
        let prev = *prev_key.peek();
        if prev != Some(key) {
            prev_key.set(Some(key));
        }
        if let Some(behavior) = autoscroll(prev, key, follow) {
            spawn(async move {
                let _ = mounted.scroll_to(behavior).await;
            });
        }
    });

    let messages = chat_state.read().messages.clone();

    rsx! {
        div { class: "py-6 h-full",
            if messages.is_empty() {
                // Empty state: centered wordmark + a row of greetings in one size.
                // Previous layout had "Grüezi" as an oversized headline with the
                // other languages demoted below it; it now sits inline with the
                // rest so no one language feels singled out.
                div { class: "flex flex-col items-center justify-center min-h-[400px] h-full text-center mx-auto",
                    // Wordmark
                    div { class: "mb-20",
                        P2aWordmark { width: 280.0 }
                    }
                    // Multilingual welcome — all greetings at equal weight.
                    div { class: "mb-4 flex flex-wrap justify-center gap-x-3 gap-y-1 text-base text-gray-700 dark:text-gray-300 max-w-md",
                        span { "Welcome" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Bienvenue" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Willkommen" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Grüezi" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "Benvenuto" }
                        span { class: "text-gray-400 dark:text-gray-600", "·" }
                        span { "欢迎" }
                    }
                    p { class: "text-sm text-gray-500 dark:text-gray-400 max-w-sm",
                        "Start a conversation to get help with your data analysis."
                    }

                    // Quick suggestions
                    div { class: "mt-6 flex flex-wrap gap-2 justify-center",
                        QuickSuggestion {
                            text: "Create a sample dataset with x and y",
                            disabled: props.suggestions_disabled,
                            on_click: props.on_suggestion
                        }
                        QuickSuggestion {
                            text: "Explain OLS regression",
                            disabled: props.suggestions_disabled,
                            on_click: props.on_suggestion
                        }
                        QuickSuggestion {
                            text: "What tools are available?",
                            disabled: props.suggestions_disabled,
                            on_click: props.on_suggestion
                        }
                    }
                }
            } else {
                // Message list
                div { class: "space-y-2 px-4",
                    for message in messages.iter() {
                        Message {
                            key: "{message.id}",
                            message: message.clone()
                        }
                    }

                    // Scroll anchor - use platform-agnostic onmounted
                    div {
                        onmounted: move |event| {
                            scroll_element.set(Some(event.data().clone()));
                        }
                    }
                }
            }
        }
    }
}

/// Props for QuickSuggestion
#[derive(Props, Clone, PartialEq)]
struct QuickSuggestionProps {
    text: String,
    /// Greyed out and deaf to clicks (a send runs)
    #[props(default)]
    disabled: bool,
    #[props(default)]
    on_click: Option<EventHandler<String>>,
}

/// Quick suggestion chip
#[component]
fn QuickSuggestion(props: QuickSuggestionProps) -> Element {
    let text = props.text.clone();
    let text_for_click = props.text.clone();

    rsx! {
        button {
            class: "px-3 py-1.5 text-sm text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-full cursor-pointer shadow-sm transition-all duration-150 hover:border-orange-500 hover:bg-orange-50 hover:text-orange-700 hover:shadow-md dark:hover:bg-orange-900/30 dark:hover:text-orange-300 active:scale-95 active:shadow-none active:bg-orange-100 dark:active:bg-orange-900/50 disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none",
            disabled: props.disabled,
            onclick: move |_| {
                if props.disabled {
                    return;
                }
                if let Some(ref handler) = props.on_click {
                    handler.call(text_for_click.clone());
                }
            },
            "{text}"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(len: usize, last_len: usize, last_images: usize, last_tools: usize) -> ScrollKey {
        ScrollKey {
            len,
            last_len,
            last_images,
            last_tools,
            last_finished: 0,
        }
    }

    #[test]
    fn autoscroll_follows_new_messages_and_growth_at_the_bottom() {
        assert_eq!(
            autoscroll(None, key(2, 0, 0, 0), false),
            Some(ScrollBehavior::Smooth)
        );
        // A new message (the user sent, a reply started): always.
        assert_eq!(
            autoscroll(Some(key(2, 5, 0, 0)), key(3, 0, 0, 0), false),
            Some(ScrollBehavior::Smooth)
        );
        // The last message grew: only while the view is at the bottom.
        assert_eq!(
            autoscroll(Some(key(3, 5, 0, 0)), key(3, 105, 0, 0), true),
            Some(ScrollBehavior::Instant)
        );
        assert_eq!(
            autoscroll(Some(key(3, 5, 0, 0)), key(3, 105, 0, 0), false),
            None
        );
        assert_eq!(
            autoscroll(Some(key(3, 5, 0, 0)), key(3, 5, 1, 0), true),
            Some(ScrollBehavior::Instant)
        );
        let mut finished = key(3, 5, 0, 1);
        finished.last_finished = 1;
        assert_eq!(
            autoscroll(Some(key(3, 5, 0, 1)), finished, true),
            Some(ScrollBehavior::Instant)
        );
        // Nothing changed.
        assert_eq!(
            autoscroll(Some(key(3, 5, 0, 0)), key(3, 5, 0, 0), true),
            None
        );
    }

    #[test]
    fn near_bottom_is_within_48_px() {
        assert!(is_near_bottom(952.0, 1500, 500));
        assert!(!is_near_bottom(800.0, 1500, 500));
        assert!(is_near_bottom(0.0, 400, 500), "no scrollbar");
    }

    #[test]
    fn following_stops_when_the_user_scrolls_up() {
        // Scrolled up away from the bottom: stop following.
        assert!(!next_at_bottom(true, 900.0, 600.0, 1500, 500));
        // A scroll down that is not yet at the bottom (a smooth scroll
        // running) keeps the state.
        assert!(next_at_bottom(true, 600.0, 700.0, 1500, 500));
        assert!(!next_at_bottom(false, 600.0, 700.0, 1500, 500));
        // Back at the bottom: follow again.
        assert!(next_at_bottom(false, 700.0, 960.0, 1500, 500));
    }
}
