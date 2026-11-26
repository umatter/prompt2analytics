//! Keyboard shortcuts help modal

use dioxus::prelude::*;

/// Props for ShortcutsModal
#[derive(Props, Clone, PartialEq)]
pub struct ShortcutsModalProps {
    /// Whether the modal is open
    pub is_open: bool,
    /// Callback when modal should close
    pub on_close: EventHandler<()>,
}

/// Keyboard shortcuts help modal
#[component]
pub fn ShortcutsModal(props: ShortcutsModalProps) -> Element {
    crate::app::use_escape_close(props.is_open, props.on_close);
    if !props.is_open {
        return rsx! {};
    }

    rsx! {
        // Container for backdrop and modal
        div {
            class: "fixed inset-0 z-50 flex items-center justify-center p-4",

            // Backdrop
            div {
                class: "k-backdrop",
                onclick: move |_| props.on_close.call(()),
            }

            // Modal
            div {
                class: "k-modal max-w-md max-h-[80vh]",

                // Header
                div { class: "k-modal-head",
                    h2 { class: "k-modal-title", "Keyboard shortcuts" }
                    button {
                        class: "k-icon-btn",
                        title: "Close",
                        aria_label: "Close",
                        onclick: move |_| props.on_close.call(()),
                        svg {
                            class: "w-4 h-4",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "square",
                                stroke_width: "2",
                                d: "M6 18L18 6M6 6l12 12"
                            }
                        }
                    }
                }

                // Content
                div { class: "k-modal-body space-y-5",
                    // Global shortcuts
                    div {
                        h3 { class: "k-section-h", "Global" }
                        ShortcutRow { keys: "Ctrl + /", description: "Show keyboard shortcuts" }
                        ShortcutRow { keys: "?", description: "Show keyboard shortcuts (outside text fields)" }
                        ShortcutRow { keys: "Ctrl + K", description: "Focus chat input" }
                        ShortcutRow { keys: "Ctrl + N", description: "New conversation" }
                        ShortcutRow { keys: "Escape", description: "Close dialog" }
                    }

                    // Chat input shortcuts
                    div {
                        h3 { class: "k-section-h", "Chat input" }
                        ShortcutRow { keys: "Enter", description: "Send message" }
                        ShortcutRow { keys: "Shift + Enter", description: "New line" }
                        ShortcutRow { keys: "\u{2191} / \u{2193}", description: "Navigate prompt history" }
                        ShortcutRow { keys: "@", description: "Autocomplete datasets" }
                        ShortcutRow { keys: "/", description: "Autocomplete tools" }
                        ShortcutRow { keys: "Tab", description: "Accept autocomplete suggestion" }
                    }

                    // Dataset sidebar shortcuts
                    div {
                        h3 { class: "k-section-h", "Datasets" }
                        ShortcutRow { keys: "Click", description: "View dataset details" }
                    }
                }

                // Footer
                div { class: "px-5 py-2.5 k-rule-t",
                    p { class: "k-mono k-muted text-[11px]",
                        "Press "
                        kbd { class: "k-kbd", "Esc" }
                        " to close"
                    }
                }
            }
        }
    }
}

/// Props for ShortcutRow
#[derive(Props, Clone, PartialEq)]
struct ShortcutRowProps {
    /// Key combination to display
    keys: &'static str,
    /// Description of what the shortcut does
    description: &'static str,
}

/// A single shortcut row with key badges and description
#[component]
fn ShortcutRow(props: ShortcutRowProps) -> Element {
    rsx! {
        div { class: "k-row",
            span { "{props.description}" }
            div { class: "flex items-center gap-1 flex-shrink-0",
                // Split keys by " + " and render each as a kbd element
                for (i, key) in props.keys.split(" + ").enumerate() {
                    if i > 0 {
                        span { class: "k-muted k-mono text-[11px]", "+" }
                    }
                    kbd { key: "{key}", class: "k-kbd", "{key}" }
                }
            }
        }
    }
}
