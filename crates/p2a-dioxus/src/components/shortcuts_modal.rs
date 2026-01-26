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
    if !props.is_open {
        return rsx! {};
    }

    rsx! {
        // Container for backdrop and modal
        div {
            class: "fixed inset-0 z-50 flex items-center justify-center p-4",

            // Backdrop
            div {
                class: "absolute inset-0 bg-black/50 backdrop-blur-sm",
                onclick: move |_| props.on_close.call(()),
            }

            // Modal
            div {
                class: "relative bg-white dark:bg-gray-900 rounded-2xl shadow-2xl w-full max-w-md max-h-[80vh] overflow-hidden border border-gray-200 dark:border-gray-800",

                // Header
                div { class: "px-6 py-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between",
                    div { class: "flex items-center gap-2",
                        // Keyboard icon
                        svg {
                            class: "w-5 h-5 text-teal-600 dark:text-teal-400",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M12 6v6m0 0v6m0-6h6m-6 0H6"
                            }
                            rect {
                                x: "2",
                                y: "6",
                                width: "20",
                                height: "12",
                                rx: "2",
                                stroke_width: "2"
                            }
                        }
                        h2 { class: "text-xl font-semibold text-gray-900 dark:text-white", "Keyboard Shortcuts" }
                    }
                    button {
                        class: "p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors",
                        onclick: move |_| props.on_close.call(()),
                        svg {
                            class: "w-5 h-5",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M6 18L18 6M6 6l12 12"
                            }
                        }
                    }
                }

                // Content
                div { class: "px-6 py-4 overflow-y-auto",
                    // Global shortcuts
                    div { class: "mb-6",
                        h3 { class: "text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3",
                            "Global"
                        }
                        div { class: "space-y-2",
                            ShortcutRow { keys: "Ctrl + /", description: "Show keyboard shortcuts" }
                            ShortcutRow { keys: "?", description: "Show keyboard shortcuts (alternative)" }
                            ShortcutRow { keys: "Ctrl + K", description: "Focus chat input" }
                            ShortcutRow { keys: "Ctrl + N", description: "New conversation" }
                            ShortcutRow { keys: "Escape", description: "Close modal/dropdown" }
                        }
                    }

                    // Chat input shortcuts
                    div { class: "mb-6",
                        h3 { class: "text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3",
                            "Chat Input"
                        }
                        div { class: "space-y-2",
                            ShortcutRow { keys: "Enter", description: "Send message" }
                            ShortcutRow { keys: "Shift + Enter", description: "New line" }
                            ShortcutRow { keys: "\u{2191} / \u{2193}", description: "Navigate prompt history" }
                            ShortcutRow { keys: "@", description: "Autocomplete datasets" }
                            ShortcutRow { keys: "/", description: "Autocomplete tools" }
                            ShortcutRow { keys: "Tab", description: "Accept autocomplete suggestion" }
                        }
                    }

                    // Dataset sidebar shortcuts
                    div {
                        h3 { class: "text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-3",
                            "Datasets"
                        }
                        div { class: "space-y-2",
                            ShortcutRow { keys: "Click", description: "View dataset details" }
                        }
                    }
                }

                // Footer
                div { class: "px-6 py-3 border-t border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50",
                    p { class: "text-xs text-gray-500 dark:text-gray-400 text-center",
                        "Press "
                        kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-gray-600 dark:text-gray-300 font-mono text-xs",
                            "Esc"
                        }
                        " to close this panel"
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
        div { class: "flex items-center justify-between py-1.5",
            span { class: "text-sm text-gray-700 dark:text-gray-300",
                "{props.description}"
            }
            div { class: "flex items-center gap-1",
                // Split keys by " + " and render each as a kbd element
                for (i, key) in props.keys.split(" + ").enumerate() {
                    if i > 0 {
                        span { class: "text-gray-400 dark:text-gray-500 text-xs", "+" }
                    }
                    kbd {
                        key: "{key}",
                        class: "px-2 py-1 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded text-xs font-mono text-gray-700 dark:text-gray-300 min-w-[1.5rem] text-center",
                        "{key}"
                    }
                }
            }
        }
    }
}
