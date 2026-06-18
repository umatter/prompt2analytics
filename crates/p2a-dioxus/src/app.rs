//! Root application component

use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::api::api;
use crate::components::{ChatPanel, ConversationSidebar, DatasetSidebar, ShortcutsModal};
use crate::state::settings::Theme;
use crate::state::{ChatState, ConversationState, SessionState, Settings};

/// Apply theme to the document
/// Strategy:
/// - System: no classes, let media queries handle it
/// - Light: add "light" class to override media query dark styles
/// - Dark: add "dark" class to force dark mode
#[cfg(target_arch = "wasm32")]
pub fn apply_theme(theme: Theme) {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(html) = document.document_element() {
                let class_list = html.class_list();

                // Remove both classes first
                let _ = class_list.remove_1("dark");
                let _ = class_list.remove_1("light");

                match theme {
                    Theme::System => {
                        // No class - let media queries handle it
                        tracing::info!("Theme: System (no class, using media queries)");
                    }
                    Theme::Light => {
                        // Add light class to force light mode (overrides media query)
                        let _ = class_list.add_1("light");
                        tracing::info!("Theme: Light (added light class)");
                    }
                    Theme::Dark => {
                        // Add dark class to force dark mode
                        let _ = class_list.add_1("dark");
                        tracing::info!("Theme: Dark (added dark class)");
                    }
                }
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn apply_theme(theme: Theme) {
    // For desktop, use document::eval to run JavaScript in the webview
    let js = match theme {
        Theme::System => {
            "document.documentElement.classList.remove('dark'); document.documentElement.classList.remove('light');"
        }
        Theme::Light => {
            "document.documentElement.classList.remove('dark'); document.documentElement.classList.add('light');"
        }
        Theme::Dark => {
            "document.documentElement.classList.remove('light'); document.documentElement.classList.add('dark');"
        }
    };

    // Use dioxus document eval to run JavaScript
    dioxus::document::eval(js);
    tracing::info!("Applied theme: {:?}", theme);
}

/// Root application component
#[component]
pub fn App() -> Element {
    // Initialize global state
    let session_state = use_signal(SessionState::new);
    let chat_state = use_signal(ChatState::new);
    let conversation_state = use_signal(ConversationState::new);
    let settings = use_signal(Settings::load);

    // UI state
    let mut sidebar_visible = use_signal(|| true);
    let mut shortcuts_open = use_signal(|| false);

    // Provide state to all child components via context
    use_context_provider(|| session_state);
    use_context_provider(|| chat_state);
    use_context_provider(|| conversation_state);
    use_context_provider(|| settings);
    use_context_provider(|| sidebar_visible);

    // Initialize session on mount
    use_effect(move || {
        let session = session_state;
        spawn(async move {
            // Do NOT hold a signal borrow across `.await`: a `session.write()`
            // (or `.read()`) guard kept alive during the session-creation network
            // call makes any concurrent access to `session_state` (e.g. ChatPanel
            // reading `loaded_datasets` during a re-render) panic with
            // `AlreadyBorrowedMut`. Clone a snapshot first so the borrow is
            // released before we await.
            let snapshot = session.read().clone();
            if let Err(e) = snapshot.initialize().await {
                tracing::error!("Failed to initialize session: {}", e);
            }
        });
    });

    // Apply theme on mount and whenever settings change
    use_effect(move || {
        let theme = settings.read().theme;
        apply_theme(theme);
    });

    let is_sidebar_visible = *sidebar_visible.read();

    // Global keyboard handler
    let handle_global_keydown = {
        let session_state = session_state;
        let mut conversation_state = conversation_state;
        move |evt: Event<KeyboardData>| {
            let key = evt.key();
            let ctrl = evt.modifiers().ctrl();
            let meta = evt.modifiers().meta(); // For Mac Cmd key
            let ctrl_or_cmd = ctrl || meta;

            // Check if we're typing in an input field (don't intercept those keys)
            // We check for common character keys that would be typed
            let is_typing = matches!(key, Key::Character(_)) && !ctrl_or_cmd;

            if is_typing {
                return;
            }

            match (&key, ctrl_or_cmd) {
                // Ctrl+/ or Cmd+/ - Show shortcuts
                (Key::Character(c), true) if c == "/" => {
                    evt.prevent_default();
                    shortcuts_open.set(true);
                }
                // ? without modifier - Show shortcuts (but not in input)
                (Key::Character(c), false) if c == "?" => {
                    // Only trigger if not in an input field
                    // The event will only reach here if the target is the main div
                    evt.prevent_default();
                    shortcuts_open.set(true);
                }
                // Ctrl+K or Cmd+K - Focus chat input
                (Key::Character(c), true) if c == "k" => {
                    evt.prevent_default();
                    // Focus the chat input textarea via JS
                    #[cfg(target_arch = "wasm32")]
                    {
                        if let Some(window) = web_sys::window() {
                            if let Some(document) = window.document() {
                                if let Some(textarea) =
                                    document.query_selector("textarea").ok().flatten()
                                {
                                    if let Some(elem) = textarea.dyn_ref::<web_sys::HtmlElement>() {
                                        let _ = elem.focus();
                                    }
                                }
                            }
                        }
                    }
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        dioxus::document::eval("document.querySelector('textarea')?.focus()");
                    }
                }
                // Ctrl+N or Cmd+N - New conversation
                (Key::Character(c), true) if c == "n" => {
                    evt.prevent_default();
                    let session_id = session_state.read().session_id.clone();
                    if let Some(sid) = session_id {
                        spawn(async move {
                            let client = api();
                            match client.create_conversation(&sid, "New Conversation").await {
                                Ok(new_conv) => {
                                    let id = new_conv.id.clone();
                                    conversation_state.write().add_conversation(new_conv);
                                    conversation_state
                                        .write()
                                        .set_current_conversation(Some(id));
                                }
                                Err(e) => {
                                    tracing::warn!("Failed to create conversation: {}", e);
                                }
                            }
                        });
                    }
                }
                // Escape - Close modals/dropdowns
                (Key::Escape, _) => {
                    if *shortcuts_open.read() {
                        evt.prevent_default();
                        shortcuts_open.set(false);
                    }
                }
                _ => {}
            }
        }
    };

    // Handle shortcuts modal close
    let handle_close_shortcuts = move |_| {
        shortcuts_open.set(false);
    };

    rsx! {
        // Include stylesheet - works for both web and desktop
        Stylesheet { href: asset!("/assets/styles.css") }

        div {
            class: "min-h-screen bg-gray-50 dark:bg-gray-900 flex",
            tabindex: "0",
            onkeydown: handle_global_keydown,

            // Conversation sidebar (collapsible)
            if is_sidebar_visible {
                div {
                    class: "w-72 flex-shrink-0",
                    ConversationSidebar {}
                }
            } else {
                // Show expand button when sidebar is hidden
                div {
                    class: "flex-shrink-0 bg-white dark:bg-gray-900 border-r border-gray-300 dark:border-gray-800",
                    button {
                        class: "m-2 p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors",
                        onclick: move |_| sidebar_visible.set(true),
                        title: "Show conversations",
                        svg {
                            class: "w-5 h-5",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M13 5l7 7-7 7M5 5l7 7-7 7"
                            }
                        }
                    }
                }
            }

            // Main chat area
            div {
                class: "flex-1 min-w-0 flex",
                // Chat panel
                div {
                    class: "flex-1 min-w-0",
                    ChatPanel {}
                }

                // Dataset sidebar (always visible for now)
                div {
                    class: "w-72 flex-shrink-0",
                    DatasetSidebar {}
                }
            }

            // Shortcuts modal
            ShortcutsModal {
                is_open: *shortcuts_open.read(),
                on_close: handle_close_shortcuts
            }
        }
    }
}
