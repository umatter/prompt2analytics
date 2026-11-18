//! Root application component

use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::api::api;
use crate::components::{ChatPanel, ConversationSidebar, DatasetSidebar, ShortcutsModal};
use crate::state::settings::Theme;
use crate::state::{
    ChatState, ConversationState, EscapeStack, SessionState, Settings, ensure_session,
    switch_blocked,
};

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

/// A global keyboard shortcut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shortcut {
    /// Show the shortcuts (Ctrl/Cmd+/ anywhere, `?` outside text fields)
    Help,
    /// Focus the chat input (Ctrl/Cmd+K)
    FocusInput,
    /// New conversation (Ctrl/Cmd+N)
    New,
    /// Close the dialog opened last (Escape)
    CloseTop,
}

/// The shortcut a key press means: `ctrl_or_cmd` is Ctrl or Cmd held;
/// `in_text_field` whether the focus is in a text field, where `?` is typed,
/// not a shortcut.
pub fn global_shortcut(key: &Key, ctrl_or_cmd: bool, in_text_field: bool) -> Option<Shortcut> {
    match (key, ctrl_or_cmd) {
        (Key::Escape, _) => Some(Shortcut::CloseTop),
        (Key::Character(c), true) => match c.as_str() {
            "/" => Some(Shortcut::Help),
            "k" => Some(Shortcut::FocusInput),
            "n" => Some(Shortcut::New),
            _ => None,
        },
        (Key::Character(c), false) if c == "?" && !in_text_field => Some(Shortcut::Help),
        _ => None,
    }
}

/// Whether an element with tag `tag` (any case) is a text field: an input,
/// a text area, a select, or any element being edited.
pub fn is_text_field(tag: &str, content_editable: bool) -> bool {
    content_editable
        || ["INPUT", "TEXTAREA", "SELECT"]
            .iter()
            .any(|t| tag.eq_ignore_ascii_case(t))
}

/// Whether the focus is in a text field (web: read synchronously).
#[cfg(target_arch = "wasm32")]
fn focus_in_text_field() -> bool {
    web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.active_element())
        .map(|el| {
            let editable = el
                .dyn_ref::<web_sys::HtmlElement>()
                .is_some_and(|h| h.is_content_editable());
            is_text_field(&el.tag_name(), editable)
        })
        .unwrap_or(false)
}

/// Whether the focus is in a text field (desktop: asked from the webview).
#[cfg(not(target_arch = "wasm32"))]
async fn focus_in_text_field_native() -> bool {
    let js = "const el = document.activeElement; \
              return !!el && (el.isContentEditable || ['INPUT','TEXTAREA','SELECT'].includes(el.tagName));";
    dioxus::document::eval(js)
        .join::<bool>()
        .await
        .unwrap_or(true)
}

/// Register a dialog with Escape while `open`: Escape calls the `on_close`
/// of the dialog opened last.
pub fn use_escape_close(open: bool, on_close: Callback<()>) {
    let mut stack = use_context::<Signal<EscapeStack<Callback<()>>>>();
    let id = use_hook(|| {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    });
    use_effect(use_reactive((&open,), move |(open,)| {
        if open {
            stack.write().push(id, on_close);
        } else {
            stack.write().remove(id);
        }
    }));
    use_drop(move || stack.write().remove(id));
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
    // The open dialogs, for Escape (use_escape_close)
    let escape_stack = use_context_provider(|| Signal::new(EscapeStack::<Callback<()>>::new()));

    // No session is created at mount: the first send, data load or new
    // conversation creates it (state::ensure_session). The native app
    // restores the session it used last, through the same gate.
    #[cfg(any(feature = "desktop", feature = "mobile"))]
    use_effect(move || {
        spawn(crate::state::restore_session(session_state));
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
        let chat_state = chat_state;
        move |evt: Event<KeyboardData>| {
            let key = evt.key();
            let ctrl = evt.modifiers().ctrl();
            let meta = evt.modifiers().meta(); // For Mac Cmd key
            let ctrl_or_cmd = ctrl || meta;

            // `?` is a shortcut only outside text fields. The web build reads
            // the focus synchronously; the desktop build asks the webview.
            #[cfg(target_arch = "wasm32")]
            let in_text_field = focus_in_text_field();
            #[cfg(not(target_arch = "wasm32"))]
            let in_text_field = false;

            match global_shortcut(&key, ctrl_or_cmd, in_text_field) {
                Some(Shortcut::Help) => {
                    #[cfg(not(target_arch = "wasm32"))]
                    if !ctrl_or_cmd {
                        spawn(async move {
                            if !focus_in_text_field_native().await {
                                shortcuts_open.set(true);
                            }
                        });
                        return;
                    }
                    evt.prevent_default();
                    shortcuts_open.set(true);
                }
                Some(Shortcut::FocusInput) => {
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
                Some(Shortcut::New) => {
                    evt.prevent_default();
                    // Refused while an answer streams or a conversation change
                    // runs, as the sidebar's "+".
                    let blocked = switch_blocked(&chat_state.read(), &conversation_state.read());
                    if let Some(why) = blocked {
                        conversation_state.write().error = Some(why.to_string());
                        return;
                    }
                    spawn(async move {
                        conversation_state.write().set_operating(true);
                        let sid = match ensure_session(session_state).await {
                            Ok(sid) => sid,
                            Err(e) => {
                                conversation_state.write().set_error(Some(e));
                                return;
                            }
                        };
                        let client = api();
                        match client.create_conversation(&sid, "New Conversation").await {
                            Ok(new_conv) => {
                                let id = new_conv.id.clone();
                                let mut conv = conversation_state.write();
                                conv.add_conversation(new_conv);
                                // A send that started meanwhile keeps its conversation.
                                if !chat_state.read().is_busy() {
                                    conv.set_current_conversation(Some(id));
                                }
                                conv.set_operating(false);
                            }
                            Err(e) => {
                                tracing::warn!("Failed to create conversation: {}", e);
                                conversation_state.write().set_error(Some(e));
                            }
                        }
                    });
                }
                Some(Shortcut::CloseTop) => {
                    let top = escape_stack.read().top().copied();
                    if let Some(close) = top {
                        evt.prevent_default();
                        close.call(());
                    }
                }
                None => {}
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ch(c: &str) -> Key {
        Key::Character(c.to_string())
    }

    #[test]
    fn global_shortcuts() {
        assert_eq!(
            global_shortcut(&ch("?"), false, false),
            Some(Shortcut::Help)
        );
        assert_eq!(global_shortcut(&ch("?"), false, true), None);
        assert_eq!(global_shortcut(&ch("/"), true, true), Some(Shortcut::Help));
        assert_eq!(global_shortcut(&ch("n"), true, false), Some(Shortcut::New));
        assert_eq!(
            global_shortcut(&ch("k"), true, true),
            Some(Shortcut::FocusInput)
        );
        assert_eq!(
            global_shortcut(&Key::Escape, false, true),
            Some(Shortcut::CloseTop)
        );
        assert_eq!(global_shortcut(&ch("n"), false, false), None);
        assert_eq!(global_shortcut(&ch("/"), false, false), None);
    }

    #[test]
    fn text_fields() {
        for tag in ["INPUT", "textarea", "Select"] {
            assert!(is_text_field(tag, false), "{tag}");
        }
        assert!(is_text_field("DIV", true));
        assert!(!is_text_field("DIV", false));
        assert!(!is_text_field("BUTTON", false));
    }
}
