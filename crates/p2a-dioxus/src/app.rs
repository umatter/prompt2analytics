//! Root application component

use dioxus::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::components::{ChatPanel, ConversationSidebar, DatasetSidebar, ShortcutsModal};
use crate::state::settings::Theme;
use crate::state::{
    ChatState, ConversationState, EscapeStack, SessionState, Settings, start_new_conversation,
    switch_blocked,
};

/// Whether `theme` shows the dark palette; System follows the OS.
pub fn theme_is_dark(theme: Theme, system_prefers_dark: bool) -> bool {
    match theme {
        Theme::Dark => true,
        Theme::Light => false,
        Theme::System => system_prefers_dark,
    }
}

/// The media query System follows.
#[cfg(target_arch = "wasm32")]
const PREFERS_DARK: &str = "(prefers-color-scheme: dark)";

#[cfg(target_arch = "wasm32")]
thread_local! {
    /// The theme applied last, read by the OS-appearance listener.
    static CURRENT_THEME: std::cell::Cell<Theme> = const { std::cell::Cell::new(Theme::Light) };
    /// Whether the OS-appearance listener is installed (once per page).
    static THEME_LISTENER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Set exactly one of `dark`/`light` on `<html>` (web).
#[cfg(target_arch = "wasm32")]
fn set_theme_class(window: &web_sys::Window, theme: Theme) {
    let prefers_dark = window
        .match_media(PREFERS_DARK)
        .ok()
        .flatten()
        .is_some_and(|mq| mq.matches());
    let dark = theme_is_dark(theme, prefers_dark);
    if let Some(html) = window.document().and_then(|d| d.document_element()) {
        let class_list = html.class_list();
        let (on, off) = if dark {
            ("dark", "light")
        } else {
            ("light", "dark")
        };
        let _ = class_list.remove_1(off);
        let _ = class_list.add_1(on);
    }
}

/// Apply theme to the document: `<html>` always carries `dark` or `light`
/// (the `dark:` utilities key on the class). System resolves through the OS
/// preference and follows its changes while System is the theme.
#[cfg(target_arch = "wasm32")]
pub fn apply_theme(theme: Theme) {
    let Some(window) = web_sys::window() else {
        return;
    };
    CURRENT_THEME.with(|t| t.set(theme));
    set_theme_class(&window, theme);
    if !THEME_LISTENER.with(|l| l.replace(true))
        && let Ok(Some(mq)) = window.match_media(PREFERS_DARK)
    {
        let on_change =
            wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::MediaQueryListEvent)>::new(
                move |_event: web_sys::MediaQueryListEvent| {
                    let theme = CURRENT_THEME.with(|t| t.get());
                    if theme == Theme::System
                        && let Some(window) = web_sys::window()
                    {
                        set_theme_class(&window, theme);
                    }
                },
            );
        let _ = mq.add_event_listener_with_callback("change", on_change.as_ref().unchecked_ref());
        // Kept for the page's life: the listener is installed once.
        on_change.forget();
    }
    tracing::info!("Applied theme: {:?}", theme);
}

/// Installs the OS-appearance listener once (`window.__p2aThemeMq`), which
/// re-applies while the theme is System.
#[cfg(not(target_arch = "wasm32"))]
macro_rules! theme_listener_js {
    () => {
        "var w = window; \
         if (!w.__p2aThemeMq) { \
           w.__p2aThemeMq = w.matchMedia('(prefers-color-scheme: dark)'); \
           w.__p2aThemeMq.addEventListener('change', function () { \
             if (w.__p2aTheme === 'system') { \
               var c = document.documentElement.classList; \
               var d = w.__p2aThemeMq.matches; \
               c.remove(d ? 'light' : 'dark'); c.add(d ? 'dark' : 'light'); \
             } \
           }); \
         } "
    };
}

/// The script the desktop webview runs for `theme`: exactly one of
/// `dark`/`light` on `<html>`; System resolves through the OS preference
/// and follows its changes.
#[cfg(not(target_arch = "wasm32"))]
pub fn theme_script(theme: Theme) -> &'static str {
    match theme {
        Theme::Light => concat!(
            theme_listener_js!(),
            "w.__p2aTheme = 'light'; \
             document.documentElement.classList.remove('dark'); \
             document.documentElement.classList.add('light');"
        ),
        Theme::Dark => concat!(
            theme_listener_js!(),
            "w.__p2aTheme = 'dark'; \
             document.documentElement.classList.remove('light'); \
             document.documentElement.classList.add('dark');"
        ),
        Theme::System => concat!(
            theme_listener_js!(),
            "w.__p2aTheme = 'system'; \
             var c = document.documentElement.classList; \
             var d = w.__p2aThemeMq.matches; \
             c.remove(d ? 'light' : 'dark'); c.add(d ? 'dark' : 'light');"
        ),
    }
}

/// Apply theme to the document (desktop: a script in the webview).
#[cfg(not(target_arch = "wasm32"))]
pub fn apply_theme(theme: Theme) {
    dioxus::document::eval(theme_script(theme));
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
                    spawn(start_new_conversation(
                        session_state,
                        conversation_state,
                        chat_state,
                    ));
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

#[cfg(test)]
mod theme_tests {
    use super::*;

    #[test]
    fn theme_is_dark_resolves_system_by_the_os() {
        assert!(!theme_is_dark(Theme::System, false));
        assert!(theme_is_dark(Theme::System, true));
        assert!(!theme_is_dark(Theme::Light, false));
        assert!(!theme_is_dark(Theme::Light, true));
        assert!(theme_is_dark(Theme::Dark, false));
        assert!(theme_is_dark(Theme::Dark, true));
    }

    #[test]
    fn theme_script_sets_one_class() {
        let light = theme_script(Theme::Light);
        assert!(light.contains("classList.add('light')"), "{light}");
        assert!(light.contains("classList.remove('dark')"), "{light}");
        let dark = theme_script(Theme::Dark);
        assert!(dark.contains("classList.add('dark')"), "{dark}");
        assert!(dark.contains("classList.remove('light')"), "{dark}");
        let system = theme_script(Theme::System);
        assert!(system.contains("prefers-color-scheme: dark"), "{system}");
        assert!(system.contains("__p2aThemeMq"), "{system}");
    }

    #[test]
    fn input_css_uses_the_dark_class() {
        let css = include_str!("input.css");
        assert!(css.contains("@custom-variant dark (&:where(.dark, .dark *));"));
        assert!(css.contains("@layer components"));
        assert!(!css.contains("html.light"));
        assert!(!css.contains("@media (prefers-color-scheme"));
        assert!(!css.contains("ring-color"));
    }
}
