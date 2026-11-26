//! Root application component

use dioxus::prelude::*;
use futures::StreamExt;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

use crate::components::{
    ChatPanel, ConversationSidebar, DatasetSidebar, ShortcutsModal, StatusBar,
};
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

/// One item per Escape pressed with its target outside the element marked `data-p2a-root`
/// (inside it the root's onkeydown handles the key). The web build listens with web_sys: the
/// page's Content-Security-Policy forbids the string evaluation `document::eval` uses there.
#[cfg(target_arch = "wasm32")]
fn escape_outside_root() -> futures::channel::mpsc::UnboundedReceiver<()> {
    let (tx, rx) = futures::channel::mpsc::unbounded();
    let listener = wasm_bindgen::closure::Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(
        move |e: web_sys::KeyboardEvent| {
            if e.key() != "Escape" {
                return;
            }
            let in_root = e
                .target()
                .and_then(|t| t.dyn_into::<web_sys::Element>().ok())
                .and_then(|el| el.closest("[data-p2a-root]").ok().flatten())
                .is_some();
            if !in_root {
                let _ = tx.unbounded_send(());
            }
        },
    );
    if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
        let _ = doc.add_event_listener_with_callback("keydown", listener.as_ref().unchecked_ref());
    }
    // Lives as long as the page, as the App does.
    listener.forget();
    rx
}

/// The desktop build: the webview reports each such Escape through `document::eval`.
#[cfg(not(target_arch = "wasm32"))]
fn escape_outside_root() -> futures::channel::mpsc::UnboundedReceiver<()> {
    let (tx, rx) = futures::channel::mpsc::unbounded();
    spawn(async move {
        let mut outside = dioxus::document::eval(ESCAPE_OUTSIDE_ROOT_JS);
        while outside.recv::<bool>().await.is_ok() {
            if tx.unbounded_send(()).is_err() {
                break;
            }
        }
    });
    rx
}

/// The desktop listener: sends `true` for each Escape outside the root; never finishes, so the
/// channel stays open.
#[cfg(not(target_arch = "wasm32"))]
const ESCAPE_OUTSIDE_ROOT_JS: &str = r#"
document.addEventListener('keydown', (e) => {
    const inRoot = e.target instanceof Element && e.target.closest('[data-p2a-root]');
    if (e.key === 'Escape' && !inRoot) { dioxus.send(true); }
});
await new Promise(() => {});
"#;

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
    // On a narrow screen (a phone) both side panels start closed and open as overlays.
    let mut sidebar_visible = use_signal(|| !narrow_viewport());
    let mut datasets_open = use_signal(|| false);
    let mut shortcuts_open = use_signal(|| false);

    // Provide state to all child components via context
    use_context_provider(|| session_state);
    use_context_provider(|| chat_state);
    use_context_provider(|| conversation_state);
    use_context_provider(|| settings);
    use_context_provider(|| sidebar_visible);
    use_context_provider(|| crate::state::AddDataRequest(Signal::new(0)));
    use_context_provider(|| crate::state::ConnectModelRequest(Signal::new(0)));
    use_context_provider(crate::exhibit::ExhibitViewer::new);
    // The open dialogs, for Escape (use_escape_close)
    let escape_stack = use_context_provider(|| Signal::new(EscapeStack::<Callback<()>>::new()));

    // No session is created at mount: the first send, data load or new
    // conversation creates it (state::ensure_session). The app restores the
    // session it used last, through the same gate: on desktop and mobile
    // across restarts, on web across a reload of the tab.
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

    // Escape pressed while the focus is outside the app's root element (on the page body, after
    // the button that had it is removed, as "Load dataset" is once the file loads) never reaches
    // the root's onkeydown. A document listener reports it; the dialog opened last is closed from
    // a task inside the Dioxus runtime.
    use_future(move || async move {
        let mut outside = escape_outside_root();
        while outside.next().await.is_some() {
            let top = escape_stack.read().top().copied();
            if let Some(close) = top {
                close.call(());
            }
        }
    });

    // Handle shortcuts modal close
    let handle_close_shortcuts = move |_| {
        shortcuts_open.set(false);
    };

    rsx! {
        // Include stylesheet - works for both web and desktop
        Stylesheet { href: asset!("/assets/fonts.css") }
        Stylesheet { href: asset!("/assets/styles.css") }

        div {
            class: "h-screen flex flex-col k-paper",
            tabindex: "0",
            "data-p2a-root": "1",
            onkeydown: handle_global_keydown,

            div { class: "flex-1 min-h-0 flex",
                // Conversation sidebar (collapsible)
                if is_sidebar_visible {
                    div { class: "k-side-conv w-[236px] flex-shrink-0 k-rule-r min-h-0",
                        ConversationSidebar {}
                    }
                } else {
                    // Show expand button when sidebar is hidden
                    div { class: "flex-shrink-0 k-rule-r k-edge",
                        button {
                            class: "k-icon-btn m-2",
                            onclick: move |_| sidebar_visible.set(true),
                            title: "Show conversations",
                            "aria-label": "Show conversations",
                            span { class: "k-mono font-bold k-orange", "»" }
                        }
                    }
                }

                // Chat panel
                div { class: "flex-1 min-w-0 min-h-0",
                    ChatPanel {}
                }

                // Dataset sidebar: always shown on a wide screen; an overlay opened from the
                // edge button on a narrow one
                if !*datasets_open.read() {
                    div { class: "k-narrow-only flex-shrink-0 k-rule-l k-edge",
                        button {
                            class: "k-icon-btn m-2",
                            onclick: move |_| datasets_open.set(true),
                            title: "Show datasets",
                            "aria-label": "Show datasets",
                            span { class: "k-mono font-bold k-orange", "«" }
                        }
                    }
                }
                div {
                    class: if *datasets_open.read() {
                        "k-side-data open w-[312px] flex-shrink-0 k-rule-l min-h-0"
                    } else {
                        "k-side-data w-[312px] flex-shrink-0 k-rule-l min-h-0"
                    },
                    div { class: "k-narrow-only flex justify-end",
                        button {
                            class: "k-icon-btn m-1",
                            onclick: move |_| datasets_open.set(false),
                            title: "Hide datasets",
                            "aria-label": "Hide datasets",
                            span { class: "k-mono text-sm k-muted", "»" }
                        }
                    }
                    div { class: "flex-1 min-h-0", DatasetSidebar {} }
                }
            }

            StatusBar {}

            // Figures and tables of the transcript, enlarged
            crate::components::ExhibitViewerModal {}

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

    /// `document::eval` runs a string through `new Function`, which the web page's
    /// Content-Security-Policy refuses (no 'unsafe-eval'); the refusal panics the app. Every call
    /// must therefore sit in desktop-only code: a `cfg(not(target_arch = "wasm32"))` within the
    /// twelve lines above it.
    #[test]
    fn document_eval_is_desktop_only() {
        fn visit(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for entry in std::fs::read_dir(dir).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    visit(&path, out);
                } else if path.extension().is_some_and(|e| e == "rs") {
                    out.push(path);
                }
            }
        }
        let mut files = Vec::new();
        visit(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        let mut bad = Vec::new();
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let lines: Vec<&str> = text.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let code = line.split("//").next().unwrap_or("");
                if !code.contains(concat!("document::", "eval(")) {
                    continue;
                }
                let gated = lines[i.saturating_sub(12)..i]
                    .iter()
                    .any(|l| l.contains("cfg(not(target_arch = \"wasm32\"))"));
                if !gated {
                    bad.push(format!("{}:{}", file.display(), i + 1));
                }
            }
        }
        assert!(
            bad.is_empty(),
            "document::eval reachable on the web: {bad:?}"
        );
    }

    /// The desktop outside-root listener answers only Escape, only outside the root the App marks.
    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn escape_outside_root_listener() {
        assert!(ESCAPE_OUTSIDE_ROOT_JS.contains("e.key === 'Escape'"));
        assert!(ESCAPE_OUTSIDE_ROOT_JS.contains("closest('[data-p2a-root]')"));
        assert!(ESCAPE_OUTSIDE_ROOT_JS.contains("dioxus.send(true)"));
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

/// Below this viewport width (CSS px) the side panels are overlays (`input.css`).
const NARROW_BELOW_PX: f64 = 900.0;

/// Whether the window is narrower than [`NARROW_BELOW_PX`] at start (web; native windows are
/// taken as wide).
pub(crate) fn narrow_viewport() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window()
            .and_then(|w| w.inner_width().ok())
            .and_then(|w| w.as_f64())
            .is_some_and(|w| w < NARROW_BELOW_PX)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = NARROW_BELOW_PX;
        false
    }
}
