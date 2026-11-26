//! Conversation sidebar component
//!
//! Displays a list of conversations and allows creating, selecting,
//! and managing conversations.

use dioxus::prelude::*;

use crate::api::Conversation;
use crate::components::P2aWordmark;
use crate::state::{
    ChatState, ConversationState, SessionState, remember_conversation, report_conversation_error,
    start_new_conversation, stored_conversation, switch_blocked,
};

/// Whether a switch may run now; when not, the reason is shown in the
/// sidebar.
fn may_switch(chat: Signal<ChatState>, mut conv: Signal<ConversationState>) -> bool {
    let blocked = switch_blocked(&chat.read(), &conv.read());
    match blocked {
        Some(why) => {
            conv.write().error = Some(why.to_string());
            false
        }
        None => true,
    }
}

/// The label of the archived section's toggle.
fn archived_toggle_label(count: usize, open: bool) -> String {
    if open {
        format!("Hide archived ({count})")
    } else {
        format!("Archived ({count})")
    }
}

/// Whether the archived section's toggle is shown (any conversation is
/// archived).
fn show_archived_toggle(count: usize) -> bool {
    count > 0
}

/// Conversation sidebar component
/// Whether a key pressed on a row's action button stops there: Enter and Space press the
/// button and must not also open the row; Escape and the rest reach the app's handler.
fn stops_at_row_actions(key: &Key) -> bool {
    *key == Key::Enter || *key == Key::Character(" ".into())
}

#[component]
pub fn ConversationSidebar() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();
    let chat_state = use_context::<Signal<ChatState>>();
    let mut sidebar_visible = use_context::<Signal<bool>>();

    // Editing state for rename
    let mut editing_id = use_signal(|| Option::<String>::None);
    let mut edit_title = use_signal(String::new);

    // Confirmation state for delete
    let mut delete_confirm_id = use_signal(|| Option::<String>::None);

    // Whether the archived conversations are listed
    let mut show_archived = use_signal(|| false);

    // Whether the conversation open before a reload was looked for (once,
    // with the first list), and the open conversation last remembered.
    let mut reopen_done = use_signal(|| false);
    let mut remembered = use_signal(|| None::<String>);

    // Load conversations when session is ready
    use_effect(move || {
        let session = session_state.read();
        let mut conv_state = conversation_state;

        if let Some(session_id) = session.session_id.clone() {
            spawn(async move {
                conv_state.write().set_loading(true);
                let state = conv_state.read().clone();
                match state.load_conversations(&session_id).await {
                    Ok(conversations) => {
                        // After a reload: open the conversation that was open,
                        // if it is still listed and nothing else was opened.
                        let reopen = (!*reopen_done.peek())
                            .then(stored_conversation)
                            .flatten()
                            .filter(|id| conversations.iter().any(|c| &c.id == id))
                            .filter(|_| conv_state.peek().current_conversation_id.is_none());
                        reopen_done.set(true);
                        conv_state.write().set_conversations(conversations);
                        conv_state.write().set_loading(false);
                        if let Some(id) = reopen {
                            conv_state
                                .write()
                                .set_current_conversation(Some(id.clone()));
                            let state = conv_state.read().clone();
                            match state.load_messages(&id).await {
                                Ok(messages) => {
                                    conv_state.write().set_current_messages(&id, messages);
                                }
                                Err(e) => report_conversation_error(
                                    session_state,
                                    conv_state,
                                    chat_state,
                                    e,
                                ),
                            }
                        }
                    }
                    Err(e) => report_conversation_error(session_state, conv_state, chat_state, e),
                }
            });
        }
    });

    // Remember the open conversation for the next reload, once the one open
    // before was looked for (until then the empty start would erase it).
    use_effect(move || {
        let current = conversation_state.read().current_conversation_id.clone();
        if *reopen_done.read() && *remembered.peek() != current {
            remember_conversation(current.as_deref());
            remembered.set(current);
        }
    });

    // Handle new conversation. Refused while an answer streams (it would
    // redirect the answer into the new conversation) or another change runs;
    // a conversation created while a send started meanwhile is listed, not
    // made current.
    let handle_new_conversation = move |_| {
        if !may_switch(chat_state, conversation_state) {
            return;
        }
        spawn(start_new_conversation(
            session_state,
            conversation_state,
            chat_state,
        ));
    };

    // Handle select conversation. Refused while an answer streams: the stream
    // writes into the selected conversation's transcript. The messages are
    // kept only if the conversation is still current when they arrive.
    // Selecting the current conversation does nothing; one with a live
    // transcript (left in this page session) is shown from it, unloaded.
    let handle_select = move |id: String| {
        if conversation_state.read().current_conversation_id.as_deref() == Some(id.as_str()) {
            return;
        }
        if !may_switch(chat_state, conversation_state) {
            return;
        }
        // a phone's overlay closes on the choice, uncovering the chat
        if crate::app::narrow_viewport() {
            let mut visible = sidebar_visible;
            visible.set(false);
        }
        let mut conv = conversation_state;
        spawn(async move {
            conv.write().set_current_conversation(Some(id.clone()));
            if chat_state.read().has_live(&id) {
                return;
            }

            // Load messages for the selected conversation
            let state = conv.read().clone();
            match state.load_messages(&id).await {
                Ok(messages) => {
                    conv.write().set_current_messages(&id, messages);
                }
                Err(e) => {
                    tracing::error!("Failed to load messages: {}", e);
                    report_conversation_error(session_state, conv, chat_state, e);
                }
            }
        });
    };

    // Handle start rename
    let mut handle_start_rename = move |id: String, title: String| {
        editing_id.set(Some(id));
        edit_title.set(title);
    };

    // Handle save rename
    let handle_save_rename = move |_| {
        let id = editing_id.read().clone();
        let title = edit_title.read().clone();
        let mut conv = conversation_state;
        let mut ed_id = editing_id;

        if let Some(conversation_id) = id {
            spawn(async move {
                let state = conv.read().clone();
                match state
                    .update_conversation_title(&conversation_id, &title)
                    .await
                {
                    Ok(updated) => {
                        conv.write().update_conversation(updated);
                    }
                    Err(e) => {
                        tracing::error!("Failed to rename: {}", e);
                        report_conversation_error(session_state, conv, chat_state, e);
                    }
                }
                ed_id.set(None);
            });
        }
    };

    // Handle cancel rename
    let mut handle_cancel_rename = move |_| {
        editing_id.set(None);
    };

    // Handle delete confirmation request
    let mut handle_request_delete = move |id: String| {
        delete_confirm_id.set(Some(id));
    };

    // Handle cancel delete
    let mut handle_cancel_delete = move |_| {
        delete_confirm_id.set(None);
    };
    // Escape cancels the delete dialog.
    let cancel_delete = use_callback(move |_: ()| delete_confirm_id.set(None));
    crate::app::use_escape_close(delete_confirm_id.read().is_some(), cancel_delete);

    // Handle confirmed delete
    let handle_confirm_delete = move |_| {
        // Deleting a conversation while a response streams into it would leave
        // the stream writing to a removed conversation; block it.
        if !may_switch(chat_state, conversation_state) {
            delete_confirm_id.set(None);
            return;
        }
        let id = delete_confirm_id.read().clone();
        let mut conv = conversation_state;
        let mut chat = chat_state;
        let mut confirm_id = delete_confirm_id;

        if let Some(conversation_id) = id {
            spawn(async move {
                conv.write().set_operating(true);
                let state = conv.read().clone();
                match state.delete_conversation(&conversation_id).await {
                    Ok(()) => {
                        conv.write().remove_conversation(&conversation_id);
                        chat.write().forget_live(&conversation_id);

                        // Auto-select the first remaining non-archived conversation
                        let remaining: Vec<_> = conv
                            .read()
                            .conversations
                            .iter()
                            .filter(|c| !c.is_archived)
                            .cloned()
                            .collect();

                        // No auto-selection when a send started meanwhile.
                        let next = remaining.first().filter(|_| !chat_state.read().is_busy());
                        if let Some(next_conv) = next {
                            let next_id = next_conv.id.clone();
                            tracing::debug!(
                                "[ConversationSidebar] Auto-selecting conversation after delete: {}",
                                next_id
                            );
                            conv.write().set_current_conversation(Some(next_id.clone()));

                            // Load messages for the newly selected conversation
                            let state = conv.read().clone();
                            match state.load_messages(&next_id).await {
                                Ok(messages) => {
                                    conv.write().set_current_messages(&next_id, messages);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        "[ConversationSidebar] Failed to load messages for auto-selected conversation: {}",
                                        e
                                    );
                                    report_conversation_error(session_state, conv, chat_state, e);
                                }
                            }
                        }

                        conv.write().set_operating(false);
                    }
                    Err(e) => {
                        report_conversation_error(session_state, conv, chat_state, e);
                        conv.write().set_operating(false);
                    }
                }
                confirm_id.set(None);
            });
        }
    };

    // Handle archive/unarchive
    // Archiving the current conversation keeps it selected and opens the
    // archived section, so it stays in view.
    let handle_toggle_archive = move |id: String, is_archived: bool| {
        let mut conv = conversation_state;
        spawn(async move {
            let state = conv.read().clone();
            match state.set_conversation_archived(&id, !is_archived).await {
                Ok(updated) => {
                    let archived_current = updated.is_archived
                        && conv.read().current_conversation_id.as_deref() == Some(id.as_str());
                    conv.write().update_conversation(updated);
                    if archived_current {
                        show_archived.set(true);
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to toggle archive: {}", e);
                    report_conversation_error(session_state, conv, chat_state, e);
                }
            }
        });
    };

    // The active conversations, then (in their own section) the archived ones
    let state = conversation_state.read();
    let (archived, conversations): (Vec<Conversation>, Vec<Conversation>) = state
        .conversations
        .iter()
        .cloned()
        .partition(|c| c.is_archived);
    let archived_open = *show_archived.read();
    let current_id = state.current_conversation_id.clone();
    let is_loading = state.is_loading;
    let sidebar_error = state.error.clone();
    // Why "+" is unavailable now, if it is (shown as its tooltip).
    let new_blocked = switch_blocked(&chat_state.read(), &state);

    // One conversation row (active or archived): select, rename, archive or
    // unarchive, delete.
    let row_current_id = current_id.clone();
    let render_row = move |conv: &Conversation| -> Element {
        let conv_id = conv.id.clone();
        let conv_title = conv.title.clone();
        let is_current = row_current_id.as_ref() == Some(&conv_id);
        let is_editing = editing_id.read().as_ref() == Some(&conv_id);
        let is_archived = conv.is_archived;
        let message_count = conv.message_count;

        rsx! {
            div {
                key: "{conv_id}",
                class: if is_current {
                    "group relative py-[7px] pr-3 pl-[34px] cursor-pointer font-semibold"
                } else {
                    "group relative py-[7px] pr-3 pl-[34px] cursor-pointer hover:bg-[var(--k-fill)]"
                },
                onclick: {
                    let id = conv_id.clone();
                    move |_| handle_select(id.clone())
                },
                // keyboard: Tab reaches the row, Enter or Space opens it
                tabindex: "0",
                role: "button",
                "aria-current": if is_current { "true" } else { "false" },
                onkeydown: {
                    let id = conv_id.clone();
                    move |e: Event<KeyboardData>| {
                        let open = e.key() == Key::Enter || e.key() == Key::Character(" ".into());
                        if open && !is_editing {
                            e.prevent_default();
                            handle_select(id.clone())
                        }
                    }
                },

                if is_current {
                    span { class: "absolute left-4 top-[6px] k-mono font-bold k-orange", "aria-hidden": "true", "›" }
                }
                // Title row
                div { class: "flex items-center gap-2",
                    if is_editing {
                        input {
                            class: "flex-1 px-2 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-teal-500",
                            value: "{edit_title}",
                            "aria-label": "Conversation title",
                            // the field takes the focus, with its text selected
                            onmounted: move |e: Event<MountedData>| async move {
                                let _ = e.set_focus(true).await;
                            },
                            onfocus: move |_| select_focused_text(),
                            oninput: move |e| edit_title.set(e.value().clone()),
                            onkeydown: move |e| {
                                if e.key() == Key::Enter {
                                    handle_save_rename(())
                                } else if e.key() == Key::Escape {
                                    handle_cancel_rename(())
                                }
                            },
                            onclick: |e| e.stop_propagation()
                        }
                        button {
                            class: "p-1 text-teal-600 hover:text-teal-700 dark:text-teal-400",
                            "aria-label": "Save the title",
                            title: "Save the title",
                            onclick: move |e| {
                                e.stop_propagation();
                                handle_save_rename(())
                            },
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M5 13l4 4L19 7"
                                }
                            }
                        }
                        button {
                            class: "p-1 text-gray-500 hover:text-gray-700 dark:text-gray-400",
                            "aria-label": "Keep the old title",
                            title: "Keep the old title",
                            onclick: move |e| {
                                e.stop_propagation();
                                handle_cancel_rename(())
                            },
                            svg {
                                class: "w-4 h-4",
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
                    } else {
                        // Title
                        span {
                            class: if is_archived {
                                "flex-1 text-sm k-muted truncate italic"
                            } else {
                                "flex-1 text-sm truncate"
                            },
                            "{conv_title}"
                        }

                        // Archived badge
                        if is_archived {
                            span { class: "k-mono text-[11px] k-muted",
                                "archived"
                            }
                        }

                        // Action buttons (shown on hover)
                        // shown on hover, and while the row or a button has the keyboard focus
                        div { class: "k-row-actions flex items-center gap-0.5 transition-opacity",
                            // Enter or Space on a button acts on the button, not on the row; other
                            // keys (Escape for an open dialog) go on to the app's handler
                            onkeydown: |e: Event<KeyboardData>| {
                                if stops_at_row_actions(&e.key()) {
                                    e.stop_propagation();
                                }
                            },
                            // Rename button
                            button {
                                class: "action-btn-rename p-1.5 rounded transition-all cursor-pointer hover:bg-teal-50 dark:hover:bg-teal-900/30",
                                title: "Rename",
                                onclick: {
                                    let id = conv_id.clone();
                                    let title = conv_title.clone();
                                    move |e: Event<MouseData>| {
                                        e.stop_propagation();
                                        handle_start_rename(id.clone(), title.clone())
                                    }
                                },
                                svg {
                                    class: "w-3.5 h-3.5 text-gray-400 transition-colors",
                                    fill: "none",
                                    stroke: "currentColor",
                                    view_box: "0 0 24 24",
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                                    }
                                }
                            }
                            // Archive button
                            button {
                                class: "action-btn-archive p-1.5 rounded transition-all cursor-pointer hover:bg-orange-50 dark:hover:bg-orange-900/30",
                                title: if is_archived { "Unarchive" } else { "Archive" },
                                onclick: {
                                    let id = conv_id.clone();
                                    move |e: Event<MouseData>| {
                                        e.stop_propagation();
                                        handle_toggle_archive(id.clone(), is_archived)
                                    }
                                },
                                svg {
                                    class: "w-3.5 h-3.5 text-gray-400 transition-colors",
                                    fill: "none",
                                    stroke: "currentColor",
                                    view_box: "0 0 24 24",
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"
                                    }
                                }
                            }
                            // Delete button
                            button {
                                class: "action-btn-delete p-1.5 rounded transition-all cursor-pointer hover:bg-red-50 dark:hover:bg-red-900/30",
                                title: "Delete",
                                onclick: {
                                    let id = conv_id.clone();
                                    move |e: Event<MouseData>| {
                                        e.stop_propagation();
                                        handle_request_delete(id.clone())
                                    }
                                },
                                svg {
                                    class: "w-3.5 h-3.5 text-gray-400 transition-colors",
                                    fill: "none",
                                    stroke: "currentColor",
                                    view_box: "0 0 24 24",
                                    path {
                                        stroke_linecap: "round",
                                        stroke_linejoin: "round",
                                        stroke_width: "2",
                                        d: "M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                                    }
                                }
                            }
                        }
                    }
                }

                // Message count
                if !is_editing {
                    div { class: "flex items-center gap-2",
                        span { class: "text-xs font-normal k-muted",
                            "{message_count} messages"
                        }
                    }
                }
            }
        }
    };

    // Track if we need to show delete dialog (read outside of sidebar div)
    let show_delete_dialog = delete_confirm_id.read().is_some();
    // The title of the conversation the dialog would delete, so the dialog names it
    let delete_title = delete_confirm_id.read().as_ref().and_then(|id| {
        conversation_state
            .read()
            .conversations
            .iter()
            .find(|c| &c.id == id)
            .map(|c| c.title.clone())
    });

    rsx! {
        // Use fragment to render sidebar and dialog as siblings
        // This ensures the dialog isn't constrained by sidebar's width

        div { class: "flex flex-col h-full k-paper",
            // Brand row, as tall as the chat header
            div { class: "flex-shrink-0 h-[60px] px-[18px] k-rule-b flex items-center justify-between",
                P2aWordmark { width: 56.0 }
                button {
                    class: "k-icon-btn",
                    onclick: move |_| sidebar_visible.set(false),
                    title: "Hide conversations",
                    "aria-label": "Hide conversations",
                    span { class: "k-mono text-sm k-muted", "«" }
                }
            }
            // New conversation
            button {
                class: "flex-shrink-0 mx-[18px] mt-3.5 mb-2 text-left px-3 py-[7px] border border-[var(--k-rule)] font-semibold text-sm hover:bg-[var(--k-fill)] disabled:opacity-50 disabled:cursor-not-allowed",
                disabled: new_blocked.is_some(),
                onclick: handle_new_conversation,
                title: new_blocked.unwrap_or("Start a new analysis"),
                span { class: "k-mono k-orange-text mr-1", "+" }
                "New analysis"
            }

            // A refused switch or a failed conversation call, dismissable
            if let Some(ref err) = sidebar_error {
                div { class: "mx-[18px] mt-3 pl-3 py-1 border-l-4 border-[var(--k-orange)] flex items-start gap-2",
                    p { class: "flex-1 text-xs", "{err}" }
                    button {
                        class: "k-muted hover:text-[var(--k-ink)] text-xs",
                        title: "Dismiss",
                        onclick: move |_| {
                            let mut conv = conversation_state;
                            conv.write().clear_error();
                        },
                        "\u{2715}"
                    }
                }
            }

            // Conversation list
            div { class: "flex-1 overflow-y-auto py-1.5",
                if is_loading {
                    div { class: "flex items-center justify-center py-8",
                        span { class: "k-mono k-orange k-spin" }
                    }
                } else if conversations.is_empty() {
                    div { class: "px-[18px] py-4 k-muted",
                        p { class: "text-sm", "No analyses yet." }
                        p { class: "text-xs mt-1", "Your conversations will appear here." }
                    }
                } else {
                    for conv in conversations.iter() {
                        {render_row(conv)}
                    }
                }
                // Archived conversations, collapsed under the list
                if !is_loading && show_archived_toggle(archived.len()) {
                    button {
                        class: "w-full mt-2 px-[18px] py-1.5 text-xs k-mono text-left k-muted hover:text-[var(--k-ink)]",
                        onclick: move |_| show_archived.set(!archived_open),
                        "{archived_toggle_label(archived.len(), archived_open)}"
                    }
                    if archived_open {
                        for conv in archived.iter() {
                            {render_row(conv)}
                        }
                    }
                }
            }
        }

        // Delete confirmation dialog - rendered OUTSIDE sidebar div as a sibling
        // Uses fixed positioning to overlay the entire viewport
        if show_delete_dialog {
            div { class: "fixed inset-0 z-[9999] flex items-center justify-center bg-black/50",
                onclick: move |_| handle_cancel_delete(()),
                div {
                    class: "k-paper border-2 border-[var(--k-rule)] p-6 m-4 w-[90vw] max-w-md",
                    role: "alertdialog",
                    "aria-modal": "true",
                    "aria-labelledby": "delete-dialog-title",
                    "aria-describedby": "delete-dialog-text",
                    onclick: |e| e.stop_propagation(),
                    // Warning icon and title
                    div { class: "flex items-center gap-3 mb-4",
                        div { class: "hidden",
                            svg {
                                class: "w-6 h-6 text-red-600 dark:text-red-400",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                                }
                            }
                        }
                        h3 { id: "delete-dialog-title", class: "text-[22px] font-extrabold k-cond",
                            "Delete this conversation?"
                        }
                    }
                    // Message
                    p { id: "delete-dialog-text", class: "text-sm mb-6",
                        if let Some(title) = delete_title {
                            "This will permanently delete “{title}” and all its messages. This action cannot be undone."
                        } else {
                            "This will permanently delete the conversation and all its messages. This action cannot be undone."
                        }
                    }
                    // Buttons - use grid to ensure both are always visible
                    div { class: "grid grid-cols-2 gap-3 mt-2",
                        button {
                            class: "k-btn justify-center",
                            onclick: move |_| handle_cancel_delete(()),
                            "Cancel"
                        }
                        button {
                            class: "k-btn-pri justify-center",
                            onclick: handle_confirm_delete,
                            "Delete"
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use dioxus::prelude::Key;

    /// Escape on a focused Delete button reaches the app's handler and closes the dialog.
    #[test]
    fn escape_passes_the_row_action_buttons() {
        assert!(!stops_at_row_actions(&Key::Escape));
        assert!(stops_at_row_actions(&Key::Enter));
        assert!(stops_at_row_actions(&Key::Character(" ".into())));
    }

    use super::*;

    #[test]
    fn archived_toggle_names_the_count() {
        assert_eq!(archived_toggle_label(3, false), "Archived (3)");
        assert_eq!(archived_toggle_label(3, true), "Hide archived (3)");
        assert!(show_archived_toggle(1));
        assert!(!show_archived_toggle(0));
    }
}

/// Select the text of the focused input. The web build calls the DOM directly: the page's
/// Content-Security-Policy forbids the string evaluation `document::eval` uses there, and the
/// refused call panicked the app when a conversation was renamed.
#[cfg(target_arch = "wasm32")]
fn select_focused_text() {
    use wasm_bindgen::JsCast;
    if let Some(input) = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.active_element())
        .and_then(|el| el.dyn_into::<web_sys::HtmlInputElement>().ok())
    {
        input.select();
    }
}

/// Select the text of the focused input (desktop: a script in the webview).
#[cfg(not(target_arch = "wasm32"))]
fn select_focused_text() {
    let _ = document::eval("document.activeElement?.select?.()");
}
