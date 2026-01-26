//! Conversation sidebar component
//!
//! Displays a list of conversations and allows creating, selecting,
//! and managing conversations.

use dioxus::prelude::*;

use crate::api::Conversation;
use crate::components::P2aIconMinimal;
use crate::state::{ConversationState, SessionState};

/// Conversation sidebar component
#[component]
pub fn ConversationSidebar() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();
    let mut sidebar_visible = use_context::<Signal<bool>>();

    // Editing state for rename
    let mut editing_id = use_signal(|| Option::<String>::None);
    let mut edit_title = use_signal(String::new);

    // Confirmation state for delete
    let mut delete_confirm_id = use_signal(|| Option::<String>::None);

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
                        conv_state.write().set_conversations(conversations);
                        conv_state.write().set_loading(false);
                    }
                    Err(e) => {
                        conv_state.write().set_error(Some(e));
                    }
                }
            });
        }
    });

    // Handle new conversation
    let handle_new_conversation = move |_| {
        let session = session_state.read();
        let mut conv = conversation_state;

        if let Some(session_id) = session.session_id.clone() {
            spawn(async move {
                conv.write().set_operating(true);
                let state = conv.read().clone();
                match state.create_conversation(&session_id, "New Conversation").await {
                    Ok(conversation) => {
                        let id = conversation.id.clone();
                        conv.write().add_conversation(conversation);
                        conv.write().set_current_conversation(Some(id));
                        conv.write().set_operating(false);
                    }
                    Err(e) => {
                        conv.write().set_error(Some(e));
                        conv.write().set_operating(false);
                    }
                }
            });
        }
    };

    // Handle select conversation
    let handle_select = move |id: String| {
        let mut conv = conversation_state;
        spawn(async move {
            conv.write().set_current_conversation(Some(id.clone()));

            // Load messages for the selected conversation
            let state = conv.read().clone();
            match state.load_messages(&id).await {
                Ok(messages) => {
                    conv.write().set_current_messages(messages);
                }
                Err(e) => {
                    tracing::error!("Failed to load messages: {}", e);
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
                match state.update_conversation_title(&conversation_id, &title).await {
                    Ok(updated) => {
                        conv.write().update_conversation(updated);
                    }
                    Err(e) => {
                        tracing::error!("Failed to rename: {}", e);
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

    // Handle confirmed delete
    let handle_confirm_delete = move |_| {
        let id = delete_confirm_id.read().clone();
        let mut conv = conversation_state;
        let mut confirm_id = delete_confirm_id;

        if let Some(conversation_id) = id {
            spawn(async move {
                conv.write().set_operating(true);
                let state = conv.read().clone();
                match state.delete_conversation(&conversation_id).await {
                    Ok(()) => {
                        conv.write().remove_conversation(&conversation_id);

                        // Auto-select the first remaining non-archived conversation
                        let remaining: Vec<_> = conv
                            .read()
                            .conversations
                            .iter()
                            .filter(|c| !c.is_archived)
                            .cloned()
                            .collect();

                        if let Some(next_conv) = remaining.first() {
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
                                    conv.write().set_current_messages(messages);
                                }
                                Err(e) => {
                                    tracing::error!(
                                        "[ConversationSidebar] Failed to load messages for auto-selected conversation: {}",
                                        e
                                    );
                                }
                            }
                        }

                        conv.write().set_operating(false);
                    }
                    Err(e) => {
                        conv.write().set_error(Some(e));
                        conv.write().set_operating(false);
                    }
                }
                confirm_id.set(None);
            });
        }
    };

    // Handle archive/unarchive
    let handle_toggle_archive = move |id: String, is_archived: bool| {
        let mut conv = conversation_state;
        spawn(async move {
            let state = conv.read().clone();
            match state.set_conversation_archived(&id, !is_archived).await {
                Ok(updated) => {
                    conv.write().update_conversation(updated);
                }
                Err(e) => {
                    tracing::error!("Failed to toggle archive: {}", e);
                }
            }
        });
    };

    // Get conversations to display (filter out archived)
    let state = conversation_state.read();
    let conversations: Vec<Conversation> = state.conversations.iter()
        .filter(|c| !c.is_archived)
        .cloned()
        .collect();
    let current_id = state.current_conversation_id.clone();
    let is_loading = state.is_loading;
    let is_operating = state.is_operating;

    // Track if we need to show delete dialog (read outside of sidebar div)
    let show_delete_dialog = delete_confirm_id.read().is_some();

    rsx! {
        // Use fragment to render sidebar and dialog as siblings
        // This ensures the dialog isn't constrained by sidebar's width

        div { class: "flex flex-col h-full w-fit min-w-[180px] max-w-[320px] bg-white dark:bg-gray-900 border-r border-gray-300 dark:border-gray-800",
            // Header - h-16 matches ChatPanel header height
            div { class: "flex-shrink-0 h-16 px-4 border-b border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-900",
                div { class: "h-full flex items-center justify-between",
                    div { class: "flex items-center gap-2",
                        P2aIconMinimal { size: 20.0 }
                        h2 { class: "text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider",
                            "Conversations"
                        }
                    }
                    div { class: "flex items-center gap-1",
                        // New conversation button
                        button {
                            class: "p-1 text-gray-500 hover:text-teal-600 dark:text-gray-400 dark:hover:text-teal-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors",
                            disabled: is_operating,
                            onclick: handle_new_conversation,
                            title: "New Conversation",
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M12 4v16m8-8H4"
                                }
                            }
                        }
                        // Collapse button
                        button {
                            class: "p-1 text-gray-400 hover:text-gray-600 dark:text-gray-500 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors",
                            onclick: move |_| sidebar_visible.set(false),
                            title: "Hide sidebar",
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M11 19l-7-7 7-7m8 14l-7-7 7-7"
                                }
                            }
                        }
                    }
                }
            }

            // Conversation list
            div { class: "flex-1 overflow-y-auto p-2",
                if is_loading {
                    div { class: "flex items-center justify-center py-8",
                        div { class: "w-6 h-6 border-2 border-gray-300 border-t-teal-500 rounded-full animate-spin" }
                    }
                } else if conversations.is_empty() {
                    div { class: "text-center py-8 px-2 text-gray-500 dark:text-gray-400",
                        p { class: "text-sm", "No conversations yet" }
                        p { class: "text-xs mt-1", "Click + to start a new one" }
                    }
                } else {
                    for conv in conversations.iter() {
                        {
                            let conv_id = conv.id.clone();
                            let conv_title = conv.title.clone();
                            let is_current = current_id.as_ref() == Some(&conv_id);
                            let is_editing = editing_id.read().as_ref() == Some(&conv_id);
                            let is_archived = conv.is_archived;
                            let message_count = conv.message_count;

                            rsx! {
                                div {
                                    key: "{conv_id}",
                                    class: if is_current {
                                        "group relative p-3 rounded-lg mb-1 cursor-pointer bg-teal-50 dark:bg-teal-900/30 border border-teal-400 dark:border-teal-700"
                                    } else {
                                        "group relative p-3 rounded-lg mb-1 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 border border-gray-200 dark:border-transparent"
                                    },
                                    onclick: {
                                        let id = conv_id.clone();
                                        move |_| handle_select(id.clone())
                                    },

                                    // Title row
                                    div { class: "flex items-center gap-2",
                                        if is_editing {
                                            input {
                                                class: "flex-1 px-2 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-teal-500",
                                                value: "{edit_title}",
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
                                                    "flex-1 text-sm font-medium text-gray-400 dark:text-gray-500 truncate italic"
                                                } else {
                                                    "flex-1 text-sm font-medium text-gray-800 dark:text-gray-200 truncate"
                                                },
                                                "{conv_title}"
                                            }

                                            // Archived badge
                                            if is_archived {
                                                span { class: "px-1.5 py-0.5 text-xs rounded bg-gray-200 dark:bg-gray-700 text-gray-500 dark:text-gray-400",
                                                    "archived"
                                                }
                                            }

                                            // Action buttons (shown on hover)
                                            div { class: "opacity-0 group-hover:opacity-100 flex items-center gap-0.5 transition-opacity",
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
                                        div { class: "mt-1 flex items-center gap-2",
                                            span { class: "text-xs text-gray-400 dark:text-gray-500",
                                                "{message_count} messages"
                                            }
                                        }
                                    }
                                }
                            }
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
                    class: "bg-white dark:bg-gray-800 rounded-xl shadow-2xl p-6 m-4 w-[90vw] max-w-md",
                    onclick: |e| e.stop_propagation(),
                    // Warning icon and title
                    div { class: "flex items-center gap-3 mb-4",
                        div { class: "p-2 rounded-full bg-red-100 dark:bg-red-900/30",
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
                        h3 { class: "text-lg font-semibold text-gray-900 dark:text-white",
                            "Delete Conversation?"
                        }
                    }
                    // Message
                    p { class: "text-sm text-gray-600 dark:text-gray-400 mb-6",
                        "This will permanently delete the conversation and all its messages. This action cannot be undone."
                    }
                    // Buttons - use grid to ensure both are always visible
                    div { class: "grid grid-cols-2 gap-3 mt-2",
                        button {
                            class: "px-4 py-2.5 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors cursor-pointer",
                            onclick: move |_| handle_cancel_delete(()),
                            "Cancel"
                        }
                        button {
                            class: "px-4 py-2.5 text-sm font-medium rounded-lg transition-colors cursor-pointer border-2 border-red-600 bg-red-600 text-white hover:bg-red-700 hover:border-red-700",
                            style: "background-color: #dc2626; color: white;",
                            onclick: handle_confirm_delete,
                            "Delete"
                        }
                    }
                }
            }
        }
    }
}
