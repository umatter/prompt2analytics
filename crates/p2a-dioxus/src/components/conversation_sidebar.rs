//! Conversation sidebar component
//!
//! Displays a list of conversations and allows creating, selecting,
//! and managing conversations.

use dioxus::prelude::*;

use crate::api::Conversation;
use crate::state::{ConversationState, SessionState};

/// Conversation sidebar component
#[component]
pub fn ConversationSidebar() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();

    // Editing state for rename
    let mut editing_id = use_signal(|| Option::<String>::None);
    let mut edit_title = use_signal(String::new);

    // Show archived toggle
    let mut show_archived = use_signal(|| false);

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

    // Handle delete conversation
    let handle_delete = move |id: String| {
        let mut conv = conversation_state;
        spawn(async move {
            conv.write().set_operating(true);
            let state = conv.read().clone();
            match state.delete_conversation(&id).await {
                Ok(()) => {
                    conv.write().remove_conversation(&id);
                    conv.write().set_operating(false);
                }
                Err(e) => {
                    conv.write().set_error(Some(e));
                }
            }
        });
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

    // Get conversations to display
    let state = conversation_state.read();
    let conversations: Vec<Conversation> = if *show_archived.read() {
        state.conversations.clone()
    } else {
        state.conversations.iter().filter(|c| !c.is_archived).cloned().collect()
    };
    let current_id = state.current_conversation_id.clone();
    let is_loading = state.is_loading;
    let is_operating = state.is_operating;

    rsx! {
        div { class: "flex flex-col h-full bg-gray-100 dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700",
            // Header
            div { class: "flex-shrink-0 p-4 border-b border-gray-200 dark:border-gray-700",
                div { class: "flex items-center justify-between mb-3",
                    h2 { class: "text-sm font-semibold text-gray-700 dark:text-gray-300 uppercase tracking-wider",
                        "Conversations"
                    }
                    button {
                        class: "p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg transition-colors",
                        disabled: is_operating,
                        onclick: handle_new_conversation,
                        title: "New Conversation",
                        svg {
                            class: "w-5 h-5",
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
                }

                // Show archived toggle
                label { class: "flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400 cursor-pointer",
                    input {
                        r#type: "checkbox",
                        class: "rounded text-blue-600 focus:ring-blue-500 dark:focus:ring-blue-400",
                        checked: *show_archived.read(),
                        onchange: move |e| show_archived.set(e.checked())
                    }
                    "Show archived"
                }
            }

            // Conversation list
            div { class: "flex-1 overflow-y-auto p-2",
                if is_loading {
                    div { class: "flex items-center justify-center py-8",
                        div { class: "w-6 h-6 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin" }
                    }
                } else if conversations.is_empty() {
                    div { class: "text-center py-8 text-gray-500 dark:text-gray-400",
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
                            let preview = conv.last_message_preview.clone();

                            rsx! {
                                div {
                                    key: "{conv_id}",
                                    class: if is_current {
                                        "group relative p-3 rounded-lg mb-1 cursor-pointer bg-blue-100 dark:bg-blue-900/40 border border-blue-300 dark:border-blue-700"
                                    } else {
                                        "group relative p-3 rounded-lg mb-1 cursor-pointer hover:bg-gray-200 dark:hover:bg-gray-800 border border-transparent"
                                    },
                                    onclick: {
                                        let id = conv_id.clone();
                                        move |_| handle_select(id.clone())
                                    },

                                    // Title row
                                    div { class: "flex items-center gap-2",
                                        if is_editing {
                                            input {
                                                class: "flex-1 px-2 py-1 text-sm rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-blue-500",
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
                                                class: "p-1 text-green-600 hover:text-green-700 dark:text-green-400",
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
                                            div { class: "opacity-0 group-hover:opacity-100 flex items-center gap-1 transition-opacity",
                                                // Rename button
                                                button {
                                                    class: "p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300",
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
                                                        class: "w-3.5 h-3.5",
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
                                                    class: "p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300",
                                                    title: if is_archived { "Unarchive" } else { "Archive" },
                                                    onclick: {
                                                        let id = conv_id.clone();
                                                        move |e: Event<MouseData>| {
                                                            e.stop_propagation();
                                                            handle_toggle_archive(id.clone(), is_archived)
                                                        }
                                                    },
                                                    svg {
                                                        class: "w-3.5 h-3.5",
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
                                                    class: "p-1 text-gray-400 hover:text-red-500",
                                                    title: "Delete",
                                                    onclick: {
                                                        let id = conv_id.clone();
                                                        move |e: Event<MouseData>| {
                                                            e.stop_propagation();
                                                            handle_delete(id.clone())
                                                        }
                                                    },
                                                    svg {
                                                        class: "w-3.5 h-3.5",
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

                                    // Message count and preview
                                    if !is_editing {
                                        div { class: "mt-1 flex items-center gap-2",
                                            span { class: "text-xs text-gray-400 dark:text-gray-500",
                                                "{message_count} messages"
                                            }
                                        }
                                        if let Some(ref prev) = preview {
                                            p { class: "mt-1 text-xs text-gray-500 dark:text-gray-400 truncate",
                                                "{prev}"
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
    }
}
