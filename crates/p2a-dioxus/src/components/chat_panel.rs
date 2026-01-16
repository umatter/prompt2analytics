//! Main chat panel component

use dioxus::prelude::*;
use wasm_bindgen::JsValue;

use crate::api::{api, stream_chat, StreamEvent};
use crate::components::{ChatInput, MessageList, SettingsModal};
use crate::state::{ChatMessage, ChatState, ConversationState, SessionState, Settings};

fn log(msg: &str) {
    web_sys::console::log_1(&JsValue::from_str(msg));
}

/// Chat panel component - main chat interface
#[component]
pub fn ChatPanel() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let mut chat_state = use_context::<Signal<ChatState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();
    let settings = use_context::<Signal<Settings>>();

    let mut settings_open = use_signal(|| false);
    let mut settings_initial_tab = use_signal(|| "llm".to_string());

    // Track the last loaded conversation to detect changes
    let mut last_loaded_conv_id = use_signal(|| Option::<String>::None);

    // Load messages when conversation changes
    use_effect(move || {
        let conv_state = conversation_state.read();
        let current_conv_id = conv_state.current_conversation_id.clone();
        let last_id = last_loaded_conv_id.read().clone();

        // If the conversation changed, load its messages
        if current_conv_id != last_id {
            if let Some(ref conv_id) = current_conv_id {
                let messages = conv_state.current_messages.clone();
                let mut chat = chat_state;

                // Clear existing messages and load from conversation
                chat.write().clear_messages();

                // Convert conversation messages to chat messages
                for msg in messages {
                    let chat_msg = ChatMessage::from_conversation_message(&msg);
                    chat.write().messages.push(chat_msg);
                }

                last_loaded_conv_id.set(current_conv_id.clone());
            } else {
                // No conversation selected, clear chat
                chat_state.write().clear_messages();
                last_loaded_conv_id.set(None);
            }
        }
    });

    // Get current status and error
    let status = chat_state.read().status.clone();
    let error = chat_state.read().error.clone();

    // Get current conversation title
    let current_conversation = conversation_state.read().get_current_conversation().cloned();

    // Handle send message
    let handle_send = move |message: String| {
        log(&format!("[ChatPanel] handle_send called with: {}", message));
        let mut session = session_state;
        let mut chat = chat_state;
        let mut conv = conversation_state;
        let settings = settings;

        spawn(async move {
            log("[ChatPanel] spawn started");

            // Check if we already have a session ID (without holding write guard across await)
            let existing_id = session.read().session_id.clone();

            let session_id = if let Some(id) = existing_id {
                log(&format!("[ChatPanel] Using existing session: {}", id));
                id
            } else {
                // Create new session - do the async work first, then update state
                log("[ChatPanel] Creating new session");
                let state = session.read().clone();
                match state.initialize().await {
                    Ok(id) => {
                        log(&format!("[ChatPanel] Created session: {}", id));
                        session.write().set_session_id(id.clone());
                        id
                    }
                    Err(e) => {
                        log(&format!("[ChatPanel] Session error: {}", e));
                        chat.write().set_error(Some(format!("Session error: {}", e)));
                        return;
                    }
                }
            };

            // Ensure we have a conversation
            let conversation_id = {
                let conv_state = conv.read();
                if let Some(id) = conv_state.current_conversation_id.clone() {
                    id
                } else {
                    // Create a new conversation if none selected
                    drop(conv_state); // Release the read lock
                    let client = api();
                    match client.create_conversation(&session_id, "New Conversation").await {
                        Ok(new_conv) => {
                            let id = new_conv.id.clone();
                            conv.write().add_conversation(new_conv);
                            conv.write().set_current_conversation(Some(id.clone()));
                            id
                        }
                        Err(e) => {
                            log(&format!("[ChatPanel] Failed to create conversation: {}", e));
                            // Continue without persistence
                            String::new()
                        }
                    }
                }
            };

            // Start processing
            log("[ChatPanel] Starting processing");
            chat.write().start_processing();
            chat.write().add_user_message(&message);
            chat.write().add_streaming_message();

            // Persist user message to conversation (fire and forget)
            if !conversation_id.is_empty() {
                let conv_id = conversation_id.clone();
                let msg = message.clone();
                spawn(async move {
                    let client = api();
                    if let Err(e) = client.add_message(&conv_id, "user", &msg).await {
                        log(&format!("[ChatPanel] Failed to persist user message: {}", e));
                    }
                });
            }

            // Build history
            let history = chat.read().build_history();
            let provider_config = settings.read().to_provider_config();
            let interpret = settings.read().interpret_results;

            log("[ChatPanel] Calling stream_chat");

            // Clone conversation_id for the closure
            let conv_id_for_done = conversation_id.clone();

            // Stream the response
            let result = stream_chat(
                "http://localhost:8080",
                &session_id,
                &message,
                history,
                provider_config,
                interpret,
                |event| {
                    log(&format!("[ChatPanel] Got event: {:?}", std::mem::discriminant(&event)));
                    match event {
                        StreamEvent::Content { text } => {
                            chat.write().append_content(&text);
                        }
                        StreamEvent::Status { message } => {
                            chat.write().set_status(Some(message));
                        }
                        StreamEvent::ToolStart { tool } => {
                            chat.write().set_active_tool(tool);
                        }
                        StreamEvent::ToolEnd { tool: _, elapsed_ms: _ } => {
                            chat.write().clear_active_tool();
                        }
                        StreamEvent::ToolResult { images } => {
                            if let Some(imgs) = images {
                                for img in imgs {
                                    chat.write().add_image(&img.data);
                                }
                            }
                        }
                        StreamEvent::Done { message } => {
                            chat.write().finalize_message(message.clone());

                            // Persist assistant message to conversation
                            if !conv_id_for_done.is_empty() {
                                let conv_id = conv_id_for_done.clone();
                                let content = message.content.clone();
                                spawn(async move {
                                    let client = api();
                                    if let Err(e) = client.add_message(&conv_id, "assistant", &content).await {
                                        log(&format!("[ChatPanel] Failed to persist assistant message: {}", e));
                                    }
                                });
                            }
                        }
                        StreamEvent::Error { error } => {
                            chat.write().set_error(Some(error));
                        }
                    }
                },
            )
            .await;

            log(&format!("[ChatPanel] stream_chat returned: {:?}", result.is_ok()));
            if let Err(e) = result {
                log(&format!("[ChatPanel] Stream error: {}", e));
                chat.write().set_error(Some(e));
            }

            chat.write().stop_processing();
            log("[ChatPanel] Processing stopped");
        });
    };

    // Handle clear error
    let handle_clear_error = move |_| {
        chat_state.write().set_error(None);
    };

    // Handle open settings
    let handle_open_settings = move |_| {
        settings_initial_tab.set("llm".to_string());
        settings_open.set(true);
    };

    // Handle open load data
    let handle_open_load_data = move |_| {
        settings_initial_tab.set("data".to_string());
        settings_open.set(true);
    };

    // Handle close settings
    let handle_close_settings = move |_| {
        settings_open.set(false);
    };

    // Handle clear messages
    let handle_clear_messages = move |_| {
        let mut chat = chat_state;
        let mut conv = conversation_state;

        // Clear local chat state
        chat.write().clear_messages();

        // Clear messages in the backend conversation
        spawn(async move {
            let conv_id = conv.read().current_conversation_id.clone();
            if let Some(id) = conv_id {
                let client = api();
                if let Err(e) = client.clear_messages(&id).await {
                    tracing::error!("Failed to clear conversation messages: {}", e);
                }
                conv.write().clear_cached_messages();
            }
        });
    };

    rsx! {
        div { class: "flex flex-col h-screen bg-white dark:bg-gray-800 shadow-xl",
            // Header
            header { class: "flex-shrink-0 px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800",
                div { class: "flex items-center justify-between",
                    div { class: "flex items-center gap-3",
                        // Logo/Title
                        h1 { class: "text-xl font-bold text-gray-900 dark:text-white",
                            "prompt2analytics"
                        }
                        // Current conversation title
                        if let Some(ref conv) = current_conversation {
                            span { class: "text-sm text-gray-500 dark:text-gray-400",
                                "/ {conv.title}"
                            }
                        }
                        // Provider badge
                        span { class: "px-2.5 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
                            "{settings.read().provider.as_str()} / {settings.read().current_model()}"
                        }
                    }

                    div { class: "flex items-center gap-2",
                        // Load Data button
                        button {
                            class: "px-3 py-1.5 text-sm font-medium text-white bg-green-600 rounded-lg hover:bg-green-700 transition-colors flex items-center gap-1.5",
                            onclick: handle_open_load_data,
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"
                                }
                            }
                            "Load Data"
                        }

                        // Clear button
                        button {
                            class: "px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 dark:text-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors",
                            onclick: handle_clear_messages,
                            "Clear"
                        }

                        // Settings button
                        button {
                            class: "px-3 py-1.5 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 dark:text-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors flex items-center gap-1.5",
                            onclick: handle_open_settings,
                            svg {
                                class: "w-4 h-4",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
                                }
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                                }
                            }
                            "Settings"
                        }
                    }
                }

                // Loaded datasets indicator
                {
                    let datasets = session_state.read().loaded_datasets.clone();
                    rsx! {
                        if !datasets.is_empty() {
                            div { class: "flex items-center gap-2 mt-2 flex-wrap",
                                span { class: "text-xs text-gray-500 dark:text-gray-400", "Datasets:" }
                                for ds in datasets.iter() {
                                    span {
                                        class: "inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300",
                                        svg {
                                            class: "w-3 h-3",
                                            fill: "none",
                                            stroke: "currentColor",
                                            view_box: "0 0 24 24",
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M4 7v10c0 2 1 3 3 3h10c2 0 3-1 3-3V7c0-2-1-3-3-3H7c-2 0-3 1-3 3z"
                                            }
                                            path {
                                                stroke_linecap: "round",
                                                stroke_linejoin: "round",
                                                stroke_width: "2",
                                                d: "M4 7h16"
                                            }
                                        }
                                        "{ds.name}"
                                        span { class: "text-green-600 dark:text-green-400",
                                            "({ds.rows}×{ds.cols})"
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Error banner
            if let Some(ref err) = error {
                div { class: "mx-4 mt-4 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 flex items-center justify-between",
                    div { class: "flex items-center gap-2",
                        svg {
                            class: "w-5 h-5 text-red-500",
                            fill: "currentColor",
                            view_box: "0 0 20 20",
                            path {
                                fill_rule: "evenodd",
                                d: "M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z",
                                clip_rule: "evenodd"
                            }
                        }
                        span { class: "text-sm text-red-700 dark:text-red-300", "{err}" }
                    }
                    button {
                        class: "p-1 text-red-500 hover:text-red-700 dark:hover:text-red-300 rounded transition-colors",
                        onclick: handle_clear_error,
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
                }
            }

            // Status indicator
            if let Some(ref stat) = status {
                div { class: "mx-4 mt-2 px-3 py-2 rounded-lg bg-blue-50 dark:bg-blue-900/20 flex items-center gap-2",
                    div { class: "w-2 h-2 rounded-full bg-blue-500 animate-pulse" }
                    span { class: "text-sm text-blue-700 dark:text-blue-300", "{stat}" }
                }
            }

            // Active tool indicator
            {
                let active_tool = chat_state.read().active_tool.clone();
                rsx! {
                    if let Some(ref tool) = active_tool {
                        div { class: "mx-4 mt-2 px-3 py-2 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 flex items-center gap-2",
                            svg {
                                class: "w-4 h-4 text-amber-600 dark:text-amber-400 animate-spin",
                                fill: "none",
                                stroke: "currentColor",
                                view_box: "0 0 24 24",
                                path {
                                    stroke_linecap: "round",
                                    stroke_linejoin: "round",
                                    stroke_width: "2",
                                    d: "M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                }
                            }
                            span { class: "text-sm font-medium text-amber-700 dark:text-amber-300",
                                "Running: "
                            }
                            span { class: "text-sm text-amber-600 dark:text-amber-400 font-mono",
                                "{tool.name}"
                            }
                        }
                    }
                }
            }

            // Message list (scrollable)
            div { class: "flex-1 overflow-y-auto",
                MessageList {}
            }

            // Input area
            div { class: "flex-shrink-0 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800",
                ChatInput { on_send: handle_send }
            }

            // Settings modal
            SettingsModal {
                is_open: *settings_open.read(),
                initial_tab: settings_initial_tab.read().clone(),
                on_close: handle_close_settings
            }
        }
    }
}
