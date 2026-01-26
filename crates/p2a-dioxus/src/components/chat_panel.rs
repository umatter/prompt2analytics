//! Main chat panel component

use dioxus::prelude::*;

use crate::api::{api, stream_chat, PersistedToolCall, StreamEvent};
use crate::app::apply_theme;
use crate::components::{ChatInput, MessageList, P2aWordmark, SettingsModal};
use crate::state::settings::Theme;
use crate::state::{ChatMessage, ChatState, ConversationState, SessionState, Settings, ToolCallInfo};

/// Chat panel component - main chat interface
#[component]
pub fn ChatPanel() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let mut chat_state = use_context::<Signal<ChatState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();
    let mut settings = use_context::<Signal<Settings>>();

    let mut settings_open = use_signal(|| false);
    let mut settings_initial_tab = use_signal(|| "llm".to_string());

    // Track which conversation we've synced from and how many messages we've synced
    let mut synced_conv_id = use_signal(|| Option::<String>::None);
    let mut synced_messages_len = use_signal(|| 0usize);

    // Sync messages from conversation state to chat state when:
    // 1. User explicitly selects a DIFFERENT conversation (not the same one)
    // 2. OR when messages are loaded for the current conversation
    // 3. AND we're not in the middle of streaming/processing
    //
    // IMPORTANT: This effect should NEVER clear chat_state.messages if they exist,
    // unless the user is switching to a different conversation.
    use_effect(move || {
        let conv_state = conversation_state.read();
        let current_conv_id = conv_state.current_conversation_id.clone();
        let current_messages = conv_state.current_messages.clone();
        let current_messages_len = current_messages.len();
        let synced_id = synced_conv_id.read().clone();
        let prev_messages_len = *synced_messages_len.read();

        // Don't interfere with active streaming/processing
        // BUT still update synced_conv_id to prevent re-triggering when processing completes
        let is_processing = chat_state.read().is_processing;
        if is_processing {
            tracing::debug!("[ChatPanel Sync] Skipping - is_processing=true, but updating synced_conv_id");
            // Update synced tracking to match current state so we don't re-trigger after processing
            if current_conv_id != synced_id {
                synced_conv_id.set(current_conv_id.clone());
                synced_messages_len.set(0); // Reset since we're not loading messages now
            }
            return;
        }

        // Check if conversation changed OR if messages were loaded for current conversation
        let conv_changed = current_conv_id != synced_id;
        let messages_loaded = current_conv_id == synced_id && current_messages_len > 0 && prev_messages_len == 0;

        // Also check if chat_state already has messages (don't clear user's in-progress work)
        let chat_has_messages = !chat_state.read().messages.is_empty();
        let chat_message_count = chat_state.read().messages.len();

        tracing::debug!(
            "[ChatPanel Sync] conv_changed={}, messages_loaded={}, chat_has_messages={}, chat_message_count={}, current_conv_id={:?}, synced_id={:?}, current_messages_len={}, prev_messages_len={}",
            conv_changed, messages_loaded, chat_has_messages, chat_message_count, current_conv_id, synced_id, current_messages_len, prev_messages_len
        );

        if conv_changed || messages_loaded {
            if let Some(conv_id) = current_conv_id.clone() {
                let mut chat = chat_state;
                let conv_id_for_async = conv_id.clone();

                // Determine if this is a real conversation switch (user selected different conv)
                let is_switching_conversations = synced_id.is_some() && synced_id != current_conv_id;

                if !current_messages.is_empty() {
                    // Load persisted messages from database
                    tracing::debug!("[ChatPanel Sync] Loading {} persisted messages", current_messages_len);
                    chat.write().clear_messages();

                    for msg in current_messages.iter() {
                        let chat_msg = ChatMessage::from_conversation_message(msg);
                        chat.write().messages.push(chat_msg);
                    }

                    // Load tool calls asynchronously
                    spawn(async move {
                        let client = api();
                        match client.get_conversation_tool_calls(&conv_id_for_async).await {
                            Ok(tool_calls) => {
                                if !tool_calls.is_empty() {
                                    tracing::debug!("[ChatPanel] Loaded {} tool calls", tool_calls.len());

                                    let mut tool_calls_by_message: std::collections::HashMap<String, Vec<PersistedToolCall>> =
                                        std::collections::HashMap::new();
                                    for tc in tool_calls {
                                        tool_calls_by_message
                                            .entry(tc.message_id.clone())
                                            .or_default()
                                            .push(tc);
                                    }

                                    let mut chat_write = chat.write();
                                    for msg in chat_write.messages.iter_mut() {
                                        if let Some(tcs) = tool_calls_by_message.get(&msg.id) {
                                            msg.tool_calls = tcs.iter().map(ToolCallInfo::from).collect();
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("[ChatPanel] Failed to load tool calls: {}", e);
                            }
                        }
                    });

                    synced_messages_len.set(current_messages_len);
                } else if is_switching_conversations && !chat_has_messages {
                    // Switching to a new/empty conversation AND chat is already empty
                    // Just update tracking, don't clear (nothing to clear)
                    tracing::debug!("[ChatPanel Sync] Switching to empty conversation, chat already empty");
                    synced_messages_len.set(0);
                } else if is_switching_conversations && chat_has_messages {
                    // Switching to a different conversation that has no persisted messages yet
                    // Clear the old conversation's messages
                    tracing::debug!("[ChatPanel Sync] Switching conversations, clearing old messages");
                    chat.write().clear_messages();
                    synced_messages_len.set(0);
                }
                // If NOT switching conversations and current_messages is empty,
                // preserve whatever is in chat_state (user might be typing/streaming)

                synced_conv_id.set(current_conv_id);
            } else {
                // No conversation selected, clear chat
                tracing::debug!("[ChatPanel Sync] No conversation selected, clearing chat");
                chat_state.write().clear_messages();
                synced_conv_id.set(None);
                synced_messages_len.set(0);
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
        tracing::debug!("[ChatPanel] handle_send called with: {}", message);
        let mut session = session_state;
        let mut chat = chat_state;
        let mut conv = conversation_state;
        let settings = settings;

        spawn(async move {
            tracing::debug!("[ChatPanel] spawn started");

            // IMPORTANT: Start processing FIRST to prevent the sync effect from
            // interfering with our messages. The sync effect checks is_processing
            // and returns early if true.
            tracing::debug!("[ChatPanel] Starting processing EARLY to protect messages");
            chat.write().start_processing();

            // Check if we already have a session ID (without holding write guard across await)
            let existing_id = session.read().session_id.clone();

            let session_id = if let Some(id) = existing_id {
                tracing::debug!("[ChatPanel] Using existing session: {}", id);
                id
            } else {
                // Create new session - do the async work first, then update state
                tracing::debug!("[ChatPanel] Creating new session");
                let state = session.read().clone();
                match state.initialize().await {
                    Ok(id) => {
                        tracing::debug!("[ChatPanel] Created session: {}", id);
                        session.write().set_session_id(id.clone());
                        id
                    }
                    Err(e) => {
                        tracing::error!("[ChatPanel] Session error: {}", e);
                        chat.write().set_error(Some(format!("Session error: {}", e)));
                        chat.write().stop_processing();
                        return;
                    }
                }
            };

            // Ensure we have a conversation
            let conversation_id = {
                let conv_state = conv.read();
                if let Some(id) = conv_state.current_conversation_id.clone() {
                    tracing::debug!("[ChatPanel] Using existing conversation: {}", id);
                    id
                } else {
                    // Create a new conversation if none selected
                    drop(conv_state); // Release the read lock
                    tracing::debug!("[ChatPanel] Creating new conversation");
                    let client = api();
                    match client.create_conversation(&session_id, "New Conversation").await {
                        Ok(new_conv) => {
                            let id = new_conv.id.clone();
                            tracing::debug!("[ChatPanel] Created conversation: {}", id);
                            conv.write().add_conversation(new_conv);
                            conv.write().set_current_conversation(Some(id.clone()));
                            id
                        }
                        Err(e) => {
                            tracing::warn!("[ChatPanel] Failed to create conversation: {}", e);
                            // Continue without persistence
                            String::new()
                        }
                    }
                }
            };

            // Now add the messages (after conversation is set up)
            tracing::debug!("[ChatPanel] Adding user message and streaming placeholder");
            chat.write().add_user_message(&message);
            chat.write().add_streaming_message();
            tracing::debug!("[ChatPanel] Messages added, count: {}", chat.read().messages.len());

            // Persist user message to conversation and update local ID
            if !conversation_id.is_empty() {
                let conv_id = conversation_id.clone();
                let msg = message.clone();
                let mut chat_for_user_id = chat;
                spawn(async move {
                    let client = api();
                    match client.add_message(&conv_id, "user", &msg).await {
                        Ok(persisted_msg) => {
                            // Find and update the user message ID
                            let mut chat_write = chat_for_user_id.write();
                            // Find the user message with this content (should be second-to-last, before streaming assistant)
                            for m in chat_write.messages.iter_mut().rev() {
                                if m.role == "user" && m.content == msg {
                                    tracing::debug!(
                                        "[ChatPanel] Updating user message ID from {} to {}",
                                        m.id,
                                        persisted_msg.id
                                    );
                                    m.id = persisted_msg.id;
                                    break;
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("[ChatPanel] Failed to persist user message: {}", e);
                        }
                    }
                });
            }

            // Build history
            let history = chat.read().build_history();
            let provider_config = settings.read().to_provider_config();
            let interpret = settings.read().interpret_results;

            tracing::debug!("[ChatPanel] Calling stream_chat");

            // Clone conversation_id for the closure
            let conv_id_for_done = conversation_id.clone();

            // Stream the response
            let result = stream_chat(
                "http://localhost:8081",
                &session_id,
                &message,
                history,
                provider_config,
                interpret,
                Some(conversation_id.clone()),
                |event| {
                    tracing::debug!("[ChatPanel] Got event: {:?}", std::mem::discriminant(&event));
                    match event {
                        StreamEvent::Content { text } => {
                            chat.write().append_content(&text);
                        }
                        StreamEvent::Status { message } => {
                            chat.write().set_status(Some(message));
                        }
                        StreamEvent::ToolStart { tool, arguments } => {
                            chat.write().set_active_tool(tool, arguments);
                        }
                        StreamEvent::ToolEnd { tool: _, elapsed_ms: _, result } => {
                            chat.write().clear_active_tool(result);
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

                            // Persist assistant message to conversation and update local ID
                            if !conv_id_for_done.is_empty() {
                                let conv_id = conv_id_for_done.clone();
                                let content = message.content.clone();
                                let mut chat_for_id = chat;
                                spawn(async move {
                                    let client = api();
                                    match client.add_message(&conv_id, "assistant", &content).await {
                                        Ok(persisted_msg) => {
                                            // Update the local message ID to match the backend ID
                                            // This ensures tool calls can be matched correctly
                                            if let Some(last) = chat_for_id.write().messages.last_mut() {
                                                if last.role == "assistant" {
                                                    tracing::debug!(
                                                        "[ChatPanel] Updating message ID from {} to {}",
                                                        last.id,
                                                        persisted_msg.id
                                                    );
                                                    last.id = persisted_msg.id;
                                                }
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!("[ChatPanel] Failed to persist assistant message: {}", e);
                                        }
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

            tracing::debug!("[ChatPanel] stream_chat returned: {:?}", result.is_ok());
            if let Err(e) = result {
                tracing::error!("[ChatPanel] Stream error: {}", e);
                chat.write().set_error(Some(e.to_string()));
            }

            chat.write().stop_processing();

            // Trigger dataset sidebar refresh in case tools created/modified datasets
            session.write().trigger_datasets_refresh();

            // Generate title for new conversations after first exchange
            if !conversation_id.is_empty() {
                let conv_id = conversation_id.clone();
                let user_msg = message.clone();
                let settings_clone = settings.read().clone();
                let mut conv_state = conv;

                // Check if conversation still has default title
                let needs_title = conv_state.read()
                    .get_current_conversation()
                    .map(|c| c.title == "New Conversation")
                    .unwrap_or(false);

                if needs_title {
                    // Get the assistant's response from chat state
                    let assistant_response = chat.read()
                        .messages
                        .iter()
                        .filter(|m| m.role == "assistant")
                        .last()
                        .map(|m| m.content.clone());

                    spawn(async move {
                        let client = api();

                        // Generate title using LLM
                        match client.generate_title(
                            &user_msg,
                            assistant_response.as_deref(),
                            Some(settings_clone.to_provider_config()),
                        ).await {
                            Ok(new_title) => {
                                tracing::info!("[ChatPanel] Generated title: {}", new_title);
                                // Update conversation with new title
                                match client.update_conversation(&conv_id, Some(&new_title), None).await {
                                    Ok(updated) => {
                                        conv_state.write().update_conversation(updated);
                                    }
                                    Err(e) => {
                                        tracing::warn!("[ChatPanel] Failed to update conversation title: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("[ChatPanel] Failed to generate title: {}", e);
                            }
                        }
                    });
                }
            }

            tracing::debug!("[ChatPanel] Processing stopped");
        });
    };

    // Handle clear error
    let handle_clear_error = move |_| {
        chat_state.write().set_error(None);
    };

    // Handle retry - re-send the last user message
    let handle_retry = {
        let handle_send = handle_send.clone();
        move |_| {
            // Find the last user message
            let last_user_msg = chat_state.read()
                .messages
                .iter()
                .filter(|m| m.role == "user")
                .last()
                .map(|m| m.content.clone());

            if let Some(msg) = last_user_msg {
                // Clear the error first
                chat_state.write().set_error(None);
                // Remove the failed assistant message if any (last message that might be incomplete)
                {
                    let mut chat = chat_state.write();
                    if let Some(last) = chat.messages.last() {
                        if last.role == "assistant" && last.content.is_empty() {
                            chat.messages.pop();
                        }
                    }
                }
                // Re-send the message
                handle_send(msg);
            }
        }
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

    // Handle theme toggle - cycles through System → Light → Dark → System
    let handle_theme_toggle = move |_| {
        let current = settings.read().theme;
        let next = match current {
            Theme::System => Theme::Light,
            Theme::Light => Theme::Dark,
            Theme::Dark => Theme::System,
        };
        settings.write().theme = next;
        settings.read().save();
        apply_theme(next);
    };

    rsx! {
        div { class: "flex flex-col h-screen bg-white dark:bg-gray-900 shadow-xl",
            // Header - fixed height for alignment with DatasetSidebar
            header { class: "flex-shrink-0 h-16 px-6 border-b border-gray-300 dark:border-gray-800 bg-gray-50 dark:bg-gray-900",
                div { class: "h-full w-full flex items-center justify-between",
                    div { class: "flex items-center gap-3",
                        // Logo/Wordmark
                        P2aWordmark { width: 140.0, class: "shrink-0" }
                        // Current conversation title
                        if let Some(ref conv) = current_conversation {
                            span { class: "text-sm text-gray-500 dark:text-gray-400",
                                "/ {conv.title}"
                            }
                        }
                        // Provider badge - with border for visual weight
                        span { class: "px-2.5 py-1 text-xs font-medium rounded-full bg-orange-100 text-orange-700 border border-orange-300 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-700",
                            "{settings.read().provider.as_str()} / {settings.read().current_model()}"
                        }
                    }

                    div { class: "flex items-center gap-2",
                        // Load Data button
                        button {
                            class: "px-3 py-1.5 text-sm font-medium text-white bg-orange-600 rounded-lg hover:bg-orange-700 transition-colors flex items-center gap-1.5",
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

                        // Theme toggle button
                        button {
                            class: "p-2 text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 dark:text-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors",
                            onclick: handle_theme_toggle,
                            title: match settings.read().theme {
                                Theme::System => "Theme: System (click to switch)",
                                Theme::Light => "Theme: Light (click to switch)",
                                Theme::Dark => "Theme: Dark (click to switch)",
                            },
                            // Show different icons based on current theme
                            match settings.read().theme {
                                Theme::System => rsx! {
                                    // Computer/monitor icon for system
                                    svg {
                                        class: "w-5 h-5",
                                        fill: "none",
                                        stroke: "currentColor",
                                        view_box: "0 0 24 24",
                                        path {
                                            stroke_linecap: "round",
                                            stroke_linejoin: "round",
                                            stroke_width: "2",
                                            d: "M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                                        }
                                    }
                                },
                                Theme::Light => rsx! {
                                    // Sun icon for light mode
                                    svg {
                                        class: "w-5 h-5",
                                        fill: "none",
                                        stroke: "currentColor",
                                        view_box: "0 0 24 24",
                                        path {
                                            stroke_linecap: "round",
                                            stroke_linejoin: "round",
                                            stroke_width: "2",
                                            d: "M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"
                                        }
                                    }
                                },
                                Theme::Dark => rsx! {
                                    // Moon icon for dark mode
                                    svg {
                                        class: "w-5 h-5",
                                        fill: "none",
                                        stroke: "currentColor",
                                        view_box: "0 0 24 24",
                                        path {
                                            stroke_linecap: "round",
                                            stroke_linejoin: "round",
                                            stroke_width: "2",
                                            d: "M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"
                                        }
                                    }
                                },
                            }
                        }

                        // Settings button (icon-only like theme toggle)
                        button {
                            class: "p-2 text-gray-600 bg-gray-100 rounded-lg hover:bg-gray-200 dark:text-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 transition-colors",
                            onclick: handle_open_settings,
                            title: "Settings",
                            svg {
                                class: "w-5 h-5",
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
                        }
                    }
                }
            }

            // Loaded datasets indicator (outside fixed-height header)
            {
                let datasets = session_state.read().loaded_datasets.clone();
                rsx! {
                    if !datasets.is_empty() {
                        div { class: "flex items-center gap-2 px-6 py-2 flex-wrap bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-800",
                            span { class: "text-xs text-gray-500 dark:text-gray-400", "Datasets:" }
                            for ds in datasets.iter() {
                                span {
                                    class: "inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300",
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
                                    span { class: "text-teal-600 dark:text-teal-400",
                                        "({ds.rows}×{ds.cols})"
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Error banner with retry button
            if let Some(ref err) = error {
                div { class: "mx-4 mt-4 px-4 py-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800",
                    div { class: "flex items-center justify-between",
                        div { class: "flex items-center gap-2 flex-1 min-w-0",
                            svg {
                                class: "w-5 h-5 text-red-500 flex-shrink-0",
                                fill: "currentColor",
                                view_box: "0 0 20 20",
                                path {
                                    fill_rule: "evenodd",
                                    d: "M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z",
                                    clip_rule: "evenodd"
                                }
                            }
                            span { class: "text-sm text-red-700 dark:text-red-300 truncate", "{err}" }
                        }
                        div { class: "flex items-center gap-2 flex-shrink-0 ml-2",
                            // Retry button
                            button {
                                class: "px-3 py-1 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors flex items-center gap-1",
                                onclick: handle_retry,
                                svg {
                                    class: "w-4 h-4",
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
                                "Retry"
                            }
                            // Dismiss button
                            button {
                                class: "p-1 text-red-500 hover:text-red-700 dark:hover:text-red-300 rounded transition-colors",
                                onclick: handle_clear_error,
                                title: "Dismiss",
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
                }
            }

            // Status indicator
            if let Some(ref stat) = status {
                div { class: "mx-4 mt-2 px-3 py-2 rounded-lg bg-teal-50 dark:bg-teal-900/20 flex items-center gap-2",
                    div { class: "w-2 h-2 rounded-full bg-teal-500 animate-pulse" }
                    span { class: "text-sm text-teal-700 dark:text-teal-300", "{stat}" }
                }
            }

            // Active tool indicator (orange for Rust analytics branding)
            {
                let active_tool = chat_state.read().active_tool.clone();
                rsx! {
                    if let Some(ref tool) = active_tool {
                        div { class: "mx-4 mt-2 px-3 py-2 rounded-lg bg-orange-50 dark:bg-orange-900/20 border border-orange-200 dark:border-orange-700 flex items-center gap-2",
                            svg {
                                class: "w-4 h-4 text-orange-600 dark:text-orange-400 animate-spin",
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
                            span { class: "text-sm font-medium text-orange-700 dark:text-orange-300",
                                "Running: "
                            }
                            span { class: "text-sm text-orange-600 dark:text-orange-400 font-mono",
                                "{tool.name}"
                            }
                        }
                    }
                }
            }

            // Message list (scrollable)
            div { class: "flex-1 overflow-y-auto",
                MessageList {
                    on_suggestion: move |text: String| {
                        handle_send(text);
                    }
                }
            }

            // Input area
            div { class: "flex-shrink-0 border-t border-gray-300 dark:border-gray-800 bg-gray-50 dark:bg-gray-900",
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
