//! Main chat panel component

use dioxus::prelude::*;

use crate::api::{
    PersistedToolCall, StreamErrorKind, StreamEvent, StreamStop, api, stop_pair, stream_chat,
};
use crate::app::apply_theme;
use crate::components::{
    ChatInput, ExportMenu, MessageList, P2aWordmark, SettingsModal, next_at_bottom,
};
use crate::state::settings::Theme;
use crate::state::{
    ChatMessage, ChatState, ConversationState, SESSION_ENDED_TEXT, SessionState, Settings,
    SyncAction, SyncInput, ToolCallInfo, busy_tracking, decide_sync, ensure_session, needs_title,
    on_session_lost, session_lost, session_lost_text, turn_persistence,
};

/// The session `old_id` ended during a send of `prompt`: the turn is ended,
/// the transcript and the conversations are reset (the sync tracking first,
/// so the sync effect keeps the banner), and Retry will send `prompt` in a
/// new session.
fn recover_lost_session(
    mut chat: Signal<ChatState>,
    conv: Signal<ConversationState>,
    session: Signal<SessionState>,
    mut synced_conv_id: Signal<Option<String>>,
    mut synced_messages_len: Signal<usize>,
    old_id: &str,
    prompt: &str,
) {
    chat.write().end_interrupted("");
    synced_conv_id.set(None);
    synced_messages_len.set(0);
    on_session_lost(session, conv, chat, old_id);
    let mut c = chat.write();
    c.clear_messages();
    c.retry_prompt = Some(prompt.to_string());
    c.set_error(Some(SESSION_ENDED_TEXT.to_string()));
}

/// Chat panel component - main chat interface
#[component]
pub fn ChatPanel() -> Element {
    let session_state = use_context::<Signal<SessionState>>();
    let mut chat_state = use_context::<Signal<ChatState>>();
    let conversation_state = use_context::<Signal<ConversationState>>();
    let mut settings = use_context::<Signal<Settings>>();

    let mut settings_open = use_signal(|| false);
    let mut settings_initial_tab = use_signal(|| "llm".to_string());

    // First-run onboarding: auto-open the settings modal on the LLM tab when
    // no provider is configured, so users land directly on the setup screen
    // instead of discovering the confusing "Not connected" header badge.
    //
    // The `first_run_handled` guard makes this strictly one-shot: `use_effect`
    // re-runs whenever any read signal changes, so without the guard a user
    // who closed the modal without picking a provider could see it reopen on
    // the next unrelated settings mutation.
    let mut first_run_handled = use_signal(|| false);
    use_effect(move || {
        if !first_run_handled() && !settings.read().provider.is_configured() {
            settings_initial_tab.set("llm".to_string());
            settings_open.set(true);
            first_run_handled.set(true);
        }
    });

    // Native: a backend that did not start is reported once, in the error banner
    // (the effect reads no signal, so it runs once, after mount).
    use_effect(move || {
        if let Some(reason) = crate::api::backend_start_error() {
            chat_state.write().set_error(Some(format!(
                "The built-in backend did not start: {reason}"
            )));
        }
    });

    // Track which conversation we've synced from and how many messages we've synced
    let mut synced_conv_id = use_signal(|| Option::<String>::None);
    let mut synced_messages_len = use_signal(|| 0usize);

    // Whether the message list's view is at the bottom (it follows new
    // text then), and the scroll position seen last.
    let mut at_bottom = use_signal(|| true);
    let last_scroll_top = use_hook(|| std::rc::Rc::new(std::cell::Cell::new(0.0f64)));

    // The Stop handle of the answer being written, if one is.
    let mut stream_stop = use_signal(|| None::<StreamStop>);

    // The conversation whose live transcript is shown (restored from the
    // cache); its stored rows do not replace it.
    let mut restored_conv = use_signal(|| Option::<String>::None);

    // Keep the transcript in step with the current conversation
    // (state::decide_sync): a conversation left keeps its live transcript
    // for this page session (plots and tool results included) and gets it
    // back when selected again; one without is rebuilt from its stored rows.
    // An answer being written is never touched.
    use_effect(move || {
        let conv_state = conversation_state.read();
        let current_conv_id = conv_state.current_conversation_id.clone();
        let current_messages_len = conv_state.current_messages.len();
        let synced_id = synced_conv_id.read().clone();
        let prev_messages_len = *synced_messages_len.read();
        let mut chat = chat_state;

        if chat.read().is_busy() {
            // Track the conversation the send created, so the end of the
            // send does not look like a switch.
            // Stored rows that arrive meanwhile (a select's load) are
            // counted, so the end of the send does not rebuild the
            // transcript from rows older than the answer just written.
            let (id, len) = busy_tracking(
                current_conv_id.as_deref(),
                synced_id.as_deref(),
                current_messages_len,
                prev_messages_len,
            );
            if id.as_deref() != synced_id.as_deref() {
                synced_conv_id.set(id);
            }
            if len != prev_messages_len {
                synced_messages_len.set(len);
            }
            return;
        }

        // A switch away from a conversation that still exists: keep its
        // live transcript (a deleted or reset one is dropped).
        if current_conv_id != synced_id
            && let Some(left) = synced_id.as_deref()
            && conv_state.conversations.iter().any(|c| c.id == left)
            && !chat.read().messages.is_empty()
        {
            chat.write().stash_live(left);
        }

        let (chat_has_messages, has_live) = {
            let c = chat.read();
            let has_live = current_conv_id.as_deref().is_some_and(|id| c.has_live(id));
            (!c.messages.is_empty(), has_live)
        };
        let restored = current_conv_id.is_some() && *restored_conv.peek() == current_conv_id;
        let action = decide_sync(&SyncInput {
            current: current_conv_id.as_deref(),
            synced: synced_id.as_deref(),
            stored_len: current_messages_len,
            prev_stored_len: prev_messages_len,
            chat_has_messages,
            busy: false,
            has_live,
            restored,
        });
        tracing::debug!(
            "[ChatPanel Sync] {:?}: current={:?} synced={:?} stored={} prev={}",
            action,
            current_conv_id,
            synced_id,
            current_messages_len,
            prev_messages_len
        );

        match action {
            SyncAction::Skip => {
                if restored && prev_messages_len != current_messages_len {
                    synced_messages_len.set(current_messages_len);
                }
            }
            SyncAction::Track => {
                restored_conv.set(None);
                synced_conv_id.set(current_conv_id);
                synced_messages_len.set(0);
            }
            SyncAction::Clear => {
                chat.write().clear_messages();
                restored_conv.set(None);
                synced_conv_id.set(current_conv_id);
                synced_messages_len.set(0);
            }
            SyncAction::Restore => {
                if let Some(id) = current_conv_id.as_deref() {
                    let mut c = chat.write();
                    c.clear_messages();
                    c.restore_live(id);
                }
                restored_conv.set(current_conv_id.clone());
                synced_conv_id.set(current_conv_id);
                synced_messages_len.set(current_messages_len);
            }
            SyncAction::Rebuild => {
                let Some(conv_id) = current_conv_id.clone() else {
                    return;
                };
                {
                    let mut c = chat.write();
                    c.clear_messages();
                    c.messages.extend(
                        conv_state
                            .current_messages
                            .iter()
                            .map(ChatMessage::from_conversation_message),
                    );
                }
                restored_conv.set(None);
                synced_conv_id.set(current_conv_id);
                synced_messages_len.set(current_messages_len);

                // Load tool calls asynchronously
                spawn(async move {
                    let client = api();
                    match client.get_conversation_tool_calls(&conv_id).await {
                        Ok(tool_calls) => {
                            if tool_calls.is_empty() {
                                return;
                            }
                            tracing::debug!("[ChatPanel] Loaded {} tool calls", tool_calls.len());
                            let mut tool_calls_by_message: std::collections::HashMap<
                                String,
                                Vec<PersistedToolCall>,
                            > = std::collections::HashMap::new();
                            for tc in tool_calls {
                                tool_calls_by_message
                                    .entry(tc.message_id.clone())
                                    .or_default()
                                    .push(tc);
                            }

                            // Into the transcript of the conversation they
                            // belong to: the one shown, or its live copy when
                            // it was left meanwhile.
                            let still_current =
                                conversation_state.read().current_conversation_id.as_deref()
                                    == Some(conv_id.as_str());
                            if chat.read().is_busy() && still_current {
                                return;
                            }
                            let mut chat_write = chat.write();
                            let messages = if still_current {
                                Some(&mut chat_write.messages)
                            } else {
                                chat_write.live_messages_mut(&conv_id)
                            };
                            for msg in messages.into_iter().flatten() {
                                if let Some(tcs) = tool_calls_by_message.get(&msg.id) {
                                    msg.set_tool_calls(
                                        tcs.iter().map(ToolCallInfo::from).collect(),
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!("[ChatPanel] Failed to load tool calls: {}", e);
                        }
                    }
                });
            }
        }
    });

    // Get current status and error
    let status = chat_state.read().status.clone();
    let is_processing = chat_state.read().is_processing;
    let error = chat_state.read().error.clone();

    // Get current conversation title
    let current_conversation = conversation_state
        .read()
        .get_current_conversation()
        .cloned();

    // Handle send message
    let mut handle_send = move |message: String| {
        tracing::debug!("[ChatPanel] handle_send called with: {}", message);
        let mut session = session_state;
        let mut chat = chat_state;
        let mut conv = conversation_state;
        let settings = settings;

        // First-run gate: if no provider is configured, surface a friendly
        // inline error and open the settings modal instead of kicking off a
        // chat that would silently 405 against an unconfigured Ollama.
        if let Some(err) = settings.read().validation_error() {
            chat.write().set_error(Some(err));
            settings_initial_tab.set("llm".to_string());
            settings_open.set(true);
            return;
        }

        // One send at a time: processing starts here, synchronously, so a
        // second click (a chip, Retry) before the task runs is refused. It
        // also keeps the sync effect away from the messages this send adds.
        if !chat.write().try_start_processing() {
            return;
        }

        spawn(async move {
            tracing::debug!("[ChatPanel] spawn started");

            // The Stop button fires this; it is checked before the stream
            // starts and ends the stream when it is running.
            let (stop, stop_signal) = stop_pair();
            stream_stop.set(Some(stop));

            // The tab's session, created on first need (single flight).
            let session_id = match ensure_session(session).await {
                Ok(id) => id,
                Err(e) => {
                    tracing::error!("[ChatPanel] Session error: {}", e);
                    chat.write()
                        .fail_before_turn(&message, format!("Session error: {}", e));
                    stream_stop.set(None);
                    return;
                }
            };

            // Ensure we have a conversation of this session (one of an ended
            // session is not posted to).
            let conversation_id = {
                let conv_state = conv.read();
                let of_other_session = conv_state
                    .get_current_conversation()
                    .is_some_and(|c| c.session_id != session_id);
                if let Some(id) = conv_state
                    .current_conversation_id
                    .clone()
                    .filter(|_| !of_other_session)
                {
                    tracing::debug!("[ChatPanel] Using existing conversation: {}", id);
                    id
                } else {
                    // Create a new conversation if none selected
                    drop(conv_state); // Release the read lock
                    tracing::debug!("[ChatPanel] Creating new conversation");
                    let client = api();
                    match client
                        .create_conversation(&session_id, "New Conversation")
                        .await
                    {
                        Ok(new_conv) => {
                            let id = new_conv.id.clone();
                            tracing::debug!("[ChatPanel] Created conversation: {}", id);
                            conv.write().add_conversation(new_conv);
                            conv.write().set_current_conversation(Some(id.clone()));
                            id
                        }
                        Err(e) if session_lost_text(&e) => {
                            recover_lost_session(
                                chat,
                                conv,
                                session,
                                synced_conv_id,
                                synced_messages_len,
                                &session_id,
                                &message,
                            );
                            chat.write().stop_processing();
                            stream_stop.set(None);
                            return;
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
            tracing::debug!(
                "[ChatPanel] Messages added, count: {}",
                chat.read().messages.len()
            );

            // Store the user's row before the stream starts, so it precedes the
            // reply the server stores; the server stores the reply itself
            // (state::turn_persistence). Without the row nothing is streamed:
            // the server would store a reply without its question.
            if turn_persistence(&conversation_id).contains(&"user") {
                match api().add_message(&conversation_id, "user", &message).await {
                    Ok(persisted_msg) => {
                        let mut chat_write = chat.write();
                        if let Some(m) = chat_write
                            .messages
                            .iter_mut()
                            .rev()
                            .find(|m| m.role == "user" && m.content == message)
                        {
                            m.id = persisted_msg.id;
                        }
                    }
                    Err(e) if session_lost_text(&e) => {
                        recover_lost_session(
                            chat,
                            conv,
                            session,
                            synced_conv_id,
                            synced_messages_len,
                            &session_id,
                            &message,
                        );
                        chat.write().stop_processing();
                        stream_stop.set(None);
                        return;
                    }
                    Err(e) => {
                        tracing::warn!("[ChatPanel] Failed to store the user message: {}", e);
                        let mut c = chat.write();
                        c.end_interrupted("Not sent.");
                        c.set_error(Some(format!("The message was not sent: {e}")));
                        c.stop_processing();
                        stream_stop.set(None);
                        return;
                    }
                }
            }

            // Build history
            let history = chat.read().build_history();
            // handle_send checked the provider before spawning, but Settings
            // stays open to edits while the turn is set up.
            let Some(provider_config) = settings.read().to_provider_config() else {
                let text = settings.read().validation_error().unwrap_or_else(|| {
                    "Choose an LLM provider in Settings to start chatting.".to_string()
                });
                let mut c = chat.write();
                c.set_error(Some(text));
                c.end_interrupted("");
                c.stop_processing();
                stream_stop.set(None);
                return;
            };
            let interpret = settings.read().interpret_results;

            tracing::debug!("[ChatPanel] Calling stream_chat");

            // Read the backend URL from the API client: the embedded server's
            // URL on native, the compiled-in one on web.
            let backend_url = api().base_url().to_string();

            // Set when the server says the session is gone.
            let mut session_ended = false;
            // Set when the Done event came (the answer finished).
            let mut saw_done = false;

            // Stopped before the stream started (while the turn was set up).
            if stop_signal.is_fired() {
                stream_stop.set(None);
                let mut c = chat.write();
                c.end_interrupted("Stopped.");
                c.stop_processing();
                c.set_status(Some("Answer stopped.".to_string()));
                return;
            }

            // Stream the response
            let result = stream_chat(
                &backend_url,
                &session_id,
                &message,
                history,
                provider_config,
                interpret,
                Some(conversation_id.clone()),
                stop_signal,
                |event| {
                    tracing::debug!(
                        "[ChatPanel] Got event: {:?}",
                        std::mem::discriminant(&event)
                    );
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
                        StreamEvent::ToolEnd { result, status, .. } => {
                            chat.write().clear_active_tool(result, status.as_deref());
                        }
                        StreamEvent::ToolResult { images } => {
                            if let Some(imgs) = images {
                                for img in imgs {
                                    chat.write().add_image(&img.data);
                                }
                            }
                        }
                        StreamEvent::Done { message } => {
                            // The server stores the reply (with its tool calls).
                            saw_done = true;
                            chat.write().finalize_message(message);
                        }
                        StreamEvent::Error { error, code } => {
                            if session_lost(code.as_deref(), &error) {
                                session_ended = true;
                            } else {
                                chat.write().set_error(Some(crate::api::error_text(
                                    &error,
                                    code.as_deref(),
                                )));
                            }
                        }
                    }
                },
            )
            .await;

            tracing::debug!("[ChatPanel] stream_chat returned: {:?}", result.is_ok());
            stream_stop.set(None);
            let mut stopped = false;
            match result {
                Ok(_) => {}
                // Stopped by the user: the partial answer stays, no error.
                Err(e) if e.kind() == StreamErrorKind::Stopped => stopped = true,
                // A silent stream was given up: the error banner offers Retry.
                Err(e) if e.kind() == StreamErrorKind::Idle => {
                    let mut c = chat.write();
                    c.end_interrupted("");
                    c.set_error(Some(e.to_string()));
                }
                Err(e) => {
                    tracing::error!("[ChatPanel] Stream error: {}", e);
                    if session_lost_text(&e.to_string()) {
                        session_ended = true;
                    } else {
                        chat.write().set_error(Some(e.to_string()));
                    }
                }
            }

            if session_ended {
                recover_lost_session(
                    chat,
                    conv,
                    session,
                    synced_conv_id,
                    synced_messages_len,
                    &session_id,
                    &message,
                );
                chat.write().stop_processing();
                return;
            }

            {
                let mut c = chat.write();
                if stopped {
                    c.end_interrupted("Stopped.");
                }
                // An answer the stream left unfinished is ended and marked.
                c.finish_stream(saw_done);
                c.stop_processing();
                if stopped {
                    c.set_status(Some("Answer stopped.".to_string()));
                }
            }

            // Trigger dataset sidebar refresh in case tools created/modified datasets
            session.write().trigger_datasets_refresh();

            // Generate title for new conversations after first exchange
            if !conversation_id.is_empty() {
                let conv_id = conversation_id.clone();
                let user_msg = message.clone();
                let settings_clone = settings.read().clone();
                let mut conv_state = conv;

                // Whether the turn's conversation (not the one current now)
                // still has the default title
                let needs_title = needs_title(&conv_state.read().conversations, &conv_id);

                if needs_title {
                    // Get the assistant's response from chat state
                    let assistant_response = chat
                        .read()
                        .messages
                        .iter()
                        .rfind(|m| m.role == "assistant")
                        .map(|m| m.content.clone());

                    spawn(async move {
                        let client = api();

                        // Generate title using LLM. Same invariant as the
                        // main chat path: we reach here only after the gate
                        // in handle_send, so the provider is configured.
                        match client
                            .generate_title(
                                &user_msg,
                                assistant_response.as_deref(),
                                settings_clone.to_provider_config(),
                            )
                            .await
                        {
                            Ok(new_title) => {
                                tracing::info!("[ChatPanel] Generated title: {}", new_title);
                                // Update conversation with new title
                                match client
                                    .update_conversation(&conv_id, Some(&new_title), None)
                                    .await
                                {
                                    Ok(updated) => {
                                        conv_state.write().update_conversation(updated);
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "[ChatPanel] Failed to update conversation title: {}",
                                            e
                                        );
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

    // A send runs or a conversation call does: chips and Retry wait.
    let busy = is_processing || conversation_state.read().is_operating;

    // Handle retry - re-send the last user message
    let handle_retry = {
        let mut handle_send = handle_send;
        move |_| {
            if chat_state.read().is_busy() || conversation_state.read().is_operating {
                return;
            }
            // The prompt to ask again; an interrupted or empty answer and its
            // prompt are taken out, so the new turn replaces them. Without a
            // configured provider handle_send only says so (and opens the
            // settings), so nothing is taken out then.
            let retry = if settings.read().validation_error().is_some() {
                Some(String::new())
            } else {
                chat_state.write().take_retry_turn()
            };
            if let Some(msg) = retry {
                chat_state.write().set_error(None);
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
        let saved = settings.read().save();
        if let Err(e) = saved {
            chat_state.write().set_error(Some(e));
        }
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
                        // Provider badge: CTA when unconfigured, model pill otherwise.
                        // Both open the Settings modal on click so users can
                        // reach the setup flow from the header at any time.
                        if settings.read().provider.is_configured() {
                            button {
                                class: "px-2.5 py-1 text-xs font-medium rounded-full bg-orange-100 text-orange-700 border border-orange-300 hover:bg-orange-200 dark:bg-orange-900/30 dark:text-orange-300 dark:border-orange-700 dark:hover:bg-orange-900/50 transition-colors",
                                onclick: move |_| {
                                    settings_initial_tab.set("llm".to_string());
                                    settings_open.set(true);
                                },
                                "{settings.read().provider.display_name()} / {settings.read().current_model()}"
                            }
                        } else {
                            button {
                                class: "px-3 py-1 text-xs font-semibold rounded-full bg-teal-600 text-white border border-teal-600 hover:bg-teal-700 transition-colors flex items-center gap-1",
                                onclick: move |_| {
                                    settings_initial_tab.set("llm".to_string());
                                    settings_open.set(true);
                                },
                                "Connect LLM provider →"
                            }
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

                        // Export menu
                        ExportMenu {}

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
                                class: "px-3 py-1 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-lg transition-colors flex items-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed",
                                disabled: busy,
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

            // While an answer is written: the Stop control
            if is_processing {
                div { class: "mx-4 mt-2 px-3 py-1.5 rounded-lg bg-gray-50 dark:bg-gray-800 flex items-center justify-between gap-2",
                    span { class: "text-sm text-gray-600 dark:text-gray-300", "Writing the answer\u{2026}" }
                    button {
                        class: "px-3 py-1 text-sm font-medium text-gray-700 dark:text-gray-200 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors",
                        title: "Stop the answer",
                        onclick: move |_| {
                            let stop = stream_stop.read().clone();
                            if let Some(stop) = stop {
                                stop.stop();
                            }
                        },
                        "Stop"
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

            // Message list (scrollable). Whether the view is at the bottom
            // is written only when it changes, not per scroll event.
            div {
                class: "flex-1 overflow-y-auto",
                onscroll: move |evt: Event<ScrollData>| {
                    let data = evt.data();
                    let top = data.scroll_top();
                    let was = *at_bottom.peek();
                    let now = next_at_bottom(
                        was,
                        last_scroll_top.get(),
                        top,
                        data.scroll_height(),
                        data.client_height(),
                    );
                    last_scroll_top.set(top);
                    if now != was {
                        at_bottom.set(now);
                    }
                },
                MessageList {
                    at_bottom: Some(at_bottom),
                    suggestions_disabled: busy,
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
