//! Chat input component with keyboard handling and autocomplete

use dioxus::prelude::*;

use crate::api::{api, DatasetMeta, ToolDefinition};
use crate::components::AutocompleteDropdown;
use crate::state::autocomplete::{AutocompleteState, Suggestion};
use crate::state::{ChatState, SessionState};

/// ChatInput component - textarea with send button and autocomplete
#[component]
pub fn ChatInput(on_send: EventHandler<String>) -> Element {
    let mut chat_state = use_context::<Signal<ChatState>>();
    let session_state = use_context::<Signal<SessionState>>();
    let mut textarea_value = use_signal(String::new);
    let mut autocomplete = use_signal(AutocompleteState::new);

    // Cache for tools (fetched once on mount)
    let mut cached_tools = use_signal(Vec::<ToolDefinition>::new);

    // Cache for datasets with column info (refreshed when datasets change)
    let mut cached_datasets = use_signal(Vec::<DatasetMeta>::new);

    let is_processing = chat_state.read().is_processing;

    // Fetch tools once on mount
    use_effect(move || {
        spawn(async move {
            match api().list_tools().await {
                Ok(tools) => {
                    tracing::debug!("[ChatInput] Loaded {} tools for autocomplete", tools.len());
                    cached_tools.set(tools);
                }
                Err(e) => {
                    tracing::warn!("[ChatInput] Failed to load tools: {}", e);
                }
            }
        });
    });

    // Fetch datasets when session is ready or datasets_refresh_counter changes
    // We need to read the signal inside the effect to make it reactive
    use_effect(move || {
        // Read inside the effect to track dependencies
        let state = session_state.read();
        let refresh_counter = state.datasets_refresh_counter;
        let session_id = state.session_id.clone();
        drop(state); // Release the read lock before spawning

        tracing::debug!(
            "[ChatInput] Dataset effect triggered, refresh={}, session={:?}",
            refresh_counter,
            session_id
        );

        if let Some(sid) = session_id {
            spawn(async move {
                match api().list_session_datasets(&sid).await {
                    Ok(datasets) => {
                        tracing::info!(
                            "[ChatInput] Loaded {} datasets for autocomplete: {:?}",
                            datasets.len(),
                            datasets.iter().map(|d| &d.name).collect::<Vec<_>>()
                        );
                        cached_datasets.set(datasets);
                    }
                    Err(e) => {
                        tracing::warn!("[ChatInput] Failed to load datasets: {}", e);
                    }
                }
            });
        } else {
            tracing::debug!("[ChatInput] No session ID yet, skipping dataset fetch");
        }
    });

    // Handle key events
    let handle_keydown = move |evt: Event<KeyboardData>| {
        let key = evt.key();
        let ac = autocomplete.read();
        let is_ac_open = ac.is_open && !ac.suggestions.is_empty();
        drop(ac);

        if is_ac_open {
            // Autocomplete is open - handle navigation
            match key {
                Key::ArrowUp => {
                    evt.prevent_default();
                    autocomplete.write().navigate_up();
                }
                Key::ArrowDown => {
                    evt.prevent_default();
                    autocomplete.write().navigate_down();
                }
                Key::Tab | Key::Enter => {
                    // Accept selected suggestion
                    let selected = autocomplete.read().get_selected().cloned();
                    if let Some(suggestion) = selected {
                        evt.prevent_default();
                        accept_suggestion(
                            &mut textarea_value,
                            &mut autocomplete,
                            &suggestion,
                        );
                    } else if key == Key::Enter && !evt.modifiers().shift() {
                        // No selection, send message normally
                        evt.prevent_default();
                        autocomplete.write().close();
                        let value = textarea_value.read().clone();
                        if !value.trim().is_empty() && !is_processing {
                            on_send.call(value.clone());
                            textarea_value.set(String::new());
                            chat_state.write().reset_history_index();
                        }
                    }
                }
                Key::Escape => {
                    evt.prevent_default();
                    autocomplete.write().close();
                }
                Key::Backspace => {
                    // Check if we're deleting the trigger character
                    let ac = autocomplete.read();
                    let value = textarea_value.read();
                    // If filter is empty and we backspace, we're deleting the trigger
                    if ac.filter_text.is_empty() && value.len() == ac.trigger_position + 1 {
                        drop(ac);
                        drop(value);
                        autocomplete.write().close();
                    }
                }
                _ => {}
            }
        } else {
            // Autocomplete is closed - normal behavior
            match key {
                Key::Enter => {
                    // Enter without shift sends the message
                    if !evt.modifiers().shift() {
                        evt.prevent_default();
                        let value = textarea_value.read().clone();
                        if !value.trim().is_empty() && !is_processing {
                            on_send.call(value.clone());
                            textarea_value.set(String::new());
                            chat_state.write().reset_history_index();
                        }
                    }
                }
                Key::ArrowUp => {
                    // Arrow up navigates history when input is empty
                    let is_empty = textarea_value.read().is_empty();
                    if is_empty {
                        evt.prevent_default();
                        chat_state.write().navigate_history_up();
                        let input = chat_state.read().input.clone();
                        textarea_value.set(input);
                    }
                }
                Key::ArrowDown => {
                    // Arrow down navigates history
                    let is_empty = textarea_value.read().is_empty();
                    if is_empty {
                        evt.prevent_default();
                        chat_state.write().navigate_history_down();
                        let input = chat_state.read().input.clone();
                        textarea_value.set(input);
                    }
                }
                _ => {}
            }
        }
    };

    // Handle input change - detect triggers and update autocomplete
    let handle_input = {
        let cached_tools = cached_tools.clone();
        let cached_datasets = cached_datasets.clone();
        move |evt: Event<FormData>| {
            let value = evt.value();
            textarea_value.set(value.clone());

            // Try to detect autocomplete triggers
            let suggestions = detect_and_generate_suggestions(
                &value,
                &mut autocomplete,
                &cached_tools.read(),
                &cached_datasets.read(),
                &chat_state.read().prompt_history,
            );

            if let Some(suggestions) = suggestions {
                autocomplete.write().set_suggestions(suggestions);
            }
        }
    };

    // Handle send button click
    let handle_send = move |_| {
        tracing::debug!("[ChatInput] Send clicked, is_processing: {}", is_processing);
        autocomplete.write().close();
        let value = textarea_value.read().clone();
        tracing::debug!(
            "[ChatInput] Value: '{}', empty: {}",
            value,
            value.trim().is_empty()
        );
        if !value.trim().is_empty() && !is_processing {
            tracing::debug!("[ChatInput] Calling on_send");
            on_send.call(value.clone());
            textarea_value.set(String::new());
            chat_state.write().reset_history_index();
        } else {
            tracing::debug!("[ChatInput] Skipped send - empty or processing");
        }
    };

    // Handle suggestion selection from dropdown click
    let handle_suggestion_select = move |suggestion: Suggestion| {
        accept_suggestion(&mut textarea_value, &mut autocomplete, &suggestion);
    };

    rsx! {
        div { class: "px-4 py-4",
            div { class: "flex gap-3 items-center",
                // Textarea container with autocomplete dropdown
                div { class: "flex-1 relative",
                    // Autocomplete dropdown (positioned above)
                    AutocompleteDropdown {
                        state: autocomplete.read().clone(),
                        on_select: handle_suggestion_select,
                    }

                    textarea {
                        class: "w-full px-4 py-3 pr-12 text-sm bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent placeholder-gray-400 dark:placeholder-gray-500 text-gray-900 dark:text-white transition-all",
                        placeholder: "Type a message... (@ for datasets, / for tools)",
                        value: "{textarea_value}",
                        disabled: is_processing,
                        oninput: handle_input,
                        onkeydown: handle_keydown,
                        rows: "1"
                    }
                }

                // Send button
                button {
                    class: if is_processing {
                        "px-4 py-3 bg-gray-400 text-white rounded-xl cursor-not-allowed flex items-center gap-2"
                    } else {
                        "px-4 py-3 bg-orange-600 hover:bg-orange-700 text-white rounded-xl transition-colors flex items-center gap-2 shadow-sm hover:shadow-md"
                    },
                    onclick: handle_send,
                    disabled: is_processing,

                    if is_processing {
                        // Loading spinner
                        svg {
                            class: "w-5 h-5 animate-spin",
                            fill: "none",
                            view_box: "0 0 24 24",
                            circle {
                                class: "opacity-25",
                                cx: "12",
                                cy: "12",
                                r: "10",
                                stroke: "currentColor",
                                stroke_width: "4"
                            }
                            path {
                                class: "opacity-75",
                                fill: "currentColor",
                                d: "M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            }
                        }
                        "Processing..."
                    } else {
                        // Send icon
                        svg {
                            class: "w-5 h-5",
                            fill: "none",
                            stroke: "currentColor",
                            view_box: "0 0 24 24",
                            path {
                                stroke_linecap: "round",
                                stroke_linejoin: "round",
                                stroke_width: "2",
                                d: "M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                            }
                        }
                        "Send"
                    }
                }
            }

            // Hint text
            div { class: "flex items-center gap-4 mt-2 text-xs text-gray-400 dark:text-gray-500 flex-wrap",
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 font-mono", "Enter" }
                    "to send"
                }
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 font-mono", "Shift" }
                    "+"
                    kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 font-mono", "Enter" }
                    "new line"
                }
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 font-mono", "@" }
                    "datasets"
                }
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 font-mono", "/" }
                    "tools"
                }
                span { class: "flex items-center gap-1",
                    kbd { class: "px-1.5 py-0.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-700 rounded text-gray-600 dark:text-gray-400 font-mono", "\u{2191}/\u{2193}" }
                    "history"
                }
            }
        }
    }
}

/// Detect triggers and generate suggestions based on input
fn detect_and_generate_suggestions(
    input: &str,
    autocomplete: &mut Signal<AutocompleteState>,
    tools: &[ToolDefinition],
    datasets: &[DatasetMeta],
    history: &[String],
) -> Option<Vec<Suggestion>> {
    // Find the last trigger character and its position
    let chars: Vec<char> = input.chars().collect();

    // Look for @ or / triggers by scanning backwards from cursor (end of input)
    // We want to find the most recent trigger that could be active

    // Check for @dataset. pattern first (column mode)
    if let Some(at_pos) = find_last_trigger(&chars, '@') {
        let after_at = &input[at_pos + 1..];

        // Check if we have a complete dataset name followed by a dot
        if let Some(dot_pos) = after_at.find('.') {
            let dataset_name = &after_at[..dot_pos];
            let filter = &after_at[dot_pos + 1..];

            // Verify this dataset exists
            if let Some(dataset) = datasets.iter().find(|d| d.name == dataset_name) {
                // Open in column mode
                let mut ac = autocomplete.write();
                if !ac.is_open
                    || !ac.is_column_mode()
                    || ac.column_dataset.as_deref() != Some(dataset_name)
                {
                    ac.open_column_mode(dataset_name, at_pos);
                }
                ac.set_filter(filter);
                drop(ac);

                // Generate column suggestions
                let filter_lower = filter.to_lowercase();
                let suggestions: Vec<Suggestion> = dataset
                    .column_names
                    .iter()
                    .filter(|col| {
                        filter.is_empty() || col.to_lowercase().contains(&filter_lower)
                    })
                    .take(10)
                    .map(|col| Suggestion::column(col, dataset_name))
                    .collect();

                return Some(suggestions);
            }
        }

        // Not column mode - dataset mode
        let filter = after_at;

        // Make sure there's no space after @ (which would cancel the trigger)
        if !filter.contains(' ') {
            let mut ac = autocomplete.write();
            if !ac.is_open || ac.trigger != Some('@') || ac.is_column_mode() {
                ac.open('@', at_pos);
            }
            ac.set_filter(filter);
            drop(ac);

            // Generate dataset suggestions
            let filter_lower = filter.to_lowercase();
            let suggestions: Vec<Suggestion> = datasets
                .iter()
                .filter(|d| filter.is_empty() || d.name.to_lowercase().contains(&filter_lower))
                .take(10)
                .map(|d| Suggestion::dataset(&d.name, d.row_count, d.column_count))
                .collect();

            return Some(suggestions);
        }
    }

    // Check for / trigger (tools)
    if let Some(slash_pos) = find_last_trigger(&chars, '/') {
        let filter = &input[slash_pos + 1..];

        // Make sure there's no space after / (which would cancel the trigger)
        if !filter.contains(' ') {
            let mut ac = autocomplete.write();
            if !ac.is_open || ac.trigger != Some('/') {
                ac.open('/', slash_pos);
            }
            ac.set_filter(filter);
            drop(ac);

            // Generate tool suggestions
            let filter_lower = filter.to_lowercase();
            let suggestions: Vec<Suggestion> = tools
                .iter()
                .filter(|t| filter.is_empty() || t.name.to_lowercase().contains(&filter_lower))
                .take(10)
                .map(|t| Suggestion::tool(&t.name, &t.description))
                .collect();

            return Some(suggestions);
        }
    }

    // No trigger found - check for history matching (free text)
    // Only show history if input is non-empty and doesn't start with trigger
    if !input.is_empty()
        && !input.starts_with('@')
        && !input.starts_with('/')
        && !history.is_empty()
    {
        let input_lower = input.to_lowercase();

        // Filter history by matching prefix
        let matching: Vec<Suggestion> = history
            .iter()
            .rev() // Most recent first
            .filter(|h| h.to_lowercase().starts_with(&input_lower) && *h != input)
            .take(5)
            .map(|h| Suggestion::history(h))
            .collect();

        if !matching.is_empty() {
            let mut ac = autocomplete.write();
            if !ac.is_open {
                ac.is_open = true;
                ac.trigger = None;
                ac.trigger_position = 0;
            }
            ac.set_filter(input);
            drop(ac);

            return Some(matching);
        }
    }

    // No suggestions - close autocomplete
    autocomplete.write().close();
    None
}

/// Find the position of the last occurrence of a trigger character
/// that could be the start of an autocomplete pattern
fn find_last_trigger(chars: &[char], trigger: char) -> Option<usize> {
    // Search backwards for the trigger
    for (i, &c) in chars.iter().enumerate().rev() {
        if c == trigger {
            // Make sure this trigger is either at the start or after a space
            if i == 0 || chars[i - 1].is_whitespace() {
                return Some(i);
            }
        }
    }
    None
}

/// Accept a suggestion and insert it into the textarea
fn accept_suggestion(
    textarea_value: &mut Signal<String>,
    autocomplete: &mut Signal<AutocompleteState>,
    suggestion: &Suggestion,
) {
    let ac = autocomplete.read();
    let current_value = textarea_value.read().clone();

    let new_value = if let Some(trigger) = ac.trigger {
        // We have a trigger - replace from trigger position
        let (start, _end) = ac.get_replacement_range();
        let before = &current_value[..start];
        let after = if ac.is_column_mode() {
            // In column mode, insert @dataset.column
            let insert_pos = start + 1 + ac.column_dataset.as_ref().map(|d| d.len()).unwrap_or(0) + 1 + ac.filter_text.len();
            if insert_pos < current_value.len() {
                &current_value[insert_pos..]
            } else {
                ""
            }
        } else {
            let insert_pos = start + 1 + ac.filter_text.len();
            if insert_pos < current_value.len() {
                &current_value[insert_pos..]
            } else {
                ""
            }
        };

        // Build the new value
        match trigger {
            '@' => {
                if ac.is_column_mode() {
                    // Insert full @dataset.column
                    format!("{}@{} {}", before, suggestion.insert_text, after.trim_start())
                } else {
                    // Insert @dataset (without trailing dot to allow column access)
                    format!("{}@{} {}", before, suggestion.insert_text, after.trim_start())
                }
            }
            '/' => {
                // Insert /tool
                format!("{}/{} {}", before, suggestion.insert_text, after.trim_start())
            }
            _ => current_value,
        }
    } else {
        // History mode - replace entire input
        suggestion.insert_text.clone()
    };

    drop(ac);
    textarea_value.set(new_value.trim_end().to_string() + " ");
    autocomplete.write().close();
}
