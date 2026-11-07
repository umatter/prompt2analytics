//! Chat input component with keyboard handling and autocomplete

use dioxus::prelude::*;

use crate::api::{DatasetMeta, ToolDefinition, api};
use crate::components::AutocompleteDropdown;
use crate::state::autocomplete::{AutocompleteState, Suggestion, find_last_trigger};
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
                        accept_suggestion(&mut textarea_value, &mut autocomplete, &suggestion);
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
        let cached_tools = cached_tools;
        let cached_datasets = cached_datasets;
        move |evt: Event<FormData>| {
            let value = evt.value();
            textarea_value.set(value.clone());

            // Try to detect autocomplete triggers
            let mut ac = autocomplete.write();
            if let Some(suggestions) = detect_and_generate_suggestions(
                &value,
                &mut ac,
                &cached_tools.read(),
                &cached_datasets.read(),
                &chat_state.read().prompt_history,
            ) {
                ac.set_suggestions(suggestions);
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

/// Detect triggers and generate suggestions based on input.
///
/// Runs on every input event, so the trigger offset in `ac` always matches the text.
fn detect_and_generate_suggestions(
    input: &str,
    ac: &mut AutocompleteState,
    tools: &[ToolDefinition],
    datasets: &[DatasetMeta],
    history: &[String],
) -> Option<Vec<Suggestion>> {
    // Look for @ or / triggers by scanning for the most recent trigger that
    // could be active. `find_last_trigger` returns a byte offset into `input`.

    // @dataset.column (column mode) and @dataset
    if let Some(at_pos) = find_last_trigger(input, '@')
        && let Some(after_at) = after_trigger(input, at_pos, '@')
    {
        if let Some((dataset_name, filter)) = after_at.split_once('.')
            && let Some(dataset) = datasets.iter().find(|d| d.name == dataset_name)
        {
            ac.track_column_mode(dataset_name, at_pos);
            ac.set_filter(filter);
            let filter_lower = filter.to_lowercase();
            return Some(
                dataset
                    .column_names
                    .iter()
                    .filter(|col| filter.is_empty() || col.to_lowercase().contains(&filter_lower))
                    .take(10)
                    .map(|col| Suggestion::column(col, dataset_name))
                    .collect(),
            );
        }
        // Not column mode: a space after the @ cancels the trigger.
        if !after_at.contains(' ') {
            ac.track('@', at_pos);
            ac.set_filter(after_at);
            let filter_lower = after_at.to_lowercase();
            return Some(
                datasets
                    .iter()
                    .filter(|d| {
                        after_at.is_empty() || d.name.to_lowercase().contains(&filter_lower)
                    })
                    .take(10)
                    .map(|d| Suggestion::dataset(&d.name, d.row_count, d.column_count))
                    .collect(),
            );
        }
    }

    // /tool; a space after the / cancels the trigger.
    if let Some(slash_pos) = find_last_trigger(input, '/')
        && let Some(filter) = after_trigger(input, slash_pos, '/')
        && !filter.contains(' ')
    {
        ac.track('/', slash_pos);
        ac.set_filter(filter);
        let filter_lower = filter.to_lowercase();
        return Some(
            tools
                .iter()
                .filter(|t| filter.is_empty() || t.name.to_lowercase().contains(&filter_lower))
                .take(10)
                .map(|t| Suggestion::tool(&t.name, &t.description))
                .collect(),
        );
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
            ac.track_history();
            ac.set_filter(input);

            return Some(matching);
        }
    }

    // No suggestions - close autocomplete
    ac.close();
    None
}

/// The text after the trigger at `position`, or `None` when `input` has no `trigger` there.
fn after_trigger(input: &str, position: usize, trigger: char) -> Option<&str> {
    input.get(position..)?.strip_prefix(trigger)
}

/// Accept a suggestion and insert it into the textarea
fn accept_suggestion(
    textarea_value: &mut Signal<String>,
    autocomplete: &mut Signal<AutocompleteState>,
    suggestion: &Suggestion,
) {
    let new_value = autocomplete
        .read()
        .apply(&textarea_value.read(), suggestion);
    textarea_value.set(new_value);
    autocomplete.write().close();
}

#[cfg(test)]
mod accept_tests {
    //! ChatInput's input handling replayed without a runtime. Each input is the whole textarea
    //! value, as `handle_input` sees it. Then the selected suggestion is accepted, as the Tab branch
    //! of `handle_keydown` does.
    use super::detect_and_generate_suggestions;
    use crate::api::{DatasetMeta, ToolDefinition};
    use crate::state::autocomplete::AutocompleteState;

    fn datasets() -> Vec<DatasetMeta> {
        vec![
            DatasetMeta {
                id: "ds-1".into(),
                session_id: "s-1".into(),
                name: "sales".into(),
                source_path: None,
                source_type: "csv".into(),
                row_count: 100,
                column_count: 2,
                column_names: vec!["price".into(), "sqft".into()],
                loaded_at: String::new(),
                file_size_bytes: None,
            },
            DatasetMeta {
                id: "ds-2".into(),
                session_id: "s-1".into(),
                name: "my data".into(),
                source_path: None,
                source_type: "csv".into(),
                row_count: 100,
                column_count: 1,
                column_names: vec!["unit price".into()],
                loaded_at: String::new(),
                file_size_bytes: None,
            },
        ]
    }

    fn tools() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "regression_ols".into(),
            description: "OLS regression".into(),
            input_schema: None,
        }]
    }

    /// Replays `inputs`, then accepts the selected suggestion; returns the new input and the
    /// state the dropdown was in when the suggestion was accepted.
    fn type_then_tab(inputs: &[&str], history: &[String]) -> (String, AutocompleteState) {
        let (tools, datasets) = (tools(), datasets());
        let mut ac = AutocompleteState::new();
        for value in inputs {
            if let Some(s) =
                detect_and_generate_suggestions(value, &mut ac, &tools, &datasets, history)
            {
                ac.set_suggestions(s);
            }
        }
        let current = inputs.last().copied().unwrap_or_default();
        let selected = ac
            .get_selected()
            .cloned()
            .expect("a suggestion is selected");
        (ac.apply(current, &selected), ac)
    }

    /// Guard: no edits before the trigger.
    #[test]
    fn test_accept_suggestion_plain() {
        assert_eq!(type_then_tab(&["@", "@s"], &[]).0, "@sales ");
        assert_eq!(
            type_then_tab(&["/", "/r", "/re"], &[]).0,
            "/regression_ols "
        );
        assert_eq!(
            type_then_tab(&["@sales.", "@sales.p"], &[]).0,
            "@sales.price "
        );
        assert_eq!(
            type_then_tab(&["an"], &["analyze sales".to_string()]).0,
            "analyze sales "
        );
    }

    /// "abc @s", then "abc " deleted: the offset 4 kept from the opening input was past the end of
    /// "@s".
    #[test]
    fn test_accept_suggestion_after_edit_before_trigger() {
        let (value, ac) = type_then_tab(&["abc @", "abc @s", "@s"], &[]);
        assert_eq!(value, "@sales ");
        assert_eq!(ac.trigger_position, 0);
    }

    /// A multi-byte character typed before an open trigger.
    #[test]
    fn test_accept_suggestion_after_multibyte_edit_before_trigger() {
        let (value, ac) = type_then_tab(&["@s", " @s", "中 @s"], &[]);
        assert_eq!(value, "中 @sales ");
        assert_eq!(ac.trigger_position, 4);
    }

    /// A space typed before an open trigger.
    #[test]
    fn test_accept_suggestion_after_space_before_trigger() {
        assert_eq!(type_then_tab(&["@s", " @s"], &[]).0, " @sales ");
    }

    /// The `/` trigger.
    #[test]
    fn test_accept_tool_suggestion_after_edit_before_trigger() {
        assert_eq!(
            type_then_tab(&["/re", "x /re"], &[]).0,
            "x /regression_ols "
        );
    }

    /// Column mode kept the offset too.
    #[test]
    fn test_accept_column_suggestion_after_edit_before_trigger() {
        let (value, ac) = type_then_tab(&["@sales.p", "y @sales.p"], &[]);
        assert_eq!(value, "y @sales.price ");
        assert_eq!(ac.trigger_position, 2);
    }

    /// Dataset and column names can hold spaces: the whole `@dataset.filter` is replaced, not just
    /// its first word.
    #[test]
    fn test_accept_column_suggestion_with_spaces_in_the_names() {
        assert_eq!(
            type_then_tab(&["@my data.unit p"], &[]).0,
            "@my data.unit price "
        );
    }

    /// History after a trigger: the history branch reset the trigger only when the dropdown was
    /// closed, so the history suggestion was inserted after an '@'.
    #[test]
    fn test_accept_history_suggestion_after_a_trigger() {
        let (value, ac) = type_then_tab(&["@s", "ana"], &["analyze sales".to_string()]);
        assert_eq!(value, "analyze sales ");
        assert_eq!(ac.trigger, None);
    }
}
