//! Autocomplete dropdown component

use dioxus::prelude::*;

use crate::state::autocomplete::{AutocompleteState, Suggestion, SuggestionType};

/// Props for the autocomplete dropdown
#[derive(Props, Clone, PartialEq)]
pub struct AutocompleteDropdownProps {
    /// The autocomplete state
    pub state: AutocompleteState,
    /// Handler when a suggestion is selected
    pub on_select: EventHandler<Suggestion>,
}

/// Autocomplete dropdown component - displays suggestions above the input
#[component]
pub fn AutocompleteDropdown(props: AutocompleteDropdownProps) -> Element {
    let state = &props.state;

    // Don't render if closed or no suggestions
    if !state.is_open || state.suggestions.is_empty() {
        return rsx! {};
    }

    rsx! {
        div {
            class: "autocomplete-dropdown",

            // Group suggestions by type if mixing
            for (idx, suggestion) in state.suggestions.iter().enumerate() {
                AutocompleteSuggestionItem {
                    key: "{idx}",
                    suggestion: suggestion.clone(),
                    is_selected: idx as i32 == state.selected_index,
                    on_click: {
                        let suggestion = suggestion.clone();
                        let on_select = props.on_select;
                        move |_| {
                            on_select.call(suggestion.clone());
                        }
                    },
                }
            }
        }
    }
}

/// Props for individual suggestion item
#[derive(Props, Clone, PartialEq)]
struct AutocompleteSuggestionItemProps {
    suggestion: Suggestion,
    is_selected: bool,
    on_click: EventHandler<()>,
}

/// Individual suggestion item in the dropdown
#[component]
fn AutocompleteSuggestionItem(props: AutocompleteSuggestionItemProps) -> Element {
    let suggestion = &props.suggestion;
    let is_selected = props.is_selected;

    let selected_class = if is_selected { "selected" } else { "" };

    rsx! {
        div {
            class: "autocomplete-item {selected_class}",
            onclick: move |_| props.on_click.call(()),

            // Icon based on type
            {render_type_icon(suggestion.suggestion_type)}

            // Text content
            div { class: "flex-1 min-w-0",
                div { class: "flex items-center gap-2",
                    span {
                        class: "k-mono k-ink text-[12.5px] font-semibold truncate",
                        "{suggestion.display_text}"
                    }
                }
                if let Some(ref desc) = suggestion.description {
                    div {
                        class: "k-muted text-xs truncate",
                        "{desc}"
                    }
                }
            }

            // Type badge
            span {
                class: "{type_badge_class(suggestion.suggestion_type)}",
                "{type_label(suggestion.suggestion_type)}"
            }
        }
    }
}

/// Render the icon for a suggestion type
fn render_type_icon(suggestion_type: SuggestionType) -> Element {
    let (path, color_class) = match suggestion_type {
        SuggestionType::Dataset => (
            "M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4",
            "k-ink",
        ),
        SuggestionType::Column => (
            "M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2",
            "k-ink",
        ),
        SuggestionType::Tool => (
            "M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z M15 12a3 3 0 11-6 0 3 3 0 016 0z",
            "k-orange-text",
        ),
        SuggestionType::History => ("M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z", "k-muted"),
    };

    rsx! {
        svg {
            class: "w-4 h-4 flex-shrink-0 {color_class}",
            fill: "none",
            stroke: "currentColor",
            view_box: "0 0 24 24",
            path {
                stroke_linecap: "round",
                stroke_linejoin: "round",
                stroke_width: "2",
                d: "{path}"
            }
        }
    }
}

/// Get badge class for suggestion type: a mono label, muted except for tools
fn type_badge_class(suggestion_type: SuggestionType) -> &'static str {
    match suggestion_type {
        SuggestionType::Tool => "k-mono text-[11px] k-orange-text flex-shrink-0",
        SuggestionType::Dataset | SuggestionType::Column | SuggestionType::History => {
            "k-mono text-[11px] k-muted flex-shrink-0"
        }
    }
}

/// Get label for suggestion type
fn type_label(suggestion_type: SuggestionType) -> &'static str {
    match suggestion_type {
        SuggestionType::Dataset => "dataset",
        SuggestionType::Column => "column",
        SuggestionType::Tool => "tool",
        SuggestionType::History => "history",
    }
}
