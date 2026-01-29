//! Autocomplete state management for chat input

/// Type of suggestion being displayed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionType {
    /// Dataset name (triggered by @)
    Dataset,
    /// Column name from a dataset (triggered by @dataset.)
    Column,
    /// Tool/command name (triggered by /)
    Tool,
    /// Previous prompt from history
    History,
}

/// A single autocomplete suggestion
#[derive(Debug, Clone, PartialEq)]
pub struct Suggestion {
    /// Text shown in the dropdown
    pub display_text: String,
    /// Text inserted when selected (may differ from display)
    pub insert_text: String,
    /// Optional description/context
    pub description: Option<String>,
    /// Type of this suggestion
    pub suggestion_type: SuggestionType,
}

impl Suggestion {
    /// Create a new dataset suggestion
    pub fn dataset(name: &str, row_count: i32, col_count: i32) -> Self {
        Self {
            display_text: name.to_string(),
            insert_text: name.to_string(),
            description: Some(format!("{} rows, {} cols", row_count, col_count)),
            suggestion_type: SuggestionType::Dataset,
        }
    }

    /// Create a new column suggestion
    pub fn column(name: &str, dataset_name: &str) -> Self {
        Self {
            display_text: name.to_string(),
            insert_text: format!("{}.{}", dataset_name, name),
            description: Some(format!("from {}", dataset_name)),
            suggestion_type: SuggestionType::Column,
        }
    }

    /// Create a new tool suggestion
    pub fn tool(name: &str, description: &str) -> Self {
        Self {
            display_text: name.to_string(),
            insert_text: name.to_string(),
            description: if description.is_empty() {
                None
            } else {
                Some(description.to_string())
            },
            suggestion_type: SuggestionType::Tool,
        }
    }

    /// Create a new history suggestion
    pub fn history(prompt: &str) -> Self {
        // Truncate long prompts for display
        let display = if prompt.len() > 60 {
            format!("{}...", &prompt[..57])
        } else {
            prompt.to_string()
        };

        Self {
            display_text: display,
            insert_text: prompt.to_string(),
            description: None,
            suggestion_type: SuggestionType::History,
        }
    }
}

/// Autocomplete dropdown state
#[derive(Debug, Clone, PartialEq)]
pub struct AutocompleteState {
    /// Whether the dropdown is currently open
    pub is_open: bool,
    /// Current list of filtered suggestions
    pub suggestions: Vec<Suggestion>,
    /// Currently selected suggestion index (-1 = none)
    pub selected_index: i32,
    /// The trigger character that opened autocomplete (@ or /)
    pub trigger: Option<char>,
    /// Position in the input where the trigger was found
    pub trigger_position: usize,
    /// Current filter text (text after trigger)
    pub filter_text: String,
    /// Dataset name when in column mode (for @dataset. pattern)
    pub column_dataset: Option<String>,
}

impl Default for AutocompleteState {
    fn default() -> Self {
        Self::new()
    }
}

impl AutocompleteState {
    /// Create a new autocomplete state
    pub fn new() -> Self {
        Self {
            is_open: false,
            suggestions: Vec::new(),
            selected_index: -1,
            trigger: None,
            trigger_position: 0,
            filter_text: String::new(),
            column_dataset: None,
        }
    }

    /// Open autocomplete with a trigger
    pub fn open(&mut self, trigger: char, position: usize) {
        self.is_open = true;
        self.trigger = Some(trigger);
        self.trigger_position = position;
        self.filter_text.clear();
        self.selected_index = if self.suggestions.is_empty() { -1 } else { 0 };
        self.column_dataset = None;
    }

    /// Open autocomplete for column mode
    pub fn open_column_mode(&mut self, dataset_name: &str, position: usize) {
        self.is_open = true;
        self.trigger = Some('@');
        self.trigger_position = position;
        self.filter_text.clear();
        self.selected_index = if self.suggestions.is_empty() { -1 } else { 0 };
        self.column_dataset = Some(dataset_name.to_string());
    }

    /// Close autocomplete
    pub fn close(&mut self) {
        self.is_open = false;
        self.suggestions.clear();
        self.selected_index = -1;
        self.trigger = None;
        self.filter_text.clear();
        self.column_dataset = None;
    }

    /// Update the filter text
    pub fn set_filter(&mut self, filter: &str) {
        self.filter_text = filter.to_string();
        // Reset selection when filter changes
        self.selected_index = if self.suggestions.is_empty() { -1 } else { 0 };
    }

    /// Set suggestions
    pub fn set_suggestions(&mut self, suggestions: Vec<Suggestion>) {
        self.suggestions = suggestions;
        self.selected_index = if self.suggestions.is_empty() { -1 } else { 0 };
    }

    /// Navigate to previous suggestion (up arrow)
    pub fn navigate_up(&mut self) {
        if self.suggestions.is_empty() {
            return;
        }
        if self.selected_index <= 0 {
            self.selected_index = self.suggestions.len() as i32 - 1;
        } else {
            self.selected_index -= 1;
        }
    }

    /// Navigate to next suggestion (down arrow)
    pub fn navigate_down(&mut self) {
        if self.suggestions.is_empty() {
            return;
        }
        if self.selected_index >= self.suggestions.len() as i32 - 1 {
            self.selected_index = 0;
        } else {
            self.selected_index += 1;
        }
    }

    /// Get the currently selected suggestion
    pub fn get_selected(&self) -> Option<&Suggestion> {
        if self.selected_index >= 0 && (self.selected_index as usize) < self.suggestions.len() {
            Some(&self.suggestions[self.selected_index as usize])
        } else {
            None
        }
    }

    /// Check if currently in column mode
    pub fn is_column_mode(&self) -> bool {
        self.column_dataset.is_some()
    }

    /// Get the full text to replace (from trigger position to current cursor)
    pub fn get_replacement_range(&self) -> (usize, usize) {
        let start = self.trigger_position;
        let end = self.trigger_position + 1 + self.filter_text.len();
        // In column mode, include the dataset name and dot
        if let Some(ref ds) = self.column_dataset {
            // @dataset.filter -> start is @, end is after filter
            let full_len = 1 + ds.len() + 1 + self.filter_text.len(); // @ + dataset + . + filter
            (start, start + full_len)
        } else {
            (start, end)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_navigate() {
        let mut state = AutocompleteState::new();
        state.set_suggestions(vec![
            Suggestion::dataset("test1", 10, 3),
            Suggestion::dataset("test2", 20, 4),
            Suggestion::dataset("test3", 30, 5),
        ]);

        assert_eq!(state.selected_index, 0);

        state.navigate_down();
        assert_eq!(state.selected_index, 1);

        state.navigate_down();
        assert_eq!(state.selected_index, 2);

        // Wrap around
        state.navigate_down();
        assert_eq!(state.selected_index, 0);

        // Up from 0 wraps to end
        state.navigate_up();
        assert_eq!(state.selected_index, 2);
    }

    #[test]
    fn test_truncate_history() {
        let long_prompt =
            "This is a very long prompt that exceeds sixty characters and should be truncated";
        let suggestion = Suggestion::history(long_prompt);
        assert!(suggestion.display_text.ends_with("..."));
        assert!(suggestion.display_text.len() <= 63);
        assert_eq!(suggestion.insert_text, long_prompt);
    }
}
