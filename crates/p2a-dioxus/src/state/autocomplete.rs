//! Autocomplete state management for chat input

use std::ops::Range;

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
            format!("{}...", crate::utils::truncate_on_char_boundary(prompt, 57))
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

/// Find the **byte offset** of the last occurrence of a trigger character
/// (`@` or `/`) that could start an autocomplete pattern — i.e. one at the
/// start of the input or immediately after whitespace.
///
/// Returns a byte offset (not a char index) so callers can slice the input
/// string directly. The trigger characters are single-byte ASCII, so the
/// offset immediately after the trigger is always a valid char boundary.
/// (The previous version returned a `Vec<char>` index that callers used as a
/// byte index — corrupting the filter text, or panicking, whenever a
/// multi-byte character preceded the trigger.) Moved from components/chat_input.rs.
pub fn find_last_trigger(input: &str, trigger: char) -> Option<usize> {
    let mut result = None;
    let mut prev_is_boundary = true; // start of string counts as a boundary
    for (byte_idx, c) in input.char_indices() {
        if c == trigger && prev_is_boundary {
            result = Some(byte_idx);
        }
        prev_is_boundary = c.is_whitespace();
    }
    result
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

    /// Keep the dropdown on `trigger` at `position` for this input. Opens it (filter and selection
    /// reset) unless it is already open on the same trigger outside column mode, and records
    /// `position` either way, so the stored offset always matches the text.
    pub fn track(&mut self, trigger: char, position: usize) {
        if !self.is_open || self.trigger != Some(trigger) || self.is_column_mode() {
            self.open(trigger, position);
        }
        self.trigger_position = position;
    }

    /// [`track`](Self::track) for `@dataset.` column mode.
    pub fn track_column_mode(&mut self, dataset_name: &str, position: usize) {
        if !self.is_open
            || !self.is_column_mode()
            || self.column_dataset.as_deref() != Some(dataset_name)
        {
            self.open_column_mode(dataset_name, position);
        }
        self.trigger_position = position;
    }

    /// Keep the dropdown open on prompt-history matches: no trigger, so accepting replaces the whole
    /// input. A trigger left from an earlier input is cleared.
    pub fn track_history(&mut self) {
        self.is_open = true;
        self.trigger = None;
        self.trigger_position = 0;
        self.column_dataset = None;
    }

    /// The bytes of `current` that accepting a suggestion replaces: the trigger, the dataset prefix
    /// in column mode, and the filter. The recorded offset is used only if `current` has the trigger
    /// there; otherwise the trigger is looked up again, so a stale offset cannot shift the range.
    /// Names may hold spaces, so the filter text is matched as recorded; if the text no longer has
    /// it, the range runs to the next whitespace.
    /// `None` in history mode or when the text no longer contains the trigger.
    pub fn replacement_range(&self, current: &str) -> Option<Range<usize>> {
        let trigger = self.trigger?;
        let start = current
            .get(self.trigger_position..)
            .filter(|rest| rest.starts_with(trigger))
            .map(|_| self.trigger_position)
            .or_else(|| find_last_trigger(current, trigger))?;
        let after_trigger = current.get(start..)?.strip_prefix(trigger)?;
        let body = match &self.column_dataset {
            Some(dataset) => after_trigger
                .strip_prefix(dataset.as_str())
                .and_then(|rest| rest.strip_prefix('.'))
                .unwrap_or(after_trigger),
            None => after_trigger,
        };
        let body_start = current.len() - body.len();
        let filter_len = if body.starts_with(self.filter_text.as_str()) {
            self.filter_text.len()
        } else {
            body.find(char::is_whitespace).unwrap_or(body.len())
        };
        Some(start..body_start + filter_len)
    }

    /// The input after accepting `suggestion`, ending in one space.
    ///
    /// With a trigger, the [`replacement_range`](Self::replacement_range) becomes the trigger plus
    /// `insert_text`, and the text after it is kept. In history mode, the suggestion replaces the
    /// whole input. If the trigger is gone, `current` comes back unchanged. Every slice is checked
    /// (`str::get`), so no offset can panic.
    pub fn apply(&self, current: &str, suggestion: &Suggestion) -> String {
        let Some(trigger) = self.trigger else {
            return format!("{} ", suggestion.insert_text.trim_end());
        };
        let Some(range) = self.replacement_range(current) else {
            return current.to_string();
        };
        let before = current.get(..range.start).unwrap_or_default();
        let after = current.get(range.end..).unwrap_or_default().trim_start();
        let value = format!("{before}{trigger}{} {after}", suggestion.insert_text);
        format!("{} ", value.trim_end())
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

    /// An offset that no longer matches the text (past its end, inside a character, or not on a
    /// trigger) is not used: the trigger is found in the text, and a text without one is unchanged.
    #[test]
    fn test_apply_with_a_stale_offset() {
        let suggestion = Suggestion::dataset("sales", 10, 3);
        for (offset, current, expected) in [
            (9, "@s", "@sales "),
            (1, "中 @s", "中 @sales "),
            (0, "x @s", "x @sales "),
            (0, "no trigger here", "no trigger here"),
        ] {
            let mut state = AutocompleteState::new();
            state.open('@', offset);
            state.set_filter("s");
            assert_eq!(state.apply(current, &suggestion), expected, "{current}");
        }
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

#[cfg(test)]
mod trigger_tests {
    use super::find_last_trigger;

    #[test]
    fn returns_byte_offset_after_multibyte_chars() {
        // "café @data" — 'é' is 2 bytes, so the '@' is at byte offset 6, not
        // char index 5. The returned offset must let `&input[pos + 1..]` slice
        // correctly to "data".
        let input = "café @data";
        let at = find_last_trigger(input, '@').unwrap();
        assert_eq!(&input[at..at + 1], "@");
        assert_eq!(&input[at + 1..], "data");
    }

    #[test]
    fn requires_start_or_whitespace_before_trigger() {
        // '@' embedded in a word (email-like) is not a trigger.
        assert_eq!(find_last_trigger("foo@bar", '@'), None);
        // At start.
        assert_eq!(find_last_trigger("@ds", '@'), Some(0));
        // Picks the last valid trigger.
        let input = "@a @b";
        let at = find_last_trigger(input, '@').unwrap();
        assert_eq!(&input[at + 1..], "b");
    }

    #[test]
    fn slash_trigger_after_multibyte() {
        let input = "λ /tool";
        let slash = find_last_trigger(input, '/').unwrap();
        assert_eq!(&input[slash + 1..], "tool");
    }
}
