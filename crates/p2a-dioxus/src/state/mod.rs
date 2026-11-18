//! Application state management

pub mod autocomplete;
pub mod chat;
pub mod conversation;
pub mod session;
pub mod settings;

pub use chat::*;
pub use conversation::*;
pub use session::*;
pub use settings::*;

/// The dialogs open now, in the order they opened: Escape closes the last
/// one (`app::use_escape_close` registers a dialog while it is open).
#[derive(Debug, Clone)]
pub struct EscapeStack<T> {
    entries: Vec<(u64, T)>,
}

impl<T> Default for EscapeStack<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> EscapeStack<T> {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Register dialog `id` as the last opened (again, if it was registered).
    pub fn push(&mut self, id: u64, close: T) {
        self.remove(id);
        self.entries.push((id, close));
    }

    /// Unregister dialog `id`; an unknown id does nothing.
    pub fn remove(&mut self, id: u64) {
        self.entries.retain(|(entry, _)| *entry != id);
    }

    /// The close action of the last opened dialog.
    pub fn top(&self) -> Option<&T> {
        self.entries.last().map(|(_, close)| close)
    }
}

#[cfg(test)]
mod tests {
    use super::EscapeStack;

    #[test]
    fn escape_stack_closes_the_last_opened() {
        let mut stack = EscapeStack::new();
        assert_eq!(stack.top(), None);
        stack.push(1, "settings");
        stack.push(2, "delete dialog");
        assert_eq!(stack.top(), Some(&"delete dialog"));
        stack.remove(2);
        assert_eq!(stack.top(), Some(&"settings"));
        stack.remove(7);
        assert_eq!(stack.top(), Some(&"settings"));
        // Registering again moves a dialog to the top, once.
        stack.push(3, "shortcuts");
        stack.push(1, "settings");
        assert_eq!(stack.top(), Some(&"settings"));
        stack.remove(1);
        assert_eq!(stack.top(), Some(&"shortcuts"));
    }
}
