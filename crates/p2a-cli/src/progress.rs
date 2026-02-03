//! Progress indicators for long-running CLI operations
//!
//! Provides spinners and progress bars for operations that may take time,
//! improving user experience by showing activity and progress.

use indicatif::{ProgressBar, ProgressStyle};
use std::time::Duration;

/// Create a spinner for indeterminate-length operations.
///
/// Use this when you don't know how long an operation will take
/// (e.g., loading a file of unknown size, running an optimization).
pub fn create_spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::default_spinner()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
            .template("{spinner:.cyan} {msg}")
            .expect("Invalid spinner template"),
    );
    spinner.set_message(message.to_string());
    spinner.enable_steady_tick(Duration::from_millis(80));
    spinner
}

/// Create a progress bar for operations with known length.
///
/// Use this when you know the total number of items to process
/// (e.g., processing N rows, running N iterations).
pub fn create_progress_bar(total: u64, message: &str) -> ProgressBar {
    let bar = ProgressBar::new(total);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .expect("Invalid progress bar template")
            .progress_chars("█▓░"),
    );
    bar.set_message(message.to_string());
    bar
}

/// Create a progress bar for byte-based operations (file loading).
pub fn create_bytes_progress_bar(total_bytes: u64, message: &str) -> ProgressBar {
    let bar = ProgressBar::new(total_bytes);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
            .expect("Invalid progress bar template")
            .progress_chars("█▓░"),
    );
    bar.set_message(message.to_string());
    bar
}

/// Wrapper that conditionally shows progress based on quiet mode.
pub struct Progress {
    bar: Option<ProgressBar>,
}

impl Progress {
    /// Create a new progress indicator (spinner) if not in quiet mode.
    pub fn spinner(message: &str, quiet: bool) -> Self {
        if quiet {
            Self { bar: None }
        } else {
            Self {
                bar: Some(create_spinner(message)),
            }
        }
    }

    /// Create a new progress bar if not in quiet mode.
    pub fn bar(total: u64, message: &str, quiet: bool) -> Self {
        if quiet {
            Self { bar: None }
        } else {
            Self {
                bar: Some(create_progress_bar(total, message)),
            }
        }
    }

    /// Update the progress message.
    pub fn set_message(&self, message: &str) {
        if let Some(ref bar) = self.bar {
            bar.set_message(message.to_string());
        }
    }

    /// Increment the progress by one.
    pub fn inc(&self, delta: u64) {
        if let Some(ref bar) = self.bar {
            bar.inc(delta);
        }
    }

    /// Set the current position.
    pub fn set_position(&self, pos: u64) {
        if let Some(ref bar) = self.bar {
            bar.set_position(pos);
        }
    }

    /// Finish the progress indicator with a message.
    pub fn finish_with_message(&self, message: &str) {
        if let Some(ref bar) = self.bar {
            bar.finish_with_message(message.to_string());
        }
    }

    /// Finish and clear the progress indicator.
    pub fn finish_and_clear(&self) {
        if let Some(ref bar) = self.bar {
            bar.finish_and_clear();
        }
    }
}
