//! Utility functions

mod markdown;
mod text;
mod time;

pub use markdown::{render_answer, render_markdown};
pub use text::{js_single_quoted, truncate_on_char_boundary};
pub use time::{local_clock, local_stamp};
