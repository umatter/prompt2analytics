//! Utility functions

mod markdown;
mod text;

pub use markdown::render_markdown;
pub use text::truncate_on_char_boundary;
