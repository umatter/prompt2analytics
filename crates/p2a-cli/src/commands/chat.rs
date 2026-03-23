//! Re-export chat functionality from the lightweight `p2a-chat` crate.
//!
//! The chat REPL is a pure HTTP/SSE client with no dependency on p2a-core,
//! so it lives in its own crate for fast standalone builds.

pub use p2a_chat::{ChatArgs, run};
