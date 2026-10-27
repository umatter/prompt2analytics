//! Helpers shared by the integration tests. A test binary that uses them
//! declares `mod common;`; the helpers it does not call are unused there,
//! which the workspace lints allow (`dead_code`).

pub mod parquet;
