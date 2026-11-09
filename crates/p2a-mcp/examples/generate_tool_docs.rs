//! Generate the MCP tool reference (`docs/mcp/tools-reference.md`) from the tool router.
//!
//! Run with: cargo run -p p2a-mcp --example generate_tool_docs
//!
//! README.md's tool table is written here too; `tests/tool_examples_doc.rs` includes this file and
//! checks both outputs.

use p2a_mcp::tools::registry::{CATEGORY_TABLE, OPTIONAL_TOOLS};
use p2a_mcp::tools::{
    ToolCategory, category_counts, generate_markdown_docs, get_registry, search_tools_scored,
    tool_count,
};
use std::path::{Path, PathBuf};

/// Marker lines of README.md's generated tool table.
pub const README_TABLE_BEGIN: &str = "<!-- BEGIN GENERATED: MCP tool table (cargo run -p p2a-mcp --example generate_tool_docs; do not edit by hand) -->";
pub const README_TABLE_END: &str = "<!-- END GENERATED: MCP tool table -->";

/// The workspace root (two levels above this crate).
pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// README.md's tool table: one row per category (ToolCategory::all() order) with every tool of the
/// category table, so it is the same in every build (the database tools are listed even without the
/// `database` feature) and names only registered tools (the registry tests require a row per tool).
pub fn readme_tool_table() -> String {
    let full = CATEGORY_TABLE.len();
    let mut md = format!(
        "The server registers {full} tools in a default build ({} without the `database` feature, which \
         CI's feature set leaves out). [docs/mcp/tools-reference.md](docs/mcp/tools-reference.md) \
         describes each one and names its R equivalent.\n\n",
        full - OPTIONAL_TOOLS.len()
    );
    md.push_str("| Category | Tools | Names |\n|---|---|---|\n");
    for c in ToolCategory::all() {
        let mut names: Vec<&str> = CATEGORY_TABLE
            .iter()
            .filter(|row| row.1 == *c)
            .map(|row| row.0)
            .collect();
        if names.is_empty() {
            continue;
        }
        names.sort_unstable();
        let list = names
            .iter()
            .map(|n| format!("`{n}`"))
            .collect::<Vec<_>>()
            .join(", ");
        md.push_str(&format!("| {c:?} | {} | {list} |\n", names.len()));
    }
    md.push_str(&format!("| **Total** | **{full}** | |\n"));
    md
}

fn section_range(doc: &str, begin: &str, end: &str) -> Result<std::ops::Range<usize>, String> {
    let start = doc
        .find(begin)
        .ok_or_else(|| format!("no `{begin}` line"))?;
    let body = start + begin.len();
    if doc[body..].contains(begin) {
        return Err(format!("`{begin}` appears twice"));
    }
    let len = doc[body..]
        .find(end)
        .ok_or_else(|| format!("no `{end}` line after `{begin}`"))?;
    Ok(body..body + len)
}

/// The text between two marker lines.
pub fn section_body<'a>(doc: &'a str, begin: &str, end: &str) -> Result<&'a str, String> {
    Ok(&doc[section_range(doc, begin, end)?])
}

/// What a section holds when `body` is current: a newline, the body, a newline.
pub fn wrap_body(body: &str) -> String {
    format!("\n{}\n", body.trim_end())
}

/// `doc` with the text between the markers replaced by `body`.
pub fn splice(doc: &str, begin: &str, end: &str, body: &str) -> Result<String, String> {
    let r = section_range(doc, begin, end)?;
    Ok(format!(
        "{}{}{}",
        &doc[..r.start],
        wrap_body(body),
        &doc[r.end..]
    ))
}

/// Writes through a temporary file and a rename, so a reader never sees a half-written file.
pub fn write_atomically(path: &Path, text: &str) -> Result<(), String> {
    let tmp = path.with_extension("md.tmp");
    std::fs::write(&tmp, text).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes README.md's tool table (any build) and docs/mcp/tools-reference.md (only with the
/// `database` feature, so the database tools are never dropped from it); returns the files written.
pub fn update() -> Result<Vec<&'static str>, String> {
    let mut written = Vec::new();
    let readme_path = repo_root().join("README.md");
    let readme = std::fs::read_to_string(&readme_path).map_err(|e| format!("README.md: {e}"))?;
    let new = splice(
        &readme,
        README_TABLE_BEGIN,
        README_TABLE_END,
        &readme_tool_table(),
    )
    .map_err(|e| format!("README.md: {e}"))?;
    if new != readme {
        write_atomically(&readme_path, &new)?;
        written.push("README.md");
    }
    if !cfg!(feature = "database") {
        return Err(format!(
            "docs/mcp/tools-reference.md not written: this build lacks the `database` feature, so its {} \
             tools would be dropped; run with the default features",
            OPTIONAL_TOOLS.len()
        ));
    }
    let reference = repo_root().join("docs/mcp/tools-reference.md");
    let want = generate_markdown_docs();
    if std::fs::read_to_string(&reference).ok().as_deref() != Some(want.as_str()) {
        write_atomically(&reference, &want)?;
        written.push("docs/mcp/tools-reference.md");
    }
    Ok(written)
}

fn main() {
    println!("MCP Tools Registry Summary");
    println!("==========================\n");

    println!("Total tools: {}\n", tool_count());

    println!("Tools by Category:");
    let counts = category_counts();
    for category in ToolCategory::all() {
        println!(
            "  {:?}: {} tools - {}",
            category,
            counts.get(category).copied().unwrap_or(0),
            category.description()
        );
    }
    println!();

    let docs = generate_markdown_docs();
    let output_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/mcp/tools-reference.md");
    if let Err(e) = std::fs::write(&output_path, &docs) {
        eprintln!("Failed to write {}: {e}", output_path.display());
        std::process::exit(1);
    }
    println!("Documentation written to: {}", output_path.display());

    println!("\nSearch Demo: top 5 for 'panel'");
    for r in search_tools_scored("panel", None, 5) {
        println!("  - {} (score {:.1})", r.tool.name, r.score);
    }

    println!("\nRegression tools (R counterpart):");
    for tool in get_registry()
        .iter()
        .filter(|t| t.category == ToolCategory::Regression)
    {
        println!(
            "  - {} (R: {})",
            tool.name,
            tool.r_equivalent.unwrap_or("-")
        );
    }
}
