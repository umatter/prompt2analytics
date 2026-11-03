//! Generate the MCP tool reference (`docs/mcp/tools-reference.md`) from the tool router.
//!
//! Run with: cargo run -p p2a-mcp --example generate_tool_docs

use p2a_mcp::tools::{
    ToolCategory, category_counts, generate_markdown_docs, get_registry, search_tools_scored,
    tool_count,
};
use std::path::Path;

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
