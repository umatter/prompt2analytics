//! Generate MCP tool documentation.
//!
//! Run with: cargo run -p p2a-mcp --example generate_tool_docs

use p2a_mcp::tools::{
    ToolCategory, category_counts, generate_markdown_docs, get_registry, search_tools, tool_count,
};
use std::fs;

fn main() {
    println!("MCP Tools Registry Summary");
    println!("==========================\n");

    // Total count
    println!("Total tools: {}\n", tool_count());

    // Category breakdown
    println!("Tools by Category:");
    let counts = category_counts();
    let mut sorted: Vec<_> = counts.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (category, count) in sorted {
        println!(
            "  {:?}: {} tools - {}",
            category,
            count,
            category.description()
        );
    }

    println!("\n");

    // Generate markdown docs
    let docs = generate_markdown_docs();

    // Write to file
    let output_path = "docs/mcp/tools-reference.md";
    if let Err(e) = fs::write(output_path, &docs) {
        eprintln!("Failed to write docs to {}: {}", output_path, e);
        // Print to stdout instead
        println!("{}", docs);
    } else {
        println!("Documentation written to: {}", output_path);
    }

    // Demo search
    println!("\nSearch Demo:");
    println!("Searching for 'panel'...");
    for tool in search_tools("panel").iter().take(5) {
        println!("  - {}: {}", tool.name, tool.description);
    }

    // Demo category filter
    println!("\nRegression tools:");
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
