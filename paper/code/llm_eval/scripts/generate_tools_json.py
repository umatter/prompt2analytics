#!/usr/bin/env python3
"""
Generate tools.json from MCP handler source files.
Parses the Rust code to extract tool definitions.
"""

import json
import re
import sys
from pathlib import Path

HANDLERS_DIR = Path("/home/umatter/tools/prompt2analytics/crates/p2a-mcp/src/tools/handlers")
OUTPUT_FILE = Path("/home/umatter/tools/prompt2analytics/paper/code/llm_eval/config/tools.json")


def parse_tool_definition(content: str, start_pos: int) -> dict | None:
    """Parse a single tool definition starting at #[tool("""
    snippet = content[start_pos:start_pos + 5000]

    # Find where description starts
    desc_start_match = re.search(r'#\[tool\(\s*description\s*=\s*"', snippet)
    if not desc_start_match:
        return None

    # Extract description by finding the closing " that ends the attribute
    # The description ends when we find `")` or `"  )` (quote followed by close paren)
    desc_content_start = desc_start_match.end()

    # Scan through to find the end of the description
    # Handle escaped quotes properly
    i = desc_content_start
    description_chars = []
    while i < len(snippet):
        char = snippet[i]
        if char == '\\' and i + 1 < len(snippet):
            # Escaped character - skip both
            next_char = snippet[i + 1]
            if next_char == '"':
                description_chars.append('"')
                i += 2
                continue
            elif next_char == 'n':
                description_chars.append(' ')
                i += 2
                continue
            else:
                description_chars.append(char)
                i += 1
                continue
        elif char == '"':
            # Check if this is followed by )] or whitespace then )] - end of description
            rest = snippet[i+1:i+10].strip()
            if rest.startswith(')') or rest.startswith(')]'):
                break
            else:
                # This quote is inside the description (shouldn't happen normally)
                description_chars.append(char)
                i += 1
        else:
            description_chars.append(char)
            i += 1

    description = ''.join(description_chars)
    # Clean up multi-line descriptions
    description = re.sub(r'\s+', ' ', description).strip()

    if not description or len(description) < 10:
        return None

    # Find the function name
    func_match = re.search(
        r'\]\s*pub\s+async\s+fn\s+(\w+)',
        snippet
    )
    if not func_match:
        return None

    func_name = func_match.group(1)

    return {
        "name": func_name,
        "description": description
    }


def extract_tools_from_file(filepath: Path) -> list[dict]:
    """Extract all tool definitions from a Rust handler file."""
    content = filepath.read_text()
    tools = []

    # Find all #[tool( occurrences
    for match in re.finditer(r'#\[tool\(', content):
        tool = parse_tool_definition(content, match.start())
        if tool:
            tools.append(tool)

    return tools


def create_openai_tool_format(tool: dict) -> dict:
    """Convert to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }


def main():
    all_tools = []

    # Process all handler files
    for handler_file in sorted(HANDLERS_DIR.glob("*.rs")):
        if handler_file.name == "mod.rs":
            continue

        tools = extract_tools_from_file(handler_file)
        print(f"{handler_file.name}: {len(tools)} tools")
        all_tools.extend(tools)

    # Remove duplicates by name
    seen = set()
    unique_tools = []
    for tool in all_tools:
        if tool["name"] not in seen:
            seen.add(tool["name"])
            unique_tools.append(tool)

    # Sort by name
    unique_tools.sort(key=lambda t: t["name"])

    # Convert to OpenAI format
    openai_tools = [create_openai_tool_format(t) for t in unique_tools]

    # Write output
    output = {"tools": openai_tools}
    OUTPUT_FILE.write_text(json.dumps(output, indent=2))

    print(f"\nTotal: {len(unique_tools)} unique tools")
    print(f"Output: {OUTPUT_FILE}")

    # Also print tool names for verification
    print("\nTool names:")
    for tool in unique_tools[:50]:
        print(f"  - {tool['name']}")
    if len(unique_tools) > 50:
        print(f"  ... and {len(unique_tools) - 50} more")


if __name__ == "__main__":
    main()
