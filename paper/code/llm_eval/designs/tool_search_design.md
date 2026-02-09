# Tool Search Design: Scaling Beyond 128 Tools

## Problem Statement

OpenAI's API limits tool calls to 128 tools per request. The p2a-mcp server exposes 268 tools. Current workaround uses a filtered 55-tool subset, but this limits functionality.

## Proposed Solution: Tool Search Meta-Tool

Instead of sending all tools upfront, we expose a small set of "meta-tools" that help the LLM discover and use the full tool library dynamically.

### Core Meta-Tools (Always Exposed)

```json
{
  "tools": [
    {
      "name": "search_tools",
      "description": "Search for analytics tools by natural language description. Returns matching tools with their full specifications. Use this when you need to find the right tool for a task.",
      "parameters": {
        "query": "Natural language description of what you want to do",
        "category": "Optional: regression, panel, causal, timeseries, stats, ml, viz, data",
        "limit": "Max tools to return (default: 10)"
      }
    },
    {
      "name": "execute_tool",
      "description": "Execute a tool discovered via search_tools. Pass the tool name and parameters.",
      "parameters": {
        "tool_name": "Name of the tool to execute",
        "arguments": "Tool-specific arguments as JSON object"
      }
    },
    {
      "name": "list_tool_categories",
      "description": "List available tool categories and their descriptions."
    }
  ]
}
```

### Workflow Example

```
User: "I have panel data and want to test whether to use fixed or random effects"

Step 1: LLM calls search_tools
─────────────────────────────
LLM: search_tools(query="panel fixed effects random effects hausman test", category="panel")

Server returns:
{
  "matches": [
    {
      "name": "hausman_test",
      "description": "Run Hausman specification test to choose between FE and RE models...",
      "parameters": {
        "dataset": "string - Name of loaded dataset",
        "y_col": "string - Dependent variable",
        "x_cols": "array - Independent variables",
        "entity_col": "string - Panel entity identifier",
        "time_col": "string - Time identifier"
      },
      "relevance_score": 0.95
    },
    {
      "name": "panel_fixed_effects",
      "description": "Estimate within-estimator fixed effects model...",
      "parameters": {...},
      "relevance_score": 0.82
    },
    {
      "name": "panel_random_effects",
      "description": "Estimate GLS random effects model...",
      "parameters": {...},
      "relevance_score": 0.80
    }
  ],
  "total_matches": 3
}

Step 2: LLM calls execute_tool
──────────────────────────────
LLM: execute_tool(
  tool_name="hausman_test",
  arguments={
    "dataset": "grunfeld",
    "y_col": "inv",
    "x_cols": ["value", "capital"],
    "entity_col": "firm",
    "time_col": "year"
  }
)

Server returns: Hausman test results...
```

## Implementation Architecture

### 1. Tool Index (Semantic Search)

Build an embedding-based index of all tools:

```rust
// p2a-mcp/src/tools/search.rs

pub struct ToolIndex {
    tools: Vec<ToolMetadata>,
    embeddings: Vec<Vec<f32>>,  // Pre-computed embeddings
}

impl ToolIndex {
    pub fn search(&self, query: &str, limit: usize) -> Vec<ToolMatch> {
        let query_embedding = self.embed(query);

        // Cosine similarity search
        let mut scores: Vec<(usize, f32)> = self.embeddings
            .iter()
            .enumerate()
            .map(|(i, emb)| (i, cosine_similarity(&query_embedding, emb)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scores.into_iter()
            .take(limit)
            .map(|(i, score)| ToolMatch {
                tool: self.tools[i].clone(),
                relevance_score: score,
            })
            .collect()
    }
}
```

### 2. Keyword + Semantic Hybrid Search

Combine exact keyword matching with semantic similarity:

```rust
pub fn hybrid_search(&self, query: &str, category: Option<&str>, limit: usize) -> Vec<ToolMatch> {
    // 1. Filter by category if specified
    let candidates = match category {
        Some(cat) => self.tools.iter().filter(|t| t.category == cat).collect(),
        None => self.tools.iter().collect(),
    };

    // 2. Keyword matching (boost exact matches)
    let keywords: HashSet<&str> = query.split_whitespace().collect();

    // 3. Semantic similarity
    let query_emb = self.embed(query);

    // 4. Combined scoring
    candidates.iter()
        .map(|tool| {
            let keyword_score = keyword_match_score(tool, &keywords);
            let semantic_score = cosine_similarity(&query_emb, &tool.embedding);
            let combined = 0.3 * keyword_score + 0.7 * semantic_score;
            ToolMatch { tool: tool.clone(), relevance_score: combined }
        })
        .sorted_by(|a, b| b.relevance_score.partial_cmp(&a.relevance_score).unwrap())
        .take(limit)
        .collect()
}
```

### 3. Category-Based Fallback

For simpler deployments without embeddings:

```rust
pub struct ToolCategories {
    categories: HashMap<String, Vec<ToolMetadata>>,
}

impl ToolCategories {
    pub fn get_category(&self, name: &str) -> Vec<&ToolMetadata> {
        self.categories.get(name).map(|v| v.iter().collect()).unwrap_or_default()
    }

    pub fn keyword_search(&self, query: &str) -> Vec<&ToolMetadata> {
        let terms: Vec<&str> = query.to_lowercase().split_whitespace().collect();

        self.categories.values()
            .flatten()
            .filter(|tool| {
                terms.iter().any(|term|
                    tool.name.contains(term) ||
                    tool.description.to_lowercase().contains(term)
                )
            })
            .collect()
    }
}
```

### 4. Tool Metadata Structure

```rust
#[derive(Clone, Serialize)]
pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    pub category: String,
    pub subcategory: Option<String>,
    pub parameters: JsonSchema,
    pub examples: Vec<ToolExample>,
    pub related_tools: Vec<String>,
    pub keywords: Vec<String>,
}

#[derive(Clone, Serialize)]
pub struct ToolExample {
    pub description: String,
    pub arguments: serde_json::Value,
    pub expected_output_summary: String,
}
```

## API Changes

### New Endpoints

```
POST /tools/search
{
  "query": "panel data hausman test",
  "category": "panel",
  "limit": 10
}

POST /tools/execute
{
  "tool_name": "hausman_test",
  "arguments": {...}
}

GET /tools/categories
```

### LLM System Prompt Update

```
You have access to a comprehensive econometrics toolkit with 268+ tools.

To find the right tool:
1. Use search_tools(query="your task description") to discover relevant tools
2. Review the returned tools and their parameters
3. Use execute_tool(tool_name, arguments) to run the selected tool

Available categories: regression, panel, causal, timeseries, stats, hypothesis,
                      ml, viz, data, munging, spatial, survival, discrete

Example workflow:
User: "Test for serial correlation in my panel data"
You: search_tools(query="serial correlation panel test", category="panel")
→ Returns: regression_bgtest, timeseries_box_test, ...
You: execute_tool("regression_bgtest", {"dataset": "mydata", ...})
```

## Embedding Options

### Option A: Local Embeddings (No External API)
- Use `rust-bert` or `candle` for local sentence embeddings
- Model: all-MiniLM-L6-v2 (22M parameters, fast)
- Pre-compute tool embeddings at build time

### Option B: External Embedding API
- Use OpenAI's text-embedding-3-small
- Cache tool embeddings (they don't change)
- Only embed user queries at runtime

### Option C: Keyword-Only (Simplest)
- No embeddings needed
- Use TF-IDF or BM25 for keyword matching
- Fast, deterministic, no dependencies

## Evaluation Metrics

Track these to measure effectiveness:

1. **Search Precision@K**: Of top K results, how many are relevant?
2. **Tool Discovery Rate**: % of tasks where correct tool is in search results
3. **Extra Roundtrips**: Average searches needed before finding right tool
4. **End-to-End Accuracy**: Task completion rate with search vs. without

## Migration Path

### Phase 1: Add search_tools (backward compatible)
- Add tool search alongside existing direct tool exposure
- For <128 tools, continue exposing directly
- For >128 tools, require search first

### Phase 2: Hybrid Mode
- Expose top 50 "core" tools directly
- Remaining 218 tools via search only
- LLM learns when to search vs. use direct tools

### Phase 3: Search-First Mode
- Only expose meta-tools by default
- All tool usage goes through search → execute
- Maximum scalability (supports 1000s of tools)

## Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Tool Search** | Scales to 1000s of tools, flexible | Extra roundtrip, search quality matters |
| **Categories** | Simple, predictable | Fixed hierarchy, may not match user mental model |
| **Dynamic Loading** | Context-aware | Complex state management, hard to predict |
| **Chunked Calls** | No code changes | Multiple API calls, higher latency/cost |

## Recommendation

Start with **Option C (Keyword-Only)** for simplicity, then upgrade to **Option A (Local Embeddings)** for better semantic matching. This gives:

1. Zero external dependencies
2. Fast search (<10ms)
3. Good enough accuracy for most queries
4. Path to semantic search later

The key insight is that tool names and descriptions are already quite descriptive, so keyword matching works surprisingly well. Embeddings help with synonyms and conceptual queries but aren't strictly necessary.
