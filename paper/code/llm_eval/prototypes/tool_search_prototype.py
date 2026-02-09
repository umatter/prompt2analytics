#!/usr/bin/env python3
"""
Prototype: Tool Search for scaling beyond 128 tools.

This demonstrates how a search_tools meta-tool could work,
allowing LLMs to discover relevant tools from a large library.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

# Load full tool library
TOOLS_FILE = Path(__file__).parent.parent / "config" / "tools_full_268.json"


@dataclass
class ToolMatch:
    name: str
    description: str
    category: str
    score: float
    keywords_matched: list[str]


class ToolSearchIndex:
    """Keyword-based tool search index."""

    # Category inference from tool name prefixes
    CATEGORY_PREFIXES = {
        "regression_": "regression",
        "panel_": "panel",
        "iv_": "causal",
        "diff_": "causal",
        "rd_": "causal",
        "treatment_": "causal",
        "synthetic_": "causal",
        "ts_": "timeseries",
        "timeseries_": "timeseries",
        "viz_": "visualization",
        "hypothesis_": "hypothesis",
        "anova_": "stats",
        "ml_": "ml",
        "spatial_": "spatial",
        "survival_": "survival",
        "discrete_": "discrete",
        "munge_": "munging",
        "cleaning_": "data",
        "db_": "database",
    }

    def __init__(self, tools_file: Path = TOOLS_FILE):
        with open(tools_file) as f:
            data = json.load(f)

        self.tools = []
        self.keyword_index = defaultdict(list)  # keyword -> [tool_idx]

        for i, tool in enumerate(data["tools"]):
            name = tool["function"]["name"]
            desc = tool["function"]["description"]
            category = self._infer_category(name)

            self.tools.append({
                "name": name,
                "description": desc,
                "category": category,
            })

            # Index by keywords from name and description
            keywords = self._extract_keywords(name, desc)
            for kw in keywords:
                self.keyword_index[kw].append(i)

    def _infer_category(self, name: str) -> str:
        """Infer tool category from name prefix."""
        for prefix, category in self.CATEGORY_PREFIXES.items():
            if name.startswith(prefix):
                return category
        return "general"

    def _extract_keywords(self, name: str, description: str) -> set[str]:
        """Extract searchable keywords from tool name and description."""
        # Split name by underscores
        name_parts = name.lower().split("_")

        # Extract words from description (alphanumeric only)
        desc_words = re.findall(r'\b[a-z]{3,}\b', description.lower())

        # Combine and deduplicate
        keywords = set(name_parts) | set(desc_words)

        # Add some domain-specific synonyms
        synonyms = {
            "ols": {"regression", "linear", "least", "squares"},
            "fe": {"fixed", "effects", "within"},
            "re": {"random", "effects", "gls"},
            "iv": {"instrumental", "variables", "endogenous", "2sls"},
            "did": {"difference", "differences", "treatment"},
            "rd": {"discontinuity", "regression", "sharp", "fuzzy"},
            "var": {"vector", "autoregression", "irf", "impulse"},
            "arima": {"autoregressive", "moving", "average", "forecast"},
            "garch": {"volatility", "heteroskedasticity", "arch"},
            "hausman": {"specification", "test", "fe", "re"},
        }

        for key, syns in synonyms.items():
            if key in keywords:
                keywords |= syns

        return keywords

    def search(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10
    ) -> list[ToolMatch]:
        """Search for tools matching query."""
        query_keywords = set(re.findall(r'\b[a-z]{2,}\b', query.lower()))

        # Score each tool
        scores = []
        for i, tool in enumerate(self.tools):
            # Skip if category filter doesn't match
            if category and tool["category"] != category:
                continue

            # Count keyword matches
            tool_keywords = self._extract_keywords(tool["name"], tool["description"])
            matched = query_keywords & tool_keywords

            if not matched:
                continue

            # Score based on matches
            # Boost exact name matches heavily
            name_parts = set(tool["name"].lower().split("_"))
            name_matches = query_keywords & name_parts

            score = len(matched) + 3 * len(name_matches)

            scores.append(ToolMatch(
                name=tool["name"],
                description=tool["description"][:200] + "..." if len(tool["description"]) > 200 else tool["description"],
                category=tool["category"],
                score=score,
                keywords_matched=list(matched),
            ))

        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:limit]

    def list_categories(self) -> dict[str, int]:
        """List available categories with tool counts."""
        counts = defaultdict(int)
        for tool in self.tools:
            counts[tool["category"]] += 1
        return dict(sorted(counts.items()))


def demo_search():
    """Demonstrate tool search functionality."""
    index = ToolSearchIndex()

    print("=" * 60)
    print("Tool Search Prototype Demo")
    print("=" * 60)

    # List categories
    print("\nAvailable categories:")
    for cat, count in index.list_categories().items():
        print(f"  {cat}: {count} tools")

    # Demo queries
    queries = [
        ("hausman test panel", None),
        ("robust standard errors", None),
        ("difference in differences", "causal"),
        ("forecast arima", "timeseries"),
        ("scatter plot", "visualization"),
        ("propensity score matching", "causal"),
        ("unit root test", None),
        ("heteroskedasticity test", None),
    ]

    for query, category in queries:
        print(f"\n{'=' * 60}")
        cat_str = f" (category={category})" if category else ""
        print(f"Query: '{query}'{cat_str}")
        print("-" * 60)

        results = index.search(query, category, limit=5)

        if not results:
            print("  No matches found")
            continue

        for i, match in enumerate(results, 1):
            print(f"  {i}. {match.name} [{match.category}] (score={match.score:.1f})")
            print(f"     Keywords: {', '.join(match.keywords_matched[:5])}")
            print(f"     {match.description[:100]}...")


def simulate_llm_workflow():
    """Simulate how an LLM would use tool search."""
    index = ToolSearchIndex()

    print("\n" + "=" * 60)
    print("Simulated LLM Workflow")
    print("=" * 60)

    # Scenario: User wants to test for serial correlation in panel data
    print("\nUser: 'Test for serial correlation in my panel residuals'")
    print("\nLLM thinking: I need to find a test for serial correlation...")
    print("LLM action: search_tools(query='serial correlation test panel')")

    results = index.search("serial correlation test panel", limit=5)

    print("\nServer response:")
    print(json.dumps([{
        "name": r.name,
        "description": r.description,
        "relevance": r.score
    } for r in results], indent=2))

    print("\nLLM thinking: 'regression_bgtest' (Breusch-Godfrey) is the right choice")
    print("LLM action: execute_tool('regression_bgtest', {dataset: '...', ...})")


if __name__ == "__main__":
    demo_search()
    simulate_llm_workflow()
