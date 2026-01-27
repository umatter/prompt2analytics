#!/usr/bin/env python3
"""
Generate speedup histogram from benchmark results.
Shows the distribution of Rust/R speedup ratios across all methods.
"""

import json
import os
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configuration
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results" / "benchmarks"
FIGURES_DIR = BASE_DIR / "figures"

# Categories and colors
CATEGORY_COLORS = {
    "regression": "#1f77b4",  # blue
    "econometrics": "#ff7f0e",  # orange
    "ml": "#2ca02c",  # green
    "munging": "#9467bd",  # purple
}

# Method to category mapping
METHOD_CATEGORIES = {
    "ols": "regression",
    "ols_hc0": "regression",
    "ols_hc1": "regression",
    "ols_hc2": "regression",
    "ols_hc3": "regression",
    "ols_robust": "regression",
    "panel_fe": "econometrics",
    "panel_re": "econometrics",
    "logit": "econometrics",
    "probit": "econometrics",
    "kmeans": "ml",
    "pca": "ml",
    "dbscan": "ml",
    "hierarchical": "ml",
    "sort": "munging",
    "filter": "munging",
    "group_by": "munging",
    "select": "munging",
    "standardize": "munging",
    "lag": "munging",
}


def parse_hyperfine(data):
    """Parse hyperfine JSON format"""
    if "results" in data:
        r = data["results"][0]
        return {
            "min_us": r.get("min", 0) * 1e6,
            "median_us": r.get("median", 0) * 1e6,
            "mean_us": r.get("mean", 0) * 1e6,
            "max_us": r.get("max", 0) * 1e6,
        }
    return data.get("timing", {})


def parse_r_bench(data):
    """Parse R bench output"""
    return data.get("timing", {})


def collect_speedups(target_n=100000):
    """Collect speedup data for all methods at given sample size"""
    speedups = []

    for r_file in glob.glob(str(RESULTS_DIR / f"r_*_n{target_n}.json")):
        try:
            with open(r_file) as f:
                r_data = json.load(f)

            # Extract method from filename
            basename = os.path.basename(r_file)
            parts = basename.replace(".json", "").split("_")
            method = "_".join(parts[1:-1])  # Between r_ and _nXXX

            # Find corresponding Rust file
            rust_file = r_file.replace("/r_", "/rust_")
            if not os.path.exists(rust_file):
                print(f"Warning: No Rust file for {method}")
                continue

            with open(rust_file) as f:
                rust_data = json.load(f)

            r_timing = parse_r_bench(r_data)
            rust_timing = parse_hyperfine(rust_data)

            r_median = r_timing.get("median_us", 0)
            rust_median = rust_timing.get("median_us", 0)

            if rust_median > 0 and r_median > 0:
                speedup = r_median / rust_median
                category = METHOD_CATEGORIES.get(method, "other")
                speedups.append({
                    "method": method,
                    "speedup": speedup,
                    "category": category,
                    "r_us": r_median,
                    "rust_us": rust_median,
                })
                print(f"  {method}: {speedup:.2f}x (R: {r_median:.0f}us, Rust: {rust_median:.0f}us)")

        except Exception as e:
            print(f"Warning: Failed to parse {r_file}: {e}")

    return speedups


def generate_histogram(speedups, output_path):
    """Generate histogram of speedup factors"""
    if not speedups:
        print("No speedup data available!")
        return

    # Sort by speedup for plotting
    speedups.sort(key=lambda x: x["speedup"], reverse=True)

    # Create figure with two subplots: histogram and bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Subplot 1: Histogram of speedup distribution ---
    values = [s["speedup"] for s in speedups]

    # Create histogram bins
    bins = np.concatenate([
        np.arange(0, 1.0, 0.2),  # 0-1x (R faster)
        np.arange(1.0, 5.0, 0.5),  # 1-5x
        np.arange(5.0, 11.0, 1.0),  # 5-10x
    ])

    ax1.hist(values, bins=bins, color="#3498db", edgecolor="white", alpha=0.8)
    ax1.axvline(x=1.0, color="red", linestyle="--", linewidth=2, label="Equal (1x)")
    ax1.axvline(x=np.median(values), color="green", linestyle="-", linewidth=2,
                label=f"Median ({np.median(values):.1f}x)")

    ax1.set_xlabel("Speedup Factor (R time / Rust time)", fontsize=12)
    ax1.set_ylabel("Number of Methods", fontsize=12)
    ax1.set_title(f"Distribution of Rust vs R Speedups (n=100,000)", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.set_xlim(0, max(values) * 1.1)

    # Add annotation for R faster region
    ax1.axvspan(0, 1, alpha=0.1, color="red")
    ax1.text(0.5, ax1.get_ylim()[1] * 0.9, "R faster", ha="center", fontsize=10, color="darkred")
    ax1.axvspan(1, max(bins), alpha=0.1, color="green")
    ax1.text(3, ax1.get_ylim()[1] * 0.9, "Rust faster", ha="center", fontsize=10, color="darkgreen")

    # --- Subplot 2: Bar chart by method ---
    methods = [s["method"].replace("_", " ").title() for s in speedups]
    vals = [s["speedup"] for s in speedups]
    colors = [CATEGORY_COLORS.get(s["category"], "gray") for s in speedups]

    bars = ax2.barh(range(len(methods)), vals, color=colors, edgecolor="white")
    ax2.set_yticks(range(len(methods)))
    ax2.set_yticklabels(methods)
    ax2.axvline(x=1.0, color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Speedup Factor", fontsize=12)
    ax2.set_title("Speedup by Method", fontsize=14)
    ax2.invert_yaxis()

    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, vals)):
        ax2.text(v + 0.1, i, f"{v:.1f}x", va="center", fontsize=9)

    # Add legend for categories
    legend_patches = [mpatches.Patch(color=c, label=cat.title())
                      for cat, c in CATEGORY_COLORS.items()]
    ax2.legend(handles=legend_patches, loc="lower right", fontsize=9)

    plt.tight_layout()

    # Save figure
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHistogram saved to: {output_path}")

    # Also save as PDF for LaTeX
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"PDF saved to: {pdf_path}")

    plt.close(fig)

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total methods: {len(speedups)}")
    print(f"  Mean speedup: {np.mean(values):.2f}x")
    print(f"  Median speedup: {np.median(values):.2f}x")
    print(f"  Min speedup: {min(values):.2f}x ({speedups[-1]['method']})")
    print(f"  Max speedup: {max(values):.2f}x ({speedups[0]['method']})")
    print(f"  Methods faster in Rust: {sum(1 for v in values if v > 1)}")
    print(f"  Methods faster in R: {sum(1 for v in values if v < 1)}")


def main():
    print("Collecting speedup data from benchmark results...")
    speedups = collect_speedups(target_n=100000)

    if not speedups:
        print("No benchmark results found. Run ./scripts/run_benchmark.sh first.")
        sys.exit(1)

    output_path = FIGURES_DIR / "speedup_histogram.png"
    generate_histogram(speedups, output_path)


if __name__ == "__main__":
    main()
