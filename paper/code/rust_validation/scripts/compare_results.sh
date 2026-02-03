#!/bin/bash
# compare_results.sh - Compare R and Rust JSON outputs
# Usage: ./scripts/compare_results.sh r_result.json rust_result.json [--tolerance 1e-6]

set -e

R_FILE=$1
RUST_FILE=$2
TOLERANCE=${3:-1e-6}

if [ ! -f "$R_FILE" ] || [ ! -f "$RUST_FILE" ]; then
    echo "Error: Both R and Rust result files required"
    exit 1
fi

# Use jq for comparison (requires jq)
if ! command -v jq &> /dev/null; then
    echo "Error: jq not found. Install with: apt install jq"
    exit 1
fi

# Python comparison script (more flexible for numeric comparison)
python3 << EOF
import json
import sys
import math

def load_json(path):
    with open(path) as f:
        return json.load(f)

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    items.extend(flatten_dict(item, f"{new_key}[{i}]", sep).items())
                else:
                    items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, v))
    return dict(items)

def compare_values(r_val, rust_val, tol):
    if isinstance(r_val, (int, float)) and isinstance(rust_val, (int, float)):
        if math.isnan(r_val) and math.isnan(rust_val):
            return True, 0.0
        if math.isnan(r_val) or math.isnan(rust_val):
            return False, float('inf')
        if r_val == 0 and rust_val == 0:
            return True, 0.0
        if r_val == 0:
            return abs(rust_val) < tol, abs(rust_val)
        rel_diff = abs((r_val - rust_val) / r_val)
        abs_diff = abs(r_val - rust_val)
        # Use relative tolerance for large values, absolute for small
        if abs(r_val) > 1:
            return rel_diff < tol, rel_diff
        else:
            return abs_diff < tol, abs_diff
    elif isinstance(r_val, str) and isinstance(rust_val, str):
        return r_val == rust_val, 0.0 if r_val == rust_val else 1.0
    else:
        return str(r_val) == str(rust_val), 0.0

try:
    r_data = load_json("$R_FILE")
    rust_data = load_json("$RUST_FILE")
    tolerance = float("$TOLERANCE")

    # Extract results section
    r_results = r_data.get('results', r_data)
    rust_results = rust_data.get('results', rust_data)

    # Flatten for comparison
    r_flat = flatten_dict(r_results)
    rust_flat = flatten_dict(rust_results)

    # Compare
    comparisons = []
    max_diff = 0.0
    all_pass = True

    # Fields to compare (numeric results)
    compare_fields = ['coefficients', 'std_errors', 't_values', 'p_values',
                      'r_squared', 'adj_r_squared', 'f_statistic',
                      'log_likelihood', 'aic', 'centers', 'loadings', 'sdev']

    for key, r_val in r_flat.items():
        # Skip metadata fields
        if any(skip in key for skip in ['timestamp', 'version', 'method', 'dataset', 'n_obs']):
            continue

        # Check if this is a field we care about
        if not any(f in key for f in compare_fields):
            continue

        if key in rust_flat:
            rust_val = rust_flat[key]
            passed, diff = compare_values(r_val, rust_val, tolerance)
            comparisons.append({
                'field': key,
                'r': r_val,
                'rust': rust_val,
                'diff': diff,
                'pass': passed
            })
            max_diff = max(max_diff, diff)
            if not passed:
                all_pass = False

    # Output comparison result
    result = {
        'r_file': "$R_FILE",
        'rust_file': "$RUST_FILE",
        'status': 'PASS' if all_pass else 'FAIL',
        'tolerance': tolerance,
        'max_diff': max_diff,
        'n_comparisons': len(comparisons),
        'comparisons': comparisons
    }

    print(json.dumps(result, indent=2))

    if not all_pass:
        print("\n--- Failed Comparisons ---", file=sys.stderr)
        for c in comparisons:
            if not c['pass']:
                print(f"  {c['field']}: R={c['r']}, Rust={c['rust']}, diff={c['diff']}", file=sys.stderr)
        sys.exit(1)

except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
EOF
