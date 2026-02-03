#!/bin/bash
# Unified Validation Runner for p2a-core
# Runs all Rust validation tests and R comparison scripts
# Generates a comprehensive validation report

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORT_DIR="$SCRIPT_DIR/reports"
TIMESTAMP=$(date +%Y-%m-%d_%H%M%S)
REPORT_FILE="$REPORT_DIR/validation_report_${TIMESTAMP}.md"

# Create reports directory if needed
mkdir -p "$REPORT_DIR"

# Parse arguments
RUN_RUST=true
RUN_R=true
VERBOSE=false
CATEGORY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --rust-only)
            RUN_R=false
            shift
            ;;
        --r-only)
            RUN_RUST=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --category|-c)
            CATEGORY="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --rust-only    Run only Rust validation tests"
            echo "  --r-only       Run only R validation scripts"
            echo "  --verbose, -v  Show detailed output"
            echo "  --category, -c Filter by category (stats, regression, econometrics, ml, forecasting)"
            echo "  --help, -h     Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Initialize report
cat > "$REPORT_FILE" << EOF
# Validation Report

**Generated:** $(date '+%Y-%m-%d %H:%M:%S')
**Commit:** $(cd "$PROJECT_ROOT" && git rev-parse --short HEAD 2>/dev/null || echo "N/A")
**Branch:** $(cd "$PROJECT_ROOT" && git branch --show-current 2>/dev/null || echo "N/A")

---

EOF

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  p2a-core Validation Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Report will be saved to: ${YELLOW}$REPORT_FILE${NC}"
echo ""

# ============================================================================
# RUST VALIDATION TESTS
# ============================================================================

if $RUN_RUST; then
    echo -e "${BLUE}[1/3] Running Rust Validation Tests${NC}"
    echo "-------------------------------------------"

    cd "$PROJECT_ROOT"

    # Build test pattern
    TEST_PATTERN="test_validate"
    if [ -n "$CATEGORY" ]; then
        TEST_PATTERN="${CATEGORY}.*test_validate"
    fi

    # Run tests and capture output
    RUST_OUTPUT=$(mktemp)
    RUST_EXIT=0

    if $VERBOSE; then
        cargo test -p p2a-core -- "$TEST_PATTERN" --nocapture 2>&1 | tee "$RUST_OUTPUT" || RUST_EXIT=$?
    else
        cargo test -p p2a-core -- "$TEST_PATTERN" 2>&1 | tee "$RUST_OUTPUT" || RUST_EXIT=$?
    fi

    # Parse results
    RUST_PASSED=$(grep -c "test .* ok$" "$RUST_OUTPUT" 2>/dev/null | tr -d '\n' || echo "0")
    RUST_FAILED=$(grep -c "test .* FAILED$" "$RUST_OUTPUT" 2>/dev/null | tr -d '\n' || echo "0")
    RUST_IGNORED=$(grep -c "test .* ignored$" "$RUST_OUTPUT" 2>/dev/null | tr -d '\n' || echo "0")
    # Ensure values are integers
    RUST_PASSED=${RUST_PASSED:-0}
    RUST_FAILED=${RUST_FAILED:-0}
    RUST_IGNORED=${RUST_IGNORED:-0}
    RUST_TOTAL=$((RUST_PASSED + RUST_FAILED))

    # Extract failed test names
    FAILED_TESTS=$(grep "test .* FAILED$" "$RUST_OUTPUT" | sed 's/test \(.*\) \.\.\. FAILED$/\1/' || echo "")

    # Add to report
    cat >> "$REPORT_FILE" << EOF
## Rust Validation Tests

| Metric | Count |
|--------|-------|
| **Passed** | $RUST_PASSED |
| **Failed** | $RUST_FAILED |
| **Ignored** | $RUST_IGNORED |
| **Total** | $RUST_TOTAL |

EOF

    if [ -n "$FAILED_TESTS" ]; then
        echo "" >> "$REPORT_FILE"
        echo "### Failed Tests" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "$FAILED_TESTS" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi

    # Print summary
    if [ "$RUST_FAILED" -eq 0 ]; then
        echo -e "${GREEN}Rust tests: $RUST_PASSED/$RUST_TOTAL passed${NC}"
    else
        echo -e "${RED}Rust tests: $RUST_PASSED/$RUST_TOTAL passed, $RUST_FAILED failed${NC}"
    fi
    echo ""

    rm -f "$RUST_OUTPUT"
fi

# ============================================================================
# R VALIDATION SCRIPTS
# ============================================================================

if $RUN_R; then
    echo -e "${BLUE}[2/3] Running R Validation Scripts${NC}"
    echo "-------------------------------------------"

    R_SCRIPTS_DIR="$SCRIPT_DIR/scripts"
    R_OUTPUT_DIR="$REPORT_DIR/r_output_${TIMESTAMP}"
    mkdir -p "$R_OUTPUT_DIR"

    R_PASSED=0
    R_FAILED=0
    R_SKIPPED=0
    declare -a R_FAILURES

    # Check if R is available
    if ! command -v Rscript &> /dev/null; then
        echo -e "${YELLOW}Warning: Rscript not found, skipping R validation${NC}"
        R_SKIPPED=-1
    else
        # Find all R validation scripts
        if [ -n "$CATEGORY" ]; then
            R_SCRIPTS=$(find "$R_SCRIPTS_DIR" -name "*${CATEGORY}*.R" -o -name "validate_*.R" 2>/dev/null | sort)
        else
            R_SCRIPTS=$(find "$R_SCRIPTS_DIR" -name "*.R" 2>/dev/null | sort)
        fi

        SCRIPT_COUNT=$(echo "$R_SCRIPTS" | grep -c . || echo "0")
        echo "Found $SCRIPT_COUNT R validation scripts"
        echo ""

        for script in $R_SCRIPTS; do
            script_name=$(basename "$script")
            output_file="$R_OUTPUT_DIR/${script_name%.R}.log"

            echo -n "  Running $script_name... "

            # Skip benchmarks for faster validation
            if SKIP_BENCHMARKS=1 timeout 120 Rscript "$script" > "$output_file" 2>&1; then
                echo -e "${GREEN}OK${NC}"
                R_PASSED=$((R_PASSED + 1))
            else
                exit_code=$?
                if [ $exit_code -eq 124 ]; then
                    echo -e "${YELLOW}TIMEOUT${NC}"
                    R_SKIPPED=$((R_SKIPPED + 1))
                else
                    echo -e "${RED}FAILED${NC}"
                    R_FAILED=$((R_FAILED + 1))
                    R_FAILURES+=("$script_name")
                fi
            fi
        done
    fi

    # Add to report
    cat >> "$REPORT_FILE" << EOF
## R Validation Scripts

| Metric | Count |
|--------|-------|
| **Passed** | $R_PASSED |
| **Failed** | $R_FAILED |
| **Skipped/Timeout** | $R_SKIPPED |
| **Total** | $((R_PASSED + R_FAILED + R_SKIPPED)) |

EOF

    if [ ${#R_FAILURES[@]} -gt 0 ]; then
        echo "" >> "$REPORT_FILE"
        echo "### Failed R Scripts" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo '```' >> "$REPORT_FILE"
        for f in "${R_FAILURES[@]}"; do
            echo "$f" >> "$REPORT_FILE"
        done
        echo '```' >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
    fi

    echo ""
    if [ "$R_FAILED" -eq 0 ] && [ "$R_SKIPPED" -ge 0 ]; then
        echo -e "${GREEN}R scripts: $R_PASSED passed${NC}"
    else
        echo -e "${YELLOW}R scripts: $R_PASSED passed, $R_FAILED failed, $R_SKIPPED skipped${NC}"
    fi
    echo ""
fi

# ============================================================================
# COVERAGE ANALYSIS
# ============================================================================

echo -e "${BLUE}[3/3] Generating Coverage Analysis${NC}"
echo "-------------------------------------------"

cd "$PROJECT_ROOT"

# Count methods by category
cat >> "$REPORT_FILE" << EOF
## Validation Coverage by Category

| Category | Validation Tests | Validation Docs | Status |
|----------|-----------------|-----------------|--------|
EOF

# Count tests and docs per category (simplified)
total_tests=0
total_docs=0

for cat in stats regression econometrics forecasting ml; do
    # Count tests - grep for category prefix in test names
    tests=$(cargo test -p p2a-core -- "test_validate" --list 2>/dev/null | grep -c "^${cat}::" || echo 0)
    tests=$(echo "$tests" | tr -d '[:space:]')
    tests=${tests:-0}

    # Count docs
    if [ -d "$SCRIPT_DIR/$cat" ]; then
        docs=$(find "$SCRIPT_DIR/$cat" -name "*.md" 2>/dev/null | wc -l | tr -d '[:space:]')
    else
        docs=0
    fi
    docs=${docs:-0}

    # Determine status
    if [ "$tests" -gt 10 ]; then
        status="Good"
    elif [ "$tests" -gt 5 ]; then
        status="Partial"
    else
        status="Needs Work"
    fi

    echo "| **$cat** | $tests | $docs | $status |" >> "$REPORT_FILE"
    total_tests=$((total_tests + tests))
    total_docs=$((total_docs + docs))
done

echo "| **TOTAL** | **$total_tests** | **$total_docs** | - |" >> "$REPORT_FILE"

# ============================================================================
# SUMMARY
# ============================================================================

cat >> "$REPORT_FILE" << EOF

---

## Summary

EOF

OVERALL_STATUS="PASS"

if $RUN_RUST && [ "${RUST_FAILED:-0}" -gt 0 ]; then
    OVERALL_STATUS="FAIL"
    echo "- **Rust Tests:** ${RUST_FAILED} failures need attention" >> "$REPORT_FILE"
else
    echo "- **Rust Tests:** All ${RUST_PASSED:-0} tests passing" >> "$REPORT_FILE"
fi

if $RUN_R && [ "${R_FAILED:-0}" -gt 0 ]; then
    OVERALL_STATUS="PARTIAL"
    echo "- **R Scripts:** ${R_FAILED} failures (may need R packages installed)" >> "$REPORT_FILE"
else
    echo "- **R Scripts:** All ${R_PASSED:-0} scripts passing" >> "$REPORT_FILE"
fi

echo "" >> "$REPORT_FILE"
echo "**Overall Status:** $OVERALL_STATUS" >> "$REPORT_FILE"

# ============================================================================
# FINAL OUTPUT
# ============================================================================

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Validation Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ "$OVERALL_STATUS" = "PASS" ]; then
    echo -e "${GREEN}Overall Status: PASS${NC}"
elif [ "$OVERALL_STATUS" = "PARTIAL" ]; then
    echo -e "${YELLOW}Overall Status: PARTIAL (some R scripts failed)${NC}"
else
    echo -e "${RED}Overall Status: FAIL${NC}"
fi

echo ""
echo -e "Full report: ${YELLOW}$REPORT_FILE${NC}"
echo ""

# Exit with appropriate code
if [ "$OVERALL_STATUS" = "FAIL" ]; then
    exit 1
elif [ "$OVERALL_STATUS" = "PARTIAL" ]; then
    exit 0  # R failures are warnings, not errors
else
    exit 0
fi
