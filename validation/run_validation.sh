#!/bin/bash
# validation/run_validation.sh -- run the validation suite and write a results report.
#
# Rust: the test_validate tests of p2a-core, p2a-cli and p2a-mcp on the CI feature sets,
# listed with `cargo test -- --list`, selected by module path and test binary
# (validation/tools/validation_status.py select) and run with --exact per test binary,
# plus every self-test of p2a-validation. A non-zero cargo exit, a compile error, fewer
# tests run than selected, or no selected test_validate test at all is a failure.
# Full runs (no --category) also lint the methods registry (validation/methods/).
#
# R: the drift check `Rscript validation/r/run.R --check`, which recomputes every R
# reference case and compares it with validation/expected/<id>.json. The legacy scripts
# validation/scripts/*.R run only with --legacy-r, each in a scratch directory with a
# copy of validation/datasets, so they never write tracked files; they are reported and
# never counted as passed.
#
# Results: validation/reports/<timestamp>-<pid>/ (gitignored): the cargo, R and lint
# logs, results.json and report.md (rendered by validation_status.py render).
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: validation/run_validation.sh [--rust-only | --r-only] [--category <cat>] [--release]
                                    [--legacy-r] [--write-status] [--verbose]
  -c is --category, -v is --verbose, -h/--help prints this and exits 0.
  <cat>: stats | regression | econometrics | forecasting | ml | cli | mcp
Exit codes: 0 PASS, 1 FAIL, 2 usage error, 3 INCOMPLETE.
Environment: P2A_RSCRIPT (default Rscript), P2A_LEGACY_R_TIMEOUT (default 300), PYTHON (default python3).
EOF
}

usage_error() {
    echo "run_validation.sh: $1" >&2
    usage >&2
    exit 2
}

ORIG_ARGS="$*"
RUST_ONLY=false
R_ONLY=false
CATEGORY=""
CATEGORY_GIVEN=false
RELEASE=false
LEGACY=false
WRITE_STATUS=false
VERBOSE=false
while [ $# -gt 0 ]; do
    case "$1" in
        --rust-only) RUST_ONLY=true ;;
        --r-only) R_ONLY=true ;;
        --category | -c)
            [ $# -ge 2 ] || usage_error "$1 needs a category"
            CATEGORY="$2"
            CATEGORY_GIVEN=true
            shift
            ;;
        --category=*)
            CATEGORY="${1#--category=}"
            CATEGORY_GIVEN=true
            ;;
        --release) RELEASE=true ;;
        --legacy-r) LEGACY=true ;;
        --write-status) WRITE_STATUS=true ;;
        --verbose | -v) VERBOSE=true ;;
        --help | -h)
            usage
            exit 0
            ;;
        *) usage_error "unknown option: $1" ;;
    esac
    shift
done

if $CATEGORY_GIVEN; then
    case "$CATEGORY" in
        stats | regression | econometrics | forecasting | ml | cli | mcp) ;;
        "") usage_error "--category needs a category" ;;
        *) usage_error "unknown category: $CATEGORY" ;;
    esac
fi
if $RUST_ONLY && $R_ONLY; then usage_error "--rust-only and --r-only exclude each other"; fi
if $LEGACY && $RUST_ONLY; then usage_error "--legacy-r runs R scripts; it cannot be combined with --rust-only"; fi
if $WRITE_STATUS && [ -n "$CATEGORY" ]; then
    usage_error "--write-status describes the whole suite; it cannot be combined with --category"
fi
RUN_RUST=true
RUN_R=true
if $R_ONLY; then RUN_RUST=false; fi
if $RUST_ONLY; then RUN_R=false; fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python3}"
P2A_RSCRIPT="${P2A_RSCRIPT:-Rscript}"
P2A_LEGACY_R_TIMEOUT="${P2A_LEGACY_R_TIMEOUT:-300}"
case "$P2A_LEGACY_R_TIMEOUT" in
    '' | *[!0-9]*) usage_error "P2A_LEGACY_R_TIMEOUT must be a number of seconds" ;;
esac
STATUS_PY="validation/tools/validation_status.py"
REPORT_DIR="validation/reports/$(date +%Y-%m-%d_%H%M%S)-$$"
RESULTS="$REPORT_DIR/results.json"
R_PACKAGES='install.packages(c("jsonlite", "plm", "fixest", "sandwich", "lmtest", "car", "tseries", "quantreg", "nlme", "aod", "strucchange", "nnet", "MASS"))'

if [ -t 1 ]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    BLUE=$'\033[0;34m'
    NC=$'\033[0m'
else
    RED=""
    GREEN=""
    YELLOW=""
    BLUE=""
    NC=""
fi

FAILURES=()
INCOMPLETE=false
SCRATCH_DIRS=()

cleanup() {
    local d
    for d in ${SCRATCH_DIRS[@]+"${SCRATCH_DIRS[@]}"}; do
        rm -rf "$d"
    done
}
trap cleanup EXIT

fail() {
    FAILURES+=("$1")
}

# run_logged <log> <command...>: run a command into <log> (also to the terminal with
# --verbose) and return its own exit code.
run_logged() {
    local log="$1"
    shift
    local rc
    set +e
    if $VERBOSE; then
        "$@" 2>&1 | tee "$log"
        rc=${PIPESTATUS[0]}
    else
        "$@" > "$log" 2>&1
        rc=$?
    fi
    set -e
    return "$rc"
}

# status_py <subcommand> <args...>: run validation_status.py and keep its exit code in
# STATUS_RC. Exit 1 is a verdict the caller handles; any other non-zero exit (2:
# malformed input or usage; 127: no Python) is recorded as a failure of the run.
STATUS_RC=0
status_py() {
    STATUS_RC=0
    "$PYTHON" "$STATUS_PY" "$@" || STATUS_RC=$?
    if [ "$STATUS_RC" -ne 0 ] && [ "$STATUS_RC" -ne 1 ]; then
        fail "validation_status.py $1 exited $STATUS_RC"
    fi
    return 0
}

# collect_line <prefix> <collect args...>: run `validation_status.py collect` and set
# COLLECT_LINE to the first line of its output that starts with <prefix>; the whole
# output stays in $COLLECT_OUT.
COLLECT_LINE=""
COLLECT_OUT=""
collect_line() {
    local prefix="$1"
    shift
    COLLECT_OUT="$REPORT_DIR/collect-output.txt"
    status_py collect "$@" > "$COLLECT_OUT"
    COLLECT_LINE="$(grep -F -- "$prefix" "$COLLECT_OUT" | head -n 1 || true)"
}

features_of() {
    case "$1" in
        p2a-core) echo "--no-default-features --features ci" ;;
        p2a-cli) echo "--no-default-features --features spectral-analysis" ;;
        p2a-mcp) echo "--no-default-features --features http,llm,auth,websocket" ;;
        *) echo "" ;;
    esac
}

slug_of() {
    case "$1" in
        --lib) echo lib ;;
        --bins) echo bins ;;
        --bin\ *) echo "bin-${1#--bin }" ;;
        --test\ *) echo "test-${1#--test }" ;;
        *) echo "$1" | tr -c 'A-Za-z0-9_.\n-' '_' ;;
    esac
}

mkdir -p "$REPORT_DIR"
echo "${BLUE}p2a validation runner${NC}"
echo "Report directory: $PROJECT_ROOT/$REPORT_DIR"
echo ""

INIT=(--command "validation/run_validation.sh${ORIG_ARGS:+ $ORIG_ARGS}" -o "$RESULTS")
if [ -n "$CATEGORY" ]; then
    INIT+=(--category "$CATEGORY")
fi
status_py collect "${INIT[@]}" > /dev/null

# ------------------------------------------------------------------------------------
# Rust
# ------------------------------------------------------------------------------------
LIST_FAILED=false
RUST_RAN=false
SELECTED_VALIDATE=0
if $RUN_RUST; then
    echo "${BLUE}Rust tests${NC}"
    if ! command -v cargo > /dev/null 2>&1; then
        echo "${YELLOW}Rust tests: INCOMPLETE (cargo not found)${NC}"
        INCOMPLETE=true
    else
        RUST_RAN=true
        if [ -z "$CATEGORY" ]; then
            PACKAGES=(p2a-core p2a-cli p2a-mcp)
        else
            case "$CATEGORY" in
                cli) PACKAGES=(p2a-cli) ;;
                mcp) PACKAGES=(p2a-mcp) ;;
                *) PACKAGES=(p2a-core) ;;
            esac
        fi
        PACKAGES+=(p2a-validation)
        for pkg in "${PACKAGES[@]}"; do
            features="$(features_of "$pkg")"
            FEAT=()
            if [ -n "$features" ]; then
                read -r -a FEAT <<< "$features"
            fi
            if $RELEASE; then
                FEAT+=(--release)
            fi
            list="$REPORT_DIR/$pkg-tests.txt"
            rc=0
            run_logged "$list" cargo test -p "$pkg" ${FEAT[@]+"${FEAT[@]}"} --tests -- --list || rc=$?
            if [ "$rc" -ne 0 ]; then
                status_py collect --package "$pkg" --exit-code "$rc" --features "$features" -o "$RESULTS" "$list" > /dev/null
                echo "${RED}Rust $pkg: FAIL (cargo test --list exited $rc; see $list)${NC}"
                fail "Rust $pkg: cargo test --list exited $rc"
                LIST_FAILED=true
                continue
            fi
            SELECT=(select --package "$pkg")
            if [ "$pkg" = p2a-validation ]; then
                SELECT+=(--all)
            elif [ -n "$CATEGORY" ]; then
                SELECT+=(--category "$CATEGORY")
            fi
            selection="$REPORT_DIR/$pkg-selected.txt"
            status_py "${SELECT[@]}" "$list" > "$selection"
            if [ "$STATUS_RC" -ne 0 ]; then
                echo "${RED}Rust $pkg: FAIL (test selection failed)${NC}"
                fail "Rust $pkg: test selection failed"
                continue
            fi
            n="$(grep -c . "$selection" || true)"
            if [ "$pkg" != p2a-validation ]; then
                SELECTED_VALIDATE=$((SELECTED_VALIDATE + n))
            fi
            if [ "$n" -eq 0 ]; then
                status_py collect --package "$pkg" --selected 0 --features "$features" -o "$RESULTS" > /dev/null
                if [ "$pkg" = p2a-validation ]; then
                    echo "${RED}Rust p2a-validation: FAIL (harness self-tests missing)${NC}"
                    fail "Rust p2a-validation: harness self-tests missing"
                else
                    echo "Rust $pkg: 0 test_validate tests"
                fi
                continue
            fi
            LOGS=()
            max_rc=0
            while IFS= read -r target; do
                NAMES=()
                while IFS=$'\t' read -r t name; do
                    [ "$t" = "$target" ] && NAMES+=("$name")
                done < "$selection"
                read -r -a TARGET_ARGS <<< "$target"
                log="$REPORT_DIR/$pkg-run-$(slug_of "$target").txt"
                rc=0
                if $VERBOSE; then
                    run_logged "$log" env P2A_VALIDATION_VERBOSE=1 cargo test -p "$pkg" ${FEAT[@]+"${FEAT[@]}"} \
                        "${TARGET_ARGS[@]}" -- --exact --nocapture "${NAMES[@]}" || rc=$?
                else
                    run_logged "$log" cargo test -p "$pkg" ${FEAT[@]+"${FEAT[@]}"} \
                        "${TARGET_ARGS[@]}" -- --exact "${NAMES[@]}" || rc=$?
                fi
                LOGS+=("$log")
                if [ "$rc" -gt "$max_rc" ]; then
                    max_rc=$rc
                fi
            done < <(cut -f1 "$selection" | sort -u)
            collect_line "collect: $pkg: " --package "$pkg" --exit-code "$max_rc" --selected "$n" \
                --features "$features" -o "$RESULTS" "${LOGS[@]}"
            line="$COLLECT_LINE"
            pkg_status="${line#collect: "$pkg": }"
            pkg_status="${pkg_status%% *}"
            detail="${line#*(}"
            detail="${detail%)}"
            if [ "$pkg_status" = pass ]; then
                echo "${GREEN}Rust $pkg: PASS ($detail)${NC}"
            else
                echo "${RED}Rust $pkg: FAIL ($detail)${NC}"
                grep -F -- "collect: $pkg: error: " "$COLLECT_OUT" | sed "s/^collect: $pkg: error: /  /" || true
                fail "Rust $pkg: $pkg_status ($detail)"
            fi
        done
        if ! $LIST_FAILED && [ "$SELECTED_VALIDATE" -eq 0 ]; then
            echo "${RED}Rust: FAIL (no test_validate tests selected)${NC}"
            fail "no test_validate tests selected"
        fi
    fi
    echo ""
fi

# ------------------------------------------------------------------------------------
# Registry lint (full runs only)
# ------------------------------------------------------------------------------------
if $RUN_RUST && [ -z "$CATEGORY" ]; then
    if ! $RUST_RAN; then
        echo "registry lint: skipped (cargo not found)"
    elif $LIST_FAILED; then
        echo "registry lint: skipped (a test list failed)"
    else
        lint_log="$REPORT_DIR/lint.txt"
        status_py lint --tests "$REPORT_DIR"/*-tests.txt > "$lint_log" 2>&1
        if [ "$STATUS_RC" -eq 0 ]; then
            echo "registry lint: $(tail -n 1 "$lint_log")"
        else
            echo "${RED}registry lint: FAIL (see $lint_log)${NC}"
            grep -E '^ERROR ' "$lint_log" | head -n 20 | sed 's/^/  /' || true
            if [ "$STATUS_RC" -eq 1 ]; then
                fail "registry lint: FAIL (see $lint_log)"
            fi
        fi
    fi
    echo ""
fi

# ------------------------------------------------------------------------------------
# R drift check
# ------------------------------------------------------------------------------------
if $RUN_R; then
    echo "${BLUE}R reference cases${NC}"
    r_log="$REPORT_DIR/r-check.txt"
    if ! command -v "$P2A_RSCRIPT" > /dev/null 2>&1; then
        echo "Rscript not found (P2A_RSCRIPT=$P2A_RSCRIPT)" > "$r_log"
        status_py collect --r-check "$r_log" -o "$RESULTS" > /dev/null
        echo "${YELLOW}R reference check: INCOMPLETE (Rscript not found; install R >= 4.3 and $R_PACKAGES)${NC}"
        INCOMPLETE=true
    else
        rc=0
        run_logged "$r_log" "$P2A_RSCRIPT" validation/r/run.R --check || rc=$?
        collect_line "collect: R reference check: " --r-check "$r_log" -o "$RESULTS"
        line="$COLLECT_LINE"
        r_status="${line#collect: R reference check: }"
        r_status="${r_status%% *}"
        if [ "$r_status" = pass ] && [ "$rc" -eq 0 ]; then
            ok="${line#*(}"
            ok="${ok%% ok*}"
            echo "${GREEN}R reference check: PASS ($ok case(s))${NC}"
        else
            echo "${RED}R reference check: FAIL${NC}"
            if grep -q '^R reference check FAILED:' "$r_log"; then
                sed -n '/^R reference check FAILED:/,$p' "$r_log" | sed -n '2,41p'
            else
                tail -n 20 "$r_log" | sed 's/^/  /'
            fi
            fail "R reference check: ${r_status:-error} (see $r_log)"
            if grep -q 'there is no package called' "$r_log"; then
                echo "  Missing R packages; install R >= 4.3 and $R_PACKAGES"
            fi
        fi
    fi
    echo ""
fi

# ------------------------------------------------------------------------------------
# Legacy R scripts (reported only)
# ------------------------------------------------------------------------------------
if $LEGACY; then
    echo "${BLUE}Legacy R scripts${NC}"
    if ! command -v timeout > /dev/null 2>&1; then
        echo "legacy R scripts not run: timeout(1) not found"
    elif ! command -v "$P2A_RSCRIPT" > /dev/null 2>&1; then
        echo "legacy R scripts not run: Rscript not found (P2A_RSCRIPT=$P2A_RSCRIPT)"
    else
        mkdir -p "$REPORT_DIR/legacy/produced"
        ENTRIES=()
        for script in validation/scripts/*.R; do
            [ -e "$script" ] || continue
            name="$(basename "$script")"
            [ "$name" = run_all_validations.R ] && continue
            scratch="$(mktemp -d)"
            SCRATCH_DIRS+=("$scratch")
            mkdir -p "$scratch/validation/expected"
            cp -R validation/datasets "$scratch/validation/datasets"
            start=$SECONDS
            rc=0
            (cd "$scratch" && SKIP_BENCHMARKS=1 timeout --kill-after=10 "$P2A_LEGACY_R_TIMEOUT" \
                "$P2A_RSCRIPT" "$PROJECT_ROOT/validation/scripts/$name") \
                > "$REPORT_DIR/legacy/${name%.R}.txt" 2>&1 || rc=$?
            seconds=$((SECONDS - start))
            case "$rc" in
                0) st=ran ;;
                124 | 137) st=timeout ;;
                *) st=error ;;
            esac
            for f in "$scratch"/validation/expected/*.csv; do
                if [ -e "$f" ]; then
                    cp "$f" "$REPORT_DIR/legacy/produced/"
                fi
            done
            rm -rf "$scratch"
            ENTRIES+=("{\"script\": \"$name\", \"status\": \"$st\", \"exit_code\": $rc, \"seconds\": $seconds}")
            echo "  $name: $st (${seconds} s)"
        done
        {
            printf '{"timeout_seconds": %s, "produced_dir": "legacy/produced", "scripts": [' "$P2A_LEGACY_R_TIMEOUT"
            sep=""
            for e in ${ENTRIES[@]+"${ENTRIES[@]}"}; do
                printf '%s\n %s' "$sep" "$e"
                sep=","
            done
            printf '\n]}\n'
        } > "$REPORT_DIR/legacy.json"
        collect_line "collect: legacy R scripts: " --legacy "$REPORT_DIR/legacy.json" -o "$RESULTS"
        echo "Legacy R scripts (reported, never counted as passed): ${COLLECT_LINE#collect: legacy R scripts: }"
    fi
    echo ""
fi

# ------------------------------------------------------------------------------------
# Verdict and report
# ------------------------------------------------------------------------------------
RENDER=(render --results "$RESULTS" -o "$REPORT_DIR/report.md")
if $WRITE_STATUS; then
    RENDER+=(-o validation/VALIDATION_STATUS.md)
fi
status_py "${RENDER[@]}"
if [ "$STATUS_RC" -ne 0 ]; then
    fail "the report could not be rendered"
fi

if [ ${#FAILURES[@]} -gt 0 ]; then
    OVERALL=FAIL
elif $INCOMPLETE; then
    OVERALL=INCOMPLETE
else
    OVERALL=PASS
fi
for f in ${FAILURES[@]+"${FAILURES[@]}"}; do
    echo "  - $f"
done
case "$OVERALL" in
    PASS) echo "${GREEN}Overall Status: PASS${NC}" ;;
    INCOMPLETE) echo "${YELLOW}Overall Status: INCOMPLETE${NC}" ;;
    *) echo "${RED}Overall Status: FAIL${NC}" ;;
esac
echo "Results: $PROJECT_ROOT/$RESULTS"
case "$OVERALL" in
    PASS) exit 0 ;;
    INCOMPLETE) exit 3 ;;
    *) exit 1 ;;
esac
