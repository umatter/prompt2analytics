#!/bin/bash
# Tests of validation/run_validation.sh with a fake cargo and a fake Rscript
# (fixtures/bin). Each scenario runs a copy of the runner in a fresh temporary tree and
# checks its exit code, its output, results.json and the calls cargo received.
#
# Run: bash validation/tools/tests/test_run_validation.sh
# Prints "ok <scenario>" or "FAIL <scenario>: <reason>" per scenario; exits 1 if any failed.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
FIX="$HERE/fixtures"
PY="${PYTHON:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

total=0
failed=0
TREE=""
OUT=""
RC=0

# A fresh tree with copies of the runner, the status tool and the fixture data.
new_tree() {
    TREE="$WORK/tree$total"
    mkdir -p "$TREE/validation/tools" "$TREE/validation/r" "$TREE/validation/scripts" \
        "$TREE/validation/expected" "$TREE/validation/datasets"
    cp "$REPO/validation/run_validation.sh" "$TREE/validation/"
    cp "$REPO/validation/tools/validation_status.py" "$REPO/validation/tools/lint_ignore_names.txt" \
        "$TREE/validation/tools/"
    : > "$TREE/validation/r/run.R"
    cp "$FIX/runner/scripts/legacy_a.R" "$TREE/validation/scripts/legacy_a.R"
    cp "$FIX/runner/expected/legacy_a.csv" "$TREE/validation/expected/legacy_a.csv"
    cp "$FIX/tree/validation/datasets/d.csv" "$TREE/validation/datasets/d.csv"
}

# runner <list dir: two|none> [VAR=value ...] -- [runner arguments ...]
runner() {
    local list="$1"
    shift
    local envs=()
    while [ $# -gt 0 ] && [ "$1" != "--" ]; do
        envs+=("$1")
        shift
    done
    [ $# -gt 0 ] && shift
    OUT="$(env PATH="$FIX/bin:$PATH" FAKE_CARGO_CALLS="$TREE/calls.txt" \
        FAKE_CARGO_LIST_DIR="$FIX/runner/$list" ${envs[@]+"${envs[@]}"} \
        bash "$TREE/validation/run_validation.sh" "$@" 2>&1 < /dev/null)"
    RC=$?
}

# The results.json the runner reports on its "Results:" line.
results_file() {
    printf '%s\n' "$OUT" | sed -n 's/^Results: //p' | tail -n 1
}

# json_get <python expression on r> -> prints its value
json_get() {
    local f
    f="$(results_file)"
    [ -n "$f" ] && [ -f "$f" ] || { echo "<no results.json>"; return; }
    "$PY" -c 'import json, sys; r = json.load(open(sys.argv[1])); print(eval(sys.argv[2]))' "$f" "$1" 2>&1
}

calls() {
    cat "$TREE/calls.txt" 2>/dev/null
}

REASONS=()
expect_rc() { [ "$RC" -eq "$1" ] || REASONS+=("exit $RC, expected $1"); }
expect_out() { printf '%s\n' "$OUT" | grep -qF -- "$1" || REASONS+=("output lacks '$1'"); }
expect_true() { eval "$1" || REASONS+=("$2"); }

finish() {
    total=$((total + 1))
    if [ ${#REASONS[@]} -eq 0 ]; then
        echo "ok $1"
    else
        failed=$((failed + 1))
        local IFS=';'
        echo "FAIL $1: ${REASONS[*]}"
        if [ -n "${VERBOSE_TEST:-}" ]; then printf '%s\n' "$OUT" | sed 's/^/    | /'; fi
    fi
    REASONS=()
}

# 1. A compile error is FAIL (the old runner printed PASS and exited 0).
new_tree
runner two FAKE_CARGO_MODE=compile_error -- --rust-only
expect_rc 1
expect_out "Overall Status: FAIL"
expect_true '[ "$(json_get "r[\"packages\"][\"p2a-core\"][\"status\"]")" = compile_error ]' \
    "results.json: p2a-core status is not compile_error"
finish compile-error

# 2. --category stats runs exactly the stats test, by module path, with --exact.
new_tree
runner two -- --rust-only --category stats
expect_rc 0
core_runs="$(calls | grep -- '-p p2a-core' | grep -v -- '--list')"
expect_true '[ "$(printf "%s\n" "$core_runs" | grep -c .)" -eq 1 ]' "not exactly one p2a-core run: $core_runs"
expect_true 'printf "%s\n" "$core_runs" | grep -qF -- "--no-default-features --features ci --lib -- --exact stats::ttest::tests::test_validate_one"' \
    "the p2a-core run does not select stats::ttest::tests::test_validate_one by --exact"
expect_true '! printf "%s\n" "$core_runs" | grep -q test_validate_two' "the stats run also ran test_validate_two"
expect_true '! calls | grep -qE -- "-p p2a-(cli|mcp)"' "p2a-cli or p2a-mcp was called"
expect_true '[ "$(calls | grep -- "-p p2a-validation" | grep -vc -- "--list")" -eq 1 ]' "not exactly one p2a-validation run"
finish category-stats

# 3. A full Rust run uses the CI feature sets and runs the registry lint.
new_tree
runner two -- --rust-only
expect_rc 0
expect_true 'calls | grep -v -- "--list" | grep -q "test_validate_one"' "test_validate_one did not run"
expect_true 'calls | grep -v -- "--list" | grep -q "test_validate_two"' "test_validate_two did not run"
expect_true 'calls | grep -- "-p p2a-cli" | grep -- "--list" | grep -qF -- "--no-default-features --features spectral-analysis"' \
    "no p2a-cli list call with --features spectral-analysis"
expect_true 'calls | grep -- "-p p2a-mcp" | grep -- "--list" | grep -qF -- "--no-default-features --features http,llm,auth,websocket"' \
    "no p2a-mcp list call with --features http,llm,auth,websocket"
expect_out "registry lint:"
expect_out "Overall Status: PASS"
finish full-rust

# 4. No test_validate test selected is FAIL.
new_tree
runner none -- --rust-only
expect_rc 1
expect_out "no test_validate tests selected"
finish no-tests

# 5. A failing test is FAIL.
new_tree
runner two FAKE_CARGO_FAIL=regression::ols::tests::test_validate_two -- --rust-only
expect_rc 1
expect_out "Rust p2a-core: FAIL"
finish test-failed

# 6. A selected test that does not run is FAIL.
new_tree
runner two FAKE_CARGO_DROP=regression::ols::tests::test_validate_two -- --rust-only
expect_rc 1
expect_out "selected 2 tests, ran 1"
finish test-dropped

# 7. R drift check passes.
new_tree
runner two FAKE_R_MODE=pass -- --r-only
expect_rc 0
expect_out "R reference check: PASS (1 case(s))"
finish r-pass

# 8. R drift check fails.
new_tree
runner two FAKE_R_MODE=fail -- --r-only
expect_rc 1
expect_out "R reference check: FAIL"
expect_out "ols_grunfeld: adj_r2[1]"
finish r-fail

# 9. No Rscript is INCOMPLETE (exit 3).
new_tree
runner two P2A_RSCRIPT="$TREE/no/Rscript" -- --r-only
expect_rc 3
expect_out "Overall Status: INCOMPLETE"
expect_out "install.packages("
finish r-missing

# 10. Legacy scripts run in scratch directories: tracked CSVs are never written.
new_tree
before="$(cd "$TREE/validation/expected" && find . -type f -exec sha256sum {} + | sort)"
runner two FAKE_R_MODE=pass FAKE_LEGACY_MODE=write -- --r-only --legacy-r
after="$(cd "$TREE/validation/expected" && find . -type f -exec sha256sum {} + | sort)"
expect_rc 0
expect_true '[ "$before" = "$after" ]' "files under validation/expected changed"
expect_true '[ ! -e "$TREE/validation/expected/new_c.csv" ]' "new_c.csv was written into validation/expected"
expect_true '[ "$(json_get "r[\"legacy_r\"][\"files\"][\"legacy_a.csv\"]")" = differs ]' "legacy_a.csv is not 'differs'"
expect_true '[ "$(json_get "r[\"legacy_r\"][\"files\"][\"new_c.csv\"]")" = untracked ]' "new_c.csv is not 'untracked'"
expect_true '[ "$(json_get "r[\"legacy_r\"][\"scripts\"][\"legacy_a.R\"][\"status\"]")" = ran ]' "legacy_a.R is not 'ran'"
expect_out "never counted as passed"
finish legacy

# 11. A legacy script that times out is reported, and does not fail the run.
new_tree
runner two FAKE_R_MODE=pass FAKE_LEGACY_MODE=sleep P2A_LEGACY_R_TIMEOUT=1 -- --r-only --legacy-r
expect_rc 0
expect_true '[ "$(json_get "r[\"legacy_r\"][\"scripts\"][\"legacy_a.R\"][\"status\"]")" = timeout ]' "legacy_a.R is not 'timeout'"
finish legacy-timeout

# 12. Usage errors exit 2.
new_tree
for args in "--bogus" "--category nope" "--rust-only --r-only" "--write-status --category stats" \
    "--write-status --r-only" "--write-status --rust-only" "--category" "--legacy-r --rust-only"; do
    # shellcheck disable=SC2086
    runner two -- $args
    [ "$RC" -eq 2 ] || REASONS+=("'$args' exited $RC, expected 2")
done
for timeout_value in 0 00 -5 abc; do
    runner two FAKE_R_MODE=pass P2A_LEGACY_R_TIMEOUT="$timeout_value" -- --r-only --legacy-r
    [ "$RC" -eq 2 ] || REASONS+=("P2A_LEGACY_R_TIMEOUT='$timeout_value' with --legacy-r exited $RC, expected 2")
done
expect_out "P2A_LEGACY_R_TIMEOUT must be a whole number of seconds >= 1 (got 'abc')"
expect_true '[ ! -e "$TREE/validation/reports" ]' "a usage error created a report directory"
finish usage

# 12b. P2A_LEGACY_R_TIMEOUT is only read with --legacy-r: a bad value does not stop other runs.
new_tree
runner two P2A_LEGACY_R_TIMEOUT=abc -- --rust-only
expect_rc 0
expect_out "Overall Status: PASS"
finish legacy-timeout-unused

# 12c. A test that reads stdin cannot swallow the names of the targets still to run.
new_tree
runner multi FAKE_CARGO_READ_STDIN=1 -- --rust-only
expect_rc 0
expect_out "Rust p2a-core: PASS (selected 3, ran 3"
expect_true 'calls | grep -v -- "--list" | grep -qF -- "--test validation_ml_advanced -- --exact test_validate_c50_classification"' \
    "the validation_ml_advanced target did not run"
finish stdin-isolated

# 12d. Colour forced in the caller's environment (CARGO_TERM_COLOR=always): the runner
# turns it off for cargo, so its logs carry no escape codes, and the selection still works.
new_tree
runner two CARGO_TERM_COLOR=always -- --rust-only
expect_rc 0
expect_out "Rust p2a-core: PASS (selected 2, ran 2"
expect_out "registry lint: lint: 0 error(s)"
expect_out "Overall Status: PASS"
report_dir="$(dirname "$(results_file)")"
ESC=$'\x1b'
expect_true '[ -f "$report_dir/p2a-core-tests.txt" ]' "no p2a-core-tests.txt in the report directory"
expect_true '! grep -qF -- "$ESC" "$report_dir"/*.txt' "a cargo log in the report directory has escape codes"
finish colour-forced

# 13. --write-status renders validation/VALIDATION_STATUS.md.
new_tree
runner two FAKE_R_MODE=pass -- --write-status
expect_rc 0
expect_true 'head -n 1 "$TREE/validation/VALIDATION_STATUS.md" 2>/dev/null | grep -q "^<!-- Generated by validation/tools/validation_status.py render"' \
    "validation/VALIDATION_STATUS.md lacks the generated header"
finish write-status

echo "test_run_validation.sh: $failed of $total scenarios failed"
[ "$failed" -eq 0 ]
