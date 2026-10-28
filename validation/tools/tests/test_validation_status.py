"""Unit tests of validation/tools/validation_status.py (select, collect, lint, render).

Run: python3 -m unittest discover -s validation/tools/tests -v
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
SCRIPT = HERE.parent / "validation_status.py"

_spec = importlib.util.spec_from_file_location("validation_status", SCRIPT)
vs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vs)


def run(*args):
    """Run the script; (exit code, stdout + stderr)."""
    p = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    return p.returncode, p.stdout + p.stderr


# The styles cargo 1.99 writes with CARGO_TERM_COLOR=always (or --color always): a status
# word right-aligned to 12 columns in bold bright green, and errors in bold bright red,
# rustc's with the message in bold.
GREEN, RED, BOLD, RESET = "\x1b[1m\x1b[92m", "\x1b[1m\x1b[91m", "\x1b[1m", "\x1b[0m"
_STATUS_LINE_RE = re.compile(r"^( *(?:Compiling|Finished|Running|Doc-tests))( .*)$")

# Lines captured from `cargo test -p p2a-validation --color always` and rustc
# --color=always (the build paths shortened); cargo_colour must reproduce them.
REAL_COLOURED_LINES = [
    "\x1b[1m\x1b[92m    Finished\x1b[0m `test` profile [optimized] target(s) in 0.29s",
    "\x1b[1m\x1b[92m     Running\x1b[0m unittests src/lib.rs "
    "(target/debug/deps/p2a_validation-7cafe02d9be3df34)",
    "\x1b[1m\x1b[92m     Running\x1b[0m tests/cases.rs (target/debug/deps/cases-5745df7f86c3d631)",
    "\x1b[1m\x1b[92m   Doc-tests\x1b[0m p2a_validation",
    "\x1b[1m\x1b[91merror[E0425]\x1b[0m\x1b[1m: cannot find value `x` in this scope\x1b[0m",
]


def cargo_colour(text: str) -> str:
    """`text` (plain cargo output) with the escape codes cargo and rustc write when colour
    is forced: status lines, rustc's `error[E...]` lines and cargo's `error:` lines."""
    out = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\n")
        end = line[len(body):]
        m = _STATUS_LINE_RE.match(body)
        e = re.match(r"^(error\[E\d+\])(: .*)$", body)
        c = re.match(r"^error:( .*)$", body)
        if m:
            body = f"{GREEN}{m.group(1)}{RESET}{m.group(2)}"
        elif e:
            body = f"{RED}{e.group(1)}{RESET}{BOLD}{e.group(2)}{RESET}"
        elif c:
            body = f"{RED}error{RESET}{BOLD}:{RESET}{c.group(1)}"
        out.append(body + end)
    return "".join(out)


class TreeCase(unittest.TestCase):
    """A fresh copy of fixtures/tree per test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tree"
        shutil.copytree(FIX / "tree", self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, rel):
        return self.root / rel

    def edit_json(self, rel, fn):
        p = self.path(rel)
        doc = json.loads(p.read_text())
        fn(doc)
        p.write_text(json.dumps(doc, indent=1))

    def method(self, doc, mid):
        return next(m for m in doc["methods"] if m["id"] == mid)

    def lint(self, *extra, tests=True):
        args = ["lint", "--root", self.root]
        if tests:
            args += ["--tests", self.path("lists/demo-tests.txt")]
        return run(*args, *extra)


class TestCollect(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name) / "results.json"

    def tearDown(self):
        self._tmp.cleanup()

    def collect(self, *args):
        code, text = run("collect", "--root", FIX / "tree", "-o", self.out, *args)
        results = json.loads(self.out.read_text()) if self.out.exists() else None
        return code, text, results

    def test_lib_and_integration_ids(self):
        parsed = vs.parse_cargo_log((FIX / "cargo/core_run_ok.txt").read_text(), "p2a-core")
        tests = parsed["tests"]
        self.assertEqual(tests["p2a-core::stats::ttest::tests::test_validate_one"], "ok")
        self.assertEqual(
            tests["p2a-core::validation_ml_advanced::test_validate_c50_classification"], "ok"
        )
        self.assertEqual(
            tests[
                "p2a-core::ml::advanced_clustering_mod::advanced::tests::test_hdbscan_full_benchmark"
            ],
            "ignored",
        )
        self.assertFalse([t for t in tests if "lib.rs" in t or "line 10" in t], tests)
        self.assertEqual(len(tests), 4)
        self.assertEqual(parsed["binaries"], ["validation_ml_advanced"])
        totals = parsed["totals"]
        self.assertEqual((totals["passed"], totals["failed"], totals["ignored"]), (3, 0, 1))

        code, text, results = self.collect(
            "--package", "p2a-core", "--selected", "3", FIX / "cargo/core_run_ok.txt"
        )
        self.assertEqual(code, 0, text)
        self.assertIn(
            "collect: p2a-core: pass (selected 3, ran 3: 3 passed, 0 failed, 1 ignored)", text
        )
        self.assertEqual(results["schema"], 1)
        self.assertEqual(results["binaries"]["p2a-core"], ["validation_ml_advanced"])
        self.assertRegex(results["date"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")

    def test_failed_run_is_fail(self):
        code, text, results = self.collect(
            "--package", "p2a-core", "--exit-code", "101", "--selected", "2",
            FIX / "cargo/core_run_failed.txt",
        )
        self.assertEqual(code, 1, text)
        pkg = results["packages"]["p2a-core"]
        self.assertEqual(pkg["status"], "fail")
        self.assertEqual((pkg["passed"], pkg["failed"], pkg["ran"]), (1, 1, 2))
        self.assertEqual(results["tests"]["p2a-core::regression::ols::tests::test_validate_two"], "failed")

    def test_compile_error_is_compile_error(self):
        code, text, results = self.collect(
            "--package", "p2a-core", "--exit-code", "101", FIX / "cargo/compile_error.txt"
        )
        self.assertEqual(code, 1, text)
        pkg = results["packages"]["p2a-core"]
        self.assertEqual(pkg["status"], "compile_error")
        self.assertTrue(any(e.startswith("error[E0425]") for e in pkg["errors"]), pkg["errors"])

    def test_compile_error_without_exit_code_is_compile_error(self):
        # A log with compile errors is never a pass, even when no exit code is given.
        code, text, results = self.collect("--package", "p2a-core", FIX / "cargo/compile_error.txt")
        self.assertEqual(code, 1, text)
        pkg = results["packages"]["p2a-core"]
        self.assertEqual(pkg["status"], "compile_error")
        self.assertIsNone(pkg["exit_code"])
        self.assertTrue(any(e.startswith("error[E0425]") for e in pkg["errors"]), pkg["errors"])
        self.assertIn("collect: p2a-core: compile_error (ran 0", text)

    def test_coloured_compile_error_log_is_compile_error(self):
        # CARGO_TERM_COLOR=always (CI sets it for the whole workflow) colours rustc's
        # error[E...] and cargo's "error: could not compile" lines.
        log = Path(self._tmp.name) / "compile_error.txt"
        log.write_text(cargo_colour((FIX / "cargo/compile_error.txt").read_text()))
        self.assertIn(f"{RED}error[E0425]{RESET}", log.read_text())
        code, text, results = self.collect("--package", "p2a-core", log)
        self.assertEqual(code, 1, text)
        pkg = results["packages"]["p2a-core"]
        self.assertEqual(pkg["status"], "compile_error")
        self.assertEqual(
            pkg["errors"],
            [
                "error[E0425]: cannot find value `x` in this scope",
                "error: could not compile `p2a-core` (lib test) due to 1 previous error",
            ],
        )

    def test_could_not_compile_in_test_output_is_not_a_compile_error(self):
        log = Path(self._tmp.name) / "run.txt"
        log.write_text(
            "     Running unittests src/lib.rs (target/debug/deps/p2a_core-0)\n"
            "test stats::tests::test_validate_x ... ok\n"
            "note: the message said it could not compile the expression\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        )
        code, text, results = self.collect("--package", "p2a-core", "--selected", "1", log)
        self.assertEqual(code, 0, text)
        self.assertEqual(results["packages"]["p2a-core"]["status"], "pass")

    def test_selected_but_not_run_is_fail(self):
        # Ignored tests did not run: a selected test that is #[ignore]d is not validated.
        code, text, results = self.collect(
            "--package", "p2a-core", "--selected", "4", FIX / "cargo/core_run_ok.txt"
        )
        self.assertEqual(code, 1, text)
        pkg = results["packages"]["p2a-core"]
        self.assertEqual(pkg["status"], "fail")
        self.assertIn("selected 4 tests, ran 3", pkg["errors"])

    def selection(self, *lines):
        path = Path(self._tmp.name) / "selected.txt"
        path.write_text("".join(line + "\n" for line in lines))
        return path

    CLI_SELECTION = (
        "--test exhibits\ttest_validate_cli_hausman_sim_corr",
        "--test exhibits\ttest_validate_cli_ols",
        "--test exhibits\ttest_validate_cli_re",
    )

    def test_selected_test_without_parsed_result_is_fail(self):
        # A real p2a-cli log run with --nocapture: a Check line split the status line of
        # test_validate_cli_hausman_sim_corr, so it has no parsed result, although
        # libtest's totals count all three tests.
        log = FIX / "cargo/cli_run_nocapture_interleaved.txt"
        parsed = vs.parse_cargo_log(log.read_text(), "p2a-cli")
        self.assertNotIn("p2a-cli::exhibits::test_validate_cli_hausman_sim_corr", parsed["tests"])
        self.assertEqual(parsed["totals"]["passed"], 3)

        sel = self.selection(*self.CLI_SELECTION)
        code, text, results = self.collect(
            "--package", "p2a-cli", "--exit-code", "0", "--selected", "3", "--selection", sel, log
        )
        self.assertEqual(code, 1, text)
        missing = "p2a-cli::exhibits::test_validate_cli_hausman_sim_corr"
        self.assertIn("collect: p2a-cli: fail (selected 3, ran 3: 3 passed, 0 failed, 0 ignored)", text)
        self.assertIn(f"collect: p2a-cli: missing: {missing}\n", text)
        self.assertIn("collect: p2a-cli: error: 1 selected test(s) without a parsed result", text)
        pkg = results["packages"]["p2a-cli"]
        self.assertEqual(pkg["status"], "fail")
        self.assertEqual(pkg["missing"], [missing])
        self.assertEqual(pkg["unexpected"], [])
        self.assertNotIn(missing, results["tests"])

    def test_selection_fully_parsed_is_pass(self):
        log = Path(self._tmp.name) / "cli.txt"
        log.write_text(
            "     Running tests/exhibits.rs (target/debug/deps/exhibits-0)\n"
            "test test_validate_cli_hausman_sim_corr ... ok\n"
            "test test_validate_cli_re ... ok\n"
            "test test_validate_cli_ols ... ok\n"
            "test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        )
        sel = self.selection(*self.CLI_SELECTION)
        code, text, results = self.collect("--package", "p2a-cli", "--selection", sel, log)
        self.assertEqual(code, 0, text)
        self.assertIn("collect: p2a-cli: pass (selected 3, ran 3", text)
        pkg = results["packages"]["p2a-cli"]
        self.assertEqual((pkg["selected"], pkg["missing"], pkg["unexpected"]), (3, [], []))

    def test_lib_selection_ids_and_unexpected_result(self):
        # --lib and --bins names take the package prefix; a parsed result for a test that
        # was not selected is a failure too (it would add a row to the page).
        sel = self.selection(
            "--lib\tstats::ttest::tests::test_validate_one",
            "--lib\tstats::ttest::tests::test_validate_three",
            "--test validation_ml_advanced\ttest_validate_c50_classification",
        )
        code, text, results = self.collect(
            "--package", "p2a-core", "--selection", sel, FIX / "cargo/core_run_ok.txt"
        )
        self.assertEqual(code, 1, text)
        pkg = results["packages"]["p2a-core"]
        self.assertEqual(pkg["status"], "fail")
        self.assertEqual(pkg["missing"], ["p2a-core::stats::ttest::tests::test_validate_three"])
        self.assertEqual(
            pkg["unexpected"],
            [
                "p2a-core::ml::advanced_clustering_mod::advanced::tests::test_hdbscan_full_benchmark",
                "p2a-core::regression::ols::tests::test_validate_two",
            ],
        )
        self.assertIn(
            "collect: p2a-core: unexpected: p2a-core::regression::ols::tests::test_validate_two\n",
            text,
        )
        self.assertIn("collect: p2a-core: error: 2 parsed result(s) for tests not selected", text)

    def test_selection_usage_errors(self):
        sel = self.selection(*self.CLI_SELECTION)
        code, text, _ = self.collect("--package", "p2a-cli", "--selected", "2", "--selection", sel)
        self.assertEqual(code, 2, text)
        self.assertIn("--selected 2 does not match the 3 line(s) of", text)
        bad = self.selection("test_validate_cli_ols")
        code, text, _ = self.collect("--package", "p2a-cli", "--selection", bad)
        self.assertEqual(code, 2, text)
        self.assertIn("not a `<target>\\t<libtest name>` line", text)
        code, text, _ = self.collect("--selection", sel)
        self.assertEqual(code, 2, text)
        self.assertIn("--selection needs --package", text)

    def test_selected_zero_is_no_tests(self):
        code, text, results = self.collect("--package", "p2a-cli", "--selected", "0")
        self.assertEqual(code, 0, text)
        self.assertEqual(results["packages"]["p2a-cli"]["status"], "no_tests")

    def test_should_panic_and_ignored_with_reason(self):
        text = (
            "     Running tests/cases.rs (target/debug/deps/cases-0)\n"
            "test check_panics_when_nothing_compared - should panic ... ok\n"
            "test check_named_rejects_extra_name - should panic ... FAILED\n"
            "test slow_case ... ignored, needs the database feature\n"
            "test result: FAILED. 1 passed; 1 failed; 1 ignored; 0 measured; 0 filtered out\n"
        )
        tests = vs.parse_cargo_log(text, "p2a-validation")["tests"]
        self.assertEqual(
            tests,
            {
                "p2a-validation::cases::check_panics_when_nothing_compared": "ok",
                "p2a-validation::cases::check_named_rejects_extra_name": "failed",
                "p2a-validation::cases::slow_case": "ignored",
            },
        )

    def test_duplicate_ids_keep_worst_status(self):
        parsed = vs.parse_cargo_log((FIX / "cargo/mcp_lib_main.txt").read_text(), "p2a-mcp")
        self.assertEqual(
            parsed["tests"],
            {"p2a-mcp::tools::handlers::regression::tests::test_validate_mcp_regression_ols": "failed"},
        )

    def test_merge_into_existing_and_replace_package(self):
        self.collect("--package", "p2a-core", FIX / "cargo/core_run_ok.txt")
        log = Path(self._tmp.name) / "cli.txt"
        log.write_text(
            "     Running tests/exhibits.rs (target/debug/deps/exhibits-0)\n"
            "test test_validate_cli_ols ... ok\n"
            "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n"
        )
        self.collect("--package", "p2a-cli", "--features", "--features spectral-analysis", log)
        code, text, results = self.collect("--package", "p2a-core", FIX / "cargo/core_run_failed.txt")
        self.assertEqual(code, 1, text)
        tests = results["tests"]
        self.assertEqual(tests["p2a-cli::exhibits::test_validate_cli_ols"], "ok")
        self.assertNotIn("p2a-core::validation_ml_advanced::test_validate_c50_classification", tests)
        self.assertEqual(tests["p2a-core::regression::ols::tests::test_validate_two"], "failed")
        self.assertEqual(set(results["packages"]), {"p2a-core", "p2a-cli"})
        self.assertEqual(results["features"], {"p2a-cli": "--features spectral-analysis"})
        self.assertEqual(results["binaries"]["p2a-core"], [])

    def test_r_check_pass_fail_error_missing(self):
        ok = vs.parse_r_check((FIX / "r/check_ok.txt").read_text())
        self.assertEqual(ok["status"], "pass")
        self.assertEqual(ok["cases"], {"harness_selftest": "ok"})
        fail = vs.parse_r_check((FIX / "r/check_fail.txt").read_text())
        self.assertEqual(fail["status"], "fail")
        self.assertEqual(fail["cases"]["ols_grunfeld"], "failed")
        self.assertEqual(fail["cases"]["harness_selftest"], "ok")
        self.assertIn(
            "committed case orphan_case has no definition in validation/r/cases/", fail["failures"]
        )
        err = vs.parse_r_check((FIX / "r/check_error.txt").read_text(encoding="utf-8"))
        self.assertEqual(err["status"], "error")
        self.assertEqual(err["missing_packages"], ["jsonlite"])
        missing = vs.parse_r_check((FIX / "r/check_missing.txt").read_text())
        self.assertEqual(missing["status"], "missing")

        code, text, results = self.collect("--r-check", FIX / "r/check_fail.txt")
        self.assertEqual(code, 1, text)
        self.assertEqual(results["r_check"]["status"], "fail")
        code, text, _ = self.collect("--r-check", FIX / "r/check_missing.txt")
        self.assertEqual(code, 0, text)

    def test_legacy_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tree"
            shutil.copytree(FIX / "tree", root)
            shutil.copy(root / "validation/expected/legacy_b.csv", root / "validation/expected/legacy_z.csv")
            legacy = vs.collect_legacy(FIX / "legacy/legacy.json", root)
        files = legacy["files"]
        self.assertEqual(files["legacy_a.csv"], "same")
        self.assertEqual(files["legacy_b.csv"], "differs")
        self.assertRegex(legacy["details"]["legacy_b.csv"], r"^row 1, column statistic: ")
        self.assertEqual(files["new_c.csv"], "untracked")
        self.assertEqual(files["legacy_z.csv"], "not_produced")
        self.assertEqual(legacy["scripts"]["validate_b.R"]["status"], "timeout")
        self.assertEqual(legacy["scripts"]["validate_a.R"]["status"], "ran")
        self.assertEqual(legacy["timeout_seconds"], 300)

    def test_compare_csv_tolerance(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            a.write_text("k,v\nx,1.0\ny,NA\n")
            b.write_text("k,v\nx,1.0000001\ny,NA\n")
            self.assertEqual(vs.compare_csv(a, b), ("differs", "row 1, column v: 1.0 vs 1.0000001"))
            b.write_text("k,v\nx,1.000000001\ny,NA\n")
            self.assertEqual(vs.compare_csv(a, b), ("same", ""))
            b.write_text("k,w\nx,1.0\ny,NA\n")
            self.assertEqual(vs.compare_csv(a, b)[0], "differs")

    @unittest.skipUnless(shutil.which("git"), "git not installed")
    def test_commit_is_marked_dirty_when_tracked_files_changed(self):
        # Isolated from the caller's git: without a ceiling, git looks for a repository
        # above the temporary directory (TMPDIR inside a work tree), and a git hook
        # exports GIT_INDEX_FILE (and may export GIT_DIR) to the suite it runs.
        isolated = {"GIT_CEILING_DIRECTORIES": os.path.realpath(self._tmp.name)}
        with mock.patch.dict(os.environ, isolated):
            for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
                        "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR"):
                os.environ.pop(var, None)
            self.check_dirty_marking()

    def check_dirty_marking(self):
        repo = Path(self._tmp.name) / "repo"
        repo.mkdir()

        def git(*args):
            subprocess.run(
                ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid",
                 "-c", "commit.gpgsign=false", *args],
                check=True, capture_output=True,
            )

        self.assertEqual(vs.git_commit(repo), "unknown")
        git("init", "-q")
        self.assertEqual(vs.git_commit(repo), "unknown")  # no commit yet
        (repo / "a.txt").write_text("one\n")
        git("add", "a.txt")
        git("commit", "-q", "--no-verify", "-m", "one")
        clean = vs.git_commit(repo)
        self.assertRegex(clean, r"^[0-9a-f]{7,}$")
        (repo / "untracked.txt").write_text("x\n")
        self.assertEqual(vs.git_commit(repo), clean)  # untracked files do not count
        (repo / "a.txt").write_text("two\n")
        self.assertEqual(vs.git_commit(repo), clean + "-dirty")
        git("add", "a.txt")
        self.assertEqual(vs.git_commit(repo), clean + "-dirty")  # staged, not committed

    def test_malformed_results_file_is_usage_error(self):
        self.out.write_text("{not json")
        code, text = run("collect", "-o", self.out, "--r-check", FIX / "r/check_ok.txt")
        self.assertEqual(code, 2, text)


class TestSelect(unittest.TestCase):
    def select(self, *args, fixture="cargo/core_list.txt", package="p2a-core"):
        code, text = run("select", "--package", package, *args, FIX / fixture)
        return code, [line for line in text.splitlines() if line]

    def test_category_stats(self):
        code, lines = self.select("--category", "stats")
        self.assertEqual(code, 0)
        self.assertEqual(lines, ["--lib\tstats::ttest::tests::test_validate_one"])

    def test_category_ml_includes_integration_binaries(self):
        code, lines = self.select("--category", "ml")
        self.assertEqual(lines, ["--test validation_ml_advanced\ttest_validate_c50_classification"])

    def test_category_econometrics_includes_diagnostics(self):
        code, lines = self.select("--category", "econometrics")
        self.assertEqual(
            lines,
            [
                "--lib\tdiagnostics::iv::tests::test_validate_weak_iv_warning",
                "--lib\teconometrics::did::tests::test_validate_did_2x2",
            ],
        )

    def test_full_selection_excludes_database_guards_and_non_validation_tests(self):
        code, lines = self.select()
        self.assertEqual(code, 0)
        self.assertEqual(len(lines), 5, lines)
        joined = "\n".join(lines)
        for absent in ("read_only", "test_other", "test_negbin_vs_r", "bench_validate"):
            self.assertNotIn(absent, joined)

    def test_all_and_targets(self):
        code, lines = self.select("--all", fixture="cargo/cli_list.txt", package="p2a-cli")
        self.assertEqual(
            lines,
            [
                "--bins\toutput::shell_quote_tests::joins_arguments",
                "--test exhibits\ttest_validate_cli_ols",
                "--test tier2\twp21_stats_viz::test_validate_cli_t_test",
            ],
        )
        ids = vs.parse_test_list((FIX / "cargo/cli_list.txt").read_text(), "p2a-cli")
        self.assertIn("p2a-cli::exhibits::test_validate_cli_ols", ids)
        self.assertIn("p2a-cli::tier2::wp21_stats_viz::test_validate_cli_t_test", ids)
        self.assertEqual(ids["p2a-cli::tier2::wp21_stats_viz::test_validate_cli_t_test"],
                         ("--test tier2", "wp21_stats_viz::test_validate_cli_t_test"))
        code, lines = self.select("--category", "cli", fixture="cargo/cli_list.txt", package="p2a-cli")
        self.assertEqual(len(lines), 2, lines)

    def test_unknown_category_is_usage_error(self):
        code, _ = self.select("--category", "nope")
        self.assertEqual(code, 2)

    def test_coloured_list_selects_like_plain(self):
        with tempfile.TemporaryDirectory() as tmp:
            for fixture, package in (("cargo/core_list.txt", "p2a-core"), ("cargo/cli_list.txt", "p2a-cli")):
                coloured = Path(tmp) / Path(fixture).name
                coloured.write_text(cargo_colour((FIX / fixture).read_text()))
                for args in ([], ["--category", "ml"], ["--all"]):
                    with self.subTest(fixture=fixture, args=args):
                        code, plain = self.select(*args, fixture=fixture, package=package)
                        self.assertEqual(code, 0)
                        code, lines = self.select(*args, fixture=coloured, package=package)
                        self.assertEqual(code, 0, lines)
                        self.assertEqual(lines, plain)
                        if args == ["--all"]:
                            self.assertTrue(lines)

    def test_list_without_running_line_is_usage_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            texts = {
                "no output": "",
                "no target line": "stats::ttest::tests::test_validate_one: test\n\n1 test, 0 benchmarks\n",
                "a compile error": (FIX / "cargo/compile_error.txt").read_text(),
            }
            for label, text in texts.items():
                with self.subTest(label):
                    path = Path(tmp) / "p2a-core.list"
                    path.write_text(text)
                    code, lines = self.select(fixture=path)
                    self.assertEqual(code, 2, lines)
                    self.assertTrue(any(str(path) in ln and "Running" in ln for ln in lines), lines)

    def test_category_of(self):
        b = {"validation_ml_advanced", "validation_boosting_methods", "validation_new_methods"}
        self.assertEqual(vs.category_of("p2a-core::diagnostics::iv::tests::test_validate_x", b), "econometrics")
        self.assertEqual(vs.category_of("p2a-core::validation_boosting_methods::test_validate_x", b), "ml")
        self.assertEqual(vs.category_of("p2a-core::validation_new_methods::test_x", b), "tests/validation_new_methods.rs")
        self.assertEqual(vs.category_of("p2a-mcp::tools::x::test_validate_y", b), "mcp")
        self.assertEqual(vs.category_of("p2a-validation::cases::x", b), "harness")


class TestLint(TreeCase):
    def assert_finding(self, code, text, level, rule, *needles, exit_code=1):
        self.assertEqual(code, exit_code, text)
        lines = [ln for ln in text.splitlines() if ln.startswith(f"{level} [{rule}]")]
        self.assertTrue(lines, text)
        for n in needles:
            self.assertTrue(any(n in ln for ln in lines), f"{n!r} not in {lines}")

    def test_good_tree_passes(self):
        code, text = self.lint()
        self.assertEqual(code, 0, text)
        self.assertIn("lint: 0 error(s), 0 warning(s)", text)
        code, text = self.lint("--strict")
        self.assertEqual(code, 0, text)

    def test_registry_errors(self):
        mutations = {
            "missing tolerance": lambda d: self.method(d, "demo_mean").pop("tolerance"),
            "extra key": lambda d: self.method(d, "demo_mean").update(statuss="ok"),
            "duplicate id": lambda d: self.method(d, "demo_var").update(id="demo_mean"),
            "bad deviation": lambda d: self.method(d, "demo_mean")["deviations"][0].update(finding="reg50"),
            "missing legacy": lambda d: self.method(d, "demo_mean").update(
                supersedes_legacy=["validation/expected/gone.csv"]
            ),
            "schema 2": lambda d: d.update(schema=2),
            "unknown category": lambda d: self.method(d, "demo_mean").update(category="statistics"),
        }
        original = self.path("validation/methods/demo.json").read_text()
        for label, fn in mutations.items():
            with self.subTest(label):
                self.path("validation/methods/demo.json").write_text(original)
                self.edit_json("validation/methods/demo.json", fn)
                code, text = self.lint()
                self.assert_finding(code, text, "ERROR", "registry")
        with self.subTest("truncated JSON"):
            self.path("validation/methods/demo.json").write_text(original[:100])
            code, text = self.lint()
            self.assert_finding(code, text, "ERROR", "registry", "not valid JSON")

    def test_registry_categories(self):
        # The runner's categories, plus `interfaces` (validation/methods/interfaces.json).
        original = self.path("validation/methods/demo.json").read_text()
        for category in vs.CATEGORIES + ("interfaces",):
            with self.subTest(category):
                self.path("validation/methods/demo.json").write_text(original)
                self.edit_json(
                    "validation/methods/demo.json",
                    lambda d, c=category: [m.update(category=c) for m in d["methods"]],
                )
                code, text = self.lint()
                self.assertEqual(code, 0, text)
                self.assertIn("lint: 0 error(s), 0 warning(s)", text)
        self.assertNotIn("interfaces", vs.CATEGORIES)  # not a runner/select --category

    def test_case_file_errors(self):
        with self.subTest("registry case without JSON"):
            self.edit_json("validation/methods/demo.json", lambda d: self.method(d, "demo_var")["cases"].append("nope_case"))
            code, text = self.lint()
            self.assert_finding(code, text, "ERROR", "case-file", "nope_case")
        cases = {
            "missing dataset": lambda d: d.update(dataset="validation/datasets/missing.csv",
                                                  datasets=["validation/datasets/missing.csv"]),
            "schema 2": lambda d: d.update(schema=2),
            "id differs": lambda d: d.update(id="other_case"),
            "dataset not in datasets": lambda d: d.update(datasets=[]),
            "text value": lambda d: d["values"].update(mean="text"),
            "names and values of unequal length": lambda d: d["values"].update(
                pair={"names": ["a", "b"], "values": [1.0]}
            ),
        }
        original = self.path("validation/expected/demo_case.json").read_text()
        for label, fn in cases.items():
            with self.subTest(label):
                self.path("validation/expected/demo_case.json").write_text(original)
                self.edit_json("validation/expected/demo_case.json", fn)
                code, text = self.lint()
                self.assert_finding(code, text, "ERROR", "case-file", "demo_case.json")

    def test_orphan_case_and_harness_exemption(self):
        doc = json.loads(self.path("validation/expected/demo_var_case.json").read_text())
        doc["id"] = "extra_case"
        self.path("validation/expected/extra_case.json").write_text(json.dumps(doc))
        code, text = self.lint()
        self.assert_finding(code, text, "ERROR", "orphan-case", "extra_case")
        self.assertNotIn("selftest_case", "\n".join(ln for ln in text.splitlines() if "orphan-case" in ln))

    def test_unknown_test(self):
        self.edit_json(
            "validation/methods/demo.json",
            lambda d: self.method(d, "demo_var")["tests"].append("demo::stats::tests::test_validate_gone"),
        )
        code, text = self.lint()
        self.assert_finding(code, text, "ERROR", "unknown-test", "test_validate_gone")
        code, text = self.lint(tests=False)
        self.assertEqual(code, 0, text)
        self.assertIn("note: tests of demo not checked", text)

    def test_coloured_test_list(self):
        # CI forces colour (CARGO_TERM_COLOR=always): the `Running` lines of the --list
        # files then start with escape codes.
        lst = self.path("lists/demo-tests.txt")
        lst.write_text(cargo_colour(lst.read_text()))
        self.assertIn(f"{GREEN}     Running{RESET}", lst.read_text())
        code, text = self.lint()
        self.assertEqual(code, 0, text)
        self.assertIn("lint: 0 error(s), 0 warning(s)", text)
        # The coloured list is read, not skipped: a test it lacks is still unknown.
        self.edit_json(
            "validation/methods/demo.json",
            lambda d: self.method(d, "demo_var")["tests"].append("demo::stats::tests::test_validate_gone"),
        )
        code, text = self.lint()
        self.assert_finding(code, text, "ERROR", "unknown-test", "test_validate_gone")
        lines = [ln for ln in text.splitlines() if "unknown-test" in ln]
        self.assertEqual(len(lines), 1, text)

    def test_test_list_without_running_line_is_usage_error(self):
        lst = self.path("lists/demo-tests.txt")
        lst.write_text("".join(ln for ln in lst.read_text().splitlines(keepends=True) if "Running" not in ln))
        code, text = self.lint()
        self.assertEqual(code, 2, text)
        self.assertIn(f"test list {lst} has no cargo `Running <target>` line", text)

    def test_tests_file_name_must_end_in_tests_txt(self):
        shutil.copy(self.path("lists/demo-tests.txt"), self.path("lists/demo.txt"))
        code, text = run("lint", "--root", self.root, "--tests", self.path("lists/demo.txt"))
        self.assertEqual(code, 2, text)

    def test_unregistered_load(self):
        src = self.path("crates/demo/src/lib.rs")
        text = src.read_text().replace(
            "        #[test]\n        fn test_selftest()",
            "        #[test]\n"
            "        fn test_validate_extra() {\n"
            '            let _ = crate::test_support::load_case("demo_case");\n'
            "        }\n\n"
            "        fn helper() {\n"
            '            let _ = crate::test_support::load_case("demo_var_case");\n'
            "        }\n\n"
            "        #[test]\n        fn test_selftest()",
        )
        src.write_text(text)
        code, out = self.lint()
        self.assert_finding(code, out, "ERROR", "unregistered-load", "test_validate_extra", "demo_case")
        lines = [ln for ln in out.splitlines() if "unregistered-load" in ln]
        self.assertEqual(len(lines), 1, out)
        load_line = text.splitlines().index('            let _ = crate::test_support::load_case("demo_case");') + 1
        self.assertIn(f"crates/demo/src/lib.rs:{load_line}: ", lines[0])

    def test_registered_test_that_loads_no_case(self):
        src = self.path("crates/demo/src/lib.rs")
        original = src.read_text()
        variants = {
            # The registered test loads nothing at all.
            "no load": 'let c = 1;',
            # It loads a harness case, which is none of demo_var's cases.
            "harness case only": 'let c = p2a_validation::Case::load("selftest_case");',
            # The case id through a helper, but only in a comment.
            "case id in a comment only": 'let c = 1; // demo_case("demo_var_case")',
        }
        for label, body in variants.items():
            with self.subTest(label):
                src.write_text(
                    original.replace('let c = p2a_validation::Case::load("demo_var_case");', body)
                )
                code, out = self.lint()
                self.assert_finding(
                    code, out, "ERROR", "registered-test-loads-no-case",
                    "demo_var", "demo::stats::tests::test_validate_demo_var", "demo_var_case",
                )
                lines = [ln for ln in out.splitlines() if "registered-test-loads-no-case" in ln]
                self.assertEqual(len(lines), 1, out)
        with self.subTest("an async test that loads its case through a helper"):
            src.write_text(
                original.replace(
                    "        #[test]\n        fn test_validate_demo_var() {\n"
                    '            let c = p2a_validation::Case::load("demo_var_case");\n',
                    "        fn demo_case(id: &str) -> p2a_validation::Case {\n"
                    "            p2a_validation::Case::load(id)\n"
                    "        }\n\n"
                    "        #[tokio::test]\n        async fn test_validate_demo_var() {\n"
                    '            let c = demo_case("demo_var_case");\n',
                )
            )
            self.assertIn("async fn test_validate_demo_var", src.read_text())
            code, out = self.lint()
            self.assertEqual(code, 0, out)
            self.assertNotIn("registered-test-loads-no-case", out)
        with self.subTest("a registered test that is not in the sources is not checked here"):
            src.write_text(original)
            self.edit_json(
                "validation/methods/demo.json",
                lambda d: self.method(d, "demo_var")["tests"].append("demo::stats::tests::test_validate_gone"),
            )
            code, out = self.lint(tests=False)
            self.assertEqual(code, 0, out)

    def test_load_in_comment_or_string_is_not_a_load(self):
        src = self.path("crates/demo/src/lib.rs")
        text = src.read_text().replace(
            "        #[test]\n        fn test_selftest()",
            "        #[test]\n"
            "        fn test_validate_quiet() {\n"
            '            // let _ = load_case("demo_case");\n'
            "            /* Case::load(\"demo_case\") */\n"
            "            let lt: &'static str = \"x\";\n"
            "        }\n\n"
            "        #[test]\n        fn test_selftest()",
        )
        src.write_text(text)
        code, out = self.lint()
        self.assertEqual(code, 0, out)

    def test_missing_doc(self):
        self.edit_json("validation/methods/demo.json", lambda d: self.method(d, "demo_var").update(doc="validation/stats/missing.md"))
        code, text = self.lint()
        self.assert_finding(code, text, "ERROR", "missing-doc", "validation/stats/missing.md")

    def test_strict_csv_citation(self):
        src = self.path("crates/demo/src/lib.rs")
        original = src.read_text()
        src.write_text(
            original.replace(
                "        #[test]\n        fn test_selftest()",
                "        #[test]\n"
                "        fn test_validate_old() {\n"
                "            // R reference values from validation/expected/legacy_b.csv\n"
                "            assert!(true);\n"
                "        }\n\n"
                "        #[test]\n        fn test_selftest()",
            )
        )
        code, text = self.lint()
        self.assert_finding(code, text, "WARNING", "csv-citation", "test_validate_old", exit_code=0)
        code, text = self.lint("--strict")
        self.assertEqual(code, 1, text)
        self.assertIn("(--strict: warnings count as errors)", text)
        src.write_text(
            original.replace(
                '            let s = "}{";\n',
                '            let s = "}{";\n            // R reference values from validation/expected/legacy_b.csv\n',
            )
        )
        code, text = self.lint("--strict")
        self.assertEqual(code, 0, text)

    def test_strict_dead_link(self):
        doc = self.path("validation/stats/demo.md")
        doc.write_text(
            doc.read_text()
            + "\nSee [x](nope.md), [site](https://example.org), [top](#demo-mean) and [ok](demo.md#x).\n"
            + "\n```text\n[y](fenced-nope.md)\n```\n"
        )
        code, text = self.lint()
        self.assert_finding(code, text, "WARNING", "dead-link", "nope.md", exit_code=0)
        lines = [ln for ln in text.splitlines() if "dead-link" in ln]
        self.assertEqual(len(lines), 1, text)

    def test_strict_dead_test_name(self):
        readme = self.path("README.md")
        readme.write_text(
            readme.read_text()
            + "\nRemoved: `test_validate_gone`. Prefix: `test_validate_demo`. Ignored: `test_type`.\n"
            + "Path: `paper/e2e/test_cases.json`. Filter: `cargo test -- stats::tests::test_validate_ghost2`.\n"
            + "\n```rust\nlet test_data = 1;\nfn test_validate_ghost() {}\n```\n"
        )
        code, text = self.lint()
        self.assertEqual(code, 0, text)
        lines = [ln for ln in text.splitlines() if ln.startswith("WARNING [dead-test-name]")]
        names = sorted(ln.split("`")[1] for ln in lines)
        self.assertEqual(names, ["test_validate_ghost", "test_validate_ghost2", "test_validate_gone"], text)
        code, text = self.lint("--strict")
        self.assertEqual(code, 1, text)

    def test_strict_missing_dataset(self):
        p = self.path("validation/datasets/README.md")
        p.write_text(p.read_text() + "\n**File**: `ghost.csv`\n")
        code, text = self.lint()
        self.assert_finding(code, text, "WARNING", "missing-dataset", "ghost.csv", exit_code=0)

    def test_fixture_directories_are_not_scanned(self):
        fx = self.path("validation/tools/tests/fixtures")
        fx.mkdir(parents=True)
        (fx / "x.md").write_text("[a](nope.md) `test_validate_nothing`\n")
        reports = self.path("validation/reports/2026")
        reports.mkdir(parents=True)
        (reports / "report.md").write_text("[a](nope.md)\n")
        code, text = self.lint("--strict")
        self.assertEqual(code, 0, text)


class TestColouredOutput(unittest.TestCase):
    """cargo output with forced colour parses exactly like the plain fixtures."""

    def test_helper_reproduces_cargo_and_rustc_codes(self):
        for real in REAL_COLOURED_LINES:
            with self.subTest(real=real):
                plain = vs.strip_ansi(real)
                self.assertNotIn("\x1b", plain)
                self.assertEqual(cargo_colour(plain), real)

    def test_strip_ansi(self):
        self.assertEqual(
            vs.strip_ansi("a\x1b[38;5;9mb\x1b[0m \x1b(B\x1b[mc \x1b]8;;https://x\x1b\\d\x1b]8;;\x07"),
            "ab c d",
        )

    def test_lists(self):
        for fixture, package in (("cargo/core_list.txt", "p2a-core"), ("cargo/cli_list.txt", "p2a-cli")):
            with self.subTest(fixture):
                plain = (FIX / fixture).read_text()
                coloured = cargo_colour(plain)
                self.assertIn(f"{GREEN}     Running{RESET}", coloured)
                self.assertTrue(vs.parse_test_list(plain, package))
                self.assertEqual(vs.parse_test_list(coloured, package), vs.parse_test_list(plain, package))
                self.assertEqual(vs.list_binaries(coloured, package), vs.list_binaries(plain, package))

    def test_run_logs(self):
        for fixture, package in (
            ("cargo/core_run_ok.txt", "p2a-core"),
            ("cargo/core_run_failed.txt", "p2a-core"),
            ("cargo/compile_error.txt", "p2a-core"),
            ("cargo/mcp_lib_main.txt", "p2a-mcp"),
        ):
            with self.subTest(fixture):
                plain = (FIX / fixture).read_text()
                coloured = cargo_colour(plain)
                self.assertNotEqual(coloured, plain)
                self.assertEqual(vs.parse_cargo_log(coloured, package), vs.parse_cargo_log(plain, package))
        ok = vs.parse_cargo_log(cargo_colour((FIX / "cargo/core_run_ok.txt").read_text()), "p2a-core")
        self.assertEqual(len(ok["tests"]), 4)  # the doc test after `Doc-tests` is not collected
        self.assertEqual(ok["binaries"], ["validation_ml_advanced"])
        err = vs.parse_cargo_log(cargo_colour((FIX / "cargo/compile_error.txt").read_text()), "p2a-core")
        self.assertEqual(len(err["compile_errors"]), 2)

    def test_r_check(self):
        for fixture in ("r/check_ok.txt", "r/check_fail.txt"):
            with self.subTest(fixture):
                plain = (FIX / fixture).read_text()
                coloured = "".join(f"{BOLD}{ln}{RESET}\n" for ln in plain.splitlines())
                self.assertEqual(vs.parse_r_check(coloured), vs.parse_r_check(plain))


class TestScanner(unittest.TestCase):
    def test_braces_in_literals_and_comments(self):
        src = (
            "fn test_a() {\n"
            '    let s = "}"; let r = r#"}"{"#; let c = \'}\'; let e = \'\\\'\';\n'
            "    // }\n"
            "    /* } /* nested } */ } */\n"
            "    let lt: &'static str = s;\n"
            '    load_case("one");\n'
            "}\n"
            "fn test_b() { Case::try_load(\"two\"); }\n"
        )
        functions, keep, _ = vs.test_functions(src)
        self.assertEqual([f[0] for f in functions], ["test_a", "test_b"])
        name, line, start, end = functions[0]
        self.assertEqual(line, 1)
        self.assertIn('load_case("one")', keep[start:end])
        self.assertNotIn("two", keep[start:end])


class TestRender(TreeCase):
    def render(self, results=None):
        path = FIX / "results/results.json"
        if results is not None:
            path = Path(self._tmp.name) / "results.json"
            path.write_text(json.dumps(results))
        out = Path(self._tmp.name) / "out.md"
        code, text = run("render", "--results", path, "--root", self.root, "-o", out)
        self.assertEqual(code, 0, text)
        return out.read_text()

    def row(self, page, first_cell):
        return next(ln for ln in page.splitlines() if ln.startswith(f"| {first_cell} "))

    def test_render_fixture(self):
        page = self.render()
        self.assertTrue(page.startswith("<!-- Generated by validation/tools/validation_status.py render"))
        self.assertIn("Commit `abc1234`, 2026-10-21T12:00:00Z, command `validation/run_validation.sh`.", page)
        self.assertIn("| **Total** | | **3** | **1** | **0** |", page)
        self.assertIn("| demo | ml | 2 | 0 | 0 |", page)
        self.assertIn("| demo | stats | 1 | 1 | 0 |", page)
        self.assertIn("Harness self-tests (p2a-validation): 2 passed, 0 failed.", page)
        methods = page.split("## Methods pinned to R")[1].split("## R reference check")[0]
        mean = next(ln for ln in methods.splitlines() if "`demo_mean`" in ln)
        var = next(ln for ln in methods.splitlines() if "`demo_var`" in ln)
        self.assertTrue(mean.endswith("| PASS (known deviations) |"), mean)
        self.assertTrue(var.endswith("| NOT RUN |"), var)
        deviations = methods.split("### Known deviations")[1]
        self.assertIn("REG-50", deviations)
        r = page.split("## R reference check")[1].split("## Other")[0]
        self.assertIn("PASS", r)
        self.assertIn("R version 4.3.3", r)
        other = page.split("## Other `test_validate_*` tests")[1].split("## Legacy")[0]
        self.assertIn("`ml::kmeans`", other)
        self.assertIn("`tests/validation_ml_advanced.rs`", other)
        self.assertIn("`stats::tests`", other)
        self.assertNotIn("test_validate_demo_mean", other)
        legacy = page.split("## Legacy reference files")[1]
        self.assertIn("| `validation/expected/legacy_a.csv` | superseded by `demo_case` | same |", legacy)
        self.assertIn("| `validation/expected/legacy_b.csv` | not asserted by any test | differs", legacy)
        self.assertIn("1 ran, 0 error, 1 timeout", legacy)
        self.assertNotIn("Fully Validated", page)
        self.assertNotIn("](", page)  # no Markdown links
        self.assertEqual(page, self.render())

    def test_render_marks_failed_registry_test(self):
        results = json.loads((FIX / "results/results.json").read_text())
        results["tests"]["demo::stats::tests::test_validate_demo_mean"] = "failed"
        page = self.render(results)
        mean = next(ln for ln in page.splitlines() if "`demo_mean`" in ln and ln.startswith("|"))
        self.assertTrue(mean.endswith("| FAIL |"), mean)

        results = json.loads((FIX / "results/results.json").read_text())
        results["r_check"]["cases"]["demo_case"] = "failed"
        page = self.render(results)
        mean = next(ln for ln in page.splitlines() if "`demo_mean`" in ln and ln.startswith("|"))
        self.assertTrue(mean.endswith("| FAIL |"), mean)

    def test_render_without_registry_or_runs(self):
        shutil.rmtree(self.path("validation/methods"))
        results = {"schema": 1, "commit": "abc1234", "date": "2026-10-21T12:00:00Z", "tests": {}}
        page = self.render(results)
        self.assertIn("No method is registered yet (`validation/methods/*.json`).", page)
        self.assertIn("Legacy R scripts were not run for this report.", page)
        self.assertIn("| **Total** | | **0** | **0** | **0** |", page)
        self.assertIn("NOT RUN.", page)


if __name__ == "__main__":
    unittest.main()
