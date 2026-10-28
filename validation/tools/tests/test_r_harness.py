"""Tests of validation/r/harness.R: the drift comparator compare_case_files() that
`Rscript validation/r/run.R --check` uses, and the case writer run_case().

Skipped when Rscript or the R package jsonlite is missing (CI installs no R).
Run: python3 -m unittest discover -s validation/tools/tests -v
"""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parents[1]
HARNESS = VALIDATION / "r" / "harness.R"
STATUS = VALIDATION / "tools" / "validation_status.py"
FIX_TREE = HERE / "fixtures" / "tree"
RSCRIPT = shutil.which("Rscript")


def _r_ready() -> bool:
    if not RSCRIPT:
        return False
    probe = 'quit(status = if (requireNamespace("jsonlite", quietly = TRUE)) 0 else 1)'
    return subprocess.run([RSCRIPT, "-e", probe], capture_output=True, check=False).returncode == 0


R_READY = _r_ready()

# The values of validation/expected/harness_selftest.json, the shapes R writes.
BASE_VALUES = {
    "n": 200,
    "inv_mean": 145.95824999999999,
    "means": {"names": ["value", "capital"], "values": [1081.6811, 276.01715000000002]},
    "inv_max": {"names": ["inv"], "values": [1486.7]},
    "inv_first3": [317.60000000000002, 391.80000000000001, 410.60000000000002],
    "with_na": [29191.650000000001, None],
}

COMPARE_R = r"""
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
d <- compare_case_files(args[2], args[3])
writeLines(c(d, "END"))
"""

# Runs cases on the fixture tree's validation/datasets/d.csv. Every case is written to
# args[3] except demo_var_case, the fixture tree's registered case, which a base-R-only
# case (packages = character(0)) overwrites in that tree's validation/expected/.
RUN_CASE_R = r"""
args <- commandArgs(trailingOnly = TRUE)
source(args[1])
root <- args[2]; out <- args[3]; gen <- args[4]; tree_expected <- args[5]
returns <- function(packages) function(ctx) {
  d <- ctx$dataset("datasets/d.csv")
  res <- list(values = list(n = nrow(d)), reference = "base R nrow()", call = "nrow(d)")
  res$packages <- packages
  res
}
case("zz_no_packages", "harness", "no packages", returns(NULL))
case("zz_empty_package", "harness", "an empty package name", returns(""))
case("zz_na_package", "harness", "an NA package name", returns(NA_character_))
case("zz_with_packages", "harness", "packages given", returns("jsonlite"))
case("demo_var_case", "stats", "Demo variance, base R only", function(ctx) {
  d <- ctx$dataset("datasets/d.csv")
  list(values = list(var = var(d$y)), reference = "base R var()", call = "var(d$y)",
       packages = character(0))
})
for (id in names(.harness$cases)) {
  dir <- if (id == "demo_var_case") tree_expected else out
  msg <- tryCatch({ run_case(.harness$cases[[id]], dir, gen, root); "written" },
                  error = function(e) conditionMessage(e))
  cat(id, ": ", msg, "\n", sep = "")
}
"""


@unittest.skipUnless(R_READY, "Rscript with the R package jsonlite is not installed")
class TestCompareCaseFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.script = self.tmp / "compare.R"
        self.script.write_text(COMPARE_R)

    def tearDown(self):
        self._tmp.cleanup()

    def write_case(self, name, values):
        doc = {
            "schema": 1,
            "id": "harness_selftest",
            "area": "harness",
            "method": "m",
            "reference": "r",
            "call": "c",
            "dataset": "validation/datasets/grunfeld.csv",
            "datasets": ["validation/datasets/grunfeld.csv"],
            "r_version": "R version 4.3.3 (2024-02-29)",
            "packages": {"jsonlite": "2.0.0"},
            "values": values,
        }
        path = self.tmp / f"{name}.json"
        path.write_text(json.dumps(doc, indent=1))
        return path

    def compare(self, committed: dict, recomputed: dict) -> list:
        """compare_case_files(committed, recomputed): the differences it reports."""
        a = self.write_case("committed", {**BASE_VALUES, **committed})
        b = self.write_case("recomputed", {**BASE_VALUES, **recomputed})
        p = subprocess.run(
            [RSCRIPT, str(self.script), str(HARNESS), str(a), str(b)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        lines = p.stdout.splitlines()
        self.assertEqual(lines[-1:], ["END"], p.stdout + p.stderr)
        return lines[:-1]

    def assert_differs(self, committed, recomputed, *needles):
        diffs = self.compare(committed, recomputed)
        self.assertTrue(diffs, f"no difference reported for {committed} vs {recomputed}")
        for needle in needles:
            self.assertTrue(any(needle in d for d in diffs), f"{needle!r} not in {diffs}")

    def test_identical_values_and_drift_within_tolerance_pass(self):
        self.assertEqual(self.compare({}, {}), [])
        self.assertEqual(self.compare({}, {"inv_mean": 145.95825000001}), [])  # 7e-14 relative

    def test_value_drift_is_reported(self):
        self.assert_differs({}, {"inv_mean": 145.9583}, "inv_mean[1]: committed 145.95824999999999")

    def test_na_moved_inside_an_unnamed_vector(self):
        self.assert_differs(
            {"with_na": [1.5, None]}, {"with_na": [None, 1.5]}, "with_na[1,2]: committed"
        )

    def test_na_dropped_from_an_unnamed_vector(self):
        self.assert_differs({"with_na": [1.5, None]}, {"with_na": [1.5]}, "with_na: length 2 vs 1")

    def test_na_scalar(self):
        self.assert_differs({"n": None}, {"n": 200}, "n[1]")
        self.assertEqual(self.compare({"n": None}, {"n": None}), [])

    def test_renamed_element_of_a_named_vector(self):
        self.assert_differs(
            {},
            {"means": {"names": ["value", "capital_stock"], "values": [1081.6811, 276.01715000000002]}},
            "means: names differ: value,capital vs value,capital_stock",
        )

    def test_names_swapped_with_their_values(self):
        swapped = {"names": ["capital", "value"], "values": [276.01715000000002, 1081.6811]}
        self.assert_differs({}, {"means": swapped}, "means: names differ", "means[1,2]")

    def test_kind_change_of_the_same_length(self):
        self.assert_differs({}, {"inv_max": [1486.7]}, "inv_max: kind differs: named vs array")
        self.assert_differs({}, {"n": [200]}, "n: kind differs: scalar vs array")
        self.assert_differs(
            {}, {"n": {"names": ["n"], "values": [200]}}, "n: kind differs: scalar vs named"
        )


@unittest.skipUnless(R_READY, "Rscript with the R package jsonlite is not installed")
class TestRunCase(unittest.TestCase):
    def test_packages_must_be_a_vector_of_package_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tree = tmp / "tree"
            shutil.copytree(FIX_TREE, tree)
            script = tmp / "run_case.R"
            script.write_text(RUN_CASE_R)
            out, gen = tmp / "expected", tmp / "generated"
            out.mkdir()
            gen.mkdir()
            tree_expected = tree / "validation" / "expected"
            p = subprocess.run(
                [RSCRIPT, str(script), str(HARNESS), str(tree / "validation"), str(out), str(gen),
                 str(tree_expected)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
            results = dict(line.split(": ", 1) for line in p.stdout.splitlines() if ": " in line)
            for bad in ("zz_no_packages", "zz_empty_package", "zz_na_package"):
                self.assertTrue(
                    results.get(bad, "").startswith(
                        f"case {bad}: packages must be a character vector of R package names"
                    ),
                    p.stdout,
                )
                self.assertFalse((out / f"{bad}.json").exists(), f"{bad}.json was written")
            self.assertEqual(results.get("zz_with_packages"), "written", p.stdout)
            doc = json.loads((out / "zz_with_packages.json").read_text())
            self.assertIsInstance(doc["packages"], dict)
            self.assertIn("jsonlite", doc["packages"])

            # A base-R-only case (packages = character(0)) records no package: an empty map.
            self.assertEqual(results.get("demo_var_case"), "written", p.stdout)
            text = (tree_expected / "demo_var_case.json").read_text()
            self.assertIn('"packages": {}', text)
            self.assertEqual(json.loads(text)["packages"], {})
            # The registered fixture case, now written by R, passes the registry lint.
            lint = subprocess.run(
                [sys.executable, str(STATUS), "lint", "--root", str(tree),
                 "--tests", str(tree / "lists" / "demo-tests.txt")],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(lint.returncode, 0, lint.stdout + lint.stderr)
            self.assertIn("lint: 0 error(s), 0 warning(s)", lint.stdout)


if __name__ == "__main__":
    unittest.main()
