"""Validation docs checked against the registry, the generated status page and cargo's test lists
(T1-WP06: INF-15, REG-57, ECON-82). Stdlib only; run with
python3 -m unittest discover -s validation/tools/tests"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATION = REPO / "validation"
TOOL = VALIDATION / "tools" / "validation_status.py"


def registry_entries():
    out = []
    for path in sorted((VALIDATION / "methods").glob("*.json")):
        out.extend(json.loads(path.read_text(encoding="utf-8"))["methods"])
    return out


def section(text, fragment):
    """Body of the first ##/### heading containing `fragment`, up to the next heading of the same or higher level."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#{2,3}) (.*)$", line)
        if m and fragment in m.group(2):
            body = []
            for nxt in lines[i + 1:]:
                h = re.match(r"^(#{1,6}) ", nxt)
                if h and len(h.group(1)) <= len(m.group(1)):
                    break
                body.append(nxt)
            return "\n".join(body)
    raise AssertionError(f"no section heading containing {fragment!r}")


LINK = re.compile(r"\]\(([^)]+\.md)\)")


class ReadmeIndexTest(unittest.TestCase):
    """The Method Index status column is derived from validation/methods/*.json (INF-15, REG-57)."""

    def setUp(self):
        self.index = section((VALIDATION / "README.md").read_text(encoding="utf-8"), "Method Index")
        self.by_doc = {}
        for entry in registry_entries():
            self.by_doc.setdefault(os.path.normpath(entry["doc"]), []).append(entry)

    def expected(self, doc):
        entries = self.by_doc.get(doc)
        if not entries:
            return "not R-pinned"
        ids = sorted({d["finding"] for e in entries for d in e.get("deviations", [])})
        return "R-pinned" if not ids else "R-pinned, known deviations (" + ", ".join(ids) + ")"

    def rows(self):
        for line in self.index.splitlines():
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 3 or cells[0] == "Method" or set(cells[0]) <= set("-: "):
                continue
            yield cells

    def test_status_cells_match_registry(self):
        problems, rows = [], 0
        for method, file_cell, status in self.rows():
            rows += 1
            link = LINK.search(file_cell)
            if link:
                doc = os.path.normpath(os.path.join("validation", link.group(1)))
                if not (REPO / doc).exists():
                    problems.append(f"{method}: {doc} does not exist")
                want = self.expected(doc)
            else:
                if file_cell != "no validation document yet":
                    problems.append(f"{method}: file cell {file_cell!r}")
                want = "not R-pinned"
            if status != want:
                problems.append(f"{method}: status {status!r}, registry says {want!r}")
        self.assertGreater(rows, 50)
        self.assertEqual(problems, [])

    def test_every_registered_doc_is_indexed(self):
        indexed = {os.path.normpath(os.path.join("validation", m.group(1))) for m in LINK.finditer(self.index)}
        self.assertEqual(sorted(set(self.by_doc) - indexed), [])


class Econ82DocsTest(unittest.TestCase):
    """ECON-82: parity docs report R's numbers, cite existing tests and claim no approximate p2a values."""

    def read(self, rel):
        return (VALIDATION / rel).read_text(encoding="utf-8")

    def test_hausman_doc_shows_phtest_statistic(self):
        self.assertIn("2.330366893675463", self.read("econometrics/hausman.md"))

    def test_no_approximate_parity_cells(self):
        for rel in ("econometrics/gmm.md", "econometrics/pggls.md", "econometrics/feglm.md"):
            with self.subTest(doc=rel):
                self.assertEqual(re.findall(r"\|\s*~\s*-?[0-9]", self.read(rel)), [])

    def test_docs_report_the_r_values(self):
        for rel, value in (("econometrics/gmm.md", "0.6719522974788301"),
                           ("econometrics/pggls.md", "0.0009197911431547233"),
                           ("econometrics/pggls.md", "-13.74653219816151"),
                           ("econometrics/feglm.md", "2.93615099246126")):
            with self.subTest(doc=rel):
                self.assertIn(value, self.read(rel))


PAGE = VALIDATION / "VALIDATION_STATUS.md"
HAND_WRITTEN = ("Fully Validated", "R cross-validated", "DGP-validated", "All Pass", "All pass")
# Methods REG-57 named that must not be listed as pinned unless a registry entry names them.
REG57_NAMES = ("loess", "nonlinear least squares", "smooth spline", "sensemakr", "e-value",
               "supsmu", "super smoother", "tukey", "resistant line")
_RUNNING = re.compile(r"^\s*Running (?:unittests )?(\S+)")


def listed_validate_counts():
    """{package: number of test_validate tests} from `cargo test ... -- --list` outputs
    (files named <package>-tests.txt), excluding the data::database read-only SQL guards.
    A test is counted once per test id, as the runner and the page count it: unit tests of
    the lib and the bins share the package's ids, an integration test is <binary>::<name>."""
    pattern = os.environ.get("P2A_TEST_LISTS", str(REPO / "target" / "*-tests.txt"))
    counts = {}
    for path in sorted(glob.glob(pattern)):
        package = Path(path).name[: -len("-tests.txt")]
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        text = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
        ids, prefix = set(), None
        for line in text.splitlines():
            m = _RUNNING.match(line)
            if m:
                target = m.group(1).replace("\\", "/")
                t = re.search(r"(?:^|/)tests/([^/]+?)(?:\.rs|/main\.rs)$", target)
                prefix = f"{t.group(1)}::" if t else ""
                continue
            if line.lstrip().startswith("Doc-tests "):
                prefix = None
                continue
            m = re.match(r"^(\S+): test$", line)
            if m and prefix is not None:
                name = m.group(1)
                last = name.rsplit("::", 1)[-1]
                if "test_validate" in last and not ("data::database::" in name and "read_only" in last):
                    ids.add(prefix + name)
        if ids:
            counts[package] = len(ids)
    return counts


def summary_counts(page):
    """{package: test_validate total} as the rendered summary shows it. The table
    validation_status.py render emits has one row per (package, category) with the cells
    Package | Category | Passed | Failed | Ignored; a package's total is the sum of the
    integer cells of its rows."""
    counts = {}
    for line in section(page, "Summary").splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) > 1 and re.fullmatch(r"p2a-[a-z]+", cells[0]):
            ints = [int(c.replace(",", "")) for c in cells[1:] if re.fullmatch(r"[0-9][0-9,]*", c)]
            if ints:
                counts[cells[0]] = counts.get(cells[0], 0) + sum(ints)
    return counts


class StatusPageTest(unittest.TestCase):
    def setUp(self):
        self.page = PAGE.read_text(encoding="utf-8")

    def test_page_is_generated(self):
        self.assertIn("validation_status.py render", "\n".join(self.page.splitlines()[:8]))

    def test_no_hand_written_status_claims(self):
        for phrase in HAND_WRITTEN:
            with self.subTest(phrase=phrase):
                self.assertNotIn(phrase, self.page)

    def test_pinned_section_lists_only_registered_methods(self):  # REG-57
        pinned = section(self.page, "Methods pinned to R").lower()
        entries = registry_entries()
        for e in entries:
            with self.subTest(method=e["id"]):
                self.assertTrue(e["id"].lower() in pinned or e["name"].lower() in pinned)
        registered = " ".join(f"{e['id']} {e['name']} {e.get('reference', '')}".lower() for e in entries)
        for name in REG57_NAMES:
            if name not in registered:
                with self.subTest(unregistered=name):
                    self.assertNotIn(name, pinned)

    @unittest.skipUnless(os.environ.get("P2A_CHECK_STATUS_PAGE") == "1",
                         "set P2A_CHECK_STATUS_PAGE=1 with cargo --list outputs in target/*-tests.txt")
    def test_summary_counts_match_cargo_list(self):  # INF-15
        listed = listed_validate_counts()
        self.assertTrue(listed, "no <package>-tests.txt list files found")
        self.assertEqual(summary_counts(self.page), listed)


FIXTURES = Path(__file__).resolve().parent / "fixtures"
UNREGISTERED_LOESS = "regression::loess::tests::test_validate_loess_against_r"


def build_render_fixture(tmp):
    """WP01's render fixture (fixtures/tree, fixtures/results/results.json) copied to
    tmp/root, with one more passing test_validate test that no registry entry names."""
    import shutil

    root = tmp / "root"
    shutil.copytree(FIXTURES / "tree", root)
    results = json.loads((FIXTURES / "results" / "results.json").read_text(encoding="utf-8"))
    package = next(t.split("::", 1)[0] for t in sorted(results["tests"]) if not t.startswith("p2a-validation::"))
    results["tests"][f"{package}::{UNREGISTERED_LOESS}"] = "ok"
    (root / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return root


def render_command(root, out):
    return [sys.executable, str(TOOL), "render", "--results", str(root / "results.json"), "-o", str(out),
            "--root", str(root)]


def render_env(root):
    return dict(os.environ)


class RenderFixtureTest(unittest.TestCase):
    """REG-57: render puts a test_validate test that no registry entry names outside 'Methods pinned to R'."""

    def test_unregistered_validate_tests_are_listed_outside_the_pinned_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = build_render_fixture(Path(tmp))
            out = Path(tmp) / "STATUS.md"
            subprocess.run(render_command(root, out), check=True, env=render_env(root), cwd=root)
            page = out.read_text(encoding="utf-8")
            self.assertNotIn("regression::loess", section(page, "Methods pinned to R"))
            # render puts these tests under "## Other `test_validate_*` tests", whose first
            # line says they are not registered as R-pinned.
            other = section(page, "Other `test_validate_*` tests")
            self.assertIn("not registered as r-pinned", other.lower())
            self.assertIn("regression::loess", other)


if __name__ == "__main__":
    unittest.main()
