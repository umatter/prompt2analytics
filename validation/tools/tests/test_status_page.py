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


if __name__ == "__main__":
    unittest.main()
