"""Tier 3 closure checks for the validation claims (T3-WP64: X-10, X-11, X-13, ECON-80, ECON-81, CAUS-98,
CAUS-99). Stdlib only; run with python3 -m unittest discover -s validation/tools/tests.

Checks that need cargo's test lists (<repo>/target/<package>-tests.txt, or the glob in P2A_TEST_LISTS) run only
with P2A_CHECK_STATUS_PAGE=1, as CI runs them (T1-WP06). Everything else reads the tree only."""
import csv
import glob
import json
import os
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
VALIDATION = REPO / "validation"
CRATES = REPO / "crates"
LABELS = VALIDATION / "tools" / "validate_property_tests.txt"
WITH_LISTS = os.environ.get("P2A_CHECK_STATUS_PAGE") == "1"
NEEDS_LISTS = "set P2A_CHECK_STATUS_PAGE=1 with cargo --list outputs in target/*-tests.txt"
SEVEN = ("X-10", "X-11", "X-13", "ECON-80", "ECON-81", "CAUS-98", "CAUS-99")


# ---------------------------------------------------------------- Task 1: shared helpers

def registry_entries():
    out = []
    for path in sorted((VALIDATION / "methods").glob("*.json")):
        for entry in json.loads(path.read_text(encoding="utf-8"))["methods"]:
            out.append(dict(entry, _file=path.name))
    return out


def expected_status(doc, entries=None):
    """The README / method-doc status vocabulary of T1-WP06 for a repository-relative doc path."""
    entries = registry_entries() if entries is None else entries
    mine = [e for e in entries if os.path.normpath(e["doc"]) == os.path.normpath(doc)]
    if not mine:
        return "not R-pinned"
    ids = sorted({d["finding"] for e in mine for d in e.get("deviations", [])})
    return "R-pinned" if not ids else "R-pinned, known deviations (" + ", ".join(ids) + ")"


def md_files():
    for path in sorted(VALIDATION.rglob("*.md")):
        rel = path.relative_to(VALIDATION).as_posix()
        if rel.startswith("reports/") or rel.startswith("tools/tests/fixtures/"):
            continue
        yield path


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


def table_rows(text):
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and not set(cells[0]) <= set("-: "):
                yield cells


_RUST_FILES = None


def rust_files():
    """(repository-relative path, Path) of every .rs file under crates/*/src and crates/*/tests (cached)."""
    global _RUST_FILES
    if _RUST_FILES is None:
        _RUST_FILES = list(_walk_rust())
    return _RUST_FILES


def _walk_rust():
    for crate in sorted(CRATES.iterdir()):
        for sub in ("src", "tests"):
            base = crate / sub
            if base.is_dir():
                for dirpath, _, files in os.walk(base):
                    for f in sorted(files):
                        if f.endswith(".rs"):
                            p = Path(dirpath) / f
                            yield p.relative_to(REPO).as_posix(), p


def read_labels():
    """{test id: (kind, reason)} from validate_property_tests.txt."""
    line_re = re.compile(r"^(?P<id>p2a-[a-z]+::\S+)\s*\|\s*(?P<kind>formula|exact|property|registered-property)\s*\|\s*(?P<reason>\S.*)$")
    out = {}
    for n, raw in enumerate(LABELS.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = line_re.match(line)
        if not m:
            raise AssertionError(f"{LABELS.name}:{n}: malformed line {raw!r}")
        if m["id"] in out:
            raise AssertionError(f"{LABELS.name}:{n}: duplicate {m['id']}")
        out[m["id"]] = (m["kind"], m["reason"])
    return out


def load_test_lists():
    """{test id: (target, raw libtest name)} from the cargo --list outputs, parsed by validation_status.py."""
    sys.path.insert(0, str(VALIDATION / "tools"))
    import validation_status as vs  # T1-WP01
    ids = {}
    pattern = os.environ.get("P2A_TEST_LISTS", str(REPO / "target" / "*-tests.txt"))
    for path in sorted(glob.glob(pattern)):
        package = Path(path).name[: -len("-tests.txt")]
        ids.update(vs.parse_test_list(Path(path).read_text(encoding="utf-8", errors="replace"), package))
    return ids, vs


# ---------------------------------------------------------------- Task 1: Rust source helpers (the X-10 rules are used from Task 8)

def blank(src):
    """Comments and string/char literals replaced by spaces; offsets and newlines kept."""
    out, i, n = list(src), 0, len(src)

    def wipe(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            wipe(i, j)
            i = j
        elif src.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif src.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            wipe(i, j)
            i = j
        elif c in "rb" and re.match(r'(b?r#*")', src[i:i + 12]) and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            m = re.match(r'b?r(#*)"', src[i:])
            end = src.find('"' + m.group(1), i + len(m.group(0)))
            end = n if end < 0 else end + 1 + len(m.group(1))
            wipe(i + len(m.group(0)), end - 1 - len(m.group(1)))
            i = end
        elif c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            wipe(i + 1, j)
            i = j + 1
        elif c == "'":
            m = re.match(r"'(\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'", src[i:i + 12])
            if m:
                wipe(i + 1, i + len(m.group(0)) - 1)
                i += len(m.group(0))
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def match_brace(code, start):
    depth = 0
    for j in range(start, len(code)):
        if code[j] == "{":
            depth += 1
        elif code[j] == "}":
            depth -= 1
            if depth == 0:
                return j
    return len(code) - 1


ASSERTION = re.compile(r"\b(?:debug_)?assert(?:_eq|_ne)?!|\bassert_[a-z_]+!|\bpanic!|\bunreachable!|\bCheck::new\b|\.finish\(\)|\bassert_[a-z_]+\(")
# Assertions that compare something (a panic! inside a guard is a failure path, not a guarded comparison).
COMPARISON = re.compile(r"\b(?:debug_)?assert(?:_eq|_ne)?!|\bassert_[a-z_]+!|\bCheck::new\b|\.finish\(\)|\bassert_[a-z_]+\(")


_PARSED = {}


def rust_tests(src):
    """(name, line, attributes, body, body_offset) for every #[test] fn, on blanked code (cached by text)."""
    if src not in _PARSED:
        _PARSED[src] = list(_rust_tests(src))
    return _PARSED[src]


def _rust_tests(src):
    code = blank(src)
    for m in re.finditer(r"#\[test\]", code):
        fm = re.compile(r"\bfn\s+([A-Za-z0-9_]+)\s*\(").search(code, m.end())
        if not fm:
            continue
        back = max(code.rfind(";", 0, m.start()), code.rfind("}", 0, m.start()), code.rfind("{", 0, m.start()))
        attrs = code[back + 1:fm.start()]
        brace = code.find("{", fm.end())
        end = match_brace(code, brace)
        yield fm.group(1), code.count("\n", 0, fm.start()) + 1, attrs, code[brace:end + 1], brace


def is_conditional_block(body, brace):
    """True if the block opened at `brace` is an `if`/`if let`/`else` branch (match arms are left to the
    swallowed-err rule: an `Err(..) => panic!(..)` arm is a failure path, not a condition)."""
    k, depth = brace - 1, 0
    while k >= 0:
        ch = body[k]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and ch in ";{}":
            break
        k -= 1
    head = body[k + 1:brace].strip()
    if not (re.match(r"^(else\b|if\b)", head) or re.search(r"\belse(\s+if\b.*)?$", head)):
        return False
    # `if ... { assert } else { panic!(..) }` always fails or asserts: not a condition on the assertion
    return not (re.match(r"^if\b", head) and else_fails(body, brace - 1))


def else_fails(body, pos):
    """True if the `if let` block starting after `pos` has an `else` branch that panics or asserts."""
    b = body.find("{", pos)
    if b < 0:
        return False
    e = match_brace(body, b)
    tail = body[e + 1:].lstrip()
    if not tail.startswith("else"):
        return False
    eb = body.find("{", e + 1)
    return eb >= 0 and re.search(r"panic!|unreachable!|assert", body[eb:match_brace(body, eb) + 1]) is not None


def body_rules(name, attrs, body, scope):
    """Rule names violated by one test. `scope` adds the X-10 rules beyond print-only."""
    if re.search(r"#\[ignore\b", attrs):
        return []
    should_panic = "should_panic" in attrs
    asserts = [a.start() for a in ASSERTION.finditer(body)]
    out = []
    if re.search(r"\b(e?println|dbg)!", body) and not asserts and not should_panic:
        out.append("print-only")
    if not scope:
        return out
    if any(not else_fails(body, m.end()) for m in re.finditer(r"\b(?:if|while)\s+let\s+Ok\s*\(", body)):
        out.append("swallowed-err")
    else:
        # catch-all arms only: `Err(_)`, `Err(e)`; an arm naming an error variant is an expected-error path
        for em in re.finditer(r"\bErr\s*\(\s*(?:_|[a-z_][a-z0-9_]*)\s*\)\s*=>", body):
            rest = body[em.end():].lstrip()
            arm = rest[:match_brace(rest, 0) + 1] if rest.startswith("{") else re.split(r",\s*\n|\n\s*\}", rest, maxsplit=1)[0]
            if not re.search(r"panic!|unreachable!|assert|\.unwrap|\.expect|return\s+Err|\?", arm):
                out.append("swallowed-err")
                break
    if not asserts and not should_panic:
        if "print-only" not in out:
            out.append("no-assertion")
        return out
    stack, all_conditional = [], True
    pos = set(asserts)
    for j, ch in enumerate(body):
        if ch == "{":
            stack.append(j > 0 and is_conditional_block(body, j))
        elif ch == "}" and stack:
            stack.pop()
        if j in pos and not any(stack[1:]):
            all_conditional = False
            break
    if all_conditional and not should_panic:
        out.append("conditional-assert")
    guard = re.compile(r"\bif\s+(?:let\s+(?:Some|Ok)\s*\(|[^{;]*?(?:\.is_some\(\)|\.is_ok\(\)|\.len\(\)\s*(?:==|>=|>)|\.converged\b))")
    for gm in guard.finditer(body):
        b = body.find("{", gm.end())
        if b < 0:
            continue
        e = match_brace(body, b)
        if COMPARISON.search(body[b:e + 1]):
            tail = body[e + 1:e + 200].lstrip()
            if tail.startswith("else"):
                eb = body.find("{", e + 1)
                if eb >= 0 and re.search(r"panic!|unreachable!|assert", body[eb:match_brace(body, eb) + 1]):
                    continue
            out.append("guarded-assert")
            break
    return out


def scan_source(rel, src, extra_names=()):
    """[(line, test name, rule)] for one Rust file."""
    validation_file = re.search(r"^crates/[^/]+/tests/validation_[^/]*\.rs$", rel) is not None
    hits = []
    for name, line, attrs, body, _ in rust_tests(src):
        scope = "test_validate" in name or validation_file or name in extra_names
        for rule in dict.fromkeys(body_rules(name, attrs, body, scope)):
            hits.append((line, name, rule))
    return hits


# ---------------------------------------------------------------- Task 1: X-13 (quoted R values, tolerance rules)

WRONG_QUOTES = [
    ("W = 7, p-value", "wilcox.test gives W = 8, p = 0.2353332665095857"),
    ("V = 40, p-value = 0.1455", "wilcox.test gives V = 33, p = 0.2361369702216528"),
    ("widened for normal approx variation", "a window tuned to pass"),
    ("Wait, need to recalculate", "an unfinished reference"),
    ("p-value = 0.0476", "exact two-sample ks.test gives p = 0.07936507936507936"),
    ("coef ~0.3-0.6", "coxph (Efron and Breslow) gives -0.05023124445706806, se 0.7086724463055264"),
    ("~0.3-0.6 | ~0.6-0.9", "coxph (Efron and Breslow) gives -0.05023124445706806, se 0.7086724463055264"),
    # plan sync: the doc quote moved to validation/econometrics/survival_models.md:78 (T2-WP09, 0b699a4)
    ("about 0.3-0.6", "coxph (Efron and Breslow) gives -0.05023124445706806, se 0.7086724463055264"),
    ("Allow tolerance of 0.001 as specified in validation doc", "the assertion used 0.01"),
    ("Allow 10% tolerance due to approximation differences", "power.t.test gives 0.4778409859409388"),
    ("1% relative tolerance for highly collinear data", "Longley is pinned at Tol::coef"),
    ("r_intercept).abs() < 30.0", "a PVCM intercept window of 30"),
]


class QuotedValuesTest(unittest.TestCase):
    """X-13: no test or validation doc quotes the R output R does not produce, or the widened windows."""

    def test_wrong_r_values_are_not_quoted(self):
        hits = []
        files = [(rel, p) for rel, p in rust_files()] + [(p.relative_to(REPO).as_posix(), p) for p in md_files()]
        for rel, path in files:
            text = path.read_text(encoding="utf-8")
            for needle, why in WRONG_QUOTES:
                if needle in text:
                    hits.append(f"{rel}: {needle!r} ({why})")
        self.assertEqual(hits, [], "\n" + "\n".join(hits))


class ToleranceTextTest(unittest.TestCase):
    """X-13 (validation/README.md:72): the tolerance rules for iterative methods and quoted values are exact."""

    def setUp(self):
        # whitespace collapsed, so that a phrase may wrap across lines
        self.text = " ".join(section((VALIDATION / "README.md").read_text(encoding="utf-8"), "Tolerances").split())

    def test_no_licence_to_widen(self):
        self.assertNotIn("expect slightly larger differences", self.text)

    def test_iterative_and_quoted_value_rules(self):
        for phrase in ("pinned to a tight R refit",
                       "stated in the test comment and in the registry entry's `tolerance`",
                       "never the pin",
                       "A test comment never carries an R value that no case file holds",
                       "Path quantities are never R-pinned",
                       "optimality conditions",
                       "not the algorithm template",
                       "(property)"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, self.text)


# ---------------------------------------------------------------- Task 2: registry deviations of fixed findings

# (registry file, finding): deviations that Tier 3 made stale (T3-WP64 Task 2; confirmed against the
# Fixes: lines of the Tier 3 range before committing).
STALE_DEVIATIONS = [
    ("regression_diagnostics.json", "REG-35"),
    ("regression_diagnostics.json", "REG-52"),
    ("regression_diagnostics.json", "REG-26"),
    ("regression_diagnostics.json", "REG-64"),
    ("regression_diagnostics.json", "REG-9"),
    ("regression_diagnostics.json", "REG-43"),
    ("regression_na.json", "REG-26"),
    ("regression_na.json", "REG-31"),
    ("causal_rd.json", "CAUS-28"),
    ("causal_rd.json", "CAUS-77"),
    ("causal_rdbw.json", "CAUS-77"),
    ("cli_flags.json", "REG-52"),
    ("cli_flags.json", "TS-15"),
    ("regression_na.json", "REG-14"),
    ("causal_na.json", "CAUS-92"),
]


def sources_of_test(test_id):
    """Raw source text of the test function(s) a registry test id names (comments kept)."""
    package, rest = test_id.split("::", 1)
    segs = rest.split("::")
    name, mods = segs[-1], segs[:-1]
    crate = CRATES / package
    candidates = []
    for k in range(len(mods), 0, -1):
        base = "/".join(mods[:k])
        for root in ("src", "tests"):
            for p in (crate / root / f"{base}.rs", crate / root / base / "mod.rs", crate / root / base / "main.rs"):
                if p.is_file():
                    candidates.append(p)
        if candidates:
            break
    if not candidates:
        candidates = [crate / "src" / "lib.rs"]
    out = []
    for p in candidates:
        src = p.read_text(encoding="utf-8")
        for tname, _, _, body, off in rust_tests(src):
            if tname == name:
                out.append(src[off:off + len(body)])
    return out


class RegistryDeviationTest(unittest.TestCase):
    def test_tier3_fixed_deviations_are_removed(self):
        left = []
        for e in registry_entries():
            for d in e.get("deviations", []):
                if (e["_file"], d["finding"]) in STALE_DEVIATIONS:
                    left.append(f"{e['_file']} {e['id']}: {d['finding']} ({d['quantity']})")
        self.assertEqual(left, [], "\n" + "\n".join(left))

    def test_deviation_comments_have_registry_deviations(self):
        """T1 1.4 rule 3: a `// deviation <id>` comment in a registered test matches a registry deviation."""
        problems = []
        entries = registry_entries()
        for e in entries:
            allowed = {d["finding"] for x in entries for d in x.get("deviations", []) if set(x["tests"]) & set(e["tests"])}
            for test_id in e["tests"]:
                for raw in sources_of_test(test_id):
                    for line in raw.splitlines():
                        c = line.split("//", 1)[1] if "//" in line else ""
                        m = re.search(r"\bdeviations?\b(.*)$", c)
                        if not m:
                            continue
                        for fid in re.findall(r"\b[A-Z]{2,5}-\d+\b", m.group(1)):
                            if fid not in allowed:
                                problems.append(f"{test_id}: comment names deviation {fid}, registry has {sorted(allowed)}")
        self.assertEqual(sorted(set(problems)), [], "\n" + "\n".join(sorted(set(problems))))


# ---------------------------------------------------------------- Task 3: method docs WP64 owns (CAUS-98, CAUS-99; ECON-80 docs)

OWNED_DOCS = ["validation/econometrics/gsynth.md", "validation/econometrics/scpi.md",
              "validation/econometrics/rdmulti.md", "validation/econometrics/tmle.md",
              "validation/econometrics/bpbounds.md", "validation/econometrics/ivmte.md"]


class OwnedDocsTest(unittest.TestCase):
    def test_status_line_matches_registry(self):
        entries = registry_entries()
        for doc in OWNED_DOCS:
            with self.subTest(doc=doc):
                lines = [l for l in (REPO / doc).read_text(encoding="utf-8").splitlines() if l.startswith("**Status**: ")]
                self.assertEqual(len(lines), 1, "exactly one **Status**: line")
                want = "**Status**: " + expected_status(doc, entries)
                self.assertTrue(lines[0].startswith(want) and (lines[0] == want or lines[0][len(want)] in ".;( "),
                                f"{lines[0]!r} does not start with {want!r}")

    def test_no_approximate_or_missing_values(self):
        for doc in OWNED_DOCS:
            text = (REPO / doc).read_text(encoding="utf-8")
            with self.subTest(doc=doc):
                self.assertNotIn("TBD", text)
                self.assertIsNone(re.search(r"\|\s*~\s*[0-9]", text), "approximate parity cell")


def read_doc(rel):
    return (VALIDATION / rel).read_text(encoding="utf-8")


class Caus98Test(unittest.TestCase):
    """CAUS-98: gsynth, SCPI and rdmulti pages no longer claim parity they do not have."""

    def test_caus98_gsynth_scpi_rdmulti(self):
        g = read_doc("econometrics/gsynth.md")
        self.assertNotIn("Within 1.0 of R", g)
        self.assertNotIn("~4.0", g)
        s = read_doc("econometrics/scpi.md")
        self.assertNotIn("Intervals are constructed as (Cattaneo et al. 2021, Eq. 3.5)", s)
        self.assertIn("CAUS-32", s)
        r = read_doc("econometrics/rdmulti.md")
        self.assertNotIn("slightly different tuning parameters", r)
        self.assertNotIn("Test Results Summary", r)
        self.assertIn("rdmc", r)
        self.assertIn("not installed", r)


class Caus99Test(unittest.TestCase):
    """CAUS-99: no pending TMLE claim; CLAUDE.md lists only the DoubleML models that exist."""

    def test_caus99_tmle_doc_and_doubleml_line(self):
        self.assertNotIn("pending real data comparison", read_doc("econometrics/tmle.md"))
        claude = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
        line = next(l for l in claude.splitlines() if "doubleml.rs" in l)
        listed = re.findall(r"\b(PLR|PLIV|IRM|IIVM)\b", line)
        src = (CRATES / "p2a-core/src/econometrics/doubleml.rs").read_text(encoding="utf-8")
        enum = re.search(r"pub enum DMLModelType\s*\{(.*?)\}", blank(src), re.S).group(1)
        variants = set(re.findall(r"\b([A-Z][A-Za-z]*)\b", enum))
        self.assertTrue(listed and set(listed) <= variants, f"CLAUDE.md lists {listed}; DMLModelType has {sorted(variants)}")


# ---------------------------------------------------------------- Task 4: ECON-80 (bounds, time-series index and registry)

class Econ80Test(unittest.TestCase):
    """ECON-80: Balke-Pearl table, time-series index and R-pinned VAR/IRF/Granger/VARMA/VECM docs."""

    def test_econ80_bounds_and_time_series(self):
        b = read_doc("econometrics/bpbounds.md")
        for value in ("-0.1946228482113902", "0.0032280386285733"):   # vitamin A (Sommer et al. 1986), closed form
            self.assertIn(value, b)
        idx = read_doc("econometrics/timeseries/README.md")
        self.assertNotIn("R custom", idx)
        self.assertNotIn("timeseries::tests::test_validate", idx)
        docs = sorted(p.name for p in (VALIDATION / "econometrics/timeseries").glob("*.md") if p.name != "README.md")
        for name in docs:
            with self.subTest(doc=name):
                self.assertIn(f"]({name})", idx)
        entries = registry_entries()
        for name in ("var.md", "irf.md", "varma.md", "vecm.md", "granger.md"):
            with self.subTest(registered=name):
                self.assertTrue(any(e["doc"].endswith("econometrics/timeseries/" + name) for e in entries))


if __name__ == "__main__":
    unittest.main()
