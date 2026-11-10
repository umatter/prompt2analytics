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
    prop = {t for t, (kind, _) in read_labels().items() if kind == "registered-property"}
    # A method whose only tests compare no R value (registered-property) is not R-pinned.
    mine = [e for e in entries if os.path.normpath(e["doc"]) == os.path.normpath(doc)
            and any(t not in prop for t in e["tests"])]
    if not mine:
        return "not R-pinned"
    ids = sorted({d["finding"] for e in mine for d in e.get("deviations", [])})
    return "R-pinned" if not ids else "R-pinned, known deviations (" + ", ".join(ids) + ")"


def md_files():
    for path in sorted(VALIDATION.rglob("*.md")):
        rel = path.relative_to(VALIDATION).as_posix()
        if rel.startswith(("reports/", "tools/tests/fixtures/")):
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
    line_re = re.compile(r"^(?P<id>p2a-[a-z]+::\S+)\s*\|\s*(?P<kind>formula|exact|property|quoted|registered-property)\s*\|\s*(?P<reason>\S.*)$")
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



_SPANS = {}


def _fn_spans(src):
    """{name: (start, end)} of every fn in the source (any attribute: #[test], #[tokio::test], helpers)."""
    if src not in _SPANS:
        _SPANS[src] = _fn_spans_uncached(src)
    return _SPANS[src]


def _fn_spans_uncached(src):
    code = blank(src)
    spans = {}
    for m in re.finditer(r"\bfn\s+([A-Za-z0-9_]+)\s*(?:<[^>{]*>)?\s*\(", code):
        depth, j = 0, m.end() - 1
        while j < len(code):
            depth += {"(": 1, ")": -1}.get(code[j], 0)
            if depth == 0:
                break
            j += 1
        brace = code.find("{", j)
        semi = code.find(";", j)
        if brace == -1 or (semi != -1 and semi < brace):
            continue
        end = match_brace(code, brace)
        spans.setdefault(m.group(1), (m.start(), end + 1))
    return spans


_CRATE_FNS = {}


def _crate_fns(package):
    """{name: source} of the fns defined exactly once in the crate (for helpers shared across files)."""
    if package not in _CRATE_FNS:
        seen, out = {}, {}
        for path in sorted((CRATES / package).rglob("*.rs")):
            if "target" in path.parts:
                continue
            src = path.read_text(encoding="utf-8")
            for name, (s, e) in _fn_spans(src).items():
                seen[name] = seen.get(name, 0) + 1
                out[name] = src[s:e]
        _CRATE_FNS[package] = {k: v for k, v in out.items() if seen[k] == 1}
    return _CRATE_FNS[package]

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


def body_rules(name, attrs, body, scope, helper_calls=()):
    """Rule names violated by one test. `scope` adds the X-10 rules beyond print-only. `helper_calls`
    are the offsets of calls to helpers that assert (they count as assertions at the call)."""
    if re.search(r"#\[ignore\b", attrs):
        return []
    should_panic = "should_panic" in attrs
    asserts = sorted({a.start() for a in ASSERTION.finditer(body)} | set(helper_calls))
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
        calls = asserting_helper_calls(rel, src, body)
        for rule in dict.fromkeys(body_rules(name, attrs, body, scope, calls)):
            hits.append((line, name, rule))
    return hits


CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\s*(?:::<[^>]*>)?\(")


_ASSERTS = {}


def asserting_helper_calls(rel, src, body, depth=3):
    """Offsets in `body` of calls to fns that assert: same file first, else defined once in the crate,
    judged transitively up to `depth` levels (a test that delegates its comparison to a helper)."""
    parts = rel.split("/")
    package = parts[1] if len(parts) > 2 and parts[0] == "crates" else None
    key = (rel, src)
    if key not in _ASSERTS:
        spans = _fn_spans(src)
        crate = _crate_fns(package) if package else {}
        code_of = {}

        def code(fn):
            if fn not in code_of:
                if fn in spans:
                    s, e = spans[fn]
                    code_of[fn] = blank(src[s:e])
                elif fn in crate:
                    code_of[fn] = blank(crate[fn])
                else:
                    code_of[fn] = None
            return code_of[fn]

        memo = {}

        def asserts(fn, level, stack):
            if fn in memo:
                return memo[fn]
            c = code(fn)
            if c is None or level > depth or fn in stack:
                return False
            hit = bool(ASSERTION.search(c)) or any(
                asserts(x, level + 1, stack | {fn}) for x in set(CALL.findall(c)) if x != fn)
            memo[fn] = hit
            return hit

        _ASSERTS[key] = lambda name: asserts(name, 1, frozenset())
    check = _ASSERTS[key]
    return [m.start() for m in CALL.finditer(body) if check(m.group(1))]
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
        enum = re.search(r"pub enum DMLModelType\s*\{(.*?)\}", blank(src), re.DOTALL).group(1)
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


# ---------------------------------------------------------------- Task 5: ECON-81 and deferred-pin clauses

# Each family of the old "Survival ... / Discrete ..." status lines (ECON-81) and a pattern its registry
# entry matches in id, name or reference.
ECON81_FAMILIES = [
    ("Kaplan-Meier", r"kaplan|\bkm\b|_km_|survfit"), ("log-rank", r"log.?rank|survdiff"), ("Cox PH", r"\bcox"),
    ("AFT", r"\baft\b|accelerated failure|survreg"), ("logit", r"logit"), ("probit", r"probit"),
    ("multinomial logit", r"multinom"), ("ordered logit/probit", r"ordered"),
    ("negative binomial", r"negbin|negative binomial|glm\.nb"), ("ZIP", r"\bzip\b|zero-inflated poisson"),
    ("ZINB", r"\bzinb\b|zero-inflated negative"), ("hurdle (zero part)", r"hurdle"),
    ("conditional logit", r"conditional.logit|mlogit"), ("mixed logit", r"mixed.logit|gmnl|rpar"),
]
ECON81_LEGACY = ["ordered_logit_test.csv", "zip_test.csv", "hurdle_poisson_test.csv", "negbin_test.csv", "logit_simple.csv"]


class Econ81Test(unittest.TestCase):
    def test_every_family_is_registered(self):
        entries = registry_entries()
        text = [f"{e['id']} {e['name']} {e['reference']}".lower() for e in entries]
        missing = [fam for fam, pat in ECON81_FAMILIES if not any(re.search(pat, t) for t in text)]
        self.assertEqual(missing, [])

    def test_competing_risks_is_pinned_or_says_not(self):
        entries = registry_entries()
        if any(re.search(r"competing|cuminc", f"{e['id']} {e['name']}".lower()) for e in entries):
            return
        surv = (VALIDATION / "econometrics/survival.md").read_text(encoding="utf-8")
        self.assertRegex(surv, r"(?im)^.*competing risks.*\bnot R-pinned \([A-Z]+-\d+")

    def test_named_legacy_csvs_are_superseded_or_unclaimed(self):
        superseded = {os.path.basename(p) for e in registry_entries() for p in e.get("supersedes_legacy", [])}
        claims = []
        for name in ECON81_LEGACY:
            if name in superseded:
                continue
            for p in md_files():
                for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                    if name in line and not re.search(r"not asserted|legacy", line, re.IGNORECASE):
                        claims.append(f"{p.relative_to(REPO)}:{n}: {line.strip()}")
        self.assertEqual(claims, [], "\n" + "\n".join(claims))


class DeferredClaimsTest(unittest.TestCase):
    """No validation doc still defers its R pin to one of the seven findings WP64 closes."""

    DEFER = re.compile(r"\b(?:comes? with|arrives? with|added (?:later )?by|adds? the (?:R-pinned )?case|left (?:open )?for|pending|waits? for|deferred to)\b[^.\n|]{0,100}?\b(" + "|".join(re.escape(s) for s in SEVEN) + r")\b")

    def test_no_doc_defers_to_a_closed_finding(self):
        hits = []
        for p in md_files():
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if self.DEFER.search(line):
                    hits.append(f"{p.relative_to(REPO)}:{n}: {line.strip()[:160]}")
        self.assertEqual(hits, [], "\n" + "\n".join(hits))


# ---------------------------------------------------------------- Task 6: indexes (README, categories, datasets)

class IndexTest(unittest.TestCase):
    def test_registered_docs_are_in_their_category_index(self):
        missing = []
        for doc in sorted({os.path.normpath(e["doc"]) for e in registry_entries()}):
            d = (REPO / doc).parent
            idx = d / "README.md"
            if d != VALIDATION and idx.is_file() and f"]({Path(doc).name})" not in idx.read_text(encoding="utf-8"):
                missing.append(f"{idx.relative_to(REPO)} lacks {Path(doc).name}")
        self.assertEqual(missing, [], "\n" + "\n".join(missing))

    def test_econometrics_index_lists_every_doc(self):
        idx = (VALIDATION / "econometrics" / "README.md").read_text(encoding="utf-8")
        docs = sorted(p.name for p in (VALIDATION / "econometrics").glob("*.md") if p.name != "README.md")
        self.assertEqual([d for d in docs if f"]({d})" not in idx], [])
        self.assertIn("](timeseries/README.md)", idx)
        self.assertNotIn("econometrics::tests::test_validate", idx)

    def test_every_generated_dataset_is_documented(self):
        readme = (VALIDATION / "datasets" / "README.md").read_text(encoding="utf-8")
        rows = {}
        for cells in table_rows(section(readme, "Generated datasets")):
            m = re.search(r"generated/([A-Za-z0-9_.-]+\.csv)", cells[0])
            if m:
                rows[m.group(1)] = cells
        cases = {}
        for p in (VALIDATION / "expected").glob("*.json"):
            c = json.loads(p.read_text(encoding="utf-8"))
            for ds in c.get("datasets", []):
                cases.setdefault(Path(ds).name, set()).add(c["id"])
        problems = []
        for csv_path in sorted((VALIDATION / "datasets" / "generated").glob("*.csv")):
            name = csv_path.name
            cells = rows.pop(name, None)
            if cells is None or len(cells) < 5:
                problems.append(f"{name}: no row (| File | Rows | Case(s) | R file | Source |)")
                continue
            with csv_path.open(newline="", encoding="utf-8") as fh:
                records = list(csv.reader(fh))
            want = (len(records) - 1, len(records[0]))
            m = re.match(r"^\s*([0-9][0-9,]*)\s*×\s*([0-9]+)", cells[1])
            if not m or (int(m.group(1).replace(",", "")), int(m.group(2))) != want:
                problems.append(f"{name}: rows cell {cells[1]!r}, file has {want[0]} × {want[1]}")
            listed = set(re.findall(r"`([a-z0-9_]+)`", cells[2]))
            if not listed or not listed <= cases.get(name, set()):
                problems.append(f"{name}: cases {sorted(listed)} vs case files {sorted(cases.get(name, set()))}")
            rfile = re.search(r"`(?:validation/)?(?:r/cases/)?([a-z0-9_]+\.R)`", cells[3])
            if not rfile or not (VALIDATION / "r" / "cases" / rfile.group(1)).is_file():
                problems.append(f"{name}: R file cell {cells[3]!r}")
            if not cells[4]:
                problems.append(f"{name}: empty source")
        problems += [f"{name}: row for a file that does not exist" for name in rows]
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_harness_packages_are_listed(self):
        packages = set()
        for p in (VALIDATION / "expected").glob("*.json"):
            packages |= set(json.loads(p.read_text(encoding="utf-8")).get("packages", {}))
        prereq = section((VALIDATION / "README.md").read_text(encoding="utf-8"), "Prerequisites")
        missing = sorted(pk for pk in packages if not re.search(rf"(?<![\w.]){re.escape(pk)}(?![\w.])", prereq))
        self.assertEqual(missing, [])


# ---------------------------------------------------------------- Task 8: X-10 and X-11 (validation tests can fail; test_validate selects pinned tests)

FIXTURES = {
    "if_let_ok": ("#[test]\nfn test_validate_a() {\n    if let Ok(r) = run() {\n        assert!(r > 0.0);\n    }\n}\n",
                  {"swallowed-err", "conditional-assert", "guarded-assert"}),
    "err_arm_prints": ("#[test]\nfn test_validate_b() {\n    match run() {\n        Ok(r) => assert!(r > 0.0),\n        Err(e) => eprintln!(\"{e}\"),\n    }\n}\n",
                       {"swallowed-err"}),
    "err_arm_panics": ("#[test]\nfn test_validate_c() {\n    match run() {\n        Ok(r) => assert!(r > 0.0),\n        Err(e) => panic!(\"{e}\"),\n    }\n}\n", set()),
    "guard_that_panics": ("#[test]\nfn test_validate_c2() {\n    let r = run().unwrap();\n    if !r.converged {\n        panic!(\"not converged\");\n    }\n    assert!(r.b > 0.0);\n}\n", set()),
    "print_only": ("#[test]\nfn test_validate_d() {\n    let r = run().unwrap();\n    println!(\"{r}\");\n}\n", {"print-only"}),
    "no_assertion": ("#[test]\nfn test_validate_e() {\n    let _ = run().unwrap();\n}\n", {"no-assertion"}),
    "guarded": ("#[test]\nfn test_validate_f() {\n    let r = run().unwrap();\n    assert_eq!(r.n, 3);\n    if let Some(i) = r.find(1.0) {\n        assert!(r.v[i] > 0.0);\n    }\n}\n",
                {"guarded-assert"}),
    "len_guard": ("#[test]\nfn test_validate_g() {\n    let r = run().unwrap();\n    if r.segments.len() == 2 {\n        assert!(r.segments[0] > 1.0);\n    }\n}\n",
                  {"conditional-assert", "guarded-assert"}),
    "if_let_ok_else_panics": ("#[test]\nfn test_validate_l() {\n    if let Ok(r) = run() {\n        assert!(r > 0.0);\n    } else {\n        panic!(\"run failed\");\n    }\n}\n", set()),
    "expected_error_variant": ("#[test]\nfn test_validate_m() {\n    match run() {\n        Err(EconError::InvalidSpecification { .. }) => {}\n        other => panic!(\"{other:?}\"),\n    }\n}\n", set()),
    "pinned": ("#[test]\nfn test_validate_h() {\n    let (case, ds) = load_case(\"x\");\n    let r = run(&ds).unwrap();\n    Check::new(&case).scalar(\"b\", r.b, Tol::coef(10)).finish();\n}\n", set()),
    "should_panic": ("#[test]\n#[should_panic(expected = \"boom\")]\nfn test_validate_i() {\n    run().unwrap();\n}\n", set()),
    "ignored_benchmark": ("#[test]\n#[ignore = \"benchmark\"]\nfn benchmark_j() {\n    println!(\"{:?}\", run());\n}\n", set()),
    "comment_and_string": ("#[test]\nfn test_validate_k() {\n    // if let Ok(r) = run() { }\n    let s = \"if let Ok(x) = y { println!() }\";\n    assert_eq!(s.len(), 33);\n}\n", set()),
    "print_only_anywhere": ("#[test]\nfn test_other() {\n    println!(\"{:?}\", run());\n}\n", {"print-only"}),
}


class ValidateBodiesTest(unittest.TestCase):
    """X-10: no validation test passes on Err, prints instead of asserting, or guards its assertions."""

    def test_rules_on_fixtures(self):
        for label, (src, want) in FIXTURES.items():
            with self.subTest(fixture=label):
                got = {rule for _, _, rule in scan_source("crates/x/src/lib.rs", src)}
                self.assertEqual(got, want)

    def test_validation_tests_can_fail(self):
        hits, pending = [], set()
        for rel, path in rust_files():
            for line, name, rule in scan_source(rel, path.read_text(encoding="utf-8"), CITED_OTHER_NAMES):
                if (rel, name, rule) in PENDING_TEST_CODE:
                    pending.add((rel, name, rule))
                    continue
                hits.append(f"{rel}:{line}: {rule}: {name}")
        self.assertEqual(hits, [], "\n" + "\n".join(hits))
        stale = sorted(PENDING_TEST_CODE - pending)
        self.assertEqual(stale, [], "fixed sites: remove them from PENDING_TEST_CODE")


# Known test-code defects left to the owner of the .rs files; the list may only shrink (a fixed site
# that stays listed fails the test). Empty since the last four sites were fixed; keep it empty.
PENDING_TEST_CODE: set[tuple[str, str, str]] = set()


# The test functions the findings cite (name at d6a3a3b). Each is gone (deleted or renamed), or it is
# registered in validation/methods/*.json (R-pinned) or labelled in validate_property_tests.txt.
# (finding, module hint in the libtest path, name)
CITED = [
    ("X-10", "regression::ols", "test_validate_robust_se_hc0"),
    ("X-10", "regression::ols", "test_validate_robust_se_hc1"),
    ("X-10", "regression::ols", "test_validate_robust_se_hc2_hc3"),
    ("X-10", "regression::ols", "test_validate_clustered_se_one_way"),
    ("X-10", "regression::ols", "test_validate_clustered_se_two_way"),
    ("X-10", "regression::ols", "test_validate_hac_vs_r"),
    ("X-10", "regression::ols", "test_validate_bootstrap_vs_r"),
    ("X-10", "regression::ols", "test_validate_driscoll_kraay_vs_r"),
    ("X-10", "regression::diagnostics", "test_validate_bgtest_vs_r"),
    ("X-10", "regression::diagnostics", "test_validate_reset_vs_r"),
    ("X-10", "regression::diagnostics", "test_validate_harvey_collier_vs_r"),
    ("X-10", "regression::diagnostics", "test_validate_wald_vs_r"),
    ("X-10", "econometrics::iv", "test_validate_iv2sls_basic"),
    ("X-10", "binary_choice", "test_validate_logit_simple"),
    ("X-10", "econometrics::did", "test_validate_did_classic_2x2"),
    ("X-10", "econometrics::spatial", "test_validate_sar_vs_r"),
    ("X-10", "econometrics::spatial", "test_validate_sem_vs_r"),
    ("X-10", "econometrics::spatial", "test_validate_sac_vs_r"),
    ("X-10", "econometrics::spatial", "test_validate_sar_coefficient_signs"),
    ("X-10", "forecasting::garch", "test_validate_garch_parameter_constraints"),
    ("X-10", "ml::clustering", "test_validate_kmeans_vs_r"),
    ("X-10", "ml::reduction", "test_validate_pca_vs_r"),
    ("X-10", "econometrics::tmle", "test_validate_tmle_vs_r"),
    ("X-10", "econometrics::tmle", "test_validate_tmle_continuous_outcome"),
    ("X-10", "econometrics::treatment", "test_validate_ipw_ate_vs_r"),
    ("X-10", "econometrics::matching", "test_validate_matching_vs_r"),
    ("X-10", "econometrics::mediation", "test_validate_mediation_vs_r"),
    ("X-10", "timeseries", "test_validate_var_vs_r"),
    ("X-10", "econometrics::feglm", "test_validate_feglm_logit_against_alpaca"),
    ("X-10", "forecasting::changepoint", "test_validate_pelt_two_changepoints"),
    ("X-10", "forecasting::changepoint", "test_validate_pelt_segment_means"),
    ("X-10", "econometrics::rd", "test_validate_fuzzy_rd_structure"),
    ("X-10", "econometrics::panel", "test_validate_arellano_bond_full_grunfeld"),
    ("X-10", "econometrics::panel", "test_validate_gmm_twostep_full_grunfeld"),
    ("X-10", "econometrics::survival", "test_validate_km_against_r"),
    ("X-11", "ordered_models", "test_validate_ordered_logit_vs_r"),
    ("X-11", "ordered_models", "test_validate_ordered_probit_vs_r"),
    ("X-11", "ordered_models", "test_validate_ordered_logit_multiple_predictors"),
    ("X-11", "econometrics::survival", "test_validate_cox_ties_against_r"),
    ("X-11", "forecasting::garch", "test_validate_garch_persistence_calculation"),
    ("X-11", "forecasting::garch", "test_validate_garch_unconditional_variance"),
    ("X-11", "forecasting::garch", "test_validate_garch_half_life"),
    ("X-11", "forecasting::garch", "test_validate_garch_conditional_variance"),
    ("X-11", "regression::smooth_spline", "test_validate_smooth_spline_against_r"),
    ("X-11", "regression::ols", "test_validate_ols_longley"),
    ("ECON-80", "timeseries", "test_validate_var_coefficient_estimation"),
    ("ECON-80", "timeseries", "test_validate_var_irf_structure"),
    ("ECON-80", "timeseries", "test_validate_var_non_orthogonalized_irf"),
    ("ECON-80", "timeseries", "test_validate_varma_structure"),
    ("ECON-80", "timeseries", "test_validate_vecm_structure"),
    ("ECON-80", "econometrics::bpbounds", "test_validate_against_r"),
    ("ECON-81", "econometrics::survival", "test_validate_logrank_against_r"),
    ("ECON-81", "count_models", "test_validate_negbin_vs_r"),
    ("ECON-81", "mixed_logit", "test_validate_gmnl_vs_r"),
    ("ECON-81", "conditional_logit", "test_validate_mlogit_vs_r"),
]
# Ordinary tests X-10 cites (they swallowed errors or asserted nothing): scanned with the X-10 rules too.
CITED_OTHER_NAMES = ("test_llc_unit_root", "test_fuzzy_rd", "test_rd_multi_dataset", "test_granger_causality")


def defined_tests(hint, name):
    """Files that define `fn name(` inside a #[test] and whose path matches the module hint."""
    parts = [p for p in hint.split("::") if p]
    found = []
    for rel, path in rust_files():
        if not all(p in rel for p in parts):
            continue
        src = path.read_text(encoding="utf-8")
        if any(t == name for t, *_ in rust_tests(src)):
            found.append(rel)
    return found


class CitedTestsTest(unittest.TestCase):
    """X-10, X-11, ECON-80, ECON-81: every cited test is gone, R-pinned or labelled."""

    def test_cited_tests_are_registered_labelled_or_gone(self):
        registered = {t for e in registry_entries() for t in e["tests"]}
        labelled = set(read_labels())
        problems = []
        for finding, hint, name in CITED:
            files = defined_tests(hint, name)
            if not files:
                continue
            ok = [t for t in registered | labelled if t.endswith("::" + name) and hint.split("::")[-1] in t]
            if not ok:
                problems.append(f"{finding}: {name} still exists ({', '.join(files)}) and is neither registered nor labelled")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


# A comparison with a committed case value: a Check, or a case accessor with a key literal.
CASE_VALUE = re.compile(r'Check::new\(|\.(?:scalar|vector|named|named_subset|names|by_names)\(\s*&?(?:"|format!|[A-Za-z_])')


def _test_files(test_id):
    package, rest = test_id.split("::", 1)
    segs = rest.split("::")
    mods = segs[:-1]
    crate = CRATES / package
    for k in range(len(mods), 0, -1):
        base = "/".join(mods[:k])
        found = [p for root in ("src", "tests")
                 for p in (crate / root / f"{base}.rs", crate / root / base / "mod.rs", crate / root / base / "main.rs")
                 if p.is_file()]
        if found:
            return found
    return [crate / "src" / "lib.rs"]


def test_sources_with_helpers(test_id, depth=3):
    """Source of the test fn, plus the fns it calls (transitively, up to `depth` levels): same-file fns
    first, otherwise a fn defined once in the crate."""
    package = test_id.split("::", 1)[0]
    name = test_id.rsplit("::", 1)[-1]
    out = []
    for p in _test_files(test_id):
        src = p.read_text(encoding="utf-8")
        local = {k: src[s:e] for k, (s, e) in _fn_spans(src).items()}
        if name not in local:
            continue
        seen, frontier = {name}, [local[name]]
        for _ in range(depth + 1):
            nxt = []
            for body in frontier:
                out.append(body)
                for call in set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*(?:::<[^>]*>)?\(", blank(body))):
                    if call in seen:
                        continue
                    found = local.get(call) or _crate_fns(package).get(call)
                    if found:
                        seen.add(call)
                        nxt.append(found)
            frontier = nxt
    return out


class ValidateFilterTest(unittest.TestCase):
    """X-10: the filter `test_validate` selects only R-pinned (registered) or explicitly labelled tests."""

    def test_label_file_is_consistent(self):
        labels = read_labels()
        registered = {t for e in registry_entries() for t in e["tests"]}
        for test_id, (kind, reason) in labels.items():
            with self.subTest(test=test_id):
                self.assertIn("test_validate", test_id.rsplit("::", 1)[-1])
                if kind == "registered-property":
                    self.assertIn(test_id, registered, "kind registered-property needs a registry entry")
                else:
                    self.assertNotIn(test_id, registered, "a registered test is R-pinned; do not label it")
                self.assertGreaterEqual(len(reason), 20, "give the reason the test keeps its name")

    def test_registered_tests_compare_a_case_value(self):
        """A registered test compares a value of its case (Check, or a case accessor), or is labelled
        registered-property: it then checks properties on R's data and does not count as R-pinned."""
        labels = read_labels()
        missing = []
        for test_id in sorted({t for e in registry_entries() for t in e["tests"]}):
            if "test_validate" not in test_id.rsplit("::", 1)[-1]:
                continue
            if labels.get(test_id, ("",))[0] == "registered-property":
                continue
            sources = test_sources_with_helpers(test_id)
            if not any(CASE_VALUE.search(s) for s in sources):
                missing.append(test_id + ("" if sources else " (source not found)"))
        self.assertEqual(missing, [], "\n" + "\n".join(missing))

    def test_readme_states_the_naming_rule(self):
        text = (VALIDATION / "README.md").read_text(encoding="utf-8")
        for phrase in ("tools/validate_property_tests.txt", "*_recovers_dgp", "test_tier3_closure.py",
                       "registered-property", "`quoted`"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    @unittest.skipUnless(WITH_LISTS, NEEDS_LISTS)
    def test_every_test_validate_test_is_pinned_or_labelled(self):
        """Every test_validate_* test cargo lists is registered or labelled; a new unregistered one fails."""
        ids, vs = load_test_lists()
        self.assertTrue(ids, "no <package>-tests.txt list files found")
        validate = {t for t in ids if "test_validate" in t.rsplit("::", 1)[-1] and not vs.is_excluded(t)
                    and not t.startswith("p2a-validation::")}
        registered = {t for e in registry_entries() for t in e["tests"]}
        labels = read_labels()
        self.assertEqual(sorted(set(labels) - set(ids)), [], "labelled tests that do not exist")
        unlisted = sorted(validate - registered - set(labels))
        self.assertEqual(unlisted, [], "register these tests, rename them (they compare no committed R case) "
                         "or label them in validate_property_tests.txt:\n" + "\n".join(unlisted))


# ---------------------------------------------------------------- Task 9: validation docs and tooling after the renames

class DocCommandsTest(unittest.TestCase):
    """Every `cargo test ... -- <filter>` in a validation doc selects at least one listed test."""

    @unittest.skipUnless(WITH_LISTS, NEEDS_LISTS)
    def test_cargo_test_filters_select_tests(self):
        ids, _ = load_test_lists()
        by_pkg = {}
        for test_id, (_, raw) in ids.items():
            by_pkg.setdefault(test_id.split("::", 1)[0], []).append(raw)
        problems = []
        for p in md_files():
            in_fence = False
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if line.lstrip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if not in_fence or "cargo test" not in line or " -- " not in line:
                    continue
                cmd = line.split("#", 1)[0]
                pkg = re.search(r"-p\s+(p2a-[a-z]+)", cmd)
                names = by_pkg.get(pkg.group(1), []) if pkg else [r for v in by_pkg.values() for r in v]
                tail = []
                for tok in cmd.split(" -- ", 1)[1].split():
                    if tok in (">", ">>", "|", "&&", "||", ";") or tok.startswith((">", "2>", "|")):
                        break
                    if tok != "\\":
                        tail.append(tok)
                exact = "--exact" in tail
                for f in (t for t in tail if not t.startswith("-") and "<" not in t):
                    if not any((r == f) if exact else (f in r) for r in names):
                        problems.append(f"{p.relative_to(REPO)}:{n}: filter {f!r} selects no test")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


class ClaudeToolingTest(unittest.TestCase):
    """The agent tooling's method registry names validation docs and test patterns that exist (X-10 renames)."""

    def test_patterns_and_docs_resolve(self):
        reg = json.loads((REPO / ".claude/tooling/validation/method_registry.json").read_text(encoding="utf-8"))
        names = set()
        for _, path in rust_files():
            names.update(re.findall(r"\bfn (test_\w+)", path.read_text(encoding="utf-8")))
        problems = []
        for key, m in reg["methods"].items():
            doc = m.get("validation_doc")
            if doc and not (REPO / doc).is_file():
                problems.append(f"{key}: validation_doc {doc} does not exist")
            module = m.get("rust_module")
            if module and not (REPO / module).exists():
                problems.append(f"{key}: rust_module {module} does not exist (WP04 split or moved it)")
            for pat in m.get("test_patterns", []):
                if not any(pat in n for n in names):
                    problems.append(f"{key}: test pattern {pat!r} matches no test")
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


if __name__ == "__main__":
    unittest.main()
