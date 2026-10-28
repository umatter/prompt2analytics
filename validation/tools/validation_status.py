#!/usr/bin/env python3
"""Validation results, methods-registry lint and status page for prompt2analytics.

Subcommands (each takes --root DIR, default: the repository containing this script):

  select   Pick tests from `cargo test -p PKG --tests -- --list` output and print
           "<target>\\t<libtest name>" lines (target: --lib, --bins, --bin X, --test X).
           Default: the test_validate tests (minus the data::database read-only guards);
           --category CAT keeps one category (stats, regression, econometrics,
           forecasting, ml, cli, mcp); --all takes every listed test.
  collect  Turn cargo test logs (--package), the R drift check log of
           `Rscript validation/r/run.R --check` (--r-check) and a legacy-script run
           (--legacy) into results.json (-o), merging into an existing file.
  lint     Check validation/methods/*.json against the case files in
           validation/expected/, the test lists (--tests <pkg>-tests.txt ...) and the
           tests that load cases; report legacy documentation debt as warnings
           (errors with --strict).
  render   Write the whole validation status page from results.json.

Exit codes: 0 ok; 1 lint errors (or warnings with --strict), or a collected package
or R check that did not pass; 2 usage error or unreadable/malformed input.

Used by validation/run_validation.sh and CI; see validation/README.md and
validation/methods/README.md. Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import re
import subprocess
import sys
from bisect import bisect_left
from collections import namedtuple
from pathlib import Path
from urllib.parse import unquote

DEFAULT_ROOT = Path(__file__).resolve().parents[2]
CATEGORIES = ("stats", "regression", "econometrics", "forecasting", "ml", "cli", "mcp")
STATUS_RANK = {"ignored": 0, "ok": 1, "failed": 2}
SCHEMA = 1

Finding = namedtuple("Finding", "level rule path line message")


class UsageError(Exception):
    """Bad arguments or unreadable/malformed input (exit code 2)."""


def read_text(path) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise UsageError(f"cannot read {path}: {e}") from e


def read_json(path):
    text = read_text(path)
    try:
        return json.loads(text)
    except ValueError as e:
        raise UsageError(f"{path} is not valid JSON: {e}") from e


# --------------------------------------------------------------------------------------
# cargo output: targets, --list lines, run lines
# --------------------------------------------------------------------------------------

TARGET_RE = re.compile(r"^\s*Running (unittests )?(\S+)( \(.*\))?\s*$")
DOC_RE = re.compile(r"^\s*Doc-tests ")
LIST_RE = re.compile(r"^(\S+): (test|bench)$")
RUN_RE = re.compile(r"(?:^|\s)test (\S+)(?: - should panic)? \.\.\. (ok|FAILED|ignored)\b")
TOTALS_RE = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;")
COMPILE_ERROR_RE = re.compile(r"^error\[E\d+\]")
RUN_STATUS = {"ok": "ok", "FAILED": "failed", "ignored": "ignored"}


def target_of(path: str, package: str):
    """(target flag, test-id prefix, integration binary or None) for a `Running` path,
    or None for a target this tool does not collect (benches, examples)."""
    p = path.replace("\\", "/")
    m = re.search(r"(?:^|/)src/(lib|main)\.rs$", p)
    if m:
        return ("--lib" if m.group(1) == "lib" else "--bins", package + "::", None)
    m = re.search(r"(?:^|/)src/bin/([^/]+?)(?:\.rs|/main\.rs)$", p)
    if m:
        return ("--bin " + m.group(1), package + "::", None)
    m = re.search(r"(?:^|/)tests/([^/]+?)(?:\.rs|/main\.rs)$", p)
    if m:
        binary = m.group(1)
        return ("--test " + binary, f"{package}::{binary}::", binary)
    return None


def _scan_list(text: str, package: str):
    """([(id, target, raw name)], binaries) of `cargo test -- --list` output."""
    entries = []
    binaries = set()
    current = None
    for line in text.splitlines():
        m = TARGET_RE.match(line)
        if m:
            current = target_of(m.group(2), package)
            if current and current[2]:
                binaries.add(current[2])
            continue
        if DOC_RE.match(line):
            current = None
            continue
        m = LIST_RE.match(line)
        if m and current is not None and m.group(2) == "test":
            target, prefix, _ = current
            entries.append((prefix + m.group(1), target, m.group(1)))
    return entries, binaries


def parse_test_list(text: str, package: str) -> dict:
    """{test id: (target flag, raw libtest name)} of `cargo test -- --list` output.
    Benches are dropped; an id listed under two targets keeps the first."""
    out = {}
    for tid, target, raw in _scan_list(text, package)[0]:
        out.setdefault(tid, (target, raw))
    return out


def list_binaries(text: str, package: str) -> set:
    return _scan_list(text, package)[1]


def parse_cargo_log(text: str, package: str) -> dict:
    """{"tests": {id: ok|failed|ignored}, "totals": {...}, "binaries": [...],
    "compile_errors": [...]} of a `cargo test` log. Doc tests are not collected."""
    tests = {}
    totals = {"passed": 0, "failed": 0, "ignored": 0, "results": 0}
    binaries = set()
    compile_errors = []
    current = None  # None: before any target; False: doc tests or an uncollected target
    for line in text.splitlines():
        if COMPILE_ERROR_RE.match(line) or "could not compile" in line:
            compile_errors.append(line.strip())
            continue
        m = TARGET_RE.match(line)
        if m:
            current = target_of(m.group(2), package) or False
            if current and current[2]:
                binaries.add(current[2])
            continue
        if DOC_RE.match(line):
            current = False
            continue
        if current is False:
            continue
        m = TOTALS_RE.match(line)
        if m:
            totals["passed"] += int(m.group(2))
            totals["failed"] += int(m.group(3))
            totals["ignored"] += int(m.group(4))
            totals["results"] += 1
            continue
        m = RUN_RE.search(line)
        if m:
            prefix = current[1] if current else package + "::"
            tid = prefix + m.group(1)
            status = RUN_STATUS[m.group(2)]
            old = tests.get(tid)
            if old is None or STATUS_RANK[status] > STATUS_RANK[old]:
                tests[tid] = status
    return {
        "tests": tests,
        "totals": totals,
        "binaries": sorted(binaries),
        "compile_errors": compile_errors,
    }


def category_of(test_id: str, binaries) -> str:
    """Category of a test id: cli (p2a-cli), mcp (p2a-mcp), harness (p2a-validation);
    otherwise the first module (diagnostics counts as econometrics), ml for the
    validation_ml_* and validation_boosting_methods binaries, tests/<bin>.rs for other
    integration binaries."""
    package, _, rest = test_id.partition("::")
    if package == "p2a-cli":
        return "cli"
    if package == "p2a-mcp":
        return "mcp"
    if package == "p2a-validation":
        return "harness"
    first = rest.split("::", 1)[0]
    if first in binaries:
        if first.startswith("validation_ml") or first == "validation_boosting_methods":
            return "ml"
        return f"tests/{first}.rs"
    if first == "diagnostics":
        return "econometrics"
    return first


def is_excluded(test_id: str) -> bool:
    """The data::database read-only SQL guards (they only compile with `database`)."""
    _, _, rest = test_id.partition("::")
    return rest.startswith("data::database::") and "read_only" in rest.rsplit("::", 1)[-1]


def is_validate(test_id: str) -> bool:
    return "test_validate" in test_id.rsplit("::", 1)[-1] and not is_excluded(test_id)


def select_tests(list_text: str, package: str, category=None, all_tests=False) -> list:
    """Sorted [(target flag, raw libtest name)] to run."""
    if category is not None and category not in CATEGORIES:
        raise UsageError(f"unknown category {category!r} (one of: {', '.join(CATEGORIES)})")
    entries, binaries = _scan_list(list_text, package)
    chosen = set()
    for tid, target, raw in entries:
        if not all_tests and not is_validate(tid):
            continue
        if category is not None and category_of(tid, binaries) != category:
            continue
        chosen.add((target, raw))
    return sorted(chosen)


# --------------------------------------------------------------------------------------
# R drift check and legacy scripts
# --------------------------------------------------------------------------------------

R_OK_RE = re.compile(r"^ok {5}(\S+)\s*$")
R_FAILED_CASE_RE = re.compile(r"^FAILED (\S+)\s*$")
R_PASS_RE = re.compile(r"^R reference check: \d+ case\(s\) match the committed files\.")
R_ITEM_ID_RE = re.compile(r"^([a-z0-9_]+): ")
R_MISSING_PKG_RE = re.compile(r"there is no package called [‘'](\S+?)[’']")


def parse_r_check(text: str) -> dict:
    """{"status": pass|fail|missing|error, "cases": {id: ok|failed}, "failures": [...],
    "missing_packages": [...]} of `Rscript validation/r/run.R --check` output."""
    lines = text.splitlines()
    cases, failures, missing = {}, [], []
    passed = failed = in_block = False
    for line in lines:
        m = R_OK_RE.match(line)
        if m:
            cases[m.group(1)] = "ok"
        m = R_FAILED_CASE_RE.match(line)
        if m:
            cases[m.group(1)] = "failed"
        if R_PASS_RE.match(line):
            passed = True
        if line.startswith("R reference check FAILED:"):
            failed = in_block = True
        elif in_block and line.startswith("  "):
            item = line[2:].rstrip()
            failures.append(item)
            m = R_ITEM_ID_RE.match(item)
            if m:
                cases[m.group(1)] = "failed"
        elif in_block and line.strip():
            in_block = False
        for pkg in R_MISSING_PKG_RE.findall(line):
            if pkg not in missing:
                missing.append(pkg)
    first = next((ln for ln in lines if ln.strip()), "")
    if failed:
        status = "fail"
    elif passed:
        status = "pass"
    elif first.startswith("Rscript not found"):
        status = "missing"
    else:
        status = "error"
    return {
        "status": status,
        "cases": dict(sorted(cases.items())),
        "failures": failures,
        "missing_packages": missing,
    }


def _number(cell: str):
    try:
        return float(cell)
    except ValueError:
        return None


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-8 * max(abs(a), abs(b)) + 1e-12


def _read_csv(path: Path) -> list:
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            return list(csv.reader(f))
    except OSError as e:
        raise UsageError(f"cannot read {path}: {e}") from e


def compare_csv(produced: Path, tracked: Path):
    """("same" | "differs", detail): same header and shape, numeric cells within
    |a - b| <= 1e-8 * max(|a|, |b|) + 1e-12, other cells equal after strip. The detail
    names the first differing cell as "<produced> vs <tracked>"."""
    a, b = _read_csv(produced), _read_csv(tracked)
    header_a = [c.strip() for c in a[0]] if a else []
    header_b = [c.strip() for c in b[0]] if b else []
    if header_a != header_b:
        return "differs", f"header {','.join(header_a)} vs {','.join(header_b)}"
    if len(a) != len(b):
        return "differs", f"{max(len(a) - 1, 0)} rows vs {max(len(b) - 1, 0)} rows"
    for r, (row_a, row_b) in enumerate(zip(a[1:], b[1:]), start=1):
        if len(row_a) != len(row_b):
            return "differs", f"row {r}: {len(row_a)} columns vs {len(row_b)} columns"
        for c, (x, y) in enumerate(zip(row_a, row_b)):
            x, y = x.strip(), y.strip()
            fx, fy = _number(x), _number(y)
            same = _close(fx, fy) if fx is not None and fy is not None else x == y
            if not same:
                column = header_a[c] if c < len(header_a) and header_a[c] else str(c + 1)
                return "differs", f"row {r}, column {column}: {x} vs {y}"
    return "same", ""


def collect_legacy(json_path: Path, root: Path) -> dict:
    doc = read_json(json_path)
    if not isinstance(doc, dict) or not isinstance(doc.get("scripts"), list):
        raise UsageError(f"{json_path}: expected an object with a `scripts` list")
    produced_dir = Path(str(doc.get("produced_dir", "produced")))
    if not produced_dir.is_absolute():
        produced_dir = json_path.parent / produced_dir
    scripts = {}
    for s in doc["scripts"]:
        if not isinstance(s, dict) or s.get("status") not in ("ran", "error", "timeout"):
            raise UsageError(f"{json_path}: malformed script entry {s!r}")
        scripts[str(s.get("script"))] = {
            "status": s["status"],
            "exit_code": s.get("exit_code"),
            "seconds": s.get("seconds"),
        }
    expected_dir = root / "validation" / "expected"
    files, details = {}, {}
    produced = sorted(produced_dir.glob("*.csv")) if produced_dir.is_dir() else []
    for p in produced:
        tracked = expected_dir / p.name
        if not tracked.is_file():
            files[p.name] = "untracked"
            continue
        status, detail = compare_csv(p, tracked)
        files[p.name] = status
        if detail:
            details[p.name] = detail
    for p in sorted(expected_dir.glob("*.csv")):
        files.setdefault(p.name, "not_produced")
    return {
        "timeout_seconds": doc.get("timeout_seconds"),
        "scripts": dict(sorted(scripts.items())),
        "files": dict(sorted(files.items())),
        "details": dict(sorted(details.items())),
    }


# --------------------------------------------------------------------------------------
# collect
# --------------------------------------------------------------------------------------


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "unknown"


def package_summary(name: str, pkg: dict) -> str:
    counts = f"{pkg['passed']} passed, {pkg['failed']} failed, {pkg['ignored']} ignored"
    if pkg.get("selected") is not None:
        detail = f"selected {pkg['selected']}, ran {pkg['ran']}: {counts}"
    else:
        detail = f"ran {pkg['ran']}: {counts}"
    return f"{name}: {pkg['status']} ({detail})"


def collect_package(results: dict, package: str, logs, exit_code, selected, features) -> dict:
    prefix = package + "::"
    tests = {k: v for k, v in results.get("tests", {}).items() if not k.startswith(prefix)}
    merged, binaries, compile_errors = {}, set(), []
    totals = {"passed": 0, "failed": 0, "ignored": 0, "results": 0}
    for log in logs:
        parsed = parse_cargo_log(read_text(log), package)
        for tid, status in parsed["tests"].items():
            old = merged.get(tid)
            if old is None or STATUS_RANK[status] > STATUS_RANK[old]:
                merged[tid] = status
        for k in totals:
            totals[k] += parsed["totals"][k]
        binaries.update(parsed["binaries"])
        compile_errors.extend(parsed["compile_errors"])
    if totals["results"]:
        passed, failed, ignored = totals["passed"], totals["failed"], totals["ignored"]
    else:
        values = list(merged.values())
        passed, failed, ignored = (values.count(s) for s in ("ok", "failed", "ignored"))
    ran = passed + failed  # an ignored test did not run
    errors = []
    if exit_code:
        status = "compile_error" if compile_errors else "fail"
        errors.extend(compile_errors[:20])
        if not compile_errors:
            errors.append(f"cargo exited {exit_code}")
    elif failed or any(s == "failed" for s in merged.values()):
        status = "fail"
    else:
        status = "pass"
    if status == "pass" and selected is not None and ran != selected:
        status = "fail"
    if selected is not None and ran != selected:
        errors.append(f"selected {selected} tests, ran {ran}")
    if status == "pass" and selected == 0:
        status = "no_tests"
    tests.update(merged)
    results["tests"] = dict(sorted(tests.items()))
    results.setdefault("binaries", {})[package] = sorted(binaries)
    if features is not None:
        results.setdefault("features", {})[package] = features
    pkg = {
        "status": status,
        "exit_code": exit_code,
        "selected": selected,
        "ran": ran,
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "errors": errors,
    }
    results.setdefault("packages", {})[package] = pkg
    return pkg


def cmd_collect(args) -> int:
    root = Path(args.root)
    out = Path(args.output)
    if out.exists():
        results = read_json(out)
        if not isinstance(results, dict) or results.get("schema") != SCHEMA:
            raise UsageError(f"{out}: not a schema-{SCHEMA} results file")
    else:
        results = {"schema": SCHEMA}
    results["commit"] = git_commit(root)
    now = datetime.datetime.now(datetime.timezone.utc)
    results["date"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if args.command is not None:
        results["command"] = args.command
    if args.category is not None:
        results["category"] = args.category
    results.setdefault("command", None)
    results.setdefault("category", None)
    for key in ("features", "packages", "binaries", "tests"):
        results.setdefault(key, {})
    if args.logs and not args.package:
        raise UsageError("cargo logs need --package")
    rc = 0
    if args.package:
        pkg = collect_package(
            results, args.package, args.logs, args.exit_code, args.selected, args.features
        )
        print("collect: " + package_summary(args.package, pkg))
        for e in pkg["errors"]:
            print(f"collect: {args.package}: error: {e}")
        if pkg["status"] in ("fail", "compile_error"):
            rc = 1
    if args.r_check:
        r = parse_r_check(read_text(args.r_check))
        results["r_check"] = r
        ok = sum(1 for s in r["cases"].values() if s == "ok")
        bad = sum(1 for s in r["cases"].values() if s == "failed")
        print(f"collect: R reference check: {r['status']} ({ok} ok, {bad} failed)")
        if r["status"] in ("fail", "error"):
            rc = 1
    if args.legacy:
        legacy = collect_legacy(Path(args.legacy), root)
        results["legacy_r"] = legacy
        print("collect: legacy R scripts: " + legacy_summary(legacy))
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    tmp.replace(out)
    return rc


def legacy_counts(legacy: dict):
    statuses = [s["status"] for s in legacy.get("scripts", {}).values()]
    files = list(legacy.get("files", {}).values())
    return (
        {k: statuses.count(k) for k in ("ran", "error", "timeout")},
        {k: files.count(k) for k in ("same", "differs", "untracked", "not_produced")},
    )


def legacy_summary(legacy: dict) -> str:
    s, f = legacy_counts(legacy)
    return (
        f"{s['ran']} ran, {s['error']} error, {s['timeout']} timeout; files: "
        f"{f['same']} same, {f['differs']} differs, {f['untracked']} untracked, "
        f"{f['not_produced']} not produced"
    )


# --------------------------------------------------------------------------------------
# Rust source scanning (lint)
# --------------------------------------------------------------------------------------

_START_RE = re.compile(
    r"(?P<lc>//)|(?P<bc>/\*)|(?P<raw>(?<![A-Za-z0-9_])b?r#*\")|(?P<str>\")|(?P<chr>')"
)
_BLOCK_RE = re.compile(r"/\*|\*/")
_STR_END_RE = re.compile(r'(?:[^"\\]|\\.)*"', re.S)
_FN_TEST_RE = re.compile(r"\bfn\s+(test_\w+)")
_LOAD_RE = re.compile(r'\b(?:Case::(?:try_)?load|load_case)\(\s*"([^"]*)"')
_CSV_CITATION_RE = re.compile(r"\breference values?\b.*validation/expected/[\w.-]+\.csv")


def _blank(text: str) -> str:
    return re.sub(r"[^\n]", " ", text)


def scan_rust(text: str):
    """(code, no_comments, comments) of Rust source: `code` has comments, strings and
    char literals blanked, `no_comments` only comments; both keep every offset.
    `comments` lists (start, end) of `//` comments. Lifetimes are code."""
    code, keep = [], []
    comments = []
    pos = 0
    n = len(text)
    while True:
        m = _START_RE.search(text, pos)
        if not m:
            break
        i = m.start()
        kind = m.lastgroup
        code.append(text[pos:i])
        keep.append(text[pos:i])
        if kind == "lc":
            end = text.find("\n", i)
            end = n if end < 0 else end
            comments.append((i, end))
            blank = _blank(text[i:end])
            code.append(blank)
            keep.append(blank)
        elif kind == "bc":
            depth, j = 1, m.end()
            while depth:
                mm = _BLOCK_RE.search(text, j)
                if not mm:
                    j = n
                    break
                depth += 1 if mm.group() == "/*" else -1
                j = mm.end()
            end = j
            blank = _blank(text[i:end])
            code.append(blank)
            keep.append(blank)
        elif kind == "raw":
            hashes = m.group().count("#")
            close = '"' + "#" * hashes
            k = text.find(close, m.end())
            end = n if k < 0 else k + len(close)
            code.append(_blank(text[i:end]))
            keep.append(text[i:end])
        elif kind == "str":
            mm = _STR_END_RE.match(text, m.end())
            end = mm.end() if mm else n
            code.append(_blank(text[i:end]))
            keep.append(text[i:end])
        else:  # a quote: char literal, or a lifetime / label
            if i + 1 < n and text[i + 1] == "\\":
                k = text.find("'", i + 3)
                end = n if k < 0 else k + 1
            elif i + 2 < n and text[i + 2] == "'":
                end = i + 3
            else:
                end = i + 1
                code.append("'")
                keep.append("'")
                pos = end
                continue
            code.append(_blank(text[i:end]))
            keep.append(text[i:end])
        pos = end
    code.append(text[pos:])
    keep.append(text[pos:])
    return "".join(code), "".join(keep), comments


def test_functions(text: str):
    """[(name, line, body_start, body_end)] of every `fn test_*` with a body."""
    code, keep, comments = scan_rust(text)
    out = []
    for m in _FN_TEST_RE.finditer(code):
        k = m.end()
        brace, semi = code.find("{", k), code.find(";", k)
        if brace < 0 or (0 <= semi < brace):
            continue
        depth, end = 0, len(code)
        for mm in re.finditer(r"[{}]", code[brace:]):
            depth += 1 if mm.group() == "{" else -1
            if depth == 0:
                end = brace + mm.end()
                break
        line = code.count("\n", 0, m.start()) + 1
        out.append((m.group(1), line, brace, end))
    return out, keep, comments


SKIP_DIRS = {"target", "node_modules", ".git"}


def all_rust_files(root: Path) -> list:
    """Every crates/**/*.rs, skipping target/ and node_modules/."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root / "crates"):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        files.extend(Path(dirpath) / f for f in sorted(filenames) if f.endswith(".rs"))
    return files


def rust_files(root: Path) -> list:
    """crates/*/src/**/*.rs and crates/*/tests/**/*.rs."""
    out = []
    for p in all_rust_files(root):
        parts = p.relative_to(root / "crates").parts
        if len(parts) >= 3 and parts[1] in ("src", "tests"):
            out.append(p)
    return out


def crate_package(root: Path, crate_dir: str) -> str:
    manifest = root / "crates" / crate_dir / "Cargo.toml"
    if manifest.is_file():
        section = None
        for line in read_text(manifest).splitlines():
            s = line.strip()
            if s.startswith("["):
                section = s
                continue
            if section == "[package]":
                m = re.match(r'name\s*=\s*"([^"]+)"', s)
                if m:
                    return m.group(1)
    return crate_dir


# --------------------------------------------------------------------------------------
# Markdown scanning (lint)
# --------------------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_LINK_RE = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_TOKEN_RE = re.compile(r"(?<![\w/.-])(test_[a-z0-9_]+)(?![\w/-]|\.[A-Za-z])")
_COLON_RE = re.compile(r"::(test_[a-z0-9_]+)")
_FN_RE = re.compile(r"\bfn (test_[a-z0-9_]+)")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_DATASET_RE = re.compile(r"\*\*File\*\*:\s*`([^`]+)`")


def markdown_lines(text: str):
    """Yield (line number, line, in_fence) with fence lines themselves marked in_fence."""
    fence = None
    for no, line in enumerate(text.splitlines(), start=1):
        m = _FENCE_RE.match(line)
        if m:
            mark = m.group(1)
            if fence is None:
                fence = mark
                yield no, line, True
                continue
            if mark[0] == fence[0] and len(mark) >= len(fence) and not line.strip()[len(mark):].strip():
                fence = None
                yield no, line, True
                continue
        yield no, line, fence is not None


def validation_markdown(root: Path):
    vdir = root / "validation"
    out = []
    for p in sorted(vdir.glob("**/*.md")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith("validation/reports/") or rel.startswith("validation/tools/tests/fixtures/"):
            continue
        out.append(p)
    return out


def test_name_tokens(text: str):
    """[(line, column, name)] of test names mentioned in Markdown (one per position)."""
    seen = {}
    for no, line, in_fence in markdown_lines(text):
        if in_fence:
            if "cargo test" in line:
                spans = [(0, line)]
                patterns = (_TOKEN_RE, _COLON_RE)
            else:
                spans = [(0, line)]
                patterns = (_FN_RE, _COLON_RE)
        else:
            spans = [(m.start(1), m.group(1)) for m in _INLINE_CODE_RE.finditer(line)]
            patterns = (_TOKEN_RE, _COLON_RE)
        for offset, span in spans:
            for pat in patterns:
                for m in pat.finditer(span):
                    col = offset + m.start(1) + 1
                    seen.setdefault((no, col), m.group(1))
    return [(no, col, name) for (no, col), name in sorted(seen.items())]


# --------------------------------------------------------------------------------------
# lint
# --------------------------------------------------------------------------------------

METHOD_KEYS = {
    "id": str,
    "name": str,
    "category": str,
    "reference": str,
    "doc": str,
    "cases": list,
    "tests": list,
    "quantities": list,
    "tolerance": str,
    "deviations": list,
    "supersedes_legacy": list,
}
ID_RE = re.compile(r"^[a-z0-9_]+$")
FINDING_RE = re.compile(r"^[A-Z]+-[0-9]+$")


def _line_of(text: str, needle: str):
    i = text.find(needle)
    return None if i < 0 else text.count("\n", 0, i) + 1


def _value_ok(v) -> bool:
    def num(x):
        return x is None or (isinstance(x, (int, float)) and not isinstance(x, bool))

    if num(v):
        return True
    if isinstance(v, list):
        return all(num(x) for x in v)
    if isinstance(v, dict) and set(v) == {"names", "values"}:
        names, values = v["names"], v["values"]
        return (
            isinstance(names, list)
            and isinstance(values, list)
            and all(isinstance(x, str) for x in names)
            and all(num(x) for x in values)
        )
    return False


def load_registry(root: Path, findings: list):
    """[(registry path, method dict)] of every well-formed method."""
    methods = []
    seen_ids = {}
    for path in sorted((root / "validation" / "methods").glob("*.json")):
        rel = path.relative_to(root).as_posix()
        text = read_text(path)

        def err(msg, line=None, rel=rel):
            findings.append(Finding("ERROR", "registry", rel, line, msg))

        try:
            doc = json.loads(text)
        except ValueError as e:
            err(f"not valid JSON: {e}")
            continue
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            err(f"schema must be {SCHEMA}")
            continue
        if not isinstance(doc.get("methods"), list):
            err("no `methods` list")
            continue
        for i, m in enumerate(doc["methods"]):
            label = f"method #{i + 1}"
            if not isinstance(m, dict):
                err(f"{label} is not an object")
                continue
            if isinstance(m.get("id"), str):
                label = f"method `{m['id']}`"
            line = _line_of(text, f'"{m["id"]}"') if isinstance(m.get("id"), str) else None
            problems = []
            missing = sorted(set(METHOD_KEYS) - set(m))
            extra = sorted(set(m) - set(METHOD_KEYS))
            if missing:
                problems.append("missing key(s) " + ", ".join(missing))
            if extra:
                problems.append("unknown key(s) " + ", ".join(extra))
            for key, typ in METHOD_KEYS.items():
                if key in m and not isinstance(m[key], typ):
                    problems.append(f"`{key}` must be a {typ.__name__}")
            if isinstance(m.get("id"), str) and not ID_RE.match(m["id"]):
                problems.append("`id` must match ^[a-z0-9_]+$")
            for key in ("cases", "tests"):
                v = m.get(key)
                if isinstance(v, list) and (not v or not all(isinstance(x, str) and x for x in v)):
                    problems.append(f"`{key}` must be a non-empty list of strings")
            if isinstance(m.get("quantities"), list) and not all(
                isinstance(x, str) for x in m["quantities"]
            ):
                problems.append("`quantities` must be a list of strings")
            for d in m.get("deviations") if isinstance(m.get("deviations"), list) else []:
                if not (
                    isinstance(d, dict)
                    and set(d) == {"finding", "quantity", "note"}
                    and all(isinstance(d[k], str) for k in d)
                    and FINDING_RE.match(d["finding"])
                ):
                    problems.append(
                        f"invalid deviation {json.dumps(d)} (keys finding ^[A-Z]+-[0-9]+$, quantity, note)"
                    )
            for p in m.get("supersedes_legacy") if isinstance(m.get("supersedes_legacy"), list) else []:
                if not (
                    isinstance(p, str)
                    and re.match(r"^validation/expected/[^/]+\.csv$", p)
                    and (root / p).is_file()
                ):
                    problems.append(f"supersedes_legacy path {p!r} is not an existing validation/expected/*.csv")
            if isinstance(m.get("id"), str):
                if m["id"] in seen_ids:
                    problems.append(f"duplicate method id (also in {seen_ids[m['id']]})")
                else:
                    seen_ids[m["id"]] = rel
            if problems:
                for p in problems:
                    err(f"{label}: {p}", line)
                continue
            methods.append((rel, m, line))
    return methods


def load_cases(root: Path, findings: list) -> dict:
    """{case id: case dict} of every validation/expected/*.json that is a valid schema-1 case."""
    cases = {}
    for path in sorted((root / "validation" / "expected").glob("*.json")):
        rel = path.relative_to(root).as_posix()

        def err(msg, rel=rel):
            findings.append(Finding("ERROR", "case-file", rel, None, msg))

        try:
            doc = json.loads(read_text(path))
        except ValueError as e:
            err(f"not valid JSON: {e}")
            continue
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            err(f"schema must be {SCHEMA} (regenerate with Rscript validation/r/run.R --write {path.stem})")
            continue
        problems = []
        if doc.get("id") != path.stem:
            problems.append(f"id {doc.get('id')!r} differs from the file name")
        for key in ("area", "method", "reference", "call", "dataset", "r_version"):
            if not isinstance(doc.get(key), str):
                problems.append(f"`{key}` must be a string")
        datasets = doc.get("datasets")
        if not isinstance(datasets, list) or not all(isinstance(d, str) for d in datasets):
            problems.append("`datasets` must be a list of paths")
            datasets = []
        if not isinstance(doc.get("packages"), dict):
            problems.append("`packages` must be an object")
        values = doc.get("values")
        if not isinstance(values, dict) or not values:
            problems.append("`values` must be a non-empty object")
        else:
            for k, v in values.items():
                if not _value_ok(v):
                    problems.append(f"value `{k}` is not a number, an array or {{names, values}}")
        dataset = doc.get("dataset")
        if isinstance(dataset, str) and dataset not in datasets:
            problems.append(f"dataset {dataset} is not listed in `datasets`")
        for d in ([dataset] if isinstance(dataset, str) else []) + datasets:
            if not (root / d).is_file():
                problems.append(f"dataset {d} does not exist")
        for p in dict.fromkeys(problems):
            err(p)
        if not problems:
            cases[path.stem] = doc
    return cases


def lint(root: Path, test_lists, strict=False) -> list:
    """All findings (ERROR, WARNING and NOTE) for the tree at `root`. `strict` does not
    change the findings; the caller counts warnings as errors."""
    root = Path(root)
    findings = []
    lists = {}
    for lf in test_lists:
        lf = Path(lf)
        if not lf.name.endswith("-tests.txt") or lf.name == "-tests.txt":
            raise UsageError(f"test list {lf} must be named <package>-tests.txt")
        package = lf.name[: -len("-tests.txt")]
        lists[package] = set(parse_test_list(read_text(lf), package))

    methods = load_registry(root, findings)
    cases = load_cases(root, findings)
    case_files = {p.stem for p in (root / "validation" / "expected").glob("*.json")}

    registered_cases = set()
    unchecked = set()
    for rel, m, line in methods:
        if not (root / m["doc"]).is_file():
            findings.append(Finding("ERROR", "missing-doc", rel, line, f"method `{m['id']}`: doc {m['doc']} does not exist"))
        for cid in m["cases"]:
            registered_cases.add(cid)
            if cid not in case_files:
                findings.append(
                    Finding("ERROR", "case-file", rel, line, f"method `{m['id']}`: case `{cid}` has no validation/expected/{cid}.json")
                )
        for tid in m["tests"]:
            package = tid.split("::", 1)[0]
            if package not in lists:
                unchecked.add(package)
            elif tid not in lists[package]:
                findings.append(
                    Finding("ERROR", "unknown-test", rel, line, f"method `{m['id']}`: test {tid} is not in the {package} test list")
                )
    for package in sorted(unchecked):
        findings.append(Finding("NOTE", "unknown-test", "", None, f"tests of {package} not checked (no --tests list)"))

    for cid, doc in sorted(cases.items()):
        if doc.get("area") != "harness" and cid not in registered_cases:
            findings.append(
                Finding(
                    "ERROR",
                    "orphan-case",
                    f"validation/expected/{cid}.json",
                    None,
                    f"case `{cid}` is listed by no method in validation/methods/*.json",
                )
            )

    # Tests that load cases, and CSV citations in tests that load none.
    registry_tests = {}
    for _, m, _ in methods:
        for cid in m["cases"]:
            registry_tests.setdefault(cid, set()).update(m["tests"])
    packages = {}
    for path in rust_files(root):
        text = read_text(path)
        if "fn test_" not in text:
            continue
        rel = path.relative_to(root).as_posix()
        crate_dir = path.relative_to(root / "crates").parts[0]
        if crate_dir not in packages:
            packages[crate_dir] = crate_package(root, crate_dir)
        package = packages[crate_dir]
        functions, keep, comments = test_functions(text)
        for name, line, start, end in functions:
            loads = [(m.group(1), keep.count("\n", 0, start + m.start()) + 1) for m in _LOAD_RE.finditer(keep, start, end)]
            for cid, load_line in loads:
                if cid not in case_files:
                    findings.append(
                        Finding("ERROR", "case-file", rel, load_line, f"test `{name}` loads case `{cid}`, which has no validation/expected/{cid}.json")
                    )
                    continue
                doc = cases.get(cid)
                if doc is not None and doc.get("area") == "harness":
                    continue
                ok = any(
                    t.startswith(package + "::") and t.rsplit("::", 1)[-1] == name
                    for t in registry_tests.get(cid, ())
                )
                if not ok:
                    findings.append(
                        Finding(
                            "ERROR",
                            "unregistered-load",
                            rel,
                            load_line,
                            f"test `{name}` loads case `{cid}`, but no validation/methods entry lists "
                            f"that case with a {package} test named `{name}`",
                        )
                    )
            if loads:
                continue
            for c_start, c_end in comments:
                if start <= c_start < end and _CSV_CITATION_RE.search(text, c_start, c_end):
                    findings.append(
                        Finding(
                            "WARNING",
                            "csv-citation",
                            rel,
                            text.count("\n", 0, c_start) + 1,
                            f"test `{name}` cites a legacy CSV as its reference values but loads no R case",
                        )
                    )

    # Documentation debt.
    for path in validation_markdown(root):
        rel = path.relative_to(root).as_posix()
        for no, line, in_fence in markdown_lines(read_text(path)):
            if in_fence:
                continue
            for m in _LINK_RE.finditer(line):
                target = m.group(1)
                if target.startswith("#") or _SCHEME_RE.match(target):
                    continue
                target = unquote(target.split("#", 1)[0])
                if not target:
                    continue
                if not (path.parent / target).exists():
                    findings.append(Finding("WARNING", "dead-link", rel, no, f"link target {target} does not exist"))

    live = set()
    for names in lists.values():
        live.update(t.rsplit("::", 1)[-1] for t in names)
    for path in all_rust_files(root):
        live.update(re.findall(r"\bfn\s+(test_\w+)", read_text(path)))
    live_sorted = sorted(live)
    ignore_file = root / "validation" / "tools" / "lint_ignore_names.txt"
    ignored = set()
    if ignore_file.is_file():
        for line in read_text(ignore_file).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                ignored.add(line)

    def is_live(name):
        i = bisect_left(live_sorted, name)
        return i < len(live_sorted) and live_sorted[i].startswith(name)

    docs = [p for p in (root / "README.md", root / "CLAUDE.md") if p.is_file()]
    for path in validation_markdown(root) + docs:
        rel = path.relative_to(root).as_posix()
        for no, col, name in test_name_tokens(read_text(path)):
            if name in ignored or is_live(name):
                continue
            findings.append(
                Finding("WARNING", "dead-test-name", rel, no, f"`{name}` (column {col}) matches no test, neither exactly nor as a prefix")
            )

    ds_readme = root / "validation" / "datasets" / "README.md"
    if ds_readme.is_file():
        for no, line in enumerate(read_text(ds_readme).splitlines(), start=1):
            for m in _DATASET_RE.finditer(line):
                if not (root / "validation" / "datasets" / m.group(1)).exists():
                    findings.append(
                        Finding("WARNING", "missing-dataset", "validation/datasets/README.md", no, f"dataset {m.group(1)} is not in validation/datasets/")
                    )
    return findings


def cmd_lint(args) -> int:
    findings = lint(Path(args.root), args.tests or [], args.strict)
    order = {"NOTE": 0, "ERROR": 1, "WARNING": 2}
    findings.sort(key=lambda f: (order[f.level], f.path, f.line or 0, f.rule, f.message))
    errors = warnings = 0
    for f in findings:
        if f.level == "NOTE":
            print(f"note: {f.message}")
            continue
        where = f.path + (f":{f.line}" if f.line else "")
        print(f"{f.level} [{f.rule}] {where}: {f.message}")
        if f.level == "ERROR":
            errors += 1
        else:
            warnings += 1
    summary = f"lint: {errors} error(s), {warnings} warning(s)"
    if args.strict:
        summary += " (--strict: warnings count as errors)"
    print(summary)
    return 1 if errors or (args.strict and warnings) else 0


# --------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------


def _cell(text) -> str:
    return str(text).replace("\n", " ").replace("|", "\\|")


def _code(text) -> str:
    return "`" + str(text).replace("`", "'") + "`"


def _counts(statuses) -> tuple:
    statuses = list(statuses)
    return tuple(statuses.count(s) for s in ("ok", "failed", "ignored"))


def render(results: dict, root: Path) -> str:
    root = Path(root)
    tests = results.get("tests", {}) or {}
    binaries = {k: set(v) for k, v in (results.get("binaries") or {}).items()}
    packages = results.get("packages") or {}
    out = [
        "<!-- Generated by validation/tools/validation_status.py render; do not edit. "
        "Regenerate with: validation/run_validation.sh --write-status -->",
        "# Validation Status",
        "",
    ]
    head = f"Commit {_code(results.get('commit', 'unknown'))}, {results.get('date', 'unknown date')}"
    if results.get("command"):
        head += f", command {_code(results['command'])}"
    if results.get("category"):
        head += f", category {_code(results['category'])}"
    out.append(head + ".")
    features = results.get("features") or {}
    if features:
        out.append("Feature sets: " + ", ".join(_code(f"{p} {f}".strip()) for p, f in sorted(features.items())) + ".")
    else:
        out.append("Feature sets: not recorded.")
    out.append("")

    # Summary.
    out += ["## Summary", "", "| Package | Category | Passed | Failed | Ignored |", "|---|---|---:|---:|---:|"]
    groups = {}
    for tid, status in tests.items():
        package = tid.split("::", 1)[0]
        if package == "p2a-validation" or not is_validate(tid):
            continue
        cat = category_of(tid, binaries.get(package, set()))
        groups.setdefault((package, cat), []).append(status)
    total = [0, 0, 0]
    for (package, cat), statuses in sorted(groups.items()):
        p, f, i = _counts(statuses)
        total = [total[0] + p, total[1] + f, total[2] + i]
        out.append(f"| {package} | {cat} | {p} | {f} | {i} |")
    out.append(f"| **Total** | | **{total[0]}** | **{total[1]}** | **{total[2]}** |")
    out.append("")
    hp, hf, _ = _counts(s for t, s in tests.items() if t.startswith("p2a-validation::"))
    out.append(
        "`test_validate` tests only; the three `data::database` read-only SQL guards are not "
        "counted (they compile only with the `database` feature). "
        f"Harness self-tests (p2a-validation): {hp} passed, {hf} failed."
    )
    if packages:
        runs = []
        for name, pkg in sorted(packages.items()):
            run = f"{_code(name)} {pkg.get('status')}"
            if pkg.get("errors"):
                run += " (" + "; ".join(_cell(e) for e in pkg["errors"][:3]) + ")"
            runs.append(run)
        out.append("Package runs: " + ", ".join(runs) + ".")
    out.append("")

    # Methods.
    findings = []
    methods = load_registry(root, findings)
    r_check = results.get("r_check") or {}
    r_cases = r_check.get("cases", {}) or {}
    out += ["## Methods pinned to R", ""]
    registered_tests = set()
    if not methods:
        out.append("No method is registered yet (`validation/methods/*.json`).")
    else:
        out += [
            "| Method | Reference | Cases | Quantities | Tolerance | Tests | Status |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, m, _ in sorted(methods, key=lambda x: x[1]["id"]):
            registered_tests.update(m["tests"])
            statuses = [tests.get(t) for t in m["tests"]]
            if "failed" in statuses or any(r_cases.get(c) == "failed" for c in m["cases"]):
                status = "FAIL"
            elif any(s in (None, "ignored") for s in statuses):
                status = "NOT RUN"
            elif m["deviations"]:
                status = "PASS (known deviations)"
            else:
                status = "PASS"
            out.append(
                "| "
                + " | ".join(
                    [
                        f"{_cell(m['name'])} ({_code(m['id'])}; {_code(m['doc'])})",
                        _cell(m["reference"]),
                        ", ".join(_code(c) for c in m["cases"]),
                        _cell(", ".join(m["quantities"])),
                        _cell(m["tolerance"]),
                        "<br>".join(_code(t) for t in m["tests"]),
                        status,
                    ]
                )
                + " |"
            )
        out += ["", "### Known deviations", ""]
        deviations = [(m["id"], d) for _, m, _ in sorted(methods, key=lambda x: x[1]["id"]) for d in m["deviations"]]
        if deviations:
            for mid, d in deviations:
                out.append(f"- {_code(mid)}: {d['finding']}, {_cell(d['quantity'])}: {_cell(d['note'])}")
        else:
            out.append("None.")
    out.append("")

    # R reference check.
    out += ["## R reference check", ""]
    st = r_check.get("status")
    ok = sum(1 for s in r_cases.values() if s == "ok")
    if st == "pass":
        out.append(f"PASS: {ok} case(s) match the committed files (`Rscript validation/r/run.R --check`).")
    elif st == "fail":
        out.append(f"FAIL: {sum(1 for s in r_cases.values() if s == 'failed')} case(s) failed, {ok} matched.")
        out += [""] + [f"- {_code(x)}" for x in r_check.get("failures", [])]
    elif st == "error":
        out.append("ERROR: the R check did not complete.")
    elif st == "missing":
        out.append("R not found: the R check did not run.")
    else:
        out.append("NOT RUN.")
    if r_check.get("missing_packages"):
        out.append("Missing R packages: " + ", ".join(_code(p) for p in r_check["missing_packages"]) + ".")
    versions, pkgs = set(), {}
    for path in sorted((root / "validation" / "expected").glob("*.json")):
        try:
            doc = json.loads(read_text(path))
        except (ValueError, UsageError):
            continue
        if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
            continue
        if isinstance(doc.get("r_version"), str):
            versions.add(doc["r_version"])
        for p, v in (doc.get("packages") or {}).items():
            pkgs.setdefault(p, set()).add(str(v))
    out.append("")
    if versions:
        line = "Case files written with " + "; ".join(sorted(versions))
        if pkgs:
            line += "; packages: " + ", ".join(f"{_code(p)} {', '.join(sorted(v))}" for p, v in sorted(pkgs.items()))
        out.append(line + ".")
    else:
        out.append("No case file in `validation/expected/`.")
    out.append("")

    # Other test_validate tests.
    out += [
        "## Other `test_validate_*` tests",
        "",
        "Not registered as R-pinned (property, DGP-recovery or quoted-value checks).",
        "",
        "| Package | Module | Passed | Failed | Ignored |",
        "|---|---|---:|---:|---:|",
    ]
    other = {}
    for tid, status in tests.items():
        package, _, rest = tid.partition("::")
        if package == "p2a-validation" or not is_validate(tid) or tid in registered_tests:
            continue
        segments = rest.split("::")
        if segments[0] in binaries.get(package, set()):
            module = f"tests/{segments[0]}.rs"
        else:
            module = "::".join(segments[:-1][:2]) or "(crate root)"
        other.setdefault((package, module), []).append(status)
    for (package, module), statuses in sorted(other.items()):
        p, f, i = _counts(statuses)
        out.append(f"| {package} | {_code(module)} | {p} | {f} | {i} |")
    out.append("")

    # Legacy reference files.
    out += ["## Legacy reference files", "", "| File | Asserted by | Legacy run |", "|---|---|---|"]
    superseded = {}
    for _, m, _ in methods:
        for p in m["supersedes_legacy"]:
            superseded.setdefault(p, []).extend(m["cases"])
    legacy = results.get("legacy_r")
    files = (legacy or {}).get("files", {}) or {}
    run_text = {"same": "same", "differs": "differs", "not_produced": "not produced"}
    for path in sorted((root / "validation" / "expected").glob("*.csv")):
        rel = f"validation/expected/{path.name}"
        if rel in superseded:
            asserted = "superseded by " + ", ".join(_code(c) for c in dict.fromkeys(superseded[rel]))
        else:
            asserted = "not asserted by any test"
        if legacy is None:
            run = "not run"
        else:
            run = run_text.get(files.get(path.name), "not run")
            detail = (legacy.get("details") or {}).get(path.name)
            if detail and files.get(path.name) == "differs":
                run += f" ({_cell(detail)})"
        out.append(f"| {_code(rel)} | {asserted} | {run} |")
    out.append("")
    if legacy is None:
        out.append("Legacy R scripts were not run for this report.")
    else:
        s, _ = legacy_counts(legacy)
        untracked = [n for n, v in files.items() if v == "untracked"]
        out.append(
            f"Legacy R scripts (`validation/scripts/*.R`, run with `--legacy-r` in scratch directories; "
            f"timeout {legacy.get('timeout_seconds')} s): {s['ran']} ran, {s['error']} error, {s['timeout']} timeout."
        )
        if untracked:
            out.append("Produced but not tracked: " + ", ".join(_code(n) for n in sorted(untracked)) + ".")
        out += ["", "| Script | Status | Seconds |", "|---|---|---:|"]
        for name, info in sorted((legacy.get("scripts") or {}).items()):
            secs = info.get("seconds")
            out.append(f"| {_code(name)} | {info.get('status')} | {'' if secs is None else secs} |")
    return "\n".join(out) + "\n"


def cmd_render(args) -> int:
    results = read_json(args.results)
    if not isinstance(results, dict) or results.get("schema") != SCHEMA:
        raise UsageError(f"{args.results}: not a schema-{SCHEMA} results file")
    text = render(results, Path(args.root))
    for out in args.output:
        Path(out).write_text(text, encoding="utf-8")
    return 0


def cmd_select(args) -> int:
    for target, name in select_tests(read_text(args.list_file), args.package, args.category, args.all):
        print(f"{target}\t{name}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="validation_status.py", description=__doc__.split("\n\n")[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    def root_arg(p):
        p.add_argument("--root", default=str(DEFAULT_ROOT), help="repository root (default: %(default)s)")

    p = sub.add_parser("select", help="pick tests from cargo --list output")
    p.add_argument("--package", required=True)
    p.add_argument("--category")
    p.add_argument("--all", action="store_true")
    root_arg(p)
    p.add_argument("list_file")
    p.set_defaults(func=cmd_select)

    p = sub.add_parser("collect", help="cargo logs, R check and legacy runs -> results.json")
    p.add_argument("--package")
    p.add_argument("--exit-code", type=int)
    p.add_argument("--selected", type=int)
    p.add_argument("--features")
    p.add_argument("--command")
    p.add_argument("--category")
    p.add_argument("--r-check")
    p.add_argument("--legacy")
    root_arg(p)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("logs", nargs="*")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("lint", help="check the methods registry and the docs")
    p.add_argument("--tests", nargs="+", action="extend", metavar="LIST_FILE")
    p.add_argument("--strict", action="store_true")
    root_arg(p)
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("render", help="results.json -> status page")
    p.add_argument("--results", required=True)
    p.add_argument("-o", "--output", required=True, action="append")
    root_arg(p)
    p.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except UsageError as e:
        print(f"validation_status.py {args.cmd}: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
