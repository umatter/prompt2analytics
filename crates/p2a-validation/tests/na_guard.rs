//! NA guard: product code must not read user columns in a way that ignores missing values or
//! refuses integer columns.
//!
//! Two polars calls do that on data loaded from a CSV file:
//! - `ChunkedArray::into_no_null_iter()` ignores the validity bitmap, so a null cell (an empty
//!   field) is read as whatever the buffer holds behind it, 0.0 in practice;
//! - `Column::f64()` without a cast refuses `Int64`, the dtype polars infers for a column of
//!   whole numbers (counts, scores, 0/1 events, integer times).
//!
//! The guard walks `crates/<crate>/src/**/*.rs` of the five product crates
//! ([`PRODUCT_CRATES`]) and reports each non-test line that calls `into_no_null_iter(`, or that
//! calls `.f64()` when none of the line and the [`CAST_LOOKBACK`] lines above it calls `cast`,
//! `strict_cast` or `cast_with_options` (matched as calls, so `forecast(` and `broadcast(` are
//! no casts). Comments, string literals and char literals are ignored. Test code is skipped
//! item by item: every `#[cfg(test)]` or `#[cfg(all(test, ..))]` item, and the files of a
//! test-only module declared out of line (`#[cfg(test)] mod tests;`). Code after a test module
//! is scanned, because some files keep product code below theirs.
//!
//! A site the guard reports is fixed by reading the column through `p2a_core::data::columns`
//! (in p2a-mcp: `tools::common::numeric_column(s)` / `numeric_matrix` with an explicit
//! `NaPolicy`); CLAUDE.md, "Reading user columns", explains the rules. Two markers, written as
//! a non-doc `//` line comment on the reported line or the line above, exempt a site:
//! - `// na-ok: <reason>` for data that cannot hold nulls (a column the code built itself, a
//!   comparison mask);
//! - `// na-legacy(T3): <reason>` or `// na-legacy(T4): <reason>` for a known site that a later
//!   tier migrates (the reason says what the site gets wrong, after its finding id if it has one).
//!
//! A comment that starts with `na-ok` or `na-legacy` in any other form is reported as a
//! malformed marker, and an `na-legacy` marker with no extraction on its line or the next as a
//! stale one. The number of `na-legacy` markers may only fall ([`MAX_LEGACY_MARKERS`]).
//!
//! The rules are checked on fixture trees in temporary directories; `test_na_guard_tree_is_clean`
//! runs them on the repository.

use std::fs;
use std::path::{Path, PathBuf};

/// The crates whose `src` trees are scanned, in [`Scan::files_per_crate`] order.
const PRODUCT_CRATES: [&str; 5] = ["p2a-core", "p2a-mcp", "p2a-cli", "p2a-chat", "p2a-dioxus"];
/// Lines above a `.f64()` call that may hold the cast it reads.
const CAST_LOOKBACK: usize = 6;
/// The `na-legacy` markers placed on the carry-over sites. Lower it when a package migrates a
/// site and removes its marker; never raise it.
const MAX_LEGACY_MARKERS: usize = 18;
/// The removed p2a-mcp helpers that read a null as NaN.
const LEGACY_HELPERS: [&str; 2] = ["extract_column_f64", "extract_numeric_matrix"];
/// Identifiers that count as a cast call before `.f64()`.
const CAST_CALLS: [&str; 3] = ["cast", "strict_cast", "cast_with_options"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    IntoNoNullIter,
    F64WithoutCast,
    MalformedMarker,
    StaleLegacyMarker,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::IntoNoNullIter => "into_no_null_iter",
            Kind::F64WithoutCast => "f64-without-cast",
            Kind::MalformedMarker => "malformed-marker",
            Kind::StaleLegacyMarker => "stale-na-legacy-marker",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Violation {
    /// Repository-relative path with `/` separators.
    file: String,
    /// 1-based line.
    line: usize,
    kind: Kind,
    /// The trimmed source line.
    code: String,
}

#[derive(Debug, Default)]
struct Scan {
    /// Every `.rs` file walked, skipped test-only files included.
    files: usize,
    files_per_crate: [usize; 5],
    /// `#[cfg(test)]` and `#[cfg(all(test, ..))]` items found.
    test_items: usize,
    violations: Vec<Violation>,
    /// (file, 1-based line) of every well-formed `na-legacy` marker.
    legacy_markers: Vec<(String, usize)>,
    /// Well-formed `na-ok` markers.
    ok_markers: usize,
}

/// One source file: its path relative to the root, its lines, and the lexer's view of them.
struct Source {
    rel: String,
    lines: Vec<String>,
    code: Vec<String>,
    comments: Vec<Option<String>>,
    is_test: Vec<bool>,
}

fn is_ident(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// The `.rs` files under `dir`, recursively, sorted by path.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn relative(root: &Path, path: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    rel.components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect::<Vec<_>>()
        .join("/")
}

/// Code text per source line, with every character of a comment, string literal or char
/// literal replaced by a space (line count and columns kept), and the text after `//` of each
/// line's non-doc line comment (`None` without one; `///` and `//!` comments are not recorded).
fn code_lines(src: &str) -> (Vec<String>, Vec<Option<String>>) {
    #[derive(Clone, Copy, PartialEq)]
    enum State {
        Code,
        LineComment { record: bool },
        BlockComment(usize),
        Str,
        RawStr(usize),
    }
    let chars: Vec<char> = src.chars().collect();
    let at = |i: usize| chars.get(i).copied();
    let mut code = Vec::new();
    let mut comments = Vec::new();
    let mut line = String::new();
    let mut comment: Option<String> = None;
    let mut state = State::Code;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '\n' {
            code.push(std::mem::take(&mut line));
            comments.push(comment.take());
            if matches!(state, State::LineComment { .. }) {
                state = State::Code;
            }
            i += 1;
            continue;
        }
        match state {
            State::Code => {
                let prev_ident = i > 0 && is_ident(chars[i - 1]);
                if c == '/' && at(i + 1) == Some('/') {
                    let doc = (at(i + 2) == Some('/') && at(i + 3) != Some('/'))
                        || at(i + 2) == Some('!');
                    state = State::LineComment { record: !doc };
                    if !doc {
                        comment = Some(String::new());
                    }
                    line.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '/' && at(i + 1) == Some('*') {
                    state = State::BlockComment(1);
                    line.push_str("  ");
                    i += 2;
                    continue;
                }
                if c == '"' {
                    state = State::Str;
                    line.push(' ');
                    i += 1;
                    continue;
                }
                if c == 'b' && at(i + 1) == Some('"') && !prev_ident {
                    state = State::Str;
                    line.push_str("  ");
                    i += 2;
                    continue;
                }
                // r"..", r#".."#, br"..", br#".."#
                let raw_start = match (c, at(i + 1)) {
                    ('r', _) if !prev_ident => Some(i + 1),
                    ('b', Some('r')) if !prev_ident => Some(i + 2),
                    _ => None,
                };
                if let Some(mut j) = raw_start {
                    let mut hashes = 0;
                    while at(j) == Some('#') {
                        hashes += 1;
                        j += 1;
                    }
                    if at(j) == Some('"') {
                        for _ in i..=j {
                            line.push(' ');
                        }
                        state = State::RawStr(hashes);
                        i = j + 1;
                        continue;
                    }
                }
                if c == '\'' {
                    // A char literal: one character, or an escape, then a closing quote. Any
                    // other quote starts a lifetime or a label and stays code.
                    let end = match at(i + 1) {
                        Some('\\') => (i + 3..(i + 14).min(chars.len()))
                            .take_while(|&k| chars[k] != '\n')
                            .find(|&k| chars[k] == '\''),
                        Some(ch) if ch != '\n' && at(i + 2) == Some('\'') => Some(i + 2),
                        _ => None,
                    };
                    if let Some(end) = end {
                        for _ in i..=end {
                            line.push(' ');
                        }
                        i = end + 1;
                        continue;
                    }
                }
                line.push(c);
                i += 1;
            }
            State::LineComment { record } => {
                if record && let Some(text) = comment.as_mut() {
                    text.push(c);
                }
                line.push(' ');
                i += 1;
            }
            State::BlockComment(depth) => {
                if c == '/' && at(i + 1) == Some('*') {
                    state = State::BlockComment(depth + 1);
                    line.push_str("  ");
                    i += 2;
                } else if c == '*' && at(i + 1) == Some('/') {
                    state = if depth == 1 {
                        State::Code
                    } else {
                        State::BlockComment(depth - 1)
                    };
                    line.push_str("  ");
                    i += 2;
                } else {
                    line.push(' ');
                    i += 1;
                }
            }
            State::Str => {
                if c == '\\' {
                    line.push(' ');
                    i += 1;
                    match at(i) {
                        Some('\n') | None => {}
                        Some(_) => {
                            line.push(' ');
                            i += 1;
                        }
                    }
                } else {
                    if c == '"' {
                        state = State::Code;
                    }
                    line.push(' ');
                    i += 1;
                }
            }
            State::RawStr(hashes) => {
                if c == '"' && (1..=hashes).all(|k| at(i + k) == Some('#')) {
                    for _ in 0..=hashes {
                        line.push(' ');
                    }
                    state = State::Code;
                    i += hashes + 1;
                } else {
                    line.push(' ');
                    i += 1;
                }
            }
        }
    }
    if !src.is_empty() && !src.ends_with('\n') {
        code.push(line);
        comments.push(comment);
    }
    (code, comments)
}

/// Whether `code` (already trimmed) starts a test-only attribute: `#[cfg(test)]`, or
/// `#[cfg(all(` followed by `test` and then `,` or `)`. `#[cfg(any(test, ..))]` is product code.
fn is_test_attribute(code: &str) -> bool {
    if code.starts_with("#[cfg(test)]") {
        return true;
    }
    let Some(rest) = code.strip_prefix("#[cfg(all(") else {
        return false;
    };
    let Some(rest) = rest.trim_start().strip_prefix("test") else {
        return false;
    };
    matches!(rest.trim_start().chars().next(), Some(',') | Some(')'))
}

/// The module name of an out-of-line module declaration `[pub[(..)] ]mod <ident>;`.
fn out_of_line_module(item: &str) -> Option<String> {
    let mut rest = item.trim();
    if let Some(after) = rest.strip_prefix("pub") {
        rest = after.trim_start();
        if let Some(after) = rest.strip_prefix('(') {
            rest = after.split_once(')')?.1.trim_start();
        }
    }
    let rest = rest.strip_prefix("mod")?;
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let rest = rest.trim_start();
    let name: String = rest.chars().take_while(|&c| is_ident(c)).collect();
    if name.is_empty() || rest[name.len()..].trim() != ";" {
        return None;
    }
    Some(name)
}

/// Lines that belong to `#[cfg(test)]` / `#[cfg(all(test, ..))]` items, the names of the
/// test-only modules those items declare out of line, and the number of items.
fn test_lines(code: &[String]) -> (Vec<bool>, Vec<String>, usize) {
    let mut is_test = vec![false; code.len()];
    let mut modules = Vec::new();
    let mut items = 0;
    let mut i = 0;
    while i < code.len() {
        let trimmed = code[i].trim_start();
        if !is_test_attribute(trimmed) {
            i += 1;
            continue;
        }
        // The attribute ends at the `]` that closes its `#[`.
        let indent = code[i].len() - trimmed.len();
        let mut depth = 0usize;
        let mut close = None;
        for (k, ch) in code[i].char_indices().skip_while(|&(k, _)| k < indent) {
            match ch {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 {
                        close = Some(k + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        // The item starts after the attribute on its line, else at the next line that is
        // neither empty nor an attribute.
        let (mut line, mut col) = match close {
            Some(c) if !code[i][c..].trim().is_empty() => (i, c),
            _ => {
                let mut j = i + 1;
                while j < code.len() && {
                    let t = code[j].trim();
                    t.is_empty() || t.starts_with("#[")
                } {
                    j += 1;
                }
                (j, 0)
            }
        };
        // The item ends at the first `;` at depth 0 before any `{`, or at the `}` that brings
        // the depth back to 0. A `;` inside brackets or parentheses (`[u8; 4]`) does not end it.
        let start = (line, col);
        let mut depth = 0usize;
        let mut nested = 0usize;
        let mut opened = false;
        let mut end = None;
        'outer: while line < code.len() {
            for (k, ch) in code[line].char_indices().skip_while(|&(k, _)| k < col) {
                match ch {
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    '}' => {
                        depth = depth.saturating_sub(1);
                        if opened && depth == 0 {
                            end = Some((line, k));
                            break 'outer;
                        }
                    }
                    '(' | '[' => nested += 1,
                    ')' | ']' => nested = nested.saturating_sub(1),
                    ';' if !opened && depth == 0 && nested == 0 => {
                        end = Some((line, k));
                        break 'outer;
                    }
                    _ => {}
                }
            }
            line += 1;
            col = 0;
        }
        let (end_line, end_col) = end.unwrap_or((code.len().saturating_sub(1), 0));
        if start.0 == end_line
            && let Some(name) = out_of_line_module(&code[end_line][start.1..=end_col])
        {
            modules.push(name);
        }
        for flag in is_test.iter_mut().take(end_line + 1).skip(i) {
            *flag = true;
        }
        items += 1;
        i = end_line + 1;
    }
    (is_test, modules, items)
}

/// Whether `code` calls one of [`CAST_CALLS`]: the identifier with no identifier character
/// before it, then optional spaces and `(`.
fn has_cast_call(code: &str) -> bool {
    CAST_CALLS
        .iter()
        .any(|name| identifier_at(code, name, |rest| rest.trim_start().starts_with('(')))
}

/// Whether `name` occurs in `code` as a whole identifier for which `after` (the text that
/// follows it) holds.
fn identifier_at(code: &str, name: &str, after: impl Fn(&str) -> bool) -> bool {
    code.match_indices(name).any(|(k, _)| {
        let before = code[..k].chars().next_back();
        let rest = &code[k + name.len()..];
        !before.is_some_and(is_ident) && !rest.chars().next().is_some_and(is_ident) && after(rest)
    })
}

/// A recorded comment as a marker: `Some(Ok(true))` for `na-ok: <reason>`, `Some(Ok(false))`
/// for `na-legacy(T3|T4): <reason>`, `Some(Err(()))` for a malformed one, `None` for no marker.
fn marker(comment: &str) -> Option<Result<bool, ()>> {
    let text = comment.trim_start();
    let reason = |rest: Option<&str>| rest.is_some_and(|r| !r.trim().is_empty());
    if text.starts_with("na-ok") {
        Some(if reason(text.strip_prefix("na-ok:")) {
            Ok(true)
        } else {
            Err(())
        })
    } else if text.starts_with("na-legacy") {
        let rest = text
            .strip_prefix("na-legacy(T3):")
            .or_else(|| text.strip_prefix("na-legacy(T4):"));
        Some(if reason(rest) { Ok(false) } else { Err(()) })
    } else {
        None
    }
}

/// Read and lex every `.rs` file of the product crates' `src` trees, flagging the lines of
/// test items and dropping the files of out-of-line test modules (still counted in `files`).
fn sources(root: &Path, scan: &mut Scan) -> Vec<Source> {
    let mut all = Vec::new();
    let mut skip: Vec<PathBuf> = Vec::new();
    for (k, krate) in PRODUCT_CRATES.iter().enumerate() {
        let mut files = Vec::new();
        rust_files(&root.join("crates").join(krate).join("src"), &mut files);
        scan.files_per_crate[k] = files.len();
        scan.files += files.len();
        for path in files {
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {} as UTF-8: {e}", path.display()));
            let (code, comments) = code_lines(&text);
            let (is_test, modules, items) = test_lines(&code);
            scan.test_items += items;
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let dir = if matches!(file_name, "mod.rs" | "lib.rs" | "main.rs") {
                path.parent().map(Path::to_path_buf)
            } else {
                path.parent()
                    .zip(path.file_stem())
                    .map(|(parent, stem)| parent.join(stem))
            };
            if let Some(dir) = dir {
                for name in modules {
                    skip.push(dir.join(format!("{name}.rs")));
                    skip.push(dir.join(name));
                }
            }
            let lines = text.lines().map(str::to_string).collect();
            all.push((
                path.clone(),
                Source {
                    rel: relative(root, &path),
                    lines,
                    code,
                    comments,
                    is_test,
                },
            ));
        }
    }
    all.into_iter()
        .filter(|(path, _)| !skip.iter().any(|s| path == s || path.starts_with(s)))
        .map(|(_, source)| source)
        .collect()
}

/// Scan the five product crates under `root`. A crate directory that does not exist counts
/// 0 files.
fn scan(root: &Path) -> Scan {
    let mut scan = Scan::default();
    for src in sources(root, &mut scan) {
        let n = src.code.len();
        let code_at = |i: usize| src.code.get(i).map_or("", String::as_str);
        let source_at = |i: usize| src.lines.get(i).map_or("", |l| l.trim()).to_string();
        // Markers of the non-test lines: Some(true) valid, Some(false) malformed.
        let mut valid = vec![false; n];
        for i in 0..n {
            if src.is_test[i] {
                continue;
            }
            let Some(comment) = src.comments[i].as_deref() else {
                continue;
            };
            match marker(comment) {
                None => {}
                Some(Err(())) => scan.violations.push(Violation {
                    file: src.rel.clone(),
                    line: i + 1,
                    kind: Kind::MalformedMarker,
                    code: source_at(i),
                }),
                Some(Ok(ok)) => {
                    valid[i] = true;
                    if ok {
                        scan.ok_markers += 1;
                    } else {
                        scan.legacy_markers.push((src.rel.clone(), i + 1));
                        let extracts =
                            |c: &str| c.contains(".f64()") || c.contains("into_no_null_iter(");
                        if !extracts(code_at(i)) && !extracts(code_at(i + 1)) {
                            scan.violations.push(Violation {
                                file: src.rel.clone(),
                                line: i + 1,
                                kind: Kind::StaleLegacyMarker,
                                code: source_at(i),
                            });
                        }
                    }
                }
            }
        }
        for i in 0..n {
            if src.is_test[i] {
                continue;
            }
            let exempt = valid[i] || (i > 0 && valid[i - 1]);
            let code = code_at(i);
            let mut flag = |kind| {
                if !exempt {
                    scan.violations.push(Violation {
                        file: src.rel.clone(),
                        line: i + 1,
                        kind,
                        code: source_at(i),
                    });
                }
            };
            if code.contains("into_no_null_iter(") {
                flag(Kind::IntoNoNullIter);
            }
            if code.contains(".f64()")
                && !(i.saturating_sub(CAST_LOOKBACK)..=i).any(|k| has_cast_call(code_at(k)))
            {
                flag(Kind::F64WithoutCast);
            }
        }
    }
    scan.violations
        .sort_by(|a, b| (&a.file, a.line, a.kind).cmp(&(&b.file, b.line, b.kind)));
    scan
}

/// Non-test code lines of the product crates that name one of [`LEGACY_HELPERS`] as a whole
/// identifier: (file, 1-based line, name).
fn legacy_helper_uses(root: &Path) -> Vec<(String, usize, &'static str)> {
    let mut hits = Vec::new();
    for src in sources(root, &mut Scan::default()) {
        for (i, code) in src.code.iter().enumerate() {
            if src.is_test[i] {
                continue;
            }
            for name in LEGACY_HELPERS {
                if identifier_at(code, name, |_| true) {
                    hits.push((src.rel.clone(), i + 1, name));
                }
            }
        }
    }
    hits
}

/// The violations as `file:line: label: code` lines under a header, then how to fix them.
fn report(v: &[Violation]) -> String {
    let mut out = format!(
        "NA guard: {} violation(s) in product code (CLAUDE.md, \"Reading user columns\")\n",
        v.len()
    );
    for x in v {
        out.push_str(&format!(
            "  {}:{}: {}: {}\n",
            x.file,
            x.line,
            x.kind.label(),
            x.code
        ));
    }
    out.push_str(
        "Read the column through p2a_core::data::columns (p2a-mcp: \
         tools::common::numeric_column(s)/numeric_matrix with a NaPolicy); if the data cannot \
         hold nulls, add `// na-ok: <reason>` on the line or the line above.\n",
    );
    out
}

// -------------------------------------------------------------------------------------------
// Fixture trees
// -------------------------------------------------------------------------------------------

/// A temporary tree, removed when dropped.
struct Fixture(PathBuf);

impl Fixture {
    fn new(tag: &str) -> Fixture {
        let dir = std::env::temp_dir().join(format!("p2a-na-guard-{}-{tag}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        Fixture(dir)
    }

    fn file(&self, rel: &str, src: &str) -> &Fixture {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, src).unwrap();
        self
    }

    fn scan(&self) -> Scan {
        scan(&self.0)
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The 1-based line of the one line of `src` that contains `needle`.
fn line_of(src: &str, needle: &str) -> usize {
    let hits: Vec<usize> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(needle))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(hits.len(), 1, "`{needle}` is on lines {hits:?}");
    hits[0]
}

/// The violations in `file`, sorted.
fn found(s: &Scan, file: &str) -> Vec<(usize, Kind)> {
    let mut v: Vec<(usize, Kind)> = s
        .violations
        .iter()
        .filter(|v| v.file == file)
        .map(|v| (v.line, v.kind))
        .collect();
    v.sort();
    v
}

#[test]
fn test_na_guard_flags_into_no_null_iter_and_bare_f64() {
    let src = "use polars::prelude::*;\n\
               pub fn bad_iter(ca: &Float64Chunked) -> Vec<f64> {\n    \
                   ca.into_no_null_iter().collect()\n\
               }\n\
               pub fn bad_f64(c: &Column) -> PolarsResult<f64> {\n    \
                   let ca = c.f64()?;\n    \
                   Ok(ca.sum().unwrap_or(0.0))\n\
               }\n";
    let fx = Fixture::new("flags");
    fx.file("crates/p2a-core/src/a.rs", src);
    let s = fx.scan();
    assert_eq!(
        found(&s, "crates/p2a-core/src/a.rs"),
        [
            (line_of(src, "into_no_null_iter"), Kind::IntoNoNullIter),
            (line_of(src, "c.f64()"), Kind::F64WithoutCast)
        ]
    );
    assert_eq!(line_of(src, "into_no_null_iter"), 3);
    assert_eq!(line_of(src, "c.f64()"), 6);
    let r = report(&s.violations);
    assert!(
        r.contains(
            "crates/p2a-core/src/a.rs:3: into_no_null_iter: ca.into_no_null_iter().collect()"
        ),
        "{r}"
    );
    assert!(
        r.contains("crates/p2a-core/src/a.rs:6: f64-without-cast: let ca = c.f64()?;"),
        "{r}"
    );
}

#[test]
fn test_na_guard_cast_must_be_a_cast_call_within_six_lines() {
    let filler = |k: usize| -> String { (0..k).map(|j| format!("    let _a{j} = 1;\n")).collect() };
    let fx = Fixture::new("cast");
    let c1 = format!(
        "pub fn f(c: &Column) -> PolarsResult<()> {{\n    let s = c.cast(&DataType::Float64)?;\n{}    let _six = s.f64()?;\n    Ok(())\n}}\n",
        filler(5)
    );
    let c2 = format!(
        "pub fn f(c: &Column) -> PolarsResult<()> {{\n    let s = c.strict_cast(&DataType::Float64)?;\n{}    let _seven = s.f64()?;\n    Ok(())\n}}\n",
        filler(6)
    );
    let c3 = "pub fn f(c: &Column) -> PolarsResult<()> {\n    \
              let _same = c.strict_cast(&DataType::Float64)?.f64()?.sum();\n    Ok(())\n}\n";
    let c4 = "pub fn f(c: &Column, x: &[f64]) -> PolarsResult<()> {\n    \
              let p = arima_forecast(&x);\n    let _fc = c.f64()?;\n    Ok(())\n}\n";
    let c5 = "pub fn f(c: &Column, a: &Array1<f64>) -> PolarsResult<()> {\n    \
              let b = a.broadcast((2, 2));\n    let _bc = c.f64()?;\n    Ok(())\n}\n";
    fx.file("crates/p2a-core/src/c1.rs", &c1)
        .file("crates/p2a-core/src/c2.rs", &c2)
        .file("crates/p2a-core/src/c3.rs", c3)
        .file("crates/p2a-core/src/c4.rs", c4)
        .file("crates/p2a-core/src/c5.rs", c5);
    let s = fx.scan();
    let all: Vec<(String, usize, Kind)> = s
        .violations
        .iter()
        .map(|v| (v.file.clone(), v.line, v.kind))
        .collect();
    assert_eq!(
        all,
        [
            (
                "crates/p2a-core/src/c2.rs".to_string(),
                line_of(&c2, "_seven"),
                Kind::F64WithoutCast
            ),
            (
                "crates/p2a-core/src/c4.rs".to_string(),
                line_of(c4, "_fc"),
                Kind::F64WithoutCast
            ),
            (
                "crates/p2a-core/src/c5.rs".to_string(),
                line_of(c5, "_bc"),
                Kind::F64WithoutCast
            ),
        ]
    );
}

#[test]
fn test_na_guard_markers_exempt_their_line_and_the_next() {
    let src = "pub fn f(c: &Column, m: &BooleanChunked) -> PolarsResult<()> {\n    \
               let _x = c.f64()?; // na-ok: the column was built two lines up\n    \
               // na-ok: mask from is_null(), never null\n    \
               let _y: Vec<bool> = m.into_no_null_iter().collect();\n    \
               // na-legacy(T4): per-column read\n    \
               let _z = c.f64()?;\n    \
               // na-ok: two lines above does not count\n\
               \n    \
               let _w = c.f64()?;\n    \
               Ok(())\n\
               }\n";
    let fx = Fixture::new("markers");
    fx.file("crates/p2a-core/src/m.rs", src);
    let s = fx.scan();
    assert_eq!(
        found(&s, "crates/p2a-core/src/m.rs"),
        [(line_of(src, "_w"), Kind::F64WithoutCast)]
    );
    assert_eq!(s.ok_markers, 3);
    assert_eq!(s.legacy_markers.len(), 1);
}

#[test]
fn test_na_guard_rejects_malformed_and_stale_markers() {
    let src = "pub fn f(c: &Column) -> PolarsResult<()> {\n    \
               let _a = c.f64()?; // na-ok\n    \
               let _b = c.f64()?; // na-ok:\n    \
               // na-legacy(T2): wrong tier\n    \
               let _c = c.f64()?;\n    \
               // na-legacy: no tier\n    \
               // na-legacy(T4): nothing to excuse\n    \
               let _d = 1;\n    \
               // the na-ok marker is explained in CLAUDE.md\n    \
               /// mark sites with `// na-ok: <reason>`\n    \
               Ok(())\n\
               }\n";
    let fx = Fixture::new("malformed");
    fx.file("crates/p2a-core/src/m.rs", src);
    let s = fx.scan();
    let mut expected = vec![
        (line_of(src, "_a"), Kind::F64WithoutCast),
        (line_of(src, "_a"), Kind::MalformedMarker),
        (line_of(src, "_b"), Kind::F64WithoutCast),
        (line_of(src, "_b"), Kind::MalformedMarker),
        (line_of(src, "wrong tier"), Kind::MalformedMarker),
        (line_of(src, "_c"), Kind::F64WithoutCast),
        (line_of(src, "no tier"), Kind::MalformedMarker),
        (line_of(src, "nothing to excuse"), Kind::StaleLegacyMarker),
    ];
    expected.sort();
    assert_eq!(found(&s, "crates/p2a-core/src/m.rs"), expected);
    assert_eq!(s.ok_markers, 0);
    assert_eq!(s.legacy_markers.len(), 1);
}

#[test]
fn test_na_guard_skips_cfg_test_items_but_not_the_code_after_them() {
    let src = "pub fn a(x: &[f64]) -> f64 {\n    x.iter().sum()\n}\n\n\
               #[cfg(test)]\n\
               mod tests {\n    \
                   #[test]\n    \
                   fn t() {\n        \
                       let _s = \"}\";\n        \
                       let _c = '{';\n        \
                       let _v: Vec<f64> = ca.into_no_null_iter().collect();\n    \
                   }\n\
               }\n\n\
               pub fn b(ca: &Float64Chunked) -> Vec<f64> {\n    \
                   ca.into_no_null_iter().collect() // fn B\n\
               }\n\n\
               #[cfg(all(test, feature = \"http\"))]\n\
               #[allow(dead_code)]\n\
               mod http_tests {\n    \
                   fn h(ca: &Float64Chunked) -> Vec<f64> {\n        \
                       ca.into_no_null_iter().collect()\n    \
                   }\n\
               }\n\n\
               #[cfg(test)] use polars::prelude::Float64Chunked;\n\n\
               pub fn c(c: &Column) -> PolarsResult<()> {\n    \
                   let _e = c.f64()?;\n    \
                   Ok(())\n\
               }\n";
    let fx = Fixture::new("cfgtest");
    fx.file("crates/p2a-core/src/synth.rs", src);
    let s = fx.scan();
    assert_eq!(
        found(&s, "crates/p2a-core/src/synth.rs"),
        [
            (line_of(src, "// fn B"), Kind::IntoNoNullIter),
            (line_of(src, "_e"), Kind::F64WithoutCast)
        ]
    );
    assert_eq!(s.test_items, 3);
}

#[test]
fn test_na_guard_skips_out_of_line_test_modules() {
    let iter =
        "pub fn f(ca: &Float64Chunked) -> Vec<f64> {\n    ca.into_no_null_iter().collect()\n}\n";
    let bare = "pub fn g(c: &Column) {\n    let _ = c.f64();\n}\n";
    let fx = Fixture::new("outofline");
    fx.file(
        "crates/p2a-core/src/db/mod.rs",
        "mod other;\n#[cfg(test)]\nmod tests;\npub use other::f;\n",
    )
    .file("crates/p2a-core/src/db/tests.rs", iter)
    .file("crates/p2a-core/src/db/other.rs", iter)
    .file(
        "crates/p2a-core/src/lib.rs",
        "#[cfg(test)]\npub(crate) mod test_support;\npub mod db;\n",
    )
    .file("crates/p2a-core/src/test_support.rs", bare)
    .file(
        "crates/p2a-core/src/stats/mod.rs",
        "#[cfg(test)]\nmod helpers;\n",
    )
    .file("crates/p2a-core/src/stats/helpers/mod.rs", bare)
    .file("crates/p2a-core/src/stats/helpers/deep.rs", bare);
    let s = fx.scan();
    let files: Vec<&str> = s.violations.iter().map(|v| v.file.as_str()).collect();
    assert_eq!(files, ["crates/p2a-core/src/db/other.rs"]);
    assert_eq!(s.files, 8);
}

#[test]
fn test_na_guard_ignores_comments_and_string_literals() {
    let body = "pub fn lits(c: &Column) -> PolarsResult<()> {\n    \
                /// Never call into_no_null_iter() or .f64() on user data.\n    \
                // ca.into_no_null_iter() in a line comment\n    \
                /* c.f64() in a block /* nested into_no_null_iter( */ comment */\n    \
                let _s = \"into_no_null_iter( and .f64() in a string\";\n    \
                let _r = r#\"raw \".f64()\" string\"#;\n    \
                let _b = b\".f64()\";\n    \
                let _ch = '{';\n    \
                let _lt: &'static str = \"x\";\n    \
                let _m = \"line one\n.f64()\nline three\";\n    \
                Ok(())\n\
                }\n";
    let fx = Fixture::new("literals");
    fx.file("crates/p2a-core/src/l.rs", body);
    assert_eq!(fx.scan().violations, []);
    let flagged = body.replace(
        "    Ok(())\n",
        "    let a = c.f64()?; // trailing note\n    Ok(())\n",
    );
    fx.file("crates/p2a-core/src/l.rs", &flagged);
    assert_eq!(
        found(&fx.scan(), "crates/p2a-core/src/l.rs"),
        [(line_of(&flagged, "trailing note"), Kind::F64WithoutCast)]
    );
}

#[test]
fn test_na_guard_scans_only_the_src_of_the_five_product_crates() {
    let src = "pub fn f(c: &Column) { let _ = c.f64(); }\n";
    let scanned = [
        "crates/p2a-core/src/lib.rs",
        "crates/p2a-mcp/src/lib.rs",
        "crates/p2a-cli/src/main.rs",
        "crates/p2a-chat/src/main.rs",
        "crates/p2a-dioxus/src/main.rs",
    ];
    let fx = Fixture::new("crates");
    for rel in scanned.iter().chain(&[
        "crates/p2a-core/tests/t.rs",
        "crates/p2a-core/benches/b.rs",
        "crates/p2a-validation/src/lib.rs",
        "crates/p2a-core/src/notes.txt",
        "validation/x.rs",
    ]) {
        fx.file(rel, src);
    }
    let s = fx.scan();
    let files: Vec<&str> = s.violations.iter().map(|v| v.file.as_str()).collect();
    let mut expected = scanned.to_vec();
    expected.sort();
    assert_eq!(files, expected);
    assert_eq!(s.files_per_crate, [1, 1, 1, 1, 1]);
}

#[test]
fn test_na_guard_tree_is_clean() {
    let s = scan(&p2a_validation::repo_root());
    assert!(s.files >= 300, "only {} files walked", s.files);
    assert!(
        s.files_per_crate.iter().all(|&n| n >= 1),
        "{:?}",
        s.files_per_crate
    );
    assert!(s.test_items >= 100, "only {} test items", s.test_items);
    assert!(s.violations.is_empty(), "{}", report(&s.violations));
    assert!(
        s.legacy_markers.len() <= MAX_LEGACY_MARKERS,
        "{} na-legacy markers, at most {MAX_LEGACY_MARKERS}: do not add na-legacy markers; \
         migrate the site through p2a_core::data::columns\n{}",
        s.legacy_markers.len(),
        s.legacy_markers
            .iter()
            .map(|(f, l)| format!("  {f}:{l}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
fn test_na_guard_legacy_mcp_helpers_are_gone() {
    let fx = Fixture::new("helpers");
    fx.file(
        "crates/p2a-mcp/src/tools/common.rs",
        "pub fn extract_column_f64(ds: &Dataset, c: &str) -> Vec<f64> {\n    vec![]\n}\n",
    )
    .file(
        "crates/p2a-mcp/src/tools/handlers/ml.rs",
        "fn f() -> Result<(), String> {\n    \
         let m = common::extract_numeric_matrix(&ds, &cols)?;\n    \
         // extract_column_f64 is gone\n    \
         Ok(())\n\
         }\n\
         fn my_extract_column_f64_like() {}\n",
    );
    let hits = legacy_helper_uses(&fx.0);
    assert_eq!(
        hits,
        [
            (
                "crates/p2a-mcp/src/tools/common.rs".to_string(),
                1,
                "extract_column_f64"
            ),
            (
                "crates/p2a-mcp/src/tools/handlers/ml.rs".to_string(),
                2,
                "extract_numeric_matrix"
            ),
        ]
    );
    let real = legacy_helper_uses(&p2a_validation::repo_root());
    assert!(
        real.is_empty(),
        "callers must use tools::common::numeric_column(s)/numeric_matrix with an explicit \
         NaPolicy:\n{}",
        real.iter()
            .map(|(f, l, n)| format!("  {f}:{l}: {n}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
