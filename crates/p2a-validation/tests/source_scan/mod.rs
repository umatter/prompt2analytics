//! The source lexer and walk shared by the repository's source guards (`na_guard.rs`,
//! `numeric_guard.rs`).
//!
//! [`code_lines`] blanks comments, string literals and char literals (line count and columns
//! kept) and records each line's non-doc `//` comment; [`test_lines`] flags the lines of
//! `#[cfg(test)]` and `#[cfg(all(test, ..))]` items and names the test modules declared out of
//! line; [`walk`] lexes every `.rs` file of the five product crates' `src` trees and drops the
//! files of out-of-line test modules (still counted). [`Fixture`] builds temporary trees for
//! the guards' self-tests.
//!
//! A `tests/<dir>/mod.rs` file is not a test target: each guard declares `mod source_scan;`.

// Each guard binary uses a different subset of these items.
#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

/// The crates whose `src` trees are scanned, in [`Walk::files_per_crate`] order.
pub const PRODUCT_CRATES: [&str; 5] = ["p2a-core", "p2a-mcp", "p2a-cli", "p2a-chat", "p2a-dioxus"];

/// One lexed source file of a product crate.
pub struct SourceFile {
    /// Repository-relative path with `/` separators.
    pub rel: String,
    /// Index into [`PRODUCT_CRATES`].
    pub krate: usize,
    /// The raw source lines.
    pub lines: Vec<String>,
    /// Code text per line (comments and literals blanked).
    pub code: Vec<String>,
    /// The non-doc `//` comment text of each line.
    pub comments: Vec<Option<String>>,
    /// Whether each line belongs to a test item.
    pub is_test: Vec<bool>,
}

/// What [`walk`] found.
#[derive(Default)]
pub struct Walk {
    /// Every `.rs` file walked, skipped test-only files included.
    pub files: usize,
    pub files_per_crate: [usize; 5],
    /// `#[cfg(test)]` and `#[cfg(all(test, ..))]` items found.
    pub test_items: usize,
    /// The files that are scanned (out-of-line test modules dropped).
    pub scanned: Vec<SourceFile>,
}

pub fn is_ident(c: char) -> bool {
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
pub fn code_lines(src: &str) -> (Vec<String>, Vec<Option<String>>) {
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
pub fn test_lines(code: &[String]) -> (Vec<bool>, Vec<String>, usize) {
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
        // Angle brackets of generics (`HashMap<u8, u8>`), so that their commas do not end a
        // field; `->` and `=>` are not brackets.
        let mut angle = 0usize;
        let mut prev = ' ';
        let mut opened = false;
        let mut end = None;
        'outer: while line < code.len() {
            for (k, ch) in code[line].char_indices().skip_while(|&(k, _)| k < col) {
                let before = std::mem::replace(&mut prev, ch);
                match ch {
                    '<' => angle += 1,
                    '>' if before != '-' && before != '=' => angle = angle.saturating_sub(1),
                    '{' => {
                        depth += 1;
                        opened = true;
                    }
                    // A `}` before the item opened a block closes the enclosing block: the item
                    // was a field or variant without a trailing comma and ends before it.
                    '}' if !opened && depth == 0 => {
                        end = Some((line, k.saturating_sub(1)));
                        break 'outer;
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
                    // A struct field or enum variant ends at its comma.
                    ',' if !opened && depth == 0 && nested == 0 && angle == 0 => {
                        end = Some((line, k));
                        break 'outer;
                    }
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
            && end_col >= start.1
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

/// Whether `name` occurs in `code` as a whole identifier for which `after` (the text that
/// follows it) holds.
pub fn identifier_at(code: &str, name: &str, after: impl Fn(&str) -> bool) -> bool {
    code.match_indices(name).any(|(k, _)| {
        let before = code[..k].chars().next_back();
        let rest = &code[k + name.len()..];
        !before.is_some_and(is_ident) && !rest.chars().next().is_some_and(is_ident) && after(rest)
    })
}

/// Read and lex every `.rs` file of the product crates' `src` trees under `root`, flagging the
/// lines of test items and dropping the files of out-of-line test modules (still counted in
/// `files`). A crate directory that does not exist counts 0 files.
pub fn walk(root: &Path) -> Walk {
    let mut walk = Walk::default();
    let mut all = Vec::new();
    let mut skip: Vec<PathBuf> = Vec::new();
    for (k, krate) in PRODUCT_CRATES.iter().enumerate() {
        let mut files = Vec::new();
        rust_files(&root.join("crates").join(krate).join("src"), &mut files);
        walk.files_per_crate[k] = files.len();
        walk.files += files.len();
        for path in files {
            let text = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("cannot read {} as UTF-8: {e}", path.display()));
            let (code, comments) = code_lines(&text);
            let (is_test, modules, items) = test_lines(&code);
            walk.test_items += items;
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
                SourceFile {
                    rel: relative(root, &path),
                    krate: k,
                    lines,
                    code,
                    comments,
                    is_test,
                },
            ));
        }
    }
    walk.scanned = all
        .into_iter()
        .filter(|(path, _)| !skip.iter().any(|s| path == s || path.starts_with(s)))
        .map(|(_, source)| source)
        .collect();
    walk
}

// -------------------------------------------------------------------------------------------
// Fixture trees
// -------------------------------------------------------------------------------------------

/// A temporary tree, removed when dropped.
pub struct Fixture(PathBuf);

impl Fixture {
    pub fn new(tag: &str) -> Fixture {
        let dir =
            std::env::temp_dir().join(format!("p2a-source-scan-{}-{tag}", std::process::id()));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();
        Fixture(dir)
    }

    pub fn file(&self, rel: &str, src: &str) -> &Fixture {
        let path = self.0.join(rel);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, src).unwrap();
        self
    }

    pub fn root(&self) -> &Path {
        &self.0
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The 1-based line of the one line of `src` that contains `needle`.
pub fn line_of(src: &str, needle: &str) -> usize {
    let hits: Vec<usize> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(needle))
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(hits.len(), 1, "`{needle}` is on lines {hits:?}");
    hits[0]
}
