//! Numeric guard: product code must not reintroduce the numeric defects that the cross-cutting
//! review findings X-6, X-14 and X-17 removed from the whole tree, nor discard the rank
//! condition of a matrix inverse.
//!
//! The guard walks `crates/<crate>/src/**/*.rs` of the five product crates with the lexer of
//! the NA guard (`source_scan`: comments, string and char literals blanked; `#[cfg(test)]` and
//! `#[cfg(all(test, ..))]` items and out-of-line test modules skipped) and reports, as
//! `file:line`, every non-test site of five kinds:
//!
//! - **partial-cmp-unwrap**: `partial_cmp(..)` followed by `.unwrap`, `.unwrap_or`,
//!   `.unwrap_or_else`, `.unwrap_or_default` or `.expect`. That comparator is no total order
//!   once a NaN appears (std's sort may panic or return an unsorted slice). Compare with
//!   `f64::total_cmp` after refusing or filtering NaN.
//! - **one-minus-cdf**: `1 - <cdf>(..)`, an upper tail computed as one minus a distribution
//!   function, which is exactly 0 below about 1e-16. Use
//!   `traits::estimator::{normal_p_value_two_sided, t_test_p_value, chi_squared_p_value,
//!   f_test_p_value}` or statrs `Normal::sf`.
//! - **abramowitz-stegun**: a constant of the Abramowitz-Stegun approximations 26.2.17,
//!   26.2.23 or 7.1.26 (accurate to about 1e-4 to 1e-7). Use
//!   `traits::estimator::{normal_quantile, z_critical}` or statrs `erf`/`erfc`.
//! - **critical-value-literal**: a hard-coded `1.96`, `2.576` or `1.645`. Use
//!   `traits::estimator::z_critical(level)`.
//! - **discarded-inverse-condition**: a `safe_inverse` call whose second element (the
//!   condition, `Some` when the matrix is rank-deficient or ill-conditioned and a generalized
//!   inverse was used) is dropped. Solve with `linalg::solve::{least_squares, sym_inverse}`
//!   and push the warning.
//!
//! Two markers, written as a non-doc `//` comment on the reported line or the line above,
//! document a deliberate exception:
//! - `// numeric-ok: <reason>` for the first four kinds (for example a constant that the R
//!   reference hard-codes, named in the reason);
//! - `// inverse-ok: <reason>` for a `safe_inverse` use that cannot change any output.
//!
//! A comment that starts with `numeric-ok` or `inverse-ok` in any other form is a
//! malformed marker, and a valid marker with no site of its kinds on its line or the next a
//! stale one; both are reported. The rules are checked on fixture trees;
//! `test_numeric_guard_tree_is_clean` runs them on the repository.
//! `test_release_profile_checks_overflow_and_bench_does_not` checks the workspace profiles:
//! release builds keep integer overflow checks, the bench profile alone turns them off.

mod source_scan;

use source_scan::{Fixture, is_ident, line_of, walk};
use std::fs;
use std::path::Path;

/// Constants of Abramowitz and Stegun 26.2.23 (normal quantile), 7.1.26 (erf) and 26.2.17
/// (normal distribution function).
const AS_CONSTANTS: [&str; 19] = [
    "2.515517",
    "0.802853",
    "0.010328",
    "1.432788",
    "0.189269",
    "0.001308",
    "0.3275911",
    "0.254829592",
    "0.284496736",
    "1.421413741",
    "1.453152027",
    "1.061405429",
    "0.2316419",
    "0.319381530",
    "0.31938153",
    "0.356563782",
    "1.781477937",
    "1.821255978",
    "1.330274429",
];
/// Callees that compute a distribution function besides names ending in `cdf`.
const CDF_CALLEES: [&str; 14] = [
    "erf",
    "gamma_lr",
    "beta_reg",
    "ptukey",
    "pnorm",
    "pt",
    "pchisq",
    "pf",
    "pbinom",
    "ppois",
    "pgamma",
    "pbeta",
    "regularized_gamma_p",
    "regularized_gamma_p_series",
];
/// Hard-coded normal critical values.
const CRITICAL_LITERALS: [&str; 3] = ["1.96", "2.576", "1.645"];
/// Code lines after a `safe_inverse` call in which a bound condition must be used.
const INVERSE_USE_WINDOW: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    PartialCmpUnwrap,
    OneMinusCdf,
    AbramowitzStegun,
    CriticalValueLiteral,
    DiscardedInverse,
    MalformedMarker,
    StaleMarker,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::PartialCmpUnwrap => "partial-cmp-unwrap",
            Kind::OneMinusCdf => "one-minus-cdf",
            Kind::AbramowitzStegun => "abramowitz-stegun",
            Kind::CriticalValueLiteral => "critical-value-literal",
            Kind::DiscardedInverse => "discarded-inverse-condition",
            Kind::MalformedMarker => "malformed-marker",
            Kind::StaleMarker => "stale-marker",
        }
    }

    fn hint(self) -> &'static str {
        match self {
            Kind::PartialCmpUnwrap => {
                "compare floats with f64::total_cmp after refusing or filtering NaN"
            }
            Kind::OneMinusCdf => {
                "use traits::estimator::{normal_p_value_two_sided, t_test_p_value, \
                 chi_squared_p_value, f_test_p_value} or statrs Normal::sf"
            }
            Kind::AbramowitzStegun => {
                "use traits::estimator::{normal_quantile, z_critical} or statrs erf/erfc"
            }
            Kind::CriticalValueLiteral => {
                "use traits::estimator::z_critical(level); if the R reference hard-codes the \
                 value, mark it // numeric-ok: <R function>"
            }
            Kind::DiscardedInverse => {
                "solve with linalg::solve::{least_squares, sym_inverse}, push the warning, or \
                 mark a prediction-only use // inverse-ok: <reason>"
            }
            Kind::MalformedMarker | Kind::StaleMarker => {
                "a marker is `// numeric-ok: <reason>` or `// inverse-ok: <reason>` on the line \
                 or the line above a site"
            }
        }
    }

    /// Whether a `numeric-ok` (`false`) or `inverse-ok` (`true`) marker exempts this kind.
    fn inverse_kind(self) -> bool {
        self == Kind::DiscardedInverse
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
    files: usize,
    files_per_crate: [usize; 5],
    test_items: usize,
    violations: Vec<Violation>,
    /// Well-formed `numeric-ok` markers (stale ones included).
    numeric_ok: usize,
    /// Well-formed `inverse-ok` markers (stale ones included).
    inverse_ok: usize,
}

// -------------------------------------------------------------------------------------------
// Text helpers over the joined code text of one file
// -------------------------------------------------------------------------------------------

/// The code text of one file's non-test lines joined with `\n` (test lines empty), as chars,
/// and the char offset at which each line starts.
struct Text {
    chars: Vec<char>,
    starts: Vec<usize>,
}

impl Text {
    fn new(code: &[String], is_test: &[bool]) -> Text {
        let mut chars = Vec::new();
        let mut starts = Vec::with_capacity(code.len());
        for (i, line) in code.iter().enumerate() {
            starts.push(chars.len());
            if !is_test[i] {
                chars.extend(line.chars());
            }
            chars.push('\n');
        }
        Text { chars, starts }
    }

    fn at(&self, i: usize) -> char {
        self.chars.get(i).copied().unwrap_or('\0')
    }

    /// The 0-based line of char offset `i`.
    fn line(&self, i: usize) -> usize {
        self.starts.partition_point(|&s| s <= i).saturating_sub(1)
    }

    fn starts_with(&self, i: usize, s: &str) -> bool {
        s.chars().enumerate().all(|(k, c)| self.at(i + k) == c)
    }

    /// Whether `word` starts at `i` as a whole identifier.
    fn word_at(&self, i: usize, word: &str) -> bool {
        self.starts_with(i, word)
            && !(i > 0 && is_ident(self.at(i - 1)))
            && !is_ident(self.at(i + word.chars().count()))
    }

    /// Every offset at which `word` starts as a whole identifier.
    fn words(&self, word: &str) -> Vec<usize> {
        let first = word.chars().next().unwrap();
        (0..self.chars.len())
            .filter(|&i| self.chars[i] == first && self.word_at(i, word))
            .collect()
    }

    fn skip_ws(&self, mut i: usize) -> usize {
        while self.at(i).is_whitespace() {
            i += 1;
        }
        i
    }

    /// The offset after the bracket group that opens at `i` (`(`, `[`, `{` or `<`), or `None`.
    fn group_end(&self, i: usize) -> Option<usize> {
        let (open, close) = match self.at(i) {
            '(' => ('(', ')'),
            '[' => ('[', ']'),
            '{' => ('{', '}'),
            '<' => ('<', '>'),
            _ => return None,
        };
        let mut depth = 0usize;
        let mut k = i;
        while k < self.chars.len() {
            let c = self.chars[k];
            if c == open {
                depth += 1;
            } else if c == close && !(close == '>' && matches!(self.at(k - 1), '-' | '=')) {
                depth -= 1;
                if depth == 0 {
                    return Some(k + 1);
                }
            }
            k += 1;
        }
        None
    }

    /// The identifier that starts at `i` (empty if none).
    fn ident(&self, i: usize) -> String {
        self.chars[i.min(self.chars.len())..]
            .iter()
            .take_while(|&&c| is_ident(c))
            .collect()
    }

    fn slice(&self, a: usize, b: usize) -> String {
        self.chars[a.min(self.chars.len())..b.min(self.chars.len())]
            .iter()
            .collect()
    }
}

/// Text without whitespace.
fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

// -------------------------------------------------------------------------------------------
// Rules
// -------------------------------------------------------------------------------------------

/// `partial_cmp(..)` followed by `.unwrap*` or `.expect`.
fn partial_cmp_sites(t: &Text) -> Vec<usize> {
    let mut out = Vec::new();
    for i in t.words("partial_cmp") {
        // `fn partial_cmp(` is a definition.
        let before = t.slice(i.saturating_sub(8), i);
        if before.trim_end().ends_with("fn") {
            continue;
        }
        let open = t.skip_ws(i + "partial_cmp".len());
        let Some(end) = t.group_end(open).filter(|_| t.at(open) == '(') else {
            continue;
        };
        let k = t.skip_ws(end);
        if t.at(k) != '.' {
            continue;
        }
        let name = t.ident(t.skip_ws(k + 1));
        if matches!(
            name.as_str(),
            "unwrap" | "unwrap_or" | "unwrap_or_else" | "unwrap_or_default" | "expect"
        ) {
            out.push(i);
        }
    }
    out
}

/// The end of the literal one (`1`, `1.`, `1.0`, `1.00`, with an optional `f64` or `_f64`
/// suffix) that starts at `i`, or `None`.
fn literal_one_end(t: &Text, i: usize) -> Option<usize> {
    if t.at(i) != '1' {
        return None;
    }
    if i > 0 && (is_ident(t.at(i - 1)) || t.at(i - 1) == '.') {
        return None;
    }
    let mut k = i + 1;
    if t.at(k) == '.' && !is_ident(t.at(k + 1)) || t.at(k) == '.' && t.at(k + 1) == '0' {
        k += 1;
        while t.at(k) == '0' {
            k += 1;
        }
    }
    if t.starts_with(k, "_f64") {
        k += 4;
    } else if t.starts_with(k, "f64") {
        k += 3;
    }
    if is_ident(t.at(k)) || t.at(k) == '.' {
        return None;
    }
    Some(k)
}

/// The end of the operand that starts at `i` (after whitespace): a bracket group or an
/// identifier, followed by postfix parts. `None` when no operand starts there.
fn operand_end(t: &Text, i: usize) -> Option<usize> {
    let i = t.skip_ws(i);
    let mut k = if t.at(i) == '(' {
        t.group_end(i)?
    } else if is_ident(t.at(i)) && !t.at(i).is_ascii_digit() {
        i + t.ident(i).chars().count()
    } else {
        return None;
    };
    loop {
        let j = t.skip_ws(k);
        if t.starts_with(j, "::") {
            let m = t.skip_ws(j + 2);
            if t.at(m) == '<' {
                k = t.group_end(m)?;
            } else if is_ident(t.at(m)) {
                k = m + t.ident(m).chars().count();
            } else {
                return Some(k);
            }
        } else if t.at(j) == '.' && t.at(j + 1) != '.' {
            let m = t.skip_ws(j + 1);
            let name = t.ident(m);
            if name.is_empty() {
                return Some(k);
            }
            k = m + name.chars().count();
        } else if matches!(t.at(j), '(' | '[') {
            k = t.group_end(j)?;
        } else if t.at(j) == '?' {
            k = j + 1;
        } else {
            return Some(k);
        }
    }
}

/// Whether the code in `[a, b)` calls a distribution function.
fn calls_cdf(t: &Text, a: usize, b: usize) -> bool {
    let mut i = a;
    while i < b {
        if is_ident(t.at(i)) && !(i > 0 && is_ident(t.at(i - 1))) {
            let name = t.ident(i);
            let after = t.skip_ws(i + name.chars().count());
            if after < b && t.at(after) == '(' {
                let cdf = name.ends_with("cdf") && !name.ends_with("inverse_cdf");
                if cdf || CDF_CALLEES.contains(&name.as_str()) {
                    return true;
                }
            }
            i += name.chars().count();
        } else {
            i += 1;
        }
    }
    false
}

/// `1 - <operand calling a cdf>`.
fn one_minus_cdf_sites(t: &Text) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 0..t.chars.len() {
        let Some(end) = literal_one_end(t, i) else {
            continue;
        };
        let m = t.skip_ws(end);
        if t.at(m) != '-' || matches!(t.at(m + 1), '=' | '>') {
            continue;
        }
        if let Some(op_end) = operand_end(t, m + 1)
            && calls_cdf(t, m + 1, op_end)
        {
            out.push(i);
        }
    }
    out
}

/// Float literal tokens of `t` equal to one of `values`: (offset, token).
fn float_literal_sites(t: &Text, values: &[&str]) -> Vec<usize> {
    let mut out = Vec::new();
    let n = t.chars.len();
    let mut i = 0;
    while i < n {
        let c = t.at(i);
        let starts =
            c.is_ascii_digit() && !(i > 0 && (is_ident(t.at(i - 1)) || t.at(i - 1) == '.'));
        if !starts {
            i += 1;
            continue;
        }
        let mut k = i;
        while t.at(k).is_ascii_digit() {
            k += 1;
        }
        if t.at(k) != '.' || !t.at(k + 1).is_ascii_digit() {
            i = k.max(i + 1);
            continue;
        }
        k += 1;
        while t.at(k).is_ascii_digit() {
            k += 1;
        }
        let token = t.slice(i, k);
        let mut e = k;
        if t.starts_with(e, "_f64") {
            e += 4;
        } else if t.starts_with(e, "f64") {
            e += 3;
        }
        if !is_ident(t.at(e)) && t.at(e) != '.' && values.contains(&token.as_str()) {
            out.push(i);
        }
        i = e.max(i + 1);
    }
    out
}

/// After the call's closing parenthesis at `end`, skip `.map_err(..)`, `.ok_or(..)`,
/// `.ok_or_else(..)`, `.context(..)`, `?`, `.unwrap()` and `.expect(..)`.
fn skip_error_chain(t: &Text, mut k: usize) -> usize {
    loop {
        let j = t.skip_ws(k);
        if t.at(j) == '?' {
            k = j + 1;
            continue;
        }
        if t.at(j) != '.' {
            return k;
        }
        let m = t.skip_ws(j + 1);
        let name = t.ident(m);
        if !matches!(
            name.as_str(),
            "map_err" | "ok_or" | "ok_or_else" | "context" | "unwrap" | "expect"
        ) {
            return k;
        }
        let open = t.skip_ws(m + name.chars().count());
        match t.group_end(open).filter(|_| t.at(open) == '(') {
            Some(e) => k = e,
            None => return k,
        }
    }
}

/// In compacted text, `<prefix>(<ident>,<second>)<suffix>` with the tuple's second binding:
/// returns the second binding when the pattern starts at the beginning of `s`.
fn tuple_second(s: &str, prefix: &str, suffix: &str) -> Option<String> {
    let rest = s.strip_prefix(prefix)?.strip_prefix('(')?;
    let (first, rest) = rest.split_once(',')?;
    let first = first.strip_prefix("mut").unwrap_or(first);
    if first.is_empty() || !first.chars().all(is_ident) {
        return None;
    }
    let (second, rest) = rest.split_once(')')?;
    let second = second.strip_prefix("mut").unwrap_or(second);
    if second.is_empty() || !second.chars().all(is_ident) || !rest.starts_with(suffix) {
        return None;
    }
    Some(second.to_string())
}

/// The second binding of a destructuring pattern anywhere in `s` (compacted): `let(a,b)`,
/// `ifletOk((a,b))`, `whileletOk((a,b))`, `Ok((a,b))` (match arms).
fn second_bindings(s: &str, prefixes: &[(&str, &str)]) -> Vec<String> {
    let mut out = Vec::new();
    for (prefix, suffix) in prefixes {
        for (k, _) in s.match_indices(prefix) {
            if k > 0 && is_ident(s[..k].chars().next_back().unwrap()) {
                continue;
            }
            if let Some(b) = tuple_second(&s[k..], prefix, suffix) {
                out.push(b);
            }
        }
    }
    out
}

/// `safe_inverse(..)` calls whose condition is dropped.
fn discarded_inverse_sites(t: &Text, code: &[String]) -> Vec<usize> {
    let mut out = Vec::new();
    for i in t.words("safe_inverse") {
        let line = t.line(i);
        let line_code = code[line].trim_start();
        if line_code.starts_with("use ") || line_code.starts_with("pub use ") {
            continue;
        }
        let before = t.slice(i.saturating_sub(8), i);
        if before.trim_end().ends_with("fn") {
            continue;
        }
        let open = t.skip_ws(i + "safe_inverse".len());
        if t.at(open) != '(' {
            // `use ..::{safe_inverse, ..}` spread over lines, or a path mention.
            continue;
        }
        let Some(call_end) = t.group_end(open) else {
            continue;
        };
        // Statement head: back to the previous `;`, `{` or `}` at depth 0.
        let mut depth = 0i64;
        let mut h = i;
        while h > 0 {
            let c = t.at(h - 1);
            match c {
                ')' | ']' => depth += 1,
                '(' | '[' => depth -= 1,
                '}' if depth == 0 => break,
                '}' => depth += 1,
                '{' if depth == 0 => break,
                '{' => depth -= 1,
                ';' if depth == 0 => break,
                _ => {}
            }
            if depth < 0 {
                break;
            }
            h -= 1;
        }
        let head = compact(&t.slice(h, i));
        let after = skip_error_chain(t, call_end);
        let j = t.skip_ws(after);
        let mut flagged = false;
        // `.0` right after the call.
        if t.at(j) == '.' && t.at(j + 1) == '0' && !t.at(j + 2).is_ascii_digit() {
            flagged = true;
        }
        // `.map(|(a, _)| ..)`, `.ok().map(..)`, `.and_then(..)` with an underscore binding.
        if !flagged && t.at(j) == '.' {
            let mut m = t.skip_ws(j + 1);
            let mut name = t.ident(m);
            if name == "ok" {
                let o = t.skip_ws(m + 2);
                if t.at(o) == '(' && t.at(t.skip_ws(o + 1)) == ')' {
                    let d = t.skip_ws(t.skip_ws(o + 1) + 1);
                    if t.at(d) == '.' {
                        m = t.skip_ws(d + 1);
                        name = t.ident(m);
                    }
                }
            }
            if matches!(name.as_str(), "map" | "and_then") {
                let o = t.skip_ws(m + name.chars().count());
                if let Some(e) = t.group_end(o).filter(|_| t.at(o) == '(') {
                    let args = compact(&t.slice(o + 1, e));
                    if let Some(b) = tuple_second(&args, "|", "|")
                        && b.starts_with('_')
                    {
                        flagged = true;
                    }
                }
            }
        }
        // Destructuring in the head, or in the arms of a `match` on the call.
        let mut names = second_bindings(
            &head,
            &[("let", "="), ("ifletOk(", ")="), ("whileletOk(", ")=")],
        );
        let mut window_from = j;
        if head.ends_with("match") && t.at(j) == '{' {
            if let Some(e) = t.group_end(j) {
                let arms = compact(&t.slice(j + 1, e - 1));
                names.extend(second_bindings(&arms, &[("Ok(", ")")]));
                window_from = j + 1;
            }
        }
        if !flagged {
            for name in &names {
                if name.starts_with('_') {
                    flagged = true;
                    break;
                }
                // The bound condition must be used in the next INVERSE_USE_WINDOW lines.
                let last = (line + INVERSE_USE_WINDOW).min(t.starts.len() - 1);
                let stop = t.starts[last] + code[last].chars().count();
                let used = (window_from..stop).any(|k| t.word_at(k, name));
                if !used {
                    flagged = true;
                    break;
                }
            }
        }
        if flagged {
            out.push(i);
        }
    }
    out
}

/// A recorded comment as a marker: `Some(Ok(false))` for `numeric-ok: <reason>`,
/// `Some(Ok(true))` for `inverse-ok: <reason>`, `Some(Err(()))` for a malformed one, `None`
/// for no marker.
fn marker(comment: &str) -> Option<Result<bool, ()>> {
    let text = comment.trim_start();
    for (word, inverse) in [("numeric-ok", false), ("inverse-ok", true)] {
        if text.starts_with(word) {
            let ok = text
                .strip_prefix(word)
                .and_then(|r| r.strip_prefix(':'))
                .is_some_and(|r| !r.trim().is_empty());
            return Some(if ok { Ok(inverse) } else { Err(()) });
        }
    }
    None
}

/// Scan the five product crates under `root`.
fn scan(root: &Path) -> Scan {
    let walked = walk(root);
    let mut scan = Scan {
        files: walked.files,
        files_per_crate: walked.files_per_crate,
        test_items: walked.test_items,
        ..Scan::default()
    };
    for src in walked.scanned {
        let n = src.code.len();
        let text = Text::new(&src.code, &src.is_test);
        let source_at = |i: usize| src.lines.get(i).map_or("", |l| l.trim()).to_string();
        // Raw sites per line, before markers.
        let mut sites: Vec<(usize, Kind)> = Vec::new();
        let mut per_line = |offsets: Vec<usize>, kind: Kind, once: bool| {
            let mut lines: Vec<usize> = offsets.into_iter().map(|o| text.line(o)).collect();
            if once {
                lines.dedup();
            }
            sites.extend(lines.into_iter().map(|l| (l, kind)));
        };
        per_line(partial_cmp_sites(&text), Kind::PartialCmpUnwrap, false);
        per_line(one_minus_cdf_sites(&text), Kind::OneMinusCdf, false);
        per_line(
            float_literal_sites(&text, &AS_CONSTANTS),
            Kind::AbramowitzStegun,
            true,
        );
        per_line(
            float_literal_sites(&text, &CRITICAL_LITERALS),
            Kind::CriticalValueLiteral,
            true,
        );
        per_line(
            discarded_inverse_sites(&text, &src.code),
            Kind::DiscardedInverse,
            false,
        );
        // Markers of the non-test lines: Some(false) numeric-ok, Some(true) inverse-ok.
        let mut markers: Vec<Option<bool>> = vec![None; n];
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
                Some(Ok(inverse)) => {
                    markers[i] = Some(inverse);
                    if inverse {
                        scan.inverse_ok += 1;
                    } else {
                        scan.numeric_ok += 1;
                    }
                    let covers = sites
                        .iter()
                        .any(|&(l, k)| (l == i || l == i + 1) && k.inverse_kind() == inverse);
                    if !covers {
                        scan.violations.push(Violation {
                            file: src.rel.clone(),
                            line: i + 1,
                            kind: Kind::StaleMarker,
                            code: source_at(i),
                        });
                    }
                }
            }
        }
        for (line, kind) in sites {
            let exempt = |l: usize| markers[l] == Some(kind.inverse_kind());
            if exempt(line) || (line > 0 && exempt(line - 1)) {
                continue;
            }
            scan.violations.push(Violation {
                file: src.rel.clone(),
                line: line + 1,
                kind,
                code: source_at(line),
            });
        }
    }
    scan.violations
        .sort_by(|a, b| (&a.file, a.line, a.kind).cmp(&(&b.file, b.line, b.kind)));
    scan
}

/// The violations as `file:line: label: code` lines under a header, then one hint per kind.
fn report(v: &[Violation]) -> String {
    let mut out = format!(
        "Numeric guard: {} violation(s) in product code (CLAUDE.md, \"Numeric conventions\")\n",
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
    let mut kinds: Vec<Kind> = v.iter().map(|x| x.kind).collect();
    kinds.sort();
    kinds.dedup();
    let mut hints: Vec<&str> = Vec::new();
    for k in kinds {
        let h = k.hint();
        if !hints.contains(&h) {
            hints.push(h);
            out.push_str(&format!("{}: {h}\n", k.label()));
        }
    }
    out
}

/// The numeric guard's scan of a fixture tree.
fn scan_fixture(fx: &Fixture) -> Scan {
    scan(fx.root())
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

const A: &str = "crates/p2a-core/src/a.rs";

/// The 1-based line of the one line of `src` whose trimmed text is `text`.
fn exact_line(src: &str, text: &str) -> usize {
    let hits: Vec<usize> = src
        .lines()
        .enumerate()
        .filter(|(_, l)| l.trim() == text)
        .map(|(i, _)| i + 1)
        .collect();
    assert_eq!(hits.len(), 1, "`{text}` is on lines {hits:?}");
    hits[0]
}

#[test]
fn test_numeric_guard_flags_partial_cmp_unwrap_forms() {
    let src = "use std::cmp::Ordering;\n\
               fn f(v: &mut Vec<f64>, w: &mut Vec<(u8, f64)>) {\n    \
                   v.sort_by(|a, b| a.partial_cmp(b).unwrap());\n    \
                   w.sort_by(|a, b| {\n        \
                       b.1\n            \
                           .partial_cmp(&a.1)\n            \
                           .unwrap_or(Ordering::Equal)\n    \
                   });\n    \
                   let m = v.iter().max_by(|a, b| a.partial_cmp(b).expect(\"finite\"));\n    \
                   let ok = v[0].partial_cmp(&v[1]) == Some(Ordering::Less);\n    \
                   v.sort_by(|a, b| a.total_cmp(b));\n    \
                   my_partial_cmp_helper(a, b).unwrap();\n\
               }\n\
               impl PartialOrd for S { fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) } }\n";
    let fx = Fixture::new("ng-partial");
    fx.file(A, src);
    assert_eq!(
        found(&scan_fixture(&fx), A),
        [
            (
                line_of(src, "a.partial_cmp(b).unwrap()"),
                Kind::PartialCmpUnwrap
            ),
            (line_of(src, ".partial_cmp(&a.1)"), Kind::PartialCmpUnwrap),
            (line_of(src, "max_by"), Kind::PartialCmpUnwrap),
        ]
    );
}

#[test]
fn test_numeric_guard_flags_one_minus_cdf_operands() {
    let flagged = [
        "let p = 2.0 * (1.0 - normal_cdf(z.abs()));",
        "let p = 1.0 - n.cdf(z);",
        "let p = 1.0 - ChiSquared::new(df).unwrap().cdf(x);",
        "let p = 1. - t.cdf(x);",
        "let p = 1 - binom.cdf(k);",
        "let p = 1.0 - erf(x);",
        "let p = 1.0 - ptukey(q, k, df);",
    ];
    let clean = [
        "let a = 1.0 - alpha / 2.0;",
        "let q = n.inverse_cdf(1.0 - alpha);",
        "let q = 1.0 - n.inverse_cdf(alpha);",
        "let a = 21.0 - n.cdf(x);",
        "let a = 0.1 - n.cdf(x);",
        "let a = 1.0 - p;",
        "let a = n.sf(z);",
        "let a = (1.0 - x) * n.cdf(z);",
        "x -= 1.0;",
    ];
    let mut src = String::from("fn f() {\n");
    for l in flagged.iter().chain(clean.iter()) {
        src.push_str(&format!("    {l}\n"));
    }
    src.push_str("    let split = 1.0\n        - t.cdf(x);\n}\n");
    let fx = Fixture::new("ng-cdf");
    fx.file(A, &src);
    let mut expected: Vec<(usize, Kind)> = flagged
        .iter()
        .map(|l| (line_of(&src, l), Kind::OneMinusCdf))
        .collect();
    expected.push((line_of(&src, "let split"), Kind::OneMinusCdf));
    assert_eq!(found(&scan_fixture(&fx), A), expected);
}

#[test]
fn test_numeric_guard_flags_abramowitz_stegun_and_critical_literals() {
    let src = "fn f(se: f64, estimate: f64) {\n    \
                   let c0 = 2.515517;\n    \
                   let a1 = 0.254829592_f64;\n    \
                   let p = 0.2316419;\n    \
                   let both = 1.432788 + 0.189269;\n    \
                   let z = 1.96;\n    \
                   let lo = estimate - 1.645 * se;\n    \
                   let w = 2.576f64;\n    \
                   let n1 = 11.96;\n    \
                   let n2 = 1.9600;\n    \
                   let n3 = 1.959964;\n    \
                   let n4 = 2.5155170;\n    \
                   // let c = 2.515517;\n    \
                   let s = \"1.96\";\n\
               }\n";
    let fx = Fixture::new("ng-literals");
    fx.file(A, src);
    assert_eq!(
        found(&scan_fixture(&fx), A),
        [
            (line_of(src, "c0"), Kind::AbramowitzStegun),
            (line_of(src, "a1"), Kind::AbramowitzStegun),
            (line_of(src, "let p"), Kind::AbramowitzStegun),
            (line_of(src, "both"), Kind::AbramowitzStegun),
            (line_of(src, "let z"), Kind::CriticalValueLiteral),
            (line_of(src, "let lo"), Kind::CriticalValueLiteral),
            (line_of(src, "let w"), Kind::CriticalValueLiteral),
        ]
    );
}

#[test]
fn test_numeric_guard_flags_discarded_safe_inverse_conditions() {
    let src = "use crate::linalg::matrix_ops::safe_inverse;\n\
               pub fn safe_inverse(m: &M) -> R { todo!() }\n\
               fn f(m: &M) -> EconResult<()> {\n    \
                   let (inv, _) = safe_inverse(&m.view())?;\n    \
                   let (inv2, _cond) = safe_inverse(&m.view()).map_err(|e| EconError::SingularMatrix {\n        \
                       context: e.to_string(),\n        \
                       suggestion: String::new(),\n    \
                   })?;\n    \
                   if let Ok((inv3, _)) = safe_inverse(&m.view()) {\n        \
                       use_it(inv3);\n    \
                   }\n    \
                   let inv4 = match safe_inverse(&m.view()) {\n        \
                       Ok((i, _)) => i,\n        \
                       Err(_) => return Ok(()),\n    \
                   };\n    \
                   let inv5 = safe_inverse(&m.view())?.0;\n    \
                   let inv6 = safe_inverse(&m.view()).map(|(i, _)| i)?;\n    \
                   let (inv7, cond7) = safe_inverse(&m.view())?;\n    \
                   let (inv8, cond8) = safe_inverse(&m.view())?;\n    \
                   if let Some(c) = cond8 {\n        \
                       warnings.push(format!(\"{c}\"));\n    \
                   }\n    \
                   // inverse-ok: prediction only\n    \
                   let (inv9, _) = safe_inverse(&m.view())?;\n    \
                   Ok(())\n\
               }\n";
    let fx = Fixture::new("ng-inverse");
    fx.file(A, src);
    assert_eq!(
        found(&scan_fixture(&fx), A),
        [
            (line_of(src, "(inv, _)"), Kind::DiscardedInverse),
            (line_of(src, "inv2"), Kind::DiscardedInverse),
            (line_of(src, "inv3, _"), Kind::DiscardedInverse),
            (line_of(src, "let inv4"), Kind::DiscardedInverse),
            (line_of(src, "inv5"), Kind::DiscardedInverse),
            (line_of(src, "inv6"), Kind::DiscardedInverse),
            (line_of(src, "inv7"), Kind::DiscardedInverse),
        ]
    );
}

#[test]
fn test_numeric_guard_markers_exempt_their_kind_on_their_line_and_the_next() {
    let src = "fn f(v: &mut Vec<f64>, x: f64) {\n    \
                   // numeric-ok: R's ansari.test computes one minus pansari\n    \
                   let p = 1.0 - pansari_cdf(x);\n    \
                   let z = 1.96; // numeric-ok: EValue hard-codes it\n    \
                   // inverse-ok: wrong kind\n    \
                   v.sort_by(|a, b| a.partial_cmp(b).unwrap());\n    \
                   // numeric-ok: too far\n\n    \
                   let w = 2.576;\n\
               }\n";
    let fx = Fixture::new("ng-markers");
    fx.file(A, src);
    let s = scan_fixture(&fx);
    assert_eq!(
        found(&s, A),
        [
            (line_of(src, "wrong kind"), Kind::StaleMarker),
            (line_of(src, "a.partial_cmp(b)"), Kind::PartialCmpUnwrap),
            (line_of(src, "too far"), Kind::StaleMarker),
            (line_of(src, "let w"), Kind::CriticalValueLiteral),
        ]
    );
    assert_eq!((s.numeric_ok, s.inverse_ok), (3, 1));
}

#[test]
fn test_numeric_guard_rejects_malformed_and_stale_markers() {
    let src = "fn f() {\n    \
                   // numeric-ok\n    \
                   // numeric-ok:\n    \
                   // inverse-ok missing colon\n    \
                   // numeric-ok: nothing here\n    \
                   let a = 1;\n    \
                   // inverse-ok: nothing here\n    \
                   let b = 2;\n    \
                   // the numeric-ok marker is explained in CLAUDE.md\n    \
                   /// numeric-ok: a doc comment\n    \
                   let c = 3;\n\
               }\n";
    let fx = Fixture::new("ng-malformed");
    fx.file(A, src);
    let s = scan_fixture(&fx);
    assert_eq!(
        found(&s, A),
        [
            (exact_line(src, "// numeric-ok"), Kind::MalformedMarker),
            (exact_line(src, "// numeric-ok:"), Kind::MalformedMarker),
            (line_of(src, "missing colon"), Kind::MalformedMarker),
            (line_of(src, "// numeric-ok: nothing"), Kind::StaleMarker),
            (line_of(src, "// inverse-ok: nothing"), Kind::StaleMarker),
        ]
    );
    assert_eq!((s.numeric_ok, s.inverse_ok), (1, 1));
}

#[test]
fn test_numeric_guard_skips_tests_comments_and_strings() {
    let src = "pub fn a() {}\n\n\
               #[cfg(test)]\n\
               mod tests {\n    \
                   fn t(v: &mut Vec<f64>, m: &M) {\n        \
                       v.sort_by(|a, b| a.partial_cmp(b).unwrap());\n        \
                       let p = 1.0 - n.cdf(z);\n        \
                       let c = 2.515517;\n        \
                       let z = 1.96;\n        \
                       let (i, _) = safe_inverse(m).unwrap();\n    \
                   }\n\
               }\n\n\
               #[cfg(all(test, feature = \"http\"))]\n\
               fn h() { let z = 1.96; }\n\n\
               /* nested /* let z = 1.96; */ let p = 1.0 - n.cdf(z); */\n\
               const S: &str = \"1.0 - x.cdf(\";\n\
               const R: &str = r#\".partial_cmp(b).unwrap()\"#;\n\
               const C: char = '{';\n\
               fn after() { let z = 1.96; }\n";
    let fx = Fixture::new("ng-skips");
    fx.file(A, src);
    assert_eq!(
        found(&scan_fixture(&fx), A),
        [(line_of(src, "fn after"), Kind::CriticalValueLiteral)]
    );
}

#[test]
fn test_numeric_guard_scans_only_the_src_of_the_five_product_crates() {
    let src = "pub fn f() -> f64 { let z = 1.96; z }\n";
    let fx = Fixture::new("ng-crates");
    let product = [
        "crates/p2a-core/src/lib.rs",
        "crates/p2a-mcp/src/lib.rs",
        "crates/p2a-cli/src/main.rs",
        "crates/p2a-chat/src/main.rs",
        "crates/p2a-dioxus/src/main.rs",
    ];
    for f in product {
        fx.file(f, src);
    }
    for f in [
        "crates/p2a-core/tests/t.rs",
        "crates/p2a-core/benches/b.rs",
        "crates/p2a-validation/src/lib.rs",
        "validation/x.rs",
    ] {
        fx.file(f, src);
    }
    let s = scan_fixture(&fx);
    let mut files: Vec<&str> = s.violations.iter().map(|v| v.file.as_str()).collect();
    files.sort();
    let mut expected = product.to_vec();
    expected.sort();
    assert_eq!(files, expected);
    assert_eq!(s.files_per_crate, [1, 1, 1, 1, 1]);
}

#[test]
fn test_numeric_guard_tree_is_clean() {
    let s = scan(&p2a_validation::repo_root());
    assert!(s.files >= 300, "only {} files walked", s.files);
    assert!(
        s.files_per_crate.iter().all(|&n| n >= 1),
        "a product crate has no files: {:?}",
        s.files_per_crate
    );
    assert!(s.test_items >= 100, "only {} test items", s.test_items);
    assert!(s.violations.is_empty(), "{}", report(&s.violations));
}

/// The keys of the `[name]` section of a TOML text (comment and empty lines skipped), or
/// `None` without such a section.
fn toml_section(text: &str, name: &str) -> Option<Vec<String>> {
    let header = format!("[{name}]");
    let mut lines = text.lines().map(str::trim);
    lines.find(|l| *l == header)?;
    Some(
        lines
            .take_while(|l| !l.starts_with('['))
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| compact(l).replace('=', " = "))
            .collect(),
    )
}

/// Every file under `dir` (recursively) readable as text: (path, text).
fn text_files(dir: &Path, out: &mut Vec<(String, String)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        if let Ok(text) = fs::read_to_string(dir) {
            out.push((dir.display().to_string(), text));
        }
        return;
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            text_files(&p, out);
        } else if let Ok(text) = fs::read_to_string(&p) {
            out.push((p.display().to_string(), text));
        }
    }
}

#[test]
fn test_release_profile_checks_overflow_and_bench_does_not() {
    let root = p2a_validation::repo_root();
    let manifest = fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let release = toml_section(&manifest, "profile.release")
        .expect("no [profile.release] section in Cargo.toml");
    assert_eq!(
        release,
        ["overflow-checks = true"],
        "Cargo.toml [profile.release] must hold exactly overflow-checks = true"
    );
    let bench =
        toml_section(&manifest, "profile.bench").expect("no [profile.bench] section in Cargo.toml");
    assert_eq!(
        bench,
        ["overflow-checks = false"],
        "Cargo.toml [profile.bench] must hold exactly overflow-checks = false"
    );
    let others: Vec<&str> = manifest
        .lines()
        .map(str::trim)
        .filter(|l| {
            l.starts_with("[profile.") && *l != "[profile.release]" && *l != "[profile.bench]"
        })
        .collect();
    assert!(
        others.is_empty(),
        "Cargo.toml has other profile sections: {others:?}"
    );
    for entry in fs::read_dir(root.join("crates")).unwrap().flatten() {
        let m = entry.path().join("Cargo.toml");
        if let Ok(text) = fs::read_to_string(&m) {
            assert!(
                !text.contains("[profile"),
                "{} has a [profile] section (per-package profiles are not allowed)",
                m.display()
            );
        }
    }
    let mut deploy = Vec::new();
    for d in ["docker", ".github", "deploy"] {
        text_files(&root.join(d), &mut deploy);
    }
    for (path, text) in deploy {
        for key in [
            "CARGO_PROFILE_RELEASE_OVERFLOW_CHECKS",
            "CARGO_PROFILE_BENCH_",
        ] {
            assert!(
                !text.contains(key),
                "{path} overrides the build profile with {key}"
            );
        }
    }
}
