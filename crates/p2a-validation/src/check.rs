//! Tolerances ([`Tol`]) and the collecting comparator ([`Check`]).

use crate::case::{Case, Value};

/// A tolerance: `|actual - expected| <= abs + rel * |expected|`. NaN equals NaN;
/// an infinite value only equals itself.
///
/// The constructors encode the tolerance table of `validation/README.md`; a looser
/// tolerance needs [`Tol::iterative`] or [`Tol::new`] with a comment giving the reason.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tol {
    /// Relative part, multiplied by `|expected|`.
    pub rel: f64,
    /// Absolute part.
    pub abs: f64,
}

impl Tol {
    /// A tolerance of `rel` relative plus `abs` absolute.
    pub const fn new(rel: f64, abs: f64) -> Tol {
        Tol { rel, abs }
    }

    /// Coefficients (validation/README.md): n < 100 -> 1e-6, 100..=1000 -> 1e-8,
    /// n > 1000 -> 1e-10 relative; abs 1e-12.
    pub fn coef(n: usize) -> Tol {
        let rel = if n < 100 {
            1e-6
        } else if n <= 1000 {
            1e-8
        } else {
            1e-10
        };
        Tol::new(rel, 1e-12)
    }

    /// Standard errors (validation/README.md): n < 100 -> 1e-5, 100..=1000 -> 1e-6,
    /// n > 1000 -> 1e-8 relative; abs 1e-14.
    pub fn se(n: usize) -> Tol {
        let rel = if n < 100 {
            1e-5
        } else if n <= 1000 {
            1e-6
        } else {
            1e-8
        };
        Tol::new(rel, 1e-14)
    }

    /// Test statistics, R², F, chi², logLik/AIC/BIC, variance components and CI bounds:
    /// the same as [`Tol::se`].
    pub fn stat(n: usize) -> Tol {
        Tol::se(n)
    }

    /// p-values: `rel = se(n).rel`, `abs = 1e-12` (tail p-values below 1e-12 compare equal).
    pub fn p_value(n: usize) -> Tol {
        Tol::new(Tol::se(n).rel, 1e-12)
    }

    /// Iterative estimators (MAP demeaning, MLE): a documented looser relative
    /// tolerance; abs 1e-12.
    pub fn iterative(rel: f64) -> Tol {
        Tol::new(rel, 1e-12)
    }

    /// Numbers printed with `decimals` decimals: abs `0.5 * 10^-decimals + 1e-12`, rel 0.
    pub fn printed(decimals: u32) -> Tol {
        let scale = 10f64.powi(i32::try_from(decimals).unwrap_or(i32::MAX));
        Tol::new(0.0, 0.5 / scale + 1e-12)
    }

    /// Counts and degrees of freedom: exact.
    pub fn exact() -> Tol {
        Tol::new(0.0, 0.0)
    }

    /// The allowed absolute difference for `expected`.
    fn allowed(&self, expected: f64) -> f64 {
        self.abs + self.rel * expected.abs()
    }

    /// Whether `actual` matches `expected` within this tolerance.
    pub fn allows(&self, actual: f64, expected: f64) -> bool {
        match (actual.is_nan(), expected.is_nan()) {
            (true, true) => true,
            (true, false) | (false, true) => false,
            (false, false) if actual.is_infinite() || expected.is_infinite() => actual == expected,
            (false, false) => (actual - expected).abs() <= self.allowed(expected),
        }
    }
}

struct Row {
    ok: bool,
    line: String,
}

/// Collects every comparison of one test and panics once, listing all failures:
/// `case <id>: <key>[<name>] rust=<a> R=<e> |diff|=<d> allowed=<t>`.
///
/// Structural problems (unknown key, wrong kind of value, length mismatch, missing,
/// extra or duplicate names) are failed comparisons too; nothing panics before
/// [`Check::finish`]. A `Check` that is dropped without `finish` panics, so a test
/// cannot forget to assert.
pub struct Check<'a> {
    case: &'a Case,
    rows: Vec<Row>,
    failures: Vec<String>,
    finished: bool,
}

impl<'a> Check<'a> {
    /// Start comparing against `case`.
    pub fn new(case: &'a Case) -> Self {
        Check {
            case,
            rows: Vec::new(),
            failures: Vec::new(),
            finished: false,
        }
    }

    fn push(&mut self, ok: bool, line: String) {
        if !ok {
            self.failures.push(line.clone());
        }
        self.rows.push(Row { ok, line });
    }

    fn compare(&mut self, label: &str, actual: f64, expected: f64, tol: Tol) {
        let ok = tol.allows(actual, expected);
        let line = format!(
            "case {}: {label} rust={actual:?} R={expected:?} |diff|={:e} allowed={:e}",
            self.case.id,
            (actual - expected).abs(),
            tol.allowed(expected)
        );
        self.push(ok, line);
    }

    fn structural(&mut self, key: &str, message: String) {
        let line = format!("case {}: {key}: {message}", self.case.id);
        self.push(false, line);
    }

    fn lookup(&mut self, key: &str) -> Option<&'a Value> {
        let case = self.case;
        let value = case.value(key);
        if value.is_none() {
            self.structural(
                key,
                format!("no value `{key}` in the case (keys: {})", case.key_list()),
            );
        }
        value
    }

    /// Compare a scalar value (`Value::Scalar`; `null` is NaN).
    pub fn scalar(&mut self, key: &str, actual: f64, tol: Tol) -> &mut Self {
        match self.lookup(key) {
            None => {}
            Some(Value::Scalar(e)) => {
                self.compare(key, actual, e.unwrap_or(f64::NAN), tol);
            }
            Some(_) => self.structural(key, format!("value `{key}` is not a scalar")),
        }
        self
    }

    /// Compare element-wise in R's order; lengths must match. Named and unnamed vectors
    /// are accepted, and a scalar counts as a vector of length 1.
    pub fn vector(&mut self, key: &str, actual: &[f64], tol: Tol) -> &mut Self {
        let expected: Vec<f64> = match self.lookup(key) {
            None => return self,
            Some(Value::Named { values, .. } | Value::Vector(values)) => {
                values.iter().map(|v| v.unwrap_or(f64::NAN)).collect()
            }
            Some(Value::Scalar(v)) => vec![v.unwrap_or(f64::NAN)],
        };
        if actual.len() != expected.len() {
            self.structural(
                key,
                format!("length {} (Rust) vs {} (R)", actual.len(), expected.len()),
            );
            return self;
        }
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            self.compare(&format!("{key}[{i}]"), a, e, tol);
        }
        self
    }

    /// The names and values of a named R vector, or a structural failure.
    fn named_value(&mut self, key: &str) -> Option<(&'a [String], Vec<f64>)> {
        match self.lookup(key)? {
            Value::Named { names, values } => Some((
                names.as_slice(),
                values.iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
            )),
            _ => {
                self.structural(key, format!("value `{key}` has no names"));
                None
            }
        }
    }

    /// Check that the Rust names and values pair up.
    fn same_length<S: AsRef<str>>(&mut self, key: &str, names: &[S], actual: &[f64]) -> bool {
        if names.len() == actual.len() {
            return true;
        }
        self.structural(
            key,
            format!(
                "{} Rust names for {} Rust values",
                names.len(),
                actual.len()
            ),
        );
        false
    }

    /// Compare the element of R's vector named `name` with the single Rust entry of that
    /// name, or record why it cannot be compared.
    fn compare_name<S: AsRef<str>>(
        &mut self,
        key: &str,
        name: &str,
        expected: f64,
        names: &[S],
        actual: &[f64],
        tol: Tol,
    ) {
        let hits: Vec<usize> = names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.as_ref() == name)
            .map(|(i, _)| i)
            .collect();
        match hits.as_slice() {
            [] => self.structural(key, format!("R name `{name}` not in the Rust names")),
            [i] => self.compare(&format!("{key}[{name}]"), actual[*i], expected, tol),
            more => self.structural(
                key,
                format!(
                    "name `{name}` occurs {} times in the Rust names",
                    more.len()
                ),
            ),
        }
    }

    /// Compare by name: every R name must occur exactly once in `names`, and every entry
    /// of `names` must be an R name. Order is irrelevant.
    pub fn named<S: AsRef<str>>(
        &mut self,
        key: &str,
        names: &[S],
        actual: &[f64],
        tol: Tol,
    ) -> &mut Self {
        let Some((r_names, expected)) = self.named_value(key) else {
            return self;
        };
        if !self.same_length(key, names, actual) {
            return self;
        }
        if actual.len() != expected.len() {
            self.structural(
                key,
                format!("length {} (Rust) vs {} (R)", actual.len(), expected.len()),
            );
        }
        for (r_name, &e) in r_names.iter().zip(&expected) {
            self.compare_name(key, r_name, e, names, actual, tol);
        }
        let mut reported: Vec<&str> = Vec::new();
        for n in names {
            let n = n.as_ref();
            if !r_names.iter().any(|r| r == n) && !reported.contains(&n) {
                reported.push(n);
                self.structural(key, format!("`{n}` is not an R name"));
            }
        }
        self
    }

    /// Compare only the R names in `subset` (for estimators that report part of R's
    /// vector, e.g. FE has no intercept). Each must be an R name and occur exactly once
    /// in `names`; other entries of `names` are ignored.
    pub fn named_subset<S: AsRef<str>>(
        &mut self,
        key: &str,
        subset: &[&str],
        names: &[S],
        actual: &[f64],
        tol: Tol,
    ) -> &mut Self {
        let Some((r_names, expected)) = self.named_value(key) else {
            return self;
        };
        if !self.same_length(key, names, actual) {
            return self;
        }
        for &s in subset {
            let r_hits: Vec<usize> = r_names
                .iter()
                .enumerate()
                .filter(|(_, r)| *r == s)
                .map(|(i, _)| i)
                .collect();
            match r_hits.as_slice() {
                [] => self.structural(key, format!("`{s}` is not an R name")),
                [i] => self.compare_name(key, s, expected[*i], names, actual, tol),
                more => self.structural(
                    key,
                    format!("name `{s}` occurs {} times in the R names", more.len()),
                ),
            }
        }
        self
    }

    /// Panics if any comparison failed (listing all of them), or if nothing was
    /// compared. With `P2A_VALIDATION_VERBOSE=1` every comparison is printed to stderr.
    pub fn finish(&mut self) {
        self.finished = true;
        if std::env::var_os("P2A_VALIDATION_VERBOSE").is_some_and(|v| v == "1") {
            for row in &self.rows {
                eprintln!("{} {}", if row.ok { "ok  " } else { "FAIL" }, row.line);
            }
        }
        if self.rows.is_empty() {
            panic!(
                "case {}: Check finished but nothing was compared",
                self.case.id
            );
        }
        if !self.failures.is_empty() {
            panic!(
                "case {}: {} of {} comparisons failed\n  {}",
                self.case.id,
                self.failures.len(),
                self.rows.len(),
                self.failures.join("\n  ")
            );
        }
    }
}

impl Drop for Check<'_> {
    fn drop(&mut self) {
        if !self.finished && !std::thread::panicking() {
            panic!("Check for case {} dropped without finish()", self.case.id);
        }
    }
}
