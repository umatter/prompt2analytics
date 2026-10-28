//! Case files: `validation/expected/<id>.json`, schema 1, written by
//! `Rscript validation/r/run.R --write <id>`.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::{SCHEMA, repo_path, validation_dir};

/// One R reference case: the metadata of the R computation and its named values.
#[derive(Debug, Clone, Deserialize)]
pub struct Case {
    /// Case-file schema; always [`SCHEMA`] for a loaded case.
    pub schema: u32,
    /// Case id, equal to the file stem.
    pub id: String,
    /// Area, e.g. `econometrics/panel`; `harness` cases are exempt from the registry.
    pub area: String,
    /// Human-readable description of the method and data.
    pub method: String,
    /// The R function(s) the values come from.
    pub reference: String,
    /// The R call that produced the values.
    pub call: String,
    /// Repository-relative path of the analysis dataset.
    pub dataset: String,
    /// Every repository-relative dataset the case read or generated.
    pub datasets: Vec<String>,
    /// `R.version.string` of the R that wrote the file.
    pub r_version: String,
    /// R packages used, with their versions.
    pub packages: BTreeMap<String, String>,
    values: BTreeMap<String, Value>,
}

/// One value of a case: a named vector, an unnamed vector or a scalar. JSON `null`
/// (R `NA`) is `None` and reads as NaN.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Value {
    // Variant order matters for untagged matching.
    /// A named vector (coefficients, SEs, CI bounds), in R's order.
    Named {
        /// Element names, as R reports them.
        names: Vec<String>,
        /// Element values, parallel to `names`.
        values: Vec<Option<f64>>,
    },
    /// An unnamed vector, in R's order.
    Vector(Vec<Option<f64>>),
    /// A single number.
    Scalar(Option<f64>),
}

/// Why a case file could not be loaded.
#[derive(Debug)]
pub enum CaseError {
    /// No file at this path.
    Missing(PathBuf),
    /// The file at this path exists but could not be read.
    Io(PathBuf, std::io::Error),
    /// The file is not valid JSON, does not match schema 1 (including a
    /// `{names, values}` value whose arrays differ in length), or holds another case.
    Parse(String),
    /// The file has another schema than [`SCHEMA`].
    Schema {
        /// The case file.
        path: PathBuf,
        /// The schema it declares (0 if absent).
        found: u32,
    },
}

/// The case id of a case-file path (its file stem).
fn id_of(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

impl fmt::Display for CaseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaseError::Missing(p) => {
                let id = id_of(p);
                write!(
                    f,
                    "case `{id}` not found at {}; run `Rscript validation/r/run.R --write {id}`",
                    p.display()
                )
            }
            CaseError::Io(p, e) => write!(f, "cannot read case file {}: {e}", p.display()),
            CaseError::Schema { path, found } => write!(
                f,
                "case file {} has schema {found}; this crate reads schema {SCHEMA} \
                 (regenerate with Rscript validation/r/run.R --write {})",
                path.display(),
                id_of(path)
            ),
            CaseError::Parse(m) => write!(f, "cannot parse case file: {m}"),
        }
    }
}

impl std::error::Error for CaseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CaseError::Io(_, e) => Some(e),
            _ => None,
        }
    }
}

/// Parse the text of a case file found at `path`, expected to hold case `id`.
fn parse_case(id: &str, path: &Path, text: &str) -> Result<Case, CaseError> {
    let doc: serde_json::Value =
        serde_json::from_str(text).map_err(|e| CaseError::Parse(e.to_string()))?;
    let schema = doc
        .get("schema")
        .and_then(serde_json::Value::as_u64)
        .map(|n| u32::try_from(n).unwrap_or(u32::MAX))
        .unwrap_or(0);
    if schema != SCHEMA {
        return Err(CaseError::Schema {
            path: path.to_path_buf(),
            found: schema,
        });
    }
    let case: Case = serde_json::from_value(doc).map_err(|e| CaseError::Parse(e.to_string()))?;
    for (k, v) in &case.values {
        if let Value::Named { names, values } = v
            && names.len() != values.len()
        {
            return Err(CaseError::Parse(format!(
                "value `{k}` has {} names for {} values",
                names.len(),
                values.len()
            )));
        }
    }
    if case.id != id {
        return Err(CaseError::Parse(format!(
            "file {} holds case `{}`, expected `{id}`",
            path.display(),
            case.id
        )));
    }
    Ok(case)
}

/// Read and parse the case file at `path`, expected to hold case `id`.
fn read_case(id: &str, path: &Path) -> Result<Case, CaseError> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_case(id, path, &text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            Err(CaseError::Missing(path.to_path_buf()))
        }
        Err(e) => Err(CaseError::Io(path.to_path_buf(), e)),
    }
}

fn to_f64(v: &Option<f64>) -> f64 {
    v.unwrap_or(f64::NAN)
}

impl Case {
    /// Load `validation/expected/<id>.json`. Panics with an actionable message, e.g.
    /// ``case `x` not found at <path>; run `Rscript validation/r/run.R --write x` ``.
    pub fn load(id: &str) -> Case {
        match Case::try_load(id) {
            Ok(case) => case,
            Err(e @ CaseError::Missing(_)) => panic!("{e}"),
            Err(e) => panic!("case `{id}`: {e}"),
        }
    }

    /// Load `validation/expected/<id>.json`, returning why it could not be loaded.
    pub fn try_load(id: &str) -> Result<Case, CaseError> {
        let path = validation_dir().join("expected").join(format!("{id}.json"));
        read_case(id, &path)
    }

    /// Absolute path of the analysis dataset (`repo_path(&self.dataset)`).
    pub fn dataset_path(&self) -> PathBuf {
        repo_path(&self.dataset)
    }

    /// The value keys, sorted.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.values.keys().map(String::as_str)
    }

    /// The value stored under `key`, or `None`.
    pub(crate) fn value(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Comma-separated list of the keys, for messages.
    pub(crate) fn key_list(&self) -> String {
        self.keys().collect::<Vec<_>>().join(", ")
    }

    fn get(&self, key: &str) -> &Value {
        self.values.get(key).unwrap_or_else(|| {
            panic!(
                "case {}: no value `{key}` (keys: {})",
                self.id,
                self.key_list()
            )
        })
    }

    /// A scalar value; `null` reads as NaN. Panics on an unknown key or a vector value.
    pub fn scalar(&self, key: &str) -> f64 {
        match self.get(key) {
            Value::Scalar(v) => to_f64(v),
            _ => panic!("case {}: value `{key}` is not a scalar", self.id),
        }
    }

    /// A named or unnamed vector in R's order; a scalar reads as a vector of length 1.
    pub fn vector(&self, key: &str) -> Vec<f64> {
        match self.get(key) {
            Value::Named { values, .. } | Value::Vector(values) => {
                values.iter().map(to_f64).collect()
            }
            Value::Scalar(v) => vec![to_f64(v)],
        }
    }

    /// The names of a named vector. Panics if the value has no names.
    pub fn names(&self, key: &str) -> &[String] {
        match self.get(key) {
            Value::Named { names, .. } => names,
            _ => panic!("case {}: value `{key}` has no names", self.id),
        }
    }

    /// The element `name` of a named vector; `name` must occur exactly once.
    pub fn named(&self, key: &str, name: &str) -> f64 {
        let Value::Named { names, values } = self.get(key) else {
            panic!("case {}: value `{key}` has no names", self.id);
        };
        let mut hits = names.iter().enumerate().filter(|(_, n)| *n == name);
        match (hits.next(), hits.next()) {
            (Some((i, _)), None) if i < values.len() => to_f64(&values[i]),
            _ => panic!(
                "case {}: value `{key}` has no single element named `{name}`",
                self.id
            ),
        }
    }

    /// The elements of a named vector in the order of `order` (each name exactly once in R).
    pub fn by_names<S: AsRef<str>>(&self, key: &str, order: &[S]) -> Vec<f64> {
        order.iter().map(|n| self.named(key, n.as_ref())).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = r#""id": "x", "area": "harness", "method": "m", "reference": "r",
        "call": "c", "dataset": "validation/datasets/grunfeld.csv",
        "datasets": ["validation/datasets/grunfeld.csv"],
        "r_version": "R version 4.3.3 (2024-02-29)", "packages": {"jsonlite": "2.0.0"}"#;

    fn parse(text: &str) -> Result<Case, CaseError> {
        parse_case("x", Path::new("validation/expected/x.json"), text)
    }

    #[test]
    fn schema_2_is_rejected() {
        let text = format!(r#"{{"schema": 2, {HEAD}, "values": {{"a": 1}}}}"#);
        assert!(matches!(
            parse(&text),
            Err(CaseError::Schema { found: 2, path }) if path == Path::new("validation/expected/x.json")
        ));
    }

    #[test]
    fn missing_schema_reads_as_0() {
        let text = format!(r#"{{{HEAD}, "values": {{"a": 1}}}}"#);
        let err = parse(&text).unwrap_err();
        assert!(matches!(err, CaseError::Schema { found: 0, .. }));
        assert_eq!(
            err.to_string(),
            "case file validation/expected/x.json has schema 0; this crate reads schema 1 \
             (regenerate with Rscript validation/r/run.R --write x)"
        );
    }

    #[test]
    fn other_id_is_a_parse_error_naming_both() {
        let text = format!(r#"{{"schema": 1, {HEAD}, "values": {{"a": 1}}}}"#)
            .replace(r#""id": "x""#, r#""id": "other""#);
        match parse(&text) {
            Err(CaseError::Parse(m)) => {
                assert!(m.contains("`other`") && m.contains("`x`"), "{m}");
                assert!(m.contains("validation/expected/x.json"), "{m}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn text_value_is_a_parse_error() {
        let text = format!(r#"{{"schema": 1, {HEAD}, "values": {{"a": "text"}}}}"#);
        let err = parse(&text).unwrap_err();
        assert!(matches!(err, CaseError::Parse(_)), "{err:?}");
        assert!(
            err.to_string().starts_with("cannot parse case file: "),
            "{err}"
        );
    }

    #[test]
    fn named_value_with_unequal_lengths_is_a_parse_error() {
        let text = format!(
            r#"{{"schema": 1, {HEAD}, "values": {{"a": 1, "v": {{"names": ["x", "y"], "values": [1]}}}}}}"#
        );
        match parse(&text) {
            Err(CaseError::Parse(m)) => {
                assert_eq!(m, "value `v` has 2 names for 1 values");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    #[test]
    fn unreadable_file_is_an_io_error_naming_the_path() {
        // A directory exists but cannot be read as a file: not `Missing`, not `Parse`.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let err = read_case("x", dir).unwrap_err();
        let CaseError::Io(p, _) = &err else {
            panic!("expected Io, got {err:?}");
        };
        assert_eq!(p, dir);
        let text = err.to_string();
        assert!(
            text.starts_with(&format!("cannot read case file {}: ", dir.display())),
            "{text}"
        );
        let missing = dir.join("no_such_case.json");
        assert!(matches!(
            read_case("no_such_case", &missing),
            Err(CaseError::Missing(p)) if p == missing
        ));
    }

    #[test]
    fn invalid_json_is_a_parse_error() {
        assert!(matches!(parse("{\"schema\": 1,"), Err(CaseError::Parse(_))));
    }

    #[test]
    fn empty_packages_map_is_a_base_r_only_case() {
        // harness.R writes `packages = character(0)` as {} and refuses NULL, which would be [].
        let head = HEAD.replace(r#""packages": {"jsonlite": "2.0.0"}"#, r#""packages": {}"#);
        let case = parse(&format!(r#"{{"schema": 1, {head}, "values": {{"a": 1}}}}"#)).unwrap();
        assert!(case.packages.is_empty());
        let head = HEAD.replace(r#""packages": {"jsonlite": "2.0.0"}"#, r#""packages": []"#);
        let text = format!(r#"{{"schema": 1, {head}, "values": {{"a": 1}}}}"#);
        assert!(matches!(parse(&text), Err(CaseError::Parse(_))));
    }

    #[test]
    fn null_reads_as_nan() {
        let text = format!(
            r#"{{"schema": 1, {HEAD}, "extra_field": true, "values": {{"a": null, "b": [1, null]}}}}"#
        );
        let case = parse(&text).unwrap();
        assert!(case.scalar("a").is_nan());
        let b = case.vector("b");
        assert_eq!(b.len(), 2);
        assert_eq!(b[0], 1.0);
        assert!(b[1].is_nan());
    }

    #[test]
    fn unknown_key_lists_the_keys() {
        let text = format!(r#"{{"schema": 1, {HEAD}, "values": {{"a": 1, "b": [2]}}}}"#);
        let case = parse(&text).unwrap();
        let payload = std::panic::catch_unwind(|| case.scalar("zz")).unwrap_err();
        let msg = payload.downcast_ref::<String>().unwrap();
        assert_eq!(msg, "case x: no value `zz` (keys: a, b)");
    }

    #[test]
    fn named_rejects_absent_and_duplicate_names() {
        let text = format!(
            r#"{{"schema": 1, {HEAD}, "values": {{"v": {{"names": ["a", "a", "b"], "values": [1, 2, 3]}}}}}}"#
        );
        let case = parse(&text).unwrap();
        assert_eq!(case.named("v", "b"), 3.0);
        for name in ["a", "c"] {
            let payload = std::panic::catch_unwind(|| case.named("v", name)).unwrap_err();
            let msg = payload.downcast_ref::<String>().unwrap();
            assert_eq!(
                msg,
                &format!("case x: value `v` has no single element named `{name}`")
            );
        }
    }
}
