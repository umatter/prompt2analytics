//! Option strings parsed into enums (Tier 3 rule R1, theme 3.6).
//!
//! Matching is case-insensitive, ' ', '-', '.' and '_' are equivalent, and every documented alias
//! (the R spelling, the CLI spelling, the MCP spelling) is accepted. Anything else is an
//! `EconError::InvalidSpecification` that names the option and lists the accepted values; no unknown
//! string silently selects a default.
//!
//! ```
//! use p2a_core::options::Choices;
//! #[derive(Debug, Clone, Copy, PartialEq)]
//! enum Kernel { Triangular, Uniform }
//! p2a_core::option_enum!(Kernel, Choices {
//!     option: "kernel",
//!     values: &[(&["triangular", "tri"], Kernel::Triangular), (&["uniform", "rectangular"], Kernel::Uniform)],
//! });
//! assert_eq!("Tri".parse::<Kernel>().unwrap(), Kernel::Triangular);
//! assert_eq!(Kernel::Uniform.to_string(), "uniform");
//! let err = "epa".parse::<Kernel>().unwrap_err().to_string();
//! assert!(err.contains("unknown kernel 'epa'; expected one of: triangular, uniform"));
//! ```

use crate::errors::{EconError, EconResult};

/// Canonical form of an option string (see the module docs).
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut pending_sep = false;
    for ch in s.trim().chars().flat_map(char::to_lowercase) {
        if matches!(ch, ' ' | '-' | '.' | '_') || ch.is_whitespace() {
            pending_sep = !out.is_empty();
        } else {
            if pending_sep {
                out.push('_');
                pending_sep = false;
            }
            out.push(ch);
        }
    }
    out
}

/// The user's input for an error message: at most 64 characters (char-boundary safe), then "...".
fn quoted(s: &str) -> String {
    const MAX_CHARS: usize = 64;
    let mut it = s.chars();
    let head: String = it.by_ref().take(MAX_CHARS).collect();
    if it.next().is_some() {
        format!("{head}...")
    } else {
        head
    }
}

/// The accepted spellings of one option; the first spelling of each entry is canonical.
pub struct Choices<T: 'static> {
    /// Option name used in error messages ("kernel").
    pub option: &'static str,
    /// (spellings, value) pairs.
    pub values: &'static [(&'static [&'static str], T)],
}

impl<T: Copy + PartialEq + 'static> Choices<T> {
    /// Parse a user string; unknown values are an error listing the canonical spellings.
    pub fn parse(&self, s: &str) -> EconResult<T> {
        let key = normalize(s);
        if !key.is_empty() {
            for (names, v) in self.values {
                if names.iter().any(|n| normalize(n) == key) {
                    return Ok(*v);
                }
            }
        }
        Err(EconError::InvalidSpecification {
            message: format!(
                "unknown {} '{}'; expected one of: {}",
                self.option,
                quoted(s),
                self.names().join(", ")
            ),
        })
    }

    /// The canonical spelling of `v` ("" if `v` is missing from the table, a programming error).
    pub fn canonical(&self, v: T) -> &'static str {
        self.values
            .iter()
            .find(|(_, x)| *x == v)
            .and_then(|(names, _)| names.first().copied())
            .unwrap_or("")
    }

    /// The canonical spellings in table order.
    pub fn names(&self) -> Vec<&'static str> {
        self.values
            .iter()
            .filter_map(|(names, _)| names.first().copied())
            .collect()
    }

    /// "a | b | c", for CLI help and tool descriptions.
    pub fn help(&self) -> String {
        self.names().join(" | ")
    }

    /// Table consistency for tests: no empty or ambiguous spelling, no value listed twice.
    pub fn validate(&self) -> Result<(), String> {
        let mut seen: Vec<String> = Vec::new();
        for (i, (names, v)) in self.values.iter().enumerate() {
            if names.is_empty() {
                return Err(format!("{}: entry {i} has no spelling", self.option));
            }
            if self.values[..i].iter().any(|(_, w)| w == v) {
                return Err(format!(
                    "{}: value of '{}' listed twice",
                    self.option, names[0]
                ));
            }
            for n in names.iter() {
                let k = normalize(n);
                if k.is_empty() {
                    return Err(format!("{}: empty spelling in '{}'", self.option, names[0]));
                }
                if seen.contains(&k) {
                    return Err(format!("{}: spelling '{n}' is ambiguous", self.option));
                }
                seen.push(k);
            }
        }
        Ok(())
    }
}

/// Implements `choices()`, `FromStr` (Err = `EconError`) and, unless `no_display` is given,
/// `Display` (canonical spelling) for an enum from a [`Choices`] table.
#[macro_export]
macro_rules! option_enum {
    ($ty:ty, $choices:expr, no_display) => {
        impl $ty {
            /// The accepted spellings of this option; the first spelling of each entry is canonical.
            pub fn choices() -> &'static $crate::options::Choices<$ty> {
                static CHOICES: $crate::options::Choices<$ty> = $choices;
                &CHOICES
            }
        }
        impl ::std::str::FromStr for $ty {
            type Err = $crate::errors::EconError;
            fn from_str(s: &str) -> ::std::result::Result<Self, Self::Err> {
                <$ty>::choices().parse(s)
            }
        }
    };
    ($ty:ty, $choices:expr) => {
        $crate::option_enum!($ty, $choices, no_display);
        impl ::std::fmt::Display for $ty {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                f.write_str(<$ty>::choices().canonical(*self))
            }
        }
    };
}

/// `None` and blank strings give `default`; anything else is parsed (unknown values are an error).
pub fn parse_or<T: std::str::FromStr<Err = EconError>>(
    s: Option<&str>,
    default: T,
) -> EconResult<T> {
    match s {
        Some(v) if !v.trim().is_empty() => v.parse(),
        _ => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    enum Kernel {
        Triangular,
        Epanechnikov,
        Uniform,
    }
    crate::option_enum!(
        Kernel,
        Choices {
            option: "kernel",
            values: &[
                (&["triangular", "tri"], Kernel::Triangular),
                (&["epanechnikov", "epa"], Kernel::Epanechnikov),
                (&["uniform", "rectangular", "rect"], Kernel::Uniform),
            ],
        }
    );

    fn message(e: EconError) -> String {
        match e {
            EconError::InvalidSpecification { message } => message,
            other => panic!("expected InvalidSpecification, got {other:?}"),
        }
    }

    #[test]
    fn test_normalize_folds_case_and_separators() {
        for s in [
            "  Quadratic Spectral ",
            "quadratic-spectral",
            "QUADRATIC_SPECTRAL",
            "quadratic.spectral",
        ] {
            assert_eq!(normalize(s), "quadratic_spectral", "{s:?}");
        }
        assert_eq!(normalize("Tukey-Hanning"), "tukey_hanning");
        assert_eq!(normalize("wild.rademacher"), "wild_rademacher");
        assert_eq!(normalize("a__b"), "a_b");
        assert_eq!(normalize("-hc1-"), "hc1");
        assert_eq!(normalize("HÇ1"), "hç1");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn test_kernel_parses_aliases_and_rejects_unknown() {
        for s in ["TRI", "Triangular", " tri "] {
            assert_eq!(s.parse::<Kernel>().unwrap(), Kernel::Triangular, "{s:?}");
        }
        for s in ["Epanechnikov", "epa"] {
            assert_eq!(s.parse::<Kernel>().unwrap(), Kernel::Epanechnikov, "{s:?}");
        }
        for s in ["rectangular", "Rect"] {
            assert_eq!(s.parse::<Kernel>().unwrap(), Kernel::Uniform, "{s:?}");
        }
        assert_eq!(
            message("qs2".parse::<Kernel>().unwrap_err()),
            "unknown kernel 'qs2'; expected one of: triangular, epanechnikov, uniform"
        );
        assert!("".parse::<Kernel>().is_err());
        let long = "x".repeat(200);
        let m = message(long.parse::<Kernel>().unwrap_err());
        assert!(m.len() < 300, "{m}");
        assert!(m.contains(&("x".repeat(64) + "...")), "{m}");
        assert!(!m.contains(&"x".repeat(65)), "{m}");
        let m = message("größe".parse::<Kernel>().unwrap_err());
        assert!(m.contains("'größe'"), "{m}");
        assert_eq!(Kernel::choices().validate(), Ok(()));
    }

    #[test]
    fn test_option_enum_display_fromstr_and_help() {
        assert_eq!(Kernel::Uniform.to_string(), "uniform");
        assert_eq!(
            Kernel::choices().help(),
            "triangular | epanechnikov | uniform"
        );
        assert_eq!(
            Kernel::choices().names(),
            vec!["triangular", "epanechnikov", "uniform"]
        );
        assert_eq!(
            Kernel::choices().canonical(Kernel::Epanechnikov),
            "epanechnikov"
        );
        for v in [Kernel::Triangular, Kernel::Epanechnikov, Kernel::Uniform] {
            assert_eq!(v.to_string().parse::<Kernel>().unwrap(), v);
        }
    }

    #[test]
    fn test_parse_or_defaults_on_none_and_blank() {
        assert_eq!(
            parse_or::<Kernel>(None, Kernel::Uniform).unwrap(),
            Kernel::Uniform
        );
        assert_eq!(
            parse_or(Some(""), Kernel::Uniform).unwrap(),
            Kernel::Uniform
        );
        assert_eq!(
            parse_or(Some("  "), Kernel::Uniform).unwrap(),
            Kernel::Uniform
        );
        assert_eq!(
            parse_or(Some("epa"), Kernel::Uniform).unwrap(),
            Kernel::Epanechnikov
        );
        assert!(parse_or(Some("bogus"), Kernel::Uniform).is_err());
    }

    #[test]
    fn test_choices_validate_detects_ambiguity() {
        let ambiguous = Choices {
            option: "k",
            values: &[(&["tri"], 1u8), (&["Tri"], 2u8)],
        };
        let e = ambiguous.validate().unwrap_err();
        assert!(e.contains("ambiguous"), "{e}");
        let twice = Choices {
            option: "k",
            values: &[(&["a"], 1u8), (&["b"], 1u8)],
        };
        assert!(twice.validate().is_err());
        let no_spelling = Choices {
            option: "k",
            values: &[(&[], 1u8)],
        };
        assert!(no_spelling.validate().is_err());
        let empty = Choices {
            option: "k",
            values: &[(&["--"], 1u8)],
        };
        assert!(empty.validate().is_err());
    }
}
