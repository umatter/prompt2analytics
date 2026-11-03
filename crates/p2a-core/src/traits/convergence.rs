//! Convergence reporting and result notes shared by every iterative estimator (Tier 3 rule R10).
//!
//! An iterative method stores a [`Convergence`], pushes [`Convergence::note`] into its warnings when it
//! stops at the iteration limit, implements [`ResultNotes`], and ends its `Display` with
//! [`notes_text`], so CLI and MCP text output show the same lines. Stopping rules are scale-free:
//! [`relative_change`], [`relative_change_vec`] or [`deviance_change`] (R's `glm.fit`).
//!
//! ```
//! use p2a_core::traits::convergence::{Convergence, relative_change};
//! let c = Convergence::new(true, 7, 100, "relative change in deviance", relative_change(10.0, 10.0), 1e-8);
//! assert_eq!(c.line(), "Converged: yes (7 iterations)");
//! assert!(c.note("logit").is_none());
//! ```

use serde::{Deserialize, Serialize};

/// The outcome of an iterative fit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Convergence {
    /// The stopping rule was met before `max_iterations`.
    pub converged: bool,
    /// Iterations performed.
    pub iterations: usize,
    /// The iteration limit.
    pub max_iterations: usize,
    /// What was compared with `tolerance`, e.g. "relative change in deviance".
    pub criterion: String,
    /// The last value of the criterion (NaN when not computed).
    pub final_change: f64,
    /// The tolerance of the stopping rule.
    pub tolerance: f64,
}

/// C-style `%.1e`: one decimal, a sign and at least two exponent digits (`3.2e-04`).
pub(crate) fn sci(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{x:.1e}");
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            let sign = if e < 0 { '-' } else { '+' };
            format!("{mantissa}e{sign}{:02}", e.unsigned_abs())
        }
        None => s,
    }
}

fn iterations_text(n: usize) -> String {
    if n == 1 {
        "1 iteration".to_string()
    } else {
        format!("{n} iterations")
    }
}

impl Convergence {
    /// Build a record; `criterion` names what `final_change` measures.
    pub fn new(
        converged: bool,
        iterations: usize,
        max_iterations: usize,
        criterion: &str,
        final_change: f64,
        tolerance: f64,
    ) -> Self {
        Self {
            converged,
            iterations,
            max_iterations,
            criterion: criterion.to_string(),
            final_change,
            tolerance,
        }
    }

    /// The warning for a fit that stopped without converging (`None` when it converged).
    pub fn note(&self, method: &str) -> Option<String> {
        if self.converged {
            return None;
        }
        let detail = if self.final_change.is_finite() {
            format!(
                "{} {}, tolerance {}",
                self.criterion,
                sci(self.final_change),
                sci(self.tolerance)
            )
        } else {
            format!(
                "{} not available, tolerance {}",
                self.criterion,
                sci(self.tolerance)
            )
        };
        Some(format!(
            "{method} did not converge after {} ({detail}); the estimates are not at the optimum",
            iterations_text(self.iterations)
        ))
    }

    /// One summary line for Display and CLI text.
    pub fn line(&self) -> String {
        if self.converged {
            format!("Converged: yes ({})", iterations_text(self.iterations))
        } else if self.final_change.is_finite() {
            format!(
                "Converged: no ({}; {} {} > {})",
                iterations_text(self.iterations),
                self.criterion,
                sci(self.final_change),
                sci(self.tolerance)
            )
        } else {
            format!("Converged: no ({})", iterations_text(self.iterations))
        }
    }
}

/// `|new - old| / max(|old|, |new|, 1e-300)`: invariant to rescaling the data; NaN if either is NaN.
pub fn relative_change(new: f64, old: f64) -> f64 {
    (new - old).abs() / old.abs().max(new.abs()).max(1e-300)
}

/// `max_i |new_i - old_i| / max(max_i |old_i|, 1e-300)`; NaN for unequal lengths or any NaN, 0 for empty.
pub fn relative_change_vec(new: &[f64], old: &[f64]) -> f64 {
    if new.len() != old.len() {
        return f64::NAN;
    }
    let (mut num, mut den) = (0.0_f64, 0.0_f64);
    for (&a, &b) in new.iter().zip(old) {
        let d = (a - b).abs();
        if d.is_nan() || b.is_nan() {
            return f64::NAN;
        }
        num = num.max(d);
        den = den.max(b.abs());
    }
    num / den.max(1e-300)
}

/// R `glm.fit`'s criterion `|dev - devold| / (|dev| + 0.1)`.
///
/// `glm.fit`'s (and fixest's) rule. The 0.1 is in deviance units, so the rule is not invariant
/// to rescaling the response or the prior weights. T3-WP01's GLM uses a scaled floor instead
/// (its re-scope amendment, WP01-8). For other estimators, [`relative_change`] is the
/// scale-free default.
pub fn deviance_change(dev: f64, devold: f64) -> f64 {
    (dev - devold).abs() / (dev.abs() + 0.1)
}

/// Uniform access to a result's warnings and convergence for Display, CLI and MCP output.
pub trait ResultNotes {
    /// Warnings to show, in order.
    fn notes_warnings(&self) -> Vec<String>;
    /// The convergence record of an iterative fit.
    fn notes_convergence(&self) -> Option<Convergence> {
        None
    }
}

/// Lines appended at the end of a Display: the convergence line (if any), then `Warning: <text>`
/// per warning; every line ends with '\n'; empty when there is nothing to report.
pub fn notes_text(r: &dyn ResultNotes) -> String {
    let mut out = String::new();
    if let Some(c) = r.notes_convergence() {
        out.push_str(&c.line());
        out.push('\n');
    }
    for w in r.notes_warnings() {
        out.push_str("Warning: ");
        out.push_str(&w);
        out.push('\n');
    }
    out
}

/// `{"converged": bool, "iterations": n}` when a convergence record exists and `"warnings": [..]`
/// when there are warnings; an empty object otherwise.
pub fn notes_json(r: &dyn ResultNotes) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    if let Some(c) = r.notes_convergence() {
        map.insert(
            "converged".to_string(),
            serde_json::Value::Bool(c.converged),
        );
        map.insert(
            "iterations".to_string(),
            serde_json::Value::from(c.iterations),
        );
    }
    let warnings = r.notes_warnings();
    if !warnings.is_empty() {
        map.insert("warnings".to_string(), serde_json::Value::from(warnings));
    }
    serde_json::Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_convergence_note_and_line() {
        let c = Convergence::new(false, 100, 100, "relative change in deviance", 3.2e-4, 1e-8);
        assert_eq!(
            c.note("logit").as_deref(),
            Some(
                "logit did not converge after 100 iterations (relative change in deviance 3.2e-04, \
                 tolerance 1.0e-08); the estimates are not at the optimum"
            )
        );
        assert_eq!(
            c.line(),
            "Converged: no (100 iterations; relative change in deviance 3.2e-04 > 1.0e-08)"
        );
        let ok = Convergence::new(true, 12, 100, "relative change in deviance", 4e-10, 1e-8);
        assert_eq!(ok.note("logit"), None);
        assert_eq!(ok.line(), "Converged: yes (12 iterations)");
        assert_eq!(
            Convergence::new(true, 1, 25, "x", 0.0, 1e-8).line(),
            "Converged: yes (1 iteration)"
        );
        let n = Convergence::new(false, 50, 50, "gradient norm", f64::NAN, 1e-6);
        assert_eq!(
            n.note("GARCH").as_deref(),
            Some(
                "GARCH did not converge after 50 iterations (gradient norm not available, \
                 tolerance 1.0e-06); the estimates are not at the optimum"
            )
        );
        assert_eq!(n.line(), "Converged: no (50 iterations)");
        let d = Convergence::default();
        assert!(!d.converged);
        assert_eq!(d.iterations, 0);
    }

    #[test]
    fn test_sci_matches_c_format() {
        assert_eq!(sci(3.2e-4), "3.2e-04");
        assert_eq!(sci(1e-8), "1.0e-08");
        assert_eq!(sci(150.0), "1.5e+02");
        assert_eq!(sci(0.0), "0.0e+00");
        assert_eq!(sci(-2.5e10), "-2.5e+10");
        assert_eq!(sci(f64::INFINITY), "inf");
        assert_eq!(sci(f64::NAN), "NaN");
    }

    #[test]
    fn test_relative_change_is_scale_free() {
        assert_eq!(relative_change(3.0, 2.0), 1.0 / 3.0);
        for s in [1e-6, 1.0, 1e6] {
            assert!((relative_change(3.0 * s, 2.0 * s) - 1.0 / 3.0).abs() <= 1e-15);
        }
        assert_eq!(relative_change(0.0, 0.0), 0.0);
        assert_eq!(relative_change(-2.0, 2.0), 2.0);
        assert!(relative_change(f64::NAN, 1.0).is_nan());
    }

    #[test]
    fn test_relative_change_vec_and_deviance_change() {
        assert_eq!(relative_change_vec(&[1.0, 4.0], &[1.0, 2.0]), 1.0);
        for s in [1e-6, 1e6] {
            let r = relative_change_vec(&[1.0 * s, 4.0 * s], &[1.0 * s, 2.0 * s]);
            assert!((r - 1.0).abs() <= 1e-15, "{r}");
        }
        assert!(relative_change_vec(&[1.0], &[1.0, 2.0]).is_nan());
        assert_eq!(relative_change_vec(&[], &[]), 0.0);
        assert!(relative_change_vec(&[1.0, f64::NAN], &[1.0, 2.0]).is_nan());
        assert!(relative_change_vec(&[1.0, 2.0], &[f64::NAN, 2.0]).is_nan());
        // glm.fit: abs(dev - devold) / (abs(dev) + 0.1)
        assert_eq!(deviance_change(10.0, 10.5), 0.5 / 10.1);
        assert_eq!(deviance_change(10.0, 10.5), 0.04950495049504951);
    }

    struct Fit {
        w: Vec<String>,
        c: Option<Convergence>,
    }

    impl ResultNotes for Fit {
        fn notes_warnings(&self) -> Vec<String> {
            self.w.clone()
        }
        fn notes_convergence(&self) -> Option<Convergence> {
            self.c.clone()
        }
    }

    #[test]
    fn test_notes_text_and_json() {
        let full = Fit {
            w: vec!["separation".to_string()],
            c: Some(Convergence::new(
                false,
                100,
                100,
                "relative change in deviance",
                3.2e-4,
                1e-8,
            )),
        };
        assert_eq!(
            notes_text(&full),
            "Converged: no (100 iterations; relative change in deviance 3.2e-04 > 1.0e-08)\n\
             Warning: separation\n"
        );
        assert_eq!(
            notes_json(&full),
            json!({"converged": false, "iterations": 100, "warnings": ["separation"]})
        );
        let empty = Fit { w: vec![], c: None };
        assert_eq!(notes_text(&empty), "");
        assert_eq!(notes_json(&empty), json!({}));
        let warn = Fit {
            w: vec!["a".to_string(), "b".to_string()],
            c: None,
        };
        assert_eq!(notes_json(&warn), json!({"warnings": ["a", "b"]}));
        assert_eq!(notes_text(&warn), "Warning: a\nWarning: b\n");
    }
}
