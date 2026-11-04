//! Discrete-choice estimators through the crate-root API, against R reference values
//! (validation/r/cases/discrete_mlogit_asc.R). Each estimator's own module pins it in detail;
//! this file checks the public entry points on R cases.

use p2a_core::data::{DataLoader, Dataset};
use p2a_core::run_multinom;
use p2a_validation::{Case, Check, Tol};

fn case_with_data(id: &str) -> (Case, Dataset) {
    let case = Case::load(id);
    let df = DataLoader::load_csv(case.dataset_path()).expect("case dataset loads");
    (case, Dataset::new(df))
}

/// nnet::multinom(y ~ x1 + x2) with reference category "B" on overlapping data (finite MLE),
/// Newton-polished in the R case.
#[test]
fn test_validate_multinom_overlapping() {
    let (case, ds) = case_with_data("disc_multinom_overlap");
    let r = run_multinom(&ds, "y", &["x1", "x2"], Some("B")).unwrap();
    assert_eq!(r.reference_category, "B");
    assert_eq!(r.categories, ["B", "A", "C"]);
    let mut names = Vec::new();
    let (mut coef, mut se) = (Vec::new(), Vec::new());
    for (j, cat) in r.categories.iter().skip(1).enumerate() {
        for (k, v) in r.variables.iter().enumerate() {
            names.push(format!("{cat}:{v}"));
            coef.push(r.coefficients[j][k]);
            se.push(r.std_errors[j][k]);
        }
    }
    Check::new(&case)
        .scalar("n", r.n_obs as f64, Tol::exact())
        .named("refB_coef_tight", &names, &coef, Tol::iterative(1e-8))
        .named("refB_se_tight", &names, &se, Tol::iterative(1e-6))
        .scalar("loglik", r.log_likelihood, Tol::stat(300)) // invariant to the reference
        .finish();
    assert!(r.converged && !r.separation && r.warnings.is_empty());
}
