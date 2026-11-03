//! Shared traits and utilities for econometric estimators.

pub mod convergence;
pub mod estimator;

pub use convergence::{
    Convergence, ResultNotes, deviance_change, notes_json, notes_text, relative_change,
    relative_change_vec,
};

pub use estimator::{
    LinearEstimator, SignificanceLevel, chi_squared_p_value, chi_squared_quantile, f_test_p_value,
    logistic_cdf, logistic_pdf, normal_cdf, normal_p_value_two_sided, normal_pdf, normal_quantile,
    t_critical, t_quantile, t_test_p_value, validate_level, z_critical,
};
