//! Shared traits and utilities for econometric estimators.

pub mod convergence;
pub mod estimator;

pub use convergence::{
    Convergence, ResultNotes, deviance_change, notes_json, notes_text, relative_change,
    relative_change_vec,
};

pub use estimator::{
    LinearEstimator, SignificanceLevel, chi_squared_p_value, f_test_p_value, logistic_cdf,
    logistic_pdf, normal_cdf, normal_pdf, t_critical, t_test_p_value,
};
