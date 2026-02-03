//! Shared traits and utilities for econometric estimators.

pub mod estimator;

pub use estimator::{
    LinearEstimator, SignificanceLevel, chi_squared_p_value, f_test_p_value, logistic_cdf,
    logistic_pdf, normal_cdf, normal_pdf, t_critical, t_test_p_value,
};
