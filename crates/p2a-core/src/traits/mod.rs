//! Shared traits and utilities for econometric estimators.

pub mod estimator;

pub use estimator::{
    LinearEstimator, SignificanceLevel,
    t_test_p_value, chi_squared_p_value, f_test_p_value, t_critical,
    normal_cdf, normal_pdf, logistic_cdf, logistic_pdf,
};
