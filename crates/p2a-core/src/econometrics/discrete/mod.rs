//! Discrete choice models: Logit, Probit, and related models.
//!
//! # Overview
//!
//! This module provides pure Rust implementations of discrete choice models:
//!
//! - **Binary choice**: Logit and Probit for binary outcomes
//! - **Multinomial logit**: For unordered categorical outcomes
//! - **Ordered logit/probit**: For ordered categorical outcomes
//! - **Count models**: Negative binomial, zero-inflated, and hurdle models
//! - **Conditional logit**: McFadden's choice model (mlogit)
//! - **Mixed logit**: Random parameters logit (gmnl, mixl)
//!
//! # Mathematical Background
//!
//! For binary outcomes y ∈ {0, 1}, the latent variable model is:
//!
//! y*_i = X_i'β + ε_i,  y_i = 1[y*_i > 0]
//!
//! - **Logit**: ε follows a logistic distribution
//! - **Probit**: ε follows a standard normal distribution
//!
//! # References
//!
//! - McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior.
//! - Train, K.E. (2009). *Discrete Choice Methods with Simulation* (2nd ed.).
//! - Cameron, A.C. & Trivedi, P.K. (2013). Regression Analysis of Count Data.
//!
//! # R Equivalents
//!
//! - `stats::glm()` with binomial family (logit/probit)
//! - `nnet::multinom()` (multinomial logit)
//! - `MASS::polr()` (ordered logit/probit)
//! - `MASS::glm.nb()` (negative binomial)
//! - `pscl::zeroinfl()`, `pscl::hurdle()` (zero-inflated and hurdle models)
//! - `mlogit::mlogit()` (conditional logit)
//! - `gmnl::gmnl()`, `mixl::mixl()` (mixed logit)

// Module declarations
mod binary_choice;
mod conditional_logit;
mod count_models;
mod mixed_logit;
mod multinomial_logit;
mod ordered_models;
mod types;

// Re-exports for types
pub use types::{DiscreteModelType, DiscreteResult, MleSettings};

// Re-exports for binary choice models
pub use binary_choice::{run_discrete_model, run_logit, run_probit};

// Re-exports for multinomial logit
pub use multinomial_logit::{run_multinom, MultinomResult};

// Re-exports for ordered models
pub use ordered_models::{run_ordered_logit, run_ordered_probit, OrderedModelType, OrderedResult};

// Re-exports for count models
pub use count_models::{
    run_hurdle, run_negbin, run_zinb, run_zip, HurdleResult, HurdleType, NegBinResult,
    ZeroInflResult, ZeroInflatedType,
};

// Re-exports for conditional logit (mlogit)
pub use conditional_logit::{run_conditional_logit, run_mlogit, MlogitResult};

// Re-exports for mixed logit
pub use mixed_logit::{
    run_gmnl, run_mixl, run_mixed_logit, MixedLogitConfig, MixedLogitResult, RandomDistribution,
    RandomParameterSpec,
};

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    use crate::data::Dataset;

    fn create_binary_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_logit_basic() {
        let dataset = create_binary_dataset();
        let result = run_logit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);
        assert!(result.variables.len() >= 1);
        assert!(result.pseudo_r_squared > 0.3);
    }

    #[test]
    fn test_probit_basic() {
        let dataset = create_binary_dataset();
        let result = run_probit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);
        assert!(result.pseudo_r_squared > 0.3);
    }

    #[test]
    fn test_logit_probit_sign_consistency() {
        let dataset = create_binary_dataset();
        let logit_result = run_logit(&dataset, "y", &["x"]).unwrap();
        let probit_result = run_probit(&dataset, "y", &["x"]).unwrap();

        let logit_x = logit_result
            .coefficients
            .iter()
            .zip(&logit_result.variables)
            .find(|(_, v)| *v == "x")
            .map(|(c, _)| *c)
            .unwrap();
        let probit_x = probit_result
            .coefficients
            .iter()
            .zip(&probit_result.variables)
            .find(|(_, v)| *v == "x")
            .map(|(c, _)| *c)
            .unwrap();

        assert!(
            logit_x.signum() == probit_x.signum(),
            "Logit and Probit should have same sign"
        );
    }

    fn create_multinomial_dataset() -> Dataset {
        let df = df! {
            "y" => ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "A", "B"],
            "x" => [1.0, 2.0, 1.5, 4.0, 5.0, 4.5, 7.0, 8.0, 9.0, 8.5, 2.5, 5.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_multinom_basic() {
        let dataset = create_multinomial_dataset();
        let result = run_multinom(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.categories.len(), 3);
        assert_eq!(result.coefficients.len(), 2);
    }

    fn create_ordered_dataset() -> Dataset {
        let df = df! {
            "y" => ["Low", "Low", "Low", "Medium", "Medium", "Medium", "High", "High", "High", "High"],
            "x" => [1.0, 2.0, 1.5, 4.0, 5.0, 4.5, 7.0, 8.0, 9.0, 8.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_ordered_logit_basic() {
        let dataset = create_ordered_dataset();
        let result = run_ordered_logit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.n_obs, 10);
        assert_eq!(result.categories.len(), 3);
        assert_eq!(result.thresholds.len(), 2);
        assert!(result.coefficients[0] > 0.0);
    }

    #[test]
    fn test_ordered_probit_basic() {
        let dataset = create_ordered_dataset();
        let result = run_ordered_probit(&dataset, "y", &["x"]).unwrap();

        assert_eq!(result.model_type, OrderedModelType::Probit);
        assert!(result.coefficients[0] > 0.0);
    }

    fn create_count_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 1.0, 0.0, 2.0, 3.0, 1.0, 5.0, 4.0, 7.0, 8.0, 2.0, 6.0],
            "x" => [1.0, 2.0, 1.5, 3.0, 4.0, 2.5, 5.0, 4.5, 6.0, 7.0, 3.5, 5.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_negbin_basic() {
        let dataset = create_count_dataset();
        let result = run_negbin(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.n_obs, 12);
        assert!(result.theta > 0.0);
    }

    fn create_hurdle_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 3.0, 4.0, 2.0, 5.0, 3.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 3.5, 5.0, 6.0, 4.5, 7.0, 5.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_hurdle_poisson_basic() {
        let dataset = create_hurdle_dataset();
        let result = run_hurdle(&dataset, "y", &["x"], None, HurdleType::Poisson).unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.model_type, HurdleType::Poisson);
        assert_eq!(result.n_zeros, 4);
    }

    fn create_zero_inflated_dataset() -> Dataset {
        let df = df! {
            "y" => [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0, 5.0, 4.0],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 3.5, 6.0, 4.5, 7.0, 6.5]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_zip_basic() {
        let dataset = create_zero_inflated_dataset();
        let result = run_zip(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.n_obs, 12);
        assert_eq!(result.model_type, ZeroInflatedType::Poisson);
    }

    #[test]
    fn test_zinb_basic() {
        let dataset = create_zero_inflated_dataset();
        let result = run_zinb(&dataset, "y", &["x"], None).unwrap();

        assert_eq!(result.model_type, ZeroInflatedType::NegBin);
        assert!(result.theta.is_some());
    }

    fn create_mlogit_dataset() -> Dataset {
        let df = df! {
            "choice_id" => [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "alt_id" => ["car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train", "car", "bus", "train",
                        "car", "bus", "train"],
            "choice" => [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                        1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "cost" => [10.0, 3.0, 5.0, 8.0, 2.0, 4.0, 15.0, 4.0, 3.0,
                      5.0, 5.0, 8.0, 12.0, 2.0, 6.0],
            "time" => [20.0, 40.0, 30.0, 15.0, 35.0, 25.0, 25.0, 45.0, 20.0,
                      10.0, 30.0, 40.0, 20.0, 30.0, 25.0]
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_mlogit_basic() {
        let dataset = create_mlogit_dataset();
        let result = run_mlogit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[],
            None,
        )
        .unwrap();

        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.n_alternatives, 3);
        assert_eq!(result.beta.len(), 2);
        assert!(result.beta[0] < 0.0);
    }

    #[test]
    fn test_conditional_logit() {
        let dataset = create_mlogit_dataset();
        let result = run_conditional_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            None,
        )
        .unwrap();

        assert_eq!(result.n_choice_situations, 5);
        assert_eq!(result.beta.len(), 2);
    }

    #[test]
    fn test_mixed_logit_basic() {
        let dataset = create_mlogit_dataset();

        let result = run_mixed_logit(
            &dataset,
            "choice_id",
            "alt_id",
            "choice",
            &["cost", "time"],
            &[RandomParameterSpec {
                name: "cost".to_string(),
                distribution: RandomDistribution::Normal,
            }],
            Some(MixedLogitConfig {
                n_draws: 50,
                halton: true,
                max_iter: 50,
                tolerance: 1e-4,
                seed: Some(42),
            }),
        )
        .unwrap();

        assert_eq!(result.variable_names.len(), 2);
        assert_eq!(result.n_choice_situations, 5);
        assert!(result.log_likelihood.is_finite());
    }
}
