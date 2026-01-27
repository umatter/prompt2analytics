//! Panel data estimators: Fixed Effects (FE) and Random Effects (RE).
//!
//! Pure Rust implementation without external formula parsing.
//! Uses column-based API for simplicity.
//!
//! # Mathematical Background
//!
//! For panel data with observations yᵢₜ for entity i at time t:
//!
//! yᵢₜ = αᵢ + Xᵢₜ'β + εᵢₜ
//!
//! ## Fixed Effects (Within) Estimator
//!
//! The FE estimator demeans the data within each entity:
//!
//! (yᵢₜ - ȳᵢ) = (Xᵢₜ - X̄ᵢ)'β + (εᵢₜ - ε̄ᵢ)
//!
//! This eliminates time-invariant unobserved heterogeneity αᵢ.
//!
//! ## Random Effects (GLS) Estimator
//!
//! The RE estimator assumes αᵢ is uncorrelated with Xᵢₜ and uses quasi-demeaning:
//!
//! (yᵢₜ - θȳᵢ) = (1-θ)α + (Xᵢₜ - θX̄ᵢ)'β + (εᵢₜ - θε̄ᵢ)
//!
//! where θ = 1 - √(σ²ₑ / (σ²ₑ + Tσ²ᵤ))
//!
//! ## Hausman Test
//!
//! Tests H₀: RE is consistent (Cov(αᵢ, Xᵢₜ) = 0) vs H₁: FE is required.
//!
//! H = (β̂ᶠᴱ - β̂ᴿᴱ)'(V̂ᶠᴱ - V̂ᴿᴱ)⁻¹(β̂ᶠᴱ - β̂ᴿᴱ) ~ χ²(k)
//!
//! # References
//!
//! - Mundlak, Y. (1978). On the pooling of time series and cross section data.
//!   *Econometrica*, 46(1), 69-85. https://doi.org/10.2307/1913646
//!
//! - Hausman, J.A. (1978). Specification tests in econometrics. *Econometrica*,
//!   46(6), 1251-1271. https://doi.org/10.2307/1913827
//!
//! - Baltagi, B.H. (2013). *Econometric Analysis of Panel Data* (5th ed.).
//!   Wiley. ISBN: 978-1118672327.
//!
//! - Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*
//!   (2nd ed.), Chapters 10-11. MIT Press.
//!
//! - Arellano, M. (2003). *Panel Data Econometrics*. Oxford University Press.
//!   ISBN: 978-0199245291.
//!
//! R equivalent: `plm::plm()` with `model = "within"` or `model = "random"`,
//! `plm::phtest()` for Hausman test

// Submodules
mod types;
mod utils;
mod linear_models;
mod specification_test;
mod gls_models;
mod dynamic_panel;
mod heterogeneous;

// Re-export core types
pub use types::{PanelResult, PanelMethod};

// Re-export linear models (FE/RE)
pub use linear_models::{run_fixed_effects, run_random_effects};

// Re-export Hausman test
pub use specification_test::{HausmanResult, run_hausman_test};

// Re-export Panel GLS (FGLS)
pub use gls_models::{PanelGlsResult, PanelGlsModel, run_panel_gls, run_fegls, run_pooled_gls};

// Re-export Arellano-Bond / System GMM
pub use dynamic_panel::{GmmResult, GmmConfig, GmmTransform, GmmStep, run_gmm, run_arellano_bond};

// Re-export Variable Coefficients Model (pvcm) and Mean Group (pmg)
pub use heterogeneous::{PvcmResult, PvcmType, run_pvcm, run_pmg};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Dataset;
    use polars::prelude::*;

    fn create_panel_dataset() -> Dataset {
        // Simple panel: 3 entities, 4 time periods each
        // y = 2*x + entity_effect + noise
        // Entity effects: A=0, B=5, C=10
        let df = df! {
            "entity" => ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C", "C", "C"],
            "time" => [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "x" => [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 4.2, 5.9, 8.1,   // A: y ≈ 2x + 0
                    7.0, 9.1, 10.9, 13.2,  // B: y ≈ 2x + 5
                    12.2, 13.8, 16.1, 17.9] // C: y ≈ 2x + 10
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_fixed_effects_basic() {
        let dataset = create_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure
        assert_eq!(result.method, PanelMethod::FixedEffects);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_groups, 3);
        assert_eq!(result.variables.len(), 1); // x only, no intercept in FE

        // The true coefficient is 2.0
        // With noise, should be close to 2.0
        assert!((result.coefficients[0] - 2.0).abs() < 0.3,
            "FE coefficient should be close to 2.0, got {}", result.coefficients[0]);

        // R-squared should be high (good fit within entities)
        assert!(result.r_squared > 0.9, "R² should be high, got {}", result.r_squared);
    }

    #[test]
    fn test_random_effects_basic() {
        let dataset = create_panel_dataset();
        let result = run_random_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure
        assert_eq!(result.method, PanelMethod::RandomEffects);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_groups, 3);

        // RE coefficient should be positive (x positively affects y)
        // Note: RE combines within and between variation, so coefficient may differ from FE
        assert!(result.coefficients[0] > 0.0,
            "RE coefficient should be positive, got {}", result.coefficients[0]);

        // R-squared should be positive (RE uses different R² calculation)
        assert!(result.r_squared > 0.0, "R² should be positive, got {}", result.r_squared);
    }

    #[test]
    fn test_hausman_test() {
        let dataset = create_panel_dataset();
        let result = run_hausman_test(&dataset, "y", &["x"], "entity").unwrap();

        // Hausman test should produce FE and RE results
        assert!(!result.fe_result.coefficients.is_empty());
        assert!(!result.re_result.coefficients.is_empty());

        // FE coefficient should be close to 2.0 (within variation)
        assert!((result.fe_result.coefficients[0] - 2.0).abs() < 0.3,
            "FE coefficient should be close to 2.0, got {}", result.fe_result.coefficients[0]);

        // Chi-squared statistic should be non-negative (or NaN if variance matrix issues)
        assert!(result.chi2_statistic >= 0.0 || result.chi2_statistic.is_nan());

        // Should have a recommendation
        assert!(!result.recommendation.is_empty());
    }

    #[test]
    fn test_panel_missing_column() {
        let dataset = create_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["nonexistent"], "entity");
        assert!(result.is_err());
    }

    #[test]
    fn test_panel_missing_entity() {
        let dataset = create_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["x"], "nonexistent");
        assert!(result.is_err());
    }

    // =====================================================================
    // Unbalanced Panel Tests (Cameron-Miller validation)
    // =====================================================================

    fn create_unbalanced_panel_dataset() -> Dataset {
        // Unbalanced panel: 3 entities with different numbers of time periods
        // Entity A: 5 periods, Entity B: 3 periods, Entity C: 4 periods
        // y = 2*x + entity_effect + noise
        let df = df! {
            "entity" => ["A", "A", "A", "A", "A",   // 5 periods
                         "B", "B", "B",              // 3 periods
                         "C", "C", "C", "C"],        // 4 periods
            "time" => [1, 2, 3, 4, 5,
                       1, 2, 3,
                       1, 2, 3, 4],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0,
                    1.0, 2.0, 3.0,
                    1.0, 2.0, 3.0, 4.0],
            "y" => [2.1, 4.2, 5.9, 8.1, 9.8,      // A: y ≈ 2x + 0
                    7.0, 9.1, 10.9,                // B: y ≈ 2x + 5
                    12.2, 13.8, 16.1, 17.9]        // C: y ≈ 2x + 10
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_fixed_effects_unbalanced() {
        let dataset = create_unbalanced_panel_dataset();
        let result = run_fixed_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure with unbalanced data
        assert_eq!(result.method, PanelMethod::FixedEffects);
        assert_eq!(result.n_obs, 12); // 5 + 3 + 4
        assert_eq!(result.n_groups, 3);

        // Degrees of freedom: n - n_groups - k = 12 - 3 - 1 = 8
        assert_eq!(result.df, 8);

        // Coefficient should still be close to 2.0
        assert!((result.coefficients[0] - 2.0).abs() < 0.5,
            "FE coefficient should be close to 2.0 with unbalanced panel, got {}", result.coefficients[0]);

        // R-squared should be high
        assert!(result.r_squared > 0.8, "R² should be high with unbalanced panel, got {}", result.r_squared);
    }

    #[test]
    fn test_random_effects_unbalanced() {
        let dataset = create_unbalanced_panel_dataset();
        let result = run_random_effects(&dataset, "y", &["x"], "entity").unwrap();

        // Check structure with unbalanced data
        assert_eq!(result.method, PanelMethod::RandomEffects);
        assert_eq!(result.n_obs, 12);
        assert_eq!(result.n_groups, 3);

        // RE coefficient should be positive
        assert!(result.coefficients[0] > 0.0,
            "RE coefficient should be positive with unbalanced panel, got {}", result.coefficients[0]);

        // Theta (quasi-demeaning factor) should be between 0 and 1
        if let Some(theta) = result.theta {
            assert!(theta >= 0.0 && theta <= 1.0,
                "Theta should be in [0, 1], got {}", theta);
        }
    }

    // =====================================================================
    // Panel GLS (FGLS) Tests
    // =====================================================================

    fn create_gls_panel_dataset() -> Dataset {
        // Panel with serial correlation in errors
        // 5 entities, 6 time periods
        let df = df! {
            "entity" => ["A", "A", "A", "A", "A", "A",
                        "B", "B", "B", "B", "B", "B",
                        "C", "C", "C", "C", "C", "C",
                        "D", "D", "D", "D", "D", "D",
                        "E", "E", "E", "E", "E", "E"],
            "time" => [1i64, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6,
                      1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6],
            "x" => [1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
                   1.2, 1.7, 2.2, 2.7, 3.2, 3.7,
                   0.8, 1.3, 1.8, 2.3, 2.8, 3.3,
                   1.1, 1.6, 2.1, 2.6, 3.1, 3.6,
                   0.9, 1.4, 1.9, 2.4, 2.9, 3.4],
            // y = 2*x + entity_effect + correlated_error
            "y" => [2.1, 3.2, 4.1, 5.2, 6.3, 7.1,   // A: alpha=0
                   4.5, 5.4, 6.6, 7.5, 8.5, 9.6,    // B: alpha=2
                   1.4, 2.5, 3.4, 4.6, 5.4, 6.5,    // C: alpha=-0.5
                   3.3, 4.2, 5.4, 6.3, 7.4, 8.3,    // D: alpha=1
                   0.7, 1.9, 2.7, 3.8, 4.8, 5.7]    // E: alpha=-1
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_panel_gls_fe() {
        let dataset = create_gls_panel_dataset();
        let result = run_panel_gls(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
            Some(PanelGlsModel::FixedEffects)
        );

        assert!(result.is_ok(), "Panel GLS FE should succeed, got: {:?}", result.err());
        let result = result.unwrap();

        assert_eq!(result.model, PanelGlsModel::FixedEffects);
        assert_eq!(result.n_obs, 30);
        assert_eq!(result.n_groups, 5);
        assert_eq!(result.n_periods, 6);

        // Coefficient should be close to 2.0
        assert!(!result.coefficients.is_empty());
        assert!((result.coefficients[0] - 2.0).abs() < 0.5,
            "Coefficient should be close to 2.0, got {}", result.coefficients[0]);
    }

    #[test]
    fn test_panel_gls_pooling() {
        let dataset = create_gls_panel_dataset();
        let result = run_pooled_gls(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
        );

        assert!(result.is_ok(), "Pooled GLS should succeed, got: {:?}", result.err());
        let result = result.unwrap();

        assert_eq!(result.model, PanelGlsModel::Pooling);
        // Should have intercept + x
        assert_eq!(result.variables.len(), 2);
        assert!(result.variables.contains(&"(Intercept)".to_string()));
    }

    // =====================================================================
    // GMM Tests
    // =====================================================================

    fn create_gmm_panel_dataset() -> Dataset {
        // Create a dynamic panel: 10 entities, 8 time periods each
        let mut entities = Vec::new();
        let mut times = Vec::new();
        let mut xs = Vec::new();
        let mut ys = Vec::new();

        let entity_effects = [0.0, 1.0, -0.5, 0.5, -1.0, 2.0, 0.3, -0.3, 0.8, -0.8];
        let n_entities = 10;
        let n_periods = 8;

        let noise_values = [
            0.1, -0.2, 0.15, -0.1, 0.05, -0.05, 0.2, -0.15, 0.12, -0.08,
            0.08, -0.12, 0.18, -0.18, 0.03, -0.07, 0.14, -0.14, 0.09, -0.06,
            0.11, -0.11, 0.16, -0.16, 0.04, -0.09, 0.13, -0.13, 0.07, -0.04,
            0.06, -0.03, 0.17, -0.17, 0.02, -0.08, 0.19, -0.19, 0.08, -0.01,
            0.05, -0.05, 0.12, -0.12, 0.09, -0.09, 0.15, -0.15, 0.04, -0.04,
            0.07, -0.07, 0.11, -0.11, 0.06, -0.06, 0.14, -0.14, 0.03, -0.03,
            0.08, -0.08, 0.13, -0.13, 0.07, -0.07, 0.16, -0.16, 0.02, -0.02,
            0.09, -0.09, 0.14, -0.14, 0.05, -0.05, 0.17, -0.17, 0.01, -0.01
        ];

        let x_values = [
            1.0, 1.5, 2.0, 1.8, 2.2, 2.5, 2.3, 2.7,
            1.2, 1.7, 2.1, 1.9, 2.3, 2.6, 2.4, 2.8,
            0.8, 1.3, 1.8, 1.6, 2.0, 2.3, 2.1, 2.5,
            1.1, 1.6, 2.2, 2.0, 2.4, 2.7, 2.5, 2.9,
            0.9, 1.4, 1.9, 1.7, 2.1, 2.4, 2.2, 2.6,
            1.3, 1.8, 2.3, 2.1, 2.5, 2.8, 2.6, 3.0,
            0.7, 1.2, 1.7, 1.5, 1.9, 2.2, 2.0, 2.4,
            1.0, 1.5, 2.0, 1.8, 2.2, 2.5, 2.3, 2.7,
            1.4, 1.9, 2.4, 2.2, 2.6, 2.9, 2.7, 3.1,
            0.6, 1.1, 1.6, 1.4, 1.8, 2.1, 1.9, 2.3
        ];

        for i in 0..n_entities {
            let alpha = entity_effects[i];
            let mut y_prev = 2.0 + alpha;

            for t in 0..n_periods {
                let idx = i * n_periods + t;
                let x = x_values[idx];
                let noise = noise_values[idx];

                let y = 0.5 * y_prev + 1.5 * x + alpha + noise;

                entities.push(format!("E{}", i));
                times.push((t + 1) as i64);
                xs.push(x);
                ys.push(y);

                y_prev = y;
            }
        }

        let df = df! {
            "entity" => entities,
            "time" => times,
            "x" => xs,
            "y" => ys
        }.unwrap();

        Dataset::new(df)
    }

    #[test]
    fn test_arellano_bond_convenience() {
        let dataset = create_gmm_panel_dataset();

        let result = run_arellano_bond(
            &dataset,
            "y",
            &["x"],
            "entity",
            "time",
        );

        assert!(result.is_ok(), "Arellano-Bond should succeed, got error: {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.transform, GmmTransform::Difference);
    }

    // =====================================================================
    // PVCM Tests
    // =====================================================================

    fn create_pvcm_dataset() -> Dataset {
        let df = df! {
            "entity" => ["A", "A", "A", "A", "A", "A",
                        "B", "B", "B", "B", "B", "B",
                        "C", "C", "C", "C", "C", "C"],
            "x" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
                   1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "y" => [3.1, 5.0, 6.9, 9.1, 10.8, 13.0,   // A: y ≈ 1 + 2x
                   5.0, 8.1, 10.9, 14.0, 17.2, 20.0,  // B: y ≈ 2 + 3x
                   4.6, 6.0, 7.4, 9.0, 10.6, 12.1]    // C: y ≈ 3 + 1.5x
        }.unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_pvcm_within_basic() {
        let dataset = create_pvcm_dataset();
        let result = run_pvcm(&dataset, "y", &["x"], "entity", PvcmType::Within);

        assert!(result.is_ok(), "pvcm within should succeed, got {:?}", result.err());

        let result = result.unwrap();

        assert_eq!(result.model_type, PvcmType::Within);
        assert_eq!(result.n_obs, 18);
        assert_eq!(result.n_entities, 3);
        assert_eq!(result.variables.len(), 2); // intercept + x

        assert!(result.individual_coefficients.contains_key("A"));
        assert!(result.individual_coefficients.contains_key("B"));
        assert!(result.individual_coefficients.contains_key("C"));
    }

    #[test]
    fn test_pmg_basic() {
        let dataset = create_pvcm_dataset();
        let result = run_pmg(&dataset, "y", &["x"], "entity");

        assert!(result.is_ok(), "pmg should succeed, got {:?}", result.err());

        let result = result.unwrap();
        assert_eq!(result.model_type, PvcmType::Within);
    }
}
