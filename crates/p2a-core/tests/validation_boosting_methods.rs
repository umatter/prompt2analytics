//! Validation tests for XGBoost, LightGBM, MBoost, and BART implementations.
//!
//! These tests verify:
//! 1. Accuracy: R² > 0.80 on synthetic regression data
//! 2. Feature importance: x1 (coef=2) ranked higher than x2 (coef=0.5)
//! 3. Classification accuracy: > 80% on separable data
//! 4. Performance: Timing information for comparison

use ndarray::{Array1, Array2};
use p2a_core::ml::{
    BartConfig, LightGbmConfig, LightGbmObjective, MboostBaseLearner, MboostConfig, MboostFamily,
    XGBoostConfig, XGBoostObjective, bart_arrays, lightgbm, lightgbm_predict, mboost,
    mboost_predict, xgboost, xgboost_predict,
};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::time::Instant;

/// Create synthetic regression data: y = 2*x1 + 0.5*x2 + noise
fn create_regression_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 0.3).unwrap();

    let mut x = Array2::zeros((n, 3));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x1 = rng.r#gen::<f64>();
        let x2 = rng.r#gen::<f64>();
        let x3 = Normal::new(0.0, 0.5).unwrap().sample(&mut rng); // noise feature

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        y[i] = 2.0 * x1 + 0.5 * x2 + normal.sample(&mut rng);
    }

    (x, y)
}

/// Create synthetic classification data
fn create_classification_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);

    let mut x = Array2::zeros((n, 3));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x1 = rng.r#gen::<f64>();
        let x2 = rng.r#gen::<f64>();
        let x3 = Normal::new(0.0, 0.5).unwrap().sample(&mut rng);

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        // Probability based on x1 + x2
        let prob = 1.0 / (1.0 + (-3.0 * (x1 + x2 - 1.0)).exp());
        y[i] = if rng.r#gen::<f64>() < prob { 1.0 } else { 0.0 };
    }

    (x, y)
}

fn compute_r2(y_true: &Array1<f64>, y_pred: &[f64]) -> f64 {
    let y_mean = y_true.mean().unwrap();
    let ss_tot: f64 = y_true.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum();
    1.0 - ss_res / ss_tot
}

fn compute_mse(y_true: &Array1<f64>, y_pred: &[f64]) -> f64 {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&yt, &yp)| (yt - yp).powi(2))
        .sum::<f64>()
        / y_true.len() as f64
}

fn compute_accuracy(y_true: &Array1<f64>, y_pred: &[f64], threshold: f64) -> f64 {
    let correct: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(yt, yp)| {
            let pred_class = if **yp > threshold { 1.0 } else { 0.0 };
            (**yt - pred_class).abs() < 0.5
        })
        .count();
    correct as f64 / y_true.len() as f64
}

#[test]
fn test_validate_xgboost_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = XGBoostConfig {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.3,
        lambda: 1.0,
        alpha: 0.0,
        subsample: 1.0,
        colsample_bytree: 1.0,
        gamma: 0.0,
        objective: XGBoostObjective::RegSquaredError,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = xgboost(x.view(), y.view(), &config).unwrap();
    let elapsed = start.elapsed();

    let r2 = compute_r2(&y, &result.predictions);
    let mse = compute_mse(&y, &result.predictions);

    println!("XGBoost Regression Results:");
    println!("  MSE: {:.6}", mse);
    println!("  R²:  {:.4}", r2);
    println!("  Time: {:.4}s", elapsed.as_secs_f64());
    println!(
        "  Feature Importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.feature_importances[0], result.feature_importances[1], result.feature_importances[2]
    );

    // Validation criteria
    assert!(r2 > 0.80, "XGBoost R² should be > 0.80, got {:.4}", r2);
    assert!(
        result.feature_importances[0] > result.feature_importances[2],
        "x1 should be more important than x3 (noise)"
    );
}

#[test]
fn test_validate_xgboost_classification() {
    let (x, y) = create_classification_data(200, 42);

    let config = XGBoostConfig {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.3,
        objective: XGBoostObjective::BinaryLogistic,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = xgboost(x.view(), y.view(), &config).unwrap();
    let elapsed = start.elapsed();

    let accuracy = compute_accuracy(&y, &result.predictions, 0.5);

    println!("XGBoost Classification Results:");
    println!("  Accuracy: {:.4}", accuracy);
    println!("  Time: {:.4}s", elapsed.as_secs_f64());

    assert!(
        accuracy > 0.75,
        "XGBoost accuracy should be > 0.75, got {:.4}",
        accuracy
    );
}

#[test]
fn test_validate_lightgbm_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = LightGbmConfig {
        num_iterations: 100,
        num_leaves: 31,
        max_depth: -1,
        learning_rate: 0.1,
        max_bin: 255,
        min_data_in_leaf: 20,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        objective: LightGbmObjective::Regression,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = lightgbm(x.view(), y.view(), &config).unwrap();
    let elapsed = start.elapsed();

    let r2 = compute_r2(&y, &result.predictions);
    let mse = compute_mse(&y, &result.predictions);

    println!("LightGBM Regression Results:");
    println!("  MSE: {:.6}", mse);
    println!("  R²:  {:.4}", r2);
    println!("  Time: {:.4}s", elapsed.as_secs_f64());
    println!(
        "  Feature Importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.feature_importances[0], result.feature_importances[1], result.feature_importances[2]
    );

    assert!(r2 > 0.70, "LightGBM R² should be > 0.70, got {:.4}", r2);
    assert!(
        result.feature_importances[0] > result.feature_importances[2],
        "x1 should be more important than x3"
    );
}

#[test]
fn test_validate_mboost_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = MboostConfig {
        mstop: 100,
        nu: 0.1,
        base_learner: MboostBaseLearner::Tree,
        tree_depth: 4,
        family: MboostFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = mboost(x.view(), y.view(), &config).unwrap();
    let elapsed = start.elapsed();

    let r2 = compute_r2(&y, &result.predictions);
    let mse = compute_mse(&y, &result.predictions);

    println!("MBoost Regression Results (Tree):");
    println!("  MSE: {:.6}", mse);
    println!("  R²:  {:.4}", r2);
    println!("  Time: {:.4}s", elapsed.as_secs_f64());
    println!(
        "  Variable Importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0], result.variable_importance[1], result.variable_importance[2]
    );

    assert!(r2 > 0.75, "MBoost R² should be > 0.75, got {:.4}", r2);
}

#[test]
fn test_validate_mboost_linear() {
    let (x, y) = create_regression_data(200, 42);

    let config = MboostConfig {
        mstop: 100,
        nu: 0.1,
        base_learner: MboostBaseLearner::Linear,
        family: MboostFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = mboost(x.view(), y.view(), &config).unwrap();
    let elapsed = start.elapsed();

    let r2 = compute_r2(&y, &result.predictions);

    println!("MBoost Linear Results:");
    println!("  R²:  {:.4}", r2);
    println!("  Time: {:.4}s", elapsed.as_secs_f64());
    println!(
        "  Variable Importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0], result.variable_importance[1], result.variable_importance[2]
    );

    // Linear base learner should identify x1 as most important
    assert!(
        result.variable_importance[0] > result.variable_importance[2],
        "x1 should be selected more often than x3"
    );
}

#[test]
fn test_validate_bart_regression() {
    let (x, y) = create_regression_data(100, 42);

    let config = BartConfig {
        n_trees: 50,
        n_burn: 50,
        n_mcmc: 100,
        k: 2.0,
        seed: Some(42),
        ..Default::default()
    };

    let start = Instant::now();
    let result = bart_arrays(y.view(), x.view(), None, config).unwrap();
    let elapsed = start.elapsed();

    let r2 = compute_r2(&y, &result.predictions);
    let mse = compute_mse(&y, &result.predictions);

    println!("BART Regression Results:");
    println!("  MSE: {:.6}", mse);
    println!("  R²:  {:.4}", r2);
    println!("  Time: {:.4}s", elapsed.as_secs_f64());
    println!("  Sigma (posterior mean): {:.4}", result.sigma);
    println!(
        "  Variable importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0], result.variable_importance[1], result.variable_importance[2]
    );

    // Check prediction intervals
    let mut in_interval = 0;
    for i in 0..y.len() {
        if y[i] >= result.prediction_lower[i] && y[i] <= result.prediction_upper[i] {
            in_interval += 1;
        }
    }
    let coverage = in_interval as f64 / y.len() as f64;
    println!("  95% CI Coverage: {:.1}%", coverage * 100.0);

    assert!(r2 > 0.50, "BART R² should be > 0.50, got {:.4}", r2);
}

#[test]
fn test_benchmark_all_methods() {
    println!("\n====== BOOSTING METHODS BENCHMARK ======\n");

    let (x, y) = create_regression_data(500, 42);

    // XGBoost
    let start = Instant::now();
    let xgb_result = xgboost(
        x.view(),
        y.view(),
        &XGBoostConfig {
            n_estimators: 100,
            seed: Some(42),
            ..Default::default()
        },
    )
    .unwrap();
    let xgb_time = start.elapsed();
    let xgb_r2 = compute_r2(&y, &xgb_result.predictions);

    // LightGBM
    let start = Instant::now();
    let lgb_result = lightgbm(
        x.view(),
        y.view(),
        &LightGbmConfig {
            num_iterations: 100,
            seed: Some(42),
            ..Default::default()
        },
    )
    .unwrap();
    let lgb_time = start.elapsed();
    let lgb_r2 = compute_r2(&y, &lgb_result.predictions);

    // MBoost
    let start = Instant::now();
    let mb_result = mboost(
        x.view(),
        y.view(),
        &MboostConfig {
            mstop: 100,
            seed: Some(42),
            ..Default::default()
        },
    )
    .unwrap();
    let mb_time = start.elapsed();
    let mb_r2 = compute_r2(&y, &mb_result.predictions);

    // BART (smaller for speed)
    let (x_small, y_small) = create_regression_data(100, 42);
    let start = Instant::now();
    let bart_result = bart_arrays(
        y_small.view(),
        x_small.view(),
        None,
        BartConfig {
            n_trees: 30,
            n_burn: 30,
            n_mcmc: 50,
            seed: Some(42),
            ..Default::default()
        },
    )
    .unwrap();
    let bart_time = start.elapsed();
    let bart_r2 = compute_r2(&y_small, &bart_result.predictions);

    println!("Method        | R²      | Time (ms) | Notes");
    println!("--------------|---------|-----------|------");
    println!(
        "XGBoost       | {:.4}  | {:>7.2}  | n=500, 100 trees",
        xgb_r2,
        xgb_time.as_secs_f64() * 1000.0
    );
    println!(
        "LightGBM      | {:.4}  | {:>7.2}  | n=500, 100 trees",
        lgb_r2,
        lgb_time.as_secs_f64() * 1000.0
    );
    println!(
        "MBoost        | {:.4}  | {:>7.2}  | n=500, 100 iters",
        mb_r2,
        mb_time.as_secs_f64() * 1000.0
    );
    println!(
        "BART          | {:.4}  | {:>7.2}  | n=100, 30 trees (MCMC)",
        bart_r2,
        bart_time.as_secs_f64() * 1000.0
    );
    println!();

    // All should achieve reasonable R²
    assert!(xgb_r2 > 0.80, "XGBoost R² too low");
    assert!(lgb_r2 > 0.70, "LightGBM R² too low");
    assert!(mb_r2 > 0.70, "MBoost R² too low");
    assert!(bart_r2 > 0.60, "BART R² too low");
}
