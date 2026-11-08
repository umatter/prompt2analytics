//! DGP-recovery checks for XGBoost, LightGBM, mboost and BART.
//!
//! None of these tests compares with R: each recovers a simulated DGP (an R-squared or accuracy
//! bound, the stronger coefficient ranked first). The tests compared with R or with published
//! algorithms are in the modules `ml::{adaboost, xgboost, xgboost_fast, mboost, bart}` and in the
//! GBM module.

use ndarray::{Array1, Array2};
use p2a_core::ml::{
    BartConfig, LightGbmConfig, LightGbmObjective, MboostBaseLearner, MboostConfig, MboostFamily,
    XGBoostConfig, XGBoostObjective, bart_arrays, lightgbm, lightgbm_predict, mboost,
    mboost_predict, xgboost, xgboost_predict,
};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};

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
fn xgboost_regression_recovers_dgp() {
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

    let result = xgboost(x.view(), y.view(), &config).unwrap();

    let r2 = compute_r2(&y, &result.predictions);

    // Validation criteria
    assert!(r2 > 0.80, "XGBoost R² should be > 0.80, got {:.4}", r2);
    assert!(
        result.feature_importances[0] > result.feature_importances[2],
        "x1 should be more important than x3 (noise)"
    );
}

#[test]
fn xgboost_classification_recovers_dgp() {
    let (x, y) = create_classification_data(200, 42);

    let config = XGBoostConfig {
        n_estimators: 100,
        max_depth: 6,
        learning_rate: 0.3,
        objective: XGBoostObjective::BinaryLogistic,
        seed: Some(42),
        ..Default::default()
    };

    let result = xgboost(x.view(), y.view(), &config).unwrap();

    let accuracy = compute_accuracy(&y, &result.predictions, 0.5);

    assert!(
        accuracy > 0.75,
        "XGBoost accuracy should be > 0.75, got {:.4}",
        accuracy
    );
}

#[test]
fn lightgbm_regression_recovers_dgp() {
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

    let result = lightgbm(x.view(), y.view(), &config).unwrap();

    let r2 = compute_r2(&y, &result.predictions);

    assert!(r2 > 0.70, "LightGBM R² should be > 0.70, got {:.4}", r2);
    assert!(
        result.feature_importances[0] > result.feature_importances[2],
        "x1 should be more important than x3"
    );
}

#[test]
fn mboost_tree_regression_recovers_dgp() {
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

    let result = mboost(x.view(), y.view(), &config).unwrap();

    let r2 = compute_r2(&y, &result.predictions);

    assert!(r2 > 0.75, "MBoost R² should be > 0.75, got {:.4}", r2);
}

#[test]
fn mboost_linear_selects_the_signal() {
    let (x, y) = create_regression_data(200, 42);

    let config = MboostConfig {
        mstop: 100,
        nu: 0.1,
        base_learner: MboostBaseLearner::Linear,
        family: MboostFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let result = mboost(x.view(), y.view(), &config).unwrap();

    // Linear base learner should identify x1 as most important
    assert!(
        result.variable_importance[0] > result.variable_importance[2],
        "x1 should be selected more often than x3"
    );
}

#[test]
fn bart_regression_recovers_dgp() {
    let (x, y) = create_regression_data(100, 42);

    let config = BartConfig {
        n_trees: 50,
        n_burn: 50,
        n_mcmc: 100,
        k: 2.0,
        seed: Some(42),
        ..Default::default()
    };

    let result = bart_arrays(y.view(), x.view(), None, config).unwrap();

    let r2 = compute_r2(&y, &result.predictions);

    assert!(r2 > 0.50, "BART R² should be > 0.50, got {:.4}", r2);
}

#[test]
fn boosting_methods_recover_dgp() {
    let (x, y) = create_regression_data(500, 42);

    // XGBoost
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
    let xgb_r2 = compute_r2(&y, &xgb_result.predictions);

    // LightGBM
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
    let lgb_r2 = compute_r2(&y, &lgb_result.predictions);

    // MBoost
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
    let mb_r2 = compute_r2(&y, &mb_result.predictions);

    // BART (smaller for speed)
    let (x_small, y_small) = create_regression_data(100, 42);
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
    let bart_r2 = compute_r2(&y_small, &bart_result.predictions);

    // All should achieve reasonable R²
    assert!(xgb_r2 > 0.80, "XGBoost R² too low");
    assert!(lgb_r2 > 0.70, "LightGBM R² too low");
    assert!(mb_r2 > 0.70, "MBoost R² too low");
    assert!(bart_r2 > 0.60, "BART R² too low");
}
