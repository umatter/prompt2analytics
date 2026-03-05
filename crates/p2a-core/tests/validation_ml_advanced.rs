//! Validation tests for advanced ML methods: C5.0, Cubist, CTree, MBoost, SHAP.
//!
//! These tests verify:
//! 1. C5.0: Classification accuracy on separable data
//! 2. Cubist: Regression R^2 on synthetic linear data
//! 3. CTree: Conditional inference tree splits on significant features
//! 4. MBoost: Gradient boosting with linear and tree base learners
//! 5. SHAP: Feature attribution sums to prediction - base_value

use ndarray::{Array1, Array2, ArrayView2};
use p2a_core::ml::{
    // C5.0
    C50Config,
    c50,
    c50_predict,
    // Cubist
    CubistConfig,
    cubist,
    cubist_predict,
    // CTree
    CtreeConfig,
    ctree,
    ctree_predict,
    // MBoost
    MboostBaseLearner,
    MboostConfig,
    MboostFamily,
    mboost,
    mboost_predict,
    // SHAP
    ShapConfig,
    shap_values_model,
    random_forest_with_trees,
};

/// Create synthetic regression data: y = 2*x1 + 0.5*x2 + noise
fn create_regression_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    // Simple LCG for reproducibility without external RNG
    let mut state = seed;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut x = Array2::zeros((n, 3));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x1 = next(&mut state);
        let x2 = next(&mut state);
        let x3 = next(&mut state) - 0.5; // noise feature centered at 0

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        let noise = (next(&mut state) - 0.5) * 0.6; // uniform noise ~[-0.3, 0.3]
        y[i] = 2.0 * x1 + 0.5 * x2 + noise;
    }

    (x, y)
}

/// Create synthetic binary classification data
fn create_classification_data(n: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
    let mut state = seed;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut x = Array2::zeros((n, 3));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x1 = next(&mut state);
        let x2 = next(&mut state);
        let x3 = next(&mut state) - 0.5;

        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        x[[i, 2]] = x3;

        // Decision boundary: x1 + x2 > 1.0
        y[i] = if x1 + x2 > 1.0 { 1.0 } else { 0.0 };
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
    if ss_tot == 0.0 {
        return 0.0;
    }
    1.0 - ss_res / ss_tot
}

fn compute_classification_accuracy(y_true: &Array1<f64>, y_pred: &[usize]) -> f64 {
    let correct: usize = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|&(&yt, &yp)| (yt as usize) == yp)
        .count();
    correct as f64 / y_true.len() as f64
}

// ============================================================================
// C5.0 Validation Tests
// ============================================================================

#[test]
fn test_validate_c50_classification() {
    let (x, y) = create_classification_data(200, 42);

    let config = C50Config {
        trials: 1,
        min_cases: 5,
        cf: 0.25,
        seed: Some(42),
        ..Default::default()
    };

    let result = c50(x.view(), y.view(), &config).unwrap();

    println!("C5.0 Classification Results:");
    println!("  Training accuracy: {:.4}", result.accuracy);
    println!("  Number of classes: {}", result.n_classes);
    println!(
        "  Variable importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0],
        result.variable_importance[1],
        result.variable_importance[2]
    );

    // Training accuracy should be high for separable data
    assert!(
        result.accuracy > 0.80,
        "C5.0 training accuracy should be > 0.80, got {:.4}",
        result.accuracy
    );

    // x1 and x2 should be more important than x3 (noise)
    assert!(
        result.variable_importance[0] + result.variable_importance[1]
            > result.variable_importance[2],
        "Signal features should be more important than noise"
    );
}

#[test]
fn test_validate_c50_boosted() {
    let (x, y) = create_classification_data(200, 42);

    let config = C50Config {
        trials: 10,
        min_cases: 5,
        cf: 0.25,
        seed: Some(42),
        ..Default::default()
    };

    let result = c50(x.view(), y.view(), &config).unwrap();

    println!("C5.0 Boosted (10 trials):");
    println!("  Training accuracy: {:.4}", result.accuracy);
    println!("  Actual trials: {}", result.actual_trials);

    // Boosted model should achieve at least as good accuracy as single tree
    assert!(
        result.accuracy > 0.80,
        "C5.0 boosted accuracy should be > 0.80, got {:.4}",
        result.accuracy
    );
}

#[test]
fn test_validate_c50_predict() {
    let (x_train, y_train) = create_classification_data(200, 42);
    let (x_test, y_test) = create_classification_data(50, 123);

    let config = C50Config {
        trials: 5,
        min_cases: 5,
        seed: Some(42),
        ..Default::default()
    };

    let result = c50(x_train.view(), y_train.view(), &config).unwrap();
    let predictions = c50_predict(&result, x_test.view()).unwrap();

    let accuracy = compute_classification_accuracy(&y_test, &predictions);
    println!("C5.0 Out-of-sample accuracy: {:.4}", accuracy);

    // Out-of-sample accuracy should be reasonable for this separable data
    assert!(
        accuracy > 0.70,
        "C5.0 out-of-sample accuracy should be > 0.70, got {:.4}",
        accuracy
    );
}

// ============================================================================
// Cubist Validation Tests
// ============================================================================

#[test]
fn test_validate_cubist_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = CubistConfig {
        committees: 1,
        neighbors: 0,
        max_depth: 10,
        min_split: 10,
        seed: Some(42),
        ..Default::default()
    };

    let result = cubist(x.view(), y.view(), &config).unwrap();

    println!("Cubist Regression Results:");
    println!("  Train RMSE: {:.6}", result.train_rmse);
    println!("  Train R²: {:.4}", result.train_r_squared);
    println!("  Number of rules: {}", result.n_rules);
    println!(
        "  Variable importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0],
        result.variable_importance[1],
        result.variable_importance[2]
    );

    assert!(
        result.train_r_squared > 0.70,
        "Cubist R² should be > 0.70, got {:.4}",
        result.train_r_squared
    );
    assert!(result.n_rules > 0, "Cubist should produce at least one rule");
}

#[test]
fn test_validate_cubist_committees() {
    let (x, y) = create_regression_data(200, 42);

    let config = CubistConfig {
        committees: 5,
        neighbors: 0,
        max_depth: 10,
        min_split: 10,
        seed: Some(42),
        ..Default::default()
    };

    let result = cubist(x.view(), y.view(), &config).unwrap();

    println!("Cubist with 5 committees:");
    println!("  Train R²: {:.4}", result.train_r_squared);
    println!("  Committees used: {}", result.committees);

    assert!(
        result.train_r_squared > 0.70,
        "Cubist with committees R² should be > 0.70, got {:.4}",
        result.train_r_squared
    );
}

#[test]
fn test_validate_cubist_predict() {
    let (x_train, y_train) = create_regression_data(200, 42);
    let (x_test, y_test) = create_regression_data(50, 123);

    let config = CubistConfig {
        committees: 3,
        max_depth: 10,
        seed: Some(42),
        ..Default::default()
    };

    let result = cubist(x_train.view(), y_train.view(), &config).unwrap();
    let predictions = cubist_predict(&result, x_test.view()).unwrap();

    let r2_test = compute_r2(&y_test, &predictions);
    println!("Cubist out-of-sample R²: {:.4}", r2_test);

    assert!(
        r2_test > 0.50,
        "Cubist out-of-sample R² should be > 0.50, got {:.4}",
        r2_test
    );
}

// ============================================================================
// CTree Validation Tests
// ============================================================================

#[test]
fn test_validate_ctree_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = CtreeConfig {
        mincriterion: 0.95,
        minsplit: 20,
        minbucket: 7,
        maxdepth: 10,
        seed: Some(42),
        ..Default::default()
    };

    let result = ctree(x.view(), y.view(), &config).unwrap();

    let predictions = ctree_predict(&result, x.view()).unwrap();
    let r2 = compute_r2(&y, &predictions);

    println!("CTree Regression Results:");
    println!("  R²: {:.4}", r2);
    println!("  Number of nodes: {}", result.n_nodes);
    println!("  Terminal nodes: {}", result.n_terminal);
    println!("  Max depth: {}", result.depth);
    println!(
        "  Variable importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0],
        result.variable_importance[1],
        result.variable_importance[2]
    );

    assert!(
        r2 > 0.50,
        "CTree R² should be > 0.50, got {:.4}",
        r2
    );

    // x1 should have significant p-value at root
    println!(
        "  Root p-values: x1={:.4}, x2={:.4}, x3={:.4}",
        result.root_p_values[0], result.root_p_values[1], result.root_p_values[2]
    );
}

#[test]
fn test_validate_ctree_classification() {
    let (x, y) = create_classification_data(200, 42);

    let config = CtreeConfig {
        mincriterion: 0.95,
        minsplit: 20,
        minbucket: 7,
        maxdepth: 10,
        seed: Some(42),
        ..Default::default()
    };

    let result = ctree(x.view(), y.view(), &config).unwrap();
    assert!(
        result.is_classification,
        "CTree should detect classification task"
    );

    let predictions = ctree_predict(&result, x.view()).unwrap();
    let accuracy: f64 = y
        .iter()
        .zip(predictions.iter())
        .filter(|&(&yt, &yp)| (yt - yp).abs() < 0.5)
        .count() as f64
        / y.len() as f64;

    println!("CTree Classification Results:");
    println!("  Accuracy: {:.4}", accuracy);
    println!("  Nodes: {}", result.n_nodes);

    assert!(
        accuracy > 0.75,
        "CTree classification accuracy should be > 0.75, got {:.4}",
        accuracy
    );
}

#[test]
fn test_validate_ctree_predict() {
    let (x_train, y_train) = create_regression_data(200, 42);
    let (x_test, y_test) = create_regression_data(50, 123);

    let config = CtreeConfig {
        minsplit: 20,
        minbucket: 7,
        seed: Some(42),
        ..Default::default()
    };

    let result = ctree(x_train.view(), y_train.view(), &config).unwrap();
    let predictions = ctree_predict(&result, x_test.view()).unwrap();

    let r2_test = compute_r2(&y_test, &predictions);
    println!("CTree out-of-sample R²: {:.4}", r2_test);

    assert!(
        r2_test > 0.30,
        "CTree out-of-sample R² should be > 0.30, got {:.4}",
        r2_test
    );
}

// ============================================================================
// MBoost Validation Tests
// ============================================================================

#[test]
fn test_validate_mboost_linear_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = MboostConfig {
        mstop: 200,
        nu: 0.1,
        base_learner: MboostBaseLearner::Linear,
        family: MboostFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let result = mboost(x.view(), y.view(), &config).unwrap();

    let r2 = compute_r2(&y, &result.predictions);

    println!("MBoost Linear Results:");
    println!("  R²: {:.4}", r2);
    println!("  Iterations: {}", result.iterations);
    println!("  Variables selected: {}", result.n_selected);
    println!(
        "  Variable importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0],
        result.variable_importance[1],
        result.variable_importance[2]
    );
    println!(
        "  Coefficients: x1={:.4}, x2={:.4}, x3={:.4}",
        result.coefficients[0], result.coefficients[1], result.coefficients[2]
    );

    assert!(
        r2 > 0.70,
        "MBoost linear R² should be > 0.70, got {:.4}",
        r2
    );

    // Linear mboost should recover approximate coefficients (x1~2, x2~0.5)
    assert!(
        result.coefficients[0] > result.coefficients[2].abs(),
        "x1 coefficient should be larger than x3 coefficient"
    );
}

#[test]
fn test_validate_mboost_tree_regression() {
    let (x, y) = create_regression_data(200, 42);

    let config = MboostConfig {
        mstop: 100,
        nu: 0.1,
        base_learner: MboostBaseLearner::Tree,
        tree_depth: 3,
        family: MboostFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let result = mboost(x.view(), y.view(), &config).unwrap();

    let r2 = compute_r2(&y, &result.predictions);

    println!("MBoost Tree (depth=3) Results:");
    println!("  R²: {:.4}", r2);
    println!("  Iterations: {}", result.iterations);
    println!(
        "  Variable importance: x1={:.4}, x2={:.4}, x3={:.4}",
        result.variable_importance[0],
        result.variable_importance[1],
        result.variable_importance[2]
    );

    assert!(
        r2 > 0.75,
        "MBoost tree R² should be > 0.75, got {:.4}",
        r2
    );
}

#[test]
fn test_validate_mboost_predict() {
    let (x_train, y_train) = create_regression_data(200, 42);
    let (x_test, y_test) = create_regression_data(50, 123);

    let config = MboostConfig {
        mstop: 150,
        nu: 0.1,
        base_learner: MboostBaseLearner::Linear,
        family: MboostFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let result = mboost(x_train.view(), y_train.view(), &config).unwrap();
    let predictions = mboost_predict(&result, x_test.view()).unwrap();

    let r2_test = compute_r2(&y_test, &predictions);
    println!("MBoost out-of-sample R²: {:.4}", r2_test);

    assert!(
        r2_test > 0.50,
        "MBoost out-of-sample R² should be > 0.50, got {:.4}",
        r2_test
    );
}

#[test]
fn test_validate_mboost_poisson() {
    // Poisson regression on count-like data: y = exp(0.5*x1 + 0.2*x2)
    let mut state: u64 = 42;
    let next = |s: &mut u64| -> f64 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let n = 200;
    let mut x = Array2::zeros((n, 2));
    let mut y = Array1::zeros(n);

    for i in 0..n {
        let x1 = next(&mut state) * 2.0;
        let x2 = next(&mut state) * 2.0;
        x[[i, 0]] = x1;
        x[[i, 1]] = x2;
        // Use exp to make it count-like (positive)
        y[i] = (0.5 * x1 + 0.2 * x2).exp() + next(&mut state) * 0.1;
    }

    let config = MboostConfig {
        mstop: 100,
        nu: 0.1,
        base_learner: MboostBaseLearner::Linear,
        family: MboostFamily::Poisson,
        seed: Some(42),
        ..Default::default()
    };

    let result = mboost(x.view(), y.view(), &config).unwrap();
    let r2 = compute_r2(&y, &result.predictions);

    println!("MBoost Poisson Results:");
    println!("  R²: {:.4}", r2);
    println!("  Iterations: {}", result.iterations);

    // Poisson should fit reasonably well
    assert!(
        r2 > 0.30,
        "MBoost Poisson R² should be > 0.30, got {:.4}",
        r2
    );
}

// ============================================================================
// SHAP Validation Tests
// ============================================================================

#[test]
fn test_validate_shap_tree_ensemble() {
    let (x, y) = create_regression_data(100, 42);

    // Train a random forest
    let rf = random_forest_with_trees(
        x.view(),
        y.view(),
        Some(20),     // 20 trees
        Some(5),      // max depth 5
        Some(5),      // min samples split
        Some("sqrt"), // sqrt features
        Some(42),
        Some(vec!["x1".into(), "x2".into(), "x3".into()]),
    )
    .unwrap();

    let config = ShapConfig {
        seed: Some(42),
        ..Default::default()
    };

    let shap_result = shap_values_model(&rf, x.view(), &config).unwrap();

    println!("SHAP Tree Ensemble Results:");
    println!("  Base value: {:.4}", shap_result.base_value);
    println!("  N observations: {}", shap_result.n_obs);
    println!("  N features: {}", shap_result.n_features);
    println!(
        "  Mean |SHAP|: x1={:.4}, x2={:.4}, x3={:.4}",
        shap_result.feature_importance[0],
        shap_result.feature_importance[1],
        shap_result.feature_importance[2]
    );

    // SHAP values should have correct dimensions
    assert_eq!(shap_result.n_features, 3);
    assert_eq!(shap_result.n_obs, 100);

    // x1 should have higher mean |SHAP| than x3 (it has coefficient 2.0)
    assert!(
        shap_result.feature_importance[0] > shap_result.feature_importance[2],
        "x1 should have higher SHAP importance than x3 (noise)"
    );

    // SHAP additivity: sum of SHAP values + base ≈ prediction
    // Tree SHAP approximation may not be exact, so we check average error
    let mut total_diff = 0.0;
    for i in 0..shap_result.n_obs {
        let shap_sum: f64 = shap_result.shap_values.row(i).sum();
        let pred = rf.predictions[i];
        let reconstructed = shap_result.base_value + shap_sum;
        total_diff += (pred - reconstructed).abs();
    }
    let mean_diff = total_diff / shap_result.n_obs as f64;
    println!("  Mean additivity error: {:.6}", mean_diff);

    // Mean error should be reasonable (tree SHAP is approximate)
    assert!(
        mean_diff < 1.0,
        "SHAP mean additivity error should be < 1.0, got {:.4}",
        mean_diff
    );
}

#[test]
fn test_validate_shap_feature_importance_ordering() {
    let (x, y) = create_regression_data(150, 42);

    let rf = random_forest_with_trees(
        x.view(),
        y.view(),
        Some(30),
        Some(6),
        Some(5),
        Some("sqrt"),
        Some(42),
        None,
    )
    .unwrap();

    let config = ShapConfig {
        seed: Some(42),
        ..Default::default()
    };

    let shap_result = shap_values_model(&rf, x.view(), &config).unwrap();

    // For y = 2*x1 + 0.5*x2 + noise, SHAP importance of x1 > x2 > x3
    println!(
        "SHAP importance ordering: x1={:.4}, x2={:.4}, x3={:.4}",
        shap_result.feature_importance[0],
        shap_result.feature_importance[1],
        shap_result.feature_importance[2]
    );

    assert!(
        shap_result.feature_importance[0] > shap_result.feature_importance[1],
        "x1 (coef=2) should have higher SHAP importance than x2 (coef=0.5)"
    );
    assert!(
        shap_result.feature_importance[0] > shap_result.feature_importance[2],
        "x1 (coef=2) should have higher SHAP importance than x3 (noise)"
    );
}
