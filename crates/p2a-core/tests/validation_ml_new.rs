//! Property tests on a hand-typed 20-row dataset (y = 2 + 3 x1 - 1.5 x2 plus hand-written noise).
//! It is not the data of validation/scripts/validate_ml_new.R, and no R output is compared here.
//! glmnet is pinned to R in `regression::glmnet::tests` (T3-WP07), CART in `ml::cart::tests`
//! (T3-WP44).

use ndarray::{Array1, Array2};
use p2a_core::ml::{AdaBoostConfig, AdaBoostType, GbmConfig, GbmFamily, adaboost, gbm};

/// A hand-typed 20-row dataset: y = 2 + 3 x1 - 1.5 x2 plus hand-written noise (x3 is noise), and
/// a class label from the sign of x1 + x2. It is not the data of validation/scripts/validate_ml_new.R.
fn generate_test_data() -> (Array2<f64>, Array1<f64>, Array1<f64>) {
    let n = 20;

    // x1, x2, x3 (x3 is noise)
    let x = Array2::from_shape_vec(
        (n, 3),
        vec![
            // x1          x2          x3 (noise)
            -0.5, -1.2, 0.3, 0.8, 0.5, -0.7, -1.2, 0.3, 1.1, 1.5, -0.8, -0.2, 0.3, 1.0, 0.5, -0.8,
            -0.5, -1.0, 2.0, 0.2, 0.8, -1.5, 1.5, -0.3, 0.5, -1.0, 0.1, 1.0, 0.8, -0.5, -0.3, -0.3,
            0.9, 0.9, 1.2, -0.8, -2.0, 0.1, 0.2, 1.8, -0.5, 0.6, 0.1, 0.9, -0.4, -1.0, -1.5, 0.0,
            1.2, 0.3, 1.2, -0.7, 0.7, -0.9, 0.6, -0.9, 0.4, 1.3, 1.1, -0.1,
        ],
    )
    .unwrap();

    // y_reg = 2 + 3*x1 - 1.5*x2 + noise
    let y_reg = Array1::from_vec(vec![
        2.0 + 3.0 * (-0.5) - 1.5 * (-1.2) + 0.1,
        2.0 + 3.0 * 0.8 - 1.5 * 0.5 - 0.2,
        2.0 + 3.0 * (-1.2) - 1.5 * 0.3 + 0.3,
        2.0 + 3.0 * 1.5 - 1.5 * (-0.8) - 0.1,
        2.0 + 3.0 * 0.3 - 1.5 * 1.0 + 0.2,
        2.0 + 3.0 * (-0.8) - 1.5 * (-0.5) - 0.3,
        2.0 + 3.0 * 2.0 - 1.5 * 0.2 + 0.1,
        2.0 + 3.0 * (-1.5) - 1.5 * 1.5 + 0.0,
        2.0 + 3.0 * 0.5 + 1.5 * 1.0 - 0.1,
        2.0 + 3.0 * 1.0 - 1.5 * 0.8 + 0.2,
        2.0 + 3.0 * (-0.3) - 1.5 * (-0.3) - 0.2,
        2.0 + 3.0 * 0.9 - 1.5 * 1.2 + 0.1,
        2.0 + 3.0 * (-2.0) - 1.5 * 0.1 - 0.3,
        2.0 + 3.0 * 1.8 - 1.5 * (-0.5) + 0.2,
        2.0 + 3.0 * 0.1 - 1.5 * 0.9 - 0.1,
        2.0 - 3.0 * 1.0 + 1.5 * 1.5 + 0.3,
        2.0 + 3.0 * 1.2 - 1.5 * 0.3 - 0.2,
        2.0 + 3.0 * (-0.7) - 1.5 * 0.7 + 0.1,
        2.0 + 3.0 * 0.6 - 1.5 * (-0.9) - 0.1,
        2.0 + 3.0 * 1.3 - 1.5 * 1.1 + 0.0,
    ]);

    // Binary classification: y = 1 if x1 + x2 > 0, else -1
    let y_class = Array1::from_vec(
        (0..n)
            .map(|i| {
                if x[[i, 0]] + x[[i, 1]] > 0.0 {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect(),
    );

    (x, y_reg, y_class)
}

#[test]
fn test_gbm_gaussian_reduces_training_loss() {
    let (x, y, _) = generate_test_data();

    let config = GbmConfig {
        n_trees: 100,
        learning_rate: 0.1,
        max_depth: 3,
        subsample: 1.0,
        family: GbmFamily::Gaussian,
        seed: Some(42),
        ..Default::default()
    };

    let result = gbm(x.view(), y.view(), &config).unwrap();

    let mse: f64 = result
        .predictions
        .iter()
        .zip(y.iter())
        .map(|(p, y)| (p - y).powi(2))
        .sum::<f64>()
        / y.len() as f64;

    let y_var: f64 = {
        let mean = y.mean().unwrap();
        y.iter().map(|yi| (yi - mean).powi(2)).sum::<f64>() / y.len() as f64
    };
    let r_squared = 1.0 - mse / y_var;

    println!("Rust GBM results:");
    println!("  Training MSE: {:.4}", mse);
    println!("  R-squared: {:.4}", r_squared);
    println!("  Final loss: {:.4}", result.final_train_loss);
    println!("  Feature importances: {:?}", result.feature_importances);

    assert!(
        r_squared > 0.7,
        "GBM R-squared {} should be > 0.7",
        r_squared
    );
    assert!(
        result.train_loss.last().unwrap() < result.train_loss.first().unwrap(),
        "Training loss should decrease"
    );
}

#[test]
fn test_adaboost_m1_fits_linear_boundary() {
    let (x, _, y_class) = generate_test_data();

    let config = AdaBoostConfig {
        n_estimators: 50,
        boost_type: AdaBoostType::M1,
        learning_rate: 1.0,
        max_depth: 1,
        ..Default::default()
    };

    let result = adaboost(x.view(), y_class.view(), &config).unwrap();

    println!("Rust AdaBoost results:");
    println!("  Estimators used: {}", result.n_estimators);
    println!("  Training accuracy: {:.4}", result.train_accuracy.unwrap());
    println!("  Feature importances: {:?}", result.feature_importances);

    assert!(
        result.train_accuracy.unwrap() > 0.8,
        "AdaBoost accuracy {} should be > 0.8",
        result.train_accuracy.unwrap()
    );
}
