//! Integration tests for p2a-mcp server.
//!
//! These tests verify end-to-end functionality of the MCP server
//! without needing external dependencies.

use p2a_mcp::AnalyticsServer;

/// Test that the server initializes correctly.
#[test]
fn test_server_initialization() {
    // Server should initialize without panicking
    let _server = AnalyticsServer::new();
}

/// Test that multiple servers can be created (for session isolation).
#[test]
fn test_multiple_servers() {
    let _server1 = AnalyticsServer::new();
    let _server2 = AnalyticsServer::new();
    let _server3 = AnalyticsServer::new();
    // All servers should be independent - no shared state panic
}

/// Test server cloning (required for concurrent access).
#[test]
fn test_server_clone() {
    let server1 = AnalyticsServer::new();
    let server2 = server1.clone();
    // Both should be valid
    drop(server1);
    drop(server2);
}

/// Test p2a-core regression functionality (simulates what MCP tools do).
#[test]
fn test_core_regression_via_dataset() {
    use p2a_core::LinearEstimator;
    use p2a_core::data::Dataset;
    use p2a_core::regression::{CovarianceType, run_ols};
    use polars::prelude::*;

    // Create test data with known relationship: y ≈ 2x + 1
    let df = df! {
        "y" => [3.1, 4.9, 7.2, 8.8, 11.1],
        "x" => [1.0, 2.0, 3.0, 4.0, 5.0]
    }
    .expect("Failed to create DataFrame");

    let dataset = Dataset::new(df);

    // Run OLS - this is what the MCP tool does internally
    let result = run_ols(&dataset, "y", &["x"], true, CovarianceType::Standard);
    assert!(result.is_ok(), "OLS should succeed");

    let ols = result.unwrap();
    // Slope should be approximately 2
    let slope = ols.coefficients()[1];
    assert!(
        (slope - 2.0).abs() < 0.5,
        "Slope should be ~2, got {}",
        slope
    );
}

/// Test error handling for missing column (what MCP tools return).
#[test]
fn test_missing_column_error() {
    use p2a_core::data::Dataset;
    use p2a_core::regression::{CovarianceType, run_ols};
    use polars::prelude::*;

    let df = df! {
        "a" => [1.0, 2.0, 3.0],
        "b" => [4.0, 5.0, 6.0]
    }
    .expect("Failed to create DataFrame");

    let dataset = Dataset::new(df);

    // Try OLS with nonexistent column "y"
    let result = run_ols(&dataset, "y", &["a"], true, CovarianceType::Standard);
    assert!(result.is_err(), "Should fail with missing column");

    // Error message should be helpful (includes typo suggestions now)
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("y") && err_msg.contains("not found"),
        "Error should mention column not found: {}",
        err_msg
    );
}

/// Test that p2a-core panel data works (used by panel MCP tools).
#[test]
fn test_core_panel_fe() {
    use p2a_core::data::Dataset;
    use p2a_core::econometrics::run_fixed_effects;
    use polars::prelude::*;

    // Simple panel dataset
    let df = df! {
        "entity" => ["A", "A", "B", "B", "C", "C"],
        "time" => [1, 2, 1, 2, 1, 2],
        "y" => [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "x" => [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    }
    .expect("Failed to create DataFrame");

    let dataset = Dataset::new(df);

    let result = run_fixed_effects(&dataset, "y", &["x"], "entity");
    assert!(
        result.is_ok(),
        "Panel FE should succeed: {:?}",
        result.err()
    );
}

/// Test that p2a-core DiD works (used by causal MCP tools).
#[test]
fn test_core_did() {
    use p2a_core::data::Dataset;
    use p2a_core::econometrics::run_did;
    use polars::prelude::*;

    // Simple DiD dataset
    let df = df! {
        "y" => [1.0, 1.5, 2.0, 3.0, 1.1, 1.6, 2.5, 4.0],
        "treatment" => [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        "post" => [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    }
    .expect("Failed to create DataFrame");

    let dataset = Dataset::new(df);

    let result = run_did(&dataset, "y", "treatment", "post", None);
    assert!(result.is_ok(), "DiD should succeed: {:?}", result.err());
}

/// Test descriptive statistics (used by stats MCP tools).
#[test]
fn test_core_descriptive() {
    use p2a_core::data::Dataset;
    use p2a_core::stats::correlation_matrix;
    use polars::prelude::*;

    let df = df! {
        "x" => [1.0, 2.0, 3.0, 4.0, 5.0],
        "y" => [2.0, 4.0, 6.0, 8.0, 10.0],
        "z" => [5.0, 4.0, 3.0, 2.0, 1.0]
    }
    .expect("Failed to create DataFrame");

    let dataset = Dataset::new(df);

    // correlation_matrix takes just the dataset (uses all numeric columns)
    let result = correlation_matrix(&dataset);
    assert!(result.is_ok(), "Correlation matrix should succeed");

    let corr = result.unwrap();
    // x and y should have correlation close to 1 (matrix is Vec<Vec<f64>>)
    assert!(
        corr.matrix[0][1].abs() > 0.99,
        "x and y should be highly correlated"
    );
    // x and z should have correlation close to -1
    assert!(
        corr.matrix[0][2] < -0.99,
        "x and z should be negatively correlated"
    );
}

/// Test hypothesis testing (used by stats MCP tools).
#[test]
fn test_core_ttest() {
    use p2a_core::stats::{Alternative, two_sample_t_test};

    let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let group2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    // two_sample_t_test(x, y, mu, alternative, var_equal, conf_level)
    let result = two_sample_t_test(&group1, &group2, 0.0, Alternative::TwoSided, false, 0.95);
    assert!(result.is_ok(), "T-test should succeed");

    let ttest = result.unwrap();
    // Groups differ by 1 on average: estimate is mean(x) - mean(y) = 3.0 - 4.0 = -1.0
    let mean_diff = ttest.estimate - ttest.estimate_2.unwrap_or(0.0);
    assert!(
        (mean_diff - (-1.0)).abs() < 0.01,
        "Mean difference should be -1, got {}",
        mean_diff
    );
}

/// Test clustering (used by ML MCP tools).
#[test]
fn test_core_kmeans() {
    use ndarray::Array2;
    use p2a_core::ml::kmeans;

    // Create clustered data
    let data = Array2::from_shape_vec(
        (6, 2),
        vec![
            0.0, 0.0, // Cluster 1
            0.1, 0.1, // Cluster 1
            5.0, 5.0, // Cluster 2
            5.1, 5.1, // Cluster 2
            10.0, 0.0, // Cluster 3
            10.1, 0.1, // Cluster 3
        ],
    )
    .unwrap();

    // kmeans(data, k, max_iter, tol, n_init, seed)
    let result = kmeans(data.view(), 3, Some(100), Some(1e-4), Some(5), Some(42));
    assert!(result.is_ok(), "K-means should succeed");

    let kmeans_result = result.unwrap();
    assert_eq!(kmeans_result.labels.len(), 6, "Should have 6 labels");
    assert_eq!(kmeans_result.centroids.nrows(), 3, "Should have 3 centers");
}
