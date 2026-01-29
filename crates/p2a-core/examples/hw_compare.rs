// Quick comparison test
use p2a_core::forecasting::{HoltWintersConfig, SeasonalType, holt_winters};

fn main() {
    let y = vec![
        112.0, 118.0, 132.0, 129.0, 121.0, 135.0, 148.0, 148.0, 136.0, 119.0, 104.0, 118.0, 115.0,
        126.0, 141.0, 135.0, 125.0, 149.0, 170.0, 170.0, 158.0, 133.0, 114.0, 140.0,
    ];

    // Test 1: Fixed parameters
    println!("=== Test 1: Fixed Parameters (additive) ===");
    let config1 = HoltWintersConfig {
        alpha: Some(0.2),
        beta: Some(0.1),
        gamma: Some(0.3),
        seasonal: SeasonalType::Additive,
        period: 12,
        use_trend: true,
        use_seasonal: true,
        ..Default::default()
    };
    let result1 = holt_winters(&y, config1).unwrap();
    println!("alpha: {}", result1.alpha);
    println!("beta: {:?}", result1.beta);
    println!("gamma: {:?}", result1.gamma);
    println!("SSE: {}", result1.sse);
    println!("Final level: {}", result1.coefficients.level);
    println!("Final trend: {:?}", result1.coefficients.trend);
    println!();

    // Test 2: Optimized parameters (additive)
    println!("=== Test 2: Optimized Parameters (additive) ===");
    let config2 = HoltWintersConfig {
        alpha: None,
        beta: None,
        gamma: None,
        seasonal: SeasonalType::Additive,
        period: 12,
        use_trend: true,
        use_seasonal: true,
        ..Default::default()
    };
    let result2 = holt_winters(&y, config2).unwrap();
    println!("alpha: {}", result2.alpha);
    println!("beta: {:?}", result2.beta);
    println!("gamma: {:?}", result2.gamma);
    println!("SSE: {}", result2.sse);
    println!("Final level: {}", result2.coefficients.level);
    println!("Final trend: {:?}", result2.coefficients.trend);
    println!();

    // Test 3: Optimized parameters (multiplicative)
    println!("=== Test 3: Optimized Parameters (multiplicative) ===");
    let config3 = HoltWintersConfig {
        alpha: None,
        beta: None,
        gamma: None,
        seasonal: SeasonalType::Multiplicative,
        period: 12,
        use_trend: true,
        use_seasonal: true,
        ..Default::default()
    };
    let result3 = holt_winters(&y, config3).unwrap();
    println!("alpha: {}", result3.alpha);
    println!("beta: {:?}", result3.beta);
    println!("gamma: {:?}", result3.gamma);
    println!("SSE: {}", result3.sse);
}
