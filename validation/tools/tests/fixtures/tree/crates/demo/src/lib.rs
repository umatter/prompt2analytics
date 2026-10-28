// Fixture source for validation_status.py lint (no Cargo.toml: cargo and rustfmt never
// see it). The package is the directory name, `demo`.
mod stats {
    #[cfg(test)]
    mod tests {
        #[test]
        fn test_validate_demo_mean() {
            let s = "}{";
            let (case, ds) = crate::test_support::load_case("demo_case");
            let _ = (s, case, ds, '}', '{');
        }

        #[test]
        fn test_validate_demo_var() {
            let c = p2a_validation::Case::load("demo_var_case");
            let _ = c;
        }

        #[test]
        fn test_selftest() {
            let c = p2a_validation::Case::load("selftest_case");
            let _ = c;
        }
    }
}
