//! Test-only helpers: load an R reference case and its dataset (validation/README.md, "R reference harness").
//!
//! The dataset is read with [`DataLoader::load_csv`], the loader `p2a data load` replays
//! (`crates/p2a-cli/src/session.rs`, `reload_dataset`), so a test computes on the same
//! doubles R read when it wrote `validation/expected/<id>.json`.

use crate::data::{DataLoader, Dataset};

/// The case's analysis dataset, loaded as `p2a data load` loads it. Panics with
/// `case <id>: cannot load dataset <path>: <error>`.
pub(crate) fn case_dataset(case: &p2a_validation::Case) -> Dataset {
    let path = case.dataset_path();
    match DataLoader::load_csv(&path) {
        Ok(df) => Dataset::new(df),
        Err(e) => panic!(
            "case {}: cannot load dataset {}: {e}",
            case.id,
            path.display()
        ),
    }
}

/// Load case `id` (`validation/expected/<id>.json`) and its dataset.
pub(crate) fn load_case(id: &str) -> (p2a_validation::Case, Dataset) {
    let case = p2a_validation::Case::load(id);
    let ds = case_dataset(&case);
    (case, ds)
}

#[cfg(test)]
mod tests {
    use polars::prelude::ChunkAgg;

    #[test]
    fn test_load_case_reads_the_selftest_dataset_as_r_does() {
        let (case, ds) = super::load_case("harness_selftest");
        assert_eq!(ds.nrows() as f64, case.scalar("n"));
        let inv = ds
            .column("inv")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .clone();
        let first3: Vec<f64> = (0..3).map(|i| inv.get(i).unwrap()).collect();
        // Bit-identical inputs: polars parses the CSV text to the doubles R read.
        assert_eq!(first3, case.vector("inv_first3"));
        let mean = inv.mean().unwrap();
        let r = case.scalar("inv_mean");
        // R accumulates in long double; polars sums in f64 (a few ulps).
        assert!(
            (mean - r).abs() <= 1e-14 * r.abs(),
            "polars {mean} vs R {r}"
        );
    }
}
