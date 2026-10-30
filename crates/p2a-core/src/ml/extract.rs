//! Input checks and target and feature extraction shared by the ML entry points (T2 design
//! 1.6: ML methods refuse missing values).

use ndarray::ArrayView2;

/// `Err` naming the first non-finite cell in row-major order:
/// "<method>: x must contain only finite values (row 4, column 0 is NaN); remove or impute
/// missing values first".
pub(crate) fn require_finite(data: &ArrayView2<f64>, method: &str) -> Result<(), String> {
    match data.indexed_iter().find(|(_, v)| !v.is_finite()) {
        Some(((i, j), v)) => Err(format!(
            "{method}: x must contain only finite values (row {i}, column {j} is {v}); \
             remove or impute missing values first"
        )),
        None => Ok(()),
    }
}

/// "Class labels: 0 = no, 1 = yes" (codes are 0..K-1 whenever labels exist).
pub(crate) fn class_labels_line(labels: &[String]) -> String {
    let pairs: Vec<String> = labels
        .iter()
        .enumerate()
        .map(|(k, label)| format!("{k} = {label}"))
        .collect();
    format!("Class labels: {}", pairs.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_require_finite_names_first_cell() {
        let ok = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        assert!(require_finite(&ok.view(), "PCA").is_ok());
        let mut bad = ok.clone();
        bad[[1, 1]] = f64::INFINITY;
        bad[[1, 0]] = f64::NAN; // row-major: (1, 0) comes first
        let e = require_finite(&bad.view(), "PCA").unwrap_err();
        assert_eq!(
            e,
            "PCA: x must contain only finite values (row 1, column 0 is NaN); remove or impute missing values first"
        );
        assert_eq!(
            class_labels_line(&["no".into(), "yes".into()]),
            "Class labels: 0 = no, 1 = yes"
        );
    }
}
