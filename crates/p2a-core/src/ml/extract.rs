//! Input checks and target and feature extraction shared by the ML entry points (T2 design
//! 1.6: ML methods refuse missing values).
//!
//! The dataset wrappers (`run_gbm`, `run_xgboost`, `run_lightgbm`, `run_adaboost`,
//! `run_mboost`, `run_cart`, `run_c50`, `run_ctree`, `run_cubist`, `run_quantile_rf`) read
//! their target and features through [`extract_xy`] with [`NaPolicy::Fail`]: p2a's trees
//! have no surrogate splits, and R's randomForest and gbm refuse missing values too.

use crate::Dataset;
use crate::data::columns::{self, ColumnRequest, MissingSummary, NaPolicy};
use crate::errors::{EconError, EconResult};
use ndarray::{Array1, Array2, ArrayView2};

/// Target and features as estimator input: `x` columns in `x` order, rows aligned with `y`.
///
/// One [`ColumnRequest`] over the target (first, so [`NaPolicy::Fail`] names it first, as
/// R's model frame lists the response first) and the features. Integer, boolean (0/1) and
/// decimal columns are numeric; text is `EconError::NonNumericColumn`, `±Inf`
/// `EconError::NonFiniteValues`, a missing value under `Fail` `EconError::NullValues`. An
/// empty `x` is `EconError::InvalidSpecification` "At least one feature column must be
/// specified".
pub fn extract_xy(
    dataset: &Dataset,
    y: &str,
    x: &[&str],
    policy: NaPolicy,
) -> EconResult<(Array2<f64>, Array1<f64>, MissingSummary)> {
    if x.is_empty() {
        return Err(no_features());
    }
    let cols = ColumnRequest::new(dataset.df())
        .numeric(y)
        .numerics(x)
        .na(policy)
        .extract()?;
    Ok((
        cols.matrix(x, false).data,
        cols.numeric_array(y),
        cols.missing().clone(),
    ))
}

/// Target of a classification wrapper (C5.0, ctree, CART with gini/entropy).
#[derive(Debug, Clone, PartialEq)]
pub struct ClassTarget {
    /// Estimator input: the stored values of a numeric target, or the 0-based level codes
    /// of a text, categorical or boolean target.
    pub y: Array1<f64>,
    /// The level labels when `y` holds codes (R `factor()` order: text in byte order,
    /// FALSE < TRUE; the levels observed in the kept rows); `None` for a numeric target.
    pub labels: Option<Vec<String>>,
}

/// As [`extract_xy`], but a text, categorical or boolean target becomes level codes
/// 0..K-1 with their labels ([`ClassTarget::labels`]). A numeric target (integers and
/// floats; booleans count as classes here) is passed as stored, with `labels: None`.
pub fn extract_x_classes(
    dataset: &Dataset,
    y: &str,
    x: &[&str],
    policy: NaPolicy,
) -> EconResult<(Array2<f64>, ClassTarget, MissingSummary)> {
    let df = dataset.df();
    if columns::is_numeric_dtype(columns::column(df, y)?.dtype(), false) {
        let (x, y, missing) = extract_xy(dataset, y, x, policy)?;
        return Ok((x, ClassTarget { y, labels: None }, missing));
    }
    if x.is_empty() {
        return Err(no_features());
    }
    let cols = ColumnRequest::new(df)
        .factor(y)
        .numerics(x)
        .na(policy)
        .extract()?;
    let factor = cols.factor(y);
    let codes: Array1<f64> = factor.codes.iter().map(|&c| f64::from(c)).collect();
    Ok((
        cols.matrix(x, false).data,
        ClassTarget {
            y: codes,
            labels: Some(factor.levels.clone()),
        },
        cols.missing().clone(),
    ))
}

/// The refusal of an empty feature list.
fn no_features() -> EconError {
    EconError::InvalidSpecification {
        message: "At least one feature column must be specified".to_string(),
    }
}

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
    use crate::DataLoader;
    use crate::ml::*;
    use polars::prelude::*;

    const X: [&str; 2] = ["x1", "x2"];

    /// The 30-row fixture of plans/T2/T2-WP17.md (R cross-check: Appendix A.2, fixture.R).
    /// Rows listed in `skip` are left out (complete-row comparisons). An empty field is
    /// missing (T2-WP01 loader).
    fn fixture_csv(skip: &[usize]) -> String {
        let mut out =
            String::from("id,x1,x2,xi,xi_f,y,yi,yi_f,cls,cls_f,cls_b,lab,y_na,x1_na,cls_na\n");
        for i in (0..30usize).filter(|i| !skip.contains(i)) {
            let x1 = ((i * 7) % 11) as f64 + 0.5 * ((i * 3) % 5) as f64;
            let x2 = ((i * 5) % 13) as f64 / 2.0;
            let xi = (i * 5) % 13;
            let noise = ((i * 7919) % 101) as f64 / 100.0 - 0.5;
            let y = 2.0 * x1 - x2 + noise;
            let yi = y.floor() as i64;
            let cls = u8::from(x1 - 0.5 * x2 + noise > 3.0);
            let lab = if cls == 1 { "yes" } else { "no" };
            let blank_if = |row: usize, v: String| if i == row { String::new() } else { v };
            out.push_str(&format!(
                "{i},{x1},{x2},{xi},{xi}.0,{y},{yi},{yi}.0,{cls},{cls}.0,{},{lab},{},{},{}\n",
                cls == 1,
                blank_if(4, y.to_string()),
                blank_if(7, x1.to_string()),
                blank_if(4, cls.to_string()),
            ));
        }
        out
    }

    /// The fixture plus y_inf (y with +Inf in row 2) and lab12 (12 text levels) for the
    /// error cases.
    fn ml_dataset() -> Dataset {
        let mut df = DataLoader::from_csv_string(&fixture_csv(&[])).unwrap();
        // na-ok: test fixture without nulls
        let y: Vec<f64> = df
            .column("y")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        let y_inf: Vec<f64> = y
            .iter()
            .enumerate()
            .map(|(i, &v)| if i == 2 { f64::INFINITY } else { v })
            .collect();
        let lab12: Vec<String> = (0..30).map(|i| format!("c{:02}", i % 12)).collect();
        df.with_column(Series::new("y_inf".into(), y_inf)).unwrap();
        df.with_column(Series::new("lab12".into(), lab12)).unwrap();
        Dataset::new(df)
    }

    type Fit = fn(&Dataset, &str, &[&str]) -> EconResult<Vec<f64>>;

    /// Every wrapper in a numeric-target mode, seeded and small. c50 is separate (class
    /// codes).
    fn numeric_fits() -> Vec<(&'static str, Fit)> {
        vec![
            ("gbm", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = GbmConfig {
                    n_trees: 20,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_gbm(d, y, x, &cfg)?.predictions)
            }),
            ("xgboost", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = XGBoostConfig {
                    n_estimators: 20,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_xgboost(d, y, x, &cfg)?.predictions)
            }),
            ("lightgbm", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = LightGbmConfig {
                    num_iterations: 20,
                    min_data_in_leaf: 5,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_lightgbm(d, y, x, &cfg)?.predictions)
            }),
            ("adaboost", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = AdaBoostConfig {
                    n_estimators: 10,
                    boost_type: AdaBoostType::R2,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_adaboost(d, y, x, &cfg)?.predictions)
            }),
            ("mboost", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = MboostConfig {
                    mstop: 20,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_mboost(d, y, x, &cfg)?.predictions)
            }),
            ("cart", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = CartConfig {
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_cart(d, y, x, &cfg)?.predictions)
            }),
            ("ctree", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = CtreeConfig {
                    minsplit: 6,
                    minbucket: 3,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_ctree(d, y, x, &cfg)?.predictions)
            }),
            ("cubist", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = CubistConfig {
                    min_split: 6,
                    min_bucket: 3,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_cubist(d, y, x, &cfg)?.predictions)
            }),
            ("quantile_rf", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = QuantileRfConfig {
                    n_trees: 20,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_quantile_rf(d, y, x, &cfg)?
                    .predictions
                    .iter()
                    .copied()
                    .collect())
            }),
        ]
    }

    fn c50_fit(d: &Dataset, y: &str, x: &[&str]) -> EconResult<Vec<f64>> {
        let cfg = C50Config {
            seed: Some(7),
            ..Default::default()
        };
        Ok(run_c50(d, y, x, &cfg)?
            .predictions
            .iter()
            .map(|&c| c as f64)
            .collect())
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

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

    #[test]
    fn test_na_ml_fixture_dtypes() {
        let d = ml_dataset();
        let dt = |c: &str| d.df().column(c).unwrap().dtype().clone();
        for c in ["id", "xi", "yi", "cls", "cls_na"] {
            assert_eq!(dt(c), DataType::Int64, "{c}");
        }
        for c in ["x1", "x2", "xi_f", "y", "yi_f", "cls_f", "y_na", "x1_na"] {
            assert_eq!(dt(c), DataType::Float64, "{c}");
        }
        assert_eq!(dt("cls_b"), DataType::Boolean);
        assert_eq!(dt("lab"), DataType::String);
        for (c, n) in [("y_na", 1), ("x1_na", 1), ("cls_na", 1), ("y", 0)] {
            assert_eq!(d.df().column(c).unwrap().null_count(), n, "{c}");
        }
    }

    #[test]
    fn test_na_extract_xy_policies() {
        let d = ml_dataset();
        let (x, y, m) = extract_xy(&d, "y_na", &["x1_na", "x2"], NaPolicy::Omit).unwrap();
        assert_eq!((x.nrows(), x.ncols(), y.len()), (28, 2, 28));
        assert_eq!(
            m,
            MissingSummary {
                n_total: 30,
                n_used: 28,
                n_dropped: 2,
                by_column: vec![("y_na".into(), 1), ("x1_na".into(), 1)]
            }
        );
        // rows 4 and 7 are gone: position 4 is row 5, position 6 is row 8 (R: y[6] = -2.47,
        // x1[9] = 3)
        assert_eq!(
            y[4],
            2.0 * 2.0 - 6.0 + (((5 * 7919) % 101) as f64 / 100.0 - 0.5)
        );
        assert_eq!(x[[6, 0]], 3.0);
        let e = extract_xy(&d, "y_na", &["x1_na", "x2"], NaPolicy::Fail).unwrap_err();
        assert!(
            matches!(&e, EconError::NullValues { column, count: 1 } if column == "y_na"),
            "{e}"
        );
        let (_, _, m) = extract_xy(&d, "yi", &["xi", "cls_b"], NaPolicy::Fail).unwrap();
        assert!(m.is_empty());
        let e = extract_xy(&d, "y", &[], NaPolicy::Fail).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message == "At least one feature column must be specified"),
            "{e}"
        );
    }

    #[test]
    fn test_na_ml_wrappers_int64_target_equals_float64() {
        // WT-2. Before the fix every wrapper fails on the Int64 column: "Column 'yi' is not
        // numeric".
        let d = ml_dataset();
        for (name, fit) in numeric_fits() {
            let a = fit(&d, "yi", &X).unwrap_or_else(|e| panic!("{name} on Int64 yi: {e}"));
            let b = fit(&d, "yi_f", &X).unwrap();
            assert_eq!(bits(&a), bits(&b), "{name}");
            let c = fit(&d, "yi", &["x1", "xi"])
                .unwrap_or_else(|e| panic!("{name} with an Int64 feature: {e}"));
            assert_eq!(
                bits(&c),
                bits(&fit(&d, "yi", &["x1", "xi_f"]).unwrap()),
                "{name} (feature)"
            );
        }
        let a = c50_fit(&d, "cls", &X).unwrap();
        assert_eq!(bits(&a), bits(&c50_fit(&d, "cls_f", &X).unwrap()), "c50");
    }

    #[test]
    fn test_na_ml_wrappers_refuse_missing_values() {
        // WT-2. Before the fix a null target is the buffer value 0.0 (Ok, n_obs = 30), and a
        // null feature is EconError::DesignError(NullValues ..). R: randomForest "NA not
        // permitted in response", gbm "Missing values are not allowed in the response"
        // (Appendix A.1).
        let d = ml_dataset();
        for (name, fit) in numeric_fits() {
            let e = fit(&d, "y_na", &X).expect_err(name);
            assert!(
                matches!(&e, EconError::NullValues { column, count: 1 } if column == "y_na"),
                "{name}: {e}"
            );
            let e = fit(&d, "y", &["x1_na", "x2"]).expect_err(name);
            assert!(
                matches!(&e, EconError::NullValues { column, count: 1 } if column == "x1_na"),
                "{name}: {e}"
            );
            let e = fit(&d, "y_inf", &X).expect_err(name);
            assert!(
                matches!(&e, EconError::NonFiniteValues { column, count: 1 } if column == "y_inf"),
                "{name}: {e}"
            );
        }
        let e = c50_fit(&d, "cls_na", &X).unwrap_err();
        assert!(
            matches!(&e, EconError::NullValues { column, count: 1 } if column == "cls_na"),
            "c50: {e}"
        );
    }

    #[test]
    fn test_na_ml_wrappers_column_errors() {
        let d = ml_dataset();
        for (name, fit) in numeric_fits()
            .into_iter()
            .filter(|(n, _)| !["ctree", "cart"].contains(n))
        {
            // before: EconError::Computation("Column 'lab' is not numeric")
            let e = fit(&d, "lab", &X).expect_err(name);
            assert!(
                matches!(&e, EconError::NonNumericColumn { column } if column == "lab"),
                "{name}: {e}"
            );
        }
        for (name, fit) in numeric_fits() {
            let e = fit(&d, "nope", &X).expect_err(name); // guard: ColumnNotFound before and after
            assert!(
                matches!(&e, EconError::ColumnNotFound { column, .. } if column == "nope"),
                "{name}: {e}"
            );
            let e = fit(&d, "y", &["x1", "lab"]).expect_err(name);
            assert!(
                matches!(&e, EconError::NonNumericColumn { column } if column == "lab"),
                "{name}: {e}"
            );
        }
    }

    #[test]
    fn test_na_binomial_wrappers_accept_boolean_target() {
        // Before: "Column 'cls_b' is not numeric". R: as.numeric(TRUE) = 1.
        let d = ml_dataset();
        let fits: Vec<(&str, Fit)> = vec![
            ("gbm", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = GbmConfig {
                    n_trees: 20,
                    family: GbmFamily::Binomial,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_gbm(d, y, x, &cfg)?.predictions)
            }),
            ("xgboost", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = XGBoostConfig {
                    n_estimators: 20,
                    objective: XGBoostObjective::BinaryLogistic,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_xgboost(d, y, x, &cfg)?.predictions)
            }),
            ("lightgbm", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = LightGbmConfig {
                    num_iterations: 20,
                    min_data_in_leaf: 5,
                    objective: LightGbmObjective::Binary,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_lightgbm(d, y, x, &cfg)?.predictions)
            }),
            ("mboost", |d: &Dataset, y: &str, x: &[&str]| {
                let cfg = MboostConfig {
                    mstop: 20,
                    family: MboostFamily::Binomial,
                    seed: Some(7),
                    ..Default::default()
                };
                Ok(run_mboost(d, y, x, &cfg)?.predictions)
            }),
        ];
        for (name, fit) in fits {
            let a = fit(&d, "cls_b", &X).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(bits(&a), bits(&fit(&d, "cls_f", &X).unwrap()), "{name}");
        }
    }

    #[test]
    fn test_na_extract_x_classes_levels() {
        let d = ml_dataset();
        let cls: Vec<f64> = extract_xy(&d, "cls", &X, NaPolicy::Fail)
            .unwrap()
            .1
            .to_vec();
        let (_, t, _) = extract_x_classes(&d, "lab", &X, NaPolicy::Fail).unwrap();
        // R: levels(factor(lab))
        assert_eq!(
            t.labels.as_deref(),
            Some(&["no".to_string(), "yes".to_string()][..])
        );
        assert_eq!(t.y.to_vec(), cls); // "yes" iff cls == 1
        let (_, t, _) = extract_x_classes(&d, "cls_b", &X, NaPolicy::Fail).unwrap();
        assert_eq!(
            t.labels.as_deref(),
            Some(&["FALSE".to_string(), "TRUE".to_string()][..])
        );
        assert_eq!(t.y.to_vec(), cls);
        let (_, t, _) = extract_x_classes(&d, "cls", &X, NaPolicy::Fail).unwrap();
        assert_eq!((t.labels, t.y.to_vec()), (None, cls));
        let e = extract_x_classes(&d, "cls_na", &X, NaPolicy::Fail).unwrap_err();
        assert!(
            matches!(&e, EconError::NullValues { column, .. } if column == "cls_na"),
            "{e}"
        );
        // a text target with a missing label is named too, and an empty feature list refused
        let e = extract_x_classes(&d, "lab", &["x1_na"], NaPolicy::Fail).unwrap_err();
        assert!(
            matches!(&e, EconError::NullValues { column, .. } if column == "x1_na"),
            "{e}"
        );
        let e = extract_x_classes(&d, "lab", &[], NaPolicy::Fail).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message == "At least one feature column must be specified"),
            "{e}"
        );
    }

    #[test]
    fn test_na_c50_class_labels() {
        // WT-2 duplicate: "ml_c50 ... string labels are rejected". Before: "Column 'lab' is
        // not numeric".
        let d = ml_dataset();
        let cfg = C50Config {
            seed: Some(7),
            ..Default::default()
        };
        let codes = run_c50(&d, "cls", &X, &cfg).unwrap();
        assert_eq!(codes.class_names, None);
        assert!(!codes.to_string().contains("Class labels:"));
        for (col, names) in [("lab", ["no", "yes"]), ("cls_b", ["FALSE", "TRUE"])] {
            let r = run_c50(&d, col, &X, &cfg).unwrap_or_else(|e| panic!("{col}: {e}"));
            assert_eq!(r.predictions, codes.predictions, "{col}");
            assert_eq!(r.class_labels, vec![0, 1]);
            assert_eq!(r.class_names, Some(names.map(String::from).to_vec()));
            assert!(
                r.to_string()
                    .contains(&format!("Class labels: 0 = {}, 1 = {}", names[0], names[1])),
                "{r}"
            );
        }
    }

    #[test]
    fn test_na_ctree_class_labels() {
        let d = ml_dataset();
        let cfg = CtreeConfig {
            minsplit: 6,
            minbucket: 3,
            seed: Some(7),
            ..Default::default()
        };
        let codes = run_ctree(&d, "cls", &X, &cfg).unwrap();
        let r = run_ctree(&d, "lab", &X, &cfg).unwrap();
        assert!(r.is_classification && codes.is_classification);
        assert_eq!(bits(&r.predictions), bits(&codes.predictions));
        assert_eq!(codes.class_names, None);
        assert_eq!(
            r.class_names,
            Some(vec!["no".to_string(), "yes".to_string()])
        );
        assert!(r.to_string().contains("Class labels: 0 = no, 1 = yes"));
        // 12 classes: ctree's heuristic (at most 10 classes, fewer than n/2) would fit a
        // regression on the codes
        let e = run_ctree(&d, "lab12", &X, &cfg).unwrap_err();
        assert!(
            matches!(&e, EconError::InvalidSpecification { message }
                if message.contains("at most 10 classes")),
            "{e}"
        );
    }

    #[test]
    fn test_na_cart_class_labels() {
        let d = ml_dataset();
        let gini = CartConfig {
            method: CartMethod::Gini,
            seed: Some(7),
            ..Default::default()
        };
        let r = run_cart(&d, "lab", &X, &gini).unwrap();
        assert_eq!(
            r.class_labels,
            Some(vec!["no".to_string(), "yes".to_string()])
        );
        assert!(r.predictions.iter().all(|&p| p == 0.0 || p == 1.0));
        assert!(r.to_string().contains("Class labels: 0 = no, 1 = yes"));
        assert_eq!(run_cart(&d, "cls", &X, &gini).unwrap().class_labels, None);
        // anova needs a numeric target
        let e = run_cart(&d, "lab", &X, &CartConfig::default()).unwrap_err();
        assert!(
            matches!(&e, EconError::NonNumericColumn { column } if column == "lab"),
            "{e}"
        );
    }
}
