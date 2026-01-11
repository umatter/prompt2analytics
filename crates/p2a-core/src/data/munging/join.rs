//! Join operations for combining datasets.
//!
//! This module provides various join operations (left, right, inner, full, anti, semi)
//! as well as concatenation operations (vertical and horizontal).
//!
//! # Join Types
//!
//! - **Left Join**: Keep all rows from left, matching rows from right
//! - **Right Join**: Keep all rows from right, matching rows from left
//! - **Inner Join**: Keep only rows that match in both datasets
//! - **Full Join**: Keep all rows from both datasets
//! - **Anti Join**: Keep rows from left that have no match in right
//! - **Semi Join**: Keep rows from left that have a match in right (no right columns)
//!
//! # Example
//!
//! ```ignore
//! use p2a_core::data::munging::*;
//!
//! // Join two datasets
//! let result = left_join(&orders, &customers, &["customer_id"], None, Some("_right"))?;
//!
//! // Concatenate datasets vertically
//! let combined = concat(&[&df1, &df2, &df3])?;
//!
//! // Concatenate datasets horizontally
//! let wide = hconcat(&[&df1, &df2])?;
//! ```

use super::error::{MungeError, MungeResult};
use crate::data::Dataset;
use polars::prelude::*;

/// Type of join operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Keep all rows from left, matching rows from right
    Left,
    /// Keep all rows from right, matching rows from left
    Right,
    /// Keep only rows that match in both datasets
    Inner,
    /// Keep all rows from both datasets
    Full,
    /// Keep rows from left that have no match in right
    Anti,
    /// Keep rows from left that have a match in right (no right columns)
    Semi,
}

impl JoinType {
    /// Convert to Polars JoinType
    fn to_polars(&self) -> polars::prelude::JoinType {
        match self {
            JoinType::Left => polars::prelude::JoinType::Left,
            JoinType::Right => polars::prelude::JoinType::Right,
            JoinType::Inner => polars::prelude::JoinType::Inner,
            JoinType::Full => polars::prelude::JoinType::Full,
            JoinType::Anti => polars::prelude::JoinType::Anti,
            JoinType::Semi => polars::prelude::JoinType::Semi,
        }
    }
}

/// Join two datasets on specified columns.
///
/// # Arguments
///
/// * `left` - Left dataset
/// * `right` - Right dataset
/// * `left_on` - Column names to join on from left dataset
/// * `right_on` - Column names to join on from right dataset (if None, uses left_on)
/// * `join_type` - Type of join (Left, Right, Inner, Full, Anti, Semi)
/// * `suffix` - Suffix for duplicate column names from right dataset
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::{join, JoinType};
///
/// // Inner join on customer_id
/// let result = join(&orders, &customers, &["customer_id"], None, JoinType::Inner, Some("_right"))?;
///
/// // Left join with different column names
/// let result = join(&orders, &customers, &["cust_id"], Some(&["customer_id"]), JoinType::Left, None)?;
/// ```
pub fn join(
    left: &Dataset,
    right: &Dataset,
    left_on: &[&str],
    right_on: Option<&[&str]>,
    join_type: JoinType,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    // Validate left_on columns exist
    for col in left_on {
        if left.df().column(*col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Determine right_on columns
    let right_on_cols: Vec<&str> = match right_on {
        Some(cols) => cols.to_vec(),
        None => left_on.to_vec(),
    };

    // Validate right_on columns exist
    for col in &right_on_cols {
        if right.df().column(*col).is_err() {
            return Err(MungeError::ColumnNotFound(col.to_string()));
        }
    }

    // Check column count matches
    if left_on.len() != right_on_cols.len() {
        return Err(MungeError::JoinError(format!(
            "left_on has {} columns but right_on has {} columns",
            left_on.len(),
            right_on_cols.len()
        )));
    }

    // Convert to owned strings for Polars (PlSmallStr needs owned data)
    let left_on_str: Vec<String> = left_on.iter().map(|s| s.to_string()).collect();
    let right_on_str: Vec<String> = right_on_cols.iter().map(|s| s.to_string()).collect();

    // Set up join arguments
    let suffix_str: PlSmallStr = suffix.unwrap_or("_right").into();

    // Perform the join
    let result = left
        .df()
        .join(
            right.df(),
            left_on_str.as_slice(),
            right_on_str.as_slice(),
            JoinArgs::new(join_type.to_polars()).with_suffix(Some(suffix_str)),
            None, // JoinTypeOptions
        )
        .map_err(|e| MungeError::JoinError(e.to_string()))?;

    Ok(Dataset::new(result))
}

/// Left join: Keep all rows from left, matching rows from right.
///
/// # Example
///
/// ```ignore
/// let result = left_join(&orders, &customers, &["customer_id"], None, Some("_cust"))?;
/// ```
pub fn left_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Left, suffix)
}

/// Right join: Keep all rows from right, matching rows from left.
///
/// # Example
///
/// ```ignore
/// let result = right_join(&orders, &customers, &["customer_id"], None, None)?;
/// ```
pub fn right_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Right, suffix)
}

/// Inner join: Keep only rows that match in both datasets.
///
/// # Example
///
/// ```ignore
/// let result = inner_join(&orders, &customers, &["customer_id"], None, None)?;
/// ```
pub fn inner_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Inner, suffix)
}

/// Full (outer) join: Keep all rows from both datasets.
///
/// # Example
///
/// ```ignore
/// let result = full_join(&orders, &customers, &["customer_id"], None, None)?;
/// ```
pub fn full_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
    suffix: Option<&str>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Full, suffix)
}

/// Anti join: Keep rows from left that have NO match in right.
///
/// # Example
///
/// ```ignore
/// // Find orders without matching customers
/// let orphan_orders = anti_join(&orders, &customers, &["customer_id"], None)?;
/// ```
pub fn anti_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Anti, None)
}

/// Semi join: Keep rows from left that have a match in right (no right columns added).
///
/// # Example
///
/// ```ignore
/// // Find orders that have matching customers (but don't add customer columns)
/// let matched_orders = semi_join(&orders, &customers, &["customer_id"], None)?;
/// ```
pub fn semi_join(
    left: &Dataset,
    right: &Dataset,
    on: &[&str],
    right_on: Option<&[&str]>,
) -> MungeResult<Dataset> {
    join(left, right, on, right_on, JoinType::Semi, None)
}

/// Concatenate datasets vertically (stack rows).
///
/// All datasets must have the same columns. Missing columns are filled with nulls
/// if `fill_null` is true.
///
/// # Arguments
///
/// * `datasets` - Slice of datasets to concatenate
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::concat;
///
/// let combined = concat(&[&jan_data, &feb_data, &mar_data])?;
/// ```
pub fn concat(datasets: &[&Dataset]) -> MungeResult<Dataset> {
    if datasets.is_empty() {
        return Err(MungeError::EmptyDataset);
    }

    if datasets.len() == 1 {
        return Ok(datasets[0].clone());
    }

    // Get DataFrames
    let dfs: Vec<DataFrame> = datasets.iter().map(|ds| ds.df().clone()).collect();

    // Use diagonal concat to handle missing columns (fills with nulls)
    let result = polars::functions::concat_df_diagonal(&dfs)
        .map_err(|e| MungeError::JoinError(format!("Concat failed: {}", e)))?;

    Ok(Dataset::new(result))
}

/// Concatenate datasets horizontally (add columns side by side).
///
/// All datasets must have the same number of rows.
///
/// # Arguments
///
/// * `datasets` - Slice of datasets to concatenate horizontally
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::hconcat;
///
/// let wide = hconcat(&[&demographics, &survey_responses])?;
/// ```
pub fn hconcat(datasets: &[&Dataset]) -> MungeResult<Dataset> {
    if datasets.is_empty() {
        return Err(MungeError::EmptyDataset);
    }

    if datasets.len() == 1 {
        return Ok(datasets[0].clone());
    }

    // Check all have the same number of rows
    let nrows = datasets[0].nrows();
    for ds in datasets.iter().skip(1) {
        if ds.nrows() != nrows {
            return Err(MungeError::RowCountMismatch {
                expected: nrows,
                found: ds.nrows(),
            });
        }
    }

    // Get DataFrames
    let dfs: Vec<DataFrame> = datasets.iter().map(|ds| ds.df().clone()).collect();

    // Use horizontal concat
    let result = polars::functions::concat_df_horizontal(&dfs, true)
        .map_err(|e| MungeError::JoinError(format!("hconcat failed: {}", e)))?;

    Ok(Dataset::new(result))
}

/// Cross join: Cartesian product of two datasets.
///
/// Every row from left is paired with every row from right.
/// Result has left.nrows() * right.nrows() rows.
///
/// # Arguments
///
/// * `left` - Left dataset
/// * `right` - Right dataset
/// * `suffix` - Suffix for duplicate column names from right dataset
///
/// # Example
///
/// ```ignore
/// use p2a_core::data::munging::cross_join;
///
/// // Create all combinations of products and stores
/// let combos = cross_join(&products, &stores, Some("_store"))?;
/// ```
pub fn cross_join(left: &Dataset, right: &Dataset, suffix: Option<&str>) -> MungeResult<Dataset> {
    let suffix_str: PlSmallStr = suffix.unwrap_or("_right").into();

    let result = left
        .df()
        .cross_join(right.df(), Some(suffix_str), None)
        .map_err(|e| MungeError::JoinError(format!("Cross join failed: {}", e)))?;

    Ok(Dataset::new(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::df;

    fn make_orders() -> Dataset {
        let df = df! {
            "order_id" => [1, 2, 3, 4, 5],
            "customer_id" => [101, 102, 101, 103, 104],
            "amount" => [100.0, 200.0, 150.0, 300.0, 50.0],
        }
        .unwrap();
        Dataset::new(df)
    }

    fn make_customers() -> Dataset {
        let df = df! {
            "customer_id" => [101, 102, 103],
            "name" => ["Alice", "Bob", "Charlie"],
        }
        .unwrap();
        Dataset::new(df)
    }

    #[test]
    fn test_inner_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = inner_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // Only orders with matching customers (order 5 has customer 104 which doesn't exist)
        assert_eq!(result.nrows(), 4);
        assert!(result.df().column("name").is_ok());
        assert!(result.df().column("order_id").is_ok());
    }

    #[test]
    fn test_left_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = left_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // All orders kept, even those without matching customers
        assert_eq!(result.nrows(), 5);
        assert!(result.df().column("name").is_ok());

        // Check that customer 104's order has null name
        let name_col = result.df().column("name").unwrap();
        let names: Vec<Option<&str>> = name_col.str().unwrap().into_iter().collect();
        // The last row (customer_id 104) should have null name
        assert!(names.iter().any(|n| n.is_none()));
    }

    #[test]
    fn test_right_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = right_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // All customers kept
        // Customer 101 has 2 orders, 102 has 1, 103 has 1 = 4 rows
        assert_eq!(result.nrows(), 4);
    }

    #[test]
    fn test_full_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = full_join(&orders, &customers, &["customer_id"], None, None).unwrap();

        // All orders + all customers (5 orders, but customer 101 appears twice)
        // Orders: 1(101), 2(102), 3(101), 4(103), 5(104)
        // Full join keeps all 5 orders
        assert_eq!(result.nrows(), 5);
    }

    #[test]
    fn test_anti_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = anti_join(&orders, &customers, &["customer_id"], None).unwrap();

        // Only order 5 (customer 104) has no matching customer
        assert_eq!(result.nrows(), 1);

        let order_ids: Vec<i32> = result
            .df()
            .column("order_id")
            .unwrap()
            .i32()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(order_ids, vec![5]);
    }

    #[test]
    fn test_semi_join() {
        let orders = make_orders();
        let customers = make_customers();

        let result = semi_join(&orders, &customers, &["customer_id"], None).unwrap();

        // Orders with matching customers, but no customer columns added
        assert_eq!(result.nrows(), 4);
        assert!(result.df().column("name").is_err()); // name column not added
        assert!(result.df().column("order_id").is_ok());
    }

    #[test]
    fn test_join_with_different_column_names() {
        let df1 = df! {
            "id" => [1, 2, 3],
            "value" => [10, 20, 30],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "ref_id" => [1, 2, 4],
            "label" => ["A", "B", "D"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = left_join(&ds1, &ds2, &["id"], Some(&["ref_id"]), None).unwrap();

        assert_eq!(result.nrows(), 3);
        assert!(result.df().column("label").is_ok());
    }

    #[test]
    fn test_concat() {
        let df1 = df! {
            "x" => [1, 2],
            "y" => ["a", "b"],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "x" => [3, 4],
            "y" => ["c", "d"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = concat(&[&ds1, &ds2]).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_concat_with_missing_columns() {
        let df1 = df! {
            "x" => [1, 2],
            "y" => ["a", "b"],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "x" => [3, 4],
            "z" => [10.0, 20.0],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        // Diagonal concat fills missing columns with nulls
        let result = concat(&[&ds1, &ds2]).unwrap();

        assert_eq!(result.nrows(), 4);
        assert_eq!(result.ncols(), 3); // x, y, z
    }

    #[test]
    fn test_hconcat() {
        let df1 = df! {
            "x" => [1, 2, 3],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "y" => ["a", "b", "c"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = hconcat(&[&ds1, &ds2]).unwrap();

        assert_eq!(result.nrows(), 3);
        assert_eq!(result.ncols(), 2);
        assert!(result.df().column("x").is_ok());
        assert!(result.df().column("y").is_ok());
    }

    #[test]
    fn test_hconcat_row_mismatch() {
        let df1 = df! {
            "x" => [1, 2, 3],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "y" => ["a", "b"],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = hconcat(&[&ds1, &ds2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_join() {
        let df1 = df! {
            "product" => ["A", "B"],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "store" => [1, 2, 3],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = cross_join(&ds1, &ds2, None).unwrap();

        // 2 products x 3 stores = 6 rows
        assert_eq!(result.nrows(), 6);
        assert!(result.df().column("product").is_ok());
        assert!(result.df().column("store").is_ok());
    }

    #[test]
    fn test_join_column_not_found() {
        let orders = make_orders();
        let customers = make_customers();

        let result = inner_join(&orders, &customers, &["nonexistent"], None, None);
        assert!(matches!(result, Err(MungeError::ColumnNotFound(_))));
    }

    #[test]
    fn test_join_with_suffix() {
        let df1 = df! {
            "id" => [1, 2],
            "value" => [10, 20],
        }
        .unwrap();
        let ds1 = Dataset::new(df1);

        let df2 = df! {
            "id" => [1, 2],
            "value" => [100, 200],
        }
        .unwrap();
        let ds2 = Dataset::new(df2);

        let result = inner_join(&ds1, &ds2, &["id"], None, Some("_other")).unwrap();

        assert!(result.df().column("value").is_ok());
        assert!(result.df().column("value_other").is_ok());
    }
}
