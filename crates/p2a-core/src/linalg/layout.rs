//! Memory layout helpers.

use ndarray::ArrayView2;

use crate::errors::EconResult;
use crate::resource;

/// The elements of `arr` in logical row-major order, whatever its memory layout: the slice
/// itself for a standard-layout view, else one pass over the elements in row order
/// (a transposed, Fortran-order or strided view). `to_owned` would keep a Fortran layout, so
/// its raw vector would be column-major.
pub(crate) fn row_major_copy(arr: &ArrayView2<f64>) -> EconResult<Vec<f64>> {
    if let Some(slice) = arr.as_slice() {
        let mut v = resource::try_with_capacity(slice.len(), "row-major copy")?;
        v.extend_from_slice(slice);
        return Ok(v);
    }
    let mut v = resource::try_with_capacity(arr.len(), "row-major copy")?;
    v.extend(arr.iter().copied());
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array, s};

    /// The upload logic before REG-33: the slice of a standard view, else `to_owned`'s raw
    /// vector.
    fn old_copy(arr: &ArrayView2<f64>) -> Vec<f64> {
        match arr.as_slice() {
            Some(s) => s.to_vec(),
            None => arr.to_owned().into_raw_vec_and_offset().0,
        }
    }

    /// REG-33: a transposed view, its Fortran-order owned copy and a strided view are copied
    /// in logical row-major order.
    #[test]
    fn test_row_major_copy_of_transposed_and_strided_views() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let want = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let t = a.t();
        assert_eq!(row_major_copy(&t).unwrap(), want);
        let f: Array2<f64> = a.t().to_owned();
        assert_eq!(row_major_copy(&f.view()).unwrap(), want);
        // the old logic sent the Fortran-order copy column-major
        assert_ne!(old_copy(&f.view()), want);
        let big = Array2::from_shape_fn((4, 5), |(i, j)| (10 * i + j) as f64);
        let strided = big.slice(s![..;2, 1..;2]);
        assert_eq!(
            row_major_copy(&strided).unwrap(),
            vec![1.0, 3.0, 21.0, 23.0]
        );
        assert_eq!(
            row_major_copy(&a.view()).unwrap(),
            a.iter().copied().collect::<Vec<_>>()
        );
    }
}
