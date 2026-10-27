//! Parquet files for the tests of the check a parquet file goes through
//! before it is decoded (tests/dataset_quotas.rs, tests/sessions.rs).

use std::path::Path;

use polars::prelude::{DataFrame, NamedFrom, ParquetWriter, Series};
use polars_parquet_format::FileMetaData;
use polars_parquet_format::thrift::protocol::{TCompactInputProtocol, TCompactOutputProtocol};

/// Write a parquet file of `rows` rows of one float column `x` at `path`,
/// the values cycling through 97 noisy levels: dictionary-encoded, so a few
/// kilobytes on disk and 8 bytes a row once decoded. `group_rows` is the
/// number of rows in a row group (`None`: the writer's default).
pub fn levels_parquet(path: &Path, rows: usize, group_rows: Option<usize>) {
    let values: Vec<f64> = (0..rows).map(|i| (i % 97) as f64 * 0.37 + 0.1).collect();
    let mut df = DataFrame::new(vec![Series::new("x".into(), values).into()]).unwrap();
    ParquetWriter::new(std::fs::File::create(path).unwrap())
        .with_row_group_size(group_rows)
        .finish(&mut df)
        .unwrap();
}

/// Rewrite the footer of the parquet file at `path` with `edit`. The pages
/// stay where they are, so the file decodes as before wherever the decoder
/// does not follow the edited fields.
pub fn rewrite_footer(path: &Path, edit: impl FnOnce(&mut FileMetaData)) {
    let bytes = std::fs::read(path).unwrap();
    let n = bytes.len();
    assert_eq!(&bytes[n - 4..], b"PAR1");
    let len = u32::from_le_bytes(bytes[n - 8..n - 4].try_into().unwrap()) as usize;
    let start = n - 8 - len;
    let mut input = TCompactInputProtocol::new(&bytes[start..n - 8], usize::MAX);
    let mut footer = FileMetaData::read_from_in_protocol(&mut input).unwrap();
    edit(&mut footer);
    let mut written = Vec::new();
    footer
        .write_to_out_protocol(&mut TCompactOutputProtocol::new(&mut written))
        .unwrap();
    let mut out = bytes[..start].to_vec();
    out.extend_from_slice(&written);
    out.extend_from_slice(&u32::try_from(written.len()).unwrap().to_le_bytes());
    out.extend_from_slice(b"PAR1");
    std::fs::write(path, out).unwrap();
}
