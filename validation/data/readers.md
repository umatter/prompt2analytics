# Stata and SAS file readers

`p2a_core::data::load_stata` (`crates/p2a-core/src/data/stata.rs`) and `p2a_core::data::load_sas`
(`crates/p2a-core/src/data/sas.rs`), reached from `p2a data load` and the MCP `load_dataset` tool
through `DataLoader::load`. Pinned to **haven 2.5.5** (`read_dta`, `read_sas`; the ReadStat C
library bundled with haven) on R 4.3.3, on committed binary fixtures.

| | |
|---|---|
| R case | `data_readers_values` in `validation/r/cases/data_readers.R` |
| Expected values | `validation/expected/data_readers_values.json` |
| Fixtures | `validation/datasets/files/` ([README](../datasets/files/README.md)), written by `validation/r/fixtures/data_readers.R` |
| Registry | `validation/methods/data_readers.json` (`data_read_dta`, `data_read_sas7bdat`) |
| Tolerance | exact (`Tol::exact()`): both readers decode the same bytes; text compared byte for byte |
| Review findings | DATA-18, DATA-19, DATA-20, DATA-21, DATA-22, DATA-47, DATA-48, DATA-50, DATA-51, DATA-73 (T3-WP51) |

## What is compared

For every fixture the case records the shape, the column names in file order, every numeric column
(R `NA` and every tagged missing value as `null`) and every text column (the strings haven returns,
with their UTF-8 byte lengths, so `run.R --check` notices a changed decoding). The Rust helper
`data::sas::fixture_check::assert_matches_haven` compares a reader's `DataFrame` with all of it.
Stata byte/int/long columns load as Int64 and compare after a cast to f64. A text cell haven reads
as `""` is `""` in both readers; `str_replace_value(column, "", null)` turns such cells into missing
values.

Differences from haven, recorded as deviations of the registry entries: Stata byte, int and long load
as Int64 (haven: doubles); tagged missing values `.a`-`.z` load as plain null (haven keeps the tag);
strL contents are not decoded (cells hold the placeholder `"<strL>"`, DATA-74) and value labels and
display formats are not read (DATA-75); SAS non-ASCII text under an encoding code outside 0, 20, 28, 29,
62, 204 is refused (ReadStat converts it with iconv).

The case also records what ReadStat does with the corrupted copies the Rust guard tests build
(`guard__*`: rows read, or -1 when ReadStat refuses the file):

| Key | Patch | ReadStat |
|---|---|---|---|
| `guard__sas_page_type_0x0280` | `sas32_iris` page 0 type 0x0280 | 150 rows |
| `guard__sas_page_type_0x4000` | `sas64_haven` page 0 type 0x4000 | 60 rows |
| `guard__sas_page_type_0x0300` | `sas64_haven` page 1 type 0x0300 | refused |
| `guard__sas_row_count_59` | row count 60 -> 59 | 59 rows |
| `guard__sas_row_count_61` | row count 60 -> 61 | refused |
| `guard__sas_type_code_3` | column 0 type code 3 | refused |
| `guard__sas_char_only_row_count_2` | `sas32_iris`, every column text, row count 2 | 2 rows |
| `guard__sas_wlatin1_byte_0x81` | `sas64_wlatin1`, byte 0x81 in a text cell | refused |
| `guard__sas_wlatin1_code_204` | `sas64_wlatin1`, encoding byte 204 | 60 rows, as code 62 |
| `guard__dta_k0_n3`, `guard__dta_k0_n2p40` | `dta118_haven`, `<K>` = 0, `<N>` = 3 or 2^40 | refused |

`sas64_wlatin1_code29__grp` is haven's read of `sas64_wlatin1` with encoding byte 29 (ISO-8859-1:
byte 0x80 is U+0080, not the euro sign).

## Tests

Tests that do not load `data_readers_values` (byte offsets, decoders, guards) are listed here but not in
`validation/methods/data_readers.json`.

| Test | Pins |
|---|---|
| `p2a-core::data::sas::tests::test_sas_fixture_offsets` | the byte offsets the SAS tests patch |
| `p2a-core::data::sas::tests::test_validate_data_readers_manifest_lists_every_fixture` | every fixture exists and is in the case |
