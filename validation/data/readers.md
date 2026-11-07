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

## Stata `.dta` (releases 117, 118, 119)

The header is read tag by tag (`<stata_dta><header><release>`, `<byteorder>`, `<K>`, `<N>`,
`<label>`, `<timestamp>`), then the 14 `<map>` offsets; the reader seeks to slot 2
(`<variable_types>`), 3 (`<varnames>`) and 9 (`<data>`) and checks the tag found there (DATA-22).
Before, it scanned forward from slot 1 (`<map>` itself) for `<K>`, which sits in the header, so every
real file failed with `Tag not found: <K>`.

| Release | `<K>` | `<N>` | label length | variable name | text |
|---|---|---|---|---|---|
| 117 (Stata 13) | u16 | u32 | u8 | 33 bytes | UTF-8 if valid, else Windows-1252 |
| 118 (Stata 14-18) | u16 | u64 | u16 | 129 bytes | UTF-8 |
| 119 (Stata 15-18, > 32,767 variables) | u32 | u64 | u16 | 129 bytes | UTF-8 |

(DATA-51.) Release 117 has no encoding field: Stata 13 writes the platform encoding (Windows-1252 on
Windows), haven and ReadStat write UTF-8. haven's own default (`encoding = NULL`) reads release 117
as Windows-1252, so it shows haven's own UTF-8 117 files as mojibake; p2a reads UTF-8 when the bytes
are valid UTF-8, and the case pins both files with the encoding that gives the intended text
(`dta117_haven`: `"UTF-8"`, `dta117_latin1`: `"windows-1252"`; both give `"Zürich"`, `"€5"`). Recorded
as a registry deviation of `data_read_dta`.

Missing values: byte above 100, int above 32,740, long above 2,147,483,620, float above 0x7EFFFFFF
(1.7014117331926443e38) and double above 0x7FDFFFFFFFFFFFFF (8.9884656743115785e307) are `.` or
`.a`-`.z` and load as null. `dta118_types` holds each type's extremes: the largest float and double
were null before (the old thresholds 1.701e38 and 8.988e307 lie below them). A file with `<K>` = 0 is
refused, whatever `<N>` says (DATA-73; Stata cannot save one, ReadStat refuses it). Formats 102-116
and releases above 119 are `UnsupportedVersion`. Not decoded (Tier 4): strL contents (DATA-74, the
placeholder `"<strL>"` stays) and value labels and display formats (DATA-75).

## SAS `.sas7bdat`

| Field | Where |
|---|---|
| 64-bit integers | byte 32 is 0x33: offsets, lengths and counts are 8 bytes, else 4 |
| header pad | byte 35 is 0x33: 4 bytes after byte 164 (DATA-18; before, the pad came from byte 32) |
| byte order | byte 37: 0x01 little-endian, 0x00 big-endian |
| header length, page size, page count | at 196, 200 and 204 plus the pad |
| page header (B = 16 or 32) | type at B, block count at B + 2, subheader count at B + 4 (DATA-20; before, swapped) |
| subheader pointer | offset, length (int), compression (u8), type (u8), from B + 8 |
| column attribute vector | offset (int), width (4), name flag (2), type (1) at int + 6 (DATA-19; before, int + 4) |
| column name pointer | text block, offset, length (u16 each) into the column-text blocks (DATA-47; before, `VAR1`..`VARn`) |

## Tests

Tests that do not load `data_readers_values` (byte offsets, decoders, guards) are listed here but not in
`validation/methods/data_readers.json`.

| Test | Pins |
|---|---|
| `p2a-core::data::stata::tests::test_stata_type_from_code` | type codes 65530/65526/100 -> byte, double, str100 |
| `p2a-core::data::stata::tests::test_stata_invalid_magic` | a non-dta file is an error |
| `p2a-core::data::stata::tests::test_stata_fixture_offsets` | the byte offsets the Stata tests patch |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_118` | `dta118_haven` = haven (DATA-22) |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_118_stata_written` | `dta118_stata_iris` (Stata 14) = haven (DATA-22, DATA-50) |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_117` | `dta117_haven` = haven with `encoding = "UTF-8"` (DATA-51) |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_117_windows_1252` | `dta117_latin1` = haven with `encoding = "windows-1252"` (DATA-51) |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_119` | `dta119_haven` = haven: u32 `<K>` (DATA-51) |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_119_big_endian` | `dta119_msf` = haven (DATA-51) |
| `p2a-core::data::stata::tests::test_stata_119_k_is_four_bytes` | a release-119 `<K>` of 65542 is read as such (DATA-51) |
| `p2a-core::data::stata::tests::test_validate_read_haven_dta_118_storage_types` | `dta118_types` = haven: byte/int/long/float/double extremes and missing codes |
| `p2a-core::data::stata::tests::test_validate_dataloader_loads_dta_fixture` | `DataLoader::load` of `dta118_haven` = haven |
| `p2a-core::data::stata::tests::test_validate_stata_k0_n_positive_is_error` | `<K>` = 0 refused for N = 3, 0, 2^40; ReadStat refuses too (DATA-73) |
| `p2a-core::data::stata::tests::test_stata_column_buffers_respect_the_allocation_ceiling` | column buffers under a 64-byte ceiling are refused (G3) |
| `p2a-core::data::stata::tests::test_stata_map_and_header_guards` | map offsets past EOF or off their tag, N and K bounds (DATA-22) |
| `p2a-core::data::stata::tests::test_stata_pre_117_formats_are_unsupported` | formats 102-116 and release 120 are `UnsupportedVersion` |
| `p2a-core::data::sas::tests::test_magic_number` | the 32-byte magic |
| `p2a-core::data::sas::tests::test_sas_invalid_magic` | a non-SAS file is refused for its magic |
| `p2a-core::data::sas::tests::test_sas_truncated_magic` | a 16-byte file is refused |
| `p2a-core::data::sas::tests::test_sas_fixture_offsets` | the byte offsets the SAS tests patch |
| `p2a-core::data::sas::tests::test_validate_data_readers_manifest_lists_every_fixture` | every fixture exists and is in the case |
| `p2a-core::data::sas::tests::test_sas_header_fields_follow_byte_35` | header length, page size, page count with and without the pad (DATA-18) |
| `p2a-core::data::sas::tests::test_sas_page_header_counts` | 60, 60 and 150 rows from the block and subheader counts (DATA-20) |
| `p2a-core::data::sas::tests::test_sas_column_types_from_the_attribute_vector` | column types from int + 6 (DATA-19) |
| `p2a-core::data::sas::tests::test_validate_sas_unknown_column_type_is_error` | type code 3 refused; ReadStat refuses too (DATA-19) |
| `p2a-core::data::sas::tests::test_validate_sas_column_names_from_the_column_text` | names of `sas64_haven` and `sas32_iris` = haven (DATA-47) |
| `p2a-core::data::sas::tests::test_sas_name_pointer_outside_text_is_error` | a name pointer outside the column text is refused (DATA-47) |
| `p2a-core::data::sas::tests::sas_rejects_zero_row_length` | T0 (DATA-49): row length 0 refused |
| `p2a-core::data::sas::tests::sas_rejects_negative_fields` | T0 (DATA-49): negative width and offset, oversized subheader and block counts |
| `p2a-core::data::sas::tests::sas_bounds_rows_and_columns_by_the_layout` | T0 (DATA-49): row count, page capacity, column widths and extents |
| `p2a-core::data::sas::tests::test_validate_sas_stops_at_row_count_for_character_only_files` | T0 (DATA-49): one row counter; ReadStat reads 2 and 59 rows too |
