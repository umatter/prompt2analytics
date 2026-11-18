# Cleaning layer: quality profile, suggestions, previews, verification, sessions

The LLM-assisted cleaning layer of `p2a-core::data` (`quality.rs`, `suggestion.rs`, `verification.rs`,
`cleaning_session.rs`) and the MCP tools over it (`data_quality_profile`, `suggest_cleaning`,
`preview_cleaning`, `verify_cleaning`, `cleaning_session_*`, `cleaning_rollback`). No R function is
its reference, so nothing here is R-pinned: the tests check internal consistency, chiefly that a
preview agrees with the operation it previews and that every suggestion can be applied as given.

## Behaviour

- **Patterns.** Each distinct text value gets at most one pattern, the first it matches in the order
  email, date, phone, numeric text: "2024-01-15" is a date only, "555-123-4567" a phone number only,
  "12.5" numeric text, and "1-2-3-" none (numeric text must parse as a number). A column reports
  `InconsistentFormat` only when its values have more than one pattern.
- **Completeness** is the share of present cells, a null and a NaN in a float column both counting as
  missing (as `describe` counts them): x = [1, null, NaN, 4], y = ["a", null, "c", "d"] gives 5/8.
- **Suggestions** name an operation `preview_cleaning` and `cleaning_session_apply` accept with the
  arguments printed under it: outliers are one or two `filter`s on the fences (`>=` the lower fence,
  `<=` the upper fence, each only when values lie beyond it; the filter also drops the rows where the
  column is missing); a column more than 70% missing, and a constant column, `drop_column`; empty
  strings `replace` with old value `""` and value `null`; mixed types and formats are guidance only
  (`none`).
- **Previews** run the operation they preview. `rows_affected` counts distinct rows (at most 100%),
  `cells_affected` the cells changed; `estimated_quality_change` is the completeness change in
  percentage points between the frame and the operation's result. Filter and drop-missing previews
  call the apply's own `filter` and `drop_na`, so an unknown operator, a missing column, `contains`
  (literal text) and NaN behave as in the apply. A frame with no rows previews 0%.
- **Verification** matches issues by kind and column: an issue on both sides whose count changed is
  one `issues_changed` line ("missing values in 'x': 12 -> 3"), not one resolved plus one introduced.
  `rows_modified` counts rows, `cells_modified` cells.
- **Sessions.** Each checkpoint records the checkpoint it was applied to; undo returns there, so undo
  after an undo and a new operation returns to the branch point, not to the undone state.

## Tests

| Test | Pins |
|---|---|
| `p2a-core::data::quality::tests::iso_date_and_phone_columns_are_consistent` | one pattern per value; date and phone columns consistent (DATA-46) |
| `p2a-core::data::quality::tests::nan_count_survives_a_refused_value_buffer` | the NaN count under a 64-byte ceiling |
| `p2a-core::data::quality::tests::completeness_counts_nan_as_missing` | 5/8 on the frame above |
| `p2a-core::data::suggestion::tests::every_suggestion_is_an_accepted_operation` | operations and their arguments (DATA-52) |
| `p2a-core::data::verification::tests::trim_preview_counts_rows_not_cells` | rows 2, cells 3, 50% (DATA-53) |
| `p2a-core::data::verification::tests::fill_na_preview_counts_nan_and_reports_completeness` | NaN filled, +25.0 points (DATA-53, DATA-54) |
| `p2a-core::data::verification::tests::filter_preview_agrees_with_filter` | preview rows = the apply's removed rows; literal `contains`; errors as warnings (DATA-53) |
| `p2a-core::data::verification::tests::empty_frame_preview` | 0 rows: 0%, no "ALL rows" |
| `p2a-core::data::verification::tests::nan_constant_fill_preview_warns` | a NaN constant fill is refused with the apply's message |
| `p2a-core::data::verification::tests::drop_columns_preview_reports_completeness` | `DropColumns` preview |
| `p2a-core::data::verification::tests::first_changes_and_changed_mask_agree` | the two change scans agree; NaN to NaN is no change |
| `p2a-core::data::verification::tests::verify_reports_a_count_change_once` | `issues_changed` (DATA-55) |
| `p2a-core::data::verification::tests::verify_counts_modified_rows` | rows 2, cells 3 (DATA-54) |
| `p2a-core::data::cleaning_session::tests::rollback_after_reapply_returns_to_the_branch_point` | undo after re-apply (DATA-31) |
| `p2a-core::data::cleaning_session::tests::rollback_to_then_apply_records_the_parent` | `rollback_to` then apply (DATA-31) |
| `p2a-mcp::tier3::wp28_cleaning::cleaning_session_undo_follows_the_branch` | the same through the MCP tools (DATA-31) |
| `p2a-mcp::tier3::wp28_cleaning::every_suggestion_applies` | every printed suggestion applies as written (DATA-52) |
| `p2a-mcp::tier3::wp28_cleaning::preview_and_apply_drop_column` | `drop_column` preview, apply and undo |

## Timing

Release builds (`--no-default-features --features ci`), v0.5.0 (`f80b0131`) against this branch, one
harness binary per side pinned to core 2 (`taskset -c 2`, `RAYON_NUM_THREADS=1`, `POLARS_MAX_THREADS=1`,
cwd `/dev/shm`); 9 interleaved processes per side, each the median of 2,001 / 301 / 41 calls (n = 100 /
1,000 / 10,000) after 10 warm-ups; the table gives the median of the 9 medians. Seeded data with noise
(ChaCha8 in the harness, `set.seed(n)` for the files R writes). Quiet re-time at the
integration (2026-11-18, merged head 156e744f, nothing else running): 9 rounds, load average 1.5-2.3;
"minima" is the ratio of the fastest of the 9 processes per side.

| Hot path | n | v0.5.0 (µs) | head (µs) | head/v0.5.0 | minima |
|---|---|---|---|---|---|
| quality profile (5 columns: 2 numeric with NaN, ISO dates, phones, words) | 100 | 153.5 | 162.8 | 1.06x | 1.06x |
| quality profile (5 columns: 2 numeric with NaN, ISO dates, phones, words) | 1,000 | 1,027.2 | 1,126.5 | 1.10x | 1.10x |
| quality profile (5 columns: 2 numeric with NaN, ISO dates, phones, words) | 10,000 | 10,104.1 | 11,026.7 | 1.09x | 1.09x |
| estimate_dataset_memory (10 columns, 4 text) | 100 | 0.07 | 0.31 | 4.45x | 4.35x |
| estimate_dataset_memory (10 columns, 4 text) | 1,000 | 0.07 | 0.30 | 4.39x | 4.26x |
| estimate_dataset_memory (10 columns, 4 text) | 10,000 | 0.07 | 0.31 | 4.54x | 4.41x |
| preview filter `>=` (Float64) | 100 | 1.7 | 24.0 | 14.39x | 14.40x |
| preview filter `>=` (Float64) | 1,000 | 10.6 | 31.5 | 2.98x | 3.00x |
| preview filter `>=` (Float64) | 10,000 | 99.3 | 208.1 | 2.10x | 2.10x |
| preview filter `contains` (text) | 100 | 5.8 | 28.7 | 4.91x | 4.92x |
| preview filter `contains` (text) | 1,000 | 55.9 | 61.9 | 1.11x | 1.10x |
| preview filter `contains` (text) | 10,000 | 680.7 | 520.3 | 0.76x | 0.76x |
| preview drop_na (3 columns, 10% missing) | 100 | 21.1 | 28.2 | 1.34x | 1.36x |
| preview drop_na (3 columns, 10% missing) | 1,000 | 209.6 | 44.1 | 0.21x | 0.21x |
| preview drop_na (3 columns, 10% missing) | 10,000 | 2,102.8 | 202.8 | 0.10x | 0.10x |
| preview fill_na median (10% missing) | 100 | 9.5 | 8.6 | 0.91x | 0.92x |
| preview fill_na median (10% missing) | 1,000 | 33.3 | 30.6 | 0.92x | 0.93x |
| preview fill_na median (10% missing) | 10,000 | 257.4 | 247.0 | 0.96x | 0.95x |
| preview trim | 100 | 3.4 | 3.2 | 0.93x | 0.94x |
| preview trim | 1,000 | 18.9 | 16.4 | 0.87x | 0.88x |
| preview trim | 10,000 | 204.3 | 184.6 | 0.90x | 0.93x |
| preview replace, no match | 100 | 4.7 | 1.9 | 0.40x | 0.40x |
| preview replace, no match | 1,000 | 30.9 | 5.2 | 0.17x | 0.17x |
| preview replace, no match | 10,000 | 310.0 | 61.5 | 0.20x | 0.19x |

Above 1.10x:
- `estimate_dataset_memory` reads polars' buffer sizes (`estimated_size`) instead of multiplying the row
  count by a fixed width: 0.07 to 0.31 µs a call (4.4-4.5x), whatever the rows.
- The filter preview runs the apply's own `filter` and materialises the kept rows, where v0.5.0 counted
  a mask of its own (which treated `contains` as a regex and an unknown operator as no rows): up to
  0.11 ms at 10,000 rows (99.3 to 208.1 µs) (the NaN count of the completeness change is vectorised). A mask-only
  filter would have to come from the munging module.
- `contains` at 100 rows and drop_na at 100 rows: the same fixed cost of the real operation (5.8 to 28.7 µs and
  21.1 to 28.2 µs); at 1,000 rows and above both are as fast or faster (drop_na 4-7x faster: one vectorised
  mask instead of a per-cell lookup).
- The quality profile (1.06-1.10x): one pattern per distinct value, a number parse for digit
  text, and the NaN count taken apart from the value buffer.

Rule 5: the filter preview runs above its pre-approved ceiling of 1.5x (14.4x / 3.0x / 2.1x for `>=`,
4.9x for `contains` at 100 rows), and `estimate_dataset_memory` and the drop_na preview at 100 rows were
in no earlier sign-off: all signed off by the user on 2026-11-18 at these figures.

