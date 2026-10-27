//! Request limits: the catalogue of cost-bearing request fields and the budgets that bound them.
//!
//! # Profiles
//!
//! [`Limits::off`] is the default for local use (stdio, CLI-style HTTP, the embedded desktop
//! server). It has no family cap, no row cap, no minimum tolerance and no `Option` budget, so it
//! checks nothing: every request that deserialises is accepted. [`Limits::hosted`] is the profile
//! of the public no-login deployment. Every hosted cap is at least 1.5 times the default of the
//! tool fields it bounds, so default requests are never refused. [`Limits::with_overrides`]
//! adjusts either profile from `key=value` pairs (`P2A_LIMIT_OVERRIDES`); [`Limits::known_keys`]
//! lists the valid keys.
//!
//! # What is checked
//!
//! - [`Limits::check_tool_arguments`] checks a tool call's JSON arguments against
//!   [`PARAM_RULES`]: each rule maps a `(tool, path)` to a cost [`Family`] (or another [`Rule`]).
//!   Rules address parameters by name, but a derived request struct also takes a JSON array,
//!   filling its fields by position. Outside the off profile the check therefore refuses
//!   arguments that are not a JSON object (`parameter` `arguments`), and a nested struct a
//!   rule path goes into (`sections[]` for `sections[].content`) given as an array (`parameter`
//!   the struct's path). The governed call path refuses top-level arguments that are not an
//!   object in every profile as well; in the off profile a nested struct given by position
//!   still runs, as nothing is checked there. A nested struct no rule goes into (only `Exempt`
//!   rules below it, or none) is not looked at in any profile. [`struct_paths`] lists the
//!   structs a tool's rules go into; the audit log replaces one given as an array by its size in
//!   every profile. A rule path must therefore not go on into a field whose schema admits an
//!   array; the schema test in `rules.rs` fails on one that does.
//! - [`Limits::check_rows`] checks a dataset's height for the tools in [`ROW_RULES`], whose cost
//!   grows faster than linearly in the number of rows ([`RowClass`]).
//! - [`Limits::check_pivot`] checks the wide table a tool in [`PIVOT_RULES`] would make of the
//!   dataset it names, counted from the dataset's distinct values, against
//!   [`Limits::pivot_columns`] and [`Limits::pivot_cells`].
//! - [`Limits::check_budget`] lets a handler check a quantity it computes itself (for example a
//!   predicted join size) against a named budget field such as [`Limits::join_output_rows`].
//! - [`Limits::check_arima_fit`] and [`Limits::check_arima_series`] let the ARIMA handlers
//!   refuse a fit before it starts, because the `arima` crate's optimiser cannot be stopped
//!   once it runs: one whose work exceeds [`Limits::arima_fit_work`], and one whose series holds
//!   a nonzero value below [`ARIMA_MIN_MAGNITUDE`], small enough that the fit's arithmetic on it
//!   can become subnormal and much slower.
//!
//! Every refusal is a [`LimitViolation`] whose message tells an LLM caller what to change. It never
//! contains filesystem paths, environment variable names or internals; `override_key` names the
//! knob for server logs.
//!
//! # Classifying a new request field
//!
//! These request fields of every registered tool must be matched by a rule:
//!
//! - an integer field, by [`Rule::Max`] with a numeric family (not [`Family::StringBytes`]);
//! - a float field, by a numeric [`Rule::Max`] (a count or size given as a float, such as the
//!   `n` of `power_t_test`) or, for a convergence tolerance, by [`Rule::Tolerance`];
//! - an array field, an array element that is itself an array (`a[]`) or a tuple, by
//!   [`Rule::ArrayLen`], [`Rule::UniqueList`] or [`Rule::SquareLen`];
//! - the integer or float elements of an array (`Vec<usize>`, `Vec<f64>`, a tuple of numbers),
//!   which the test lists at the element path `a[]`, by [`Rule::ArrayElemMax`] on the array's
//!   path `a`, or by a numeric [`Rule::Max`] on `a[]`;
//! - any of these, alternatively, by `Rule::Exempt` on the same path with the reason the value
//!   costs nothing. An `Exempt` on `a[]` covers the elements only, never the array's length.
//!
//! Add a [`ParamRule`] to `PARAM_RULES` in `rules.rs`: a generic `"*"` rule when the field name
//! means the same thing in every tool, a tool-specific rule otherwise. The schema test in
//! `rules.rs` (built with the `http` feature) fails and names each unclassified `(tool, path)`.
//! It also fails, rather than skipping the field, when a field is nested deeper than the walk
//! goes (`MAX_DEPTH` in the schema test, eight levels of objects or arrays).
//!
//! A field that can hold any JSON can only be `Exempt`: a `serde_json::Value`, but also an
//! untagged enum with a `Value` arm, or a field whose schema refers to a definition the test
//! cannot find. Give each of its cost sub-fields a rule of its own by path, because the schema
//! does not list them and the test cannot find them. A rule path cannot address the values of a
//! map (`HashMap`, `BTreeMap`), so a map whose values are integers, floats, arrays, objects or
//! any JSON can only be `Exempt`; give a cost field a struct or a list instead. A map at the top
//! level of a request (a `#[serde(flatten)]` map) cannot even be exempted, because a rule path is
//! never empty, so the test fails on it whatever its values. Do not give a cost field a
//! `#[serde(alias = ...)]`. If one is unavoidable, add the same rule for the alias name, because
//! the schema, and so the schema test, does not list aliases.
//!
//! # Installation
//!
//! `transport::start_transport` (for stdio) and, on every HTTP start path (`start_http_transport`,
//! `start_embedded_server`, `bind` + `serve`), `transport::http::serve` (for the bound socket)
//! compute the process's limits with [`resolve`] (the deployment's resource profile plus
//! `P2A_LIMIT_OVERRIDES`) and install them with [`install`]. A caller that serves
//! `transport::http::build_app`'s router itself installs them (see `build_app`). Each server holds
//! its limits while any of its connections or websocket sessions is served, and a start whose
//! limits differ from those of a server still running in the process is refused
//! (`transport::http::hold_limits`). Every tool
//! call is checked against [`current`] before its arguments are deserialised: governed HTTP-side
//! calls in `AnalyticsServer::call_tool_with_session` (the governor carries the limits onto its
//! blocking thread), stdio calls in the server's `ServerHandler::call_tool`. Handlers read named
//! budgets through [`current`]. Until something is installed, [`current`] returns the off profile.

// `LimitViolation` (five `String`s and a `&'static str`) is the stable error type of this API,
// which the handlers use. It is built only when a request is refused, once per tool call at
// most, so the size of `Result<(), LimitViolation>` does not matter here and boxing it would
// change that API.
#![allow(
    clippy::result_large_err,
    reason = "LimitViolation is the fixed, rarely constructed error type of the limits API"
)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock, PoisonError, RwLock};

use p2a_core::polars::prelude::{DataFrame, PolarsError};
use serde_json::Value;

mod distinct;
mod rules;
pub use rules::{ARIMA_MIN_MAGNITUDE, PARAM_RULES, PIVOT_RULES, ROW_RULES, arima_fit_work};

/// Generates [`Family`], [`Family::ALL`], [`Family::key`] and `Family::hosted_cap` from one list
/// (`Variant => "key", hosted cap;`), so that the enum, its list of variants and its keys and
/// caps cannot drift apart.
macro_rules! families {
    ($($(#[$attr:meta])* $variant:ident => $key:literal, $cap:expr;)*) => {
        /// A family of cost parameters that share one cap (for example every bootstrap
        /// replication count).
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Family {
            $($(#[$attr])* $variant,)*
        }

        impl Family {
            /// Every family, in declaration order.
            pub const ALL: &'static [Family] = &[$(Family::$variant,)*];

            /// The family's override key, also shown in refusals (for example `replications`).
            pub fn key(self) -> &'static str {
                match self {
                    $(Family::$variant => $key,)*
                }
            }

            /// The family's cap in the hosted profile.
            fn hosted_cap(self) -> u64 {
                match self {
                    $(Family::$variant => $cap,)*
                }
            }
        }
    };
}

families! {
    /// Bootstrap and permutation replications.
    Replications => "replications", 10_000;
    /// Monte Carlo simulation sizes.
    SimulationSize => "simulation_size", 100_000;
    /// Trees of a forest or committee members.
    Trees => "trees", 5_000;
    /// Boosting rounds and additive terms.
    BoostingIterations => "boosting_iterations", 10_000;
    /// MCMC or simulation draws, burn-in included.
    McmcDraws => "mcmc_draws", 20_000;
    /// Cross-validation folds.
    Folds => "folds", 20;
    /// Iteration limits of optimisers.
    Iterations => "iterations", 100_000;
    /// Random restarts.
    Restarts => "restarts", 100;
    /// Grid sizes (density grids, lambda paths, MTE grids).
    GridPoints => "grid_points", 10_000;
    /// Histogram and binning bins.
    Bins => "bins", 1_000;
    /// Lags, lag orders and seasonal periods.
    Lags => "lags", 1_000;
    /// Forecast and response horizons.
    Horizon => "horizon", 5_000;
    /// Polynomial and ARMA model orders.
    ModelOrder => "model_order", 20;
    /// Integer kernel bandwidths (HAC, Driscoll-Kraay, GMM, sphet).
    Bandwidth => "bandwidth", 1_000;
    /// Smoother span and window widths.
    SpanWidth => "span_width", 1_001;
    /// Components, factors, ranks and clusters.
    Components => "components", 500;
    /// Nearest-neighbour counts.
    Neighbors => "neighbors", 100;
    /// Rows a tool generates or samples.
    GeneratedRows => "generated_rows", 1_000_000;
    /// Image width or height in pixels.
    ImageDimension => "image_dimension", 4_000;
    /// Lists of names (columns, covariates, ...) and other short lists.
    ListLength => "list_length", 500;
    /// Numeric arrays passed by the caller.
    ArrayLength => "array_length", 100_000;
    /// Byte length of caller strings that are stored, repeated or compiled.
    StringBytes => "string_bytes", 4_096;
    /// Recursion depth of tree builders.
    TreeDepth => "tree_depth", 100;
    /// The float sample size `n` (per group in two-sample and ANOVA designs) that
    /// `power_t_test` and `power_anova_test` are given. The cap is the top of the bracket in
    /// which those tools solve for `n`, so every sample size they can return is accepted.
    /// `power_prop_test`'s `n` is not capped. The key is `power_sample_size`, not
    /// `sample_size`, which other tools use as a request field name. p2a-core computes the t
    /// and F quantiles of the power calculations with bounded work at any `n`, so the cap is
    /// kept as defense in depth, not as a cost bound, and raising it with
    /// `P2A_LIMIT_OVERRIDES` does not let a call run past its deadline.
    PowerSampleSize => "power_sample_size", 1_000_000;
    /// Groups of a planned design (`power_anova_test` `groups`). As with
    /// [`Family::PowerSampleSize`], p2a-core's F quantile does bounded work at any group count,
    /// so the cap is kept as defense in depth, not as a cost bound, and can be raised with
    /// `P2A_LIMIT_OVERRIDES`.
    DesignGroups => "design_groups", 10;
}

/// Hosted [`Limits::arima_fit_work`]: the slowest fit it allows takes about 16 s on a release
/// build, a quarter of the hosted 60 s deadline, so that a slower or shared CPU still finishes
/// it in time.
///
/// Measured on a laptop CPU (Intel Core 7 150U) under heavy load from parallel builds: at the
/// optimiser's bound of 7,961 evaluations a work unit ([`arima_fit_work`]) costs at most about
/// 40 us, for an MA-only fit on the slower cores (8 to 25 us for other orders), so 400,000 units
/// take about 16 s. Whole fits at this size whose optimiser ran to that bound took up to 9.1 s;
/// fits that converge stop far sooner. It allows ARIMA(1,1,1) on 15,384 observations,
/// ARIMA(0,1,0) on 40,000 and the `model_order` caps p = d = q = 20 on 214.
///
/// The margin assumes the default 60 s deadline: with a lower `P2A_TOOL_TIMEOUT_SECS`, or on a
/// throttled shared CPU, lower `arima_fit_work` in proportion with `P2A_LIMIT_OVERRIDES`. Raised
/// above about 3.5 times this value, it lets an allowed fit run past the deadline and keep its
/// slot, since the fit has no checkpoint.
const ARIMA_FIT_WORK_HOSTED: u64 = 400_000;

/// Hosted [`Limits::pivot_columns`]: 20 times `list_length`, the most columns most tools take in
/// one call, so far above a wide table an analysis uses. It bounds what the cell budget cannot:
/// a one-row pivot has as many cells as columns, but each column costs far more than a cell.
///
/// Measured on a release build of the hosted server, a one-row pivot's peak RSS grows by about
/// 870 bytes per output column (100 MiB for 120,001 columns), and the dataset quotas charge each
/// kept column 1,536 bytes. At this cap one pivot costs about 8.5 MiB at its peak and 15 MB of
/// the 200 MiB session quota.
const PIVOT_COLUMNS_HOSTED: usize = 10_000;

/// Hosted [`Limits::pivot_cells`]. polars' pivot groups every row of the long table by its index
/// and `on` values, then fills one buffer with every output cell (an `Option` of the value for
/// numbers, an `AnyValue` otherwise) and builds the columns from it, so its peak memory grows
/// with the rows it reads as well as with the cells it makes. It reads at most as many rows as
/// it makes cells ([`Limits::check_pivot`] refuses a taller dataset), so at a given number of
/// cells the dense pivot, one row for each pair of index values and `on` value, costs the most.
///
/// Measured on a release build of the hosted server, a fresh process for each measurement, a
/// dense pivot of about 1,000,000 cells raised the peak RSS above the loaded long table by 150
/// to 200 MiB for f64 values and by 195 to 260 MiB for string values (of 6, 20 and 45 bytes),
/// as a 1,000 x 1,000, a 100 x 10,000 or a 9,900 x 101 wide table; about 2,000,000 cells raised
/// it by 310 MiB for numbers and 415 MiB for strings. A sparse pivot, with far fewer rows than
/// cells, costs about 25 bytes per cell for numbers and 65 for strings. At this value one pivot
/// peaks at most about 260 MiB above what the session already holds, well under the 700 MiB
/// soft limit of the hosted 1 GB machine, and a wide table of 100,000 rows and 10 columns, of
/// 10,000 rows and 100 columns or of 1,000 rows and 1,000 columns still fits. The allocation
/// ceiling (`P2A_MAX_ALLOC_MB`) bounds p2a-core's own buffers, not polars', so it does not
/// apply to a pivot.
const PIVOT_CELLS_HOSTED: u64 = 1_000_000;

/// Number of families; the length of [`Family::ALL`].
const FAMILY_COUNT: usize = Family::ALL.len();

// `families!` lists the variants in declaration order and gives none an explicit discriminant,
// so `family as usize` is the family's position in `Family::ALL` and indexes the caps array.
const _: () = {
    let mut i = 0;
    while i < FAMILY_COUNT {
        assert!(Family::ALL[i] as usize == i);
        i += 1;
    }
};

impl Family {
    /// Position in [`Family::ALL`] (declaration order), used to index the caps array.
    fn index(self) -> usize {
        self as usize
    }
}

/// How one request field is checked.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rule {
    /// A number must not exceed the family cap by absolute value. For
    /// [`Family::StringBytes`], a string's UTF-8 length must not exceed it instead.
    Max(Family),
    /// A number must be at least [`Limits::min_tolerance`] (catches 0, negatives, `1e-300`).
    Tolerance,
    /// An array's length must not exceed the family cap.
    ArrayLen(Family),
    /// Every numeric element of an array must not exceed the family cap by absolute value.
    ArrayElemMax(Family),
    /// A list's length must not exceed the family cap, and no string may occur twice.
    UniqueList(Family),
    /// An array of length `n` produces an `n x n` result: `n * n` must not exceed
    /// [`Limits::table_cells`].
    SquareLen,
    /// Reviewed and not a cost parameter; the string says why.
    Exempt(&'static str),
}

/// One classified request field: `tool` (or `"*"` for every tool) and the field's path.
///
/// A path is segments separated by `.`; each segment is a field name followed by zero or more
/// `[]` (each element of an array): `a`, `a[]`, `a[].b`, `a[].b.c[]`, `a[][]`. A tool-specific
/// rule for a path replaces every `"*"` rule for the same path for that tool; several rules may
/// share one `(tool, path)`, and all of them apply. Where a path goes on into a field (`a[]` in
/// `a[].b`), a request that holds a JSON array there is refused outside the off profile, since
/// serde would fill that struct by position, and the audit log records only its size
/// ([`struct_paths`]); do not give such a path to a value that may rightly be an array there
/// (the schema test in `rules.rs` names the tool and path of one that may).
#[derive(Debug, Clone, Copy)]
pub struct ParamRule {
    /// Tool name, or `"*"` for every tool that has the field and no tool-specific rule for it.
    pub tool: &'static str,
    /// Field path (see the type documentation).
    pub path: &'static str,
    /// The check applied to every value the path resolves to.
    pub rule: Rule,
}

/// How a method's cost grows with the number of dataset rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowClass {
    /// Memory grows with the square of the row count (n x n matrices).
    Dense,
    /// Run time grows with the cube of the row count (or several n x n matrices per step).
    Cubic,
    /// Run time grows with the square of the row count.
    Quadratic,
}

impl RowClass {
    /// The override key of the class's row cap.
    pub fn key(self) -> &'static str {
        match self {
            RowClass::Dense => "dense_rows",
            RowClass::Cubic => "cubic_rows",
            RowClass::Quadratic => "quadratic_rows",
        }
    }

    /// Why the class is row-limited, as shown in refusals.
    fn reason(self) -> &'static str {
        match self {
            RowClass::Dense => "its memory grows with the square of the row count",
            RowClass::Cubic => "its run time grows with the cube of the row count",
            RowClass::Quadratic => "its run time grows with the square of the row count",
        }
    }
}

/// A tool whose dataset height is limited by a [`RowClass`].
#[derive(Debug, Clone, Copy)]
pub struct RowRule {
    /// Tool name.
    pub tool: &'static str,
    /// The request field that names the dataset.
    pub dataset_field: &'static str,
    /// The tool's cost class.
    pub class: RowClass,
}

/// A tool that reshapes a dataset from long to wide ([`PIVOT_RULES`]). Its output has one row per
/// distinct combination of the index columns, the index columns, and one column per distinct
/// value of the `on` column, filled from the values column. [`Limits::check_pivot`] checks that
/// shape against [`Limits::pivot_columns`] and [`Limits::pivot_cells`] before the tool runs.
#[derive(Debug, Clone, Copy)]
pub struct PivotRule {
    /// Tool name.
    pub tool: &'static str,
    /// The request field that names the dataset.
    pub dataset_field: &'static str,
    /// The request field that lists the index columns.
    pub index_field: &'static str,
    /// The request field that names the column whose values become column names.
    pub on_field: &'static str,
    /// The request field that names the values column.
    pub values_field: &'static str,
}

/// The active request limits: family caps, row caps, the minimum tolerance and named budgets.
///
/// `None` means unlimited. Counts are `usize`; products checked with `checked_mul` and every
/// `*_bytes`, `*_secs` and `*_mb` field are `u64`. The budgets that have a value in both profiles
/// are not `Option`; their consumers read them directly.
#[derive(Debug, Clone, PartialEq)]
pub struct Limits {
    /// Family caps, indexed by position in [`Family::ALL`].
    caps: [Option<u64>; FAMILY_COUNT],
    /// Rows for [`RowClass::Dense`] tools.
    pub dense_rows: Option<usize>,
    /// Rows for [`RowClass::Cubic`] tools.
    pub cubic_rows: Option<usize>,
    /// Rows for [`RowClass::Quadratic`] tools.
    pub quadratic_rows: Option<usize>,
    /// Smallest accepted convergence tolerance ([`Rule::Tolerance`]).
    pub min_tolerance: Option<f64>,
    /// Pixels of a rendered image (width x height).
    pub image_pixels: Option<u64>,
    /// Cells of a computed table or square matrix.
    pub table_cells: Option<u64>,
    /// Groups compared pairwise.
    pub max_groups: Option<usize>,
    /// Predicted output rows of a join.
    pub join_output_rows: Option<u64>,
    /// Levels expanded by one-hot encoding.
    pub onehot_levels: Option<usize>,
    /// Output columns of a pivot: the index columns plus one per distinct value of the `on`
    /// column; checked by [`Limits::check_pivot`] before the pivot.
    pub pivot_columns: Option<usize>,
    /// Output cells of a pivot: its rows (one per distinct combination of the index columns)
    /// times its columns, and at least the rows of the dataset it reads; checked by
    /// [`Limits::check_pivot`] before the pivot.
    pub pivot_cells: Option<u64>,
    /// Parts produced by `str_split`.
    pub str_split_parts: Option<usize>,
    /// Target width of `str_pad`.
    pub str_pad_width: Option<usize>,
    /// Cells of generated data.
    pub generated_cells: Option<u64>,
    /// Estimated bytes of generated data.
    pub generated_bytes: Option<u64>,
    /// Estimated bytes of an export.
    pub export_bytes: Option<u64>,
    /// Size on disk of a file `load_dataset` or a dataset reload reads, checked before parsing.
    pub load_file_bytes: Option<u64>,
    /// Bytes of a tool's text output.
    pub response_bytes: Option<u64>,
    /// Bytes of a session's data directory.
    pub session_disk_bytes: Option<u64>,
    /// Total links of a spatial neighbour structure.
    pub spatial_links: Option<u64>,
    /// Work of an ARIMA fit, as [`arima_fit_work`] counts it (observations x
    /// ((p + q + 3)^2 + d)); checked by the ARIMA handlers with [`Limits::check_arima_fit`]
    /// before the fit starts. While it is set, [`Limits::check_arima_series`] also refuses a
    /// series below [`ARIMA_MIN_MAGNITUDE`].
    pub arima_fit_work: Option<u64>,
    /// Rows a SQL tool returns.
    pub db_result_rows: Option<usize>,
    /// Bytes a SQL tool returns.
    pub db_result_bytes: Option<u64>,
    /// Wall-clock limit of a SQL statement.
    pub db_timeout_secs: Option<u64>,
    /// Memory limit of the SQL engine.
    pub db_memory_mb: Option<u64>,
    /// Threads of the SQL engine (`None`: engine default).
    pub db_threads: Option<u32>,
    /// Whether the free-form SQL tools (`db_duckdb_query`, `db_query_file`) may run at all.
    pub sql_freeform: bool,
    /// Bytes of one conversation message.
    pub message_bytes: u64,
    /// Conversations per session.
    pub conversations_per_session: Option<usize>,
    /// Messages per conversation.
    pub messages_per_conversation: Option<usize>,
    /// Conversation-store bytes per session (titles, messages, tool calls and results).
    pub session_db_bytes: Option<u64>,
    /// LLM HTTP timeout (total for non-streaming, read-idle for streaming).
    pub llm_timeout_secs: u64,
    /// Upper clamp of a request's `max_tool_iterations`.
    pub llm_max_tool_iterations: usize,
    /// Bytes of dataset context in the LLM prompt.
    pub llm_context_bytes: u64,
    /// Bytes of conversation history in the LLM prompt.
    pub llm_history_bytes: u64,
    /// Concurrent chat streams per session.
    pub llm_streams_per_session: Option<usize>,
    /// Whether API keys from the server's environment may be used.
    pub llm_server_keys: bool,
    /// Whether loopback LLM endpoints (the Ollama default) may be used.
    pub llm_allow_loopback: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Limits::off()
    }
}

/// Generates the override keys of the row caps, the minimum tolerance and the named budgets (the
/// field names), the setter for them and the "all unlimited" test, from one list, so that the key
/// list, the setter and [`Limits::is_off`] cannot drift apart.
macro_rules! named_limits {
    (
        optional { $($opt:ident),* $(,)? }
        required { $($req:ident),* $(,)? }
        flags { $($flag:ident),* $(,)? }
    ) => {
        /// Keys of the row caps, the minimum tolerance and the named budgets: the `Option` limits,
        /// then the required budgets, then the flags, each group in the order listed below.
        const NAMED_KEYS: &[&str] = &[
            $(stringify!($opt),)*
            $(stringify!($req),)*
            $(stringify!($flag),)*
        ];

        impl Limits {
            /// Sets a named limit; returns `Ok(false)` when `key` is not one.
            fn set_named(&mut self, key: &str, value: &str) -> Result<bool, String> {
                match key {
                    $(stringify!($opt) => self.$opt = parse_optional(key, value)?,)*
                    $(stringify!($req) => self.$req = parse_required(key, value)?,)*
                    $(stringify!($flag) => self.$flag = parse_flag(key, value)?,)*
                    _ => return Ok(false),
                }
                Ok(true)
            }

            /// True when every `Option` named limit is `None`.
            fn named_unlimited(&self) -> bool {
                true $(&& self.$opt.is_none())*
            }
        }
    };
}

named_limits! {
    optional {
        dense_rows, cubic_rows, quadratic_rows, min_tolerance, image_pixels, table_cells,
        max_groups, join_output_rows, onehot_levels, pivot_columns, pivot_cells,
        str_split_parts, str_pad_width,
        generated_cells, generated_bytes, export_bytes, load_file_bytes, response_bytes,
        session_disk_bytes, spatial_links, arima_fit_work, db_result_rows, db_result_bytes,
        db_timeout_secs, db_memory_mb, db_threads, conversations_per_session,
        messages_per_conversation, session_db_bytes, llm_streams_per_session,
    }
    required {
        message_bytes, llm_timeout_secs, llm_max_tool_iterations, llm_context_bytes,
        llm_history_bytes,
    }
    flags { sql_freeform, llm_server_keys, llm_allow_loopback }
}

impl Limits {
    /// The profile that checks nothing (local use; the default).
    pub fn off() -> Self {
        Limits {
            caps: [None; FAMILY_COUNT],
            dense_rows: None,
            cubic_rows: None,
            quadratic_rows: None,
            min_tolerance: None,
            image_pixels: None,
            table_cells: None,
            max_groups: None,
            join_output_rows: None,
            onehot_levels: None,
            pivot_columns: None,
            pivot_cells: None,
            str_split_parts: None,
            str_pad_width: None,
            generated_cells: None,
            generated_bytes: None,
            export_bytes: None,
            load_file_bytes: None,
            response_bytes: None,
            session_disk_bytes: None,
            spatial_links: None,
            arima_fit_work: None,
            db_result_rows: None,
            db_result_bytes: None,
            db_timeout_secs: None,
            db_memory_mb: None,
            db_threads: None,
            sql_freeform: true,
            message_bytes: 2_097_152,
            conversations_per_session: None,
            messages_per_conversation: None,
            session_db_bytes: None,
            llm_timeout_secs: 900,
            llm_max_tool_iterations: 50,
            llm_context_bytes: 65_536,
            llm_history_bytes: 524_288,
            llm_streams_per_session: None,
            llm_server_keys: true,
            llm_allow_loopback: true,
        }
    }

    /// The profile of the public no-login deployment.
    pub fn hosted() -> Self {
        const MIB: u64 = 1024 * 1024;
        let mut caps = [None; FAMILY_COUNT];
        for &f in Family::ALL {
            caps[f.index()] = Some(f.hosted_cap());
        }
        Limits {
            caps,
            dense_rows: Some(5_000),
            cubic_rows: Some(2_000),
            quadratic_rows: Some(50_000),
            min_tolerance: Some(1e-15),
            image_pixels: Some(16_000_000),
            table_cells: Some(1_000_000),
            max_groups: Some(100),
            join_output_rows: Some(2_000_000),
            onehot_levels: Some(500),
            pivot_columns: Some(PIVOT_COLUMNS_HOSTED),
            pivot_cells: Some(PIVOT_CELLS_HOSTED),
            str_split_parts: Some(100),
            str_pad_width: Some(10_000),
            generated_cells: Some(10_000_000),
            generated_bytes: Some(200 * MIB),
            export_bytes: Some(64 * MIB),
            load_file_bytes: Some(64 * MIB),
            response_bytes: Some(16 * MIB),
            session_disk_bytes: Some(64 * MIB),
            spatial_links: Some(2_000_000),
            arima_fit_work: Some(ARIMA_FIT_WORK_HOSTED),
            db_result_rows: Some(100_000),
            db_result_bytes: Some(64 * MIB),
            db_timeout_secs: Some(20),
            db_memory_mb: Some(256),
            db_threads: Some(1),
            sql_freeform: false,
            message_bytes: 262_144,
            conversations_per_session: Some(50),
            messages_per_conversation: Some(1_000),
            session_db_bytes: Some(16 * MIB),
            llm_timeout_secs: 120,
            llm_max_tool_iterations: 25,
            llm_context_bytes: 24_576,
            llm_history_bytes: 131_072,
            llm_streams_per_session: Some(1),
            llm_server_keys: false,
            llm_allow_loopback: false,
        }
    }

    /// Applies `key=value` overrides (`P2A_LIMIT_OVERRIDES`).
    ///
    /// Each entry may hold several comma-separated pairs; pieces are trimmed, empty pieces are
    /// skipped, and the last occurrence of a key wins. `Option` limits take `none` or `unlimited`
    /// (any case) for no limit; booleans take `true` or `false` (any case); `min_tolerance` takes a
    /// finite float above zero. An unknown key or an invalid value is an error.
    pub fn with_overrides(mut self, overrides: &[String]) -> Result<Self, String> {
        for entry in overrides {
            for piece in entry.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                let Some((key, value)) = piece.split_once('=') else {
                    return Err(format!(
                        "limit override `{piece}` is not of the form key=value"
                    ));
                };
                self.set(key.trim(), value.trim())?;
            }
        }
        Ok(self)
    }

    /// Every key [`Limits::with_overrides`] accepts: the family keys, then the row caps,
    /// `min_tolerance` and the named budgets (their field names).
    pub fn known_keys() -> Vec<&'static str> {
        Family::ALL
            .iter()
            .map(|f| f.key())
            .chain(NAMED_KEYS.iter().copied())
            .collect()
    }

    /// True when nothing is checked: no family cap, no row cap, no minimum tolerance and no
    /// `Option` budget. The non-`Option` budgets and the booleans are not considered.
    pub fn is_off(&self) -> bool {
        self.caps.iter().all(Option::is_none) && self.named_unlimited()
    }

    /// The cap of a family (`None`: unlimited).
    pub fn cap(&self, family: Family) -> Option<u64> {
        self.caps[family.index()]
    }

    /// The row cap of a row class (`None`: unlimited).
    pub fn row_cap(&self, class: RowClass) -> Option<usize> {
        match class {
            RowClass::Dense => self.dense_rows,
            RowClass::Cubic => self.cubic_rows,
            RowClass::Quadratic => self.quadratic_rows,
        }
    }

    /// Sets one limit by key.
    fn set(&mut self, key: &str, value: &str) -> Result<(), String> {
        if let Some(&family) = Family::ALL.iter().find(|f| f.key() == key) {
            self.caps[family.index()] = parse_optional(key, value)?;
            return Ok(());
        }
        if self.set_named(key, value)? {
            return Ok(());
        }
        Err(format!(
            "unknown limit `{key}`; known limits: {}",
            Limits::known_keys().join(", ")
        ))
    }
}

/// A value type of a limit, parsed from an override.
trait LimitValue: Sized {
    fn parse_limit(key: &str, value: &str) -> Result<Self, String>;
}

fn invalid_value(key: &str, value: &str, expected: &str) -> String {
    format!("invalid value `{value}` for limit `{key}`: expected {expected}")
}

fn parse_integer<T: TryFrom<u64>>(key: &str, value: &str, max: u64) -> Result<T, String> {
    let expected = || format!("a non-negative integer no larger than {max}");
    let n: u64 = value
        .parse()
        .map_err(|_| invalid_value(key, value, &expected()))?;
    T::try_from(n).map_err(|_| invalid_value(key, value, &expected()))
}

impl LimitValue for u64 {
    fn parse_limit(key: &str, value: &str) -> Result<Self, String> {
        parse_integer(key, value, u64::MAX)
    }
}

impl LimitValue for usize {
    fn parse_limit(key: &str, value: &str) -> Result<Self, String> {
        parse_integer(key, value, usize::MAX as u64)
    }
}

impl LimitValue for u32 {
    fn parse_limit(key: &str, value: &str) -> Result<Self, String> {
        parse_integer(key, value, u64::from(u32::MAX))
    }
}

impl LimitValue for f64 {
    fn parse_limit(key: &str, value: &str) -> Result<Self, String> {
        match value.parse::<f64>() {
            Ok(x) if x.is_finite() && x > 0.0 => Ok(x),
            _ => Err(invalid_value(key, value, "a finite number above 0")),
        }
    }
}

fn is_unlimited(value: &str) -> bool {
    value.eq_ignore_ascii_case("none") || value.eq_ignore_ascii_case("unlimited")
}

fn parse_optional<T: LimitValue>(key: &str, value: &str) -> Result<Option<T>, String> {
    if is_unlimited(value) {
        Ok(None)
    } else {
        T::parse_limit(key, value).map(Some)
    }
}

fn parse_required<T: LimitValue>(key: &str, value: &str) -> Result<T, String> {
    if is_unlimited(value) {
        Err(format!("limit `{key}` cannot be unlimited"))
    } else {
        T::parse_limit(key, value)
    }
}

fn parse_flag(key: &str, value: &str) -> Result<bool, String> {
    if value.eq_ignore_ascii_case("true") {
        Ok(true)
    } else if value.eq_ignore_ascii_case("false") {
        Ok(false)
    } else {
        Err(invalid_value(key, value, "true or false"))
    }
}

/// Tool-specific rules for `tool` (table order), then the `"*"` rules whose path has no
/// tool-specific rule for `tool` (table order).
pub fn rules_for(tool: &str) -> Vec<&'static ParamRule> {
    select_rules(tool, PARAM_RULES)
}

/// The nested request structs that the rules for `tool` read fields of, as rule paths: the part
/// of each non-`Exempt` rule path before a `.` (`columns[]` and `columns[].distribution` for
/// `columns[].distribution.value`), each once, in [`rules_for`] order. Outside the off profile
/// [`Limits::check_tool_arguments`] refuses a JSON array at one of them (serde would fill the
/// struct by field position), and the audit log replaces one by its size in every profile
/// ([`crate::audit::sanitize_arguments`]), as its values carry no names to be redacted by.
pub fn struct_paths(tool: &str) -> Vec<&'static str> {
    let mut out = Vec::new();
    for rule in rules_for(tool) {
        if matches!(rule.rule, Rule::Exempt(_)) {
            continue;
        }
        for (dot, _) in rule.path.match_indices('.') {
            let prefix = &rule.path[..dot];
            if !out.contains(&prefix) {
                out.push(prefix);
            }
        }
    }
    out
}

/// The row rule of `tool`, if its dataset height is limited.
pub fn row_rule(tool: &str) -> Option<&'static RowRule> {
    ROW_RULES.iter().find(|r| r.tool == tool)
}

/// The pivot rule of `tool`, if it reshapes a dataset from long to wide. The request path
/// filters [`PIVOT_RULES`] itself; the tests look rules up here.
#[cfg(test)]
pub fn pivot_rule(tool: &str) -> Option<&'static PivotRule> {
    PIVOT_RULES.iter().find(|r| r.tool == tool)
}

fn select_rules<'a>(tool: &str, rules: &'a [ParamRule]) -> Vec<&'a ParamRule> {
    let mut selected: Vec<&'a ParamRule> = rules.iter().filter(|r| r.tool == tool).collect();
    let specific = selected.len();
    for rule in rules.iter().filter(|r| r.tool == "*") {
        if !selected[..specific].iter().any(|s| s.path == rule.path) {
            selected.push(rule);
        }
    }
    selected
}

/// A request that exceeds a limit. `message` is written for the caller (an LLM or a person)
/// and is the error's `Display`; `override_key` names the knob for server logs.
#[derive(Debug, Clone, thiserror::Error)]
#[error("{message}")]
pub struct LimitViolation {
    /// The tool that was called.
    pub tool: String,
    /// The rule path, `dataset_field` (row limits) or budget key that was exceeded
    /// (`arguments` for arguments that are not a JSON object, the struct's path, such as
    /// `sections[]`, for a nested struct given as a JSON array, `column` for an ARIMA series
    /// below [`ARIMA_MIN_MAGNITUDE`], and for the pivot budgets the request field that decides
    /// the refused size: the `on` field, the index field or the dataset field of the
    /// [`PivotRule`]).
    pub parameter: String,
    /// What the request asked for.
    pub requested: String,
    /// The limit that applies.
    pub limit: String,
    /// The `P2A_LIMIT_OVERRIDES` key that raises the limit; empty when no override lifts the
    /// refusal (arguments, or a nested struct, that are not a JSON object).
    pub override_key: &'static str,
    /// The refusal shown to the caller.
    pub message: String,
}

/// Longest echo of a caller string in a refusal, in characters.
const ECHO_CHARS: usize = 64;

/// Cuts a caller string to [`ECHO_CHARS`] characters (at a char boundary, never mid-character).
/// The result borrows from `s`, so echoing allocates nothing.
///
/// Handlers use it for every caller-supplied name they echo in a refusal.
pub(crate) fn echo(s: &str) -> &str {
    match s.char_indices().nth(ECHO_CHARS) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}

/// Collects every non-null value `path` resolves to inside `value` (path grammar: see
/// [`ParamRule`]). A missing key, `null` or a type that does not fit a segment (`[]` on a
/// non-array, a field on a scalar) yields nothing, because serde refuses that shape itself.
///
/// A field on an array is an error: serde fills a struct from a JSON array by field position,
/// so the values there would reach the handler unchecked. The error is the part of `path` that
/// the array was to resolve (a suffix of `path` after a `.`, or `path` itself when `value` is
/// an array).
fn resolve_path<'v, 'p>(
    value: &'v Value,
    path: &'p str,
    out: &mut Vec<&'v Value>,
) -> Result<(), &'p str> {
    if value.is_array() {
        return Err(path);
    }
    let (segment, rest) = match path.split_once('.') {
        Some((segment, rest)) => (segment, Some(rest)),
        None => (path, None),
    };
    let mut name = segment;
    let mut depth = 0;
    while let Some(stripped) = name.strip_suffix("[]") {
        name = stripped;
        depth += 1;
    }
    if let Some(field) = value.as_object().and_then(|o| o.get(name)) {
        expand(field, depth, rest, out)?;
    }
    Ok(())
}

/// Descends `depth` array levels from `value`, then resolves `rest` (or collects the value).
/// The error is [`resolve_path`]'s.
fn expand<'v, 'p>(
    value: &'v Value,
    depth: usize,
    rest: Option<&'p str>,
    out: &mut Vec<&'v Value>,
) -> Result<(), &'p str> {
    if depth > 0 {
        for item in value.as_array().into_iter().flatten() {
            expand(item, depth - 1, rest, out)?;
        }
    } else if let Some(rest) = rest {
        resolve_path(value, rest, out)?;
    } else if !value.is_null() {
        out.push(value);
    }
    Ok(())
}

/// The refusal of a nested struct at `parameter` (a rule path's prefix, such as `sections[]`)
/// given as a JSON array; `tool_echo` is the tool name as [`echo`] cut it.
fn positional_struct_violation(tool_echo: &str, parameter: &str) -> LimitViolation {
    LimitViolation {
        tool: tool_echo.to_string(),
        parameter: parameter.to_string(),
        requested: "an array".to_string(),
        limit: "a JSON object".to_string(),
        override_key: "",
        message: format!(
            "Parameter `{parameter}` of `{tool_echo}` is an array, but it must be a JSON object \
             that names each field. Send each value under its field name, as the tool's input \
             schema lists them."
        ),
    }
}

/// True when a JSON number exceeds `cap` by absolute value.
fn number_exceeds(n: &serde_json::Number, cap: u64) -> bool {
    magnitude_exceeds(n.as_u64(), n.as_i64(), n.as_f64(), cap)
}

/// [`number_exceeds`] on the number's `u64`, `i64` and `f64` readings. A number with none of
/// them exceeds every cap, so the check fails closed.
fn magnitude_exceeds(u: Option<u64>, i: Option<i64>, f: Option<f64>, cap: u64) -> bool {
    match (u, i, f) {
        (Some(u), _, _) => u > cap,
        (None, Some(i), _) => i.unsigned_abs() > cap,
        (None, None, Some(f)) => f.abs() > cap as f64,
        // Reachable only with serde_json's `arbitrary_precision` feature (for example 1e400).
        (None, None, None) => true,
    }
}

/// True when a tolerance, given as the number's `f64` reading, is refused against `min`: below
/// it, NaN, or without an `f64` reading, so that the check fails closed.
fn below_min_tolerance(f: Option<f64>, min: f64) -> bool {
    match f {
        Some(x) if x >= min => false,
        // `None` is reachable only with serde_json's `arbitrary_precision` feature (for example
        // 1e400 or -1e400, which have no finite reading); `tolerance_refusal` words it so. A
        // NaN reading, which JSON cannot produce, is refused as well.
        _ => true,
    }
}

/// The refusal of the number `n` at `path`, which exceeds `cap` by magnitude: the wording of
/// [`Rule::Max`] and [`Rule::ArrayElemMax`]. `subject` sits between the parameter and the verb:
/// `"= {n}"` for a scalar, `"contains {n}, which"` for an element of an array. `plural` words
/// the advice for the array's values.
fn magnitude_refusal(
    path: &str,
    subject: &str,
    n: &serde_json::Number,
    cap: u64,
    key: &str,
    tool_echo: &str,
    plural: bool,
) -> String {
    // The cap bounds the magnitude, so "smaller" would point a negative value the wrong way.
    if n.to_string().starts_with('-') {
        let values = if plural { "values" } else { "a value" };
        format!(
            "Parameter `{path}` {subject} is outside this server's limit of ±{cap} ({key}) for \
             `{tool_echo}`. Use {values} of smaller magnitude."
        )
    } else {
        let values = if plural {
            "smaller values"
        } else {
            "a smaller value"
        };
        format!(
            "Parameter `{path}` {subject} exceeds this server's limit of {cap} ({key}) for \
             `{tool_echo}`. Use {values}."
        )
    }
}

/// The refusal of the tolerance `shown` (the number as the request wrote it) at `path`.
/// `readable` is false for a number with no `f64` reading, which may lie above the minimum as
/// well as below it, so the message says it cannot be read instead.
fn tolerance_refusal(path: &str, shown: &str, readable: bool, min: f64, tool_echo: &str) -> String {
    if readable {
        format!(
            "Parameter `{path}` = {shown} is below this server's minimum of {min:e} \
             (min_tolerance) for `{tool_echo}`. Use a larger tolerance such as 1e-8."
        )
    } else {
        format!(
            "Parameter `{path}` = {shown} cannot be read as a finite number by this server \
             (min_tolerance) for `{tool_echo}`. Use a tolerance such as 1e-8."
        )
    }
}

/// What a tool call's arguments are when they are not the JSON object every tool takes:
/// `Some("an array")`, `Some("a string")`, `Some("a number")` or `Some("a boolean")`; `None` for
/// an object, and for `null`, which is arguments left out (the empty object).
///
/// A derived request struct deserialises a JSON array too, filling its fields in declaration
/// order, so top-level positional arguments would reach a handler past every check that reads
/// parameters by name: this catalogue, the row classes and the audit redaction's
/// [`REDACT_KEYS`](crate::audit::REDACT_KEYS). The governed call path
/// (`AnalyticsServer::call_tool_with_session`) refuses them in every profile with
/// [`non_object_message`]; stdio receives arguments as an object map only. This looks at the
/// top level only: [`Limits::check_tool_arguments`] refuses a nested struct given by position
/// where a rule goes into it.
pub(crate) fn non_object_kind(args: &Value) -> Option<&'static str> {
    match args {
        Value::Object(_) | Value::Null => None,
        Value::Array(_) => Some("an array"),
        Value::String(_) => Some("a string"),
        Value::Number(_) => Some("a number"),
        Value::Bool(_) => Some("a boolean"),
    }
}

/// The refusal of `tool`'s arguments when they are `kind` (from [`non_object_kind`]).
pub(crate) fn non_object_message(tool: &str, kind: &str) -> String {
    format!(
        "The arguments of `{}` are {kind}, but tool arguments must be a JSON object that names \
         each parameter. Send each value under its parameter name, as the tool's input schema \
         lists them.",
        echo(tool)
    )
}

/// What a pivot request asks for, read from its JSON arguments by a [`PivotRule`]'s field names.
struct PivotRequest<'a> {
    /// The index columns (at least one).
    index: Vec<&'a str>,
    /// The column whose values become column names.
    on: &'a str,
    /// The column whose values fill the new columns.
    values: &'a str,
}

impl<'a> PivotRequest<'a> {
    /// `None` when `args` do not have the request's shape, or name no index column; serde or
    /// the core refuses those.
    fn read(rule: &PivotRule, args: &'a Value) -> Option<Self> {
        let index = args
            .get(rule.index_field)?
            .as_array()?
            .iter()
            .map(Value::as_str)
            .collect::<Option<Vec<_>>>()?;
        if index.is_empty() {
            return None;
        }
        Some(PivotRequest {
            index,
            on: args.get(rule.on_field)?.as_str()?,
            values: args.get(rule.values_field)?.as_str()?,
        })
    }
}

/// The refusal of a pivot whose `column` (the request's `parameter`, `on` or `index`) had
/// distinct values that could not be counted (polars' error): the check fails closed. `budget`
/// is the budget the count was for, its key and value; `error` is logged, not shown.
fn uncountable(
    tool: &str,
    parameter: &str,
    column: &str,
    (knob, limit): (&'static str, u64),
    error: &PolarsError,
) -> LimitViolation {
    tracing::info!(tool, parameter, %error, "pivot shape could not be counted");
    LimitViolation {
        message: format!(
            "`{tool}` cannot count the distinct values of `{parameter}` ({column}), which this \
             server needs to check its limit of {limit} ({knob}) before pivoting. Cast the column \
             to strings first, for example with munge_cast."
        ),
        tool: tool.to_string(),
        parameter: parameter.to_string(),
        requested: "an uncountable column".to_string(),
        limit: limit.to_string(),
        override_key: knob,
    }
}

impl Limits {
    /// Checks a tool call's JSON arguments against [`PARAM_RULES`]; returns the first violation
    /// in [`rules_for`] order. The off profile passes everything. Otherwise arguments that are
    /// not an object ([`non_object_kind`]; `null` passes as the empty object) are refused with
    /// `parameter` `arguments`, because no rule could resolve against them, and so is a nested
    /// struct that a rule's path goes into when it is a JSON array (`parameter` its path, such
    /// as `sections[]`): serde would fill it by field position, past the rules on its fields.
    pub fn check_tool_arguments(&self, tool: &str, args: &Value) -> Result<(), LimitViolation> {
        self.check_with(tool, args, PARAM_RULES)
    }

    /// Checks a dataset's height for a tool in [`ROW_RULES`]; other tools pass.
    pub fn check_rows(
        &self,
        tool: &str,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), LimitViolation> {
        let Some(rule) = row_rule(tool) else {
            return Ok(());
        };
        let Some(cap) = self.row_cap(rule.class) else {
            return Ok(());
        };
        if n_rows <= cap {
            return Ok(());
        }
        let tool = echo(tool);
        Err(LimitViolation {
            message: format!(
                "The dataset for `{tool}` has {n_rows} rows and {n_cols} columns, but this server \
                 allows at most {cap} rows for this method because {}. Use a sample of at most \
                 {cap} rows, for example with munge_sample.",
                rule.class.reason()
            ),
            tool: tool.to_string(),
            parameter: rule.dataset_field.to_string(),
            requested: n_rows.to_string(),
            limit: cap.to_string(),
            override_key: rule.class.key(),
        })
    }

    /// Checks a quantity a handler computed (for example predicted join rows) against a named
    /// budget; `knob` is the budget's key and `limit` its value (`None` passes).
    pub fn check_budget(
        &self,
        tool: &str,
        knob: &'static str,
        requested: u64,
        limit: Option<u64>,
    ) -> Result<(), LimitViolation> {
        match limit {
            Some(limit) if requested > limit => {
                let tool = echo(tool);
                Err(LimitViolation {
                    message: format!(
                        "This request to `{tool}` needs {requested} ({knob}), which exceeds this \
                         server's limit of {limit}. Use a smaller request."
                    ),
                    tool: tool.to_string(),
                    parameter: knob.to_string(),
                    requested: requested.to_string(),
                    limit: limit.to_string(),
                    override_key: knob,
                })
            }
            _ => Ok(()),
        }
    }

    /// Checks an ARIMA fit of `n_obs` observations at orders `p`, `d` and `q`
    /// ([`arima_fit_work`]) against the [`Limits::arima_fit_work`] budget (`None` passes). The
    /// ARIMA handlers call it before they fit, because the fit cannot be stopped once it starts.
    /// The refusal says how many observations the orders allow (when that is at least the
    /// `p + d + q + 10` the fit needs) and how small `p + q` must be for the observations.
    pub fn check_arima_fit(
        &self,
        tool: &str,
        n_obs: usize,
        p: usize,
        d: usize,
        q: usize,
    ) -> Result<(), LimitViolation> {
        const KNOB: &str = "arima_fit_work";
        let Some(limit) = self.arima_fit_work else {
            return Ok(());
        };
        let work = arima_fit_work(n_obs, p, d, q);
        if work <= limit {
            return Ok(());
        }
        // `work > limit` here, so `n_obs >= 1`, and every order costs at least 3^2 per
        // observation.
        let max_obs = limit / arima_fit_work(1, p, d, q);
        let min_obs = p.saturating_add(d).saturating_add(q).saturating_add(10);
        let fewer_fit = usize::try_from(max_obs).is_ok_and(|m| m >= min_obs);
        let per_obs = limit / u64::try_from(n_obs).unwrap_or(u64::MAX);
        let room = u64::try_from(d)
            .ok()
            .and_then(|d| per_obs.checked_sub(d))
            .and_then(|r| r.isqrt().checked_sub(3));
        let fewer = format!(
            "at most {max_obs} observations at these orders, for example the most recent ones"
        );
        let advice = match (fewer_fit, room) {
            (true, Some(0)) => format!("Use {fewer}, or set p and q to 0."),
            (true, Some(s)) => {
                format!("Use {fewer}, or lower the orders so that p + q is at most {s}.")
            }
            (true, None) => format!("Use {fewer}."),
            (false, Some(0)) => "Set p and q to 0.".to_string(),
            (false, Some(s)) => format!("Lower the orders so that p + q is at most {s}."),
            (false, None) => "Use fewer observations and lower orders.".to_string(),
        };
        let tool = echo(tool);
        Err(LimitViolation {
            message: format!(
                "`{tool}` with p = {p}, d = {d} and q = {q} on {n_obs} observations needs {work} \
                 ({KNOB}: observations x ((p + q + 3)^2 + d)), which exceeds this server's limit \
                 of {limit}: the ARIMA fit cannot be stopped once it starts. {advice}"
            ),
            tool: tool.to_string(),
            parameter: KNOB.to_string(),
            requested: work.to_string(),
            limit: limit.to_string(),
            override_key: KNOB,
        })
    }

    /// Checks the column an ARIMA fit reads for a nonzero value smaller than
    /// [`ARIMA_MIN_MAGNITUDE`] while the [`Limits::arima_fit_work`] budget is set (`None`
    /// passes): the fit's products and squares of such values can be subnormal, and arithmetic
    /// on those is far slower than the budget's time per unit allows for. The floor keeps the
    /// column, its differences, and their products and squares with ordinary-sized coefficients
    /// out of that range. It does not cover the MA residual recursion at an intercept of
    /// exactly 0.0 over a stretch of exact zeros, which can sit in the subnormal range (one pass
    /// measured about 28 times slower); [`ARIMA_MIN_MAGNITUDE`] says why that is not believed to
    /// last a whole fit, and records it as a residual risk for a change of the optimiser. The
    /// ARIMA handlers call it after [`Limits::check_arima_fit`]. `column` names the series in
    /// the refusal.
    pub fn check_arima_series(
        &self,
        tool: &str,
        column: &str,
        values: impl IntoIterator<Item = f64>,
    ) -> Result<(), LimitViolation> {
        const KNOB: &str = "arima_fit_work";
        if self.arima_fit_work.is_none() {
            return Ok(());
        }
        let Some(found) = values
            .into_iter()
            .map(f64::abs)
            .find(|&m| m != 0.0 && m < ARIMA_MIN_MAGNITUDE)
        else {
            return Ok(());
        };
        let (tool, column) = (echo(tool), echo(column));
        Err(LimitViolation {
            message: format!(
                "`{tool}` cannot fit column `{column}`: it holds a nonzero value of magnitude \
                 {found:e}, below this server's floor of {ARIMA_MIN_MAGNITUDE:e} for an ARIMA fit \
                 ({KNOB}). Values that small can make the fit's arithmetic many times slower, \
                 and the fit cannot be stopped once it starts. Set values that small to 0, or \
                 rescale the column, for example multiply it by a power of ten."
            ),
            tool: tool.to_string(),
            parameter: "column".to_string(),
            requested: format!("{found:e}"),
            limit: format!("{ARIMA_MIN_MAGNITUDE:e}"),
            override_key: KNOB,
        })
    }

    /// Checks the wide table a pivot ([`PIVOT_RULES`]) would make of `df` for the request `args`
    /// against [`Limits::pivot_columns`] and [`Limits::pivot_cells`] (`None` passes), before
    /// anything is pivoted: polars groups every row of `df` by its index and `on` values and fills
    /// a buffer of rows x columns cells before the dataset quotas look at the result, so a long
    /// table well within the quotas could otherwise make one call reach its peak memory
    /// unchecked.
    ///
    /// The columns are the index columns plus one per distinct value of the `on` column (a null
    /// is one value, as in the pivot); the rows are the distinct combinations of the index
    /// columns. Each row of `df` fills one cell, and a second row for the same cell makes the
    /// pivot fail, so a `df` with more rows than `pivot_cells` is refused before anything is
    /// counted. The distinct values are then counted in hash maps that stop at the first count
    /// no budget can pass (`distinct::count`): counting takes memory in proportion to the
    /// budgets, not to the rows of `df`, and a refusal returns as soon as it is certain, giving
    /// that count as a lower bound (`at least`).
    ///
    /// Arguments of another shape, an empty index and a column `df` lacks (an index column, `on`
    /// or the values column) pass, so that serde or the core refuses them as usual. A column
    /// whose distinct values cannot be counted is refused, so that the check fails closed. The
    /// refusal names `on` for the column budget, and for the cell budget the dataset field when
    /// `df` alone has too many rows, `index` otherwise.
    pub fn check_pivot(
        &self,
        rule: &PivotRule,
        args: &Value,
        df: &DataFrame,
    ) -> Result<(), LimitViolation> {
        // The budget the count of `on` values serves: the column budget, or else the cell
        // budget, which needs the columns too.
        let on_budget = match (self.pivot_columns, self.pivot_cells) {
            (None, None) => return Ok(()),
            (Some(cap), _) => ("pivot_columns", cap as u64),
            (None, Some(cap)) => ("pivot_cells", cap),
        };
        let Some(request) = PivotRequest::read(rule, args) else {
            return Ok(());
        };
        let named = [request.on, request.values];
        if named
            .iter()
            .chain(&request.index)
            .any(|c| df.column(c).is_err())
        {
            return Ok(());
        }
        let tool = echo(rule.tool);
        let on = echo(request.on);
        let index_columns = request.index.len() as u64;
        let height = df.height() as u64;

        if let Some(cap) = self.pivot_cells
            && height > cap
        {
            return Err(LimitViolation {
                message: format!(
                    "The dataset has {height} rows, and `{tool}` fills one cell of the wide \
                     table from each row (one row for each pair of `index` values and `on` \
                     value), so it would make at least {height} cells, which exceeds this \
                     server's limit of {cap} cells (pivot_cells). Pivot fewer rows, for example \
                     after munge_filter or munge_sample; where a pair has several rows, \
                     aggregate them first with munge_group_by."
                ),
                tool: tool.to_string(),
                parameter: rule.dataset_field.to_string(),
                requested: distinct::Count::at_least(height).to_string(),
                limit: cap.to_string(),
                override_key: "pivot_cells",
            });
        }

        // Count the `on` values up to the first count that cannot pass: one that makes more
        // columns than pivot_columns, or more cells than pivot_cells, since a dataset with an
        // `on` value has at least one row.
        let stop = [self.pivot_columns.map(|cap| cap as u64), self.pivot_cells]
            .into_iter()
            .flatten()
            .map(|cap| cap.saturating_sub(index_columns).saturating_add(1))
            .min()
            .unwrap_or(u64::MAX);
        let on_values = distinct::count(df, &[request.on], stop)
            .map_err(|e| uncountable(tool, "on", on, on_budget, &e))?;
        let columns = on_values.plus(index_columns);
        if let Some(cap) = self.pivot_columns.map(|cap| cap as u64)
            && columns.n > cap
        {
            return Err(LimitViolation {
                message: format!(
                    "Parameter `on` names column `{on}`, which has {on_values} distinct values, \
                     so `{tool}` would make {columns} output columns (index columns included), \
                     which exceeds this server's limit of {cap} columns (pivot_columns). Pivot a \
                     column with fewer distinct values, or first keep only the rows whose `{on}` \
                     values you need, for example with munge_filter."
                ),
                tool: tool.to_string(),
                parameter: "on".to_string(),
                requested: columns.to_string(),
                limit: cap.to_string(),
                override_key: "pivot_columns",
            });
        }

        let Some(cap) = self.pivot_cells else {
            return Ok(());
        };
        // The rows are at most the dataset's height; count them only when that could not pass,
        // up to the first count that cannot (`columns.n` is at least 1, the index column).
        if height.saturating_mul(columns.n) <= cap {
            return Ok(());
        }
        let index_echo = crate::tools::common::echo_list(
            &request.index,
            ", ",
            crate::tools::common::REFUSAL_NAMES,
        );
        let rows = distinct::count(df, &request.index, cap / columns.n + 1)
            .map_err(|e| uncountable(tool, "index", &index_echo, ("pivot_cells", cap), &e))?;
        let cells = rows.times(columns);
        if cells.n <= cap {
            return Ok(());
        }
        Err(LimitViolation {
            message: format!(
                "`{tool}` would make {rows} rows (one per distinct combination of the `index` \
                 columns {index_echo}) by {columns} columns (the `on` column `{on}` has \
                 {on_values} distinct values), {cells} cells, which exceeds this server's limit \
                 of {cap} cells (pivot_cells). Pivot fewer rows or fewer `on` values, for example \
                 after munge_filter or munge_sample."
            ),
            tool: tool.to_string(),
            parameter: "index".to_string(),
            requested: cells.to_string(),
            limit: cap.to_string(),
            override_key: "pivot_cells",
        })
    }

    fn check_with(
        &self,
        tool: &str,
        args: &Value,
        rules: &[ParamRule],
    ) -> Result<(), LimitViolation> {
        if self.is_off() {
            return Ok(());
        }
        let tool_echo = echo(tool);
        // Fail closed: every rule path is a name, which only an object can resolve.
        if let Some(kind) = non_object_kind(args) {
            return Err(LimitViolation {
                tool: tool_echo.to_string(),
                parameter: "arguments".to_string(),
                requested: kind.to_string(),
                limit: "a JSON object".to_string(),
                override_key: "",
                message: non_object_message(tool, kind),
            });
        }
        let mut values = Vec::new();
        for rule in select_rules(tool, rules) {
            // An `Exempt` checks nothing. It is selected only so that a tool-specific one shadows
            // a generic rule on its path; resolving it would walk every element of an exempt
            // array path for no effect.
            if matches!(rule.rule, Rule::Exempt(_)) {
                continue;
            }
            values.clear();
            if let Err(rest) = resolve_path(args, rule.path, &mut values) {
                // `args` is an object here, so `rest` follows a `.` in the path: what comes
                // before it is the struct that was given by position.
                let parameter = rule
                    .path
                    .strip_suffix(rest)
                    .and_then(|p| p.strip_suffix('.'))
                    .unwrap_or(rule.path);
                return Err(positional_struct_violation(tool_echo, parameter));
            }
            for value in &values {
                self.check_value(tool_echo, rule, value)?;
            }
        }
        Ok(())
    }

    /// Applies one rule to one resolved value; `tool_echo` is the tool name as [`echo`] cut it.
    fn check_value(
        &self,
        tool_echo: &str,
        rule: &ParamRule,
        value: &Value,
    ) -> Result<(), LimitViolation> {
        let path = rule.path;
        let violation =
            |requested: String, limit: String, override_key: &'static str, message: String| {
                Err(LimitViolation {
                    tool: tool_echo.to_string(),
                    parameter: path.to_string(),
                    requested,
                    limit,
                    override_key,
                    message,
                })
            };
        match rule.rule {
            Rule::Exempt(_) => Ok(()),
            Rule::Max(Family::StringBytes) => {
                let (Some(cap), Some(s)) = (self.cap(Family::StringBytes), value.as_str()) else {
                    return Ok(());
                };
                let len = s.len() as u64;
                if len <= cap {
                    return Ok(());
                }
                violation(
                    len.to_string(),
                    cap.to_string(),
                    Family::StringBytes.key(),
                    format!(
                        "Parameter `{path}` is {len} bytes long, which exceeds this server's limit \
                         of {cap} bytes (string_bytes) for `{tool_echo}`. Use a shorter value."
                    ),
                )
            }
            Rule::Max(family) => {
                let (Some(cap), Some(n)) = (self.cap(family), value.as_number()) else {
                    return Ok(());
                };
                if !number_exceeds(n, cap) {
                    return Ok(());
                }
                let key = family.key();
                let subject = format!("= {n}");
                let message = magnitude_refusal(path, &subject, n, cap, key, tool_echo, false);
                violation(n.to_string(), cap.to_string(), key, message)
            }
            Rule::ArrayLen(family) | Rule::UniqueList(family) => {
                let (Some(cap), Some(items)) = (self.cap(family), value.as_array()) else {
                    return Ok(());
                };
                let key = family.key();
                let len = items.len() as u64;
                if len > cap {
                    return violation(
                        len.to_string(),
                        cap.to_string(),
                        key,
                        format!(
                            "Parameter `{path}` has {len} elements, which exceeds this server's \
                             limit of {cap} ({key}) for `{tool_echo}`. Use fewer elements."
                        ),
                    );
                }
                if matches!(rule.rule, Rule::UniqueList(_)) {
                    let mut seen = HashSet::with_capacity(items.len());
                    for name in items.iter().filter_map(Value::as_str) {
                        if !seen.insert(name) {
                            let name = echo(name);
                            return violation(
                                name.to_string(),
                                "no duplicates".to_string(),
                                key,
                                format!(
                                    "Parameter `{path}` lists `{name}` more than once for \
                                     `{tool_echo}`. List each name once."
                                ),
                            );
                        }
                    }
                }
                Ok(())
            }
            Rule::ArrayElemMax(family) => {
                let (Some(cap), Some(items)) = (self.cap(family), value.as_array()) else {
                    return Ok(());
                };
                let key = family.key();
                for n in items.iter().filter_map(Value::as_number) {
                    if number_exceeds(n, cap) {
                        let subject = format!("contains {n}, which");
                        let message =
                            magnitude_refusal(path, &subject, n, cap, key, tool_echo, true);
                        return violation(n.to_string(), cap.to_string(), key, message);
                    }
                }
                Ok(())
            }
            Rule::Tolerance => {
                let (Some(min), Some(n)) = (self.min_tolerance, value.as_number()) else {
                    return Ok(());
                };
                let reading = n.as_f64();
                if !below_min_tolerance(reading, min) {
                    return Ok(());
                }
                let shown = n.to_string();
                let message = tolerance_refusal(path, &shown, reading.is_some(), min, tool_echo);
                violation(shown, format!("{min:e}"), "min_tolerance", message)
            }
            Rule::SquareLen => {
                let (Some(cells), Some(items)) = (self.table_cells, value.as_array()) else {
                    return Ok(());
                };
                let n = items.len() as u64;
                match n.checked_mul(n) {
                    Some(square) if square <= cells => Ok(()),
                    _ => violation(
                        n.to_string(),
                        cells.to_string(),
                        "table_cells",
                        format!(
                            "Parameter `{path}` has {n} elements, which would produce a {n} x {n} \
                             result and exceeds this server's limit of {cells} cells (table_cells) \
                             for `{tool_echo}`. Use at most {} elements.",
                            cells.isqrt()
                        ),
                    ),
                }
            }
        }
    }
}

/// The process-wide limits, set once at startup by [`install`].
static INSTALLED: RwLock<Option<Arc<Limits>>> = RwLock::new(None);

/// The shared off profile returned while nothing is installed (allocated once).
static OFF: OnceLock<Arc<Limits>> = OnceLock::new();

thread_local! {
    /// The test override set by [`with_limits`] on this thread.
    static OVERRIDE: RefCell<Option<Arc<Limits>>> = const { RefCell::new(None) };
}

/// The active limits: this thread's [`with_limits`] override if one is set, else the limits
/// passed to [`install`], else the off profile.
pub fn current() -> Arc<Limits> {
    if let Some(limits) = OVERRIDE.with(|o| o.borrow().clone()) {
        return limits;
    }
    current_from(&INSTALLED)
}

/// Limits for this process: the profile from `config.resource_profile()` plus
/// `config.limits.overrides`. An unknown override key or an invalid value is an error (startup
/// fails).
pub fn resolve(config: &crate::config::ServerConfig) -> Result<Limits, String> {
    use crate::config::ResourceProfile;
    match config.resource_profile() {
        ResourceProfile::Off => Limits::off(),
        ResourceProfile::Hosted => Limits::hosted(),
    }
    .with_overrides(&config.limits.overrides)
}

/// Installs the process-wide limits (startup; a later call replaces them).
pub fn install(limits: Limits) {
    install_into(&INSTALLED, limits);
}

/// Runs `f` with `limits` as this thread's active limits and restores the previous ones
/// afterwards, also when `f` panics. Calls nest.
///
/// This is meant for tests. The override is thread-local: it does not follow an async task
/// that moves to another worker thread, and threads spawned inside `f` do not see it.
pub fn with_limits<R>(limits: Limits, f: impl FnOnce() -> R) -> R {
    /// Restores the previous override on drop (normal return or unwinding).
    struct Restore(Option<Arc<Limits>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            // `try_with`: never panic while the thread's locals are being torn down.
            let _ = OVERRIDE.try_with(|o| *o.borrow_mut() = previous);
        }
    }
    let previous = OVERRIDE.with(|o| o.replace(Some(Arc::new(limits))));
    let _restore = Restore(previous);
    f()
}

fn install_into(slot: &RwLock<Option<Arc<Limits>>>, limits: Limits) {
    *slot.write().unwrap_or_else(PoisonError::into_inner) = Some(Arc::new(limits));
}

fn current_from(slot: &RwLock<Option<Arc<Limits>>>) -> Arc<Limits> {
    let installed = slot.read().unwrap_or_else(PoisonError::into_inner);
    match installed.as_ref() {
        Some(limits) => Arc::clone(limits),
        None => Arc::clone(OFF.get_or_init(|| Arc::new(Limits::off()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde_json::json;

    const T: &[ParamRule] = &[
        ParamRule {
            tool: "*",
            path: "reps",
            rule: Rule::Max(Family::Replications),
        },
        ParamRule {
            tool: "special",
            path: "reps",
            rule: Rule::Max(Family::Folds),
        },
        ParamRule {
            tool: "*",
            path: "cols",
            rule: Rule::UniqueList(Family::ListLength),
        },
        ParamRule {
            tool: "*",
            path: "tol",
            rule: Rule::Tolerance,
        },
        ParamRule {
            tool: "*",
            path: "label",
            rule: Rule::Max(Family::StringBytes),
        },
        ParamRule {
            tool: "*",
            path: "items[].name",
            rule: Rule::Max(Family::StringBytes),
        },
        ParamRule {
            tool: "*",
            path: "items[].spec.cats",
            rule: Rule::ArrayLen(Family::ListLength),
        },
        ParamRule {
            tool: "*",
            path: "grid[]",
            rule: Rule::ArrayLen(Family::ListLength),
        },
        ParamRule {
            tool: "*",
            path: "spans",
            rule: Rule::ArrayLen(Family::ListLength),
        },
        ParamRule {
            tool: "*",
            path: "spans",
            rule: Rule::ArrayElemMax(Family::SpanWidth),
        },
        ParamRule {
            tool: "*",
            path: "vals",
            rule: Rule::SquareLen,
        },
        ParamRule {
            tool: "*",
            path: "free",
            rule: Rule::Exempt("test"),
        },
    ];

    #[test]
    fn max_rule_message_and_fields() {
        let h = Limits::hosted();
        let v = h
            .check_with(
                "mediation_analysis",
                &json!({"reps": 1_000_000_000_000u64}),
                T,
            )
            .unwrap_err();
        assert_eq!(
            v.message,
            "Parameter `reps` = 1000000000000 exceeds this server's limit of 10000 (replications) for `mediation_analysis`. Use a smaller value."
        );
        assert_eq!(
            (
                v.tool.as_str(),
                v.parameter.as_str(),
                v.requested.as_str(),
                v.limit.as_str(),
                v.override_key
            ),
            (
                "mediation_analysis",
                "reps",
                "1000000000000",
                "10000",
                "replications"
            )
        );
        assert_eq!(v.to_string(), v.message);
        assert!(h.check_with("t", &json!({"reps": 10_000}), T).is_ok());
        assert!(h.check_with("t", &json!({"reps": 10_001}), T).is_err());
        // `requested` is the JSON number's own `to_string()` (serde_json 1.0.148 writes "1e+30").
        assert_eq!(
            h.check_with("t", &json!({"reps": 1e30}), T)
                .unwrap_err()
                .requested,
            json!(1e30).to_string()
        );
        // absolute value; a negative value is told to shrink in magnitude, not to get smaller
        let v = h
            .check_with(
                "mediation_analysis",
                &json!({"reps": -1_000_000_000_000i64}),
                T,
            )
            .unwrap_err();
        assert_eq!(
            v.message,
            "Parameter `reps` = -1000000000000 is outside this server's limit of ±10000 (replications) for `mediation_analysis`. Use a value of smaller magnitude."
        );
        assert_eq!(
            (v.requested.as_str(), v.limit.as_str(), v.override_key),
            ("-1000000000000", "10000", "replications")
        );
        let v = h.check_with("t", &json!({"reps": -10_001}), T).unwrap_err();
        assert!(v.message.contains("limit of ±10000"), "{}", v.message);
        assert!(h.check_with("t", &json!({"reps": -10_000}), T).is_ok());
        assert!(
            h.check_with("t", &json!({"reps": null, "other": 1e30}), T)
                .is_ok()
        );
        // not a number: serde's job
        assert!(h.check_with("t", &json!({"reps": "1e30"}), T).is_ok());
        // not an object: refused (see `arguments_that_are_not_an_object_are_refused`)
        assert!(h.check_with("t", &json!([1, 2]), T).is_err());
    }

    #[test]
    fn arguments_that_are_not_an_object_are_refused() {
        // A derived request struct deserialises a JSON array too, filling its fields by
        // position, so positional arguments must not pass a check that reads them by name.
        let h = Limits::hosted();
        for (args, kind) in [
            (json!(["small", "y", ["x"], 20_000, "pairs", 1]), "an array"),
            (json!("small"), "a string"),
            (json!(20_000), "a number"),
            (json!(true), "a boolean"),
        ] {
            let v = h
                .check_tool_arguments("regression_bootstrap_cov", &args)
                .unwrap_err();
            assert_eq!(
                (
                    v.tool.as_str(),
                    v.parameter.as_str(),
                    v.requested.as_str(),
                    v.limit.as_str(),
                ),
                (
                    "regression_bootstrap_cov",
                    "arguments",
                    kind,
                    "a JSON object"
                ),
                "{args}"
            );
            assert_eq!(
                v.message,
                format!(
                    "The arguments of `regression_bootstrap_cov` are {kind}, but tool arguments \
                     must be a JSON object that names each parameter. Send each value under its \
                     parameter name, as the tool's input schema lists them."
                ),
                "{args}"
            );
        }
        // `null` is arguments left out, the empty object.
        assert!(h.check_tool_arguments("t", &Value::Null).is_ok());
        // The off profile checks nothing; the governed call path refuses the shape in every
        // profile (`AnalyticsServer::call_tool_with_session`).
        assert!(Limits::off().check_tool_arguments("t", &json!([1])).is_ok());
        // The tool name is echoed cut, like every other echo.
        let v = h
            .check_tool_arguments(&"t".repeat(10_000), &json!([1]))
            .unwrap_err();
        assert_eq!(v.tool.len(), ECHO_CHARS);
        assert!(v.message.len() < 300, "{}", v.message);
    }

    #[test]
    fn a_nested_struct_given_by_position_is_refused() {
        // serde fills a nested request struct from a JSON array by field position too, so a rule
        // that goes on into a field must not find nothing there and pass.
        let h = Limits::hosted();
        let long = "x".repeat(5_000);
        let cats: Vec<String> = (0..501).map(|i| i.to_string()).collect();
        for (args, parameter) in [
            // `items[]` by position: `items[].name` and `items[].spec.cats` cannot resolve.
            (json!({"items": [[long.clone()]]}), "items[]"),
            (
                json!({"items": [{"name": "ok"}, [long.clone()]]}),
                "items[]",
            ),
            // `items[].spec` by position inside an element given by name.
            (
                json!({"items": [{"name": "ok", "spec": [cats]}]}),
                "items[].spec",
            ),
        ] {
            let v = h.check_with("t", &args, T).unwrap_err();
            assert_eq!(
                (
                    v.tool.as_str(),
                    v.parameter.as_str(),
                    v.requested.as_str(),
                    v.limit.as_str(),
                    v.override_key,
                ),
                ("t", parameter, "an array", "a JSON object", ""),
                "{args}"
            );
            assert_eq!(
                v.message,
                format!(
                    "Parameter `{parameter}` of `t` is an array, but it must be a JSON object \
                     that names each field. Send each value under its field name, as the tool's \
                     input schema lists them."
                ),
                "{args}"
            );
        }
        // The same values by name are checked (and these pass).
        let by_name = json!({"items": [{"name": "ok", "spec": {"cats": ["a"]}}]});
        assert!(h.check_with("t", &by_name, T).is_ok());
        // A scalar or null where a struct belongs cannot be deserialised into one: serde's job.
        for args in [
            json!({"items": ["x", 1, true, null]}),
            json!({"items": [{"spec": "x"}]}),
        ] {
            assert!(h.check_with("t", &args, T).is_ok(), "{args}");
        }
        // An array at the end of a path, or in an element position, is a value, not a struct.
        assert!(h.check_with("t", &json!({"grid": [[1, 2]]}), T).is_ok());
        // An `Exempt` path is never resolved, so it refuses nothing.
        let free = &[ParamRule {
            tool: "*",
            path: "free[].a",
            rule: Rule::Exempt("test"),
        }];
        assert!(h.check_with("t", &json!({"free": [[1]]}), free).is_ok());
        // The off profile checks nothing.
        assert!(
            Limits::off()
                .check_with("t", &json!({"items": [[long]]}), T)
                .is_ok()
        );
        // The tool name is echoed cut.
        let v = h
            .check_with(&"t".repeat(10_000), &json!({"items": [[1]]}), T)
            .unwrap_err();
        assert_eq!(v.tool.len(), ECHO_CHARS);
        assert!(v.message.len() < 300, "{}", v.message);
    }

    #[test]
    fn a_number_without_a_u64_i64_or_f64_reading_exceeds_every_cap() {
        // Only serde_json's `arbitrary_precision` feature produces such a number (1e400); the
        // check must refuse it rather than let it through.
        assert!(magnitude_exceeds(None, None, None, u64::MAX));
        assert!(magnitude_exceeds(Some(11), Some(11), Some(11.0), 10));
        assert!(!magnitude_exceeds(Some(10), Some(10), Some(10.0), 10));
        assert!(magnitude_exceeds(None, Some(-11), Some(-11.0), 10));
        assert!(!magnitude_exceeds(None, None, Some(-9.5), 10));
    }

    #[test]
    fn a_tolerance_without_an_f64_reading_is_below_every_minimum() {
        // Only serde_json's `arbitrary_precision` feature produces such a number (-1e400); the
        // minimum-tolerance check must refuse it rather than let it through.
        assert!(below_min_tolerance(None, 1e-12));
        assert!(!below_min_tolerance(Some(1e-12), 1e-12));
        assert!(below_min_tolerance(Some(9.999e-13), 1e-12));
        assert!(!below_min_tolerance(Some(1e-8), 1e-12));
        assert!(below_min_tolerance(Some(0.0), 1e-12));
        assert!(below_min_tolerance(Some(-1.0), 1e-12));
        assert!(below_min_tolerance(Some(f64::NAN), 1e-12));
        // the hosted minimum itself is accepted
        let h = Limits::hosted();
        assert!(h.check_with("t", &json!({"tol": 1e-15}), T).is_ok());
    }

    #[test]
    fn a_tolerance_without_an_f64_reading_is_not_called_below_the_minimum() {
        // 1e400 has no f64 reading either, and it is not below the minimum.
        let unreadable = tolerance_refusal("tol", "1e400", false, 1e-15, "t");
        assert!(
            unreadable.contains("cannot be read as a finite number")
                && !unreadable.contains("below"),
            "{unreadable}"
        );
        let below = tolerance_refusal("tol", "1e-16", true, 1e-15, "t");
        assert_eq!(
            below,
            "Parameter `tol` = 1e-16 is below this server's minimum of 1e-15 (min_tolerance) for \
             `t`. Use a larger tolerance such as 1e-8."
        );
        // the refusal of a readable tolerance is that message
        let v = Limits::hosted()
            .check_with("t", &json!({"tol": 1e-16}), T)
            .unwrap_err();
        assert_eq!(
            v.message,
            tolerance_refusal("tol", &json!(1e-16).to_string(), true, 1e-15, "t")
        );
    }

    #[test]
    fn tool_specific_exempt_replaces_generic_rule_for_that_path() {
        let table = &[
            ParamRule {
                tool: "*",
                path: "reps",
                rule: Rule::Max(Family::Replications),
            },
            ParamRule {
                tool: "free",
                path: "reps",
                rule: Rule::Exempt("test"),
            },
            ParamRule {
                tool: "*",
                path: "grid[]",
                rule: Rule::Exempt("test"),
            },
        ];
        let h = Limits::hosted();
        assert!(h.check_with("t", &json!({"reps": 10_001}), table).is_err());
        assert!(
            h.check_with("free", &json!({"reps": 10_001}), table)
                .is_ok()
        );
        // an Exempt on an element path checks nothing
        let grid = json!({"grid": vec![vec![1e30; 3]; 1_000]});
        assert!(h.check_with("t", &grid, table).is_ok());
        // check_with skips an Exempt, but it is still selected, and so shadows the generic rule
        let selected = select_rules("free", table);
        let on_reps = |r: &&ParamRule| r.path == "reps";
        assert!(
            selected
                .iter()
                .copied()
                .filter(on_reps)
                .all(|r| matches!(r.rule, Rule::Exempt(_)))
        );
        assert_eq!(selected.iter().copied().filter(on_reps).count(), 1);
    }

    #[test]
    fn refusals_echo_at_most_64_characters_of_the_tool_name() {
        let h = Limits::hosted();
        let long = "t".repeat(100);
        let v = h
            .check_with(&long, &json!({"reps": 10_001}), T)
            .unwrap_err();
        assert_eq!(v.tool, "t".repeat(64));
        assert!(v.message.contains(&format!("`{}`", "t".repeat(64))));
        assert!(!v.message.contains(&"t".repeat(65)), "{}", v.message);
        // a short name is echoed whole, borrowed from the caller's string
        let short = "regression_ols";
        assert!(std::ptr::eq(echo(short), short));
        assert_eq!(echo(&"é".repeat(65)), "é".repeat(64));
    }

    #[test]
    fn tool_specific_rule_replaces_generic_rule_for_that_path() {
        let h = Limits::hosted();
        let v = h
            .check_with("special", &json!({"reps": 21}), T)
            .unwrap_err();
        assert_eq!(v.override_key, "folds");
        assert!(h.check_with("special", &json!({"reps": 20}), T).is_ok());
    }

    #[test]
    fn list_string_tolerance_square_and_nested_rules() {
        let h = Limits::hosted();
        let dup = h
            .check_with("multivariate_cancor", &json!({"cols": ["a", "b", "a"]}), T)
            .unwrap_err();
        assert!(
            dup.message.contains("lists `a` more than once") && dup.limit == "no duplicates",
            "{}",
            dup.message
        );
        let long: Vec<String> = (0..501).map(|i| format!("c{i}")).collect();
        assert_eq!(
            h.check_with("t", &json!({"cols": long}), T)
                .unwrap_err()
                .requested,
            "501"
        );
        // non-strings: no duplicate check
        assert!(h.check_with("t", &json!({"cols": [1, 1]}), T).is_ok());
        // echoed name is char-truncated
        let name = "é".repeat(80);
        let v = h
            .check_with("t", &json!({"cols": [name.clone(), name]}), T)
            .unwrap_err();
        assert_eq!(v.requested, "é".repeat(64));

        // 4096 bytes
        assert!(
            h.check_with("t", &json!({"label": "é".repeat(2048)}), T)
                .is_ok()
        );
        // 4098 bytes
        let v = h
            .check_with("t", &json!({"label": "é".repeat(2049)}), T)
            .unwrap_err();
        assert_eq!(
            (v.requested.as_str(), v.override_key),
            ("4098", "string_bytes")
        );
        // StringBytes ignores numbers
        assert!(h.check_with("t", &json!({"label": 1e30}), T).is_ok());

        assert!(h.check_with("panel_hdfe", &json!({"tol": 1e-8}), T).is_ok());
        for bad in [json!(0.0), json!(-1.0), json!(1e-16)] {
            let v = h
                .check_with("panel_hdfe", &json!({"tol": bad}), T)
                .unwrap_err();
            assert_eq!(v.override_key, "min_tolerance");
            assert!(v.message.contains("1e-15"), "{}", v.message);
        }

        assert!(
            h.check_with("t", &json!({"vals": vec![0.0; 1000]}), T)
                .is_ok()
        );
        let v = h
            .check_with("linalg_toeplitz", &json!({"vals": vec![0.0; 1001]}), T)
            .unwrap_err();
        assert!(
            v.message.contains("1001 x 1001") && v.message.contains("at most 1000 elements"),
            "{}",
            v.message
        );

        let nested =
            json!({"items": [{"name": "ok", "spec": {"cats": ["a"]}}, {"name": "x".repeat(4097)}]});
        assert_eq!(
            h.check_with("t", &nested, T).unwrap_err().parameter,
            "items[].name"
        );
        let cats: Vec<String> = (0..501).map(|i| i.to_string()).collect();
        assert_eq!(
            h.check_with("t", &json!({"items": [{"spec": {"cats": cats}}]}), T)
                .unwrap_err()
                .parameter,
            "items[].spec.cats"
        );
        // [] on non-array
        assert!(
            h.check_with("t", &json!({"items": {"name": "x".repeat(5000)}}), T)
                .is_ok()
        );
        let inner: Vec<u8> = vec![0; 501];
        assert_eq!(
            h.check_with("t", &json!({"grid": [[1], inner]}), T)
                .unwrap_err()
                .parameter,
            "grid[]"
        );

        assert!(h.check_with("t", &json!({"spans": [3, 1001]}), T).is_ok());
        let v = h
            .check_with("timeseries_spectrum", &json!({"spans": [3, 1003]}), T)
            .unwrap_err();
        assert_eq!(
            v.message,
            "Parameter `spans` contains 1003, which exceeds this server's limit of 1001 (span_width) for `timeseries_spectrum`. Use smaller values."
        );
        assert_eq!(v.override_key, "span_width");
        assert!(h.check_with("t", &json!({"spans": [3, -1001]}), T).is_ok());
        let v = h
            .check_with("timeseries_spectrum", &json!({"spans": [3, -1003]}), T)
            .unwrap_err();
        assert_eq!(
            v.message,
            "Parameter `spans` contains -1003, which is outside this server's limit of ±1001 (span_width) for `timeseries_spectrum`. Use values of smaller magnitude."
        );
        assert_eq!((v.requested.as_str(), v.limit.as_str()), ("-1003", "1001"));
        assert!(h.check_with("t", &json!({"free": 1e300}), T).is_ok());
    }

    #[test]
    fn off_profile_checks_nothing() {
        let o = Limits::off();
        let absurd = json!({"reps": 1e30, "cols": ["a", "a"], "tol": 0.0, "label": "x".repeat(100_000),
                            "vals": vec![0; 5000], "spans": [1_000_001]});
        assert!(o.check_with("t", &absurd, T).is_ok());
        assert!(o.check_rows("ml_tsne", 1_000_000_000, 10).is_ok());
        assert!(
            o.check_budget(
                "munge_join",
                "join_output_rows",
                u64::MAX,
                o.join_output_rows
            )
            .is_ok()
        );
    }

    #[test]
    fn rows_and_budgets() {
        let h = Limits::hosted();
        assert!(h.check_rows("ml_tsne", 2_000, 3).is_ok());
        let v = h.check_rows("ml_tsne", 2_001, 3).unwrap_err();
        assert_eq!(
            (
                v.parameter.as_str(),
                v.requested.as_str(),
                v.limit.as_str(),
                v.override_key
            ),
            ("dataset", "2001", "2000", "cubic_rows")
        );
        assert!(
            v.message.contains("2001 rows and 3 columns") && v.message.contains("munge_sample"),
            "{}",
            v.message
        );
        assert!(h.check_rows("ml_hierarchical", 5_000, 2).is_ok());
        assert_eq!(
            h.check_rows("ml_hierarchical", 5_001, 2)
                .unwrap_err()
                .override_key,
            "dense_rows"
        );
        assert!(h.check_rows("ml_svm", 50_000, 2).is_ok());
        assert_eq!(
            h.check_rows("ml_svm", 50_001, 2).unwrap_err().override_key,
            "quadratic_rows"
        );
        // no row rule
        assert!(h.check_rows("regression_ols", 10_000_000, 50).is_ok());

        let v = h
            .check_budget(
                "munge_join",
                "join_output_rows",
                3_000_000,
                h.join_output_rows,
            )
            .unwrap_err();
        assert_eq!(
            v.message,
            "This request to `munge_join` needs 3000000 (join_output_rows), which exceeds this server's limit of 2000000. Use a smaller request."
        );
        assert_eq!(
            (v.parameter.as_str(), v.override_key),
            ("join_output_rows", "join_output_rows")
        );
        assert!(
            h.check_budget(
                "munge_join",
                "join_output_rows",
                2_000_000,
                h.join_output_rows
            )
            .is_ok()
        );
        assert!(h.check_budget("x", "anything", u64::MAX, None).is_ok());
    }

    #[test]
    fn hosted_family_caps_match_design() {
        let h = Limits::hosted();
        let expect: &[(Family, &str, u64)] = &[
            (Family::Replications, "replications", 10_000),
            (Family::SimulationSize, "simulation_size", 100_000),
            (Family::Trees, "trees", 5_000),
            (Family::BoostingIterations, "boosting_iterations", 10_000),
            (Family::McmcDraws, "mcmc_draws", 20_000),
            (Family::Folds, "folds", 20),
            (Family::Iterations, "iterations", 100_000),
            (Family::Restarts, "restarts", 100),
            (Family::GridPoints, "grid_points", 10_000),
            (Family::Bins, "bins", 1_000),
            (Family::Lags, "lags", 1_000),
            (Family::Horizon, "horizon", 5_000),
            (Family::ModelOrder, "model_order", 20),
            (Family::Bandwidth, "bandwidth", 1_000),
            (Family::SpanWidth, "span_width", 1_001),
            (Family::Components, "components", 500),
            (Family::Neighbors, "neighbors", 100),
            (Family::GeneratedRows, "generated_rows", 1_000_000),
            (Family::ImageDimension, "image_dimension", 4_000),
            (Family::ListLength, "list_length", 500),
            (Family::ArrayLength, "array_length", 100_000),
            (Family::StringBytes, "string_bytes", 4_096),
            (Family::TreeDepth, "tree_depth", 100),
            (Family::PowerSampleSize, "power_sample_size", 1_000_000),
            (Family::DesignGroups, "design_groups", 10),
        ];
        assert_eq!(Family::ALL.len(), expect.len());
        for &(f, key, cap) in expect {
            assert_eq!(f.key(), key);
            assert_eq!(h.cap(f), Some(cap), "{key}");
            assert_eq!(Limits::off().cap(f), None, "{key}");
        }
        let keys: std::collections::HashSet<_> = Family::ALL.iter().map(|f| f.key()).collect();
        assert_eq!(keys.len(), Family::ALL.len());
    }

    #[test]
    fn hosted_and_off_budgets_match_design() {
        let h = Limits::hosted();
        assert_eq!(
            (h.dense_rows, h.cubic_rows, h.quadratic_rows),
            (Some(5_000), Some(2_000), Some(50_000))
        );
        assert_eq!(h.min_tolerance, Some(1e-15));
        assert_eq!(
            (h.image_pixels, h.table_cells, h.max_groups),
            (Some(16_000_000), Some(1_000_000), Some(100))
        );
        assert_eq!(
            (h.join_output_rows, h.onehot_levels),
            (Some(2_000_000), Some(500))
        );
        assert_eq!(
            (h.pivot_columns, h.pivot_cells),
            (Some(10_000), Some(1_000_000))
        );
        assert_eq!(
            (h.str_split_parts, h.str_pad_width),
            (Some(100), Some(10_000))
        );
        assert_eq!(
            (h.generated_cells, h.generated_bytes),
            (Some(10_000_000), Some(200 * 1024 * 1024))
        );
        assert_eq!(
            (h.export_bytes, h.response_bytes, h.session_disk_bytes),
            (
                Some(64 * 1024 * 1024),
                Some(16 * 1024 * 1024),
                Some(64 * 1024 * 1024)
            )
        );
        assert_eq!(h.load_file_bytes, Some(64 * 1024 * 1024));
        assert_eq!(h.session_db_bytes, Some(16 * 1024 * 1024));
        assert!(!h.sql_freeform);
        assert_eq!(h.spatial_links, Some(2_000_000));
        assert_eq!(h.arima_fit_work, Some(400_000));
        assert_eq!(
            (h.db_result_rows, h.db_result_bytes),
            (Some(100_000), Some(64 * 1024 * 1024))
        );
        assert_eq!(
            (h.db_timeout_secs, h.db_memory_mb, h.db_threads),
            (Some(20), Some(256), Some(1))
        );
        assert_eq!(
            (
                h.message_bytes,
                h.conversations_per_session,
                h.messages_per_conversation
            ),
            (262_144, Some(50), Some(1_000))
        );
        assert_eq!((h.llm_timeout_secs, h.llm_max_tool_iterations), (120, 25));
        assert_eq!(
            (
                h.llm_context_bytes,
                h.llm_history_bytes,
                h.llm_streams_per_session
            ),
            (24_576, 131_072, Some(1))
        );
        assert!(!h.llm_server_keys && !h.llm_allow_loopback);
        assert!(!h.is_off());

        let o = Limits::off();
        assert!(o.is_off());
        assert_eq!(o, Limits::default());
        assert_eq!(
            (
                o.dense_rows,
                o.min_tolerance,
                o.table_cells,
                o.db_threads,
                o.response_bytes
            ),
            (None, None, None, None, None)
        );
        assert_eq!(o.load_file_bytes, None);
        assert_eq!((o.pivot_columns, o.pivot_cells), (None, None));
        assert_eq!(
            (o.export_bytes, o.session_disk_bytes, o.session_db_bytes),
            (None, None, None)
        );
        assert!(o.sql_freeform);
        assert_eq!(
            (
                o.message_bytes,
                o.llm_timeout_secs,
                o.llm_max_tool_iterations
            ),
            (2_097_152, 900, 50)
        );
        assert_eq!(
            (
                o.llm_context_bytes,
                o.llm_history_bytes,
                o.llm_streams_per_session
            ),
            (65_536, 524_288, None)
        );
        assert!(o.llm_server_keys && o.llm_allow_loopback);
    }

    #[test]
    fn overrides_parse_and_apply() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let l = Limits::hosted()
            .with_overrides(&s(&[
                "replications=50000",
                " dense_rows = 100 , llm_server_keys=TRUE",
            ]))
            .unwrap();
        assert_eq!(l.cap(Family::Replications), Some(50_000));
        assert_eq!(l.dense_rows, Some(100));
        assert!(l.llm_server_keys);
        let l = Limits::hosted()
            .with_overrides(&s(&[
                "replications=none",
                "db_threads=unlimited",
                "min_tolerance=1e-12",
                "",
            ]))
            .unwrap();
        assert_eq!(
            (l.cap(Family::Replications), l.db_threads, l.min_tolerance),
            (None, None, Some(1e-12))
        );
        assert_eq!(
            Limits::hosted()
                .with_overrides(&s(&["trees=1", "trees=2"]))
                .unwrap()
                .cap(Family::Trees),
            Some(2)
        );
        let off = Limits::off().with_overrides(&s(&["folds=5"])).unwrap();
        assert!(!off.is_off());

        let err = Limits::hosted()
            .with_overrides(&s(&["bogus=1"]))
            .unwrap_err();
        assert!(
            err.contains("unknown limit `bogus`") && err.contains("replications"),
            "{err}"
        );
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["replications"]))
                .unwrap_err()
                .contains("key=value")
        );
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["replications=abc"]))
                .unwrap_err()
                .contains("invalid value `abc`")
        );
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["llm_timeout_secs=none"]))
                .unwrap_err()
                .contains("cannot be unlimited")
        );
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["llm_server_keys=yes"]))
                .is_err()
        );
        let l = Limits::hosted()
            .with_overrides(&s(&["sql_freeform=TRUE", "session_db_bytes=none"]))
            .unwrap();
        assert!(l.sql_freeform && l.session_db_bytes.is_none());
        assert!(
            !Limits::off()
                .with_overrides(&s(&["sql_freeform=false"]))
                .unwrap()
                .sql_freeform
        );
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["sql_freeform=none"]))
                .is_err()
        );
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["min_tolerance=0"]))
                .is_err()
        );
        // u32 overflow
        assert!(
            Limits::hosted()
                .with_overrides(&s(&["db_threads=4294967296"]))
                .is_err()
        );
    }

    #[test]
    fn every_known_key_is_settable() {
        const BOOLS: &[&str] = &["llm_server_keys", "llm_allow_loopback", "sql_freeform"];
        let keys = Limits::known_keys();
        // `set` tries the family keys first, so a named budget sharing a family key could never
        // be set.
        let unique: HashSet<_> = keys.iter().collect();
        assert_eq!(unique.len(), keys.len(), "duplicate limit keys in {keys:?}");
        for f in Family::ALL {
            assert!(keys.contains(&f.key()));
        }
        for k in [
            "dense_rows",
            "cubic_rows",
            "quadratic_rows",
            "min_tolerance",
            "image_pixels",
            "table_cells",
            "max_groups",
            "join_output_rows",
            "onehot_levels",
            "pivot_columns",
            "pivot_cells",
            "str_split_parts",
            "str_pad_width",
            "generated_cells",
            "generated_bytes",
            "export_bytes",
            "load_file_bytes",
            "response_bytes",
            "session_disk_bytes",
            "spatial_links",
            "arima_fit_work",
            "db_result_rows",
            "db_result_bytes",
            "db_timeout_secs",
            "db_memory_mb",
            "db_threads",
            "sql_freeform",
            "message_bytes",
            "conversations_per_session",
            "messages_per_conversation",
            "session_db_bytes",
            "llm_timeout_secs",
            "llm_max_tool_iterations",
            "llm_context_bytes",
            "llm_history_bytes",
            "llm_streams_per_session",
            "llm_server_keys",
            "llm_allow_loopback",
        ] {
            assert!(keys.contains(&k), "{k}");
        }
        assert_eq!(keys.len(), Family::ALL.len() + 38);
        for k in keys {
            let v = if BOOLS.contains(&k) { "true" } else { "1" };
            Limits::off()
                .with_overrides(&[format!("{k}={v}")])
                .unwrap_or_else(|e| panic!("{k}: {e}"));
        }
    }

    #[test]
    fn install_and_current_use_the_given_slot() {
        static SLOT: std::sync::RwLock<Option<std::sync::Arc<Limits>>> =
            std::sync::RwLock::new(None);
        assert!(current_from(&SLOT).is_off());
        install_into(&SLOT, Limits::hosted());
        assert_eq!(current_from(&SLOT).cap(Family::Folds), Some(20));
        install_into(&SLOT, Limits::off());
        assert!(current_from(&SLOT).is_off());
    }

    #[test]
    fn with_limits_is_thread_local_nestable_and_panic_safe() {
        let outer = with_limits(Limits::hosted(), || {
            let inner = with_limits(Limits::off(), || current().is_off());
            (inner, current().cap(Family::Trees))
        });
        assert_eq!(outer, (true, Some(5_000)));
        // a panic inside the inner override restores the outer override (drop guard)
        let restored = with_limits(Limits::off(), || {
            let r = std::panic::catch_unwind(|| {
                with_limits(Limits::hosted(), || -> () { panic!("boom") })
            });
            assert!(r.is_err());
            current().is_off()
        });
        assert!(restored);
        // the override is per thread
        let other = with_limits(Limits::off(), || {
            std::thread::spawn(|| with_limits(Limits::hosted(), || current().cap(Family::Trees)))
                .join()
                .unwrap()
        });
        assert_eq!(other, Some(5_000));
    }

    #[cfg(feature = "http")]
    fn open_public() -> crate::config::ServerConfig {
        let mut c = crate::config::ServerConfig::default();
        // `ServerConfig::deployment_mode()` is Local for the default stdio transport, whatever
        // the http fields say; `resolve` reads the config as given, so the transport is Http.
        c.transport = crate::config::TransportType::Http;
        c.http.addr = "0.0.0.0:8080".parse().unwrap();
        c.http.allow_unauthenticated = true;
        c
    }

    #[cfg(feature = "http")]
    #[test]
    fn resolve_open_public_is_hosted() {
        let l = resolve(&open_public()).unwrap();
        assert!(!l.is_off());
        assert_eq!(l.cap(Family::Replications), Some(10_000));
    }

    #[test]
    fn resolve_local_is_off() {
        assert!(
            resolve(&crate::config::ServerConfig::default())
                .unwrap()
                .is_off()
        );
    }

    #[test]
    fn resolve_explicit_hosted_profile_in_local_mode() {
        let mut c = crate::config::ServerConfig::default();
        c.limits.profile = Some(crate::config::ResourceProfile::Hosted);
        assert_eq!(resolve(&c).unwrap().cap(Family::Folds), Some(20));
    }

    #[cfg(feature = "http")]
    #[test]
    fn resolve_applies_overrides() {
        let mut c = open_public();
        c.limits.overrides = vec!["replications=50".into()];
        assert_eq!(resolve(&c).unwrap().cap(Family::Replications), Some(50));
    }

    #[cfg(feature = "http")]
    #[test]
    fn resolve_rejects_unknown_override_key() {
        let mut c = open_public();
        c.limits.overrides = vec!["no_such_knob=1".into()];
        assert!(resolve(&c).unwrap_err().contains("no_such_knob"));
    }

    #[test]
    fn resolve_applies_overrides_in_local_mode() {
        // Overrides apply to the off profile too (stdio builds have no http fields).
        let mut c = crate::config::ServerConfig::default();
        c.limits.overrides = vec!["folds=7".into()];
        let l = resolve(&c).unwrap();
        assert_eq!(l.cap(Family::Folds), Some(7));
        assert_eq!(l.cap(Family::Replications), None);
        c.limits.overrides = vec!["no_such_knob=1".into()];
        assert!(resolve(&c).unwrap_err().contains("no_such_knob"));
    }

    #[test]
    fn with_limits_is_visible_to_current_on_this_thread_only() {
        with_limits(Limits::hosted(), || {
            assert_eq!(current().cap(Family::Trees), Some(5_000));
            // Valid only while no lib test calls install(). The tests above use the private
            // install_into() on a test-local slot. If a later lib test calls install(), drop
            // this line.
            std::thread::spawn(|| assert!(current().is_off()))
                .join()
                .unwrap();
        });
    }

    /// The pivot budgets (PR-H1), checked on frames against the shape polars' pivot makes.
    mod pivot_budgets {
        use super::super::*;
        use p2a_core::data::{Dataset, munging::pivot};
        use p2a_core::polars::prelude::{DataFrame, PolarsError, df};
        use serde_json::json;

        fn rule() -> &'static PivotRule {
            pivot_rule("munge_pivot").expect("munge_pivot has a pivot rule")
        }

        fn args(index: &[&str]) -> Value {
            json!({"dataset": "d", "index": index, "on": "key", "values": "val"})
        }

        fn with(overrides: &str) -> Limits {
            Limits::hosted()
                .with_overrides(&[overrides.to_string()])
                .unwrap()
        }

        /// A long frame with one row per pair of `ids` ids and `keys` keys: `idx`, `name`
        /// (which `idx` determines), `key` and a noisy `val`.
        fn long(ids: usize, keys: usize) -> DataFrame {
            let n = ids * keys;
            let idx: Vec<i64> = (0..n).map(|i| (i / keys) as i64).collect();
            let name: Vec<String> = (0..n).map(|i| format!("n{}", i / keys)).collect();
            let key: Vec<String> = (0..n).map(|i| format!("k{}", i % keys)).collect();
            let val: Vec<f64> = (0..n)
                .map(|i| (i as f64 * 0.73).sin() * 5.0 + i as f64 * 0.1)
                .collect();
            df!("idx" => idx, "name" => name, "key" => key, "val" => val).unwrap()
        }

        /// The rows and columns polars' pivot makes of `df` over `index`.
        fn pivoted(df: &DataFrame, index: &[&str]) -> (u64, u64) {
            let out = pivot(&Dataset::new(df.clone()), index, "key", "val").unwrap();
            (out.nrows() as u64, out.ncols() as u64)
        }

        #[test]
        fn the_column_budget_counts_the_index_and_each_distinct_on_value() {
            let df = long(3, 4);
            assert_eq!(pivoted(&df, &["idx"]), (3, 5));
            assert!(
                with("pivot_columns=5")
                    .check_pivot(rule(), &args(&["idx"]), &df)
                    .is_ok()
            );
            // The count stops at the first value that cannot pass, so a refusal gives a lower
            // bound, here the exact count.
            let v = with("pivot_columns=4")
                .check_pivot(rule(), &args(&["idx"]), &df)
                .unwrap_err();
            assert_eq!(
                (
                    v.tool.as_str(),
                    v.parameter.as_str(),
                    v.requested.as_str(),
                    v.limit.as_str(),
                    v.override_key
                ),
                ("munge_pivot", "on", "at least 5", "4", "pivot_columns")
            );
            assert_eq!(
                v.message,
                "Parameter `on` names column `key`, which has at least 4 distinct values, so \
                 `munge_pivot` would make at least 5 output columns (index columns included), \
                 which exceeds this server's limit of 4 columns (pivot_columns). Pivot a column \
                 with fewer distinct values, or first keep only the rows whose `key` values you \
                 need, for example with munge_filter."
            );
            // Each index column is an output column.
            assert_eq!(pivoted(&df, &["idx", "name"]), (3, 6));
            assert!(
                with("pivot_columns=6")
                    .check_pivot(rule(), &args(&["idx", "name"]), &df)
                    .is_ok()
            );
            let v = with("pivot_columns=5")
                .check_pivot(rule(), &args(&["idx", "name"]), &df)
                .unwrap_err();
            assert_eq!(v.requested, "at least 6");
        }

        #[test]
        fn on_values_are_told_apart_as_the_pivot_does() {
            // A null is a column of its own.
            let df = df!(
                "idx" => [0i64, 0, 1, 1, 2],
                "key" => [Some("a"), None, Some("a"), Some("b"), None],
                "val" => [0.4, 1.3, 2.1, 0.7, 1.9]
            )
            .unwrap();
            assert_eq!(pivoted(&df, &["idx"]), (3, 4));
            let limits = with("pivot_columns=4");
            assert!(limits.check_pivot(rule(), &args(&["idx"]), &df).is_ok());
            let v = with("pivot_columns=3")
                .check_pivot(rule(), &args(&["idx"]), &df)
                .unwrap_err();
            assert_eq!(v.requested, "at least 4");
            // Float values: -0.0 is 0.0, and every NaN is one column.
            let df = df!(
                "idx" => [0i64, 1, 2, 3, 4, 5],
                "key" => [Some(0.0), Some(-0.0), Some(f64::NAN), Some(-f64::NAN), None, Some(2.5)],
                "val" => [0.4, 1.3, 2.1, 0.7, 1.9, 1.1]
            )
            .unwrap();
            assert_eq!(pivoted(&df, &["idx"]), (6, 5));
            assert!(
                with("pivot_columns=5")
                    .check_pivot(rule(), &args(&["idx"]), &df)
                    .is_ok()
            );
            let v = with("pivot_columns=4")
                .check_pivot(rule(), &args(&["idx"]), &df)
                .unwrap_err();
            assert_eq!(v.requested, "at least 5");
        }

        #[test]
        fn the_cell_budget_counts_rows_times_columns() {
            let df = long(4, 4);
            assert_eq!(pivoted(&df, &["idx"]), (4, 5));
            assert!(
                with("pivot_cells=20")
                    .check_pivot(rule(), &args(&["idx"]), &df)
                    .is_ok()
            );
            let v = with("pivot_cells=19")
                .check_pivot(rule(), &args(&["idx"]), &df)
                .unwrap_err();
            assert_eq!(
                (
                    v.tool.as_str(),
                    v.parameter.as_str(),
                    v.requested.as_str(),
                    v.limit.as_str(),
                    v.override_key
                ),
                ("munge_pivot", "index", "at least 20", "19", "pivot_cells")
            );
            assert_eq!(
                v.message,
                "`munge_pivot` would make at least 4 rows (one per distinct combination of the \
                 `index` columns idx) by 5 columns (the `on` column `key` has 4 distinct \
                 values), at least 20 cells, which exceeds this server's limit of 19 cells \
                 (pivot_cells). Pivot fewer rows or fewer `on` values, for example after \
                 munge_filter or munge_sample."
            );
            // Without a column budget the cell budget still counts the columns.
            let v = with("pivot_columns=none,pivot_cells=19")
                .check_pivot(rule(), &args(&["idx"]), &df)
                .unwrap_err();
            assert_eq!(v.override_key, "pivot_cells");
            assert_eq!(v.requested, "at least 20");
            // Too many `on` values for the cell budget alone stop the count of the `on` values:
            // the dataset has a row, so the wide table at least one.
            let wide = long(1, 12);
            assert_eq!(pivoted(&wide, &["idx"]), (1, 13));
            let v = with("pivot_columns=none,pivot_cells=12")
                .check_pivot(rule(), &args(&["idx"]), &wide)
                .unwrap_err();
            assert_eq!(
                (v.parameter.as_str(), v.requested.as_str()),
                ("index", "at least 13")
            );
            assert!(
                v.message.contains(
                    "by at least 13 columns (the `on` column `key` has at least \
                     12 distinct values)"
                ),
                "{}",
                v.message
            );
            assert!(
                with("pivot_columns=none,pivot_cells=13")
                    .check_pivot(rule(), &args(&["idx"]), &wide)
                    .is_ok()
            );
        }

        #[test]
        fn a_dataset_taller_than_the_cell_budget_is_refused_before_counting() {
            // Each row of the long table fills one cell of the wide table.
            let df = long(4, 4);
            assert_eq!(df.height(), 16);
            let v = with("pivot_cells=15")
                .check_pivot(rule(), &args(&["idx"]), &df)
                .unwrap_err();
            assert_eq!(
                (
                    v.parameter.as_str(),
                    v.requested.as_str(),
                    v.limit.as_str(),
                    v.override_key
                ),
                ("dataset", "at least 16", "15", "pivot_cells")
            );
            assert_eq!(
                v.message,
                "The dataset has 16 rows, and `munge_pivot` fills one cell of the wide table \
                 from each row (one row for each pair of `index` values and `on` value), so it \
                 would make at least 16 cells, which exceeds this server's limit of 15 cells \
                 (pivot_cells). Pivot fewer rows, for example after munge_filter or \
                 munge_sample; where a pair has several rows, aggregate them first with \
                 munge_group_by."
            );
            // A long table with several rows for a pair would make fewer cells than rows, but
            // the pivot refuses it.
            let repeated = df!(
                "idx" => [0i64; 16],
                "key" => ["k"; 16],
                "val" => (0..16).map(|i| i as f64 * 0.3 + 0.1).collect::<Vec<_>>()
            )
            .unwrap();
            assert!(pivot(&Dataset::new(repeated.clone()), &["idx"], "key", "val").is_err());
            let v = with("pivot_cells=15")
                .check_pivot(rule(), &args(&["idx"]), &repeated)
                .unwrap_err();
            assert_eq!(v.parameter, "dataset");
            assert!(
                with("pivot_cells=16")
                    .check_pivot(rule(), &args(&["idx"]), &repeated)
                    .is_ok()
            );
        }

        #[test]
        fn several_index_columns_are_counted_by_their_combinations() {
            // `name` is determined by `idx`: 10 combinations, where the product of the
            // distinct counts is 100 and the height 20. The pivot has 10 x 4 = 40 cells.
            let df = long(10, 2);
            assert_eq!(pivoted(&df, &["idx", "name"]), (10, 4));
            let index = args(&["idx", "name"]);
            assert!(
                with("pivot_cells=40")
                    .check_pivot(rule(), &index, &df)
                    .is_ok()
            );
            let v = with("pivot_cells=39")
                .check_pivot(rule(), &index, &df)
                .unwrap_err();
            assert_eq!(v.requested, "at least 40");
            assert!(
                v.message.contains(
                    "at least 10 rows (one per distinct combination of the `index` columns idx, \
                     name)"
                ),
                "{}",
                v.message
            );
            // Independent index columns: every combination is a row.
            let df = df!(
                "idx" => [0i64, 0, 1, 1, 2, 2],
                "name" => ["a", "b", "a", "b", "a", "b"],
                "key" => ["x", "y", "y", "x", "x", "y"],
                "val" => [0.4, 1.3, 2.1, 0.7, 1.9, 1.2]
            )
            .unwrap();
            assert_eq!(pivoted(&df, &["idx", "name"]), (6, 4));
            assert!(
                with("pivot_cells=24")
                    .check_pivot(rule(), &index, &df)
                    .is_ok()
            );
            assert_eq!(
                with("pivot_cells=23")
                    .check_pivot(rule(), &index, &df)
                    .unwrap_err()
                    .requested,
                "at least 24"
            );
        }

        #[test]
        fn requests_of_another_shape_are_left_to_serde_and_the_core() {
            let df = long(3, 4);
            let tight = with("pivot_columns=1,pivot_cells=1");
            assert!(tight.check_pivot(rule(), &args(&["idx"]), &df).is_err());
            for a in [
                json!({"index": "idx", "on": "key", "values": "val"}),
                json!({"index": [1], "on": "key", "values": "val"}),
                json!({"index": [], "on": "key", "values": "val"}),
                json!({"index": ["idx"], "on": 3, "values": "val"}),
                json!({"index": ["idx"], "on": "key", "values": 1}),
                json!({"index": ["idx"], "on": "key", "values": ["val"]}),
                json!({"index": ["idx"], "on": "key"}),
                // Columns the frame lacks are the core's error.
                json!({"index": ["nope"], "on": "key", "values": "val"}),
                json!({"index": ["idx"], "on": "nope", "values": "val"}),
                json!({"index": ["idx"], "on": "key", "values": "nope"}),
            ] {
                assert!(tight.check_pivot(rule(), &a, &df).is_ok(), "{a}");
            }
            // The off profile, and a profile without pivot budgets, check nothing.
            let off = [Limits::off(), with("pivot_columns=none,pivot_cells=none")];
            for limits in off {
                assert!(limits.check_pivot(rule(), &args(&["idx"]), &df).is_ok());
            }
        }

        #[test]
        fn refusals_echo_cut_column_names() {
            let long_name = "é".repeat(100);
            let df = df!(
                "idx" => [0i64, 0, 1],
                long_name.as_str() => ["a", "b", "c"],
                "val" => [0.4, 1.3, 2.1]
            )
            .unwrap();
            let a = json!({"index": ["idx"], "on": long_name, "values": "val"});
            let v = with("pivot_columns=2")
                .check_pivot(rule(), &a, &df)
                .unwrap_err();
            assert!(v.message.contains(&"é".repeat(64)), "{}", v.message);
            assert!(!v.message.contains(&"é".repeat(65)), "{}", v.message);
        }

        /// No column type this build can hold fails to be counted, so the refusal check_pivot
        /// gives when counting fails is tested on the helper that builds it.
        #[test]
        fn the_uncountable_refusal_names_the_parameter_and_the_limit() {
            let e = PolarsError::ComputeError("n_unique is not supported for this type".into());
            let v = uncountable("munge_pivot", "on", "key", ("pivot_columns", 10), &e);
            assert_eq!(
                (
                    v.parameter.as_str(),
                    v.limit.as_str(),
                    v.override_key,
                    v.requested.as_str()
                ),
                ("on", "10", "pivot_columns", "an uncountable column")
            );
            assert_eq!(
                v.message,
                "`munge_pivot` cannot count the distinct values of `on` (key), which this server \
                 needs to check its limit of 10 (pivot_columns) before pivoting. Cast the column \
                 to strings first, for example with munge_cast."
            );
        }
    }
}
