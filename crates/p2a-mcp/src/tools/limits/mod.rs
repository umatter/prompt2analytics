//! Request limits: the catalogue of cost-bearing request fields and the budgets that bound them.
//!
//! # Profiles
//!
//! [`Limits::off`] is the default for local use (stdio, CLI-style HTTP, the embedded desktop
//! server). It has no family cap, no row cap, no minimum tolerance and no `Option` budget, so it
//! checks nothing: every request that deserialises is accepted. [`Limits::hosted`] is the profile
//! of the public no-login deployment. Its values are those of the Tier 0 design (section 4), and
//! every hosted cap is at least 1.5 times the default of the tool fields it bounds, so default
//! requests are never refused. [`Limits::with_overrides`] adjusts either profile from
//! `key=value` pairs (`P2A_LIMIT_OVERRIDES`); [`Limits::known_keys`] lists the valid keys.
//!
//! # Classifying a new request field
//!
//! Every integer field, every array field and every float field whose name contains `tol` or
//! `eps` of every registered tool must be matched by a rule. Add a [`ParamRule`] to
//! `PARAM_RULES` in `rules.rs`: a generic `"*"` rule when the field name means the same thing in
//! every tool, a tool-specific rule otherwise, or `Rule::Exempt` with the reason the field costs
//! nothing. The schema test in `rules.rs` (built with the `http` feature) fails and names each
//! unclassified `(tool, path)`.
//!
//! # Installation
//!
//! This module only defines and checks limits. T0-WP06 installs the configured profile at startup
//! with [`install`] and enforces it before every tool call; handlers read named budgets through
//! [`current`]. Until something is installed, [`current`] returns the off profile.

mod rules;

/// A family of cost parameters that share one cap (for example every bootstrap replication count).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Family {
    /// Bootstrap and permutation replications.
    Replications,
    /// Monte Carlo simulation sizes.
    SimulationSize,
    /// Trees of a forest or committee members.
    Trees,
    /// Boosting rounds and additive terms.
    BoostingIterations,
    /// MCMC or simulation draws, burn-in included.
    McmcDraws,
    /// Cross-validation folds.
    Folds,
    /// Iteration limits of optimisers.
    Iterations,
    /// Random restarts.
    Restarts,
    /// Grid sizes (density grids, lambda paths, MTE grids).
    GridPoints,
    /// Histogram and binning bins.
    Bins,
    /// Lags, lag orders and seasonal periods.
    Lags,
    /// Forecast and response horizons.
    Horizon,
    /// Polynomial and ARMA model orders.
    ModelOrder,
    /// Integer kernel bandwidths (HAC, Driscoll-Kraay, GMM, sphet).
    Bandwidth,
    /// Smoother span and window widths.
    SpanWidth,
    /// Components, factors, ranks and clusters.
    Components,
    /// Nearest-neighbour counts.
    Neighbors,
    /// Rows a tool generates or samples.
    GeneratedRows,
    /// Image width or height in pixels.
    ImageDimension,
    /// Lists of names (columns, covariates, ...) and other short lists.
    ListLength,
    /// Numeric arrays passed by the caller.
    ArrayLength,
    /// Byte length of caller strings that are stored, repeated or compiled.
    StringBytes,
    /// Recursion depth of tree builders.
    TreeDepth,
}

/// Number of families; the length of [`Family::ALL`].
const FAMILY_COUNT: usize = 23;

// `Family::ALL` lists every family once, in declaration order, so `family as usize` indexes it.
const _: () = {
    assert!(Family::ALL.len() == FAMILY_COUNT);
    let mut i = 0;
    while i < FAMILY_COUNT {
        assert!(Family::ALL[i] as usize == i);
        i += 1;
    }
};

impl Family {
    /// Every family, in declaration order.
    pub const ALL: &'static [Family] = &[
        Family::Replications,
        Family::SimulationSize,
        Family::Trees,
        Family::BoostingIterations,
        Family::McmcDraws,
        Family::Folds,
        Family::Iterations,
        Family::Restarts,
        Family::GridPoints,
        Family::Bins,
        Family::Lags,
        Family::Horizon,
        Family::ModelOrder,
        Family::Bandwidth,
        Family::SpanWidth,
        Family::Components,
        Family::Neighbors,
        Family::GeneratedRows,
        Family::ImageDimension,
        Family::ListLength,
        Family::ArrayLength,
        Family::StringBytes,
        Family::TreeDepth,
    ];

    /// The family's override key, also shown in refusals (for example `replications`).
    pub fn key(self) -> &'static str {
        match self {
            Family::Replications => "replications",
            Family::SimulationSize => "simulation_size",
            Family::Trees => "trees",
            Family::BoostingIterations => "boosting_iterations",
            Family::McmcDraws => "mcmc_draws",
            Family::Folds => "folds",
            Family::Iterations => "iterations",
            Family::Restarts => "restarts",
            Family::GridPoints => "grid_points",
            Family::Bins => "bins",
            Family::Lags => "lags",
            Family::Horizon => "horizon",
            Family::ModelOrder => "model_order",
            Family::Bandwidth => "bandwidth",
            Family::SpanWidth => "span_width",
            Family::Components => "components",
            Family::Neighbors => "neighbors",
            Family::GeneratedRows => "generated_rows",
            Family::ImageDimension => "image_dimension",
            Family::ListLength => "list_length",
            Family::ArrayLength => "array_length",
            Family::StringBytes => "string_bytes",
            Family::TreeDepth => "tree_depth",
        }
    }

    /// The family's cap in the hosted profile.
    fn hosted_cap(self) -> u64 {
        match self {
            Family::Replications => 10_000,
            Family::SimulationSize => 100_000,
            Family::Trees => 5_000,
            Family::BoostingIterations => 10_000,
            Family::McmcDraws => 20_000,
            Family::Folds => 20,
            Family::Iterations => 100_000,
            Family::Restarts => 100,
            Family::GridPoints => 10_000,
            Family::Bins => 1_000,
            Family::Lags => 1_000,
            Family::Horizon => 5_000,
            Family::ModelOrder => 20,
            Family::Bandwidth => 1_000,
            Family::SpanWidth => 1_001,
            Family::Components => 500,
            Family::Neighbors => 100,
            Family::GeneratedRows => 1_000_000,
            Family::ImageDimension => 4_000,
            Family::ListLength => 500,
            Family::ArrayLength => 100_000,
            Family::StringBytes => 4_096,
            Family::TreeDepth => 100,
        }
    }

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
/// share one `(tool, path)`, and all of them apply.
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
    /// Size on disk of a file `load_dataset` reads, checked before parsing.
    pub load_file_bytes: Option<u64>,
    /// Bytes of a tool's text output.
    pub response_bytes: Option<u64>,
    /// Bytes of a session's data directory.
    pub session_disk_bytes: Option<u64>,
    /// Total links of a spatial neighbour structure.
    pub spatial_links: Option<u64>,
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
        /// Keys of the row caps, the minimum tolerance and the named budgets, in declaration order.
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
        max_groups, join_output_rows, onehot_levels, str_split_parts, str_pad_width,
        generated_cells, generated_bytes, export_bytes, load_file_bytes, response_bytes,
        session_disk_bytes, spatial_links, db_result_rows, db_result_bytes, db_timeout_secs,
        db_memory_mb, db_threads, conversations_per_session, messages_per_conversation,
        session_db_bytes, llm_streams_per_session,
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
            str_split_parts: None,
            str_pad_width: None,
            generated_cells: None,
            generated_bytes: None,
            export_bytes: None,
            load_file_bytes: None,
            response_bytes: None,
            session_disk_bytes: None,
            spatial_links: None,
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

    /// The profile of the public no-login deployment (Tier 0 design, section 4).
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
            str_split_parts: Some(100),
            str_pad_width: Some(10_000),
            generated_cells: Some(10_000_000),
            generated_bytes: Some(200 * MIB),
            export_bytes: Some(64 * MIB),
            load_file_bytes: Some(64 * MIB),
            response_bytes: Some(16 * MIB),
            session_disk_bytes: Some(64 * MIB),
            spatial_links: Some(2_000_000),
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

#[cfg(test)]
mod tests {
    use super::*;

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
            "str_split_parts",
            "str_pad_width",
            "generated_cells",
            "generated_bytes",
            "export_bytes",
            "load_file_bytes",
            "response_bytes",
            "session_disk_bytes",
            "spatial_links",
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
        assert_eq!(keys.len(), Family::ALL.len() + 35);
        for k in keys {
            let v = if BOOLS.contains(&k) { "true" } else { "1" };
            Limits::off()
                .with_overrides(&[format!("{k}={v}")])
                .unwrap_or_else(|e| panic!("{k}: {e}"));
        }
    }
}
