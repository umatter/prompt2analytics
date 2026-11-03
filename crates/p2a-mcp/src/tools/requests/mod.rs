//! Request types for MCP tools, organized by category.
//!
//! Each module contains the request structs for a category of tools.
//! Request structs define the input parameters for each MCP tool.
//!
//! # Migration Status
//!
//! Request types are being migrated incrementally from server.rs.
//! Modules marked with "(migrated)" have their types defined here.
//! Other modules are placeholders awaiting migration.
//!
//! - `data`: (migrated) Data management requests
//! - `utils`: (migrated) Utility requests (seed, reports)
//! - `database`: (migrated) Database requests (SQLite, DuckDB)
//! - `viz`: (migrated) Visualization requests
//! - `ml`: (migrated) Machine learning requests
//! - `stats`: (migrated) Statistics requests (loglin, contrasts, weighted stats, robust stats, splines)
//! - `regression`: (migrated) Regression requests (OLS, diagnostics, HAC, bootstrap, quantile, NLS, LOESS, GLS)
//! - `panel`: (migrated) Panel data requests (FE, RE, Hausman, PVCM, PMG, GMM, panel GLS, unit root, HDFE)
//! - `discrete`: (migrated) Discrete choice requests (logit, probit, multinomial, mlogit, mixed logit, ordered, count models, FEGLM)
//! - `causal`: (migrated) Causal inference requests (IV, DiD, treatment effects, matching, TMLE, synth, RD, mediation)
//! - `timeseries`: (migrated) Time series requests (ACF, VAR, ARIMA, GARCH, decomposition, filtering)
//! - `spatial`: (migrated) Spatial econometrics requests (neighbors, Moran, SAR, SEM, spatial probit, panel)
//! - `munging`: (migrated) Data munging requests (filter, select, join, concat, string ops, transformations)

// Migrated modules
pub mod causal;
pub mod cleaning;
pub mod data;
pub mod database;
pub mod discrete;
pub mod hypothesis;
pub mod ml;
pub mod munging;
pub mod panel;
pub mod regression;
pub mod search;
pub mod spatial;
pub mod stats;
pub mod survival;
pub mod timeseries;
pub mod utils;
pub mod viz;

// Re-export migrated types
pub use causal::*;
pub use cleaning::*;
pub use data::*;
pub use database::*;
pub use discrete::*;
pub use hypothesis::*;
pub use ml::*;
pub use munging::*;
pub use panel::*;
pub use regression::*;
pub use search::*;
pub use spatial::*;
pub use stats::*;
pub use survival::*;
pub use timeseries::*;
pub use utils::*;
pub use viz::*;

#[cfg(test)]
mod tests {
    const SOURCES: &[(&str, &str)] = &[
        ("causal.rs", include_str!("causal.rs")),
        ("cleaning.rs", include_str!("cleaning.rs")),
        ("data.rs", include_str!("data.rs")),
        ("database.rs", include_str!("database.rs")),
        ("discrete.rs", include_str!("discrete.rs")),
        ("hypothesis.rs", include_str!("hypothesis.rs")),
        ("ml.rs", include_str!("ml.rs")),
        ("munging.rs", include_str!("munging.rs")),
        ("panel.rs", include_str!("panel.rs")),
        ("regression.rs", include_str!("regression.rs")),
        ("search.rs", include_str!("search.rs")),
        ("spatial.rs", include_str!("spatial.rs")),
        ("stats.rs", include_str!("stats.rs")),
        ("survival.rs", include_str!("survival.rs")),
        ("timeseries.rs", include_str!("timeseries.rs")),
        ("utils.rs", include_str!("utils.rs")),
        ("viz.rs", include_str!("viz.rs")),
        (
            "../handlers/search.rs",
            include_str!("../handlers/search.rs"),
        ),
    ];

    /// Every struct deriving Deserialize carries #[serde(deny_unknown_fields)] among the
    /// attribute lines directly above it (TOOL-13). A multi-line attribute there reads as
    /// missing: keep both one-line.
    #[test]
    fn test_every_request_struct_denies_unknown_fields() {
        let mut missing = Vec::new();
        let mut structs = 0;
        for (file, src) in SOURCES {
            let lines: Vec<&str> = src.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                let t = line.trim_start();
                if !(t.starts_with("pub struct ") || t.starts_with("struct ")) {
                    continue;
                }
                let (mut deser, mut deny, mut j) = (false, false, i);
                while j > 0
                    && (lines[j - 1].trim_start().starts_with("#[")
                        || lines[j - 1].trim_start().starts_with("//"))
                {
                    let a = lines[j - 1].trim_start();
                    deser |= a.starts_with("#[derive(") && a.contains("Deserialize");
                    deny |= a.starts_with("#[serde(") && a.contains("deny_unknown_fields");
                    j -= 1;
                }
                if deser {
                    structs += 1;
                    if !deny {
                        missing.push(format!("{file}:{}: {t}", i + 1));
                    }
                }
            }
        }
        assert!(structs >= 270, "guard saw only {structs} request structs");
        assert!(
            missing.is_empty(),
            "request structs without #[serde(deny_unknown_fields)]:\n{}",
            missing.join("\n")
        );
    }
}
