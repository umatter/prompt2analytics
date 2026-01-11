//! Econometrics module with pure Rust implementations.
//!
//! Provides panel data estimators, instrumental variables, causal inference,
//! discrete choice models, treatment effects, and multivariate time series.

mod panel;
mod iv;
mod did;
mod discrete;
mod timeseries;
mod hdfe;
mod treatment;
mod mediation;

pub use panel::{PanelResult, HausmanResult, run_fixed_effects, run_random_effects, run_hausman_test};
pub use iv::{IVResult, run_iv2sls, FirstStageDiagnostics, run_first_stage_diagnostics};
pub use did::{DiDResult, run_did};
pub use discrete::{DiscreteResult, run_logit, run_probit};
pub use timeseries::{VarResult, VarmaResult, VecmResult, VarIrfResult, run_var, run_varma, run_vecm, run_var_irf};
pub use hdfe::{HdfeResult, HdfeConfig, FactorInfo, run_hdfe};
pub use treatment::{
    Estimand, DRMethod, IpwConfig, DoublyRobustConfig,
    PropensityScoreSummary, IpwResult, DoublyRobustResult,
    run_ipw_treatment, run_doubly_robust,
};
pub use mediation::{MediationConfig, MediationResult, run_mediation_analysis};
