//! Causal inference commands

use clap::Subcommand;
use p2a_core::{
    AttEstimationMethod,
    BandwidthMethod,
    ComparisonGroup,
    DRMethod,
    DistanceMethod,
    DoublyRobustConfig,
    Estimand,
    GModel,
    GsynthConfig,
    GsynthEstimator,
    GsynthForce,
    IpwConfig,
    KernelType,
    MatchMethod,
    MediationConfig,
    PredictorSpec,
    QModel,
    RdConfig,
    SEMethod,
    StaggeredDidConfig,
    StdRegConfig,
    StdRegEstimand,
    StdRegModel,
    SynthConfig,
    TmleConfig,
    VOptimization,
    VceType,
    // Diagnostics
    diagnostics::{
        WarningSeverity, did_diagnostics, ipw_diagnostics, iv_diagnostics, matching_diagnostics,
        rd_diagnostics, staggered_did_diagnostics,
    },
    run_did,
    run_doubly_robust,
    run_fuzzy_rd,
    // Generalized Synthetic Control
    run_gsynth,
    run_ipw_treatment,
    run_iv2sls,
    run_mediation_analysis,
    run_rd,
    // Staggered DiD (Callaway-Sant'Anna)
    run_staggered_did,
    // Regression Standardization / G-computation
    run_stdreg,
    run_synthetic_control,
    // TMLE
    tmle,
};

use crate::output::{
    OutputFormat, add_notes, add_result_notes, core_message, format_regression_results,
    missing_json, parse_core_option, print_error, print_missing_note, print_notes,
    print_result_notes, result_notes,
};
use crate::session::SessionManager;
use p2a_core::diagnostics::IdentificationReport;
use p2a_core::econometrics::matching::{MOrder, MatchMethodKind, MatchOptions, match_it_with};

/// Identification warnings of severity Caution and above, as
/// `[{"code", "severity", "title", "message"}]`.
fn identification_json(report: &IdentificationReport) -> serde_json::Value {
    report
        .warnings
        .iter()
        .filter(|w| w.severity >= WarningSeverity::Caution)
        .map(|w| {
            serde_json::json!({
                "code": w.code,
                "severity": w.severity.to_string(),
                "title": w.title,
                "message": w.message,
            })
        })
        .collect::<Vec<_>>()
        .into()
}

/// "Identification warnings:" and one "  [SEVERITY] title: message" line per warning of
/// severity Caution and above; nothing when there is none.
fn print_identification_warnings(report: &IdentificationReport) {
    let warnings: Vec<_> = report
        .warnings
        .iter()
        .filter(|w| w.severity >= WarningSeverity::Caution)
        .collect();
    if warnings.is_empty() {
        return;
    }
    println!("\nIdentification warnings:");
    for w in warnings {
        let label = match w.severity {
            WarningSeverity::Critical => "CRITICAL",
            WarningSeverity::Warning => "WARNING",
            WarningSeverity::Caution => "CAUTION",
            WarningSeverity::Info => "INFO",
        };
        println!("  [{}] {}: {}", label, w.title, w.message);
    }
}

#[derive(Subcommand)]
pub enum CausalCommands {
    /// Two-Stage Least Squares (Instrumental Variables)
    #[command(after_help = "\
EXAMPLES:
    # IV regression with education instrumented by distance to college
    p2a --session s.json causal iv mydata -y wage --exog experience --endog education --instruments distance_college
")]
    Iv {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Exogenous variable columns
        #[arg(long, num_args = 1..)]
        exog: Vec<String>,

        /// Endogenous variable columns
        #[arg(long, num_args = 1..)]
        endog: Vec<String>,

        /// Instrument columns
        #[arg(long, num_args = 1..)]
        instruments: Vec<String>,
    },

    /// Difference-in-Differences
    #[command(after_help = "\
EXAMPLES:
    # Classic 2x2 DiD
    p2a --session s.json causal did mydata -y outcome --treat treatment --post post_period

    # DiD with controls
    p2a --session s.json causal did mydata -y outcome --treat treatment --post post_period -x age income
")]
    Did {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(long)]
        treat: String,

        /// Post-treatment period indicator column
        #[arg(long)]
        post: String,

        /// Control variable columns (optional)
        #[arg(short = 'x', long, num_args = 0..)]
        controls: Vec<String>,

        /// Cluster standard errors on this column (CR1, as sandwich::vcovCL); HC1 without it
        #[arg(long)]
        cluster: Option<String>,
    },

    /// Synthetic Control Method (Abadie et al.)
    #[command(after_help = "\
EXAMPLES:
    # Synthetic control for California tobacco program
    p2a --session s.json causal synth mydata -y cigsale --unit state --time year --treated California \\
        --treatment-time 1989 -p income beer retprice

    # With placebo tests
    p2a --session s.json causal synth mydata -y gdp --unit country --time year --treated Spain \\
        --treatment-time 1986 -p population unemployment --placebos
")]
    Synth {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Unit identifier column (e.g., "state", "country")
        #[arg(long)]
        unit: String,

        /// Time period column (must be integer)
        #[arg(long)]
        time: String,

        /// Name/ID of the treated unit
        #[arg(long)]
        treated: String,

        /// Treatment time (first post-treatment period)
        #[arg(long)]
        treatment_time: i64,

        /// Predictor columns (will use pre-treatment mean for matching)
        #[arg(short = 'p', long, num_args = 1..)]
        predictors: Vec<String>,

        /// V matrix: datadriven (nested optimisation, default; also data_driven, optimal,
        /// nested) or equal (also uniform; equal weights on the standardised predictors);
        /// custom V is MCP-only
        #[arg(long, default_value = "datadriven", value_parser = parse_v_method)]
        v_method: VOptimization,

        /// Run placebo tests for inference
        #[arg(long)]
        placebos: bool,
    },

    /// Inverse Probability Weighting treatment effect
    #[command(after_help = "\
EXAMPLES:
    # IPW for ATE with default trimming
    p2a --session s.json causal ipw mydata -y outcome -t treatment -x age income education

    # IPW for ATT with custom trimming
    p2a --session s.json causal ipw mydata -y outcome -t treatment -x age income --estimand att --trim 0.1
")]
    Ipw {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Estimand: ate (default) or att
        #[arg(long, default_value = "ate", value_parser = parse_core_option::<Estimand>)]
        estimand: Estimand,

        /// Trimming threshold for propensity scores, in [0, 0.5) (default: 0.05)
        #[arg(long, default_value = "0.05")]
        trim: f64,

        /// Standard errors: 0 (default) gives the M-estimation SE with the propensity score
        /// estimated, as WeightIt; N >= 2 a bootstrap that resamples every row and refits the
        /// propensity model and the trimming in each replicate
        #[arg(long, default_value = "0")]
        bootstrap: usize,

        /// Seed of the bootstrap draws (default: 42)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Doubly robust treatment effect (AIPW)
    #[command(after_help = "\
EXAMPLES:
    # AIPW (doubly robust)
    p2a --session s.json causal doubly-robust mydata -y outcome -t treatment -x age income --method aipw

    # Regression-only method for ATT
    p2a --session s.json causal doubly-robust mydata -y outcome -t treatment -x age income --method regression --estimand att
")]
    DoublyRobust {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Method: aipw (default; also dr, doubly_robust), ipw or regression (also reg, ra,
        /// outcome_regression)
        #[arg(long, default_value = "aipw", value_parser = parse_core_option::<DRMethod>)]
        method: DRMethod,

        /// Estimand: ate (default) or att
        #[arg(long, default_value = "ate", value_parser = parse_core_option::<Estimand>)]
        estimand: Estimand,

        /// Trimming threshold for propensity scores, in [0, 0.5) (default: 0.05)
        #[arg(long, default_value = "0.05")]
        trim: f64,

        /// Bootstrap replications for the SEs of --method ipw and regression (default: 999),
        /// each refitting the propensity and outcome models; 0 gives analytic SEs
        /// (M-estimation for ipw, the stacked-equation sandwich for regression) and 1 is an
        /// error. aipw uses its analytic influence-function SE and ignores it
        #[arg(long, default_value = "999")]
        bootstrap: usize,

        /// Seed of the bootstrap draws (default: 42)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Causal mediation analysis
    #[command(after_help = "\
EXAMPLES:
    # Mediation analysis (treatment -> mediator -> outcome)
    p2a --session s.json causal mediation mydata -y outcome -t treatment -m mediator -x age income
")]
    Mediation {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Mediator variable column
        #[arg(short = 'm', long)]
        mediator: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 0..)]
        covariates: Vec<String>,

        /// Number of bootstrap replications (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,

        /// Seed of the bootstrap draws (default: 42)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Regression Discontinuity (sharp design)
    #[command(after_help = "\
EXAMPLES:
    # Sharp RD with MSE-optimal bandwidth
    p2a --session s.json causal rd mydata -y test_score -r running_var -c 50

    # With local quadratic polynomial
    p2a --session s.json causal rd mydata -y test_score -r running_var -c 50 -p 2

    # Given bandwidths (h for the estimate, b for the bias correction) and HC1 variance
    p2a --session s.json causal rd mydata -y score -r x -c 0 --h 0.3 --b 0.3 --vce hc1
")]
    Rd {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Running variable column
        #[arg(short = 'r', long)]
        running: String,

        /// Cutoff value
        #[arg(short = 'c', long)]
        cutoff: f64,

        /// Polynomial order (default: 1 = local linear)
        #[arg(short = 'p', long, default_value = "1")]
        poly_order: usize,

        /// Kernel, as rdrobust: triangular (default), epanechnikov or uniform (aliases tri,
        /// epa, uni)
        #[arg(long, default_value = "triangular", value_parser = parse_kernel_arg)]
        kernel: KernelType,

        /// Bandwidth selector, as rdrobust::rdbwselect: mserd (default), msetwo, msesum,
        /// msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1 or cercomb2; not used for h when
        /// --h is given
        #[arg(long, default_value = "mserd", value_parser = parse_bwselect_arg)]
        bwselect: BandwidthMethod,

        /// Main bandwidth h (both sides); default: p2a's MSE-type selector
        #[arg(long = "h", visible_alias = "bandwidth")]
        h: Option<f64>,

        /// Bias bandwidth b; default: h
        #[arg(long = "b", visible_alias = "bias-bandwidth")]
        b: Option<f64>,

        /// Variance estimator: nn (default, as rdrobust), hc0, hc1, hc2 or hc3
        #[arg(long, default_value = "nn", value_parser = parse_vce_arg)]
        vce: VceType,
    },

    /// Fuzzy Regression Discontinuity
    #[command(after_help = "\
EXAMPLES:
    # Fuzzy RD (imperfect compliance at cutoff)
    p2a --session s.json causal fuzzy-rd mydata -y outcome -r score -t enrolled -c 60

    # One given bandwidth for both stages, bias bandwidth 0.5, HC1 variance
    p2a --session s.json causal fuzzy-rd mydata -y outcome -r score -t enrolled -c 60 --h 0.3 --b 0.5 --vce hc1

    # CER-optimal bandwidths, separate left and right
    p2a --session s.json causal fuzzy-rd mydata -y y -r x -t d -c 0 --bwselect certwo
")]
    FuzzyRd {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Running variable column
        #[arg(short = 'r', long)]
        running: String,

        /// Treatment indicator column
        #[arg(short = 't', long)]
        treatment: String,

        /// Cutoff value
        #[arg(short = 'c', long)]
        cutoff: f64,

        /// Polynomial order (default: 1)
        #[arg(short = 'p', long, default_value = "1")]
        poly_order: usize,

        /// Kernel, as rdrobust: triangular (default), epanechnikov or uniform (aliases tri,
        /// epa, uni)
        #[arg(long, default_value = "triangular", value_parser = parse_kernel_arg)]
        kernel: KernelType,

        /// Bandwidth selector for both stages, as rdrobust::rdbwselect(fuzzy = ..): mserd
        /// (default), msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1 or
        /// cercomb2; not used for h when --h is given
        #[arg(long, default_value = "mserd", value_parser = parse_bwselect_arg)]
        bwselect: BandwidthMethod,

        /// Main bandwidth h (both sides); default: p2a's MSE-type selector
        #[arg(long = "h", visible_alias = "bandwidth")]
        h: Option<f64>,

        /// Bias bandwidth b; default: h
        #[arg(long = "b", visible_alias = "bias-bandwidth")]
        b: Option<f64>,

        /// Variance estimator: nn (default, as rdrobust), hc0, hc1, hc2 or hc3
        #[arg(long, default_value = "nn", value_parser = parse_vce_arg)]
        vce: VceType,
    },

    /// Staggered Difference-in-Differences (Callaway-Sant'Anna)
    #[command(after_help = "\
EXAMPLES:
    # Staggered DiD with never-treated comparison
    p2a --session s.json causal staggered-did mydata -y outcome --treatment-time first_treat \\
        --time year --unit state

    # Doubly robust (DRDID) with covariates
    p2a --session s.json causal staggered-did mydata -y outcome --treatment-time first_treat \\
        --time year --unit state -x population gdp --method dr
")]
    StaggeredDid {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment timing column: the period a unit is first treated (0 or negative: never).
        #[arg(long)]
        treatment_time: String,

        /// Time period column.
        #[arg(long)]
        time: String,

        /// Unit identifier column.
        #[arg(long)]
        unit: String,

        /// Covariate columns for conditional parallel trends (optional).
        #[arg(short = 'x', long, num_args = 0..)]
        covariates: Vec<String>,

        /// Comparison group, as did's control_group.
        ///
        /// never_treated (default) or not_yet_treated (also nevertreated, never, notyettreated,
        /// notyet, nyt).
        #[arg(long, default_value = "never_treated", value_parser = parse_core_option::<ComparisonGroup>)]
        comparison: ComparisonGroup,

        /// Estimation method, as did's est_method (DRDID).
        ///
        /// or = outcome_regression (default; also reg), ipw = inverse probability weighting
        /// (also std_ipw), dr = doubly_robust (also aipw, drdid).
        #[arg(long, default_value = "or", value_parser = parse_core_option::<AttEstimationMethod>)]
        method: AttEstimationMethod,

        /// Periods before treatment in which units may anticipate it (did's anticipation).
        #[arg(long, default_value_t = 0)]
        anticipation: usize,

        /// Unit-bootstrap replications for SEs (at least 50), or 0 for analytic SEs.
        ///
        /// The bootstrap is a multiplier bootstrap of the influence functions with covariates;
        /// 0 gives analytic influence-function SEs (DRDID's with covariates).
        #[arg(long, default_value = "999")]
        bootstrap: usize,

        /// Seed of the bootstrap draws (default: 42)
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Targeted Maximum Likelihood Estimation (TMLE)
    #[command(after_help = "\
EXAMPLES:
    # TMLE for ATE
    p2a --session s.json causal tmle mydata -y outcome -t treatment -x age income education

    # With custom propensity score truncation
    p2a --session s.json causal tmle mydata -y outcome -t treatment -x age income --truncate-lower 0.05 --truncate-upper 0.95
")]
    Tmle {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Outcome (Q) model: logistic (default; also logit, binomial, binary) or linear (also
        /// gaussian, continuous, ols); logistic: Y outside [0, 1] is rescaled to [0, 1] first,
        /// as the tmle package does
        #[arg(long, default_value = "logistic", value_parser = parse_core_option::<QModel>)]
        qmodel: QModel,

        /// Propensity score truncation lower bound (default: 0.01)
        #[arg(long, default_value = "0.01")]
        truncate_lower: f64,

        /// Propensity score truncation upper bound (default: 0.99)
        #[arg(long, default_value = "0.99")]
        truncate_upper: f64,
    },

    /// Generalized Synthetic Control (gsynth)
    #[command(after_help = "\
EXAMPLES:
    # Interactive FE with auto-selected factors
    p2a --session s.json causal gsynth mydata -y gdp -d treatment --unit country --time year

    # Matrix completion estimator with bootstrap SEs
    p2a --session s.json causal gsynth mydata -y gdp -d treatment --unit country --time year \\
        --estimator mc --bootstrap-se
")]
    Gsynth {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (0/1)
        #[arg(short = 'd', long)]
        treatment: String,

        /// Unit identifier column
        #[arg(long)]
        unit: String,

        /// Time period column
        #[arg(long)]
        time: String,

        /// Covariate columns (optional)
        #[arg(short = 'x', long, num_args = 0..)]
        covariates: Vec<String>,

        /// Number of latent factors (0 = auto-select via cross-validation)
        #[arg(long, default_value = "0")]
        n_factors: usize,

        /// Fixed effects (gsynth force): none, unit (also individual), time or two_way (default;
        /// also twoway, two-way)
        #[arg(long, default_value = "two_way", value_parser = parse_core_option::<GsynthForce>)]
        force: GsynthForce,

        /// Estimator: ife (interactive fixed effects, default; also gsynth) or mc (matrix
        /// completion; also matrix_completion)
        #[arg(long, default_value = "ife", value_parser = parse_core_option::<GsynthEstimator>)]
        estimator: GsynthEstimator,

        /// Compute bootstrap standard errors
        #[arg(long)]
        bootstrap_se: bool,

        /// Number of bootstrap replications (default: 200)
        #[arg(long, default_value = "200")]
        n_bootstrap: usize,
    },

    /// Propensity Score Matching (MatchIt)
    #[command(after_help = "\
EXAMPLES:
    # Nearest neighbor 1:1 matching with caliper
    p2a --session s.json causal matching mydata -t treatment -x age income education --caliper 0.25

    # Coarsened exact matching
    p2a --session s.json causal matching mydata -t treatment -x age income education --method cem

    # Full matching
    p2a --session s.json causal matching mydata -t treatment -x age income --method full
")]
    Matching {
        /// Dataset name
        dataset: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns for matching
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Matching method, as MatchIt: nearest (default; also nn), cem (coarsened exact, Sturges
        /// bins), full (optimal full matching; also optimal) or subclass (also subclassification)
        #[arg(long, default_value = "nearest", value_parser = parse_core_option::<MatchMethodKind>)]
        method: MatchMethodKind,

        /// Distance: logit (default; also glm, logistic, ps), probit, mahalanobis or euclidean
        #[arg(long, default_value = "logit", value_parser = parse_core_option::<DistanceMethod>)]
        distance: DistanceMethod,

        /// Order in which treated units are matched (MatchIt m.order): largest (also
        /// descending), smallest (also ascending), data or random; default largest for
        /// logit/probit, data otherwise
        #[arg(long, value_parser = parse_core_option::<MOrder>)]
        m_order: Option<MOrder>,

        /// Matching ratio for nearest neighbor (default: 1)
        #[arg(long, default_value = "1")]
        ratio: usize,

        /// Caliper for nearest neighbor (in SD units of propensity score; optional)
        #[arg(long)]
        caliper: Option<f64>,

        /// Sample with replacement (for nearest neighbor)
        #[arg(long)]
        replace: bool,

        /// Number of subclasses (for subclass method, default: 5)
        #[arg(long, default_value = "5")]
        n_subclasses: usize,
    },

    /// Regression Standardization / G-computation (stdReg)
    #[command(after_help = "\
EXAMPLES:
    # G-computation for ATE with bootstrap SEs
    p2a --session s.json causal stdreg mydata -y outcome -t treatment -x age income

    # Logistic outcome model for ATT
    p2a --session s.json causal stdreg mydata -y binary_outcome -t treatment -x age income \\
        --model logistic --estimand att
")]
    Stdreg {
        /// Dataset name
        dataset: String,

        /// Outcome variable column
        #[arg(short = 'y', long)]
        outcome: String,

        /// Treatment indicator column (binary 0/1)
        #[arg(short = 't', long)]
        treatment: String,

        /// Covariate columns
        #[arg(short = 'x', long, num_args = 1..)]
        covariates: Vec<String>,

        /// Outcome model: linear (default; also ols, gaussian), logistic (also logit, binomial)
        /// or poisson (also count, log_linear)
        #[arg(long, default_value = "linear", value_parser = parse_core_option::<StdRegModel>)]
        model: StdRegModel,

        /// Estimand: ate (default), att, atc or levels (also potential_outcomes)
        #[arg(long, default_value = "ate", value_parser = parse_core_option::<StdRegEstimand>)]
        estimand: StdRegEstimand,

        /// SE method: bootstrap (default; also boot), delta or sandwich (also robust)
        #[arg(long, default_value = "bootstrap", value_parser = parse_core_option::<SEMethod>)]
        se_method: SEMethod,

        /// Number of bootstrap replications (default: 999)
        #[arg(long, default_value = "999")]
        bootstrap: usize,

        /// Include treatment-covariate interactions in outcome model
        #[arg(long)]
        interactions: bool,

        /// Seed of the bootstrap draws (default: 42)
        #[arg(long)]
        seed: Option<u64>,
    },
}

pub fn execute(
    cmd: &CausalCommands,
    format: &OutputFormat,
    _quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        CausalCommands::Iv {
            dataset,
            dep_var,
            exog,
            endog,
            instruments,
        } => execute_iv(dataset, dep_var, exog, endog, instruments, format, session),
        CausalCommands::Did {
            dataset,
            outcome,
            treat,
            post,
            controls,
            cluster,
        } => execute_did(
            dataset,
            outcome,
            treat,
            post,
            controls,
            cluster.as_deref(),
            format,
            session,
        ),
        CausalCommands::Synth {
            dataset,
            outcome,
            unit,
            time,
            treated,
            treatment_time,
            predictors,
            v_method,
            placebos,
        } => execute_synth(
            dataset,
            outcome,
            unit,
            time,
            treated,
            *treatment_time,
            predictors,
            v_method.clone(),
            *placebos,
            format,
            session,
        ),
        CausalCommands::Ipw {
            dataset,
            outcome,
            treatment,
            covariates,
            estimand,
            trim,
            bootstrap,
            seed,
        } => execute_ipw(
            dataset, outcome, treatment, covariates, *estimand, *trim, *bootstrap, *seed, format,
            session,
        ),
        CausalCommands::DoublyRobust {
            dataset,
            outcome,
            treatment,
            covariates,
            method,
            estimand,
            trim,
            bootstrap,
            seed,
        } => execute_dr(
            dataset, outcome, treatment, covariates, *method, *estimand, *trim, *bootstrap, *seed,
            format, session,
        ),
        CausalCommands::Mediation {
            dataset,
            outcome,
            treatment,
            mediator,
            covariates,
            bootstrap,
            seed,
        } => execute_mediation(
            dataset, outcome, treatment, mediator, covariates, *bootstrap, *seed, format, session,
        ),
        CausalCommands::Rd {
            dataset,
            outcome,
            running,
            cutoff,
            poly_order,
            kernel,
            bwselect,
            h,
            b,
            vce,
        } => execute_rd(
            dataset,
            outcome,
            running,
            *cutoff,
            *poly_order,
            RdOptions {
                kernel: *kernel,
                bwselect: *bwselect,
                vce: *vce,
                h: *h,
                b: *b,
            },
            format,
            session,
        ),
        CausalCommands::FuzzyRd {
            dataset,
            outcome,
            running,
            treatment,
            cutoff,
            poly_order,
            kernel,
            bwselect,
            h,
            b,
            vce,
        } => execute_fuzzy_rd(
            dataset,
            outcome,
            running,
            treatment,
            *cutoff,
            *poly_order,
            RdOptions {
                kernel: *kernel,
                bwselect: *bwselect,
                vce: *vce,
                h: *h,
                b: *b,
            },
            format,
            session,
        ),
        CausalCommands::StaggeredDid {
            dataset,
            outcome,
            treatment_time,
            time,
            unit,
            covariates,
            comparison,
            method,
            anticipation,
            bootstrap,
            seed,
        } => execute_staggered_did(
            dataset,
            outcome,
            treatment_time,
            time,
            unit,
            covariates,
            *comparison,
            *method,
            *anticipation,
            *bootstrap,
            *seed,
            format,
            session,
        ),
        CausalCommands::Tmle {
            dataset,
            outcome,
            treatment,
            covariates,
            qmodel,
            truncate_lower,
            truncate_upper,
        } => execute_tmle(
            dataset,
            outcome,
            treatment,
            covariates,
            *qmodel,
            *truncate_lower,
            *truncate_upper,
            format,
            session,
        ),
        CausalCommands::Gsynth {
            dataset,
            outcome,
            treatment,
            unit,
            time,
            covariates,
            n_factors,
            force,
            estimator,
            bootstrap_se,
            n_bootstrap,
        } => execute_gsynth(
            dataset,
            outcome,
            treatment,
            unit,
            time,
            covariates,
            *n_factors,
            *force,
            *estimator,
            *bootstrap_se,
            *n_bootstrap,
            format,
            session,
        ),
        CausalCommands::Matching {
            dataset,
            treatment,
            covariates,
            method,
            distance,
            m_order,
            ratio,
            caliper,
            replace,
            n_subclasses,
        } => execute_matching(
            dataset,
            treatment,
            covariates,
            *method,
            *distance,
            *m_order,
            *ratio,
            caliper.as_ref().copied(),
            *replace,
            *n_subclasses,
            format,
            session,
        ),
        CausalCommands::Stdreg {
            dataset,
            outcome,
            treatment,
            covariates,
            model,
            estimand,
            se_method,
            bootstrap,
            interactions,
            seed,
        } => execute_stdreg(
            dataset,
            outcome,
            treatment,
            covariates,
            *model,
            *estimand,
            *se_method,
            *bootstrap,
            *interactions,
            *seed,
            format,
            session,
        ),
    }
}

fn execute_iv(
    dataset_name: &str,
    dep_var: &str,
    exog: &[String],
    endog: &[String],
    instruments: &[String],
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let exog_refs: Vec<&str> = exog.iter().map(|s| s.as_str()).collect();
            let endog_refs: Vec<&str> = endog.iter().map(|s| s.as_str()).collect();
            let inst_refs: Vec<&str> = instruments.iter().map(|s| s.as_str()).collect();

            match run_iv2sls(ds, dep_var, &exog_refs, &endog_refs, &inst_refs, true) {
                Ok(result) => {
                    // IVResult uses fields, not methods
                    let coeffs = &result.coefficients;
                    let ses = &result.std_errors;
                    let p_vals = &result.p_values;
                    let t_vals = &result.t_stats;

                    let var_names = &result.variables;

                    let coef_table: Vec<(String, f64, f64, f64, f64)> = var_names
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.clone(), coeffs[i], ses[i], t_vals[i], p_vals[i]))
                        .collect();

                    let output = format_regression_results(
                        "2SLS / IV Regression",
                        &coef_table,
                        result.r_squared,
                        result.r_squared, // IVResult has no adj_r_squared
                        result.n_obs,
                        format,
                    );

                    // Identification diagnostics; a failure is reported as a warning, not
                    // shown as an empty (warning-free) list.
                    let diagnostics = iv_diagnostics(ds, &result);
                    let mut warnings = result.warnings.clone();
                    if let Err(e) = &diagnostics {
                        warnings.push(format!(
                            "identification diagnostics could not be computed: {e}"
                        ));
                    }

                    // Print first-stage diagnostics
                    match format {
                        OutputFormat::Json => {
                            let mut json: serde_json::Value = serde_json::from_str(&output)?;
                            json["first_stage"] = serde_json::json!({
                                "f_type": result.first_stage_f_type,
                                "f_stats": result.first_stage_f_stats,
                                "df": result.first_stage_df,
                                "p_values": result.first_stage_p_values,
                                "strong_instruments": result.strong_instruments,
                            });
                            json["diagnostics"] = match &diagnostics {
                                Ok(r) => identification_json(r),
                                Err(_) => serde_json::Value::Null,
                            };
                            add_notes(&mut json, None, &warnings);
                            if let Some(m) = missing_json(&result.missing) {
                                json["missing"] = m;
                            }
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("{}", output);
                            if !result.first_stage_f_stats.is_empty() {
                                println!(
                                    "\nFirst-stage diagnostics ({}):",
                                    result.first_stage_f_type
                                );
                                for (i, f_stat) in result.first_stage_f_stats.iter().enumerate() {
                                    let endog_name = if i < endog.len() {
                                        &endog[i]
                                    } else {
                                        "Unknown"
                                    };
                                    match result.first_stage_df.get(i) {
                                        Some((df1, df2)) => println!(
                                            "  {} F-statistic: {:.4} on {} and {} DF",
                                            endog_name, f_stat, df1, df2
                                        ),
                                        None => {
                                            println!("  {} F-statistic: {:.4}", endog_name, f_stat)
                                        }
                                    }
                                }
                                if result.strong_instruments {
                                    println!("  Instruments are strong (all F > 10)");
                                } else {
                                    println!("  Warning: Some instruments may be weak (F < 10)");
                                }
                            }

                            print_notes(None, &warnings, format);

                            if let Ok(diag_report) = &diagnostics {
                                print_identification_warnings(diag_report);
                            }
                            print_missing_note(&result.missing, format);
                        }
                    }
                }
                Err(e) => {
                    print_error(&format!("2SLS failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_did(
    dataset_name: &str,
    outcome: &str,
    treat: &str,
    post: &str,
    controls: &[String],
    cluster: Option<&str>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let control_refs: Vec<&str> = controls.iter().map(|s| s.as_str()).collect();
            let controls_opt = if control_refs.is_empty() {
                None
            } else {
                Some(control_refs.as_slice())
            };

            match run_did(ds, outcome, treat, post, controls_opt, cluster) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Difference-in-Differences",
                            "att": result.att,
                            "std_error": result.std_error,
                            "t_stat": result.t_stat,
                            "p_value": result.p_value,
                            "n_obs": result.n_obs,
                            "treated_pre_mean": result.treated_pre_mean,
                            "treated_post_mean": result.treated_post_mean,
                            "control_pre_mean": result.control_pre_mean,
                            "control_post_mean": result.control_post_mean,
                            "df": result.df,
                            "cluster_var": result.cluster_var,
                            "n_clusters": result.n_clusters,
                            "aliased": result.aliased,
                            "diagnostics": identification_json(&did_diagnostics(&result)),
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nDifference-in-Differences Results");
                        println!("{}", "=".repeat(50));
                        println!(
                            "\nAverage Treatment Effect on Treated (ATT): {:.6}",
                            result.att
                        );
                        println!("Standard Error: {:.6}", result.std_error);
                        println!("t-statistic: {:.4}", result.t_stat);
                        println!("p-value: {:.4}", result.p_value);
                        println!("\nObservations: {}", result.n_obs);
                        if let (Some(cvar), Some(nc)) = (&result.cluster_var, result.n_clusters) {
                            println!("Clustered SEs: {cvar} ({nc} clusters)");
                        }
                        println!("\nGroup means:");
                        println!("  Pre-treatment (treated): {:.4}", result.treated_pre_mean);
                        println!(
                            "  Post-treatment (treated): {:.4}",
                            result.treated_post_mean
                        );
                        println!("  Pre-treatment (control): {:.4}", result.control_pre_mean);
                        println!(
                            "  Post-treatment (control): {:.4}",
                            result.control_post_mean
                        );
                        if !result.aliased.is_empty() {
                            println!(
                                "\nAliased (dropped, coefficient NA): {}",
                                result.aliased.join(", ")
                            );
                        }
                        let (converged, warnings) = result_notes(&result);
                        if converged.is_some() || !warnings.is_empty() {
                            println!();
                        }
                        print_result_notes(&result, format);

                        // Run identification diagnostics
                        let diag_report = did_diagnostics(&result);
                        print_identification_warnings(&diag_report);
                    }
                },
                Err(e) => {
                    print_error(&format!("DiD failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_synth(
    dataset_name: &str,
    outcome: &str,
    unit_col: &str,
    time_col: &str,
    treated_unit: &str,
    treatment_time: i64,
    predictors: &[String],
    v_opt: VOptimization,
    run_placebos: bool,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error(
                "No session active. Use --session <file> to enable dataset storage.",
                format,
            );
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            // Build predictor specs (using pre-treatment mean for all)
            let predictor_specs: Vec<PredictorSpec> = predictors
                .iter()
                .map(|col| PredictorSpec::new(col))
                .collect();

            let mut config = SynthConfig::default();
            config.treated_unit = treated_unit.to_string();
            config.treatment_time = treatment_time;
            config.optimization_window = None;
            config.v_method = v_opt;
            config.tolerance = 1e-6;
            config.max_iter = 1000;
            config.run_placebos = run_placebos;
            config.weight_threshold = 0.001;

            match run_synthetic_control(ds, outcome, unit_col, time_col, &predictor_specs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Synthetic Control",
                            "treated_unit": result.treated_unit,
                            "treatment_time": result.treatment_time,
                            "unit_weights": result.unit_weights,
                            "all_unit_weights": result.all_unit_weights,
                            "predictor_loss": result.predictor_loss,
                            "predictor_balance": result.predictor_balance.iter().map(|b| {
                                serde_json::json!({
                                    "predictor": b.predictor,
                                    "treated_value": b.treated_value,
                                    "synthetic_value": b.synthetic_value,
                                    "percent_diff": b.percent_diff,
                                })
                            }).collect::<Vec<_>>(),
                            "pre_treatment_mspe": result.pre_treatment_mspe,
                            "pre_treatment_rmspe": result.pre_treatment_rmspe,
                            "treatment_effects": result.treatment_effects.iter().map(|e| {
                                serde_json::json!({
                                    "time": e.time,
                                    "actual": e.actual,
                                    "synthetic": e.synthetic,
                                    "effect": e.effect,
                                })
                            }).collect::<Vec<_>>(),
                            "average_effect": result.average_effect,
                            "cumulative_effect": result.cumulative_effect,
                            "placebo_results": result.placebo_results.as_ref().map(|p| {
                                serde_json::json!({
                                    "treated_rank": p.treated_rank,
                                    "p_value": p.p_value,
                                    "n_units": p.n_units,
                                })
                            }),
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("{}", result);
                    }
                },
                Err(e) => {
                    print_error(&format!("Synthetic Control failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_ipw(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    est: Estimand,
    trim: f64,
    bootstrap: usize,
    seed: Option<u64>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let mut config = IpwConfig::default();
            config.estimand = est;
            config.trim = trim;
            config.bootstrap = bootstrap;
            config.normalized = true;
            config.seed = Some(seed.unwrap_or(42));

            match run_ipw_treatment(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "IPW Treatment Effect",
                            "estimand": format!("{:?}", result.estimand),
                            "effect": result.effect,
                            "std_error": result.std_error,
                            "se_method": result.se_method,
                            "se_mest": result.se_mest,
                            "se_naive": result.se_naive,
                            "se_eif": result.se_eif,
                            "ci_lower": result.ci_lower,
                            "ci_upper": result.ci_upper,
                            "t_stat": result.t_stat,
                            "p_value": result.p_value,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_trimmed": result.n_trimmed,
                            "diagnostics": identification_json(&ipw_diagnostics(&result)),
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nIPW Treatment Effect ({:?})", result.estimand);
                        println!("{}", "=".repeat(50));
                        println!("Effect: {:.6}", result.effect);
                        println!("Std Error: {:.6}", result.std_error);
                        println!("SE method: {}", result.se_method);
                        println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                        println!(
                            "t-stat: {:.4}, p-value: {:.4}",
                            result.t_stat, result.p_value
                        );
                        println!(
                            "N: {} (treated: {}, control: {}; trimmed: {})",
                            result.n_obs, result.n_treated, result.n_control, result.n_trimmed
                        );
                        print_result_notes(&result, format);

                        // Run identification diagnostics
                        let diag_report = ipw_diagnostics(&result);
                        print_identification_warnings(&diag_report);
                    }
                },
                Err(e) => print_error(&format!("IPW failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_dr(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    dr_method: DRMethod,
    est: Estimand,
    trim: f64,
    bootstrap: usize,
    seed: Option<u64>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let mut config = DoublyRobustConfig::default();
            config.method = dr_method;
            config.estimand = est;
            config.trim = trim;
            config.bootstrap = bootstrap;
            config.seed = Some(seed.unwrap_or(42));

            match run_doubly_robust(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": format!("{:?}", result.method),
                            "estimand": format!("{:?}", result.estimand),
                            "effect": result.effect,
                            "std_error": result.std_error,
                            "se_method": result.se_method,
                            "ci_lower": result.ci_lower,
                            "ci_upper": result.ci_upper,
                            "t_stat": result.t_stat,
                            "p_value": result.p_value,
                            "n_obs": result.n_obs,
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!(
                            "\nDoubly Robust Treatment Effect ({:?}, {:?})",
                            result.method, result.estimand
                        );
                        println!("{}", "=".repeat(50));
                        println!("Effect: {:.6}", result.effect);
                        println!("Std Error: {:.6}", result.std_error);
                        println!("SE method: {}", result.se_method);
                        println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                        println!(
                            "t-stat: {:.4}, p-value: {:.4}",
                            result.t_stat, result.p_value
                        );
                        println!("N: {}", result.n_obs);
                        print_result_notes(&result, format);
                    }
                },
                Err(e) => print_error(&format!("Doubly robust failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_mediation(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    mediator: &str,
    covariates: &[String],
    bootstrap: usize,
    seed: Option<u64>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let mut config = MediationConfig::default();
            config.bootstrap = bootstrap;
            config.trim = 0.05;
            config.seed = Some(seed.unwrap_or(42));

            match run_mediation_analysis(ds, outcome, treatment, mediator, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Causal Mediation Analysis",
                            "total_effect": result.total_effect,
                            "direct_effect": result.direct_effect,
                            "indirect_effect": result.indirect_effect,
                            "proportion_mediated": result.proportion_mediated,
                            "se_total": result.se_total,
                            "se_direct": result.se_direct,
                            "se_indirect": result.se_indirect,
                            "p_total": result.p_total,
                            "p_direct": result.p_direct,
                            "p_indirect": result.p_indirect,
                            "direct_effect_treated": result.direct_effect_treated,
                            "indirect_effect_control": result.indirect_effect_control,
                            "se_direct_treated": result.se_direct_treated,
                            "se_indirect_control": result.se_indirect_control,
                            "p_direct_treated": result.p_direct_treated,
                            "p_indirect_control": result.p_indirect_control,
                            "n_obs": result.n_obs,
                        });
                        add_result_notes(&mut json, &result);
                        if let Some(m) = missing_json(&result.missing) {
                            json["missing"] = m;
                        }
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nCausal Mediation Analysis");
                        println!("{}", "=".repeat(50));
                        println!(
                            "Total Effect (ATE): {:.6} (SE: {:.6}, p: {:.4})",
                            result.total_effect, result.se_total, result.p_total
                        );
                        println!(
                            "Direct Effect (NDE): {:.6} (SE: {:.6}, p: {:.4})",
                            result.direct_effect, result.se_direct, result.p_direct
                        );
                        println!(
                            "Indirect Effect (NIE): {:.6} (SE: {:.6}, p: {:.4})",
                            result.indirect_effect, result.se_indirect, result.p_indirect
                        );
                        println!(
                            "Direct Effect θ(1): {:.6} (SE: {:.6}, p: {:.4})",
                            result.direct_effect_treated,
                            result.se_direct_treated,
                            result.p_direct_treated
                        );
                        println!(
                            "Indirect Effect δ(0): {:.6} (SE: {:.6}, p: {:.4})",
                            result.indirect_effect_control,
                            result.se_indirect_control,
                            result.p_indirect_control
                        );
                        println!(
                            "Proportion Mediated: {:.2}%",
                            result.proportion_mediated * 100.0
                        );
                        println!("N: {}", result.n_obs);
                        print_result_notes(&result, format);
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("Mediation analysis failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// The kernel, selector, variance and bandwidth flags of `causal rd` and `causal fuzzy-rd`.
#[derive(Clone, Copy)]
struct RdOptions {
    kernel: KernelType,
    bwselect: BandwidthMethod,
    vce: VceType,
    h: Option<f64>,
    b: Option<f64>,
}

impl RdOptions {
    /// `RdConfig::default()` with polynomial order `p` and these options.
    fn config(&self, p: usize) -> RdConfig {
        let mut config = RdConfig::default();
        config.p = p;
        config.h = self.h;
        config.b = self.b;
        config.kernel = self.kernel;
        config.bwselect = self.bwselect;
        config.vce = self.vce;
        config
    }
}

/// `--v-method` of `causal synth`: `VOptimization::from_options(Some(s), None)`, or the core
/// message.
fn parse_v_method(s: &str) -> Result<VOptimization, String> {
    VOptimization::from_options(Some(s), None).map_err(core_message)
}

/// `--vce` as a `VceType`, or the core message.
fn parse_vce_arg(vce: &str) -> Result<VceType, String> {
    VceType::parse(vce).map_err(core_message)
}

/// `--kernel` as a `KernelType`, or the core message.
fn parse_kernel_arg(kernel: &str) -> Result<KernelType, String> {
    KernelType::parse(kernel).map_err(core_message)
}

/// `--bwselect` as a `BandwidthMethod`, or the core message.
fn parse_bwselect_arg(bwselect: &str) -> Result<BandwidthMethod, String> {
    BandwidthMethod::parse(bwselect).map_err(core_message)
}

fn execute_rd(
    dataset_name: &str,
    outcome: &str,
    running: &str,
    cutoff: f64,
    poly_order: usize,
    options: RdOptions,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let config = options.config(poly_order);

            match run_rd(ds, outcome, running, cutoff, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Sharp RD",
                            "outcome": result.outcome,
                            "running_var": result.running_var,
                            "cutoff": result.cutoff,
                            "tau_robust": result.tau_robust,
                            "se_robust": result.se_robust,
                            "p_robust": result.p_robust,
                            "ci_robust": result.ci_robust,
                            "n_left": result.n_left,
                            "n_right": result.n_right,
                            "h_left": result.h_left,
                            "h_right": result.h_right,
                            "tau_conventional": result.tau_conventional,
                            "se_conventional": result.se_conventional,
                            "p_conventional": result.p_conventional,
                            "ci_conventional": result.ci_conventional,
                            "tau_bc": result.tau_bc,
                            "bias": result.bias,
                            "n_eff_left": result.n_eff_left,
                            "n_eff_right": result.n_eff_right,
                            "n_b_left": result.n_b_left,
                            "n_b_right": result.n_b_right,
                            "b_left": result.b_left,
                            "b_right": result.b_right,
                            "kernel": result.kernel.to_string(),
                            "bwselect": result.bwselect.name(),
                            "bandwidth_source": result.bandwidth_label(),
                            "vce": result.vce.to_string(),
                            "p": result.p,
                            "q": result.q,
                            "level": result.level,
                            "diagnostics": identification_json(&rd_diagnostics(&result)),
                        });
                        // the notes leave out the missing-value note, which has its own key
                        add_result_notes(&mut json, &result);
                        if let Some(m) = missing_json(&result.missing) {
                            json["missing"] = m;
                        }
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nRegression Discontinuity Estimation");
                        println!("{}", "=".repeat(50));
                        println!(
                            "Outcome: {}, Running var: {}",
                            result.outcome, result.running_var
                        );
                        println!("Cutoff: {:.4}", result.cutoff);
                        println!("\nRobust RD Estimate: {:.6}", result.tau_robust);
                        println!("Std Error: {:.6}", result.se_robust);
                        println!("p-value: {:.4}", result.p_robust);
                        println!(
                            "95% CI: [{:.6}, {:.6}]",
                            result.ci_robust.0, result.ci_robust.1
                        );
                        println!();
                        println!("Conventional Estimate: {:.6}", result.tau_conventional);
                        println!("Conventional Std Error: {:.6}", result.se_conventional);
                        println!(
                            "Conventional 95% CI: [{:.6}, {:.6}]",
                            result.ci_conventional.0, result.ci_conventional.1
                        );
                        println!("\nN left: {}, N right: {}", result.n_left, result.n_right);
                        println!(
                            "Effective N (h): {} left, {} right",
                            result.n_eff_left, result.n_eff_right
                        );
                        println!(
                            "Effective N (b): {} left, {} right",
                            result.n_b_left, result.n_b_right
                        );
                        println!(
                            "Bandwidth h: left={:.4}, right={:.4}",
                            result.h_left, result.h_right
                        );
                        println!(
                            "Bandwidth b: left={:.4}, right={:.4}",
                            result.b_left, result.b_right
                        );
                        println!("Kernel: {}", result.kernel);
                        println!("Bandwidth source: {}", result.bandwidth_label());
                        println!("VCE: {}", result.vce);
                        print_result_notes(&result, format);

                        // Run identification diagnostics
                        print_identification_warnings(&rd_diagnostics(&result));
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("RD estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_fuzzy_rd(
    dataset_name: &str,
    outcome: &str,
    running: &str,
    treatment: &str,
    cutoff: f64,
    poly_order: usize,
    options: RdOptions,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let config = options.config(poly_order);

            match run_fuzzy_rd(ds, outcome, running, treatment, cutoff, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let rd = &result.outcome_rd;
                        let mut json = serde_json::json!({
                            "method": "Fuzzy RD",
                            "tau_fuzzy": result.tau_fuzzy,
                            "se_fuzzy": result.se_fuzzy,
                            "p_fuzzy": result.p_fuzzy,
                            "ci_fuzzy": result.ci_fuzzy,
                            "treatment": result.treatment,
                            "tau_fuzzy_bc": result.tau_fuzzy_bc,
                            "se_fuzzy_robust": result.se_fuzzy_robust,
                            "p_fuzzy_robust": result.p_fuzzy_robust,
                            "ci_fuzzy_robust": result.ci_fuzzy_robust,
                            "first_stage_tau": result.first_stage_tau,
                            "first_stage_se": result.first_stage.se_conventional,
                            "first_stage_f": result.first_stage_f,
                            "reduced_form_tau": rd.tau_conventional,
                            "h_left": rd.h_left,
                            "h_right": rd.h_right,
                            "b_left": rd.b_left,
                            "b_right": rd.b_right,
                            "n_eff_left": rd.n_eff_left,
                            "n_eff_right": rd.n_eff_right,
                            "kernel": rd.kernel.to_string(),
                            "bwselect": rd.bwselect.name(),
                            "bandwidth_source": rd.bandwidth_label(),
                            "vce": rd.vce.to_string(),
                            "level": result.level,
                            "diagnostics": identification_json(&rd_diagnostics(rd)),
                        });
                        // the notes leave out the missing-value note, which has its own key
                        add_result_notes(&mut json, &result);
                        if let Some(m) = missing_json(&result.missing) {
                            json["missing"] = m;
                        }
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        let rd = &result.outcome_rd;
                        println!("\nFuzzy Regression Discontinuity Estimation");
                        println!("{}", "=".repeat(50));
                        println!("Treatment: {}", result.treatment);
                        println!(
                            "\nFuzzy RD Estimate (LATE, conventional): {:.6}",
                            result.tau_fuzzy
                        );
                        println!("Std Error: {:.6}", result.se_fuzzy);
                        println!("p-value: {:.4}", result.p_fuzzy);
                        println!(
                            "95% CI: [{:.6}, {:.6}]",
                            result.ci_fuzzy.0, result.ci_fuzzy.1
                        );
                        println!();
                        println!("Robust LATE (bias-corrected): {:.6}", result.tau_fuzzy_bc);
                        println!("Robust Std Error: {:.6}", result.se_fuzzy_robust);
                        println!("Robust p-value: {:.4}", result.p_fuzzy_robust);
                        println!(
                            "Robust 95% CI: [{:.6}, {:.6}]",
                            result.ci_fuzzy_robust.0, result.ci_fuzzy_robust.1
                        );
                        println!(
                            "\nFirst stage: {:.6} (SE {:.6}), F = {:.2}",
                            result.first_stage_tau,
                            result.first_stage.se_conventional,
                            result.first_stage_f
                        );
                        println!(
                            "Bandwidth h: left={:.4}, right={:.4}; b: left={:.4}, right={:.4} \
                             (both stages)",
                            rd.h_left, rd.h_right, rd.b_left, rd.b_right
                        );
                        println!("Kernel: {}", rd.kernel);
                        println!("Bandwidth source: {}", rd.bandwidth_label());
                        println!("VCE: {}", rd.vce);
                        print_result_notes(&result, format);

                        // Run identification diagnostics on the outcome RD
                        print_identification_warnings(&rd_diagnostics(rd));
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("Fuzzy RD estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_staggered_did(
    dataset_name: &str,
    outcome: &str,
    treatment_time: &str,
    time_col: &str,
    unit_col: &str,
    covariates: &[String],
    comparison_group: ComparisonGroup,
    estimation_method: AttEstimationMethod,
    anticipation: usize,
    bootstrap: usize,
    seed: Option<u64>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();
            let covariates_opt = if cov_refs.is_empty() {
                None
            } else {
                Some(cov_refs.as_slice())
            };

            let mut config = StaggeredDidConfig::default();
            config.comparison_group = comparison_group;
            config.estimation_method = estimation_method;
            config.anticipation = anticipation;
            config.bootstrap = bootstrap;
            config.seed = Some(seed.unwrap_or(42));

            match run_staggered_did(
                ds,
                outcome,
                treatment_time,
                time_col,
                unit_col,
                covariates_opt,
                config,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Callaway-Sant'Anna Staggered DiD",
                            "comparison_group": format!("{}", result.config.comparison_group),
                            "estimation_method": format!("{}", result.config.estimation_method),
                            "overall_att": {
                                "att": result.overall_att.att,
                                "std_error": result.overall_att.std_error,
                                "ci_lower": result.overall_att.ci_lower,
                                "ci_upper": result.overall_att.ci_upper,
                                "p_value": result.overall_att.p_value,
                            },
                            "event_study": result.event_study.iter().map(|e| {
                                serde_json::json!({
                                    "relative_time": e.key,
                                    "att": e.att,
                                    "std_error": e.std_error,
                                    "ci_lower": e.ci_lower,
                                    "ci_upper": e.ci_upper,
                                    "p_value": e.p_value,
                                })
                            }).collect::<Vec<_>>(),
                            "cohorts": result.cohorts,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_never_treated": result.n_never_treated,
                            "pretrend_test": result.pretrend_test.as_ref().map(|pt| {
                                serde_json::json!({
                                    "chi2": pt.chi2,
                                    "df": pt.df,
                                    "p_value": pt.p_value,
                                    "f_stat": pt.f_stat,
                                    "df2": pt.df2,
                                    "method": pt.method,
                                })
                            }),
                            "covariates": result.covariates,
                            "anticipation": result.config.anticipation,
                            "diagnostics": identification_json(&staggered_did_diagnostics(&result)),
                        });
                        add_result_notes(&mut json, &result);
                        // The text output's Display already ends with the missing-value note.
                        if let Some(m) = missing_json(&result.missing) {
                            json["missing"] = m;
                        }
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("{}", result);

                        // Run identification diagnostics
                        let diag_report = staggered_did_diagnostics(&result);
                        print_identification_warnings(&diag_report);
                    }
                },
                Err(e) => print_error(&format!("Staggered DiD failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_tmle(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    q_model: QModel,
    truncate_lower: f64,
    truncate_upper: f64,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();

            let mut config = TmleConfig::default();
            config.q_model = q_model;
            config.g_model = GModel::Logistic;
            config.truncate_ps = (truncate_lower, truncate_upper);

            match tmle(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Targeted Maximum Likelihood Estimation (TMLE)",
                            "ate": result.ate,
                            "ate_se": result.ate_se,
                            "ci_lower": result.ate_ci_lower,
                            "ci_upper": result.ate_ci_upper,
                            "z_stat": result.z_stat,
                            "p_value": result.ate_p_value,
                            "significance": format!("{}", result.significance),
                            "fluctuation_coef": result.fluctuation_coef,
                            "fluctuation_coef_control": result.fluctuation_coef_control,
                            "targeting_iterations": result.targeting_iterations,
                            "outcome_range": result.outcome_range,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_truncated": result.n_truncated,
                            "q_model_converged": result.q_model_converged,
                            "g_model_converged": result.g_model_converged,
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nTargeted Maximum Likelihood Estimation (TMLE)");
                        println!("{}", "=".repeat(50));
                        println!("ATE: {:.6}", result.ate);
                        println!("Std Error: {:.6}", result.ate_se);
                        println!(
                            "95% CI: [{:.6}, {:.6}]",
                            result.ate_ci_lower, result.ate_ci_upper
                        );
                        println!(
                            "Z-stat: {:.4}, p-value: {:.4}{}",
                            result.z_stat,
                            result.ate_p_value,
                            result.significance.stars()
                        );
                        println!(
                            "\nFluctuation coefficients (epsilon): {:.6} (treated), {:.6} (control)",
                            result.fluctuation_coef, result.fluctuation_coef_control
                        );
                        println!(
                            "N: {} (treated: {}, control: {})",
                            result.n_obs, result.n_treated, result.n_control
                        );
                        if result.n_truncated > 0 {
                            println!("Propensity scores truncated: {}", result.n_truncated);
                        }
                        println!(
                            "Q-model converged: {}, G-model converged: {}",
                            result.q_model_converged, result.g_model_converged
                        );
                        print_result_notes(&result, format);
                    }
                },
                Err(e) => print_error(&format!("TMLE failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_gsynth(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    unit_col: &str,
    time_col: &str,
    covariates: &[String],
    n_factors: usize,
    gsynth_force: GsynthForce,
    gsynth_estimator: GsynthEstimator,
    bootstrap_se: bool,
    n_bootstrap: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();

            let mut config = GsynthConfig::default();
            config.n_factors = n_factors;
            config.cross_validate = n_factors == 0;
            config.force = gsynth_force;
            config.estimator = gsynth_estimator;
            config.bootstrap_se = bootstrap_se;
            config.n_bootstrap = n_bootstrap;

            match run_gsynth(
                ds, outcome, treatment, unit_col, time_col, &cov_refs, config,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Generalized Synthetic Control (gsynth)",
                            "att": result.att,
                            "att_se": result.att_se,
                            "att_ci": result.att_ci,
                            "p_value": result.p_value,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_pre_periods": result.n_pre_periods,
                            "n_post_periods": result.n_post_periods,
                            "n_factors": result.n_factors,
                            "estimator": result.estimator.to_string(),
                            "force": result.force.to_string(),
                            "beta": result.beta,
                            "covariate_names": result.covariate_names,
                            "cv_mspe": result.cv_mspe,
                            "cv_lambda": result.cv_lambda,
                            "lambda": result.lambda,
                            "sigma2": result.sigma2,
                            "mu": result.mu,
                            "dynamic_counts": result.dynamic_counts,
                            "dynamic_se": result.dynamic_se,
                            "bootstrap": result.bootstrap,
                            "dynamic_effects": result.dynamic_effects.iter().map(|(t, e)| {
                                serde_json::json!({ "time": t, "effect": e })
                            }).collect::<Vec<_>>(),
                            "unit_effects": result.unit_effects.iter().map(|ue| {
                                serde_json::json!({
                                    "unit": ue.unit,
                                    "treatment_time": ue.treatment_time,
                                    "att": ue.att,
                                    "se": ue.se,
                                })
                            }).collect::<Vec<_>>(),
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("{}", result);
                    }
                },
                Err(e) => print_error(&format!("gsynth failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_matching(
    dataset_name: &str,
    treatment: &str,
    covariates: &[String],
    method: MatchMethodKind,
    distance_method: DistanceMethod,
    m_order: Option<MOrder>,
    ratio: usize,
    caliper: Option<f64>,
    replace: bool,
    n_subclasses: usize,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();

            let match_method = match method {
                MatchMethodKind::Nearest => MatchMethod::NearestNeighbor {
                    ratio,
                    caliper,
                    replace,
                },
                // MatchIt's default bins (nclass.Sturges)
                MatchMethodKind::Cem => MatchMethod::CoarsenedExact {
                    cutpoints: None,
                    n_bins: None,
                },
                MatchMethodKind::Full => MatchMethod::Full {
                    min_ratio: 0.5,
                    max_ratio: 2.0,
                },
                MatchMethodKind::Subclass => MatchMethod::Subclass { n_subclasses },
            };
            let mut options = MatchOptions::default();
            options.m_order = m_order;

            match match_it_with(
                ds,
                treatment,
                &cov_refs,
                match_method,
                Some(distance_method),
                &options,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": format!("{}", result.method),
                            "distance": format!("{}", result.distance),
                            "m_order": result.m_order.map(|m| m.to_string()),
                            "ess_treated": result.ess_treated,
                            "ess_control": result.ess_control,
                            "total_distance": result.total_distance,
                            "caliper_abs": result.caliper_abs,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                            "n_matched_treated": result.n_matched_treated,
                            "n_matched_control": result.n_matched_control,
                            "n_discarded_treated": result.n_discarded_treated,
                            "n_discarded_control": result.n_discarded_control,
                            "effective_sample_size": result.effective_sample_size,
                            "caliper_used": result.caliper_used,
                            "balance_before": {
                                "mean_abs_std_diff": result.balance_before.mean_abs_std_diff,
                                "max_abs_std_diff": result.balance_before.max_abs_std_diff,
                                "n_imbalanced": result.balance_before.n_imbalanced,
                            },
                            "balance_after": {
                                "mean_abs_std_diff": result.balance_after.mean_abs_std_diff,
                                "max_abs_std_diff": result.balance_after.max_abs_std_diff,
                                "n_imbalanced": result.balance_after.n_imbalanced,
                            },
                            "diagnostics": identification_json(&matching_diagnostics(&result)),
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        // the Display prints both balance tables and the notes
                        println!("{}", result);

                        // Run identification diagnostics
                        let diag_report = matching_diagnostics(&result);
                        print_identification_warnings(&diag_report);
                    }
                },
                Err(e) => print_error(&format!("Matching failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn execute_stdreg(
    dataset_name: &str,
    outcome: &str,
    treatment: &str,
    covariates: &[String],
    model_type: StdRegModel,
    stdreg_estimand: StdRegEstimand,
    se_meth: SEMethod,
    bootstrap: usize,
    interactions: bool,
    seed: Option<u64>,
    format: &OutputFormat,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    let dataset = match session {
        Some(mgr) => mgr.get_dataset(dataset_name),
        None => {
            print_error("No session active. Use --session <file>.", format);
            return Ok(());
        }
    };

    match dataset {
        Some(ds) => {
            let cov_refs: Vec<&str> = covariates.iter().map(|s| s.as_str()).collect();

            let mut config = StdRegConfig::default();
            config.model_type = model_type;
            config.estimand = stdreg_estimand;
            config.se_method = se_meth;
            config.n_bootstrap = bootstrap;
            config.include_interactions = interactions;
            config.seed = Some(seed.unwrap_or(42));

            match run_stdreg(ds, outcome, treatment, &cov_refs, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Regression Standardization (G-computation)",
                            "estimand": format!("{}", result.estimand),
                            "model_type": format!("{}", result.model_type),
                            "se_method": format!("{}", result.se_method),
                            "ate": result.ate,
                            "se": result.se,
                            "ci_lower": result.ci_lower,
                            "ci_upper": result.ci_upper,
                            "z_stat": result.z_stat,
                            "p_value": result.p_value,
                            "significance": format!("{}", result.significance),
                            "ey1": result.ey1,
                            "ey0": result.ey0,
                            "ey1_se": result.ey1_se,
                            "ey0_se": result.ey0_se,
                            "risk_ratio": result.risk_ratio,
                            "risk_ratio_ci": result.risk_ratio_ci,
                            "odds_ratio": result.odds_ratio,
                            "odds_ratio_ci": result.odds_ratio_ci,
                            "nnt": result.nnt,
                            "n_obs": result.n_obs,
                            "n_treated": result.n_treated,
                            "n_control": result.n_control,
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nRegression Standardization / G-computation");
                        println!("{}", "=".repeat(50));
                        println!("Estimand: {}", result.estimand);
                        println!("Model: {}", result.model_type);
                        println!("SE Method: {}", result.se_method);
                        println!();
                        println!(
                            "Treatment Effect ({:?}): {:.6}",
                            result.estimand, result.ate
                        );
                        println!("Std Error: {:.6}", result.se);
                        println!("95% CI: [{:.6}, {:.6}]", result.ci_lower, result.ci_upper);
                        println!(
                            "Z-stat: {:.4}, p-value: {:.4}{}",
                            result.z_stat,
                            result.p_value,
                            result.significance.stars()
                        );
                        println!();
                        println!("Potential Outcomes:");
                        println!("  E[Y(1)] = {:.6} (SE: {:.6})", result.ey1, result.ey1_se);
                        println!("  E[Y(0)] = {:.6} (SE: {:.6})", result.ey0, result.ey0_se);
                        if let Some(rr) = result.risk_ratio {
                            println!("  Risk Ratio: {:.4}", rr);
                            if let Some((lo, hi)) = result.risk_ratio_ci {
                                println!("  RR 95% CI: [{:.4}, {:.4}]", lo, hi);
                            }
                        }
                        if let Some(or) = result.odds_ratio {
                            println!("  Odds Ratio: {:.4}", or);
                            if let Some((lo, hi)) = result.odds_ratio_ci {
                                println!("  OR 95% CI: [{:.4}, {:.4}]", lo, hi);
                            }
                        }
                        if let Some(nnt) = result.nnt {
                            println!("  Number Needed to Treat: {:.2}", nnt);
                        }
                        println!();
                        println!(
                            "N: {} (treated: {}, control: {})",
                            result.n_obs, result.n_treated, result.n_control
                        );
                        print_result_notes(&result, format);
                    }
                },
                Err(e) => print_error(&format!("Regression standardization failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// CLI-14 closure: the --help text of every typed option names each canonical value (no drift
    /// when an enum grows).
    #[test]
    fn test_option_help_lists_every_choice() {
        use clap::CommandFactory;
        use p2a_core::*;
        let root = crate::Cli::command();
        let help = |path: &[&str], id: &str| -> String {
            let mut cmd = &root;
            for p in path {
                cmd = cmd
                    .find_subcommand(p)
                    .unwrap_or_else(|| panic!("no subcommand {p}"))
            }
            cmd.get_arguments()
                .find(|a| a.get_id() == id)
                // --help text: the long help where the doc has a short first paragraph
                .and_then(|a| a.get_long_help().or(a.get_help()))
                .map(|h| h.to_string())
                .unwrap_or_else(|| panic!("{path:?} has no documented --{id}"))
        };
        let ts = super::super::timeseries::option_names();
        let rows: Vec<(Vec<&str>, &str, Vec<&str>)> = vec![
            (
                vec!["causal", "rd"],
                "kernel",
                KernelType::choices().names(),
            ),
            (
                vec!["causal", "rd"],
                "bwselect",
                BandwidthMethod::choices().names(),
            ),
            (vec!["causal", "rd"], "vce", VceType::choices().names()),
            (
                vec!["causal", "fuzzy-rd"],
                "kernel",
                KernelType::choices().names(),
            ),
            (
                vec!["causal", "fuzzy-rd"],
                "bwselect",
                BandwidthMethod::choices().names(),
            ),
            (
                vec!["causal", "staggered-did"],
                "comparison",
                ComparisonGroup::choices().names(),
            ),
            (
                vec!["causal", "staggered-did"],
                "method",
                AttEstimationMethod::choices().names(),
            ),
            (
                vec!["causal", "ipw"],
                "estimand",
                Estimand::choices().names(),
            ),
            (
                vec!["causal", "doubly-robust"],
                "method",
                DRMethod::choices().names(),
            ),
            (
                vec!["causal", "doubly-robust"],
                "estimand",
                Estimand::choices().names(),
            ),
            (vec!["causal", "tmle"], "qmodel", QModel::choices().names()),
            (
                vec!["causal", "stdreg"],
                "model",
                StdRegModel::choices().names(),
            ),
            (
                vec!["causal", "stdreg"],
                "estimand",
                StdRegEstimand::choices().names(),
            ),
            (
                vec!["causal", "stdreg"],
                "se_method",
                SEMethod::choices().names(),
            ),
            (
                vec!["causal", "gsynth"],
                "force",
                GsynthForce::choices().names(),
            ),
            (
                vec!["causal", "gsynth"],
                "estimator",
                GsynthEstimator::choices().names(),
            ),
            (
                vec!["causal", "matching"],
                "method",
                MatchMethodKind::choices().names(),
            ),
            (
                vec!["causal", "matching"],
                "distance",
                DistanceMethod::choices().names(),
            ),
            (
                vec!["causal", "matching"],
                "m_order",
                MOrder::choices().names(),
            ),
            (
                vec!["timeseries", "garch"],
                "distribution",
                forecasting::garch::GarchDistribution::choices().names(),
            ),
            (
                vec!["timeseries", "vecm"],
                "ecdet",
                econometrics::timeseries::vecm::VecmDeterministic::choices().names(),
            ),
            (
                vec!["spatial", "sar"],
                "weight_style",
                WeightStyle::choices().names(),
            ),
            (
                vec!["spatial", "sem"],
                "weight_style",
                WeightStyle::choices().names(),
            ),
            (
                vec!["spatial", "sac"],
                "weight_style",
                WeightStyle::choices().names(),
            ),
            (
                vec!["spatial", "moran"],
                "weight_style",
                WeightStyle::choices().names(),
            ),
            (
                vec!["spatial", "moran"],
                "alternative",
                MoranAlternative::choices().names(),
            ),
            (
                vec!["ml", "hierarchical"],
                "linkage",
                Linkage::choices().names(),
            ),
            (
                vec!["viz", "dendrogram"],
                "linkage",
                Linkage::choices().names(),
            ),
        ];
        for (path, id, names) in rows.into_iter().chain(ts) {
            let h = help(&path, id);
            for n in names {
                assert!(h.contains(n), "{path:?} --{id}: help {h:?} lacks {n:?}")
            }
        }
    }

    /// The R, CLI and MCP spellings of every newly typed option parse.
    #[test]
    fn test_options_accept_r_cli_and_mcp_spellings() {
        use clap::Parser;
        let ok =
            |a: &[&str]| crate::Cli::try_parse_from(a).unwrap_or_else(|e| panic!("{a:?}: {e}"));
        let rd = ["p2a", "causal", "rd", "d", "-y", "y", "-r", "x", "-c", "0"];
        for extra in [
            ["--kernel", "EPA"],
            ["--bwselect", "cer_two"],
            ["--vce", "HC1"],
        ] {
            ok(&[&rd[..], &extra[..]].concat());
        }
        let sd = [
            "p2a",
            "causal",
            "staggered-did",
            "d",
            "-y",
            "y",
            "--treatment-time",
            "g",
            "--time",
            "t",
            "--unit",
            "u",
        ];
        for extra in [
            ["--comparison", "notyet"],
            ["--comparison", "Not Yet Treated"],
            ["--method", "DR"],
            ["--method", "outcome_regression"],
        ] {
            ok(&[&sd[..], &extra[..]].concat());
        }
        match ok(&[&sd[..], &["--comparison", "not_yet_treated"][..]].concat()).command {
            crate::Commands::Causal(CausalCommands::StaggeredDid { comparison, .. }) => {
                assert_eq!(comparison, p2a_core::ComparisonGroup::NotYetTreated)
            }
            _ => panic!("not staggered-did"),
        }
        let more: [&[&str]; 16] = [
            &[
                "p2a", "causal", "tmle", "d", "-y", "y", "-t", "t", "-x", "a", "--qmodel",
                "gaussian",
            ],
            &[
                "p2a",
                "causal",
                "stdreg",
                "d",
                "-y",
                "y",
                "-t",
                "t",
                "-x",
                "a",
                "--model",
                "binomial",
                "--estimand",
                "potential_outcomes",
                "--se-method",
                "boot",
            ],
            &[
                "p2a",
                "causal",
                "synth",
                "d",
                "-y",
                "y",
                "--unit",
                "u",
                "--time",
                "t",
                "--treated",
                "T",
                "--treatment-time",
                "5",
                "-p",
                "a",
                "--v-method",
                "data_driven",
            ],
            &[
                "p2a",
                "causal",
                "gsynth",
                "d",
                "-y",
                "y",
                "-d",
                "t",
                "--unit",
                "u",
                "--time",
                "t",
                "--force",
                "twoWay",
                "--estimator",
                "matrix_completion",
            ],
            &[
                "p2a",
                "causal",
                "matching",
                "d",
                "-t",
                "t",
                "-x",
                "a",
                "--method",
                "nn",
                "--distance",
                "glm",
                "--m-order",
                "descending",
            ],
            &[
                "p2a",
                "ts",
                "changepoint",
                "d",
                "--col",
                "y",
                "--change-type",
                "both",
            ],
            &[
                "p2a",
                "ts",
                "holt-winters",
                "d",
                "--col",
                "y",
                "--period",
                "4",
                "--seasonal",
                "mult",
            ],
            &[
                "p2a",
                "ts",
                "ar",
                "d",
                "--col",
                "y",
                "--method",
                "Yule-Walker",
            ],
            &[
                "p2a",
                "ts",
                "decompose",
                "d",
                "--col",
                "y",
                "--period",
                "4",
                "--decompose-type",
                "Multiplicative",
            ],
            &[
                "p2a",
                "ts",
                "struct-ts",
                "d",
                "--col",
                "y",
                "--model-type",
                "bsm",
            ],
            &[
                "p2a",
                "ts",
                "garch",
                "d",
                "--col",
                "y",
                "--distribution",
                "student_t",
            ],
            &[
                "p2a", "ts", "vecm", "d", "--cols", "a", "b", "--ecdet", "constant",
            ],
            &[
                "p2a",
                "spatial",
                "sar",
                "d",
                "-y",
                "y",
                "-x",
                "a",
                "--coord-x",
                "cx",
                "--coord-y",
                "cy",
                "--weight-style",
                "rowstd",
            ],
            &[
                "p2a",
                "spatial",
                "moran",
                "d",
                "-y",
                "y",
                "--coord-x",
                "cx",
                "--coord-y",
                "cy",
                "--alternative",
                "two-sided",
            ],
            &[
                "p2a",
                "ml",
                "hierarchical",
                "d",
                "--cols",
                "a",
                "b",
                "--linkage",
                "ward",
            ],
            &[
                "p2a",
                "viz",
                "dendrogram",
                "d",
                "--cols",
                "a",
                "b",
                "--linkage",
                "wpgmc",
                "-f",
                "t.png",
            ],
        ];
        for a in more {
            ok(a);
        }
    }

    #[test]
    fn test_identification_json_keeps_caution_and_above() {
        use p2a_core::diagnostics::{IdentificationReport, IdentificationWarning};
        let w = |code: &str, s| IdentificationWarning {
            code: code.into(),
            severity: s,
            title: format!("t {code}"),
            message: format!("m {code}"),
            assumption: String::new(),
            remediation: vec![],
            diagnostics: None,
        };
        let mut rep = IdentificationReport::new("IV");
        rep.warnings = vec![
            w("A", WarningSeverity::Info),
            w("B", WarningSeverity::Caution),
            w("C", WarningSeverity::Critical),
        ];
        assert_eq!(
            identification_json(&rep),
            json!([
                {"code": "B", "severity": "Caution", "title": "t B", "message": "m B"},
                {"code": "C", "severity": "Critical", "title": "t C", "message": "m C"}])
        );
    }
}
