//! Regression analysis commands

use clap::{Subcommand, ValueEnum};
use p2a_core::regression::CovarianceType;
use p2a_core::regression::gls::{GlsCorrelation, GlsMethod, GlsOptions, run_gls_with};
use p2a_core::regression::quantreg::{QuantRegMethod, QuantRegSe};
use p2a_core::regression::smooth_spline::{SmoothSplineConfig, SplineCriterion, smooth_spline};
use p2a_core::regression::step::StepDirection;
use p2a_core::regression::{OlsClusteredResult, QuantRegConfig, quantreg};
use p2a_core::run_loess;
use p2a_core::{HacResult, run_diagnostics, run_ols};
// Additional regression methods
use ndarray::Array1;
use p2a_core::traits::t_test_p_value;
use p2a_core::{
    model_asymptotic, model_exponential_decay, model_exponential_growth, model_logistic_growth,
    model_michaelis_menten, model_power, run_nls,
};
use p2a_core::{run_line, run_step, run_supsmu};

use crate::output::{
    CoefRow, OutputFormat, RegressionReport, StatKind, add_notes, add_result_notes,
    check_core_option, format_regression_report, missing_json, parse_core_option, print_error,
    print_missing_note, print_notes, print_result_notes, se_label, validate_column_exists,
    validate_columns_exist, validate_sample_size,
};
use crate::progress::Progress;
use crate::session::SessionManager;
use p2a_core::data::columns::{ColumnRequest, NaPolicy};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum RobustSE {
    Standard,
    HC0,
    HC1,
    HC2,
    HC3,
}

impl From<RobustSE> for CovarianceType {
    fn from(val: RobustSE) -> Self {
        match val {
            RobustSE::Standard => CovarianceType::Standard,
            RobustSE::HC0 => CovarianceType::HC0,
            RobustSE::HC1 => CovarianceType::HC1,
            RobustSE::HC2 => CovarianceType::HC2,
            RobustSE::HC3 => CovarianceType::HC3,
        }
    }
}

#[derive(Subcommand)]
pub enum RegressionCommands {
    /// Ordinary Least Squares regression
    #[command(after_help = "\
EXAMPLES:
    # Simple regression
    p2a --session s.json reg ols mydata -y price -x sqft

    # Multiple regression with HC1 robust SEs
    p2a --session s.json reg ols mydata -y price -x sqft bedrooms bathrooms --robust hc1

    # Without intercept
    p2a --session s.json reg ols mydata -y price -x sqft --intercept false
")]
    Ols {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Include an intercept (true/false; default true). With false the model goes through
        /// the origin and R² is uncentered, as R's lm(y ~ 0 + x)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        intercept: bool,

        /// Standard errors: hc1 (default, heteroskedasticity-robust), hc0, hc2, hc3, or standard (classical, as R's summary(lm))
        #[arg(short, long, default_value = "hc1")]
        robust: RobustSE,
    },

    /// Clustered standard errors regression
    #[command(after_help = "\
EXAMPLES:
    # Cluster by firm
    p2a --session s.json reg clustered mydata -y revenue -x employees --cluster firm_id

    # Two-way clustering (firm and year)
    p2a --session s.json reg clustered mydata -y revenue -x employees --cluster firm_id --cluster2 year
")]
    Clustered {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Cluster variable column
        #[arg(long)]
        cluster: String,

        /// Second cluster variable: two-way clustered SEs (Cameron, Gelbach and Miller 2011),
        /// as sandwich::vcovCL(cluster = ~c1 + c2, type = "HC1")
        #[arg(long, value_name = "COLUMN")]
        cluster2: Option<String>,

        /// Include an intercept (true/false; default true). With false the model goes through
        /// the origin and R² is uncentered, as R's lm(y ~ 0 + x)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        intercept: bool,
    },

    /// Run regression diagnostics
    #[command(after_help = "\
EXAMPLES:
    # Full diagnostics (Jarque-Bera, Breusch-Pagan, Durbin-Watson, VIF)
    p2a --session s.json reg diagnostics mydata -y price -x sqft bedrooms bathrooms
")]
    Diagnostics {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,
    },

    /// Quantile regression
    #[command(after_help = "\
EXAMPLES:
    # Median regression (tau = 0.5)
    p2a --session s.json reg quantreg mydata -y price -x sqft bedrooms

    # 90th percentile regression
    p2a --session s.json reg quantreg mydata -y price -x sqft bedrooms -t 0.9
")]
    Quantreg {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Quantile (tau) value (default: 0.5 = median)
        #[arg(short = 't', long, default_value = "0.5")]
        tau: f64,

        /// Algorithm: auto (default: br up to 5,000 rows, fn above; rq's default is br at
        /// every n), br (Barrodale-Roberts simplex), fn (Frisch-Newton interior point) or irls
        /// (an approximation)
        #[arg(long, value_parser = parse_core_option::<QuantRegMethod>, default_value = "auto")]
        method: QuantRegMethod,

        /// Standard errors as summary.rq: nid (default at every n), iid, ker, boot
        #[arg(long, value_parser = parse_core_option::<QuantRegSe>, default_value = "nid")]
        se: QuantRegSe,

        /// Seed of the xy-pair bootstrap (--se boot only)
        #[arg(long)]
        seed: Option<u64>,

        /// Include an intercept (true/false; default true). With false the model goes through
        /// the origin and R² is uncentered, as R's lm(y ~ 0 + x)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        intercept: bool,
    },

    /// LOESS (local polynomial regression)
    #[command(after_help = "\
EXAMPLES:
    # Default LOESS (span=0.75, degree=2)
    p2a --session s.json reg loess mydata -y price -x sqft

    # More local fit with lower span
    p2a --session s.json reg loess mydata -y price -x sqft --span 0.3 --degree 1
")]
    Loess {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Span (smoothing parameter, 0-1, default: 0.75)
        #[arg(long, default_value = "0.75")]
        span: f64,

        /// Polynomial degree (1 = local linear, 2 = local quadratic)
        #[arg(long, default_value = "2")]
        degree: usize,
    },

    /// Nonlinear Least Squares regression
    #[command(after_help = "\
EXAMPLES:
    # Exponential decay: y = a * exp(-b * x) + c
    p2a --session s.json reg nls mydata -y response -x time --model exponential_decay --start 10,0.5,0

    # Michaelis-Menten: y = Vmax * x / (Km + x)
    p2a --session s.json reg nls mydata -y rate -x substrate --model michaelis_menten --start 100,5

    # Logistic growth: y = K / (1 + exp(-r * (x - x0)))
    p2a --session s.json reg nls mydata -y population -x time --model logistic_growth --start 1000,0.5,10
")]
    Nls {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Model type: exponential_decay (y = a * exp(-b * x) + c), exponential_growth,
        /// michaelis_menten, logistic_growth, power, asymptotic
        #[arg(long, default_value = "exponential_decay")]
        model: String,

        /// Comma-separated start values, as many as the model has parameters (R's nls needs
        /// them)
        #[arg(long)]
        start: String,
    },

    /// Generalized Least Squares regression
    #[command(after_help = "\
EXAMPLES:
    # Auto-detect AR(1) correlation
    p2a --session s.json reg gls mydata -y gdp -x investment exports --correlation ar1_auto

    # Fixed AR(1) with rho=0.7
    p2a --session s.json reg gls mydata -y gdp -x investment --correlation ar1 --rho 0.7
")]
    Gls {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Correlation structure: ar1_auto (default; AR(1) with rho estimated), ar1,
        /// compound_symmetry (cs) or identity
        #[arg(long, value_parser = gls_correlation_arg, default_value = "ar1_auto")]
        correlation: GlsCorrelationArg,

        /// Correlation parameter (rho for ar1/cs; estimated when not given; refused with
        /// ar1_auto and identity)
        #[arg(long)]
        rho: Option<f64>,

        /// Likelihood for rho and the reported fit: reml (default, as nlme) or ml
        #[arg(long, value_parser = parse_core_option::<GlsMethod>, default_value = "reml")]
        method: GlsMethod,

        /// Groups for the correlation (nlme form = ~ 1 | g); compound symmetry without
        /// groups needs --rho
        #[arg(long, value_name = "COL")]
        group: Option<String>,

        /// Include an intercept (true/false; default true). With false the model goes through
        /// the origin and R² is uncentered, as R's lm(y ~ 0 + x)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        intercept: bool,
    },

    /// Stepwise regression model selection
    #[command(after_help = "\
EXAMPLES:
    # Bidirectional stepwise with AIC
    p2a --session s.json reg step mydata -y price -x sqft bedrooms bathrooms age pool

    # Forward selection with BIC
    p2a --session s.json reg step mydata -y price -x sqft bedrooms bathrooms --direction forward --use-bic

    # Backward elimination keeping sqft always
    p2a --session s.json reg step mydata -y price -x sqft bedrooms bathrooms --direction backward --required-vars sqft
")]
    Step {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Candidate predictor variables (upper scope)
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Variables always included (lower scope)
        #[arg(long, num_args = 0..)]
        required_vars: Vec<String>,

        /// Direction: forward, backward, or both (default: both)
        #[arg(long, value_parser = check_core_option::<StepDirection>, default_value = "both")]
        direction: String,

        /// Use BIC (k = log n) instead of AIC (true/false; default false)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = false,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        use_bic: bool,

        /// Include an intercept (true/false; default true). With false the model goes through
        /// the origin and R² is uncentered, as R's lm(y ~ 0 + x)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        intercept: bool,
    },

    /// Smoothing spline regression
    #[command(after_help = "\
EXAMPLES:
    # Smoothing chosen by generalized cross-validation (R's default)
    p2a --session s.json reg smooth-spline mydata -y response -x time

    # Leave-one-out cross-validation instead
    p2a --session s.json reg smooth-spline mydata -y response -x time --criterion cv

    # Fixed degrees of freedom
    p2a --session s.json reg smooth-spline mydata -y response -x time --df 5
")]
    SmoothSpline {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Equivalent degrees of freedom (if not set, --criterion chooses the smoothing)
        #[arg(long)]
        df: Option<f64>,

        /// Smoothing-parameter criterion when --df is not given: gcv (R's default cv = FALSE)
        /// or cv (leave-one-out, R's cv = TRUE)
        #[arg(long, value_parser = parse_core_option::<SplineCriterion>, default_value = "gcv")]
        criterion: SplineCriterion,
    },

    /// Friedman's SuperSmoother
    #[command(after_help = "\
EXAMPLES:
    # Auto-select span via cross-validation
    p2a --session s.json reg supsmu mydata -y response -x time

    # Fixed span with bass smoothing
    p2a --session s.json reg supsmu mydata -y response -x time --span 0.05 --bass 5
")]
    Supsmu {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Fixed span (0-1), if not set uses cross-validation
        #[arg(long)]
        span: Option<f64>,

        /// Bass parameter (0-10, higher = smoother)
        #[arg(long, default_value = "0")]
        bass: f64,
    },

    /// Tukey's resistant line (robust regression)
    #[command(after_help = "\
EXAMPLES:
    # Basic resistant line
    p2a --session s.json reg line mydata -y price -x sqft

    # With additional polishing iterations
    p2a --session s.json reg line mydata -y price -x sqft --iter 3
")]
    Line {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable column (single predictor)
        #[arg(short = 'x', long)]
        indep_var: String,

        /// Number of polishing iterations
        #[arg(long, default_value = "1")]
        iter: usize,
    },

    /// HAC (Newey-West) standard errors for OLS
    #[command(after_help = "\
EXAMPLES:
    # Auto-select bandwidth
    p2a --session s.json reg hac mydata -y returns -x market_return

    # Fixed lag length
    p2a --session s.json reg hac mydata -y returns -x market_return smb hml --lags 4
")]
    Hac {
        /// Dataset name
        dataset: String,

        /// Dependent variable column
        #[arg(short = 'y', long)]
        dep_var: String,

        /// Independent variable columns
        #[arg(short = 'x', long, num_args = 1..)]
        indep_vars: Vec<String>,

        /// Number of lags (if not set, uses automatic bandwidth)
        #[arg(long)]
        lags: Option<usize>,

        /// Include an intercept (true/false; default true). With false the model goes through
        /// the origin and R² is uncentered, as R's lm(y ~ 0 + x)
        #[arg(
            long,
            value_name = "BOOL",
            num_args = 0..=1,
            default_value_t = true,
            default_missing_value = "true",
            action = clap::ArgAction::Set,
            value_parser = clap::builder::BoolishValueParser::new()
        )]
        intercept: bool,
    },
}

pub fn execute(
    cmd: &RegressionCommands,
    format: &OutputFormat,
    quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    match cmd {
        RegressionCommands::Ols {
            dataset,
            dep_var,
            indep_vars,
            intercept,
            robust,
        } => execute_ols(
            dataset, dep_var, indep_vars, *intercept, *robust, format, quiet, session,
        ),
        RegressionCommands::Clustered {
            dataset,
            dep_var,
            indep_vars,
            cluster,
            cluster2,
            intercept,
        } => execute_clustered(
            dataset,
            dep_var,
            indep_vars,
            cluster,
            cluster2.as_deref(),
            *intercept,
            format,
            session,
        ),
        RegressionCommands::Diagnostics {
            dataset,
            dep_var,
            indep_vars,
        } => execute_diagnostics(dataset, dep_var, indep_vars, format, session),
        RegressionCommands::Quantreg {
            dataset,
            dep_var,
            indep_vars,
            tau,
            method,
            se,
            seed,
            intercept,
        } => {
            if seed.is_some() && *se != QuantRegSe::Boot {
                print_error(
                    "--seed applies to --se boot only (the other standard errors draw nothing)",
                    format,
                );
                return Ok(());
            }
            let mut config = QuantRegConfig::default();
            config.tau = *tau;
            config.intercept = *intercept;
            config.algorithm = *method;
            config.se = *se;
            config.seed = *seed;
            execute_quantreg(dataset, dep_var, indep_vars, &config, format, session)
        }
        RegressionCommands::Loess {
            dataset,
            dep_var,
            indep_var,
            span,
            degree,
        } => execute_loess(dataset, dep_var, indep_var, *span, *degree, format, session),
        RegressionCommands::Nls {
            dataset,
            dep_var,
            indep_var,
            model,
            start,
        } => execute_nls(dataset, dep_var, indep_var, model, start, format, session),
        RegressionCommands::Gls {
            dataset,
            dep_var,
            indep_vars,
            correlation,
            rho,
            method,
            group,
            intercept,
        } => {
            if rho.is_some()
                && matches!(
                    correlation,
                    GlsCorrelationArg::Ar1Auto
                        | GlsCorrelationArg::Structure(GlsCorrelation::Identity)
                )
            {
                print_error(
                    "--rho applies to --correlation ar1 and compound_symmetry only (ar1_auto \
                     estimates rho; identity has none)",
                    format,
                );
                return Ok(());
            }
            let mut options = GlsOptions::default();
            match correlation {
                GlsCorrelationArg::Ar1Auto => {
                    options.correlation = GlsCorrelation::Ar1;
                    options.rho = None;
                }
                GlsCorrelationArg::Structure(GlsCorrelation::Identity) => {
                    options.correlation = GlsCorrelation::Identity;
                    options.rho = None;
                }
                GlsCorrelationArg::Structure(c) => {
                    options.correlation = *c;
                    options.rho = *rho;
                }
            }
            options.method = *method;
            options.group_col = group.clone();
            execute_gls(
                dataset, dep_var, indep_vars, &options, *intercept, format, session,
            )
        }
        RegressionCommands::Step {
            dataset,
            dep_var,
            indep_vars,
            required_vars,
            direction,
            use_bic,
            intercept,
        } => execute_step(
            dataset,
            dep_var,
            indep_vars,
            required_vars,
            direction,
            *use_bic,
            *intercept,
            format,
            session,
        ),
        RegressionCommands::SmoothSpline {
            dataset,
            dep_var,
            indep_var,
            df,
            criterion,
        } => execute_smooth_spline(
            dataset, dep_var, indep_var, *df, *criterion, format, session,
        ),
        RegressionCommands::Supsmu {
            dataset,
            dep_var,
            indep_var,
            span,
            bass,
        } => execute_supsmu(dataset, dep_var, indep_var, *span, *bass, format, session),
        RegressionCommands::Line {
            dataset,
            dep_var,
            indep_var,
            iter,
        } => execute_line(dataset, dep_var, indep_var, *iter, format, session),
        RegressionCommands::Hac {
            dataset,
            dep_var,
            indep_vars,
            lags,
            intercept,
        } => execute_hac(
            dataset, dep_var, indep_vars, *lags, *intercept, format, session,
        ),
    }
}

fn execute_ols(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    intercept: bool,
    robust: RobustSE,
    format: &OutputFormat,
    quiet: bool,
    session: Option<&mut SessionManager>,
) -> anyhow::Result<()> {
    log::info!(
        "Running OLS regression: {} ~ {}",
        dep_var,
        indep_vars.join(" + ")
    );
    log::debug!(
        "Parameters: dataset={}, intercept={}, robust={:?}",
        dataset_name,
        intercept,
        robust
    );

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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();
            let cov_type: CovarianceType = robust.into();

            // Validate dependent variable exists
            if let Err(e) = validate_column_exists(ds, dep_var, "dependent variable") {
                print_error(&e, format);
                return Ok(());
            }

            // Validate independent variables exist
            if let Err(e) = validate_columns_exist(ds, &x_cols, "independent variables") {
                print_error(&e, format);
                return Ok(());
            }

            // Validate minimum sample size
            let min_obs = x_cols.len() + if intercept { 1 } else { 0 } + 1;
            if let Err(e) = validate_sample_size(ds.nrows(), min_obs, "OLS regression") {
                print_error(&e, format);
                return Ok(());
            }

            log::debug!(
                "Design matrix: {} observations × {} predictors",
                ds.nrows(),
                x_cols.len()
            );

            // Show progress spinner for regression
            let progress = Progress::spinner("Running OLS regression...", quiet);

            match run_ols(ds, dep_var, &x_cols, intercept, cov_type) {
                Ok(result) => {
                    progress.finish_and_clear();
                    log::info!(
                        "OLS completed: n={}, R²={:.4}, Adj R²={:.4}",
                        result.n_obs,
                        result.r_squared,
                        result.adj_r_squared
                    );
                    log::debug!("Degrees of freedom: {}", result.df_resid);

                    // The table comes from the result's coefficient records, so
                    // an aliased column prints the core's NaN, never a made-up 0.
                    let se_type = se_label(result.cov_type);
                    let r2 = R2Labels::new(intercept);
                    let report = RegressionReport {
                        method: "OLS Regression",
                        coefficients: result.coefficients.iter().map(CoefRow::from).collect(),
                        statistic: StatKind::T,
                        r_squared: (r2.r_squared, result.r_squared),
                        adj_r_squared: (r2.adj_r_squared, result.adj_r_squared),
                        n_obs: result.n_obs,
                        se_type: &se_type,
                        se_type_json: result.cov_type.short_label(),
                        r_squared_type: r2.json_type,
                        extra_json: serde_json::Map::new(),
                        warnings: &result.warnings,
                    };
                    println!("{}", format_regression_report(&report, format));
                }
                Err(e) => {
                    progress.finish_and_clear();
                    print_error(&format!("Regression failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_clustered(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    cluster: &str,
    cluster2: Option<&str>,
    intercept: bool,
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            match OlsClusteredResult::fit(ds, dep_var, &x_cols, cluster, cluster2, intercept) {
                Ok(result) => {
                    let second = result.cluster_var_2.as_deref().zip(result.n_clusters_2);
                    let (se_type, se_type_json) =
                        clustered_se_label(&result.cluster_var_1, result.n_clusters_1, second);
                    let mut extra_json = serde_json::Map::new();
                    extra_json.insert("n_clusters".to_string(), result.n_clusters_1.into());
                    if let Some(g2) = result.n_clusters_2 {
                        extra_json.insert("n_clusters_2".to_string(), g2.into());
                    }
                    // The text output shows the drop as the first warning.
                    if let Some(m) = missing_json(&result.ols.missing) {
                        extra_json.insert("missing".to_string(), m);
                    }
                    let method = match cluster2 {
                        None => format!("Clustered OLS (cluster: {cluster})"),
                        Some(c2) => format!("Clustered OLS (clusters: {cluster}, {c2})"),
                    };
                    let r2 = R2Labels::new(intercept);
                    let report = RegressionReport {
                        method: &method,
                        coefficients: result.ols.coefficients.iter().map(CoefRow::from).collect(),
                        statistic: StatKind::T,
                        r_squared: (r2.r_squared, result.ols.r_squared),
                        adj_r_squared: (r2.adj_r_squared, result.ols.adj_r_squared),
                        n_obs: result.ols.n_obs,
                        se_type: &se_type,
                        se_type_json: &se_type_json,
                        r_squared_type: r2.json_type,
                        extra_json,
                        warnings: &result.ols.warnings,
                    };
                    println!("{}", format_regression_report(&report, format));
                }
                Err(e) => {
                    print_error(&format!("Clustered regression failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

/// R² labels of a least-squares fit: with an intercept R² is centered (`"ordinary"`),
/// without one it is uncentered as in `summary.lm` (`"uncentered"`).
struct R2Labels {
    r_squared: &'static str,
    adj_r_squared: &'static str,
    json_type: &'static str,
}

impl R2Labels {
    fn new(intercept: bool) -> R2Labels {
        if intercept {
            R2Labels {
                r_squared: "R-squared",
                adj_r_squared: "Adj. R-squared",
                json_type: "ordinary",
            }
        } else {
            R2Labels {
                r_squared: "R-squared (uncentered)",
                adj_r_squared: "Adj. R-squared (uncentered)",
                json_type: "uncentered",
            }
        }
    }
}

/// The `Standard errors:` text and the JSON `se_type` of a cluster-robust fit,
/// whose covariance is `sandwich::vcovCL(type = "HC1")`: the CR1 factor
/// `G/(G-1) (n-1)/(n-k)`, per dimension for two-way clustering.
///
/// One-way: `("clustered by firm (CR1, G = 10)", "cluster:firm")`; two-way:
/// `("clustered by firm and year (two-way CR1, G = 10 and 20)", "cluster:firm+year")`.
fn clustered_se_label(var1: &str, g1: usize, second: Option<(&str, usize)>) -> (String, String) {
    match second {
        None => (
            format!("clustered by {var1} (CR1, G = {g1})"),
            format!("cluster:{var1}"),
        ),
        Some((var2, g2)) => (
            format!("clustered by {var1} and {var2} (two-way CR1, G = {g1} and {g2})"),
            format!("cluster:{var1}+{var2}"),
        ),
    }
}

fn execute_diagnostics(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            // run_diagnostics takes 3 args (no intercept param)
            match run_diagnostics(ds, dep_var, &x_cols) {
                Ok(diag) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "n_obs": diag.n_obs,
                            "n_params": diag.n_params,
                            "jarque_bera": diag.jarque_bera.as_ref().map(|jb| {
                                serde_json::json!({
                                    "statistic": jb.statistic,
                                    "p_value": jb.p_value,
                                })
                            }),
                            "breusch_pagan": diag.breusch_pagan.as_ref().map(|bp| {
                                serde_json::json!({
                                    "statistic": bp.statistic,
                                    "p_value": bp.p_value,
                                })
                            }),
                            "durbin_watson": diag.durbin_watson.as_ref().map(|dw| dw.statistic),
                            "vif": diag.vif.as_ref().map(|vifs| {
                                vifs.iter().map(|v| serde_json::json!({
                                    "variable": v.variable,
                                    "vif": v.vif,
                                })).collect::<Vec<_>>()
                            }),
                            "condition_number": diag.condition_number,
                            "scaled_condition_number": diag.scaled_condition_number,
                        });
                        let mut json = json;
                        add_notes(&mut json, None, &diag.warnings);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nRegression Diagnostics");
                        println!("{}", "=".repeat(50));
                        println!(
                            "Observations: {}, Parameters: {}",
                            diag.n_obs, diag.n_params
                        );

                        if let Some(ref jb) = diag.jarque_bera {
                            println!("\nJarque-Bera Test (Normality):");
                            println!("  Statistic: {:.4}", jb.statistic);
                            println!("  P-value: {:.4}", jb.p_value);
                        }

                        if let Some(ref bp) = diag.breusch_pagan {
                            println!("\nBreusch-Pagan Test (Heteroskedasticity):");
                            println!("  Statistic: {:.4}", bp.statistic);
                            println!("  P-value: {:.4}", bp.p_value);
                        }

                        if let Some(ref dw) = diag.durbin_watson {
                            println!("\nDurbin-Watson Statistic: {:.4}", dw.statistic);
                            println!("  {}", dw.interpretation);
                        }

                        if let Some(ref vifs) = diag.vif {
                            // The intercept has no VIF (car::vif omits it); a NaN or
                            // infinite VIF of a regressor still prints.
                            let regressors: Vec<_> = vifs
                                .iter()
                                .filter(|v| v.variable != "(Intercept)")
                                .collect();
                            if !regressors.is_empty() {
                                println!("\nVariance Inflation Factors:");
                                for vif_result in regressors {
                                    println!("  {}: {:.4}", vif_result.variable, vif_result.vif);
                                }
                            }
                        }

                        if let Some(cond) = diag.condition_number {
                            println!("\nCondition Number: {:.4}", cond);
                        }
                        print_notes(None, &diag.warnings, format);
                    }
                },
                Err(e) => {
                    print_error(&format!("Diagnostics failed: {}", e), format);
                }
            }
        }
        None => {
            print_error(&format!("Dataset '{}' not found", dataset_name), format);
        }
    }
    Ok(())
}

fn execute_quantreg(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    config: &QuantRegConfig,
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            match quantreg(ds, dep_var, &x_cols, config) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Quantile Regression",
                            "tau": result.tau,
                            "algorithm": result.algorithm.label(),
                            "se_type": result.se_type.to_string(),
                            "se_method": result.se_method,
                            "coefficients": result.coefficients.iter().map(|c| {
                                serde_json::json!({
                                    "name": c.name,
                                    "estimate": c.estimate,
                                    "std_error": c.std_error,
                                    "t_value": c.t_value,
                                    "p_value": c.p_value,
                                    "ci_lower_95": c.ci_lower_95,
                                    "ci_upper_95": c.ci_upper_95,
                                })
                            }).collect::<Vec<_>>(),
                            "objective": result.objective,
                            "nonunique": result.nonunique,
                            "n_obs": result.n_obs,
                        });
                        if let Some(m) = missing_json(&result.missing) {
                            json["missing"] = m;
                        }
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nQuantile Regression (tau = {:.2})", result.tau);
                        println!("{}", "=".repeat(60));
                        println!("Algorithm: {}", result.algorithm.label());
                        println!("Standard errors: {}", result.se_method);

                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(60));

                        for c in &result.coefficients {
                            let sig = if c.p_value < 0.001 {
                                "***"
                            } else if c.p_value < 0.01 {
                                "**"
                            } else if c.p_value < 0.05 {
                                "*"
                            } else {
                                ""
                            };
                            println!(
                                "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                c.name, c.estimate, c.std_error, c.t_value, c.p_value, sig
                            );
                        }

                        println!("\n---");
                        println!("Objective: {:.6}", result.objective);
                        println!("Observations: {}", result.n_obs);
                        print_result_notes(&result, format);
                        print_missing_note(&result.missing, format);
                    }
                },
                Err(e) => print_error(&format!("Quantile regression failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_loess(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    span: f64,
    degree: usize,
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
        Some(ds) => match run_loess(ds, dep_var, indep_var, span, degree, false) {
            Ok(result) => match format {
                OutputFormat::Json => {
                    let mut json = serde_json::json!({
                        "method": "LOESS",
                        "span": result.span,
                        "degree": result.degree,
                        "n_obs": result.n_obs,
                        "enp": result.enp,
                        "rss": result.rss,
                        "fitted_values_sample": result.fitted.iter().take(10).collect::<Vec<_>>(),
                        "residuals_sample": result.residuals.iter().take(10).collect::<Vec<_>>(),
                    });
                    if let Some(m) = missing_json(&result.missing) {
                        json["missing"] = m;
                    }
                    add_result_notes(&mut json, &result);
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                _ => {
                    println!("\nLOESS (Local Polynomial Regression)");
                    println!("{}", "=".repeat(50));
                    println!("\nSpan: {:.4}", result.span);
                    println!("Degree: {}", result.degree);
                    println!("Observations: {}", result.n_obs);
                    println!("Effective params: {:.2}", result.enp);

                    println!("\nFitted values (first 10):");
                    for (i, fitted) in result.fitted.iter().take(10).enumerate() {
                        println!("  {}: {:.6}", i + 1, fitted);
                    }
                    print_result_notes(&result, format);
                    print_missing_note(&result.missing, format);
                }
            },
            Err(e) => print_error(&format!("LOESS failed: {}", e), format),
        },
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_nls(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    model_name: &str,
    start_str: &str,
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
            // Parse starting values
            let start_vals: Vec<f64> = start_str
                .split(',')
                .filter_map(|s| s.trim().parse::<f64>().ok())
                .collect();

            if start_vals.is_empty() {
                print_error(
                    "Invalid starting values. Use comma-separated numbers.",
                    format,
                );
                return Ok(());
            }

            let start = Array1::from_vec(start_vals.clone());

            // Select model function and parameter names
            let (model_fn, param_names): (fn(&Array1<f64>, &Array1<f64>) -> f64, Vec<&str>) =
                match model_name.to_lowercase().as_str() {
                    "exponential_decay" | "exp_decay" => {
                        if start_vals.len() != 3 {
                            print_error(
                                "exponential_decay requires 3 starting values: a, b, c (y = a * exp(-b * x) + c)",
                                format,
                            );
                            return Ok(());
                        }
                        (model_exponential_decay, vec!["a", "b", "c"])
                    }
                    "exponential_growth" | "exp_growth" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "exponential_growth requires 2 starting values: a, b (y = a * exp(b * x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_exponential_growth, vec!["a", "b"])
                    }
                    "michaelis_menten" | "mm" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "michaelis_menten requires 2 starting values: Vmax, Km (y = Vmax * x / (Km + x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_michaelis_menten, vec!["Vmax", "Km"])
                    }
                    "logistic_growth" | "logistic" => {
                        if start_vals.len() != 3 {
                            print_error(
                                "logistic_growth requires 3 starting values: K, r, x0 (y = K / (1 + exp(-r * (x - x0))))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_logistic_growth, vec!["K", "r", "x0"])
                    }
                    "power" => {
                        if start_vals.len() != 2 {
                            print_error(
                                "power requires 2 starting values: a, b (y = a * x^b)",
                                format,
                            );
                            return Ok(());
                        }
                        (model_power, vec!["a", "b"])
                    }
                    "asymptotic" => {
                        if start_vals.len() != 3 {
                            print_error(
                                "asymptotic requires 3 starting values: a, b, c (y = a - b * exp(-c * x))",
                                format,
                            );
                            return Ok(());
                        }
                        (model_asymptotic, vec!["a", "b", "c"])
                    }
                    _ => {
                        print_error(
                            &format!(
                                "Unknown model: {}. Options: exponential_decay, exponential_growth, michaelis_menten, logistic_growth, power, asymptotic",
                                model_name
                            ),
                            format,
                        );
                        return Ok(());
                    }
                };

            match run_nls(ds, dep_var, indep_var, model_fn, &start, &param_names) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Nonlinear Least Squares",
                            "model": model_name,
                            "converged": result.converged,
                            "iterations": result.iterations,
                            "parameters": result.param_names.iter().zip(result.coefficients.iter())
                                .map(|(name, val)| serde_json::json!({
                                    "name": name,
                                    "estimate": val,
                                })).collect::<Vec<_>>(),
                            "std_errors": result.std_errors,
                            "rss": result.rss,
                            "n_obs": result.n_obs,
                        });
                        let mut json = json;
                        add_notes(&mut json, Some((result.converged, result.iterations)), &[]);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nNonlinear Least Squares");
                        println!("{}", "=".repeat(50));
                        println!("Model: {}", model_name);

                        println!("\nParameters:");
                        println!("{:<12} {:>15} {:>15}", "Parameter", "Estimate", "Std Error");
                        println!("{}", "-".repeat(45));

                        for (i, name) in result.param_names.iter().enumerate() {
                            let se = if i < result.std_errors.len() {
                                format!("{:.6}", result.std_errors[i])
                            } else {
                                "N/A".to_string()
                            };
                            println!("{:<12} {:>15.6} {:>15}", name, result.coefficients[i], se);
                        }

                        println!("\n---");
                        println!("RSS: {:.6}", result.rss);
                        println!("Observations: {}", result.n_obs);
                        print_notes(Some((result.converged, result.iterations)), &[], format);
                    }
                },
                Err(e) => print_error(&format!("NLS failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_gls(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    options: &GlsOptions,
    intercept: bool,
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
            let x_refs: Vec<&str> = indep_vars.iter().map(String::as_str).collect();
            let mut var_names = Vec::new();
            if intercept {
                var_names.push("(Intercept)".to_string());
            }
            var_names.extend(indep_vars.iter().cloned());
            match run_gls_with(ds, dep_var, &x_refs, intercept, options) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "model": "GLS Regression",
                            "method": result.method.label(),
                            "correlation": result.correlation,
                            "rho_estimated": result.rho_estimated,
                            "n_groups": result.n_groups,
                            "log_likelihood": result.log_likelihood,
                            "correlation_param": result.correlation_param,
                            "names": var_names,
                            "coefficients": result.coefficients,
                            "std_errors": result.std_errors,
                            "t_values": result.t_values,
                            "p_values": result.p_values,
                            "r_squared": result.r_squared,
                            "adj_r_squared": result.adj_r_squared,
                            "sigma": result.sigma,
                            "aic": result.aic,
                            "bic": result.bic,
                            "n_obs": result.n_obs,
                        });
                        add_result_notes(&mut json, &result);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nGeneralized Least Squares Regression");
                        println!("{}", "=".repeat(60));
                        println!("Correlation: {}", result.correlation);
                        if let Some(rho_val) = result.correlation_param {
                            if result.rho_estimated {
                                println!(
                                    "Rho: {:.4} (estimated, {})",
                                    rho_val,
                                    result.method.label()
                                );
                            } else {
                                println!("Rho: {:.4} (user-supplied)", rho_val);
                            }
                        }
                        if let Some(g) = result.n_groups {
                            println!("Groups: {g}");
                        }
                        println!("Log-likelihood: {:.4}", result.log_likelihood);
                        println!("\nCoefficients:");
                        println!(
                            "{:<15} {:>12} {:>12} {:>10} {:>10}",
                            "Variable", "Coef", "Std Err", "t", "P>|t|"
                        );
                        println!("{}", "-".repeat(60));

                        for (i, name) in var_names.iter().enumerate() {
                            if i < result.coefficients.len() {
                                let sig = if result.p_values[i] < 0.001 {
                                    "***"
                                } else if result.p_values[i] < 0.01 {
                                    "**"
                                } else if result.p_values[i] < 0.05 {
                                    "*"
                                } else {
                                    ""
                                };
                                println!(
                                    "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                    name,
                                    result.coefficients[i],
                                    result.std_errors[i],
                                    result.t_values[i],
                                    result.p_values[i],
                                    sig
                                );
                            }
                        }

                        println!("\n---");
                        let uncentered = if intercept { "" } else { " (uncentered)" };
                        println!("R-squared{uncentered}: {:.4}", result.r_squared);
                        println!("Adj R-squared{uncentered}: {:.4}", result.adj_r_squared);
                        println!("Sigma: {:.4}", result.sigma);
                        println!("AIC: {:.4}, BIC: {:.4}", result.aic, result.bic);
                        println!("Observations: {}", result.n_obs);
                        print_result_notes(&result, format);
                    }
                },
                Err(e) => print_error(&format!("GLS failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_step(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    required_vars: &[String],
    direction: &str,
    use_bic: bool,
    intercept: bool,
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
            let scope_lower: Vec<&str> = required_vars.iter().map(|s| s.as_str()).collect();
            let scope_upper: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            match run_step(
                ds,
                dep_var,
                &scope_lower,
                &scope_upper,
                direction,
                use_bic,
                intercept,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let json = serde_json::json!({
                            "method": "Stepwise Regression",
                            "direction": format!("{}", result.direction),
                            "criterion": result.criterion_name,
                            "initial_variables": result.initial_variables,
                            "final_variables": result.final_variables,
                            "n_steps": result.n_steps,
                            "final_aic": result.final_model.aic,
                            "final_r_squared": result.final_model.r_squared,
                            "steps": result.steps.iter().map(|s| {
                                serde_json::json!({
                                    "step": s.step,
                                    "action": s.action,
                                    "df": s.df,
                                    "rss": s.rss,
                                    "criterion": s.criterion,
                                })
                            }).collect::<Vec<_>>(),
                        });
                        let mut json = json;
                        add_result_notes(&mut json, &result.final_model);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nStepwise Model Selection");
                        println!("{}", "=".repeat(60));
                        println!("Direction: {}", result.direction);
                        println!("Criterion: {} (k = {:.2})", result.criterion_name, result.k);

                        println!("\nStep History:");
                        println!(
                            "{:>5} {:>20} {:>6} {:>12} {:>12}",
                            "Step", "Action", "Df", "RSS", &result.criterion_name
                        );
                        println!("{}", "-".repeat(60));

                        for step in &result.steps {
                            let action = step.action.as_deref().unwrap_or("<initial>");
                            println!(
                                "{:>5} {:>20} {:>6} {:>12.4} {:>12.4}",
                                step.step, action, step.df, step.rss, step.criterion
                            );
                        }

                        println!("\nInitial model: {}", result.initial_variables.join(" + "));
                        println!("Final model: {}", result.final_variables.join(" + "));
                        println!("Total steps: {}", result.n_steps);
                        let uncentered = if intercept { "" } else { " (uncentered)" };
                        println!(
                            "\nFinal model R-squared{uncentered}: {:.4}",
                            result.final_model.r_squared
                        );
                        print_result_notes(&result.final_model, format);
                    }
                },
                Err(e) => print_error(&format!("Stepwise selection failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_smooth_spline(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    df: Option<f64>,
    criterion: SplineCriterion,
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

    let Some(ds) = dataset else {
        print_error(&format!("Dataset '{}' not found", dataset_name), format);
        return Ok(());
    };
    // R's smooth.spline refuses missing or infinite values: so does this.
    let fit = ColumnRequest::new(ds.df())
        .numeric(indep_var)
        .numeric(dep_var)
        .na(NaPolicy::Fail)
        .extract()
        .and_then(|cols| {
            let config = df
                .map(SmoothSplineConfig::with_df)
                .unwrap_or_else(|| SmoothSplineConfig::with_criterion(criterion));
            smooth_spline(cols.numeric(indep_var), cols.numeric(dep_var), config)
        });
    let result = match fit {
        Ok(result) => result,
        Err(e) => {
            print_error(&format!("Smoothing spline failed: {}", e), format);
            return Ok(());
        }
    };
    let criterion_name = result
        .criterion
        .map(|c| SplineCriterion::choices().canonical(c));
    match format {
        OutputFormat::Json => {
            let mut json = serde_json::json!({
                "method": "Smoothing Spline",
                "criterion": criterion_name,
                "df": result.df,
                "spar": result.spar,
                "lambda": result.lambda,
                "cv_crit": result.cv_crit,
                "n_obs": result.n_obs,
                "n_knots": result.n_knots,
                "fitted_sample": result.y.iter().take(10).collect::<Vec<_>>(),
            });
            add_result_notes(&mut json, &result);
            println!("{}", serde_json::to_string_pretty(&json)?);
        }
        _ => {
            println!("\nSmoothing Spline");
            println!("{}", "=".repeat(50));
            if let Some(name) = criterion_name {
                println!("Criterion: {}", name.to_uppercase());
            }
            println!("Equivalent df: {:.4}", result.df);
            println!("Smoothing param (spar): {:.4}", result.spar);
            println!("Lambda: {:.6e}", result.lambda);
            if let Some(cv) = result.cv_crit {
                println!("CV criterion: {:.6}", cv);
            }
            println!("Observations: {}", result.n_obs);
            println!("Knots: {}", result.n_knots);

            println!("\nFitted values (first 10):");
            for (i, (x, y)) in result.x.iter().zip(result.y.iter()).take(10).enumerate() {
                println!("  {}: x={:.4}, y={:.6}", i + 1, x, y);
            }
            print_result_notes(&result, format);
        }
    }
    Ok(())
}

fn execute_supsmu(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    span: Option<f64>,
    bass: f64,
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
            // x and y with R's deletion of incomplete pairs; too few complete pairs is
            // reported against the dataset's row count
            let cols = match ColumnRequest::new(ds.df())
                .numeric(indep_var)
                .numeric(dep_var)
                .na(NaPolicy::Omit)
                .min_rows(4, &format!("SuperSmoother of '{dep_var}' on '{indep_var}'"))
                .extract()
            {
                Ok(c) => c,
                Err(e) => {
                    print_error(&format!("SuperSmoother failed: {e}"), format);
                    return Ok(());
                }
            };

            match run_supsmu(
                cols.numeric(indep_var),
                cols.numeric(dep_var),
                None,
                span,
                false,
                bass,
            ) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "SuperSmoother",
                            "bass": result.bass,
                            "periodic": result.periodic,
                            "n": result.n,
                            "fitted_sample": result.y.iter().take(10).collect::<Vec<_>>(),
                        });
                        if let Some(m) = missing_json(cols.missing()) {
                            json["missing"] = m;
                        }
                        add_notes(&mut json, None, &[]);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nFriedman's SuperSmoother");
                        println!("{}", "=".repeat(50));
                        println!("Bass parameter: {:.1}", result.bass);
                        println!("Observations: {}", result.n);

                        println!("\nSmoothed values (first 10):");
                        for (i, (x_val, y_val)) in
                            result.x.iter().zip(result.y.iter()).take(10).enumerate()
                        {
                            println!("  {}: x={:.4}, y={:.6}", i + 1, x_val, y_val);
                        }
                        print_missing_note(cols.missing(), format);
                    }
                },
                Err(e) => print_error(&format!("SuperSmoother failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_line(
    dataset_name: &str,
    dep_var: &str,
    indep_var: &str,
    iter: usize,
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
            // x and y with R's deletion of incomplete pairs; too few complete pairs is
            // reported against the dataset's row count
            let cols = match ColumnRequest::new(ds.df())
                .numeric(indep_var)
                .numeric(dep_var)
                .na(NaPolicy::Omit)
                .min_rows(
                    3,
                    &format!("Tukey's resistant line of '{dep_var}' on '{indep_var}'"),
                )
                .extract()
            {
                Ok(c) => c,
                Err(e) => {
                    print_error(&format!("Resistant line failed: {e}"), format);
                    return Ok(());
                }
            };

            match run_line(cols.numeric(indep_var), cols.numeric(dep_var), Some(iter)) {
                Ok(result) => match format {
                    OutputFormat::Json => {
                        let mut json = serde_json::json!({
                            "method": "Tukey's Resistant Line",
                            "intercept": result.intercept,
                            "slope": result.slope,
                            "n": result.n,
                            "iterations": result.iter,
                            "residuals_sample": result.residuals.iter().take(10).collect::<Vec<_>>(),
                        });
                        if let Some(m) = missing_json(cols.missing()) {
                            json["missing"] = m;
                        }
                        add_notes(&mut json, None, &[]);
                        println!("{}", serde_json::to_string_pretty(&json)?);
                    }
                    _ => {
                        println!("\nTukey's Resistant Line");
                        println!("{}", "=".repeat(50));
                        println!("Intercept: {:.6}", result.intercept);
                        println!("Slope: {:.6}", result.slope);
                        println!("Observations: {}", result.n);
                        println!("Polishing iterations: {}", result.iter);

                        println!("\nResiduals (first 10):");
                        for (i, resid) in result.residuals.iter().take(10).enumerate() {
                            println!("  {}: {:.6}", i + 1, resid);
                        }
                        print_missing_note(cols.missing(), format);
                    }
                },
                Err(e) => print_error(&format!("Resistant line failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

fn execute_hac(
    dataset_name: &str,
    dep_var: &str,
    indep_vars: &[String],
    lags: Option<usize>,
    intercept: bool,
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
            let x_cols: Vec<&str> = indep_vars.iter().map(|s| s.as_str()).collect();

            // Run HAC estimation directly using the Dataset-based API
            match HacResult::fit(
                ds,
                dep_var,
                &x_cols,
                lags,
                Some("bartlett"),
                false,
                intercept,
            ) {
                Ok(hac_result) => {
                    // Also run OLS for R-squared
                    let ols_result =
                        run_ols(ds, dep_var, &x_cols, intercept, CovarianceType::Standard);
                    let (r_squared, n_obs_display) = match &ols_result {
                        Ok(ols) => (ols.r_squared, ols.n_obs),
                        Err(_) => (f64::NAN, hac_result.n_obs),
                    };

                    // t = c / s from the coefficients and HAC standard errors. A
                    // zero or NaN SE gives an infinite or NaN t (printed as such,
                    // null in JSON), never a made-up 0.
                    let coefficients: Vec<f64> = hac_result.coefficients.to_vec();
                    let std_errors: Vec<f64> = hac_result.std_errors.to_vec();
                    let t_values: Vec<f64> = coefficients
                        .iter()
                        .zip(std_errors.iter())
                        .map(|(c, s)| c / s)
                        .collect();

                    // lm's df.residual: n - rank
                    let df = hac_result.df_resid as f64;
                    let p_values: Vec<f64> =
                        t_values.iter().map(|&t| t_test_p_value(t, df)).collect();

                    match format {
                        OutputFormat::Json => {
                            let mut json = serde_json::json!({
                                "method": "OLS with HAC Standard Errors",
                                "kernel": hac_result.kernel.to_string(),
                                "df_resid": hac_result.df_resid,
                                "bandwidth": hac_result.bandwidth,
                                "names": hac_result.names,
                                "coefficients": coefficients,
                                "std_errors": std_errors,
                                "t_values": t_values,
                                "p_values": p_values,
                                "n_obs": n_obs_display,
                                "r_squared": r_squared,
                            });
                            if !hac_result.warnings.is_empty() {
                                json["warnings"] = serde_json::json!(hac_result.warnings);
                            }
                            add_result_notes(&mut json, &hac_result);
                            println!("{}", serde_json::to_string_pretty(&json)?);
                        }
                        _ => {
                            println!("\nOLS with HAC (Newey-West) Standard Errors");
                            println!("{}", "=".repeat(60));
                            println!("Kernel: {}", hac_result.kernel);
                            println!("Bandwidth: {}", hac_result.bandwidth);

                            println!("\nCoefficients:");
                            println!(
                                "{:<15} {:>12} {:>12} {:>10} {:>10}",
                                "Variable", "Coef", "HAC SE", "t", "P>|t|"
                            );
                            println!("{}", "-".repeat(60));

                            for (i, name) in hac_result.names.iter().enumerate() {
                                if i < coefficients.len() {
                                    let sig = if p_values[i] < 0.001 {
                                        "***"
                                    } else if p_values[i] < 0.01 {
                                        "**"
                                    } else if p_values[i] < 0.05 {
                                        "*"
                                    } else {
                                        ""
                                    };
                                    println!(
                                        "{:<15} {:>12.6} {:>12.6} {:>10.4} {:>9.4} {}",
                                        name,
                                        coefficients[i],
                                        std_errors[i],
                                        t_values[i],
                                        p_values[i],
                                        sig
                                    );
                                }
                            }

                            println!("\n---");
                            let uncentered = if intercept { "" } else { " (uncentered)" };
                            println!("R-squared{uncentered}: {:.4}", r_squared);
                            println!("Observations: {}", n_obs_display);
                            if !hac_result.warnings.is_empty() {
                                println!("\nWarnings:");
                                for w in &hac_result.warnings {
                                    println!("  - {w}");
                                }
                            }
                        }
                    }
                }
                Err(e) => print_error(&format!("HAC estimation failed: {}", e), format),
            }
        }
        None => print_error(&format!("Dataset '{}' not found", dataset_name), format),
    }
    Ok(())
}

/// `reg gls --correlation`: `ar1_auto` (AR(1) with rho estimated, `--rho` ignored) or a
/// correlation structure.
#[derive(Debug, Clone, Copy)]
pub enum GlsCorrelationArg {
    Ar1Auto,
    Structure(GlsCorrelation),
}

fn gls_correlation_arg(s: &str) -> Result<GlsCorrelationArg, String> {
    let name = p2a_core::options::normalize(s);
    if name == "ar1_auto" || name == "auto" {
        return Ok(GlsCorrelationArg::Ar1Auto);
    }
    parse_core_option::<GlsCorrelation>(s)
        .map(GlsCorrelationArg::Structure)
        .map_err(|e| format!("{e}; or ar1_auto (AR(1) with rho estimated)"))
}

#[cfg(test)]
mod tests {
    use super::clustered_se_label;

    #[test]
    fn clustered_label() {
        assert_eq!(
            clustered_se_label("firm", 10, None),
            (
                "clustered by firm (CR1, G = 10)".to_string(),
                "cluster:firm".to_string()
            )
        );
        assert_eq!(
            clustered_se_label("firm", 10, Some(("year", 20))),
            (
                "clustered by firm and year (two-way CR1, G = 10 and 20)".to_string(),
                "cluster:firm+year".to_string()
            )
        );
    }
}
