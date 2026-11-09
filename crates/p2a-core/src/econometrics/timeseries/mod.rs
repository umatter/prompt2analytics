//! Time series models: VAR, VARMA, VECM, impulse responses and the Granger causality test.
//!
//! Pure Rust implementation of multivariate time series models.
//! Uses column-based API for simplicity.
//!
//! # Mathematical Background
//!
//! ## Vector Autoregression VAR(p)
//!
//! A VAR(p) model for k-dimensional time series yₜ:
//!
//! yₜ = c + A₁yₜ₋₁ + A₂yₜ₋₂ + ... + Aₚyₜ₋ₚ + εₜ
//!
//! where Aᵢ are k×k coefficient matrices and εₜ ~ N(0, Σ).
//!
//! Estimation is by equation-by-equation OLS (equivalent to GLS when errors
//! are contemporaneously correlated).
//!
//! ## Vector Error Correction Model (VECM)
//!
//! For cointegrated variables, VECM represents the VAR in error correction form:
//!
//! Δyₜ = αβ'yₜ₋₁ + Γ₁Δyₜ₋₁ + ... + Γₚ₋₁Δyₜ₋ₚ₊₁ + εₜ
//!
//! where β'yₜ₋₁ are the r cointegrating relations and α is the loading matrix.
//!
//! ## Impulse Response Functions (IRF)
//!
//! The response of yᵢ,ₜ₊ₕ to a one-unit shock in εⱼ,ₜ:
//!
//! Orthogonalized IRF uses Cholesky decomposition of Σ to identify structural shocks.
//!
//! # References
//!
//! - Sims, C.A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.
//!   <https://doi.org/10.2307/1912017>. Foundational VAR paper.
//!
//! - Engle, R.F., & Granger, C.W.J. (1987). Co-integration and error correction:
//!   Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
//!   <https://doi.org/10.2307/1913236>. Cointegration and VECM.
//!
//! - Johansen, S. (1991). Estimation and hypothesis testing of cointegration
//!   vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6),
//!   1551-1580. <https://doi.org/10.2307/2938278>. Johansen cointegration test.
//!
//! - Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*.
//!   Springer. ISBN: 978-3540401728. Comprehensive VAR/VECM textbook.
//!
//! - Hamilton, J.D. (1994). *Time Series Analysis*. Princeton University Press.
//!   ISBN: 978-0691042893. Chapters 10-11 on VAR models.
//!
//! - Kilian, L., & Lütkepohl, H. (2017). *Structural Vector Autoregressive Analysis*.
//!   Cambridge University Press. ISBN: 978-1107196575.
//!
//! R equivalent: `vars::VAR()`, `vars::vec2var()`, `urca::ca.jo()`
//!
//! # Layout
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`var`] | [`VarResult`], [`run_var`]; impulse responses [`VarIrfResult`], [`var::compute_irf`], [`run_var_irf`] |
//! | [`varma`] | [`VarmaResult`], [`run_varma`] |
//! | [`vecm`] | [`VecmResult`], [`run_vecm`] |
//! | [`granger`] | [`GrangerResult`], [`granger_test`], [`run_granger_test`], [`granger_test_bidirectional`] |
//! | [`common`] | Column extraction shared by the VAR family (crate-internal) |

pub mod common;
pub mod granger;
pub mod var;
pub mod varma;
pub mod vecm;

pub use granger::{GrangerResult, granger_test, granger_test_bidirectional, run_granger_test};
pub use var::{VarIrfResult, VarResult, run_var, run_var_irf};
pub use varma::{VarmaConfig, VarmaResult, run_varma, run_varma_with};
pub use vecm::{VecmResult, run_vecm};
