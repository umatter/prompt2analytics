# validation/r/cases/ts_vecm.R -- Johansen VECM references (T3: ECON-52).
# urca::ca.jo (trace and maximal-eigenvalue tests, eigenvalues, cointegrating vectors V,
# loadings W, PI) and urca::cajorls (normalised beta, alpha, Gamma by OLS on the r-restricted VECM).
# Two datasets: urca's Danish money-demand data (Johansen-Juselius 1990, the urca example, here
# without its seasonal dummies) and a simulated 3-variable system with one cointegrating relation.
# Every fit uses spec = "transitory" (p2a implements the transitory form only; urca's "longrun"
# gives the same eigenvalues, statistics, beta and alpha). The levels VAR (vars::vec2var) and
# urca's @GAMMA slot are not implemented by p2a and not recorded.
suppressPackageStartupMessages(library(urca))

.vecm_values <- function(x, ecdet, K, spec, r) {
  out <- list()
  for (type in c("trace", "eigen")) {
    cj <- ca.jo(x, type = type, ecdet = ecdet, K = K, spec = spec)
    out[[paste0(type, "_stat")]] <- vec(cj@teststat)          # urca order: H0 r <= k-1, ..., r = 0
    out[[paste0(type, "_cv10")]] <- vec(cj@cval[, 1])
    out[[paste0(type, "_cv5")]] <- vec(cj@cval[, 2])
    out[[paste0(type, "_cv1")]] <- vec(cj@cval[, 3])
  }
  cj <- ca.jo(x, type = "trace", ecdet = ecdet, K = K, spec = spec)
  out$eigenvalues <- vec(cj@lambda)
  out$v_normalised <- vec(cj@V)         # column-major, columns normalised on the first variable
  out$w_loadings <- vec(cj@W)
  out$pi <- vec(cj@PI)
  ls <- cajorls(cj, r = r)
  out$beta <- vec(ls$beta)
  cf <- coef(ls$rlm)                    # rows: ect1..ectr, (constant), lagged differences
  out$rlm_coef <- vec(cf)
  out$rlm_rownames_n <- nrow(cf)
  out$n_obs <- nrow(x)
  out
}

case("vecm_denmark", "econometrics/timeseries", "Johansen VECM, Danish money demand (urca example), K = 2, ecdet = const, r = 1", function(ctx) {
  data(denmark, package = "urca")
  d <- ctx$gen_dataset("vecm_denmark", denmark[, c("LRM", "LRY", "IBO", "IDE")])
  x <- as.matrix(d)
  v <- .vecm_values(x, ecdet = "const", K = 2, spec = "transitory", r = 1)
  vn <- .vecm_values(x, ecdet = "none", K = 2, spec = "transitory", r = 1)
  names(vn) <- paste0("none_", names(vn))
  v2 <- .vecm_values(x, ecdet = "const", K = 2, spec = "transitory", r = 2)   # beta normalised by its top 2 x 2 block
  names(v2) <- paste0("r2_", names(v2))
  list(values = c(v,
                  vn[c("none_eigenvalues", "none_trace_stat", "none_trace_cv5", "none_beta", "none_rlm_coef")],
                  v2[c("r2_beta", "r2_rlm_coef", "r2_rlm_rownames_n")]),
       reference = "urca::ca.jo(type = 'trace' | 'eigen', ecdet, K = 2, spec = 'transitory'); urca::cajorls(r = 1 | 2)",
       call = "ca.jo(denmark[, c('LRM','LRY','IBO','IDE')], ecdet = 'const', K = 2, spec = 'transitory')",
       packages = "urca")
})

case("vecm_sim3", "econometrics/timeseries", "Johansen VECM, simulated 3 variables, one cointegrating relation, K = 3", function(ctx) {
  set.seed(20261101)
  n <- 250
  e <- matrix(rnorm(3 * n), n, 3)
  x1 <- cumsum(e[, 1])
  x2 <- 0.5 * x1 + arima.sim(list(ar = 0.5), n) + 0.2 * e[, 2]
  x3 <- cumsum(e[, 3]) + 0.3 * x1
  d <- ctx$gen_dataset("vecm_sim3", data.frame(x1 = x1, x2 = as.numeric(x2), x3 = x3))
  x <- as.matrix(d)
  v <- .vecm_values(x, ecdet = "none", K = 3, spec = "transitory", r = 1)
  vt <- .vecm_values(x, ecdet = "trend", K = 3, spec = "transitory", r = 1)
  names(vt) <- paste0("trend_", names(vt))
  list(values = c(v, vt[c("trend_eigenvalues", "trend_trace_stat", "trend_trace_cv5", "trend_eigen_stat", "trend_beta",
                          "trend_rlm_coef")]),
       reference = "urca::ca.jo; urca::cajorls",
       call = "ca.jo(sim3, ecdet = 'none' | 'trend', K = 3, spec = 'transitory')",
       packages = "urca")
})
