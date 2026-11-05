# validation/r/mc/varma_long_var_order.R -- Monte Carlo behind the default long-VAR order of the
# Hannan-Rissanen VARMA estimator (VarmaConfig::long_var_order in
# crates/p2a-core/src/econometrics/timeseries/varma.rs; validation/econometrics/timeseries/varma.md,
# "Why 2 ln n").
#
# Not an R reference case: nothing here is pinned by a Rust test, and run.R does not run it.
#   Rscript validation/r/mc/varma_long_var_order.R
#
# The estimator is the one of validation/r/cases/ts_varma.R (.hr_varma), cut down to the
# coefficients: step 1 a long VAR(m) with intercept by lm.fit on t = m + 1..n, step 2 lm.fit of y_t
# on [1, y_{t-1..p}, ehat_{t-1..q}] over t = m + max(p, q) + 1..n. Three long-VAR orders m are
# compared, each reduced to floor((n - 2) / (2K + 1)) as p2a's default is:
#   floor(ln(n)^2)   Gomez & Maravall (2001), statsmodels' hannan_rissanen default;
#   floor(2 ln n)    p2a's default, max(floor(2 ln n), 2 max(p, q));
#   AIC              vars::VARselect(lag.max = 10, type = "const"), at least 1.
# Designs (all VARMA(1,1), Gaussian innovations, 100 burn-in rows as in the case):
#   case       the DGP of the varma_hr case (eigenvalues of M_1: 0.3, -0.25);
#   strong MA  the case's A_1 and Sigma, M_1 with eigenvalues 0.8 and -0.7 (inverse MA roots);
#   K = 3      a trivariate VARMA(1,1), inverse MA roots of modulus 0.42, 0.27 and 0.24.
# Entry: root mean squared error of the K (1 + 2K) step-2 coefficients, sqrt(mean over
# replications of the squared Frobenius norm of the error), the true constant being 0.
# 200 replications per cell; set.seed(20261104) at the start of each design (its cells n = 100,
# 400, 2,000 then draw in turn), and the three orders are applied to the same series.
# Runtime: about 20 s.
suppressPackageStartupMessages(library(vars))

hr <- function(Y, p, q, m) {
  n <- nrow(Y); K <- ncol(Y)
  E <- matrix(NA_real_, n, K)
  Z <- cbind(1, do.call(cbind, lapply(1:m, function(l) Y[(m + 1 - l):(n - l), , drop = FALSE])))
  E[(m + 1):n, ] <- lm.fit(Z, Y[(m + 1):n, , drop = FALSE])$residuals
  s <- max(p, q); tt <- (m + s + 1):n
  X <- cbind(1, do.call(cbind, lapply(seq_len(p), function(l) Y[tt - l, , drop = FALSE])),
             do.call(cbind, lapply(seq_len(q), function(l) E[tt - l, , drop = FALSE])))
  lm.fit(X, Y[tt, , drop = FALSE])$coefficients     # (1 + K p + K q) x K, column j = equation j
}

simulate <- function(n, A, M, S) {
  K <- nrow(A); nb <- n + 100
  e <- matrix(rnorm(K * nb), ncol = K) %*% chol(S)
  y <- matrix(0, nb, K)
  for (t in 2:nb) y[t, ] <- A %*% y[t - 1, ] + e[t, ] + M %*% e[t - 1, ]
  y <- y[-(1:100), , drop = FALSE]; colnames(y) <- paste0("y", 1:K); y
}

orders <- function(Y) {
  n <- nrow(Y); K <- ncol(Y); cap <- floor((n - 2) / (2 * K + 1))
  aic <- max(1, unname(VARselect(Y, lag.max = 10, type = "const")$selection["AIC(n)"]))
  pmin(c(lnsq = floor(log(n)^2), twoln = max(floor(2 * log(n)), 2), aic = aic), cap)
}

dgps <- list(
  case = list(A = matrix(c(0.5, 0.1, -0.2, 0.4), 2), M = matrix(c(0.3, 0, 0.2, -0.25), 2),
              S = matrix(c(1, 0.3, 0.3, 0.8), 2)),
  "strong MA" = list(A = matrix(c(0.5, 0.1, -0.2, 0.4), 2), M = matrix(c(0.8, 0, 0.3, -0.7), 2),
                     S = matrix(c(1, 0.3, 0.3, 0.8), 2)),
  "K = 3" = list(A = rbind(c(0.5, 0.1, 0), c(-0.2, 0.4, 0.1), c(0, 0.2, 0.3)),
                 M = rbind(c(0.3, 0, 0.1), c(0.1, -0.25, 0), c(0, 0.2, 0.4)),
                 S = matrix(0.3, 3, 3) + diag(0.7, 3)))
ns <- c(100, 400, 2000); R <- 200

t0 <- proc.time()[["elapsed"]]
rows <- list()
for (dn in names(dgps)) {
  g <- dgps[[dn]]; K <- nrow(g$A)
  stopifnot(max(Mod(eigen(g$A)$values)) < 1, max(Mod(eigen(g$M)$values)) < 1)
  truth <- rbind(0, t(g$A), t(g$M))                  # rows: const, y.l1, e.l1; columns: equations
  set.seed(20261104)
  for (n in ns) {
    sq <- matrix(NA_real_, R, 3, dimnames = list(NULL, c("lnsq", "twoln", "aic")))
    for (r in 1:R) {
      Y <- simulate(n, g$A, g$M, g$S); m <- orders(Y)
      for (k in names(m)) sq[r, k] <- sum((hr(Y, 1, 1, m[[k]]) - truth)^2)
    }
    rmse <- sqrt(colMeans(sq))
    rows[[length(rows) + 1]] <- data.frame(DGP = dn, n = n, lnsq = rmse[["lnsq"]],
                                           twoln = rmse[["twoln"]], aic = rmse[["aic"]])
  }
}
tab <- do.call(rbind, rows)
names(tab) <- c("DGP", "n", "floor(ln(n)^2)", "floor(2 ln n)", "AIC")
tab[3:5] <- lapply(tab[3:5], sprintf, fmt = "%.3f")
print(tab, row.names = FALSE)
cat(sprintf("\nelapsed %.0f s\n", proc.time()[["elapsed"]] - t0))
