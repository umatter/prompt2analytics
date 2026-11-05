# validation/r/cases/causal_doubleml.R -- double/debiased ML references (T3-WP30: CAUS-103, CAUS-99
# part (PLIV, IIVM); moved in CAUS-60, CAUS-104). Package copy of design Appendix A.26 (canonical):
# same draws, dataset and keys; keys added (`*_tight`, R2, CIs, p-values, the remaining nuisance
# RMSEs, Jacobians, IRM with lasso outcome learners, IIVM under one-sided noncompliance).
# DoubleML is not installed: this file writes out the DoubleML cross-fitting estimators
# (Chernozhukov et al. 2018; Bach et al. 2022, 2024) with K = 5 fixed folds (the `fold` column)
# and deterministic learners: OLS, lasso and ridge (glmnet at a fixed lambda, standardize = TRUE),
# logistic regression (glm) for binary nuisances; propensities m trimmed to [0.01, 0.99]
# (DoubleML's trimming_rule = "truncate", trimming_threshold = 0.01); dml2 (pooled scores).
#   PLR (partialling out):  theta = sum (D - m)(Y - l) / sum (D - m)^2; psi = (Y - l - theta (D - m))(D - m)
#   IRM (ATE):              psi_b = g1 - g0 + D (Y - g1)/m - (1 - D)(Y - g0)/(1 - m); theta = mean(psi_b)
#   PLIV (partialling out): theta = sum rZ rY / sum rZ rD
#   IIVM (LATE):            theta = -mean(psi_b) / mean(psi_a) with the Wald-type scores of DoubleML
#                           (psi_a = -(r1 - r0 + Z (D - r1)/m - (1 - Z)(D - r0)/(1 - m)), psi = psi_a theta + psi_b)
# SE = sqrt(mean(psi^2) / J^2 / n), J = mean(d psi / d theta); CI theta -/+ qnorm(0.975) SE;
# p = 2 pnorm(-|theta / SE|). Nuisance RMSEs and R2 from the cross-fitted (out-of-fold) predictions
# over all n observations (g0, g1, r0, r1 over their own arm, as DoubleML's evaluate_learners)
# (CAUS-60). `*_tight`: the same estimators with each lasso / ridge fit replaced by the exact
# minimiser of glmnet's objective (lasso: KKT on glmnet's active set, checked to 1e-12; ridge:
# closed form with glmnet's penalty lambda / sd(y)), the T2 design 2.3 convention for iterative fits.
suppressPackageStartupMessages(library(glmnet))

.dml_psd <- function(v) sqrt(mean((v - mean(v))^2))
# Exact minimiser of glmnet's gaussian objective at one lambda (standardize = TRUE, intercept = TRUE):
# (1/2n) ||y - b0 - X b||^2 + lambda (alpha ||s b||_1 + (1 - alpha)/(2 sd(y)) ||s b||_2^2), s = sd(X).
.dml_exact_coef <- function(learner, Xtr, ytr, lambda) {
  n <- nrow(Xtr); xm <- colMeans(Xtr); xs <- apply(Xtr, 2, .dml_psd)
  Xs <- sweep(sweep(Xtr, 2, xm), 2, xs, "/"); ym <- mean(ytr); yc <- ytr - ym
  G <- crossprod(Xs) / n; c0 <- drop(crossprod(Xs, yc)) / n
  if (learner == "ridge") {
    bs <- solve(G + (lambda / .dml_psd(ytr)) * diag(ncol(Xs)), c0)
  } else {
    g <- glmnet(Xtr, ytr, alpha = 1, lambda = lambda, standardize = TRUE, control = list(thresh = 1e-14))
    bs <- as.numeric(coef(g))[-1] * xs
    for (it in 1:50) {
      A <- which(bs != 0); s <- sign(bs[A]); bn <- numeric(length(bs))
      if (length(A)) bn[A] <- solve(G[A, A, drop = FALSE], c0[A] - lambda * s)
      grad <- c0 - drop(G %*% bn)
      viol <- setdiff(which(abs(grad) > lambda * (1 + 1e-12)), A); flip <- A[sign(bn[A]) != s]
      if (!length(viol) && !length(flip)) { bs <- bn; break }
      bs <- bn; if (length(viol)) bs[viol] <- sign(grad[viol]) * 1e-300; bs[flip] <- 0
    }
    A <- which(bs != 0); grad <- c0 - drop(G %*% bs)
    stopifnot(all(abs(grad[A] - lambda * sign(bs[A])) < 1e-12), all(abs(grad[-A]) <= lambda * (1 + 1e-12)))
  }
  b <- bs / xs; c(ym - sum(xm * b), b)
}
.dml_fit_pred <- function(learner, Xtr, ytr, Xte, lambda = 0.05, tight = FALSE) {
  if (tight && learner %in% c("lasso", "ridge")) return(drop(cbind(1, Xte) %*% .dml_exact_coef(learner, Xtr, ytr, lambda)))
  switch(learner,
    ols = drop(cbind(1, Xte) %*% qr.coef(qr(cbind(1, Xtr)), ytr)),
    lasso = drop(predict(glmnet(Xtr, ytr, alpha = 1, lambda = lambda, standardize = TRUE, control = list(thresh = 1e-14)), Xte)),
    ridge = drop(predict(glmnet(Xtr, ytr, alpha = 0, lambda = lambda, standardize = TRUE, control = list(thresh = 1e-14)), Xte)),
    logit = drop(plogis(cbind(1, Xte) %*% coef(glm(ytr ~ Xtr, family = binomial, control = glm.control(epsilon = 1e-14))))))
}
.dml_crossfit <- function(learner, X, y, fold, subset = rep(TRUE, length(y)), tight = FALSE) {
  out <- numeric(length(y))
  for (k in sort(unique(fold))) { tr <- fold != k & subset; te <- fold == k; out[te] <- .dml_fit_pred(learner, X[tr, , drop = FALSE], y[tr], X[te, , drop = FALSE], tight = tight) }
  out
}
.dml_trim <- function(p) pmin(pmax(p, 0.01), 0.99)
.dml_se <- function(psi, J) sqrt(mean(psi^2) / J^2 / length(psi))
.dml_rmse <- function(r) sqrt(mean(r^2))
.dml_r2 <- function(y, pred) 1 - sum((y - pred)^2) / sum((y - mean(y))^2)
.dml_inf <- function(th, se) list(ci = vec(th + c(-1, 1) * qnorm(0.975) * se), p = 2 * pnorm(-abs(th / se)))

case("doubleml_models", "causal/dml", "DoubleML PLR (OLS, lasso, ridge learners), IRM (logistic propensity), PLIV, IIVM with K = 5 fixed folds", function(ctx) {
  set.seed(20261201)
  n <- 400; p <- 6
  X <- matrix(rnorm(n * p), n, p)
  dcont <- 0.5 * X[, 1] - 0.3 * X[, 2] + rnorm(n)
  y <- 0.7 * dcont + X[, 1] + 0.5 * X[, 3] + rnorm(n)
  dbin <- rbinom(n, 1, plogis(0.4 * X[, 1] - 0.4 * X[, 2]))
  yb <- 1 + 1.0 * dbin + X[, 1] + 0.5 * X[, 3] + rnorm(n)
  z <- rnorm(n) + 0.3 * X[, 4]; u <- rnorm(n)
  dz <- 0.8 * z + 0.3 * X[, 1] + u; yz <- 0.5 * dz + X[, 1] - 0.6 * u + rnorm(n)
  zb <- rbinom(n, 1, plogis(0.3 * X[, 4])); db <- as.integer(runif(n) < plogis(-1 + 2.5 * zb + 0.3 * X[, 1]))
  yzb <- 1 + 0.9 * db + X[, 1] + rnorm(n)
  fold <- rep_len(1:5, n)[sample.int(n)]
  d <- ctx$gen_dataset("doubleml_sim", data.frame(round(X, 10), dcont = round(dcont, 10), y = round(y, 10), dbin = dbin, yb = round(yb, 10),
                                                  z = round(z, 10), dz = round(dz, 10), yz = round(yz, 10), zb = zb, db = db, yzb = round(yzb, 10), fold = fold))
  X <- as.matrix(d[, paste0("X", 1:p)]); f <- d$fold
  v <- list()
  # PLR with each learner for l = E[Y|X] and m = E[D|X]; `_tight` with the exact glmnet minimiser
  for (lr in c("ols", "lasso", "ridge")) for (tight in if (lr == "ols") FALSE else c(FALSE, TRUE)) {
    l <- .dml_crossfit(lr, X, d$y, f, tight = tight); m <- .dml_crossfit(lr, X, d$dcont, f, tight = tight)
    rd <- d$dcont - m; ry <- d$y - l
    th <- sum(rd * ry) / sum(rd^2); psi <- (ry - th * rd) * rd; J <- -mean(rd^2); se <- .dml_se(psi, J)
    key <- function(s) paste0("plr_", lr, "_", s, if (tight) "_tight" else "")
    v[[key("theta")]] <- th; v[[key("se")]] <- se
    v[[key("rmse_l")]] <- .dml_rmse(ry); v[[key("rmse_m")]] <- .dml_rmse(rd)
    if (!tight) {
      v[[key("r2_l")]] <- .dml_r2(d$y, l); v[[key("r2_m")]] <- .dml_r2(d$dcont, m)
      v[[key("jacobian")]] <- -J; v[[key("mean_resid_m")]] <- mean(rd)
      inf <- .dml_inf(th, se); v[[key("ci95")]] <- inf$ci; v[[key("p")]] <- inf$p
    }
  }
  stopifnot(abs(v$plr_lasso_theta - v$plr_lasso_theta_tight) < 1e-9, abs(v$plr_ridge_theta - v$plr_ridge_theta_tight) < 1e-9,
            abs(v$plr_lasso_theta - v$plr_ols_theta) > 1e-3)   # lasso is not OLS (CAUS-103)
  # IRM (ATE): g0, g1 by OLS on each arm of the training folds; m by logistic regression
  g0 <- .dml_crossfit("ols", X, d$yb, f, d$dbin == 0); g1 <- .dml_crossfit("ols", X, d$yb, f, d$dbin == 1)
  mm <- .dml_trim(.dml_crossfit("logit", X, d$dbin, f))
  psib <- g1 - g0 + d$dbin * (d$yb - g1) / mm - (1 - d$dbin) * (d$yb - g0) / (1 - mm)
  th <- mean(psib); v$irm_theta <- th; v$irm_se <- .dml_se(psib - th, -1)
  v$irm_ps_range <- vec(range(mm)); v$irm_rmse_g0 <- sqrt(mean(((d$yb - g0)[d$dbin == 0])^2)); v$irm_rmse_g1 <- sqrt(mean(((d$yb - g1)[d$dbin == 1])^2))
  v$irm_r2_g0 <- .dml_r2(d$yb[d$dbin == 0], g0[d$dbin == 0]); v$irm_r2_g1 <- .dml_r2(d$yb[d$dbin == 1], g1[d$dbin == 1])
  gD <- ifelse(d$dbin == 1, g1, g0)   # the outcome model of the observed arm (NuisanceDiagnostics.outcome_*)
  v$irm_rmse_outcome <- .dml_rmse(d$yb - gD); v$irm_r2_outcome <- .dml_r2(d$yb, gD)
  v$irm_rmse_m <- .dml_rmse(d$dbin - mm); v$irm_r2_m <- .dml_r2(d$dbin, mm); v$irm_mean_resid_m <- mean(d$dbin - mm)
  v$irm_mean_psi_sq <- mean((psib - th)^2)   # what the pre-fix Display printed as "Jacobian (Var(D-m))" (CAUS-104)
  inf <- .dml_inf(th, v$irm_se); v$irm_ci95 <- inf$ci; v$irm_p <- inf$p
  # IRM with lasso outcome learners (the binary nuisance m stays logistic)
  for (tight in c(FALSE, TRUE)) {
    g0 <- .dml_crossfit("lasso", X, d$yb, f, d$dbin == 0, tight = tight); g1 <- .dml_crossfit("lasso", X, d$yb, f, d$dbin == 1, tight = tight)
    pb <- g1 - g0 + d$dbin * (d$yb - g1) / mm - (1 - d$dbin) * (d$yb - g0) / (1 - mm)
    sfx <- if (tight) "_tight" else ""
    v[[paste0("irm_lasso_theta", sfx)]] <- mean(pb); v[[paste0("irm_lasso_se", sfx)]] <- .dml_se(pb - mean(pb), -1)
    v[[paste0("irm_lasso_rmse_g0", sfx)]] <- sqrt(mean(((d$yb - g0)[d$dbin == 0])^2))
  }
  # PLIV
  l <- .dml_crossfit("ols", X, d$yz, f); r <- .dml_crossfit("ols", X, d$dz, f); mz <- .dml_crossfit("ols", X, d$z, f)
  ry <- d$yz - l; rD <- d$dz - r; rz <- d$z - mz
  th <- sum(rz * ry) / sum(rz * rD); psi <- (ry - th * rD) * rz
  v$pliv_theta <- th; v$pliv_se <- .dml_se(psi, -mean(rD * rz))
  v$pliv_rmse_l <- .dml_rmse(ry); v$pliv_rmse_r <- .dml_rmse(rD); v$pliv_rmse_m <- .dml_rmse(rz); v$pliv_jacobian <- mean(rD * rz)
  v$pliv_r2_l <- .dml_r2(d$yz, l); v$pliv_r2_r <- .dml_r2(d$dz, r); v$pliv_r2_m <- .dml_r2(d$z, mz); v$pliv_mean_resid_r <- mean(rD)
  inf <- .dml_inf(th, v$pliv_se); v$pliv_ci95 <- inf$ci; v$pliv_p <- inf$p
  # IIVM (LATE)
  iivm <- function(D, onesided = FALSE) {
    g0 <- .dml_crossfit("ols", X, d$yzb, f, d$zb == 0); g1 <- .dml_crossfit("ols", X, d$yzb, f, d$zb == 1)
    r0 <- if (onesided) rep(0, n) else .dml_crossfit("logit", X, D, f, d$zb == 0)   # no D = 1 among Z = 0: r0 = 0
    r1 <- .dml_crossfit("logit", X, D, f, d$zb == 1)
    mz <- .dml_trim(.dml_crossfit("logit", X, d$zb, f))
    pb <- g1 - g0 + d$zb * (d$yzb - g1) / mz - (1 - d$zb) * (d$yzb - g0) / (1 - mz)
    pa <- -(r1 - r0 + d$zb * (D - r1) / mz - (1 - d$zb) * (D - r0) / (1 - mz))
    th <- -mean(pb) / mean(pa)
    list(theta = th, se = .dml_se(pa * th + pb, mean(pa)), jacobian = -mean(pa), mz = mz, g0 = g0, g1 = g1, r0 = r0, r1 = r1, D = D)
  }
  a <- iivm(d$db)
  v$iivm_theta <- a$theta; v$iivm_se <- a$se
  z0 <- d$zb == 0; z1 <- d$zb == 1
  v$iivm_jacobian <- a$jacobian; v$iivm_ps_range <- vec(range(a$mz))
  v$iivm_rmse_g0 <- .dml_rmse((d$yzb - a$g0)[z0]); v$iivm_rmse_g1 <- .dml_rmse((d$yzb - a$g1)[z1])
  v$iivm_rmse_r0 <- .dml_rmse((d$db - a$r0)[z0]); v$iivm_rmse_r1 <- .dml_rmse((d$db - a$r1)[z1]); v$iivm_rmse_m <- .dml_rmse(d$zb - a$mz)
  v$iivm_r2_g0 <- .dml_r2(d$yzb[z0], a$g0[z0]); v$iivm_r2_g1 <- .dml_r2(d$yzb[z1], a$g1[z1])
  v$iivm_r2_r0 <- .dml_r2(d$db[z0], a$r0[z0]); v$iivm_r2_r1 <- .dml_r2(d$db[z1], a$r1[z1]); v$iivm_r2_m <- .dml_r2(d$zb, a$mz)
  gZ <- ifelse(z1, a$g1, a$g0); rZ <- ifelse(z1, a$r1, a$r0)   # observed-arm models (NuisanceDiagnostics)
  v$iivm_rmse_outcome <- .dml_rmse(d$yzb - gZ); v$iivm_r2_outcome <- .dml_r2(d$yzb, gZ)
  v$iivm_rmse_treatment <- .dml_rmse(d$db - rZ); v$iivm_r2_treatment <- .dml_r2(d$db, rZ); v$iivm_mean_resid_treatment <- mean(d$db - rZ)
  inf <- .dml_inf(a$theta, a$se); v$iivm_ci95 <- inf$ci; v$iivm_p <- inf$p
  # IIVM under one-sided noncompliance: treatment db * zb (no unit with Z = 0 is treated), so the
  # r0 training outcome is constant 0 in every fold and r0 = 0 (DoubleML's subgroups = list(always_takers = FALSE))
  D1 <- d$db * d$zb; stopifnot(all(D1[z0] == 0))
  b <- iivm(D1, onesided = TRUE)
  v$iivm_onesided_theta <- b$theta; v$iivm_onesided_se <- b$se; v$iivm_onesided_jacobian <- b$jacobian
  v$fold_sizes <- vec(as.numeric(table(f)))
  list(values = v, reference = "DoubleML (Chernozhukov et al. 2018; Bach et al. 2022, 2024) PLR / IRM / PLIV / IIVM scores with fixed folds, written out (DoubleML not installed); glmnet lambda = 0.05, thresh 1e-14; exact glmnet minimisers (*_tight)",
       call = "cross-fitting with K = 5 folds from the `fold` column", packages = c("glmnet"))
})
