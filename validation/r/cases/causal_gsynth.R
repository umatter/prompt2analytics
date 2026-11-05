# validation/r/cases/causal_gsynth.R -- generalized synthetic control references (T3-WP31: CAUS-44;
# moved in CAUS-41, CAUS-43). Package copy of design Appendix A.27 (canonical; same case id, dataset and
# keys, plus the keys listed under "added").
#
# R reference: gsynth 1.4.0, a wrapper of fect 2.4.5. gsynth's default estimator "gsynth" is Xu (2017):
# interactive fixed effects estimated on the never-treated units only (fect method "gsynth"), each
# treated unit's loadings (and, with unit effects, its alpha) by OLS on its own pre-treatment residuals.
# estimator = "mc" is matrix completion (Athey et al. 2021) on all untreated cells. (gsynth 1.4's
# estimator = "ife" is the EM variant; p2a does not implement it.) se = FALSE except in the bootstrap keys.
# Every pinned fit uses tol = 1e-12: gsynth iterates to tol = 1e-3 by default (an absolute bound on
# ||beta_k - beta_{k-1}||, T2 design 2.3); the default-tol fit is recorded as a cross-check.
#
# Written-out references (the .gs_* / .gsmc_* functions below) reproduce gsynth/fect and are checked against
# them inside each case (stopifnot): IFE to 1e-12, MC to 1e-8, both bootstraps on gsynth's own random
# draws (fect_boot's sequential path, set.seed(seed + 1)). They provide what gsynth does not return or
# does not make reproducible:
#   * the LOO cross-validation MSPE at full convergence (fect cv.method = "loo" fits at max(tol, 1e-3));
#   * matrix-completion cross-validation on deterministic folds (fect folds are random): fold f holds out
#     every treated unit's pre-treatment cells at positions p (0-based in its pre-window) with p %% k == f;
#     every fit starts from the lambda = Inf fit of its fold (two-way FE + X OLS on the observed cells,
#     fect's initialFit; soft-impute with L = 0 from the full-data one, relative change below 1e-14) and
#     stops at a relative change of the fitted matrix below 1e-12 (mc_cv_user_mspe, the solution; the
#     iteration cap 1e5 is never reached) or below 1e-8 / after 1000 iterations (mc_cv_user_mspe_tol8,
#     with mc_cv_user_gap their largest relative difference); the lambda with the smallest pooled MSPE is
#     selected (first in grid order on ties; fect's default cv.rule = "1se" scores time-weighted
#     residuals on its random folds and is not reproduced);
#   * the explicit bootstrap draws, so that p2a can replay gsynth's bootstrap exactly.
# added keys: *_count, *_rmse, *_att_unit, ife_loo_*, mc_l001_cov_*, mc_eigen_max, mc_grid, mc_cv_user_*,
# boot_*, and the case gsynth_staggered (staggered adoption, the CAUS-41 failure mode).
suppressPackageStartupMessages({ library(gsynth); library(fect) })

.gs_panel <- function(d, xcols) {
  ids <- sort(unique(d$id)); times <- sort(unique(d$time))
  TT <- length(times); N <- length(ids)
  idx <- cbind(match(d$time, times), match(d$id, ids))
  m <- function(v) { z <- matrix(NA_real_, TT, N); z[idx] <- d[[v]]; z }
  list(Y = m("Y"), D = m("D"), X = lapply(xcols, m), TT = TT, N = N, ids = ids, times = times)
}
.gs_demean <- function(M, force) {   # force: 0 none, 1 unit, 2 time, 3 two-way (fect coding)
  mu <- mean(M); out <- M - mu
  if (force %in% c(1, 3)) out <- out - matrix(colMeans(M) - mu, nrow(M), ncol(M), byrow = TRUE)
  if (force %in% c(2, 3)) out <- out - matrix(rowMeans(M) - mu, nrow(M), ncol(M))
  out
}
.gs_pca <- function(U, r) {           # F'F / T = I, Lambda = U'F / T
  TT <- nrow(U)
  if (r == 0) return(list(F = matrix(0, TT, 0), L = matrix(0, ncol(U), 0), FE = U * 0))
  s <- svd(U)
  F <- sqrt(TT) * s$u[, 1:r, drop = FALSE]
  L <- t(U) %*% F / TT
  list(F = F, L = L, FE = F %*% t(L))
}
# Bai (2009) IFE on the controls (fect inter_fe): beta from the two-way within OLS, then alternate
# factors | beta and beta | factors until ||beta_k - beta_{k-1}||_2 < tol.
.gs_ife <- function(Y, X, r, force, tol, maxit = 5000) {
  p <- length(X); TT <- nrow(Y); N <- ncol(Y)
  YY <- .gs_demean(Y, force)
  XX <- if (p) sapply(X, function(x) c(.gs_demean(x, force))) else NULL
  beta <- numeric(0); it <- 0
  if (p) {
    beta <- c(solve(crossprod(XX), crossprod(XX, c(YY))))
    repeat {
      it <- it + 1
      pf <- .gs_pca(YY - matrix(XX %*% beta, TT, N), r)
      b <- c(solve(crossprod(XX), crossprod(XX, c(YY - pf$FE))))
      ch <- sqrt(sum((b - beta)^2)); beta <- b
      if (ch < tol || it >= maxit) break
    }
  }
  R <- Y; for (j in seq_len(p)) R <- R - X[[j]] * beta[j]
  pf <- .gs_pca(.gs_demean(R, force), r)
  mu <- mean(R)
  xi <- if (force %in% c(2, 3)) rowMeans(R) - mu else rep(0, TT)
  al <- if (force %in% c(1, 3)) colMeans(R) - mu else rep(0, N)
  fit <- mu + matrix(al, TT, N, byrow = TRUE) + matrix(xi, TT, N) + pf$FE
  nfe <- c(1, N, TT, N + TT - 1)[force + 1]; ndim <- c(0, 1, 1, 2)[force + 1]
  df <- TT * N - nfe - p - r * (N + TT - ndim) + r^2
  list(beta = beta, mu = mu, xi = xi, F = pf$F, e = R - fit, fit = fit, sigma2 = sum((R - fit)^2) / df, it = it)
}
# the gsynth estimator on a panel (treated = columns with any D == 1, absorbing)
.gs_estimate <- function(Y, D, X, r, force, tol) {
  tr <- which(colSums(D) > 0); co <- which(colSums(D) == 0); TT <- nrow(Y)
  est <- .gs_ife(Y[, co, drop = FALSE], lapply(X, function(x) x[, co, drop = FALSE]), r, force, tol)
  U <- Y[, tr, drop = FALSE]
  for (j in seq_along(X)) U <- U - X[[j]][, tr, drop = FALSE] * est$beta[j]
  U <- U - est$mu
  if (force %in% c(2, 3)) U <- U - est$xi
  Fh <- est$F; if (force %in% c(1, 3)) Fh <- cbind(Fh, 1)
  eff <- U
  for (j in seq_along(tr)) {
    pre <- which(D[, tr[j]] == 0)
    if (ncol(Fh) > 0) eff[, j] <- U[, j] - Fh %*% solve(crossprod(Fh[pre, , drop = FALSE]), crossprod(Fh[pre, , drop = FALSE], U[pre, j]))
  }
  Dt <- D[, tr, drop = FALSE]
  list(att.avg = sum(eff * Dt) / sum(Dt), eff = eff, est = est, tr = tr, co = co,
       rmse = sqrt(mean(eff[Dt == 0]^2)), att_unit = colSums(eff * Dt) / colSums(Dt))
}
# leave-one-pre-period-out CV over the treated units (Xu 2017; fect cv.method = "loo"); mu is removed
# for every force (fect's loo keeps mu for force = "none"; not used here)
.gs_loo <- function(Y, D, X, r, force, tol) {
  tr <- which(colSums(D) > 0); co <- which(colSums(D) == 0)
  est <- .gs_ife(Y[, co, drop = FALSE], lapply(X, function(x) x[, co, drop = FALSE]), r, force, tol)
  U <- Y[, tr, drop = FALSE]
  for (j in seq_along(X)) U <- U - X[[j]][, tr, drop = FALSE] * est$beta[j]
  U <- U - est$mu
  if (force %in% c(2, 3)) U <- U - est$xi
  Fh <- est$F; if (force %in% c(1, 3)) Fh <- cbind(Fh, 1)
  pres <- lapply(seq_along(tr), function(j) which(D[, tr[j]] == 0))
  res <- c()
  for (lv in sort(unique(unlist(pres)))) for (j in seq_along(tr)) {
    if (!(lv %in% pres[[j]])) next
    keep <- setdiff(pres[[j]], lv)
    res <- c(res, if (ncol(Fh) == 0) U[lv, j] else U[lv, j] - sum(Fh[lv, ] * solve(crossprod(Fh[keep, , drop = FALSE]), crossprod(Fh[keep, , drop = FALSE], U[keep, j]))))
  }
  mean(res^2)
}
# matrix completion: min 1/2 sum_obs (Y - mu - a_i - x_t - X b - L)^2 + lambda T N ||L||_*  by soft-impute
.gsmc_svt <- function(E, tau) {
  s <- svd(E); dd <- pmax(s$d - tau, 0); k <- sum(dd > 0)
  if (k == 0) return(E * 0)
  s$u[, 1:k, drop = FALSE] %*% (dd[1:k] * t(s$v[, 1:k, drop = FALSE]))
}
.gsmc_fit <- function(Y, X, II, lambda, force, tol, maxit, start) {
  TT <- nrow(Y); N <- ncol(Y); p <- length(X)
  XX <- if (p) sapply(X, function(x) c(.gs_demean(x, force))) else NULL
  fit <- start; L <- matrix(0, TT, N); beta <- numeric(0); it <- 0; conv <- FALSE
  repeat {
    it <- it + 1
    Z <- Y * II + fit * (1 - II)
    R <- Z - L
    if (p) beta <- c(solve(crossprod(XX), crossprod(XX, c(.gs_demean(R, force)))))
    Xb <- matrix(0, TT, N); for (j in seq_len(p)) Xb <- Xb + X[[j]] * beta[j]
    Rb <- R - Xb; mu <- mean(Rb)
    al <- if (force %in% c(1, 3)) colMeans(Rb) - mu else rep(0, N)
    xi <- if (force %in% c(2, 3)) rowMeans(Rb) - mu else rep(0, TT)
    FE <- mu + matrix(al, TT, N, byrow = TRUE) + matrix(xi, TT, N)
    L <- if (is.finite(lambda)) .gsmc_svt(Z - FE - Xb, lambda * TT * N) else matrix(0, TT, N)
    new <- FE + Xb + L
    ch <- sqrt(sum((new - fit)^2)) / sqrt(sum(fit^2)); fit <- new
    if (ch < tol) { conv <- TRUE; break }
    if (it >= maxit) break
  }
  list(fit = fit, beta = beta, mu = mu, L = L, it = it, conv = conv)
}
.gsmc_start <- function(Y, X, II, force) .gsmc_fit(Y, X, II, Inf, force, 1e-14, 10000, matrix(mean(Y[II == 1]), nrow(Y), ncol(Y)))$fit
.gsmc_grid <- function(Y, II, start, nlambda) {   # fect_cv's grid
  eig <- svd((Y - start) * II / (nrow(Y) * ncol(Y)))$d
  lmax <- log10(max(eig)); by <- 3 / (nlambda - 2)
  list(grid = c(10^(lmax - (0:(nlambda - 2)) * by), 0), eig_max = max(eig))
}
.gsmc_cv <- function(Y, D, X, force, grid, k, tol = 1e-8, maxit = 1000, need_conv = FALSE) {
  II <- 1 - D; tr <- which(colSums(D) > 0)
  pres <- lapply(tr, function(i) which(D[, i] == 0))
  st0 <- .gsmc_start(Y, X, II, force)
  sapply(grid, function(lam) {
    res <- c()
    for (f in 0:(k - 1)) {
      IIcv <- II; held <- NULL
      for (j in seq_along(tr)) { h <- pres[[j]][((seq_along(pres[[j]]) - 1) %% k) == f]; IIcv[h, tr[j]] <- 0; held <- rbind(held, cbind(h, tr[j])) }
      st <- .gsmc_fit(Y, X, IIcv, Inf, force, 1e-14, 10000, st0)$fit   # lambda = Inf fit of the fold, from the full-data one
      m <- .gsmc_fit(Y, X, IIcv, lam, force, tol, maxit, st)
      if (need_conv) stopifnot(m$conv)
      res <- c(res, Y[held] - m$fit[held])
    }
    mean(res^2)
  })
}
.gs_values <- function(g, prefix) {
  out <- list()
  out[[paste0(prefix, "_att_avg")]] <- unname(g$att.avg)
  out[[paste0(prefix, "_att")]] <- vec(unname(g$att))
  out[[paste0(prefix, "_time")]] <- vec(as.numeric(g$time))
  out[[paste0(prefix, "_count")]] <- vec(as.numeric(g$count))
  if (!is.null(g$beta) && !all(is.na(g$beta))) out[[paste0(prefix, "_beta")]] <- vec(as.numeric(g$beta))
  if (!is.null(g$r.cv)) out[[paste0(prefix, "_r")]] <- unname(g$r.cv)
  if (!is.null(g$mu)) out[[paste0(prefix, "_mu")]] <- unname(g$mu)
  out[[paste0(prefix, "_sigma2")]] <- unname(g$sigma2)
  out[[paste0(prefix, "_rmse")]] <- unname(g$rmse)
  Dm <- g$D.dat; tr <- which(colSums(Dm) > 0)   # g$D is the variable name; D.dat is the T x N matrix
  out[[paste0(prefix, "_att_unit")]] <- vec(unname(colSums(g$eff[, tr, drop = FALSE] * Dm[, tr, drop = FALSE]) / colSums(Dm[, tr, drop = FALSE])))
  out
}
.gs_fit <- function(f, d, ...) suppressMessages(gsynth(f, data = d, index = c("id", "time"), se = FALSE, ...))
.gs_close <- function(a, b, tol) isTRUE(all(abs(a - b) <= tol * pmax(1, abs(b))))

case("gsynth_simdata", "econometrics/synth", "gsynth IFE (r = 2 fixed; LOO CV over r = 0..5), with and without covariates, MC (fixed and cross-validated lambda), bootstrap draws, on gsynth::simdata", function(ctx) {
  data(gsynth, package = "gsynth", envir = environment())   # loads simdata, turnout
  d <- ctx$gen_dataset("gsynth_simdata", simdata[, c("id", "time", "Y", "D", "X1", "X2")])
  P <- .gs_panel(d, c("X1", "X2")); X2l <- P$X; X0l <- list()
  v <- list()
  g1 <- .gs_fit(Y ~ D, d, force = "two-way", CV = FALSE, r = 2, tol = 1e-12)
  v <- c(v, .gs_values(g1, "ife_r2_nocov"))
  w1 <- .gs_estimate(P$Y, P$D, X0l, 2, 3, 1e-12)
  stopifnot(.gs_close(w1$att.avg, g1$att.avg, 1e-12), .gs_close(w1$est$sigma2, g1$sigma2, 1e-12), .gs_close(w1$rmse, g1$rmse, 1e-12))
  g2 <- .gs_fit(Y ~ D + X1 + X2, d, force = "two-way", CV = FALSE, r = 2, tol = 1e-12)
  v <- c(v, .gs_values(g2, "ife_r2_cov"))
  w2 <- .gs_estimate(P$Y, P$D, X2l, 2, 3, 1e-12)
  stopifnot(.gs_close(w2$att.avg, g2$att.avg, 1e-12), .gs_close(w2$est$beta, c(g2$beta), 1e-12), .gs_close(w2$est$sigma2, g2$sigma2, 1e-12))
  g2d <- .gs_fit(Y ~ D + X1 + X2, d, force = "two-way", CV = FALSE, r = 2)
  v$ife_r2_cov_default_tol_att_avg <- unname(g2d$att.avg); v$ife_r2_cov_default_tol_beta <- vec(as.numeric(g2d$beta))
  stopifnot(.gs_close(.gs_estimate(P$Y, P$D, X2l, 2, 3, 1e-3)$att.avg, g2d$att.avg, 1e-12))   # gsynth's criterion and start
  set.seed(7)   # gsynth 1.4 (fect) cross-validates on random folds: recorded, not pinned by p2a
  g3 <- .gs_fit(Y ~ D + X1 + X2, d, force = "two-way", CV = TRUE, r = c(0, 5), tol = 1e-12)
  v <- c(v, .gs_values(g3, "ife_cv_cov"))
  # LOO CV (fect cv.method = "loo"): deterministic; fect fits the CV models at max(tol, 1e-3)
  for (cv in list(list(x = X2l, f = Y ~ D + X1 + X2, tag = "cov"), list(x = X0l, f = Y ~ D, tag = "nocov"))) {
    fl <- suppressMessages(fect(cv$f, data = d, index = c("id", "time"), method = "gsynth", force = "two-way", CV = TRUE, r = c(0, 5), cv.method = "loo", se = FALSE, tol = 1e-12))
    loose <- sapply(0:5, function(r) .gs_loo(P$Y, P$D, cv$x, r, 3, 1e-3))
    stopifnot(.gs_close(loose, unname(fl$CV.out[, "MSPE"]), 1e-12))
    tight <- sapply(0:5, function(r) .gs_loo(P$Y, P$D, cv$x, r, 3, 1e-13))
    stopifnot(which.min(tight) - 1 == fl$r.cv)
    v[[paste0("ife_loo_", cv$tag, "_r")]] <- unname(fl$r.cv)
    v[[paste0("ife_loo_", cv$tag, "_mspe_fect")]] <- vec(unname(fl$CV.out[, "MSPE"]))
    v[[paste0("ife_loo_", cv$tag, "_mspe")]] <- vec(tight)
    ga <- .gs_fit(cv$f, d, force = "two-way", CV = FALSE, r = unname(fl$r.cv), tol = 1e-12)
    v[[paste0("ife_loo_", cv$tag, "_att_avg")]] <- unname(ga$att.avg)
  }
  # matrix completion, fixed lambda (0.3 exceeds the largest singular value: L = 0, the two-way FE + X fit)
  g4 <- .gs_fit(Y ~ D + X1 + X2, d, force = "two-way", estimator = "mc", CV = FALSE, lambda = 0.3, tol = 1e-12)
  v <- c(v, .gs_values(g4, "mc_l03_cov"))
  g4b <- .gs_fit(Y ~ D + X1 + X2, d, force = "two-way", estimator = "mc", CV = FALSE, lambda = 0.01, tol = 1e-12)
  v <- c(v, .gs_values(g4b, "mc_l001_cov"))
  II <- 1 - P$D; st <- .gsmc_start(P$Y, X2l, II, 3)
  m4 <- .gsmc_fit(P$Y, X2l, II, 0.01, 3, 1e-14, 1e5, st)
  stopifnot(.gs_close(sum((P$Y - m4$fit) * P$D) / sum(P$D), g4b$att.avg, 1e-8), .gs_close(m4$beta, c(g4b$beta), 1e-8))
  sv <- svd(m4$L)$d; v$mc_l001_cov_rank <- sum(sv > 1e-8 * max(sv))
  # matrix completion: fect's lambda grid (nlambda = 10) and p2a's deterministic CV on a 3-value user grid
  gr <- .gsmc_grid(P$Y, II, st, 10)
  stopifnot(.gs_close(gr$eig_max, max(g4$eigen.all), 1e-10))
  v$mc_eigen_max <- gr$eig_max; v$mc_grid <- vec(gr$grid)
  ug <- gr$grid[1:3]
  # at the solution (every fold fit to a relative change below 1e-12; the problem is convex, so the start
  # does not matter there) and, as a record of the gap, at tolerance 1e-8 with at most 1000 iterations
  ms <- .gsmc_cv(P$Y, P$D, X2l, 3, ug, k = 5, tol = 1e-12, maxit = 1e5, need_conv = TRUE)
  ms8 <- .gsmc_cv(P$Y, P$D, X2l, 3, ug, k = 5)
  v$mc_cv_user_grid <- vec(ug); v$mc_cv_user_mspe <- vec(ms); v$mc_cv_user_mspe_tol8 <- vec(ms8)
  v$mc_cv_user_gap <- max(abs(ms8 - ms) / abs(ms))
  lsel <- ug[which.min(ms)]; stopifnot(lsel == ug[which.min(ms8)]); v$mc_cv_user_lambda <- lsel
  ms_fit <- .gsmc_fit(P$Y, X2l, II, lsel, 3, 1e-14, 1e5, st)
  v$mc_cv_user_att_avg <- sum((P$Y - ms_fit$fit) * P$D) / sum(P$D)
  g5 <- .gs_fit(Y ~ D, d, force = "unit", CV = FALSE, r = 2, tol = 1e-12)
  v <- c(v, .gs_values(g5, "ife_r2_unit"))
  # parametric bootstrap (gsynth's default for estimator "gsynth"): replay gsynth's draws
  gb <- suppressWarnings(suppressMessages(gsynth(Y ~ D, data = d, index = c("id", "time"), force = "two-way", CV = FALSE, r = 2,
          se = TRUE, inference = "parametric", nboots = 50, parallel = FALSE, seed = 1, tol = 1e-12)))
  B <- 50; tr <- which(colSums(P$D) > 0); co <- which(colSums(P$D) == 0); Ntr <- length(tr); Nco <- length(co)
  base <- .gs_estimate(P$Y, P$D, X0l, 2, 3, 1e-12)
  Yct <- P$Y; Yct[, tr] <- P$Y[, tr] - base$eff; Yct[, co] <- base$est$fit
  set.seed(1 + 1)
  fake <- integer(B); pseudo <- matrix(0L, B, Nco); err <- array(NA_real_, c(P$TT, Ntr, B))
  for (b in 1:B) {
    fake[b] <- sample(co, 1, replace = FALSE); pseudo[b, ] <- sample(co[co != fake[b]], Nco, replace = TRUE)
    id <- c(rep(fake[b], Ntr), pseudo[b, ])
    err[, , b] <- .gs_estimate(P$Y[, id], P$D[, c(tr, pseudo[b, ])], X0l, 2, 3, 1e-12)$eff
  }
  bco <- matrix(0L, B, Nco); pick <- matrix(0L, B, Ntr); res <- matrix(0L, B, Nco); att_b <- numeric(B)
  for (b in 1:B) {
    bco[b, ] <- sample(co, Nco, replace = TRUE)
    for (w in 1:Ntr) pick[b, w] <- sample(1:B, 1, replace = TRUE)
    res[b, ] <- sample(1:Nco, Nco, replace = TRUE)
    id <- c(tr, bco[b, ]); Yb <- Yct[, id]
    for (w in 1:Ntr) Yb[, w] <- Yb[, w] + err[, w, pick[b, w]]
    Yb[, Ntr + 1:Nco] <- Yb[, Ntr + 1:Nco] + base$est$e[, res[b, ]]
    att_b[b] <- .gs_estimate(Yb, P$D[, id], X0l, 2, 3, 1e-12)$att.avg
  }
  stopifnot(.gs_close(att_b, c(gb$att.avg.boot), 1e-10), .gs_close(sd(att_b), gb$est.avg[1, "S.E."], 1e-10))
  v$boot_ife_nocov_r2_se <- unname(gb$est.avg[1, "S.E."])
  v$boot_ife_nocov_r2_ci <- vec(unname(gb$est.avg[1, c("CI.lower", "CI.upper")]))
  v$boot_ife_nocov_r2_p <- unname(gb$est.avg[1, "p.value"])
  v$boot_ife_att_se <- vec(unname(gb$est.att[, "S.E."]))   # per event time (gsynth est.att), same order as *_time
  v$boot_ife_att_b <- vec(att_b)
  v$boot_ife_fake <- vec(fake); v$boot_ife_pseudo <- vec(c(t(pseudo))); v$boot_ife_bco <- vec(c(t(bco)))
  v$boot_ife_pick <- vec(c(t(pick))); v$boot_ife_res <- vec(c(t(res)))
  # nonparametric unit bootstrap (gsynth's inference for estimator "mc"), lambda = 0.01
  gm <- suppressWarnings(suppressMessages(gsynth(Y ~ D + X1 + X2, data = d, index = c("id", "time"), force = "two-way", estimator = "mc",
          CV = FALSE, lambda = 0.01, se = TRUE, nboots = 20, parallel = FALSE, seed = 1, tol = 1e-12)))
  Bm <- 20; set.seed(1 + 1); units <- matrix(0L, Bm, P$N); attm <- numeric(Bm)
  for (b in 1:Bm) {
    fco <- sample(co, Nco, replace = TRUE); ftr <- sample(tr, Ntr, replace = TRUE); id <- c(ftr, fco); units[b, ] <- id
    Xb <- lapply(X2l, function(x) x[, id]); IIb <- 1 - P$D[, id]
    m <- .gsmc_fit(P$Y[, id], Xb, IIb, 0.01, 3, 1e-14, 1e5, .gsmc_start(P$Y[, id], Xb, IIb, 3))
    attm[b] <- sum((P$Y[, id] - m$fit) * P$D[, id]) / sum(P$D[, id])
  }
  stopifnot(.gs_close(attm, c(gm$att.avg.boot), 1e-8))
  v$boot_mc_l001_se <- unname(gm$est.avg[1, "S.E."]); v$boot_mc_att_b <- vec(attm); v$boot_mc_units <- vec(c(t(units)))
  v$boot_mc_l001_ci <- vec(unname(gm$est.avg[1, c("CI.lower", "CI.upper")])); v$boot_mc_l001_p <- unname(gm$est.avg[1, "p.value"])
  list(values = v, reference = "gsynth::gsynth(force = 'two-way' | 'unit', estimator = 'gsynth' | 'mc', CV, r, lambda, se, inference); fect::fect(method = 'gsynth', cv.method = 'loo'); written-out IFE, MC, LOO and k-fold MC CV checked against them",
       call = "gsynth(Y ~ D + X1 + X2, data = simdata, index = c('id','time'), force = 'two-way', ...)", packages = c("gsynth", "fect"))
})

case("gsynth_staggered", "econometrics/synth", "gsynth with staggered adoption (treated from t = 11, 13, 15), two-way and unit effects, one covariate, LOO CV", function(ctx) {
  set.seed(4231)
  Nco <- 30; Ntr <- 6; TT <- 20; adopt <- c(11, 11, 13, 13, 15, 15); N <- Nco + Ntr
  f1 <- cumsum(rnorm(TT, 0, 0.6)); f2 <- sin((1:TT) / 3) + rnorm(TT, 0, 0.2)
  xi <- 0.25 * (1:TT) + rnorm(TT, 0, 0.3)
  lam1 <- c(rnorm(Nco), rnorm(Ntr, 0.8, 0.5)); lam2 <- c(rnorm(Nco), rnorm(Ntr, -0.5, 0.5))
  alpha <- c(rnorm(Nco), rnorm(Ntr, 1, 0.5))
  rows <- vector("list", N * TT); k <- 0
  for (i in 1:N) for (t in 1:TT) {
    dd <- if (i > Nco && t >= adopt[i - Nco]) 1 else 0
    eff <- if (dd == 1) 1 + 0.5 * (t - adopt[i - Nco]) else 0
    x <- 0.5 * lam1[i] * f1[t] + rnorm(1)
    k <- k + 1
    rows[[k]] <- data.frame(id = i, time = t, Y = 2 + alpha[i] + xi[t] + lam1[i] * f1[t] + lam2[i] * f2[t] + 1.5 * x + eff + rnorm(1), D = dd, X = x)
  }
  d <- ctx$gen_dataset("gsynth_staggered", do.call(rbind, rows))
  P <- .gs_panel(d, "X"); v <- list()
  g1 <- .gs_fit(Y ~ D + X, d, force = "two-way", CV = FALSE, r = 2, tol = 1e-12)
  v <- c(v, .gs_values(g1, "tw_r2_cov"))
  w1 <- .gs_estimate(P$Y, P$D, P$X, 2, 3, 1e-12)
  stopifnot(.gs_close(w1$att.avg, g1$att.avg, 1e-12), .gs_close(w1$att_unit, c(v$tw_r2_cov_att_unit), 1e-12))
  g2 <- .gs_fit(Y ~ D, d, force = "two-way", CV = FALSE, r = 2, tol = 1e-12)
  v <- c(v, .gs_values(g2, "tw_r2_nocov"))
  g3 <- .gs_fit(Y ~ D + X, d, force = "unit", CV = FALSE, r = 2, tol = 1e-12)
  v <- c(v, .gs_values(g3, "unit_r2_cov"))
  stopifnot(.gs_close(.gs_estimate(P$Y, P$D, P$X, 2, 1, 1e-12)$att.avg, g3$att.avg, 1e-12))
  fl <- suppressMessages(fect(Y ~ D + X, data = d, index = c("id", "time"), method = "gsynth", force = "two-way", CV = TRUE, r = c(0, 5), cv.method = "loo", se = FALSE, tol = 1e-12))
  stopifnot(.gs_close(sapply(0:5, function(r) .gs_loo(P$Y, P$D, P$X, r, 3, 1e-3)), unname(fl$CV.out[, "MSPE"]), 1e-12))
  tight <- sapply(0:5, function(r) .gs_loo(P$Y, P$D, P$X, r, 3, 1e-13))
  stopifnot(which.min(tight) - 1 == fl$r.cv)
  v$loo_cov_r <- unname(fl$r.cv); v$loo_cov_mspe_fect <- vec(unname(fl$CV.out[, "MSPE"])); v$loo_cov_mspe <- vec(tight)
  v$true_att <- mean(with(d[d$D == 1, ], 1 + 0.5 * (time - adopt[id - Nco])))
  list(values = v, reference = "gsynth::gsynth(force = 'two-way' | 'unit', se = FALSE); fect::fect(method = 'gsynth', cv.method = 'loo')",
       call = "gsynth(Y ~ D + X, data = d, index = c('id','time'), force = 'two-way', CV = FALSE, r = 2, tol = 1e-12)", packages = c("gsynth", "fect"))
})
