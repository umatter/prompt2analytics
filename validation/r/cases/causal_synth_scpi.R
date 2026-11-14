# validation/r/cases/causal_synth_scpi.R -- synthetic-control weights as the exact simplex QP and the SCPI
# quantities p2a computes (T3-WP31: CAUS-40, CAUS-88; X-12, X-14, X-3 and CAUS-77 parts; closes CAUS-80
# and CAUS-82 as side effects). Planner case of T3-WP31 (canonical).
#
# The reference is the QP itself, as Abadie, Diamond and Hainmueller (2010) define the weights for a given V:
#   W*(V) = argmin_w (X1 - X0 w)' V (X1 - X0 w)  s.t. w >= 0, sum(w) = 1,
# with X1, X0 the pre-period means of the predictors (p2a's PredictorSpec default), each predictor divided
# by its standard deviation across the treated unit and the donors, as R Synth 1.1-10's synth() does
# (divisor <- sqrt(apply(cbind(X0, X1), 1, var))). Synth solves the QP with kernlab::ipop, an interior-point
# method stopped at a margin, so its weights are not exact; the reference here is the exact optimum.
# H = X0' V X0 has rank <= k < J, so the objective gets the tie-breaking ridge delta * I,
# delta = 1e-10 * trace(H) / J; the problem is then strictly convex and its minimiser is invariant to
# rescaling all predictors (or V) by a common factor. quadprog::solve.QP finds the active set; the weights
# are then solved exactly on the free set (bordered KKT system) and the KKT conditions are asserted.
# sv2_start is the second start of Synth's data-driven V search ("regression-based" V): with Xs the
# standardised predictors (treated first) and Z the pre-period outcomes, Beta = solve(Xall'Xall) Xall' Z'
# for Xall = cbind(1, t(Xs)), and SV2 = diag(B B') / sum(diag(B B')) with B = Beta without its intercept
# row; transcribed from Synth 1.1-10's synth() source (printed in R 4.3.3) and checked against it below.
#
# SCPI quantities (scpi_quantities): the weights, synthetic path, effects and pre-treatment fit of
# run_scpi on the pre-period outcomes (same QP, H = Y0'Y0). The variance formulas this case used to
# write out were p2a's own heuristic interval, replaced by scpi's construction (CAUS-32).
#
# SCPI intervals (scpi_default_intervals): scpi 3.0.1's scpi() with its defaults (simplex weights,
# u.missp = TRUE, u.sigma = "HC1", u.order = 1, u.lags = 0, e.method = "all", e.order = 1,
# e.lags = 0, u.alpha = e.alpha = 0.05, sims = 200, rho = NULL -> "type-2", rho.max = 0.2) on the
# outcome alone (scdata: constant = FALSE, cointegrated.data = FALSE). scpi 4.x needs R >= 4.4; 3.0.1
# is the CRAN archive version for R 4.3.3. The in-sample bounds are the u.alpha / 2 and
# 1 - u.alpha / 2 quantiles (type 7) over the simulations of -max and -min of p_t' delta subject to
# delta' Q delta - 2 G' delta <= 0, sum(delta) = 0 and w + delta >= lb (ECOS); G = Sigma^(1/2) z with
# z the standard normal draws scpi takes with rnorm() after set.seed() and scest(), written to the case
# (`draws`, J x sims, column-major) so the comparison does not depend on R's generator. The
# out-of-sample bounds are scpi.out() at e.alpha / 2 (gaussian, ls) and the qreg bounds; scpi's
# `bounds$<method>` are the in-sample plus the out-of-sample bounds. The in-sample programs are also
# re-solved accurately (`insample_tight_*`, see .scpi_insample_tight below).
suppressPackageStartupMessages(library(quadprog))

.sc_simplex_qp <- function(H, cvec, ridge = 1e-10) {
  J <- ncol(H); sc <- sum(diag(H)) / J; stopifnot(sc > 0)
  G <- H / sc + ridge * diag(J); cvec <- cvec / sc     # scale-free: (H, c) and (s H, s c) give the same weights
  delta <- ridge * sc
  sol <- solve.QP(Dmat = G, dvec = cvec, Amat = cbind(rep(1, J), diag(J)), bvec = c(1, rep(0, J)), meq = 1)$solution
  Fs <- which(sol > 1e-9 * max(sol))
  K <- rbind(cbind(G[Fs, Fs, drop = FALSE], 1), c(rep(1, length(Fs)), 0))
  z <- solve(K, c(cvec[Fs], 1))
  w <- rep(0, J); w[Fs] <- z[seq_along(Fs)]; mu <- z[length(Fs) + 1]
  eta <- c(G %*% w - cvec) + mu
  stopifnot(all(w[Fs] > 0), all(eta[-Fs] >= -1e-12 * max(abs(G))))
  list(w = w, delta = delta, free = length(Fs))
}

case("synth_weights_qp", "econometrics/synth", "Synthetic-control unit weights as the exact simplex QP on standardised predictors (equal and custom V; predictors in units and x 1e4); Synth's regression-based V start", function(ctx) {
  set.seed(3101)
  J <- 8; TT <- 20; T0 <- 14
  donors <- sprintf("D%02d", 1:J); units <- c(donors, "TR")
  lv <- cbind(rnorm(J, 5, 1), rnorm(J, 20, 4), runif(J, 0.2, 0.8))
  mix <- c(0, 0.5, 0, 0, 0.3, 0, 0.2, 0); off <- c(0.08, -0.3, 0.01)   # treated just outside the donors' hull
  outcome <- function(x, t) 1 + 0.8 * x[, 1] + 0.25 * x[, 2] + 4 * x[, 3] + 0.1 * t   # predictors drive the outcome
  rows <- list()
  for (t in 1:TT) {
    xd <- sweep(lv, 2, c(0.05, 0.2, 0.002) * t, "+") + matrix(rnorm(3 * J, 0, c(0.1, 0.4, 0.01)), J, 3, byrow = TRUE)
    yd <- outcome(xd, t) + rnorm(J, 0, 0.3)
    xt <- colSums(mix * xd) + off + rnorm(3, 0, c(0.05, 0.2, 0.005))
    yt <- outcome(matrix(xt, 1), t) + rnorm(1, 0, 0.3) + if (t > T0) 2 else 0
    rows[[t]] <- data.frame(unit = units, time = t, y = c(yd, yt), x1 = c(xd[, 1], xt[1]), x2 = c(xd[, 2], xt[2]), x3 = c(xd[, 3], xt[3]))
  }
  d0 <- do.call(rbind, rows); d0 <- d0[order(d0$unit, d0$time), ]
  for (v in c("x1", "x2", "x3")) d0[[paste0(v, "_big")]] <- d0[[v]] * 1e4
  d <- ctx$gen_dataset("synth_qp_panel", d0)
  pre <- d$time < 15
  pm <- function(col) { m <- tapply(d[[col]][pre], d$unit[pre], mean); m[units] }
  # Synth's standardisation: each predictor divided by its SD across all units (n - 1 denominator);
  # the predictor balance (equal_balance_synthetic) is on the predictors' own scale
  stdz <- function(X) { stopifnot(all(apply(X[, donors], 1, sd) > 0)); X / sqrt(apply(X, 1, var)) }
  Xraw <- rbind(pm("x1"), pm("x2"), pm("x3")); rownames(Xraw) <- c("x1", "x2", "x3")
  Xall <- stdz(Xraw)
  Xall_big <- stdz(rbind(pm("x1_big"), pm("x2_big"), pm("x3_big")))
  Xall_x2 <- stdz(rbind(pm("x1"), pm("x2_big"), pm("x3")))   # only x2 rescaled
  Ymat <- sapply(units, function(u) d$y[d$unit == u][order(d$time[d$unit == u])])   # TT x (J + 1)
  fit <- function(X, v) {
    X0 <- X[, donors]; X1 <- X[, "TR"]
    H <- t(X0) %*% (v * X0); cv <- c(t(X0) %*% (v * X1))
    q <- .sc_simplex_qp(H, cv); w <- q$w; names(w) <- donors
    syn <- c(Ymat[, donors] %*% w)
    gap <- Ymat[, "TR"] - syn
    list(w = w, loss = sum(v * (X1 - X0 %*% w)^2), pre_mspe = mean(gap[1:T0]^2), effects = gap[(T0 + 1):TT],
         synthetic = syn, free = q$free)
  }
  ve <- rep(1 / 3, 3); vc <- c(0.6, 0.3, 0.1)
  e <- fit(Xall, ve); c2 <- fit(Xall, vc); eb <- fit(Xall_big, ve); ex <- fit(Xall_x2, ve)
  stopifnot(max(abs(eb$w - e$w)) < 1e-10, abs(eb$loss / e$loss - 1) < 1e-10, max(abs(ex$w - e$w)) < 1e-10, e$loss > 0, e$free <= 4)
  # Synth's regression-based start (SV2), from the standardised predictors and the pre-period outcomes
  Xs <- cbind(Xall[, "TR"], Xall[, donors]); Z <- cbind(Ymat[1:T0, "TR"], Ymat[1:T0, donors])
  XA <- cbind(1, t(Xs))
  Beta <- try(solve(crossprod(XA), crossprod(XA, t(Z))), silent = TRUE)
  sv2 <- if (inherits(Beta, "try-error")) rep(1 / nrow(Xs), nrow(Xs)) else { B <- Beta[-1, , drop = FALSE]; dd <- rowSums(B^2); dd / sum(dd) }
  # the same quantity as Synth's own code computes it (Beta <- solve(t(Xall) %*% Xall) %*% t(Xall) %*% t(Zall))
  Bs <- (solve(t(XA) %*% XA) %*% t(XA) %*% t(Z))[-1, ]; Vs <- diag(Bs %*% t(Bs)); stopifnot(max(abs(Vs / sum(Vs) - sv2)) < 1e-10)
  list(values = list(
    equal_w = named(e$w), equal_predictor_loss = e$loss, equal_pre_mspe = e$pre_mspe, equal_effects = vec(e$effects),
    equal_average_effect = mean(e$effects), equal_synthetic = vec(e$synthetic), equal_balance_synthetic = vec(c(Xraw[, donors] %*% e$w)),
    equal_n_positive = e$free,
    custom_w = named(c2$w), custom_predictor_loss = c2$loss, custom_pre_mspe = c2$pre_mspe, custom_average_effect = mean(c2$effects),
    equal_big_w = named(eb$w), equal_big_predictor_loss = eb$loss,
    sv2_start = named(setNames(sv2, rownames(Xraw)))),
    reference = "exact simplex QP (quadprog::solve.QP + KKT polish) on predictors standardised as Synth 1.1-10, Abadie et al. (2010) weights for a given V; Synth's regression-based V start",
    call = "solve.QP(Dmat = X0'VX0 + delta I, dvec = X0'VX1, Amat = cbind(1, I), bvec = c(1, 0, ..), meq = 1)", packages = "quadprog")
})

case("scpi_quantities", "econometrics/scpi", "p2a SCPI: simplex weights of the pre-period outcomes, synthetic path, effects, pre-treatment fit", function(ctx) {
  d <- ctx$dataset("datasets/generated/synth_qp_panel.csv")
  donors <- sprintf("D%02d", 1:8); T0 <- 14; TT <- 20
  Y <- sapply(c(donors, "TR"), function(u) d$y[d$unit == u][order(d$time[d$unit == u])])
  X <- Y[, donors]; y <- Y[, "TR"]
  w <- .sc_simplex_qp(crossprod(X[1:T0, ]), c(crossprod(X[1:T0, ], y[1:T0])))$w; names(w) <- donors
  syn <- c(X %*% w); e <- y[1:T0] - syn[1:T0]; eff <- y[(T0 + 1):TT] - syn[(T0 + 1):TT]
  tss <- sum((y[1:T0] - mean(y[1:T0]))^2)
  list(values = list(
    w = named(w), synthetic = vec(syn), effect = vec(eff),
    pre_rmspe = sqrt(mean(e^2)), pre_r2 = 1 - sum(e^2) / tss),
    reference = "simplex weights by the exact simplex QP (quadprog + KKT polish)",
    call = "run_scpi(treated = y[TR], donors = y[D01..D08], treatment_period = 14, ...)", packages = "quadprog")
})

# scpi's in-sample programs re-solved accurately. scpi passes them to ECOS in the weights x, with
# the quadratic constraint's constant d = 2 G'w + w'Qw folded in; ECOS then mostly ends with "Close
# to optimal solution found" (its reduced tolerances), about 1e-5 from the optimum. The same
# programs in delta = x - w (min/max p'delta subject to delta'Q delta <= 2 G'delta, sum(delta) =
# 1 - sum(w), delta >= lb - w, the quadratic as a second-order cone) solve to about 1e-8 with ECOS
# at tolerance 1e-12.
.scpi_insample_tight <- function(B, P, w, rho, Sigma, z) {
  J <- ncol(B); Q <- crossprod(B) / nrow(B)
  ev <- eigen(Q, symmetric = TRUE); L <- ev$vectors %*% diag(sqrt(pmax(ev$values, 0)))   # Q = L L'
  zeta <- scpi:::sqrtm(Sigma) %*% z
  lb <- ifelse(w < rho, w, 0)
  # x = (delta, t): 2 G'delta >= t, delta >= lb - w, (1 + t, 1 - t, 2 L'delta) in the cone
  A <- methods::as(matrix(c(rep(1, J), 0), 1), "CsparseMatrix"); b <- 1 - sum(w)
  ctl <- ECOSolveR::ecos.control(maxit = 500L, feastol = 1e-12, reltol = 1e-12, abstol = 1e-12)
  dims <- list(l = 1L + J, q = list(J + 2L), e = 0L)
  lo <- hi <- matrix(NA_real_, ncol(z), nrow(P))
  for (s in seq_len(ncol(z))) {
    G <- c(zeta[, s])
    Gm <- methods::as(rbind(c(-2 * G, 1), cbind(-diag(J), 0), c(rep(0, J), -1), c(rep(0, J), 1),
                            cbind(-2 * t(L), 0)), "CsparseMatrix")
    h <- c(0, w - lb, 1, 1, rep(0, J))
    for (k in seq_len(nrow(P))) for (sg in c(-1, 1)) {
      o <- ECOSolveR::ECOS_csolve(c = c(sg * P[k, ], 0), G = Gm, h = h, dims = dims, A = A, b = b, control = ctl)
      stopifnot(o$infostring %in% c("Optimal solution found", "Close to optimal solution found"))
      v <- -sum(P[k, ] * o$x[1:J])
      if (sg < 0) lo[s, k] <- v else hi[s, k] <- v
    }
  }
  list(lo = lo, hi = hi)
}

case("scpi_default_intervals", "econometrics/scpi", "scpi 3.0.1 scpi() at its defaults: weights, rho, Sigma, pseudo-residual mean, R's standard normal draws, in-sample bounds (scpi's, and its programs re-solved in delta = x - w at ECOS tolerance 1e-12), out-of-sample (gaussian, ls, qreg) bounds and the prediction intervals; u.alpha = e.alpha = 0.025; u.missp = FALSE", function(ctx) {
  d <- ctx$dataset("datasets/generated/synth_qp_panel.csv")
  donors <- sprintf("D%02d", 1:8); J <- 8; sims <- 200; seed <- 8101
  sd0 <- scpi::scdata(df = d, id.var = "unit", time.var = "time", outcome.var = "y", period.pre = 1:14,
                      period.post = 15:20, unit.tr = "TR", unit.co = donors, constant = FALSE,
                      cointegrated.data = FALSE)
  fit <- function(...) {
    set.seed(seed)
    r <- suppressWarnings(scpi::scpi(sd0, w.constr = list(name = "simplex"), sims = sims, cores = 1, verbose = FALSE, ...))
    stopifnot(all(r$inference.results$failed.sims == 0))
    r
  }
  r <- fit(); ra <- fit(u.alpha = 0.025, e.alpha = 0.025); rm <- fit(u.missp = FALSE)
  # the draws scpi took: scest() (CVXR) consumes the generator first, then rnorm(J * sims)
  set.seed(seed); invisible(scpi::scest(sd0, w.constr = list(name = "simplex"))); z <- matrix(rnorm(J * sims), J, sims)
  ir <- r$inference.results; b <- ir$bounds
  w <- c(r$est.results$w); names(w) <- donors
  B <- sd0$B; P <- sd0$P
  tight <- .scpi_insample_tight(B, P, w, ir$rho, ir$Sigma, z)
  tight_m <- .scpi_insample_tight(B, P, w, rm$inference.results$rho, rm$inference.results$Sigma, z)
  q <- function(m, p) vec(apply(m, 2, quantile, probs = p))
  # each bound in scpi's `bounds` is the in-sample plus the out-of-sample one
  e_of <- function(rr, k, col) vec(rr$inference.results$bounds[[k]][, col] - rr$inference.results$bounds$insample[, col])
  ci <- function(rr, k, col) vec(rr$inference.results[[k]][, col])
  list(values = list(
    w = named(w), rho = ir$rho, sigma = vec(c(ir$Sigma)), u_mean = vec(ir$u.mean), draws = vec(c(z)),
    y_post_fit = vec(r$est.results$Y.post.fit),
    insample_lb = vec(b$insample[, 1]), insample_ub = vec(b$insample[, 2]),
    insample_tight_lb = q(tight$lo, 0.025), insample_tight_ub = q(tight$hi, 0.975),
    e_gaussian_lb = e_of(r, "subgaussian", 1), e_gaussian_ub = e_of(r, "subgaussian", 2),
    e_ls_lb = e_of(r, "ls", 1), e_ls_ub = e_of(r, "ls", 2),
    e_qreg_lb = e_of(r, "qreg", 1), e_qreg_ub = e_of(r, "qreg", 2),
    ci_gaussian_lb = ci(r, "CI.all.gaussian", 1), ci_gaussian_ub = ci(r, "CI.all.gaussian", 2),
    ci_ls_lb = ci(r, "CI.all.ls", 1), ci_ls_ub = ci(r, "CI.all.ls", 2),
    ci_qreg_lb = ci(r, "CI.all.qreg", 1), ci_qreg_ub = ci(r, "CI.all.qreg", 2),
    a025_insample_tight_lb = q(tight$lo, 0.0125), a025_insample_tight_ub = q(tight$hi, 0.9875),
    a025_e_gaussian_lb = e_of(ra, "subgaussian", 1), a025_e_gaussian_ub = e_of(ra, "subgaussian", 2),
    a025_e_ls_lb = e_of(ra, "ls", 1), a025_e_ls_ub = e_of(ra, "ls", 2),
    a025_e_qreg_lb = e_of(ra, "qreg", 1), a025_e_qreg_ub = e_of(ra, "qreg", 2),
    a025_ci_gaussian_lb = ci(ra, "CI.all.gaussian", 1), a025_ci_gaussian_ub = ci(ra, "CI.all.gaussian", 2),
    nomissp_sigma = vec(c(rm$inference.results$Sigma)),
    nomissp_insample_lb = vec(rm$inference.results$bounds$insample[, 1]),
    nomissp_insample_ub = vec(rm$inference.results$bounds$insample[, 2]),
    nomissp_insample_tight_lb = q(tight_m$lo, 0.025), nomissp_insample_tight_ub = q(tight_m$hi, 0.975)),
    reference = "scpi::scpi() 3.0.1 defaults (ECOS via CVXR for the weights, ECOSolveR for the in-sample programs; Qtools::rrq for the quantile regressions); set.seed(8101) before each call; the in-sample programs also re-solved in delta = x - w with ECOSolveR at tolerance 1e-12",
    call = "scpi(scdata(df, outcome.var = 'y', period.pre = 1:14, period.post = 15:20, unit.tr = 'TR', constant = FALSE), w.constr = list(name = 'simplex'), sims = 200, cores = 1)",
    packages = c("scpi", "CVXR", "ECOSolveR", "Qtools", "quantreg"))
})
