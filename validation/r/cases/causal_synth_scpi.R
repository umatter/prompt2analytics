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
# SCPI: the scpi package is not installed and p2a's prediction interval is not the Cattaneo-Feng-Titiunik
# construction (CAUS-32, open). The case writes out what p2a computes, so the tests pin p2a's own formulas:
# simplex weights of the pre-period outcomes (same QP, H = Y0'Y0), in-sample variance SSE / T0,
# "gaussian" SSE / (T0 - J), "subgaussian" 1.5 (median|e| / qnorm(0.75))^2, leave-one-out and k-fold
# (contiguous blocks of ceiling(T0 / k)) re-fitted out-of-sample MSE, the critical value qt(1 - alpha/2,
# T0 - 1) and effect +/- crit * sqrt(in + out).
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

case("scpi_quantities", "econometrics/scpi", "p2a SCPI: simplex weights, variance components (in-sample, gaussian, subgaussian, LOO, 5-fold), t critical values, intervals", function(ctx) {
  d <- ctx$dataset("datasets/generated/synth_qp_panel.csv")
  donors <- sprintf("D%02d", 1:8); T0 <- 14; TT <- 20
  Y <- sapply(c(donors, "TR"), function(u) d$y[d$unit == u][order(d$time[d$unit == u])])
  X <- Y[, donors]; y <- Y[, "TR"]
  wfit <- function(rows) { Xs <- X[rows, , drop = FALSE]; .sc_simplex_qp(crossprod(Xs), c(crossprod(Xs, y[rows])))$w }
  w <- wfit(1:T0); names(w) <- donors
  syn <- c(X %*% w); e <- y[1:T0] - syn[1:T0]; eff <- y[(T0 + 1):TT] - syn[(T0 + 1):TT]
  in_var <- sum(e^2) / T0
  var_g <- sum(e^2) / (T0 - 8)
  var_sg <- 1.5 * (median(abs(e)) / qnorm(0.75))^2
  loo <- sapply(1:T0, function(i) { wi <- wfit(setdiff(1:T0, i)); (y[i] - sum(X[i, ] * wi))^2 })
  k <- 5; fs <- ceiling(T0 / k); kf <- c()
  for (f in 0:(k - 1)) {
    test <- (f * fs + 1):min((f + 1) * fs, T0); if (min((f + 1) * fs, T0) < f * fs + 1) next
    wf <- wfit(setdiff(1:T0, test)); kf <- c(kf, (y[test] - c(X[test, , drop = FALSE] %*% wf))^2)
  }
  crit95 <- qt(0.975, T0 - 1); crit90 <- qt(0.95, T0 - 1)
  se_sg <- sqrt(in_var + var_sg)
  tss <- sum((y[1:T0] - mean(y[1:T0]))^2)
  dfs <- c(1, 2, 3, 6, 10, 13, 29, 30, 100)
  list(values = list(
    w = named(w), synthetic = vec(syn), effect = vec(eff), in_sample_var = in_var,
    var_gaussian = var_g, var_subgaussian = var_sg, var_loo = mean(loo), var_kfold5 = mean(kf),
    crit_95 = crit95, crit_90 = crit90, pi_lower_subgaussian = vec(eff - crit95 * se_sg), pi_upper_subgaussian = vec(eff + crit95 * se_sg),
    pi_lower_subgaussian_90 = vec(eff - crit90 * se_sg),
    pre_rmspe = sqrt(mean(e^2)), pre_r2 = 1 - sum(e^2) / tss,
    qt_df = vec(dfs), qt_975 = vec(qt(0.975, dfs)), qt_95 = vec(qt(0.95, dfs)), qt_995 = vec(qt(0.995, dfs))),
    reference = "p2a's SCPI formulas written out (not CFT 2021, CAUS-32); weights by the exact simplex QP (quadprog + KKT polish); stats::qt",
    call = "run_scpi(treated = y[TR], donors = y[D01..D08], treatment_period = 14, ...)", packages = "quadprog")
})
