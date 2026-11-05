# validation/r/cases/causal_weighting.R -- balancing-weight references (T3-WP23: CAUS-49, CAUS-50,
# CAUS-79, CAUS-95, CAUS-92; moved in CAUS-94).
#  * entropy balancing: WeightIt::weightit(method = 'ebal', estimand = 'ATE' | 'ATT' | 'ATC'),
#    moments = 1 (means), ESS per group; and the exact dual solution (Newton to |grad| < 1e-14),
#    which p2a is pinned to (WeightIt's optim values are a cross-check);
#  * stable balancing weights (SBW, Zubizarreta 2015) and energy balancing (Huling & Mak 2022):
#    WeightIt's implementations need the optweight / osqp packages, which are not installed here, so
#    the reference is the published quadratic program solved exactly with quadprog::solve.QP:
#      SBW (ATT, controls): min sum_i (w_i - 1/n0)^2  s.t. sum w = 1, w >= 0,
#                           |sum_i w_i x_ij - xbar1_j| <= delta * sd_j  (delta = 0 is exact balance;
#                           sd_j is the full-sample standard deviation of covariate j);
#                           ATC mirrors it; ATE solves it once per group with the pooled means as target;
#      energy (ATT):         min_w  E(w) = 2/(n0 n1) sum_i w_i sum_j |x_i - z_j| - 1/n0^2 w' D00 w - const,
#                           the energy distance between the weighted controls and the treated sample
#                           (standardised X), s.t. sum w = n0, w >= 0 (Huling & Mak 2022, eq. 6);
#                           ATC mirrors it; ATE is the "improved" form WeightIt uses (Huling & Mak 2022,
#                           section 3.2): E(F1w, Fn) + E(F0w, Fn) + E(F1w, F0w), with WeightIt's P and q at
#                           lambda = 0 and min.w = 0 (WeightIt's defaults lambda = 1e-4, min.w = 1e-8 move
#                           the ATT effect by 1.3e-5 on this data).
#    -D is only conditionally positive definite (on sum(u) = 0), so every QP is written in w = 1 + Z u,
#    Z an orthonormal basis of the sum-zero subspace of each group, with constraints Z u >= -1.
suppressPackageStartupMessages({ library(WeightIt); library(quadprog) })

.w_sim <- function(ctx) {
  set.seed(20261104)
  n <- 300
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rbinom(n, 1, 0.5)
  tr <- rbinom(n, 1, plogis(-0.2 + 0.7 * x1 - 0.5 * x2 + 0.4 * x3))
  y <- 2 + 1.5 * tr + x1 + x2 + rnorm(n)
  ctx$gen_dataset("weighting_sim", data.frame(treat = tr, x1 = round(x1, 10), x2 = round(x2, 10), x3 = x3, y = round(y, 10)))
}

.ess <- function(w) sum(w)^2 / sum(w^2)

# Entropy balancing to the target means `tgt` with base weights 1: w_i proportional to exp(lambda' x_i),
# lambda solving the convex dual min log sum_i exp(lambda' (x_i - tgt)) by Newton to |grad| < 1e-14.
.ebal_exact <- function(X, tgt) {
  Xc <- sweep(X, 2, tgt); lam <- rep(0, ncol(X))
  for (it in 1:100) {
    s <- drop(Xc %*% lam); p <- exp(s - max(s)); p <- p / sum(p)
    gr <- colSums(Xc * p); if (max(abs(gr)) < 1e-14) break
    H <- crossprod(Xc * p, Xc) - tcrossprod(gr)
    lam <- lam - solve(H, gr)
  }
  s <- drop(Xc %*% lam); p <- exp(s - max(s)); p / sum(p) * nrow(X)
}

case("weights_ebal", "econometrics/weighting", "Entropy balancing (WeightIt ebal), ATE / ATT / ATC, simulated n = 300", function(ctx) {
  d <- .w_sim(ctx)
  v <- list()
  for (est in c("ATE", "ATT", "ATC")) {
    w <- weightit(treat ~ x1 + x2 + x3, data = d, method = "ebal", estimand = est)$weights
    tag <- tolower(est)
    v[[paste0(tag, "_weights_head")]] <- vec(head(w, 12))
    v[[paste0(tag, "_weights_sum_t")]] <- sum(w[d$treat == 1]); v[[paste0(tag, "_weights_sum_c")]] <- sum(w[d$treat == 0])
    v[[paste0(tag, "_ess_t")]] <- .ess(w[d$treat == 1]); v[[paste0(tag, "_ess_c")]] <- .ess(w[d$treat == 0])
    for (x in c("x1", "x2", "x3")) {
      v[[paste0(tag, "_wmean_t_", x)]] <- weighted.mean(d[[x]][d$treat == 1], w[d$treat == 1])
      v[[paste0(tag, "_wmean_c_", x)]] <- weighted.mean(d[[x]][d$treat == 0], w[d$treat == 0])
    }
    v[[paste0(tag, "_effect")]] <- weighted.mean(d$y[d$treat == 1], w[d$treat == 1]) - weighted.mean(d$y[d$treat == 0], w[d$treat == 0])
    # exact dual solution (pinned); the WeightIt values above stop at optim's reltol (cross-check)
    X <- as.matrix(d[, c("x1", "x2", "x3")]); t1 <- d$treat == 1
    tgt <- switch(est, ATE = colMeans(X), ATT = colMeans(X[t1, ]), ATC = colMeans(X[!t1, ]))
    wt <- if (est == "ATT") rep(1, sum(t1)) else .ebal_exact(X[t1, ], tgt)
    wc <- if (est == "ATC") rep(1, sum(!t1)) else .ebal_exact(X[!t1, ], tgt)
    v[[paste0(tag, "_exact_weights_t_head")]] <- vec(head(wt, 8)); v[[paste0(tag, "_exact_weights_c_head")]] <- vec(head(wc, 8))
    v[[paste0(tag, "_exact_ess_t")]] <- .ess(wt); v[[paste0(tag, "_exact_ess_c")]] <- .ess(wc)
    v[[paste0(tag, "_exact_effect")]] <- weighted.mean(d$y[t1], wt) - weighted.mean(d$y[!t1], wc)
  }
  list(values = v, reference = "WeightIt::weightit(method = 'ebal', estimand = 'ATE' | 'ATT' | 'ATC'); ESS = (sum w)^2 / sum w^2 per group",
       call = "weightit(treat ~ x1 + x2 + x3, data = weighting_sim, method = 'ebal', estimand)", packages = c("WeightIt"))
})

.sbw_qp <- function(X0, xbar1, sdv, delta) {
  n0 <- nrow(X0); k <- ncol(X0)
  D <- diag(n0); dvec <- rep(1 / n0, n0)                     # min 1/2 w'Dw - d'w  ==  min sum (w - 1/n0)^2 / 2 + const
  Aeq <- matrix(1, n0, 1); beq <- 1
  if (delta == 0) {
    A <- cbind(Aeq, X0, diag(n0)); b <- c(beq, xbar1, rep(0, n0)); meq <- 1 + k
  } else {
    A <- cbind(Aeq, X0, -X0, diag(n0)); b <- c(beq, xbar1 - delta * sdv, -(xbar1 + delta * sdv), rep(0, n0)); meq <- 1
  }
  w <- solve.QP(D, dvec, A, b, meq = meq)$solution
  w[abs(w) < 1e-14] <- 0
  w
}

case("weights_sbw", "econometrics/weighting", "Stable balancing weights (ATT, ATC, ATE), exact (delta = 0) and approximate (delta = 0.1 sd) balance, quadprog QP", function(ctx) {
  d <- ctx$dataset("datasets/generated/weighting_sim.csv")
  X <- as.matrix(d[, c("x1", "x2", "x3")]); t1 <- d$treat == 1
  sdv <- apply(X, 2, sd)
  v <- list(sd_all = vec(sdv))
  for (delta in c(0, 0.1)) {
    tag <- if (delta == 0) "exact" else "approx"
    w0 <- .sbw_qp(X[!t1, ], colMeans(X[t1, ]), sdv, delta)
    v[[paste0("att_", tag, "_weights_head")]] <- vec(head(w0, 12))
    v[[paste0("att_", tag, "_weighted_means")]] <- vec(colSums(X[!t1, ] * w0))
    v[[paste0("att_", tag, "_objective")]] <- sum((w0 - 1 / sum(!t1))^2)
    v[[paste0("att_", tag, "_n_zero")]] <- sum(w0 == 0)
    v[[paste0("att_", tag, "_effect")]] <- mean(d$y[t1]) - sum(w0 * d$y[!t1])
    # ATC: the treated reweighted to the control means
    w1 <- .sbw_qp(X[t1, ], colMeans(X[!t1, ]), sdv, delta)
    v[[paste0("atc_", tag, "_weights_head")]] <- vec(head(w1, 12))
    v[[paste0("atc_", tag, "_objective")]] <- sum((w1 - 1 / sum(t1))^2)
    v[[paste0("atc_", tag, "_n_zero")]] <- sum(w1 == 0)
    v[[paste0("atc_", tag, "_effect")]] <- sum(w1 * d$y[t1]) - mean(d$y[!t1])
    # ATE: each group reweighted to the pooled means
    wt <- .sbw_qp(X[t1, ], colMeans(X), sdv, delta); wc <- .sbw_qp(X[!t1, ], colMeans(X), sdv, delta)
    v[[paste0("ate_", tag, "_weights_t_head")]] <- vec(head(wt, 8)); v[[paste0("ate_", tag, "_weights_c_head")]] <- vec(head(wc, 8))
    v[[paste0("ate_", tag, "_effect")]] <- sum(wt * d$y[t1]) - sum(wc * d$y[!t1])
    v[[paste0("ate_", tag, "_objective_t")]] <- sum((wt - 1 / sum(t1))^2)
    v[[paste0("ate_", tag, "_objective_c")]] <- sum((wc - 1 / sum(!t1))^2)
  }
  list(values = v, reference = "Zubizarreta (2015) stable balancing weights, solved with quadprog::solve.QP (optweight / sbw not installed)",
       call = "solve.QP(diag(n0), rep(1/n0, n0), [1, X0, I], [1, xbar1, 0], meq = 1 + k)", packages = c("quadprog"))
})

.energy_att <- function(Xs0, Xs1) {
  n0 <- nrow(Xs0); n1 <- nrow(Xs1)
  D00 <- as.matrix(dist(Xs0)); D01 <- as.matrix(dist(rbind(Xs0, Xs1)))[seq_len(n0), n0 + seq_len(n1)]
  D11 <- as.matrix(dist(Xs1))
  # E(w) = 2/(n0 n1) sum_i w_i sum_j |x_i - z_j| - 1/n0^2 w' D00 w - 1/n1^2 1' D11 1, sum(w) = n0, w >= 0.
  # -D00 is only conditionally positive definite (on sum(u) = 0), so write w = 1 + Z u with Z an
  # orthonormal basis of the sum-zero subspace and solve the QP in u (constraints Z u >= -1).
  Z <- qr.Q(qr(cbind(1, diag(n0))))[, -1]
  H <- -2 / n0^2 * crossprod(Z, D00 %*% Z)
  g <- crossprod(Z, 2 / (n0 * n1) * rowSums(D01) - 2 / n0^2 * rowSums(D00))
  sol <- solve.QP(H, -drop(g), t(Z), rep(-1, n0))
  w <- drop(1 + Z %*% sol$solution)
  w[abs(w) < 1e-10] <- 0
  e <- 2 / (n0 * n1) * sum(w * rowSums(D01)) - 1 / n0^2 * drop(t(w) %*% D00 %*% w) - 1 / n1^2 * sum(D11)
  e1 <- 2 / (n0 * n1) * sum(rowSums(D01)) - 1 / n0^2 * sum(D00) - 1 / n1^2 * sum(D11)
  list(w = w, energy = e, energy_unweighted = e1)
}

# Energy QP: min w'Pw + q'w over the groups `grp` (sum of w in group g = n_g, w >= 0), as w = 1 + Z u.
.energy_qp <- function(P, q, grp) {
  m <- length(q); lev <- sort(unique(grp))
  Z <- matrix(0, m, m - length(lev)); col <- 0
  for (g in lev) {
    idx <- which(grp == g); k <- length(idx)
    Z[idx, col + seq_len(k - 1)] <- qr.Q(qr(cbind(1, diag(k))))[, -1, drop = FALSE]
    col <- col + k - 1
  }
  H <- crossprod(Z, (2 * P) %*% Z)
  g <- crossprod(Z, 2 * P %*% rep(1, m) + q)
  sol <- solve.QP(H, -drop(g), t(Z), rep(-1, m))
  w <- drop(1 + Z %*% sol$solution)
  w[abs(w) < 1e-10] <- 0
  w
}
.edist <- function(xa, xb, wa, wb) {   # energy distance between samples a and b with mean-1 weights
  na <- nrow(xa); nb <- nrow(xb)
  Dab <- as.matrix(dist(rbind(xa, xb)))[seq_len(na), na + seq_len(nb), drop = FALSE]
  2 / (na * nb) * drop(t(wa) %*% Dab %*% wb) - 1 / na^2 * drop(t(wa) %*% as.matrix(dist(xa)) %*% wa) -
    1 / nb^2 * drop(t(wb) %*% as.matrix(dist(xb)) %*% wb)
}

case("weights_energy", "econometrics/weighting", "Energy balancing weights (ATT, ATC, improved ATE), Huling & Mak (2022), quadprog QP on standardised covariates", function(ctx) {
  d <- ctx$dataset("datasets/generated/weighting_sim.csv")
  Xr <- as.matrix(d[, c("x1", "x2", "x3")])
  X <- scale(Xr); t1 <- d$treat == 1
  n <- nrow(X); n1 <- sum(t1); n0 <- sum(!t1)
  Dm <- as.matrix(dist(X))
  # ATT: controls reweighted towards the treated sample (the design's helper, unchanged values)
  r <- .energy_att(X[!t1, ], X[t1, ]); w0 <- r$w
  # ATC: treated reweighted towards the control sample
  w1 <- .energy_qp(-Dm[t1, t1] / n1^2, 2 / (n0 * n1) * rowSums(Dm[t1, !t1]), rep(1, n1))
  # ATE, improved (WeightIt weightit2energy with lambda = 0)
  s0 <- ifelse(!t1, 1 / n0, 0); s1 <- ifelse(t1, 1 / n1, 0)
  nn <- tcrossprod(cbind(s0, s1)) + tcrossprod(s0 - s1)
  w <- .energy_qp(-Dm * nn, drop((rep(2 / n, n)) %*% Dm) * (s0 + s1), as.integer(t1))
  v <- list(att_weights_head = vec(head(w0, 12)), att_energy = r$energy,
            att_energy_unweighted = r$energy_unweighted,
            att_weighted_means = vec(colSums(as.matrix(d[!t1, c("x1", "x2", "x3")]) * w0) / sum(!t1)),
            att_effect = mean(d$y[t1]) - sum(w0 * d$y[!t1]) / n0, att_n_zero = sum(w0 == 0),
            atc_weights_head = vec(head(w1, 12)), atc_energy = .edist(X[t1, ], X[!t1, ], w1, rep(1, n0)),
            atc_weighted_means = vec(colSums(Xr[t1, ] * w1) / n1),
            atc_effect = sum(w1 * d$y[t1]) / n1 - mean(d$y[!t1]), atc_n_zero = sum(w1 == 0),
            ate_weights_head = vec(head(w, 12)),
            ate_energy = .edist(X[t1, ], X, w[t1], rep(1, n)) + .edist(X[!t1, ], X, w[!t1], rep(1, n)) + .edist(X[t1, ], X[!t1, ], w[t1], w[!t1]),
            ate_weighted_means_t = vec(colSums(Xr[t1, ] * w[t1]) / n1), ate_weighted_means_c = vec(colSums(Xr[!t1, ] * w[!t1]) / n0),
            ate_effect = sum(w[t1] * d$y[t1]) / n1 - sum(w[!t1] * d$y[!t1]) / n0, ate_n_zero = sum(w == 0))
  list(values = v, reference = "Huling & Mak (2022) energy balancing (ATT, ATC, improved ATE as WeightIt weightit2energy with lambda = 0, min.w = 0), solved with quadprog (WeightIt 'energy' needs osqp, not installed)",
       call = "solve.QP on the energy-distance quadratic form in w = 1 + Z u", packages = c("quadprog"))
})
