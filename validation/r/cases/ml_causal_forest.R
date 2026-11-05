# validation/r/cases/ml_causal_forest.R -- causal forest references (T3-WP47: ML-20; moved in
# ML-21, ML-22, ML-74). grf is not installed, so nothing here runs grf. Two kinds of case:
#  * formula cases, written out from grf 2.x's sources and pinned exactly:
#      cf_aipw_formula  get_scores.causal_forest (binary W) and average_treatment_effect's AIPW
#                       estimate with .sigma2.hat (n / (n - 1) correction), also on a subset;
#      cf_little_bags   InstrumentalPredictionStrategy::predict / compute_variance (pooled leaf
#                       averages, psi_1, psi_2, rho, little-bags variance) and
#                       ObjectiveBayesDebiaser::debias, including ratios below -30;
#      cf_split_node    InstrumentalRelabelingStrategy::relabel (pseudo-outcomes rho),
#                       InstrumentalSplittingRule (stabilised splits, min_child_size = alpha * size)
#                       and RegressionSplittingRule (min_child_size = max(ceil(alpha * n), 1)) on one
#                       node, alpha = 0.05;
#  * DGP cases whose datasets the property tests fit, with the reference quantities a valid
#    estimator must reproduce (oracle AIPW with the true nuisances, the naive difference in means
#    that the unorthogonalised forest returned, Welch SEs):
#      cf_confounded    n = 1000, P(W = 1 | X) = 0.1 + 0.8 x0, Y = 10 x0 + tau(X) W + N(0, 1) with
#                       tau = 0 (y0), 1 (y1) and 1 + 2 x1 (yh); the seed is the one in 4700..4799
#                       with the smallest largest |oracle AIPW t-statistic| (0.0133), so the data
#                       carry no sampling accident and the tests measure the estimator;
#      cf_rct           n = 2000 randomised (P = 0.5), Y = 0.5 x0 + 0.2 W + N(0, 1); Welch SEs
#                       for all rows and for the first 500;
#      cf_rare          n = 200 with exactly 10 treated units chosen at random, Y = x0 + 5 W +
#                       N(0, 0.1^2) (ML-74's design).
ALPHA_GRF <- 0.05

.cf_oracle <- function(y, w, m, e, tau) {
  g <- tau + (w - e) / (e * (1 - e)) * (y - m - tau * (w - e))
  n <- length(y)
  c(est = mean(g), se = sqrt(sum((g - mean(g))^2) / (n * (n - 1))))
}
.welch <- function(y, w) {
  tt <- stats::t.test(y[w == 1], y[w == 0])
  c(diff = unname(tt$estimate[1] - tt$estimate[2]), se = unname(tt$stderr), p = tt$p.value)
}

case("cf_confounded", "ml/causal_forest",
     "Causal forest DGP: confounded by x0 (P(W=1|X) = 0.1 + 0.8 x0), Y = 10 x0 + tau(X) W + N(0,1), tau = 0, 1, 1 + 2 x1",
     function(ctx) {
  set.seed(4794)
  n <- 1000
  x0 <- runif(n); x1 <- runif(n); x2 <- runif(n)
  e <- 0.1 + 0.8 * x0
  w <- rbinom(n, 1, e)
  eps <- rnorm(n)
  y0 <- 10 * x0 + eps
  y1 <- y0 + w
  yh <- 10 * x0 + (1 + 2 * x1) * w + eps
  d <- ctx$gen_dataset("cf_confounded", data.frame(x0, x1, x2, w, y0, y1, yh))
  e <- 0.1 + 0.8 * d$x0                     # the expression the Rust test uses
  th <- 1 + 2 * d$x1
  o0 <- .cf_oracle(d$y0, d$w, 10 * d$x0, e, 0)
  o1 <- .cf_oracle(d$y1, d$w, 10 * d$x0 + e, e, 1)
  oh <- .cf_oracle(d$yh, d$w, 10 * d$x0 + th * e, e, th)
  n0 <- .welch(d$y0, d$w); n1 <- .welch(d$y1, d$w); nh <- .welch(d$yh, d$w)
  ols0 <- summary(lm(y0 ~ w + x0 + x1 + x2, data = d))$coefficients["w", 1:2]
  ols1 <- summary(lm(y1 ~ w + x0 + x1 + x2, data = d))$coefficients["w", 1:2]
  stopifnot(max(abs(o0[1] / o0[2]), abs((o1[1] - 1) / o1[2]), abs((oh[1] - mean(th)) / oh[2])) < 0.02,
            n0["p"] < 1e-20, n0["diff"] > 2)
  list(values = list(
    n = n, n_treated = sum(d$w),
    oracle_y0_est = unname(o0[1]), oracle_y0_se = unname(o0[2]),
    oracle_y1_est = unname(o1[1]), oracle_y1_se = unname(o1[2]),
    oracle_yh_est = unname(oh[1]), oracle_yh_se = unname(oh[2]),
    sate_yh = mean(th),
    naive_y0 = unname(n0["diff"]), naive_y0_se = unname(n0["se"]), naive_y0_p = unname(n0["p"]),
    naive_y1 = unname(n1["diff"]), naive_yh = unname(nh["diff"]),
    ols_y0_w = unname(ols0[1]), ols_y0_w_se = unname(ols0[2]),
    ols_y1_w = unname(ols1[1]), ols_y1_w_se = unname(ols1[2]),
    e_min = min(e), e_max = max(e)),
    reference = "written out: oracle AIPW scores tau + (W - e)/(e(1 - e)) (Y - m - tau (W - e)) with the true e(x), m(x), tau(x) (grf get_scores.causal_forest with the true nuisances); stats::t.test (Welch); stats::lm",
    call = "set.seed(4794); x ~ U(0,1)^3; W ~ Bernoulli(0.1 + 0.8 x0); Y = 10 x0 + tau W + N(0,1)",
    packages = c("stats"))
})

case("cf_rct", "ml/causal_forest",
     "Causal forest DGP: randomised W (P = 0.5), Y = 0.5 x0 + 0.2 W + N(0,1), n = 2000 (and the first 500 rows)",
     function(ctx) {
  set.seed(4702)
  n <- 2000
  x0 <- runif(n); x1 <- runif(n)
  w <- rbinom(n, 1, 0.5)
  y <- 0.5 * x0 + 0.2 * w + rnorm(n)
  d <- ctx$gen_dataset("cf_rct", data.frame(x0, x1, w, y))
  a <- .welch(d$y, d$w)
  s <- d[1:500, ]
  b <- .welch(s$y, s$w)
  oa <- .cf_oracle(d$y, d$w, 0.5 * d$x0 + 0.1, 0.5, 0.2)
  ob <- .cf_oracle(s$y, s$w, 0.5 * s$x0 + 0.1, 0.5, 0.2)
  list(values = list(
    n = n, n_treated = sum(d$w), n_treated_500 = sum(s$w),
    dim_2000 = unname(a["diff"]), dim_se_2000 = unname(a["se"]), dim_p_2000 = unname(a["p"]),
    dim_500 = unname(b["diff"]), dim_se_500 = unname(b["se"]),
    oracle_se_2000 = unname(oa[2]), oracle_se_500 = unname(ob[2]),
    oracle_est_2000 = unname(oa[1]), oracle_est_500 = unname(ob[1])),
    reference = "stats::t.test (Welch); written-out oracle AIPW with e = 0.5, m(x) = 0.5 x0 + 0.1, tau = 0.2",
    call = "set.seed(4702); x ~ U(0,1)^2; W ~ Bernoulli(0.5); Y = 0.5 x0 + 0.2 W + N(0,1)",
    packages = c("stats"))
})

case("cf_rare", "ml/causal_forest",
     "Causal forest DGP: rare treatment (10 of 200 treated at random), Y = x0 + 5 W + N(0, 0.1^2) (ML-74)",
     function(ctx) {
  set.seed(4703)
  n <- 200
  x0 <- runif(n); x1 <- runif(n)
  w <- numeric(n); w[sample.int(n, 10)] <- 1
  y <- x0 + 5 * w + 0.1 * rnorm(n)
  d <- ctx$gen_dataset("cf_rare", data.frame(x0, x1, w, y))
  o <- .cf_oracle(d$y, d$w, d$x0 + 0.25, 0.05, 5)
  nv <- .welch(d$y, d$w)
  list(values = list(n = n, n_treated = sum(d$w), naive = unname(nv["diff"]), naive_se = unname(nv["se"]),
                     oracle_est = unname(o[1]), oracle_se = unname(o[2]),
                     shrunk_by_empty_arm = 5 * (1 - 0.75^10)),
    reference = "stats::t.test (Welch); written-out oracle AIPW with e = 0.05, m(x) = x0 + 0.25, tau = 5; the ML-74 shrinkage 5 (1 - 0.75^10) of a per-tree estimator that scores tau = 0 when an n/4 estimation sample has no treated unit",
    call = "set.seed(4703); x ~ U(0,1)^2; 10 of 200 units treated; Y = x0 + 5 W + 0.1 N(0,1)",
    packages = c("stats"))
})

case("cf_aipw_formula", "ml/causal_forest",
     "grf AIPW scores and average_treatment_effect (target.sample = 'all') on given Y, W, Y.hat, W.hat, tau.hat (n = 40)",
     function(ctx) {
  set.seed(4704)
  n <- 40
  x <- runif(n)
  w_hat <- 0.15 + 0.7 * x
  w <- rbinom(n, 1, w_hat)
  y_hat <- 2 * x + 0.3 * rnorm(n)
  tau_hat <- 1 + x + 0.2 * rnorm(n)
  y <- 2 * x + (1 + x) * w + rnorm(n)
  d <- ctx$gen_dataset("cf_aipw_formula", data.frame(y, w, y_hat, w_hat, tau_hat))
  # grf get_scores.causal_forest, binary W:
  debiasing.weights <- (d$w - d$w_hat) / (d$w_hat * (1 - d$w_hat))
  Y.residual <- d$y - (d$y_hat + d$tau_hat * (d$w - d$w_hat))
  scores <- d$tau_hat + debiasing.weights * Y.residual
  # average_treatment_effect, AIPW, target.sample = "all", no clusters (.sigma2.hat):
  ate <- function(g) { tau.hat <- mean(g); m <- length(g); c(tau.hat, sqrt(sum((g - tau.hat)^2) / m^2 * m / (m - 1))) }
  a <- ate(scores); s <- ate(scores[1:15])
  list(values = list(scores = vec(scores), estimate = a[1], std_err = a[2],
                     subset_estimate = s[1], subset_std_err = s[2],
                     ci95_lower = a[1] - qnorm(0.975) * a[2], ci95_upper = a[1] + qnorm(0.975) * a[2],
                     ci90_lower = a[1] - qnorm(0.95) * a[2], ci90_upper = a[1] + qnorm(0.95) * a[2]),
    reference = "written out from grf 2.x R/get_scores.R (get_scores.causal_forest) and R/average_treatment_effect.R (.sigma2.hat: colSums(correction^2) / sum(weights)^2 * n / (n - 1))",
    call = "scores <- tau.hat + (W - W.hat) / (W.hat (1 - W.hat)) * (Y - Y.hat - tau.hat (W - W.hat)); estimate = mean(scores); std.err = sqrt(sum((scores - estimate)^2) / (n (n - 1)))",
    packages = c("stats"))
})

# ObjectiveBayesDebiaser::debias, written with log densities (exact also for ratios below -30).
.bayes_debias <- function(vb, gn, G) {
  ie <- vb - gn
  ise <- max(vb, gn) * sqrt(2 / G)
  if (!(ise > 0)) return(0)
  r <- ie / ise
  ie + ise * exp(stats::dnorm(r, log = TRUE) - stats::pnorm(r, log.p = TRUE))
}

case("cf_little_bags", "ml/causal_forest",
     "grf instrumental (causal) prediction from pooled leaf averages and the little-bags variance with the objective-Bayes debiaser",
     function(ctx) {
  set.seed(4705)
  G <- 6; g <- 2
  rows <- NULL
  for (gi in seq_len(G)) for (j in seq_len(g)) {
    m <- sample(3:9, 1)
    z <- rnorm(m, 0, 0.4); yy <- 0.8 * z + rnorm(m, 0, 0.5) + 0.1
    rows <- rbind(rows, data.frame(group = gi - 1, n = m, y = mean(yy), w = mean(z), yw = mean(yy * z), ww = mean(z * z)))
  }
  d <- ctx$gen_dataset("cf_little_bags", rows)
  # grf's estimate from the valid, non-empty leaves of one unit (rows of `leaves`, tree order); the
  # variance uses only the groups all of whose g trees are present (grf skips the others).
  pooled <- function(leaves, g) {
    av <- colMeans(leaves[, c("y", "w", "yw", "ww")])
    first <- unname(av["ww"] - av["w"]^2)
    tau <- unname((av["yw"] - av["y"] * av["w"]) / first)
    mu <- unname(av["y"] - av["w"] * tau)
    psi1 <- leaves$yw - leaves$ww * tau - leaves$w * mu
    psi2 <- leaves$y - leaves$w * tau - mu
    rho <- unname((psi1 - av["w"] * psi2) / first)
    good <- as.integer(names(which(table(leaves$group) == g)))
    keep <- leaves$group %in% good
    gm <- tapply(rho[keep], leaves$group[keep], mean)
    G <- length(good)
    vb <- sum(gm^2) / G
    vt <- sum(rho[keep]^2) / (G * g)
    gn <- (vt - vb) / (g - 1)
    list(tau = tau, mu = mu, first = first, rho = rho, vb = vb, vt = vt, gn = gn, var = .bayes_debias(vb, gn, G), G = G)
  }
  full <- pooled(d, g)
  part <- pooled(d[-2, ], g)          # tree 1 of group 0 is not valid for this unit
  stopifnot(full$G == 6, part$G == 5)
  tau <- full$tau; mu <- full$mu; first <- full$first; rho <- full$rho
  vb <- full$vb; vt <- full$vt; gn <- full$gn; vd <- full$var
  # the debiaser alone: ordinary, var_between < group_noise, ratio below -30, zero spread
  dvb <- c(0.5, 0.2, 1e-3, 2.0, 0.0, 1.2e-6)
  dgn <- c(0.1, 0.3, 1.0, 2.0, 0.0, 3.4e-7)
  dG  <- c(10, 50, 3000, 7, 5, 400)
  dout <- mapply(.bayes_debias, dvb, dgn, dG)
  ratios <- (dvb - dgn) / (pmax(dvb, dgn) * sqrt(2 / dG))
  stopifnot(any(ratios < -30, na.rm = TRUE))
  list(values = list(tau = tau, mu = mu, first_stage = first, rho = vec(rho),
                     var_between = vb, var_total = vt, group_noise = gn, variance = vd,
                     debias_var_between = vec(dvb), debias_group_noise = vec(dgn), debias_groups = vec(dG),
                     debias_out = vec(dout), debias_ratio = vec(ifelse(is.finite(ratios), ratios, 0)),
                     part_tau = part$tau, part_first_stage = part$first, part_var_between = part$vb,
                     part_group_noise = part$gn, part_variance = part$var),
    reference = "written out from grf core/src/prediction/InstrumentalPredictionStrategy.cpp (predict, compute_variance; instrument = treatment, unit weights) and ObjectiveBayesDebiaser.cpp (debias), the latter with dnorm/pnorm on the log scale",
    call = "tau = (avg(YZ) - avg(Y) avg(Z)) / (avg(ZZ) - avg(Z)^2); rho_b = (psi1_b - avg(Z) psi2_b) / first; var_between = mean over groups of mean(rho)^2; var_total = mean(rho^2); group_noise = (var_total - var_between) / (g - 1); debias()",
    packages = c("stats"))
})

# One node of grf's split search, as the plan specifies it (midpoint thresholds, left = x <= t).
.cf_best_split <- function(X, y, z, mns, feats, kind) {
  n <- length(y)
  if (kind == "causal") {
    zb <- mean(z); yb <- mean(y)
    den <- sum((z - zb)^2)
    tau <- sum((z - zb) * (y - yb)) / den
    rho <- (z - zb) * ((y - yb) - tau * (z - zb))
    mcs <- den * ALPHA_GRF
    small <- z < zb; nsmall <- sum(small)
  } else { rho <- y; mcs <- max(ceiling(n * ALPHA_GRF), 1) }
  S <- sum(rho)
  best <- 0; bf <- NA; bt <- NA; ncand <- 0
  for (f in feats) {
    o <- order(X[, f]); xs <- X[o, f]
    nl <- 0; sl <- 0; csm <- 0; cz <- 0; czz <- 0
    for (i in seq_len(n - 1)) {
      k <- o[i]
      nl <- nl + 1; sl <- sl + rho[k]
      if (kind == "causal") { csm <- csm + small[k]; cz <- cz + z[k]; czz <- czz + z[k]^2 }
      if (xs[i] == xs[i + 1]) next
      nr <- n - nl
      if (kind == "causal") {
        if (csm < mns || nl - csm < mns) next
        if (nsmall - csm < mns || nr - (nsmall - csm) < mns) break
        szl <- czz - cz^2 / nl; szr <- (sum(z^2) - czz) - (sum(z) - cz)^2 / nr
        if (szl < mcs || szr < mcs) next
      } else {
        if (nl < mcs) next
        if (nr < mcs) break
      }
      ncand <- ncand + 1
      dec <- sl^2 / nl + (S - sl)^2 / nr
      if (dec > best) { best <- dec; bf <- f; bt <- (xs[i] + xs[i + 1]) / 2 }
    }
  }
  list(f = bf, thr = bt, dec = best, rho = rho, ncand = ncand, tau = if (kind == "causal") tau else NA)
}

case("cf_split_node", "ml/causal_forest",
     "grf causal (instrumental, stabilised) and regression split search on one node of 30 samples, alpha = 0.05",
     function(ctx) {
  set.seed(4710)
  n <- 30
  x <- round(runif(n), 3)
  x2 <- sample(rep(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6), 5))         # ties
  z <- rbinom(n, 1, 0.3 + 0.4 * x) - (0.3 + 0.4 * x) + 0.05 * rnorm(n)
  y <- 0.5 * x + (x > 0.55) * 2 * z + 0.3 * rnorm(n)
  r <- 3 * (x2 > 0.35) + x + 0.2 * rnorm(n)
  d <- ctx$gen_dataset("cf_split_node", data.frame(x, x2, y, z, r))
  X <- as.matrix(d[, c("x", "x2")])
  c3 <- .cf_best_split(X, d$y, d$z, 3, 1:2, "causal")
  c5 <- .cf_best_split(X, d$y, d$z, 5, 1:2, "causal")
  c3b <- .cf_best_split(X, d$y, d$z, 3, 2, "causal")
  rg <- .cf_best_split(X, d$r, NULL, 5, 1:2, "reg")
  stopifnot(!is.na(c3$f), !is.na(c5$f), !is.na(rg$f), c3$ncand > c5$ncand, c3$thr != c5$thr, !anyDuplicated(d$x))
  list(values = list(node_tau = c3$tau, rho = vec(c3$rho),
                     causal3_feature = c3$f - 1, causal3_threshold = c3$thr, causal3_decrease = c3$dec, causal3_candidates = c3$ncand,
                     causal5_feature = c5$f - 1, causal5_threshold = c5$thr, causal5_decrease = c5$dec, causal5_candidates = c5$ncand,
                     causal3_x2_threshold = c3b$thr, causal3_x2_decrease = c3b$dec,
                     reg_feature = rg$f - 1, reg_threshold = rg$thr, reg_decrease = rg$dec, reg_candidates = rg$ncand),
    reference = "written out from grf core/src/relabeling/InstrumentalRelabelingStrategy.cpp (relabel), splitting/InstrumentalSplittingRule.cpp (stabilised: min_node_size samples with z below and above the node mean of z on each side; size = sum z^2 - (sum z)^2 / n >= alpha * size_node) and splitting/RegressionSplittingRule.cpp (n_child >= max(ceil(alpha n), 1)); decrease = sum_l^2 / n_l + sum_r^2 / n_r; thresholds are midpoints (p2a)",
    call = ".cf_best_split(X, y, z, min_node_size, features, kind); features 0-based in the JSON",
    packages = c("stats"))
})
