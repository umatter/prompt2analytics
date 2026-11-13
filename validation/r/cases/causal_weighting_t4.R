# validation/r/cases/causal_weighting_t4.R -- Tier 4 weighting references: WeightIt::trim
# (upper tail by default, lower tail opt-in, focal group untouched), balance tables standardised by
# the fixed unadjusted SD (cobalt's s.d.denom), and twang's GBM path and iteration choice.
#  * Logistic weights are computed from glm(epsilon = 1e-14) as WeightIt's glm method defines them
#    (ATE 1/e, 1/(1-e); ATT 1, e/(1-e); ATC (1-e)/e, 1). WeightIt's own weights stop at glm's default
#    epsilon; cobalt::bal.tab on them is recorded as `*_cobalt` (cross-check, about 1e-9 away).
#  * Trimming: WeightIt::trim(w, at, lower, treat): at <- max(at, 1 - at); a type-3 quantile of the
#    weights of the groups whose weights are not all equal (the focal group of ATT / ATC is left at 1),
#    weights above it set to it; with lower = TRUE also the (1 - at) quantile from below.
#  * Balance (cobalt bal.tab, binary = "std"): SMD = weighted mean difference / an SD fixed from the
#    unweighted sample: treated SD for ATT, control SD for ATC, sqrt((v1 + v0) / 2) for ATE;
#    continuous variances with n - 1, binary p (1 - p). Variance ratio: weighted variances
#    sum v (x - m)^2 / (1 - sum v^2), v = w / sum w; NA for binary covariates.
#  * twang (not installed here): gbm(distribution = "bernoulli", bag.fraction = 1, train.fraction = 1,
#    n.minobsinnode = 10, shrinkage = 0.01), the balance criterion twang's ps.summary.new2 defines:
#    ES = weighted mean difference / SD (treated SD for ATT, SD of the full sample for ATE, control SD
#    for ATC by symmetry; n - 1, every numeric covariate continuous), KS = weighted two-sample KS.
#    The iteration is the global argmin of the criterion over 1..n.trees (twang searches a 25-point
#    grid and refines between neighbours; the global argmin is never worse).
suppressPackageStartupMessages({ library(WeightIt); library(cobalt); library(gbm) })

.glm_w <- function(d, est) {
  e <- fitted(glm(treat ~ x1 + x2 + x3, data = d, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100)))
  t1 <- d$treat == 1
  switch(est, ATE = ifelse(t1, 1 / e, 1 / (1 - e)), ATT = ifelse(t1, 1, e / (1 - e)), ATC = ifelse(t1, (1 - e) / e, 1))
}
.ess <- function(w) sum(w)^2 / sum(w^2)
.wvar <- function(x, w) { v <- w / sum(w); m <- sum(v * x); sum(v * (x - m)^2) / (1 - sum(v^2)) }
.is_bin <- function(x) all(x %in% c(0, 1))
.sd_denom <- function(x, t1, denom, bin_prop = TRUE) {
  v <- function(z) if (bin_prop && .is_bin(x)) mean(z) * (1 - mean(z)) else var(z)
  switch(denom, treated = sqrt(v(x[t1])), control = sqrt(v(x[!t1])), pooled = sqrt((v(x[t1]) + v(x[!t1])) / 2),
         all = sqrt(v(x)))
}
.denom_of <- function(est) switch(est, ATE = "pooled", ATT = "treated", ATC = "control")
.bal <- function(d, w, est, covs = c("x1", "x2", "x3")) {
  t1 <- d$treat == 1
  smd <- sapply(covs, function(v) { x <- d[[v]]; (weighted.mean(x[t1], w[t1]) - weighted.mean(x[!t1], w[!t1])) / .sd_denom(x, t1, .denom_of(est)) })
  vr <- sapply(covs, function(v) { x <- d[[v]]; if (.is_bin(x)) NA_real_ else .wvar(x[t1], w[t1]) / .wvar(x[!t1], w[!t1]) })
  list(smd = smd, vr = vr)
}

case("weights_trim_balance", "econometrics/weighting", "WeightIt glm weights: trim (upper / lower, focal untouched) and fixed-SD balance tables, weighting_sim", function(ctx) {
  d <- ctx$dataset("datasets/generated/weighting_sim.csv")
  t1 <- d$treat == 1
  v <- list()
  for (est in c("ATE", "ATT", "ATC")) {
    tag <- tolower(est); w <- .glm_w(d, est)
    b0 <- .bal(d, rep(1, nrow(d)), est); b1 <- .bal(d, w, est)
    v[[paste0(tag, "_smd_before")]] <- vec(b0$smd); v[[paste0(tag, "_smd_after")]] <- vec(b1$smd)
    v[[paste0(tag, "_vr_after")]] <- vec(b1$vr[1:2])
    v[[paste0(tag, "_mean_abs_smd_before")]] <- mean(abs(b0$smd)); v[[paste0(tag, "_mean_abs_smd_after")]] <- mean(abs(b1$smd))
    v[[paste0(tag, "_max_abs_smd_after")]] <- max(abs(b1$smd))
    v[[paste0(tag, "_sd_denom")]] <- vec(sapply(c("x1", "x2", "x3"), function(x) .sd_denom(d[[x]], t1, .denom_of(est))))
    W <- weightit(treat ~ x1 + x2 + x3, data = d, method = "glm", estimand = est)
    bt <- bal.tab(W, binary = "std", disp.v.ratio = TRUE, un = TRUE, s.d.denom = .denom_of(est))$Balance
    v[[paste0(tag, "_smd_after_cobalt")]] <- vec(bt[c("x1", "x2", "x3"), "Diff.Adj"])
    for (spec in list(list(at = 0.95, lower = FALSE), list(at = 0.95, lower = TRUE), list(at = 0.9, lower = FALSE))) {
      if (est != "ATE" && spec$lower) next
      wt <- suppressMessages(trim(w, at = spec$at, lower = spec$lower, treat = d$treat))
      s <- sprintf("%s_trim%s%s", tag, sub("0.", "", format(spec$at)), if (spec$lower) "_lower" else "")
      v[[paste0(s, "_weights_head")]] <- vec(head(wt, 12))
      v[[paste0(s, "_max")]] <- max(wt); v[[paste0(s, "_min")]] <- min(wt)
      v[[paste0(s, "_n_changed")]] <- sum(wt != w)
      v[[paste0(s, "_sum_t")]] <- sum(wt[t1]); v[[paste0(s, "_sum_c")]] <- sum(wt[!t1])
      v[[paste0(s, "_ess_t")]] <- .ess(wt[t1]); v[[paste0(s, "_ess_c")]] <- .ess(wt[!t1])
      v[[paste0(s, "_smd_after")]] <- vec(.bal(d, wt, est)$smd)
    }
  }
  # trim(W) on WeightIt's own object: same caps (cross-check of the focal rule)
  v$ate_trim95_max_weightit <- max(suppressMessages(trim(weightit(treat ~ x1 + x2 + x3, data = d, method = "glm", estimand = "ATE"), at = 0.95))$weights)
  list(values = v, reference = "WeightIt::trim(at, lower, treat) (type-3 quantile, focal group untouched); cobalt::bal.tab(binary = 'std', s.d.denom = treated | control | pooled)",
       call = "trim(w, at = 0.95 | 0.9, lower, treat = d$treat); bal.tab(weightit(treat ~ x1 + x2 + x3, method = 'glm', estimand), binary = 'std')",
       packages = c("WeightIt", "cobalt"))
})

case("balance_sd_cbps_sim", "econometrics/weighting", "Fixed balance-table SD denominators (cobalt s.d.denom) on cbps_sim", function(ctx) {
  d <- ctx$dataset("datasets/generated/cbps_sim.csv")
  t1 <- d$treat == 1
  v <- list()
  for (den in c("treated", "control", "pooled")) v[[paste0("sd_", den)]] <- vec(sapply(c("x1", "x2", "x3"), function(x) .sd_denom(d[[x]], t1, den)))
  list(values = v, reference = "cobalt s.d.denom (treated / control / pooled; binary p (1 - p))",
       call = "sd(x[t1]); sd(x[!t1]); sqrt((var(x[t1]) + var(x[!t1])) / 2); binary sqrt(p (1 - p))", packages = c("cobalt"))
})

.twang_crit <- function(d, f, est, covs = c("x1", "x2", "x3")) {
  t1 <- d$treat == 1; ps <- plogis(f)
  w <- switch(est, ATT = ifelse(t1, 1, ps / (1 - ps)), ATE = ifelse(t1, 1 / ps, 1 / (1 - ps)), ATC = ifelse(t1, (1 - ps) / ps, 1))
  den <- switch(est, ATT = "treated", ATE = "all", ATC = "control")
  es <- sapply(covs, function(v) { x <- d[[v]]; (weighted.mean(x[t1], w[t1]) - weighted.mean(x[!t1], w[!t1])) / .sd_denom(x, t1, den, bin_prop = FALSE) })
  ks <- sapply(covs, function(v) {
    x <- d[[v]]; sw <- ifelse(t1, w / sum(w[t1]), -w / sum(w[!t1])); o <- order(x)
    cs <- abs(cumsum(sw[o])); keep <- c(diff(x[o]) != 0, TRUE); max(cs[keep]) })
  list(es = es, ks = ks, w = w, ps = ps)
}
.twang_run <- function(d, M, depth, est, stop) {
  g <- gbm(treat ~ x1 + x2 + x3, data = d, distribution = "bernoulli", n.trees = M, interaction.depth = depth,
           shrinkage = 0.01, bag.fraction = 1, train.fraction = 1, n.minobsinnode = 10, verbose = FALSE)
  F <- predict(g, d, n.trees = seq_len(M), type = "link")
  crit <- sapply(seq_len(M), function(m) { b <- .twang_crit(d, F[, m], est)
    switch(stop, es.mean = mean(abs(b$es)), es.max = max(abs(b$es)), ks.mean = mean(b$ks), ks.max = max(b$ks)) })
  list(F = F, crit = crit, opt = which.min(crit), init = g$initF)
}

case("twang_gbm_sim", "econometrics/weighting", "twang: GBM propensity path (gbm bernoulli, depth 1 and 3) and the balance-optimal iteration, weighting_sim", function(ctx) {
  d <- ctx$dataset("datasets/generated/weighting_sim.csv")
  v <- list()
  for (spec in list(list(depth = 1, est = "ATT", stop = "es.mean"), list(depth = 3, est = "ATE", stop = "ks.max"),
                    list(depth = 1, est = "ATC", stop = "es.max"))) {
    r <- .twang_run(d, 3000, spec$depth, spec$est, spec$stop)
    s <- sprintf("d%d_%s_%s", spec$depth, tolower(spec$est), gsub(".", "", spec$stop, fixed = TRUE))
    v[[paste0(s, "_initf")]] <- r$init
    for (m in c(1, 10, 100, 1000, 3000)) v[[sprintf("%s_link_m%d_head", s, m)]] <- vec(head(r$F[, m], 10))
    v[[paste0(s, "_optimal_n_trees")]] <- r$opt
    v[[paste0(s, "_criterion_at_opt")]] <- r$crit[r$opt]
    v[[paste0(s, "_criterion_runner_up_gap")]] <- sort(r$crit)[2] - r$crit[r$opt]
    v[[paste0(s, "_criterion_m1")]] <- r$crit[1]; v[[paste0(s, "_criterion_m3000")]] <- r$crit[3000]
    b <- .twang_crit(d, r$F[, r$opt], spec$est); t1 <- d$treat == 1
    v[[paste0(s, "_ps_head")]] <- vec(head(b$ps, 10))
    v[[paste0(s, "_es_after")]] <- vec(b$es); v[[paste0(s, "_ks_after")]] <- vec(b$ks)
    v[[paste0(s, "_es_before")]] <- vec(sapply(c("x1", "x2", "x3"), function(x) { z <- d[[x]]
      (mean(z[t1]) - mean(z[!t1])) / .sd_denom(z, t1, switch(spec$est, ATT = "treated", ATE = "all", ATC = "control"), bin_prop = FALSE) }))
    v[[paste0(s, "_ess_t")]] <- .ess(b$w[t1]); v[[paste0(s, "_ess_c")]] <- .ess(b$w[!t1])
  }
  list(values = v, reference = "gbm::gbm(distribution = 'bernoulli', bag.fraction = 1, n.minobsinnode = 10, shrinkage = 0.01) + twang's ES/KS criterion (ps.summary.new2), global argmin over 1..n.trees",
       call = "gbm(treat ~ x1 + x2 + x3, data = weighting_sim, distribution = 'bernoulli', n.trees = 3000, interaction.depth = 1 | 3, ...)",
       packages = c("gbm"))
})
