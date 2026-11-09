# validation/r/cases/causal_ltmle.R -- longitudinal TMLE reference (T3-WP27: CAUS-106, CAUS-66;
# moved in CAUS-17, CAUS-18, CAUS-19).
#
# The estimator is written out here so that every intermediate (H, cumulative g, epsilons, Q and g
# coefficients) can be pinned; it is cross-checked against ltmle 1.3-0 in the case. It follows the
# defaults of ltmle 1.x (Lendle, Schwab, Petersen & van der Laan 2017, JSS 81(1), section 2;
# ltmle(stratify = FALSE, gbounds = c(0.01, 1), SL.library = "glm", variance.method = "ic")):
#  * history: the nodes in time order L_1, A_1, L_2, A_2, ... (ltmle's default formulas use every
#    parent node, main terms, in data order).
#  * g_t: logistic glm of A_t on the history before it, fitted on all subjects; predicted with
#    A_1..A_{t-1} set to the regime (SetA); probability of following the regime = g or 1 - g; the
#    CUMULATIVE product c_t = prod_{s <= t} is bounded to gbounds (not each factor).
#  * H_t = I(Abar_t = dbar_t) / bounded c_t.
#  * Q_t: quasibinomial glm of the pseudo-outcome (Q*_{T+1} = Y) on the history through A_t, fitted on
#    all subjects (stratify = FALSE), predicted on the link scale with A_1..A_t set to the regime.
#  * fluctuation: intercept-only quasibinomial glm with offset logit Q_t(d) and weights H_t among the
#    subjects who follow the regime through A_t; Q*_t = plogis(logit Q_t(d) + eps_t) for every subject
#    is the next pseudo-outcome.
#  * psi = mean(Q*_1); EIC = sum_t H_t (Q*_{t+1} - Q*_t) + Q*_1 - psi; SE = sd(EIC) / sqrt(n)
#    (ltmle's variance.method = "ic"; the default "tmle" substitution variance is not reproduced);
#    contrasts difference the EICs.
#  * a Y outside [0, 1] is rescaled by its observed range before the logistic fits; psi and the EIC
#    are scaled back (ltmle's Yrange = NULL behaviour).
#  * "linear" is p2a's Gaussian variant (not an ltmle option): lm for Q_t; eps_t = the H_t-weighted
#    mean of (pseudo-outcome - Q_t(d)) among the followers; Q*_t = Q_t(d) + eps_t; the same EIC.
# Every glm uses glm.control(epsilon = 1e-14, maxit = 100) (tight reference).
# Planning checks of this estimator (plan T3-WP27): saturated models reproduce the hand-computed
# g-formula to 1e-16; Monte Carlo on this DGP (400 x n = 600): ATE bias -0.006 (truth 0.3462),
# mean EIC SE 0.0507 vs MC SD 0.0489, 95% coverage 0.948; bootstrap SE (B = 400) 0.0528 vs 0.0531.

.lt_data <- function(ctx) {
  set.seed(20261206)
  n <- 600
  L0 <- rnorm(n); A1 <- rbinom(n, 1, plogis(0.3 * L0)); L1 <- 0.5 * L0 + 0.6 * A1 + rnorm(n)
  A2 <- rbinom(n, 1, plogis(-0.2 + 0.5 * L1 + 0.8 * A1)); Y <- rbinom(n, 1, plogis(-1 + 0.5 * L1 + 0.6 * A2 + 0.7 * A1 + 0.2 * L0))
  # drawn after Y, so L0..Y are the draws of the design's plan-check copy
  Yc <- 2 + 0.8 * L1 + 1.5 * A2 + 1.0 * A1 + 0.3 * L0 + rnorm(n)
  # randomized two-period design with a known effect: always - never = 2 + 3 + 0.5 = 5.5 (CAUS-17/18 repro DGP)
  R1 <- rbinom(n, 1, 0.5); Lr <- 0.5 * R1 + 0.4 * L0 + rnorm(n); R2 <- rbinom(n, 1, 0.5)
  Yr <- 1 + 2 * R1 + 3 * R2 + Lr + 0.3 * L0 + rnorm(n)
  # third time point with treatment persistence (history beyond one lag)
  L2 <- 0.4 * L1 + 0.5 * A2 + 0.2 * A1 + rnorm(n)
  A3 <- rbinom(n, 1, plogis(-0.1 + 0.4 * L2 + 0.9 * A2 + 0.3 * A1))
  Y3 <- rbinom(n, 1, plogis(-1.2 + 0.4 * L2 + 0.5 * A3 + 0.4 * A2 + 0.3 * A1 + 0.2 * L1 + 0.1 * L0))
  ctx$gen_dataset("ltmle_sim", data.frame(L0 = round(L0, 10), A1 = A1, L1 = round(L1, 10), A2 = A2, Y = Y, Yc = round(Yc, 10),
                                          R1 = R1, Lr = round(Lr, 10), R2 = R2, Yr = round(Yr, 10),
                                          L2 = round(L2, 10), A3 = A3, Y3 = Y3))
}
.lt_tight <- glm.control(epsilon = 1e-14, maxit = 100)
.lt_glm <- function(resp, cols, data, family, ctl) {
  df <- data[, cols, drop = FALSE]; df$resp_ <- resp
  glm(reformulate(cols, "resp_"), data = df, family = family, control = ctl)
}
# d: data; Lc: list of covariate-name vectors per time point; Ac: treatment names; Ycol: outcome;
# abar: n x T regime matrix (row i = the treatments the regime assigns to subject i).
.ltmle <- function(d, Lc, Ac, Ycol, abar, gbounds = c(0.01, 1), q = "logistic", per_factor_bounds = FALSE,
                   set_prior_a = TRUE, ctl = .lt_tight, check_ic = TRUE) {
  n <- nrow(d); TT <- length(Ac)
  y <- d[[Ycol]]; sc <- c(0, 1)
  if (q == "logistic" && (min(y) < 0 || max(y) > 1)) { sc <- range(y); y <- (y - sc[1]) / diff(sc) }
  hist_g <- function(t) unlist(lapply(seq_len(t), function(s) c(Lc[[s]], if (s < t) Ac[s])))
  hist_q <- function(t) c(hist_g(t), Ac[t])
  setA <- function(upto) { nd <- d; for (s in seq_len(upto)) nd[[Ac[s]]] <- abar[, s]; nd }
  gd <- matrix(NA_real_, n, TT); gfits <- vector("list", TT)
  for (t in seq_len(TT)) {
    gfits[[t]] <- .lt_glm(d[[Ac[t]]], hist_g(t), d, binomial, ctl)
    p1 <- predict(gfits[[t]], newdata = setA(t - 1), type = "response")
    gd[, t] <- ifelse(abar[, t] == 1, p1, 1 - p1)
  }
  cum_unb <- t(apply(gd, 1, cumprod))
  cum <- if (per_factor_bounds) t(apply(pmin(pmax(gd, gbounds[1]), gbounds[2]), 1, cumprod)) else pmin(pmax(cum_unb, gbounds[1]), gbounds[2])
  follow <- matrix(FALSE, n, TT)
  for (t in seq_len(TT)) follow[, t] <- (if (t > 1) follow[, t - 1] else TRUE) & d[[Ac[t]]] == abar[, t]
  H <- follow / cum
  out_of_bounds <- cum_unb < gbounds[1] | cum_unb > gbounds[2]
  qnext <- y; ic <- numeric(n); eps <- numeric(TT); qfits <- vector("list", TT)
  for (t in rev(seq_len(TT))) {
    nd <- if (set_prior_a) setA(t) else { z <- d; z[[Ac[t]]] <- abar[, t]; z }
    sub <- follow[, t]
    if (q == "logistic") {
      qfits[[t]] <- suppressWarnings(.lt_glm(qnext, hist_q(t), d, quasibinomial, ctl))
      eta <- predict(qfits[[t]], newdata = nd, type = "link")
      fl <- data.frame(r = qnext, o = eta, w = H[, t])[sub, ]
      eps[t] <- unname(coef(suppressWarnings(glm(r ~ 1, offset = o, weights = w, data = fl, family = quasibinomial, control = ctl))))
      qs <- plogis(eta + eps[t])
    } else {
      df <- d[, hist_q(t), drop = FALSE]; df$resp_ <- qnext
      qfits[[t]] <- lm(reformulate(hist_q(t), "resp_"), data = df)
      m <- predict(qfits[[t]], newdata = nd)
      eps[t] <- sum(H[sub, t] * (qnext[sub] - m[sub])) / sum(H[sub, t])
      qs <- m + eps[t]
    }
    ic <- ic + H[, t] * (qnext - qs)
    qnext <- qs
  }
  psi <- mean(qnext); ic <- ic + qnext - psi
  if (check_ic) stopifnot(abs(mean(ic)) < 1e-9)                     # the targeting solved the EIC equation
  list(psi = psi * diff(sc) + sc[1], ic = ic * diff(sc), eps = eps, n_follow = colSums(follow), cum_unb = cum_unb,
       n_bounded = colSums(out_of_bounds), n_bounded_follow = colSums(out_of_bounds & follow),
       H = H, cum = cum, gfits = gfits, qfits = qfits, scale = sc)
}
.lt_se <- function(ic) sd(ic) / sqrt(length(ic))
# ltmle::ltmle itself on the same nodes (glm, stratify = FALSE, variance.method = "ic"); the regime
# means and EIC SEs must equal the written-out values to 1e-8 relative (ltmle fits at glm's default
# control, about 1e-11 away from the tight fits).
.lt_pkg <- function(d, Lc, Ac, Ycol, abar, gbounds = c(0.01, 1)) {
  nodes <- c(unlist(lapply(seq_along(Ac), function(s) c(Lc[[s]], Ac[s]))), Ycol)
  lnodes <- unlist(Lc[-1])
  f <- suppressWarnings(suppressMessages(ltmle::ltmle(d[, nodes], Anodes = Ac, Lnodes = lnodes, Ynodes = Ycol,
         abar = abar, gbounds = gbounds, SL.library = "glm", stratify = FALSE, variance.method = "ic",
         estimate.time = FALSE)))
  s <- summary(f)$treatment
  c(psi = unname(s$estimate), se = unname(s$std.dev))
}
.lt_same <- function(pkg, mine, what) {
  ref <- c(mine$psi, .lt_se(mine$ic))
  if (any(abs(pkg - ref) > 1e-8 * abs(ref)))
    stop(sprintf("ltmle cross-check %s: ltmle %s, written out %s", what,
                 paste(format(pkg, digits = 17), collapse = "/"), paste(format(ref, digits = 17), collapse = "/")))
}

case("ltmle_two_period", "causal/tmle", "LTMLE as ltmle's defaults (written out, cross-checked against ltmle): static, mixed static and dynamic regimes at two and three time points, cumulative gbounds, continuous outcome (ltmle rescaling and p2a's linear variant), a randomized design with a known effect; ATE and contrasts with EIC standard errors", function(ctx) {
  d <- .lt_data(ctx); n <- nrow(d)
  Lc <- list("L0", "L1"); Ac <- c("A1", "A2")
  st <- function(...) matrix(rep(c(...), each = n), n)                # static regime matrix
  always <- st(1, 1); never <- st(0, 0)
  dynamic <- cbind(as.integer(d$L0 > 0), as.integer(d$L1 > 0))     # A_t = 1{L_{t-1} > 0}
  r11 <- .ltmle(d, Lc, Ac, "Y", always); r00 <- .ltmle(d, Lc, Ac, "Y", never); rdy <- .ltmle(d, Lc, Ac, "Y", dynamic)
  r10 <- .ltmle(d, Lc, Ac, "Y", st(1, 0))
  ate <- r11$psi - r00$psi; ate_se <- .lt_se(r11$ic - r00$ic); z <- ate / ate_se
  stopifnot(sum(r11$n_bounded_follow, r00$n_bounded_follow, rdy$n_bounded_follow) == 0)   # default bounds never bind for followers here
  # cumulative gbounds (0.1, 0.5) bind for both regimes; bounding each factor instead gives another psi
  gb <- c(0.1, 0.5)
  b11 <- .ltmle(d, Lc, Ac, "Y", always, gbounds = gb); b00 <- .ltmle(d, Lc, Ac, "Y", never, gbounds = gb)
  f00 <- .ltmle(d, Lc, Ac, "Y", never, gbounds = gb, per_factor_bounds = TRUE)
  stopifnot(all(b11$n_bounded_follow > 0), all(b00$n_bounded_follow > 0), abs(f00$psi - b00$psi) > 1e-4)
  # SetA matters: predicting Q_2 with the observed A1 instead gives another psi
  stopifnot(abs(.ltmle(d, Lc, Ac, "Y", dynamic, set_prior_a = FALSE)$psi - rdy$psi) > 1e-4)
  # continuous outcome: ltmle's rescaled logistic fits and p2a's linear variant
  c11 <- .ltmle(d, Lc, Ac, "Yc", always); c00 <- .ltmle(d, Lc, Ac, "Yc", never)
  l11 <- .ltmle(d, Lc, Ac, "Yc", always, q = "linear"); l00 <- .ltmle(d, Lc, Ac, "Yc", never, q = "linear")
  stopifnot(min(d$Yc) < 0, max(d$Yc) > 1)
  # randomized design, linear Q (correct specification): truth 5.5
  Lr <- list("L0", "Lr"); Ar <- c("R1", "R2")
  x11 <- .ltmle(d, Lr, Ar, "Yr", always, q = "linear"); x00 <- .ltmle(d, Lr, Ar, "Yr", never, q = "linear")
  rand_se <- .lt_se(x11$ic - x00$ic)
  stopifnot(abs(x11$psi - x00$psi - 5.5) < 3 * rand_se)
  # three time points
  L3 <- list("L0", "L1", "L2"); A3 <- c("A1", "A2", "A3")
  dyn3 <- cbind(as.integer(d$L0 > 0), as.integer(d$L1 > 0), as.integer(d$L2 > 0))
  t111 <- .ltmle(d, L3, A3, "Y3", st(1, 1, 1)); t000 <- .ltmle(d, L3, A3, "Y3", st(0, 0, 0))
  tdy <- .ltmle(d, L3, A3, "Y3", dyn3); t101 <- .ltmle(d, L3, A3, "Y3", st(1, 0, 1))
  # ltmle 1.3-0 cross-check of the logistic-Q regime means and EIC SEs (the linear variant has no
  # ltmle counterpart)
  .lt_same(.lt_pkg(d, Lc, Ac, "Y", always), r11, "always")
  .lt_same(.lt_pkg(d, Lc, Ac, "Y", never), r00, "never")
  .lt_same(.lt_pkg(d, Lc, Ac, "Y", dynamic), rdy, "dynamic")
  .lt_same(.lt_pkg(d, Lc, Ac, "Y", always, gbounds = gb), b11, "always, gbounds (0.1, 0.5)")
  .lt_same(.lt_pkg(d, Lc, Ac, "Y", never, gbounds = gb), b00, "never, gbounds (0.1, 0.5)")
  .lt_same(.lt_pkg(d, Lc, Ac, "Yc", always), c11, "always, rescaled Yc")
  .lt_same(.lt_pkg(d, Lc, Ac, "Yc", never), c00, "never, rescaled Yc")
  .lt_same(.lt_pkg(d, L3, A3, "Y3", st(1, 1, 1)), t111, "three periods, always")
  .lt_same(.lt_pkg(d, L3, A3, "Y3", st(0, 0, 0)), t000, "three periods, never")
  .lt_same(.lt_pkg(d, L3, A3, "Y3", dyn3), tdy, "three periods, dynamic")
  .lt_same(.lt_pkg(d, L3, A3, "Y3", st(1, 0, 1)), t101, "three periods, (1, 0, 1)")
  v <- list(
    g1_coef = named(coef(r11$gfits[[1]])), g2_coef = named(coef(r11$gfits[[2]])),
    q2_coef = named(coef(r11$qfits[[2]])), q1_coef_always = named(coef(r11$qfits[[1]])), q1_coef_never = named(coef(r00$qfits[[1]])),
    psi_always = r11$psi, psi_never = r00$psi, psi_dynamic = rdy$psi, psi_10 = r10$psi,
    se_always = .lt_se(r11$ic), se_never = .lt_se(r00$ic), se_dynamic = .lt_se(rdy$ic), se_10 = .lt_se(r10$ic),
    eps_always = vec(r11$eps), eps_never = vec(r00$eps), eps_dynamic = vec(rdy$eps), eps_10 = vec(r10$eps),
    n_follow_always = vec(r11$n_follow), n_follow_never = vec(r00$n_follow), n_follow_dynamic = vec(rdy$n_follow), n_follow_10 = vec(r10$n_follow),
    h_always_2_head = vec(r11$H[1:8, 2]), h_dynamic_1_head = vec(rdy$H[1:8, 1]), cumg_never_2_head = vec(r00$cum[1:8, 2]),
    ate = ate, ate_se = ate_se, ate_z = z, ate_p = 2 * pnorm(-abs(z)),
    ate_ci_lower = ate - qnorm(0.975) * ate_se, ate_ci_upper = ate + qnorm(0.975) * ate_se,
    ate_dynamic_vs_never = rdy$psi - r00$psi, ate_dynamic_vs_never_se = .lt_se(rdy$ic - r00$ic),
    ic_ate_head = vec((r11$ic - r00$ic)[1:5]),
    psi_always_gb = b11$psi, psi_never_gb = b00$psi, ate_gb = b11$psi - b00$psi, ate_se_gb = .lt_se(b11$ic - b00$ic),
    n_bounded_always_gb = vec(b11$n_bounded), n_bounded_never_gb = vec(b00$n_bounded),
    n_bounded_follow_always_gb = vec(b11$n_bounded_follow), n_bounded_follow_never_gb = vec(b00$n_bounded_follow),
    yc_range = vec(c11$scale),
    psi_always_cont = c11$psi, psi_never_cont = c00$psi, ate_cont = c11$psi - c00$psi, ate_se_cont = .lt_se(c11$ic - c00$ic),
    eps_always_cont = vec(c11$eps),
    psi_always_lin = l11$psi, psi_never_lin = l00$psi, ate_lin = l11$psi - l00$psi, ate_se_lin = .lt_se(l11$ic - l00$ic),
    eps_always_lin = vec(l11$eps), eps_never_lin = vec(l00$eps),
    psi_always_rand = x11$psi, psi_never_rand = x00$psi, ate_rand = x11$psi - x00$psi, ate_se_rand = rand_se, truth_rand = 5.5,
    t3_g3_coef = named(coef(t111$gfits[[3]])), t3_q2_coef_always = named(coef(t111$qfits[[2]])),
    t3_psi_always = t111$psi, t3_psi_never = t000$psi, t3_psi_dynamic = tdy$psi, t3_psi_101 = t101$psi,
    t3_se_always = .lt_se(t111$ic), t3_se_never = .lt_se(t000$ic), t3_se_dynamic = .lt_se(tdy$ic), t3_se_101 = .lt_se(t101$ic),
    t3_eps_always = vec(t111$eps), t3_eps_never = vec(t000$eps), t3_eps_dynamic = vec(tdy$eps), t3_eps_101 = vec(t101$eps),
    t3_n_follow_always = vec(t111$n_follow), t3_n_follow_never = vec(t000$n_follow), t3_n_follow_dynamic = vec(tdy$n_follow),
    t3_n_follow_101 = vec(t101$n_follow), t3_n_bounded_never = vec(t000$n_bounded), t3_n_bounded_follow_never = vec(t000$n_bounded_follow),
    t3_ate = t111$psi - t000$psi, t3_ate_se = .lt_se(t111$ic - t000$ic))
  list(values = v,
       reference = "LTMLE (Lendle et al. 2017) as ltmle 1.x with its defaults (stratify = FALSE, gbounds = c(0.01, 1) on the cumulative g, SetA predictions, weighted-intercept fluctuation) and variance.method = 'ic', written out with glm and cross-checked against ltmle::ltmle 1.3-0 (logistic-Q regime means and EIC SEs to 1e-8 relative); the linear Q model is p2a's Gaussian variant written out with lm",
       call = "ltmle(d, Anodes = c('A1','A2'), Lnodes = 'L1', Ynodes = 'Y', abar = list(c(1,1), c(0,0)), SL.library = 'glm', stratify = FALSE, variance.method = 'ic'); abar = c(1,0) and the rule A_t = 1{L_{t-1} > 0}; gbounds = c(0.1, 0.5); Ynodes = 'Yc' (rescaled, and p2a's linear variant); randomized R1, Lr, R2, Yr (linear); Anodes = c('A1','A2','A3'), Lnodes = c('L1','L2'), Ynodes = 'Y3'",
       packages = c("ltmle"))
})

# ltmle_saturation: an eight-node design (an R port of the sparse_followers fixture of ltmle.rs's
# tests: two uniform covariates per node, A_t ~ plogis(0.3 L_t1 + 0.2 A_{t-1}), Y ~ plogis(latent -
# mean(latent)) with latent = 1 + 0.2 L_11 + U(-0.5, 0.5) + sum_t (0.5 - 0.2 (t - 1) / 8) A_t) whose
# never regime has a handful of followers at the last node.
# Seed search (T3-LTMLE-TOL, 2026-11-09): the first seed in 1..200 such that ltmle::ltmle's never
# regime has at least one no-variation Q node, the written-out estimator's never mean moves by more
# than 1e-3 between glm epsilon 1e-8 and 1e-14, and ltmle succeeds. Seed 1: no no-variation node;
# seed 2: the written-out never mean moves by 1e-5 only; seed 3 meets all three (9 never followers,
# 7 no-variation nodes, written-out never mean 7.4e-13 at 1e-8 and 0.0169 at 1e-14).
.lt_sat_data <- function(seed = 3, n = 2500, TT = 8) {
  set.seed(seed)
  L <- vector("list", TT); A <- vector("list", TT); prev <- numeric(n)
  for (t in seq_len(TT)) {
    L[[t]] <- cbind(runif(n, -1, 1), runif(n, -1, 1))
    A[[t]] <- rbinom(n, 1, plogis(0.3 * L[[t]][, 1] + 0.2 * prev)); prev <- A[[t]]
  }
  latent <- 1 + 0.2 * L[[1]][, 1] + runif(n, -0.5, 0.5)
  for (t in seq_len(TT)) latent <- latent + (0.5 - 0.2 * (t - 1) / TT) * A[[t]]
  Y <- rbinom(n, 1, plogis(latent - mean(latent)))
  d <- data.frame(row.names = seq_len(n))
  for (t in seq_len(TT)) {
    d[[paste0("L", t, "a")]] <- round(L[[t]][, 1], 10); d[[paste0("L", t, "b")]] <- round(L[[t]][, 2], 10)
    d[[paste0("A", t)]] <- A[[t]]
  }
  d$Y <- Y
  list(d = d, mean_latent = mean(latent))
}
# ltmle::ltmle itself; its score re-solve restarts from rnorm, so the seed is set before each call.
.lt_sat_fit <- function(d, Lc, Ac, abar) {
  nodes <- c(unlist(lapply(seq_along(Ac), function(s) c(Lc[[s]], Ac[s]))), "Y")
  w <- character(0)
  set.seed(20261109)
  fit <- suppressMessages(withCallingHandlers(ltmle::ltmle(d[, nodes], Anodes = Ac, Lnodes = unlist(Lc[-1]), Ynodes = "Y",
           abar = abar, SL.library = "glm", stratify = FALSE, variance.method = "ic", estimate.time = FALSE),
           warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") }))
  list(fit = fit, warnings = w)
}

case("ltmle_saturation", "causal/tmle", "LTMLE on a regime with few followers: ltmle::ltmle itself as the reference (its no-variation rule, score check and positivity warning), with the written-out estimator's tolerance dependence and the analytic estimands recorded", function(ctx) {
  g <- .lt_sat_data(3); d <- ctx$gen_dataset("ltmle_saturation", g$d); n <- nrow(d); TT <- 8
  stopifnot(all(d$Y %in% c(0, 1)), is.integer(d$A1))
  Lc <- lapply(1:TT, function(s) c(paste0("L", s, "a"), paste0("L", s, "b"))); Ac <- paste0("A", 1:TT)
  a1 <- .lt_sat_fit(d, Lc, Ac, rep(1, TT)); a0 <- .lt_sat_fit(d, Lc, Ac, rep(0, TT))
  ab <- .lt_sat_fit(d, Lc, Ac, list(rep(1, TT), rep(0, TT)))
  s1 <- summary(a1$fit)$treatment; s0 <- summary(a0$fit)$treatment; ate <- summary(ab$fit)$effect.measures$ATE
  const <- function(f) as.numeric(which(sapply(f$fit$fit$Q, function(q) inherits(q, "no.Y.variation"))))
  resolved <- function(f) as.numeric(which(sapply(f$fit$fit$Qstar, function(q) is.list(q) && !is.null(q$ltmle.msg))))
  positivity <- function(f) any(grepl("positivity", f$warnings))
  # the written-out estimator (glm fits at the given epsilon); its EIC check is off: on this design
  # its targeting does not solve the EIC equation, which is why ltmle::ltmle is the reference here
  w1 <- .ltmle(d, Lc, Ac, "Y", matrix(1, n, TT))
  w8 <- suppressWarnings(.ltmle(d, Lc, Ac, "Y", matrix(0, n, TT), ctl = glm.control(epsilon = 1e-8, maxit = 100), check_ic = FALSE))
  w14 <- suppressWarnings(.ltmle(d, Lc, Ac, "Y", matrix(0, n, TT), ctl = glm.control(epsilon = 1e-14, maxit = 100), check_ic = FALSE))
  stopifnot(length(const(a0)) > 0, abs(w8$psi - w14$psi) > 1e-3, positivity(a0), positivity(ab))
  # analytic estimands: Y depends on L_1a, U(-0.5, 0.5) and the treatments, given mean(latent)
  est <- function(c0, b) integrate(function(l) sapply(l, function(li) integrate(function(u) plogis(c0 + b * li + u), -0.5, 0.5)$value), -1, 1)$value / 2
  m <- g$mean_latent
  v <- list(
    psi_always_sat = unname(s1$estimate), psi_never_sat = unname(s0$estimate),
    se_always_sat = unname(s1$std.dev), se_never_sat = unname(s0$std.dev),
    ate_sat = unname(ate$estimate), ate_se_sat = unname(ate$std.dev),
    positivity_warning = as.numeric(positivity(ab)),
    positivity_warning_always = as.numeric(positivity(a1)), positivity_warning_never = as.numeric(positivity(a0)),
    rare_warning = as.numeric(any(grepl("rare", ab$warnings))),
    n_below_gbound_always = sum(w1$cum_unb[, TT] < 0.01), n_below_gbound_never = sum(w14$cum_unb[, TT] < 0.01),
    # per node 1..8, 1 where ltmle predicted the Q response as a constant (no.Y.variation) or re-solved
    # the score equation (an empty vector is not a case value)
    constant_q_always = vec(as.numeric(1:TT %in% const(a1))), constant_q_never = vec(as.numeric(1:TT %in% const(a0))),
    resolved_always = vec(as.numeric(1:TT %in% resolved(a1))), resolved_never = vec(as.numeric(1:TT %in% resolved(a0))),
    n_follow_always = vec(w1$n_follow), n_follow_never = vec(w14$n_follow),
    writtenout_psi_always = w1$psi,
    writtenout_psi_never_1e8 = w8$psi, writtenout_psi_never_1e14 = w14$psi, writtenout_icmean_never_1e14 = mean(w14$ic),
    truth_always = est(1 + sum(0.5 - 0.2 * (0:(TT - 1)) / TT) - m, 0.2), truth_never = est(1 - m, 0.2))
  list(values = v,
       reference = "ltmle::ltmle 1.3-0 (glm, stratify = FALSE, gbounds = c(0.01, 1), variance.method = 'ic'): regime means and SEs from single-regime calls, the ATE and its SE from the two-regime call, its no-variation Q nodes (class no.Y.variation), its score re-solves (ltmle.msg) and its positivity warning; the written-out glm estimator of ltmle_two_period does not solve the EIC equation on this design and is recorded for documentation only; the analytic estimands are context, not pins",
       call = "ltmle(d, Anodes = A1..A8, Lnodes = L2a..L8b, Ynodes = 'Y', abar = rep(1, 8) / rep(0, 8) / list(rep(1, 8), rep(0, 8)), SL.library = 'glm', stratify = FALSE, variance.method = 'ic'), set.seed(20261109) before each call",
       packages = c("ltmle"))
})
