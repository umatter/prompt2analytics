# validation/r/cases/causal_stdreg.R -- regression standardisation, TMLE and C-TMLE references
# (T3-WP29: CAUS-38, CAUS-45, CAUS-56, CAUS-57, CAUS-101; contributes CAUS-66, CAUS-73, X-10).
#  * stdreg: stats::glm / stats::lm outcome models, then standardisation of the predictions under
#    A = 1 and A = 0 over the sample (ATE), the treated (ATT) and the controls (ATC), as
#    stdReg::stdGlm does. stdReg is not installed: its point estimates are written out, and its
#    delta-method (stacked estimating equation sandwich) standard errors are recorded as
#    *_stdreg_*_se for the Tier 4 finding CAUS-39 (not asserted by T3-WP29).
#  * TMLE: tmle 2.1.1's fluctuation with two clever covariates H1W = A/g and H0W = (1 - A)/(1 - g)
#    (logistic for a binary Y, linear for a continuous Y), written out with glm / lm; a continuous Y
#    under the logistic Q is rescaled to [0, 1] by its range first, as tmle does. tmle::tmle 2.1.1
#    itself is recorded as tmle_pkg_*: on a binary Y it agrees to 1e-8 (checked here), on the rescaled
#    continuous Y it bounds Q differently (tmle_pkg_rescaled_*, informational).
#  * C-TMLE: p2a's greedy forward / backward / prespecified selection of the propensity covariates
#    with a fixed initial Q and one clever covariate H = A/g - (1 - A)/(1 - g), written out with the
#    real V-fold risk of CAUS-57: for each fold v, Q, g and the fluctuation epsilon are fitted on the
#    training folds and the loss is evaluated on the validation fold. ctmle 0.1.2 is installed but
#    not used as a pin: ctmle::ctmleDiscrete also updates Q between clever covariates. Folds:
#    observation i (0-based) is in fold i %% V. Criteria (Gruber & van der Laan 2010; ctmle's names):
#      RSS    = sum over folds of the validation-fold RSS (ctmle's `likelihood`), SE sqrt(V) sd(fold RSS);
#      VarIC  = mean over folds of the validation-fold variance of the influence curve
#               (ctmle's `varIC.cv`), SE sd(fold varIC) / sqrt(V);
#      PenRSS = RSS + penalty * log(n) * k (k covariates in g), SE as RSS.
#    The reported ATE at each step and at the selected step is the full-data TMLE with that g.
# All fits use glm.control(epsilon = 1e-14, maxit = 100): the p2a tests compare at the solution.

.sr_ctl <- glm.control(epsilon = 1e-14, maxit = 100)    # tight refit: p2a tests use tolerance 1e-14

# ---------------------------------------------------------------- TMLE and C-TMLE data and helpers
.sr_ct_data <- function(ctx) {
  set.seed(20261210)
  n <- 300
  w1 <- rnorm(n); w2 <- rnorm(n); w3 <- rnorm(n); w4 <- runif(n, -1, 1)
  a <- rbinom(n, 1, plogis(-0.2 + 0.8 * w1 + 1.0 * w2))           # w1 confounder, w2 instrument
  yb <- rbinom(n, 1, plogis(-0.5 + 1.0 * a + 0.9 * w1 + 0.8 * w3)) # w3 precision variable, w4 noise
  yc <- 1 + 2 * a + 1.2 * w1 + 1.0 * w3 + rnorm(n)
  ctx$gen_dataset("ctmle_sim", data.frame(yb = yb, yc = round(yc, 10), a = a, w1 = round(w1, 10),
                                          w2 = round(w2, 10), w3 = round(w3, 10), w4 = round(w4, 10)))
}
.sr_W <- c("w1", "w2", "w3", "w4")
.sr_clip <- function(p, lo, hi) pmin(pmax(p, lo), hi)
.sr_qbound <- function(q) .sr_clip(q, 1e-10, 1 - 1e-10)                  # p2a's guard on Q before qlogis
# initial Q fitted on rows `rows` (logistic: binomial glm, quasibinomial for a Y in [0, 1] that is not
# 0/1, which gives the binomial estimates without the non-integer warning; linear: lm); predictions
# for every row of d
.sr_fitQ <- function(d, y, logistic, rows = seq_len(nrow(d)), ctl = .sr_ctl) {
  f <- as.formula(paste(y, "~", paste(c(.sr_W, "a"), collapse = " + ")))
  fam <- if (all(d[[y]] %in% c(0, 1))) binomial else quasibinomial
  m <- if (logistic) glm(f, data = d[rows, ], family = fam, control = ctl) else lm(f, data = d[rows, ])
  list(m = m, QA = predict(m, newdata = d, type = "response"),
       Q1 = predict(m, newdata = transform(d, a = 1), type = "response"),
       Q0 = predict(m, newdata = transform(d, a = 0), type = "response"))
}
# g fitted on rows `rows` with the covariates S (indices into .sr_W); bounded predictions for every row
.sr_fitg <- function(d, S, rows, gb, ctl = .sr_ctl) {
  f <- as.formula(paste("a ~", if (length(S)) paste(.sr_W[S], collapse = " + ") else "1"))
  m <- glm(f, data = d[rows, ], family = binomial, control = ctl)
  list(m = m, g = .sr_clip(predict(m, newdata = d, type = "response"), gb[1], gb[2]))
}
# fluctuation fitted on rows `rows`; epsilon(s) and Q*(A,W), Q*(1,W), Q*(0,W) for every row.
# One-dimensional (C-TMLE): H = A/g - (1 - A)/(1 - g). Bivariate (TMLE, as tmle 2.1.1):
# H1 = A/g, H0 = (1 - A)/(1 - g), Q*(1) = Q(1) + e1 / g, Q*(0) = Q(0) + e0 / (1 - g) on the link scale.
.sr_fluct <- function(y, a, Q, g, logistic, rows, ctl = .sr_ctl, bivariate = FALSE) {
  H <- a / g - (1 - a) / (1 - g)
  if (bivariate) {
    H1 <- a / g; H0 <- (1 - a) / (1 - g)
    if (logistic) {
      off <- qlogis(.sr_qbound(Q$QA))
      fit <- glm(y[rows] ~ -1 + H0[rows] + H1[rows], offset = off[rows], family = quasibinomial, control = ctl)
      e <- unname(coef(fit))
      list(eps = e[2], eps1 = e[2], eps0 = e[1], H = H, QsA = plogis(off + e[1] * H0 + e[2] * H1),
           Qs1 = plogis(qlogis(.sr_qbound(Q$Q1)) + e[2] / g), Qs0 = plogis(qlogis(.sr_qbound(Q$Q0)) + e[1] / (1 - g)))
    } else {
      r <- y - Q$QA
      e <- unname(coef(lm(r[rows] ~ -1 + H0[rows] + H1[rows])))
      list(eps = e[2], eps1 = e[2], eps0 = e[1], H = H, QsA = Q$QA + e[1] * H0 + e[2] * H1,
           Qs1 = Q$Q1 + e[2] / g, Qs0 = Q$Q0 + e[1] / (1 - g))
    }
  } else if (logistic) {
    off <- qlogis(.sr_qbound(Q$QA))
    fit <- glm(y[rows] ~ -1 + H[rows], offset = off[rows], family = quasibinomial, control = ctl)
    eps <- unname(coef(fit))
    list(eps = eps, H = H, QsA = plogis(off + eps * H),
         Qs1 = plogis(qlogis(.sr_qbound(Q$Q1)) + eps / g), Qs0 = plogis(qlogis(.sr_qbound(Q$Q0)) - eps / (1 - g)))
  } else {
    eps <- sum(H[rows] * (y[rows] - Q$QA[rows])) / sum(H[rows]^2)
    list(eps = eps, H = H, QsA = Q$QA + eps * H, Qs1 = Q$Q1 + eps / g, Qs0 = Q$Q0 - eps / (1 - g))
  }
}
.sr_tmle_full <- function(d, y, S, logistic, gb, Q = .sr_fitQ(d, y, logistic), ctl = .sr_ctl) {
  n <- nrow(d); all <- seq_len(n)
  g <- .sr_fitg(d, S, all, gb, ctl)$g
  f <- .sr_fluct(d[[y]], d$a, Q, g, logistic, all, ctl)
  ate <- mean(f$Qs1 - f$Qs0)
  ic <- f$H * (d[[y]] - f$QsA) + f$Qs1 - f$Qs0 - ate
  list(ate = ate, se = sd(ic) / sqrt(n), eps = f$eps, g = g, ic = ic, f = f)
}

# ---------------------------------------------------------------- TMLE (two clever covariates)
case("tmle_glm", "causal/tmle",
     "TMLE of the ATE with glm Q and g: tmle 2.1.1's bivariate logistic fluctuation (binary Y; a continuous Y rescaled to [0, 1]) and linear fluctuation (continuous Y); IC-based SE",
     function(ctx) {
  d <- .sr_ct_data(ctx); n <- nrow(d)
  gb <- c(0.01, 0.99)
  at <- c(which(d$a == 1)[1:3], which(d$a == 0)[1:3])            # three treated and three control rows
  one <- function(y, logistic, ctl) {
    Q <- .sr_fitQ(d, y, logistic, ctl = ctl)
    gm <- .sr_fitg(d, 1:4, seq_len(n), gb, ctl)
    graw <- fitted(gm$m)
    f <- .sr_fluct(d[[y]], d$a, Q, gm$g, logistic, seq_len(n), ctl, bivariate = TRUE)
    ate <- mean(f$Qs1 - f$Qs0)
    ic <- f$H * (d[[y]] - f$QsA) + f$Qs1 - f$Qs0 - ate
    se <- sd(ic) / sqrt(n)
    list(q_coef = vec(coef(Q$m)), g_coef = vec(coef(gm$m)), epsilon1 = f$eps1, epsilon0 = f$eps0,
         ate = ate, se = se,
         ci_lower = ate - qnorm(0.975) * se, ci_upper = ate + qnorm(0.975) * se, z = ate / se,
         p = 2 * pnorm(-abs(ate / se)), n_truncated = sum(graw < gb[1] | graw > gb[2]),
         rows_at = vec(at - 1), q_init_at = vec(Q$QA[at]), q_star_at = vec(f$QsA[at]), q_star1_at = vec(f$Qs1[at]),
         q_star0_at = vec(f$Qs0[at]), g_at = vec(gm$g[at]), ic_at = vec(ic[at]), ic_mean = mean(ic))
  }
  lb <- one("yb", TRUE, .sr_ctl); lc <- one("yc", FALSE, .sr_ctl)
  names(lb) <- paste0("logit_", names(lb)); names(lc) <- paste0("linear_", names(lc))
  # CAUS-45: a continuous Y with the logistic Q, rescaled as R's tmle (Gruber & van der Laan 2010):
  # Y* = (Y - a) / (b - a); quasibinomial Q, logistic fluctuation; ATE and SE scaled back by b - a.
  rng <- range(d$yc); wd <- diff(rng); d$ycs <- (d$yc - rng[1]) / wd
  ls <- one("ycs", TRUE, .sr_ctl)
  rs <- list(rescaled_ate = wd * ls$ate, rescaled_se = wd * ls$se,
             rescaled_epsilon1 = ls$epsilon1, rescaled_epsilon0 = ls$epsilon0,   # logit scale
             rescaled_range = vec(rng))
  # cross-check: tmle 2.1.1 (two clever covariates H0W, H1W; gbound as here)
  W <- d[, .sr_W]
  pk <- suppressWarnings(tmle::tmle(Y = d$yb, A = d$a, W = W, Qform = "Y ~ A + w1 + w2 + w3 + w4",
                                    gform = "A ~ w1 + w2 + w3 + w4", cvQinit = FALSE, family = "binomial", gbound = gb))
  # tmle also bounds Q to 1% beyond the observed range and to [1 - alpha, alpha] (informational)
  pkr <- suppressWarnings(tmle::tmle(Y = d$yc, A = d$a, W = W, Qform = "Y ~ A + w1 + w2 + w3 + w4",
                                     gform = "A ~ w1 + w2 + w3 + w4", cvQinit = FALSE, family = "gaussian",
                                     fluctuation = "logistic", gbound = gb))
  # the written-out bivariate TMLE is tmle 2.1.1's estimator: it agrees to 1e-8 (tmle_pkg_* is pinned too)
  stopifnot(abs(lb$logit_ate / pk$estimates$ATE$psi - 1) < 1e-8,
            abs(lb$logit_se / sqrt(pk$estimates$ATE$var.psi) - 1) < 1e-8)
  v <- c(lb, lc, rs, list(tmle_pkg_ate = pk$estimates$ATE$psi, tmle_pkg_se = sqrt(pk$estimates$ATE$var.psi),
                          tmle_pkg_rescaled_ate = pkr$estimates$ATE$psi,
                          tmle_pkg_rescaled_se = sqrt(pkr$estimates$ATE$var.psi)))
  list(values = v, dataset = "validation/datasets/generated/ctmle_sim.csv",
       reference = "TMLE with tmle 2.1.1's two clever covariates H1W = A/g, H0W = (1 - A)/(1 - g) (van der Laan & Rose 2011, ch. 4) written out with stats::glm (epsilon 1e-14); a continuous Y with the logistic Q rescaled to [0, 1] by its range (Gruber & van der Laan 2010); tmle::tmle(Qform, gform, cvQinit = FALSE) 2.1.1 recorded as tmle_pkg_* (agrees to 1e-8 on the binary Y)",
       call = "glm(Y ~ w1 + w2 + w3 + w4 + a, binomial); glm(a ~ w1 + w2 + w3 + w4, binomial); glm(Y ~ -1 + H0W + H1W, offset = qlogis(Q), quasibinomial)",
       packages = c("tmle"))
})

