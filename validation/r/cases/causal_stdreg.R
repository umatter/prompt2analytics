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

# ---------------------------------------------------------------- stdreg
.sr_std_data <- function(ctx) {
  set.seed(20261209)
  n <- 300
  x1 <- rnorm(n); x2 <- runif(n, -1, 1)
  a <- rbinom(n, 1, plogis(0.4 * x1 - 0.3 * x2))
  yc <- 1 + 0.8 * a + 0.5 * x1 - 0.4 * x2 + 0.3 * a * x1 + rnorm(n)
  yb <- rbinom(n, 1, plogis(-0.5 + 0.9 * a + 0.6 * x1 - 0.5 * x2))
  yp <- rpois(n, exp(5.3 + 0.2 * a + 0.1 * x1 + 0.05 * x2))     # mean about 220: CAUS-38's large counts
  x1 <- round(x1, 10); x2 <- round(x2, 10)
  ys <- as.integer(x1 > 0.3)                                    # x1 separates ys completely
  ctx$gen_dataset("stdreg_sim", data.frame(yc = round(yc, 10), yb = yb, yp = yp, ys = ys, a = a,
                                           x1 = x1, x2 = x2, x1dup = 2 * x1 + 1))
}
# standardised means and the three contrasts of a fitted model
.sr_standardise <- function(fit, d) {
  p1 <- predict(fit, newdata = transform(d, a = 1), type = "response")
  p0 <- predict(fit, newdata = transform(d, a = 0), type = "response")
  t <- d$a == 1
  list(ey1 = mean(p1), ey0 = mean(p0), ate = mean(p1) - mean(p0),
       att = mean(p1[t]) - mean(p0[t]), atc = mean(p1[!t]) - mean(p0[!t]),
       att_ey1 = mean(p1[t]), att_ey0 = mean(p0[t]), p1 = p1, p0 = p0)
}
# stdReg::stdGlm delta method: stacked estimating equations (score_i(beta), m_i(a) - mu_a), a in (1, 0)
.sr_stdreg_sandwich <- function(fit, d, s) {
  X <- model.matrix(fit); n <- nrow(X); k <- ncol(X)
  mu <- fitted(fit); y <- fit$y
  eta <- fit$linear.predictors
  fam <- family(fit)
  w <- fam$mu.eta(eta)^2 / fam$variance(mu)
  r <- (y - mu) / fam$mu.eta(eta)
  disp <- if (fam$family == "gaussian") sum(r^2 * w) / n else 1
  score <- X * as.vector(w * r) / disp
  X1 <- model.matrix(delete.response(terms(fit)), transform(d, a = 1))
  X0 <- model.matrix(delete.response(terms(fit)), transform(d, a = 0))
  d1 <- fam$mu.eta(as.vector(X1 %*% coef(fit))); d0 <- fam$mu.eta(as.vector(X0 %*% coef(fit)))
  U <- cbind(score, s$p1 - s$ey1, s$p0 - s$ey0)
  J <- matrix(0, k + 2, k + 2)
  J[1:k, 1:k] <- crossprod(X * as.vector(w), X) / n / disp
  J[k + 1, 1:k] <- -colMeans(X1 * d1); J[k + 2, 1:k] <- -colMeans(X0 * d0)
  J[k + 1, k + 1] <- 1; J[k + 2, k + 2] <- 1
  Ji <- solve(J)
  V <- Ji %*% (crossprod(U) / n) %*% t(Ji) / n
  g <- c(rep(0, k), 1, -1)
  c(ey1_se = sqrt(V[k + 1, k + 1]), ey0_se = sqrt(V[k + 2, k + 2]), ate_se = sqrt(drop(t(g) %*% V %*% g)))
}
.sr_std_values <- function(fit, d, prefix) {
  s <- .sr_standardise(fit, d)
  se <- .sr_stdreg_sandwich(fit, d, s)
  v <- list(vec(coef(fit)), s$ey1, s$ey0, s$ate, s$att, s$atc, s$att_ey1, s$att_ey0,
            se[["ey1_se"]], se[["ey0_se"]], se[["ate_se"]])
  names(v) <- paste0(prefix, c("_coef", "_ey1", "_ey0", "_ate", "_att", "_atc", "_att_ey1", "_att_ey0",
                               "_stdreg_ey1_se", "_stdreg_ey0_se", "_stdreg_ate_se"))
  v
}

case("stdreg_glm", "causal/stdreg",
     "Regression standardisation with glm outcome models: logistic (with and without A x W interactions) and Poisson with large counts; ATE/ATT/ATC; stdReg delta-method SEs recorded",
     function(ctx) {
  d <- .sr_std_data(ctx)
  fl <- glm(yb ~ x1 + x2 + a, data = d, family = binomial, control = .sr_ctl)
  fli <- glm(yb ~ x1 + x2 + a + a:x1 + a:x2, data = d, family = binomial, control = .sr_ctl)
  fp <- glm(yp ~ x1 + x2 + a, data = d, family = poisson, control = .sr_ctl)
  stopifnot(fl$converged, fp$converged, fli$converged, mean(d$yp) > 150)
  v <- c(.sr_std_values(fl, d, "logit"), .sr_std_values(fli, d, "logit_int"), .sr_std_values(fp, d, "pois"),
         list(logit_r2 = 1 - fl$deviance / fl$null.deviance, pois_r2 = 1 - fp$deviance / fp$null.deviance,
              pois_mean_y = mean(d$yp)))
  list(values = v, dataset = "validation/datasets/generated/stdreg_sim.csv",
       reference = "stats::glm (epsilon 1e-14) + standardisation as stdReg::stdGlm (stdReg not installed: point estimates and the stacked-equation delta method written out)",
       call = "glm(yb ~ x1 + x2 + a, binomial); glm(yb ~ x1 + x2 + a + a:x1 + a:x2, binomial); glm(yp ~ x1 + x2 + a, poisson); mean(predict(fit, a = 1)) - mean(predict(fit, a = 0))",
       packages = character(0))
})

case("stdreg_linear", "causal/stdreg",
     "Regression standardisation with a linear outcome model: with A x W interactions, and with an aliased covariate (lm: NA coefficient, estimable predictions)",
     function(ctx) {
  d <- .sr_std_data(ctx)
  fi <- lm(yc ~ x1 + x2 + a + a:x1 + a:x2, data = d)
  fa <- lm(yc ~ x1 + x2 + x1dup + a, data = d)
  stopifnot(is.na(coef(fa)[["x1dup"]]))
  si <- .sr_standardise(fi, d)
  fa_w <- tryCatch({ predict(fa, newdata = transform(d, a = 1)); "" }, warning = function(w) conditionMessage(w))
  sa <- .sr_standardise(fa, d)
  v <- list(int_coef = vec(coef(fi)), int_ey1 = si$ey1, int_ey0 = si$ey0, int_ate = si$ate,
            int_att = si$att, int_atc = si$atc, int_r2 = summary(fi)$r.squared,
            alias_coef = vec(coef(fa)), alias_ey1 = sa$ey1, alias_ey0 = sa$ey0, alias_ate = sa$ate,
            alias_rank = fa$rank, alias_df_residual = fa$df.residual,
            alias_predict_warns = as.numeric(nzchar(fa_w)))
  list(values = v, dataset = "validation/datasets/generated/stdreg_sim.csv",
       reference = "stats::lm + standardisation as stdReg::stdGlm (gaussian); aliased column as lm (dqrdc2, NA coefficient; R 4.3.3 predict.lm does not warn because the counterfactual rows are estimable)",
       call = "lm(yc ~ x1 + x2 + a + a:x1 + a:x2); lm(yc ~ x1 + x2 + x1dup + a)", packages = character(0))
})

case("stdreg_separation", "causal/stdreg",
     "Regression standardisation with a logistic outcome model under complete separation (glm(maxit = 100) warns; the standardised contrast is numerically 0)",
     function(ctx) {
  d <- .sr_std_data(ctx)
  msgs <- character(0)
  fit <- withCallingHandlers(glm(ys ~ x1 + x2 + a, data = d, family = binomial, control = glm.control(maxit = 100)),
                             warning = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning") })
  stopifnot(any(msgs == "glm.fit: fitted probabilities numerically 0 or 1 occurred"))
  s <- .sr_standardise(fit, d)
  eps10 <- 10 * .Machine$double.eps
  v <- list(sep_ate = s$ate, sep_ey1 = s$ey1, sep_ey0 = s$ey0,
            sep_fitted_extreme = sum(fitted(fit) < eps10 | fitted(fit) > 1 - eps10),
            sep_converged = as.numeric(fit$converged),
            sep_warn_fitted01 = as.numeric(any(msgs == "glm.fit: fitted probabilities numerically 0 or 1 occurred")),
            sep_warn_noconv = as.numeric(any(msgs == "glm.fit: algorithm did not converge")),
            sep_mean_ys = mean(d$ys))
  list(values = v, dataset = "validation/datasets/generated/stdreg_sim.csv",
       reference = "stats::glm(binomial, epsilon 1e-8, maxit = 100) warnings under complete separation; standardisation as stdReg",
       call = "glm(ys ~ x1 + x2 + a, binomial, control = glm.control(maxit = 100)) with ys = 1{x1 > 0.3}", packages = character(0))
})

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

# ---------------------------------------------------------------- C-TMLE (written out)
# V-fold risk of the g-model with covariates S: Q (Qv[[v]]), g and epsilon fitted on the training folds,
# loss on the validation fold
.sr_cv_risk <- function(d, y, S, logistic, gb, V, pen, Qv) {
  n <- nrow(d); fold <- (seq_len(n) - 1) %% V + 1
  r <- s <- e <- numeric(V)
  for (v in seq_len(V)) {
    tr <- which(fold != v); te <- which(fold == v)
    g <- .sr_fitg(d, S, tr, gb)$g
    f <- .sr_fluct(d[[y]], d$a, Qv[[v]], g, logistic, tr)
    r[v] <- sum((d[[y]][te] - f$QsA[te])^2)
    s[v] <- var(f$H[te] * (d[[y]][te] - f$QsA[te]) + f$Qs1[te] - f$Qs0[te])
    e[v] <- f$eps
  }
  list(rss = sum(r), rss_se = sqrt(V) * sd(r), varic = mean(s), varic_se = sd(s) / sqrt(V),
       penrss = sum(r) + pen * log(n) * length(S), r = r, s = s, e = e)
}
.sr_ctmle <- function(d, y, logistic, crit, order = "forward", pre = NULL, V = 5, gb = c(0.025, 0.975), pen = 1) {
  n <- nrow(d); fold <- (seq_len(n) - 1) %% V + 1
  Qv <- lapply(seq_len(V), function(v) .sr_fitQ(d, y, logistic, which(fold != v)))
  Qfull <- .sr_fitQ(d, y, logistic)
  val <- function(S) {
    cr <- .sr_cv_risk(d, y, S, logistic, gb, V, pen, Qv)
    c(crit = cr[[crit]], se = if (crit == "varic") cr$varic_se else cr$rss_se, rss = cr$rss, varic = cr$varic)
  }
  p <- length(.sr_W)
  cur <- if (order == "backward") 1:p else integer(0)
  path <- list(val(cur)); moved <- integer(0); first_step <- NULL
  repeat {
    cand <- if (order == "forward") setdiff(1:p, cur) else if (order == "backward") cur else
      { k <- length(moved) + 1; if (k > length(pre)) integer(0) else pre[k] }
    if (!length(cand)) break
    sets <- lapply(cand, function(j) if (order == "backward") setdiff(cur, j) else sort(c(cur, j)))
    vals <- t(sapply(sets, val))
    if (is.null(first_step)) first_step <- vals[, "crit"]
    b <- which.min(vals[, "crit"])                                   # first minimum, as p2a's strict `<`
    cur <- sets[[b]]; moved <- c(moved, cand[b]); path[[length(path) + 1]] <- vals[b, ]
  }
  get <- function(nm) sapply(path, `[[`, nm)
  crits <- get("crit"); ses <- get("se")
  ncov <- if (order == "backward") p - (seq_along(crits) - 1) else seq_along(crits) - 1
  sel <- which.min(crits)
  thr <- crits[sel] + ses[sel]; ok <- which(crits <= thr)
  onese <- ok[which.min(ncov[ok])]
  covs_at <- function(k) if (order == "backward") setdiff(1:p, moved[seq_len(k - 1)]) else sort(moved[seq_len(k - 1)])
  fin <- .sr_tmle_full(d, y, covs_at(sel), logistic, gb, Qfull)
  fin1 <- .sr_tmle_full(d, y, covs_at(onese), logistic, gb, Qfull)
  steps <- lapply(seq_along(crits), function(k) .sr_tmle_full(d, y, covs_at(k), logistic, gb, Qfull))
  list(first_step = vec(first_step), path_crit = vec(crits), path_se = vec(ses),
       path_rss = vec(get("rss")), path_varic = vec(get("varic")), path_ncov = vec(ncov),
       moved = vec(moved - 1), selected_step = sel - 1, selected_mask = vec(as.numeric(1:p %in% covs_at(sel))),
       ate = fin$ate, se = fin$se, epsilon = fin$eps, ci_lower = fin$ate - qnorm(0.975) * fin$se,
       ci_upper = fin$ate + qnorm(0.975) * fin$se, p = 2 * pnorm(-abs(fin$ate / fin$se)),
       onese_step = onese - 1, onese_selected_mask = vec(as.numeric(1:p %in% covs_at(onese))), onese_ate = fin1$ate,
       path_ate = vec(sapply(steps, `[[`, "ate")), path_ate_se = vec(sapply(steps, `[[`, "se")))
}
.sr_prefix <- function(l, p) { names(l) <- paste0(p, "_", names(l)); l }

case("ctmle_cv", "causal/ctmle",
     "C-TMLE forward / backward / prespecified selection of g with a real 5-fold CV risk (Q, g and epsilon fitted on the training folds); RSS, VarIC and PenRSS criteria",
     function(ctx) {
  d <- .sr_ct_data(ctx); n <- nrow(d); V <- 5
  fold <- (seq_len(n) - 1) %% V + 1
  # fold-level values of one candidate g = {w1, w3} (logistic Q), for the recompute-from-folds tests
  Qv <- lapply(seq_len(V), function(v) .sr_fitQ(d, "yb", TRUE, which(fold != v)))
  c13 <- .sr_cv_risk(d, "yb", c(1, 3), TRUE, c(0.025, 0.975), V, 2, Qv)
  v <- c(list(cv13_fold_rss = vec(c13$r), cv13_fold_varic = vec(c13$s), cv13_fold_epsilon = vec(c13$e),
              cv13_rss = c13$rss, cv13_rss_se = c13$rss_se, cv13_varic = c13$varic, cv13_varic_se = c13$varic_se,
              cv13_penrss2 = c13$penrss),
         .sr_prefix(.sr_ctmle(d, "yb", TRUE, "rss"), "logit_rss"),
         .sr_prefix(.sr_ctmle(d, "yb", TRUE, "varic"), "logit_varic"),
         .sr_prefix(.sr_ctmle(d, "yb", TRUE, "penrss", pen = 2), "logit_penrss"),
         .sr_prefix(.sr_ctmle(d, "yc", FALSE, "rss"), "linear_rss"),
         .sr_prefix(.sr_ctmle(d, "yc", FALSE, "varic"), "linear_varic"),
         .sr_prefix(.sr_ctmle(d, "yb", TRUE, "rss", order = "backward"), "logit_backward"),
         .sr_prefix(.sr_ctmle(d, "yb", TRUE, "rss", order = "pre", pre = c(3, 1)), "logit_pre31"))
  list(values = v, dataset = "validation/datasets/generated/ctmle_sim.csv",
       reference = "C-TMLE greedy selection of g with a fixed initial Q, criteria of Gruber & van der Laan (2010) as ctmle::ctmleDiscrete names them (CV likelihood, varIC.cv), written out; ctmle 0.1.2 (installed) is not used as a pin because it updates Q between clever covariates; glm epsilon 1e-14",
       call = "for each candidate g and fold v (i %% 5): Q, g, epsilon on the training folds; RSS and var(IC) on fold v",
       packages = character(0))
})
