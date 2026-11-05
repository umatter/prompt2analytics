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
