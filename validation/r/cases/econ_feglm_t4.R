# validation/r/cases/econ_feglm_t4.R -- T4-WP10 FEGLM references (ECON-20, ECON-66, ECON-67, ECON-85).
#  * feglm_mle: logit, probit and Poisson with id + time and with id alone, Gaussian with id + time, on
#    feglm_panel (feglm_families' data). The reference is glm with factor dummies on the rows alpaca and
#    fixest keep (groups whose outcome is all 0 or all 1 dropped), fitted to epsilon 1e-16: coefficients,
#    inverse-Hessian SEs (alpaca's; dispersion 1 for binomial and Poisson), log-likelihood, AIC, BIC,
#    deviance, the intercept-only null log-likelihood and deviance, residual df, n.
#  * feglm_separation: a two-way panel where dropping the all-1 and all-0 entities leaves a period whose
#    outcome is all 0 (alpaca and fixest drop it too: the rule is applied until nothing changes), an
#    entity whose counts are all 0, and a singleton entity with a positive count (alpaca keeps it,
#    fixest drops it as a singleton).
# alpaca is called through its namespace only: attaching it would mask fixest::feglm in econ_panel_t3.R.

glm_ref <- function(v, pre, d, y, fe, family) {
  f <- as.formula(paste(y, "~ x + x2 +", paste0("factor(", fe, ")", collapse = " + ")))
  g <- glm(f, data = d, family = family, control = glm.control(epsilon = 1e-16, maxit = 100))
  stopifnot(g$converged)
  g0 <- glm(as.formula(paste(y, "~ 1")), data = d, family = family, control = glm.control(epsilon = 1e-16, maxit = 100))
  s <- summary(g)$coefficients
  v[[paste0(pre, "_coef")]] <- named(coef(g)[c("x", "x2")])
  v[[paste0(pre, "_se")]] <- named(s[c("x", "x2"), "Std. Error"])
  v[[paste0(pre, "_loglik")]] <- as.numeric(logLik(g))
  v[[paste0(pre, "_loglik_null")]] <- as.numeric(logLik(g0))
  v[[paste0(pre, "_aic")]] <- AIC(g); v[[paste0(pre, "_bic")]] <- BIC(g)
  v[[paste0(pre, "_deviance")]] <- deviance(g); v[[paste0(pre, "_null_deviance")]] <- deviance(g0)
  v[[paste0(pre, "_df_resid")]] <- df.residual(g); v[[paste0(pre, "_nobs")]] <- nobs(g)
  v[[paste0(pre, "_df_absorbed")]] <- g$rank - 2L
  v
}
# the rows alpaca keeps (its own drop rule, applied until nothing changes)
kept <- function(d, y, fe, family) {
  a <- alpaca::feglm(as.formula(paste(y, "~ x + x2 |", paste(fe, collapse = " + "))), d, family = family,
                     control = alpaca::feglmControl(dev.tol = 1e-12, center.tol = 1e-12, iter.max = 100))
  keep <- rep(TRUE, nrow(d))
  repeat {
    changed <- FALSE
    for (f in fe) {
      m <- ave(d[[y]][keep], d[[f]][keep], FUN = mean)
      bad <- if (family$family == "binomial") m == 0 | m == 1 else m == 0
      if (any(bad)) { keep[which(keep)[bad]] <- FALSE; changed <- TRUE }
    }
    if (!changed) break
  }
  stopifnot(sum(keep) == a$nobs[["nobs"]])
  list(rows = d[keep, ], alpaca = a)
}

case("feglm_mle", "econometrics/feglm", "FE-GLM MLE: logit, probit, Poisson (id + time, id), Gaussian (id + time) on feglm_panel; glm with factor dummies on alpaca's rows", function(ctx) {
  d <- ctx$dataset("datasets/generated/feglm_panel.csv")
  v <- list()
  for (fam in list(list("logit", binomial("logit"), "y_bin"), list("probit", binomial("probit"), "y_bin"),
                   list("poisson", poisson(), "y_cnt"))) {
    for (fe in list(c("id", "time"), "id")) {
      pre <- paste0(fam[[1]], if (length(fe) == 2L) "2" else "1")
      k <- kept(d, fam[[3]], fe, fam[[2]])
      v <- glm_ref(v, pre, k$rows, fam[[3]], fe, fam[[2]])
      v[[paste0(pre, "_n_positive")]] <- sum(k$rows[[fam[[3]]]] > 0)
      v[[paste0(pre, "_alpaca_coef")]] <- named(coef(k$alpaca))
      v[[paste0(pre, "_alpaca_se")]] <- named(setNames(sqrt(diag(vcov(k$alpaca))), names(coef(k$alpaca))))
    }
  }
  v <- glm_ref(v, "gauss2", d, "y_gauss", c("id", "time"), gaussian())
  l <- lm(y_gauss ~ x + x2 + factor(id) + factor(time), d)
  v$gauss2_dispersion <- summary(l)$sigma^2
  v$n_obs <- nrow(d)
  list(values = v, reference = "stats::glm with factor dummies on the rows alpaca::feglm keeps (epsilon 1e-16); alpaca::feglm",
       call = "glm(y ~ x + x2 + factor(id) [+ factor(time)], family, feglm_panel[alpaca rows]); alpaca::feglm(y ~ x + x2 | id [+ time])",
       packages = c("stats", "alpaca"))
})

case("feglm_separation", "econometrics/feglm", "FE-GLM drop rule: all-0/all-1 entities, a period left all 0 after them, an all-0 count entity, a singleton with a positive count", function(ctx) {
  set.seed(20261113)
  N <- 30L; TT <- 6L
  d <- data.frame(id = rep(seq_len(N), each = TT), time = rep(seq_len(TT), N))
  a <- rnorm(N, 0, 0.5); b <- rnorm(TT, 0, 0.3)
  d$x <- 0.5 * a[d$id] + rnorm(N * TT); d$x2 <- rnorm(N * TT)
  d$y_bin <- rbinom(N * TT, 1, plogis(0.8 * d$x - 0.4 * d$x2 + a[d$id] + b[d$time]))
  d$y_cnt <- rpois(N * TT, exp(0.3 * d$x - 0.2 * d$x2 + 0.5 * a[d$id] + b[d$time]))
  d$y_bin[d$id == 1] <- 1; d$y_bin[d$id == 2] <- 0
  d$y_bin[d$time == TT & d$id > 1] <- 0            # period TT: only entity 1 has y = 1
  d$y_cnt[d$id == 3] <- 0
  d <- rbind(d, data.frame(id = N + 1L, time = 1L, x = 0.25, x2 = -0.5, y_bin = 1, y_cnt = 3))
  d$y_gauss <- 0.6 * d$x - 0.3 * d$x2 + c(a, 0)[d$id] + b[d$time] + rnorm(nrow(d))
  d <- ctx$gen_dataset("feglm_sep", d)
  v <- list()
  for (fam in list(list("logit", binomial("logit"), "y_bin"), list("poisson", poisson(), "y_cnt"))) {
    k <- kept(d, fam[[3]], c("id", "time"), fam[[2]])
    v <- glm_ref(v, fam[[1]], k$rows, fam[[3]], c("id", "time"), fam[[2]])
    v[[paste0(fam[[1]], "_n_dropped")]] <- nrow(d) - nrow(k$rows)
    v[[paste0(fam[[1]], "_n_positive")]] <- sum(k$rows[[fam[[3]]]] > 0)
    fx <- suppressMessages(fixest::feglm(as.formula(paste(fam[[3]], "~ x + x2 | id + time")), d, family = fam[[2]]))
    v[[paste0(fam[[1]], "_nobs_fixest")]] <- nobs(fx)
  }
  # logit: entities 1 and 2, period 6 of the other 28, then 4 entities left all 0 or all 1 by that, and
  # the singleton (6 + 6 + 28 + 20 + 1); Poisson: entity 3 only (alpaca keeps the singleton, fixest drops it)
  stopifnot(v$logit_n_dropped == 61L, v$logit_nobs_fixest == v$logit_nobs,
            v$poisson_n_dropped == 6L, v$poisson_nobs_fixest == v$poisson_nobs - 1L)

  v <- glm_ref(v, "gauss", d, "y_gauss", c("id", "time"), gaussian())
  v$n_obs <- nrow(d)
  list(values = v, reference = "stats::glm with factor dummies on the rows alpaca::feglm keeps (epsilon 1e-16); fixest::feglm nobs (drops the singleton too)",
       call = "glm(y ~ x + x2 + factor(id) + factor(time), family, feglm_sep[alpaca rows])", packages = c("stats", "alpaca", "fixest"))
})
