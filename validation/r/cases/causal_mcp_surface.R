# validation/r/cases/causal_mcp_surface.R -- references for the MCP causal surface (TOOL-4, TOOL-15).
#  * mte_mcp_covariate: the MCP iv_mte tool takes one instrument z and covariates x. With x = z2 the
#    covariate enters the propensity model and the second stage (TOOL-4): P = probit(d ~ z1 + z2),
#    y ~ [1, P, P^2, P^3, z2] by OLS, MTE(u) = d/du of the cubic, Heckman & Vytlacil (2005) ATE / ATT /
#    ATU weights and LATE over the propensity support on a 1001-point midpoint grid, delta-method SEs
#    with the full HC1 covariance of the OLS fit (sandwich::vcovHC; the econ_mte.R conventions of
#    T3-WP16, which commits mte_sim.csv). The MCP tool runs the first stage at IVMTEConfig's default
#    tolerance 1e-8; the pins are the `*_tight` fits at the probit solution (glm at epsilon 1e-14,
#    restarted from its own estimate until the coefficients stop changing: a single epsilon-1e-14 fit
#    stops 3e-10 short); the default-control values are recorded only. `nox_*` is the same tool call
#    with x ignored (what the handler returned before TOOL-4): P = probit(d ~ z1), y ~ [1, P, P^2, P^3].
.mcp_mte <- function(y, P, Xcov, ngrid = 1001, deg = 3) {
  Xp <- sapply(seq_len(deg), function(j) P^j); colnames(Xp) <- c("P", "I(P^2)", "I(P^3)")[seq_len(deg)]
  X <- if (is.null(Xcov)) Xp else cbind(Xp, Xcov)
  ols <- lm(y ~ X); b <- coef(ols); V <- sandwich::vcovHC(ols, type = "HC1"); names(b) <- c("(Intercept)", colnames(X))
  grid <- (seq_len(ngrid) - 0.5) / ngrid
  k <- length(b); g0 <- matrix(0, ngrid, k)
  for (j in seq_len(deg)) g0[, 1 + j] <- j * grid^(j - 1)
  inl <- grid >= min(P) & grid <= max(P)
  w <- list(ate = rep(1, ngrid), att = sapply(grid, function(u) mean(P > u)) / mean(P),
            atu = sapply(grid, function(u) mean(P <= u)) / mean(1 - P), late_support = as.numeric(inl))
  w <- lapply(w, function(x) x / sum(x))
  te <- function(wt) { g <- colSums(g0 * wt); c(est = sum(g * b), se = sqrt(drop(t(g) %*% V %*% g))) }
  list(b = b, te = lapply(w, te), p_min = min(P), p_max = max(P))
}

case("mte_mcp_covariate", "econometrics/mte", "MTE by local IV through the MCP tool: instrument z1, covariate z2 in the probit propensity and the second stage; and the same call with the covariate ignored", function(ctx) {
  d <- ctx$dataset("datasets/generated/mte_sim.csv")
  # The MCP tool runs the first stage at IVMTEConfig's default tolerance, glm.control()'s epsilon = 1e-8
  # (T3-WP16); the *_tight values (the probit solution) are the pins, the default-control values are recorded.
  fit <- function(form, cov, ctl, polish = FALSE) {
    fs <- glm(form, data = d, family = binomial("probit"), control = ctl)
    if (polish) for (i in 1:50) {                           # restart until the coefficients stop changing
      g <- glm(form, data = d, family = binomial("probit"), start = coef(fs), control = ctl)
      done <- max(abs(coef(g) - coef(fs))) < 1e-16
      fs <- g
      if (done) break
    }
    list(fs = fs, mte = .mcp_mte(d$y, fitted(fs), cov))
  }
  f <- fit(d ~ z1 + z2, cbind(z2 = d$z2), glm.control())
  ft <- fit(d ~ z1 + z2, cbind(z2 = d$z2), glm.control(epsilon = 1e-14), polish = TRUE)
  f0 <- fit(d ~ z1, NULL, glm.control())
  f0t <- fit(d ~ z1, NULL, glm.control(epsilon = 1e-14), polish = TRUE)
  v <- list(n = nrow(d), n_grid = 1001, fs_iter = f$fs$iter, fs_coef = named(coef(f$fs)), mte_coef = named(f$mte$b),
            p_min = f$mte$p_min, p_max = f$mte$p_max)
  for (e in names(f$mte$te)) v[[e]] <- named(f$mte$te[[e]])
  v$fs_coef_tight <- named(coef(ft$fs))
  v$mte_coef_tight <- named(ft$mte$b)
  for (e in names(ft$mte$te)) v[[paste0(e, "_tight")]] <- named(ft$mte$te[[e]])
  v$nox_fs_iter <- f0$fs$iter; v$nox_fs_coef <- named(coef(f0$fs)); v$nox_mte_coef <- named(f0$mte$b)
  for (e in names(f0$mte$te)) v[[paste0("nox_", e)]] <- named(f0$mte$te[[e]])
  v$nox_fs_coef_tight <- named(coef(f0t$fs)); v$nox_mte_coef_tight <- named(f0t$mte$b)
  for (e in names(f0t$mte$te)) v[[paste0("nox_", e, "_tight")]] <- named(f0t$mte$te[[e]])
  list(values = v, reference = "local IV MTE (Brinch, Mogstad & Wiswall 2017), probit propensity with the covariate, cubic in P plus the covariate by OLS; Heckman & Vytlacil (2005) weights; delta method with the full HC1 covariance (sandwich::vcovHC(type = 'HC1')); written out. Pins: the probit solution (*_tight: epsilon 1e-14, restarted until the coefficients stop changing); glm.control() defaults recorded",
       call = "glm(d ~ z1 + z2, binomial('probit')); lm(y ~ P + P^2 + P^3 + z2); sandwich::vcovHC(type = 'HC1')", packages = c("sandwich"))
})
