# validation/r/cases/econ_mte.R -- MTE by local IV (T3-WP16: ECON-70; moved in ECON-26, ECON-27,
# ECON-87). No R package (ivmte computes bounds by LP; localIV is not installed): P(Z) by probit of
# D on Z; Y on [1, P, P^2, P^3] (Brinch, Mogstad & Wiswall 2017); MTE(u) = d E[Y | P = u] / du;
# ATE = int_0^1 MTE; ATT/ATU weights w_ATT(u) = Pr(P > u) / E[P], w_ATU(u) = Pr(P <= u) / E[1 - P]
# (Heckman & Vytlacil 2005, Table IB) on a 1001-point midpoint grid; delta-method SEs with the
# full HC1 coefficient covariance (sandwich::vcovHC; Var(Y | P) varies with P by construction). PRTE for the policy P' = min(P + 0.1, 1): w(u) = Pr(P' > u) - Pr(P > u)
# (Heckman & Vytlacil 2005, Table IB, normalised). Extrapolation shares: weight on grid points
# outside [min P, max P]. A narrow-support subsample (|z1| < 0.5) and a bootstrap on a committed
# resample-index matrix (units resampled, both stages refitted).
.mte_fit <- function(dd, ngrid = 1001, deg = 3) {
  fs <- glm(d ~ z1 + z2, data = dd, family = binomial("probit"), control = glm.control(epsilon = 1e-14))
  P <- fitted(fs)
  Xp <- sapply(seq_len(deg), function(j) P^j); colnames(Xp) <- c("P", "I(P^2)", "I(P^3)")[seq_len(deg)]
  ols <- lm(dd$y ~ Xp); b <- coef(ols); names(b) <- c("(Intercept)", colnames(Xp)); V <- sandwich::vcovHC(ols, type = "HC1")
  grid <- (seq_len(ngrid) - 0.5) / ngrid
  gmte <- cbind(0, sapply(seq_len(deg), function(j) j * grid^(j - 1)))
  inl <- grid >= min(P) & grid <= max(P)
  w <- list(ate = rep(1, ngrid), att = sapply(grid, function(u) mean(P > u)) / mean(P),
            atu = sapply(grid, function(u) mean(P <= u)) / mean(1 - P), late_support = as.numeric(inl))
  w <- lapply(w, function(x) x / sum(x))
  te <- function(wt) { g <- colSums(gmte * wt); c(est = sum(g * b), se = sqrt(drop(t(g) %*% V %*% g))) }
  list(fs = fs, P = P, b = b, V = V, grid = grid, gmte = gmte, inl = inl, w = w, te = te)
}

case("mte_local_iv", "econometrics/mte", "MTE by local IV, cubic polynomial in P(Z) (probit first stage); ATE, ATT, ATU, LATE(p_min, p_max), PRTE with delta-method SEs; extrapolation shares", function(ctx) {
  set.seed(20261205)
  n <- 2000
  z1 <- rnorm(n); z2 <- rbinom(n, 1, 0.5); ud <- runif(n)
  pz <- pnorm(-0.2 + 0.9 * z1 + 0.6 * z2); dd <- as.integer(ud < pz)
  y1 <- 2 + 1.5 * (1 - ud) + rnorm(n); y0 <- 1 + 0.5 * ud + rnorm(n)
  y <- ifelse(dd == 1, y1, y0)
  d <- ctx$gen_dataset("mte_sim", data.frame(y = round(y, 10), d = dd, z1 = round(z1, 10), z2 = z2))
  f <- .mte_fit(d)
  P <- f$P; inl <- f$inl
  u3 <- c(0.1, 0.5, 0.9); g3 <- cbind(0, 1, 2 * u3, 3 * u3^2)
  wprte <- sapply(f$grid, function(u) mean(pmin(P + 0.1, 1) > u) - mean(P > u)); wprte <- wprte / sum(wprte)
  v <- list(n = nrow(d), n_grid = length(f$grid), fs_coef = named(coef(f$fs)), mte_coef = named(f$b),
            p_min = min(P), p_max = max(P), n_grid_in_support = sum(inl),
            ate = named(f$te(f$w$ate)), att = named(f$te(f$w$att)), atu = named(f$te(f$w$atu)),
            late_support = named(f$te(f$w$late_support)), prte_shift = named(f$te(wprte)),
            mte_at = vec(drop(g3 %*% f$b)), mte_se_at = vec(sqrt(diag(g3 %*% f$V %*% t(g3)))),
            share_ate_weight_outside_support = sum(f$w$ate[!inl]),
            share_att_weight_outside_support = sum(f$w$att[!inl]),
            share_atu_weight_outside_support = sum(f$w$atu[!inl]))
  # Narrow propensity support: the subsample |z1| < 0.5.
  fn <- .mte_fit(d[abs(d$z1) < 0.5, ])
  v$narrow_n <- length(fn$P); v$narrow_p_min <- min(fn$P); v$narrow_p_max <- max(fn$P)
  v$narrow_n_grid_in_support <- sum(fn$inl)
  for (e in c("ate", "att", "atu", "late_support")) {
    v[[paste0("narrow_", e)]] <- named(fn$te(fn$w[[e]]))
    if (e != "late_support") v[[paste0("narrow_share_", e, "_weight_outside_support")]] <- sum(fn$w[[e]][!fn$inl])
  }
  list(values = v, reference = "local IV MTE with a cubic polynomial in the probit propensity (Brinch, Mogstad & Wiswall 2017); Heckman & Vytlacil (2005) weights on a 1001-point grid; delta method with the full HC1 covariance (sandwich::vcovHC(type = 'HC1'))",
       call = "glm(d ~ z1 + z2, binomial('probit')); lm(y ~ P + P^2 + P^3); sandwich::vcovHC(type = 'HC1')", packages = c("sandwich"))
})

case("mte_bootstrap", "econometrics/mte", "MTE bootstrap: units resampled with a committed 400 x 25 index matrix, probit and cubic polynomial refitted per replicate; SD and type-7 percentile intervals", function(ctx) {
  d <- ctx$dataset("datasets/generated/mte_sim.csv")[1:400, ]
  n <- nrow(d); R <- 25
  set.seed(20261207)
  U <- matrix(sample(n, n * R, replace = TRUE), n, R)
  colnames(U) <- sprintf("b%02d", seq_len(R))
  U <- as.matrix(ctx$gen_dataset("mte_boot_indices", as.data.frame(U)))   # 1-based row indices, one column per replicate
  est <- function(dd) { f <- .mte_fit(dd); c(sapply(f$w, function(w) f$te(w)["est"]), mte_mid = sum(f$gmte[501, ] * f$b)) }
  full <- est(d)
  reps <- t(sapply(seq_len(R), function(b) est(d[U[, b], ])))
  stopifnot(all(is.finite(reps)))
  nm <- c("ate", "att", "atu", "late_support", "mte_mid")
  colnames(reps) <- nm; names(full) <- nm
  ci <- apply(reps, 2, quantile, probs = c(0.025, 0.975), type = 7)
  list(values = list(n = n, replications = R, estimate = named(full), boot_se = named(apply(reps, 2, sd)),
                     boot_ci_lower = named(ci[1, ]), boot_ci_upper = named(ci[2, ]), boot_mean = named(colMeans(reps))),
       dataset = "validation/datasets/generated/mte_boot_indices.csv",
       reference = "nonparametric bootstrap of the local IV MTE: rows resampled with the committed index matrix, glm probit and lm refitted; sd() and quantile(type = 7)",
       call = "for (b in 1:25) est(d[U[, b], ]) with est = probit + lm(y ~ P + P^2 + P^3) + HV weights", packages = character(0))
})
