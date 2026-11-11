# validation/r/cases/econ_panel_t4.R -- T4-WP11 panel references.
#  * pggls_unbalanced: plm::pggls pooling / within / fd on an unbalanced panel (first, last and interior
#    periods missing), so each entity's weight is the inverse of its own observed-period block of Omega.
#  * pvcm_swamy_grunfeld: plm::pvcm(model = 'random') on Grunfeld, whose Delta = D1 - D2 has a negative
#    eigenvalue (plm then uses D1), and plm::pmg(model = 'mg') standard errors.
#  * hdfe_hc_leverage: sandwich::vcovHC HC0-HC3 on lm with factor dummies (the leverage includes the
#    fixed effects): Grunfeld one- and two-way, a short unbalanced simulated panel one- and two-way, and
#    the two-component design of hdfe_disconnected.
suppressPackageStartupMessages({ library(plm); library(sandwich) })

case("pggls_unbalanced", "econometrics/panel_gls", "plm::pggls pooling / within / fd on an unbalanced N = 40, T = 6 panel (AR(1), heteroskedastic errors; first, last and interior periods missing)", function(ctx) {
  set.seed(20261111)
  N <- 40L; TT <- 6L
  d <- data.frame(id = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
  d$x1 <- rnorm(N * TT); d$x2 <- rnorm(N * TT) + 0.5 * d$x1
  a <- rnorm(N); e <- numeric(N * TT)
  for (i in seq_len(N)) { s <- exp(rnorm(1, 0, 0.4)); u <- numeric(TT); u[1] <- rnorm(1, 0, s)
    for (k in 2:TT) u[k] <- 0.5 * u[k - 1] + rnorm(1, 0, s); e[(i - 1) * TT + seq_len(TT)] <- u }
  d$y <- 1 + 0.8 * d$x1 - 0.4 * d$x2 + a[d$id] + e
  drop <- (d$id %in% 1:8 & d$t == 1) | (d$id %in% 9:16 & d$t == 6) | (d$id %in% 17:22 & d$t == 3) |
          (d$id %in% 23:25 & d$t %in% c(1, 2))
  d <- ctx$gen_dataset("pggls_unbalanced", d[!drop, ])
  pd <- pdata.frame(d, index = c("id", "t"))
  stopifnot(!pdim(pd)$balanced)
  v <- list()
  for (m in c("pooling", "within", "fd")) {
    g <- pggls(y ~ x1 + x2, data = pd, model = m); s <- summary(g)$CoefTable
    v[[paste0(m, "_coef")]] <- named(coef(g)); v[[paste0(m, "_se")]] <- named(sqrt(diag(vcov(g))))
    v[[paste0(m, "_z")]] <- named(setNames(s[, 3], rownames(s))); v[[paste0(m, "_p")]] <- named(setNames(s[, 4], rownames(s)))
    v[[paste0(m, "_df_resid")]] <- g$df.residual; v[[paste0(m, "_rss")]] <- sum(resid(g)^2)
    v[[paste0(m, "_r2")]] <- summary(g)$rsqr; v[[paste0(m, "_nobs")]] <- length(resid(g))
    v[[paste0(m, "_omega_min_eigenvalue")]] <- min(eigen(g$sigma, symmetric = TRUE, only.values = TRUE)$values)
  }
  # pooling and fd: every period block of Omega is positive definite (plain inverses); within: the
  # cell-averaged Omega of an unbalanced panel is indefinite, and pggls's generalized Cholesky
  # (bdsmatrix, tolerance 1e-10 x the largest diagonal) zeroes the non-positive pivots
  stopifnot(v$pooling_omega_min_eigenvalue > 0, v$fd_omega_min_eigenvalue > 0, v$within_omega_min_eigenvalue < 0)
  v$n_obs <- nrow(d)
  list(values = v, reference = "plm::pggls(model = 'pooling' | 'within' | 'fd') (unrestricted T x T FGLS, per-entity blocks Omega_ii)",
       call = "pggls(y ~ x1 + x2, data = pdata.frame(d, index = c('id', 't')), model)", packages = "plm")
})

case("pvcm_swamy_grunfeld", "econometrics/pvcm", "plm::pvcm(model = 'random') on Grunfeld (Delta = D1 - D2 not positive semidefinite: plm uses D1) and plm::pmg(model = 'mg')", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  pd <- pdata.frame(g, index = c("firm", "year"))
  r <- pvcm(inv ~ value + capital, data = pd, model = "random"); sr <- summary(r)
  m <- pmg(inv ~ value + capital, data = pd, model = "mg"); sm <- summary(m)$CoefTable
  # plm's Delta is D1 here: rebuild D1 - D2 and check that it has a negative eigenvalue
  w <- pvcm(inv ~ value + capital, data = pd, model = "within")
  cm <- as.matrix(coef(w)); dev <- sweep(cm, 2, colMeans(cm))
  D1 <- crossprod(dev) / (nrow(cm) - 1)
  stopifnot(isTRUE(all.equal(unname(r$Delta), unname(D1), tolerance = 1e-12)))
  D2 <- Reduce("+", lapply(split(g, g$firm), function(s) vcov(lm(inv ~ value + capital, s)))) / nrow(cm)
  ev <- eigen(D1 - D2, only.values = TRUE)$values
  stopifnot(min(ev) < 0)
  rc <- sr$coefficients
  list(values = list(
    random_coef = named(coef(r)), random_se = named(setNames(sqrt(diag(vcov(r))), names(coef(r)))),
    random_z = named(setNames(rc[, 3], rownames(rc))), random_p = named(setNames(rc[, 4], rownames(rc))),
    random_delta = vec(r$Delta), random_delta_diag = named(setNames(diag(r$Delta), names(coef(r)))),
    d1_minus_d2_eigenvalues = vec(ev),
    random_chisq = unname(r$chisq.test$statistic), random_chisq_df = unname(r$chisq.test$parameter),
    random_chisq_p = unname(r$chisq.test$p.value),
    mg_coef = named(coef(m)), mg_se = named(sqrt(diag(vcov(m)))),
    mg_z = named(setNames(sm[, 3], rownames(sm))), mg_p = named(setNames(sm[, 4], rownames(sm))),
    n_obs = nrow(g), n_groups = nrow(cm)
  ), reference = "plm::pvcm(model = 'random') (Swamy 1970; Delta <- D1 when D1 - D2 has a negative eigenvalue); plm::pmg(model = 'mg') (Pesaran and Smith 1995)",
  call = "pvcm(inv ~ value + capital, data = pdata.frame(grunfeld, index = c('firm', 'year')), model = 'random'); pmg(..., model = 'mg')", packages = "plm")
})

case("hdfe_hc_leverage", "econometrics/hdfe", "HC0-HC3 with the fixed effects in the leverage: sandwich::vcovHC on lm with factor dummies (Grunfeld one- and two-way, an unbalanced short panel, two connected components)", function(ctx) {
  hc <- function(fit, vars) {
    out <- list()
    for (ty in c("HC0", "HC1", "HC2", "HC3")) out[[tolower(ty)]] <- named(sqrt(diag(vcovHC(fit, type = ty)))[vars])
    out
  }
  put <- function(v, prefix, fit, vars) {
    v[[paste0(prefix, "_coef")]] <- named(coef(fit)[vars])
    s <- hc(fit, vars); for (ty in names(s)) v[[paste0(prefix, "_se_", ty)]] <- s[[ty]]
    v[[paste0(prefix, "_df_resid")]] <- df.residual(fit); v[[paste0(prefix, "_max_hat")]] <- max(hatvalues(fit))
    v
  }
  v <- list()
  # simulated short unbalanced panel first (generated dataset), Grunfeld and the disconnected design after
  set.seed(20261112)
  N <- 100L; TT <- 4L
  d <- data.frame(id = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
  a <- rnorm(N); b <- rnorm(TT)
  d$x1 <- 0.5 * a[d$id] + rnorm(N * TT); d$x2 <- rnorm(N * TT)
  d$y <- 0.6 * d$x1 - 0.3 * d$x2 + a[d$id] + b[d$t] + rnorm(N * TT) * (1 + abs(d$x1))
  keep <- !(d$id %in% seq(3L, N, by = 7L) & d$t == 4L) & !(d$id %in% seq(5L, N, by = 11L) & d$t == 1L)
  d <- ctx$gen_dataset("hdfe_hc_short", d[keep, ])
  stopifnot(min(table(d$id)) >= 2L, any(table(d$id) == 2L))
  v <- put(v, "short_oneway", lm(y ~ x1 + x2 + factor(id), d), c("x1", "x2"))
  v <- put(v, "short_twoway", lm(y ~ x1 + x2 + factor(id) + factor(t), d), c("x1", "x2"))
  v$short_n_obs <- nrow(d)
  dc <- ctx$dataset("datasets/generated/hdfe_disconnected.csv")
  v <- put(v, "disconnected", lm(y ~ x + z + factor(firm) + factor(year), dc), c("x", "z"))
  g <- ctx$dataset("datasets/grunfeld.csv")
  v <- put(v, "grunfeld_firm", lm(inv ~ value + capital + factor(firm), g), c("value", "capital"))
  v <- put(v, "grunfeld_twoway", lm(inv ~ value + capital + factor(firm) + factor(year), g), c("value", "capital"))
  list(values = v, dataset = "validation/datasets/generated/hdfe_hc_short.csv",
       reference = "sandwich::vcovHC(lm(y ~ x + factor(fe1) [+ factor(fe2)]), type = 'HC0' | 'HC1' | 'HC2' | 'HC3') (hat values of the dummy-variable regression)",
       call = "vcovHC(lm(y ~ x1 + x2 + factor(id) + factor(t), hdfe_hc_short), type)", packages = c("sandwich", "stats"))
})
