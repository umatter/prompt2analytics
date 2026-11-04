# validation/r/cases/econ_panel_t3.R -- T3-WP17 panel references (ECON-69, ECON-72, ECON-36; X-16 part).
#  * hdfe_map_scale: two-way FE on an outcome of order 1 and the same outcome times 1e9 (fixest feols,
#    feglm(gaussian)); the MAP stopping rule must not depend on the scale (ECON-69).
#  * panel_absorbed_regressors: a time-invariant regressor z, a regressor w = a_i + b_t absorbed only by
#    the two-way effects and x3 = x1 + 2 x2: plm and fixest drop them (ECON-72).
#  * hdfe_disconnected: two-way FE whose firm-year graph has two connected components; lm with dummies
#    gives the exact residual df (X-16, the absorbed-df fix of 5f86b49).
#  * pggls_sim, pvcm_pmg_sim, feglm_families: plm::pggls, plm::pvcm/pmg, fixest::feglm and glm with dummies
#    (ECON-36 test pins; quantities p2a does not reproduce are registry deviations ECON-20/33/34/35/66/67).
suppressPackageStartupMessages({ library(plm); library(fixest) })

case("hdfe_map_scale", "econometrics/hdfe", "Two-way FE (firm + year, unbalanced) on y and y * 1e9: fixest feols / feglm(gaussian)", function(ctx) {
  set.seed(20261103)
  n <- 1000L
  firm <- sample.int(100L, n, replace = TRUE); year <- sample.int(20L, n, replace = TRUE)
  stopifnot(length(unique(firm)) == 100L, length(unique(year)) == 20L)
  af <- rnorm(100L); ay <- rnorm(20L)
  x <- 0.5 * af[firm] + 0.3 * ay[year] + rnorm(n)
  y <- 0.5 * x + af[firm] + ay[year] + rnorm(n)
  d <- ctx$gen_dataset("hdfe_scale", data.frame(firm = firm, year = year, x = x, y = y, y_big = y * 1e9))
  h <- feols(y ~ x | firm + year, d, vcov = "iid")
  hb <- feols(y_big ~ x | firm + year, d, vcov = "iid")
  gb <- feglm(y_big ~ x | firm + year, d, family = gaussian(), vcov = "iid")
  l <- lm(y_big ~ x + factor(firm) + factor(year), d)
  list(values = list(
    coef = named(coef(h)), se_iid = named(se(h)),
    coef_big = named(coef(hb)), se_iid_big = named(se(hb)),
    coef_feglm_big = named(coef(gb)), se_feglm_big_lm = named(c(x = summary(l)$coefficients["x", "Std. Error"])),
    deviance_big = deviance(l), dispersion_big = summary(l)$sigma^2,
    r2_within = unname(r2(h, "wr2")), df_resid = df.residual(l), n_obs = nobs(h)
  ), reference = "fixest::feols(y ~ x | firm + year, vcov = 'iid'); fixest::feglm(family = gaussian); lm with factor dummies for the classical SE",
  call = "feols(y_big ~ x | firm + year, hdfe_scale, vcov = 'iid')", packages = "fixest")
})

case("panel_absorbed_regressors", "econometrics/panel", "Regressors without within variation (time-invariant z, two-way-absorbed w) and a collinear x3: plm and fixest drop them", function(ctx) {
  set.seed(20261104)
  N <- 30L; TT <- 6L
  d <- data.frame(id = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
  a <- rnorm(N); b <- rnorm(TT)
  d$x1 <- 0.5 * a[d$id] + rnorm(N * TT); d$x2 <- rnorm(N * TT)
  zf <- rnorm(N); d$z <- zf[d$id]
  d$w <- a[d$id] + b[d$t]
  d$x3 <- d$x1 + 2 * d$x2
  d$y <- 1 + 0.5 * d$x1 - 0.3 * d$x2 + 0.8 * d$z + 0.4 * d$w + a[d$id] + b[d$t] + rnorm(N * TT)
  d <- ctx$gen_dataset("panel_absorbed", d)
  pd <- pdata.frame(d, index = c("id", "t"))
  fe <- plm(y ~ x1 + x2 + z, pd, model = "within")
  fe0 <- plm(y ~ x1 + x2, pd, model = "within")
  stopifnot(identical(names(coef(fe)), c("x1", "x2")), isTRUE(all.equal(coef(fe), coef(fe0), tolerance = 1e-13)))
  fe3 <- plm(y ~ x1 + x2 + x3, pd, model = "within")
  stopifnot(isTRUE(fe3$aliased[["x3"]]))
  re <- plm(y ~ x1 + x2 + z, pd, model = "random"); ec <- ercomp(re)
  re3 <- plm(y ~ x1 + x2 + x3, pd, model = "random")
  stopifnot(!("x3" %in% names(coef(re3))))
  h2 <- suppressMessages(feols(y ~ x1 + x2 + z + w | id + t, d, vcov = "iid"))
  stopifnot(identical(h2$collin.var, c("z", "w")))
  h1 <- suppressMessages(feols(y ~ x1 + x2 + z + w | id, d, vcov = "iid"))
  stopifnot(identical(h1$collin.var, "z"))
  h3 <- suppressMessages(feols(y ~ x1 + x2 + x3 | id + t, d, vcov = "iid"))
  g2 <- suppressMessages(feglm(y ~ x1 + x2 + z + w | id + t, d, family = gaussian(), vcov = "iid"))
  hz <- phtest(fe, re)
  list(values = list(
    hausman_chisq = unname(hz$statistic), hausman_df = unname(hz$parameter), hausman_p = unname(hz$p.value),
    fe_coef = named(coef(fe)), fe_se = named(sqrt(diag(vcov(fe)))), fe_df_resid = df.residual(fe),
    fe_x3_coef = named(coef(fe3)),
    re_coef = named(coef(re)), re_se = named(sqrt(diag(vcov(re)))),
    re_sigma2_idios = unname(ec$sigma2["idios"]), re_sigma2_id = unname(ec$sigma2["id"]), re_theta = unname(ec$theta[1]),
    re_x3_coef = named(coef(re3)),
    hdfe2_coef = named(coef(h2)), hdfe2_se_iid = named(se(h2)), hdfe2_df_resid = unname(degrees_freedom(h2, "resid")),
    hdfe1_coef = named(coef(h1)), hdfe1_se_iid = named(se(h1)),
    hdfe2_x3_coef = named(coef(h3)),
    feglm2_coef = named(coef(g2))
  ), reference = "plm::plm(model = 'within' | 'random') (time-invariant regressors removed, aliased dropped); fixest::feols / feglm $collin.var",
  call = "plm(y ~ x1 + x2 + z, model = 'within'); feols(y ~ x1 + x2 + z + w | id + t)", packages = c("plm", "fixest"))
})

case("hdfe_disconnected", "econometrics/hdfe", "Two-way FE with two disconnected firm-year components: exact residual df from lm with dummies", function(ctx) {
  set.seed(20261105)
  d <- rbind(expand.grid(year = 1:5, firm = 1:8), expand.grid(year = 6:10, firm = 9:16))[, c("firm", "year")]
  n <- nrow(d); af <- rnorm(16); ay <- rnorm(10)
  d$x <- 0.4 * af[d$firm] + rnorm(n); d$z <- rnorm(n)
  d$y <- 0.7 * d$x - 0.2 * d$z + af[d$firm] + ay[d$year] + rnorm(n, 0, 0.8)
  d <- ctx$gen_dataset("hdfe_disconnected", d)
  l <- lm(y ~ x + z + factor(firm) + factor(year), d)
  stopifnot(l$rank == 2L + 1L + 15L + 9L - 1L)   # one redundant year dummy per extra component
  s <- summary(l)$coefficients
  naive_df <- n - 2L - (16L + 10L - 1L)
  list(values = list(
    coef = named(coef(l)[c("x", "z")]), se = named(s[c("x", "z"), "Std. Error"]), df_resid = df.residual(l),
    df_absorbed = 16L + 10L - 2L, se_naive_df = named(s[c("x", "z"), "Std. Error"] * sqrt(df.residual(l) / naive_df)),
    n_obs = n
  ), reference = "stats::lm(y ~ x + z + factor(firm) + factor(year)) (rank-based df: n0 + n1 - 2 components)",
  call = "lm(y ~ x + z + factor(firm) + factor(year), hdfe_disconnected)", packages = "stats")
})

case("pggls_sim", "econometrics/panel_gls", "plm::pggls pooling / within / fd on a simulated N = 40, T = 5 panel (AR(1), heteroskedastic errors)", function(ctx) {
  set.seed(20261101)
  N <- 40L; TT <- 5L
  d <- data.frame(id = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
  d$x1 <- rnorm(N * TT); d$x2 <- rnorm(N * TT) + 0.5 * d$x1
  a <- rnorm(N); e <- numeric(N * TT)
  for (i in seq_len(N)) { s <- exp(rnorm(1, 0, 0.4)); u <- numeric(TT); u[1] <- rnorm(1, 0, s)
    for (k in 2:TT) u[k] <- 0.5 * u[k - 1] + rnorm(1, 0, s); e[(i - 1) * TT + seq_len(TT)] <- u }
  d$y <- 1 + 0.8 * d$x1 - 0.4 * d$x2 + a[d$id] + e
  d <- ctx$gen_dataset("pggls_panel", d)
  pd <- pdata.frame(d, index = c("id", "t"))
  v <- list()
  for (m in c("pooling", "within", "fd")) {
    g <- pggls(y ~ x1 + x2, data = pd, model = m)
    v[[paste0(m, "_coef")]] <- named(coef(g)); v[[paste0(m, "_se")]] <- named(sqrt(diag(vcov(g))))
    v[[paste0(m, "_df_resid")]] <- g$df.residual; v[[paste0(m, "_rss")]] <- sum(resid(g)^2)
    v[[paste0(m, "_r2")]] <- summary(g)$rsqr
  }
  v$n_obs <- nrow(d)
  list(values = v, reference = "plm::pggls(model = 'pooling' | 'within' | 'fd') (unrestricted T x T FGLS)",
       call = "pggls(y ~ x1 + x2, data = pdata.frame(d, index = c('id', 't')), model)", packages = "plm")
})

case("pvcm_pmg_sim", "econometrics/pvcm", "plm::pvcm (within, random) and plm::pmg (mg) with heterogeneous slopes, N = 12, T = 20 (positive-definite Swamy Delta)", function(ctx) {
  set.seed(20261102)
  N <- 12L; TT <- 20L
  d <- data.frame(id = rep(seq_len(N), each = TT), t = rep(seq_len(TT), N))
  d$x1 <- rnorm(N * TT, 2, 1); d$x2 <- rnorm(N * TT)
  b0 <- rnorm(N, 1, 1.5); b1 <- rnorm(N, 0.8, 0.5); b2 <- rnorm(N, -0.5, 0.4)
  d$y <- b0[d$id] + b1[d$id] * d$x1 + b2[d$id] * d$x2 + rnorm(N * TT, 0, 0.5)
  d <- ctx$gen_dataset("pvcm_panel", d)
  pd <- pdata.frame(d, index = c("id", "t"))
  w <- pvcm(y ~ x1 + x2, data = pd, model = "within")
  r <- pvcm(y ~ x1 + x2, data = pd, model = "random")
  stopifnot(all(eigen(r$Delta, only.values = TRUE)$values > 0))
  m <- pmg(y ~ x1 + x2, data = pd, model = "mg")
  cw <- as.matrix(coef(w)); sw <- as.matrix(w$std.error)
  v <- list()
  for (i in seq_len(N)) { v[[sprintf("within_coef_%d", i)]] <- named(cw[i, ]); v[[sprintf("within_se_%d", i)]] <- named(sw[i, ]) }
  v$random_coef <- named(coef(r)); v$random_se <- named(setNames(sqrt(diag(vcov(r))), names(coef(r))))
  v$random_delta_diag <- named(diag(r$Delta)); v$random_delta <- vec(r$Delta)
  v$random_chisq <- unname(summary(r)$chisq.test$statistic); v$random_chisq_df <- unname(summary(r)$chisq.test$parameter)
  v$mg_coef <- named(coef(m)); v$mg_se <- named(sqrt(diag(vcov(m))))
  v$n_obs <- nrow(d)
  list(values = v, reference = "plm::pvcm(model = 'within' | 'random') (Swamy 1970); plm::pmg(model = 'mg')",
       call = "pvcm(y ~ x1 + x2, data = pdata.frame(d, index = c('id', 't')), model); pmg(..., model = 'mg')", packages = "plm")
})

case("feglm_families", "econometrics/feglm", "fixest::feglm gaussian / logit / probit / poisson with id + time effects; glm with dummies", function(ctx) {
  set.seed(20261106)
  N <- 60L; TT <- 8L
  d <- data.frame(id = rep(seq_len(N), each = TT), time = rep(seq_len(TT), N))
  a <- rnorm(N, 0, 0.5); b <- rnorm(TT, 0, 0.3)
  d$x <- 0.5 * a[d$id] + rnorm(N * TT); d$x2 <- rnorm(N * TT)
  d$y_gauss <- 0.6 * d$x - 0.3 * d$x2 + a[d$id] + b[d$time] + rnorm(N * TT)
  d$y_bin <- rbinom(N * TT, 1, plogis(0.8 * d$x - 0.4 * d$x2 + a[d$id] + b[d$time]))
  d$y_cnt <- rpois(N * TT, exp(0.3 * d$x - 0.2 * d$x2 + 0.5 * a[d$id] + b[d$time]))
  d <- ctx$gen_dataset("feglm_panel", d)
  v <- list()
  lg <- lm(y_gauss ~ x + x2 + factor(id) + factor(time), d); sg <- summary(lg)
  gg <- feglm(y_gauss ~ x + x2 | id + time, d, family = gaussian(), vcov = "iid")
  v$gauss_coef <- named(coef(gg)); v$gauss_se_lm <- named(sg$coefficients[c("x", "x2"), "Std. Error"])
  v$gauss_deviance <- deviance(lg); v$gauss_dispersion <- sg$sigma^2; v$gauss_df_resid <- df.residual(lg)
  v$gauss_loglik_glm <- as.numeric(logLik(glm(y_gauss ~ x + x2 + factor(id) + factor(time), data = d)))
  for (fam in list(list("logit", binomial("logit"), "y_bin"), list("probit", binomial("probit"), "y_bin"), list("poisson", poisson(), "y_cnt"))) {
    f <- suppressMessages(feglm(as.formula(paste(fam[[3]], "~ x + x2 | id + time")), d, family = fam[[2]], vcov = "iid"))
    gl <- glm(as.formula(paste(fam[[3]], "~ x + x2 + factor(id) + factor(time)")), data = d, family = fam[[2]])
    v[[paste0(fam[[1]], "_coef")]] <- named(coef(f)); v[[paste0(fam[[1]], "_se")]] <- named(se(f))
    v[[paste0(fam[[1]], "_nobs")]] <- nobs(f)
    v[[paste0(fam[[1]], "_glm_coef")]] <- named(coef(gl)[c("x", "x2")])
    v[[paste0(fam[[1]], "_glm_se")]] <- named(summary(gl)$coefficients[c("x", "x2"), "Std. Error"])
  }
  v$n_obs <- nrow(d)
  list(values = v, reference = "fixest::feglm(| id + time, vcov = 'iid'); stats::glm with factor dummies",
       call = "feglm(y ~ x + x2 | id + time, feglm_panel, family)", packages = c("fixest", "stats"))
})
