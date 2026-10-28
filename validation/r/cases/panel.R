# validation/r/cases/panel.R -- plm / fixest reference cases for the panel exhibits.
suppressPackageStartupMessages({ library(plm); library(fixest) })

# Wald tests of the slopes with each Arellano covariance, as pwaldtest with a
# cluster vcov; the F form then has df2 = G - 1 (plm's df2adj), as summary(m,
# vcov = ...) reports. Keys <stat>_arellano_<type>, <p>_arellano_<type> and,
# for the F form, <df2>_arellano.
.arellano_tests <- function(m, test, stat, p, df2 = NULL) {
  out <- list()
  for (type in c("HC0", "HC1", "sss")) {
    w <- pwaldtest(m, test = test, vcov = vcovHC(m, method = "arellano", type = type))
    sfx <- tolower(type)
    out[[paste0(stat, "_arellano_", sfx)]] <- unname(w$statistic)
    out[[paste0(p, "_arellano_", sfx)]] <- unname(w$p.value)
    if (!is.null(df2)) out[[paste0(df2, "_arellano")]] <- unname(w$parameter[2])
  }
  out
}

.plm_fe_values <- function(fe) {
  s <- summary(fe); V <- vcov(fe); se <- sqrt(diag(V)); b <- coef(fe); df <- df.residual(fe)
  c(list(
    coef = named(b), se = named(se), t = named(b / se),
    p = named(2 * pt(abs(b / se), df, lower.tail = FALSE)),
    ci95_lower = named(b - qt(0.975, df) * se), ci95_upper = named(b + qt(0.975, df) * se),
    r2 = unname(s$r.squared["rsq"]), adj_r2 = unname(s$r.squared["adjrsq"]),
    f_stat = unname(s$fstatistic$statistic), f_df1 = unname(s$fstatistic$parameter[1]),
    f_df2 = unname(s$fstatistic$parameter[2]), f_p = unname(s$fstatistic$p.value),
    sigma2 = deviance(fe) / df, df_resid = df, n_obs = nobs(fe), n_groups = pdim(fe)$nT$n,
    se_arellano_hc0 = named(sqrt(diag(vcovHC(fe, method = "arellano", type = "HC0")))),
    se_arellano_hc1 = named(sqrt(diag(vcovHC(fe, method = "arellano", type = "HC1")))),
    se_arellano_sss = named(sqrt(diag(vcovHC(fe, method = "arellano", type = "sss"))))
  ), .arellano_tests(fe, "F", "f_stat", "f_p", "f_df2"))
}

.plm_re_values <- function(re) {
  s <- summary(re); V <- vcov(re); se <- sqrt(diag(V)); b <- coef(re); ec <- ercomp(re)
  th <- ec$theta; z <- b / se
  wald <- s$fstatistic   # plm reports a Wald chi-squared for random models
  Xs <- model.matrix(re, model = "random"); ys <- pmodel.response(re, model = "random")
  c(list(
    coef = named(b), se = named(se), z = named(z),
    p = named(2 * pnorm(abs(z), lower.tail = FALSE)),
    ci95_lower = named(b - qnorm(0.975) * se), ci95_upper = named(b + qnorm(0.975) * se),
    sigma2_idios = unname(ec$sigma2["idios"]), sigma2_id = unname(ec$sigma2["id"]),
    theta_mean = mean(th), theta_min = min(th), theta_max = max(th),
    resid_var_transformed = sum((ys - Xs %*% b)^2) / (nrow(Xs) - ncol(Xs)),
    r2 = unname(s$r.squared["rsq"]), adj_r2 = unname(s$r.squared["adjrsq"]),
    r2_overall_cor = unname(cor(pmodel.response(re, model = "pooling"), model.matrix(re, model = "pooling") %*% b)[1]^2),
    wald_chisq = unname(wald$statistic), wald_df = unname(wald$parameter), wald_p = unname(wald$p.value),
    wald_f = unname(pwaldtest(re, test = "F")$statistic), wald_f_df2 = unname(pwaldtest(re, test = "F")$parameter[2]),
    wald_f_p = unname(pwaldtest(re, test = "F")$p.value),
    df_resid = df.residual(re), n_obs = nobs(re), n_groups = pdim(re)$nT$n,
    se_arellano_hc0 = named(sqrt(diag(vcovHC(re, method = "arellano", type = "HC0")))),
    se_arellano_hc1 = named(sqrt(diag(vcovHC(re, method = "arellano", type = "HC1")))),
    se_arellano_sss = named(sqrt(diag(vcovHC(re, method = "arellano", type = "sss"))))
  ), .arellano_tests(re, "Chisq", "wald_chisq", "wald_p"),
  .arellano_tests(re, "F", "wald_f", "wald_f_p", "wald_f_df2"))
}

.phtest_values <- function(fe, re) {
  h <- phtest(fe, re)
  cn <- names(coef(fe))
  dV <- vcov(fe)[cn, cn] - vcov(re)[cn, cn]
  list(chisq = unname(h$statistic), df = unname(h$parameter), p = unname(h$p.value),
       coef_diff = named(coef(fe)[cn] - coef(re)[cn]),
       vdiff_eigen = vec(sort(eigen(dV, symmetric = TRUE, only.values = TRUE)$values)),
       quad_form_signed = as.numeric(t(coef(fe)[cn] - coef(re)[cn]) %*% solve(dV) %*% (coef(fe)[cn] - coef(re)[cn])))
}

.panel_cases <- function(id_prefix, label, load) {
  case(paste0("panel_fe_", id_prefix), "econometrics/panel", paste("Fixed effects (within),", label), function(ctx) {
    pd <- pdata.frame(load(ctx), index = c("firm", "year"))
    fe <- plm(inv ~ value + capital, data = pd, model = "within")
    list(values = .plm_fe_values(fe), reference = "plm::plm(model = 'within'); plm::vcovHC(method = 'arellano')",
         call = "plm(inv ~ value + capital, data = pdata.frame(d, index = c('firm', 'year')), model = 'within')",
         packages = "plm")
  })
  case(paste0("panel_re_", id_prefix), "econometrics/panel", paste("Random effects (Swamy-Arora),", label), function(ctx) {
    pd <- pdata.frame(load(ctx), index = c("firm", "year"))
    re <- plm(inv ~ value + capital, data = pd, model = "random")
    list(values = .plm_re_values(re), reference = "plm::plm(model = 'random', random.method = 'swar'); plm::ercomp; plm::vcovHC(method = 'arellano')",
         call = "plm(inv ~ value + capital, data = pdata.frame(d, index = c('firm', 'year')), model = 'random')",
         packages = "plm")
  })
  case(paste0("panel_hausman_", id_prefix), "econometrics/panel", paste("Hausman test (FE vs RE),", label), function(ctx) {
    pd <- pdata.frame(load(ctx), index = c("firm", "year"))
    fe <- plm(inv ~ value + capital, data = pd, model = "within")
    re <- plm(inv ~ value + capital, data = pd, model = "random")
    list(values = .phtest_values(fe, re), reference = "plm::phtest(fe, re) (method = 'chisq', classical vcov)",
         call = "phtest(plm(..., model = 'within'), plm(..., model = 'random'))", packages = "plm")
  })
}

.grunfeld <- function(ctx) ctx$dataset("datasets/grunfeld.csv")
.grunfeld_unbalanced <- function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  ctx$gen_dataset("grunfeld_unbalanced", g[(g$firm + g$year) %% 7 != 0, ])
}
.panel_cases("grunfeld", "Grunfeld (balanced, N = 10, T = 20)", .grunfeld)
.panel_cases("grunfeld_unbalanced", "Grunfeld without rows where (firm + year) %% 7 == 0", .grunfeld_unbalanced)

# A panel whose entity effects are correlated with x1: the Hausman test must reject.
.sim_corr <- function(ctx) {
  set.seed(20261021)
  N <- 60L; TT <- 5L
  a <- rnorm(N)
  d <- data.frame(firm = rep(seq_len(N), each = TT), year = rep(2001L:2005L, N))
  d$value <- a[d$firm] + rnorm(N * TT)              # x1, correlated with the effect
  d$capital <- rnorm(N * TT)                         # x2
  d$inv <- 1 + 1.0 * d$value - 0.5 * d$capital + 1.5 * a[d$firm] + rnorm(N * TT)
  ctx$gen_dataset("panel_corr_effects", d[, c("firm", "year", "inv", "value", "capital")])
}
.panel_cases("sim_corr", "simulated N = 60, T = 5, effects correlated with x1", .sim_corr)

# A panel without entity effects where the Swamy-Arora sigma_u^2 estimate is
# negative: plm truncates it to 0, so theta = 0 and RE equals pooled OLS.
.sim_no_effect <- function(ctx) {
  set.seed(3)
  N <- 30L; TT <- 4L
  d <- data.frame(firm = rep(seq_len(N), each = TT), year = rep(2001L:2004L, N))
  d$value <- rnorm(N * TT); d$capital <- rnorm(N * TT)
  d$inv <- 2 + 0.5 * d$value - 0.3 * d$capital + rnorm(N * TT)
  ctx$gen_dataset("panel_no_effect", d)
}
case("panel_re_sim_no_effect", "econometrics/panel", "Random effects with sigma_u^2 truncated to 0 (simulated, no entity effect)", function(ctx) {
  pd <- pdata.frame(.sim_no_effect(ctx), index = c("firm", "year"))
  re <- plm(inv ~ value + capital, data = pd, model = "random")
  v <- .plm_re_values(re)
  stopifnot(v$sigma2_id == 0, v$theta_mean == 0)
  ols <- lm(inv ~ value + capital, data = as.data.frame(pd))
  v$coef_pooled_ols <- named(coef(ols))
  list(values = v, reference = "plm::plm(model = 'random'); stats::lm (pooled, theta = 0)",
       call = "plm(inv ~ value + capital, data = pdata.frame(d, index = c('firm', 'year')), model = 'random')", packages = "plm")
})

case("panel_hdfe_grunfeld", "econometrics/hdfe", "Two-way fixed effects (firm + year), Grunfeld", function(ctx) {
  g <- ctx$dataset("datasets/grunfeld.csv")
  h <- feols(inv ~ value + capital | firm + year, g, vcov = "iid")
  s_iid <- se(summary(h, vcov = "iid")); s_het <- se(summary(h, vcov = "hetero")); s_cl <- se(summary(h, vcov = ~firm))
  wf <- fitstat(h, "wf")[[1]]
  tw <- plm(inv ~ value + capital, data = pdata.frame(g, index = c("firm", "year")), model = "within", effect = "twoways")
  h1 <- feols(inv ~ value + capital | firm, g, vcov = "iid")
  list(values = list(
    coef = named(coef(h)), se_iid = named(s_iid), se_hetero = named(s_het), se_cluster_firm = named(s_cl),
    t_hetero = named(coef(h) / s_het), p_hetero = named(2 * pt(abs(coef(h) / s_het), degrees_freedom(h, "resid"), lower.tail = FALSE)),
    r2_within = unname(r2(h, "wr2")), adj_r2_within_plm = unname(summary(tw)$r.squared["adjrsq"]),
    adj_r2_within_fixest = unname(r2(h, "war2")), r2_full = unname(r2(h, "r2")),
    within_f = unname(wf$stat), within_f_df1 = unname(wf$df1), within_f_df2 = unname(wf$df2), within_f_p = unname(wf$p),
    df_resid = unname(degrees_freedom(h, "resid")), n_obs = nobs(h),
    coef_firm_only = named(coef(h1)), se_firm_only_iid = named(se(h1))
  ), reference = "fixest::feols(inv ~ value + capital | firm + year); plm::plm(effect = 'twoways') for the adjusted within R-squared",
  call = "feols(inv ~ value + capital | firm + year, grunfeld, vcov = 'iid' / 'hetero' / ~firm)", packages = c("fixest", "plm"))
})
