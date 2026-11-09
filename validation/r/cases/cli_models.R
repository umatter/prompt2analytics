# validation/r/cases/cli_models.R -- references for the T3-WP60 CLI tests (CLI-3, CLI-4, CLI-23
# competing-risks part, CLI-29). The CLI runs one process per command, so these cases pin what a fit
# returns through the binary, including on a dataset that an earlier `munge` invocation derived and
# persisted in the session.
#  * cli_discrete_poisson: glm(family = poisson) on T2's committed negbin.csv, complete and with
#    missing cells (na.omit), at glm.control(epsilon = 1e-10, maxit = 100) (p2a's default control).
#  * cli_competing_risks: the Aalen-Johansen cumulative incidence of two competing causes, its
#    standard errors and log-scale limits as survival::survfit(Surv(time, factor(status))) computes
#    them (pstate, std.err, lower/upper at conf.int 0.95 and 0.5).
#  * cli_hac_aliased: `reg hac ols_aliased -y y -x x1 x2 x3 --lags 4`; x2 = 2 x1 is aliased, so the
#    t-based p-values use df.residual = n - rank = 117, not n - 4.
#  * cli_munge_ols_grunfeld: lm with HC1 SEs (the CLI default `--robust hc1`) on two derived Grunfeld
#    datasets: `munge select g --columns inv value --name sub` then `reg ols sub -y inv -x value`, and
#    `munge filter g -c year -o ge -v 1945 -n post` then `reg ols post -y inv -x value capital`; the first
#    rows of `munge sort g --by inv` (ascending inv).

case("cli_discrete_poisson", "cli", "glm(family = poisson) on the count data of disc_negbin, complete and with missing cells (na.omit)", function(ctx) {
  d <- ctx$dataset("datasets/generated/negbin.csv")          # committed by T2 (case disc_negbin)
  stopifnot(is.integer(d$y), nrow(d) == 400)
  one <- function(f) {
    m <- glm(f, family = poisson(), data = d, control = glm.control(epsilon = 1e-10, maxit = 100))
    stopifnot(m$converged)
    s <- summary(m)$coefficients
    list(coef = named(coef(m)), se = named(s[, 2]), z = named(s[, 3]), p = named(s[, 4]),
         loglik = as.numeric(logLik(m)), aic = AIC(m), deviance = deviance(m),
         null_deviance = m$null.deviance, n = nobs(m))
  }
  a <- one(y ~ x1 + x2); b <- one(y_na ~ x1_na + x2)
  v <- a; for (k in names(b)) v[[paste0("na_", k)]] <- b[[k]]
  v$na_n_dropped <- nrow(d) - b$n                             # 5 (disc_negbin's stopifnot)
  list(values = v, reference = "stats::glm(family = poisson) at glm.control(epsilon = 1e-10, maxit = 100) (p2a's default control, registry deviation ECON-60; the fit is at the MLE either way); na.omit",
       call = "glm(y ~ x1 + x2, family = poisson, data = negbin)", packages = character(0))
})
