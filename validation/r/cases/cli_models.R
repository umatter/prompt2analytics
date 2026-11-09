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

case("cli_competing_risks", "cli", "Aalen-Johansen CIF of two competing causes; survfit's standard errors and log-scale limits at conf 0.95 and 0.5", function(ctx) {
  suppressPackageStartupMessages(library(survival))
  set.seed(20261023)
  n <- 200
  t1 <- rexp(n, 0.10); t2 <- rexp(n, 0.06); cens <- runif(n, 2, 18)
  tt <- pmin(t1, t2, cens)
  status <- ifelse(tt == cens, 0L, ifelse(tt == t1, 1L, 2L))
  d <- ctx$gen_dataset("cr_sim", data.frame(time = round(tt, 6), event_type = status))
  stopifnot(!anyDuplicated(d$time), all(d$time > 0))
  fit <- survfit(Surv(time, factor(event_type, levels = 0:2)) ~ 1, data = d)
  st <- fit$states                                        # "(s0)", "1", "2"
  stopifnot(identical(st, c("(s0)", "1", "2")), fit$conf.type == "log")
  times <- fit$time                                       # every distinct time, censored ones included
  # the CIF identity: the cause-specific increments weighted by the overall KM S(t_j-)
  ut <- sort(unique(d$time)); stopifnot(isTRUE(all.equal(ut, times, tolerance = 0)))
  nrisk <- sapply(ut, function(u) sum(d$time >= u))
  s_prev <- 1; cif <- matrix(0, length(ut), 2); cum <- c(0, 0)
  for (j in seq_along(ut)) {
    dk <- sapply(1:2, function(k) sum(d$time == ut[j] & d$event_type == k))
    cum <- cum + s_prev * dk / nrisk[j]
    s_prev <- s_prev * (1 - sum(dk) / nrisk[j])
    cif[j, ] <- cum
  }
  stopifnot(max(abs(cif[, 1] - fit$pstate[, st == "1"])) < 1e-14, max(abs(cif[, 2] - fit$pstate[, st == "2"])) < 1e-14)
  v <- list(n = n, n_censored = sum(d$event_type == 0), n_cause1 = sum(d$event_type == 1), n_cause2 = sum(d$event_type == 2),
            times = vec(ut))
  for (k in 1:2) {
    v[[paste0("cif_", k)]] <- vec(fit$pstate[, st == as.character(k)])
    v[[paste0("survfit_se_", k)]] <- vec(fit$std.err[, st == as.character(k)])
  }
  for (lev in c(0.95, 0.5)) {
    f <- survfit(Surv(time, factor(event_type, levels = 0:2)) ~ 1, data = d, conf.int = lev)
    stopifnot(identical(f$states, c("(s0)", "1", "2")), f$conf.type == "log")
    tag <- sprintf("survfit_ci%d", round(100 * lev))
    for (k in 1:2) {
      v[[sprintf("%s_lower_%d", tag, k)]] <- vec(f$lower[, k + 1])
      v[[sprintf("%s_upper_%d", tag, k)]] <- vec(f$upper[, k + 1])
    }
  }
  list(values = v, reference = "survival::survfit(Surv(time, factor(event_type))): pstate, std.err and log-scale limits at conf.int 0.95 and 0.5 (multi-state Aalen-Johansen)",
       call = "survfit(Surv(time, factor(event_type, levels = 0:2)) ~ 1, data = cr_sim, conf.int = level)", packages = "survival")
})

case("cli_hac_aliased", "cli", "Newey-West (lag 4, Bartlett, no prewhitening) t and p on df.residual = n - rank for a design with an aliased column", function(ctx) {
  suppressPackageStartupMessages(library(sandwich))
  d <- ctx$dataset("datasets/generated/ols_aliased.csv")     # committed with case ols_aliased
  m <- lm(y ~ x1 + x2 + x3, data = d)                        # x2 = 2 x1 is aliased (NA)
  stopifnot(is.na(coef(m)["x2"]), m$rank == 3)
  b <- coef(m)[!is.na(coef(m))]
  se <- sqrt(diag(NeweyWest(m, lag = 4, prewhite = FALSE)))
  t <- b / se
  list(values = list(coef = named(b), se_nw = named(se), t_nw = named(t),
                     p_nw = named(2 * pt(abs(t), df.residual(m), lower.tail = FALSE)),
                     p_nw_df_n_minus_k = named(2 * pt(abs(t), nrow(d) - length(coef(m)), lower.tail = FALSE)),
                     df_resid = df.residual(m), r2 = summary(m)$r.squared, n = nrow(d)),
       reference = "sandwich::NeweyWest(lm(y ~ x1 + x2 + x3), lag = 4, prewhite = FALSE) on the identified columns; t-based p on df.residual (n - rank); p_nw_df_n_minus_k (n - number of columns) is what the CLI printed before",
       call = "coeftest(lm(y ~ x1 + x2 + x3, ols_aliased), vcov = NeweyWest(m, lag = 4, prewhite = FALSE))", packages = "sandwich")
})

case("cli_munge_ols_grunfeld", "cli", "lm(HC1) on munge-derived Grunfeld datasets (select, filter) and the munge sort order", function(ctx) {
  suppressPackageStartupMessages(library(sandwich))
  g <- ctx$dataset("datasets/grunfeld.csv")
  fit <- function(m) {
    ct <- coef(m); se <- sqrt(diag(vcovHC(m, type = "HC1"))); t <- ct / se
    p <- 2 * pt(abs(t), df.residual(m), lower.tail = FALSE)
    s <- summary(m)
    list(coef = named(ct), se_hc1 = named(se), t_hc1 = named(t), p_hc1 = named(p),
         r2 = s$r.squared, adj_r2 = s$adj.r.squared, n = nobs(m), df_resid = df.residual(m))
  }
  sel <- fit(lm(inv ~ value, data = g[, c("inv", "value")]))
  post_d <- g[g$year >= 1945, ]
  post <- fit(lm(inv ~ value + capital, data = post_d))
  v <- list()
  for (k in names(sel)) v[[paste0("sel_", k)]] <- sel[[k]]
  for (k in names(post)) v[[paste0("post_", k)]] <- post[[k]]
  o <- order(g$inv)
  stopifnot(!anyDuplicated(g$inv[o][1:6]))          # Grunfeld has five tied inv pairs, none among the 6 smallest
  v$sorted_inv_head <- vec(g$inv[o][1:5])
  v$sorted_firm_head <- vec(g$firm[o][1:5])
  v$sorted_year_head <- vec(g$year[o][1:5])
  v$n_rows <- nrow(g); v$post_rows <- nrow(post_d)
  list(values = v, reference = "stats::lm on g[, c('inv','value')] and g[g$year >= 1945, ]; sandwich::vcovHC(type = 'HC1'); t-based p on df.residual; order(g$inv)",
       call = "lm(inv ~ value + capital, data = subset(grunfeld, year >= 1945))", packages = "sandwich")
})
