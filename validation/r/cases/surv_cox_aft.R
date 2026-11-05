# validation/r/cases/surv_cox_aft.R -- Cox PH (Breslow and Efron ties, model-based, robust Lin-Wei and
# clustered SEs, convergence at any covariate scale, aliased covariates) and AFT (survreg: weibull,
# exponential, lognormal, loglogistic) references. T3-WP19: ECON-45, ECON-47, ECON-48, X-15; moved in
# ECON-46. Package copy of T3 design Appendix A.15 (canonical; it supersedes A.15).
# Data: survival::lung as committed by T2-WP09 (validation/datasets/generated/lung.csv; integer days,
# status 1/2, one NA in ph.ecog and in inst) and a 10-row heavy-ties fixture written here.
# Rows: R's na.omit over the model's columns, which is what p2a's extractor drops (227 rows for
# age + sex + ph.ecog; 226 when cluster(inst) is added; 228 for age + sex).
suppressPackageStartupMessages(library(survival))

.cox_ctl <- function() coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 100)

# Wald test from fit$wald.test (summary.coxph rounds its waldtest to 2 decimals) on the fit's variance
# (the robust one when robust or cluster); robust score test fit$rscore (score residuals at beta = 0).
.cox_tests <- function(f, ties, suffix = "") {
  out <- list()
  df <- sum(!is.na(coef(f)))
  out[[paste0(ties, "_wald_test", suffix)]] <- f$wald.test
  out[[paste0(ties, "_wald_p", suffix)]] <- pchisq(f$wald.test, df, lower.tail = FALSE)
  if (!is.null(f$rscore)) {
    out[[paste0(ties, "_rscore", suffix)]] <- f$rscore
    out[[paste0(ties, "_rscore_p", suffix)]] <- pchisq(f$rscore, df, lower.tail = FALSE)
  }
  out
}

case("surv_cox_robust", "econometrics/survival", "Cox PH, Efron and Breslow ties: model-based, robust (Lin-Wei) and cluster(inst) SEs and tests; scale invariance of convergence; aliased covariate; lung", function(ctx) {
  l <- ctx$dataset("datasets/generated/lung.csv")
  d <- l[complete.cases(l[, c("time", "status", "age", "sex", "ph.ecog")]), ]   # 227 rows
  fm <- Surv(time, status) ~ age + sex + ph.ecog
  v <- list()
  for (ties in c("efron", "breslow")) {
    f <- coxph(fm, data = d, ties = ties, control = .cox_ctl())
    fr <- coxph(fm, data = d, ties = ties, robust = TRUE, control = .cox_ctl())
    fc <- coxph(update(fm, . ~ . + cluster(inst)), data = d, ties = ties, control = .cox_ctl())   # drops the inst NA row
    s <- summary(f); sr <- summary(fr); sc <- summary(fc)
    stopifnot(fc$n == 226, f$n == 227, isTRUE(all.equal(coef(f), coef(fr), tolerance = 0)))
    v[[paste0(ties, "_coef")]] <- named(coef(f))
    v[[paste0(ties, "_se")]] <- named(s$coefficients[, "se(coef)"])
    v[[paste0(ties, "_z")]] <- named(s$coefficients[, "z"])
    v[[paste0(ties, "_p")]] <- named(s$coefficients[, "Pr(>|z|)"])
    v[[paste0(ties, "_hr_lower")]] <- named(s$conf.int[, "lower .95"])
    v[[paste0(ties, "_hr_upper")]] <- named(s$conf.int[, "upper .95"])
    v[[paste0(ties, "_loglik")]] <- f$loglik[2]
    v[[paste0(ties, "_loglik_null")]] <- f$loglik[1]
    v[[paste0(ties, "_lr_test")]] <- unname(s$logtest["test"])
    v[[paste0(ties, "_lr_p")]] <- unname(s$logtest["pvalue"])
    v[[paste0(ties, "_score_test")]] <- unname(s$sctest["test"])
    v[[paste0(ties, "_score_p")]] <- unname(s$sctest["pvalue"])
    v <- c(v, .cox_tests(f, ties))
    v[[paste0(ties, "_iter")]] <- f$iter
    # robust = TRUE: Lin-Wei sandwich from the dfbeta residuals; z, p and the HR interval use it
    v[[paste0(ties, "_se_robust")]] <- named(sr$coefficients[, "robust se"])
    v[[paste0(ties, "_z_robust")]] <- named(sr$coefficients[, "z"])
    v[[paste0(ties, "_p_robust")]] <- named(sr$coefficients[, "Pr(>|z|)"])
    v[[paste0(ties, "_hr_lower_robust")]] <- named(sr$conf.int[, "lower .95"])
    v[[paste0(ties, "_hr_upper_robust")]] <- named(sr$conf.int[, "upper .95"])
    v <- c(v, .cox_tests(fr, ties, "_robust"))
    # cluster(inst): dfbeta residuals summed within institution
    v[[paste0(ties, "_coef_cluster")]] <- named(coef(fc))
    v[[paste0(ties, "_se_model_cluster")]] <- named(sc$coefficients[, "se(coef)"])
    v[[paste0(ties, "_se_cluster")]] <- named(sc$coefficients[, "robust se"])
    v[[paste0(ties, "_z_cluster")]] <- named(sc$coefficients[, "z"])
    v[[paste0(ties, "_p_cluster")]] <- named(sc$coefficients[, "Pr(>|z|)"])
    v <- c(v, .cox_tests(fc, ties, "_cluster"))
  }
  v$n <- nrow(d); v$n_event <- sum(d$status == 2)
  dc <- d[!is.na(d$inst), ]
  v$n_cluster_obs <- nrow(dc); v$n_cluster_event <- sum(dc$status == 2); v$n_clusters <- length(unique(dc$inst))
  # X-15: one covariate rescaled; coxph converges on |1 - loglik_old / loglik_new| <= eps (default 1e-9,
  # Efron) in the same number of iterations at every scale (full lung: age and sex have no NA, 228 rows)
  for (sc_ in list(c("em6", 1e-6), c("e0", 1), c("days", 365250), c("e6", 1e6))) {
    l$age_s <- l$age * as.numeric(sc_[2])
    f2 <- coxph(Surv(time, status) ~ age_s + sex, data = l)
    v[[paste0("scale_", sc_[1], "_coef")]] <- named(setNames(coef(f2), c("age", "sex")))
    v[[paste0("scale_", sc_[1], "_iter")]] <- f2$iter
  }
  ft <- coxph(Surv(time, status) ~ age + sex, data = l, control = .cox_ctl())   # tight fit of the scale model (228 rows)
  v$scale_tight_coef <- named(setNames(coef(ft), c("age", "sex")))
  v$scale_n <- nrow(l)
  # aliased covariate: age2 = 2 * age + sex; coxph reports NA for the later of the collinear columns
  l$age2 <- 2 * l$age + l$sex
  fa <- coxph(Surv(time, status) ~ age + sex + age2, data = l, control = .cox_ctl())
  stopifnot(is.na(coef(fa)["age2"]))
  v$aliased_coef <- named(coef(fa))
  se_a <- setNames(sqrt(diag(vcov(fa))), names(coef(fa)))
  se_a[is.na(coef(fa))] <- NA                     # R prints 0 for the aliased column; p2a reports NaN (D1)
  v$aliased_se <- named(se_a)
  v$aliased_loglik <- fa$loglik[2]
  v$aliased_lr_test <- 2 * diff(fa$loglik)         # on 2 df (the identified columns)
  list(values = v, reference = "survival::coxph(ties, robust = TRUE | cluster(inst), coxph.control(eps = 1e-12, toler.chol = 1e-13)); summary.coxph",
       call = "coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung, ties, robust)", packages = "survival")
})

case("surv_cox_ties_small", "econometrics/survival", "Cox PH with heavy ties (10 rows, 5 distinct times), Efron and Breslow", function(ctx) {
  toy <- ctx$gen_dataset("cox_ties_small", data.frame(
    time = c(1, 1, 2, 2, 2, 3, 4, 4, 5, 5), event = c(1, 1, 1, 0, 1, 1, 1, 0, 1, 1), x = c(0, 1, 0, 0, 1, 1, 0, 1, 0, 1)))
  v <- list()
  for (ties in c("efron", "breslow")) {
    f <- coxph(Surv(time, event) ~ x, data = toy, ties = ties, control = .cox_ctl())
    rv <- coxph(Surv(time, event) ~ x, data = toy[nrow(toy):1, ], ties = ties, control = .cox_ctl())
    stopifnot(isTRUE(all.equal(coef(f), coef(rv), tolerance = 1e-14)))   # row order is irrelevant in R
    v[[paste0(ties, "_coef")]] <- named(coef(f))
    v[[paste0(ties, "_se")]] <- named(setNames(sqrt(diag(vcov(f))), "x"))
    v[[paste0(ties, "_loglik")]] <- f$loglik[2]
    v[[paste0(ties, "_loglik_null")]] <- f$loglik[1]
    v[[paste0(ties, "_score_test")]] <- f$score
  }
  v$n <- nrow(toy)
  list(values = v, reference = "survival::coxph(ties = 'efron' | 'breslow', coxph.control(eps = 1e-12, toler.chol = 1e-13))",
       call = "coxph(Surv(time, event) ~ x, data = cox_ties_small, ties)", packages = "survival")
})

case("surv_aft", "econometrics/survival", "Accelerated failure time (survreg): weibull, exponential, lognormal, loglogistic on lung; scale invariance; aliased covariate", function(ctx) {
  l <- ctx$dataset("datasets/generated/lung.csv")
  d <- l[complete.cases(l[, c("time", "status", "age", "sex", "ph.ecog")]), ]   # 227 rows
  ctl <- survreg.control(rel.tolerance = 1e-12, maxiter = 100)
  v <- list()
  for (dist in c("weibull", "exponential", "lognormal", "loglogistic")) {
    f <- survreg(Surv(time, status) ~ age + sex + ph.ecog, data = d, dist = dist, control = ctl)
    tab <- summary(f)$table
    cn <- names(coef(f))
    v[[paste0(dist, "_coef")]] <- named(coef(f))
    v[[paste0(dist, "_se")]] <- named(tab[cn, "Std. Error"])
    v[[paste0(dist, "_z")]] <- named(tab[cn, "z"])
    v[[paste0(dist, "_p")]] <- named(tab[cn, "p"])
    v[[paste0(dist, "_scale")]] <- f$scale
    if (dist != "exponential") {
      v[[paste0(dist, "_se_log_scale")]] <- tab["Log(scale)", "Std. Error"]
      v[[paste0(dist, "_z_log_scale")]] <- tab["Log(scale)", "z"]
      v[[paste0(dist, "_p_log_scale")]] <- tab["Log(scale)", "p"]
    }
    v[[paste0(dist, "_loglik")]] <- f$loglik[2]          # on the time scale (log-T log-likelihood minus sum(status * log t))
    v[[paste0(dist, "_loglik_null")]] <- f$loglik[1]     # intercept-only fit; not reported by p2a (not pinned)
    v[[paste0(dist, "_aic")]] <- AIC(f)                  # k = 5, exponential k = 4
    v[[paste0(dist, "_bic")]] <- BIC(f)
  }
  v$n <- nrow(d); v$n_event <- sum(d$status == 2)
  # convention 5: age x 1e-6 and x 1e6 give the rescaled solution (weibull, tight)
  for (sc_ in list(c("em6", 1e-6), c("e6", 1e6))) {
    ds <- d; ds$age <- ds$age * as.numeric(sc_[2])
    f <- survreg(Surv(time, status) ~ age + sex + ph.ecog, data = ds, dist = "weibull", control = ctl)
    v[[paste0("weibull_scale_", sc_[1], "_coef")]] <- named(coef(f))
  }
  # aliased covariate (228 rows): survreg reports NA for age2
  l$age2 <- 2 * l$age + l$sex
  fa <- survreg(Surv(time, status) ~ age + sex + age2, data = l, dist = "weibull", control = ctl)
  stopifnot(is.na(coef(fa)["age2"]))
  v$aliased_coef <- named(coef(fa))
  v$aliased_scale <- fa$scale
  v$aliased_loglik <- fa$loglik[2]
  v$aliased_n <- nrow(l)
  list(values = v, reference = "survival::survreg(dist, survreg.control(rel.tolerance = 1e-12)); summary.survreg; AIC; BIC",
       call = "survreg(Surv(time, status) ~ age + sex + ph.ecog, data = lung, dist)", packages = "survival")
})
