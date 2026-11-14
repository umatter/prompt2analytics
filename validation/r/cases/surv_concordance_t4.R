# validation/r/cases/surv_concordance_t4.R -- T4-WP13 (ECON-89): the concordance of a Cox fit and its
# infinitesimal-jackknife standard error, as survival::coxph reports them (fit$concordance, computed by
# survival::concordancefit with timewt = "n", reverse = TRUE). A pair (i, j) is comparable when i has the
# event and t_i < t_j, or t_i = t_j with j censored; concordant when eta_i > eta_j, discordant when
# eta_i < eta_j, tied.x when equal; pairs of events at one time are tied.y and not comparable.
# C = (concordant + tied.x / 2) / (concordant + discordant + tied.x); with per-observation counts
# c_i, d_i, t_i over the comparable pairs it belongs to, somers = (C - D) / npair and
# std = sqrt(sum_i ((c_i - d_i) - somers (c_i + d_i + t_i))^2 / npair^2 / 4)
# (checked against a brute-force pair loop to the last digit).
suppressPackageStartupMessages(library(survival))

case("surv_concordance_t4", "econometrics/survival",
     "Cox concordance (Harrell's C as coxph reports it) and its infinitesimal-jackknife SE; lung age + sex, age + sex + ph.ecog, and the 10-row heavy-ties fixture; lung age + sex clustered on inst (grouped IJ SE)",
     function(ctx) {
  ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 100)
  cc <- function(f) {
    k <- f$concordance
    list(counts = named(k[c("concordant", "discordant", "tied.x", "tied.y", "tied.xy")]),
         c = unname(k["concordance"]), se = unname(k["std"]))
  }
  v <- list()
  toy <- ctx$dataset("datasets/generated/cox_ties_small.csv")
  for (ties in c("efron", "breslow")) {
    k <- cc(coxph(Surv(time, event) ~ x, data = toy, ties = ties, control = ctl))
    v[[paste0("ties_small_", ties, "_counts")]] <- k$counts
    v[[paste0("ties_small_", ties, "_c")]] <- k$c
    v[[paste0("ties_small_", ties, "_se")]] <- k$se
  }
  l <- ctx$dataset("datasets/generated/lung.csv")
  d2 <- l[complete.cases(l[, c("time", "status", "age", "sex")]), ]               # 228 rows
  d3 <- l[complete.cases(l[, c("time", "status", "age", "sex", "ph.ecog")]), ]    # 227 rows
  for (ties in c("efron", "breslow")) {
    k <- cc(coxph(Surv(time, status) ~ age + sex, data = d2, ties = ties, control = ctl))
    v[[paste0("lung_age_sex_", ties, "_counts")]] <- k$counts
    v[[paste0("lung_age_sex_", ties, "_c")]] <- k$c
    v[[paste0("lung_age_sex_", ties, "_se")]] <- k$se
  }
  k <- cc(coxph(Surv(time, status) ~ age + sex + ph.ecog, data = d3, ties = "efron", control = ctl))
  v$lung_age_sex_ecog_efron_counts <- k$counts
  v$lung_age_sex_ecog_efron_c <- k$c
  v$lung_age_sex_ecog_efron_se <- k$se
  # Clustered fit: coxph(cluster = ) reports the grouped IJ SE (per-observation terms summed within a
  # cluster before squaring).
  d4 <- l[complete.cases(l[, c("time", "status", "age", "sex", "inst")]), ]       # 227 rows
  k <- cc(coxph(Surv(time, status) ~ age + sex, data = d4, cluster = inst, control = ctl))
  v$lung_age_sex_cluster_inst_counts <- k$counts
  v$lung_age_sex_cluster_inst_c <- k$c
  v$lung_age_sex_cluster_inst_se <- k$se
  v$n_age_sex <- nrow(d2); v$n_age_sex_ecog <- nrow(d3); v$n_cluster_inst <- nrow(d4)
  list(values = v,
       reference = "survival::coxph(..., coxph.control(eps = 1e-12, toler.chol = 1e-13))$concordance (concordancefit: timewt = 'n', infinitesimal-jackknife std)",
       call = "coxph(Surv(time, status) ~ age + sex [+ ph.ecog], data = lung, ties); coxph(Surv(time, event) ~ x, data = cox_ties_small, ties); coxph(Surv(time, status) ~ age + sex, data = lung, cluster = inst)",
       packages = "survival", dataset = "validation/datasets/generated/lung.csv")
})

# A covariate that varies overall but is constant within every risk set at an event time (it varies
# only among rows censored before the first event): the information at beta = 0 is singular in that
# column. coxph's cholesky2 (toler.chol) leaves its coefficient NA, without a warning, and the other
# coefficients equal the fit without it (checked: coxph(Surv(time, status) ~ z) gives the same digits).
case("surv_cox_info_singular_t4", "econometrics/survival",
     "Cox PH with a covariate that has no variation within the event-time risk sets: NA coefficient, the rest as the fit without it",
     function(ctx) {
  ctl <- coxph.control(eps = 1e-12, toler.chol = 1e-13, iter.max = 100)
  d <- ctx$gen_dataset("cox_info_singular", data.frame(
    time = c(1, 2, 3, 4, 5, 6, 7, 8), status = c(0, 0, 1, 1, 0, 1, 1, 1),
    x = c(5, -3, 0, 0, 0, 0, 0, 0), z = c(0.3, 1.1, -0.4, 0.8, 0.2, -1.3, 0.6, 0.9)))
  f <- coxph(Surv(time, status) ~ x + z, data = d, control = ctl)
  g <- coxph(Surv(time, status) ~ z, data = d, control = ctl)
  stopifnot(is.na(coef(f)["x"]), isTRUE(all.equal(coef(f)[["z"]], coef(g)[["z"]], tolerance = 0)))
  k <- f$concordance
  list(values = list(
         x_is_na = as.numeric(is.na(coef(f)["x"])),
         coef_z = unname(coef(f)["z"]), se_z = sqrt(vcov(f)["z", "z"]),
         loglik_null = f$loglik[1], loglik = f$loglik[2],
         score_test = unname(f$score), wald_test = unname(f$wald.test),
         concordance = unname(k["concordance"]), concordance_se = unname(k["std"])),
       reference = "survival::coxph(Surv(time, status) ~ x + z, coxph.control(eps = 1e-12, toler.chol = 1e-13))",
       call = "coxph(Surv(time, status) ~ x + z, data = cox_info_singular)", packages = "survival")
})
