# validation/r/cases/survival.R -- survfit / survdiff references (T2-WP09: ECON-6, ECON-7, ECON-77).
suppressPackageStartupMessages(library(survival))

.lung <- function(ctx) {
  l <- survival::lung
  # status is coded 1 = censored, 2 = dead (R's Surv convention for 1/2 codes); time is integer days.
  ctx$gen_dataset("lung", l[, c("inst", "time", "status", "age", "sex", "ph.ecog", "ph.karno", "wt.loss")])
}

.km_values <- function(fit, times, prefix) {
  s <- summary(fit, times = times, extend = TRUE)
  out <- list()
  out[[paste0(prefix, "_surv")]] <- vec(s$surv)
  out[[paste0(prefix, "_std_err")]] <- vec(s$std.err)
  out[[paste0(prefix, "_lower")]] <- vec(s$lower)
  out[[paste0(prefix, "_upper")]] <- vec(s$upper)
  out[[paste0(prefix, "_n_risk")]] <- vec(s$n.risk)
  q <- quantile(fit, probs = 0.5)
  out[[paste0(prefix, "_median")]] <- unname(q$quantile)
  out[[paste0(prefix, "_median_lower")]] <- unname(q$lower)
  out[[paste0(prefix, "_median_upper")]] <- unname(q$upper)
  out
}

case("surv_km_lung", "econometrics/survival", "Kaplan-Meier with log, log-log and plain CIs, lung (integer time, 1/2 status)", function(ctx) {
  d <- .lung(ctx)
  stopifnot(is.integer(d$time), all(d$status %in% 1:2))
  times <- c(5, 11, 100, 310, 500, 765)
  v <- list()
  for (ct in c("log", "log-log", "plain")) {
    fit <- survfit(Surv(time, status) ~ 1, data = d, conf.type = ct)
    v <- c(v, .km_values(fit, times, gsub("-", "", ct)))
  }
  fit <- survfit(Surv(time, status) ~ 1, data = d)
  v$n <- fit$n; v$n_event <- sum(fit$n.event)
  v$unique_times <- length(fit$time)
  # all event-time estimates (for a full-curve comparison of the default log CI)
  v$curve_time <- vec(fit$time); v$curve_surv <- vec(fit$surv); v$curve_std_err_log <- vec(fit$std.err)
  v$curve_lower <- vec(ifelse(is.finite(fit$lower), fit$lower, NA)); v$curve_upper <- vec(ifelse(is.finite(fit$upper), fit$upper, NA))
  list(values = v, reference = "survival::survfit(Surv(time, status) ~ 1, conf.type = 'log' (default) | 'log-log' | 'plain'); quantile.survfit",
       call = "survfit(Surv(time, status) ~ 1, data = lung, conf.type = ...)", packages = "survival")
})

case("surv_km_lung_by_sex", "econometrics/survival", "Kaplan-Meier by group (sex), log-log CI and median CI", function(ctx) {
  d <- .lung(ctx)
  fit <- survfit(Surv(time, status) ~ sex, data = d, conf.type = "log-log")
  v <- list()
  for (s in 1:2) {
    f1 <- survfit(Surv(time, status) ~ 1, data = d[d$sex == s, ], conf.type = "log-log")
    v <- c(v, .km_values(f1, c(100, 300, 500), paste0("sex", s)))
  }
  list(values = v, reference = "survival::survfit(Surv(time, status) ~ sex, conf.type = 'log-log')",
       call = "survfit(Surv(time, status) ~ sex, data = lung, conf.type = 'log-log')", packages = "survival")
})

case("surv_logrank_lung", "econometrics/survival", "Log-rank test, lung by sex and by ph.ecog (one NA group label)", function(ctx) {
  d <- .lung(ctx)
  s1 <- survdiff(Surv(time, status) ~ sex, data = d)
  s2 <- survdiff(Surv(time, status) ~ ph.ecog, data = d)
  stopifnot(sum(is.na(d$ph.ecog)) == 1)
  list(values = list(
    sex_chisq = s1$chisq, sex_df = length(s1$n) - 1, sex_p = pchisq(s1$chisq, length(s1$n) - 1, lower.tail = FALSE),
    sex_obs = vec(s1$obs), sex_exp = vec(s1$exp), sex_n = vec(as.numeric(s1$n)), sex_var = vec(as.vector(s1$var)),
    ecog_chisq = s2$chisq, ecog_df = length(s2$n) - 1, ecog_p = pchisq(s2$chisq, length(s2$n) - 1, lower.tail = FALSE),
    ecog_obs = vec(s2$obs), ecog_exp = vec(s2$exp), ecog_n = vec(as.numeric(s2$n)), ecog_n_used = sum(s2$n)),
    reference = "survival::survdiff(Surv(time, status) ~ group) (rho = 0); rows with NA group dropped",
    call = "survdiff(Surv(time, status) ~ sex, data = lung); survdiff(Surv(time, status) ~ ph.ecog, data = lung)",
    packages = "survival")
})

# The 12-row dataset of the existing test_validate_logrank_against_r (survival.rs), 0/1 status.
case("surv_logrank_small", "econometrics/survival", "Log-rank test, 12 rows with ties and 0/1 status", function(ctx) {
  d <- ctx$gen_dataset("logrank_small", data.frame(
    time = c(1, 2, 3, 5, 6, 7, 2, 3, 4, 5, 8, 9),
    event = c(1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1),
    group = c(rep("0", 6), rep("1", 6))))
  s <- survdiff(Surv(time, event) ~ group, data = d)
  list(values = list(chisq = s$chisq, p = pchisq(s$chisq, 1, lower.tail = FALSE), obs = vec(s$obs), exp = vec(s$exp)),
       reference = "survival::survdiff", call = "survdiff(Surv(time, event) ~ group)", packages = "survival")
})

# The 10-row dataset of the existing test_validate_km_against_r: S drops to 0 at the last time.
case("surv_km_small", "econometrics/survival", "Kaplan-Meier, 10 rows ending in an event (S = 0), log and log-log CIs", function(ctx) {
  d <- ctx$gen_dataset("km_small", data.frame(time = 1:10, event = c(1, 1, 0, 1, 1, 0, 1, 0, 1, 1)))
  v <- list()
  for (ct in c("log", "log-log")) {
    f <- survfit(Surv(time, event) ~ 1, data = d, conf.type = ct)
    tag <- gsub("-", "", ct)
    v[[paste0(tag, "_time")]] <- vec(f$time); v[[paste0(tag, "_surv")]] <- vec(f$surv)
    v[[paste0(tag, "_std_err_surv")]] <- vec(summary(f, times = f$time)$std.err)
    v[[paste0(tag, "_lower")]] <- vec(f$lower); v[[paste0(tag, "_upper")]] <- vec(f$upper)
    q <- quantile(f, 0.5)
    v[[paste0(tag, "_median")]] <- unname(q$quantile); v[[paste0(tag, "_median_lower")]] <- unname(q$lower)
    v[[paste0(tag, "_median_upper")]] <- unname(q$upper)
  }
  list(values = v, reference = "survival::survfit(conf.type = 'log' | 'log-log'); quantile.survfit",
       call = "survfit(Surv(time, event) ~ 1, data = d)", packages = "survival")
})

# (addition, T2-WP09) quantile.survfit's median rules: the midpoint when S equals 0.5 over an interval,
# the "last point" rule when the curve ends at 0.5, NA when the curve never reaches 0.5; the
# confidence limits at S = 1 (log-log is NA there); and survdiff's exclusion of a group whose
# expected count is 0 (its members leave before the first event).
case("surv_km_median_rules", "econometrics/survival", "Kaplan-Meier medians by quantile.survfit (midpoint, last-point, unreached), CIs at S = 1, log-rank with a zero-expected group", function(ctx) {
  time  <- c(1, 2, 3, 4, 5, 6, 7,   1, 2, 3, 4,   1, 2, 3, 4, 5)
  event <- c(0, 1, 1, 1, 0, 1, 1,   1, 1, 0, 0,   1, 0, 0, 0, 0)
  group <- c(rep("mid", 7), rep("last", 4), rep("never", 5))
  nv <- group == "never"
  d <- ctx$gen_dataset("km_median_rules", data.frame(
    time = time, event = event, group = group,
    time_early = ifelse(nv, time / 100, time),      # "never" leaves before the first event ...
    event_early = ifelse(nv, 0, event),             # ... censored, so its expected count is 0
    group_no_mid = ifelse(group == "mid", NA, group)))
  stopifnot(is.integer(d$time), is.integer(d$event), is.double(d$time_early), sum(is.na(d$group_no_mid)) == 7)
  v <- list()
  for (g in c("mid", "last", "never")) {
    for (ct in c("log", "log-log", "plain")) {
      f <- survfit(Surv(time, event) ~ 1, data = d[d$group == g, ], conf.type = ct)
      tag <- paste0(g, "_", gsub("-", "", ct))
      if (ct == "log") {
        v[[paste0(g, "_time")]] <- vec(f$time); v[[paste0(g, "_surv")]] <- vec(f$surv)
        v[[paste0(g, "_n_risk")]] <- vec(f$n.risk)
      }
      v[[paste0(tag, "_lower")]] <- vec(f$lower); v[[paste0(tag, "_upper")]] <- vec(f$upper)
      q <- quantile(f, 0.5)
      v[[paste0(tag, "_median")]] <- unname(q$quantile)
      v[[paste0(tag, "_median_lower")]] <- unname(q$lower)
      v[[paste0(tag, "_median_upper")]] <- unname(q$upper)
    }
  }
  # stratified fit: strata in factor(group) order = last < mid < never (C collation)
  fs <- survfit(Surv(time, event) ~ group, data = d)
  v$strata_n <- vec(as.numeric(fs$n))
  v$strata_median <- vec(unname(quantile(fs, 0.5)$quantile[, 1]))
  # log-rank where "never" has expected count 0: survdiff drops it from the test (df = 1)
  s <- survdiff(Surv(time_early, event_early) ~ group, data = d)
  s_sub <- survdiff(Surv(time, event) ~ group, data = d[d$group != "never", ])
  v$lr_zero_chisq <- s$chisq; v$lr_zero_df <- sum(s$exp > 0) - 1
  v$lr_zero_p <- pchisq(s$chisq, sum(s$exp > 0) - 1, lower.tail = FALSE)
  v$lr_zero_exp <- vec(s$exp); v$lr_zero_obs <- vec(s$obs)
  v$lr_subset_chisq <- s_sub$chisq
  # only "last" has expected events once "mid" (NA label) is dropped: R returns chisq 0 on 0 df
  s1 <- survdiff(Surv(time_early, event_early) ~ group_no_mid, data = d)
  v$lr_one_group_chisq <- s1$chisq; v$lr_one_group_n_used <- sum(s1$n)
  list(values = v, reference = "survival::survfit(conf.type = 'log' | 'log-log' | 'plain'); survival:::quantile.survfit (findq); survival::survdiff",
       call = "survfit(Surv(time, event) ~ 1, data = d[d$group == g, ], conf.type = ct); quantile(fit, 0.5); survdiff(Surv(time_early, event_early) ~ group)",
       packages = "survival")
})
