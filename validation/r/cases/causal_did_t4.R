# validation/r/cases/causal_did_t4.R -- Tier 4 (T4-WP16) references for the DiD family: did 2.5.1's
# default varying base period and its default overall aggregation (aggte type = "group"), did's sample
# rules (units treated in the first period dropped; small groups), Goodman-Bacon with always-treated units
# and a period coded 0, and fixest's within R-squared of the ETWFE fit.
#
# Unlike causal_did.R (written before `did` was installed), these cases call did::att_gt / did::aggte
# directly. On cs_panel the universal-base cells, the simple/group/dynamic aggregates and the Wald
# statistic of did 2.5.1 equal causal_did.R's transcription to 1e-15 (checked by the planner); the
# varying-base cases below therefore share the dataset and add only what changes.
suppressPackageStartupMessages(library(did))

# att_gt (no covariates, outcome regression, analytic SEs) and the three aggregations, flattened.
.did_t4_fit <- function(d, control, base) {
  a <- suppressWarnings(suppressMessages(did::att_gt(
    yname = "y", tname = "time", idname = "unit", gname = "first_treat", data = d,
    control_group = control, base_period = base, est_method = "reg", bstrap = FALSE, cband = FALSE)))
  q <- sum(a$group > a$t & !is.na(a$se))
  v <- list(cell_group = vec(a$group), cell_time = vec(a$t), cell_att = vec(a$att), cell_se = vec(a$se),
            pretrend_w = as.numeric(a$W), pretrend_df = q,
            pretrend_p = pchisq(as.numeric(a$W), q, lower.tail = FALSE), n_units = a$n)
  for (ty in c("group", "simple", "dynamic")) {
    g <- suppressMessages(did::aggte(a, type = ty, bstrap = FALSE, cband = FALSE))
    v[[paste0(ty, "_overall_att")]] <- g$overall.att
    v[[paste0(ty, "_overall_se")]] <- g$overall.se
    if (ty != "simple") {
      v[[paste0(ty, "_key")]] <- vec(g$egt); v[[paste0(ty, "_att")]] <- vec(g$att.egt)
      v[[paste0(ty, "_se")]] <- vec(g$se.egt)
    }
  }
  v
}

.did_t4_prefix <- function(v, p) setNames(v, paste0(p, "_", names(v)))

case("did_cs_varying", "causal/did", "Callaway-Sant'Anna with did 2.5.1 defaults: base_period 'varying' (and 'universal' for comparison), never and not-yet comparisons, aggte group (default overall), simple and dynamic, analytic SEs and the Wald pre-test", function(ctx) {
  d <- ctx$dataset("datasets/generated/cs_panel.csv")
  v <- c(.did_t4_prefix(.did_t4_fit(d, "nevertreated", "varying"), "never_varying"),
         .did_t4_prefix(.did_t4_fit(d, "notyettreated", "varying"), "notyet_varying"),
         .did_t4_prefix(.did_t4_fit(d, "nevertreated", "universal"), "never_universal"),
         .did_t4_prefix(.did_t4_fit(d, "notyettreated", "universal"), "notyet_universal"))
  list(values = v,
       reference = "did 2.5.1 att_gt(est_method = 'reg', base_period = 'varying' | 'universal', bstrap = FALSE, cband = FALSE); aggte(type = 'group' | 'simple' | 'dynamic', bstrap = FALSE, cband = FALSE)",
       call = "att_gt(yname = 'y', tname = 'time', idname = 'unit', gname = 'first_treat', data = cs_panel, control_group = 'nevertreated' | 'notyettreated', base_period = 'varying', bstrap = FALSE); aggte(type = 'group')",
       packages = c("did"))
})

case("did_cs_small_cohorts", "causal/did", "Callaway-Sant'Anna with did's sample rules: four units per cohort (no per-cell minimum; did warns about groups under 5 units), three units treated in the first period (dropped), varying base, analytic SEs", function(ctx) {
  set.seed(20261113)
  N <- 33; TT <- 8
  cohort <- c(rep(3, 4), rep(5, 4), rep(7, 4), rep(1, 3), rep(0, 18))   # 1 = treated in the first period
  unit <- rep(seq_len(N), each = TT); time <- rep(seq_len(TT), N)
  g <- cohort[unit]
  a <- rnorm(N)[unit]
  e <- unlist(lapply(seq_len(N), function(i) as.numeric(arima.sim(list(ar = 0.5), TT))))
  te <- ifelse(g > 0 & time >= g, 2 + 0.5 * (time - g), 0)
  d <- ctx$gen_dataset("cs_small_panel", data.frame(unit = unit, time = time, first_treat = g,
                                                     y = round(1 + a + 0.3 * time + te + e, 10)))
  w <- character(0)
  a1 <- withCallingHandlers(did::att_gt(yname = "y", tname = "time", idname = "unit", gname = "first_treat", data = d,
          control_group = "nevertreated", est_method = "reg", bstrap = FALSE, cband = FALSE),
        warning = function(x) { w <<- c(w, conditionMessage(x)); invokeRestart("muffleWarning") })
  stopifnot(any(grepl("Dropped 3 units that were already treated in the first period", w)),
            any(grepl("Check groups: 3,5,7", w)), a1$n == 30)
  v <- .did_t4_fit(d, "nevertreated", "varying")
  v$n_dropped_first_period <- 3
  list(values = v,
       reference = "did 2.5.1 att_gt(est_method = 'reg', base_period = 'varying', bstrap = FALSE) (drops the 3 units first treated in period 1; warns about groups 3, 5, 7); aggte(type = 'group' | 'simple' | 'dynamic')",
       call = "att_gt(yname = 'y', tname = 'time', idname = 'unit', gname = 'first_treat', data = cs_small_panel, control_group = 'nevertreated', bstrap = FALSE)",
       packages = c("did"))
})

case("bacon_always_treated", "causal/did", "Goodman-Bacon decomposition with always-treated units (first treated in the first period, coded 0) and never-treated units, periods 0-9; the Theorem 1 identity with fixest's TWFE coefficient", function(ctx) {
  suppressPackageStartupMessages({ library(bacondecomp); library(fixest) })
  set.seed(20261113)
  N <- 40; TT <- 10; times <- 0:(TT - 1)
  cohort <- rep(c(NA, 0, 3, 5, 7), each = 8)            # NA never treated, 0 always treated
  unit <- rep(seq_len(N), each = TT); time <- rep(times, N)
  g <- cohort[unit]
  a <- rnorm(N)[unit]; lam <- rnorm(TT)[time + 1]
  D <- as.integer(!is.na(g) & time >= g)
  te <- ifelse(D == 1, 1 + 0.3 * (time - g), 0)
  d <- ctx$gen_dataset("bacon_always_panel", data.frame(unit = unit, time = time, D = D,
                                                         y = round(a + lam + te + rnorm(N * TT, 0, 0.5), 10)))
  b <- bacon(y ~ D, data = d, id_var = "unit", time_var = "time", quietly = TRUE)
  b <- b[order(b$treated, b$untreated), ]
  twfe <- unname(coef(feols(y ~ D | unit + time, data = d))["D"])
  stopifnot(abs(sum(b$weight * b$estimate) - twfe) < 1e-10)
  never <- b$untreated == 99999
  v <- list(component_treated = vec(b$treated), component_control = vec(ifelse(never, -1, b$untreated)),
            component_control_never = vec(as.numeric(never)),
            component_estimate = vec(b$estimate), component_weight = vec(b$weight),
            twfe = twfe, weighted_sum = sum(b$weight * b$estimate))
  for (ty in c("Treated vs Untreated", "Earlier vs Later Treated", "Later vs Earlier Treated", "Later vs Always Treated")) {
    k <- gsub(" ", "_", tolower(ty)); s <- b$type == ty
    v[[paste0("type_weight_", k)]] <- sum(b$weight[s])
    v[[paste0("type_estimate_", k)]] <- sum(b$weight[s] * b$estimate[s]) / sum(b$weight[s])
  }
  d2 <- d; d2$D[d2$unit == 20 & d2$time == 8] <- 0      # a treatment reversal
  msg <- tryCatch({ bacon(y ~ D, data = d2, id_var = "unit", time_var = "time", quietly = TRUE); "" },
                  error = function(e) conditionMessage(e))
  stopifnot(msg == "Treatment not weakly increasing with time")
  list(values = v, reference = "bacondecomp::bacon (Goodman-Bacon 2021, Theorem 1; 'Later vs Always Treated' for units treated from the first period); fixest::feols two-way fixed effects",
       call = "bacon(y ~ D, data = bacon_always_panel, id_var = 'unit', time_var = 'time'); feols(y ~ D | unit + time)",
       packages = c("bacondecomp", "fixest"))
})

case("etwfe_within_r2", "econometrics/did", "ETWFE within R-squared (fixest r2(type = 'wr2')) for the cgroup notyet / never fits, with a time-invariant control, on the balanced and the unbalanced panel", function(ctx) {
  suppressPackageStartupMessages({ library(etwfe); library(fixest) })
  d <- ctx$dataset("datasets/generated/etwfe_panel.csv")
  du <- ctx$dataset("datasets/generated/etwfe_panel_unbalanced.csv")
  f <- function(d, cg = "notyet", ctrl = FALSE)
    etwfe(if (ctrl) y ~ xc else y ~ 0, tvar = "year", gvar = "first_treat", data = d, ivar = "id", cgroup = cg, vcov = ~id)
  fits <- list(notyet = f(d), never = f(d, "never"), notyet_xc = f(d, ctrl = TRUE), unbal = f(du), unbal_xc = f(du, ctrl = TRUE))
  v <- list()
  for (nm in names(fits)) {
    v[[paste0(nm, "_wr2")]] <- unname(r2(fits[[nm]], "wr2")); v[[paste0(nm, "_nobs")]] <- nobs(fits[[nm]])
  }
  list(dataset = "validation/datasets/generated/etwfe_panel.csv", values = v,
       reference = "etwfe::etwfe(ivar = id, vcov = ~id) and fixest::r2(type = 'wr2')",
       call = "r2(etwfe(y ~ 0 | y ~ xc, tvar = year, gvar = first_treat, data = <panel>, ivar = id, cgroup = 'notyet' | 'never'), 'wr2')",
       packages = c("etwfe", "fixest"))
})

case("did_cs_cov_overall", "causal/did", "Callaway-Sant'Anna with covariates x1 + x2 (DRDID doubly robust, outcome regression, IPW), anticipation 0 and 1, never and not-yet comparisons: aggte's default overall (type = 'group') and the simple overall, did 2.5.1 defaults (varying base)", function(ctx) {
  d <- ctx$dataset("datasets/generated/cs_cov_panel.csv")
  v <- list()
  for (control in c("never", "notyet")) for (delta in 0:1) for (m in c("or", "ipw", "dr")) {
    a <- suppressWarnings(suppressMessages(did::att_gt(yname = "y", tname = "time", idname = "unit", gname = "first_treat",
           xformla = ~ x1 + x2, data = d, control_group = if (control == "never") "nevertreated" else "notyettreated",
           anticipation = delta, est_method = c(or = "reg", ipw = "ipw", dr = "dr")[[m]], bstrap = FALSE, cband = FALSE)))
    tag <- sprintf("%s_ant%d_%s", control, delta, m)
    for (ty in c("group", "simple")) {
      g <- suppressMessages(did::aggte(a, type = ty, bstrap = FALSE, cband = FALSE))
      v[[paste0(tag, "_", ty, "_overall_att")]] <- g$overall.att
      v[[paste0(tag, "_", ty, "_overall_se")]] <- g$overall.se
    }
  }
  list(values = v,
       reference = "did 2.5.1 att_gt(xformla = ~ x1 + x2, est_method = 'reg' | 'ipw' | 'dr', anticipation = 0 | 1, bstrap = FALSE) with DRDID; aggte(type = 'group' | 'simple')",
       call = "aggte(att_gt(yname = 'y', tname = 'time', idname = 'unit', gname = 'first_treat', xformla = ~ x1 + x2, data = cs_cov_panel, bstrap = FALSE), type = 'group')",
       packages = c("did", "DRDID"))
})
