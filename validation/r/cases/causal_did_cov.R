# validation/r/cases/causal_did_cov.R -- Callaway-Sant'Anna with covariates and anticipation, the
# Goodman-Bacon decomposition against the TWFE coefficient, and the 2x2 DiD regression with HC1 and
# cluster-robust standard errors (T3-WP21: CAUS-34, CAUS-83, CAUS-54, CAUS-102; moved in CAUS-35,
# CAUS-36, CAUS-37, CAUS-109, CAUS-10). Package copy of design Appendix A.17; canonical.
#
# This file writes out the DRDID 1.3.0 panel estimators (reg_did_panel, std_ipw_did_panel,
# drdid_panel; source read from the CRAN tarballs) for one (g, t) cell, as did 2.5.1's att_gt
# calls them; that transcription is the tight reference. did 2.5.1 (with DRDID 1.4.0) is then run
# on the same blocks as a cross-check (`did_*` keys; every value within 1e-6 relative):
#  * covariates: model.matrix(xformla) of the EARLIER of the two periods (min(t, base));
#  * OR  (reg_did_panel): dY ~ [1, X] by OLS on the comparison units; ATT = mean_{D=1}(dY - X b);
#    influence function including the OLS estimation term;
#  * propensity score: logit MLE (DRDID's fastglm; here glm.fit at epsilon 1e-14, the same MLE),
#    ps = pmin(ps, 1 - 1e-6), comparison units with ps >= 0.995 trimmed (trim.level), Hessian
#    n (X' diag(ps (1 - ps)) X)^-1;
#  * IPW (std_ipw_did_panel): normalised weights; influence function with the propensity term;
#  * DR  (drdid_panel): IPW-weighted OR residuals, weights normalised by their sums; influence
#    function with the OLS and propensity terms.
# did's per-cell guards are asserted not to fire (overlap: max ps < 0.999; control design full rank).
# Anticipation delta (did's `anticipation`): base period g - 1 - delta (universal), comparison
# units G == 0 or (not-yet) G > max(t, base) + delta and G != g, post cells t >= g (did's
# post = 1(g <= t); aggte's simple/group aggregations use t >= g), pre-test cells t < g except the
# base (did drops zero-variance cells), so the anticipation cells are pre-test cells as in did.
# Aggregation and SEs: T2-WP12's .cs_aggregate (causal_did.R) for delta = 0; .dc_aggregate below
# (the same code with the base g - 1 - delta), checked equal to .cs_aggregate at delta = 0.

.dc_panel <- function(ctx) {
  set.seed(20261129)
  N <- 150; TT <- 8
  x1 <- rnorm(N); x2 <- rbinom(N, 1, 0.5)
  pg <- cbind(plogis(-1 + 0.6 * x1), plogis(-1 - 0.3 * x1 + 0.5 * x2), plogis(-1.2 + 0.4 * x2))
  g <- sapply(seq_len(N), function(i) { r <- runif(1); cp <- cumsum(c(pg[i, ] / (1 + sum(pg[i, ])))); c(3, 5, 7, 0)[findInterval(r, c(cp, 1)) + 1] })
  unit <- rep(seq_len(N), each = TT); time <- rep(seq_len(TT), N); G <- g[unit]
  a <- rnorm(N)[unit]
  te <- ifelse(G > 0 & time >= G - 1, 1 + 0.5 * pmax(time - G, 0), 0)           # one period of anticipation
  y <- 1 + a + 0.3 * time + 0.4 * x1[unit] * time + 0.3 * x2[unit] * (time > 4) + te + rnorm(N * TT)
  # Package additions, drawn after y so x1, x2 and y are those of design A.17:
  # x3 varies over time and is shifted by treatment (CAUS-35: did uses the earlier period's value);
  # D is the absorbing treatment indicator for the Bacon decomposition.
  x3 <- x1[unit] + 0.3 * time + 0.8 * (G > 0 & time >= G) + rnorm(N * TT, sd = 0.5)
  D <- as.numeric(G > 0 & time >= G)
  ctx$gen_dataset("cs_cov_panel", data.frame(unit = unit, time = time, first_treat = G, x1 = round(x1[unit], 10), x2 = x2[unit], y = round(y, 10),
                                             x3 = round(x3, 10), D = D))
}

# One cell, DRDID 1.3.0 formulas; X includes the intercept column; i.weights all 1.
.drdid_cell <- function(dY, D, X) {
  n <- length(D); iw <- rep(1, n)
  ctrl <- D == 0
  stopifnot(rcond(crossprod(X[ctrl, , drop = FALSE])) >= .Machine$double.eps)   # did's regression guard
  # OR (reg_did_panel)
  b <- qr.coef(qr(X[ctrl, , drop = FALSE]), dY[ctrl])
  od <- drop(X %*% b)
  w_t <- iw * D
  att_or <- mean(w_t * dY) / mean(w_t) - mean(w_t * od) / mean(w_t)
  wols_x <- iw * (1 - D) * X; wols_eX <- iw * (1 - D) * (dY - od) * X
  alr_ols <- wols_eX %*% solve(crossprod(wols_x, X) / n)
  inf_t <- (w_t * dY - w_t * mean(w_t * dY) / mean(w_t)) / mean(w_t)
  inf_c1 <- w_t * od - w_t * mean(w_t * od) / mean(w_t)
  if_or <- drop(inf_t - (inf_c1 + alr_ols %*% colMeans(w_t * X)) / mean(w_t))
  # propensity score
  pf <- suppressWarnings(glm.fit(X, D, family = binomial(), control = glm.control(epsilon = 1e-14, maxit = 100)))
  stopifnot(pf$converged, max(pf$fitted.values) < 0.999)                             # did's overlap guard
  ps <- pmin(pf$fitted.values, 1 - 1e-6)
  trim <- rep(TRUE, n); trim[D == 0] <- ps[D == 0] < 0.995
  H <- chol2inv(chol(crossprod(X, (ps * (1 - ps) * iw) * X))) * n
  alr_ps <- (iw * (D - ps) * X) %*% H
  w_tt <- trim * iw * D
  w_c <- trim * iw * ps * (1 - D) / (1 - ps)
  # IPW (std_ipw_did_panel)
  et <- mean(w_tt * dY) / mean(w_tt); ec <- mean(w_c * dY) / mean(w_c)
  att_ipw <- et - ec
  if_ipw <- drop((w_tt * dY - w_tt * et) / mean(w_tt) -
                 ((w_c * dY - w_c * ec) + alr_ps %*% colMeans(w_c * (dY - ec) * X)) / mean(w_c))
  # DR (drdid_panel)
  r <- dY - od
  et <- mean(w_tt * r) / mean(w_tt); ec <- mean(w_c * r) / mean(w_c)
  att_dr <- et - ec
  inf_treat <- ((w_tt * r - w_tt * et) - alr_ols %*% colMeans(w_tt * X)) / mean(w_tt)
  inf_cont <- ((w_c * r - w_c * ec) + alr_ps %*% colMeans(w_c * (r - ec) * X) - alr_ols %*% colMeans(w_c * X)) / mean(w_c)
  list(or = att_or, ipw = att_ipw, dr = att_dr, if_or = if_or, if_ipw = if_ipw, if_dr = drop(inf_treat - inf_cont),
       n_trim = sum(!trim), max_ps = max(pf$fitted.values))
}

# All ATT(g, t) cells. `xrow`: "earlier" (did: min(t, base)) or "period_t" (the pre-fix p2a rule, for the record).
.dc_attgt <- function(d, control = "never", delta = 0, covs = c("x1", "x2"), xrow = "earlier") {
  units <- sort(unique(d$unit)); n <- length(units)
  G <- d$first_treat[match(units, d$unit)]
  periods <- sort(unique(d$time))
  Y <- sapply(periods, function(t) d$y[d$time == t][order(d$unit[d$time == t])])
  Xp <- lapply(periods, function(t) { s <- d[d$time == t, ]; cbind(1, as.matrix(s[order(s$unit), covs, drop = FALSE])) })
  meth <- c("or", "ipw", "dr")
  cells <- list(); inf <- setNames(lapply(meth, function(m) NULL), meth); n_trim <- 0; max_ps <- 0
  for (g in sort(unique(G[G > 0]))) for (t in periods) {
    base <- g - 1 - delta
    if (!(base %in% periods)) next
    ctrl <- if (control == "never") G == 0 else (G == 0 | (G > max(t, base) + delta & G != g))
    sub <- (G == g) | ctrl
    if (t == base) { est <- list(or = 0, ipw = 0, dr = 0); IF <- lapply(meth, function(m) rep(0, n)) } else {
      D <- as.numeric(G[sub] == g); dY <- Y[sub, which(periods == t)] - Y[sub, which(periods == base)]
      xp <- if (xrow == "earlier") min(t, base) else t
      est <- .drdid_cell(dY, D, Xp[[which(periods == xp)]][sub, , drop = FALSE])
      n_trim <- n_trim + est$n_trim; max_ps <- max(max_ps, est$max_ps)
      IF <- lapply(meth, function(m) { v <- rep(0, n); v[sub] <- (n / sum(sub)) * est[[paste0("if_", m)]]; v })
    }
    names(IF) <- meth
    cells[[length(cells) + 1]] <- data.frame(group = g, time = t, att_or = est$or, att_ipw = est$ipw, att_dr = est$dr,
                                             n_treated = sum(G[sub] == g), n_comparison = sum(ctrl))
    for (m in meth) inf[[m]] <- cbind(inf[[m]], IF[[m]])
  }
  cells <- do.call(rbind, cells)
  for (m in meth) cells[[paste0("se_", m)]] <- sqrt(colMeans(inf[[m]]^2) / n)
  for (m in meth) cells[[paste0("se_", m)]][cells$time == cells$group - 1 - delta] <- NA   # did: NA for the reference cells (CAUS-117, Task 1a; sync)
  list(cells = cells, inf = inf, G = G, n = n, n_trim = n_trim, max_ps = max_ps)
}

# T2-WP12's .cs_aggregate with the universal base g - 1 - delta (identical at delta = 0, checked below).
.dc_aggregate <- function(fit, delta) {
  cl <- fit$cells; inf <- fit$inf; G <- fit$G; n <- fit$n
  glist <- sort(unique(cl$group)); pg_g <- sapply(glist, function(g) mean(G == g)); pg <- pg_g[match(cl$group, glist)]
  keep <- which(cl$group <= cl$time); w <- pg[keep] / sum(pg[keep])
  simple_att <- sum(cl$att[keep] * w)
  simple_if <- inf[, keep] %*% w + .wif(keep, pg, G, cl$group) %*% as.matrix(cl$att[keep])
  att_g <- se_g <- numeric(length(glist))
  for (i in seq_along(glist)) {
    wg <- which(cl$group == glist[i] & cl$group <= cl$time)
    att_g[i] <- mean(cl$att[wg]); se_g[i] <- .agg_se(inf[, wg, drop = FALSE] %*% rep(1 / length(wg), length(wg)), n)
  }
  e_all <- sort(unique(cl$time - cl$group)); att_e <- se_e <- numeric(length(e_all))
  for (i in seq_along(e_all)) {
    we <- which(cl$time - cl$group == e_all[i]); ww <- pg[we] / sum(pg[we])
    att_e[i] <- sum(cl$att[we] * ww)
    se_e[i] <- .agg_se(inf[, we, drop = FALSE] %*% ww + .wif(we, pg, G, cl$group) %*% as.matrix(cl$att[we]), n)
  }
  se_e[se_e <= sqrt(.Machine$double.eps) * 10] <- NA          # did::aggte: NA at the reference event time (CAUS-117)
  pre <- which(cl$time < cl$group & cl$time != cl$group - 1 - delta)
  V <- crossprod(inf[, pre]) / n
  W <- n * drop(t(cl$att[pre]) %*% solve(V) %*% cl$att[pre])
  list(simple_att = simple_att, simple_se = .agg_se(simple_if, n), group_att = att_g, group_se = se_g,
       event_e = e_all, event_att = att_e, event_se = se_e,
       pretrend_w = W, pretrend_df = length(pre), pretrend_p = pchisq(W, length(pre), lower.tail = FALSE))
}

.dc_method_fit <- function(fit, m) {
  cl <- fit$cells
  list(cells = data.frame(group = cl$group, time = cl$time, att = cl[[paste0("att_", m)]], se = cl[[paste0("se_", m)]]),
       inf = fit$inf[[m]], G = fit$G, n = fit$n)
}

# did 2.5.1 itself on one block: att_gt cells (in the transcription's cell order) and the simple,
# group and dynamic aggregates of aggte, all with analytic (bstrap = FALSE) standard errors.
.dc_did <- function(d, control, delta, m, xformla) {
  est <- c(or = "reg", ipw = "ipw", dr = "dr")[[m]]
  a <- suppressMessages(did::att_gt(yname = "y", tname = "time", idname = "unit", gname = "first_treat", xformla = xformla,
                                    data = d, control_group = if (control == "never") "nevertreated" else "notyettreated",
                                    anticipation = delta, est_method = est, base_period = "universal", bstrap = FALSE, cband = FALSE))
  ag <- function(type) suppressMessages(did::aggte(a, type = type, bstrap = FALSE, cband = FALSE))
  s <- ag("simple"); g <- ag("group"); e <- ag("dynamic")
  list(group = a$group, time = a$t, att = a$att, se = a$se, simple_att = s$overall.att, simple_se = s$overall.se,
       group_att = g$att.egt, group_se = g$se.egt, event_e = e$egt, event_att = e$att.egt, event_se = e$se.egt)
}
# Largest relative difference between two vectors (NA only where both are NA).
.dc_maxrel <- function(x, y) {
  stopifnot(length(x) == length(y), identical(is.na(x), is.na(y)))
  ok <- !is.na(x)
  if (!any(ok)) return(0)
  max(abs(x[ok] - y[ok]) / pmax(abs(y[ok]), 1e-300))
}
# Record did's values for one block as did_<tag>_*_<m> and check them against the transcription.
.dc_did_keys <- function(v, d, tag, control, delta, m, xformla, cells, ag) {
  w <- .dc_did(d, control, delta, m, xformla)
  i <- match(paste(cells$group, cells$time), paste(w$group, w$time))
  stopifnot(!anyNA(i), length(i) == length(w$att))
  p <- paste0("did_", tag)
  v[[paste0(p, "_cell_att_", m)]] <- vec(w$att[i]); v[[paste0(p, "_cell_se_", m)]] <- vec(w$se[i])
  rel <- c(.dc_maxrel(w$att[i], cells[[paste0("att_", m)]]), .dc_maxrel(w$se[i], cells[[paste0("se_", m)]]))
  if (!is.null(ag)) {
    stopifnot(identical(as.numeric(w$event_e), as.numeric(ag$event_e)))
    for (k in c("simple_att", "simple_se", "group_att", "group_se", "event_att", "event_se")) {
      v[[paste0(p, "_", sub("_(att|se)$", "", k), "_", sub("^.*_", "", k), "_", m)]] <- vec(w[[k]])
      rel <- c(rel, .dc_maxrel(as.numeric(w[[k]]), as.numeric(ag[[k]])))
    }
  }
  v[[paste0(p, "_maxrel_", m)]] <- max(rel)
  stopifnot(max(rel) < 1e-6)   # else did/DRDID changed a panel estimator or the transcription is wrong
  v
}

case("did_cs_covariates", "causal/did", "Callaway-Sant'Anna with covariates (OR, IPW, DR as DRDID: cells, influence-function SEs, aggregates, pre-test), anticipation 0 and 1, never and not-yet comparisons, time-varying covariate from the earlier period", function(ctx) {
  d <- .dc_panel(ctx)
  v <- list()
  for (ctl in c("never", "notyet")) for (delta in c(0, 1)) {
    fit <- .dc_attgt(d, ctl, delta); cl <- fit$cells
    stopifnot(fit$n_trim == 0, fit$max_ps < 0.995)         # trim.level and the overlap guard are inert here
    tag <- paste0(ctl, "_ant", delta)
    v[[paste0(tag, "_cell_group")]] <- vec(cl$group); v[[paste0(tag, "_cell_time")]] <- vec(cl$time)
    v[[paste0(tag, "_cell_n_treated")]] <- vec(cl$n_treated); v[[paste0(tag, "_cell_n_comparison")]] <- vec(cl$n_comparison)
    for (m in c("or", "ipw", "dr")) {
      mf <- .dc_method_fit(fit, m)
      v[[paste0(tag, "_cell_att_", m)]] <- vec(mf$cells$att); v[[paste0(tag, "_cell_se_", m)]] <- vec(mf$cells$se)
      ag <- .dc_aggregate(mf, delta)
      if (delta == 0) {
        ref <- .cs_aggregate(mf)                             # T2-WP12's aggregation, unchanged
        for (k in names(ag)) stopifnot(isTRUE(all.equal(ag[[k]], ref[[k]], tolerance = 1e-13)))
      }
      v[[paste0(tag, "_simple_att_", m)]] <- ag$simple_att; v[[paste0(tag, "_simple_se_", m)]] <- ag$simple_se
      v[[paste0(tag, "_group_att_", m)]] <- vec(ag$group_att); v[[paste0(tag, "_group_se_", m)]] <- vec(ag$group_se)
      v[[paste0(tag, "_event_att_", m)]] <- vec(ag$event_att); v[[paste0(tag, "_event_se_", m)]] <- vec(ag$event_se)
      v[[paste0(tag, "_pretrend_w_", m)]] <- ag$pretrend_w; v[[paste0(tag, "_pretrend_df_", m)]] <- ag$pretrend_df
      v[[paste0(tag, "_pretrend_p_", m)]] <- ag$pretrend_p
      if (m == "or") v[[paste0(tag, "_event_e")]] <- vec(ag$event_e)
      v <- .dc_did_keys(v, d, tag, ctl, delta, m, ~ x1 + x2, cl, ag)        # did 2.5.1 cross-check
    }
  }
  # Time-varying covariate x3 (CAUS-35): covariates from min(t, base) as did; the period-t rule of the
  # pre-fix code is recorded, not compared.
  fit <- .dc_attgt(d, "never", 0, covs = c("x1", "x3")); cl <- fit$cells
  stopifnot(fit$n_trim == 0, fit$max_ps < 0.995)
  for (m in c("or", "ipw", "dr")) {
    v[[paste0("tv_never_ant0_cell_att_", m)]] <- vec(cl[[paste0("att_", m)]]); v[[paste0("tv_never_ant0_cell_se_", m)]] <- vec(cl[[paste0("se_", m)]])
    v <- .dc_did_keys(v, d, "tv_never_ant0", "never", 0, m, ~ x1 + x3, cl, NULL)  # did 2.5.1 cross-check (cells)
  }
  pt <- .dc_attgt(d, "never", 0, covs = c("x1", "x3"), xrow = "period_t")$cells
  for (m in c("or", "ipw", "dr")) v[[paste0("tv_period_t_cell_att_", m)]] <- vec(pt[[paste0("att_", m)]])
  # Without covariates the three estimators coincide with T2's unconditional estimator.
  u <- .dc_attgt(d, "never", 0, covs = character(0))$cells
  stopifnot(isTRUE(all.equal(u$att_or, u$att_ipw, tolerance = 1e-12)), isTRUE(all.equal(u$att_or, u$att_dr, tolerance = 1e-12)),
            isTRUE(all.equal(u$se_or, u$se_dr, tolerance = 1e-10)))
  v$nocov_never_ant0_cell_att <- vec(u$att_or); v$nocov_never_ant0_cell_se <- vec(u$se_or)
  list(values = v,
       reference = "did 2.5.1 att_gt cell estimators (DRDID 1.3.0 reg_did_panel, std_ipw_did_panel, drdid_panel) with covariates from the earlier period and anticipation, written out; aggregation as T2-WP12's causal_did.R; cross-checked against did 2.5.1 att_gt/aggte with DRDID 1.4.0 (did_* keys)",
       call = "att_gt(yname = 'y', tname = 'time', idname = 'unit', gname = 'first_treat', xformla = ~ x1 + x2 | ~ x1 + x3, est_method = 'reg' | 'ipw' | 'dr', control_group = 'nevertreated' | 'notyettreated', anticipation = 0 | 1, base_period = 'universal', bstrap = FALSE); aggte(type = 'simple' | 'group' | 'dynamic', bstrap = FALSE)",
       packages = c("did", "DRDID"))
})

case("bacon_twfe", "causal/did", "Goodman-Bacon decomposition (2x2 estimates, Theorem 1 weights) and the TWFE coefficient it decomposes, balanced staggered panel with never-treated units", function(ctx) {
  suppressPackageStartupMessages({ library(bacondecomp); library(fixest) })
  d <- .dc_panel(ctx)
  b <- bacon(y ~ D, data = d, id_var = "unit", time_var = "time", quietly = TRUE)
  b$untreated[b$untreated == 99999] <- 0                     # p2a codes never treated as 0
  b <- b[order(b$treated, b$untreated), ]
  twfe <- unname(coef(feols(y ~ D | unit + time, data = d))["D"])
  stopifnot(abs(sum(b$weight * b$estimate) - twfe) < 1e-10)  # Theorem 1 identity on this panel
  v <- list(component_treated = vec(b$treated), component_control = vec(b$untreated),
            component_estimate = vec(b$estimate), component_weight = vec(b$weight),
            twfe = twfe, weighted_sum = sum(b$weight * b$estimate))
  for (ty in c("Treated vs Untreated", "Earlier vs Later Treated", "Later vs Earlier Treated")) {
    k <- gsub(" ", "_", tolower(ty)); s <- b$type == ty
    v[[paste0("type_weight_", k)]] <- sum(b$weight[s]); v[[paste0("type_estimate_", k)]] <- sum(b$weight[s] * b$estimate[s]) / sum(b$weight[s])
  }
  list(values = v, reference = "bacondecomp::bacon (Goodman-Bacon 2021, Theorem 1); fixest::feols two-way fixed effects",
       call = "bacon(y ~ D, data = d, id_var = 'unit', time_var = 'time'); feols(y ~ D | unit + time, data = d)",
       packages = c("bacondecomp", "fixest"))
})

case("did_2x2_training", "causal/did", "2x2 difference-in-differences regression, HC1 and cluster-robust (CR1) standard errors, employee_training (the paper's diff_in_diff example)", function(ctx) {
  suppressPackageStartupMessages({ library(sandwich); library(lmtest) })
  d <- ctx$dataset("datasets/employee_training.csv")
  v <- list()
  specs <- list(base = character(0), tenure = "tenure", aliased = c("tenure", "years_since_hire"))
  for (s in names(specs)) {
    rhs <- paste(c("training_group", "post_training", "training_group:post_training", specs[[s]]), collapse = " + ")
    m <- lm(as.formula(paste("performance ~", rhs)), data = d)
    cf <- coef(m)
    ok <- !is.na(cf)
    hc1 <- sqrt(diag(vcovHC(m, type = "HC1"))); cl <- sqrt(diag(vcovCL(m, cluster = ~employee_id, type = "HC1")))
    se_hc1 <- se_cl <- setNames(rep(NA_real_, length(cf)), names(cf)); se_hc1[ok] <- hc1; se_cl[ok] <- cl
    tcl <- cf / se_cl; dfr <- df.residual(m)
    v[[paste0(s, "_coef")]] <- named(cf); v[[paste0(s, "_se_hc1")]] <- named(se_hc1); v[[paste0(s, "_se_cl")]] <- named(se_cl)
    v[[paste0(s, "_t_hc1")]] <- named(cf / se_hc1); v[[paste0(s, "_p_hc1")]] <- named(2 * pt(-abs(cf / se_hc1), dfr))
    v[[paste0(s, "_t_cl")]] <- named(tcl); v[[paste0(s, "_p_cl")]] <- named(2 * pt(-abs(tcl), dfr))
    v[[paste0(s, "_df_resid")]] <- dfr; v[[paste0(s, "_rank")]] <- m$rank
    v[[paste0(s, "_r2")]] <- summary(m)$r.squared; v[[paste0(s, "_adj_r2")]] <- summary(m)$adj.r.squared
  }
  gm <- with(d, tapply(performance, list(training_group, post_training), mean))
  v$group_means <- vec(c(gm["0", "0"], gm["0", "1"], gm["1", "0"], gm["1", "1"]))  # control pre, control post, treated pre, treated post
  v$n_clusters <- length(unique(d$employee_id))
  list(values = v, reference = "stats::lm; sandwich::vcovHC(type = 'HC1'), sandwich::vcovCL(cluster = ~employee_id, type = 'HC1'); t-tests on df.residual",
       call = "lm(performance ~ training_group + post_training + training_group:post_training [+ tenure [+ years_since_hire]], data = employee_training)",
       packages = c("sandwich", "lmtest"))
})
