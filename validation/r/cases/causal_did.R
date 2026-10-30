# validation/r/cases/causal_did.R -- Callaway-Sant'Anna ATT(g,t) with analytic influence-function
# standard errors and did::aggte-style aggregation (T2-WP12: CAUS-2, CAUS-86).
#
# The R package `did` is not installed on the reference machine. This file implements the
# formulas of did 2.1 (att_gt with est_method "reg" and no covariates, base_period = "universal",
# bstrap = FALSE; aggte type "simple", "group", "dynamic") directly, following the package
# source: DRDID::reg_did_panel influence function, rescaled by n / n_subset; SE = sqrt(mean(IF^2)/n);
# aggregation influence functions include the weight-estimation term (did:::wif).

.cs_panel <- function(ctx) {
  set.seed(20261021)
  N <- 90; TT <- 8
  cohort <- rep(c(3, 5, 7, 0), length.out = N)              # 0 = never treated
  unit <- rep(seq_len(N), each = TT); time <- rep(seq_len(TT), N)
  g <- cohort[unit]
  a <- rnorm(N)[unit]
  e <- unlist(lapply(seq_len(N), function(i) as.numeric(arima.sim(list(ar = 0.5), TT))))
  te <- ifelse(g > 0 & time >= g, 2 + 0.5 * (time - g), 0)
  y <- 1 + a + 0.3 * time + te + e
  ctx$gen_dataset("cs_panel", data.frame(unit = unit, time = time, first_treat = g, y = round(y, 10)))
}

.cs_attgt <- function(d, control = c("never", "notyet")) {
  control <- match.arg(control)
  units <- sort(unique(d$unit)); n <- length(units)
  G <- d$first_treat[match(units, d$unit)]
  Y <- sapply(sort(unique(d$time)), function(t) d$y[d$time == t][order(d$unit[d$time == t])])  # n x T
  periods <- sort(unique(d$time)); groups <- sort(unique(G[G > 0]))
  cells <- list(); inf <- NULL
  for (g in groups) for (t in periods) {
    base <- g - 1
    if (!(base %in% periods)) next
    if (t == base) { att <- 0; IF <- rep(0, n) } else {
      ctrl <- if (control == "never") G == 0 else (G == 0 | (G > max(t, base) & G != g))   # did: & G != glist[g]
      sub <- (G == g) | ctrl
      D <- as.numeric(G[sub] == g)
      dY <- Y[sub, which(periods == t)] - Y[sub, which(periods == base)]
      p <- mean(D); mu1 <- mean(dY[D == 1]); mu0 <- mean(dY[D == 0])
      att <- mu1 - mu0
      ifs <- D * (dY - mu1) / p - (1 - D) * (dY - mu0) / (1 - p)
      IF <- rep(0, n); IF[sub] <- (n / sum(sub)) * ifs
    }
    cells[[length(cells) + 1]] <- data.frame(group = g, time = t, att = att)
    inf <- cbind(inf, IF)
  }
  cells <- do.call(rbind, cells)
  cells$se <- sqrt(colMeans(inf^2) / n)
  list(cells = cells, inf = inf, G = G, n = n)
}

.wif <- function(keepers, pg, G, group) {
  if1 <- sapply(keepers, function(k) (1 * (G == group[k]) - pg[k]) / sum(pg[keepers]))
  if2 <- rowSums(sapply(keepers, function(k) 1 * (G == group[k]) - pg[k])) %*% t(pg[keepers] / (sum(pg[keepers])^2))
  if1 - if2
}
.agg_se <- function(IF, n) sqrt(mean(IF^2) / n)

.cs_aggregate <- function(fit) {
  cl <- fit$cells; inf <- fit$inf; G <- fit$G; n <- fit$n
  glist <- sort(unique(cl$group))
  pg_g <- sapply(glist, function(g) mean(G == g))
  pg <- pg_g[match(cl$group, glist)]
  # simple
  keep <- which(cl$group <= cl$time)
  w <- pg[keep] / sum(pg[keep])
  simple_att <- sum(cl$att[keep] * w)
  simple_if <- inf[, keep] %*% w + .wif(keep, pg, G, cl$group) %*% as.matrix(cl$att[keep])
  # group (selective)
  att_g <- se_g <- numeric(length(glist)); IFg <- matrix(0, n, length(glist))
  for (i in seq_along(glist)) {
    wg <- which(cl$group == glist[i] & cl$group <= cl$time)
    att_g[i] <- mean(cl$att[wg]); IFg[, i] <- inf[, wg, drop = FALSE] %*% rep(1 / length(wg), length(wg))
    se_g[i] <- .agg_se(IFg[, i], n)
  }
  sel_att <- sum(att_g * pg_g) / sum(pg_g)
  wif_g <- .wif(seq_along(glist), pg_g, G, glist)
  sel_if <- IFg %*% (pg_g / sum(pg_g)) + wif_g %*% as.matrix(att_g)
  # dynamic (event study)
  e_all <- sort(unique(cl$time - cl$group))
  att_e <- se_e <- numeric(length(e_all))
  for (i in seq_along(e_all)) {
    we <- which(cl$time - cl$group == e_all[i])
    ww <- pg[we] / sum(pg[we])
    att_e[i] <- sum(cl$att[we] * ww)
    IFe <- inf[, we, drop = FALSE] %*% ww + .wif(we, pg, G, cl$group) %*% as.matrix(cl$att[we])
    se_e[i] <- .agg_se(IFe, n)
  }
  # pre-trend Wald on the pre-period cells (structural zeros at t = g - 1 dropped)
  pre <- which(cl$time < cl$group & cl$time != cl$group - 1)
  V <- crossprod(inf[, pre]) / n
  W <- n * drop(t(cl$att[pre]) %*% solve(V) %*% cl$att[pre])
  diag_W <- sum((cl$att[pre] / cl$se[pre])^2)
  list(simple_att = simple_att, simple_se = .agg_se(simple_if, n),
       group_att = att_g, group_se = se_g, selective_att = sel_att, selective_se = .agg_se(sel_if, n),
       event_e = e_all, event_att = att_e, event_se = se_e,
       pretrend_w = W, pretrend_df = length(pre), pretrend_p = pchisq(W, length(pre), lower.tail = FALSE),
       pretrend_diag_w = diag_W, pretrend_diag_p = pchisq(diag_W, length(pre), lower.tail = FALSE))
}

case("did_cs_analytic", "causal/did", "Callaway-Sant'Anna ATT(g,t), analytic IF SEs, simple/group/dynamic aggregation and pre-trend Wald", function(ctx) {
  d <- .cs_panel(ctx)
  v <- list()
  for (ctl in c("never", "notyet")) {
    fit <- .cs_attgt(d, ctl); ag <- .cs_aggregate(fit)
    cl <- fit$cells
    v[[paste0(ctl, "_cell_group")]] <- vec(cl$group); v[[paste0(ctl, "_cell_time")]] <- vec(cl$time)
    v[[paste0(ctl, "_cell_att")]] <- vec(cl$att); v[[paste0(ctl, "_cell_se")]] <- vec(cl$se)
    for (k in names(ag)) v[[paste0(ctl, "_", k)]] <- if (length(ag[[k]]) > 1) vec(ag[[k]]) else ag[[k]]
  }
  # the independence formula p2a used before the fix, for the record (simple aggregation)
  fit <- .cs_attgt(d, "never"); cl <- fit$cells; keep <- cl$group <= cl$time
  w <- table(fit$G)[as.character(cl$group[keep])]; w <- as.numeric(w) / sum(as.numeric(w))
  v$never_simple_se_independent <- sqrt(sum(w^2 * cl$se[keep]^2))
  list(values = v,
       reference = "did 2.1 formulas (att_gt est_method 'reg' without covariates, base_period 'universal', bstrap = FALSE; aggte 'simple' | 'group' | 'dynamic'), implemented in this file",
       call = "att_gt(yname = 'y', tname = 'time', idname = 'unit', gname = 'first_treat', control_group = 'nevertreated' | 'notyettreated', base_period = 'universal', bstrap = FALSE)",
       packages = character(0))
})
