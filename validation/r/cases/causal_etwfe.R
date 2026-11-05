# validation/r/cases/causal_etwfe.R -- Extended TWFE references (T3-WP22: CAUS-12; moved in CAUS-13,
# CAUS-14, CAUS-15, CAUS-61, CAUS-62). Package copy (canonical; supersedes design Appendix A.18).
#
# Model: etwfe::etwfe(fml, tvar = year, gvar = first_treat, ivar = id, cgroup, tref, gref,
# vcov = ~id | "iid"), i.e. fixest::feols with unit and period fixed effects (p2a's run_etwfe absorbs
# unit effects; etwfe's default ivar = NULL absorbs cohort effects instead, which gives the same cells
# on a balanced panel but not on an unbalanced one). fixest 0.14 defaults to iid SEs, so the clustered
# vcov is requested explicitly (CR1: G/(G-1) * (n-1)/(n-K), K = coefficients + periods, t df G-1).
# Aggregates: etwfe::emfx(type = 'simple' | 'event' | 'group' | 'calendar') (marginaleffects, numerical
# Jacobian) and, as the tight reference, the same averages with the exact delta-method gradient
# written out in .etwfe_delta() (mean over emfx's rows of the per-row effect gradient; SE sqrt(g'Vg)).
# Anticipation (not an etwfe option) is pinned as etwfe on the onset first_treat - 1 (did's rule).
suppressPackageStartupMessages({ library(etwfe); library(fixest) })

.etwfe_panel <- function(ctx, name, balanced = TRUE, all_treated = FALSE) {
  set.seed(20261103)
  N <- 120; TT <- 8
  cohorts <- if (all_treated) c(4, 6, 8) else c(0, 4, 6, 8)        # 0 = never treated (etwfe convention)
  g <- sample(cohorts, N, replace = TRUE)
  a <- rnorm(N)
  d <- expand.grid(year = 1:TT, id = 1:N)[, c("id", "year")]
  d$first_treat <- g[d$id]
  d$xc <- round(rnorm(N)[d$id], 10)                                  # time-invariant control
  te <- with(d, ifelse(first_treat > 0 & year >= first_treat, 1 + 0.4 * (year - first_treat) + 0.3 * (first_treat == 6), 0))
  d$y <- round(a[d$id] + 0.2 * d$year + te + 0.5 * d$xc * (d$year > 4) + rnorm(nrow(d)), 10)
  d$treat <- as.integer(d$first_treat > 0 & d$year >= d$first_treat)  # run_etwfe's treat_col (checked, absorbing)
  if (!balanced) d <- d[!((d$id %% 7 == 0 & d$year == 3) | (d$id %% 11 == 0 & d$year <= 2)), ]
  ctx$gen_dataset(name, d)
}

.etwfe_fit <- function(d, cgroup = "notyet", ctrl = FALSE, vcov = ~id, gvar = "first_treat", tref = NULL, gref = NULL) {
  fml <- if (ctrl) y ~ xc else y ~ 0
  args <- list(fml = fml, tvar = "year", gvar = gvar, data = d, ivar = "id", cgroup = cgroup, vcov = vcov)
  if (!is.null(tref)) args$tref <- tref
  if (!is.null(gref)) args$gref <- gref
  do.call(etwfe, args)
}

# Rows emfx averages over (etwfe 0.6.2 emfx with post_only = TRUE, compress = FALSE).
.etwfe_emfx_rows <- function(dat, a, type) {
  if (a$cgroup == "never") {
    dat <- dat[dat[[a$gvar]] != a$gref, , drop = FALSE]
    if (type != "event") dat <- dat[dat$.Dtreat & dat[[a$tvar]] >= dat[[a$gvar]], , drop = FALSE]
  } else dat <- dat[!is.na(dat$.Dtreat) & dat$.Dtreat, , drop = FALSE]
  dat
}

# Exact delta method: per-row effect = cell coefficient + sum_x cell:x_dm coefficient * x_dm (0 when the
# row's cell has no coefficient: the never-treated reference period g - 1, or a cell at tref).
.etwfe_delta <- function(m, type) {
  a <- attr(m, "etwfe"); dat <- eval(m$call$data, m$call_env)
  rows <- .etwfe_emfx_rows(dat, a, type)
  b <- coef(m); V <- vcov(m)
  Gm <- matrix(0, nrow(rows), length(b), dimnames = list(NULL, names(b)))
  cell <- paste0(".Dtreat:", a$gvar, "::", rows[[a$gvar]], ":", a$tvar, "::", rows[[a$tvar]])
  dm <- grep("_dm$", names(rows), value = TRUE)
  for (i in seq_len(nrow(rows))) if (cell[i] %in% names(b)) {
    Gm[i, cell[i]] <- 1
    for (x in dm) { nm <- paste0(cell[i], ":", x); if (nm %in% names(b)) Gm[i, nm] <- rows[[x]][i] }
  }
  key <- switch(type, simple = rep(1, nrow(rows)), event = rows[[a$tvar]] - rows[[a$gvar]],
                group = rows[[a$gvar]], calendar = rows[[a$tvar]])
  ks <- sort(unique(key))
  gr <- t(vapply(ks, function(k) colMeans(Gm[key == k, , drop = FALSE]), numeric(length(b))))
  se <- sqrt(rowSums((gr %*% V) * gr)); se[rowSums(abs(gr)) == 0] <- NA
  est <- as.numeric(gr %*% b)
  list(key = ks, est = est, se = as.numeric(se), p = 2 * pnorm(-abs(est / se)),
       n = as.numeric(table(key)[as.character(ks)]))
}

.etwfe_values <- function(m, prefix, types = c("simple", "event", "group", "calendar"), cells = FALSE) {
  out <- list()
  b <- coef(m)
  out[[paste0(prefix, "_coef")]] <- named(b)
  out[[paste0(prefix, "_se")]] <- named(se(m))
  out[[paste0(prefix, "_pvalue")]] <- named(pvalue(m))
  if (cells) {                                   # cells as (cohort, time) vectors (anticipation case)
    nm <- grep("^\\.Dtreat:[^:]+::[0-9]+:year::[0-9]+$", names(b), value = TRUE)
    out[[paste0(prefix, "_cell_onset")]] <- vec(as.numeric(sub("^\\.Dtreat:[^:]+::([0-9]+):.*$", "\\1", nm)))
    out[[paste0(prefix, "_cell_time")]] <- vec(as.numeric(sub("^.*:year::([0-9]+)$", "\\1", nm)))
    out[[paste0(prefix, "_cell_coef")]] <- vec(b[nm]); out[[paste0(prefix, "_cell_se")]] <- vec(se(m)[nm])
  }
  for (ty in types) {
    e <- suppressWarnings(emfx(m, type = ty))
    x <- .etwfe_delta(m, ty)
    stopifnot(max(abs(x$est - e$estimate)) < 1e-12)
    out[[paste0(prefix, "_emfx_", ty, "_estimate")]] <- vec(e$estimate)
    out[[paste0(prefix, "_emfx_", ty, "_se")]] <- vec(e$std.error)
    out[[paste0(prefix, "_delta_", ty, "_se")]] <- vec(x$se)
    out[[paste0(prefix, "_delta_", ty, "_p")]] <- vec(x$p)
    out[[paste0(prefix, "_delta_", ty, "_n")]] <- vec(x$n)
    if (ty == "simple") for (lv in c(0.95, 0.90))
      out[[sprintf("%s_delta_simple_ci%02d", prefix, round(100 * lv))]] <- vec(x$est + c(-1, 1) * qnorm((1 + lv) / 2) * x$se)
    if (ty != "simple") out[[paste0(prefix, "_emfx_", ty, "_key")]] <- vec(x$key)
  }
  out[[paste0(prefix, "_nobs")]] <- nobs(m)
  out
}

case("etwfe_staggered", "econometrics/did", "ETWFE (etwfe(ivar = id) + emfx), balanced staggered panel with never-treated: cgroup notyet / never, clustered and iid SEs, a time-invariant control, tref, anticipation", function(ctx) {
  d <- .etwfe_panel(ctx, "etwfe_panel")
  d$first_treat_ant <- ifelse(d$first_treat > 0, d$first_treat - 1, 0)   # anticipation = 1: onset g - 1
  m_iid <- .etwfe_fit(d, vcov = "iid")
  v <- c(.etwfe_values(.etwfe_fit(d), "notyet"),
         .etwfe_values(m_iid, "notyet_iid", types = "simple"),
         .etwfe_values(.etwfe_fit(d, "never"), "never"),
         .etwfe_values(.etwfe_fit(d, ctrl = TRUE), "notyet_xc"),
         .etwfe_values(.etwfe_fit(d, tref = 4), "tref4", types = c("simple", "event")),
         .etwfe_values(.etwfe_fit(d, gvar = "first_treat_ant"), "ant1", types = c("simple", "event"), cells = TRUE),
         list(n_clusters = length(unique(d$id)), notyet_iid_df = degrees_freedom(m_iid, "resid")))
  list(dataset = "validation/datasets/generated/etwfe_panel.csv", values = v,
       reference = "etwfe::etwfe(fml = y ~ 0 | y ~ xc, tvar = year, gvar = first_treat, ivar = id, cgroup, tref, vcov = ~id | 'iid'); etwfe::emfx(type); exact delta method written out",
       call = "etwfe(y ~ 0, tvar = year, gvar = first_treat, data = etwfe_panel, ivar = id, cgroup = 'notyet' | 'never', vcov = ~id)",
       packages = c("etwfe", "fixest", "marginaleffects"))
})

case("etwfe_unbalanced", "econometrics/did", "ETWFE on an unbalanced panel (CAUS-15, with and without a control) and on a panel where every unit is eventually treated (CAUS-13; default and user gref, cgroup never with gref)", function(ctx) {
  du <- .etwfe_panel(ctx, "etwfe_panel_unbalanced", balanced = FALSE)
  da <- .etwfe_panel(ctx, "etwfe_panel_all_treated", all_treated = TRUE)
  v <- c(.etwfe_values(.etwfe_fit(du), "unbal"),
         .etwfe_values(.etwfe_fit(du, "never"), "unbal_never", types = c("simple", "event")),
         .etwfe_values(.etwfe_fit(du, ctrl = TRUE), "unbal_xc"),
         .etwfe_values(.etwfe_fit(da), "alltreated"),
         .etwfe_values(.etwfe_fit(da, gref = 6), "alltreated_gref6", types = "simple"),
         .etwfe_values(.etwfe_fit(da, "never", gref = 8), "alltreated_never_gref8", types = c("simple", "event")))
  list(dataset = "validation/datasets/generated/etwfe_panel_unbalanced.csv", values = v,
       reference = "etwfe::etwfe(ivar = id, vcov = ~id); etwfe::emfx; exact delta method written out",
       call = "etwfe(y ~ 0, tvar = year, gvar = first_treat, data = <panel>, ivar = id, vcov = ~id)",
       packages = c("etwfe", "fixest", "marginaleffects"))
})
