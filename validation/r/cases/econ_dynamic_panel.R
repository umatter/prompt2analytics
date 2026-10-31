# validation/r/cases/econ_dynamic_panel.R -- plm::pgmm (Arellano-Bond difference GMM) references
# (T2-WP19: ECON-28, ECON-29, ECON-38, ECON-39; ECON-36 GMM part). x is strictly exogenous and
# enters as its own instrument in differences, as p2a's run_gmm documents.
#
# p2a drops incomplete rows (na.omit) before building the panel, so every case computes on
# d[complete.cases(d), ]. plm's lag() is time-aware (the lag of 2004 is 2003, missing if 2003 is
# not observed), so equations that need an unobserved period are dropped, as in p2a. pgmm counts
# instrument columns that are zero for every equation (periods no unit observes); p2a drops them,
# so the pinned instrument count and Sargan df are the non-zero columns (`*_n_instruments_nonzero`,
# `*_sargan_df_nonzero`); pgmm's own count is recorded as `*_n_instruments`.
suppressPackageStartupMessages(library(plm))

.ab_panel <- function(ctx, name, drop_rows = FALSE) {
  set.seed(606)
  N <- 100; TT <- 8
  a <- rnorm(N)
  d <- do.call(rbind, lapply(seq_len(N), function(i) {
    x <- rnorm(TT); y <- numeric(TT); y[1] <- a[i] + rnorm(1)
    for (t in 2:TT) y[t] <- 0.5 * y[t - 1] + 0.3 * x[t] + a[i] + rnorm(1)
    data.frame(id = i, year = 2000L + seq_len(TT) - 1L, y = round(y, 10), x = round(x, 10))
  }))
  if (drop_rows) {
    d <- d[!(d$id %% 9 == 0 & d$year == 2004L), ]            # interior gaps
    d <- d[!(d$id %% 13 == 0 & d$year <= 2001L), ]           # late entry
  }
  ctx$gen_dataset(name, d)
}

# One pgmm fit, the quantities p2a reports, under `prefix`.
.ab_fit <- function(pd, form, st, prefix, collapse = FALSE) {
  m <- suppressWarnings(pgmm(form, data = pd, effect = "individual", model = st, transformation = "d", collapse = collapse))
  s <- suppressWarnings(summary(m, robust = FALSE)); sr <- suppressWarnings(summary(m, robust = TRUE))
  out <- list()
  k <- length(coef(m))
  nz <- sum(colSums(abs(Reduce(`+`, lapply(m$W, abs)))) > 0)
  out[[paste0(prefix, "_coef")]] <- named(coef(m))
  out[[paste0(prefix, "_se")]] <- named(setNames(s$coefficients[, 2], rownames(s$coefficients)))
  out[[paste0(prefix, "_se_robust")]] <- named(setNames(sr$coefficients[, 2], rownames(sr$coefficients)))
  out[[paste0(prefix, "_sargan")]] <- unname(s$sargan$statistic); out[[paste0(prefix, "_sargan_df")]] <- unname(s$sargan$parameter)
  out[[paste0(prefix, "_sargan_df_nonzero")]] <- nz - k
  out[[paste0(prefix, "_sargan_p")]] <- s$sargan$p.value
  out[[paste0(prefix, "_sargan_p_nonzero")]] <- pchisq(unname(s$sargan$statistic), df = nz - k, lower.tail = FALSE)
  out[[paste0(prefix, "_ar1")]] <- unname(sr$m1$statistic); out[[paste0(prefix, "_ar1_p")]] <- sr$m1$p.value
  out[[paste0(prefix, "_ar2")]] <- unname(sr$m2$statistic); out[[paste0(prefix, "_ar2_p")]] <- sr$m2$p.value
  out[[paste0(prefix, "_n_instruments")]] <- ncol(m$W[[1]])
  out[[paste0(prefix, "_n_instruments_nonzero")]] <- nz
  out[[paste0(prefix, "_n_obs")]] <- sum(sapply(m$residuals, function(r) sum(!is.na(r) & r != 0)))
  if (st == "onestep") {
    # pgmm's non-robust one-step vcov (m$B1) omits sigma^2 and carries a 1/N factor; p2a reports
    # the Arellano-Bond (1991) / Stata xtabond form sigma2 * (X'Z A Z'X)^-1, A = (sum_i Z_i' H Z_i)^-1,
    # with sigma2 = sum(residuals^2) / (2 (n_obs - k)) from the differenced residuals.
    res <- unlist(lapply(m$residuals, function(r) r[!is.na(r) & r != 0]))
    n_eq <- length(res)
    sigma2 <- sum(res^2) / (2 * (n_eq - k))
    V <- sigma2 * length(m$W) * m$B1
    out[[paste0(prefix, "_sigma2")]] <- sigma2
    out[[paste0(prefix, "_se_textbook")]] <- named(setNames(sqrt(diag(V)), names(coef(m))))
    # Arellano-Bond (1991) Sargan statistic of the one-step fit (homoskedastic errors):
    # (sum_i Z_i' e_i)' (sum_i Z_i' H Z_i)^-1 (sum_i Z_i' e_i) / sigma2; m$A1 = N (sum_i Z_i' H Z_i)^-1.
    z <- Reduce(`+`, mapply(crossprod, m$W, m$residuals, SIMPLIFY = FALSE))
    sh <- drop(t(z) %*% m$A1 %*% z) / (length(m$W) * sigma2)
    out[[paste0(prefix, "_sargan_homoskedastic")]] <- sh
    out[[paste0(prefix, "_sargan_homoskedastic_p")]] <- pchisq(sh, df = nz - k, lower.tail = FALSE)
  }
  out
}

.ab_values <- function(d, form = y ~ lag(y, 1) + x | lag(y, 2:99), index = c("id", "year")) {
  pd <- pdata.frame(d[complete.cases(d), ], index = index)
  c(.ab_fit(pd, form, "onestep", "onestep"), .ab_fit(pd, form, "twosteps", "twostep"))
}

case("dpd_ab_balanced", "econometrics/panel", "Arellano-Bond difference GMM (pgmm), balanced N = 100, T = 8", function(ctx) {
  d <- .ab_panel(ctx, "ab_balanced")
  pd <- pdata.frame(d, index = c("id", "year"))
  v <- c(.ab_values(d),
         # variants of the instrument and lag specification (p2a lags, max_lag, min_lag, collapse, no x)
         .ab_fit(pd, y ~ lag(y, 1:2) + x | lag(y, 2:99), "twosteps", "lags2_twostep"),
         .ab_fit(pd, y ~ lag(y, 1) + x | lag(y, 2:3), "twosteps", "maxlag3_twostep"),
         .ab_fit(pd, y ~ lag(y, 1) + x | lag(y, 3:99), "onestep", "minlag3_onestep"),
         .ab_fit(pd, y ~ lag(y, 1) + x | lag(y, 2:99), "twosteps", "collapse_twostep", collapse = TRUE),
         .ab_fit(pd, y ~ lag(y, 1) | lag(y, 2:99), "onestep", "nox_onestep"))
  list(values = v, reference = "plm::pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), effect = 'individual', transformation = 'd', model = 'onestep' | 'twostep'); summary(robust = FALSE | TRUE) (Windmeijer for two-step); sargan; mtest; variants lag(y, 1:2), lag(y, 2:3), lag(y, 3:99), collapse = TRUE, no x",
       call = "pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), data = pdata.frame(d, index = c('id', 'year')), effect = 'individual', model = ...)",
       packages = "plm")
})

case("dpd_ab_gaps", "econometrics/panel", "Arellano-Bond difference GMM (pgmm) with interior gaps and late entry", function(ctx) {
  d <- .ab_panel(ctx, "ab_gaps", drop_rows = TRUE)
  pd <- pdata.frame(d, index = c("id", "year"))
  v <- c(.ab_values(d),
         .ab_fit(pd, y ~ lag(y, 1:2) + x | lag(y, 2:99), "twosteps", "lags2_twostep"),
         .ab_fit(pd, y ~ lag(y, 1) + x | lag(y, 2:99), "twosteps", "collapse_twostep", collapse = TRUE))
  list(values = v, reference = "plm::pgmm on an unbalanced panel (equations need y_t, y_{t-1}, y_{t-2}; instruments only from observed lags)",
       call = "pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), data = pdata.frame(d_with_gaps, ...), ...)", packages = "plm")
})

case("dpd_ab_common_gap", "econometrics/panel", "Arellano-Bond difference GMM (pgmm): a year no unit observes, missing cells, text unit ids", function(ctx) {
  set.seed(606)
  N <- 100; TT <- 8
  a <- rnorm(N)
  d <- do.call(rbind, lapply(seq_len(N), function(i) {
    x <- rnorm(TT); y <- numeric(TT); y[1] <- a[i] + rnorm(1)
    for (t in 2:TT) y[t] <- 0.5 * y[t - 1] + 0.3 * x[t] + a[i] + rnorm(1)
    data.frame(id = i, unit = sprintf("u%03d", i), year = 2000L + seq_len(TT) - 1L, y = round(y, 10), x = round(x, 10))
  }))
  d <- d[d$year != 2003L, ]                                   # no unit observes 2003
  d$y[d$id %% 10 == 1 & d$year == 2005L] <- NA                # missing outcome cells
  d$x[d$id %% 10 == 2 & d$year == 2006L] <- NA                # missing regressor cells
  d <- ctx$gen_dataset("ab_common_gap", d)
  stopifnot(!any(d$year == 2003L), sum(!complete.cases(d)) == 20L, is.character(d$unit))
  v <- .ab_values(d[, c("id", "year", "y", "x")])
  v$n_rows_dropped <- sum(!complete.cases(d))
  list(values = v, reference = "plm::pgmm on the complete rows; plm's time-aware lag() drops the equations that need 2003, and the 2003 instrument columns are zero",
       call = "pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), data = pdata.frame(d[complete.cases(d), ], index = c('id', 'year')), effect = 'individual', model = ...)",
       packages = "plm")
})

case("dpd_ab_grunfeld", "econometrics/panel", "Arellano-Bond difference GMM (pgmm), Grunfeld: 173 instruments for 10 firms (rank-deficient weights)", function(ctx) {
  d <- ctx$dataset("datasets/grunfeld.csv")
  v <- .ab_values(d, inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99), index = c("firm", "year"))
  # pgmm inverts both weighting matrices with MASS::ginv (singular values <= sqrt(eps) * max set to 0).
  pd <- pdata.frame(d, index = c("firm", "year"))
  m <- suppressWarnings(pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99), data = pd, effect = "individual", model = "onestep", transformation = "d"))
  S1 <- Reduce(`+`, lapply(m$W, function(w) t(w) %*% plm:::FSM(nrow(w), "G") %*% w))
  S2 <- Reduce(`+`, mapply(function(w, r) { g <- crossprod(w, r); g %*% t(g) }, m$W, m$residuals, SIMPLIFY = FALSE))
  rk <- function(M) { s <- svd(M)$d; sum(s > sqrt(.Machine$double.eps) * s[1]) }
  v$rank_one_step_weight <- rk(S1); v$rank_two_step_weight <- rk(S2)
  list(values = v, reference = "plm::pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99)), model = 'onestep' | 'twostep'; both weighting matrices are singular and pgmm uses MASS::ginv",
       call = "pgmm(inv ~ lag(inv, 1) + value + capital | lag(inv, 2:99), data = pdata.frame(grunfeld, index = c('firm', 'year')), effect = 'individual', model = ...)",
       packages = c("plm", "MASS"))
})

# Panel unit-root extraction (ECON-40): per-unit ADF tau (intercept, one lagged difference) on each
# unit's observations in time order, with the rows that miss y dropped and gaps spliced, as
# plm::purtest does (its IPS statistic is the same on the gapped and on the relabelled panel).
# The panel statistics themselves (LLC, the IPS standardisation, Fisher p-values) are Tier 4
# (ECON-41/43/44) and are not pinned here.
case("panel_pur_extraction", "econometrics/panel", "Panel unit root: per-unit ADF tau after extraction (text unit ids, missing cells, a year no unit observes)", function(ctx) {
  suppressPackageStartupMessages(library(urca))
  d <- ctx$dataset("datasets/generated/ab_common_gap.csv")
  keep <- complete.cases(d[, c("unit", "year", "y")])
  dd <- d[keep, ]
  units <- split(dd, factor(dd$unit, levels = sort(unique(dd$unit), method = "radix")))   # C order
  tau <- vapply(units, function(u) { u <- u[order(u$year), ]; unname(ur.df(u$y, type = "drift", lags = 1)@teststat[1]) }, 0)
  list(values = list(unit_tau = vec(tau), n_panels = length(units),
                     avg_t = mean(vapply(units, nrow, 0L)), n_rows_dropped = sum(!keep)),
       reference = "urca::ur.df(y, type = 'drift', lags = 1) per unit on the spliced series (the per-unit ADF regression of p2a's IPS and Fisher tests)",
       call = "lapply(split(d_complete, d_complete$unit), function(u) ur.df(u$y[order(u$year)], type = 'drift', lags = 1)@teststat[1])",
       packages = "urca")
})
