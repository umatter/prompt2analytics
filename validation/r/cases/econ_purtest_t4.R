# validation/r/cases/econ_purtest_t4.R -- panel unit-root statistics (T4-WP09: ECON-41, ECON-43, ECON-44, ECON-76).
# plm::purtest's LLC (levinlin), IPS (Wtbar), Maddala-Wu and Hadri statistics with fixed lags; per-unit ADF
# p-values from urca::punitroot(N = Inf) (MacKinnon 1996, plm's default p.approx with urca installed); per-unit
# KPSS statistics of tseries::kpss.test for Hadri's long-run-variance option.
suppressPackageStartupMessages({ library(plm); library(urca); library(tseries) })

.pur <- function(pd, ...) suppressWarnings(purtest(pd$y, ...))
.stat <- function(r) unname(r$statistic$statistic)
.pval <- function(r) unname(r$statistic$p.value)

case("purtest_stat_ar05", "econometrics/panel_unit_root", "plm::purtest LLC / IPS / Maddala-Wu / Hadri statistics, fixed lags, N = 10, T = 50", function(ctx) {
  set.seed(20261111)
  N <- 10; TT <- 50
  d <- do.call(rbind, lapply(seq_len(N), function(i) {
    e <- rnorm(TT + 50); y <- numeric(TT + 50); for (t in 2:(TT + 50)) y[t] <- 0.5 * y[t - 1] + e[t]
    w <- cumsum(rnorm(TT))
    data.frame(id = i, t = seq_len(TT), y = round(i + y[-(1:50)], 10), y_rw = round(w, 10)) }))
  d <- ctx$gen_dataset("purtest_ar05", d)
  v <- list()
  for (col in c("y", "y_rw")) {
    pd <- pdata.frame(data.frame(id = d$id, t = d$t, y = d[[col]]), index = c("id", "t"))
    s <- if (col == "y") "" else "rw_"
    for (exo in c("intercept", "trend", "none")) for (dfc in c(FALSE, TRUE)) {
      tag <- paste0(s, exo, if (dfc) "_dfcor" else "")
      r <- .pur(pd, test = "levinlin", exo = exo, lags = 1, dfcor = dfc)
      v[[paste0("llc_", tag, "_stat")]] <- .stat(r); v[[paste0("llc_", tag, "_p")]] <- .pval(r)
      r <- .pur(pd, test = "madwu", exo = exo, lags = 1, dfcor = dfc)
      v[[paste0("madwu_", tag, "_stat")]] <- .stat(r); v[[paste0("madwu_", tag, "_p")]] <- .pval(r)
      v[[paste0("madwu_", tag, "_ptrho")]] <- vec(sapply(r$idres, function(u) u$p.trho))
      v[[paste0("madwu_", tag, "_trho")]] <- vec(sapply(r$idres, function(u) u$trho))
      if (exo != "none") {
        for (lg in c(1, 3)) {
          r <- .pur(pd, test = "ips", exo = exo, lags = lg, dfcor = dfc)
          v[[paste0("ips_", tag, "_lags", lg, "_stat")]] <- .stat(r); v[[paste0("ips_", tag, "_lags", lg, "_p")]] <- .pval(r)
          v[[paste0("ips_", tag, "_lags", lg, "_moments")]] <- vec(rowMeans(r$adjval))
        }
        for (hc in c(TRUE, FALSE)) {
          r <- .pur(pd, test = "hadri", exo = exo, Hcons = hc, dfcor = dfc)
          htag <- paste0("hadri_", tag, if (hc) "_hcons" else "_homo")
          v[[paste0(htag, "_stat")]] <- .stat(r); v[[paste0(htag, "_p")]] <- .pval(r)
          if (hc) v[[paste0(htag, "_lm")]] <- vec(sapply(r$idres, function(u) u$LM))
        }
      }
    }
    # Hadri's long-run-variance option: per-unit KPSS statistics (Bartlett kernel), short and long lags
    for (nl in c("Level", "Trend")) for (ls in c(TRUE, FALSE)) {
      k <- lapply(split(d[[col]], d$id), function(x) suppressWarnings(kpss.test(x, null = nl, lshort = ls)))
      tag <- paste0("kpss_", s, tolower(nl), if (ls) "_short" else "_long")
      v[[tag]] <- vec(sapply(k, function(u) unname(u$statistic)))
      v[[paste0(tag, "_lag")]] <- unname(k[[1]]$parameter)
    }
  }
  # MacKinnon (1994) per-unit p-values (plm's p.approx without urca), the response surface without tables
  pd <- pdata.frame(data.frame(id = d$id, t = d$t, y = d$y), index = c("id", "t"))
  for (exo in c("intercept", "trend", "none")) {
    r <- .pur(pd, test = "madwu", exo = exo, lags = 1, p.approx = "MacKinnon1994")
    v[[paste0("madwu_", exo, "_m94_stat")]] <- .stat(r); v[[paste0("madwu_", exo, "_m94_p")]] <- .pval(r)
    v[[paste0("madwu_", exo, "_m94_ptrho")]] <- vec(sapply(r$idres, function(u) u$p.trho))
  }
  # Scale invariance: the LLC statistic of 1000 * y (the case's own data)
  pd <- pdata.frame(data.frame(id = d$id, t = d$t, y = 1000 * d$y), index = c("id", "t"))
  v$llc_intercept_x1000_stat <- .stat(.pur(pd, test = "levinlin", exo = "intercept", lags = 1))
  v$n_units <- N; v$n_obs <- nrow(d)
  list(values = v, reference = "plm::purtest(test = 'levinlin' | 'ips' | 'madwu' | 'hadri', exo, lags = 1 | 3, dfcor, Hcons); tseries::kpss.test(null, lshort)",
       call = "purtest(pdata$y, test, exo, lags, dfcor, Hcons); kpss.test(x, null, lshort) per unit", packages = c("plm", "urca", "tseries"))
})

case("purtest_stat_short", "econometrics/panel_unit_root", "plm::purtest IPS / LLC / Maddala-Wu on a short panel (T = 15): IPS table rows T = 10, 15, LLC table floor", function(ctx) {
  set.seed(20261112)
  N <- 8; TT <- 15
  d <- do.call(rbind, lapply(seq_len(N), function(i) {
    e <- arima.sim(list(ar = 0.3), TT); data.frame(id = i, t = seq_len(TT), y = round(cumsum(e) + 0.2 * i, 10)) }))
  d <- ctx$gen_dataset("purtest_short", d)
  pd <- pdata.frame(d, index = c("id", "t"))
  v <- list()
  for (exo in c("intercept", "trend")) for (lg in c(1, 2, 4)) {
    r <- .pur(pd, test = "ips", exo = exo, lags = lg, dfcor = TRUE)
    tag <- paste0("ips_", exo, "_lags", lg)
    v[[paste0(tag, "_stat")]] <- .stat(r); v[[paste0(tag, "_p")]] <- .pval(r); v[[paste0(tag, "_moments")]] <- vec(rowMeans(r$adjval))
  }
  for (exo in c("intercept", "trend")) {
    r <- .pur(pd, test = "levinlin", exo = exo, lags = 1, dfcor = TRUE)
    v[[paste0("llc_", exo, "_stat")]] <- .stat(r); v[[paste0("llc_", exo, "_p")]] <- .pval(r)
    r <- .pur(pd, test = "madwu", exo = exo, lags = 1, dfcor = TRUE)
    v[[paste0("madwu_", exo, "_stat")]] <- .stat(r); v[[paste0("madwu_", exo, "_p")]] <- .pval(r)
    v[[paste0("madwu_", exo, "_ptrho")]] <- vec(sapply(r$idres, function(u) u$p.trho))
  }
  v$n_units <- N; v$n_obs <- nrow(d)
  list(values = v, reference = "plm::purtest(test = 'ips' | 'levinlin' | 'madwu', exo, lags = 1 | 2 | 4, dfcor = TRUE)",
       call = "purtest(pdata$y, test, exo, lags, dfcor = TRUE)", packages = c("plm", "urca"))
})

case("mackinnon_punitroot_grid", "econometrics/panel_unit_root", "urca::punitroot(q, N = Inf, trend = 'nc' | 'c' | 'ct'): MacKinnon (1996) asymptotic p-values of the Dickey-Fuller t on a grid", function(ctx) {
  d <- ctx$gen_dataset("mackinnon_grid", data.frame(q = round(seq(-6.5, 3, by = 0.05), 10)))
  v <- list(q = vec(d$q))
  for (tr in c("nc", "c", "ct")) v[[paste0("p_", tr)]] <- vec(sapply(d$q, function(q) punitroot(q, N = Inf, trend = tr)))
  padf <- getFromNamespace("padf", "plm")
  for (ex in c("none", "intercept", "trend")) v[[paste0("p94_", ex)]] <- vec(padf(d$q, exo = ex, p.approx = "MacKinnon1994"))
  list(values = v, reference = "urca::punitroot(q, N = Inf, trend, statistic = 't')", call = "punitroot(q, N = Inf, trend = 'nc' | 'c' | 'ct'); plm:::padf(q, exo, p.approx = 'MacKinnon1994')", packages = c("urca", "plm"))
})
