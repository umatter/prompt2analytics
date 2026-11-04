# validation/r/cases/econ_purtest.R -- panel unit-root lag selection (T3-WP17: ECON-88, ECON-42).
# plm::purtest(exo, lags = 'AIC' | 'SIC' | 'Hall', pmax = 4, dfcor, fixedT = TRUE): per-unit lag orders chosen
# on the common sample t = pmax + 2..T (plm:::lagsel), then the per-unit ADF regression with the chosen order
# (plm:::tsadf). dfcor = FALSE (plm's default) divides the residual sum of squares by n, dfcor = TRUE by n - K
# (urca::ur.df's convention, p2a's default); Hall's t-rule depends on it, AIC/SIC do not.
# The IPS, LLC and Maddala-Wu statistics are recorded for Tier 4 (ECON-43, ECON-41, ECON-44); T3 pins the lags
# and the per-unit ADF t-statistics.
suppressPackageStartupMessages(library(plm))
case("purtest_lag_selection", "econometrics/panel_unit_root", "plm::purtest IPS / LLC / Maddala-Wu with lags = 'AIC' | 'SIC' | 'Hall', pmax = 4", function(ctx) {
  set.seed(20261204)
  N <- 12; TT <- 40
  d <- do.call(rbind, lapply(seq_len(N), function(i) { e <- arima.sim(list(ar = c(0.4, -0.2)), TT); rho <- if (i <= 6) 0.85 else 1
    y <- numeric(TT); y[1] <- e[1]; for (t in 2:TT) y[t] <- rho * y[t - 1] + e[t]; data.frame(id = i, t = seq_len(TT), y = round(y, 10)) }))
  d <- ctx$gen_dataset("purtest_panel", d)
  pd <- pdata.frame(d, index = c("id", "t"))
  pt <- function(...) suppressWarnings(purtest(pd$y, pmax = 4, ...))
  v <- list()
  for (dfc in c(FALSE, TRUE)) for (lg in c("AIC", "SIC", "Hall")) {
    sfx <- if (dfc) "_dfcor" else ""
    for (tst in c("ips", "levinlin", "madwu")) {
      r <- pt(test = tst, exo = "intercept", lags = lg, dfcor = dfc)
      tag <- paste0(tst, "_", tolower(lg), sfx)
      v[[paste0(tag, "_stat")]] <- unname(r$statistic$statistic); v[[paste0(tag, "_p")]] <- unname(r$statistic$p.value)
      v[[paste0(tag, "_lags")]] <- vec(sapply(r$idres, function(u) u$lags))
      if (tst == "ips") {
        v[[paste0("trho_", tolower(lg), sfx)]] <- vec(sapply(r$idres, function(u) u$trho))
        v[[paste0("ptrho_", tolower(lg), sfx)]] <- vec(sapply(r$idres, function(u) u$p.trho))
      }
    }
  }
  r <- pt(test = "ips", exo = "trend", lags = "AIC", dfcor = TRUE)
  v$trend_lags_aic_dfcor <- vec(sapply(r$idres, function(u) u$lags)); v$trend_trho_aic_dfcor <- vec(sapply(r$idres, function(u) u$trho))
  r <- pt(test = "madwu", exo = "none", lags = "SIC", dfcor = TRUE)
  v$none_lags_sic_dfcor <- vec(sapply(r$idres, function(u) u$lags)); v$none_trho_sic_dfcor <- vec(sapply(r$idres, function(u) u$trho))
  stopifnot(!identical(v$ips_hall_lags, v$ips_hall_dfcor_lags))   # the dfcor convention changes Hall's choice
  v$n_units <- N; v$n_obs <- nrow(d)
  list(values = v, reference = "plm::purtest(test, exo, lags = 'AIC' | 'SIC' | 'Hall', pmax = 4, dfcor, fixedT = TRUE)",
       call = "purtest(pdata$y, test, exo = 'intercept', lags, pmax = 4, dfcor)", packages = "plm")
})
