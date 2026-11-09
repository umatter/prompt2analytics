# validation/r/cases/mcp_dispatch.R -- references for the MCP dispatcher tests of T3-WP58 (TOOL-31,
# TOOL-32): one representative computation per handler module through call_tool_with_session, where
# no earlier case already pins it.
#  * mcp_competing_risks: the Aalen-Johansen cumulative incidence of two competing events, as
#    survival::survfit(Surv(time, factor(event, 0:2)) ~ 1) (multi-state pstate), on a small sample
#    with tied times and an integer time column. Only the point estimates are pinned (ECON-78: the
#    CIF variance is Tier 4).
#  * mcp_dispatch_smoke: a 48-row dataset with integer and missing cells. stats::kmeans (nstart = 50,
#    global optimum on three separated blobs), Moran's I with rook neighbours on an 8 x 6 grid and
#    row-standardised weights written out (spdep is not installed: I = (n / S0) z'Wz / z'z,
#    E[I] = -1 / (n - 1)), changepoint::cpt.mean(method = "PELT") on a series with one mean shift,
#    the lag-0 cross-correlation (= cor) and the string operations of the munging tools (vowel counts
#    and lengths). The seasonal (seas), VAR (y1, y2) and factor (f1..f4) columns feed smoke tests only:
#    Holt-Winters, VAR IRF and factanal values are pinned by their owners (T3-WP37, WP12; factanal has
#    Tier 4 findings STAT-23, STAT-49), not through the dispatcher.
suppressPackageStartupMessages({ library(survival); library(changepoint) })

case("mcp_competing_risks", "econometrics/survival", "Aalen-Johansen cumulative incidence of two competing events (survfit multi-state), integer times with ties", function(ctx) {
  set.seed(20261023)
  n <- 40
  t1 <- rexp(n, 0.08); t2 <- rexp(n, 0.05); cens <- runif(n, 5, 30)
  tt <- pmin(t1, t2, cens)
  ev <- ifelse(tt == cens, 0L, ifelse(tt == t1, 1L, 2L))
  d <- ctx$gen_dataset("cr_small", data.frame(id = seq_len(n), time = as.integer(ceiling(tt)), event = ev))
  stopifnot(is.integer(d$time), is.integer(d$event), anyDuplicated(d$time) > 0)
  f <- survfit(Surv(time, factor(event, levels = 0:2)) ~ 1, data = d)
  ps <- f$pstate                                  # columns: (s0), 1, 2 at every distinct time
  stopifnot(ncol(ps) == 3, all(f$states == c("(s0)", "1", "2")))
  list(values = list(times = vec(f$time), cif1 = vec(ps[, 2]), cif2 = vec(ps[, 3]),
                     n_obs = nrow(d), n_censored = sum(d$event == 0), n_event1 = sum(d$event == 1),
                     n_event2 = sum(d$event == 2)),
       reference = "survival::survfit(Surv(time, factor(event, levels = 0:2)) ~ 1)$pstate (Aalen-Johansen)",
       call = "survfit(Surv(time, factor(event, 0:2)) ~ 1, data = cr_small)", packages = "survival")
})

case("mcp_dispatch_smoke", "mcp", "One computation per MCP handler module: kmeans, Moran's I, cpt.mean, ccf lag 0, string operations", function(ctx) {
  set.seed(20261024)
  n <- 48; id <- seq_len(n)
  cl <- rep(1:3, each = 16)
  centre_a <- c(0, 8, 0)[cl]; centre_b <- c(0, 8, 8)[cl]
  a <- round(centre_a + rnorm(n, 0, 0.5), 6); b <- round(centre_b + rnorm(n, 0, 0.5), 6)
  gx <- rep(1:8, times = 6); gy <- rep(1:6, each = 8)
  v <- round(0.4 * gx + 0.3 * gy + rnorm(n, 0, 0.5), 6)
  s <- round(c(rnorm(24, 0, 1), rnorm(24, 6, 1)), 6)
  u <- round(rnorm(n), 6); w <- round(0.6 * u + rnorm(n, 0, 0.8), 6)
  seas <- round(10 + 0.2 * id + c(2, -1, 0.5, -1.5)[(id - 1) %% 4 + 1] + rnorm(n, 0, 0.3), 6)
  y1 <- numeric(n); y2 <- numeric(n); e1 <- rnorm(n); e2 <- rnorm(n)
  for (i in 2:n) { y1[i] <- 0.5 * y1[i - 1] + 0.2 * y2[i - 1] + e1[i]; y2[i] <- -0.3 * y1[i - 1] + 0.4 * y2[i - 1] + e2[i] }
  y1 <- round(y1, 6); y2 <- round(y2, 6)
  L <- rnorm(n)
  f1 <- round(0.9 * L + rnorm(n, 0, 0.45), 6); f2 <- round(0.8 * L + rnorm(n, 0, 0.6), 6)
  f3 <- round(0.7 * L + rnorm(n, 0, 0.7), 6); f4 <- round(0.6 * L + rnorm(n, 0, 0.8), 6)
  fruit <- c("apple", "banana", "cherry", "date")[(id - 1) %% 4 + 1]
  txt <- ifelse(fruit == "date", "date", paste0(fruit, "-", id * 7))
  k <- as.integer(rpois(n, 3))
  xn <- round(rnorm(n, 5, 2), 6); xn[c(5, 17, 33)] <- NA
  d <- ctx$gen_dataset("mcp_dispatch", data.frame(id = id, k = k, xn = xn, a = a, b = b, gx = gx, gy = gy, v = v,
                                                  s = s, u = u, w = w, seas = seas, y1 = y1, y2 = y2,
                                                  f1 = f1, f2 = f2, f3 = f3, f4 = f4, txt = txt))
  stopifnot(is.integer(d$id), is.integer(d$k), is.integer(d$gx), sum(is.na(d$xn)) == 3)
  km <- kmeans(cbind(d$a, d$b), centers = 3, nstart = 50, iter.max = 100)
  # rook neighbours on the 8 x 6 unit grid, row-standardised
  D <- as.matrix(dist(cbind(d$gx, d$gy))); W <- (abs(D - 1) < 1e-9) * 1; W <- W / rowSums(W)
  z <- d$v - mean(d$v)
  moran_i <- (n / sum(W)) * drop(t(z) %*% W %*% z) / sum(z^2)
  cp <- cpts(cpt.mean(d$s, method = "PELT"))
  vowels <- lengths(regmatches(d$txt, gregexpr("[aeiou]", d$txt)))
  list(values = list(
    kmeans_tot_withinss = km$tot.withinss, kmeans_sizes = vec(sort(km$size)),
    moran_i = moran_i, moran_expected = -1 / (n - 1), n_rook_links = sum(W > 0),
    cpt_mean_location = vec(cp), ccf_lag0 = cor(d$u, d$w),
    vowel_count = vec(vowels), txt_length = vec(nchar(d$txt)),
    xn_missing = sum(is.na(d$xn)), n = n),
    reference = "stats::kmeans(nstart = 50); Moran's I written out (rook, style W); changepoint::cpt.mean(PELT); cor; regmatches/nchar",
    call = "kmeans(cbind(a, b), 3, nstart = 50); cpt.mean(s, method = 'PELT'); cor(u, w)",
    packages = c("changepoint"))
})
