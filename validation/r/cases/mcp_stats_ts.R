# validation/r/cases/mcp_stats_ts.R -- references for the MCP statistics, hypothesis and time-series
# handlers that no earlier tier pins (T3-WP56: STAT-46, STAT-77 handler tests; TOOL-26 factor scores).
# The datasets carry missing cells and Int64 columns (Tier 2 section 2.2), so the handler tests run the
# NA policy of each tool through the real dispatch path:
#  * mcp_groups_t3: an Int64 response by a three-level factor, with two missing responses and one
#    missing label: aov, TukeyHSD, pairwise.wilcox.test, model.tables, the SE of a contrast (se.contrast);
#    two sample columns with one missing cell each: wilcox.test (rank sum), mood.test;
#  * mcp_series_t3: a seasonal series (period 12), an AR(1) series and white noise (n = 96, Int64 time
#    index): Box.test, decompose (additive, multiplicative), ar (yule-walker, burg), runmed (keep,
#    constant; median recorded only, TS-34), embed, filter, diffinv, the window and dplyr-style lag that
#    the tools document (lag recorded only, TS-29), toeplitz, ARMAtoMA, acf2AR;
#  * mcp_factanal_t3: n = 1200, 6 variables, 2 correlated factors: factanal(varimax) with Bartlett and
#    regression scores at the optimum (tight L-BFGS-B and varimax eps 1e-14);
#  * mcp_misc_t3: cor() and mad(center = mean) on T3-WP40's robust_weighted_t3, mcnemar.test and the
#    qchisq(0.95, df) outlier thresholds;
#  * mcp_hw_additive_t3: additive HoltWinters forecasts with 95% prediction intervals on WP37's
#    AirPassengers column (the interval pin of ts_holt_winters).
case("mcp_groups_t3", "stats/mcp", "aov, TukeyHSD, pairwise.wilcox.test, model.tables and a contrast SE on an Int64 response with missing cells and a missing label; wilcox.test and mood.test on two sample columns", function(ctx) {
  set.seed(20261201)
  n <- 33
  g <- rep(c("ctl", "trtA", "trtB"), each = 11)
  vals <- sort(sample(30:130, n))                                  # distinct integers: no ties in any rank test
  score <- rnorm(n) + rep(c(0, 0.9, 1.5), each = 11)
  vi <- integer(n); vi[order(score)] <- vals
  v <- round(rnorm(n, mean = rep(c(10, 11.2, 12.1), each = 11), sd = 1.4), 6)
  s1 <- round(rnorm(n, 0, 1), 5); s2 <- round(rnorm(n, 0.6, 1.3), 5)
  g[5] <- NA; vi[c(9, 20)] <- NA; v[14] <- NA; s1[3] <- NA; s2[30] <- NA
  d <- ctx$gen_dataset("mcp_groups_t3", data.frame(id = 1:n, g = g, vi = vi, v = v, s1 = s1, s2 = s2))
  stopifnot(is.integer(d$vi), is.integer(d$id), sum(is.na(d$g)) == 1, sum(is.na(d$vi)) == 2)
  dd <- d[complete.cases(d[, c("vi", "g")]), ]; dd$g <- factor(dd$g)
  stopifnot(nrow(dd) == 30, !anyDuplicated(dd$vi))
  fit <- aov(vi ~ g, data = dd); tab <- summary(fit)[[1]]
  tk <- TukeyHSD(fit, conf.level = 0.95)$g
  pw <- pairwise.wilcox.test(dd$vi, dd$g, exact = FALSE, p.adjust.method = "holm")$p.value
  low <- function(m) vec(c(m[1, 1], m[2, 1], m[2, 2]))              # trtA-ctl, trtB-ctl, trtB-trtA
  mt <- model.tables(fit, "means"); me <- model.tables(fit, "effects")
  mse <- tab[["Mean Sq"]][2]; nk <- as.numeric(table(dd$g))
  cvec <- c(1, -1, 0)                                              # ctl - trtA
  se_c <- se.contrast(fit, list(dd$g == "ctl", dd$g == "trtA"))
  stopifnot(abs(se_c - sqrt(mse * sum(cvec^2 / nk))) < 1e-12)
  s1 <- d$s1; s2 <- d$s2
  w2 <- wilcox.test(s1, s2, exact = FALSE, correct = TRUE)
  md <- mood.test(s1, s2, alternative = "less")
  v <- list(n_complete = nrow(dd), group_n = vec(nk),
            aov_ss = vec(tab[["Sum Sq"]]), aov_df = vec(tab[["Df"]]), aov_ms = vec(tab[["Mean Sq"]]),
            tukey_diff = vec(tk[, "diff"]), tukey_lwr = vec(tk[, "lwr"]), tukey_upr = vec(tk[, "upr"]), tukey_padj = vec(tk[, "p adj"]),
            pairwise_wilcox_holm = low(pw),
            means = named(mt$tables$g[seq_along(nk)]), grand_mean = unname(mt$tables[["Grand mean"]]),
            effects = named(me$tables$g[seq_along(nk)]),
            contrast_ctl_trtA_est = sum(cvec * tapply(dd$vi, dd$g, mean)), contrast_ctl_trtA_se = se_c,
            wilcox_W = unname(w2$statistic), wilcox_p = w2$p.value, wilcox_n1 = sum(!is.na(s1)), wilcox_n2 = sum(!is.na(s2)),
            mood_less_z = unname(md$statistic), mood_less_p = md$p.value)
  list(values = v,
       reference = "stats::aov, TukeyHSD, pairwise.wilcox.test(exact = FALSE), model.tables, se.contrast on complete.cases(vi, g); wilcox.test(exact = FALSE), mood.test with NA removed per sample",
       call = "aov(vi ~ g, data = d[complete.cases(d$vi, d$g), ]); TukeyHSD(fit); wilcox.test(s1, s2, exact = FALSE)",
       packages = character(0))
})

case("mcp_series_t3", "timeseries/mcp", "Box.test, decompose, ar, runmed, embed, filter, diffinv, window, toeplitz, ARMAtoMA and acf2AR on a seasonal series, an AR(1) series and white noise (n = 96)", function(ctx) {
  set.seed(20261203)
  n <- 96; tt <- 1:n
  e <- as.numeric(arima.sim(list(ar = 0.5), n, sd = 0.8))
  y <- round(60 + 0.25 * tt + 5 * sin(2 * pi * tt / 12) + 2 * cos(2 * pi * tt / 6) + e, 6)
  z <- round(as.numeric(arima.sim(list(ar = 0.6), n)), 6)
  w <- round(rnorm(n), 6)
  d <- ctx$gen_dataset("mcp_series_t3", data.frame(t = tt, y = y, z = z, w = w))
  stopifnot(is.integer(d$t), all(d$y > 0))
  y <- d$y; z <- d$z; w <- d$w
  lb <- Box.test(w, lag = 10, type = "Ljung-Box"); bp <- Box.test(w, lag = 10, type = "Box-Pierce", fitdf = 2)
  yt <- ts(y, frequency = 12)
  da <- decompose(yt); dm <- decompose(yt, "multiplicative")
  popvar <- function(r) { r <- r[!is.na(r)]; mean((r - mean(r))^2) }    # the tool's "random_variance" (divisor = count)
  yw <- ar(z, method = "yule-walker"); bg <- ar(z, method = "burg")
  rk <- runmed(z, 5, endrule = "keep"); rc <- runmed(z, 5, endrule = "constant"); rmd <- runmed(z, 5, endrule = "median")
  em <- embed(z, 3)
  f4 <- as.numeric(stats::filter(z, rep(0.25, 4), sides = 2)); fr <- as.numeric(stats::filter(z, c(0.5, -0.2), method = "recursive"))
  di <- diffinv(z, xi = 1.5)
  tsym <- toeplitz(c(1, 0.5, 0.25)); tasym <- toeplitz(c(1, 2, 3), c(1, 4, 5))
  psi <- ARMAtoMA(ar = c(0.5, -0.2), ma = 0.4, lag.max = 10)
  a2 <- acf2AR(c(1, 0.6, 0.3, 0.1))
  rm <- function(m) vec(as.numeric(t(m)))                                  # row-major
  list(values = list(lb_stat = unname(lb$statistic), lb_df = unname(lb$parameter), lb_p = lb$p.value,
                     bp_stat = unname(bp$statistic), bp_df = unname(bp$parameter), bp_p = bp$p.value,
                     add_figure = vec(da$figure), add_trend = vec(as.numeric(da$trend)), add_seasonal_head = vec(as.numeric(da$seasonal)[1:5]),
                     add_random_head = vec(as.numeric(da$random)[1:12]), add_trend_mean = mean(da$trend, na.rm = TRUE),
                     add_random_popvar = popvar(as.numeric(da$random)),
                     mult_figure = vec(dm$figure), mult_trend_mean = mean(dm$trend, na.rm = TRUE), mult_random_popvar = popvar(as.numeric(dm$random)),
                     yw_order = yw$order, yw_ar = vec(as.numeric(yw$ar)), yw_var = yw$var.pred, yw_mean = yw$x.mean,
                     burg_order = bg$order, burg_ar = vec(as.numeric(bg$ar)), burg_var = bg$var.pred,
                     runmed_keep_head = vec(rk[1:20]), runmed_constant_head = vec(rc[1:20]), runmed_median_head = vec(rmd[1:20]),
                     embed3_head = rm(em[1:10, ]), embed3_rows = nrow(em),
                     filter4_head = vec(f4[1:20]), filter_recursive_head = vec(fr[1:20]),
                     diffinv_head = vec(di[1:20]), diffinv_n = length(di),
                     window_11_20 = vec(z[11:20]), lag_dplyr3_head = vec(c(rep(NA, 3), z)[1:20]),
                     toeplitz_sym = rm(tsym), toeplitz_asym = rm(tasym),
                     arma_to_ma = vec(psi), acf2ar = rm(a2), acf2ar_pacf = vec(diag(a2))),
       reference = "stats::Box.test; stats::decompose(ts(y, frequency = 12)); stats::ar(method = 'yule-walker' | 'burg'); stats::runmed(k = 5); stats::embed; stats::filter; stats::diffinv; stats::toeplitz; stats::ARMAtoMA; stats::acf2AR",
       call = "Box.test(w, lag = 10); decompose(ts(y, frequency = 12)); ar(z); runmed(z, 5, endrule = 'keep'); embed(z, 3)",
       packages = character(0))
})

case("mcp_factanal_t3", "multivariate/factanal", "factanal(2 factors, varimax) with Bartlett and regression scores, n = 1200, 6 variables; tight L-BFGS-B and varimax", function(ctx) {
  set.seed(20261204)
  n <- 1200
  F1 <- rnorm(n); F2 <- 0.4 * F1 + sqrt(1 - 0.16) * rnorm(n)
  L <- rbind(c(0.85, 0), c(0.75, 0.1), c(0.7, 0), c(0, 0.8), c(0.15, 0.7), c(0, 0.65))
  X <- cbind(F1, F2) %*% t(L) + matrix(rnorm(n * 6), n, 6) %*% diag(sqrt(pmax(0.2, 1 - rowSums(L^2))))
  colnames(X) <- paste0("f", 1:6)
  d <- ctx$gen_dataset("mcp_factanal_t3", as.data.frame(round(X, 8)))
  X <- as.matrix(d)
  tight <- list(opt = list(factr = 1, pgtol = 0, maxit = 10000), rotate = list(eps = 1e-14))     # the varimax optimum
  fb <- factanal(X, 2, rotation = "varimax", scores = "Bartlett", control = tight)
  fr <- factanal(X, 2, rotation = "varimax", scores = "regression", control = tight)
  fd <- factanal(X, 2, rotation = "varimax", scores = "Bartlett")                        # R defaults: cross-check
  Lam <- unclass(fr$loadings); zz <- scale(X, TRUE, TRUE)
  model_reg <- zz %*% t(t(Lam) %*% solve(Lam %*% t(Lam) + diag(fr$uniquenesses)))          # Lambda' Sigma^-1 z (p2a's formula)
  stopifnot(max(abs(model_reg - fr$scores)) < 1e-10)                                        # equal at the ML optimum
  rows <- c(1, 2, 3, 999, 1000, 1001, 1199, 1200)
  rm <- function(m) vec(as.numeric(t(m)))
  list(values = list(rows = vec(rows), bartlett_rows = rm(fb$scores[rows, ]), regression_rows = rm(fr$scores[rows, ]),
                     bartlett_colmeans = vec(colMeans(fb$scores)), uniquenesses = vec(fb$uniquenesses),
                     loadings = rm(unclass(fb$loadings)), statistic = unname(fb$STATISTIC), pval = unname(fb$PVAL),
                     dof = unname(fb$dof), bartlett_default_maxdiff = max(abs(fd$scores - fb$scores)), n = n),
       reference = "stats::factanal(factors = 2, rotation = 'varimax', scores = 'Bartlett' | 'regression', control = list(opt = list(factr = 1, pgtol = 0), rotate = list(eps = 1e-14)))",
       call = "factanal(X, 2, rotation = 'varimax', scores = 'Bartlett')", packages = character(0))
})

case("mcp_misc_t3", "stats/mcp", "cor and mad(center = mean) on robust_weighted_t3; mcnemar.test; qchisq(0.95, df)", function(ctx) {
  r <- ctx$dataset("datasets/generated/robust_weighted_t3.csv")
  cm <- cor(as.matrix(r[, c("x1", "x2", "w")]))
  m1 <- mcnemar.test(matrix(c(40, 5, 15, 30), 2)); m0 <- mcnemar.test(matrix(c(40, 5, 15, 30), 2), correct = FALSE)
  list(values = list(cor_x1_x2_w = vec(as.numeric(cm)), mad_x1_center_mean = mad(r$x1, center = mean(r$x1)),
                     mean_x1 = mean(r$x1),
                     mcnemar_b15_c5_stat = unname(m1$statistic), mcnemar_b15_c5_p = m1$p.value,
                     mcnemar_b15_c5_uncorrected_stat = unname(m0$statistic), mcnemar_b15_c5_uncorrected_p = m0$p.value,
                     qchisq95_df1 = qchisq(0.95, 1), qchisq95_df2 = qchisq(0.95, 2), qchisq95_df3 = qchisq(0.95, 3)),
       reference = "stats::cor; stats::mad(center = mean(x)); stats::mcnemar.test (b = 15, c = 5); stats::qchisq",
       call = "cor(r[, c('x1','x2','w')]); mad(x1, center = mean(x1)); mcnemar.test(matrix(c(40, 5, 15, 30), 2))",
       packages = character(0))
})

case("mcp_hw_additive_t3", "timeseries/mcp", "HoltWinters additive (alpha 0.3, beta 0.05, gamma 0.4) on AirPassengers: 12-step forecasts and 95% prediction intervals, the interval pin of ts_holt_winters (T3-WP37-13)", function(ctx) {
  d <- ctx$dataset("datasets/generated/holtwinters_series.csv")
  f <- HoltWinters(ts(d$airpass, frequency = 12, start = c(1949, 1)), alpha = 0.3, beta = 0.05, gamma = 0.4, seasonal = "additive")
  p <- predict(f, n.ahead = 12, prediction.interval = TRUE, level = 0.95)
  list(values = list(hw_add_sse = f$SSE, hw_add_fit = vec(p[, "fit"]), hw_add_upr = vec(p[, "upr"]), hw_add_lwr = vec(p[, "lwr"])),
       reference = "stats::HoltWinters(seasonal = 'additive', alpha = 0.3, beta = 0.05, gamma = 0.4) + predict(prediction.interval = TRUE, level = 0.95)",
       call = "predict(HoltWinters(AirPassengers, 0.3, 0.05, 0.4, seasonal = 'additive'), 12, prediction.interval = TRUE)", packages = character(0))
})
