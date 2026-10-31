# validation/r/cases/stats_mcp.R -- references for the MCP statistics handlers that extract their
# own columns: loglin, weighted.mean, cov.wt, mauchly.test, fivenum/IQR/mad/ecdf, spline/approx,
# isoreg, medpolish and prop.trend.test (T2-WP20: TOOL-8, TOOL-9, TOOL-24, TOOL-17 and the
# X-2/X-8/TOOL-3 sites in crates/p2a-mcp/src/tools/handlers/{stats,hypothesis}.rs).

# A 3 x 2 x 2 count table in long form with one duplicated cell (xtabs sums it), a row with a
# missing f1 label, a row with a missing f2 code and a row with a missing count. f2 holds the
# integer codes 2 and 10 (Int64 in polars), so its levels must be ordered numerically (2 < 10).
case("stat_loglin_na", "stats/loglin", "Log-linear models (IPF) on a long-form count table with missing labels and counts", function(ctx) {
  set.seed(211)
  grid <- expand.grid(f1 = c("a", "b", "c"), f2 = c(2, 10), f3 = c("x", "y"), stringsAsFactors = FALSE)
  lam <- with(grid, 20 * (1 + (f1 == "b") + 0.5 * (f1 == "c")) * ifelse(f2 == 10, 1.6, 1) *
                    ifelse(f3 == "y", 0.7, 1) * ifelse(f1 == "a" & f2 == 10, 1.8, 1) *
                    ifelse(f1 == "c" & f3 == "y", 1.5, 1))
  grid$cnt <- rpois(nrow(grid), lam)
  extra <- data.frame(f1 = c("b", NA, "c", "a"), f2 = c(10, 2, NA, 2), f3 = c("y", "x", "x", "y"),
                      cnt = c(4, 9, 6, NA))
  d0 <- rbind(grid, extra)
  d <- ctx$gen_dataset("loglin_na", data.frame(id = seq_len(nrow(d0)), d0))
  stopifnot(is.integer(d$f2), is.integer(d$cnt), sum(!complete.cases(d$cnt, d$f1, d$f2, d$f3)) == 3)
  tab <- xtabs(cnt ~ f1 + f2 + f3, data = d)            # drops the rows with a missing label or count
  stopifnot(identical(dimnames(tab)$f2, c("2", "10")))
  models <- list(indep = list(1, 2, 3), m12_3 = list(c(1, 2), 3), m12_13 = list(c(1, 2), c(1, 3)))
  v <- list()
  for (k in names(models)) {
    f <- loglin(tab, models[[k]], fit = TRUE, print = FALSE)
    v[[paste0(k, "_lrt")]] <- f$lrt; v[[paste0(k, "_pearson")]] <- f$pearson; v[[paste0(k, "_df")]] <- f$df
    # loglin reports no p-values; p2a reports the chi-square upper tails of the two statistics
    v[[paste0(k, "_p_lrt")]] <- pchisq(f$lrt, f$df, lower.tail = FALSE)
    v[[paste0(k, "_p_pearson")]] <- pchisq(f$pearson, f$df, lower.tail = FALSE)
  }
  cc <- complete.cases(d$cnt, d$f1, d$f2, d$f3)
  v$total <- sum(tab); v$dims <- vec(dim(tab)); v$n_used <- sum(cc); v$n_total <- nrow(d)
  v$n_missing_cnt <- sum(is.na(d$cnt)); v$n_missing_f1 <- sum(is.na(d$f1)); v$n_missing_f2 <- sum(is.na(d$f2))
  list(values = v, reference = "stats::loglin(xtabs(cnt ~ f1 + f2 + f3), margin, fit = TRUE); eps = 0.1, iter = 20 (defaults)",
       call = "loglin(xtabs(cnt ~ f1 + f2 + f3, data = d), list(1, 2, 3) | list(c(1, 2), 3) | list(c(1, 2), c(1, 3)))",
       packages = character(0))
})

# weighted.mean with one missing value and one missing weight in different rows (TOOL-9),
# and cov.wt on integer columns. R's weighted.mean(na.rm = TRUE) drops only missing values and
# returns NA when a kept weight is missing; the MCP tool drops incomplete (value, weight) pairs,
# which is weighted.mean on complete.cases(v, w). cov.wt refuses missing values and weights.
case("stat_weighted_na", "stats/weighted", "weighted.mean on complete (value, weight) pairs; cov.wt on integer columns", function(ctx) {
  set.seed(307)
  n <- 12
  v <- round(rnorm(n, 10, 3), 2); w <- round(runif(n, 0.5, 3), 2)
  v[3] <- NA; w[8] <- NA
  a <- round(rnorm(n, 50, 10)); b <- round(0.6 * a + rnorm(n, 0, 6))
  wc <- round(runif(n, 0.2, 2), 3)
  b_na <- b; b_na[5] <- NA
  d <- ctx$gen_dataset("weighted_na", data.frame(id = 1:n, v = v, w = w, a = a, b = b, wc = wc, b_na = b_na))
  stopifnot(is.integer(d$a), is.integer(d$b), is.integer(d$b_na), is.double(d$v), is.double(d$w))
  cc <- complete.cases(d$v, d$w)
  X <- cbind(a = d$a, b = d$b)
  cw <- cov.wt(X, wt = d$wc, cor = TRUE)
  cw_ml <- cov.wt(X, wt = d$wc, cor = TRUE, method = "ML")
  e_x <- tryCatch({ cov.wt(cbind(d$a, d$b_na), wt = d$wc); "no error" }, error = conditionMessage)
  e_w <- tryCatch({ cov.wt(X, wt = d$w); "no error" }, error = conditionMessage)
  stopifnot(e_x == "'x' must contain finite values only", e_w != "no error")
  # Cross-check values: what today's handler computes (each column's nulls dropped on its own,
  # then paired by position), and R's own weighted.mean(na.rm = TRUE) (NA: a kept weight is missing).
  pos <- weighted.mean(d$v[!is.na(d$v)], d$w[!is.na(d$w)])
  list(values = list(
    wmean = weighted.mean(d$v[cc], d$w[cc]), wmean_n = sum(cc), wmean_sum_w = sum(d$w[cc]),
    wmean_int = weighted.mean(d$a, d$wc), wmean_int_sum_w = sum(d$wc),
    wmean_na_rm_r = weighted.mean(d$v, d$w, na.rm = TRUE), wmean_mispaired = pos,
    cov = vec(cw$cov), center = named(cw$center), cor = vec(cw$cor), wt = vec(cw$wt), n_obs = cw$n.obs,
    cov_ml = vec(cw_ml$cov), cor_ml = vec(cw_ml$cor)),
    reference = "stats::weighted.mean on complete.cases(v, w); stats::cov.wt(cor = TRUE, method = 'unbiased' | 'ML')",
    call = "weighted.mean(v[cc], w[cc]); cov.wt(cbind(a, b), wt = wc, cor = TRUE)", packages = character(0))
})

# Mauchly's test on four repeated measures (integer-valued, Int64 in polars) with two missing
# cells: lm(cbind(...) ~ 1) drops the two incomplete subjects (TOOL-24). The variances grow
# across conditions, so sphericity is rejected.
case("stat_mauchly_na", "stats/mauchly", "Mauchly's sphericity test (X = ~1) with incomplete subjects", function(ctx) {
  set.seed(419)
  n <- 16
  subj <- rnorm(n, 0, 2)
  m <- sapply(1:4, function(j) round(10 * (20 + 2 * j + subj * (1 + 0.6 * j) + rnorm(n, 0, c(0.5, 1, 2.5, 5)[j]))))
  colnames(m) <- paste0("t", 1:4)
  m[5, "t2"] <- NA; m[11, "t4"] <- NA
  d <- ctx$gen_dataset("mauchly_na", data.frame(id = 1:n, m))
  stopifnot(all(sapply(d[paste0("t", 1:4)], is.integer)))
  fit <- lm(cbind(t1, t2, t3, t4) ~ 1, data = d)
  mt <- mauchly.test(fit, X = ~1)
  sp <- stats:::sphericity(SSD(fit), X = ~1)
  # p2a reports the first-order chi-square approximation; R adds Box's second-order term (w2):
  # z and the first-order p are R's own intermediate quantities (stats:::mauchly.test.SSD).
  s <- SSD(fit); p <- 4; pp <- p - 1; nd <- s$df
  Tm <- stats:::Thin.row(stats:::Proj(diag(p)) - stats:::Proj(matrix(1, p, 1)))
  U <- solve(Tm %*% t(Tm), Tm %*% s$SSD %*% t(Tm))
  logW <- log(det(U)) - pp * log(sum(diag(U)) / pp)
  rho <- 1 - (2 * pp^2 + pp + 2) / (6 * pp * nd)
  z <- -nd * rho * logW; f <- pp * (pp + 1) / 2 - 1
  stopifnot(isTRUE(all.equal(exp(logW), unname(mt$statistic), tolerance = 1e-12)), mt$p.value < 0.05)
  list(values = list(w = unname(mt$statistic), p_r = mt$p.value, chisq = z, df = f,
                     p_first_order = pchisq(z, f, lower.tail = FALSE),
                     gg = sp$GG.eps, hf = sp$HF.eps, lb = 1 / pp, n_used = nobs(fit), n_total = nrow(d)),
       reference = "stats::mauchly.test(lm(cbind(t1, t2, t3, t4) ~ 1), X = ~1); stats:::sphericity (Greenhouse-Geisser, Huynh-Feldt)",
       call = "mauchly.test(lm(cbind(t1, t2, t3, t4) ~ 1, data = d), X = ~1)", packages = character(0))
})

# The single-column descriptive tools (documented as na.rm = TRUE) and spline/approx on (x, y)
# pairs with missing values in different rows (R drops incomplete pairs: regularize.values).
# vi is an integer column (Int64 in polars) with missing cells; s is text.
case("stat_descriptive_na", "stats/robust", "fivenum, IQR, mad, ecdf, spline and approx with missing values", function(ctx) {
  set.seed(613)
  n <- 15
  v <- round(rnorm(n, 20, 5), 3); v[c(2, 9, 13)] <- NA
  vi <- round(rnorm(n, 100, 15)); vi[c(4, 11)] <- NA
  xs <- round(sort(runif(n, 0, 10)), 3); ys <- round(sin(xs) + rnorm(n, 0, 0.1), 4)
  xs[5] <- NA; ys[10] <- NA
  s <- rep(c("lo", "mid", "hi"), length.out = n)
  d <- ctx$gen_dataset("descriptive_na", data.frame(id = 1:n, v = v, vi = vi, s = s, xs = xs, ys = ys))
  stopifnot(is.integer(d$vi), is.double(d$v), !anyDuplicated(d$xs[!is.na(d$xs)]))
  at <- c(15, 20, 25)
  cc <- complete.cases(d$xs, d$ys)
  xo <- c(1.5, 4, 6.5, 8)
  list(values = list(
    fivenum_v = vec(fivenum(d$v)), iqr_v = IQR(d$v, na.rm = TRUE), mad_v = mad(d$v, na.rm = TRUE),
    mad_mean_v = mad(d$v, center = mean(d$v, na.rm = TRUE), na.rm = TRUE), mean_v = mean(d$v, na.rm = TRUE),
    ecdf_v_at = vec(ecdf(d$v[!is.na(d$v)])(at)), n_v = sum(!is.na(d$v)),
    fivenum_vi = vec(fivenum(d$vi)), iqr_vi = IQR(d$vi, na.rm = TRUE), mad_vi = mad(d$vi, na.rm = TRUE), n_vi = sum(!is.na(d$vi)),
    approx_y = vec(approx(d$xs, d$ys, xout = xo)$y),
    spline_natural_y = vec(spline(d$xs, d$ys, method = "natural", xout = xo)$y),
    n_pairs = sum(cc)),
    reference = "stats::fivenum, IQR(na.rm = TRUE), mad(na.rm = TRUE), ecdf; approx and spline(method = 'natural') drop incomplete (x, y) pairs",
    call = "fivenum(v); IQR(v, na.rm = TRUE); mad(v, na.rm = TRUE); ecdf(v)(at); approx(xs, ys, xout); spline(xs, ys, method = 'natural', xout)",
    packages = character(0))
})

# Isotonic regression on an integer response (x without ties: STAT-52, the pooling of tied x, is
# Tier 4) and Tukey's median polish on integer columns, one of them with a missing cell. isoreg
# refuses missing values ("missing values not allowed"); medpolish skips them only with na.rm = TRUE.
case("stat_isoreg_medpolish_int", "stats/robust", "isoreg and medpolish on integer columns", function(ctx) {
  set.seed(523)
  n <- 10
  x <- round(runif(n, 0, 20), 2)
  y <- round(2 * x + rnorm(n, 0, 8))
  y_na <- y; y_na[6] <- NA
  re <- rnorm(n, 0, 6); ce <- c(-5, 0, 3, 9)
  m <- sapply(ce, function(cj) round(40 + re + cj + rnorm(n, 0, 2)))
  colnames(m) <- paste0("c", 1:4)
  m[4, "c3"] <- NA
  d <- ctx$gen_dataset("isoreg_medpolish_int", data.frame(id = 1:n, x = x, y = y, y_na = y_na, m))
  stopifnot(!anyDuplicated(d$x), is.integer(d$y), is.integer(d$y_na), all(sapply(d[paste0("c", 1:4)], is.integer)))
  ir <- isoreg(d$x, d$y)
  e <- tryCatch({ isoreg(d$x, d$y_na); "no error" }, error = conditionMessage)
  stopifnot(e == "missing values not allowed")
  mp <- medpolish(as.matrix(d[paste0("c", 1:4)]), na.rm = TRUE, trace.iter = FALSE)
  mp3 <- medpolish(as.matrix(d[c("c1", "c2", "c4")]), trace.iter = FALSE)
  e2 <- tryCatch({ medpolish(as.matrix(d[paste0("c", 1:4)]), trace.iter = FALSE); "no error" }, error = conditionMessage)
  stopifnot(e2 != "no error")
  list(values = list(isoreg_yf = vec(ir$yf), isoreg_x_sorted = vec(sort(d$x)), isoreg_n_knots = length(ir$iKnots),
                     isoreg_knots_r = vec(ir$iKnots),
                     mp_overall = mp$overall, mp_row = vec(mp$row), mp_col = named(mp$col),
                     mp3_overall = mp3$overall, mp3_row = vec(mp3$row), mp3_col = named(mp3$col)),
       reference = "stats::isoreg(x, y); stats::medpolish(na.rm = TRUE | FALSE), eps = 0.01, maxiter = 10",
       call = "isoreg(x, y); medpolish(cbind(c1, c2, c3, c4), na.rm = TRUE); medpolish(cbind(c1, c2, c4))",
       packages = character(0))
})

# Chi-squared test for trend in proportions on integer count columns. Group 4 has a missing
# success count; R's prop.trend.test refuses it ("0 (non-NA) cases"), the MCP tool drops the
# group and keeps every other group's score (default: its row position, 1, 2, 3, 5, 6, 7).
# trials_neg and successes_frac exist only for the tool's count checks.
case("stat_prop_trend_na", "stats/proportions", "prop.trend.test on integer count columns with a missing group", function(ctx) {
  d <- ctx$gen_dataset("prop_trend_na", data.frame(
    group = 1:7, dose = c(0, 1, 2, 4, 8, 16, 32),
    successes = c(3, 5, 9, NA, 14, 18, 23), trials = c(40, 42, 41, 39, 40, 43, 41),
    trials_neg = c(40, 42, 41, 39, -1, 43, 41), successes_frac = c(3, 5.5, 9, 11, 14, 18, 23)))
  stopifnot(is.integer(d$successes), is.integer(d$trials), is.integer(d$trials_neg), is.double(d$successes_frac))
  e <- tryCatch({ prop.trend.test(d$successes, d$trials); "no error" }, error = conditionMessage)
  stopifnot(e == "0 (non-NA) cases")
  cc <- complete.cases(d$successes, d$trials)
  pt <- prop.trend.test(d$successes[cc], d$trials[cc], score = which(cc))
  pd <- prop.trend.test(d$successes[cc], d$trials[cc], score = d$dose[cc])
  list(values = list(chisq = unname(pt$statistic), df = unname(pt$parameter), p = pt$p.value,
                     scores = vec(which(cc)),
                     dose_chisq = unname(pd$statistic), dose_p = pd$p.value, dose_scores = vec(d$dose[cc]),
                     n_used = sum(cc), n_total = nrow(d)),
       reference = "stats::prop.trend.test on the complete groups, score = the groups' own scores",
       call = "prop.trend.test(successes[cc], trials[cc], score = which(cc) | dose[cc])", packages = character(0))
})
