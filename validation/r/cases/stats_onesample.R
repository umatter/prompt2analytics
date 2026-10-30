# validation/r/cases/stats_onesample.R -- shapiro.test and the one/two-sample test wrappers
# (T2-WP06: STAT-1, the X-2/X-8/TOOL-9 sites in ttest.rs, ks.rs, vartest.rs, mood.rs,
# fligner.rs, bartlett.rs, mahalanobis.rs, pairwise.rs, cortest.rs).

# Shapiro-Wilk: one column per sample size, padded with NA to a common length. The
# id column keeps every CSV line non-empty (a line of only empty fields is skipped by R).
case("stat_shapiro_wilk", "stats/normality", "Shapiro-Wilk W and p-value, n = 3 to 5000", function(ctx) {
  set.seed(42)
  cols <- list(
    n3_124 = c(1, 2, 4),
    n3_123 = c(1, 2, 3),
    n4 = c(2.1, 3.4, 1.9, 5.6),
    n5 = c(1.2, 2.3, 3.1, 4.5, 5.2),
    n8_rnorm = round(rnorm(8), 6),
    n10_unif = (1:10) / 10,
    n11_rnorm = round(rnorm(11, 5, 2), 6),
    n12_rnorm = round(rnorm(12), 6),
    n15_exp = round(rexp(15), 6),
    n20_rnorm = round(rnorm(20), 6),
    n30_exp = round(rexp(30), 6),
    n50_rnorm = round(rnorm(50), 6),
    n100_qnorm = qnorm(ppoints(100)),
    n200_exp = round(rexp(200), 6),
    n40_ties = round(rnorm(40) * 2) / 2,
    n25_int = as.numeric(round(rnorm(25, 50, 10))),     # written as integers: an Int64 column for polars
    n5000_rnorm = round(rnorm(5000), 6)
  )
  len <- max(lengths(cols))
  df <- data.frame(id = seq_len(len))
  for (k in names(cols)) { v <- rep(NA_real_, len); v[seq_along(cols[[k]])] <- cols[[k]]; df[[k]] <- v }
  d <- ctx$gen_dataset("shapiro_cases", df)
  stopifnot(is.integer(d$n25_int))
  v <- list()
  for (k in names(cols)) {
    s <- shapiro.test(d[[k]])            # shapiro.test drops NA itself (x[complete.cases(x)])
    v[[paste0(k, "_w")]] <- unname(s$statistic)
    v[[paste0(k, "_p")]] <- s$p.value
    v[[paste0(k, "_n")]] <- sum(!is.na(d[[k]]))
  }
  list(values = v, reference = "stats::shapiro.test (swilk.c, Royston 1995 AS R94)",
       call = "shapiro.test(d[[col]])", packages = character(0))
})

# t-tests, F test, Mood, KS, Fligner, Bartlett, cor.test on data with missing cells and integer columns.
case("stat_two_sample_na", "stats/two_sample", "t, var, Mood, KS, Fligner, Bartlett and cor tests with NA and integer columns", function(ctx) {
  x <- c(5.1, 6.2, NA, 7.3, 5.9, 6.8, 7.7, 6.1)
  y <- c(4.0, NA, 5.5, 6.1, 4.8, 5.2, 6.6, 5.0)
  set.seed(3)
  xi <- as.numeric(c(round(rnorm(20, 50, 8)), NA, NA))        # Int64 in polars, with nulls
  yi <- as.numeric(c(round(rnorm(18, 45, 12)), NA, NA, NA, NA))
  g <- c(rep(c("a", "b", "c"), length.out = 21), NA)
  # A tie-free integer pair for Mood: mood.test scores ties as Mielke (1967) and p2a does not yet
  # (STAT-29, Tier 4), so the tied xi/yi Mood values are a registry deviation and this pair pins
  # Mood's missing-value and Int64 handling. Drawn after xi/yi so their values are unchanged.
  set.seed(17)
  mi <- c(sample(c(10:30, 70:99), 15), rep(NA, 7))            # wide spread, 15 values
  mj <- c(NA, sample(31:69, 12), rep(NA, 9))                  # narrow spread, 12 values, disjoint from mi
  d <- ctx$gen_dataset("two_sample_na", data.frame(
    id = 1:22, x = c(x, rep(NA, 14)), y = c(y, rep(NA, 14)), xi = xi, yi = yi, g = g, mi = mi, mj = mj))
  stopifnot(is.integer(d$xi), is.integer(d$mi), is.integer(d$mj),
            !anyDuplicated(c(d$mi[!is.na(d$mi)], d$mj[!is.na(d$mj)])))
  one <- t.test(d$x); two <- t.test(d$x, d$y); two_eq <- t.test(d$x, d$y, var.equal = TRUE)
  pr <- t.test(d$x, d$y, paired = TRUE)
  ti <- t.test(d$xi, d$yi)
  vt <- var.test(d$xi, d$yi); mt <- mood.test(d$xi, d$yi)
  mt_nt <- mood.test(d$mi, d$mj)
  ks2 <- suppressWarnings(ks.test(d$xi, d$yi))
  fl <- fligner.test(xi ~ g, data = d); bt <- bartlett.test(xi ~ g, data = d)
  ct <- cor.test(d$x, d$y)
  okg <- complete.cases(d$xi, d$g)
  gvar <- tapply(d$xi[okg], d$g[okg], var); gn <- tapply(d$xi[okg], d$g[okg], length)
  # Cross-check values (not reported by the R functions; p2a reports them): Mood's T for the
  # tie-free pair, the group variances and pooled variance of Bartlett, the group sizes.
  mz <- c(d$mi[!is.na(d$mi)], d$mj[!is.na(d$mj)]); mN <- length(mz); mm <- sum(!is.na(d$mi))
  list(values = list(
    one_t = unname(one$statistic), one_df = unname(one$parameter), one_p = one$p.value,
    one_ci = vec(one$conf.int), one_mean = unname(one$estimate), one_se = one$stderr,
    welch_t = unname(two$statistic), welch_df = unname(two$parameter), welch_p = two$p.value, welch_ci = vec(two$conf.int),
    welch_est = vec(two$estimate), welch_se = two$stderr,
    pooled_t = unname(two_eq$statistic), pooled_df = unname(two_eq$parameter), pooled_p = two_eq$p.value,
    pooled_ci = vec(two_eq$conf.int), pooled_se = two_eq$stderr,
    paired_t = unname(pr$statistic), paired_df = unname(pr$parameter), paired_p = pr$p.value, paired_ci = vec(pr$conf.int),
    paired_est = unname(pr$estimate), paired_se = pr$stderr,
    int_welch_t = unname(ti$statistic), int_welch_df = unname(ti$parameter), int_welch_p = ti$p.value,
    int_welch_ci = vec(ti$conf.int), int_welch_est = vec(ti$estimate), int_welch_se = ti$stderr,
    var_f = unname(vt$statistic), var_df = vec(unname(vt$parameter)), var_p = vt$p.value, var_ci = vec(vt$conf.int),
    var_est = unname(vt$estimate), var_x = var(d$xi, na.rm = TRUE), var_y = var(d$yi, na.rm = TRUE),
    mood_z = unname(mt$statistic), mood_p = mt$p.value,
    mood_nt_z = unname(mt_nt$statistic), mood_nt_p = mt_nt$p.value,
    mood_nt_T = sum((rank(mz)[seq_len(mm)] - (mN + 1) / 2)^2),
    ks_d = unname(ks2$statistic), ks_p = ks2$p.value,
    fligner_chisq = unname(fl$statistic), fligner_df = unname(fl$parameter), fligner_p = fl$p.value,
    bartlett_k2 = unname(bt$statistic), bartlett_df = unname(bt$parameter), bartlett_p = bt$p.value,
    bartlett_group_var = named(gvar), bartlett_pooled_var = sum(gvar * (gn - 1)) / (sum(gn) - length(gn)),
    group_n = named(gn),
    cor_r = unname(ct$estimate), cor_t = unname(ct$statistic), cor_df = unname(ct$parameter), cor_p = ct$p.value,
    cor_ci = vec(ct$conf.int),
    n_x = sum(!is.na(d$x)), n_y = sum(!is.na(d$y)), n_pairs = sum(complete.cases(d$x, d$y)),
    n_xi = sum(!is.na(d$xi)), n_yi = sum(!is.na(d$yi)), n_xi_g = sum(complete.cases(d$xi, d$g)),
    n_mi = sum(!is.na(d$mi)), n_mj = sum(!is.na(d$mj))),
    reference = "stats::t.test, var.test, mood.test, ks.test, fligner.test, bartlett.test, cor.test (NA removed as R does)",
    call = "t.test(x); t.test(x, y); t.test(x, y, paired = TRUE); var.test(xi, yi); mood.test(mi, mj); ...", packages = character(0))
})

# Pairwise t-tests and Mahalanobis distances on integer columns with missing cells.
case("stat_pairwise_mahalanobis_na", "stats/multi_sample", "pairwise.t.test and mahalanobis with NA and integer columns", function(ctx) {
  set.seed(9)
  n <- 30
  g <- rep(c("g1", "g2", "g3"), each = 10)
  y <- round(rnorm(n, mean = rep(c(10, 12, 15), each = 10), sd = 2))
  y[c(4, 18)] <- NA
  a <- round(rnorm(n, 5, 1), 3); b <- round(a * 0.5 + rnorm(n), 3)
  a[7] <- NA
  d <- ctx$gen_dataset("pairwise_mahalanobis_na", data.frame(id = 1:n, y = y, g = g, a = a, b = b))
  stopifnot(is.integer(d$y))
  pw <- pairwise.t.test(d$y, d$g, p.adjust.method = "holm")
  pw_raw <- pairwise.t.test(d$y, d$g, p.adjust.method = "none")
  ok <- complete.cases(d$y, d$g)
  gm <- tapply(d$y[ok], d$g[ok], mean); gn <- tapply(d$y[ok], d$g[ok], length); gs <- tapply(d$y[ok], d$g[ok], sd)
  psd <- sqrt(sum(gs^2 * (gn - 1)) / sum(gn - 1))
  # Cross-check values (pairwise.t.test reports p-values only; p2a also reports t, df, sizes, means, pooled SD).
  pair_t <- c(gm[2] - gm[1], gm[3] - gm[1], gm[3] - gm[2]) /
    (psd * sqrt(c(1 / gn[2] + 1 / gn[1], 1 / gn[3] + 1 / gn[1], 1 / gn[3] + 1 / gn[2])))
  cc <- complete.cases(d$a, d$b)
  X <- as.matrix(d[cc, c("a", "b")])
  md <- mahalanobis(X, colMeans(X), cov(X))
  list(values = list(pairwise_p = vec(pw$p.value[lower.tri(pw$p.value, diag = TRUE)]),
                     pairwise_p_raw = vec(pw_raw$p.value[lower.tri(pw_raw$p.value, diag = TRUE)]),
                     pairwise_group_n = named(gn), pairwise_group_mean = named(gm),
                     pairwise_pooled_sd = psd, pairwise_t = vec(pair_t), pairwise_df = sum(gn - 1),
                     mahalanobis_complete = vec(md), mahalanobis_rows = vec(which(cc) - 1),
                     mahalanobis_center = named(colMeans(X)), n_complete = sum(cc)),
       reference = "stats::pairwise.t.test(p.adjust = 'holm' | 'none'); stats::mahalanobis(X, colMeans(X), cov(X)) on complete rows",
       call = "pairwise.t.test(y, g); mahalanobis(X, colMeans(X), cov(X))", packages = character(0))
})
