# validation/r/cases/stats_descriptive_t4.R -- references for T4-WP04
# (STAT-20 Pearson one-sided intervals; STAT-22 Kendall tau-b, exact T and tie-corrected z;
#  STAT-47 Spearman exact / Edgeworth (AS 89) p and S; STAT-19 pairwise-complete correlation
#  matrix; STAT-48 describe quantiles (type 7); STAT-37 / STAT-65 density grid, kernels scaled
#  so that bw is the kernel's standard deviation, bw.nrd0).
#
# The density references are R's own binned approximation (density.default: linear binning by
# BinDist on 2^k >= max(n, 512) points over [from - 4 bw, to + 4 bw], FFT convolution, approx()
# onto the requested grid). `kde_*` keys hold the exact kernel density estimate at the same x,
# mean(K((x0 - x) / bw)) / bw with R's kernels, for a path that does not bin.
# ---------------------------------------------------------------------------------------------

.ct <- function(...) suppressWarnings(cor.test(...))
.ci <- function(r) vec(unname(r$conf.int))
.rkern <- function(kernel, t, bw) switch(kernel,
  gaussian = dnorm(t, sd = bw),
  epanechnikov = { a <- bw * sqrt(5); ifelse(abs(t) < a, 3/4 * (1 - (t / a)^2) / a, 0) },
  rectangular = { a <- bw * sqrt(3); ifelse(abs(t) < a, 0.5 / a, 0) },
  triangular = { a <- bw * sqrt(6); ifelse(abs(t) < a, (1 - abs(t) / a) / a, 0) },
  biweight = { a <- bw * sqrt(7); ifelse(abs(t) < a, 15/16 * (1 - (t / a)^2)^2 / a, 0) },
  cosine = { a <- bw / sqrt(1/3 - 2/pi^2); ifelse(abs(t) < a, (1 + cos(pi * t / a)) / (2 * a), 0) },
  optcosine = { a <- bw / sqrt(1 - 8/pi^2); ifelse(abs(t) < a, pi/4 * cos(pi * t / (2 * a)) / a, 0) })
.kde <- function(x, bw, kernel, at) vapply(at, function(a) mean(.rkern(kernel, a - x, bw)), 0)

case("stat_cor_test_t4", "stats/cortest", "cor.test: Pearson one-sided intervals; Spearman exact (n <= 9), Edgeworth (n < 1290) and t (ties, n >= 1290) p-values and S; Kendall tau-b, exact T (n < 50, no ties) and tie-corrected z", function(ctx) {
  set.seed(20261113)
  p2x <- round(rnorm(10), 3); p2y <- round(0.4 * p2x + rnorm(10), 3)
  s8x <- round(rnorm(8), 3); s8y <- round(0.5 * s8x + rnorm(8), 3)
  s40x <- round(rnorm(40), 4); s40y <- round(0.25 * s40x + rnorm(40), 4)
  s1300x <- round(rnorm(1300), 8); s1300y <- round(0.05 * s1300x + rnorm(1300), 8)
  k60x <- round(rnorm(60), 4); k60y <- round(0.3 * k60x + rnorm(60), 4)
  t60x <- round(rnorm(60)); t60y <- round(0.5 * t60x + rnorm(60))
  kx <- c(0.1603, 0.5681, 0.5066, 1.0145, 0.6190, -0.6551, -0.7773, 0.4089, 1.7813, 0.8909, -0.3686,
          -0.3427, 0.2088, -1.1010, -1.2737, 0.5803, -1.0766, 2.2568, -0.6514, 0.1311, -1.1748,
          -0.5524, 1.6524, 0.8165, 1.4044, -0.5720, -0.5585, 0.5073, 0.8963, 1.1749, 0.2317, -0.3034,
          0.6418, -0.3841, 0.1446, 0.8489, 0.8137, -0.4105, 0.8105, -1.3586)
  ky <- c(0.2247, -0.3042, 0.5043, 1.4535, -0.1108, -0.0848, 0.5737, -0.8745, 1.5106, -0.3778,
          -0.1653, 0.0670, 1.1093, -2.0171, 0.3239, 1.2860, -1.4404, 1.2230, 0.4424, 0.1354, -1.5561,
          -0.7617, 1.4101, -0.5054, 1.5702, -0.0184, -0.0037, -0.4206, 0.3890, -0.9830, 0.5311,
          0.9519, 1.6406, -1.0129, -0.1247, -0.0295, 0.1230, -0.3533, -0.9709, 0.0493)
  sets <- list(
    pear  = list(x = 1:10, y = c(1.2, 2.1, 2.9, 4.0, 5.1, 5.9, 7.2, 7.8, 9.1, 10.0)),   # r > 0.7616
    pear2 = list(x = p2x, y = p2y),
    perm  = list(x = 1:10, y = c(3, 1, 2, 6, 4, 5, 9, 10, 7, 8)),                       # finding repro
    s8    = list(x = s8x, y = s8y),                                                     # Spearman exact enumeration
    s40   = list(x = s40x, y = s40y),                                                   # Edgeworth
    s1300 = list(x = s1300x, y = s1300y),                                               # t (n >= 1290)
    perf7 = list(x = c(0.3, 1.1, 1.9, 2.4, 3.8, 4.2, 5.7), y = c(10.2, 11.5, 11.9, 13.4, 14.1, 15.8, 16.3)),
    perf12 = list(x = c(0.21, 0.55, 0.93, 1.48, 1.62, 2.05, 2.77, 3.14, 3.59, 4.02, 4.48, 5.13),
                  y = c(1.3, 2.2, 2.6, 3.9, 4.1, 5.5, 5.8, 6.4, 7.7, 8.2, 8.9, 9.6)),
    perft = list(x = c(1, 1, 2, 3, 4, 4, 5), y = c(2, 2, 4.5, 6, 8.1, 8.1, 9)),         # ties, rho = tau = 1
    tie12 = list(x = c(1, 2, 2, 3, 3, 3, 4, 4, 5, 2, 3, 1), y = c(3, 3, 4, 4, 4, 5, 5, 2, 3, 4, 5, 5)),
    kd40  = list(x = kx, y = ky),                                                       # Kendall exact (n < 50)
    k60   = list(x = k60x, y = k60y),                                                   # Kendall z, no ties
    t60   = list(x = t60x, y = t60y))                                                   # ties
  long <- do.call(rbind, lapply(names(sets), function(s)
    data.frame(set = s, x = as.numeric(sets[[s]]$x), y = as.numeric(sets[[s]]$y))))
  d <- ctx$gen_dataset("cor_test_t4", long)
  g <- function(s) list(x = d$x[d$set == s], y = d$y[d$set == s])
  nod <- function(s) { v <- g(s); !anyDuplicated(v$x) && !anyDuplicated(v$y) }
  stopifnot(nod("pear"), nod("pear2"), nod("perm"), nod("s8"), nod("s40"), nod("s1300"), nod("perf7"),
            nod("perf12"), nod("kd40"), nod("k60"), !nod("perft"), !nod("tie12"), !nod("t60"),
            cor(g("pear")$x, g("pear")$y) > tanh(1))
  out <- list()
  alts <- c(two = "two.sided", less = "less", greater = "greater")
  for (s in c("pear", "pear2")) {
    v <- g(s)
    for (a in names(alts)) {
      r <- .ct(v$x, v$y, method = "pearson", alternative = alts[[a]])
      out[[sprintf("pearson_%s_%s_p", s, a)]] <- r$p.value
      out[[sprintf("pearson_%s_%s_ci", s, a)]] <- .ci(r)
      if (a == "two") {
        out[[sprintf("pearson_%s_r", s)]] <- unname(r$estimate)
        out[[sprintf("pearson_%s_t", s)]] <- unname(r$statistic)
        out[[sprintf("pearson_%s_ci90", s)]] <- .ci(.ct(v$x, v$y, conf.level = 0.9))
      }
    }
  }
  for (s in c("perm", "s8", "s40", "s1300", "perf7", "perf12", "perft", "tie12")) {
    v <- g(s)
    for (a in names(alts)) {
      r <- .ct(v$x, v$y, method = "spearman", alternative = alts[[a]])
      out[[sprintf("spearman_%s_%s_p", s, a)]] <- r$p.value
      if (a == "two") { out[[sprintf("spearman_%s_rho", s)]] <- unname(r$estimate)
                        out[[sprintf("spearman_%s_S", s)]] <- unname(r$statistic) }
    }
  }
  for (s in c("perm", "kd40", "k60", "perf7", "perf12", "perft", "tie12", "t60")) {
    v <- g(s)
    for (a in names(alts)) {
      r <- .ct(v$x, v$y, method = "kendall", alternative = alts[[a]])
      out[[sprintf("kendall_%s_%s_p", s, a)]] <- r$p.value
      if (a == "two") { out[[sprintf("kendall_%s_tau", s)]] <- unname(r$estimate)
                        out[[sprintf("kendall_%s_stat", s)]] <- unname(r$statistic)
                        out[[sprintf("kendall_%s_is_T", s)]] <- as.numeric(names(r$statistic) == "T") }
    }
  }
  out$n_s1300 <- length(g("s1300")$x)
  list(values = out,
       reference = "stats::cor.test (R's defaults: Spearman exact = n < 1290 without ties, AS 89 enumeration for n <= 9 and Edgeworth series above; Kendall exact = n < 50 without ties, tau-b and the tie-corrected variance with continuity = FALSE otherwise)",
       call = "cor.test(x, y, method = m, alternative = a) per set; Pearson also conf.level = 0.9",
       packages = character(0))
})

case("stat_cor_matrix_t4", "stats/correlation", "Pearson correlation matrix of the numeric columns with missing values: pairwise-complete (and complete-case) coefficients and pair counts", function(ctx) {
  set.seed(20261114)
  n <- 40
  a <- round(rnorm(n, 10, 2), 3); b <- round(0.6 * a + rnorm(n, 0, 1.5), 3)
  cc <- round(-0.3 * a + 0.2 * b + rnorm(n), 3); i <- as.integer(round(5 + 0.8 * b + rnorm(n, 0, 2)))
  a[c(3, 17)] <- NA; b[c(8, 25, 33)] <- NA; cc[c(3, 40)] <- NA; i[c(11, 12)] <- NA
  lab <- rep(c("u", "v"), length.out = n)
  d <- ctx$gen_dataset("cor_matrix_t4", data.frame(id = 1:n, a = a, b = b, lab = lab, c = cc, i = i))
  small <- ctx$gen_dataset("cor_matrix_small_t4", data.frame(id = 1:5, a = c(1, 2, 3, 4, 100), b = c(2, 1, 4, 3, NA)))
  stopifnot(is.integer(d$i), is.double(d$a), is.integer(small$b))
  m <- as.matrix(d[, c("a", "b", "c", "i")])
  pw <- cor(m, use = "pairwise.complete.obs"); co <- cor(m, use = "complete.obs")
  np <- crossprod(!is.na(m))
  list(values = list(
    pairwise = vec(as.vector(t(pw))), complete = vec(as.vector(t(co))), pair_n = vec(as.vector(t(np))),
    complete_n = sum(complete.cases(m)), n_rows = n,
    small_ab = cor(small$a, small$b, use = "pairwise.complete.obs")),
    reference = "stats::cor(use = 'pairwise.complete.obs') and cor(use = 'complete.obs'); pair counts crossprod(!is.na(m))",
    call = "cor(d[, c('a', 'b', 'c', 'i')], use = 'pairwise.complete.obs'); row-major 4 x 4",
    dataset = "validation/datasets/generated/cor_matrix_t4.csv",
    packages = character(0))
})

case("stat_describe_t4", "stats/descriptive", "Descriptive statistics of numeric columns: type-7 quartiles and median, mean, sd, min, max over the present values", function(ctx) {
  set.seed(20261115)
  n <- 12
  e <- round(rnorm(n, 50, 10), 2)                        # even n, no missing
  o <- round(rexp(n, 0.2), 3); o[7] <- NA                # 11 present
  ni <- as.integer(round(rnorm(n, 100, 15))); ni[c(2, 9)] <- NA
  d <- ctx$gen_dataset("describe_t4", data.frame(id = 1:n, e = e, o = o, ni = ni, g = rep(c("p", "q", "r"), 4)))
  stopifnot(is.integer(d$ni))
  st <- function(v) { v <- v[!is.na(v)]
    vec(c(length(v), mean(v), sd(v), min(v), unname(quantile(v, c(0.25, 0.5, 0.75), type = 7)), max(v))) }
  list(values = list(e = st(d$e), o = st(d$o), ni = st(d$ni), four = vec(quantile(c(1, 2, 3, 4), c(0.25, 0.5, 0.75), type = 7))),
       reference = "base mean, sd, min, max and stats::quantile(type = 7) over the non-missing values (summary() quartiles)",
       call = "c(n, mean, sd, min, q25, median, q75, max) per column, na.rm",
       packages = character(0))
})

case("stat_density_t4", "stats/robust", "density(): grids that are not powers of two, every kernel with bw as its standard deviation, bw.nrd0 with its fallbacks", function(ctx) {
  set.seed(1)
  big <- round(rnorm(200), 5)
  sets <- list(big = big, x9 = c(1, 2, 3.5, 4, 4.2, 5, 6.5, 7, 9), ties = c(1:4, rep(5, 7), 6:9),
               iqr0 = c(rep(3, 20), 1, 10), const = rep(3, 5), zero = c(0, 0, 0),
               dx = c(1.21, 0.37, -0.45, 2.08, 1.66, 0.92, -1.13, 0.05, 1.48, 2.61, 0.74, -0.28, 1.02, 3.05,
                      0.51, -0.66, 1.87, 1.33, 0.19, 2.34, -0.91, 0.88, 1.55, 0.62, 2.79, -0.12, 1.11, 0.43,
                      1.94, 0.27))                                  # robust.rs DX (30 values)
  long <- do.call(rbind, lapply(names(sets), function(s) data.frame(set = s, x = as.numeric(sets[[s]]))))
  d <- ctx$gen_dataset("density_t4", long)
  g <- function(s) d$x[d$set == s]
  out <- list()
  b <- g("big")
  for (nn in c(100, 512, 1000)) {
    r <- density(b, bw = 0.4, n = nn)
    out[[sprintf("big_n%d_x", nn)]] <- vec(r$x); out[[sprintf("big_n%d_y", nn)]] <- vec(r$y)
  }
  r <- density(b, bw = 0.4, n = 100, from = -3, to = 3)
  out$big_ft_x <- vec(r$x); out$big_ft_y <- vec(r$y); out$kde_big_ft_y <- vec(.kde(b, 0.4, "gaussian", r$x))
  for (k in c("epanechnikov", "rectangular", "triangular", "biweight", "cosine", "optcosine")) {
    r <- density(b, bw = 0.4, n = 100, kernel = k)
    out[[sprintf("big_%s_y", k)]] <- vec(r$y)
  }
  r <- density(b); out$big_default_bw <- r$bw; out$big_default_y <- vec(r$y)
  x9 <- g("x9")
  for (k in c("gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine", "optcosine")) {
    r <- density(x9, bw = 0.5, n = 7, kernel = k)
    out[[sprintf("x9_%s_y", k)]] <- vec(r$y); out[[sprintf("kde_x9_%s_y", k)]] <- vec(.kde(x9, 0.5, k, r$x))
  }
  out$x9_x <- vec(density(x9, bw = 0.5, n = 7)$x)
  for (s in c("big", "x9", "ties", "iqr0", "const", "zero")) out[[sprintf("nrd0_%s", s)]] <- bw.nrd0(g(s))
  r <- density(g("iqr0")); out$iqr0_default_x <- vec(r$x); out$iqr0_default_y <- vec(r$y)
  dx <- g("dx")
  r <- density(dx, bw = 0.4, n = 50); out$dx_n50_x <- vec(r$x); out$dx_n50_y <- vec(r$y)
  out$kde_dx_n50_y <- vec(.kde(dx, 0.4, "gaussian", r$x))
  r <- density(c(dx, Inf, -Inf), bw = 0.4, n = 50); out$dx_inf_n50_y <- vec(r$y); out$dx_inf_n <- r$n
  r <- density(dx, bw = 0.4, n = 50, kernel = "epanechnikov"); out$dx_epan_n50_y <- vec(r$y)
  list(values = out,
       reference = "stats::density.default (R 4.3.3: BinDist linear binning, FFT, approx) and stats::bw.nrd0; kde_* are the exact kernel sums with R's kernels",
       call = "density(x, bw, n, kernel, from, to); bw.nrd0(x)",
       packages = character(0))
})

case("stat_density_mcp_na_t4", "stats/robust", "density() of the stats_density tool's defaults on the descriptive_na columns (missing values dropped)", function(ctx) {
  d <- ctx$dataset("datasets/generated/descriptive_na.csv")
  out <- list()
  for (col in c("v", "vi")) {
    v <- d[[col]][!is.na(d[[col]])]
    r <- density(v, n = 512)
    out[[sprintf("dens_%s_bw", col)]] <- r$bw; out[[sprintf("dens_%s_x", col)]] <- vec(r$x)
    out[[sprintf("dens_%s_y", col)]] <- vec(r$y)
  }
  list(values = out, reference = "stats::density(na.omit(x), n = 512) with bw.nrd0",
       call = "density(v[!is.na(v)], n = 512); the same for vi", packages = character(0))
})
