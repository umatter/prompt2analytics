# validation/r/cases/data_viz_t3.R -- chart statistics and export-table references (DATA-61,
# DATA-86, DATA-24 core, DATA-62, DATA-85).
#  * stats::cor on a well-scaled predictor, the same predictor shifted by 1.7e9 (epoch seconds) and
#    by 1e12, and both variables scaled by 1e-9 (the scatter-plot correlation, DATA-61);
#  * grDevices::boxplot.stats (coef = 1.5, fivenum hinges) per group, including n mod 4 in {0, 3},
#    outliers in data order, a constant group and an all-missing group (the box plot, DATA-62);
#  * grDevices::nclass.Sturges and graphics::hist(x, breaks = seq(min, max, length.out = k + 1)) counts
#    with its defaults: p2a's equal-width bins from min to max, right-closed with the lowest value
#    included and R's fuzz at the breaks (the histogram, DATA-85);
#  * stats::rstandard / cooks.distance / hatvalues for lm(mpg ~ wt + hp + qsec) and lm(mpg ~ wt) on
#    mtcars and for a model with one leverage-one observation; qnorm(ppoints(n)) for the Q-Q plot
#    (residual diagnostics, DATA-24 core, X-14 part);
#  * stats::lm on Grunfeld, two models, for the LaTeX/HTML/Markdown table exporters (DATA-86).

case("viz_scatter_cor", "visualization", "Scatter-plot Pearson correlation at large offsets and tiny scale (stats::cor)", function(ctx) {
  set.seed(52061)
  n <- 200
  x0 <- 1:n + round(runif(n), 3)
  y <- 3 + 0.25 * x0 + rnorm(n, sd = 8)
  d <- ctx$gen_dataset("viz_scatter_cor", data.frame(
    x0 = x0, x_epoch = 1.7e9 + x0, x_big = 1e12 + x0, y = y,
    x_tiny = x0 * 1e-9, y_tiny = y * 1e-9, y_const = rep(5.25, n)))
  stopifnot(nrow(d) == n, all(d$y_const == 5.25))
  list(values = list(
    n = nrow(d),
    cor_x0 = cor(d$x0, d$y),
    cor_epoch = cor(d$x_epoch, d$y),
    cor_big = cor(d$x_big, d$y),
    cor_tiny = cor(d$x_tiny, d$y_tiny),
    cor_const = suppressWarnings(cor(d$x0, d$y_const))),   # NA: zero standard deviation
    reference = "stats::cor (method = 'pearson')",
    call = "cor(x, y) for x in x0, x0 + 1.7e9, x0 + 1e12 and (x0, y) * 1e-9; cor(x0, rep(5.25, n))",
    packages = character(0))
})

case("viz_boxplot_groups", "visualization", "Box-plot statistics per group (grDevices::boxplot.stats, coef = 1.5)", function(ctx) {
  set.seed(5262)
  norm37 <- round(rnorm(35, 50, 5), 2)
  norm37 <- append(norm37, 85.5, after = 9)      # position 10
  norm37 <- append(norm37, 12.25, after = 29)    # position 30
  groups <- list(
    data62 = c(1, 7, 12, 15, 15, 16, 18, 18),    # DATA-62's example
    n4 = c(3, 1, 4, 2),                          # n mod 4 = 0
    n12 = c(32, 1:9, 30, 31),                    # n mod 4 = 0, three high values
    n7 = c(2.5, 9.75, 4, 4, 6.5, 1, 30),         # n mod 4 = 3, one outlier
    norm37 = norm37,                             # n mod 4 = 1, outliers in data order
    const3 = c(5, 5, 5),
    allna = c(NA, NA))
  long <- data.frame(group = rep(names(groups), lengths(groups)), value = unlist(groups, use.names = FALSE),
                     stringsAsFactors = FALSE)
  d <- ctx$gen_dataset("viz_boxplot_groups", long)
  v <- list()
  for (g in names(groups)) {
    b <- boxplot.stats(d$value[d$group == g])
    # allna: the stats are NA; not stored, because run.R's comparator cannot hold an all-null vector
    if (b$n > 0) v[[paste0(g, "_stats")]] <- vec(b$stats)
    v[[paste0(g, "_n")]] <- b$n
    v[[paste0(g, "_n_out")]] <- length(b$out)
    if (length(b$out)) v[[paste0(g, "_out")]] <- vec(b$out)
  }
  # +-Inf count in n and the hinges and are outliers; values only, a CSV cannot carry the Inf through R's reader
  b <- boxplot.stats(c(1, 2, 3, 4, 5, Inf))
  v$inf6_stats <- vec(b$stats)
  v$inf6_n <- b$n
  v$inf6_n_out <- length(b$out)
  stopifnot(v$data62_n_out == 0, v$n7_n_out == 1, v$norm37_n_out == 2, v$allna_n == 0, v$inf6_n_out == 1)
  list(values = v, reference = "grDevices::boxplot.stats (fivenum hinges, coef = 1.5)",
       call = "boxplot.stats(value[group == g]) for each group", packages = character(0))
})

case("viz_histogram_bins", "visualization", "Histogram bin count (nclass.Sturges) and equal-width bin counts", function(ctx) {
  set.seed(5285)
  norm100 <- rnorm(100, 10, 2)
  small50 <- 3e-11 + 1e-11 * rnorm(50)           # range ~ 4e-11: below the old absolute 1e-10 cut-off
  # a decimal grid, values on the breaks up to rounding, and two values just off a break
  grid13 <- as.numeric(c(sprintf("%.1f", 0:10 / 10), "0.400000000001", "0.799999999999"))
  long <- data.frame(sample = rep(c("norm100", "small50", "grid13"), c(100, 50, 13)),
                     value = c(norm100, small50, grid13), stringsAsFactors = FALSE)
  d <- ctx$gen_dataset("viz_histogram_samples", long)
  v <- list(sturges_n = vec(c(1, 5, 8, 16, 100, 1000, 1024)))
  v$sturges_k <- vec(sapply(v$sturges_n, function(n) nclass.Sturges(seq_len(n))))
  for (s in c("norm100", "small50", "grid13")) {
    x <- d$value[d$sample == s]
    k <- nclass.Sturges(x)
    br <- seq(min(x), max(x), length.out = k + 1)
    h <- hist(x, breaks = br, plot = FALSE)
    v[[paste0(s, "_k")]] <- k
    v[[paste0(s, "_breaks")]] <- vec(br)
    v[[paste0(s, "_counts")]] <- vec(h$counts)
  }
  list(values = v, reference = "grDevices::nclass.Sturges; graphics::hist(breaks = seq(min, max, length.out = k + 1)) with its defaults (right = TRUE, include.lowest = TRUE)",
       call = "k <- nclass.Sturges(x); hist(x, breaks = seq(min(x), max(x), length.out = k + 1), plot = FALSE)$counts",
       packages = character(0))
})

case("viz_residual_diag", "visualization", "Standardized residuals and Cook's distance (stats::rstandard, cooks.distance), mtcars", function(ctx) {
  mt <- datasets::mtcars
  m3 <- lm(mpg ~ wt + hp + qsec, data = mt)
  m1 <- lm(mpg ~ wt, data = mt)
  mt$d1 <- as.numeric(seq_len(nrow(mt)) == 1)
  ml <- lm(mpg ~ wt + d1, data = mt)             # observation 1 has leverage one
  d <- ctx$gen_dataset("viz_residual_diag", data.frame(
    fitted3 = unname(fitted(m3)), resid3 = unname(resid(m3)), hat3 = unname(hatvalues(m3)),
    fitted1 = unname(fitted(m1)), resid1 = unname(resid(m1)), hat1 = unname(hatvalues(m1)),
    fitted_l1 = unname(fitted(ml)), resid_l1 = unname(resid(ml)), hat_l1 = unname(hatvalues(ml))))
  # the read-back columns are the model's own doubles
  stopifnot(identical(d$resid3, unname(resid(m3))), identical(d$hat3, unname(hatvalues(m3))),
            identical(d$hat_l1[1], 1), is.nan(rstandard(ml)[1]))
  list(values = list(
    n = nrow(d), p3 = m3$rank, p1 = m1$rank, p_l1 = ml$rank,
    sigma3 = summary(m3)$sigma, sigma1 = summary(m1)$sigma, sigma_l1 = summary(ml)$sigma,
    rstandard3 = vec(rstandard(m3)), cooks3 = vec(cooks.distance(m3)),
    rstandard1 = vec(rstandard(m1)), cooks1 = vec(cooks.distance(m1)),
    # NaN at observation 1 (JSON null); named() so that run.R --check compares the null position
    rstandard_l1 = named(rstandard(ml)), cooks_l1 = named(cooks.distance(ml)),
    qq_ppoints_1 = vec(qnorm(ppoints(1))), qq_ppoints_8 = vec(qnorm(ppoints(8))),
    qq_ppoints_11 = vec(qnorm(ppoints(11))), qq_ppoints_32 = vec(qnorm(ppoints(32)))),
    reference = "stats::rstandard.lm; stats::cooks.distance.lm; stats::hatvalues; stats::qnorm(stats::ppoints(n))",
    call = "lm(mpg ~ wt + hp + qsec, mtcars); lm(mpg ~ wt, mtcars); lm(mpg ~ wt + d1, mtcars) with d1 = 1 for row 1",
    packages = character(0))
})

