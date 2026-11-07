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

