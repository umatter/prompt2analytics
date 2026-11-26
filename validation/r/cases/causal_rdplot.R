# validation/r/cases/causal_rdplot.R -- rdrobust::rdplot references: the variance-mimicking evenly
# spaced bins (esmv, and esmvpr under mass points), bins set by hand, the global polynomial and
# the binned means. Uses the rd_sim dataset of causal_rd.R.
suppressPackageStartupMessages(library(rdrobust))

# rdplot's breaks come from seq(c, max(x), by = range / J), which can stop one rounding step
# short of max(x); findInterval then puts max(x) into an extra bin J + 1 with no break and no
# midpoint (NA), which rdplot does not draw. p2a closes the last bin at max(x) instead. The
# merged_* values fold each such bin into the bin before it (counts added, means weighted);
# the raw rdplot bins are kept as raw_*.
.rdplot_merge <- function(b) {
  b <- b[, c("rdplot_mean_bin", "rdplot_mean_x", "rdplot_mean_y", "rdplot_N", "rdplot_min_bin", "rdplot_max_bin")]
  for (i in rev(which(is.na(b$rdplot_mean_bin)))) {
    stopifnot(i > 1, !is.na(b$rdplot_mean_bin[i - 1]))
    n1 <- b$rdplot_N[i - 1]; n2 <- b$rdplot_N[i]
    b$rdplot_mean_x[i - 1] <- (n1 * b$rdplot_mean_x[i - 1] + n2 * b$rdplot_mean_x[i]) / (n1 + n2)
    b$rdplot_mean_y[i - 1] <- (n1 * b$rdplot_mean_y[i - 1] + n2 * b$rdplot_mean_y[i]) / (n1 + n2)
    b$rdplot_N[i - 1] <- n1 + n2
    b <- b[-i, ]
  }
  b
}

.rdplot_values <- function(r, prefix) {
  raw <- r$vars_bins
  b <- .rdplot_merge(raw)
  out <- list()
  out[[paste0(prefix, "_raw_bins")]] <- nrow(raw)
  out[[paste0(prefix, "_raw_na_bins")]] <- sum(is.na(raw$rdplot_mean_bin))
  out[[paste0(prefix, "_j")]] <- vec(r$J)
  out[[paste0(prefix, "_n")]] <- vec(r$N)
  out[[paste0(prefix, "_coef_left")]] <- vec(r$coef[, 1])
  out[[paste0(prefix, "_coef_right")]] <- vec(r$coef[, 2])
  out[[paste0(prefix, "_bin_mid")]] <- vec(b$rdplot_mean_bin)
  out[[paste0(prefix, "_bin_mean_x")]] <- vec(b$rdplot_mean_x)
  out[[paste0(prefix, "_bin_mean_y")]] <- vec(b$rdplot_mean_y)
  out[[paste0(prefix, "_bin_n")]] <- vec(b$rdplot_N)
  out[[paste0(prefix, "_bin_lower")]] <- vec(b$rdplot_min_bin)
  out[[paste0(prefix, "_bin_upper")]] <- vec(b$rdplot_max_bin)
  out[[paste0(prefix, "_curve_x")]] <- vec(r$vars_poly$rdplot_x[c(1, 250, 500, 501, 750, 1000)])
  out[[paste0(prefix, "_curve_y")]] <- vec(r$vars_poly$rdplot_y[c(1, 250, 500, 501, 750, 1000)])
  out
}

case("rdplot_sim", "causal/rd", "RD plot, rdrobust::rdplot defaults (esmv, p = 4), mass points (esmvpr), nbins, p = 2, cutoff 0.2", function(ctx) {
  d <- .rd_sim(ctx)
  v <- list()
  v <- c(v, .rdplot_values(rdplot(d$y, d$x, c = 0, hide = TRUE), "default"))
  r <- rdplot(d$ym, d$xm, c = 0, hide = TRUE)
  stopifnot(r$binselect == "IMSE-optimal evenly-spaced method using polynomial regression" ||
            grepl("polynomial regression", r$binselect))
  v <- c(v, .rdplot_values(r, "mass"))
  v <- c(v, .rdplot_values(rdplot(d$y, d$x, c = 0, nbins = c(10, 12), hide = TRUE), "nbins"))
  v <- c(v, .rdplot_values(rdplot(d$y, d$x, c = 0, p = 2, hide = TRUE), "p2"))
  v <- c(v, .rdplot_values(rdplot(d$y, d$x, c = 0.2, hide = TRUE), "c02"))
  list(dataset = "validation/datasets/generated/rd_sim.csv", values = v,
       reference = "rdrobust::rdplot(y, x, c, p, nbins) 4.0.0 (binselect = 'esmv', masspoints = 'adjust', kernel = 'uni')",
       call = "rdplot(d$y, d$x, c = 0, hide = TRUE)", packages = "rdrobust")
})
