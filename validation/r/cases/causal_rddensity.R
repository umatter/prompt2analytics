# validation/r/cases/causal_rddensity.R -- rddensity::rddensity and rdbwdensity references: the
# manipulation test of Cattaneo, Jansson & Ma (2020) with its defaults (p = 2, q = 3, triangular
# kernel, jackknife variance, bwselect = "comb", mass points, binomial windows) and options.
suppressPackageStartupMessages(library(rddensity))

.rddensity_values <- function(r, bw, prefix) {
  out <- list()
  out[[paste0(prefix, "_h")]] <- vec(c(r$h$left, r$h$right))
  out[[paste0(prefix, "_n")]] <- vec(c(r$N$left, r$N$right))
  out[[paste0(prefix, "_n_eff")]] <- vec(c(r$N$eff_left, r$N$eff_right))
  out[[paste0(prefix, "_hat")]] <- vec(c(r$hat$left, r$hat$right, r$hat$diff))
  jk <- r$opt$vce == "jackknife"
  sdv <- if (jk) r$sd_jk else r$sd_asy
  out[[paste0(prefix, "_se")]] <- vec(c(sdv$left, sdv$right, sdv$diff))
  out[[paste0(prefix, "_t")]] <- if (jk) r$test$t_jk else r$test$t_asy
  out[[paste0(prefix, "_p")]] <- if (jk) r$test$p_jk else r$test$p_asy
  out[[paste0(prefix, "_bino_left")]] <- vec(r$bino$LeftWindow)
  out[[paste0(prefix, "_bino_right")]] <- vec(r$bino$RightWindow)
  out[[paste0(prefix, "_bino_n_left")]] <- vec(r$bino$LeftN)
  out[[paste0(prefix, "_bino_n_right")]] <- vec(r$bino$RightN)
  out[[paste0(prefix, "_bino_p")]] <- vec(r$bino$pval)
  if (!is.null(bw)) out[[paste0(prefix, "_bw")]] <- vec(bw$h[, 1])
  out
}

case("rddensity_sim", "causal/rd", "Density manipulation test, rddensity::rddensity: defaults, mass points, sorting, kernels, plugin variance, bwselect, p = 1, h given, cutoff 0.2", function(ctx) {
  set.seed(7)
  n <- 1500
  x <- rnorm(n, 0, 0.5)
  # sorting: 30% of the units just below the cutoff move just above it
  s <- x
  move <- which(s > -0.15 & s < 0 & runif(n) < 0.3)
  s[move] <- -s[move] * 0.8
  xm <- round(x, 2)                                    # mass points
  d <- ctx$gen_dataset("rddensity_sim", data.frame(id = seq_len(n), x = round(x, 10), s = round(s, 10), xm = xm))
  stopifnot(!anyDuplicated(d$x), anyDuplicated(d$xm) > 0)
  v <- list()
  run <- function(tag, X, ...) {
    r <- rddensity(X, ...)
    args <- list(...)
    bw <- if (is.null(args$h)) do.call(rdbwdensity, c(list(X = X), args[setdiff(names(args), c("bwselect", "h"))])) else NULL
    v <<- c(v, .rddensity_values(r, bw, tag))
  }
  run("default", d$x)
  run("mass", d$xm)
  run("sorted", d$s)
  run("uniform", d$x, kernel = "uniform")
  run("epa", d$x, kernel = "epanechnikov")
  run("plugin", d$x, vce = "plugin")
  run("each", d$x, bwselect = "each")
  run("diff", d$x, bwselect = "diff")
  run("sum", d$x, bwselect = "sum")
  run("p1", d$x, p = 1)
  run("h", d$x, h = c(0.3, 0.4))
  run("c02", d$x, c = 0.2)
  list(dataset = "validation/datasets/generated/rddensity_sim.csv", values = v,
       reference = "rddensity::rddensity(X, c, p, kernel, vce, bwselect, h) 3.0 and rddensity::rdbwdensity (Cattaneo, Jansson & Ma 2020)",
       call = "rddensity(d$x)", packages = "rddensity")
})
