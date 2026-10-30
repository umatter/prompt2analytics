# validation/r/cases/causal_rd.R -- rdrobust references with fixed bandwidths (T2-WP11: CAUS-1,
# CAUS-5, CAUS-29, CAUS-30, CAUS-76). Bandwidths are passed explicitly: p2a's bandwidth selector
# is CAUS-28 (Tier 3), so these cases pin the estimator, the bias correction and the variance only.
suppressPackageStartupMessages(library(rdrobust))

.rd_values <- function(r, prefix) {
  out <- list()
  out[[paste0(prefix, "_coef_conv")]] <- r$coef[1]; out[[paste0(prefix, "_coef_bc")]] <- r$coef[2]
  out[[paste0(prefix, "_se_conv")]] <- r$se[1]; out[[paste0(prefix, "_se_robust")]] <- r$se[3]
  out[[paste0(prefix, "_z_conv")]] <- r$z[1]; out[[paste0(prefix, "_z_robust")]] <- r$z[3]
  out[[paste0(prefix, "_p_conv")]] <- r$pv[1]; out[[paste0(prefix, "_p_robust")]] <- r$pv[3]
  out[[paste0(prefix, "_ci_conv")]] <- vec(r$ci[1, ]); out[[paste0(prefix, "_ci_robust")]] <- vec(r$ci[3, ])
  out[[paste0(prefix, "_n_h")]] <- vec(r$N_h); out[[paste0(prefix, "_n_b")]] <- vec(r$N_b)
  out[[paste0(prefix, "_beta_p_left")]] <- vec(r$beta_Y_p_l); out[[paste0(prefix, "_beta_p_right")]] <- vec(r$beta_Y_p_r)
  out
}

.rd_sim <- function(ctx) {
  set.seed(42)
  n <- 1000
  x <- runif(n, -1, 1)
  y <- 0.5 + 0.8 * x - 0.6 * x^2 + 0.4 * (x >= 0) + 0.3 * x * (x >= 0) + rnorm(n, 0, 0.5)
  # fuzzy take-up: P(D = 1) jumps from 0.2 to 0.8 at the cutoff; outcome responds to D
  dd <- as.integer(runif(n) < ifelse(x >= 0, 0.8, 0.2))
  yf <- 0.5 + 0.8 * x + 0.6 * dd + rnorm(n, 0, 0.5)
  xm <- round(x, 2)                                     # mass points: 201 distinct values
  ym <- 0.5 + 0.8 * xm + 0.4 * (xm >= 0) + rnorm(n, 0, 0.5)
  ctx$gen_dataset("rd_sim", data.frame(x = round(x, 10), y = round(y, 10), d = dd, yf = round(yf, 10),
                                         xm = xm, ym = round(ym, 10)))
}

case("rd_sharp_fixed_bw", "causal/rd", "Sharp RD, rdrobust with h and b given: kernels, VCE types, p = 1 and 2", function(ctx) {
  d <- .rd_sim(ctx)
  stopifnot(is.integer(d$d))
  v <- list()
  for (k in c("triangular", "epanechnikov", "uniform")) {
    for (vce in c("nn", "hc0", "hc1", "hc2", "hc3")) {
      r <- rdrobust(d$y, d$x, c = 0, p = 1, h = 0.3, b = 0.3, kernel = k, vce = vce)
      v <- c(v, .rd_values(r, paste0(substr(k, 1, 3), "_", vce, "_hb")))
    }
  }
  r <- rdrobust(d$y, d$x, c = 0, p = 1, h = 0.3, b = 0.5, kernel = "triangular", vce = "nn")
  v <- c(v, .rd_values(r, "tri_nn_h03_b05"))
  r <- rdrobust(d$y, d$x, c = 0, p = 1, h = 0.5, b = 0.3, kernel = "triangular", vce = "nn")
  v <- c(v, .rd_values(r, "tri_nn_h05_b03"))
  r <- rdrobust(d$y, d$x, c = 0, p = 2, h = 0.5, b = 0.6, kernel = "triangular", vce = "nn")
  v <- c(v, .rd_values(r, "tri_nn_p2"))
  r <- rdrobust(d$y, d$x, c = 0, p = 1, h = c(0.25, 0.35), b = c(0.4, 0.45), kernel = "triangular", vce = "hc1")
  v <- c(v, .rd_values(r, "tri_hc1_asym"))
  r <- rdrobust(d$y, d$x, c = 0, p = 1, h = 0.3, b = 0.3, level = 90)
  v$level90_ci_robust <- vec(r$ci[3, ])
  # T2-WP11 addition (CAUS-76): rows with a missing outcome or running variable are dropped, as
  # rdrobust does (na.ok <- complete.cases(x, y)). Two y cells and two x cells inside h = 0.3 are
  # set to NA here; the Rust test nulls the same 0-based rows of the same CSV.
  inside <- which(abs(d$x) < 0.3)
  na_y <- inside[c(3, 10)]; na_x <- inside[c(20, 41)]
  dn <- d; dn$y[na_y] <- NA; dn$x[na_x] <- NA
  r <- rdrobust(dn$y, dn$x, c = 0, p = 1, h = 0.3, b = 0.3, vce = "nn")
  stopifnot(sum(r$N) == nrow(d) - 4)
  v <- c(v, .rd_values(r, "na_tri_nn_hb"))
  v$na_rows_y <- vec(na_y - 1); v$na_rows_x <- vec(na_x - 1)
  list(dataset = "validation/datasets/generated/rd_sim.csv", values = v,
       reference = "rdrobust::rdrobust(y, x, c = 0, p, h, b, kernel, vce) 4.0.0 (CCT 2014 bias correction and robust variance)",
       call = "rdrobust(d$y, d$x, c = 0, p = 1, h = 0.3, b = 0.3, kernel = ..., vce = ...)", packages = "rdrobust")
})

case("rd_mass_points_nn", "causal/rd", "Sharp RD with a running variable on a 0.01 grid: NN residuals with tied neighbours", function(ctx) {
  d <- .rd_sim(ctx)
  v <- list()
  for (m in c(3, 1, 5)) {
    r <- rdrobust(d$ym, d$xm, c = 0, p = 1, h = 0.3, b = 0.4, vce = "nn", nnmatch = m)
    v <- c(v, .rd_values(r, paste0("nn", m)))
  }
  list(dataset = "validation/datasets/generated/rd_sim.csv", values = v,
       reference = "rdrobust::rdrobust(vce = 'nn', nnmatch = 1 | 3 | 5) with mass points (all ties at the J-th distance kept)",
       call = "rdrobust(d$ym, d$xm, c = 0, h = 0.3, b = 0.4, vce = 'nn', nnmatch = m)", packages = "rdrobust")
})
