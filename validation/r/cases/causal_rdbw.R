# validation/r/cases/causal_rdbw.R -- rdbwselect references (T3: CAUS-28; moved in CAUS-100).
# rdrobust::rdbwselect on the committed rd_sim dataset of T2-WP11: every bwselect choice
# (mserd, msetwo, msesum, msecomb1, msecomb2, cerrd, certwo, cersum, cercomb1, cercomb2), the
# three kernels, p = 1 and 2, vce nn and hc1, scaleregul 1 and 0, sharp and fuzzy, and the
# rdrobust point estimate when the bandwidths are selected (h and b not given).
# T3-WP20 additions (marked below): p = 0, q = 3, vce hc0/hc2/hc3, nnmatch = 1, the selected
# bandwidths flowing into rdrobust (msetwo, rho = 1, scaleregul = 0, hc1, mass points, h selected
# with b given), the mass-point counts, fuzzy hc1, fuzzy cerrd, one-sided compliance, and the
# two-cutoff rd_multi references (per-cutoff rdrobust and p2a's pooling rule written out).
suppressPackageStartupMessages(library(rdrobust))

.bw_all <- function(y, x, .tag, ...) {
  prefix <- .tag
  b <- rdbwselect(y, x, c = 0, all = TRUE, ...)
  bws <- b$bws
  out <- list()
  for (i in seq_len(nrow(bws))) {
    nm <- gsub("[^a-z0-9]", "", tolower(rownames(bws)[i]))
    out[[paste0(prefix, "_", nm)]] <- vec(as.numeric(bws[i, ]))   # h left, h right, b left, b right
  }
  out
}

# T3-WP20: the rdrobust quantities a selected-bandwidth run is pinned on.
.rd_sel <- function(r, prefix) {
  out <- list()
  out[[paste0(prefix, "_h")]] <- vec(r$bws[1, ]); out[[paste0(prefix, "_b")]] <- vec(r$bws[2, ])
  out[[paste0(prefix, "_coef_conv")]] <- r$coef[1]; out[[paste0(prefix, "_coef_bc")]] <- r$coef[2]
  out[[paste0(prefix, "_se_conv")]] <- r$se[1]; out[[paste0(prefix, "_se_robust")]] <- r$se[3]
  out[[paste0(prefix, "_p_robust")]] <- r$pv[3]; out[[paste0(prefix, "_ci_robust")]] <- vec(r$ci[3, ])
  out[[paste0(prefix, "_n_h")]] <- vec(r$N_h); out[[paste0(prefix, "_n_b")]] <- vec(r$N_b)
  out
}

case("rd_bwselect_sharp", "econometrics/rd", "rdbwselect bandwidths (all methods), sharp RD on rd_sim", function(ctx) {
  d <- ctx$dataset("datasets/generated/rd_sim.csv")
  v <- c(.bw_all(d$y, d$x, "tri_p1_nn"),
         .bw_all(d$y, d$x, "epa_p1_nn", kernel = "epanechnikov"),
         .bw_all(d$y, d$x, "uni_p1_nn", kernel = "uniform"),
         .bw_all(d$y, d$x, "tri_p2_nn", p = 2),
         .bw_all(d$y, d$x, "tri_p1_hc1", vce = "hc1"),
         .bw_all(d$y, d$x, "tri_p1_noreg", scaleregul = 0),
         .bw_all(d$ym, d$xm, "mass_tri_p1_nn"))
  r <- rdrobust(d$y, d$x, c = 0)                    # default: mserd, triangular, p = 1, nn
  v$auto_h <- vec(r$bws[1, ]); v$auto_b <- vec(r$bws[2, ])
  v$auto_coef_conv <- r$coef[1]; v$auto_coef_bc <- r$coef[2]
  v$auto_se_conv <- r$se[1]; v$auto_se_robust <- r$se[3]
  r2 <- rdrobust(d$y, d$x, c = 0, bwselect = "cerrd")
  v$cerrd_h <- vec(r2$bws[1, ]); v$cerrd_b <- vec(r2$bws[2, ]); v$cerrd_coef_bc <- r2$coef[2]; v$cerrd_se_robust <- r2$se[3]

  # ---- T3-WP20 additions ----
  v <- c(v, .bw_all(d$y, d$x, "tri_p0_nn", p = 0),          # CER factor N^0 = 1
         .bw_all(d$y, d$x, "tri_p1_q3", q = 3),
         .bw_all(d$y, d$x, "tri_p1_hc0", vce = "hc0"),
         .bw_all(d$y, d$x, "tri_p1_hc2", vce = "hc2"),
         .bw_all(d$y, d$x, "tri_p1_hc3", vce = "hc3"),
         .bw_all(d$y, d$x, "tri_p1_nn1", nnmatch = 1))
  v$auto_n_h <- vec(r$N_h); v$auto_n_b <- vec(r$N_b); v$auto_p_robust <- r$pv[3]; v$auto_ci_robust <- vec(r$ci[3, ])
  v$cerrd_coef_conv <- r2$coef[1]; v$cerrd_se_conv <- r2$se[1]; v$cerrd_n_h <- vec(r2$N_h)
  v <- c(v, .rd_sel(rdrobust(d$y, d$x, c = 0, bwselect = "msetwo"), "msetwo_auto"),
         .rd_sel(rdrobust(d$y, d$x, c = 0, rho = 1), "rho1_auto"),                      # b = h_selected
         .rd_sel(rdrobust(d$y, d$x, c = 0, scaleregul = 0), "noreg_auto"),
         .rd_sel(rdrobust(d$y, d$x, c = 0, vce = "hc1"), "hc1_auto"),
         .rd_sel(suppressWarnings(rdrobust(d$ym, d$xm, c = 0)), "mass_auto"))
  # p2a honours a given b when h is selected; rdrobust ignores b without h, so the reference is
  # rdrobust at the selected h with that b.
  v <- c(v, .rd_sel(rdrobust(d$y, d$x, c = 0, h = r$bws[1, 1], b = 0.5), "hsel_b05"))
  bm <- suppressWarnings(rdbwselect(d$ym, d$xm, c = 0))
  stopifnot(min(1 - bm$M / bm$N) >= 0.2)            # mass points on both sides: bwcheck = 10 applies
  v$mass_counts_n <- vec(bm$N); v$mass_counts_m <- vec(bm$M)
  b0 <- rdbwselect(d$y, d$x, c = 0)
  v$tri_p1_nn_counts_n <- vec(b0$N); v$tri_p1_nn_counts_m <- vec(b0$M)
  # rd_multi on two cutoffs: group 0 = rows (y, x) at c = 0, group 1 = rows (yf, x + 2) at c = 2,
  # stacked in that order; per-cutoff estimates are rdrobust with selected (mserd) bandwidths.
  x2 <- d$x + 2
  stopifnot(max(d$x) < 1, min(x2) > 1)              # nearest-cutoff assignment equals the groups
  r1 <- rdrobust(d$yf, x2, c = 2)
  v <- c(v, .rd_sel(r1, "multi_c2"))
  # p2a's pooling rule (rdmulti.rs), written out on the two rdrobust results (rdmulti is not installed;
  # rdmc's pooled estimate is a different estimator, see validation/econometrics/rdmulti.md)
  tau <- c(r$coef[2], r1$coef[2]); se <- c(r$se[3], r1$se[3]); nh <- c(sum(r$N_h), sum(r1$N_h))
  pool <- function(w) {
    w <- w / sum(w); e <- sum(w * tau); s <- sqrt(sum(w^2 * se^2)); z <- qnorm(0.975)
    c(e, s, e - z * s, e + z * s, 2 * pnorm(-abs(e / s)), w)      # effect, se, ci, p, weights
  }
  v$multi_pool_sample <- vec(pool(nh)); v$multi_pool_iv <- vec(pool(1 / se^2)); v$multi_pool_equal <- vec(pool(c(1, 1)))
  tiv <- sum(tau / se^2) / sum(1 / se^2); Q <- sum((tau - tiv)^2 / se^2)
  v$multi_het <- vec(c(Q, pchisq(Q, df = 1, lower.tail = FALSE)))
  g0 <- rdrobust(d$y, d$x, c = 0, h = 0.3); g1 <- rdrobust(d$yf, x2, c = 2, h = 0.3)
  v$multi_g03_coef_bc <- vec(c(g0$coef[2], g1$coef[2])); v$multi_g03_se_robust <- vec(c(g0$se[3], g1$se[3]))
  v$multi_c2_ci90_robust <- vec(rdrobust(d$yf, x2, c = 2, level = 90)$ci[3, ])
  r3 <- rdrobust(d$yf, x2, c = 2, bwselect = "cerrd")
  v$multi_c2_cerrd_h <- vec(r3$bws[1, ]); v$multi_c2_cerrd_coef_bc <- r3$coef[2]
  list(values = v, reference = "rdrobust::rdbwselect(all = TRUE, kernel, p, vce, scaleregul); rdrobust::rdrobust with selected bandwidths",
       call = "rdbwselect(rd_sim$y, rd_sim$x, c = 0, all = TRUE)", packages = "rdrobust")
})

case("rd_bwselect_fuzzy", "econometrics/rd", "rdbwselect bandwidths, fuzzy RD on rd_sim (yf with take-up d)", function(ctx) {
  d <- ctx$dataset("datasets/generated/rd_sim.csv")
  v <- .bw_all(d$yf, d$x, "fuzzy_tri_p1_nn", fuzzy = d$d)
  r <- rdrobust(d$yf, d$x, c = 0, fuzzy = d$d)
  v$fuzzy_auto_h <- vec(r$bws[1, ]); v$fuzzy_auto_b <- vec(r$bws[2, ])
  v$fuzzy_auto_coef_conv <- r$coef[1]; v$fuzzy_auto_coef_bc <- r$coef[2]; v$fuzzy_auto_se_robust <- r$se[3]

  # ---- T3-WP20 additions ----
  v <- c(v, .bw_all(d$yf, d$x, "fuzzy_tri_p1_hc1", fuzzy = d$d, vce = "hc1"))
  v$fuzzy_auto_se_conv <- r$se[1]; v$fuzzy_auto_p_robust <- r$pv[3]; v$fuzzy_auto_ci_robust <- vec(r$ci[3, ])
  v$fuzzy_auto_n_h <- vec(r$N_h); v$fuzzy_auto_first_stage_conv <- r$tau_T[1]
  rc <- rdrobust(d$yf, d$x, c = 0, fuzzy = d$d, bwselect = "cerrd")
  v$fuzzy_cerrd_h <- vec(rc$bws[1, ]); v$fuzzy_cerrd_b <- vec(rc$bws[2, ])
  v$fuzzy_cerrd_coef_bc <- rc$coef[2]; v$fuzzy_cerrd_se_robust <- rc$se[3]
  # one-sided compliance (d1 = 0 left of the cutoff): var(T_l) = 0, so rdbwselect selects on yf alone
  d1 <- d$d * (d$x >= 0)
  b1 <- rdbwselect(d$yf, d$x, c = 0, fuzzy = d1)$bws; bs <- rdbwselect(d$yf, d$x, c = 0)$bws
  stopifnot(identical(unname(b1), unname(bs)))
  v$fuzzy_onesided_mserd <- vec(b1[1, ])
  r1 <- rdrobust(d$yf, d$x, c = 0, fuzzy = d1)
  v$fuzzy_onesided_coef_conv <- r1$coef[1]; v$fuzzy_onesided_coef_bc <- r1$coef[2]; v$fuzzy_onesided_se_robust <- r1$se[3]
  list(values = v, reference = "rdrobust::rdbwselect(fuzzy = d, all = TRUE); rdrobust(fuzzy = d)",
       call = "rdbwselect(rd_sim$yf, rd_sim$x, c = 0, fuzzy = rd_sim$d, all = TRUE)", packages = "rdrobust")
})
