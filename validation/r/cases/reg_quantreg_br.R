# validation/r/cases/reg_quantreg_br.R -- quantile regression references (T3-WP10: REG-52; moved in
# REG-26, REG-64). quantreg::rq(method = "br") with summary.rq(se = "iid" | "nid" | "ker") on T1-WP04's
# quantreg_hetero (p = 2) and T1-WP02's mtcars (p = 4, no intercept, location model); the Frisch-Newton
# interior point rq.fit.fnb on the dispersion-standardised outcome; the xy-pair bootstrap of boot.rq on a
# committed resample index matrix. Package copy of T3 design Appendix A.5 (canonical, T3 rule).
suppressPackageStartupMessages(library(quantreg))

.rq_tag <- function(tau) sprintf("tau%02d", round(100 * tau))
# Evaluate expr and collect its warning messages (R's own texts), so the cases can count them.
.rq_warn <- function(expr) {
  msgs <- character(0)
  val <- withCallingHandlers(expr, warning = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning") })
  list(value = val, warnings = msgs)
}
# summary.rq table for one se type, plus the 95% interval coef -/+ qt(0.975, rdf) * se that p2a reports
# (REG-64). summary.rq gives no interval for iid/nid/ker; the interval is written out here.
.rq_table <- function(f, se, tag, v) {
  s <- .rq_warn(summary(f, se = se, covariance = TRUE))
  tab <- s$value$coefficients; nm <- rownames(tab); rdf <- s$value$rdf
  q <- qt(0.975, rdf)
  v[[paste0(tag, "_se_", se)]] <- named(setNames(tab[, 2], nm))
  v[[paste0(tag, "_t_", se)]] <- named(setNames(tab[, 3], nm))
  v[[paste0(tag, "_p_", se)]] <- named(setNames(tab[, 4], nm))
  v[[paste0(tag, "_ci_lower_", se)]] <- named(setNames(tab[, 1] - q * tab[, 2], nm))
  v[[paste0(tag, "_ci_upper_", se)]] <- named(setNames(tab[, 1] + q * tab[, 2], nm))
  # scale: 1 / sparsity for iid, mean(f) for nid and ker (summary.rq covariance = TRUE)
  v[[paste0(tag, "_scale_", se)]] <- unname(s$value$scale)
  v[[paste0(tag, "_nonunique_", se)]] <- sum(s$warnings == "Solution may be nonunique")
  if (se == "nid") v[[paste0(tag, "_nonpositive_fis")]] <-
    sum(as.numeric(sub(" non-positive fis$", "", grep("non-positive fis$", s$warnings, value = TRUE))))
  v
}
# bandwidth.rq(hs = TRUE) halved until tau -/+ h lies in [0, 1] (summary.rq's nid and ker rule)
.rq_h <- function(tau, n) { h <- bandwidth.rq(tau, n, hs = TRUE); while ((tau - h < 0) || (tau + h > 1)) h <- h / 2; h }
# Powell kernel bandwidth of summary.rq(se = "ker"), written out from quantreg 6.1's summary.rq
.rq_ker_h <- function(f, tau, n) {
  u <- residuals(f); h <- .rq_h(tau, n)
  (qnorm(tau + h) - qnorm(tau - h)) * min(sqrt(var(u)), (quantile(u, 0.75) - quantile(u, 0.25)) / 1.34)
}
# p2a's Frisch-Newton runs rq.fit.fnb on y / s, s = mean(|y - median(y)|), and rescales (scale-free
# stopping rule, REG-52); written out here around quantreg's own rq.fit.fnb.
.fn_scale <- function(y) { s <- mean(abs(y - median(y))); if (s > 0) s else 1 }
.fn_std <- function(X, y, tau) { s <- .fn_scale(y); z <- rq.fit.fnb(X, y / s, tau); list(coef = s * z$coef, nit = z$nit[1], s = s) }

case("rq_br_se", "regression/quantreg", "quantreg::rq(method = 'br') and summary.rq(se = 'iid' | 'nid' | 'ker') on quantreg_hetero; rq.fit.fnb; scale equivariance; location model", function(ctx) {
  d <- ctx$dataset("datasets/generated/quantreg_hetero.csv")
  X <- cbind("(Intercept)" = 1, x = d$x); n <- nrow(d)
  v <- list(n = n, rdf = n - 2)
  for (tau in c(0.25, 0.5, 0.75)) {
    tag <- .rq_tag(tau)
    fw <- .rq_warn(rq(y ~ x, tau = tau, data = d, method = "br")); f <- fw$value
    v[[paste0(tag, "_coef")]] <- named(coef(f))
    v[[paste0(tag, "_objective")]] <- f$rho
    v[[paste0(tag, "_nonunique")]] <- sum(fw$warnings == "Solution may be nonunique")
    v[[paste0(tag, "_hs_bandwidth")]] <- .rq_h(tau, n)
    v[[paste0(tag, "_ker_bandwidth")]] <- unname(.rq_ker_h(f, tau, n))
    for (se in c("iid", "nid", "ker")) v <- .rq_table(f, se, tag, v)
    v[[paste0(tag, "_ci_rank")]] <- vec(summary(f, se = "rank")$coefficients[, 2:3])   # R's n < 1001 default; not computed by p2a
    fs <- .fn_std(X, d$y, tau)
    v[[paste0("fn_", tag, "_coef")]] <- named(setNames(fs$coef, colnames(X)))
    v[[paste0("fn_", tag, "_nit")]] <- fs$nit
    v[[paste0("fn_r_", tag, "_coef")]] <- named(coef(rq(y ~ x, tau = tau, data = d, method = "fn")))   # R's absolute gap rule
  }
  v$fn_scale <- .fn_scale(d$y)
  # scale equivariance: y times 1e-6 and 1e6 (the doubles of `y * 1e-6` in Rust). br coefficients and the
  # ker SEs scale exactly; summary.rq's iid and nid SEs use the absolute fuzz eps = sqrt(.Machine$double.eps),
  # so at 1e-6 they are R's own (scale-dependent) values, which p2a reproduces and flags with a warning.
  for (sc in c(-6, 6)) {
    ds <- d; ds$y <- d$y * 10^sc; key <- if (sc < 0) "m6" else "p6"
    g <- rq(y ~ x, tau = 0.5, data = ds)
    v[[paste0("scale_", key, "_tau50_coef")]] <- named(coef(g))
    for (se in c("iid", "nid", "ker"))
      v[[paste0("scale_", key, "_tau50_se_", se)]] <- named(setNames(.rq_warn(summary(g, se = se))$value$coefficients[, 2], names(coef(g))))
    v[[paste0("scale_", key, "_fn_tau50_coef")]] <- named(setNames(.fn_std(X, ds$y, 0.5)$coef, colnames(X)))
    v[[paste0("scale_", key, "_fn_r_tau50_coef")]] <- named(coef(rq(y ~ x, tau = 0.5, data = ds, method = "fn")))   # R's absolute gap rule
  }
  # summary.rq halves the Hall-Sheather bandwidth until tau -/+ h lies in [0, 1] (tau = 0.02, n = 100)
  f02 <- rq(y ~ x, tau = 0.02, data = d)
  v$tau02_coef <- named(coef(f02))
  v$tau02_hs_bandwidth_unhalved <- bandwidth.rq(0.02, n, hs = TRUE)
  v$tau02_hs_bandwidth <- .rq_h(0.02, n)
  v$tau02_se_nid <- named(setNames(summary(f02, se = "nid")$coefficients[, 2], names(coef(f02))))
  v$tau02_se_ker <- named(setNames(summary(f02, se = "ker")$coefficients[, 2], names(coef(f02))))
  # location model: the median and lower quartile of n = 100 values are nonunique; br returns y(50) and y(25)
  for (tau in c(0.25, 0.5)) {
    lw <- .rq_warn(rq(y ~ 1, tau = tau, data = d))
    v[[paste0("loc_", .rq_tag(tau), "_coef")]] <- unname(coef(lw$value))
    v[[paste0("loc_", .rq_tag(tau), "_nonunique")]] <- sum(lw$warnings == "Solution may be nonunique")
  }
  # n = 5 (the first five rows): summary.rq(se = "iid") stops ("variable lengths differ"); nid works and warns
  f5 <- rq(y ~ x, tau = 0.5, data = d[1:5, ])
  s5 <- .rq_warn(summary(f5, se = "nid"))
  v$small5_coef <- named(coef(f5))
  v$small5_se_nid <- named(setNames(s5$value$coefficients[, 2], names(coef(f5))))
  v$small5_nonpositive_fis <- sum(as.numeric(sub(" non-positive fis$", "", grep("non-positive fis$", s5$warnings, value = TRUE))))
  v$small5_iid_fails <- as.numeric(inherits(try(summary(f5, se = "iid"), silent = TRUE), "try-error"))
  stopifnot(v$loc_tau50_coef == sort(d$y)[50], v$loc_tau25_coef == sort(d$y)[25], v$tau75_nonunique_iid == 1,
            v$tau02_hs_bandwidth < v$tau02_hs_bandwidth_unhalved, v$small5_iid_fails == 1, v$small5_nonpositive_fis == 1)
  list(values = v, dataset = "validation/datasets/generated/quantreg_hetero.csv",
       reference = "quantreg::rq(method = 'br'); quantreg::summary.rq(se = 'iid' | 'nid' | 'ker' | 'rank', covariance = TRUE); quantreg::rq.fit.fnb on y / mean(|y - median(y)|); quantreg::bandwidth.rq",
       call = "rq(y ~ x, tau, data = quantreg_hetero, method = 'br'); summary(f, se = 'nid')", packages = "quantreg")
})

case("rq_br_mtcars", "regression/quantreg", "quantreg::rq(method = 'br') with three regressors, without intercept and as a location model on mtcars; summary.rq iid, nid, ker", function(ctx) {
  mt <- ctx$dataset("datasets/generated/mtcars.csv"); n <- nrow(mt)
  v <- list(n = n, rdf = n - 4)
  for (tau in c(0.25, 0.5, 0.75)) {
    tag <- .rq_tag(tau)
    fw <- .rq_warn(rq(mpg ~ wt + hp + qsec, tau = tau, data = mt)); f <- fw$value
    v[[paste0(tag, "_coef")]] <- named(coef(f))
    v[[paste0(tag, "_objective")]] <- f$rho
    v[[paste0(tag, "_nonunique")]] <- sum(fw$warnings == "Solution may be nonunique")
    for (se in c("iid", "nid", "ker")) v <- .rq_table(f, se, tag, v)
    g <- rq(mpg ~ wt + hp - 1, tau = tau, data = mt)
    v[[paste0("noint_", tag, "_coef")]] <- named(coef(g))
    v[[paste0("noint_", tag, "_se_nid")]] <- named(setNames(summary(g, se = "nid")$coefficients[, 2], names(coef(g))))
    lw <- .rq_warn(rq(mpg ~ 1, tau = tau, data = mt))
    v[[paste0("loc_", tag, "_coef")]] <- unname(coef(lw$value))
    v[[paste0("loc_", tag, "_nonunique")]] <- sum(lw$warnings == "Solution may be nonunique")
  }
  stopifnot(v$tau25_nonpositive_fis == 1, v$loc_tau50_nonunique == 1)
  list(values = v, reference = "quantreg::rq(method = 'br'); quantreg::summary.rq(se = 'iid' | 'nid' | 'ker', covariance = TRUE)",
       call = "rq(mpg ~ wt + hp + qsec, tau, data = mtcars); rq(mpg ~ wt + hp - 1, tau); rq(mpg ~ 1, tau)", packages = "quantreg")
})

case("rq_boot_xy", "regression/quantreg", "summary.rq(se = 'boot') xy-pair bootstrap SEs (quantreg::boot.rq, bsmethod = 'xy') on a committed 100 x 50 resample index matrix", function(ctx) {
  d <- ctx$dataset("datasets/generated/quantreg_hetero.csv")
  n <- nrow(d); R <- 50
  set.seed(20261021)
  U <- matrix(sample(n, n * R, replace = TRUE), n, R)
  colnames(U) <- sprintf("b%02d", seq_len(R))
  Ud <- ctx$gen_dataset("rq_boot_indices", as.data.frame(U))      # 1-based row indices, one column per replicate
  U <- as.matrix(Ud)
  X <- cbind("(Intercept)" = 1, x = d$x)
  v <- list(n = n, replications = R)
  for (tau in c(0.25, 0.5, 0.75)) {
    B <- boot.rq(X, d$y, tau, R = R, U = U)$B                     # summary.rq(se = "boot"): cov <- cov(B$B)
    v[[paste0(.rq_tag(tau), "_se_boot")]] <- named(setNames(sqrt(diag(cov(B))), colnames(X)))
    v[[paste0(.rq_tag(tau), "_boot_mean")]] <- named(setNames(colMeans(B), colnames(X)))
  }
  list(values = v, dataset = "validation/datasets/generated/rq_boot_indices.csv",
       reference = "quantreg::boot.rq(bsmethod = 'xy', U = U); summary.rq(se = 'boot') takes sqrt(diag(cov(B)))",
       call = "sqrt(diag(cov(boot.rq(cbind(1, x), y, tau, R = 50, U = U)$B)))", packages = "quantreg")
})
