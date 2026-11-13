# validation/r/cases/regression_smoothers_t4.R -- T4-WP08 references: smooth.spline (REG-29, REG-30),
# stats::line (REG-12), loess one.delta and family = "symmetric" (REG-60), weighted and one-sided
# bounded nls (REG-48), step() to the intercept-only model (REG-54).
#  * smooth_spline_t4: R's smooth.spline algorithm (x scaled to [0, 1], .nknots.smspl knots, cubic
#    B-splines, the roughness penalty integrated as sgram.f does (0.333 for 1/3), ratio = tr(X'WX) /
#    tr(Omega) over the inner coefficients, lambda = ratio 256^(3 spar - 1), sbart's Brent search on spar
#    in [-1.5, 1.5], tol 1e-4, eps 2e-8, leverages from the band of the inverse) written out with
#    splines::splineDesign. As sbart, a search returns the best spar and the lambda, fit, df and
#    leverages of its last evaluation. The a_*, b_*, c_* keys are the written-out algorithm; the r_* keys
#    are smooth.spline's own output, which they reproduce to 1e-8 (searched) and 2e-11 (fixed spar).
#  * line_tukey_t4: stats::line (R >= 3.5 tukeyline), iter = 1, 2, 3, on three data sets.
#  * loess_t4: stats::loess(surface = "direct", statistics = "exact"), gaussian and symmetric.
#  * nls_weighted_t4: stats::nls(weights =), and algorithm = "port" with a lower-only bound.
#  * step_null_t4: stats::step from the full model when no predictor helps (selects y ~ 1).
suppressPackageStartupMessages(library(splines))
.or_else <- function(a, b) if (is.null(a)) b else a

.ss_omega <- function(knot, nk, third) {
  uk <- unique(knot); Om <- matrix(0, nk, nk)
  for (k in seq_len(length(uk) - 1)) {
    a <- uk[k]; b <- uk[k + 1]; h <- b - a
    L <- splineDesign(knot, a, ord = 4, derivs = 2, outer.ok = TRUE)[1, ]
    R <- splineDesign(knot, b - 1e-9 * h, ord = 4, derivs = 2, outer.ok = TRUE)[1, ]
    R <- L + (R - L) / (1 - 1e-9)            # B'' is linear on [a, b): its left limit at b
    s1 <- L; s2 <- R - L
    Om <- Om + h * (outer(s1, s1) + (outer(s2, s1) + outer(s1, s2)) * 0.5 + outer(s2, s2) * third)
  }
  Om
}
.ss_setup <- function(x, y, w = NULL, third = 0.333, all.knots = FALSE) {
  n <- length(x); no.w <- is.null(w)
  w <- if (no.w) rep(1, n) else (w * sum(w > 0)) / sum(w)
  tol <- 1e-6 * IQR(x); xx <- round((x - mean(x)) / tol)
  o <- order(x); xxs <- xx[o]; nd <- c(TRUE, xxs[-n] < xxs[-1]); ux <- x[o][nd]; nx <- length(ux)
  ox <- match(xx, xxs[nd])
  wbar <- as.vector(tapply(w, ox, sum)); swy <- as.vector(tapply(w * y, ox, sum)); swy2 <- as.vector(tapply(w * y^2, ox, sum))
  ybar <- swy / ifelse(wbar > 0, wbar, 1); yssw <- sum(swy2 - wbar * ybar^2)
  r.ux <- ux[nx] - ux[1]; xbar <- (ux - ux[1]) / r.ux
  nkn <- if (all.knots) nx else stats:::.nknots.smspl(nx)
  knot <- c(rep(xbar[1], 3), if (all.knots) xbar else xbar[seq.int(1, nx, length.out = nkn)], rep(xbar[nx], 3))
  nk <- nkn + 2
  X <- splineDesign(knot, xbar, ord = 4, outer.ok = TRUE)
  XWX <- crossprod(X * sqrt(wbar)); XWy <- drop(crossprod(X, wbar * ybar))
  Om <- .ss_omega(knot, nk, third); i <- 3:(nk - 3)
  list(x = x, y = y, w = w, no.w = no.w, n = n, ux = ux, nx = nx, ox = ox, wbar = wbar, ybar = ybar, yssw = yssw,
       r.ux = r.ux, xbar = xbar, knot = knot, nk = nk, nknots = nkn, X = X, XWX = XWX, XWy = XWy, Om = Om,
       ratio = sum(diag(XWX)[i]) / sum(diag(Om)[i]))
}
.ss_solve <- function(S, lambda) {
  M <- S$XWX + lambda * S$Om; Minv <- solve(M); cf <- drop(Minv %*% S$XWy)
  ty <- drop(S$X %*% cf); lev <- rowSums((S$X %*% Minv) * S$X) * S$wbar
  list(lambda = lambda, coef = cf, ty = ty, lev = lev, df = sum(lev))
}
.ss_crit <- function(S, f, icrit, dofoff = 0) {      # sslvrg.f: 1 GCV, 2 CV, 3 df matching
  sumw <- sum(S$wbar)
  if (icrit == 1) { rss <- S$yssw + sum(S$wbar * (S$ybar - f$ty)^2); (rss / sumw) / (1 - f$df / sumw)^2 }
  else if (icrit == 2) sum(S$wbar * ((S$ybar - f$ty) / (1 - f$lev))^2) / sumw
  else 3 + (dofoff - f$df)^2
}
.ss_brent <- function(S, icrit, dofoff = 0, low = -1.5, high = 1.5, tol = 1e-4, eps = 2e-8, maxit = 500) {
  BIG <- 1e100; cg <- 0.381966011250105151795
  ev <- function(sp) { f <- .ss_solve(S, S$ratio * 256^(3 * sp - 1)); cr <- .ss_crit(S, f, icrit, dofoff); if (!is.finite(cr)) cr <- 2 * BIG; list(f = f, cr = cr) }
  a <- low; b <- high; x <- a + cg * (b - a); v <- x; w <- x; d <- 0; e <- 0
  r0 <- ev(x); last <- r0; fx <- r0$cr; fv <- fx; fw <- fx; it <- 0
  repeat {
    xm <- (a + b) * .5; tol1 <- eps * abs(x) + tol / 3; tol2 <- tol1 * 2; it <- it + 1
    if (abs(x - xm) <= tol2 - (b - a) * .5 || it > maxit) break
    golden <- abs(e) <= tol1 || fx >= BIG || fv >= BIG || fw >= BIG
    if (!golden) {
      r <- (x - w) * (fx - fv); q <- (x - v) * (fx - fw); p <- (x - v) * q - (x - w) * r; q <- (q - r) * 2
      if (q > 0) p <- -p; q <- abs(q); r <- e; e <- d
      if (abs(p) >= abs(.5 * q * r) || q == 0 || p <= q * (a - x) || p >= q * (b - x)) golden <- TRUE
      else { d <- p / q; u <- x + d; if (u - a < tol2 || b - u < tol2) d <- if (xm - x >= 0) abs(tol1) else -abs(tol1) }
    }
    if (golden) { e <- if (x >= xm) a - x else b - x; d <- cg * e }
    u <- x + (if (abs(d) >= tol1) d else (if (d >= 0) abs(tol1) else -abs(tol1)))
    ru <- ev(u); last <- ru; fu <- ru$cr
    if (fu <= fx) { if (u >= x) a <- x else b <- x; v <- w; fv <- fw; w <- x; fw <- fx; x <- u; fx <- fu }
    else { if (u < x) a <- u else b <- u
      if (fu <= fw || w == x) { v <- w; fv <- fw; w <- u; fw <- fu }
      else if (fu <= fv || v == x || v == w) { v <- u; fv <- fu } }
  }
  list(spar = x, crit = fx, iter = it, f = last$f)   # sbart: the fit of the last evaluation
}
.ss_cvcrit <- function(S, f, cv) {                     # smooth.spline's cv.crit, on the observations
  r <- S$y - f$ty[S$ox]
  if (cv) { ww <- S$wbar; ww[ww == 0] <- 1; r <- r / (1 - (f$lev[S$ox] * S$w) / ww[S$ox]); if (S$no.w) mean(r^2) else weighted.mean(r^2, S$w) }
  else (if (S$no.w) mean(r^2) else weighted.mean(r^2, S$w)) / (1 - f$df / S$n)^2
}
.ss_predict <- function(S, f, xout) {                  # predict.smooth.spline: linear beyond the data
  xs <- (xout - S$ux[1]) / S$r.ux; inside <- xs >= 0 & xs <= 1; out <- numeric(length(xs))
  if (any(inside)) out[inside] <- drop(splineDesign(S$knot, xs[inside], ord = 4, outer.ok = TRUE) %*% f$coef)
  if (any(!inside)) {
    e <- drop(splineDesign(S$knot, c(0, 1), ord = 4, outer.ok = TRUE) %*% f$coef)
    s <- drop(splineDesign(S$knot, c(0, 1), ord = 4, derivs = c(1, 1), outer.ok = TRUE) %*% f$coef)
    lo <- xs < 0; out[lo] <- e[1] + s[1] * xs[lo]; hi <- xs > 1; out[hi] <- e[2] + s[2] * (xs[hi] - 1)
  }
  out
}

case("smooth_spline_t4", "regression/smoothers", "smooth.spline's algorithm (written out) and smooth.spline itself: spar, lambda, df, GCV, CV, weights and ties", function(ctx) {
  set.seed(20261113)
  n <- 120; xa <- sort(runif(n, 0, 50)); ya <- sin(xa / 8) + 0.3 * rnorm(n)
  A <- ctx$gen_dataset("smooth_spline_t4a", data.frame(x = round(xa, 10), y = round(ya, 10)))
  nb <- 150; xb <- round(runif(nb, 0, 10), 1); yb <- cos(xb / 2) + rnorm(nb, 0, 0.25); wb <- round(runif(nb, 0.5, 2), 2)
  B <- ctx$gen_dataset("smooth_spline_t4b", data.frame(x = xb, y = round(yb, 10), w = wb))
  nc <- 30; yc <- sin((1:nc) / 5) + 0.2 * rnorm(nc)
  C <- ctx$gen_dataset("smooth_spline_t4c", data.frame(x = 1:nc, y = round(yc, 10)))
  v <- list()
  SA <- .ss_setup(A$x, A$y); v$a_nx <- SA$nx; v$a_nknots <- SA$nknots; v$a_ratio <- SA$ratio
  xout <- c(-5, 0.7, 12.3, 25, 49.9, 55)
  f <- .ss_solve(SA, SA$ratio * 256^(3 * 0.5 - 1))
  v$a_spar05_lambda <- f$lambda; v$a_spar05_df <- f$df; v$a_spar05_fit <- vec(f$ty); v$a_spar05_lev <- vec(f$lev)
  v$a_spar05_gcv_crit <- .ss_cvcrit(SA, f, FALSE); v$a_spar05_coef <- vec(f$coef)
  v$a_predict_x <- vec(xout); v$a_spar05_predict <- vec(.ss_predict(SA, f, xout))
  f <- .ss_solve(SA, 1e-4); v$a_lambda1e4_df <- f$df; v$a_lambda1e4_fit <- vec(f$ty)
  for (spec in list(list(tag = "df6", ic = 3, df = 6), list(tag = "gcv", ic = 1, df = 0), list(tag = "cv", ic = 2, df = 0))) {
    b <- .ss_brent(SA, spec$ic, spec$df); p <- paste0("a_", spec$tag, "_")
    v[[paste0(p, "spar")]] <- b$spar; v[[paste0(p, "lambda")]] <- b$f$lambda; v[[paste0(p, "df")]] <- b$f$df
    v[[paste0(p, "crit")]] <- b$crit; v[[paste0(p, "cv_crit")]] <- .ss_cvcrit(SA, b$f, spec$tag == "cv")
    v[[paste0(p, "fit")]] <- vec(b$f$ty); v[[paste0(p, "lev")]] <- vec(b$f$lev)
    s <- switch(spec$tag, df6 = smooth.spline(A$x, A$y, df = 6), gcv = smooth.spline(A$x, A$y), cv = smooth.spline(A$x, A$y, cv = TRUE))
    q <- paste0("r_a_", spec$tag, "_")
    v[[paste0(q, "spar")]] <- s$spar; v[[paste0(q, "lambda")]] <- s$lambda; v[[paste0(q, "df")]] <- s$df
    v[[paste0(q, "cv_crit")]] <- s$cv.crit; v[[paste0(q, "fit")]] <- vec(s$y)
  }
  s <- smooth.spline(A$x, A$y, spar = 0.5)
  v$r_a_spar05_lambda <- s$lambda; v$r_a_spar05_df <- s$df; v$r_a_spar05_ratio <- s$ratio; v$r_a_spar05_fit <- vec(s$y)
  SK <- .ss_setup(A$x, A$y, all.knots = TRUE); f <- .ss_solve(SK, SK$ratio * 256^(3 * 0.5 - 1))
  v$a_allknots_ratio <- SK$ratio; v$a_allknots_spar05_df <- f$df; v$a_allknots_spar05_fit <- vec(f$ty)
  SB <- .ss_setup(B$x, B$y, B$w); v$b_nx <- SB$nx; v$b_nknots <- SB$nknots; v$b_ux <- vec(SB$ux)
  for (spec in list(list(tag = "gcv", ic = 1, df = 0), list(tag = "df5", ic = 3, df = 5))) {
    b <- .ss_brent(SB, spec$ic, spec$df); p <- paste0("b_", spec$tag, "_")
    v[[paste0(p, "spar")]] <- b$spar; v[[paste0(p, "lambda")]] <- b$f$lambda; v[[paste0(p, "df")]] <- b$f$df
    v[[paste0(p, "cv_crit")]] <- .ss_cvcrit(SB, b$f, FALSE); v[[paste0(p, "fit")]] <- vec(b$f$ty); v[[paste0(p, "lev")]] <- vec(b$f$lev)
    v[[paste0(p, "residuals")]] <- vec(B$y - b$f$ty[SB$ox])
    s <- if (spec$tag == "gcv") smooth.spline(B$x, B$y, B$w) else smooth.spline(B$x, B$y, B$w, df = 5)
    q <- paste0("r_b_", spec$tag, "_")
    v[[paste0(q, "spar")]] <- s$spar; v[[paste0(q, "lambda")]] <- s$lambda; v[[paste0(q, "df")]] <- s$df
    v[[paste0(q, "cv_crit")]] <- s$cv.crit; v[[paste0(q, "fit")]] <- vec(s$y)
  }
  SC <- .ss_setup(C$x, C$y); v$c_nknots <- SC$nknots
  b <- .ss_brent(SC, 3, 6); v$c_df6_spar <- b$spar; v$c_df6_lambda <- b$f$lambda; v$c_df6_df <- b$f$df; v$c_df6_fit <- vec(b$f$ty)
  s <- smooth.spline(C$x, C$y, df = 6); v$r_c_df6_df <- s$df; v$r_c_df6_fit <- vec(s$y)
  list(values = v, dataset = "validation/datasets/generated/smooth_spline_t4a.csv",
       reference = "stats::smooth.spline's algorithm (R 4.3 sbart.c / sslvrg.f / sgram.f: scaled x, .nknots.smspl, the penalty with 0.333, ratio, spar search returning its last evaluation, criteria) written out with splines::splineDesign (a_*, b_*, c_*); stats::smooth.spline itself (r_*)",
       call = "smooth.spline(x, y, spar = 0.5 | df = 6 | cv = FALSE | cv = TRUE); smooth.spline(x, y, w) on tied x", packages = character(0))
})

.tukeyline <- function(x, y, iter) {                   # R >= 3.5 line.c, checked against line() below
  n <- length(x); z <- sort(x)
  q <- function(p) 0.5 * (z[floor((n - 1) * p) + 1] + z[ceiling((n - 1) * p) + 1])
  xb <- q(1/3); xt <- q(2/3); xL <- median(x[x <= xb]); xR <- median(x[x >= xt])
  slope <- 0; w <- y
  for (j in seq_len(iter)) { slope <- slope + (median(w[x >= xt]) - median(w[x <= xb])) / (xR - xL); w <- y - slope * x }
  c(median(w), slope)
}

case("line_tukey_t4", "regression/line", "stats::line (R >= 3.5 tukeyline), iter = 1, 2, 3: the REG-12 example, tied x, n = 31", function(ctx) {
  set.seed(20261114)
  e <- ctx$gen_dataset("line_t4_eleven", data.frame(x = 1:11, y = c(2, 4, 3, 7, 5, 9, 8, 12, 10, 15, 30)))
  xt <- round(runif(20, 0, 4)); t <- ctx$gen_dataset("line_t4_ties", data.frame(x = xt, y = round(1 + 0.5 * xt + rnorm(20), 10)))
  x3 <- runif(31, -2, 2); u <- ctx$gen_dataset("line_t4_n31", data.frame(x = round(x3, 10), y = round(2 - x3 + rnorm(31, 0, 0.5), 10)))
  v <- list()
  for (d in list(list("eleven", e), list("ties", t), list("n31", u))) for (it in 1:3) {
    cf <- coef(line(d[[2]]$x, d[[2]]$y, iter = it))
    stopifnot(max(abs(cf - .tukeyline(d[[2]]$x, d[[2]]$y, it))) == 0)
    v[[sprintf("%s_iter%d_coef", d[[1]], it)]] <- vec(cf)
  }
  l <- line(u$x, u$y, iter = 1); v$n31_iter1_fitted <- vec(fitted(l)); v$n31_iter1_residuals <- vec(residuals(l))
  list(values = v, dataset = "validation/datasets/generated/line_t4_eleven.csv",
       reference = "stats::line(iter = 1 | 2 | 3) (R >= 3.5 tukeyline)", call = "line(x, y, iter)", packages = character(0))
})

case("loess_t4", "regression/smoothers", "stats::loess(surface = 'direct', statistics = 'exact'): one.delta, enp, s; family = 'symmetric' (iterations 4 and 2)", function(ctx) {
  set.seed(20261115)
  n <- 100; x <- sort(runif(n, 0, 10)); y <- sin(x) + rnorm(n, 0, 0.3); y[c(10, 40, 75)] <- y[c(10, 40, 75)] + c(4, -5, 6)
  d <- ctx$gen_dataset("loess_t4", data.frame(x = round(x, 10), y = round(y, 10)))
  v <- list(n = n)
  for (deg in 1:2) for (sp in c(0.3, 0.75)) {
    tag <- sprintf("d%d_s%02d", deg, round(100 * sp))
    ex <- loess(y ~ x, d, span = sp, degree = deg, control = loess.control(surface = "direct", statistics = "exact"))
    ap <- loess(y ~ x, d, span = sp, degree = deg, control = loess.control(surface = "direct"))
    v[[paste0(tag, "_trace")]] <- ex$trace.hat; v[[paste0(tag, "_one_delta")]] <- ex$one.delta
    v[[paste0(tag, "_enp")]] <- ex$enp; v[[paste0(tag, "_s")]] <- ex$s
    v[[paste0(tag, "_approx_one_delta")]] <- ap$one.delta; v[[paste0(tag, "_approx_enp")]] <- ap$enp; v[[paste0(tag, "_approx_s")]] <- ap$s
    for (it in c(4, 2)) {
      r <- loess(y ~ x, d, span = sp, degree = deg, family = "symmetric",
                 control = loess.control(surface = "direct", statistics = "exact", iterations = it))
      rt <- sprintf("%s_sym%d", tag, it)
      v[[paste0(rt, "_fit")]] <- vec(fitted(r)); v[[paste0(rt, "_s")]] <- r$s; v[[paste0(rt, "_enp")]] <- r$enp
      v[[paste0(rt, "_one_delta")]] <- r$one.delta; v[[paste0(rt, "_robust_weights")]] <- vec(.or_else(r$pars$robust, r$robust))
    }
  }
  list(values = v, reference = "stats::loess(control = loess.control(surface = 'direct', statistics = 'exact')), family = 'gaussian' | 'symmetric' (loess.control(iterations = 4 | 2)); the approx_* keys are R's default statistics = 'approximate'",
       call = "loess(y ~ x, span, degree, family, control = loess.control(surface = 'direct', statistics = 'exact', iterations))", packages = character(0))
})

case("nls_weighted_t4", "regression/nls", "stats::nls with weights (Gauss-Newton) and algorithm = 'port' with a lower-only and an upper-only bound", function(ctx) {
  set.seed(20261116)
  n <- 40; x <- seq(0, 8, length.out = n); s <- 0.2 * (1 + x / 2)
  y <- 10 * exp(-0.5 * x) + 2 + rnorm(n, 0, s); w <- 1 / (1 + x / 2)^2
  d <- ctx$gen_dataset("nls_weighted_t4", data.frame(x = round(x, 10), y = round(y, 10), w = round(w, 10)))
  ctl <- nls.control(tol = 1e-7, maxiter = 500, minFactor = 1e-10)
  f <- nls(y ~ a * exp(-b * x) + c, data = d, start = list(a = 8, b = 0.3, c = 1), weights = w, control = ctl)
  sm <- summary(f)$coefficients
  u <- nls(y ~ a * exp(-b * x) + c, data = d, start = list(a = 8, b = 0.3, c = 1), control = ctl)
  v <- list(w_coef = named(coef(f)), w_se = named(sm[, 2]), w_t = named(sm[, 3]), w_p = named(sm[, 4]),
            w_rss = deviance(f), w_sigma = summary(f)$sigma, w_df = df.residual(f),
            w_fitted = vec(fitted(f)), w_residuals = vec(d$y - fitted(f)),
            u_coef = named(coef(u)), u_rss = deviance(u), n = n)
  lo <- nls(y ~ a * exp(-b * x) + c, data = d, start = list(a = 8, b = 0.3, c = 2.5), weights = w, algorithm = "port",
            lower = c(-Inf, -Inf, 2.5), control = ctl)
  v$lower_coef <- named(coef(lo)); v$lower_rss <- deviance(lo)
  up <- nls(y ~ a * exp(-b * x) + c, data = d, start = list(a = 8, b = 0.3, c = 1), algorithm = "port",
            upper = c(Inf, 0.45, Inf), control = ctl)
  v$upper_coef <- named(coef(up)); v$upper_rss <- deviance(up)
  list(values = v, reference = "stats::nls(weights = w) (Gauss-Newton, nls.control(tol = 1e-7, minFactor = 1e-10)); nls(algorithm = 'port', lower = c(-Inf, -Inf, 2.5), weights = w); nls(algorithm = 'port', upper = c(Inf, 0.45, Inf))",
       call = "nls(y ~ a * exp(-b * x) + c, start = list(a = 8, b = 0.3, c = 1), weights = w)", packages = character(0))
})

case("step_null_t4", "regression/step", "stats::step when no predictor helps: backward and both from the full model reach y ~ 1; forward stays there", function(ctx) {
  set.seed(20261117)
  n <- 60; d <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = rnorm(n), y = rnorm(n))
  d <- ctx$gen_dataset("step_null_t4", data.frame(lapply(d, function(c) round(c, 10))))
  full <- lm(y ~ x1 + x2 + x3, data = d); null <- lm(y ~ 1, data = d)
  sc <- list(lower = ~ 1, upper = ~ x1 + x2 + x3)
  bw <- step(full, direction = "backward", trace = 0); bo <- step(full, direction = "both", trace = 0, scope = sc)
  fw <- step(null, direction = "forward", trace = 0, scope = sc)
  bic <- step(full, direction = "backward", trace = 0, k = log(n))
  drop_order <- function(s) vec(match(sub("^- ", "", as.character(s$anova$Step[-1])), c("x1", "x2", "x3")))
  v <- list(n = n,
            backward_aic = vec(bw$anova$AIC), backward_dropped = drop_order(bw), backward_final_p = length(coef(bw)),
            both_aic = vec(bo$anova$AIC), both_final_p = length(coef(bo)),
            forward_aic = vec(fw$anova$AIC), forward_final_p = length(coef(fw)),
            bic_aic = vec(bic$anova$AIC), bic_final_p = length(coef(bic)),
            null_extract_aic = extractAIC(null)[2], full_extract_aic = extractAIC(full)[2],
            null_extract_bic = extractAIC(null, k = log(n))[2])
  last1 <- lm(as.formula(paste("y ~", c("x1", "x2", "x3")[setdiff(1:3, v$backward_dropped[1:2])])), data = d)
  dr <- drop1(last1); v$drop1_last_aic <- vec(dr$AIC); v$drop1_last_rss <- vec(dr$RSS)
  list(values = v, reference = "stats::step(direction = 'backward' | 'both' | 'forward', k = 2 | log(n)); stats::drop1; stats::extractAIC",
       call = "step(lm(y ~ x1 + x2 + x3), direction = 'backward')", packages = character(0))
})
