# validation/r/cases/ml_ppr_svm.R -- PPR and fast-SVM references (T3-WP45: ML-34, ML-103; moved in ML-32).
#  * stats::ppr (Friedman's SMART, R's ppr.f): sm.method = 'supsmu' (bass 0 and 5, cv span and span 0.3),
#    'spline' (df 5) and 'gcvspline' (gcvpen 1 and 2); optlevel 0, 1, 2, 3; case weights; one predictor;
#    alpha, beta, gofn, gof, fitted values, edf, yb, ys and predict() on new rows.
#  * The two ridge smoothers of ppr.f on one (x, y): the super smoother (stats::supsmu, the same Fortran
#    routine ppr calls) and ppr's regression spline (15 B-spline coefficients at ppr.f's knot rule, sbart
#    with spar in [0, 1.5], tol 1e-2, eps .000244), reproduced by smooth.spline(all.knots = <ppr knots>,
#    control.spar = <ppr's>): df matching (icrit 3) and GCV (icrit 1).
#  * e1071::svm(type = 'C-classification', kernel = 'radial' | 'linear', cost 0.1 / 10, gamma 0.5,
#    scale = FALSE, tolerance = 1e-10), shrinking TRUE and FALSE (LIBSVM 3.x): decision values, bias,
#    support vectors, dual coefficients and predictions on labels coded 1 and 2 (a ring for the radial
#    kernel, a noisy half-plane for the linear one). Every quantity is oriented so that the larger
#    label (2) is the positive class.
suppressPackageStartupMessages(library(e1071))

.f32 <- function(v) readBin(writeBin(as.numeric(v), raw(), size = 4), "double", size = 4, n = length(v))

case("ppr_methods", "ml/ppr", "stats::ppr: supsmu (cv span, bass 5, span 0.3), spline (df 5), gcvspline (gcvpen 1, 2), optlevel 0-3, weights, one predictor, predict", function(ctx) {
  set.seed(20261125)
  n <- 200; x1 <- runif(n, -1, 1); x2 <- runif(n, -1, 1); x3 <- runif(n, -1, 1)
  y <- sin(2 * (0.8 * x1 + 0.6 * x2)) + (x2 - x3)^2 + rnorm(n, 0, 0.1)
  w <- runif(n, 0.5, 2)
  d <- ctx$gen_dataset("ppr_sim", data.frame(y = round(y, 10), x1 = round(x1, 10), x2 = round(x2, 10),
                                             x3 = round(x3, 10), w = round(w, 10)))
  newx <- data.frame(x1 = c(-1.2, -0.5, 0, 0.3, 0.7, 1.3, 0.05), x2 = c(0.9, -0.2, 0, 0.6, -0.8, 1.1, -0.35),
                     x3 = c(-0.4, 0.1, 0, -0.9, 0.5, 1.4, 0.2))
  v <- list(predict_newx = vec(as.numeric(as.matrix(newx))))   # 7 x 3, column-major (x1, x2, x3)
  put <- function(nm, f) {
    v[[paste0(nm, "_alpha")]] <<- vec(as.numeric(f$alpha)); v[[paste0(nm, "_beta")]] <<- vec(as.numeric(f$beta))
    v[[paste0(nm, "_gofn")]] <<- vec(f$gofn); v[[paste0(nm, "_gof")]] <<- f$gof
    v[[paste0(nm, "_fitted")]] <<- vec(fitted(f)); v[[paste0(nm, "_rss")]] <<- sum(residuals(f)^2)
    v[[paste0(nm, "_yb")]] <<- f$yb; v[[paste0(nm, "_ys")]] <<- f$ys
    if (any(f$edf > 0)) v[[paste0(nm, "_edf")]] <<- vec(f$edf)
  }
  fit <- function(...) ppr(y ~ x1 + x2 + x3, data = d, nterms = 2, max.terms = 3, ...)
  specs <- list(supsmu_b0 = list(), supsmu_b5 = list(bass = 5), supsmu_span03 = list(span = 0.3),
                spline_df5 = list(sm.method = "spline", df = 5), gcvspline = list(sm.method = "gcvspline"),
                gcvspline_pen2 = list(sm.method = "gcvspline", gcvpen = 2),
                supsmu_ol1 = list(optlevel = 1), supsmu_ol3 = list(optlevel = 3))
  for (nm in names(specs)) {
    f <- do.call(fit, specs[[nm]]); put(nm, f)
    if (nm %in% c("supsmu_b0", "spline_df5")) v[[paste0(nm, "_predict")]] <- vec(predict(f, newx))
  }
  # optlevel 0 does no backfitting and no backward elimination: R's object reports the first nterms
  # directions, but its fitted values use every term of the forward pass (max.terms = 3).
  f0 <- fit(optlevel = 0); f03 <- ppr(y ~ x1 + x2 + x3, data = d, nterms = 3, max.terms = 3, optlevel = 0)
  stopifnot(isTRUE(all.equal(fitted(f0), fitted(f03), tolerance = 0)), all(f0$gofn == 0))
  put("supsmu_ol0", f03)
  stopifnot(sum(residuals(fit(optlevel = 2))^2) != sum(residuals(fit(optlevel = 3))^2))
  fw <- ppr(y ~ x1 + x2 + x3, data = d, weights = w, nterms = 2, max.terms = 3); put("supsmu_weighted", fw)
  f1 <- ppr(y ~ x1, data = d, nterms = 1); put("supsmu_p1", f1)
  list(values = v, reference = "stats::ppr(nterms = 2, max.terms = 3, sm.method, bass, span, df, gcvpen, optlevel, weights) (Friedman's SMART, R's ppr.f)",
       call = "ppr(y ~ x1 + x2 + x3, data = ppr_sim, nterms = 2, max.terms = 3, ...)", packages = character(0))
})

case("ppr_smoothers", "ml/ppr", "ppr.f ridge smoothers: stats::supsmu (cv, bass 5, span 0.3, weights) and ppr's 15-coefficient regression spline (df 5, GCV) via smooth.spline", function(ctx) {
  set.seed(20261127)
  n <- 100; x <- runif(n, 0, 3); y <- sin(2 * x) + 0.3 * x + rnorm(n, 0, 0.2); w <- runif(n, 0.5, 2)
  d <- ctx$gen_dataset("ppr_smooth", data.frame(x = round(x, 10), y = round(y, 10), w = round(w, 10)))
  o <- order(d$x); xs <- d$x[o]; ys <- d$y[o]; ws <- d$w[o]
  stopifnot(!anyDuplicated(xs))
  v <- list()
  v$supsmu_cv <- vec(supsmu(xs, ys)$y); v$supsmu_bass5 <- vec(supsmu(xs, ys, bass = 5)$y)
  v$supsmu_span03 <- vec(supsmu(xs, ys, span = 0.3)$y); v$supsmu_weighted <- vec(supsmu(xs, ys, wt = ws)$y)
  # ppr.f spline(): dx = (x - x1)/(xn - x1); nk = min(n, 15) coefficients; inner knots at the
  # single-precision fraction p = (n - 1) * real(i - 4) / real(nk - 3), i = 5..nk.
  dx <- (xs - xs[1]) / (xs[n] - xs[1]); nk <- min(n, 15)
  inner <- vapply(5:nk, function(i) { p <- .f32(.f32((n - 1) * (i - 4)) / .f32(nk - 3)); ip <- floor(p); p <- p - ip
                                     (1 - p) * dx[ip + 1] + p * dx[ip + 2] }, 0)
  ctl <- list(low = 0, high = 1.5, tol = 1e-2, eps = .f32(0.000244), maxit = 500)
  s5 <- smooth.spline(dx, ys, all.knots = c(0, inner, 1), df = 5, control.spar = ctl)
  sg <- smooth.spline(dx, ys, all.knots = c(0, inner, 1), control.spar = ctl)
  sg2 <- smooth.spline(dx, ys, all.knots = c(0, inner, 1), penalty = 2, control.spar = ctl)
  v$spline_knots <- vec(c(0, inner, 1))
  for (nm in c("spline_df5", "spline_gcv", "spline_gcv_pen2")) {
    s <- switch(nm, spline_df5 = s5, spline_gcv = sg, spline_gcv_pen2 = sg2)
    v[[paste0(nm, "_y")]] <- vec(s$y); v[[paste0(nm, "_edf")]] <- s$df; v[[paste0(nm, "_spar")]] <- s$spar
    v[[paste0(nm, "_lambda")]] <- s$lambda; v[[paste0(nm, "_ratio")]] <- s$ratio; v[[paste0(nm, "_iter")]] <- s$iparms[["iter"]]
    v[[paste0(nm, "_crit")]] <- s$crit; v[[paste0(nm, "_coef")]] <- vec(s$fit$coef)
    # smooth.spline returns the smooth of its last Brent evaluation, not of the spar it reports;
    # the same spar refitted has no such gap.
    r <- smooth.spline(dx, ys, all.knots = c(0, inner, 1), spar = s$spar)
    v[[paste0(nm, "_refit_y")]] <- vec(r$y); v[[paste0(nm, "_refit_edf")]] <- r$df
  }
  list(values = v, reference = "stats::supsmu; stats::smooth.spline(all.knots = ppr.f knots, control.spar = list(low = 0, high = 1.5, tol = 0.01, eps = 0.000244)) = ppr.f spline()",
       call = "supsmu(x, y, span, bass, wt); smooth.spline(dx, y, all.knots, df = 5 | GCV, penalty, control.spar)", packages = character(0))
})
