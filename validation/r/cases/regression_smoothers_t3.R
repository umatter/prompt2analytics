# validation/r/cases/regression_smoothers_t3.R -- smoother, GLS and regression-helper references
# (T3-WP11: REG-15, REG-46, REG-53, REG-55, REG-44; moved in REG-14, REG-45, REG-31, REG-9, REG-43;
# TOOL-19 core; X-16 sensemakr part). Package copy of design Appendix A.6 (canonical; adds keys and
# the REG-44 planner cases below).
#  * smoothers_gcv_periodic_loess: smooth.spline GCV vs CV (cv = FALSE | TRUE, df = 6), supsmu (cv, span 0.3,
#    bass 5; periodic and not), loess on a predictor with a 60% mass point at 0 (surface = "direct"; where the
#    neighbourhood has zero width R returns 0 with a warning, and the reference is lowess()'s rule, the mean of
#    the tied responses).
#  * gls_compsymm: nlme::gls with corCompSymm(form = ~ 1 | g), REML and ML, rho estimated and fixed at 0.3;
#    corAR1(form = ~ 1 | g) by REML.
#  * loess_x10_direct, loess_skewed_direct: stats::loess(surface = "direct") on the old test data (span * n
#    = 7.5, REG-45) and on lognormal x (REG-14); predictions at interior points.
#  * supsmu_sine: stats::supsmu on the legacy validate_regression_diag.R DGP (supersedes supsmu_test.csv).
#  * line_tukey_outliers: stats::line on the old line test's data (pinned by Tier 4, REG-12).
#  * nls_expdecay, nls_mm: stats::nls (Gauss-Newton, tol 1e-7 / 1e-8: the tightest R reaches) on the old nls test data.
#  * sensemakr_darfur: sensemakr 0.1.6 on darfur without village effects (classical OLS t, the bound for
#    female with kd = 1); the raw benchmark partial R2 of female with the treatment is the X-16 pin.
#  * evalue_formulas: EValue 4.1.4 evalues.RR / OR / HR / MD.
suppressPackageStartupMessages({ library(nlme); library(sensemakr); library(EValue) })

case("smoothers_gcv_periodic_loess", "regression/smoothers", "smooth.spline GCV vs CV, supsmu periodic, loess with a mass point (R's behaviour)", function(ctx) {
  set.seed(20261127)
  n <- 100; x <- sort(runif(n)); y <- sin(2 * pi * x) + rnorm(n, 0, 0.3)
  xm <- c(rep(0, 60), runif(40, 0, 1)); ym <- 1 + 2 * xm + rnorm(100, 0, 0.3)
  d <- ctx$gen_dataset("smoothers_t3", data.frame(x = round(x, 10), y = round(y, 10), xm = round(xm, 10), ym = round(ym, 10)))
  s_gcv <- smooth.spline(d$x, d$y, cv = FALSE); s_cv <- smooth.spline(d$x, d$y, cv = TRUE); s_df6 <- smooth.spline(d$x, d$y, df = 6)
  sp <- supsmu(d$x, d$y, periodic = TRUE); sn <- supsmu(d$x, d$y, periodic = FALSE)
  v <- list(gcv_df = s_gcv$df, gcv_lambda = s_gcv$lambda, gcv_crit = s_gcv$cv.crit, gcv_fit_head = vec(head(predict(s_gcv, d$x)$y, 10)),
            cv_df = s_cv$df, cv_lambda = s_cv$lambda, cv_crit = s_cv$cv.crit,
            dfmatch6_df = s_df6$df, dfmatch6_lambda = s_df6$lambda,
            supsmu_periodic_y = vec(sp$y), supsmu_plain_y_head = vec(head(sn$y, 10)), supsmu_plain_y = vec(sn$y),
            supsmu_span03_periodic_y = vec(supsmu(d$x, d$y, span = 0.3, periodic = TRUE)$y),
            supsmu_span03_plain_y = vec(supsmu(d$x, d$y, span = 0.3)$y),
            supsmu_bass5_y = vec(supsmu(d$x, d$y, bass = 5)$y),
            supsmu_bass5_periodic_y = vec(supsmu(d$x, d$y, bass = 5, periodic = TRUE)$y))
  lo <- tryCatch(suppressWarnings(loess(ym ~ xm, data = d, span = 0.5, degree = 1)), error = function(e) NULL)
  v$loess_masspoint_fit_head <- if (is.null(lo)) vec(NA) else vec(head(fitted(lo), 10))
  lod <- suppressWarnings(loess(ym ~ xm, data = d, span = 0.5, degree = 1, control = loess.control(surface = "direct")))
  lw <- lowess(d$xm, d$ym, f = 0.5, iter = 0)          # lowess's rule at a zero-width neighbourhood: weight 1 on the ties
  zero_width <- d$xm == 0                               # 60 ties > q = floor(100 * 0.5) = 50
  expected <- fitted(lod); expected[zero_width] <- lw$y[1]
  v$loess_masspoint_direct_fit <- vec(fitted(lod)); v$loess_masspoint_expected <- vec(expected)
  v$loess_masspoint_tie_mean <- mean(d$ym[zero_width]); v$lowess_masspoint_y0 <- lw$y[1]
  v$loess_masspoint_zero_width <- vec(as.numeric(zero_width))
  list(values = v, reference = "stats::smooth.spline(cv = FALSE | TRUE, df = 6); stats::supsmu(span = 'cv' | 0.3, bass = 0 | 5, periodic); stats::loess(span = 0.5, degree = 1, surface = 'direct') with 60% ties at 0, lowess(f = 0.5, iter = 0) at the ties",
       call = "smooth.spline(x, y, cv); supsmu(x, y, periodic = TRUE); loess(ym ~ xm, span = 0.5, degree = 1, control = loess.control(surface = 'direct'))", packages = character(0))
})

case("gls_compsymm", "regression", "nlme::gls with corCompSymm(form = ~ 1 | g), REML and ML: rho estimated", function(ctx) {
  set.seed(20261120)
  G <- 30; m <- 5
  g <- rep(1:G, each = m); u <- rnorm(G, 0, 0.8)[g]
  x <- rnorm(G * m); z <- rnorm(G)[g]
  y <- 1 + 0.5 * x - 0.3 * z + u + rnorm(G * m, 0, 1)
  d <- ctx$gen_dataset("gls_cs", data.frame(g = g, y = round(y, 10), x = round(x, 10), z = round(z, 10)))
  ctl <- glsControl(tolerance = 1e-12, msTol = 1e-12)
  v <- list()
  for (meth in c("REML", "ML")) {
    f <- gls(y ~ x + z, data = d, correlation = corCompSymm(form = ~ 1 | g), method = meth, control = ctl)
    tag <- tolower(meth)
    v[[paste0(tag, "_coef")]] <- named(coef(f)); v[[paste0(tag, "_se")]] <- named(sqrt(diag(vcov(f))))
    v[[paste0(tag, "_rho")]] <- unname(coef(f$modelStruct$corStruct, unconstrained = FALSE))
    v[[paste0(tag, "_sigma")]] <- f$sigma; v[[paste0(tag, "_loglik")]] <- as.numeric(logLik(f))
    v[[paste0(tag, "_aic")]] <- AIC(f); v[[paste0(tag, "_bic")]] <- BIC(f)
    ff <- gls(y ~ x + z, data = d, correlation = corCompSymm(0.3, form = ~ 1 | g, fixed = TRUE), method = meth)
    v[[paste0("cs03_", tag, "_coef")]] <- named(coef(ff)); v[[paste0("cs03_", tag, "_se")]] <- named(sqrt(diag(vcov(ff))))
    v[[paste0("cs03_", tag, "_sigma")]] <- ff$sigma; v[[paste0("cs03_", tag, "_loglik")]] <- as.numeric(logLik(ff))
    v[[paste0("cs03_", tag, "_aic")]] <- AIC(ff); v[[paste0("cs03_", tag, "_bic")]] <- BIC(ff)
  }
  a <- gls(y ~ x + z, data = d, correlation = corAR1(form = ~ 1 | g), control = ctl)
  v$ar1g_rho <- unname(coef(a$modelStruct$corStruct, unconstrained = FALSE))
  v$ar1g_coef <- named(coef(a)); v$ar1g_se <- named(sqrt(diag(vcov(a)))); v$ar1g_sigma <- a$sigma
  v$ar1g_loglik <- as.numeric(logLik(a))
  # Tight refit: the profile (restricted) log-likelihood that nlme maximises, written out (it reproduces
  # logLik(gls) to 1e-13 at any fixed rho) and maximised by optimize(tol = 1e-12). nlme's own optimiser
  # stops up to 2e-6 short in rho (AR(1) within g), so these keys are the pins and nlme's the cross-check.
  X <- cbind(1, d$x, d$z); N <- nrow(X); p <- ncol(X); blocks <- split(seq_len(N), d$g)
  prof <- function(rho, type, reml) {
    wh <- function(vv) { vv <- as.matrix(vv); out <- vv
      for (i in blocks) { mm <- length(i); vi <- vv[i, , drop = FALSE]
        if (type == "cs") out[i, ] <- (vi - matrix((1 - sqrt((1 - rho) / (1 + (mm - 1) * rho))) * colMeans(vi), mm, ncol(vv), byrow = TRUE)) / sqrt(1 - rho)
        else { w <- vi; if (mm > 1) w[-1, ] <- (vi[-1, , drop = FALSE] - rho * vi[-mm, , drop = FALSE]) / sqrt(1 - rho^2); out[i, ] <- w } }
      out }
    ld <- sum(sapply(blocks, function(i) { mm <- length(i); if (type == "cs") (mm - 1) * log(1 - rho) + log(1 + (mm - 1) * rho) else (mm - 1) * log(1 - rho^2) }))
    Xs <- wh(X); ys <- drop(wh(d$y)); q <- qr(Xs); b <- qr.coef(q, ys); sse <- sum((ys - Xs %*% b)^2)
    ll <- if (reml) -(N - p) / 2 * (log(2 * pi) + 1 + log(sse / (N - p))) - 0.5 * ld - sum(log(abs(diag(qr.R(q)))))
          else -N / 2 * (log(2 * pi) + 1 + log(sse / N)) - 0.5 * ld
    list(ll = ll, b = setNames(drop(b), c("(Intercept)", "x", "z")), se = setNames(sqrt(diag(chol2inv(qr.R(q))) * sse / (N - p)), c("(Intercept)", "x", "z")),
         sigma = sqrt(sse / (if (reml) N - p else N)))
  }
  lo_cs <- -1 / (max(table(d$g)) - 1)
  for (spec in list(c("cs", "reml"), c("cs", "ml"), c("ar1", "reml"))) {
    lo <- if (spec[1] == "cs") lo_cs else -1
    o <- optimize(function(r) prof(r, spec[1], spec[2] == "reml")$ll, c(lo + 1e-8, 1 - 1e-8), maximum = TRUE, tol = 1e-12)
    r <- prof(o$maximum, spec[1], spec[2] == "reml")
    tag <- if (spec[1] == "cs") paste0(spec[2], "_profile") else "ar1g_profile"
    v[[paste0(tag, "_rho")]] <- o$maximum; v[[paste0(tag, "_coef")]] <- named(r$b); v[[paste0(tag, "_se")]] <- named(r$se)
    v[[paste0(tag, "_sigma")]] <- r$sigma; v[[paste0(tag, "_loglik")]] <- r$ll
  }
  v$n <- nrow(d); v$n_groups <- length(unique(d$g))
  list(values = v, reference = "nlme::gls(correlation = corCompSymm(form = ~ 1 | g) | corCompSymm(0.3, fixed = TRUE) | corAR1(form = ~ 1 | g), method = 'REML' | 'ML', glsControl(tolerance = 1e-12, msTol = 1e-12)); *_profile: nlme's profile likelihood maximised by optimize(tol = 1e-12)",
       call = "gls(y ~ x + z, data = gls_cs, correlation = corCompSymm(form = ~ 1 | g))", packages = "nlme")
})

case("loess_x10_direct", "regression/smoothers", "stats::loess(surface = 'direct') on x = 1:10 (span * n = 7.5): degrees 0-2, span > 1, predictions", function(ctx) {
  d <- ctx$gen_dataset("loess_x10", data.frame(x = 1:10, y = c(2.5, 3.8, 6.1, 7.9, 10.2, 12.0, 14.3, 16.1, 18.0, 20.2)))
  dir <- loess.control(surface = "direct")
  f2 <- loess(y ~ x, data = d, span = 0.75, degree = 2, control = dir)
  f1 <- loess(y ~ x, data = d, span = 0.75, degree = 1, control = dir)
  f0 <- loess(y ~ x, data = d, span = 0.5, degree = 0, control = dir)
  f12 <- loess(y ~ x, data = d, span = 1.2, degree = 2, control = dir)
  nx <- c(2.5, 5.5, 9.5)
  v <- list(fit_s075_d2 = vec(fitted(f2)), trace_s075_d2 = f2$trace.hat, enp_r_s075_d2 = f2$enp,
            fit_s075_d1 = vec(fitted(f1)), trace_s075_d1 = f1$trace.hat,
            fit_s05_d0 = vec(fitted(f0)), fit_s12_d2 = vec(fitted(f12)),
            predict_x = vec(nx), predict_s075_d2 = vec(predict(f2, data.frame(x = nx))), n = nrow(d))
  list(values = v, reference = "stats::loess(control = loess.control(surface = 'direct')); trace.hat exact (n < 500)",
       call = "loess(y ~ x, span = 0.75, degree = 2, control = loess.control(surface = 'direct'))", packages = character(0))
})

case("loess_skewed_direct", "regression/smoothers", "stats::loess(surface = 'direct') on lognormal x (neighbourhoods mostly to one side)", function(ctx) {
  set.seed(20261128)
  x <- rlnorm(100); y <- sin(x) + rnorm(100, 0, 0.3)
  d <- ctx$gen_dataset("loess_skewed", data.frame(x = round(x, 10), y = round(y, 10)))
  dir <- loess.control(surface = "direct")
  f2 <- loess(y ~ x, data = d, span = 0.75, degree = 2, control = dir)
  f1 <- loess(y ~ x, data = d, span = 0.3, degree = 1, control = dir)
  v <- list(fit_s075_d2 = vec(fitted(f2)), trace_s075_d2 = f2$trace.hat, fit_s03_d1 = vec(fitted(f1)), trace_s03_d1 = f1$trace.hat, n = nrow(d))
  list(values = v, reference = "stats::loess(control = loess.control(surface = 'direct'))",
       call = "loess(y ~ x, span = 0.75, degree = 2, control = loess.control(surface = 'direct'))", packages = character(0))
})

case("supsmu_sine", "regression/smoothers", "stats::supsmu on the legacy sine DGP (n = 100 on [0, 2 pi]): cv, span 0.3, bass 5, weights, ties", function(ctx) {
  set.seed(42); n <- 100
  x <- sort(runif(n, 0, 2 * pi)); y <- sin(x) + rnorm(n, 0, 0.3)
  w <- round(runif(n, 0.5, 2), 3)
  d <- ctx$gen_dataset("supsmu_sine", data.frame(x = x, y = y, w = w, xr = round(x, 1)))
  s <- supsmu(d$x, d$y); s3 <- supsmu(d$x, d$y, span = 0.3); sb <- supsmu(d$x, d$y, bass = 5)
  sw <- supsmu(d$x, d$y, wt = d$w); st <- supsmu(d$xr, d$y)
  v <- list(cv_x = vec(s$x), cv_y = vec(s$y), span03_y = vec(s3$y), bass5_y = vec(sb$y), weighted_y = vec(sw$y),
            ties_x = vec(st$x), ties_y = vec(st$y), n = n)
  list(values = v, reference = "stats::supsmu(span = 'cv' | 0.3, bass = 0 | 5, wt)",
       call = "supsmu(x, y); supsmu(x, y, span = 0.3); supsmu(x, y, bass = 5); supsmu(x, y, wt = w); supsmu(round(x, 1), y)", packages = character(0))
})

case("line_tukey_outliers", "regression/line", "stats::line (R >= 3.5 tukeyline) on the old line test's data, iter = 1, 2, 3", function(ctx) {
  y <- c(2.69, 2.72, 3.29, 3.82, 14.41, 5.00, 4.87, 4.88, 6.28, 6.47, 7.98, 8.13, 8.54, 9.21, 1.17,
         9.71, 10.22, 11.09, 11.57, 11.97, 12.50, 13.61, 13.35, 13.95, 26.70, 14.85, 15.42, 16.69, 16.47, 17.15)
  d <- ctx$gen_dataset("line_outliers", data.frame(x = 1:30, y = y))
  v <- list(iter1_coef = vec(coef(line(d$x, d$y, iter = 1))), iter2_coef = vec(coef(line(d$x, d$y, iter = 2))),
            iter3_coef = vec(coef(line(d$x, d$y, iter = 3))), ols_coef = vec(coef(lm(y ~ x, data = d))), n = nrow(d))
  list(values = v, reference = "stats::line(iter = 1 | 2 | 3) (R >= 3.5); stats::lm", call = "line(x, y, iter)", packages = character(0))
})

case("nls_expdecay", "regression/nls", "stats::nls(y ~ a * exp(-b * x) + c) on the old nls test data", function(ctx) {
  d <- ctx$gen_dataset("nls_expdecay", data.frame(x = seq(0, 5, by = 0.5), y = c(11.8, 9.7, 8.0, 6.5, 5.7, 4.8, 4.2, 3.7, 3.4, 3.1, 2.9)))
  f <- nls(y ~ a * exp(-b * x) + c, data = d, start = list(a = 8, b = 0.3, c = 1), control = nls.control(tol = 1e-7, maxiter = 500))
  s <- summary(f)$coefficients
  v <- list(coef = named(coef(f)), se = named(s[, 2]), t = named(s[, 3]), p = named(s[, 4]),
            rss = deviance(f), sigma = summary(f)$sigma, df = df.residual(f), n = nrow(d))
  list(values = v, reference = "stats::nls (Gauss-Newton, nls.control(tol = 1e-7); the tightest tolerance R reaches on these data)", call = "nls(y ~ a * exp(-b * x) + c, start = list(a = 8, b = 0.3, c = 1))", packages = character(0))
})

case("nls_mm", "regression/nls", "stats::nls(V ~ Vmax * S / (Km + S)) on the old Michaelis-Menten test data", function(ctx) {
  d <- ctx$gen_dataset("nls_mm", data.frame(S = c(0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0), V = c(28.6, 65.0, 100.0, 133.3, 166.7, 181.8, 190.5, 196.1)))
  f <- nls(V ~ Vmax * S / (Km + S), data = d, start = list(Vmax = 150, Km = 0.05), control = nls.control(tol = 1e-8, maxiter = 500))
  s <- summary(f)$coefficients
  v <- list(coef = named(coef(f)), se = named(s[, 2]), t = named(s[, 3]), p = named(s[, 4]),
            rss = deviance(f), sigma = summary(f)$sigma, df = df.residual(f), n = nrow(d))
  list(values = v, reference = "stats::nls (Gauss-Newton, nls.control(tol = 1e-8); the tightest tolerance R reaches on these data)", call = "nls(V ~ Vmax * S / (Km + S), start = list(Vmax = 150, Km = 0.05))", packages = character(0))
})

case("sensemakr_darfur", "regression/sensemakr", "sensemakr on darfur (no village effects): partial R2, RV, benchmark female (kd = 1)", function(ctx) {
  dd <- sensemakr::darfur
  dd <- data.frame(peacefactor = as.numeric(dd$peacefactor), directlyharmed = dd$directlyharmed, age = dd$age,
                   farmer_dar = dd$farmer_dar, herder_dar = dd$herder_dar, pastvoted = dd$pastvoted,
                   hhsize_darfur = dd$hhsize_darfur, female = dd$female)
  d <- ctx$gen_dataset("sensemakr_darfur", dd)
  m <- lm(peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female, data = d)
  s <- sensemakr(m, treatment = "directlyharmed", benchmark_covariates = "female", kd = 1)
  tt <- summary(m)$coefficients["directlyharmed", ]; dof <- df.residual(m)
  md <- lm(directlyharmed ~ age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female, data = d)
  b <- s$bounds
  v <- list(estimate = unname(tt[1]), se = unname(tt[2]), t = unname(tt[3]), dof = dof,
            r2yd_x = unname(partial_r2(m, covariates = "directlyharmed")),
            rv_q = unname(robustness_value(m, covariates = "directlyharmed", q = 1, alpha = 1)),
            rv_q05 = unname(robustness_value(m, covariates = "directlyharmed", q = 0.5, alpha = 1)),
            rv_qa = unname(robustness_value(m, covariates = "directlyharmed", q = 1, alpha = 0.05)),
            female_r2dxj = unname(partial_r2(md, covariates = "female")),
            female_r2yxj = unname(partial_r2(m, covariates = "female")),
            bound_r2dz_x = b$r2dz.x, bound_r2yz_dx = b$r2yz.dx, bound_adjusted_estimate = b$adjusted_estimate,
            bound_adjusted_se = b$adjusted_se, bound_adjusted_lower = b$adjusted_lower_CI, bound_adjusted_upper = b$adjusted_upper_CI,
            n = nrow(d))
  list(values = v, reference = "sensemakr::sensemakr(lm, treatment = 'directlyharmed', benchmark_covariates = 'female', kd = 1); partial_r2; robustness_value",
       call = "sensemakr(lm(peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female), 'directlyharmed', 'female')", packages = "sensemakr")
})

case("evalue_formulas", "regression/evalue", "EValue evalues.RR / OR / HR / MD (VanderWeele and Ding 2017)", function(ctx) {
  inp <- data.frame(id = 1:8,
                    type = c("RR", "RR", "RR", "OR", "OR", "HR", "HR", "MD"),
                    est = c(2.5, 0.6, 2.0, 2.5, 4.0, 1.5, 4.0, 0.5),
                    lo = c(1.8, 0.45, 0.9, 1.5, 2.0, 1.2, 2.0, NA),
                    hi = c(3.5, 0.8, 3.0, 4.0, 8.0, 1.9, 8.0, NA),
                    rare = c(NA, NA, NA, 1, 0, 1, 0, NA),
                    se = c(NA, NA, NA, NA, NA, NA, NA, 0.1))
  d <- ctx$gen_dataset("evalue_inputs", inp)
  ev <- function(r) {                 # rows "RR" and "E-values"; columns point, lower, upper
    m <- suppressMessages(switch(d$type[r], RR = evalues.RR(d$est[r], d$lo[r], d$hi[r]),
                OR = evalues.OR(d$est[r], d$lo[r], d$hi[r], rare = d$rare[r] == 1),
                HR = evalues.HR(d$est[r], d$lo[r], d$hi[r], rare = d$rare[r] == 1),
                MD = evalues.MD(d$est[r], se = d$se[r])))
    list(rr = unname(m[1, ]), e = unname(m[2, ]))
  }
  v <- list()
  for (r in seq_len(nrow(d))) {
    e <- ev(r)
    v[[sprintf("row%d_rr", r)]] <- e$rr[1]                                  # point estimate on the RR scale
    v[[sprintf("row%d_rr_ci", r)]] <- vec(e$rr[2:3])                        # CI limits on the RR scale
    v[[sprintf("row%d_e_point", r)]] <- e$e[1]
    v[[sprintf("row%d_e_ci", r)]] <- e$e[2:3][!is.na(e$e[2:3])][1]          # the limit closest to the null (1 if the CI crosses it)
  }
  v$evalue_rr_3_9 <- unname(evalues.RR(3.9)[2, 1])
  list(values = v, reference = "EValue::evalues.RR, evalues.OR(rare), evalues.HR(rare), evalues.MD(se); e_ci is EValue's E-value for the confidence limit closest to the null",
       call = "evalues.RR(2.5, 1.8, 3.5); evalues.OR(4, 2, 8, rare = FALSE); evalues.HR(1.5, 1.2, 1.9, rare = TRUE); evalues.MD(0.5, se = 0.1)", packages = "EValue")
})
