# validation/r/cases/regression_sensitivity_t4.R -- Tier 4 (T4-WP07): E-values for hazard ratios with a
# common outcome (EValue's toRR.HR), sensemakr's benchmark bounds, adjusted SE/t/CI and RV_alpha, logit and
# probit marginal effects at the mean (delta method with the full gradient) and average comparisons.
#  * evalue_hr_common: EValue 4.1.4 evalues.HR(rare = FALSE), (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)).
#  * sensemakr_darfur_bounds: sensemakr 0.1.6 on the darfur data of case sensemakr_darfur (no village effects);
#    bounds for benchmarks female and age at kd = ky = 1, 2, 3, kd = 2 / ky = 1, and q = 0.5 / alpha = 0.1;
#    a ky = 8 bound clamps r2yz.dx to 1; adjusted_t and adjusted_estimate on a 5 x 5 contour grid.
#  * sensemakr_rv_alpha: robustness_value(t, dof, q, alpha), including the extreme-RV branch.
#  * me_mem_contrast: glm logit and probit; MEM and its SE with G = f(zbar) I + beta f'(zbar) xbar';
#    average comparisons mean(F(X^(v2) b) - F(X^(v1) b)) with gradient mean(f(z2) X2 - f(z1) X1); both
#    cross-checked against marginaleffects::slopes(newdata = "mean") and avg_comparisons.
suppressPackageStartupMessages({ library(sensemakr); library(EValue) })

case("evalue_hr_common", "regression/evalue", "EValue evalues.HR with a common outcome (VanderWeele 2020 conversion)", function(ctx) {
  inp <- data.frame(id = 1:3, est = c(0.6, 1.3, 0.5), lo = c(0.4, 0.9, NA), hi = c(0.9, 1.8, 0.8))
  d <- ctx$gen_dataset("evalue_hr_inputs", inp)
  v <- list()
  for (r in seq_len(nrow(d))) {
    m <- suppressMessages(evalues.HR(d$est[r], d$lo[r], d$hi[r], rare = FALSE))
    v[[sprintf("hr%d_rr", r)]] <- unname(m[1, 1])
    v[[sprintf("hr%d_rr_ci", r)]] <- vec(m[1, 2:3][!is.na(m[1, 2:3])])
    v[[sprintf("hr%d_e_point", r)]] <- unname(m[2, 1])
    v[[sprintf("hr%d_e_ci", r)]] <- unname(m[2, 2:3][!is.na(m[2, 2:3])][1])
  }
  toRR <- function(h) (1 - 0.5^sqrt(h)) / (1 - 0.5^sqrt(1 / h))
  stopifnot(abs(v$hr1_rr - toRR(0.6)) < 1e-15)
  list(values = v, reference = "EValue::evalues.HR(est, lo, hi, rare = FALSE); RR = (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR)) (VanderWeele 2020)",
       call = "evalues.HR(0.6, 0.4, 0.9, rare = FALSE); evalues.HR(1.3, 0.9, 1.8, rare = FALSE); evalues.HR(0.5, NA, 0.8, rare = FALSE)", packages = "EValue")
})

case("sensemakr_darfur_bounds", "regression/sensemakr", "sensemakr benchmark bounds, adjusted SE, t and CI (darfur, benchmarks female and age)", function(ctx) {
  d <- ctx$dataset("datasets/generated/sensemakr_darfur.csv")
  m <- lm(peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female, data = d)
  dof <- df.residual(m)
  md <- lm(directlyharmed ~ age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female, data = d)
  v <- list(age_r2dxj = unname(partial_r2(md, covariates = "age")), age_r2yxj = unname(partial_r2(m, covariates = "age")),
            female_r2yxj = unname(partial_r2(m, covariates = "female")))
  cfgs <- list(k1 = list(kd = 1, ky = 1, q = 1, alpha = 0.05), k2 = list(kd = 2, ky = 2, q = 1, alpha = 0.05),
               k3 = list(kd = 3, ky = 3, q = 1, alpha = 0.05), kd2ky1 = list(kd = 2, ky = 1, q = 1, alpha = 0.05),
               q05a10 = list(kd = 1, ky = 1, q = 0.5, alpha = 0.10))
  for (nm in names(cfgs)) {
    g <- cfgs[[nm]]
    s <- sensemakr(m, treatment = "directlyharmed", benchmark_covariates = c("female", "age"), kd = g$kd, ky = g$ky, q = g$q, alpha = g$alpha)
    b <- s$bounds; lab <- c("female", "age")
    v[[paste0(nm, "_rv_q")]] <- unname(s$sensitivity_stats$rv_q)
    v[[paste0(nm, "_rv_qa")]] <- unname(s$sensitivity_stats$rv_qa)
    v[[paste0(nm, "_r2dz_x")]] <- named(setNames(b$r2dz.x, lab))
    v[[paste0(nm, "_r2yz_dx")]] <- named(setNames(b$r2yz.dx, lab))
    v[[paste0(nm, "_adj_est")]] <- named(setNames(b$adjusted_estimate, lab))
    v[[paste0(nm, "_adj_se")]] <- named(setNames(b$adjusted_se, lab))
    v[[paste0(nm, "_adj_t")]] <- named(setNames(as.numeric(b$adjusted_t), lab))
    v[[paste0(nm, "_lower")]] <- named(setNames(b$adjusted_lower_CI, lab))
    v[[paste0(nm, "_upper")]] <- named(setNames(b$adjusted_upper_CI, lab))
  }
  # ky = 8: the implied r2yz.dx exceeds 1; sensemakr warns and sets it to 1 (adjusted SE 0, t infinite)
  s8 <- withCallingHandlers(sensemakr(m, treatment = "directlyharmed", benchmark_covariates = "female", kd = 1, ky = 8),
                            warning = function(w) invokeRestart("muffleWarning"))
  stopifnot(s8$bounds$r2yz.dx == 1, s8$bounds$adjusted_se == 0)
  v$clamp_r2dz_x <- s8$bounds$r2dz.x; v$clamp_r2yz_dx <- s8$bounds$r2yz.dx; v$clamp_adj_est <- s8$bounds$adjusted_estimate
  # kd = 400 is impossible (r2dz.x >= 1): sensemakr stops
  stopifnot(inherits(try(sensemakr(m, "directlyharmed", benchmark_covariates = "female", kd = 400), silent = TRUE), "try-error"))
  # contour grid (generate_contour_data(grid_size = 5, max_r2 = 0.5)): rows r2dz.x, columns r2yz.dx
  est <- unname(coef(m)["directlyharmed"]); se <- summary(m)$coefficients["directlyharmed", 2]
  grid <- seq(0, 0.5, length.out = 5)
  v$grid <- vec(grid)
  v$contour_adj_est <- vec(as.numeric(t(outer(grid, grid, Vectorize(function(a, b) adjusted_estimate(estimate = est, se = se, dof = dof, r2dz.x = a, r2yz.dx = b))))))
  v$contour_adj_t <- vec(as.numeric(t(outer(grid, grid, Vectorize(function(a, b) as.numeric(adjusted_t(estimate = est, se = se, dof = dof, r2dz.x = a, r2yz.dx = b)))))))
  v$n <- nrow(d)
  list(values = v, dataset = "validation/datasets/generated/sensemakr_darfur.csv",
       reference = "sensemakr::sensemakr(lm, 'directlyharmed', benchmark_covariates = c('female', 'age'), kd, ky, q, alpha)$bounds (ovb_partial_r2_bound, adjusted_estimate, adjusted_se, adjusted_t, CI with qt(alpha/2, df)); adjusted_estimate and adjusted_t on a grid",
       call = "sensemakr(lm(peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female), 'directlyharmed', c('female', 'age'), kd = 1:3)", packages = "sensemakr")
})

case("sensemakr_rv_alpha", "regression/sensemakr", "sensemakr robustness_value with alpha (qt(alpha/2, dof - 1), extreme-RV branch)", function(ctx) {
  inp <- data.frame(id = 1:6, t = c(30, 30, 3, 2, -4, 2.6579965959341383), dof = c(10, 10, 10, 100, 50, 1268),
                    q = c(1, 1, 1, 1, 0.5, 1), alpha = c(0.05, 1, 0.05, 0.05, 0.1, 0.05))
  d <- ctx$gen_dataset("sensemakr_rv_inputs", inp)
  rv <- vapply(seq_len(nrow(d)), function(r) as.numeric(robustness_value(d$t[r], d$dof[r], q = d$q[r], alpha = d$alpha[r])), 0)
  xrv <- vapply(seq_len(nrow(d)), function(r) isTRUE(attr(robustness_value(d$t[r], d$dof[r], q = d$q[r], alpha = d$alpha[r]), "is.xrv")), TRUE)
  stopifnot(xrv[1], !xrv[3])
  list(values = list(rv = vec(rv), is_xrv = vec(as.numeric(xrv))),
       reference = "sensemakr::robustness_value.numeric(t, dof, q, alpha): f_crit = |qt(alpha/2, dof - 1)| / sqrt(dof - 1); extreme_robustness_value when f_q > 1 / f_crit",
       call = "robustness_value(30, 10, q = 1, alpha = 0.05); ...", packages = "sensemakr")
})

case("me_mem_contrast", "regression/marginal_effects", "Logit and probit marginal effects at the mean (full delta-method gradient) and average comparisons", function(ctx) {
  set.seed(20261111)
  n <- 400
  x1 <- round(rnorm(n, 1, 1), 6); x2 <- round(runif(n, 0, 4), 6)
  y <- rbinom(n, 1, plogis(-1.0 + 0.8 * x1 - 0.5 * x2))
  d <- ctx$gen_dataset("me_mem_contrast", data.frame(y = y, x1 = x1, x2 = x2))
  ctl <- glm.control(epsilon = 1e-14, maxit = 100)
  v <- list(); vals <- c(0, 1, 2.5)
  for (link in c("logit", "probit")) {
    fit <- glm(y ~ x1 + x2, family = binomial(link), data = d, control = ctl)
    b <- coef(fit); V <- vcov(fit); X <- model.matrix(fit)
    Fc <- if (link == "logit") plogis else pnorm
    f <- if (link == "logit") dlogis else dnorm
    fp <- if (link == "logit") function(z) dlogis(z) * (1 - 2 * plogis(z)) else function(z) -z * dnorm(z)
    xb <- colMeans(X); zb <- sum(xb * b)
    mem <- b * f(zb); G <- diag(f(zb), length(b)) + outer(b * fp(zb), xb)
    mem_se <- sqrt(diag(G %*% V %*% t(G))); mz <- mem / mem_se
    s <- as.data.frame(marginaleffects::slopes(fit, newdata = "mean"))
    s <- s[match(c("x1", "x2"), s$term), ]
    stopifnot(max(abs(s$estimate / mem[2:3] - 1)) < 1e-7, max(abs(s$std.error / mem_se[2:3] - 1)) < 1e-6)
    v[[paste0(link, "_coef")]] <- named(b)
    v[[paste0(link, "_mem")]] <- named(mem); v[[paste0(link, "_mem_se")]] <- named(mem_se)
    v[[paste0(link, "_mem_z")]] <- named(mz); v[[paste0(link, "_mem_p")]] <- named(2 * pnorm(-abs(mz)))
    est <- se <- numeric(0)
    for (i in 1:2) for (j in (i + 1):3) {
      X1 <- X; X1[, "x1"] <- vals[i]; X2 <- X; X2[, "x1"] <- vals[j]
      z1 <- drop(X1 %*% b); z2 <- drop(X2 %*% b)
      e <- mean(Fc(z2) - Fc(z1)); g <- colMeans(f(z2) * X2 - f(z1) * X1); s2 <- sqrt(drop(t(g) %*% V %*% g))
      ac <- as.data.frame(marginaleffects::avg_comparisons(fit, variables = list(x1 = c(vals[i], vals[j]))))
      stopifnot(abs(ac$estimate / e - 1) < 1e-12, abs(ac$std.error / s2 - 1) < 1e-6)
      est <- c(est, e); se <- c(se, s2)
    }
    v[[paste0(link, "_contrast_est")]] <- vec(est); v[[paste0(link, "_contrast_se")]] <- vec(se)
    v[[paste0(link, "_contrast_p")]] <- vec(2 * pnorm(-abs(est / se)))
  }
  v$contrast_values <- vec(vals); v$n <- n
  list(values = v, reference = "stats::glm(binomial(logit | probit)); MEM and SE by the delta method with G = f(zbar) I + beta f'(zbar) xbar'; average comparisons of x1 (pairs of 0, 1, 2.5) with gradient mean(f(z2) X2 - f(z1) X1); cross-checked against marginaleffects::slopes(newdata = 'mean') and avg_comparisons",
       call = "glm(y ~ x1 + x2, binomial(link), epsilon = 1e-14); slopes(fit, newdata = 'mean'); avg_comparisons(fit, variables = list(x1 = c(0, 1)))", packages = c("marginaleffects"))
})
