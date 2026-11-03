# validation/r/cases/linalg_glm.R -- references for the shared solve/inverse API and the shared GLM
# (T3-WP01: REG-59; the primitives REG-4, REG-18, ECON-59..61, CAUS-20, CAUS-38, CAUS-113 build on).
#  * base::qr (LINPACK dqrdc2, tol = 1e-7, the lm.fit path): rank, pivot, qraux, aliased columns,
#    lm.fit / lm.wfit coefficients (NA for aliased columns), zero weights;
#  * stats::glm.fit (IRLS, Fisher scoring) for binomial logit / probit / cloglog, poisson log with
#    offset, poisson identity, quasipoisson, quasibinomial with prior weights and fractional
#    responses, gaussian identity and log: coefficients, SEs (summary.glm), deviance, AIC, logLik at
#    epsilon = 1e-14, and the iteration counts at R's default glm.control (epsilon 1e-8, maxit 25);
#    a separated logit fit (R warns "fitted probabilities numerically 0 or 1 occurred") and an
#    aliased glm design (glm.fit's QR tolerance is min(1e-7, epsilon / 1000); the aliased fit uses
#    epsilon = 1e-10, so tol = 1e-13);
#  * the TMLE fluctuation forms glm(y ~ -1 + H, offset, quasibinomial) and
#    glm(y ~ 1, offset, weights, quasibinomial) (glm() refits the null deviance when an offset and
#    an intercept are present);
#  * a dummy-trap design [1, x1, d, 1 - d]: rank 3, the identified x1 coefficient and the RSS that
#    any generalized inverse of X'X reproduces.
# T3-WP01 package copy (canonical): the design's Appendix A.1 plus the keys marked "WP01 plan".
case("linalg_qr_glm", "linalg", "dqrdc2 rank/pivot/aliasing (lm.fit, lm.wfit) and glm.fit families (logit, probit, poisson + offset, quasibinomial + weights, separation)", function(ctx) {
  set.seed(20261210)
  n <- 120
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- x1 - 2 * x2                    # exactly collinear with x1, x2
  x4 <- rnorm(n); off <- log(runif(n, 0.5, 2))
  yb <- rbinom(n, 1, plogis(-0.3 + 0.8 * x1 - 0.5 * x2))
  yp <- rpois(n, exp(0.2 + 0.3 * x1 + off))
  wq <- runif(n, 0.5, 3); yq <- round(plogis(0.4 * x1 - 0.2 * x4 + rnorm(n, 0, 0.3)), 6)
  ysep <- as.integer(x4 > 0.1)                                          # completely separated by x4
  yc <- 1 + 2 * x1 - x2 + rnorm(n)
  w0 <- c(rep(0, 5), runif(n - 5, 0.5, 2))
  d <- ctx$gen_dataset("linalg_glm", data.frame(x1 = round(x1, 10), x2 = round(x2, 10), x3 = round(x1, 10) - 2 * round(x2, 10), x4 = round(x4, 10),
                                                off = round(off, 10), yb = yb, yp = yp, wq = round(wq, 10), yq = yq, ysep = ysep, yc = round(yc, 10), w0 = round(w0, 10)))
  X <- cbind(1, d$x1, d$x3, d$x2, d$x4)                                 # x3 before x2: dqrdc2 keeps x3 and drops x2
  q <- qr(X, tol = 1e-7)
  v <- list(qr_rank = q$rank, qr_pivot = vec(q$pivot), qr_qraux = vec(q$qraux), qr_r_diag = vec(diag(q$qr)))
  # WP01 plan: the entries of the identified block only (qraux and R[5, 5] of the aliased column are rounding noise)
  v$qr_qraux_rank <- vec(q$qraux[seq_len(q$rank)]); v$qr_r_diag_rank <- vec(diag(q$qr)[seq_len(q$rank)])
  # WP01 plan: a rescaled duplicate (2 * x1) and a zero column are moved to the end; n < p
  q2 <- qr(cbind(1, d$x1, 2 * d$x1, d$x2, 0, d$x4), tol = 1e-7); v$qr2_rank <- q2$rank; v$qr2_pivot <- vec(q2$pivot)
  q3 <- qr(X[1:3, ], tol = 1e-7); v$qr3_rank <- q3$rank; v$qr3_pivot <- vec(q3$pivot); v$qr3_qraux <- vec(q3$qraux)
  f <- lm.fit(X, d$yc); v$lmfit_coef <- vec(ifelse(is.na(f$coefficients), NA, f$coefficients)); v$lmfit_rank <- f$rank; v$lmfit_df <- f$df.residual
  v$lmfit_resid_ss <- sum(f$residuals^2)
  m <- lm(yc ~ x1 + x3 + x2 + x4, data = d); v$lm_se <- vec(sqrt(diag(vcov(m))))   # identified columns only
  fw <- lm.wfit(cbind(1, d$x1, d$x2), d$yc, d$w0); v$lmwfit_coef <- vec(fw$coefficients); v$lmwfit_df <- fw$df.residual
  v$lmwfit_resid_head <- vec(fw$residuals[1:8]); v$lmwfit_fitted_head <- vec(fw$fitted.values[1:8])   # WP01 plan: rows 1-5 have weight 0
  v$lmw_se <- vec(unname(sqrt(diag(vcov(lm(yc ~ x1 + x2, data = d, weights = w0))))))                  # WP01 plan
  ctl <- glm.control(epsilon = 1e-14, maxit = 100)
  specs <- list(logit = list(yb ~ x1 + x2, binomial("logit"), NULL, NULL),
                probit = list(yb ~ x1 + x2, binomial("probit"), NULL, NULL),
                cloglog = list(yb ~ x1 + x2, binomial("cloglog"), NULL, NULL),          # WP01 plan
                poisson = list(yp ~ x1, poisson, "off", NULL),
                poisid = list(yp ~ x1, poisson(link = "identity"), NULL, NULL),        # WP01 plan
                qpois = list(yp ~ x1, quasipoisson, "off", NULL),                      # WP01 plan
                quasibin = list(yq ~ x1 + x4, quasibinomial, NULL, "wq"),
                gaussian = list(yc ~ x1 + x2, gaussian, NULL, NULL),
                gausslog = list(I(yp + 0.5) ~ x1, gaussian(link = "log"), NULL, NULL))  # WP01 plan
  fit1 <- function(s, control) {
    dd <- d
    dd$.w <- if (is.null(s[[4]])) rep(1, nrow(dd)) else dd[[s[[4]]]]
    if (is.null(s[[3]])) suppressWarnings(glm(s[[1]], data = dd, family = s[[2]], weights = .w, control = control))
    else { dd$.off <- dd[[s[[3]]]]; suppressWarnings(glm(s[[1]], data = dd, family = s[[2]], offset = .off, weights = .w, control = control)) }
  }
  fits <- lapply(specs, fit1, control = ctl)
  fits$aliased <- glm(yb ~ x1 + x3 + x2, data = d, family = binomial, control = glm.control(epsilon = 1e-10))  # qr tol = min(1e-7, epsilon / 1000) = 1e-13: x2 aliased
  for (nm in names(fits)) {
    g <- fits[[nm]]; s <- summary(g)
    v[[paste0(nm, "_coef")]] <- vec(unname(coef(g)))
    v[[paste0(nm, "_se")]] <- vec(unname(s$coefficients[, 2]))
    v[[paste0(nm, "_deviance")]] <- g$deviance; v[[paste0(nm, "_null_deviance")]] <- g$null.deviance
    v[[paste0(nm, "_dispersion")]] <- s$dispersion; v[[paste0(nm, "_df_residual")]] <- g$df.residual
    if (!(nm %in% c("quasibin", "qpois"))) {
      v[[paste0(nm, "_aic")]] <- g$aic
      v[[paste0(nm, "_loglik")]] <- as.numeric(logLik(g))                                         # WP01 plan
    }
    if (nm %in% names(specs)) v[[paste0(nm, "_iter_default")]] <- fit1(specs[[nm]], glm.control())$iter   # WP01 plan: epsilon 1e-8, maxit 25
  }
  v$logit_fitted_head <- vec(unname(fitted(fits$logit)[1:5])); v$probit_fitted_head <- vec(unname(fitted(fits$probit)[1:5]))  # WP01 plan
  v$logit_start0_iter <- glm(yb ~ x1 + x2, data = d, family = binomial, start = c(0, 0, 0))$iter                             # WP01 plan
  # WP01 plan: binomial on a fractional response with non-integer prior weights (R warns "non-integer #successes";
  # same coefficients as quasibinomial; the AIC uses round(w * y) and round(w), R's round-half-even)
  gb <- suppressWarnings(glm(yq ~ x1 + x4, data = d, weights = wq, family = binomial, control = ctl))
  v$binfrac_coef <- vec(unname(coef(gb))); v$binfrac_aic <- gb$aic
  # WP01 plan: iteration counts under rescaling at R's default control (glm.fit's |dev - devold| / (|dev| + 0.1) rule)
  v$logit_x1e6_iter <- glm(yb ~ I(x1 * 1e6) + x2, data = d, family = binomial)$iter
  v$logit_x1em6_iter <- glm(yb ~ I(x1 * 1e-6) + x2, data = d, family = binomial)$iter
  v$gauss_ye6_iter <- glm(I(yc * 1e6) ~ x1 + x2, data = d, family = gaussian)$iter
  v$gauss_yem6_iter <- glm(I(yc * 1e-6) ~ x1 + x2, data = d, family = gaussian)$iter
  # TMLE fluctuation forms (WP01 plan): no intercept, clever covariate H = x4; intercept-only with prior weights
  d$.flo <- 0.2 + 0.5 * d$x1
  fl <- glm(yq ~ -1 + x4, offset = .flo, data = d, family = quasibinomial, control = ctl)
  fi <- suppressWarnings(glm(yq ~ 1, offset = .flo, weights = wq, data = d, family = quasibinomial, control = ctl))
  for (nm in c("fluct", "fluct_int")) {
    g <- if (nm == "fluct") fl else fi; s <- summary(g)
    v[[paste0(nm, "_coef")]] <- vec(unname(coef(g))); v[[paste0(nm, "_se")]] <- vec(unname(s$coefficients[, 2]))
    v[[paste0(nm, "_deviance")]] <- g$deviance; v[[paste0(nm, "_null_deviance")]] <- g$null.deviance
    v[[paste0(nm, "_dispersion")]] <- s$dispersion; v[[paste0(nm, "_df_residual")]] <- g$df.residual
  }
  # dummy trap (WP01 plan): [1, x1, ysep, 1 - ysep]
  Xt <- cbind(1, d$x1, d$ysep, 1 - d$ysep)
  mt <- lm(yc ~ x1 + ysep + I(1 - ysep), data = d)
  stopifnot(qr(Xt)$rank == 3, is.na(coef(mt)[4]))
  v$trap_rank <- qr(Xt)$rank; v$trap_x1_coef <- unname(coef(mt)[2]); v$trap_rss <- sum(resid(mt)^2)
  sep <- withCallingHandlers(glm(ysep ~ x4, data = d, family = binomial), warning = function(w) invokeRestart("muffleWarning"))
  v$sep_fitted_extreme <- sum(fitted(sep) < 10 * .Machine$double.eps | fitted(sep) > 1 - 10 * .Machine$double.eps)
  v$sep_converged_r <- as.numeric(sep$converged)
  v$sep_iter <- sep$iter                                                                             # WP01 plan
  list(values = v, reference = "base::qr (dqrdc2, tol 1e-7); stats::lm.fit / lm.wfit; stats::glm.fit (epsilon 1e-14) via glm",
       call = "qr(cbind(1, x1, x3, x2, x4)); glm(..., family)", packages = character(0))
})
