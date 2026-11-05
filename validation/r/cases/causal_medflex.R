# validation/r/cases/causal_medflex.R -- natural effect models on the difference, ratio and
# odds-ratio scales (T3-WP26: CAUS-22, CAUS-73, CAUS-74; moved in CAUS-23, CAUS-75).
# This file writes out the imputation-based estimator of Vansteelandt, Bekaert & Lange (2012) as
# medflex 0.6-11 implements it (neImpute.default, neModel, neEffdecomp, read from the package
# source) for a binary exposure coded as a factor, and cross-checks the written-out natural effect
# model coefficients and robust variance against medflex itself (neImpute + neModel):
#  1. outcome working model Y ~ A + M + C [+ A:M]: gaussian (lm), poisson or binomial glm;
#  2. expanded data, two rows per unit in unit order, A0 = 0 then 1 (direct exposure), A1 = A
#     (observed, mediator-determining); every row is imputed, Yhat = E[Y | A = A0, M, C]
#     (neImpute predicts all rows, also the one with A0 = A);
#  3. natural effect model Yhat ~ A0 + A1 + C [+ A0:A1 + A0:C]: gaussian, or quasipoisson (log
#     link, ratio scale) or quasibinomial (logit link, odds-ratio scale). With A:M in the working
#     model, E[Y(a0, M(a1)) | C] has an a0 x C term (b_am * alpha_C), so A0:C enters (p2a's choice);
#  4. effects are neEffdecomp's contrasts at covLev = colMeans(C), held fixed (medflex's default
#     level is 0; at the means the difference-scale NDE is the population-average NDE of
#     VanderWeele 2015): NDE (pure direct) = a0 + a0:C cbar, NIE (total indirect) = a1 + a0:a1,
#     TE = NDE + NIE on the link scale; exp() on the ratio scales;
#  5. robust SEs as neModel(se = "robust"): the stacked sandwich of the natural effect model and
#     the working model (medflex's estfun/bread scaling cancels); the clean form p2a implements is
#     checked against medflex's literal expression and the cross derivative against numDeriv;
#  6. bootstrap as neModel(se = "bootstrap") (units resampled, everything refitted, effects at the
#     full-sample cbar); p2a reports the sd of the link-scale draws and medflex's default
#     normal-type intervals (confint(type = "norm") calls boot::boot.ci, i.e. boot::norm.ci:
#     2 t0 - mean(t) -/+ z sd(t), link scale, exp() on the ratio scales). 50 resamples are drawn by a 32-bit LCG that
#     Rust reproduces exactly: s <- (69069 s + 1) mod 2^32, row = floor(s n / 2^32) (0-based).
# Proportion mediated: NIE/TE (difference), VanderWeele (2015) NDE (NIE - 1) / (NDE NIE - 1) (ratio
# scales); never clamped.

.mf_ctl <- glm.control(epsilon = 1e-14, maxit = 100)

.mf_lcg_rows <- function(n, B, seed) {
  s <- seed
  out <- matrix(0, n, B)
  for (b in seq_len(B)) for (i in seq_len(n)) {
    s <- (69069 * s + 1) %% 4294967296
    out[i, b] <- floor(s * n / 4294967296)
  }
  out
}

# One natural effect model fit. wfam: working-model family ("gaussian", "poisson", "binomial";
# canonical links); scale: natural effect model, "diff" (gaussian), "ratio" (quasipoisson, log),
# "odds" (quasibinomial, logit).
.mf_fit <- function(d, yname, mname, wfam, scale, inter, cvars, cbar) {
  w_rhs <- paste(c("a", mname, cvars, if (inter) paste0("a:", mname)), collapse = " + ")
  wf <- as.formula(paste(yname, "~", w_rhs))
  om <- switch(wfam,
               gaussian = glm(wf, data = d, family = gaussian()),
               poisson = glm(wf, data = d, family = poisson(), control = .mf_ctl),
               binomial = glm(wf, data = d, family = binomial(), control = .mf_ctl))
  n <- nrow(d)
  ex <- d[rep(seq_len(n), each = 2), , drop = FALSE]
  ex$a0 <- rep(c(0, 1), times = n)
  ex$a1 <- ex$a
  ex$id <- rep(seq_len(n), each = 2)
  nd <- ex; nd$a <- ex$a0
  ex$yhat <- unname(suppressWarnings(predict(om, newdata = nd, type = "response")))
  ne_rhs <- paste(c("a0", "a1", cvars, if (inter) c("a0:a1", paste0("a0:", cvars))), collapse = " + ")
  nf <- as.formula(paste("yhat ~", ne_rhs))
  ne <- suppressWarnings(switch(scale,
               diff = glm(nf, data = ex, family = gaussian()),
               ratio = glm(nf, data = ex, family = quasipoisson(), control = .mf_ctl),
               odds = glm(nf, data = ex, family = quasibinomial(), control = .mf_ctl)))
  th <- coef(ne)
  K <- matrix(0, 3, length(th), dimnames = list(c("nde", "nie", "te"), names(th)))
  K["nde", "a0"] <- 1
  K["nie", "a1"] <- 1
  if (inter) {
    K["nde", paste0("a0:", cvars)] <- cbar
    K["nie", "a0:a1"] <- 1
  }
  K["te", ] <- K["nde", ] + K["nie", ]
  thz <- ifelse(is.na(th), 0, th)                        # aliased coefficients enter as 0 (predict() rule)
  eff <- drop(K %*% thz)
  list(om = om, ne = ne, ex = ex, nd = nd, K = K, eff = eff)
}

# Stacked sandwich: the clean form (what Rust implements) and medflex's literal expression.
.mf_robust <- function(f, literal = TRUE) {
  om <- f$om; ne <- f$ne; ex <- f$ex
  okw <- !is.na(coef(om)); okn <- !is.na(coef(ne))
  X <- model.matrix(om)[, okw, drop = FALSE]
  Xa0 <- model.matrix(delete.response(terms(om)), f$nd)[, okw, drop = FALSE]
  Z <- model.matrix(ne)[, okn, drop = FALSE]
  mu <- fitted(om); y <- om$y
  nu <- fitted(ne)
  v_w <- om$family$variance(mu)                          # canonical links: mu.eta = variance
  v_n <- switch(ne$family$family, gaussian = rep(1, length(nu)), quasipoisson = nu, quasibinomial = nu * (1 - nu))
  A_b <- crossprod(X, X * v_w)
  A_t <- crossprod(Z, Z * v_n)
  me_w <- om$family$mu.eta(drop(Xa0 %*% coef(om)[okw]))
  G <- crossprod(Z, Xa0 * me_w)
  U <- X * (y - mu)
  S <- rowsum(Z * (ex$yhat - nu), ex$id, reorder = FALSE)
  Phi <- (S + U %*% solve(A_b, t(G))) %*% solve(A_t)       # rows: (A_t^{-1} (S_i + G A_b^{-1} U_i))'
  V <- crossprod(Phi)
  # medflex 0.6-11 neModel(se = "robust"), impData branch, literally (estfun/bread on the
  # non-aliased columns, as sandwich does)
  if (literal) {
    est <- cbind(as.matrix(aggregate(sandwich::estfun(ne), by = list(ex$id), FUN = mean)[, -1]), sandwich::estfun(om))
    meat <- crossprod(est) / nrow(est)
    bi <- as.matrix(Matrix::bdiag(lapply(list(ne, om), function(x) solve(sandwich::bread(x) * nrow(x$model) / sum(summary(x)$df[1:2])))))
    bi[seq_len(ncol(Z)), ncol(Z) + seq_len(ncol(X))] <- -t(sandwich::estfun(ne) / resid(ne, type = "response")) %*% (Xa0 * me_w) / nrow(Xa0)
    bread <- solve(bi)
    Vm <- (bread %*% meat %*% t(bread) / nrow(est))[seq_len(ncol(Z)), seq_len(ncol(Z))]
    stopifnot(max(abs(V - Vm)) <= 1e-8 * max(abs(V)))
  }
  # numerical check of the cross derivative G = d S(theta, beta) / d beta' at the estimates
  sfun <- function(b) colSums(Z * (om$family$linkinv(drop(Xa0 %*% b)) - nu))
  Gn <- numDeriv::jacobian(sfun, coef(om)[okw])
  stopifnot(max(abs(Gn - G)) <= 1e-6 * max(abs(G)))
  Kr <- f$K[, okn, drop = FALSE]
  se <- sqrt(diag(Kr %*% V %*% t(Kr)))
  list(V = V, se = se)
}

# medflex 0.6-11 itself on the same configuration (exposure as a factor): its natural effect model
# coefficients and robust variance must equal the written-out ones. neModel evaluates its own
# call, so the arguments go in by value (do.call).
.mf_medflex <- function(d, yname, wfam, scale, inter, cvars, f, r) {
  d2 <- d; d2$a <- factor(d2$a)
  wf <- as.formula(paste(yname, "~", paste(c("a", "m", cvars, if (inter) "a:m"), collapse = " + ")))
  om <- glm(wf, data = d2, family = switch(wfam, gaussian = gaussian(), poisson = poisson(), binomial = binomial()),
            control = .mf_ctl)
  expData <- medflex::neImpute(om, data = d2)   # neModelEst looks this name up in its caller's caller
  nf <- as.formula(paste(yname, "~", paste(c("a0", "a1", cvars, if (inter) c("a0:a1", paste0("a0:", cvars))), collapse = " + ")))
  nfam <- switch(scale, diff = gaussian(), ratio = quasipoisson(), odds = quasibinomial())
  ne <- suppressWarnings(do.call(medflex::neModel, list(formula = nf, family = nfam, expData = expData, se = "robust",
                                                        control = .mf_ctl)))
  stopifnot(max(abs(unname(coef(ne)) - unname(coef(f$ne)))) <= 1e-8 * max(abs(coef(f$ne))),
            max(abs(unname(vcov(ne)) - unname(r$V))) <= 1e-6 * max(abs(r$V)))
}

.mf_pm <- function(scale, e) {
  if (scale == "diff") e[["nie"]] / e[["te"]] else { nde <- exp(e[["nde"]]); nie <- exp(e[["nie"]]); nde * (nie - 1) / (nde * nie - 1) }
}

case("medflex_scales", "causal/mediation",
     "Imputation-based natural effect models (medflex): difference, ratio and odds-ratio scales, A:M interaction, robust and bootstrap SEs, aliased confounders",
     function(ctx) {
  set.seed(20261130)
  n <- 500
  c1 <- rnorm(n); c2 <- rbinom(n, 1, 0.5)
  a <- rbinom(n, 1, plogis(-0.2 + 0.5 * c1))
  m <- 0.5 + 0.8 * a + 0.4 * c1 - 0.3 * c2 + rnorm(n)
  yc <- 1 + 0.6 * a + 0.5 * m + 0.3 * c1 + rnorm(n)
  yb <- rbinom(n, 1, plogis(-1 + 0.7 * a + 0.4 * m + 0.3 * c1 - 0.2 * c2))
  yp <- rpois(n, exp(-0.5 + 0.4 * a + 0.3 * m + 0.2 * c1))
  # T3-WP26 additions, drawn after the design's columns, so those columns are unchanged
  yi <- 1 + 0.4 * a + 0.5 * m + 0.8 * a * m + 0.3 * c1 - 0.2 * c2 + rnorm(n)    # A:M interaction (CAUS-23)
  yo <- 1 - 0.45 * a + 0.5 * m + 0.3 * c1 + rnorm(n)                             # NDE, NIE of opposite sign (CAUS-75)
  c1r <- round(c1, 10)
  m_lin <- 0.1 + 0.3 * c1r                                                       # mediator determined by c1 (CAUS-73)
  y_lin <- 1 + 0.6 * a + 0.5 * m_lin + 0.2 * c2 + rnorm(n)
  d <- ctx$gen_dataset("medflex_sim", data.frame(
    a = a, m = round(m, 10), c1 = c1r, c2 = c2, yc = round(yc, 10), yb = yb, yp = yp,
    yi = round(yi, 10), yo = round(yo, 10), m_lin = m_lin, y_lin = round(y_lin, 10), c1dup = c1r))
  stopifnot(identical(d$c1, d$c1dup), max(abs(d$m_lin - (0.1 + 0.3 * d$c1))) < 1e-15)
  cv <- c("c1", "c2")
  cbar <- colMeans(d[, cv])
  v <- list(n = n, n_treated = sum(d$a == 1), cbar = named(cbar))
  fm <- lm(m ~ a + c1 + c2, data = d)
  v$med_coef <- named(coef(fm)); v$med_se <- named(sqrt(diag(vcov(fm)))); v$med_r2 <- summary(fm)$r.squared
  v$mean_m_control <- mean(d$m[d$a == 0]); v$mean_m_treated <- mean(d$m[d$a == 1])
  # (outcome, working family, scale, A:M); the first three use the scale's default family
  configs <- list(diff = c("yc", "gaussian", "diff", "FALSE"), ratio = c("yp", "poisson", "ratio", "FALSE"),
                  odds = c("yb", "binomial", "odds", "FALSE"),
                  diff_int = c("yi", "gaussian", "diff", "TRUE"), ratio_int = c("yp", "poisson", "ratio", "TRUE"),
                  odds_int = c("yb", "binomial", "odds", "TRUE"),
                  diff_opp = c("yo", "gaussian", "diff", "FALSE"),
                  rr_bin = c("yb", "binomial", "ratio", "FALSE"),        # risk ratio from a logistic working model
                  rd_bin = c("yb", "binomial", "diff", "FALSE"))         # risk difference from a logistic working model
  fits <- list()
  for (nm in names(configs)) {
    cf <- configs[[nm]]; scale <- cf[3]; inter <- as.logical(cf[4])
    f <- .mf_fit(d, cf[1], "m", cf[2], scale, inter, cv, cbar)
    r <- .mf_robust(f)
    fits[[nm]] <- f
    tr <- if (scale == "diff") identity else exp
    z <- qnorm(0.975)
    v[[paste0(nm, "_outcome_coef")]] <- named(coef(f$om))
    v[[paste0(nm, "_outcome_se")]] <- named(sqrt(diag(vcov(f$om))))
    v[[paste0(nm, "_ne_coef")]] <- named(coef(f$ne))
    for (e in c("nde", "nie", "te")) v[[paste0(nm, "_", e)]] <- unname(tr(f$eff[[e]]))   # the design keys: <cfg>_nde, _nie, _te
    v[[paste0(nm, "_robust_se")]] <- named(r$se)                          # link-scale SEs
    v[[paste0(nm, "_robust_ci_lower")]] <- named(tr(f$eff - z * r$se))
    v[[paste0(nm, "_robust_ci_upper")]] <- named(tr(f$eff + z * r$se))
    v[[paste0(nm, "_robust_p")]] <- named(2 * pnorm(-abs(f$eff / r$se)))
    v[[paste0(nm, "_pm")]] <- .mf_pm(scale, f$eff)
    v[[paste0(nm, "_deviance_r2")]] <- 1 - f$om$deviance / f$om$null.deviance
  }
  # Cross-checks of the written-out estimator (stopifnot; nothing below is a Rust pin on its own)
  # (0) medflex 0.6-11 itself: natural effect model coefficients to 1e-8, robust variance to 1e-6
  for (nm in c("diff", "ratio", "odds", "diff_int", "ratio_int", "odds_int")) {
    cf <- configs[[nm]]
    .mf_medflex(d, cf[1], cf[2], cf[3], as.logical(cf[4]), cv, fits[[nm]], .mf_robust(fits[[nm]], literal = FALSE))
  }
  # (a) no interaction: the NDE coefficient is the working model's A coefficient (gaussian and log
  #     links), and its robust SE is the HC0 SE of that coefficient
  for (nm in c("diff", "ratio")) {
    f <- fits[[nm]]
    stopifnot(abs(f$eff[["nde"]] - coef(f$om)[["a"]]) < 1e-10)
    stopifnot(abs(v[[paste0(nm, "_robust_se")]]$values[1] - sqrt(sandwich::vcovHC(f$om, type = "HC0")["a", "a"])) < 1e-10)
  }
  # (b) gaussian with A:M: VanderWeele's regression formulas with the lm mediator model (CAUS-23)
  fi <- fits$diff_int; b <- coef(fi$om); al <- coef(fm)
  vw_nde <- unname(b["a"] + b["a:m"] * (al["(Intercept)"] + sum(al[cv] * cbar)))
  vw_nie <- unname(al["a"] * (b["m"] + b["a:m"]))
  stopifnot(abs(fi$eff[["nde"]] - vw_nde) < 1e-10, abs(fi$eff[["nie"]] - vw_nie) < 1e-10)
  v$diff_int_vanderweele <- named(c(nde = vw_nde, nie = vw_nie))
  v$diff_int_nde_raw_control_mean <- unname(b["a"] + b["a:m"] * mean(d$m[d$a == 0]))   # the pre-fix formula (CAUS-23)
  # Bootstrap machinery on odds_int: 50 LCG resamples, cbar fixed at the full-sample means
  B <- 50; rows <- .mf_lcg_rows(n, B, 20261130)
  draws <- t(sapply(seq_len(B), function(bb) .mf_fit(d[rows[, bb] + 1, , drop = FALSE], "yb", "m", "binomial", "odds", TRUE, cv, cbar)$eff))
  v$boot_rows_head <- vec(rows[1:5, 1]); v$boot_rows_tail <- vec(rows[(n - 4):n, B])
  v$odds_int_boot_draws_nde <- vec(draws[, "nde"])
  v$odds_int_boot_se <- named(apply(draws, 2, sd))
  # medflex's default normal-type interval (boot::norm.ci, bias-corrected), link scale, exp()
  nci <- sapply(c("nde", "nie", "te"), function(e) boot::norm.ci(t0 = fits$odds_int$eff[[e]], t = draws[, e], conf = 0.95)[2:3])
  v$odds_int_boot_ci_lower <- named(exp(nci[1, ]))
  v$odds_int_boot_ci_upper <- named(exp(nci[2, ]))
  # Aliased confounders (CAUS-73): m_lin = 0.1 + 0.3 c1 and c1dup = c1
  for (inter in c(FALSE, TRUE)) {
    nm <- if (inter) "alias_int" else "alias"
    ca <- c("c1", "c2", "c1dup")
    f <- .mf_fit(d, "y_lin", "m_lin", "gaussian", "diff", inter, ca, colMeans(d[, ca]))
    r <- .mf_robust(f, literal = FALSE)
    v[[paste0(nm, "_outcome_coef")]] <- named(coef(f$om))
    v[[paste0(nm, "_ne_coef")]] <- named(coef(f$ne))
    for (e in c("nde", "nie", "te")) v[[paste0(nm, "_", e)]] <- unname(f$eff[[e]])
    v[[paste0(nm, "_robust_se")]] <- named(r$se)
    # dropping the duplicate changes nothing estimable
    g <- .mf_fit(d, "y_lin", "m_lin", "gaussian", "diff", inter, cv, cbar)
    stopifnot(max(abs(g$eff - f$eff)) < 1e-10, max(abs(.mf_robust(g, literal = FALSE)$se - r$se)) < 1e-10)
  }
  v$alias_med_coef <- named(coef(lm(m_lin ~ a + c1 + c2 + c1dup, data = d)))
  # run_mediation_analysis (Huber IPW, mediation.rs) on the same data, outcome yo, trim 0.05: glm
  # propensity scores and the Hajek total effect (a formula cross-check; causalweight is not
  # installed, and the direct effect is not pinned because of CAUS-26, Tier 4)
  p1 <- fitted(glm(a ~ c1 + c2, binomial, data = d, control = .mf_ctl))
  p2 <- fitted(glm(a ~ c1 + c2 + m, binomial, data = d, control = .mf_ctl))
  keep <- p1 > 0.05 & p1 < 0.95 & p2 > 0.05 & p2 < 0.95
  stopifnot(min(abs(c(p1, p2) - 0.05), abs(c(p1, p2) - 0.95)) > 1e-4)
  D <- d$a[keep]; Y <- d$yo[keep]; q1 <- p1[keep]
  v$ipw_yo_total <- sum(D * Y / q1) / sum(D / q1) - sum((1 - D) * Y / (1 - q1)) / sum((1 - D) / (1 - q1))
  v$ipw_yo_kept <- sum(keep)
  # the shared type-7 percentile rule of the mediation bootstrap (X-18): a fixed draw vector
  x <- ((1:199 * 37) %% 199) / 10 + sqrt(1:199) / 7
  v$pct_draws <- vec(x)
  v$pct_ci <- vec(quantile(x, c(0.025, 0.975), type = 7))
  v$pct_sd <- sd(x)
  list(values = v,
       reference = paste("Vansteelandt, Bekaert & Lange (2012) imputation estimator as medflex 0.6-11 neImpute + neModel",
                         "(se = 'robust' | 'bootstrap') + neEffdecomp(covLev = colMeans(C)), written out;",
                         "cross-checked against medflex 0.6-11 (neImpute + neModel), sandwich::vcovHC(HC0), VanderWeele (2015) and numDeriv"),
       call = paste("glm(y ~ a + m + c1 + c2 [+ a:m], gaussian|poisson|binomial); expand A0 in {0,1}, A1 = A;",
                    "glm(yhat ~ a0 + a1 + c1 + c2 [+ a0:a1 + a0:c1 + a0:c2], gaussian|quasipoisson|quasibinomial)"),
       packages = c("sandwich", "numDeriv", "Matrix", "boot", "medflex"))
})
