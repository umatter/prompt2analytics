# validation/r/cases/causal_effects_t4.R -- T4-WP15 references: regression adjustment (ATT and the
# stacked-equation sandwich SE), Huber (2014) IPW mediation (all four potential-outcome means), and
# regression-standardisation delta-method and sandwich SEs for ATE/ATT/ATC and E[Y(a)].
#  * ra_ipw_t4 (weighting_sim, the dataset of ipw_aipw_sim): OLS outcome models per arm on
#    (1, x1, x2, x3); RA ATE / ATT / ATC = mean(m1 - m0) over all / treated / controls. Sandwich SE of
#    the stacked equations [D X (Y - X b1); (1 - D) X (Y - X b0); S (X (b1 - b0) - tau)], S the
#    estimand's subset (HC0 meat, the trimmed set held fixed: trim 0 here). Equal to the
#    stdReg-style sandwich of the fully interacted lm (checked).
#  * mediation_huber_sim: Huber (2014) normalised IPW with logistic p(X) = P(D = 1 | X) and
#    p(M, X) = P(D = 1 | M, X) (glm, epsilon 1e-14), p2a's trimming rule (both scores in
#    (trim, 1 - trim)); E[Y(1,M(1))] = hm(D / p), E[Y(0,M(0))] = hm((1 - D) / (1 - p)),
#    E[Y(1,M(0))] = hm(D (1 - pM) / (pM (1 - p))), E[Y(0,M(1))] = hm((1 - D) pM / ((1 - pM) p)),
#    hm(w) = sum(w Y) / sum(w). causalweight (medweight, whose effects these are) is not installed:
#    the formulas are written out. theta(0) = Y(1,M(0)) - Y(0,M(0)) (NDE), delta(1) = Y(1,M(1)) -
#    Y(1,M(0)) (NIE), theta(1), delta(0); theta(0) + delta(1) = theta(1) + delta(0) = total.
#  * stdreg_se_t4 (stdreg_sim, the dataset of stdreg_glm): for lm (as glm gaussian) and glm logit /
#    Poisson fits (epsilon 1e-14), with and without A x W interactions, and the subsets S = all /
#    treated / controls: E[Y(a)] = mean_S g(X_a b); delta method Var = G V G' + sum_S (m_i - mu)^2 / n_S^2
#    with V = vcov(fit) (model-based; lm: sigma^2 (X'X)^-1, df n - k) and G = mean_S g'(X_a b) X_a;
#    sandwich = stdReg::stdGlm's stacked estimating equations (score, S (m_i(a) - mu_a)), HC0 meat.
#    G V G' alone is marginaleffects::avg_comparisons' SE (checked for the interacted fits).
.t4_ctl <- glm.control(epsilon = 1e-14, maxit = 100)

case("ra_ipw_t4", "econometrics/treatment",
     "Regression adjustment ATE/ATT/ATC with OLS outcome models per arm and the stacked-equation sandwich SE (weighting_sim)",
     function(ctx) {
  d <- ctx$dataset("datasets/generated/weighting_sim.csv")
  X <- cbind(1, d$x1, d$x2, d$x3); D <- d$treat; Y <- d$y; n <- nrow(d); k <- ncol(X)
  b1 <- qr.coef(qr(X[D == 1, ]), Y[D == 1]); b0 <- qr.coef(qr(X[D == 0, ]), Y[D == 0])
  m1 <- drop(X %*% b1); m0 <- drop(X %*% b0)
  r1 <- D * (Y - m1); r0 <- (1 - D) * (Y - m0)
  ra <- function(S) {
    t <- as.numeric(S); nS <- sum(t); tau <- sum(t * (m1 - m0)) / nS
    U <- cbind(X * r1, X * r0, t * (m1 - m0 - tau))
    J <- matrix(0, 2 * k + 1, 2 * k + 1)
    J[1:k, 1:k] <- crossprod(X * D, X) / n; J[k + 1:k, k + 1:k] <- crossprod(X * (1 - D), X) / n
    J[2 * k + 1, 1:k] <- -colSums(t * X) / n; J[2 * k + 1, k + 1:k] <- colSums(t * X) / n
    J[2 * k + 1, 2 * k + 1] <- nS / n
    Ji <- solve(J)
    c(tau = tau, se = sqrt((Ji %*% (crossprod(U) / n) %*% t(Ji) / n)[2 * k + 1, 2 * k + 1]))
  }
  a <- ra(rep(TRUE, n)); t1 <- ra(D == 1); t0 <- ra(D == 0)
  f <- lm(y ~ (x1 + x2 + x3) * treat, data = d)
  X1 <- model.matrix(delete.response(terms(f)), transform(d, treat = 1))
  X0 <- model.matrix(delete.response(terms(f)), transform(d, treat = 0))
  stopifnot(abs(mean((X1 - X0) %*% coef(f)) - a[["tau"]]) < 1e-12)
  list(values = list(ra_ate = a[["tau"]], ra_att = t1[["tau"]], ra_atc = t0[["tau"]],
                     ra_ate_se_sandwich = a[["se"]], ra_att_se_sandwich = t1[["se"]], ra_atc_se_sandwich = t0[["se"]],
                     n = n, n_treated = sum(D)),
       reference = "OLS per arm (qr), regression adjustment over all / treated / controls; stacked estimating-equation sandwich (HC0) written out; equals the fully interacted lm's standardised contrast",
       call = "lm(y ~ x1 + x2 + x3, subset = treat == d) for d in 1, 0; mean(m1 - m0) over S", packages = character(0))
})

case("mediation_huber_sim", "econometrics/mediation",
     "Huber (2014) IPW mediation: total, theta(0) (NDE), delta(1) (NIE), theta(1), delta(0) and the four potential-outcome means, trim 0 and 0.05, treatment-mediator interaction",
     function(ctx) {
  set.seed(20261111)
  n <- 800
  x1 <- rnorm(n); x2 <- runif(n, -1, 1)
  d <- rbinom(n, 1, plogis(0.2 + 0.7 * x1 - 0.5 * x2))
  m <- 0.3 + 0.8 * d + 0.5 * x1 + 0.3 * x2 + rnorm(n)
  y <- 1 + 0.5 * d + 0.6 * m + 0.7 * d * m + 0.4 * x1 - 0.3 * x2 + rnorm(n)
  dat <- ctx$gen_dataset("mediation_huber_sim", data.frame(y = round(y, 10), d = as.integer(d), m = round(m, 10),
                                                         x1 = round(x1, 10), x2 = round(x2, 10)))
  p <- fitted(glm(d ~ x1 + x2, data = dat, family = binomial, control = .t4_ctl))
  pm <- fitted(glm(d ~ x1 + x2 + m, data = dat, family = binomial, control = .t4_ctl))
  h <- function(trim) {
    keep <- p > trim & p < 1 - trim & pm > trim & pm < 1 - trim
    Y <- dat$y[keep]; D <- dat$d[keep]; e <- p[keep]; em <- pm[keep]
    hm <- function(w) sum(w * Y) / sum(w)
    y11 <- hm(D / e); y00 <- hm((1 - D) / (1 - e))
    y10 <- hm(D * (1 - em) / (em * (1 - e))); y01 <- hm((1 - D) * em / ((1 - em) * e))
    margin <- min(abs(c(p, pm) - trim), abs(c(p, pm) - (1 - trim)))
    list(total = y11 - y00, theta0 = y10 - y00, delta1 = y11 - y10, theta1 = y11 - y01, delta0 = y01 - y00,
         means = vec(c(y11, y10, y00, y01)), kept = sum(keep), margin = margin)
  }
  v <- list()
  for (tr in c(0, 0.05)) {
    r <- h(tr); s <- if (tr == 0) "trim0" else "trim05"
    if (tr > 0) stopifnot(r$margin > 1e-4)                    # no score sits at a trimming bound
    for (q in c("total", "theta0", "delta1", "theta1", "delta0", "means", "kept")) v[[paste0(q, "_", s)]] <- r[[q]]
  }
  v$n <- n; v$n_treated <- sum(dat$d)
  list(values = v,
       reference = "Huber (2014) normalised IPW weights with glm(binomial, epsilon 1e-14) propensities p(X) and p(M, X) (causalweight::medweight's effects; causalweight not installed, written out), p2a's trimming rule",
       call = "glm(d ~ x1 + x2, binomial); glm(d ~ x1 + x2 + m, binomial); weighted means of y", packages = character(0))
})

case("stdreg_se_t4", "causal/stdreg",
     "Regression standardisation SEs: delta method (model vcov plus the averaging term) and stdReg's stacked sandwich, for E[Y(1)], E[Y(0)] and the contrast over all / treated / controls, lm and glm (logit, Poisson), with and without A x W interactions",
     function(ctx) {
  d <- ctx$dataset("datasets/generated/stdreg_sim.csv")
  ses <- function(fit, S) {
    X <- model.matrix(fit); n <- nrow(X); k <- ncol(X); fam <- family(fit); b <- coef(fit)
    X1 <- model.matrix(delete.response(terms(fit)), transform(d, a = 1))
    X0 <- model.matrix(delete.response(terms(fit)), transform(d, a = 0))
    e1 <- drop(X1 %*% b); e0 <- drop(X0 %*% b); p1 <- fam$linkinv(e1); p0 <- fam$linkinv(e0)
    t <- as.numeric(S); nS <- sum(t); ey1 <- sum(t * p1) / nS; ey0 <- sum(t * p0) / nS
    G1 <- colSums(t * X1 * fam$mu.eta(e1)) / nS; G0 <- colSums(t * X0 * fam$mu.eta(e0)) / nS
    u1 <- t * (p1 - ey1); u0 <- t * (p0 - ey0); V <- vcov(fit)
    dv <- function(G, u) sqrt(drop(t(G) %*% V %*% G) + sum(u^2) / nS^2)
    mu <- fitted(fit); eta <- fit$linear.predictors
    w <- fam$mu.eta(eta)^2 / fam$variance(mu); r <- (fit$y - mu) / fam$mu.eta(eta)
    U <- cbind(X * as.vector(w * r), u1, u0)
    J <- matrix(0, k + 2, k + 2); J[1:k, 1:k] <- crossprod(X * as.vector(w), X) / n
    J[k + 1, 1:k] <- -G1 * nS / n; J[k + 2, 1:k] <- -G0 * nS / n; J[k + 1, k + 1] <- J[k + 2, k + 2] <- nS / n
    Ji <- solve(J); Vs <- Ji %*% (crossprod(U) / n) %*% t(Ji) / n; g <- c(rep(0, k), 1, -1)
    list(est = vec(c(ey1, ey0, ey1 - ey0)),
         delta = vec(c(dv(G1, u1), dv(G0, u0), dv(G1 - G0, u1 - u0))),
         sandwich = vec(c(sqrt(Vs[k + 1, k + 1]), sqrt(Vs[k + 2, k + 2]), sqrt(drop(t(g) %*% Vs %*% g)))),
         gvg = sqrt(drop(t(G1 - G0) %*% V %*% (G1 - G0))))
  }
  fits <- list(lin = glm(yc ~ x1 + x2 + a, data = d, family = gaussian),
               lin_int = glm(yc ~ x1 + x2 + a + a:x1 + a:x2, data = d, family = gaussian),
               logit = glm(yb ~ x1 + x2 + a, data = d, family = binomial, control = .t4_ctl),
               logit_int = glm(yb ~ x1 + x2 + a + a:x1 + a:x2, data = d, family = binomial, control = .t4_ctl),
               pois = glm(yp ~ x1 + x2 + a, data = d, family = poisson, control = .t4_ctl))
  stopifnot(all(vapply(fits, function(f) f$converged, TRUE)))
  v <- list()
  for (nm in names(fits)) for (est in c("ate", "att", "atc")) {
    S <- switch(est, ate = rep(TRUE, nrow(d)), att = d$a == 1, atc = d$a == 0)
    s <- ses(fits[[nm]], S)
    for (q in c("est", "delta", "sandwich")) v[[paste(nm, est, q, sep = "_")]] <- s[[q]]
    if (nm %in% c("lin_int", "logit_int") && est != "atc") {
      me <- marginaleffects::avg_comparisons(fits[[nm]], variables = "a", newdata = if (est == "att") subset(d, a == 1) else d)
      stopifnot(abs(me$std.error - s$gvg) < 1e-9 * s$gvg)
    }
  }
  # the sandwich ATE values agree with stdreg_glm's *_stdreg_* values (same formulas)
  list(values = v,
       reference = "stats::lm (as glm gaussian) / stats::glm (epsilon 1e-14); delta method with vcov(fit) and the averaging term, written out (G V G' = marginaleffects::avg_comparisons, checked); stdReg::stdGlm's stacked-equation sandwich written out (stdReg not installed)",
       call = "glm(yc|yb|yp ~ x1 + x2 + a [+ a:x1 + a:x2]); standardisation over all / a == 1 / a == 0",
       packages = c("marginaleffects"))
})
