# validation/r/cases/causal_cbps.R -- CBPS references (T3-WP24: CAUS-55, CAUS-11; the CBPS parts of
# CAUS-92 and CAUS-113).
#
# CBPS::CBPS(method = 'exact' | 'over', ATT = 0 | 1, standardize = TRUE, twostep = TRUE) on a simulated
# sample (n = 400). 'exact' solves the k balance conditions (just identified). 'over' minimises
# g' W g over the 2k score and balance conditions, with W = ginv(V) fixed at the start value
# (two-step GMM). CBPS whitens the covariates first (standardise the non-intercept columns, then
# take the left singular vectors U of the SVD), estimates on that scale and maps the coefficients back.
#
# Values:
#  * `<method>_<estimand>_*`: CBPS() as shipped (package values). Its optimisers stop early: optim
#    BFGS has reltol 1e-8, and the start scale alpha comes from optimize() with tol = eps^0.25. So
#    these values only cross-check p2a at about 1e-4.
#  * `*_tight_*` (pinned by p2a): the package's own CBPS.2Treat with four patches (three numerical,
#    one that only returns the start value). Each patch must apply exactly as often as stated,
#    otherwise the case stops (a new CBPS version needs a review):
#      1. The start scale alpha (1 site). CBPS takes alpha = optimize(f, c(0.8, 1.1))$minimum with
#         f(alpha) = g(alpha b0)' ginv(V(alpha b0)) g(alpha b0), where b0 is the glm start. The
#         computed f carries rounding noise of about 1e-12 relative (ginv), so optimize() places
#         alpha only to about 1e-7 at any tol. That moves W and the 'over' estimates by about 1e-7.
#         The patch keeps optimize(.., tol = 1e-12) to choose the basin and then bisects the
#         analytic derivative f'(alpha) (.cbps_alpha_refine below) on alpha +- 1e-6, so alpha
#         reaches the minimiser of f to rounding. p2a does the same. deparse() needs
#         width.cutoff = 500, or the call is split over two lines and the patch silently misses.
#      2. optim(.., reltol = 1e-15) (6 sites).
#      3. The ATT gradient. In gmm.gradient, CBPS 0.24 writes the balance block as
#         1/n.t * t(X * dw) %*% X. The derivative of its own moment (1/n) X' w is 1/n * t(X * dw) %*% X,
#         so the block is n/n_t times too large. BFGS then stops away from the minimum of CBPS's
#         objective (neither that gradient nor the objective's own is zero there). This affects only
#         ATT = 1 with method = 'over':
#         the exact fits use bal.gradient and the ATE fits use the ATE block, and both are correct.
#    CBPS.fit's whitening is reproduced line by line, the patched CBPS.2Treat is called on U, and
#    the coefficients are mapped back exactly as CBPS.fit maps them.
#  * Variance. CBPS.2Treat returns the sandwich on the whitened scale (n times the covariance of the
#    whitened coefficients). CBPS.fit's back-transformation of that matrix is not scale-equivariant:
#    its x3 SE stays at 0.1094 when x3 is multiplied by 10, and the bootstrap SE of x3 is 0.22. The
#    pinned SE (`*_tight_se`) is the delta method T V_u T' / n, where T maps whitened coefficients
#    to the original scale. CBPS's own transform of the same matrix is kept as
#    `*_tight_se_reported` for the record and is not pinned.
#  * J is CBPS's J: g' W g at the estimate, with no factor n, and with W fixed at the start value
#    for 'exact' too. `over_*_tight_hansen_p` is Hansen's test P(chi2_k > n J), which CBPS does not
#    report.
#  * `logit_*`: glm(binomial) and the weights its scores give (p2a method 'logit').
suppressPackageStartupMessages({library(CBPS); library(MASS)})

.cbps_patch <- function(src, from, to, times) {
  hits <- sum(lengths(regmatches(src, gregexpr(from, src, fixed = TRUE))))
  if (hits != times) stop(sprintf("CBPS.2Treat patch '%s': %d matches, expected %d", from, hits, times))
  gsub(from, to, src, fixed = TRUE)
}

# Derivative of CBPS's start-scale objective f(alpha) = g(b)' V(b)^+ g(b), b = alpha * b0, with unit
# sample weights. Probabilities are clipped to [1e-6, 1 - 1e-6] as CBPS clips them, with zero
# derivative where clipped. g = (U'(t - p), U'w) / n and
# V = (1/n) [[U'aU, s U'cU], [s U'cU, s^2 U'eU]], where
#   ATE: w = 1/(p - 1 + t), a = p(1-p), c = 1, e = 1/(p(1-p)), s = 1;
#   ATT: w = s (t - p)/(1 - p), a = p(1-p), c = p, e = p/(1-p), s = n/n_t.
# f' = 2 g'V^+ g' - g'V^+ V' V^+ g.
.cbps_alpha_fprime <- function(alpha, b0, X, treat, att) {
  probs.min <- 1e-06
  n <- nrow(X); eta <- drop(X %*% b0); p <- plogis(alpha * eta)
  inside <- p > probs.min & p < 1 - probs.min
  p <- pmin(1 - probs.min, pmax(probs.min, p))
  dp <- ifelse(inside, p * (1 - p) * eta, 0)
  s <- if (att) n / sum(treat == 1) else 1
  if (att) {
    w <- s * (treat - p) / (1 - p); dw <- ifelse(treat == 1, 0, -s / (1 - p)^2) * dp
    a <- p * (1 - p); cc <- p; e <- p / (1 - p); da <- (1 - 2 * p) * dp; dc <- dp; de <- dp / (1 - p)^2
  } else {
    w <- 1 / (p - 1 + treat); dw <- -dp / (p - 1 + treat)^2
    a <- p * (1 - p); cc <- rep(1, n); e <- 1 / (p * (1 - p)); da <- (1 - 2 * p) * dp; dc <- rep(0, n)
    de <- -(1 - 2 * p) / (p * (1 - p))^2 * dp
  }
  g <- c(crossprod(X, treat - p), crossprod(X, w)) / n
  dg <- c(-crossprod(X, dp), crossprod(X, dw)) / n
  blk <- function(h) crossprod(X * h, X) / n
  V <- rbind(cbind(blk(a), blk(cc) * s), cbind(blk(cc) * s, blk(e) * s^2))
  dV <- rbind(cbind(blk(da), blk(dc) * s), cbind(blk(dc) * s, blk(de) * s^2))
  Vi <- ginv(V)
  drop(2 * t(g) %*% Vi %*% dg - t(g) %*% Vi %*% dV %*% Vi %*% g)
}
# optimize()'s point a0 refined by bisection on the sign of f' within a0 +- 1e-6 (clamped to the
# interval); a0 is kept when f' does not change sign there (a minimum on the boundary).
.cbps_alpha_refine <- function(a0, b0, X, treat, ATT, probs.min) {
  stopifnot(probs.min == 1e-06)
  att <- ATT != 0
  lo <- max(0.8, a0 - 1e-6); hi <- min(1.1, a0 + 1e-6)
  if (!(.cbps_alpha_fprime(lo, b0, X, treat, att) < 0 && .cbps_alpha_fprime(hi, b0, X, treat, att) > 0)) return(a0)
  repeat {
    mid <- (lo + hi) / 2
    if (mid <= lo || mid >= hi) break
    if (.cbps_alpha_fprime(mid, b0, X, treat, att) > 0) hi <- mid else lo <- mid
  }
  (lo + hi) / 2
}

.cbps_tight_fun <- local({
  src <- deparse(CBPS:::CBPS.2Treat, width.cutoff = 500L)
  src <- .cbps_patch(src, "beta.curr * optimize(alpha.func, interval = c(0.8, 1.1))$min",
                     "beta.curr * .cbps_alpha_refine(optimize(alpha.func, interval = c(0.8, 1.1), tol = 1e-12)$minimum, beta.curr, X, treat, ATT, probs.min)", 1L)
  src <- .cbps_patch(src, "control = list(maxit = iterations)",
                     "control = list(maxit = iterations, reltol = 1e-15)", 6L)
  src <- .cbps_patch(src, "1/n.t * t(X * dw * sample.weights) %*% X",
                     "1/n * t(X * dw * sample.weights) %*% X", 1L)
  # 4. return the start value (alpha * glm coefficients) and the glm coefficients
  src <- .cbps_patch(src, "class(output) <- c(\"CBPS\", \"glm\", \"lm\")",
                     "output$gmm.init <- gmm.init; output$glm.coef <- glm1$coef; class(output) <- c(\"CBPS\", \"glm\", \"lm\")", 1L)
  f <- eval(parse(text = src))
  environment(f) <- asNamespace("CBPS")
  f
})

.cbps_sim <- function(ctx) {
  set.seed(20261102)
  n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n); x3 <- rbinom(n, 1, 0.4)
  p <- plogis(-0.3 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3)
  tr <- rbinom(n, 1, p)
  y <- 1 + 2 * tr + x1 + 0.5 * x2 + rnorm(n)
  ctx$gen_dataset("cbps_sim", data.frame(treat = tr, x1 = x1, x2 = x2, x3 = x3, y = y))
}

.cbps_nm <- c("(Intercept)", "x1", "x2", "x3")

# CBPS's standardised weights (standardize = TRUE), from fitted probabilities.
.cbps_weights <- function(tr, p, att) {
  if (att) ifelse(tr == 1, 1 / sum(tr == 1), (p / (1 - p)) / sum((p / (1 - p))[tr == 0]))
  else ifelse(tr == 1, (1 / p) / sum((1 / p)[tr == 1]), (1 / (1 - p)) / sum((1 / (1 - p))[tr == 0]))
}

# Package values: CBPS() as shipped.
.cbps_package <- function(d, method, att, prefix) {
  invisible(utils::capture.output(f <- suppressWarnings(CBPS(treat ~ x1 + x2 + x3, data = d, method = method, ATT = att,
                                                             iterations = 10000, standardize = TRUE, twostep = TRUE))))
  out <- list()
  out[[paste0(prefix, "_coef")]] <- named(setNames(as.numeric(f$coefficients), .cbps_nm))
  out[[paste0(prefix, "_se_reported")]] <- named(setNames(sqrt(diag(f$var)), .cbps_nm))
  out[[paste0(prefix, "_J")]] <- f$J
  out[[paste0(prefix, "_ps_head")]] <- vec(head(f$fitted.values, 10))
  out[[paste0(prefix, "_weights_head")]] <- vec(head(f$weights, 10))
  out[[paste0(prefix, "_deviance")]] <- f$deviance
  out
}

# Reference values: the patched CBPS.2Treat on CBPS.fit's whitened design.
.cbps_reference <- function(d, method, att, prefix) {
  X <- model.matrix(~ x1 + x2 + x3, d)
  X.orig <- X
  n <- nrow(X); k <- ncol(X)
  x.sd <- apply(as.matrix(X[, -1]), 2, sd)
  x.mean <- apply(as.matrix(X[, -1]), 2, mean)
  X[, -1] <- apply(as.matrix(X[, -1]), 2, FUN = function(x) (x - mean(x)) / sd(x))
  svd1 <- svd(X)
  U <- svd1$u
  invisible(utils::capture.output(o <- .cbps_tight_fun(factor(d$treat), U, method, k, ginv(crossprod(U)),
                                                       method == "exact", 10000, att, TRUE, TRUE, rep(1, n))))
  d.inv <- svd1$d
  d.inv[d.inv > 1e-05] <- 1 / d.inv[d.inv > 1e-05]
  d.inv[d.inv <= 1e-05] <- 0
  beta <- svd1$v %*% diag(d.inv) %*% coef(o)
  beta[-1, ] <- beta[-1, ] / x.sd
  beta[1, ] <- beta[1, ] - matrix(x.mean %*% beta[-1, ])
  # delta method: beta = Tm beta_u
  M <- diag(k); M[1, -1] <- -x.mean / x.sd; M[cbind(2:k, 2:k)] <- 1 / x.sd
  Tm <- M %*% svd1$v %*% diag(d.inv)
  stopifnot(max(abs(Tm %*% coef(o) - beta)) < 1e-12)
  vc <- Tm %*% o$var %*% t(Tm) / n
  Dx.inv <- diag(c(1, x.sd))
  vc_reported <- Dx.inv %*% ginv(t(X.orig) %*% X.orig) %*% t(X.orig) %*% U %*% svd1$v %*% ginv(diag(svd1$d)) %*%
    o$var %*% ginv(diag(svd1$d)) %*% t(svd1$v) %*% t(U) %*% X.orig %*% ginv(t(X.orig) %*% X.orig) %*% Dx.inv
  p <- as.numeric(o$fitted.values)
  stopifnot(max(abs(o$weights - .cbps_weights(d$treat, p, att))) < 1e-15)
  out <- list()
  out[[paste0(prefix, "_coef")]] <- named(setNames(as.numeric(beta), .cbps_nm))
  out[[paste0(prefix, "_se")]] <- named(setNames(sqrt(diag(vc)), .cbps_nm))
  out[[paste0(prefix, "_se_reported")]] <- named(setNames(sqrt(diag(vc_reported)), .cbps_nm))
  out[[paste0(prefix, "_J")]] <- o$J
  if (method == "over") out[[paste0(prefix, "_hansen_p")]] <- pchisq(n * o$J, df = k, lower.tail = FALSE)
  out[[paste0(prefix, "_ps_head")]] <- vec(head(p, 10))
  out[[paste0(prefix, "_weights_head")]] <- vec(head(as.numeric(o$weights), 10))
  out[[paste0(prefix, "_weights_sum_treated")]] <- sum(o$weights[d$treat == 1])
  out[[paste0(prefix, "_deviance")]] <- o$deviance
  out$alpha <- as.numeric(o$gmm.init[1] / o$glm.coef[1])
  stopifnot(max(abs(o$gmm.init / o$glm.coef - out$alpha)) < 1e-12)
  out
}

case("cbps_sim_methods", "econometrics/weighting", "CBPS exact / over, ATE and ATT, simulated n = 400 (package values and patched tight refit)", function(ctx) {
  d <- .cbps_sim(ctx)
  v <- list()
  for (m in c("exact", "over")) for (a in c(0, 1)) {
    est <- if (a == 0) "ate" else "att"
    v <- c(v, .cbps_package(d, m, a, paste0(m, "_", est)))
    r <- .cbps_reference(d, m, a, paste0(m, "_", est, "_tight"))
    alpha <- r$alpha; r$alpha <- NULL
    v <- c(v, r)
    if (m == "exact") v[[paste0(est, "_tight_alpha")]] <- alpha
  }
  g <- glm(treat ~ x1 + x2 + x3, data = d, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
  v$logit_coef <- named(coef(g))
  v$logit_se <- named(sqrt(diag(vcov(g))))
  v$logit_deviance <- deviance(g)
  v$logit_ate_weights_head <- vec(head(.cbps_weights(d$treat, fitted(g), FALSE), 10))
  v$logit_att_weights_head <- vec(head(.cbps_weights(d$treat, fitted(g), TRUE), 10))
  v$n_treated <- sum(d$treat == 1)
  list(values = v,
       reference = paste("CBPS::CBPS(method = 'exact' | 'over', ATT = 0 | 1, standardize = TRUE, twostep = TRUE);",
                         "*_tight = CBPS.2Treat with alpha = optimize(tol = 1e-12) refined by bisection on the analytic f'(alpha), optim(reltol = 1e-15) and the ATT gmm.gradient balance block divided by n (not n.t), on CBPS.fit's whitened design;",
                         "*_tight_se = delta method from CBPS.2Treat's whitened-scale sandwich; logit_* = glm(binomial, epsilon = 1e-14)"),
       call = "CBPS(treat ~ x1 + x2 + x3, data = cbps_sim, method, ATT)", packages = c("CBPS", "MASS"))
})
