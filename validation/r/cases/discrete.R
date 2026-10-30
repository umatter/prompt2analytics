# validation/r/cases/discrete.R -- MASS::polr, MASS::glm.nb and pscl::zeroinfl references
# (T2-WP10: ECON-4, ECON-5, ECON-14, ECON-15, ECON-19; ECON-12 sites in ordered_models.rs and
# count_models.rs). Mixed logit: discrete_mixed.R (T2-WP18).
#
# The package optimisers stop early (polr/zeroinfl: optim BFGS, whose gradient is still ~1e-6 at
# reltol = 1e-15; glm.nb: theta.ml stops at |Newton step| < eps^0.25 and SE.theta is evaluated one
# step before the reported theta). Every case therefore (1) fits with the package, recorded as
# `*_pkg` for a looser cross-check, (2) polishes that fit with Newton steps on the exact
# log-likelihood (analytic gradient, numDeriv::jacobian of it) until max |gradient| < 1e-9, and
# (3) takes standard errors at the polished optimum: from numDeriv::hessian of the log-likelihood
# (polr, zeroinfl; glm.nb's beta SEs are the expected information (X'WX)^-1, as summary.negbin),
# SE(theta) = 1/sqrt(theta.ml information) at the final mu (MASS). The analytic ordered-model
# Hessian below is the formula p2a implements; the case asserts it equals numDeriv's.
suppressPackageStartupMessages({ library(MASS); library(pscl); library(numDeriv) })

.polish <- function(par, gr, tol = 1e-9, max_steps = 20) {
  for (s in seq_len(max_steps)) {
    g <- gr(par)
    if (max(abs(g)) < tol) break
    H <- numDeriv::jacobian(gr, par); H <- (H + t(H)) / 2
    par <- par - solve(H, g)
  }
  stopifnot(max(abs(gr(par))) < tol)
  par
}
.z_p <- function(est, se) { z <- est / se; list(z = z, p = 2 * pnorm(-abs(z))) }

# ---- ordered logit / probit ---------------------------------------------------------------
.links <- function(link) {
  if (link == "logistic") list(F = plogis, f = dlogis, fp = function(u) dlogis(u) * (1 - 2 * plogis(u)))
  else list(F = pnorm, f = dnorm, fp = function(u) -u * dnorm(u))
}
.polr_ll <- function(par, X, y, J, link) {
  k <- ncol(X); b <- par[1:k]; zeta <- par[(k + 1):(k + J - 1)]
  F <- .links(link)$F
  eta <- drop(X %*% b); cuts <- c(-Inf, zeta, Inf)
  sum(log(F(cuts[y + 1] - eta) - F(cuts[y] - eta)))
}
.polr_grad <- function(par, X, y, J, link) {
  k <- ncol(X); b <- par[1:k]; zeta <- par[(k + 1):(k + J - 1)]
  L <- .links(link)
  eta <- drop(X %*% b); cuts <- c(-Inf, zeta, Inf)
  u <- cuts[y + 1] - eta; l <- cuts[y] - eta
  p <- L$F(u) - L$F(l); fu <- L$f(u); fl <- L$f(l)
  gz <- vapply(1:(J - 1), function(j) sum((fu / p)[y == j]) - sum((fl / p)[y == j + 1]), 0)
  c(colSums(X * ((fl - fu) / p)), gz)
}
# Analytic observed Hessian (the p2a algorithm): per observation in category c (1-based), with
# u = zeta_c - eta, l = zeta_{c-1} - eta, p = F(u) - F(l):
#   dp/db = x (f(l) - f(u)), dp/dzeta_c = f(u), dp/dzeta_{c-1} = -f(l)
#   d2p/db db' = x x' (f'(u) - f'(l)), d2p/db dzeta_c = -x f'(u), d2p/db dzeta_{c-1} = x f'(l),
#   d2p/dzeta_c^2 = f'(u), d2p/dzeta_{c-1}^2 = -f'(l)
#   H_i = d2p / p - dp dp' / p^2
.polr_hess <- function(par, X, y, J, link) {
  k <- ncol(X); b <- par[1:k]; zeta <- par[(k + 1):(k + J - 1)]
  L <- .links(link)
  eta <- drop(X %*% b); cuts <- c(-Inf, zeta, Inf)
  H <- matrix(0, k + J - 1, k + J - 1)
  for (i in seq_along(y)) {
    c <- y[i]; u <- cuts[c + 1] - eta[i]; l <- cuts[c] - eta[i]
    p <- L$F(u) - L$F(l); fu <- L$f(u); fl <- L$f(l)
    fpu <- if (is.finite(u)) L$fp(u) else 0; fpl <- if (is.finite(l)) L$fp(l) else 0
    x <- X[i, ]
    dp <- numeric(k + J - 1); d2 <- matrix(0, k + J - 1, k + J - 1)
    dp[1:k] <- x * (fl - fu); d2[1:k, 1:k] <- outer(x, x) * (fpu - fpl)
    if (c <= J - 1) { jc <- k + c; dp[jc] <- fu; d2[jc, jc] <- fpu; d2[1:k, jc] <- -x * fpu; d2[jc, 1:k] <- -x * fpu }
    if (c >= 2) { jl <- k + c - 1; dp[jl] <- -fl; d2[jl, jl] <- -fpl; d2[1:k, jl] <- x * fpl; d2[jl, 1:k] <- x * fpl }
    H <- H + d2 / p - outer(dp, dp) / p^2
  }
  H
}
.polr_values <- function(d, yname, xnames, link) {
  frm <- as.formula(paste0("factor(", yname, ") ~ ", paste(xnames, collapse = " + ")))
  f <- polr(frm, data = d, method = link, Hess = TRUE, control = list(reltol = 1e-15, maxit = 10000))
  dd <- d[complete.cases(d[, c(yname, xnames)]), ]
  X <- as.matrix(dd[, xnames, drop = FALSE]); fy <- factor(dd[[yname]]); y <- as.integer(fy); J <- nlevels(fy)
  par0 <- c(coef(f), f$zeta)
  par <- .polish(par0, function(p) .polr_grad(p, X, y, J, link))
  names(par) <- names(par0)
  H <- numDeriv::hessian(function(p) .polr_ll(p, X, y, J, link), par)
  Ha <- .polr_hess(par, X, y, J, link)
  stopifnot(max(abs(Ha - H)) < 1e-6 * max(abs(H)))        # the analytic Hessian p2a implements
  se <- sqrt(diag(solve(-Ha))); names(se) <- names(par)
  se_nd <- sqrt(diag(solve(-H))); names(se_nd) <- names(par)
  kb <- names(coef(f)); kz <- names(f$zeta)
  ll <- .polr_ll(par, X, y, J, link); n <- nrow(dd); tab <- as.vector(table(fy))
  ll0 <- sum(tab * log(tab / n)); npar <- length(par)
  zp <- .z_p(par[kb], se[kb])
  list(coef = named(par[kb]), zeta = named(par[kz]), se_coef = named(se[kb]), se_zeta = named(se[kz]),
       z = named(zp$z), p = named(zp$p), se_coef_numderiv = named(se_nd[kb]),
       coef_pkg = named(coef(f)), zeta_pkg = named(f$zeta), se_coef_pkg = named(sqrt(diag(vcov(f)))[kb]),
       loglik = ll, loglik_pkg = as.numeric(logLik(f)), loglik_null = ll0,
       loglik_null_pkg = as.numeric(logLik(polr(factor(dd[[yname]]) ~ 1, method = link))),
       pseudo_r2 = 1 - ll / ll0, aic = -2 * ll + 2 * npar, bic = -2 * ll + log(n) * npar,
       counts = vec(tab), max_abs_grad = max(abs(.polr_grad(par, X, y, J, link))),
       max_abs_grad_pkg = max(abs(.polr_grad(par0, X, y, J, link))), n = n)
}

case("disc_ordered", "econometrics/discrete", "Ordered logit and probit (MASS::polr), y coded 1/2/3 (integer column)", function(ctx) {
  set.seed(2026)
  n <- 400
  x1 <- rnorm(n); x2 <- rnorm(n)
  ystar <- 0.9 * x1 - 0.7 * x2 + rlogis(n)
  y <- cut(ystar, c(-Inf, -0.5, 0.8, Inf), labels = FALSE)
  lab <- c("low", "mid", "high")[y]               # not alphabetical in ordinal order
  # No further random draws below: the columns added for T2-WP10 are functions of the draws above.
  y_wide <- cut(ystar, quantile(ystar, seq(0, 1, length.out = 14)), labels = FALSE, include.lowest = TRUE) - 3L
  y_na <- y; y_na[c(5, 17, 123, 301)] <- NA
  x1r <- round(x1, 8); x1_na <- x1r; x1_na[c(9, 200, 377)] <- NA
  d <- ctx$gen_dataset("ordered", data.frame(y = y, y_label = lab, x1 = x1r, x2 = round(x2, 8),
                                             y_wide = y_wide, y_na = y_na, x1_na = x1_na))
  stopifnot(is.integer(d$y), is.integer(d$y_wide), is.integer(d$y_na), min(d$y_wide) == -2, max(d$y_wide) == 10,
            all(table(d$y_wide) > 0), sum(!complete.cases(d$y_na, d$x1_na, d$x2)) == 7)
  lo <- .polr_values(d, "y", c("x1", "x2"), "logistic"); pr <- .polr_values(d, "y", c("x1", "x2"), "probit")
  names(lo) <- paste0("logit_", names(lo)); names(pr) <- paste0("probit_", names(pr))
  # Text labels in ordinal order give the integer fit; the alphabetical order R would use for the
  # string labels (high < low < mid) is a different model.
  stopifnot(identical(match(d$y_label, c("low", "mid", "high")), d$y), identical(sort(unique(d$y_label)), c("high", "low", "mid")))
  d$y_alpha <- factor(d$y_label)
  al <- .polr_values(d, "y_alpha", c("x1", "x2"), "logistic")
  # 13 integer levels -2..10: numeric order, not string order ("-1" < "-2", "10" < "2")
  wd <- .polr_values(d, "y_wide", c("x1", "x2"), "logistic")
  # Missing values: polr drops incomplete rows (na.omit)
  na <- .polr_values(d, "y_na", c("x1_na", "x2"), "logistic")
  list(values = c(lo, pr,
    list(alpha_labels_coef = al$coef, alpha_labels_zeta = al$zeta, alpha_labels_loglik = al$loglik,
         wide_coef = wd$coef, wide_zeta = wd$zeta, wide_se_coef = wd$se_coef, wide_loglik = wd$loglik,
         wide_levels = vec(sort(unique(d$y_wide))), wide_counts = wd$counts,
         na_coef = na$coef, na_zeta = na$zeta, na_se_coef = na$se_coef, na_loglik = na$loglik, na_n_used = na$n,
         na_n_dropped = nrow(d) - na$n, na_missing_y = sum(is.na(d$y_na)), na_missing_x1 = sum(is.na(d$x1_na)))),
       reference = "MASS::polr(method = 'logistic' | 'probit', control = list(reltol = 1e-15)), polished by Newton steps on the exact log-likelihood; SEs from the observed information (analytic, equal to numDeriv::hessian)",
       call = "polr(factor(y) ~ x1 + x2, data = d, method = ...)", packages = c("MASS", "numDeriv"))
})
