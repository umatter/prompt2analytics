# validation/r/cases/discrete_newton.R -- nnet::multinom and MASS::polr references for the Newton
# loops of run_multinom and run_ordered_model (T1-WP07: RF-31, RF-32).
#  * disc_multinom_mn8: an ordinary 8-category fit (35 parameters). Its starting Hessian is well
#    conditioned but not positive definite, and faer 0.22's full-pivot LU inverted it wrongly (RF-31).
#  * disc_multinom_sep5: 5 categories x 6 rows with x ~ N(category, 1); at the MLE a linear predictor
#    exceeds ln(1e10) = 23.03, where the old exp(x'b).min(1e10) clamp distorted the fit (RF-32).
#  * disc_multinom_j4: 4 overlapping categories, 9 parameters (a control the old code already fit).
#  * disc_polr_j5 / _j16 / _j30: J categories x 6 rows, x ~ N(category / (J / 5), 1); MASS::polr logit
#    and probit. p2a's ordered fit is not polr's until ECON-4 (T2-WP10); these cases carry the
#    reference for that package and the counts and null log-likelihood T1-WP07 pins.
# Multinomial references are re-solved by Newton's method on the analytic score and information from
# nnet's estimate (*_tight); nnet's own vcov at reltol 1e-16 agrees to 1e-6 (asserted).
suppressPackageStartupMessages({ library(nnet); library(MASS) })

.mn_tight <- function(d, xs) {
  lev <- sort(unique(as.character(d$y)))            # p2a sorts the labels as strings; reference = lev[1]
  X <- cbind("(Intercept)" = 1, as.matrix(d[, xs, drop = FALSE]))
  k <- ncol(X); J <- length(lev) - 1; n <- nrow(X)
  Y <- sapply(lev[-1], function(l) as.numeric(as.character(d$y) == l))
  probs <- function(B) { V <- X %*% B; m <- pmax(0, apply(V, 1, max)); E <- exp(V - m); E / (exp(-m) + rowSums(E)) }
  info <- function(P) {
    I <- matrix(0, k * J, k * J)
    for (a in 1:J) for (b in 1:J) {
      w <- if (a == b) P[, a] * (1 - P[, a]) else -P[, a] * P[, b]
      I[(a - 1) * k + 1:k, (b - 1) * k + 1:k] <- crossprod(X * w, X)
    }
    I
  }
  m <- multinom(factor(as.character(d$y), levels = lev) ~ ., data = d[, xs, drop = FALSE], trace = FALSE,
                Hess = TRUE, maxit = 10000, reltol = 1e-16, abstol = 1e-300, MaxNWts = 100000)
  stopifnot(m$convergence == 0)
  B <- t(coef(m)); if (J == 1) B <- matrix(coef(m), ncol = 1)
  for (it in 1:100) {
    P <- probs(B)
    step <- solve(info(P), as.vector(crossprod(X, Y - P)))
    B <- B + matrix(step, k, J)
    if (max(abs(step) / pmax(abs(as.vector(B)), 1e-300)) < 1e-15) break
  }
  P <- probs(B); g <- as.vector(crossprod(X, Y - P)); I <- info(P)
  V <- solve(I); se <- sqrt(diag(V))
  stopifnot(max(abs(g) * se) < 1e-9)
  nm <- as.vector(outer(colnames(X), lev[-1], function(v, l) paste0(l, ":", v)))
  se_nnet <- sqrt(diag(vcov(m))); names(se_nnet) <- nm
  stopifnot(max(abs(se_nnet / se - 1)) < 1e-6)
  Pf <- cbind(1 - rowSums(P), P)
  ll <- sum(log(Pf[cbind(seq_len(n), match(as.character(d$y), lev))]))
  cnt <- table(factor(as.character(d$y), levels = lev))
  # Hessian of the log-likelihood at the start beta = 0, the matrix the first Newton step inverts
  H0 <- -info(matrix(1 / (J + 1), n, J))
  list(n = n, counts = named(setNames(as.numeric(cnt), lev)),
       coef_tight = named(setNames(as.vector(B), nm)), se_tight = named(setNames(se, nm)),
       z_tight = named(setNames(as.vector(B) / se, nm)),
       p_tight = named(setNames(2 * pnorm(-abs(as.vector(B) / se)), nm)),
       loglik = ll, null_loglik = sum(cnt * log(cnt / n)),
       aic = -2 * ll + 2 * k * J, bic = -2 * ll + k * J * log(n),
       loglik_pkg = as.numeric(logLik(m)),
       max_abs_xb = max(abs(X %*% B)),
       start_hessian_kappa = kappa(H0, exact = TRUE),
       start_hessian_max_eigen = max(eigen(H0, symmetric = TRUE, only.values = TRUE)$values))
}

case("disc_multinom_mn8", "econometrics/discrete", "nnet::multinom, 8 categories, 4 regressors (n = 500, 35 parameters)", function(ctx) {
  set.seed(8)
  n <- 500
  X <- data.frame(x1 = rnorm(n), x2 = rnorm(n), x3 = sample(1:5, n, TRUE), x4 = rnorm(n))
  B <- cbind(c(0, .3, -.2, .5, -.4, .1, .2, -.3), matrix(rnorm(8 * 4, 0, .4), 8, 4)); B[1, ] <- 0
  eta <- cbind(1, as.matrix(X)) %*% t(B); P <- exp(eta); P <- P / rowSums(P)
  y <- apply(P, 1, function(p) sample(1:8, 1, prob = p))
  d <- ctx$gen_dataset("multinom_mn8", data.frame(y = y, X))
  v <- .mn_tight(d, c("x1", "x2", "x3", "x4"))
  stopifnot(v$start_hessian_kappa < 1e4, v$start_hessian_max_eigen < 0)   # well conditioned, not PD
  list(values = v, reference = "nnet::multinom(factor(y) ~ x1 + x2 + x3 + x4) re-solved by Newton's method on the analytic score and information (*_tight; nnet's vcov at reltol 1e-16 agrees to 1e-6)",
       call = "multinom(factor(y) ~ x1 + x2 + x3 + x4, data = multinom_mn8, Hess = TRUE, reltol = 1e-16)", packages = "nnet")
})

case("disc_multinom_sep5", "econometrics/discrete", "nnet::multinom, 5 well-separated categories x 6 rows (a linear predictor above ln(1e10) at the MLE)", function(ctx) {
  set.seed(12)
  cats <- rep(1:5, each = 6)
  x <- rnorm(length(cats), cats, 1)
  d0 <- data.frame(y = sprintf("c%03d", cats), x = round(x, 5))
  d <- ctx$gen_dataset("multinom_sep5", d0[sample(nrow(d0)), ])
  v <- .mn_tight(d, "x")
  stopifnot(v$max_abs_xb > log(1e10))      # the property RF-32 needs
  list(values = v, reference = "nnet::multinom(factor(y) ~ x) re-solved by Newton's method on the analytic score and information (*_tight; nnet's vcov at reltol 1e-16 agrees to 1e-6)",
       call = "multinom(factor(y) ~ x, data = multinom_sep5, Hess = TRUE, reltol = 1e-16)", packages = "nnet")
})

case("disc_multinom_j4", "econometrics/discrete", "nnet::multinom, 4 overlapping categories, 2 regressors (n = 400, 9 parameters)", function(ctx) {
  set.seed(4)
  n <- 400
  x1 <- round(rnorm(n), 6); x2 <- round(rnorm(n), 6)
  eta <- cbind(0, 0.4 + 0.9 * x1 - 0.3 * x2, -0.2 + 0.2 * x1 + 0.7 * x2, 0.1 - 0.6 * x1 + 0.4 * x2)
  p <- exp(eta) / rowSums(exp(eta))
  y <- c("a", "b", "c", "d")[apply(p, 1, function(pr) sample.int(4, 1, prob = pr))]
  d <- ctx$gen_dataset("multinom_j4", data.frame(y = y, x1 = x1, x2 = x2))
  v <- .mn_tight(d, c("x1", "x2"))
  stopifnot(v$max_abs_xb < 10)
  list(values = v, reference = "nnet::multinom(factor(y) ~ x1 + x2) re-solved by Newton's method on the analytic score and information (*_tight; nnet's vcov at reltol 1e-16 agrees to 1e-6)",
       call = "multinom(factor(y) ~ x1 + x2, data = multinom_j4, Hess = TRUE, reltol = 1e-16)", packages = "nnet")
})

.polr_case <- function(ctx, J, seed) {
  set.seed(seed)
  cc <- rep(1:J, each = 6)
  x <- round(rnorm(length(cc), mean = cc / (J / 5), sd = 1), 6)
  d <- ctx$gen_dataset(sprintf("ordered_j%d", J), data.frame(y = sprintf("c%03d", cc), x = x))
  lev <- sort(unique(d$y)); yf <- factor(d$y, levels = lev, ordered = TRUE)
  cnt <- table(yf); n <- nrow(d)
  v <- list(n = n, counts = named(setNames(as.numeric(cnt), lev)), null_loglik = sum(cnt * log(cnt / n)))
  for (m in c("logistic", "probit")) {
    f <- polr(yf ~ x, data = d, method = m, Hess = TRUE, control = list(reltol = 1e-14, maxit = 10000))
    stopifnot(f$convergence == 0)
    tag <- if (m == "logistic") "logit" else "probit"
    se <- sqrt(diag(vcov(f)))
    v[[paste0(tag, "_coef")]] <- named(coef(f))
    v[[paste0(tag, "_zeta")]] <- named(f$zeta)
    v[[paste0(tag, "_se_coef")]] <- named(se[names(coef(f))])
    v[[paste0(tag, "_se_zeta")]] <- named(se[names(f$zeta)])
    v[[paste0(tag, "_loglik")]] <- as.numeric(logLik(f))
    v[[paste0(tag, "_null_loglik")]] <- as.numeric(logLik(polr(yf ~ 1, method = m)))
  }
  stopifnot(abs(v$logit_null_loglik - v$null_loglik) < 1e-6 * abs(v$null_loglik))
  list(values = v, reference = "MASS::polr(method = 'logistic' | 'probit', Hess = TRUE, control = list(reltol = 1e-14)); null = sum n_j log(n_j / n)",
       call = sprintf("polr(factor(y, ordered = TRUE) ~ x, data = ordered_j%d, method = 'logistic' | 'probit', Hess = TRUE)", J),
       packages = "MASS")
}
case("disc_polr_j5", "econometrics/discrete", "MASS::polr logit and probit, 5 categories x 6 rows (control below the 16-parameter trigger)", function(ctx) .polr_case(ctx, 5, 5))
case("disc_polr_j16", "econometrics/discrete", "MASS::polr logit and probit, 16 categories x 6 rows (16 parameters)", function(ctx) .polr_case(ctx, 16, 16))
case("disc_polr_j30", "econometrics/discrete", "MASS::polr logit and probit, 30 categories x 6 rows (30 parameters)", function(ctx) .polr_case(ctx, 30, 30))
