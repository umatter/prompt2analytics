# validation/r/cases/spatial_ml_weights.R -- spatial cross-section references. spdep and spatialreg are not
# installed, so their formulas are written out here exactly as those packages implement them:
#  * W: 4 nearest neighbours (asymmetric) on the coordinates; spdep::nb2listw styles W, B, C, U, minmax, S
#    (S = Tiefelsdorf, Griffith & Boots 1999; C and S scale to eff.n = n - #islands under zero.policy = TRUE;
#    minmax divides by min(max row sum, max column sum), Kelejian & Prucha 2010);
#  * log|I - rho W| = Re(sum(log(1 - rho * lambda))) over the complex eigenvalues (spatialreg method "eigen");
#  * SAR (lagsarlm), SDM (lagsarlm Durbin = TRUE), SEM (errorsarlm) and SAC (sacsarlm) by their concentrated
#    likelihoods; the maximiser is polished on the analytic score, so the references are accurate to ~1e-12
#    (the flat likelihood limits a value-based search such as optimize() or p2a's golden section to ~1e-8);
#  * spatialreg::impacts: S_k = (I - rho W)^-1 (beta_k I + theta_k W), direct = mean(diag(S_k)),
#    total = mean(rowSums(S_k)), indirect = total - direct (mixedImpactsExact; theta = 0 for the SAR);
#  * SAR information-matrix SEs (Anselin 1988, spatialreg's asymptotic covariance, sar_se) are recorded as the
#    convention reference; p2a reports the observed information of the full log-likelihood at the MLE
#    (sar_se_obs, sdm_se_obs, sem_lambda_se_obs, sac_se_obs); errorsarlm beta SEs are
#    sqrt(s2 diag((X*'X*)^-1)), s2 = SSE / n;
#  * spdep::moran.test(randomisation = TRUE, alternative = "greater", zero.policy = TRUE, adjust.n = TRUE)
#    and the geary() statistic C = (n1 / (2 S0)) sum w_ij (x_i - x_j)^2 / sum z^2 with the adjusted n1;
#    Geary's randomisation variance is the written-out Cliff-Ord formula, checked here against an exact
#    permutation variance (whether spdep::geary.test takes K over n or over n_adj was not checked).

.sp_knn_b <- function(xy, k) { D <- as.matrix(dist(xy)); diag(D) <- Inf; B <- matrix(0, nrow(xy), nrow(xy))
  for (i in seq_len(nrow(xy))) B[i, order(D[i, ])[1:k]] <- 1; B }
# The k-th and (k+1)-th neighbour distances are well separated, so ties cannot change the neighbour sets.
.sp_knn_gap <- function(xy, k) { D <- as.matrix(dist(xy)); diag(D) <- Inf
  min(apply(D, 1, function(r) { s <- sort(r); s[k + 1] - s[k] })) }
.sp_style <- function(B, s) {
  n <- nrow(B); eff_n <- sum(rowSums(B) > 0); rs <- rowSums(B); rs[rs == 0] <- 1
  switch(s, B = B, W = B / rs, C = B * eff_n / sum(B), U = B / sum(B),
         minmax = B / min(max(rowSums(B)), max(colSums(B))),
         S = { q <- sqrt(rowSums(B^2)); q[q == 0] <- 1; Bs <- B / q; Bs * eff_n / sum(Bs) })
}
.sp_logdet <- function(rho, ev) Re(sum(log(1 - rho * ev)))
.sp_tr_aw <- function(rho, ev) Re(sum(ev / (1 - rho * ev)))   # tr(W (I - rho W)^-1)

.sp_data <- function(ctx) {
  set.seed(20261202)
  n <- 80
  xy <- cbind(runif(n), runif(n)); x1 <- rnorm(n); x2 <- rnorm(n)
  d <- data.frame(id = 1:n, cx = round(xy[, 1], 10), cy = round(xy[, 2], 10), x1 = round(x1, 10), x2 = round(x2, 10))
  W <- .sp_style(.sp_knn_b(as.matrix(d[, c("cx", "cy")]), 4), "W")
  set.seed(20261203)
  X <- cbind(1, d$x1, d$x2)
  d$y <- round(drop(solve(diag(n) - 0.4 * W, X %*% c(1, 0.8, -0.5) + W %*% (0.3 * d$x1) + rnorm(n, 0, 0.7))), 10)
  ctx$gen_dataset("spatial_sim", d)
}

# Concentrated SAR likelihood (Ord 1975): beta(rho) = (X'X)^-1 X'(y - rho W y), s2 = SSE / n.
.sp_sar_fit <- function(y, X, W, ev) {
  n <- length(y); k <- ncol(X); wy <- drop(W %*% y)
  qx <- qr(X); stopifnot(qx$rank == k)
  ey <- qr.resid(qx, y); ew <- qr.resid(qx, wy)
  a <- sum(ey^2); b <- sum(ey * ew); cc <- sum(ew^2)
  s2f <- function(r) (a - 2 * r * b + r^2 * cc) / n
  ll <- function(r) -n / 2 * log(2 * pi * s2f(r)) - n / 2 + .sp_logdet(r, ev)
  score <- function(r) -(r * cc - b) / s2f(r) - .sp_tr_aw(r, ev)
  lo <- 1 / min(Re(ev)); hi <- 1 / max(Re(ev))
  r0 <- optimize(ll, c(lo + 1e-8, hi - 1e-8), maximum = TRUE, tol = 1e-12)$maximum
  rho <- uniroot(score, c(r0 - 1e-3, r0 + 1e-3), tol = 1e-15, maxiter = 1000)$root
  stopifnot(abs(rho - r0) < 1e-6)
  Ay <- y - rho * wy; beta <- qr.coef(qx, Ay); names(beta) <- colnames(X)
  e <- drop(Ay - X %*% beta); s2 <- sum(e^2) / n
  A <- diag(n) - rho * W; Ainv <- solve(A); WA <- W %*% Ainv; WAXb <- WA %*% X %*% beta
  I <- matrix(0, k + 2, k + 2)
  I[1:k, 1:k] <- crossprod(X) / s2; I[1:k, k + 1] <- I[k + 1, 1:k] <- crossprod(X, WAXb) / s2
  I[k + 1, k + 1] <- sum(diag(WA %*% WA)) + sum(diag(crossprod(WA))) + sum(WAXb^2) / s2
  I[k + 1, k + 2] <- I[k + 2, k + 1] <- sum(diag(WA)) / s2; I[k + 2, k + 2] <- n / (2 * s2^2)
  se <- sqrt(diag(solve(I)))[1:(k + 1)]; names(se) <- c(colnames(X), "rho")
  llv <- ll(rho); npar <- k + 2
  list(rho = rho, beta = beta, s2 = s2, se = se, ll = llv, aic = -2 * llv + 2 * npar, bic = -2 * llv + log(n) * npar,
       e = e, Ainv = Ainv, rho_optimize = r0)
}
# Observed information of the full SAR/SDM log-likelihood at the MLE, minus the analytic Hessian; cross-checked
# against stats::optimHess of the full log-likelihood (base R).
.sp_sar_obs_se <- function(y, X, W, ev, f) {
  n <- length(y); k <- ncol(X); wy <- drop(W %*% y); e <- f$e; s2 <- f$s2
  I <- matrix(0, k + 2, k + 2)
  I[1:k, 1:k] <- crossprod(X) / s2
  I[1:k, k + 1] <- I[k + 1, 1:k] <- crossprod(X, wy) / s2
  I[1:k, k + 2] <- I[k + 2, 1:k] <- crossprod(X, e) / s2^2
  I[k + 1, k + 1] <- Re(sum((ev / (1 - f$rho * ev))^2)) + sum(wy^2) / s2
  I[k + 1, k + 2] <- I[k + 2, k + 1] <- sum(e * wy) / s2^2
  I[k + 2, k + 2] <- -n / (2 * s2^2) + sum(e^2) / s2^3
  ll <- function(p) { u <- y - p[k + 1] * wy - drop(X %*% p[1:k])
    -n / 2 * log(2 * pi * p[k + 2]) + .sp_logdet(p[k + 1], ev) - sum(u^2) / (2 * p[k + 2]) }
  stopifnot(max(abs(I + optimHess(c(f$beta, f$rho, s2), ll)) / pmax(abs(I), 1)) < 1e-4)
  se <- sqrt(diag(solve(I)))[1:(k + 1)]; names(se) <- c(colnames(X), "rho"); se
}
# spatialreg mixedImpactsExact (theta = NULL: the SAR's lagImpactsExact).
.sp_impacts <- function(Ainv, W, beta, theta = NULL) {
  n <- nrow(W); p <- length(beta); out <- matrix(0, 3, p, dimnames = list(c("direct", "indirect", "total"), names(beta)))
  for (j in seq_len(p)) { S <- Ainv %*% (diag(n) * beta[j] + if (is.null(theta)) 0 else W * theta[j])
    out["direct", j] <- mean(diag(S)); out["total", j] <- mean(rowSums(S)); out["indirect", j] <- out["total", j] - out["direct", j] }
  out
}

case("spatial_sar_sdm_sem", "econometrics/spatial", "SAR / SDM / SEM / SAC by concentrated ML on a 4-NN row-standardised W, impacts, complex-eigenvalue log-determinant", function(ctx) {
  d <- .sp_data(ctx); n <- nrow(d); xy <- as.matrix(d[, c("cx", "cy")])
  stopifnot(.sp_knn_gap(xy, 4) > 1e-6)
  W <- .sp_style(.sp_knn_b(xy, 4), "W"); y <- d$y
  X <- cbind("(Intercept)" = 1, x1 = d$x1, x2 = d$x2)
  ev <- eigen(W, only.values = TRUE)$values
  # Near-defective eigenvalues of a kNN W come back with |Im| up to ~1e-6 (LAPACK and faer alike); the
  # genuine complex pairs have |Im| > 0.01, so the count uses a 1e-4 threshold with a clear gap around it.
  aim <- abs(Im(ev)); cplx <- aim > 1e-4
  stopifnot(!any(aim > 1e-5 & aim < 1e-3))
  v <- list(n_complex_eigen = sum(cplx), rho_range = vec(c(1 / min(Re(ev)), 1 / max(Re(ev)))))
  rhos <- c(-0.8, 0.5, 0.9)
  v$logdet_rho <- vec(rhos); v$logdet <- vec(vapply(rhos, .sp_logdet, 0, ev = ev))
  v$logdet_realparts <- vec(vapply(rhos, function(r) sum(log(abs(1 - r * Re(ev)))), 0))   # the pre-fix value
  v$logdet_direct <- vec(vapply(rhos, function(r) as.numeric(determinant(diag(n) - r * W)$modulus), 0))
  v$tr_inv_rho05 <- Re(sum(1 / (1 - 0.5 * ev))); v$tr_aw_rho05 <- .sp_tr_aw(0.5, ev)
  # SAR
  s <- .sp_sar_fit(y, X, W, ev)
  v$sar_rho <- s$rho; v$sar_beta <- named(s$beta); v$sar_s2 <- s$s2; v$sar_loglik <- s$ll; v$sar_aic <- s$aic; v$sar_bic <- s$bic
  v$sar_se <- named(s$se)                                             # expected information: the convention reference
  v$sar_se_obs <- named(.sp_sar_obs_se(y, X, W, ev, s))
  v$sar_residuals <- vec(s$e)
  imp <- .sp_impacts(s$Ainv, W, s$beta[-1])
  v$sar_impact_direct <- named(imp["direct", ]); v$sar_impact_indirect <- named(imp["indirect", ]); v$sar_impact_total <- named(imp["total", ])
  # SDM: lag.x1, lag.x2 (W is row-standardised, so the intercept is not lagged, as spatialreg::create_WX)
  Xd <- cbind(X, lag.x1 = drop(W %*% d$x1), lag.x2 = drop(W %*% d$x2))
  sd <- .sp_sar_fit(y, Xd, W, ev)
  v$sdm_rho <- sd$rho; v$sdm_beta <- named(sd$beta); v$sdm_s2 <- sd$s2; v$sdm_loglik <- sd$ll; v$sdm_aic <- sd$aic; v$sdm_bic <- sd$bic
  v$sdm_se_obs <- named(.sp_sar_obs_se(y, Xd, W, ev, sd))
  imp <- .sp_impacts(sd$Ainv, W, sd$beta[c("x1", "x2")], sd$beta[c("lag.x1", "lag.x2")])
  v$sdm_impact_direct <- named(imp["direct", ]); v$sdm_impact_indirect <- named(imp["indirect", ]); v$sdm_impact_total <- named(imp["total", ])
  # SEM: y = X b + u, u = lambda W u + e; score n e'W u / SSE - tr(W B^-1)
  wy <- drop(W %*% y); WX <- W %*% X
  sem_core <- function(l) { Bs <- y - l * wy; Xs <- X - l * WX; q <- qr(Xs); b <- qr.coef(q, Bs); e <- drop(Bs - Xs %*% b)
    list(b = b, e = e, u = drop(y - X %*% b), sse = sum(e^2), q = q) }
  llsem <- function(l) { c <- sem_core(l); -n / 2 * log(2 * pi * c$sse / n) - n / 2 + .sp_logdet(l, ev) }
  semscore <- function(l) { c <- sem_core(l); n * sum(c$e * (W %*% c$u)) / c$sse - .sp_tr_aw(l, ev) }
  lo <- 1 / min(Re(ev)); hi <- 1 / max(Re(ev))
  l0 <- optimize(llsem, c(lo + 1e-8, hi - 1e-8), maximum = TRUE, tol = 1e-12)$maximum
  lam <- uniroot(semscore, c(l0 - 1e-3, l0 + 1e-3), tol = 1e-15, maxiter = 1000)$root
  c <- sem_core(lam); s2 <- c$sse / n; bse <- sqrt(diag(chol2inv(qr.R(c$q))) * s2)
  names(c$b) <- names(bse) <- colnames(X); z <- c$b / bse
  v$sem_lambda <- lam; v$sem_beta <- named(c$b); v$sem_beta_se <- named(bse); v$sem_beta_z <- named(z)
  v$sem_beta_p <- named(2 * pnorm(abs(z), lower.tail = FALSE)); v$sem_s2 <- s2
  v$sem_loglik <- llsem(lam); v$sem_aic <- -2 * v$sem_loglik + 2 * (ncol(X) + 2); v$sem_bic <- -2 * v$sem_loglik + log(n) * (ncol(X) + 2)
  # lambda's SE from the (lambda, s2) block of the observed information; the beta-lambda block is set to its
  # expectation 0, as errorsarlm's block-diagonal asymptotic covariance
  wu <- drop(W %*% c$u)
  I2 <- matrix(c(Re(sum((ev / (1 - lam * ev))^2)) + sum(wu^2) / s2, sum(c$e * wu) / s2^2,
                 sum(c$e * wu) / s2^2, -n / (2 * s2^2) + sum(c$e^2) / s2^3), 2)
  v$sem_lambda_se_obs <- sqrt(solve(I2)[1, 1])
  # SAC: y = rho W y + X b + u, u = lambda W u + e
  sac_core <- function(r, l) { Ay <- y - r * wy; By <- Ay - l * drop(W %*% Ay); Xs <- X - l * WX
    q <- qr(Xs); b <- qr.coef(q, By); e <- drop(By - Xs %*% b); u <- drop(Ay - X %*% b)
    list(b = b, e = e, u = u, sse = sum(e^2)) }
  llsac <- function(p) { c <- sac_core(p[1], p[2]); -n / 2 * log(2 * pi * c$sse / n) - n / 2 + .sp_logdet(p[1], ev) + .sp_logdet(p[2], ev) }
  sacgrad <- function(p) { c <- sac_core(p[1], p[2]); Bwy <- wy - p[2] * drop(W %*% wy)
    c(n * sum(c$e * Bwy) / c$sse - .sp_tr_aw(p[1], ev), n * sum(c$e * (W %*% c$u)) / c$sse - .sp_tr_aw(p[2], ev)) }
  grid <- seq(-0.95, 0.95, by = 0.05); best <- c(0, 0); bv <- -Inf
  for (r in grid) for (l in grid) { val <- llsac(c(r, l)); if (is.finite(val) && val > bv) { bv <- val; best <- c(r, l) } }
  o <- optim(best, function(p) -llsac(p), function(p) -sacgrad(p), method = "BFGS", control = list(reltol = 1e-16, maxit = 10000))
  p <- o$par
  for (it in 1:5) { h <- 1e-6; H <- cbind((sacgrad(p + c(h, 0)) - sacgrad(p - c(h, 0))) / (2 * h), (sacgrad(p + c(0, h)) - sacgrad(p - c(0, h))) / (2 * h))
    p <- p - solve(H, sacgrad(p)) }
  stopifnot(max(abs(sacgrad(p))) < 1e-9, max(abs(p - o$par)) < 1e-5)
  c <- sac_core(p[1], p[2]); names(c$b) <- colnames(X)
  v$sac_rho <- p[1]; v$sac_lambda <- p[2]; v$sac_beta <- named(c$b); v$sac_s2 <- c$sse / n
  v$sac_loglik <- llsac(p); v$sac_aic <- -2 * v$sac_loglik + 2 * (ncol(X) + 3); v$sac_bic <- -2 * v$sac_loglik + log(n) * (ncol(X) + 3)
  # SAC standard errors: minus the Hessian of the full log-likelihood by central differences of its analytic
  # gradient, step 1e-5 max(|theta_i|, 1), symmetrised; cross-checked against optimHess of the log-likelihood
  sacfullgrad <- function(th) { k <- ncol(X); b <- th[1:k]; r <- th[k + 1]; l <- th[k + 2]; sg <- th[k + 3]
    u <- y - r * wy - drop(X %*% b); e <- u - l * drop(W %*% u); Xs <- X - l * WX
    c(drop(crossprod(Xs, e)) / sg,
      -.sp_tr_aw(r, ev) + sum(e * (wy - l * drop(W %*% wy))) / sg,
      -.sp_tr_aw(l, ev) + sum(e * (W %*% u)) / sg,
      -n / (2 * sg) + sum(e^2) / (2 * sg^2)) }
  sacfullll <- function(th) { k <- ncol(X); u <- y - th[k + 1] * wy - drop(X %*% th[1:k]); e <- u - th[k + 2] * drop(W %*% u)
    -n / 2 * log(2 * pi * th[k + 3]) + .sp_logdet(th[k + 1], ev) + .sp_logdet(th[k + 2], ev) - sum(e^2) / (2 * th[k + 3]) }
  th <- c(c$b, p[1], p[2], v$sac_s2); m <- length(th); H <- matrix(0, m, m)
  for (i in seq_len(m)) { h <- 1e-5 * max(abs(th[i]), 1); d <- numeric(m); d[i] <- h
    H[, i] <- (sacfullgrad(th + d) - sacfullgrad(th - d)) / (2 * h) }
  H <- (H + t(H)) / 2
  stopifnot(max(abs(H - optimHess(th, sacfullll)) / pmax(abs(H), 1)) < 1e-4)
  se <- sqrt(diag(solve(-H)))[1:(ncol(X) + 2)]; names(se) <- c(colnames(X), "rho", "lambda")
  v$sac_se_obs <- named(se)
  list(values = v, reference = "spatialreg::lagsarlm (method 'eigen', Durbin = TRUE), errorsarlm, sacsarlm and impacts (mixedImpactsExact), written out (spdep/spatialreg not installed)",
       call = "concentrated SAR/SDM/SEM/SAC likelihoods on nb2listw(knn2nb(knearneigh(cbind(cx, cy), 4)), style = 'W'); maximiser polished on the analytic score",
       packages = character(0))
})

case("spatial_weights_moran", "spatial/weights", "spdep weight styles W, B, C, U, minmax, S; Moran's I and Geary's C with islands (zero.policy, adjust.n)", function(ctx) {
  d <- .sp_data(ctx); n <- nrow(d); xy <- as.matrix(d[, c("cx", "cy")])
  D <- as.matrix(dist(xy)); Dut <- D[upper.tri(D)]
  stopifnot(.sp_knn_gap(xy, 4) > 1e-6, min(abs(Dut - 0.2)) > 1e-6, min(abs(Dut - 0.12)) > 1e-6)   # < and <= agree
  mor <- function(z, W, n_adj) {
    z <- z - mean(z); S0 <- sum(W); S1 <- 0.5 * sum((W + t(W))^2); S2 <- sum((rowSums(W) + colSums(W))^2)
    I <- (n_adj / S0) * drop(t(z) %*% W %*% z) / sum(z^2); EI <- -1 / (n_adj - 1); nn <- n_adj
    vn <- (nn^2 * S1 - nn * S2 + 3 * S0^2) / ((nn^2 - 1) * S0^2) - EI^2
    k <- (sum(z^4) / length(z)) / (sum(z^2) / length(z))^2           # kurtosis over all n (spdep moran())
    vr <- (nn * ((nn^2 - 3 * nn + 3) * S1 - nn * S2 + 3 * S0^2) - k * ((nn^2 - nn) * S1 - 2 * nn * S2 + 6 * S0^2)) / ((nn - 1) * (nn - 2) * (nn - 3) * S0^2) - EI^2
    zr <- (I - EI) / sqrt(vr)
    c(I = I, E = EI, var_norm = vn, var_rand = vr, z_rand = zr, p_rand = pnorm(zr, lower.tail = FALSE), p_two_sided = 2 * pnorm(abs(zr), lower.tail = FALSE))
  }
  gearyC <- function(x, W, n_adj) { z <- x - mean(x); (n_adj - 1) / (2 * sum(W)) * sum(W * outer(x, x, "-")^2) / sum(z^2) }
  # Cliff-Ord randomisation variance of Geary's C; nn = units with neighbours, K over all n as in mor()
  gearyVar <- function(x, W, nn) { z <- x - mean(x); S0 <- sum(W); S1 <- 0.5 * sum((W + t(W))^2); S2 <- sum((rowSums(W) + colSums(W))^2)
    K <- (sum(z^4) / length(z)) / (sum(z^2) / length(z))^2
    ((nn - 1) * S1 * (nn^2 - 3 * nn + 3 - (nn - 1) * K) - 0.25 * (nn - 1) * S2 * (nn^2 + 3 * nn - 6 - K * (nn^2 - nn + 2)) + S0^2 * (nn^2 - 3 - K * (nn - 1)^2)) / (nn * (nn - 2) * (nn - 3) * S0^2) }
  geary_z_p <- function(C, vc) { z <- (C - 1) / sqrt(vc); c(z = z, p = 2 * pnorm(-abs(z))) }
  # gearyVar against the exact variance over all 5040 permutations of 7 values on an asymmetric W
  perms <- function(v) { if (length(v) <= 1) return(matrix(v, 1, 1)); do.call(rbind, lapply(seq_along(v), function(i) cbind(v[i], perms(v[-i])))) }
  x7 <- d$x1[1:7]; W7 <- .sp_style(.sp_knn_b(xy[1:7, ], 2), "W"); P7 <- perms(1:7)
  C7 <- apply(P7, 1, function(o) gearyC(x7[o], W7, 7))
  stopifnot(abs(mean(C7) - 1) < 1e-10, abs(mean((C7 - mean(C7))^2) - gearyVar(x7, W7, 7)) < 1e-10)
  # styles on a distance band (0, 0.2]: row sums vary, so W, C, U, minmax and S all differ
  Bb <- (D < 0.2) * 1; diag(Bb) <- 0
  stopifnot(all(rowSums(Bb) > 0), isSymmetric(Bb))
  v <- list(band_links = sum(Bb), band_row_sums = vec(rowSums(Bb)), band_col_sum_max = max(colSums(Bb)))
  for (s in c("W", "B", "C", "U", "minmax", "S")) { Ws <- .sp_style(Bb, s)
    v[[paste0("style_", s, "_row1")]] <- vec(Ws[1, Ws[1, ] != 0]); v[[paste0("style_", s, "_sum")]] <- sum(Ws)
    v[[paste0("style_", s, "_row_sums")]] <- vec(rowSums(Ws))
    v[[paste0("style_", s, "_moran_I")]] <- unname(mor(d$x1, Ws, n)["I"]) }
  # minmax on an asymmetric list where the column maximum is smaller: 0 -> {1,2,3}, 1 -> {2}, 2 -> {3}, 3 -> {1}
  Ba <- matrix(0, 4, 4); Ba[1, 2:4] <- 1; Ba[2, 3] <- 1; Ba[3, 4] <- 1; Ba[4, 2] <- 1
  v$asym_minmax_weight <- unique(.sp_style(Ba, "minmax")[Ba > 0]); v$asym_minmax_sum <- sum(.sp_style(Ba, "minmax"))
  # Moran's I and Geary's C on the 4-NN row-standardised W
  W <- .sp_style(.sp_knn_b(xy, 4), "W")
  v$moran <- named(mor(d$x1, W, n)); v$geary_C <- gearyC(d$x1, W, n)
  v$geary_var_rand <- gearyVar(d$x1, W, n); gzp <- geary_z_p(v$geary_C, v$geary_var_rand)
  v$geary_z <- unname(gzp["z"]); v$geary_p <- unname(gzp["p"])
  # islands: a distance band (0, 0.12] leaves 4 units without neighbours (zero.policy = TRUE, adjust.n = TRUE)
  Bd <- (D < 0.12) * 1; diag(Bd) <- 0
  isl <- rowSums(Bd) == 0; Wd <- .sp_style(Bd, "W")
  v$n_islands <- sum(isl); v$islands <- vec(which(isl) - 1)            # 0-based ids
  v$moran_islands <- named(mor(d$x1, Wd, n - sum(isl))); v$geary_C_islands <- gearyC(d$x1, Wd, n - sum(isl))
  v$geary_var_rand_islands <- gearyVar(d$x1, Wd, n - sum(isl)); gzp <- geary_z_p(v$geary_C_islands, v$geary_var_rand_islands)
  v$geary_z_islands <- unname(gzp["z"]); v$geary_p_islands <- unname(gzp["p"])
  v$islands_style_C_sum <- sum(.sp_style(Bd, "C")); v$islands_style_S_sum <- sum(.sp_style(Bd, "S")); v$islands_style_U_sum <- sum(.sp_style(Bd, "U"))
  # k-NN ties on a 5 x 5 integer grid: brute force with R's stable order (ties by the lower index), which is
  # spdep::knearneigh(x, 4, use_kd_tree = FALSE); spdep's default kd-tree search (dbscan) can order exact ties differently
  gxy <- as.matrix(expand.grid(0:4, 0:4)); Dg <- as.matrix(dist(gxy)); diag(Dg) <- Inf
  v$knn_grid_ties <- vec(as.vector(t(apply(Dg, 1, function(r) sort(order(r)[1:4]))) - 1))
  list(values = v, reference = "spdep::nb2listw(style, zero.policy = TRUE), spdep::moran.test(randomisation = TRUE, alternative = 'greater', adjust.n = TRUE) and spdep::geary() formulas, written out (spdep not installed)",
       call = "nb2listw(dnearneigh(coords, 0, 0.2), style); moran.test(x1, nb2listw(knn2nb(knearneigh(coords, 4))))", packages = character(0))
})
