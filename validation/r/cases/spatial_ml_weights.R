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
