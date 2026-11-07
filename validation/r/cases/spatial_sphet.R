# validation/r/cases/spatial_sphet.R -- spatial GMM references (SPAT-20, SPAT-27, SPAT-7, SPAT-8,
# SPAT-9). sphet and spdep are not installed: the Kelejian-Prucha estimators are
# written out exactly as p2a implements them:
#  * W: 5 nearest neighbours on the coordinates, row-standardised (asymmetric);
#  * instruments H = [1, X, W X, W^2 X] over the NON-CONSTANT columns of X only (W 1 = 1 for a
#    row-standardised W, so W 1 is never added: SPAT-7; sphet::stslshac lags x[, -1] the same way);
#  * SAR by spatial 2SLS (KP 1998 step 1); SEs classical s2 (Zh'Zh)^-1 with s2 = u'u / (n - k),
#    heteroskedasticity-robust (Zh'Zh)^-1 sum u_i^2 zh_i zh_i' (Zh'Zh)^-1, and spatial HAC
#    (Kelejian & Prucha 2007) (Zh'Zh)^-1 Zh' Omega Zh (Zh'Zh)^-1, Omega_ij = K(d_ij / d) u_i u_j with
#    a fixed distance bandwidth d and the triangular K(z) = 1 - z or Epanechnikov K(z) = 1 - z^2
#    (|z| < 1; sphet's kernelsfun.R), Zh = P_H Z (the formula of sphet's tsls(HAC = TRUE)); the sphet
#    sources consulted are CRAN's sphet 2.1-1 (R/kernelsfun.R, R/twostagels.R, R/gm_functions.R);
#  * SEM: the GM estimator of rho on OLS residuals -- homoskedastic KP (1999), three moments, least
#    squares in (rho, rho^2, sigma^2) with rho^2 tied to rho; heteroskedastic KP (2010), two moments
#    m1 = eb'eb - e'De, m2 = eb'e with D = diag(W'W), i.e. D_ii = sum_j w_ji^2 (the column sums of
#    W^2, as sphet's gg_het) -- then OLS on (I - rho W) y, (I - rho W) X (feasible GLS); SEs
#    classical (s2 = e'e/(n - k)), White and spatial HAC on the transformed data;
#  * SARAR: KP (1998) GS2SLS: 2SLS, GM rho on its residuals, 2SLS of (I - rho W) y on (I - rho W) Z
#    with the same H;
#  * the GM objectives are quartics in rho (sigma^2 enters linearly and is concentrated out), so the
#    reference minimiser is EXACT: the real roots of the derivative cubic in [-0.99, 0.99] (polyroot,
#    Newton-polished) and the two bounds, whichever gives the smallest objective; optim / optimize are
#    run as cross-checks (stopifnot);
#  * SEM SE of rho, exog_first_step = TRUE: a_r = 0, the value of alpha_r under exogenous X (KP 2010, Theorem 2);
#    sphet's Psi plugs in the sample alpha_r, which is O_p(n^-1/2), so the two differ by O(1/n) relative;
#  * SE of rho: the GMM sandwich (J'J)^-1 J' Psi J (J'J)^-1 / n of the identity-weighted GM estimator,
#    J = d m / d theta, with Psi the variance of the moments of KP (2010, Theorem 2) as sphet's
#    psirhorho_het(step1.c = TRUE) computes it (het), and its homoskedastic analogue with the mu3/mu4
#    terms of sphet's psirhorho_hom for the three KP (1999) moments (A1 = I, A2 = W'W, A3 = W); the
#    first-step estimator enters through a_r = (I - rho W')^-1 T alpha_r, T = n Zh (Zh'Zh)^-1,
#    alpha_r = -n^-1 Z'(I - rho W')(A_r + A_r') eps.

.knn_w <- function(xy, k) { D <- as.matrix(dist(xy)); diag(D) <- Inf; B <- matrix(0, nrow(xy), nrow(xy))
  for (i in seq_len(nrow(xy))) B[i, order(D[i, ])[1:k]] <- 1; B / rowSums(B) }

.sphet_data <- function(ctx) {
  set.seed(20261208)
  n <- 150
  xy <- cbind(runif(n), runif(n)); x1 <- rnorm(n); x2 <- rnorm(n)
  W <- .knn_w(xy, 5); X <- cbind(1, x1, x2); b <- c(1, 0.8, -0.5)
  het <- 0.5 + 0.5 * abs(x1)
  v <- rnorm(n) * het
  e_sp <- solve(diag(n) - 0.3 * W, v)                                 # spatially correlated, heteroskedastic
  y_sar <- solve(diag(n) - 0.4 * W, X %*% b + e_sp)
  y_sem <- X %*% b + solve(diag(n) - 0.5 * W, rnorm(n) * het)
  y_sarar <- solve(diag(n) - 0.3 * W, X %*% b + solve(diag(n) - 0.4 * W, rnorm(n)))
  ctx$gen_dataset("sphet_sim", data.frame(id = 1:n, cx = round(xy[, 1], 10), cy = round(xy[, 2], 10), x1 = round(x1, 10), x2 = round(x2, 10),
                                          y_sar = round(drop(y_sar), 10), y_sem = round(drop(y_sem), 10), y_sarar = round(drop(y_sarar), 10)))
}

.tsls <- function(y, Z, H) {
  P <- H %*% solve(crossprod(H), crossprod(H, Z))
  d <- solve(crossprod(P, Z), crossprod(P, y))
  list(d = drop(d), P = P, u = drop(y - Z %*% d))
}
.se_classical <- function(P, u, k) sqrt(diag(sum(u^2) / (length(u) - k) * solve(crossprod(P))))
.se_white <- function(P, u) { A <- solve(crossprod(P)); sqrt(diag(A %*% crossprod(P * u) %*% A)) }
# sphet's distance kernels (kernelsfun.R), z = d_ij / bandwidth, 0 for z >= 1. Quadratic spectral is 1 at z = 0, its
# limit (sphet's formula is NaN there; only coincident points reach it, the diagonal being the White term).
.hac_kernel <- function(z, kernel) {
  k <- switch(kernel,
    triangular = 1 - z, rectangular = rep(1, length(z)), epanechnikov = 1 - z^2, bisquare = (1 - z^2)^2,
    parzen = ifelse(z <= 0.5, 1 - 6 * z^2 + 6 * z^3, 2 * (1 - z)^3),
    tukey_hanning = (1 + cos(pi * z)) / 2,
    quadratic_spectral = { a <- 6 * pi * z / 5; ifelse(z == 0, 1, 25 / (12 * pi^2 * z^2) * (sin(a) / a - cos(a))) })
  k[z >= 1] <- 0
  k
}
.se_hac <- function(P, u, D, d, kernel) {
  K <- matrix(.hac_kernel(as.vector(D / d), kernel), nrow(D), ncol(D))
  A <- solve(crossprod(P)); Om <- K * outer(u, u)
  sqrt(diag(A %*% t(P) %*% Om %*% P %*% A))
}
# Variable bandwidth: unit i's bandwidth is its largest distance to its W-neighbours (W's neighbour sets play the
# role of sphet's distance object); only those pairs carry a weight; K is symmetrised, which leaves diag(V) as it is.
.se_hac_variable <- function(P, u, D, W, kernel) {
  n <- nrow(D); K <- diag(n)
  for (i in seq_len(n)) { nb <- which(W[i, ] > 0); K[i, nb] <- .hac_kernel(D[i, nb] / max(D[i, nb]), kernel) }
  K <- (K + t(K)) / 2
  A <- solve(crossprod(P)); Om <- K * outer(u, u)
  sqrt(diag(A %*% t(P) %*% Om %*% P %*% A))
}

# Exact minimiser of f(r) = || a - b r - c r^2 ||^2 on [lo, hi]: the real roots of f'(r) / 2 =
# -b'a + (b'b - 2 c'a) r + 3 b'c r^2 + 2 c'c r^3 inside the interval, and the two bounds.
.sphet_quartic_min <- function(a, b, c, lo = -0.99, hi = 0.99) {
  f <- function(r) sum((a - b * r - c * r^2)^2)
  d <- c(-sum(b * a), sum(b * b) - 2 * sum(c * a), 3 * sum(b * c), 2 * sum(c * c))
  dd <- function(r) d[1] + d[2] * r + d[3] * r^2 + d[4] * r^3
  ddd <- function(r) d[2] + 2 * d[3] * r + 3 * d[4] * r^2
  z <- polyroot(d); r <- Re(z[abs(Im(z)) < 1e-6])
  r <- vapply(r, function(x) { for (i in 1:20) { s <- dd(x) / ddd(x); if (!is.finite(s)) break; x <- x - s }; x }, 0)
  cand <- c(r[r > lo & r < hi], lo, hi)
  fv <- vapply(cand, f, 0); i <- which.min(fv)
  list(rho = cand[i], objective = fv[i], boundary = cand[i] %in% c(lo, hi))
}

# Homoskedastic KP (1999): moments m = g - G (rho, rho^2, sigma^2)'.
.sphet_gm_parts_hom <- function(u, W) {
  n <- length(u); ub <- drop(W %*% u); ubb <- drop(W %*% ub); trWW <- sum(W * W)
  G <- rbind(c(2 * sum(u * ub), -sum(ub * ub), n), c(2 * sum(ubb * ub), -sum(ubb * ubb), trWW),
             c(sum(u * ubb) + sum(ub * ub), -sum(ub * ubb), 0)) / n
  g <- c(sum(u * u), sum(ub * ub), sum(u * ub)) / n
  list(G = G, g = g)
}
.gm_rho <- function(u, W) {
  p <- .sphet_gm_parts_hom(u, W); G <- p$G; g <- p$g; n <- length(u)
  G3 <- G[, 3]; M <- diag(3) - tcrossprod(G3) / sum(G3^2)            # sigma^2 concentrated out
  q <- .sphet_quartic_min(drop(M %*% g), drop(M %*% G[, 1]), drop(M %*% G[, 2]))
  sigma2 <- sum(G3 * (g - G[, 1] * q$rho - G[, 2] * q$rho^2)) / sum(G3^2)
  # cross-check: the unconstrained 2-D least squares of the design (optim) finds the same interior minimum
  f <- function(p) { a <- c(p[1], p[1]^2, p[2]); sum((G %*% a - g)^2) }
  o <- optim(c(0, sum(u^2) / n), f, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  o <- optim(o$par, f, method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 20000))
  o <- optim(o$par, f, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  stopifnot(!q$boundary, abs(o$par[1] - q$rho) < 1e-7, abs(o$par[2] - sigma2) < 1e-7)
  list(rho = q$rho, sigma2 = sigma2, objective = f(c(q$rho, sigma2)), G = G)
}
# Heteroskedastic KP (2010): m1 = eb'eb - e'De, m2 = eb'e (per n), D = diag(W'W) (sphet's gg_het).
.sphet_gm_parts_het <- function(u, W) {
  n <- length(u); ub <- drop(W %*% u); ubb <- drop(W %*% ub); Dg <- colSums(W^2)
  G <- rbind(c(2 * (sum(ubb * ub) - sum(Dg * u * ub)), -(sum(ubb * ubb) - sum(Dg * ub * ub))),
             c(sum(u * ubb) + sum(ub * ub), -sum(ub * ubb))) / n
  g <- c(sum(ub * ub) - sum(Dg * u * u), sum(u * ub)) / n
  list(G = G, g = g)
}
.gm_rho_het <- function(u, W) {
  p <- .sphet_gm_parts_het(u, W); G <- p$G; g <- p$g
  q <- .sphet_quartic_min(g, G[, 1], G[, 2])
  ub <- drop(W %*% u); ubb <- drop(W %*% ub); Dg <- colSums(W^2)
  f <- function(r) { e <- u - r * ub; eb <- ub - r * ubb; m1 <- sum(eb^2) - sum(Dg * e^2); m2 <- sum(eb * e); (m1^2 + m2^2) / length(u)^2 }
  o <- optimize(f, c(-0.99, 0.99), tol = 1e-12)
  stopifnot(!q$boundary, abs(o$minimum - q$rho) < 1e-7, abs(f(q$rho) - q$objective) < 1e-14)
  list(rho = q$rho, objective = q$objective, G = G)
}

# SE of the GM rho. u: first-step residuals; Z, H: first-step regressors and instruments (H = Z = X for the
# SEM's OLS); Zh = P_H Z.
.sphet_rho_se <- function(rho, u, W, Z, H, gm, het, exog_first_step = FALSE) {
  n <- length(u); I <- diag(n)
  eps <- drop(u - rho * W %*% u)
  Zh <- H %*% solve(crossprod(H), crossprod(H, Z))
  Tm <- n * Zh %*% solve(crossprod(Zh))                        # T = H P
  Zt <- (I - rho * W) %*% Z                                      # Z~ = (I - rho W) Z
  IrWti <- solve(I - rho * t(W))                                 # (I - rho W')^-1 (step-1 residuals)
  WW <- crossprod(W)
  A <- if (het) { A1 <- WW; diag(A1) <- 0; list(A1, W) } else list(I, WW, W)
  B <- lapply(A, function(a) a + t(a))
  avec <- lapply(B, function(b) if (exog_first_step) rep(0, n) else drop(IrWti %*% Tm %*% (-crossprod(Zt, b %*% eps) / n)))
  m <- length(A); Psi <- matrix(0, m, m)
  if (het) {
    ga <- eps^2
    for (r in 1:m) for (s in 1:m) Psi[r, s] <- (sum(B[[r]] * B[[s]] * outer(ga, ga)) / 2 + sum(avec[[r]] * ga * avec[[s]])) / n
  } else {
    s2 <- sum(eps^2) / n; mu3 <- sum(eps^3) / n; mu4 <- sum(eps^4) / n
    dA <- lapply(A, diag)
    for (r in 1:m) for (s in 1:m) Psi[r, s] <- (s2^2 * sum(B[[r]] * B[[s]]) / 2 + s2 * sum(avec[[r]] * avec[[s]]) +
        (mu4 - 3 * s2^2) * sum(dA[[r]] * dA[[s]]) + mu3 * (sum(avec[[r]] * dA[[s]]) + sum(avec[[s]] * dA[[r]]))) / n
  }
  G <- gm$G
  J <- if (het) cbind(G[, 1] + 2 * rho * G[, 2]) else cbind(G[, 1] + 2 * rho * G[, 2], G[, 3])
  JJi <- solve(crossprod(J))
  V <- JJi %*% t(J) %*% Psi %*% J %*% JJi / n
  sqrt(V[1, 1])
}

case("sphet_gmm_hac", "econometrics/spatial", "Kelejian-Prucha spatial 2SLS / GM / GS2SLS with classical, White and spatial-HAC SEs (written out; sphet not installed)", function(ctx) {
  d <- .sphet_data(ctx); n <- nrow(d)
  xy <- as.matrix(d[, c("cx", "cy")]); W <- .knn_w(xy, 5); D <- as.matrix(dist(xy))
  X <- cbind(1, d$x1, d$x2); Xn <- X[, -1]
  H <- cbind(X, W %*% Xn, W %*% W %*% Xn)
  v <- list(h_rank = qr(H)$rank, h_cols = ncol(H))
  # the instrument set before SPAT-7 lagged the constant as well: 9 columns of rank 7
  H_old <- cbind(X, W %*% X, W %*% W %*% X); v$h_old_rank <- qr(H_old)$rank; v$h_old_cols <- ncol(H_old)
  # SAR
  Z <- cbind(X, W %*% d$y_sar); s <- .tsls(d$y_sar, Z, H)
  v$sar_coef <- vec(s$d); v$sar_se_classical <- vec(.se_classical(s$P, s$u, ncol(Z))); v$sar_se_white <- vec(.se_white(s$P, s$u))
  v$sar_p_classical <- vec(2 * pnorm(-abs(s$d / .se_classical(s$P, s$u, ncol(Z)))))   # upper tail, not 1 - pnorm (x1: 1.7e-22)
  v$sar_se_hac_tri_015 <- vec(.se_hac(s$P, s$u, D, 0.15, "triangular")); v$sar_se_hac_epa_015 <- vec(.se_hac(s$P, s$u, D, 0.15, "epanechnikov"))
  v$sar_se_hac_tri_030 <- vec(.se_hac(s$P, s$u, D, 0.30, "triangular"))
  for (kn in list(c("rect", "rectangular"), c("bisq", "bisquare"), c("parzen", "parzen"), c("th", "tukey_hanning"), c("qs", "quadratic_spectral")))
    v[[paste0("sar_se_hac_", kn[1], "_015")]] <- vec(.se_hac(s$P, s$u, D, 0.15, kn[2]))
  v$sar_se_hac_epa_var <- vec(.se_hac_variable(s$P, s$u, D, W, "epanechnikov"))
  # a bandwidth below the smallest pairwise distance keeps only the diagonal: HAC = White
  v$min_pair_distance <- min(D[upper.tri(D)])
  stopifnot(max(abs(.se_hac(s$P, s$u, D, 0.5 * v$min_pair_distance, "triangular") - .se_white(s$P, s$u))) < 1e-14)
  # the row order must not matter for the HAC SEs (SPAT-20): recompute on a permutation
  set.seed(1); pm <- sample.int(n); Wp <- W[pm, pm]; Xp <- X[pm, ]; Hp <- cbind(Xp, Wp %*% Xp[, -1], Wp %*% Wp %*% Xp[, -1])
  sp <- .tsls(d$y_sar[pm], cbind(Xp, Wp %*% d$y_sar[pm]), Hp)
  stopifnot(max(abs(.se_hac(sp$P, sp$u, D[pm, pm], 0.15, "triangular") - .se_hac(s$P, s$u, D, 0.15, "triangular"))) < 1e-12)
  stopifnot(max(abs(.se_hac_variable(sp$P, sp$u, D[pm, pm], Wp, "epanechnikov") - .se_hac_variable(s$P, s$u, D, W, "epanechnikov"))) < 1e-12)
  # SEM: OLS residuals -> GM rho (homoskedastic KP 1999 and heteroskedastic KP 2010) -> FGLS
  for (tag in c("hom", "het")) {
    u0 <- drop(d$y_sem - X %*% qr.coef(qr(X), d$y_sem))
    g <- if (tag == "hom") .gm_rho(u0, W) else .gm_rho_het(u0, W)
    ys <- d$y_sem - g$rho * drop(W %*% d$y_sem); Xs <- X - g$rho * (W %*% X)
    b <- qr.coef(qr(Xs), ys); e <- drop(ys - Xs %*% b)
    v[[paste0("sem_", tag, "_rho")]] <- g$rho; v[[paste0("sem_", tag, "_gm_objective")]] <- g$objective
    if (tag == "hom") v$sem_hom_gm_sigma2 <- g$sigma2
    v[[paste0("sem_", tag, "_rho_se")]] <- .sphet_rho_se(g$rho, u0, W, X, X, g, tag == "het", exog_first_step = TRUE)
    v[[paste0("sem_", tag, "_coef")]] <- vec(b)
    v[[paste0("sem_", tag, "_se_classical")]] <- vec(sqrt(diag(sum(e^2) / (n - ncol(X)) * solve(crossprod(Xs)))))
    A <- solve(crossprod(Xs)); v[[paste0("sem_", tag, "_se_white")]] <- vec(sqrt(diag(A %*% crossprod(Xs * e) %*% A)))
    v[[paste0("sem_", tag, "_se_hac_tri_015")]] <- vec(.se_hac(Xs, e, D, 0.15, "triangular"))
  }
  # SARAR GS2SLS (homoskedastic GM, then the heteroskedastic GM)
  Z <- cbind(X, W %*% d$y_sarar); s1 <- .tsls(d$y_sarar, Z, H)
  v$sarar_step1_coef <- vec(s1$d)
  for (tag in c("hom", "het")) {
    g <- if (tag == "hom") .gm_rho(s1$u, W) else .gm_rho_het(s1$u, W)
    ys <- d$y_sarar - g$rho * drop(W %*% d$y_sarar); Zs <- Z - g$rho * (W %*% Z)
    s2 <- .tsls(ys, Zs, H)
    p <- if (tag == "hom") "sarar_" else "sarar_het_"
    v[[paste0(p, "rho")]] <- g$rho; v[[paste0(p, "coef")]] <- vec(s2$d)
    v[[paste0(p, "rho_se")]] <- .sphet_rho_se(g$rho, s1$u, W, Z, H, g, tag == "het")
    v[[paste0(p, "se_classical")]] <- vec(.se_classical(s2$P, s2$u, ncol(Z))); v[[paste0(p, "se_white")]] <- vec(.se_white(s2$P, s2$u))
    v[[paste0(p, "se_hac_tri_015")]] <- vec(.se_hac(s2$P, s2$u, D, 0.15, "triangular"))
  }
  list(values = v, reference = "Kelejian & Prucha (1998, 1999, 2007, 2010) estimators written out (sphet/spdep not installed; kernels, heteroskedastic moments and Psi as sphet 2.1-1's kernelsfun.R, gg_het and psirhorho_het); coefficient order: (Intercept), x1, x2, then lambda (Wy) for SAR/SARAR",
       call = "S2SLS / GM / GS2SLS on sphet_sim with a row-standardised 5-NN W", packages = character(0))
})
