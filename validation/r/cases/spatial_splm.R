# validation/r/cases/spatial_splm.R -- spatial panel references (T3-WP34: SPAT-10, SPAT-12; moved in SPAT-13).
# splm is not installed: the estimators are written out from the published likelihoods and moment conditions
# (Elhorst 2003; Baltagi, Song & Koh 2003; Kapoor, Kelejian & Prucha 2007; Millo & Piras 2012; Baltagi 1981).
# R packages anchor the non-spatial special cases: plm::plm(model = "within", effect = ...) for the within
# coefficients and nlme::lme(method = "ML") for the random-effects ML fit.
# Balanced panel, N = 49 units, T = 6; W = 4 nearest neighbours, row-standardised (10 complex eigenvalues, so
# log|I - r W| = Re(sum(log(1 - r ev))) over the complex spectrum); stacked by time, units in id order within
# each period (W_NT = I_T (x) W). Parameter order of every *_se vector: beta, rho (lag), phi (random effects),
# lambda (error), sigma2.
#  * spml (ML), within (entity, time or two-way demeaning of y and X, then W applied to the demeaned data) and
#    pooled:  l = -NT/2 log(2 pi s2) + T log|A| + T log|B| - e'e / (2 s2),  e = B_NT (A_NT y - X b),
#    A = I - rho W (lag), B = I - lambda W (error); b and s2 = e'e/(NT) concentrated out; rho/lambda by
#    optimize / optim, then Newton steps on the analytic profile gradient (score below) to |grad| < 1e-9.
#  * spml random effects (individual), u = A y - X b = iota_T (x) mu + eps, c = T phi, phi = s2_mu / s2:
#      Omega / s2 = Jbar_T (x) M^-1 + E_T (x) K^-1,  K = B'B (K = I without spatial error),
#      'b'   (Baltagi et al. 2003, mu not spatially correlated): M = (c I + K^-1)^-1, ldet = log|cK + I| - T log|K|
#      'kkp' (KKP 2007, mu spatially correlated):               M = K / (c + 1),     ldet = N log(c + 1) - T log|K|
#      'none':                                                   M = I / (c + 1),     ldet = N log(c + 1)
#    l = -NT/2 log(2 pi s2) + T log|A| - ldet/2 - (T ubar'M ubar + sum_t (u_t - ubar)'K(u_t - ubar)) / (2 s2).
#  * SEs: inverse of minus the observed information, computed as the Richardson Jacobian (numDeriv::jacobian)
#    of the analytic score, which is checked against numDeriv::grad of the log-likelihood. (numDeriv::hessian
#    of the log-likelihood itself is noise-dominated for parameters near zero, because its step is relative to
#    |x|: the pooled intercept SE came out 1.8% off.)
#  * spgm (Kapoor, Kelejian & Prucha 2007 and Baltagi's 2SLS family), lag model y = rho W y + X b + u:
#    w2sls (within 2SLS, H = [X~, W X~, W^2 X~]), b2sls (between 2SLS on unit means with an intercept,
#    H = [1, Xbar, W Xbar, W^2 Xbar]), ec2sls (Baltagi 1981: 2SLS of Omega^-1/2 y on Omega^-1/2 Z with
#    instruments [Q0 H, P0 H]) and g2sls (2SLS with instruments Omega^-1/2 H), with
#    sigma_nu^2 = u_w'u_w / (N (T - 1)) from w2sls and sigma_1^2 = T u_b'u_b / N from b2sls; Sargan
#    u'P_H u / (u'u / m) (m = N (T - 1) within, N between; the transformed models have unit variance, so
#    u*'P_H u*); within error model by the KKP within GM moments (KP 1999 moments on within residuals with
#    n = N (T - 1) and the trace term (T - 1) tr(W'W)); sigma2 concentrated out, the GM rho is the exact root
#    of the cubic first-order condition (checked against optim), then within FGLS; within GS2SLS (lag + error):
#    w2sls, the within GM on its residuals, then 2SLS of the spatially filtered model with the same H.
suppressPackageStartupMessages({ library(numDeriv); library(plm); library(nlme) })

.knn_w4 <- function(xy) { D <- as.matrix(dist(xy)); diag(D) <- Inf; B <- matrix(0, nrow(xy), nrow(xy))
  for (i in seq_len(nrow(xy))) B[i, order(D[i, ])[1:4]] <- 1; B / rowSums(B) }

.splm_data <- function(ctx) {
  set.seed(20261209)
  N <- 49; TT <- 6
  xy <- cbind(runif(N), runif(N)); W <- .knn_w4(xy)
  mu <- rnorm(N, 0, 0.8); b <- c(0.8, -0.5)
  rows <- list()
  for (t in 1:TT) {
    x1 <- rnorm(N) + 0.3 * mu; x2 <- rnorm(N)
    Xb <- cbind(x1, x2) %*% b
    y_lag <- solve(diag(N) - 0.4 * W, Xb + mu + rnorm(N, 0, 0.7))
    y_err <- Xb + mu + solve(diag(N) - 0.5 * W, rnorm(N, 0, 0.7))
    y_re <- 1 + Xb + 0.9 * mu + solve(diag(N) - 0.4 * W, rnorm(N, 0, 0.7))
    rows[[t]] <- data.frame(id = 1:N, time = t, cx = round(xy[, 1], 10), cy = round(xy[, 2], 10), x1 = round(x1, 10), x2 = round(x2, 10),
                            y_lag = round(drop(y_lag), 10), y_err = round(drop(y_err), 10), y_re = round(drop(y_re), 10))
  }
  d <- do.call(rbind, rows)
  ctx$gen_dataset("splm_sim", d[sample.int(nrow(d)), ])          # shuffled rows: p2a must sort by (time, id)
}

.logdet <- function(r, ev) Re(sum(log(1 - r * ev)))
.dlogdet <- function(r, ev) -Re(sum(ev / (1 - r * ev)))
.within <- function(v, id, time, effect) {
  switch(effect, individual = v - ave(v, id), time = v - ave(v, time), twoways = v - ave(v, id) - ave(v, time) + mean(v), pooling = v)
}
.jac_sym <- function(f, x) { H <- numDeriv::jacobian(f, x); (H + t(H)) / 2 }
# Newton polish of a profile log-likelihood from the optim/optimize point, on its analytic gradient.
.polish <- function(psi, grad) {
  for (i in 1:50) { g <- grad(psi); if (max(abs(g)) < 1e-9) break; psi <- psi - solve(.jac_sym(grad, psi), g) }
  stopifnot(max(abs(grad(psi))) < 1e-9); psi
}
.check_score <- function(score, loglik, th) stopifnot(max(abs(score(th) - numDeriv::grad(loglik, th))) < 1e-5)

.spml_fe <- function(y, X, W, N, TT, lag, err) {
  ev <- eigen(W, only.values = TRUE)$values; WN <- kronecker(diag(TT), W); n <- N * TT; k <- ncol(X)
  lo <- 1 / min(Re(ev)) + 1e-6; hi <- 1 / max(Re(ev)) - 1e-6
  Wy <- drop(WN %*% y); WX <- WN %*% X
  core <- function(r, l) { Ay <- y - r * Wy; By <- Ay - l * drop(WN %*% Ay); BX <- X - l * WX
    b <- qr.coef(qr(BX), By); e <- drop(By - BX %*% b); list(b = b, e = e, s2 = sum(e^2) / n) }
  ll <- function(r, l) { c <- core(r, l); -n / 2 * log(2 * pi * c$s2) - n / 2 + TT * .logdet(r, ev) + TT * .logdet(l, ev) }
  unpack <- function(th) list(b = th[1:k], r = if (lag) th[k + 1] else 0, l = if (err) th[k + lag + 1] else 0, s2 = th[length(th)])
  full <- function(th) { p <- unpack(th); r0 <- y - p$r * Wy - drop(X %*% p$b); e <- r0 - p$l * drop(WN %*% r0)
    -n / 2 * log(2 * pi * p$s2) + TT * .logdet(p$r, ev) + TT * .logdet(p$l, ev) - sum(e^2) / (2 * p$s2) }
  score <- function(th) { p <- unpack(th); r0 <- y - p$r * Wy - drop(X %*% p$b); Wr <- drop(WN %*% r0); e <- r0 - p$l * Wr
    g <- drop(crossprod(X - p$l * WX, e)) / p$s2
    if (lag) g <- c(g, TT * .dlogdet(p$r, ev) + sum(e * (Wy - p$l * drop(WN %*% Wy))) / p$s2)
    if (err) g <- c(g, TT * .dlogdet(p$l, ev) + sum(e * Wr) / p$s2)
    c(g, -n / (2 * p$s2) + sum(e^2) / (2 * p$s2^2)) }
  th_of <- function(psi) { r <- if (lag) psi[1] else 0; l <- if (err) psi[length(psi)] else 0; c <- core(r, l); c(c$b, if (lag) r, if (err) l, c$s2) }
  pgrad <- function(psi) { s <- score(th_of(psi)); s[(k + 1):(k + lag + err)] }
  if (lag && !err) psi <- optimize(function(r) ll(r, 0), c(lo, hi), maximum = TRUE, tol = 1e-12)$maximum
  else if (err && !lag) psi <- optimize(function(l) ll(0, l), c(lo, hi), maximum = TRUE, tol = 1e-12)$maximum
  else if (lag && err) { o <- optim(c(0.2, 0.2), function(p) -ll(p[1], p[2]), method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
    psi <- optim(o$par, function(p) -ll(p[1], p[2]), method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 20000))$par }
  else psi <- numeric(0)
  if (length(psi)) { p0 <- psi; psi <- .polish(psi, pgrad); stopifnot(max(abs(psi - p0)) < 1e-4) }
  th <- th_of(psi); .check_score(score, full, th)
  r <- if (lag) psi[1] else 0; l <- if (err) psi[length(psi)] else 0
  list(b = th[1:k], rho = r, lambda = l, s2 = th[length(th)], se = sqrt(diag(solve(-.jac_sym(score, th)))), loglik = ll(r, l))
}

.spml_re <- function(y, X, W, N, TT, lag, type) {
  ev <- eigen(W, only.values = TRUE)$values; WN <- kronecker(diag(TT), W); n <- N * TT; k <- ncol(X); err <- type != "none"
  lo <- 1 / min(Re(ev)) + 1e-6; hi <- 1 / max(Re(ev)) - 1e-6
  Wy <- drop(WN %*% y); WW <- crossprod(W); Ws <- W + t(W)
  parts <- function(phi, l) { c_ <- TT * phi
    if (err) { e <- eigen(crossprod(diag(N) - l * W), symmetric = TRUE); V <- e$vectors; dk <- e$values } else { V <- diag(N); dk <- rep(1, N) }
    m <- switch(type, b = dk / (c_ * dk + 1), kkp = dk / (c_ + 1), none = rep(1 / (c_ + 1), N))
    ldet <- switch(type, b = sum(log(c_ * dk + 1)) - TT * sum(log(dk)), kkp = N * log(c_ + 1) - TT * sum(log(dk)), none = N * log(c_ + 1))
    list(c = c_, V = V, dk = dk, m = m, ldet = ldet) }
  mf <- function(P, f) P$V %*% (f * t(P$V))
  quad <- function(u, P) { U <- matrix(u, N, TT); ub <- rowMeans(U); a <- drop(crossprod(P$V, ub)); Et <- crossprod(P$V, U - ub)
    TT * sum(P$m * a^2) + sum(P$dk * Et^2) }
  apply_oi <- function(v, P) { U <- matrix(v, N, TT); ub <- rowMeans(U); as.vector(mf(P, P$dk) %*% (U - ub) + drop(mf(P, P$m) %*% ub)) }
  whiten <- function(v, P) { U <- matrix(v, N, TT); ub <- rowMeans(U); as.vector(mf(P, sqrt(P$dk)) %*% (U - ub) + drop(mf(P, sqrt(P$m)) %*% ub)) }
  unpack <- function(th) { i <- k + lag; list(b = th[1:k], r = if (lag) th[k + 1] else 0, phi = th[i + 1], l = if (err) th[i + 2] else 0, s2 = th[length(th)]) }
  full <- function(th) { p <- unpack(th); P <- parts(p$phi, p$l); u <- y - p$r * Wy - drop(X %*% p$b)
    -n / 2 * log(2 * pi * p$s2) + TT * .logdet(p$r, ev) - 0.5 * P$ldet - quad(u, P) / (2 * p$s2) }
  score <- function(th) { p <- unpack(th); P <- parts(p$phi, p$l); u <- y - p$r * Wy - drop(X %*% p$b); s2 <- p$s2; c_ <- P$c
    Oiu <- apply_oi(u, P); U <- matrix(u, N, TT); ub <- rowMeans(U); E <- U - ub; a <- drop(crossprod(P$V, ub))
    g <- drop(crossprod(X, Oiu)) / s2
    if (lag) g <- c(g, TT * .dlogdet(p$r, ev) + sum(Wy * Oiu) / s2)
    dldet_dc <- switch(type, b = sum(P$dk / (c_ * P$dk + 1)), N / (c_ + 1))
    dQ_dc <- switch(type, b = -TT * sum(P$m^2 * a^2), kkp = -TT * sum(P$dk * a^2) / (c_ + 1)^2, none = -TT * sum(ub^2) / (c_ + 1)^2)
    g <- c(g, TT * (-0.5 * dldet_dc - dQ_dc / (2 * s2)))
    if (err) { Kp <- -Ws + 2 * p$l * WW; Kt <- crossprod(P$V, Kp %*% P$V); dKp <- diag(Kt)
      dldet <- switch(type, b = c_ * sum(dKp / (c_ * P$dk + 1)) - TT * sum(dKp / P$dk), kkp = -TT * sum(dKp / P$dk))
      ga <- a / (c_ * P$dk + 1)
      dQ <- sum(E * (Kp %*% E)) + TT * switch(type, b = drop(t(ga) %*% Kt %*% ga), kkp = drop(t(a) %*% Kt %*% a) / (c_ + 1))
      g <- c(g, -0.5 * dldet - dQ / (2 * s2)) }
    c(g, -n / (2 * s2) + quad(u, P) / (2 * s2^2)) }
  # psi = (rho?, log phi, lambda?)
  th_of <- function(psi) { r <- if (lag) psi[1] else 0; phi <- exp(psi[lag + 1]); l <- if (err) psi[lag + 2] else 0
    P <- parts(phi, l); Ay <- y - r * Wy; b <- qr.coef(qr(apply(X, 2, whiten, P = P)), whiten(Ay, P))
    u <- Ay - drop(X %*% b); c(b, if (lag) r, phi, if (err) l, quad(u, P) / n) }
  prof <- function(psi) { th <- th_of(psi); p <- unpack(th); P <- parts(p$phi, p$l); -n / 2 * log(2 * pi * p$s2) + TT * .logdet(p$r, ev) - 0.5 * P$ldet - n / 2 }
  pgrad <- function(psi) { th <- th_of(psi); s <- score(th)[(k + 1):(k + lag + 1 + err)]; s[lag + 1] <- s[lag + 1] * th[k + lag + 1]; s }
  start <- c(if (lag) 0, 0, if (err) 0.2)
  pen <- function(p) { if (lag && (p[1] <= lo || p[1] >= hi)) return(1e300); if (err && (p[lag + 2] <= lo || p[lag + 2] >= hi)) return(1e300); -prof(p) }
  if (length(start) == 1) psi <- optimize(function(p) -prof(p), c(-20, 20), tol = 1e-12)$minimum
  else { o <- optim(start, pen, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
    psi <- optim(o$par, pen, method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 20000))$par }
  p0 <- psi; psi <- .polish(psi, pgrad); stopifnot(max(abs(psi - p0)) < 1e-4)
  th <- th_of(psi); .check_score(score, full, th); p <- unpack(th)
  list(b = p$b, rho = p$r, phi = p$phi, lambda = p$l, s2 = p$s2, sigma2_mu = p$phi * p$s2, se = sqrt(diag(solve(-.jac_sym(score, th)))), loglik = full(th))
}

.tsls2 <- function(y, Z, H) { P <- H %*% solve(crossprod(H), crossprod(H, Z)); d <- solve(crossprod(P, Z), crossprod(P, y)); list(d = drop(d), P = P, u = drop(y - Z %*% d)) }
.quad_PH <- function(H, v) sum(v * drop(H %*% solve(crossprod(H), crossprod(H, v))))

# KKP within GM on within residuals u (KP 1999 moments, n = N (T - 1), trace term (T - 1) tr(W'W)); returns the
# GM rho (spatial-error coefficient) and sigma2.
.kkp_within_gm <- function(u, WN, W, N, TT) {
  nn <- N * (TT - 1)
  ub <- drop(WN %*% u); ubb <- drop(WN %*% ub); trWW <- (TT - 1) * sum(W * W)
  G <- rbind(c(2 * sum(u * ub), -sum(ub * ub), nn), c(2 * sum(ubb * ub), -sum(ubb * ubb), trWW), c(sum(u * ubb) + sum(ub * ub), -sum(ub * ubb), 0)) / nn
  g <- c(sum(u * u), sum(ub * ub), sum(u * ub)) / nn
  fo <- function(p) sum((G %*% c(p[1], p[1]^2, p[2]) - g)^2)
  o <- optim(c(0, sum(u^2) / nn), fo, method = "BFGS", control = list(reltol = 1e-15, maxit = 10000))
  o <- optim(o$par, fo, method = "Nelder-Mead", control = list(reltol = 1e-15, maxit = 20000))
  # sigma2 enters linearly: with c = G[, 3] and P = I - cc'/(c'c), f(rho) = |P (G1 rho + G2 rho^2 - g)|^2 is a quartic;
  # the GM rho is the real root in (-1, 1) of its derivative with the smallest f
  cc <- G[, 3]; Pp <- diag(3) - tcrossprod(cc) / sum(cc^2); a1 <- drop(Pp %*% G[, 1]); a2 <- drop(Pp %*% G[, 2]); h <- drop(Pp %*% g)
  fq <- function(r) sum((a1 * r + a2 * r^2 - h)^2); dfq <- function(r) 4 * sum(a2^2) * r^3 + 6 * sum(a1 * a2) * r^2 + 2 * (sum(a1^2) - 2 * sum(a2 * h)) * r - 2 * sum(a1 * h)
  rts <- polyroot(c(-2 * sum(a1 * h), 2 * (sum(a1^2) - 2 * sum(a2 * h)), 6 * sum(a1 * a2), 4 * sum(a2^2)))
  rts <- Re(rts[abs(Im(rts)) < 1e-8 & abs(Re(rts)) < 1]); rho <- rts[which.min(sapply(rts, fq))]
  for (i in 1:5) rho <- rho - dfq(rho) / (12 * sum(a2^2) * rho^2 + 12 * sum(a1 * a2) * rho + 2 * (sum(a1^2) - 2 * sum(a2 * h)))
  s2gm <- sum(cc * (g - G[, 1] * rho - G[, 2] * rho^2)) / sum(cc^2)
  stopifnot(abs(rho - o$par[1]) < 1e-6, abs(s2gm - o$par[2]) < 1e-6)
  list(rho = rho, s2 = s2gm)
}

case("splm_spml_spgm", "econometrics/splm", "Spatial panels: spml ML (within/time/two-way/pooled lag, error, SARAR; random effects without, with Baltagi and with KKP spatial error, and with a spatial lag) and spgm (w2sls, b2sls, ec2sls, g2sls, within GM error)", function(ctx) {
  d0 <- .splm_data(ctx)
  d <- d0[order(d0$time, d0$id), ]; N <- length(unique(d$id)); TT <- length(unique(d$time))
  xy <- as.matrix(d[d$time == 1, c("cx", "cy")]); W <- .knn_w4(xy); WN <- kronecker(diag(TT), W)
  v <- list()
  put <- function(tag, f, keys) for (key in keys) v[[paste0(tag, "_", key)]] <<- if (key %in% c("coef", "se")) vec(f[[if (key == "coef") "b" else "se"]]) else f[[key]]
  for (eff in c("individual", "time", "twoways", "pooling")) {
    Xd <- if (eff == "pooling") cbind(1, d$x1, d$x2) else cbind(.within(d$x1, d$id, d$time, eff), .within(d$x2, d$id, d$time, eff))
    for (spec in c("none", "lag", "error", "sarar")) {
      if (eff != "individual" && spec == "sarar") next
      if (eff == "pooling" && spec == "none") next
      yv <- if (spec %in% c("error", "none")) d$y_err else d$y_lag
      yd <- if (eff == "pooling") yv else .within(yv, d$id, d$time, eff)
      f <- .spml_fe(yd, Xd, W, N, TT, lag = spec %in% c("lag", "sarar"), err = spec %in% c("error", "sarar"))
      tag <- paste0(eff, "_", spec)
      if (spec == "none") {
        pm <- plm(y_err ~ x1 + x2, data = d, index = c("id", "time"), model = "within", effect = eff)
        stopifnot(max(abs(coef(pm) - f$b)) < 1e-10)
        v[[paste0(tag, "_coef_plm")]] <- vec(coef(pm)); put(tag, f, c("s2", "se", "loglik"))
      } else put(tag, f, c("coef", "rho", "lambda", "s2", "se", "loglik"))
    }
  }
  X3 <- cbind(1, d$x1, d$x2)
  for (cfg in list(c("random_b", "y_re", "FALSE", "b"), c("random_kkp", "y_re", "FALSE", "kkp"), c("random_none", "y_re", "FALSE", "none"),
                   c("random_lag", "y_lag", "TRUE", "none"), c("random_sarar_b", "y_lag", "TRUE", "b"))) {
    f <- .spml_re(d[[cfg[2]]], X3, W, N, TT, lag = as.logical(cfg[3]), type = cfg[4])
    put(cfg[1], f, c("coef", if (cfg[3] == "TRUE") "rho", "phi", if (cfg[4] != "none") "lambda", "s2", "sigma2_mu", "se", "loglik"))
  }
  m <- lme(y_re ~ x1 + x2, random = ~ 1 | id, data = d, method = "ML",
           control = lmeControl(tolerance = 1e-12, msTol = 1e-12, niterEM = 100, msMaxIter = 1000))
  v$random_none_lme_coef <- vec(fixef(m)); v$random_none_lme_s2 <- m$sigma^2
  v$random_none_lme_sigma2_mu <- as.numeric(getVarCov(m)); v$random_none_lme_loglik <- as.numeric(logLik(m))
  stopifnot(max(abs(v$random_none_lme_coef - v$random_none_coef)) < 1e-8, abs(v$random_none_lme_loglik - v$random_none_loglik) < 1e-8,
            abs(v$random_none_lme_sigma2_mu / v$random_none_sigma2_mu - 1) < 1e-8)
  # spgm, lag model on y_lag
  Xw <- cbind(.within(d$x1, d$id, d$time, "individual"), .within(d$x2, d$id, d$time, "individual"))
  yw <- .within(d$y_lag, d$id, d$time, "individual")
  Zw <- cbind(Xw, WN %*% yw); Hw <- cbind(Xw, WN %*% Xw, WN %*% WN %*% Xw)
  w <- .tsls2(yw, Zw, Hw); s2nu <- sum(w$u^2) / (N * (TT - 1))
  v$w2sls_coef <- vec(w$d); v$w2sls_se <- vec(sqrt(diag(sum(w$u^2) / (N * (TT - 1) - ncol(Zw)) * solve(crossprod(w$P)))))
  v$w2sls_sargan <- .quad_PH(Hw, w$u) / (sum(w$u^2) / (N * (TT - 1))); v$w2sls_n_instruments <- ncol(Hw)
  mb <- function(v_) ave(v_, d$id)[d$time == 1]
  Xb <- cbind(1, mb(d$x1), mb(d$x2)); yb <- mb(d$y_lag)
  Zb <- cbind(Xb, W %*% yb); Hb <- cbind(Xb, W %*% Xb[, -1], W %*% W %*% Xb[, -1])
  bb <- .tsls2(yb, Zb, Hb); s21 <- TT * sum(bb$u^2) / N
  v$b2sls_coef <- vec(bb$d); v$b2sls_se <- vec(sqrt(diag(sum(bb$u^2) / (N - ncol(Zb)) * solve(crossprod(bb$P)))))
  v$b2sls_sargan <- .quad_PH(Hb, bb$u) / (sum(bb$u^2) / N); v$b2sls_n_instruments <- ncol(Hb)
  v$sigma2_nu <- s2nu; v$sigma2_1 <- s21
  X <- cbind(1, d$x1, d$x2); Z <- cbind(X, WN %*% d$y_lag); Hfull <- cbind(X, WN %*% X[, -1], WN %*% WN %*% X[, -1])
  Q0 <- function(m) apply(as.matrix(m), 2, function(c) c - ave(c, d$id)); P0 <- function(m) apply(as.matrix(m), 2, function(c) ave(c, d$id))
  om <- function(m) Q0(m) / sqrt(s2nu) + P0(m) / sqrt(s21)
  ys <- drop(om(d$y_lag)); Zs <- om(Z)
  Hec <- cbind(Q0(Hfull[, -1]), P0(Hfull)); ec <- .tsls2(ys, Zs, Hec)
  v$ec2sls_coef <- vec(ec$d); v$ec2sls_se <- vec(sqrt(diag(solve(crossprod(ec$P)))))
  v$ec2sls_sargan <- .quad_PH(Hec, ec$u); v$ec2sls_n_instruments <- ncol(Hec)
  Hg <- om(Hfull); g2 <- .tsls2(ys, Zs, Hg)
  v$g2sls_coef <- vec(g2$d); v$g2sls_se <- vec(sqrt(diag(solve(crossprod(g2$P)))))
  v$g2sls_sargan <- .quad_PH(Hg, g2$u); v$g2sls_n_instruments <- ncol(Hg)
  # spgm within error model: within OLS residuals -> KKP within GM -> within FGLS
  ye <- .within(d$y_err, d$id, d$time, "individual")
  u <- drop(ye - Xw %*% qr.coef(qr(Xw), ye)); nn <- N * (TT - 1)
  gm <- .kkp_within_gm(u, WN, W, N, TT); rho <- gm$rho; s2gm <- gm$s2
  ys2 <- ye - rho * drop(WN %*% ye); Xs2 <- Xw - rho * (WN %*% Xw)
  bfg <- qr.coef(qr(Xs2), ys2); e <- drop(ys2 - Xs2 %*% bfg)
  v$spgm_within_error_rho <- rho; v$spgm_within_error_sigma2 <- s2gm; v$spgm_within_error_coef <- vec(bfg)
  v$spgm_within_error_se <- vec(sqrt(diag(sum(e^2) / (nn - ncol(Xw)) * solve(crossprod(Xs2)))))
  # spgm within GS2SLS (lag + error): w2sls -> KKP within GM on its residuals -> 2SLS of the spatially filtered
  # model (I - lambda W) y~ on (I - lambda W) Z~ with the same instruments H = [X~, W X~, W^2 X~] (Kelejian & Prucha 1998)
  gg <- .kkp_within_gm(w$u, WN, W, N, TT)
  yf <- yw - gg$rho * drop(WN %*% yw); Zf <- Zw - gg$rho * (WN %*% Zw); gs <- .tsls2(yf, Zf, Hw)
  v$gs2sls_coef <- vec(gs$d); v$gs2sls_se <- vec(sqrt(diag(sum(gs$u^2) / (nn - ncol(Zf)) * solve(crossprod(gs$P)))))
  v$gs2sls_lambda <- gg$rho; v$gs2sls_sigma2 <- gg$s2; v$gs2sls_sargan <- .quad_PH(Hw, gs$u) / (sum(gs$u^2) / nn)
  list(values = v, reference = "spatial panel ML (Elhorst 2003; Baltagi, Song & Koh 2003; Millo & Piras 2012, splm::spml) and GM/2SLS (Kapoor, Kelejian & Prucha 2007; Baltagi 1981; splm::spgm) written out (splm not installed); plm::plm(model = 'within') and nlme::lme(method = 'ML') for the non-spatial special cases; coefficients x1, x2 (within) or (Intercept), x1, x2 (pooled, random, 2SLS), then rho (2SLS)",
       call = "spml / spgm on splm_sim, W = 4-NN row-standardised", packages = c("numDeriv", "plm", "nlme"))
})
