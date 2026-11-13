# validation/r/cases/stats_misc_t4.R -- T4-WP05 references: spec.pgram and spec.ar (df, bandwidth,
# ordinates, taper, padding, circular smoothing), spline (fmm, natural, periodic, hyman), splinefun
# (monoH.FC), approx (ties, constant method), isoreg (tied x, iKnots, yc), cancor (coefficients,
# no centring, p < q), factanal (zero df, promax scores).
# Findings: STAT-39, STAT-68, STAT-40, STAT-69, STAT-71, STAT-52, STAT-16, STAT-49 (and the
# T3-T4-candidates-1 promax-scores question).

case("stat_spectrum_t4", "timeseries/spectrum", "spec.pgram df, bandwidth and ordinates (raw and smoothed, taper, demean, padding, odd n, fast padding) and spec.ar (fixed order and AIC)", function(ctx) {
  d <- ctx$dataset("datasets/generated/spectrum_t3.csv")             # n = 96 (nextn(96) = 96)
  x <- d$x
  set.seed(20261113)
  e <- rnorm(97); z <- numeric(97); z[1] <- e[1]
  for (t in 2:97) z[t] <- 0.6 * z[t - 1] + e[t]
  g <- ctx$gen_dataset("spectrum_t4", data.frame(t = 1:97, z = round(z + 0.02 * (1:97), 8)))
  z97 <- g$z; z75 <- z97[1:75]                                      # nextn(75) = 75, nextn(97) = 100
  stopifnot(nextn(96) == 96, nextn(75) == 75, nextn(97) == 100, nextn(144) == 144)
  sp <- function(y, ...) spec.pgram(y, plot = FALSE, ...)
  out <- list()
  put <- function(key, s) {
    out[[paste0(key, "_freq")]] <<- vec(s$freq); out[[paste0(key, "_spec")]] <<- vec(s$spec)
    out[[paste0(key, "_df")]] <<- s$df; out[[paste0(key, "_bw")]] <<- s$bandwidth; out[[paste0(key, "_n_used")]] <<- s$n.used
  }
  put("raw_t10", sp(x, taper = 0.1))                                  # raw periodogram, tapered: df != 2
  put("s33_t10", sp(x, spans = c(3, 3), taper = 0.1))                 # = stat_spectrum_t3 (cross-check)
  put("s35_t25_demean", sp(x, spans = c(3, 5), taper = 0.25, detrend = FALSE, demean = TRUE))
  put("s7_t0_none", sp(x, spans = 7, taper = 0, detrend = FALSE))
  put("s33_pad", sp(x, spans = c(3, 3), pad = 0.5))                   # N0 = 96, N = 144
  put("s33_n75", sp(z75, spans = c(3, 3)))                            # odd n
  put("raw_n75", sp(z75, taper = 0))
  put("s33_n97_fast", sp(z97, spans = c(3, 3)))                       # fast = TRUE pads to 100
  put("s33_n97_exact", sp(z97, spans = c(3, 3), fast = FALSE))
  a2 <- spec.ar(x, n.freq = 50, order = 2, plot = FALSE)
  fit2 <- ar(x, aic = FALSE, order.max = 2, method = "yule-walker")
  aa <- spec.ar(x, n.freq = 50, plot = FALSE)
  fit <- ar(x, method = "yule-walker")
  out$ar2_freq <- vec(a2$freq); out$ar2_spec <- vec(a2$spec); out$ar2_coef <- vec(fit2$ar); out$ar2_var_pred <- fit2$var.pred
  out$aic_order <- fit$order; out$aic_spec <- vec(aa$spec); out$aic_var_pred <- fit$var.pred
  list(values = out, dataset = "validation/datasets/generated/spectrum_t3.csv",
       reference = "stats::spec.pgram (modified Daniell spans, spec.taper, u2/u4 correction, circular kernapply, df.kernel, bandwidth.kernel, nextn padding); stats::spec.ar (ar.yw, var.pred n.used/(n.used - (order + 1)))",
       call = "spec.pgram(x, spans, taper, pad, fast, detrend, demean); spec.ar(x, n.freq = 50[, order = 2])", packages = character(0))
})

case("stat_spline_t4", "stats/spline", "spline fmm, natural, periodic and hyman, splinefun monoH.FC, inside and outside the knots; tied x (ties = mean); approx constant method at knots and with ties", function(ctx) {
  sx <- c(0.0, 1.3, 0.6, 2.2, 3.1, 4.0, 2.7, 5.2, 4.6, 6.0)
  sy <- c(1.00, 0.35, 0.92, -0.41, -0.88, -0.62, -0.79, 0.37, -0.12, 0.81)
  xo <- c(-0.5, 0.15, 0.95, 1.75, 2.45, 2.9, 3.55, 4.3, 4.9, 5.6, 5.95, 6.5)
  fx <- 0:5; fy <- c(0, 0.8, 0.9, 0.1, -0.8, -1)
  px <- c(0, 1, 2.5, 3, 4.5, 6); py <- c(0, 1, 0.5, -0.5, -1, 0)
  pxo <- c(-1, 0.5, 1.7, 2.8, 3.9, 5.2, 7.3)
  ux <- seq(0, 2 * pi, length.out = 9); uy <- sin(ux); uy[9] <- uy[1]
  uxo <- c(0.3, 1.1, 2.0, 3.3, 4.4, 5.9)
  hx <- 0:5; hy <- c(0, 0, 0, 1, 1, 1); hxo <- c(-0.5, seq(0, 5, by = 0.25), 5.5)
  set.seed(7); mx <- sort(round(runif(12, 0, 10), 3)); my <- round(cumsum(abs(rnorm(12))), 4)
  mxo <- c(-1, round(seq(min(mx), max(mx), length.out = 15), 4), 11)
  tx <- c(1, 2, 2, 3, 4, 5, 5, 6); ty <- c(1.2, 0.4, 1.0, -0.3, 0.6, 1.4, 2.0, 0.9); txo <- c(0.5, 1.5, 2, 2.5, 4.5, 5, 6.5)
  # one (x, y) column pair per set, NA-padded: the MCP tool reads a pair and drops the padding rows
  sets <- list(s = list(sx, sy), f = list(fx, fy), p = list(px, py), u = list(round(ux, 15), round(uy, 15)), h = list(hx, hy),
               m = list(mx, my), t = list(tx, ty))
  len <- max(sapply(sets, function(z) length(z[[1]])))
  pad <- function(v) c(v, rep(NA_real_, len - length(v)))
  wide <- data.frame(id = seq_len(len))
  for (k in names(sets)) { wide[[paste0(k, "x")]] <- pad(sets[[k]][[1]]); wide[[paste0(k, "y")]] <- pad(sets[[k]][[2]]) }
  d <- ctx$gen_dataset("spline_t4", wide)
  get <- function(s) { x <- d[[paste0(s, "x")]]; y <- d[[paste0(s, "y")]]; k <- !is.na(x) & !is.na(y); data.frame(x = x[k], y = y[k]) }
  S <- get("s"); F <- get("f"); P <- get("p"); U <- get("u"); H <- get("h"); M <- get("m"); T <- get("t")
  stopifnot(U$y[1] == U$y[9])
  sv <- function(x, y, m, xout) vec(spline(x, y, method = m, xout = xout)$y)
  v <- list(
    fmm_s = sv(S$x, S$y, "fmm", xo), natural_s = sv(S$x, S$y, "natural", xo),
    fmm_f_n11 = vec(spline(F$x, F$y, n = 11, method = "fmm")$y), fmm_f_out = sv(F$x, F$y, "fmm", c(-1, 6)),
    natural_f_out = sv(F$x, F$y, "natural", c(-1, 6)),
    periodic_p = sv(P$x, P$y, "periodic", pxo), periodic_u = sv(U$x, U$y, "periodic", uxo),
    hyman_h = sv(H$x, H$y, "hyman", hxo), hyman_m = sv(M$x, M$y, "hyman", mxo),
    monofc_h = vec(splinefun(H$x, H$y, method = "monoH.FC")(hxo)), monofc_m = vec(splinefun(M$x, M$y, method = "monoH.FC")(mxo)),
    monofc_s = vec(splinefun(S$x, S$y, method = "monoH.FC")(xo)),
    hyman_nonmonotone_error = as.numeric(inherits(try(spline(S$x, S$y, method = "hyman"), silent = TRUE), "try-error")),
    ties_natural = suppressWarnings(sv(T$x, T$y, "natural", txo)), ties_fmm = suppressWarnings(sv(T$x, T$y, "fmm", txo)),
    ties_default_n = suppressWarnings(length(spline(T$x, T$y)$x)),
    approx_const_f0 = vec(approx(1:3, c(10, 20, 30), xout = c(1, 2, 2.5, 3), method = "constant", f = 0)$y),
    approx_const_f05 = vec(approx(1:3, c(10, 20, 30), xout = c(1, 2, 2.5, 3), method = "constant", f = 0.5)$y),
    approx_const_f1 = vec(approx(1:3, c(10, 20, 30), xout = c(1, 2, 2.5, 3), method = "constant", f = 1)$y),
    approx_ties_linear = suppressWarnings(vec(approx(c(1, 2, 2, 3), c(10, 20, 40, 30), xout = c(2, 1.5, 2.5))$y)),
    approx_ties_const = suppressWarnings(vec(approx(c(1, 2, 2, 3), c(10, 20, 40, 30), xout = c(2, 1.5, 2.5), method = "constant")$y)),
    approx_t_rule2 = suppressWarnings(vec(approx(T$x, T$y, xout = txo, rule = 2)$y)),
    xo = vec(xo), pxo = vec(pxo), uxo = vec(uxo), hxo = vec(hxo), mxo = vec(mxo), txo = vec(txo))
  list(values = v, reference = "stats::spline (method = 'fmm' | 'natural' | 'periodic' | 'hyman'; spline_coef / spline_eval), stats::splinefun(method = 'monoH.FC'), stats::approx (regularize.values ties = mean)",
       call = "spline(x, y, method, xout); splinefun(x, y, 'monoH.FC')(xout); approx(x, y, xout, method, f, rule)", packages = character(0))
})

case("stat_isoreg_t4", "stats/isoreg", "isoreg: yc and iKnots with tied x, and invariance to the row order within a tie", function(ctx) {
  d <- ctx$dataset("datasets/generated/rank_isoreg.csv")
  get <- function(s) d[d$set == s, ]
  t <- get("tiedx"); ir4 <- get("ir4"); doc <- get("doc")
  perm <- c(5, 4, 1, 3, 2)
  it <- isoreg(t$x, t$y); ip <- isoreg(t$x[perm], t$y[perm]); i4 <- isoreg(ir4$x, ir4$y); id <- isoreg(doc$y)
  stopifnot(identical(it$yf, ip$yf), identical(it$iKnots, ip$iKnots))
  list(values = list(yc_tiedx = vec(it$yc), yc_ir4 = vec(i4$yc), yf_tiedx_perm = vec(ip$yf), iknots_tiedx_perm = vec(ip$iKnots),
                     iknots_doc = vec(id$iKnots), yc_doc = vec(id$yc), perm = vec(perm)),
       reference = "stats::isoreg (order(x, -y); iKnots 1-based block ends; yc = cumsum(c(0, y)) of the ordered y)",
       call = "isoreg(x, y)", packages = character(0))
})

case("stat_cancor_t4", "multivariate/cancor", "cancor: the ?cancor-sized 10-row example, no centring, and p < q (coefficients sign-normalised)", function(ctx) {
  a <- matrix(c(191, 195, 181, 183, 176, 208, 189, 197, 188, 192,
                155, 149, 148, 153, 144, 157, 150, 159, 152, 150,
                50, 52, 49, 51, 47, 55, 51, 54, 51, 51), ncol = 3)
  b <- matrix(c(26, 21, 24, 27, 29, 19, 25, 21, 23, 25,
                16, 14, 15, 17, 18, 12, 16, 15, 14, 16), ncol = 2)
  g <- ctx$gen_dataset("cancor_t4", data.frame(a1 = a[, 1], a2 = a[, 2], a3 = a[, 3], b1 = b[, 1], b2 = b[, 2]))
  d <- ctx$dataset("datasets/generated/cancor_t3.csv")
  A <- as.matrix(g[, c("a1", "a2", "a3")]); B <- as.matrix(g[, c("b1", "b2")])
  X <- as.matrix(d[, c("x1", "x2", "x3")]); Y <- as.matrix(d[, c("y1", "y2")])
  norm <- function(cc) { r <- seq_along(cc$cor); s <- sign(cc$xcoef[1, r])
    list(cor = vec(cc$cor), x = vec(as.numeric(sweep(cc$xcoef[, r, drop = FALSE], 2, s, "*"))),
         y = vec(as.numeric(sweep(cc$ycoef[, r, drop = FALSE], 2, s, "*")))) }
  e <- norm(cancor(A, B)); nc <- norm(cancor(X, Y, xcenter = FALSE, ycenter = FALSE)); sw <- norm(cancor(Y, X))
  list(values = list(ex_cor = e$cor, ex_xcoef = e$x, ex_ycoef = e$y, nc_cor = nc$cor, nc_xcoef = nc$x, nc_ycoef = nc$y,
                     sw_cor = sw$cor, sw_xcoef = sw$x, sw_ycoef = sw$y),
       dataset = "validation/datasets/generated/cancor_t4.csv",
       reference = "stats::cancor (QR of the [centred] data, SVD of Qx'Qy); first min(p, q) coefficient columns, column signs so that the first x coefficient is positive",
       call = "cancor(A, B); cancor(X, Y, xcenter = FALSE, ycenter = FALSE); cancor(Y, X)", packages = character(0))
})

case("stat_factanal_t4", "stats/factanal", "factanal: a saturated model (dof = 0) reports no test; a constant column is an error; promax regression and Bartlett scores", function(ctx) {
  d <- ctx$dataset("datasets/generated/factanal_promax.csv")
  X <- as.matrix(d)
  tight <- list(opt = list(factr = 1, pgtol = 0, maxit = 10000))
  f3 <- factanal(X[, 1:3], 1)                                      # tight control cannot start here
  stopifnot(f3$dof == 0, is.null(f3$STATISTIC), is.null(f3$PVAL))
  Xc <- cbind(X[, 1:6], v13 = 1)
  const_err <- inherits(try(factanal(Xc, 2), silent = TRUE), "try-error")
  stopifnot(const_err)
  # promax scores on R's tight loadings (factanal's own varimax stops at eps 1e-5: STAT-23)
  f0 <- factanal(X, 3, rotation = "none", control = tight)
  L0 <- unclass(f0$loadings)
  vt <- varimax(L0, eps = 1e-14); Lv <- unclass(vt$loadings); Q <- Lv * abs(Lv)^3
  U <- lm.fit(Lv, Q)$coefficients; U <- U %*% diag(sqrt(diag(solve(t(U) %*% U))))
  Lp <- Lv %*% U; Rp <- vt$rotmat %*% U
  o <- order(-colSums(Lp^2)); s <- ifelse(colSums(Lp[, o]) < 0, -1, 1)
  Ls <- Lp[, o] %*% diag(s); Rs <- Rp[, o] %*% diag(s); Phi <- solve(t(Rs) %*% Rs)
  zz <- scale(X, TRUE, TRUE); cv <- cor(X)
  reg <- zz %*% solve(cv, Ls)                                      # factanal's regression scores (no Phi with stats::promax)
  reg_phi <- reg %*% Phi                                           # factanal's formula when the loadings carry Phi
  dd <- 1 / f0$uniquenesses; tmp <- t(Ls * dd); bart <- t(solve(tmp %*% Ls, tmp %*% t(zz)))
  fp <- factanal(X, 3, rotation = "promax", scores = "regression")
  stopifnot(max(abs(fp$scores - zz %*% solve(cv, unclass(fp$loadings)))) < 1e-10)  # R 4.3.3: no Phi for stats::promax
  rm <- function(m) vec(as.numeric(t(m)))
  list(values = list(df0_uniquenesses = vec(f3$uniquenesses), df0_loadings = vec(as.numeric(unclass(f3$loadings))), df0_dof = f3$dof,
                     df0_has_test = 0, constant_column_error = 1,
                     promax_reg_rows10 = rm(reg[1:10, ]), promax_reg_phi_rows10 = rm(reg_phi[1:10, ]), promax_bartlett_rows10 = rm(bart[1:10, ]),
                     promax_reg_colmeans = vec(colMeans(reg)), n = nrow(X)),
       reference = "stats::factanal (dof = 0: no STATISTIC/PVAL; zero-variance column: error); regression scores zz %*% solve(cv, Lambda) [%*% Phi], Bartlett scores, on R's tight promax loadings",
       call = "factanal(X[, 1:3], 1); factanal(cbind(X[, 1:6], 1), 2); factanal(X, 3, rotation = 'promax', scores = ...)", packages = character(0))
})
