# validation/r/cases/econ_sysgmm.R -- System GMM (Blundell-Bond) references (T3-WP15: ECON-30).
# plm::pgmm(transformation = "ld"): difference equations instrumented by lagged levels (as T2's
# Arellano-Bond cases) plus level equations instrumented by lagged differences. Same panel,
# estimation conventions and outputs as validation/r/cases/econ_dynamic_panel.R (T2-WP19), which
# defines .ab_fit-style quantities; the panel is the committed ab_balanced / ab_gaps datasets.
#
# Package copy (T3-WP15; canonical, supersedes design Appendix A.10). Every key of the design copy
# is kept with the same value. Added:
#   * per fit: the equations used (total, difference, level) and the entities with an equation;
#     for one-step fits sigma2 from the difference residuals (Arellano-Bond 1991). pgmm's own
#     non-robust one-step vcov (m$B1) omits sigma2 and carries a 1/N factor, and pgmm's H weight
#     is not the error covariance of the level equations when var(eta_i) > 0, so `onestep_se` is
#     recorded and not pinned (p2a refuses classical one-step SEs for system GMM);
#   * pgmm-pinned variants: lag(y, 1:2), lag(y, 2:3), lag(y, 3:99), collapse = TRUE, no x;
#   * lags3_spec_* and minlag4_spec_*: with lag(y, 1:3) or lag(y, 4:99) pgmm 2.6.7 fills its
#     level-instrument matrix with matrix(diag(..), nrow = T - TL2) of the wrong length (R warns
#     "data length [36] is not a sub-multiple or multiple of the number of rows [5]"), so its
#     estimates are not the system estimator. These keys come from .sys_spec below, the estimator
#     written out in R, which the case first checks against pgmm on the default, lag(y, 1:2) and
#     lag(y, 3:99) fits.
# No case here has an instrument column that is zero for every equation (stopifnot), so pgmm's
# instrument count and Sargan df are p2a's.
suppressPackageStartupMessages(library(plm))

.sys_fit <- function(pd, form, st, prefix, collapse = FALSE) {
  m <- suppressWarnings(pgmm(form, data = pd, effect = "individual", model = st, transformation = "ld", collapse = collapse))
  s <- suppressWarnings(summary(m, robust = FALSE)); sr <- suppressWarnings(summary(m, robust = TRUE))
  out <- list()
  out[[paste0(prefix, "_coef")]] <- named(coef(m))
  out[[paste0(prefix, "_se")]] <- named(setNames(s$coefficients[, 2], rownames(s$coefficients)))
  out[[paste0(prefix, "_se_robust")]] <- named(setNames(sr$coefficients[, 2], rownames(sr$coefficients)))
  out[[paste0(prefix, "_sargan")]] <- unname(s$sargan$statistic)
  out[[paste0(prefix, "_sargan_df")]] <- unname(s$sargan$parameter)
  out[[paste0(prefix, "_sargan_p")]] <- s$sargan$p.value
  out[[paste0(prefix, "_ar1")]] <- unname(sr$m1$statistic); out[[paste0(prefix, "_ar2")]] <- unname(sr$m2$statistic)
  out[[paste0(prefix, "_ar1_p")]] <- sr$m1$p.value; out[[paste0(prefix, "_ar2_p")]] <- sr$m2$p.value
  out[[paste0(prefix, "_n_instruments")]] <- ncol(m$W[[1]])
  out[[paste0(prefix, "_wald_coef")]] <- unname(s$wald.coef$statistic)
  # T3-WP15 additions
  k <- length(coef(m))
  stopifnot(sum(colSums(abs(Reduce(`+`, lapply(m$W, abs)))) > 0) == ncol(m$W[[1]]))
  Kt <- length(m$args$namest)                                  # level rows per entity on pgmm's grid
  rD <- unlist(lapply(m$residuals, function(r) r[seq_len(Kt - 1)]), use.names = FALSE)     # difference rows
  rL <- unlist(lapply(m$residuals, function(r) r[Kt - 1 + seq_len(Kt)]), use.names = FALSE) # level rows
  out[[paste0(prefix, "_n_obs")]] <- sum(unlist(m$residuals, use.names = FALSE) != 0)
  out[[paste0(prefix, "_n_obs_diff")]] <- sum(rD != 0)
  out[[paste0(prefix, "_n_obs_level")]] <- sum(rL != 0)
  out[[paste0(prefix, "_n_groups")]] <- sum(vapply(m$residuals, function(r) any(r != 0), TRUE))
  if (st == "onestep") {
    # sigma_eps^2 from the one-step difference residuals, as Arellano-Bond (1991) and p2a's
    # difference GMM. p2a reports it, but no homoskedastic one-step SE or Sargan for system GMM:
    # pgmm's H weight is not the error covariance of the level equations when var(eta_i) > 0.
    eD <- rD[rD != 0]
    out[[paste0(prefix, "_sigma2")]] <- sum(eD^2) / (2 * (length(eD) - k))
  }
  out
}

.sys_values <- function(d) {
  pd <- pdata.frame(d[complete.cases(d), ], index = c("id", "year"))
  form <- y ~ lag(y, 1) + x | lag(y, 2:99)
  c(.sys_fit(pd, form, "onestep", "onestep"), .sys_fit(pd, form, "twosteps", "twostep"))
}

# System GMM written out in R (the T3-WP15 specification): per entity, a difference equation at t
# when t - t_min >= TL1 = max(lags + 1, min_lag) and y is observed at t, ..., t - lags - 1
# (instruments y_s, s in [t - max_lag, t - min_lag], one column per (t, s)), and a level equation
# at t when t - t_min >= TL2 = TL1 - 1 and y is observed at t, ..., t - lags (instrument
# y_{t-1} - y_{t-2}, one column per t); x enters in differences (difference rows) and in levels
# (level rows) as its own instrument. Weight (sum_i Z_i' H_i Z_i)^-1 with pgmm's FSM "full" H
# (difference block 2/-1, level block identity, cross term +1 at the same t and -1 at t-1), then
# (sum_i Z_i' e_i e_i' Z_i)^-1; Windmeijer (2005) SEs and the Sargan statistic as vcovHC.pgmm and
# sargan. Full-rank weights only (stopifnot); this balanced panel has them.
.sys_spec <- function(d, x = "x", lags = 1L, min_lag = 2L) {
  tmin <- min(d$year); P <- max(d$year) - tmin + 1
  tl1 <- max(lags + 1L, min_lag); tl2 <- tl1 - 1L; k <- lags + length(x)
  rows <- list()
  for (g in sort(unique(d$id))) {
    di <- d[d$id == g, ]; yy <- setNames(di$y, di$year); xx <- setNames(di[[x]], di$year)
    Y <- function(s) yy[[as.character(s)]]; has <- function(s) all(as.character(s) %in% names(yy))
    for (t in di$year) if (t - tmin >= tl1 && has((t - lags - 1):t)) {
      s <- (t - (P - 1)):(t - min_lag); s <- s[s >= tmin & as.character(s) %in% names(yy)]
      rows[[length(rows) + 1]] <- list(g = g, t = t, D = TRUE, lhs = Y(t) - Y(t - 1),
        X = c(sapply(seq_len(lags), function(j) Y(t - j) - Y(t - j - 1)), xx[[as.character(t)]] - xx[[as.character(t - 1)]]),
        keys = sprintf("a%05d_%05d", t, s), v = vapply(s, Y, 0))
    }
    for (t in di$year) if (t - tmin >= tl2 && has((t - lags):t)) {
      iv <- has(c(t - 1, t - 2))
      rows[[length(rows) + 1]] <- list(g = g, t = t, D = FALSE, lhs = Y(t),
        X = c(sapply(seq_len(lags), function(j) Y(t - j)), xx[[as.character(t)]]),
        keys = if (iv) sprintf("b%05d", t) else character(0), v = if (iv) Y(t - 1) - Y(t - 2) else numeric(0))
    }
  }
  keys <- sort(unique(unlist(lapply(rows, `[[`, "keys")))); L <- length(keys) + 2L; n <- length(rows)
  Z <- matrix(0, n, L); X <- matrix(0, n, k); lhs <- numeric(n)
  grp <- vapply(rows, function(r) r$g, 0); tt <- vapply(rows, function(r) r$t, 0); isD <- vapply(rows, function(r) r$D, TRUE)
  for (a in seq_len(n)) {
    r <- rows[[a]]; lhs[a] <- r$lhs; X[a, ] <- r$X; Z[a, match(r$keys, keys)] <- r$v
    Z[a, length(keys) + if (r$D) 1L else 2L] <- r$X[k]
  }
  H <- function(i) outer(i, i, Vectorize(function(a, b) {
    if (isD[a] && isD[b]) c(2, -1, 0)[min(abs(tt[a] - tt[b]), 2) + 1]
    else if (!isD[a] && !isD[b]) as.numeric(tt[a] == tt[b])
    else { td <- if (isD[a]) tt[a] else tt[b]; tl <- if (isD[a]) tt[b] else tt[a]; (tl == td) - (tl == td - 1) }
  }))
  blocks <- split(seq_len(n), grp)
  inv <- function(M) { stopifnot(min(eigen(M, symmetric = TRUE, only.values = TRUE)$values) > 1e-9); solve(M) }
  W1 <- inv(Reduce(`+`, lapply(blocks, function(i) t(Z[i, , drop = FALSE]) %*% H(i) %*% Z[i, , drop = FALSE])))
  ZX <- crossprod(Z, X); Zy <- crossprod(Z, lhs)
  B1 <- solve(t(ZX) %*% W1 %*% ZX); b1 <- drop(B1 %*% t(ZX) %*% W1 %*% Zy); e1 <- drop(lhs - X %*% b1)
  g1 <- lapply(blocks, function(i) crossprod(Z[i, , drop = FALSE], e1[i]))
  S2 <- Reduce(`+`, lapply(g1, tcrossprod)); W2 <- inv(S2)
  B2 <- solve(t(ZX) %*% W2 %*% ZX); b2 <- drop(B2 %*% t(ZX) %*% W2 %*% Zy); e2 <- drop(lhs - X %*% b2)
  V1r <- B1 %*% (t(ZX) %*% W1 %*% S2 %*% W1 %*% ZX) %*% B1
  u <- W2 %*% crossprod(Z, e2)
  D <- sapply(seq_len(k), function(j) {
    q <- Reduce(`+`, Map(function(i, g) { a <- crossprod(Z[i, , drop = FALSE], X[i, j]); -(a * sum(g * u) + g * sum(a * u)) }, blocks, g1))
    -B2 %*% t(ZX) %*% W2 %*% q
  })
  V2r <- B2 + D %*% B2 + t(D %*% B2) + D %*% V1r %*% t(D)
  z2 <- crossprod(Z, e2)
  list(b1 = b1, b2 = b2, se2r = sqrt(diag(V2r)), sargan2 = drop(t(z2) %*% W2 %*% z2),
       n_obs = n, n_obs_diff = sum(isD), n_obs_level = sum(!isD), n_instruments = L)
}

.spec_values <- function(s, prefix) {
  out <- list()
  out[[paste0(prefix, "_onestep_coef")]] <- vec(s$b1); out[[paste0(prefix, "_twostep_coef")]] <- vec(s$b2)
  out[[paste0(prefix, "_twostep_se_robust")]] <- vec(s$se2r); out[[paste0(prefix, "_twostep_sargan")]] <- s$sargan2
  for (nm in c("n_obs", "n_obs_diff", "n_obs_level", "n_instruments")) out[[paste0(prefix, "_", nm)]] <- s[[nm]]
  out
}

case("dpd_sys_balanced", "econometrics/panel", "System GMM (pgmm transformation = 'ld'), balanced N = 100, T = 8", function(ctx) {
  d <- ctx$dataset("datasets/generated/ab_balanced.csv")
  pd <- pdata.frame(d, index = c("id", "year"))
  v <- c(.sys_values(d),
         .sys_fit(pd, y ~ lag(y, 1:2) + x | lag(y, 2:99), "twosteps", "lags2_twostep"),
         .sys_fit(pd, y ~ lag(y, 1) + x | lag(y, 2:3), "twosteps", "maxlag3_twostep"),
         .sys_fit(pd, y ~ lag(y, 1) + x | lag(y, 3:99), "onestep", "minlag3_onestep"),
         .sys_fit(pd, y ~ lag(y, 1) + x | lag(y, 2:99), "twosteps", "collapse_twostep", collapse = TRUE),
         .sys_fit(pd, y ~ lag(y, 1) | lag(y, 2:99), "onestep", "nox_onestep"))
  # .sys_spec reproduces pgmm wherever pgmm builds its level instruments correctly (TL2 <= 2).
  agree <- function(s, key) { r <- v[[key]]$values; stopifnot(max(abs(s - r) / abs(r)) < 1e-9) }
  s0 <- .sys_spec(d); agree(s0$b1, "onestep_coef"); agree(s0$b2, "twostep_coef"); agree(s0$se2r, "twostep_se_robust")
  s2 <- .sys_spec(d, lags = 2L); agree(s2$b2, "lags2_twostep_coef"); agree(s2$se2r, "lags2_twostep_se_robust")
  s3 <- .sys_spec(d, min_lag = 3L); agree(s3$b1, "minlag3_onestep_coef")
  v <- c(v, .spec_values(.sys_spec(d, lags = 3L), "lags3_spec"), .spec_values(.sys_spec(d, min_lag = 4L), "minlag4_spec"))
  list(values = v, reference = "plm::pgmm(transformation = 'ld', effect = 'individual'); summary.pgmm (robust = FALSE/TRUE); sargan; mtest; variants lag(y, 1:2), lag(y, 2:3), lag(y, 3:99), collapse = TRUE, no x; lags3_spec/minlag4_spec: the system estimator written out in R (pgmm 2.6.7 mis-builds its level instruments when lags >= 3 or min_lag >= 4)",
       call = "pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), data = pdata.frame(ab_balanced, index = c('id','year')), transformation = 'ld', model = 'onestep' | 'twosteps')",
       packages = "plm")
})

case("dpd_sys_gaps", "econometrics/panel", "System GMM (pgmm transformation = 'ld'), panel with interior gaps and late entry", function(ctx) {
  d <- ctx$dataset("datasets/generated/ab_gaps.csv")
  pd <- pdata.frame(d[complete.cases(d), ], index = c("id", "year"))
  v <- c(.sys_values(d),
         .sys_fit(pd, y ~ lag(y, 1:2) + x | lag(y, 2:99), "twosteps", "lags2_twostep"),
         .sys_fit(pd, y ~ lag(y, 1) + x | lag(y, 2:99), "twosteps", "collapse_twostep", collapse = TRUE))
  list(values = v, reference = "plm::pgmm(transformation = 'ld')",
       call = "pgmm(y ~ lag(y, 1) + x | lag(y, 2:99), data = pdata.frame(ab_gaps), transformation = 'ld')",
       packages = "plm")
})

# The textbook EmplUK system-GMM example (Blundell-Bond style), as a second, published reference.
case("dpd_sys_empluk", "econometrics/panel", "System GMM on plm::EmplUK (log emp, wage, capital), two-step", function(ctx) {
  data("EmplUK", package = "plm")
  e <- EmplUK
  e$lemp <- log(e$emp); e$lwage <- log(e$wage); e$lcap <- log(e$capital)
  d <- ctx$gen_dataset("sysgmm_empluk", e[, c("firm", "year", "lemp", "lwage", "lcap")])
  pd <- pdata.frame(d, index = c("firm", "year"))
  form <- lemp ~ lag(lemp, 1) + lwage + lcap | lag(lemp, 2:99)
  v <- c(.sys_fit(pd, form, "onestep", "onestep"), .sys_fit(pd, form, "twosteps", "twostep"))
  list(values = v, reference = "plm::pgmm(transformation = 'ld') on EmplUK",
       call = "pgmm(lemp ~ lag(lemp, 1) + lwage + lcap | lag(lemp, 2:99), EmplUK, transformation = 'ld')",
       packages = "plm")
})
