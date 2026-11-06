# validation/r/cases/econ_gmm.R -- general GMM references (T3-WP16: ECON-68; moved in ECON-21).
# gmm::gmm(y ~ x1 + x2 + w, ~ z1 + z2 + z3 + w, type = 'twoStep' | 'iterative' | 'cue',
# vcov = 'iid' | 'MDS' | 'HAC', wmatrix = 'optimal' | 'ident', weightsMatrix = W) and gmm::specTest.
#  * HAC as gmm's default (Quadratic Spectral, bwAndrews with the AR(1) approximation, VAR(1)
#    prewhitening; recorded, p2a does not implement it) and as p2a's default lag rule (Bartlett,
#    m = floor(4 (n / 100)^(2 / 9)) lags, which is sandwich bandwidth m + 1, no prewhitening), the
#    latter also with VAR(1) prewhitening and by CUE.
#  * A user-fixed weighting matrix (W = (Z'Z / n)^-1) and the identity, both with the sandwich variance.
#  * Normal-distribution moments E[x - mu], E[(x - mu)^2 - s2], E[(x - mu)^3] on x = w (nonlinear GMM
#    with the analytic gradient, BFGS at reltol 1e-15, i.e. the tight optimum; identity first step).
suppressPackageStartupMessages(library(gmm))

.gmm_specj <- function(f) { j <- unclass(specTest(f)$test); c(as.numeric(j[1, 1]), as.numeric(j[1, 2])) }

case("gmm_linear_iv", "econometrics/gmm", "Linear GMM-IV: two-step, iterative, CUE; iid / MDS / HAC weights; identity and fixed weights; normal-distribution moments", function(ctx) {
  set.seed(20261115)
  n <- 500
  z1 <- rnorm(n); z2 <- rnorm(n); z3 <- rnorm(n); w <- rnorm(n); u <- rnorm(n)
  x1 <- 0.6 * z1 + 0.3 * z2 + 0.5 * u + rnorm(n); x2 <- 0.4 * z2 - 0.4 * z3 + 0.3 * w + rnorm(n)
  y <- 1 + 0.8 * x1 - 0.5 * x2 + 0.3 * w + u * sqrt(0.5 + 0.5 * z1^2)
  d <- ctx$gen_dataset("gmm_iv", data.frame(y = round(y, 10), x1 = round(x1, 10), x2 = round(x2, 10), w = round(w, 10),
                                            z1 = round(z1, 10), z2 = round(z2, 10), z3 = round(z3, 10)))
  f <- y ~ x1 + x2 + w; h <- ~ z1 + z2 + z3 + w
  m_nw <- floor(4 * (nrow(d) / 100)^(2 / 9))                       # p2a's default HAC lag count (5 at n = 500)
  w_2sls <- solve(crossprod(model.matrix(h, d)) / nrow(d))          # a fixed, non-identity weighting matrix
  v <- list(n = nrow(d), hac_nw_lags = m_nw)
  fits <- list(
    twostep_mds = gmm(f, h, data = d, type = "twoStep", vcov = "MDS"),
    iter_mds = gmm(f, h, data = d, type = "iterative", vcov = "MDS", crit = 1e-12, itermax = 1000),
    cue_mds = gmm(f, h, data = d, type = "cue", vcov = "MDS", control = list(reltol = 1e-15, maxit = 10000)),
    cue_hac = gmm(f, h, data = d, type = "cue", vcov = "HAC", control = list(reltol = 1e-15, maxit = 10000)),
    twostep_iid = gmm(f, h, data = d, type = "twoStep", vcov = "iid"),
    twostep_hac = gmm(f, h, data = d, type = "twoStep", vcov = "HAC"),
    twostep_hac_bartlett = gmm(f, h, data = d, type = "twoStep", vcov = "HAC", kernel = "Bartlett", bw = m_nw + 1, prewhite = 0),
    twostep_hac_bartlett_pw = gmm(f, h, data = d, type = "twoStep", vcov = "HAC", kernel = "Bartlett", bw = m_nw + 1, prewhite = 1),
    cue_hac_bartlett = gmm(f, h, data = d, type = "cue", vcov = "HAC", kernel = "Bartlett", bw = m_nw + 1, prewhite = 0,
                           control = list(reltol = 1e-15, maxit = 10000)),
    twostep_mds_uncentered = gmm(f, h, data = d, type = "twoStep", vcov = "MDS", centeredVcov = FALSE),
    ident_mds = gmm(f, h, data = d, wmatrix = "ident", vcov = "MDS"),
    fixedw_mds = gmm(f, h, data = d, weightsMatrix = w_2sls, vcov = "MDS"))
  for (nm in names(fits)) {
    fit <- fits[[nm]]
    v[[paste0(nm, "_coef")]] <- named(coef(fit)); v[[paste0(nm, "_se")]] <- named(sqrt(diag(vcov(fit))))
    j <- .gmm_specj(fit); v[[paste0(nm, "_J")]] <- j[1]; v[[paste0(nm, "_J_p")]] <- j[2]
  }
  v$twostep_hac_bw <- attr(fits$twostep_hac$v, "Spec")$bw          # Andrews bandwidth at the final estimate
  v$cue_hac_bw <- attr(fits$cue_hac$v, "Spec")$bw                  # CUE: bandwidth fixed at the 2SLS first step
  tsls <- gmm(f, h, data = d, type = "twoStep", vcov = "iid", wmatrix = "optimal")
  v$iid_equals_2sls_coef <- named(coef(tsls))
  # Normal-distribution moments on x = w (identity first step, as gmm's nonlinear twoStep).
  xw <- d$w
  g <- function(th, x) { e <- x - th[1]; cbind(e, e^2 - th[2], e^3) }
  Dg <- function(th, x) { e <- x - th[1]; rbind(c(-1, 0), c(-2 * mean(e), -1), c(-3 * mean(e^2), 0)) }
  t0 <- c(mu = mean(xw), sigma_sq = mean((xw - mean(xw))^2))
  nfits <- list(
    normal_twostep = gmm(g, xw, t0 = t0, gradv = Dg, type = "twoStep", vcov = "MDS", method = "BFGS",
                         control = list(reltol = 1e-15, maxit = 10000)),
    normal_iter = gmm(g, xw, t0 = t0, gradv = Dg, type = "iterative", vcov = "MDS", crit = 1e-12, itermax = 1000,
                      method = "BFGS", control = list(reltol = 1e-15, maxit = 10000)))
  for (nm in names(nfits)) {
    fit <- nfits[[nm]]
    v[[paste0(nm, "_coef")]] <- named(coef(fit)); v[[paste0(nm, "_se")]] <- named(sqrt(diag(vcov(fit))))
    j <- .gmm_specj(fit); v[[paste0(nm, "_J")]] <- j[1]; v[[paste0(nm, "_J_p")]] <- j[2]
  }
  list(values = v, reference = "gmm::gmm(type = 'twoStep' | 'iterative' | 'cue', vcov = 'MDS' | 'iid' | 'HAC', wmatrix = 'optimal' | 'ident', weightsMatrix, centeredVcov); gmm::specTest",
       call = "gmm(y ~ x1 + x2 + w, ~ z1 + z2 + z3 + w, data = gmm_iv, ...); gmm(g, w, t0, gradv = Dg, ...) for the normal moments",
       packages = c("gmm", "sandwich"))
})
