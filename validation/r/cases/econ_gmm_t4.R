# validation/r/cases/econ_gmm_t4.R -- T4-WP13: general GMM HAC weights on sandwich's lag-count
# convention for every kernel. p2a's lag count L is gmm(kernel = K, bw = L + 1): gmm's .myKernHAC calls
# sandwich::weightsAndrews(bw = L + 1, tol = 1e-7) (weights kweights(0:(n - 1) / (L + 1)), cut after the
# last |w| > 1e-7, with n the rows after prewhitening) and vcovHAC(prewhite, adjust = FALSE). Two-step
# linear IV on the gmm_iv data of gmm_linear_iv, L = floor(4 (n / 100)^(2 / 9)) = 5 (bw = 6), each
# kernel with and without VAR(1) prewhitening. Bartlett equals gmm_linear_iv's twostep_hac_bartlett(_pw).
suppressPackageStartupMessages(library(gmm))

case("gmm_hac_kernels_t4", "econometrics/gmm",
     "Two-step GMM-IV with HAC weights for the Bartlett, Parzen, Quadratic Spectral, Truncated and Tukey-Hanning kernels at bw = L + 1, with and without prewhitening",
     function(ctx) {
  d <- ctx$dataset("datasets/generated/gmm_iv.csv")
  f <- y ~ x1 + x2 + w; h <- ~ z1 + z2 + z3 + w
  L <- floor(4 * (nrow(d) / 100)^(2 / 9))
  v <- list(n = nrow(d), lags = L)
  kernels <- c(bartlett = "Bartlett", parzen = "Parzen", qs = "Quadratic Spectral",
               truncated = "Truncated", tukey_hanning = "Tukey-Hanning")
  for (k in names(kernels)) for (pw in 0:1) {
    fit <- gmm(f, h, data = d, type = "twoStep", vcov = "HAC", kernel = kernels[[k]], bw = L + 1, prewhite = pw)
    key <- paste0(k, if (pw == 1) "_pw" else "")
    j <- unclass(specTest(fit)$test)
    v[[paste0(key, "_coef")]] <- named(coef(fit))
    v[[paste0(key, "_se")]] <- named(sqrt(diag(vcov(fit))))
    v[[paste0(key, "_J")]] <- as.numeric(j[1, 1])
    v[[paste0(key, "_J_p")]] <- as.numeric(j[1, 2])
    v[[paste0(key, "_n_weights")]] <- length(attr(fit$v, "Spec")$weights)
  }
  list(values = v,
       reference = "gmm::gmm(type = 'twoStep', vcov = 'HAC', kernel, bw = L + 1, prewhite = 0 | 1); gmm::specTest",
       call = "gmm(y ~ x1 + x2 + w, ~ z1 + z2 + z3 + w, data = gmm_iv, type = 'twoStep', vcov = 'HAC', kernel = K, bw = 6, prewhite = p)",
       packages = c("gmm", "sandwich"))
})
