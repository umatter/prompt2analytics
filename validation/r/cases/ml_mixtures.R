# validation/r/cases/ml_mixtures.R -- mixture-model and exemplar-clustering references (T3-WP43: ML-5,
# ML-50, ML-53, ML-59, ML-111, ML-61; moved in ML-8, ML-58). Package copy, canonical over the design's
# Appendix A.38 (see the package plan for the differences).
#
# EM stops where its log-likelihood rule says so, which leaves the parameters about 1e-6 from the fixed
# point. Every EM case therefore records the package's own result AND a "tight" fixed point, reached by
# iterating the package's own M- and E-steps (mclust::mstep/estep) or the written-out EM until no
# parameter moves by more than 1e-15. Rust pins the tight values; the package values are the looser
# cross-check.
#  * mclust::Mclust(G = 3, modelNames = VVV (full), EEE (tied), VVI (diagonal), VII (spherical)), d = 3;
#  * mixtools::normalmixEM (univariate) and mvnormalmixEM (full covariances) from fixed starts, plus the
#    written-out univariate EM from p2a's deterministic start (seed = None);
#  * flexmix::flexmix(y ~ x) from a fixed cluster start: minprior = 0.1 removes the 10-point third
#    component at iteration 3; minprior = 0.05 (the default) keeps it. A written-out FLXfit replicates
#    both to 1e-12;
#  * apcluster::apcluster(negDistMat(r = 2), q = 0.5, nonoise = TRUE): lam = 0.9 and 0.5 with
#    maxits = 2000, convits = 100 (lam = 0.5 oscillates and stops at maxits), and lam = 0.9 at p2a's
#    defaults maxits = 200, convits = 15 (where p2a used to stop with zero exemplars).
suppressPackageStartupMessages({ library(mclust); library(mixtools); library(flexmix); library(apcluster) })

.mix3 <- function(ctx) {
  set.seed(20261111)
  S1 <- matrix(c(1, 0.6, 0.2, 0.6, 1, 0.3, 0.2, 0.3, 1), 3); S2 <- matrix(c(0.5, -0.2, 0, -0.2, 0.8, 0.1, 0, 0.1, 0.4), 3)
  m <- rbind(MASS::mvrnorm(120, c(0, 0, 0), S1), MASS::mvrnorm(100, c(4, 1, -1), S2), MASS::mvrnorm(80, c(1, 5, 2), S1))
  ctx$gen_dataset("mixture_3d", data.frame(x1 = round(m[, 1], 10), x2 = round(m[, 2], 10), x3 = round(m[, 3], 10)))
}

# Fixed point of mclust's own M- and E-steps, from posterior z.
.mclust_tight <- function(X, mn, z) {
  prev <- NULL
  for (it in 1:20000) {
    m <- mstep(X, mn, z); e <- estep(X, mn, m$parameters)
    cur <- c(m$parameters$pro, m$parameters$mean, m$parameters$variance$sigma)
    if (!is.null(prev) && max(abs(cur - prev)) < 1e-15) break
    prev <- cur; z <- e$z
  }
  stopifnot(it < 20000)
  list(parameters = m$parameters, loglik = e$loglik, z = e$z)
}

case("gmm_mclust_3d", "ml/clustering", "Gaussian mixtures G = 3 in d = 3: mclust VVV (full), EEE (tied), VVI (diagonal), VII (spherical)", function(ctx) {
  d <- .mix3(ctx); X <- as.matrix(d); n <- nrow(X)
  v <- list()
  for (mn in c("VVV", "EEE", "VVI", "VII")) {
    f <- Mclust(X, G = 3, modelNames = mn, control = emControl(tol = c(1e-12, 1e-12)), verbose = FALSE)
    o <- order(f$parameters$mean[1, ])
    v[[paste0(mn, "_loglik")]] <- f$loglik; v[[paste0(mn, "_bic")]] <- f$bic; v[[paste0(mn, "_df")]] <- f$df
    v[[paste0(mn, "_pro")]] <- vec(f$parameters$pro[o]); v[[paste0(mn, "_mean")]] <- vec(f$parameters$mean[, o])
    v[[paste0(mn, "_sigma")]] <- vec(f$parameters$variance$sigma[, , o])
    v[[paste0(mn, "_classification")]] <- vec(match(f$classification, o))
    t <- .mclust_tight(X, mn, f$z)
    ot <- order(t$parameters$mean[1, ]); stopifnot(identical(ot, o))
    v[[paste0(mn, "_tight_loglik")]] <- t$loglik
    v[[paste0(mn, "_tight_bic")]] <- 2 * t$loglik - f$df * log(n)
    v[[paste0(mn, "_tight_pro")]] <- vec(t$parameters$pro[o]); v[[paste0(mn, "_tight_mean")]] <- vec(t$parameters$mean[, o])
    v[[paste0(mn, "_tight_sigma")]] <- vec(t$parameters$variance$sigma[, , o])
    v[[paste0(mn, "_tight_classification")]] <- vec(match(apply(t$z, 1, which.max), o))
    # Any k-means partition reaches the same optimum (what p2a's default k-means start relies on).
    for (s in 1:10) { set.seed(s); km <- kmeans(X, 3, nstart = 5)
      r <- me(X, mn, unmap(km$cluster), control = emControl(tol = c(1e-12, 1e-12))); stopifnot(abs(r$loglik - t$loglik) < 1e-8) }
  }
  list(values = v, reference = "mclust::Mclust(G = 3, modelNames = 'VVV' | 'EEE' | 'VVI' | 'VII', emControl(tol = 1e-12)); *_tight: mclust::mstep/estep iterated to a 1e-15 fixed point; components ordered by the first mean",
       call = "Mclust(mixture_3d, G = 3, modelNames)", packages = "mclust")
})

# Written-out univariate EM (the EM for a normal mixture; standard M-step), iterated to a 1e-15 fixed point.
