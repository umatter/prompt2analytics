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
.uni_em_tight <- function(x, lambda, mu, sigma) {
  n <- length(x); k <- length(mu)
  for (it in 1:200000) {
    dens <- sapply(1:k, function(j) lambda[j] * dnorm(x, mu[j], sigma[j])); post <- dens / rowSums(dens)
    nj <- colSums(post); lam <- nj / n; m <- colSums(post * x) / nj
    s <- sqrt(colSums(post * (x - matrix(m, n, k, byrow = TRUE))^2) / nj)
    ch <- max(abs(c(lam - lambda, m - mu, s - sigma))); lambda <- lam; mu <- m; sigma <- s
    if (ch < 1e-15) break
  }
  stopifnot(it < 200000)
  dens <- sapply(1:k, function(j) lambda[j] * dnorm(x, mu[j], sigma[j]))
  list(lambda = lambda, mu = mu, sigma = sigma, loglik = sum(log(rowSums(dens))))
}

case("mixtools_em", "ml/clustering", "mixtools normalmixEM (univariate) and mvnormalmixEM (full covariance) from fixed starts", function(ctx) {
  d <- ctx$dataset("datasets/generated/mixture_3d.csv"); X <- as.matrix(d); x <- X[, 1]; n <- length(x)
  v <- list()
  invisible(capture.output(u <- normalmixEM(x, lambda = c(0.4, 0.3, 0.3), mu = c(-0.5, 1.5, 4), sigma = c(1, 1, 1), epsilon = 1e-12, maxit = 10000)))
  stopifnot(u$restarts == 0, length(u$all.loglik) - 1 < 10000)
  v$uni_loglik <- u$loglik; v$uni_lambda <- vec(u$lambda); v$uni_mu <- vec(u$mu); v$uni_sigma <- vec(u$sigma)
  v$uni_iterations <- length(u$all.loglik) - 1
  t <- .uni_em_tight(x, u$lambda, u$mu, u$sigma)
  v$uni_tight_loglik <- t$loglik; v$uni_tight_lambda <- vec(t$lambda); v$uni_tight_mu <- vec(t$mu); v$uni_tight_sigma <- vec(t$sigma)
  # p2a's deterministic start (seed = None): equal weights, the population sd, means at the centres of
  # k equal-width bins of the range. It reaches the same optimum.
  k <- 3; mu0 <- min(x) + ((0:(k - 1)) + 0.5) / k * (max(x) - min(x)); s0 <- sqrt(mean((x - mean(x))^2))
  td <- .uni_em_tight(x, rep(1 / k, k), mu0, rep(s0, k))
  stopifnot(max(abs(c(td$lambda, td$mu, td$sigma) - c(t$lambda, t$mu, t$sigma))) < 1e-12)
  v$uni_det_start_mu <- vec(mu0); v$uni_det_start_sigma <- s0
  v$uni_det_start_loglik <- sum(log(rowSums(sapply(1:k, function(j) dnorm(x, mu0[j], s0) / k))))   # loglik_history[1] for seed = None
  invisible(capture.output(mv <- mvnormalmixEM(X, lambda = c(0.4, 0.33, 0.27), mu = list(c(0, 0, 0), c(4, 1, -1), c(1, 5, 2)),
                      sigma = list(diag(3), diag(3), diag(3)), epsilon = 1e-12, maxit = 10000)))
  stopifnot(mv$restarts == 0)
  v$mv_loglik <- mv$loglik; v$mv_lambda <- vec(mv$lambda); v$mv_mu <- vec(unlist(mv$mu)); v$mv_sigma <- vec(unlist(mv$sigma))
  v$mv_iterations <- length(mv$all.loglik) - 1
  tm <- .mclust_tight(X, "VVV", mv$posterior)                       # same model, mixtools' component order
  v$mv_tight_loglik <- tm$loglik; v$mv_tight_lambda <- vec(tm$parameters$pro); v$mv_tight_mu <- vec(tm$parameters$mean)
  v$mv_tight_sigma <- vec(tm$parameters$variance$sigma)
  list(values = v, reference = "mixtools::normalmixEM(epsilon = 1e-12); mixtools::mvnormalmixEM(epsilon = 1e-12), given starts; *_tight: written-out EM (univariate) / mclust::mstep+estep VVV (multivariate) to a 1e-15 fixed point",
       call = "normalmixEM(x1, lambda, mu, sigma); mvnormalmixEM(X, lambda, mu, sigma)", packages = c("mixtools", "mclust"))
})

# Written-out flexmix::FLXfit for FLXMRglm(gaussian) + FLXPconstant, classify = "weighted" (flexmix 2.3.21).
.flex_rep <- function(X, y, cluster, k, minprior, tol, itmax) {
  N <- nrow(X); post <- matrix(0.1, N, k); for (K in 1:k) post[cluster == K, K] <- 0.9
  post <- post / rowSums(post); llh <- -Inf; removed <- integer(0)
  for (iter in 1:itmax) {
    prior <- colMeans(post); nok <- which(prior < minprior)
    if (length(nok)) {
      removed <- c(removed, iter); prior <- prior[-nok]; prior <- prior / sum(prior)
      post <- post[, -nok, drop = FALSE]; post <- post / rowSums(post); k <- ncol(post)
    }
    coef <- matrix(0, ncol(X), k); sig <- numeric(k)
    for (j in 1:k) { fit <- lm.wfit(X, y, post[, j]); coef[, j] <- fit$coefficients
      sig[j] <- sqrt(sum(fit$weights * fit$residuals^2 / mean(fit$weights)) / (N - fit$rank)) }
    lp <- sapply(1:k, function(j) dnorm(y, X %*% coef[, j], sig[j], log = TRUE)) + matrix(log(prior), N, k, byrow = TRUE)
    mx <- apply(lp, 1, max); lse <- mx + log(rowSums(exp(lp - mx))); post <- exp(lp - lse)
    llh_old <- llh; llh <- sum(lse)
    if (abs(llh - llh_old) / (abs(llh) + 0.1) < tol) break
  }
  list(k = k, iter = iter, llh = llh, prior = colMeans(post), coef = coef, sigma = sig, removed = removed)
}

case("flexmix_minprior", "ml/clustering", "flexmix regression mixture from a fixed 3-cluster start: minprior = 0.1 removes the third component, minprior = 0.05 keeps it", function(ctx) {
  set.seed(20261112)
  n <- 200; x <- runif(n, 0, 10); g <- sample(1:2, n, replace = TRUE, prob = c(0.6, 0.4))
  y <- ifelse(g == 1, 1 + 0.5 * x, 6 - 0.4 * x) + rnorm(n, 0, 0.6)
  start <- c(ifelse(g == 1, 1, 2)[1:190], rep(3, 10))                   # a third start component of 10 points (5%)
  d <- ctx$gen_dataset("flexmix_lines", data.frame(y = round(y, 10), x = round(x, 10), start = start))
  X <- cbind(1, d$x); v <- list()
  for (cfg in list(list(tag = "prune", mp = 0.1), list(tag = "keep", mp = 0.05))) {
    f <- flexmix(y ~ x, data = d, cluster = d$start, control = list(minprior = cfg$mp, tolerance = 1e-12, iter.max = 5000))
    stopifnot(f@converged)
    r <- .flex_rep(X, d$y, d$start, 3, cfg$mp, 1e-12, 5000)
    p <- parameters(f); o <- order(p["coef.(Intercept)", ])
    stopifnot(r$k == f@k, r$iter == f@iter, abs(r$llh - as.numeric(logLik(f))) < 1e-9, max(abs(rbind(r$coef, r$sigma) - p)) < 1e-12)
    tg <- cfg$tag
    v[[paste0(tg, "_k")]] <- f@k; v[[paste0(tg, "_k0")]] <- max(d$start); v[[paste0(tg, "_iterations")]] <- f@iter
    v[[paste0(tg, "_removed_at")]] <- vec(if (length(r$removed)) r$removed else 0)
    v[[paste0(tg, "_loglik")]] <- as.numeric(logLik(f)); v[[paste0(tg, "_df")]] <- attr(logLik(f), "df")
    v[[paste0(tg, "_aic")]] <- AIC(f); v[[paste0(tg, "_bic")]] <- BIC(f)
    v[[paste0(tg, "_prior")]] <- vec(prior(f)[o]); v[[paste0(tg, "_coef")]] <- vec(p[1:2, o]); v[[paste0(tg, "_sigma")]] <- vec(p[3, o])
    v[[paste0(tg, "_cluster")]] <- vec(match(clusters(f), o))
  }
  # A start posterior that puts component 3 on exactly two points: lm.wfit fits them exactly, sigma = 0,
  # and flexmix stops with "Log-likelihood: NaN" (p2a floors sigma and warns instead).
  P <- matrix(0, n, 3); P[cbind(1:n, ifelse(d$start == 3, 1, d$start))] <- 1; P[1:2, ] <- 0; P[1:2, 3] <- 1
  e <- tryCatch(flexmix(y ~ x, data = d, cluster = P, control = list(minprior = 0, tolerance = 1e-12, iter.max = 50)), error = function(e) e)
  stopifnot(inherits(e, "error"), grepl("NaN", conditionMessage(e)))
  list(values = v, reference = "flexmix::flexmix(y ~ x, cluster = start, control = list(minprior = 0.1 | 0.05, tolerance = 1e-12)); a written-out FLXfit agrees to 1e-12",
       call = "flexmix(y ~ x, data = flexmix_lines, cluster = start)", packages = "flexmix")
})

# The same three blobs plus noise as ml_density_clustering.R's blobs_noise (same seed and draws), written
# under this package's own name so that WP43 does not depend on WP41's dataset in the same wave.
