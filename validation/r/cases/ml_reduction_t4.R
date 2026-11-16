# validation/r/cases/ml_reduction_t4.R -- PCA on the SVD path, cmdscale's GOF and additive constant, exact t-SNE
# (T4-WP22: ML-35, ML-98, ML-37, ML-97).
#  * stats::prcomp(x) (centred, not scaled) on n <= 4p data: every sdev, summary()'s proportion of variance and the
#    total variance, for n > p (20 x 8) and n < p (6 x 10).
#  * stats::cmdscale(k, eig = TRUE[, add = TRUE]) on eurodist and on Manhattan distances of eight points: GOF (both
#    measures), the additive constant ac, |points| and the eigenvalues of the (modified) double-centred matrix.
#  * Rtsne::Rtsne(theta = 0, pca = FALSE, normalize = FALSE, Y_init) -- bhtsne's exact path: perplexity calibration of
#    beta = 1/(2 sigma^2) (start 1, doubling/halving while a bound is open, 200 steps, tol 1e-5), P symmetrised and
#    normalised, exaggeration 12 until iteration 250, momentum 0.5 then 0.8 at 250, eta 200 on the gradient without the
#    factor 4, gains +0.2 / x0.8 with floor 0.01, Y centred each step. The embedding after 1 and 10 iterations
#    (later ones are chaotic in the summation order). Y_init is part of the generated dataset.
suppressPackageStartupMessages(library(Rtsne))

case("pca_svd_path", "ml/pca", "stats::prcomp (centred, unscaled) on n <= 4p: sdev, proportion of variance, total variance", function(ctx) {
  set.seed(20261115)
  n <- 20; p <- 8
  z <- matrix(rnorm(n * 3), n, 3); load <- matrix(rnorm(3 * p), 3, p)
  x <- z %*% load + matrix(rnorm(n * p, sd = 0.5), n, p)
  colnames(x) <- paste0("v", 1:p)
  d <- ctx$gen_dataset("pca_svd_20x8", as.data.frame(round(x, 10)))
  X <- as.matrix(d)
  set.seed(20261116)
  w <- matrix(rnorm(6 * 10), 6, 10); colnames(w) <- paste0("w", 1:10)
  dw <- ctx$gen_dataset("pca_wide_6x10", as.data.frame(round(w, 10)))
  W <- as.matrix(dw)
  pr <- prcomp(X); pw <- prcomp(W)
  imp <- summary(pr)$importance; impw <- summary(pw)$importance
  v <- list(
    tall_sdev = vec(pr$sdev),                                   # 8 values
    tall_prop = vec(pr$sdev^2 / sum(pr$sdev^2)),                # summary()$importance[2, ] unrounded
    tall_total_variance = sum(pr$sdev^2),
    tall_rotation_abs_k2 = vec(abs(pr$rotation[, 1:2])),        # column-major, p x 2
    wide_sdev = vec(pw$sdev),                                   # min(n, p) = 6 values; the last is roundoff
    wide_prop = vec(pw$sdev^2 / sum(pw$sdev^2)),
    wide_total_variance = sum(pw$sdev^2)
  )
  stopifnot(all(abs(round(v$tall_prop, 5) - imp[2, ]) < 1e-12), all(abs(round(v$wide_prop, 5) - impw[2, ]) < 1e-12))
  list(values = v, dataset = "validation/datasets/generated/pca_svd_20x8.csv",
       reference = "stats::prcomp(x) (center = TRUE, scale. = FALSE); proportion of variance = sdev^2 / sum(sdev^2) over all min(n, p) components",
       call = "prcomp(pca_svd_20x8); prcomp(pca_wide_6x10)", packages = character(0))
})

case("cmdscale_gof_add", "ml/cmdscale", "stats::cmdscale(k = 2, eig = TRUE), add = FALSE and TRUE (Cailliez): GOF, ac, |points|, eig; eurodist and Manhattan distances", function(ctx) {
  e <- as.matrix(ctx$dataset("datasets/generated/eurodist_matrix.csv"))
  dimnames(e) <- NULL
  stopifnot(max(abs(e - as.matrix(eurodist))) == 0)
  x8 <- matrix(c(0.12,1.05,2.31, 0.48,0.31,1.74, 2.91,3.17,0.52, 3.36,2.62,1.18,
                 1.07,0.74,2.95, 5.21,0.43,0.66, 4.83,1.12,1.41, 0.71,2.38,3.27), ncol = 3, byrow = TRUE)
  colnames(x8) <- c("a", "b", "c")
  d8 <- ctx$gen_dataset("mds_points8", as.data.frame(x8))
  m <- as.matrix(dist(as.matrix(d8), "manhattan")); dimnames(m) <- NULL
  v <- list()
  put <- function(tag, f) {
    v[[paste0(tag, "_gof")]] <<- vec(f$GOF)
    v[[paste0(tag, "_ac")]] <<- f$ac
    v[[paste0(tag, "_points_abs")]] <<- vec(abs(f$points))      # column-major, n x k
    v[[paste0(tag, "_eig")]] <<- vec(f$eig)
  }
  put("euro_k2", cmdscale(e, k = 2, eig = TRUE))
  put("euro_k2_add", cmdscale(e, k = 2, eig = TRUE, add = TRUE))
  put("manh_k2", cmdscale(m, k = 2, eig = TRUE))
  put("manh_k3_add", cmdscale(m, k = 3, eig = TRUE, add = TRUE))
  list(values = v, dataset = "validation/datasets/generated/eurodist_matrix.csv",
       reference = "stats::cmdscale(d, k, eig = TRUE, add); GOF = sum(ev[1:k]) / c(sum(abs(eig)), sum(pmax(eig, 0))); add = TRUE: Cailliez's constant from the 2n x 2n eigenproblem",
       call = "cmdscale(eurodist, k = 2, eig = TRUE[, add = TRUE]); cmdscale(dist(mds_points8, 'manhattan'), k = 2 | 3, eig = TRUE[, add = TRUE])",
       packages = character(0))
})

case("tsne_rtsne_exact", "ml/tsne", "Rtsne::Rtsne exact (theta 0), no PCA or normalisation, given Y_init: embedding after 1 and 10 iterations", function(ctx) {
  set.seed(20261117)
  n <- 60; d <- 5
  X <- matrix(rnorm(n * d), n, d); X[1:20, 1] <- X[1:20, 1] + 4; X[21:40, 2] <- X[21:40, 2] + 4
  Y0 <- matrix(rnorm(n * 2, sd = 1e-4), n, 2)
  b <- ctx$gen_dataset("tsne_blobs", data.frame(x1 = round(X[, 1], 10), x2 = round(X[, 2], 10), x3 = round(X[, 3], 10),
                                               x4 = round(X[, 4], 10), x5 = round(X[, 5], 10),
                                               y0_1 = signif(Y0[, 1], 12), y0_2 = signif(Y0[, 2], 12)))
  set.seed(20261118)
  n2 <- 300; d2 <- 20
  U <- matrix(rnorm(n2 * d2), n2, d2); V0 <- matrix(rnorm(n2 * 2, sd = 1e-4), n2, 2)
  ud <- as.data.frame(round(U, 10)); names(ud) <- paste0("u", 1:d2)
  ud$y0_1 <- signif(V0[, 1], 12); ud$y0_2 <- signif(V0[, 2], 12)
  u <- ctx$gen_dataset("tsne_unit", ud)
  run <- function(df, xcols, perp, K) {
    Rtsne(as.matrix(df[, xcols]), dims = 2, perplexity = perp, theta = 0, pca = FALSE, normalize = FALSE,
          Y_init = as.matrix(df[, c("y0_1", "y0_2")]), max_iter = K, eta = 200, exaggeration_factor = 12,
          stop_lying_iter = 250L, mom_switch_iter = 250L, momentum = 0.5, final_momentum = 0.8,
          check_duplicates = FALSE, num_threads = 1, verbose = FALSE)$Y
  }
  v <- list(
    blobs_y_iter1 = vec(run(b, paste0("x", 1:5), 10, 1)),       # column-major, 60 x 2
    blobs_y_iter10 = vec(run(b, paste0("x", 1:5), 10, 10)),
    unit_y_iter1 = vec(run(u, paste0("u", 1:d2), 30, 1))        # 300 x 2
  )
  list(values = v, dataset = "validation/datasets/generated/tsne_blobs.csv",
       reference = "Rtsne::Rtsne(theta = 0, pca = FALSE, normalize = FALSE, Y_init, eta = 200, exaggeration_factor = 12, stop_lying_iter = 250, mom_switch_iter = 250) (bhtsne exact path)",
       call = "Rtsne(X, dims = 2, perplexity = 10 | 30, theta = 0, pca = FALSE, normalize = FALSE, Y_init = cbind(y0_1, y0_2), max_iter = 1 | 10, eta = 200, ...)",
       packages = "Rtsne")
})
