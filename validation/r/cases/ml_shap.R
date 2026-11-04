# validation/r/cases/ml_shap.R -- exact Shapley references for SHAP (T3-WP48: ML-38, ML-99, ML-100,
# ML-139, ML-107). No SHAP package is a case dependency (treeshap, fastshap, kernelshap, shapr are
# missing), so the references are the definitions, written out and enumerated over every coalition:
#  * shap_tree_rpart: a regression tree grown by rpart::rpart (anova, cp = 0, maxdepth 4, no
#    competitors or surrogates) on a simulated dataset with an interaction, exported as a node table
#    (0-based ids in preorder, feature -1 for leaves, x[feature] <= threshold goes left, cover = rpart's
#    n per node). For each explained point x the game is the tree's path-dependent conditional
#    expectation v(S) = E[f(x) | x_S] of Lundberg, Erion & Lee (2020, Algorithm 1, EXPVALUE: follow x at
#    splits on features in S, otherwise weight both children by cover). Shapley values
#    phi_i = sum_{S not containing i} |S|! (M - |S| - 1)! / M! (v(S + i) - v(S)) and the Shapley
#    interaction index Phi_ij = sum_{S without i, j} |S|! (M - |S| - 2)! / (2 (M - 1)!) (v(S+i+j) -
#    v(S+i) - v(S+j) + v(S)), Phi_ii = phi_i - sum_{j != i} Phi_ij (Lundberg et al. 2020, eqs. 8-9),
#    by enumeration of all 2^M coalitions. These are what TreeSHAP (Algorithm 2) computes exactly.
#  * shap_kernel_sim: the interventional game v(S) = mean_b f(x_S, b_notS) over a background sample
#    (Lundberg & Lee 2017; Janzing, Minorics & Bloebaum 2020) for four fixed models, by enumeration;
#    Kernel SHAP with every coalition and the efficiency constraint (shap.KernelExplainer with
#    nsamples >= 2^M - 2) returns exactly these values. Rows 1-4 are explained, rows 5-20 are the
#    background. The models are written out identically in the Rust tests:
#      f1(x)  = exp(x0 / 2) + x0^2                                         (M = 1)
#      f4(x)  = x0 x1 x2 + 2 x3 + 0.5 x0^2                                 (M = 4)
#      f11(x) = x0 x1 x2 + sin(x3) x4 + x5 x6 x7 x8 + sum_{j=5..10} (j - 4)/4 x_j   (M = 11)
#      f12(x) = f11(x0..x10) + 0.5 x11 x0                                  (M = 12)
#    Matrices are stored row-major (row = explained point), interaction arrays as [point, i, j].

.shap_subset_values <- function(M, v) {
  # v(S) for every subset, indexed by bitmask + 1 (bit j - 1 set <=> feature j in S)
  vapply(0:(2^M - 1), function(mask) v(which(bitwAnd(mask, 2^(0:(M - 1))) > 0)), 0)
}
.shap_from_values <- function(M, vals) {
  sizes <- vapply(0:(2^M - 1), function(mask) sum(bitwAnd(mask, 2^(0:(M - 1))) > 0), 0)
  w <- ifelse(sizes < M, factorial(sizes) * factorial(pmax(M - sizes - 1, 0)) / factorial(M), 0)
  vapply(1:M, function(i) {
    b <- 2^(i - 1)
    without <- which(bitwAnd(0:(2^M - 1), b) == 0) - 1      # masks not containing i
    sum(w[without + 1] * (vals[without + b + 1] - vals[without + 1]))
  }, 0)
}
.shap_interactions_from_values <- function(M, vals, phi) {
  sizes <- vapply(0:(2^M - 1), function(mask) sum(bitwAnd(mask, 2^(0:(M - 1))) > 0), 0)
  Phi <- matrix(0, M, M)
  for (i in 1:M) for (j in 1:M) if (i != j) {
    bi <- 2^(i - 1); bj <- 2^(j - 1)
    S <- which(bitwAnd(0:(2^M - 1), bi + bj) == 0) - 1
    w <- factorial(sizes[S + 1]) * factorial(pmax(M - sizes[S + 1] - 2, 0)) / (2 * factorial(M - 1))
    Phi[i, j] <- sum(w * (vals[S + bi + bj + 1] - vals[S + bi + 1] - vals[S + bj + 1] + vals[S + 1]))
  }
  for (i in 1:M) Phi[i, i] <- phi[i] - sum(Phi[i, -i])
  Phi
}

case("shap_tree_rpart", "ml/shap",
     "TreeSHAP: exact path-dependent Shapley and interaction values of an rpart tree (enumeration)",
     function(ctx) {
  suppressPackageStartupMessages(library(rpart))
  set.seed(4801)
  n <- 80
  d <- data.frame(x0 = runif(n), x1 = runif(n), x2 = runif(n), x3 = runif(n))
  d$y <- 2 * d$x0 + 3 * d$x1 * d$x2 + sin(3 * d$x3) + rnorm(n, 0, 0.3)
  d <- ctx$gen_dataset("shap_tree_sim", d)
  feats <- c("x0", "x1", "x2", "x3"); M <- 4
  fit <- rpart(y ~ x0 + x1 + x2 + x3, data = d, method = "anova",
               control = rpart.control(cp = 0, minsplit = 8, minbucket = 3, maxdepth = 4, xval = 0,
                                       maxcompete = 0, maxsurrogate = 0))
  ff <- fit$frame; nodes <- as.integer(rownames(ff)); is_leaf <- ff$var == "<leaf>"
  sp <- fit$splits; stopifnot(nrow(sp) == sum(!is_leaf))
  k <- length(nodes); pos <- setNames(seq_len(k) - 1L, nodes)
  feature <- rep(-1L, k); threshold <- rep(0, k); left <- rep(-1L, k); right <- rep(-1L, k)
  isp <- 0L
  for (r in seq_len(k)) if (!is_leaf[r]) {
    isp <- isp + 1L
    feature[r] <- match(as.character(ff$var[r]), feats) - 1L
    threshold[r] <- sp[isp, "index"]
    l <- pos[[as.character(2L * nodes[r])]]; rr <- pos[[as.character(2L * nodes[r] + 1L)]]
    # rpart ncat = -1: x < index goes left; +1: x >= index goes left. Store as "x <= threshold
    # goes left" (no data point equals a midpoint split).
    if (sp[isp, "ncat"] == -1) { left[r] <- l; right[r] <- rr } else { left[r] <- rr; right[r] <- l }
  }
  tab <- data.frame(feature = feature, threshold = threshold, left = left, right = right,
                    value = ff$yval, cover = ff$n)
  predict_tab <- function(x) { j <- 1L
    while (tab$feature[j] >= 0) j <- 1L + if (x[tab$feature[j] + 1L] <= tab$threshold[j]) tab$left[j] else tab$right[j]
    tab$value[j] }
  expvalue <- function(x, S) {
    g <- function(j, w) {
      if (tab$feature[j] < 0) return(w * tab$value[j])
      a <- tab$left[j] + 1L; b <- tab$right[j] + 1L
      if ((tab$feature[j] + 1L) %in% S) return(g(if (x[tab$feature[j] + 1L] <= tab$threshold[j]) a else b, w))
      g(a, w * tab$cover[a] / tab$cover[j]) + g(b, w * tab$cover[b] / tab$cover[j])
    }
    g(1L, 1)
  }
  X <- as.matrix(d[, feats])
  pred <- apply(X, 1, predict_tab)
  stopifnot(max(abs(pred - predict(fit, d))) == 0)
  # a feature repeats on some root-to-leaf path (exercises UNWIND)
  rep_path <- FALSE
  walk <- function(j, seen) { if (tab$feature[j + 1] < 0) return(invisible())
    if (tab$feature[j + 1] %in% seen) rep_path <<- TRUE
    walk(tab$left[j + 1], c(seen, tab$feature[j + 1])); walk(tab$right[j + 1], c(seen, tab$feature[j + 1])) }
  walk(0L, integer(0)); stopifnot(rep_path)
  ev <- sum(tab$value[tab$feature < 0] * tab$cover[tab$feature < 0]) / tab$cover[1]
  phi <- matrix(0, n, M); inter <- array(0, c(10, M, M))
  for (i in 1:n) {
    vals <- .shap_subset_values(M, function(S) expvalue(X[i, ], S))
    phi[i, ] <- .shap_from_values(M, vals)
    if (i <= 10) inter[i, , ] <- .shap_interactions_from_values(M, vals, phi[i, ])
  }
  stopifnot(max(abs(rowSums(phi) + ev - pred)) < 1e-12)
  x_new <- rbind(c(-0.5, 0.5, 0.5, 0.5), c(1.5, 1.5, 1.5, 1.5), c(0.6051048, 0.9, 0.2, 0.15),
                 c(0.1, 0.1, 0.9, 0.9), c(0.75, 0.62, 0.85, 0.5))
  phi_new <- t(apply(x_new, 1, function(x) .shap_from_values(M, .shap_subset_values(M, function(S) expvalue(x, S)))))
  list(values = list(
         node_feature = vec(tab$feature), node_threshold = vec(tab$threshold),
         node_left = vec(tab$left), node_right = vec(tab$right),
         node_value = vec(tab$value), node_cover = vec(tab$cover),
         expected_value = ev, pred = vec(pred),
         phi = vec(as.vector(t(phi))),                       # row-major n x M
         interactions = vec(aperm(inter, c(3, 2, 1))),       # [point, i, j], j fastest
         x_new = vec(as.vector(t(x_new))),
         pred_new = vec(apply(x_new, 1, predict_tab)),
         phi_new = vec(as.vector(t(phi_new)))),
       reference = "Shapley values and Shapley interaction index of the path-dependent game E[f(x) | x_S] (Lundberg, Erion & Lee 2020, Algorithm 1 and eqs. 8-9) by enumeration of all coalitions; tree from rpart::rpart",
       call = "rpart(y ~ x0 + x1 + x2 + x3, method = 'anova', control = rpart.control(cp = 0, minsplit = 8, minbucket = 3, maxdepth = 4, xval = 0, maxcompete = 0, maxsurrogate = 0))",
       packages = c("rpart"))
})

case("shap_kernel_sim", "ml/shap",
     "Kernel SHAP: exact interventional Shapley values of four fixed models (M = 1, 4, 11, 12; enumeration)",
     function(ctx) {
  set.seed(4802)
  n <- 20
  X <- matrix(rnorm(n * 12), n, 12, dimnames = list(NULL, paste0("x", 0:11)))
  d <- ctx$gen_dataset("shap_kernel_sim", as.data.frame(X))
  X <- as.matrix(d)
  f1 <- function(Z) exp(Z[, 1] / 2) + Z[, 1]^2
  f4 <- function(Z) Z[, 1] * Z[, 2] * Z[, 3] + 2 * Z[, 4] + 0.5 * Z[, 1]^2
  f11 <- function(Z) Z[, 1] * Z[, 2] * Z[, 3] + sin(Z[, 4]) * Z[, 5] + Z[, 6] * Z[, 7] * Z[, 8] * Z[, 9] +
    drop(Z[, 6:11, drop = FALSE] %*% ((1:6) / 4))
  f12 <- function(Z) f11(Z[, 1:11, drop = FALSE]) + 0.5 * Z[, 12] * Z[, 1]
  ex <- X[1:4, , drop = FALSE]; bg <- X[5:20, , drop = FALSE]
  game <- function(f, x, B) function(S) { Z <- B; if (length(S)) Z[, S] <- matrix(x[S], nrow(B), length(S), byrow = TRUE); mean(f(Z)) }
  run <- function(f, M) {
    B <- bg[, 1:M, drop = FALSE]
    phi <- t(vapply(1:4, function(i) .shap_from_values(M, .shap_subset_values(M, game(f, ex[i, 1:M], B))), numeric(M)))
    if (M == 1) phi <- t(phi)
    list(phi = phi, base = mean(f(B)), fx = f(ex[, 1:M, drop = FALSE]))
  }
  r1 <- run(f1, 1); r4 <- run(f4, 4); r11 <- run(f11, 11); r12 <- run(f12, 12)
  for (r in list(r1, r4, r11, r12)) stopifnot(max(abs(rowSums(r$phi) + r$base - r$fx)) < 1e-10)
  vals <- list()
  for (nm in c("m1", "m4", "m11", "m12")) {
    r <- get(sub("m", "r", nm))
    vals[[paste0(nm, "_phi")]] <- vec(as.vector(t(r$phi)))
    vals[[paste0(nm, "_base")]] <- r$base
    vals[[paste0(nm, "_fx")]] <- vec(r$fx)
  }
  list(values = vals,
       reference = "exact interventional Shapley values v(S) = mean_b f(x_S, b_notS) over rows 5-20 (Lundberg & Lee 2017; Janzing et al. 2020) by enumeration of all coalitions; equal to Kernel SHAP with every coalition (shap.KernelExplainer, nsamples >= 2^M - 2)",
       call = "written out (no SHAP package is a case dependency); models f1, f4, f11, f12 in the file header",
       packages = c("base"))
})
