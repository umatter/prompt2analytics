# validation/r/cases/ml_boosting_t3.R -- boosting and BART references (T3-WP46: ML-43, ML-63, ML-136,
# ML-106, WT-13). xgboost, lightgbm, mboost, BART and adabag are NOT case dependencies; every reference here is
# either an installed R function (rpart, glm, lm, integrate) or a published algorithm written out in
# plain R, named in the case's `reference`:
#  * ml_adaboost_samme  -- AdaBoost.M1 (Freund & Schapire 1997) and SAMME (Zhu, Zou, Rosset & Hastie 2009,
#    Algorithm 2; adabag::boosting(coeflearn = "Zhu", boos = FALSE)) with rpart weak learners
#    (method "class"/"anova", case weights, maxdepth = d, minsplit = 2, minbucket = 1, cp = 0, xval = 0,
#    no competitors/surrogates);
#  * ml_xgboost_l1      -- XGBoost exact greedy (Chen & Guestrin 2016, Alg. 1) and the histogram variant of
#    p2a's xgboost_fast, leaf weight -ThresholdL1(G, alpha) / (H + lambda) and the gain on ThresholdL1(G)
#    (xgboost src/tree/param.h CalcWeight / CalcGain), written out in R;
#  * ml_mboost_bols     -- mboost(y ~ bols(x1) + bols(x2) + bols(x3), family, boost_control(mstop, nu))
#    written out in R from mboost 2.9 R/family.R (Gaussian, Binomial() = type "adaboost", Binomial(type =
#    "glm"), Poisson), plus the converged limits lm / glm / glm / 2; a componentwise least-squares tree
#    base learner on one variable (p2a's `Tree` learner; not mboost's ctree-based btree);
#  * ml_bart_probit_single_leaf -- probit BART (BART::pbart, Albert & Chib 1993 data augmentation) with
#    trees that cannot split: the exact posterior of Phi(binaryOffset + f), f ~ N(0, 9 / k^2), by quadrature;
#  * ml_bart_probit_glm -- glm(binomial(link = "probit")) fitted probabilities on a linear-index DGP.

.thr <- function(G, a) if (G > a) G - a else if (G < -a) G + a else 0
.xgb_exact <- function(X, y, M, md, eta, lam, a, gam = 0, mcw = 1) {
  wopt <- function(G, H) if (H < 1e-10) 0 else -.thr(G, a) / (H + lam)
  sc <- function(G, H) if (H > 0) .thr(G, a)^2 / (H + lam) else 0
  build <- function(g, idx, depth) {
    G <- sum(g[idx]); H <- length(idx)
    if (depth >= md || length(idx) <= 1 || H < mcw) return(list(leaf = TRUE, w = wopt(G, H)))
    best <- -Inf; bs <- NULL
    for (j in seq_len(ncol(X))) {
      o <- idx[order(X[idx, j])]; xs <- X[o, j]; GL <- 0; HL <- 0
      for (i in seq_len(length(o) - 1)) {
        GL <- GL + g[o[i]]; HL <- HL + 1
        if (abs(xs[i + 1] - xs[i]) < 1e-10) next
        GR <- G - GL; HR <- H - HL
        if (HL < mcw || HR < mcw) next
        gn <- 0.5 * (sc(GL, HL) + sc(GR, HR) - sc(G, H)) - gam
        if (gn > best) { best <- gn; bs <- list(j = j, t = (xs[i] + xs[i + 1]) / 2, L = o[1:i], R = o[(i + 1):length(o)]) }
      }
    }
    if (best <= 0 || is.null(bs)) return(list(leaf = TRUE, w = wopt(G, H)))
    list(leaf = FALSE, j = bs$j, t = bs$t, gain = best, l = build(g, bs$L, depth + 1), r = build(g, bs$R, depth + 1))
  }
  pt <- function(tr, x) { while (!tr$leaf) tr <- if (x[tr$j] <= tr$t) tr$l else tr$r; tr$w }
  f <- rep(mean(y), nrow(X)); trees <- list()
  for (m in seq_len(M)) { tr <- build(f - y, seq_along(y), 0); f <- f + eta * apply(X, 1, function(x) pt(tr, x)); trees[[m]] <- tr }
  list(f = f, root = trees[[1]])
}
.xgb_fast <- function(X, y, M, md, eta, lam, a, gam = 0, mcw = 1) {
  bins_of <- function(v) {
    n <- length(v); s <- sort(v); nb <- min(256, n); edges <- numeric(0)
    for (i in seq_len(nb - 1)) { val <- s[min(floor(i * n / nb), n - 1) + 1]; if (!length(edges) || val > tail(edges, 1)) edges <- c(edges, val) }
    list(edges = edges, bins = sapply(v, function(x) { k <- which(x <= edges); if (length(k)) k[1] - 1 else length(edges) }))
  }
  B <- lapply(seq_len(ncol(X)), function(j) bins_of(X[, j]))
  grow <- function(g, idx, depth) {
    G <- sum(g[idx]); H <- length(idx); w <- -.thr(G, a) / (H + lam)
    if (depth >= md || length(idx) < 2 || H < mcw) return(list(leaf = TRUE, w = w))
    best <- NULL; bg <- -Inf
    for (j in seq_len(ncol(X))) {
      nb <- length(B[[j]]$edges); bj <- B[[j]]$bins[idx]
      gs <- sapply(0:nb, function(k) sum(g[idx][bj == k])); hs <- sapply(0:nb, function(k) sum(bj == k))
      lg <- 0; lh <- 0; jb <- gam; jbin <- NA
      for (k in 0:(nb - 1)) {
        lg <- lg + gs[k + 1]; lh <- lh + hs[k + 1]; rg <- G - lg; rh <- H - lh
        if (lh < mcw || rh < mcw) next
        gn <- 0.5 * (.thr(lg, a)^2 / (lh + lam) + .thr(rg, a)^2 / (rh + lam) - .thr(G, a)^2 / (H + lam))
        if (gn > jb) { jb <- gn; jbin <- k }
      }
      if (!is.na(jbin) && (is.null(best) || jb > bg)) { bg <- jb; best <- list(j = j, bin = jbin) }
    }
    if (is.null(best)) return(list(leaf = TRUE, w = w))
    L <- idx[B[[best$j]]$bins[idx] <= best$bin]; R <- idx[B[[best$j]]$bins[idx] > best$bin]
    list(leaf = FALSE, j = best$j, bin = best$bin, l = grow(g, L, depth + 1), r = grow(g, R, depth + 1))
  }
  pr <- function(tr, i) { while (!tr$leaf) tr <- if (B[[tr$j]]$bins[i] <= tr$bin) tr$l else tr$r; tr$w }
  f <- rep(mean(y), nrow(X)); root <- NULL
  for (m in seq_len(M)) { tr <- grow(f - y, seq_along(y), 0); if (m == 1) root <- tr; f <- f + eta * sapply(seq_along(y), function(i) pr(tr, i)) }
  list(f = f, root = root)
}

case("ml_xgboost_l1", "ml/xgboost", "XGBoost exact and histogram trees with the L1 (alpha) leaf weight and gain", function(ctx) {
  set.seed(4637)
  n <- 16
  x1 <- round(runif(n), 6); x2 <- round(rnorm(n), 6)
  y <- 1.5 * x1 + 0.4 * x2 + rnorm(n, sd = 0.25); y[3] <- y[3] + 3
  d <- ctx$gen_dataset("xgb_small", data.frame(x1, x2, y))
  X <- cbind(d$x1, d$x2)
  e0 <- .xgb_exact(X, d$y, 3, 2, 0.3, 1, 0); e3 <- .xgb_exact(X, d$y, 3, 2, 0.3, 1, 3)
  f0 <- .xgb_fast(X, d$y, 1, 1, 0.3, 1, 0); f3 <- .xgb_fast(X, d$y, 1, 1, 0.3, 1, 3)
  g0 <- .xgb_fast(X, d$y, 4, 2, 0.3, 1, 0); g3 <- .xgb_fast(X, d$y, 4, 2, 0.3, 1, 3)
  stopifnot(e0$root$j != e3$root$j)                 # alpha in the gain moves the first split
  list(values = list(
         exact_a0_pred = vec(e0$f), exact_a3_pred = vec(e3$f),
         exact_a0_root = vec(c(e0$root$j - 1, e0$root$t, e0$root$gain)),
         exact_a3_root = vec(c(e3$root$j - 1, e3$root$t, e3$root$gain)),
         fast_a0_stump = vec(c(f0$root$j - 1, f0$root$bin, f0$root$l$w, f0$root$r$w)),
         fast_a3_stump = vec(c(f3$root$j - 1, f3$root$bin, f3$root$l$w, f3$root$r$w)),
         fast_a0_pred4 = vec(g0$f), fast_a3_pred4 = vec(g3$f)),
       reference = "XGBoost exact greedy (Chen & Guestrin 2016, Alg. 1; eq. 5 and 7) and p2a's histogram variant (bins at the sorted values, x <= edge) with xgboost's CalcWeight = -ThresholdL1(G, alpha) / (H + lambda) and CalcGain = ThresholdL1(G, alpha)^2 / (H + lambda) (src/tree/param.h), written out in R; xgboost is not a case dependency",
       call = "reg:squarederror, base_score = mean(y), eta = 0.3, lambda = 1, alpha in {0, 3}, gamma = 0, min_child_weight = 1; exact: 3 rounds, max_depth 2; histogram: 1 round max_depth 1 and 4 rounds max_depth 2",
       packages = character(0))
})

