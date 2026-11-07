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

.mb_ngrad <- list(
  gaussian = function(y, f) y - f,
  binomial = function(y, f) { yt <- 2 * y - 1; e <- exp(-2 * yt * f); -(-2 * yt * e) / (log(2) * (1 + e)) },
  binomial_glm = function(y, f) { p <- plogis(f); dlogis(f) * (y - p) / (p * (1 - p)) },
  poisson = function(y, f) y - exp(f))
.mb_off <- list(
  gaussian = function(y) mean(y),
  binomial = function(y) { p <- mean(y > 0); 1/2 * log(p / (1 - p)) },
  binomial_glm = function(y) qlogis(mean((y + 0.5) / 2)),
  poisson = function(y) log(mean(y)))
.mb_resp <- list(gaussian = identity,
  binomial = function(f) { f <- pmin(abs(f), 36) * sign(f); exp(f) / (exp(f) + exp(-f)) },
  binomial_glm = plogis, poisson = exp)
.mb_bols <- function(X, y, fam, mstop, nu) {
  n <- nrow(X); p <- ncol(X); f0 <- .mb_off[[fam]](y); f <- rep(f0, n); slope <- rep(0, p); icpt <- f0; sel <- rep(0, p); step1 <- NULL
  for (m in seq_len(mstop)) {
    u <- .mb_ngrad[[fam]](y, f); best <- Inf
    for (j in 1:p) {
      xc <- X[, j] - mean(X[, j]); b <- sum(xc * (u - mean(u))) / sum(xc^2); a <- mean(u) - b * mean(X[, j])
      rss <- sum((u - a - b * X[, j])^2)
      if (rss < best) { best <- rss; bj <- j; bb <- b; ba <- a }
    }
    if (m == 1) step1 <- c(bj - 1, ba, bb, best)
    f <- f + nu * (ba + bb * X[, bj]); slope[bj] <- slope[bj] + nu * bb; icpt <- icpt + nu * ba; sel[bj] <- sel[bj] + 1
  }
  list(slope = slope, intercept = icpt, resp = .mb_resp[[fam]](f), sel = sel, offset = f0, step1 = step1)
}
# p2a's componentwise least-squares tree learner on one variable (depth d, children >= minn rows)
.mb_tree <- function(X, y, mstop, nu, depth, minn) {
  n <- nrow(X); f <- rep(mean(y), n)
  fit1 <- function(x, u) {
    o <- order(x); xs <- x[o]; us <- u[o]; pred <- numeric(n)
    node <- function(lo, hi, dep) {         # rows o[lo..hi] (1-based, inclusive)
      m <- hi - lo + 1; tot <- sum(us[lo:hi])
      best <- -Inf; bi <- NA; ls <- 0
      if (dep < depth) for (i in lo:(hi - 1)) {
        ls <- ls + us[i]; ln <- i - lo + 1; rn <- m - ln
        if (ln < minn || rn < minn) next
        if (abs(xs[i + 1] - xs[i]) < 1e-10) next
        g <- ls^2 / ln + (tot - ls)^2 / rn
        if (g > best) { best <- g; bi <- i }
      }
      if (is.na(bi)) { mu <- tot / m; pred[o[lo:hi]] <<- mu; return(sum((us[lo:hi] - mu)^2)) }
      node(lo, bi, dep + 1) + node(bi + 1, hi, dep + 1)
    }
    rss <- node(1, n, 0); list(rss = rss, pred = pred)
  }
  sel <- rep(0, ncol(X))
  for (m in seq_len(mstop)) {
    u <- y - f; best <- Inf
    for (j in seq_len(ncol(X))) { ft <- fit1(X[, j], u); if (ft$rss < best) { best <- ft$rss; bp <- ft$pred; bj <- j } }
    f <- f + nu * bp; sel[bj] <- sel[bj] + 1
  }
  list(f = f, sel = sel)
}

case("ml_mboost_bols", "ml/mboost", "Componentwise boosting with bols learners (Gaussian, Binomial adaboost and glm, Poisson), converged limits, tree learner depth", function(ctx) {
  set.seed(4636)
  n <- 200
  x1 <- runif(n); x2 <- runif(n); x3 <- runif(n) - 0.5
  yg <- 2 * x1 + 0.5 * x2 + rnorm(n, sd = 0.3)
  yb <- rbinom(n, 1, plogis(-0.4 + 1.6 * x1 - 1.1 * x2))
  yc <- rpois(n, exp(0.2 + 0.9 * x1 - 0.6 * x2))
  d <- ctx$gen_dataset("mboost_sim", data.frame(x1, x2, x3, yg, yb, yc))
  X <- cbind(d$x1, d$x2, d$x3)
  vals <- list()
  for (fam in c("gaussian", "binomial", "binomial_glm", "poisson")) {
    y <- switch(fam, gaussian = d$yg, binomial = d$yb, binomial_glm = d$yb, poisson = d$yc)
    r <- .mb_bols(X, y, fam, 100, 0.1); rc <- .mb_bols(X, y, fam, 2000, 0.5)
    vals[[paste0(fam, "_offset")]] <- r$offset
    vals[[paste0(fam, "_step1")]] <- vec(r$step1)
    vals[[paste0(fam, "_m100_coef")]] <- vec(r$slope)
    vals[[paste0(fam, "_m100_intercept")]] <- r$intercept
    vals[[paste0(fam, "_m100_sel")]] <- vec(r$sel)
    vals[[paste0(fam, "_m100_resp")]] <- vec(r$resp)
    vals[[paste0(fam, "_m2000_coef")]] <- vec(c(rc$intercept, rc$slope))
  }
  gl <- coef(glm(yb ~ x1 + x2 + x3, data = d, family = binomial(), control = glm.control(epsilon = 1e-14, maxit = 100)))
  gp <- coef(glm(yc ~ x1 + x2 + x3, data = d, family = poisson(), control = glm.control(epsilon = 1e-14, maxit = 100)))
  gg <- coef(lm(yg ~ x1 + x2 + x3, data = d))
  vals$gaussian_limit <- named(gg); vals$binomial_limit <- named(gl / 2); vals$binomial_glm_limit <- named(gl); vals$poisson_limit <- named(gp)
  t1 <- .mb_tree(X, d$yg, 30, 0.1, 1, 5); t2 <- .mb_tree(X, d$yg, 30, 0.1, 2, 5)
  stopifnot(max(abs(t1$f - t2$f)) > 1e-3)
  vals$tree_d1_m30_pred <- vec(t1$f); vals$tree_d2_m30_pred <- vec(t2$f); vals$tree_d2_m30_sel <- vec(t2$sel)
  list(values = vals,
       reference = "mboost::mboost(y ~ bols(x1) + bols(x2) + bols(x3), family = Gaussian() | Binomial() | Binomial(type = 'glm') | Poisson(), control = boost_control(mstop, nu)) written out in R from mboost 2.9 R/family.R (ngradient, offset, response); limits from stats::lm / stats::glm (epsilon 1e-14); Binomial() limit = glm logit coefficients / 2; tree learner = p2a's least-squares tree on one variable; mboost is not a case dependency",
       call = "bols(xj) = OLS on [1, xj], selection by RSS; mstop = 100, nu = 0.1 (paths), mstop = 2000, nu = 0.5 (limits); tree: mstop = 30, nu = 0.1, depth 1 and 2, children >= 5 rows",
       packages = character(0))
})

.ada_ctl <- function(md) rpart::rpart.control(maxdepth = md, minsplit = 2, minbucket = 1, cp = 0, xval = 0,
                                               maxcompete = 0, maxsurrogate = 0, usesurrogate = 0)
.ada_tree <- function(d, y, w, md, method) {
  dd <- data.frame(yy = if (method == "class") factor(y) else y, x1 = d$x1, x2 = d$x2, x3 = d$x3)
  f <- rpart::rpart(yy ~ x1 + x2 + x3, data = dd, weights = w, method = method, control = .ada_ctl(md))
  if (method == "class") as.numeric(as.character(predict(f, dd, type = "class"))) else unname(predict(f, dd))
}
.samme <- function(d, y, M, md, lr) {
  K <- length(unique(y)); n <- length(y); w <- rep(1 / n, n); alpha <- err <- numeric(0); H <- list(); w1 <- NULL
  for (m in seq_len(M)) {
    h <- .ada_tree(d, y, w, md, "class"); miss <- as.numeric(h != y); e <- sum(w * miss) / sum(w)
    if (e >= 1 - 1 / K) break
    ec <- min(max(e, 1e-10), 1 - 1e-10); a <- lr * (log((1 - ec) / ec) + log(K - 1))
    w <- w * exp(a * miss); w <- w / sum(w); if (m == 1) w1 <- w
    alpha <- c(alpha, a); err <- c(err, e); H[[m]] <- h
  }
  cls <- sort(unique(y))
  sc <- sapply(cls, function(k) Reduce(`+`, Map(function(a, h) a * (h == k), alpha, H)))
  pred <- cls[apply(sc, 1, which.max)]
  list(alpha = alpha, err = err, w1 = w1, pred = pred, acc = mean(pred == y))
}
.m1 <- function(d, y, M, md, lr) {
  n <- length(y); w <- rep(1 / n, n); alpha <- err <- numeric(0); score <- rep(0, n)
  for (m in seq_len(M)) {
    h <- .ada_tree(d, y, w, md, "class"); miss <- as.numeric(h != y); e <- sum(w * miss) / sum(w)
    if (e >= 0.5) break
    ec <- min(max(e, 1e-10), 1 - 1e-10); a <- lr * 0.5 * log((1 - ec) / ec)
    score <- score + a * h; w <- w * exp(a * ifelse(miss == 1, 1, -1)); w <- w / sum(w)
    alpha <- c(alpha, a); err <- c(err, e)
  }
  list(alpha = alpha, err = err, score = score)
}

case("ml_adaboost_samme", "ml/adaboost", "AdaBoost.M1 and SAMME with depth-limited rpart weak learners; the first AdaBoost.R2 learner", function(ctx) {
  set.seed(46)
  n <- 120
  x1 <- runif(n); x2 <- runif(n); x3 <- rnorm(n)
  s <- 2.2 * x1 - 1.5 * x2 + rnorm(n, sd = 0.45)
  y3 <- cut(s, quantile(s, c(0, 1/3, 2/3, 1)), include.lowest = TRUE, labels = FALSE) - 1
  yb <- ifelse(s + rnorm(n, sd = 0.3) > median(s), 1, -1)
  yr <- 1.3 * x1 + x2 * x3 + rnorm(n, sd = 0.3)
  d <- ctx$gen_dataset("adaboost_sim", data.frame(x1, x2, x3, y3, yb, yr))
  s1 <- .samme(d, d$y3, 8, 1, 1); s2 <- .samme(d, d$y3, 8, 2, 1); s25 <- .samme(d, d$y3, 8, 2, 0.5)
  m1 <- .m1(d, d$yb, 8, 1, 1); m3 <- .m1(d, d$yb, 8, 3, 1)
  k2 <- .samme(d, d$yb, 8, 1, 1)
  stopifnot(length(s1$alpha) == 8, length(s2$alpha) == 8, length(m1$alpha) == 8, length(m3$alpha) == 8,
            isTRUE(all.equal(k2$alpha, 2 * m1$alpha, tolerance = 1e-14)),
            all(k2$pred == ifelse(m1$score >= 0, 1, -1)),          # SAMME with K = 2 is M1
            any(s1$pred != s2$pred), !isTRUE(all.equal(m1$score, m3$score)))   # depth matters
  n1 <- rep(1 / n, n)
  list(values = list(
         samme_d1_alpha = vec(s1$alpha), samme_d1_err = vec(s1$err), samme_d1_w1 = vec(s1$w1),
         samme_d1_pred = vec(s1$pred), samme_d1_acc = s1$acc,
         samme_d2_alpha = vec(s2$alpha), samme_d2_err = vec(s2$err), samme_d2_pred = vec(s2$pred), samme_d2_acc = s2$acc,
         samme_d2_lr05_alpha = vec(s25$alpha), samme_d2_lr05_pred = vec(s25$pred),
         m1_d1_alpha = vec(m1$alpha), m1_d1_score = vec(m1$score),
         m1_d3_alpha = vec(m3$alpha), m1_d3_score = vec(m3$score),
         r2_tree1_d1 = vec(.ada_tree(d, d$yr, n1, 1, "anova")), r2_tree1_d2 = vec(.ada_tree(d, d$yr, n1, 2, "anova"))),
       reference = "SAMME (Zhu et al. 2009, Alg. 2; adabag::boosting(coeflearn = 'Zhu', boos = FALSE)) and AdaBoost.M1 (Freund & Schapire 1997) written out in R with rpart::rpart weak learners (case weights, maxdepth = d, minsplit = 2, minbucket = 1, cp = 0); adabag is not a case dependency",
       call = "rpart(y ~ x1 + x2 + x3, weights = w, method = 'class', control = rpart.control(maxdepth = d, minsplit = 2, minbucket = 1, cp = 0, xval = 0, maxcompete = 0, maxsurrogate = 0, usesurrogate = 0)); alpha = lr * (log((1 - e) / e) + log(K - 1)); w <- w * exp(alpha * miss)",
       packages = c("rpart"))
})

