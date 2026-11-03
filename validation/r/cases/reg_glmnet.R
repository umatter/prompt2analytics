# validation/r/cases/reg_glmnet.R -- glmnet references (T3-WP07: REG-41, REG-42; moved in REG-6,
# REG-7, REG-40; REG-8 core part, REG-44 glmnet part). This package copy supersedes the T3 design's
# Appendix A.3 (same data draws and keys, plus the keys below).
#
#  * gaussian glmnet, alpha 1 / 0.5 / 0, standardize TRUE / FALSE, intercept TRUE / FALSE, at
#    lambda = 1, 0.3, 0.1, 0.03 (glmnet 5.1, thresh 1e-14);
#  * `*_tight`: the exact minimiser of glmnet's objective, solved from the KKT conditions on the active
#    set that glmnet returns (an active-set refinement loop that stops only when every KKT condition
#    holds to 1e-12). glmnet's own coordinate descent stops at max_j xv_j * delta_j^2 < thresh, i.e.
#    coefficient changes of about 1e-7 standardised units, so its reported coefficients are within
#    1.0e-7 of the minimiser (absolute, measured on these data); the tight values are what a converged solver
#    must reproduce to 1e-9;
#  * the default lasso path (fdev / devmax stopping), the ridge and elastic-net lambda_max, the n < p
#    lambda.min.ratio rule, dfmax, nlambda = 1;
#  * binomial lasso / elastic net, standardize TRUE / FALSE, intercept TRUE / FALSE, and its default path;
#  * cv.glmnet with a fixed foldid (gaussian: 5 equal folds; binomial: 7 unequal folds, so the
#    fold-size weights of cvm / cvsd matter);
#  * KKT data (K1..K30, yk; AR(0.85) design): the default lasso path where the pre-WP07 active-set
#    coordinate descent leaves variables out (REG-41), with the exact path solutions.
suppressPackageStartupMessages(library(glmnet))

.glm_sim <- function(ctx) {
  set.seed(20261122)
  n <- 150; p <- 8
  X <- matrix(rnorm(n * p), n, p); X[, 2] <- X[, 1] * 0.6 + rnorm(n, 0, 0.8); X[, 5] <- X[, 5] * 3 + 2
  y <- 1 + 2 * X[, 1] - 1.5 * X[, 3] + 0.5 * X[, 5] + rnorm(n, 0, 1.5)
  yb <- as.integer(runif(n) < plogis(-0.5 + X[, 1] - 0.8 * X[, 3]))
  foldid <- rep_len(1:5, n)[sample.int(n)]
  foldid7 <- (seq_len(n) %% 7) + 1                       # 7 folds of 21 or 22 rows (no RNG)
  set.seed(20261130)                                     # KKT block (REG-41)
  pk <- 30
  K <- matrix(0, n, pk); K[, 1] <- rnorm(n)
  for (j in 2:pk) K[, j] <- 0.85 * K[, j - 1] + sqrt(1 - 0.85^2) * rnorm(n)
  bk <- c(1.5, -1, 0.8, 0.5, -0.5, 0.3, -0.3, 0.2, rep(0, pk - 8))[sample.int(pk)]
  yk <- drop(K %*% bk) + rnorm(n, 0, 1)
  colnames(K) <- paste0("K", 1:pk)
  ctx$gen_dataset("glmnet_sim", data.frame(y = round(y, 10), yb = yb, round(X, 10), foldid = foldid,
                                           foldid7 = foldid7, round(K, 10), yk = round(yk, 10)))
}

# glmnet's standardisation (glmnetpp standard1 / lstandard1, weights 1/n):
#   intercept: x centred; scale = population sd; y centred.
#   no intercept: x NOT centred, scale = population sd of the centred column; y not centred.
#   standardize = FALSE: scale 1.  Gaussian ys (the ridge-penalty scale): population sd of y with an
#   intercept, sqrt(mean(y^2)) without one.
.gn_std <- function(X, standardize, intercept) {
  m <- colMeans(X); s <- if (standardize) sqrt(colMeans(sweep(X, 2, m)^2)) else rep(1, ncol(X))
  ctr <- if (intercept) m else rep(0, ncol(X))
  list(Z = sweep(sweep(X, 2, ctr), 2, s, "/"), ctr = ctr, s = s)
}
.gn_vec <- function(f) as.numeric(as.matrix(coef(f)))   # (p + 1) x nlambda, column-major, intercept first

# Exact gaussian elastic-net minimiser of glmnet's objective (in y units, Z the standardised design):
#   (1/2n) ||yt - Z B||^2 + lambda * (alpha ||B||_1 + (1 - alpha) / (2 ys) ||B||^2)
# KKT: Z_A'(yt - Z B)/n - lambda (1 - alpha)/ys B_A = lambda alpha sign(B_A); |Z_j'(yt - Z B)/n| <= lambda alpha off A.
.gn_gauss_tight <- function(X, y, lambda, alpha, standardize, intercept, b_start) {
  n <- nrow(X); st <- .gn_std(X, standardize, intercept); Z <- st$Z
  ym <- if (intercept) mean(y) else 0; yt <- y - ym; ys <- if (intercept) sqrt(mean((y - mean(y))^2)) else sqrt(mean(y^2))
  l2 <- lambda * (1 - alpha) / ys; l1 <- lambda * alpha
  B <- b_start * st$s                                    # raw -> standardised scale
  A <- which(B != 0); sgn <- sign(B)
  if (alpha == 0) A <- seq_len(ncol(X))
  for (it in 1:50) {
    B <- rep(0, ncol(X))
    if (length(A)) {
      ZA <- Z[, A, drop = FALSE]
      B[A] <- solve(crossprod(ZA) / n + diag(l2, length(A)), crossprod(ZA, yt) / n - l1 * sgn[A])
    }
    g <- drop(crossprod(Z, yt - drop(Z %*% B))) / n
    flip <- A[sign(B[A]) != sgn[A] & alpha > 0]
    enter <- setdiff(which(abs(g) > l1 * (1 + 1e-12)), A)
    if (!length(flip) && !length(enter)) break
    A <- sort(c(setdiff(A, flip), enter)); sgn[enter] <- sign(g[enter])
  }
  inact <- setdiff(seq_len(ncol(X)), A)
  stopifnot(it < 50, all(abs(g[inact]) <= l1 * (1 + 1e-12)),
            all(abs(g[A] - l2 * B[A] - l1 * sign(B[A])) <= 1e-12 * max(1, l1)))
  beta <- B / st$s
  c(if (intercept) ym - sum(st$ctr * beta) else 0, beta)
}

# Exact binomial elastic-net minimiser of glmnet's objective (y not standardised):
#   -(1/n) sum(y eta - log(1 + e^eta)) + lambda (alpha ||b||_1 + (1 - alpha)/2 ||b||^2), eta = b0 + Z b
.gn_binom_tight <- function(X, y, lambda, alpha, standardize, intercept, coef_start) {
  n <- nrow(X); p <- ncol(X); st <- .gn_std(X, standardize, intercept); Z <- st$Z
  b <- coef_start[-1] * st$s; b0 <- if (intercept) coef_start[1] + sum(st$ctr * coef_start[-1]) else 0
  A <- which(b != 0); sgn <- sign(b); l1 <- lambda * alpha; l2 <- lambda * (1 - alpha)
  for (outer in 1:50) {
    for (it in 1:200) {                                # Newton on (b0, b_A) with fixed signs
      eta <- b0 + drop(Z[, A, drop = FALSE] %*% b[A]); pr <- 1 / (1 + exp(-eta)); w <- pr * (1 - pr)
      W <- if (intercept) cbind(1, Z[, A, drop = FALSE]) else Z[, A, drop = FALSE]
      pen <- c(if (intercept) 0, rep(l2, length(A)))
      grad <- -drop(crossprod(W, y - pr)) / n + pen * c(if (intercept) b0, b[A]) + c(if (intercept) 0, l1 * sgn[A])
      if (!ncol(W)) break
      H <- crossprod(W, W * w) / n + diag(pen, ncol(W))
      step <- solve(H, grad)
      if (intercept) { b0 <- b0 - step[1]; b[A] <- b[A] - step[-1] } else b[A] <- b[A] - step
      if (max(abs(step)) < 1e-15) break
    }
    eta <- b0 + drop(Z %*% b); pr <- 1 / (1 + exp(-eta))
    g <- drop(crossprod(Z, y - pr)) / n
    flip <- A[sign(b[A]) != sgn[A] & alpha > 0]
    enter <- setdiff(which(abs(g) > l1 * (1 + 1e-12)), A)
    if (!length(flip) && !length(enter)) break
    b[flip] <- 0; A <- sort(c(setdiff(A, flip), enter)); sgn[enter] <- sign(g[enter])
  }
  inact <- setdiff(seq_len(p), A)
  stopifnot(outer < 50, all(abs(g[inact]) <= l1 * (1 + 1e-12)),
            all(abs(g[A] - l2 * b[A] - l1 * sign(b[A])) <= 1e-11 * max(1, l1)),
            !intercept || abs(sum(y - pr)) / n < 1e-12)
  beta <- b / st$s
  c(if (intercept) b0 - sum(st$ctr * beta) else 0, beta)
}
.gn_tight_path <- function(fit, X, y, lambdas, alpha, standardize, intercept, family = "gaussian") {
  cm <- as.matrix(coef(fit))
  unlist(lapply(seq_along(lambdas), function(k)
    if (family == "gaussian") .gn_gauss_tight(X, y, lambdas[k], alpha, standardize, intercept, cm[-1, k])
    else .gn_binom_tight(X, y, lambdas[k], alpha, standardize, intercept, cm[, k])))
}
.gn_dev_ratio <- function(X, y, coefs, family, intercept) {
  cm <- matrix(coefs, ncol(X) + 1); eta <- sweep(X %*% cm[-1, , drop = FALSE], 2, cm[1, ], "+")
  if (family == "gaussian") {
    null <- if (intercept) sum((y - mean(y))^2) else sum(y^2)
    1 - colSums((y - eta)^2) / null
  } else {
    pr <- 1 / (1 + exp(-eta)); dev <- -2 * colSums(y * log(pr) + (1 - y) * log(1 - pr))
    p0 <- if (intercept) mean(y) else 0.5
    1 - dev / (-2 * sum(y * log(p0) + (1 - y) * log(1 - p0)))
  }
}
# cv.glmnet's cvm / cvsd (grouped = TRUE): per-fold means, weighted by fold size; cvsd with N - 1 folds.
.gn_cvstats <- function(err, foldid) {
  K <- max(foldid); fm <- t(sapply(1:K, function(k) colMeans(err[foldid == k, , drop = FALSE]))); w <- tabulate(foldid, K)
  cvm <- colSums(fm * w) / sum(w)
  cvsd <- sqrt(colSums(sweep(fm, 2, cvm)^2 * w) / sum(w) / (K - 1))
  list(cvm = cvm, cvsd = cvsd)
}
# predict.glmnet(s = lambda, exact = FALSE): coefficients linearly interpolated in lambda between the
# two bracketing path values (glmnet:::lambda.interp), clamped to the first / last fit outside the path.
.gn_interp <- function(cm, path, s) {
  li <- glmnet:::lambda.interp(path, s)
  cm[, li$left, drop = FALSE] * rep(li$frac, each = nrow(cm)) + cm[, li$right, drop = FALSE] * rep(1 - li$frac, each = nrow(cm))
}
# cv.glmnet: each fold is fitted on its OWN default lambda path (cv.glmnet passes the user's `lambda`,
# NULL here, to the fold fits), and predicted at the full-data lambdas by .gn_interp. `exact = TRUE`
# refits the fold paths to the KKT solution first.
.gn_cv <- function(X, y, foldid, lambda, family, exact, alpha = 1) {
  err <- matrix(NA_real_, nrow(X), length(lambda)); flen <- integer(max(foldid))
  for (k in 1:max(foldid)) {
    tr <- foldid != k
    f <- glmnet(X[tr, ], y[tr], family = family, alpha = alpha, control = list(thresh = 1e-14))
    flen[k] <- length(f$lambda)
    cm <- if (exact) matrix(.gn_tight_path(f, X[tr, ], y[tr], f$lambda, alpha, TRUE, TRUE, family), ncol(X) + 1) else .gn_vec(f)
    cm <- .gn_interp(matrix(cm, ncol(X) + 1), f$lambda, lambda)
    eta <- sweep(X[!tr, , drop = FALSE] %*% cm[-1, , drop = FALSE], 2, cm[1, ], "+")
    err[!tr, ] <- if (family == "gaussian") (y[!tr] - eta)^2 else {
      pr <- pmin(pmax(1 / (1 + exp(-eta)), 1e-5), 1 - 1e-5); -2 * (y[!tr] * log(pr) + (1 - y[!tr]) * log(1 - pr)) }
  }
  c(.gn_cvstats(err, foldid), list(fold_path_length = flen))
}
.gn_opt <- function(lambda, cvm, cvsd) {
  imin <- match(max(lambda[cvm <= min(cvm)]), lambda)
  i1se <- match(max(lambda[cvm <= (cvm + cvsd)[imin]]), lambda)
  c(imin, i1se)
}

case("glmnet_paths", "regression/glmnet", "glmnet gaussian (lasso, elastic net, ridge; standardize TRUE/FALSE; intercept TRUE/FALSE) at fixed lambdas with exact KKT refits, default paths, binomial, cv.glmnet with foldid, KKT path data", function(ctx) {
  d <- .glm_sim(ctx); X <- as.matrix(d[, paste0("X", 1:8)]); n <- nrow(X)
  ctl <- list(thresh = 1e-14)
  v <- list()
  lams <- c(1, 0.3, 0.1, 0.03)
  for (a in c(1, 0.5, 0)) for (st in c(TRUE, FALSE)) for (ic in c(TRUE, FALSE)) {
    f <- glmnet(X, d$y, alpha = a, lambda = lams, standardize = st, intercept = ic, control = ctl)
    tag <- paste0("a", a * 10, "_", if (st) "std" else "raw", if (ic) "" else "_noint")
    v[[paste0(tag, "_coef")]] <- vec(.gn_vec(f))                        # (p + 1) x 4, column-major, intercept first
    tight <- .gn_tight_path(f, X, d$y, lams, a, st, ic)
    stopifnot(max(abs(tight - .gn_vec(f))) < 1e-6)
    v[[paste0(tag, "_coef_tight")]] <- vec(tight)
    v[[paste0(tag, "_dev_ratio")]] <- vec(f$dev.ratio)
    v[[paste0(tag, "_dev_ratio_tight")]] <- vec(.gn_dev_ratio(X, d$y, tight, "gaussian", ic))
  }
  # default lasso path: lambda_max = max_j |x~_j'(y - ybar)| / n, ratio 1e-4 (n >= p), fdev / devmax stop after mnlam = 5
  fp <- glmnet(X, d$y, alpha = 1, control = ctl)
  v$path_lambda <- vec(fp$lambda); v$path_df <- vec(fp$df); v$path_dev_ratio <- vec(fp$dev.ratio)
  pt <- .gn_tight_path(fp, X, d$y, fp$lambda, 1, TRUE, TRUE)
  v$path_coef_tight <- vec(pt); v$path_dev_ratio_tight <- vec(.gn_dev_ratio(X, d$y, pt, "gaussian", TRUE))
  # coef / predict at s off the path (predict.glmnet exact = FALSE: linear interpolation in lambda, clamped)
  s <- c(2.5, 0.5, 0.05, 1e-4); v$interp_s <- vec(s)
  v$interp_coef <- vec(as.numeric(as.matrix(coef(fp, s = s))))
  v$interp_coef_tight <- vec(.gn_interp(matrix(pt, 9), fp$lambda, s))
  v$interp_pred <- vec(as.numeric(predict(fp, X[1:3, ], s = 0.5)))
  fr <- glmnet(X, d$y, alpha = 0, control = ctl); v$ridge_path_lambda_max <- fr$lambda[1]; v$ridge_path_length <- length(fr$lambda)
  fe <- glmnet(X, d$y, alpha = 0.5, control = ctl); v$enet_path_lambda_max <- fe$lambda[1]
  fw <- glmnet(as.matrix(d[1:20, paste0("K", 1:30)]), d$yk[1:20], control = ctl)   # n = 20 < p = 30: ratio 0.01
  v$wide_path_lambda <- vec(fw$lambda); v$wide_path_df <- vec(fw$df)
  v$dfmax2_path_df <- vec(glmnet(X, d$y, alpha = 1, control = list(thresh = 1e-14, dfmax = 2))$df)
  v$fdev0_path_length <- length(glmnet(X, d$y, alpha = 1, control = list(thresh = 1e-14, fdev = 0))$lambda)
  f1 <- suppressWarnings(glmnet(X, d$y, nlambda = 1, control = ctl)); v$nlambda1_coef <- vec(.gn_vec(f1)); v$nlambda1_dev_ratio <- f1$dev.ratio
  # binomial
  blams <- c(0.1, 0.03, 0.01)
  for (cfg in list(list("binom", 1, TRUE, TRUE), list("binom_raw", 1, FALSE, TRUE), list("binom_noint", 1, TRUE, FALSE), list("binom_a5", 0.5, TRUE, TRUE))) {
    fb <- glmnet(X, d$yb, family = "binomial", alpha = cfg[[2]], lambda = blams, standardize = cfg[[3]], intercept = cfg[[4]], control = ctl)
    v[[paste0(cfg[[1]], "_coef")]] <- vec(.gn_vec(fb))
    tight <- .gn_tight_path(fb, X, d$yb, blams, cfg[[2]], cfg[[3]], cfg[[4]], "binomial")
    stopifnot(max(abs(tight - .gn_vec(fb))) < 1e-6)
    v[[paste0(cfg[[1]], "_coef_tight")]] <- vec(tight)
    v[[paste0(cfg[[1]], "_dev_ratio")]] <- vec(fb$dev.ratio)
    v[[paste0(cfg[[1]], "_dev_ratio_tight")]] <- vec(.gn_dev_ratio(X, d$yb, tight, "binomial", cfg[[4]]))
  }
  # default binomial path: lambda_max = max_j |x~_j'(y - p0)| / n with p0 = ybar (intercept) or 0.5 (none);
  # stop when the dev.ratio gain is below fdev (absolute, unlike gaussian's relative rule)
  fbp <- glmnet(X, d$yb, family = "binomial", alpha = 1, control = ctl)
  v$binom_path_lambda <- vec(fbp$lambda); v$binom_path_df <- vec(fbp$df); v$binom_path_dev_ratio <- vec(fbp$dev.ratio)
  bpt <- .gn_tight_path(fbp, X, d$yb, fbp$lambda, 1, TRUE, TRUE, "binomial")
  v$binom_path_coef_tight <- vec(bpt); v$binom_path_dev_ratio_tight <- vec(.gn_dev_ratio(X, d$yb, bpt, "binomial", TRUE))
  v$binom_interp_prob <- vec(as.numeric(predict(fbp, X[1:3, ], s = 0.05, type = "response")))
  bi <- .gn_interp(matrix(bpt, 9), fbp$lambda, 0.05); v$binom_interp_prob_tight <- vec(1 / (1 + exp(-(bi[1] + X[1:3, ] %*% bi[-1]))))
  v$binom_noint_sub_lambda_max <- glmnet(X[1:120, ], d$yb[1:120], family = "binomial", intercept = FALSE, control = ctl)$lambda[1]   # ybar = 0.45
  # cv.glmnet, gaussian lasso, foldid (5 equal folds)
  cv <- cv.glmnet(X, d$y, alpha = 1, foldid = d$foldid, control = ctl)
  stopifnot(isTRUE(all.equal(cv$lambda, fp$lambda, tolerance = 0)))
  v$cv_lambda_min <- cv$lambda.min; v$cv_lambda_1se <- cv$lambda.1se; v$cv_cvm <- vec(cv$cvm); v$cv_cvsd <- vec(cv$cvsd)
  v$cv_coef_min <- vec(as.numeric(as.matrix(coef(cv, s = "lambda.min"))))
  v$cv_coef_1se <- vec(as.numeric(as.matrix(coef(cv, s = "lambda.1se"))))
  cr <- .gn_cv(X, d$y, d$foldid, cv$lambda, "gaussian", exact = FALSE)       # the reconstruction is cv.glmnet itself
  stopifnot(max(abs(cr$cvm - cv$cvm) / cv$cvm) < 1e-12, max(abs(cr$cvsd - cv$cvsd) / cv$cvsd) < 1e-12)
  ct <- .gn_cv(X, d$y, d$foldid, cv$lambda, "gaussian", exact = TRUE)
  v$cv_fold_path_length <- vec(ct$fold_path_length)
  v$cv_cvm_tight <- vec(ct$cvm); v$cv_cvsd_tight <- vec(ct$cvsd)
  v$cv_index <- vec(c(cv$index)); v$cv_index_tight <- vec(.gn_opt(cv$lambda, ct$cvm, ct$cvsd))
  # cv.glmnet, binomial lasso, foldid7 (unequal folds; deviance with predictions bounded at 1e-5)
  cvb <- cv.glmnet(X, d$yb, family = "binomial", alpha = 1, foldid = d$foldid7, control = ctl)
  v$cvb_lambda <- vec(cvb$lambda); v$cvb_cvm <- vec(cvb$cvm); v$cvb_cvsd <- vec(cvb$cvsd); v$cvb_index <- vec(c(cvb$index))
  cbr <- .gn_cv(X, d$yb, d$foldid7, cvb$lambda, "binomial", exact = FALSE)
  stopifnot(max(abs(cbr$cvm - cvb$cvm) / cvb$cvm) < 1e-12, max(abs(cbr$cvsd - cvb$cvsd) / cvb$cvsd) < 1e-12)
  cbt <- .gn_cv(X, d$yb, d$foldid7, cvb$lambda, "binomial", exact = TRUE)
  v$cvb_fold_path_length <- vec(cbt$fold_path_length)
  v$cvb_cvm_tight <- vec(cbt$cvm); v$cvb_cvsd_tight <- vec(cbt$cvsd); v$cvb_index_tight <- vec(.gn_opt(cvb$lambda, cbt$cvm, cbt$cvsd))
  # KKT data: default lasso path on K1..K30, exact path solutions
  K <- as.matrix(d[, paste0("K", 1:30)])
  fk <- glmnet(K, d$yk, alpha = 1, control = ctl)
  v$kkt_lambda <- vec(fk$lambda); v$kkt_df <- vec(fk$df)
  v$kkt_coef_tight <- vec(.gn_tight_path(fk, K, d$yk, fk$lambda, 1, TRUE, TRUE))
  list(values = v, reference = "glmnet::glmnet(alpha, lambda, standardize, intercept, family, control = list(thresh = 1e-14)); glmnet::cv.glmnet(foldid); exact KKT refits (*_tight) written out",
       call = "glmnet(X, y, alpha, lambda = c(1, 0.3, 0.1, 0.03), standardize, intercept, control = list(thresh = 1e-14))", packages = "glmnet")
})
