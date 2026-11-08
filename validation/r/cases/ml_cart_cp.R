# validation/r/cases/ml_cart_cp.R -- rpart complexity tables with real cross-validation (T3-WP44:
# ML-71; moved in ML-69, ML-70, ML-72). rpart(xval = <fold vector>) makes the cross-validation
# deterministic: p2a takes the same fold assignment (a `folds` column) and must reproduce the cp
# table (CP, nsplit, rel error, xerror, xstd), the frame (n, dev, yval, complexity, split variable,
# split point and direction), the training predictions, the prune at a given cp and the variable
# importance of the primary splits (surrogates and competitors off: maxsurrogate = 0, maxcompete = 0).
suppressPackageStartupMessages(library(rpart))

.wp44_frame_values <- function(fit, prefix, xnames, ylevels = NULL) {
  ff <- fit$frame
  out <- list()
  v <- match(as.character(ff$var), xnames) - 1; v[is.na(v)] <- -1          # 0-based feature, -1 = leaf
  out[[paste0(prefix, "_frame_id")]] <- vec(as.numeric(rownames(ff)))       # rpart node numbers, preorder
  out[[paste0(prefix, "_frame_var")]] <- vec(v)
  out[[paste0(prefix, "_frame_n")]] <- vec(ff$n)
  out[[paste0(prefix, "_frame_dev")]] <- vec(ff$dev)
  out[[paste0(prefix, "_frame_yval")]] <- vec(if (is.null(ylevels)) ff$yval else as.numeric(ylevels[ff$yval]))
  out[[paste0(prefix, "_frame_complexity")]] <- vec(ff$complexity)
  if (!is.null(fit$splits)) {
    out[[paste0(prefix, "_split_index")]] <- vec(fit$splits[, "index"])     # primary splits only (no competitors/surrogates)
    out[[paste0(prefix, "_split_ncat")]] <- vec(fit$splits[, "ncat"])       # -1: x < index goes to the left son
  }
  out
}
.wp44_cp_values <- function(fit, prefix) {
  cp <- fit$cptable
  out <- list()
  out[[paste0(prefix, "_cp")]] <- vec(cp[, "CP"]); out[[paste0(prefix, "_nsplit")]] <- vec(cp[, "nsplit"])
  out[[paste0(prefix, "_rel_error")]] <- vec(cp[, "rel error"])
  if ("xerror" %in% colnames(cp)) {
    out[[paste0(prefix, "_xerror")]] <- vec(cp[, "xerror"]); out[[paste0(prefix, "_xstd")]] <- vec(cp[, "xstd"])
  }
  vi <- fit$variable.importance
  if (is.numeric(vi) && length(vi) > 0) out[[paste0(prefix, "_importance")]] <- named(vi)   # absent for a tree without splits
  out
}
.wp44_one_se <- function(cpt) {
  i <- which.min(cpt[, "xerror"]); thr <- cpt[i, "xerror"] + cpt[i, "xstd"]
  c(min_xerror_cp = cpt[i, "CP"], one_se_cp = cpt[min(which(cpt[, "xerror"] <= thr)), "CP"])
}

case("cart_cp_anova", "ml/trees", "rpart regression tree cp table with 10 fixed folds, prune at cp, importance", function(ctx) {
  set.seed(20261107)
  n <- 300
  x1 <- runif(n); x2 <- runif(n); x3 <- rnorm(n); x4 <- sample(1:4, n, replace = TRUE)
  y <- 2 * (x1 > 0.5) + 1.5 * (x2 > 0.3) * (x1 <= 0.5) + 0.3 * x3 + rnorm(n, 0, 0.7)
  folds <- rep_len(1:10, n)[sample.int(n)]
  d <- ctx$gen_dataset("cart_anova", data.frame(y = round(y, 10), x1 = round(x1, 10), x2 = round(x2, 10), x3 = round(x3, 10), x4 = x4, folds = folds))
  xn <- c("x1", "x2", "x3", "x4")
  ctl <- rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = d$folds, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)
  fit <- rpart(y ~ x1 + x2 + x3 + x4, data = d, method = "anova", control = ctl)
  v <- c(.wp44_cp_values(fit, "anova"), .wp44_frame_values(fit, "anova", xn))
  v$anova_pred <- vec(predict(fit, d))
  v$anova_root_risk <- fit$frame$dev[1]
  s <- .wp44_one_se(fit$cptable); v$anova_min_xerror_cp <- s[["min_xerror_cp"]]; v$anova_one_se_cp <- s[["one_se_cp"]]
  pr <- prune(fit, cp = 0.02)
  v <- c(v, .wp44_cp_values(pr, "anova_pruned_002"), .wp44_frame_values(pr, "anova_pruned_002", xn))
  v$anova_pruned_002_pred <- vec(predict(pr, d))
  v$anova_pruned_002_pred_head <- vec(head(predict(pr, d), 10))
  # prune at a cp below every complexity: rpart returns the tree unchanged (cp table included)
  stopifnot(identical(prune(fit, cp = 0.0005)$cptable, fit$cptable))
  # without cross-validation the table has three columns and the same values
  fit0 <- rpart(y ~ x1 + x2 + x3 + x4, data = d, method = "anova", control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = 0, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0))
  stopifnot(ncol(fit0$cptable) == 3, isTRUE(all.equal(fit0$cptable[, 1:3], fit$cptable[, 1:3], tolerance = 0)))
  list(values = v, reference = "rpart::rpart(method = 'anova', xval = fold vector, maxsurrogate = 0, maxcompete = 0); rpart::prune",
       call = "rpart(y ~ x1 + x2 + x3 + x4, data = cart_anova, control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = folds, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0))", packages = "rpart")
})

case("cart_cp_class", "ml/trees", "rpart classification tree (gini) cp table with fixed folds; class labels coded 3 and 17 (ML-70)", function(ctx) {
  set.seed(20261108)
  n <- 300
  x1 <- runif(n); x2 <- runif(n); x3 <- rnorm(n)
  p <- plogis(-1 + 3 * (x1 > 0.4) + 1.5 * x3 * (x2 > 0.5))
  cl <- ifelse(runif(n) < p, 17L, 3L)
  folds <- rep_len(1:10, n)[sample.int(n)]
  d <- ctx$gen_dataset("cart_class", data.frame(cls = cl, x1 = round(x1, 10), x2 = round(x2, 10), x3 = round(x3, 10), folds = folds))
  xn <- c("x1", "x2", "x3")
  ctl <- rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = d$folds, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)
  fit <- rpart(factor(cls) ~ x1 + x2 + x3, data = d, method = "class", control = ctl, parms = list(split = "gini"))
  lev <- attr(fit, "ylevels")
  stopifnot(identical(lev, c("3", "17")))
  v <- c(.wp44_cp_values(fit, "class"), .wp44_frame_values(fit, "class", xn, lev))
  v$class_pred <- vec(as.numeric(as.character(predict(fit, d, type = "class"))))
  v$class_pred_head <- vec(as.numeric(as.character(predict(fit, d, type = "class")))[1:10])
  pp <- predict(fit, d, type = "prob")
  v$class_prob3 <- vec(pp[, "3"]); v$class_prob17 <- vec(pp[, "17"])
  v$class_prob17_head <- vec(pp[1:10, "17"])
  v$class_root_risk <- fit$frame$dev[1]
  s <- .wp44_one_se(fit$cptable); v$class_min_xerror_cp <- s[["min_xerror_cp"]]; v$class_one_se_cp <- s[["one_se_cp"]]
  # ML-70: rpart's tree does not depend on how the classes are coded (order-preserving recodes give the
  # same frame and cp table; the reversed coding gives the same cp table)
  for (codes in list(c(0, 1), c(-5, 1e6), c(0.1, 0.25))) {
    d2 <- d; d2$cls <- ifelse(d$cls == 3, codes[1], codes[2])
    f2 <- rpart(factor(cls) ~ x1 + x2 + x3, data = d2, method = "class", control = ctl, parms = list(split = "gini"))
    stopifnot(isTRUE(all.equal(f2$cptable, fit$cptable, tolerance = 0)), identical(f2$frame$n, fit$frame$n))
  }
  d3 <- d; d3$cls <- ifelse(d$cls == 3, 20, 1)
  f3 <- rpart(factor(cls) ~ x1 + x2 + x3, data = d3, method = "class", control = ctl, parms = list(split = "gini"))
  stopifnot(isTRUE(all.equal(unname(f3$cptable), unname(fit$cptable), tolerance = 1e-14)))
  # the information (entropy) split rule on the same data
  fi <- rpart(factor(cls) ~ x1 + x2 + x3, data = d, method = "class", control = ctl, parms = list(split = "information"))
  v <- c(v, .wp44_cp_values(fi, "info"))
  v$info_frame_n <- vec(fi$frame$n)
  list(values = v, reference = "rpart::rpart(method = 'class', parms = list(split = 'gini' | 'information'), xval = fold vector)",
       call = "rpart(factor(cls) ~ x1 + x2 + x3, data = cart_class, method = 'class', control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = folds, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0))", packages = "rpart")
})

case("cart_class_ties", "ml/trees", "rpart class ties: the lowest class label wins, probabilities in label order (ML-69)", function(ctx) {
  set.seed(20261110)
  # classes coded 9, 4 and 7: the 4/7 group is a 10/10 tie after the first split
  x <- c(runif(10, 0, 1), runif(20, 2, 3)); z <- runif(30)
  cls <- c(rep(9, 10), rep(c(7, 4), 10))
  d <- ctx$gen_dataset("cart_ties", data.frame(cls = cls, x = round(x, 10), z = round(z, 10)))
  ctl <- rpart.control(cp = 0.001, minsplit = 2, minbucket = 1, maxdepth = 1, xval = 0, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)
  fit <- rpart(factor(cls) ~ x + z, data = d, method = "class", control = ctl)
  lev <- attr(fit, "ylevels"); stopifnot(identical(lev, c("4", "7", "9")))
  v <- .wp44_frame_values(fit, "ties", c("x", "z"), lev)
  v$ties_pred <- vec(as.numeric(as.character(predict(fit, d, type = "class"))))
  pp <- predict(fit, d, type = "prob")
  v$ties_prob_row30 <- vec(pp[30, ])                 # the tied leaf, in label order 4, 7, 9
  v$ties_prob_row1 <- vec(pp[1, ])
  # a root that cannot split (minsplit > n): 10/10/10 tie at the root
  fit0 <- rpart(factor(cls) ~ x + z, data = d, method = "class", control = rpart.control(minsplit = 40, xval = 0))
  v$ties_root_yval <- as.numeric(lev[fit0$frame$yval[1]])
  v$ties_root_dev <- fit0$frame$dev[1]
  v$ties_root_cp <- vec(fit0$cptable[, "CP"]); v$ties_root_rel_error <- vec(fit0$cptable[, "rel error"])   # unsplittable root
  stopifnot(v$ties_root_yval == 4, rownames(fit$frame)[2] == "2", fit$frame$yval[2] == 1)
  list(values = v, reference = "rpart::rpart(method = 'class'): ginidev takes the first class of minimal loss",
       call = "rpart(factor(cls) ~ x + z, data = cart_ties, method = 'class', control = rpart.control(cp = 0.001, minsplit = 2, minbucket = 1, maxdepth = 1, xval = 0))", packages = "rpart")
})

case("cart_prune_xor", "ml/trees", "rpart weakest-link pruning: a weak root split above two strong splits survives prune(cp = 0.05) (ML-72)", function(ctx) {
  set.seed(20261111)
  x0 <- rep(c(0, 1), each = 20); x1 <- rep(0:19, 2)
  y <- 10 * xor(x1 >= 10, x0 == 1) + rnorm(40, 0, 0.5)
  d <- ctx$gen_dataset("cart_xor", data.frame(y = round(y, 10), x0 = x0, x1 = x1))
  ctl <- rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = 0, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)
  fit <- rpart(y ~ x0 + x1, data = d, method = "anova", control = ctl)
  v <- c(.wp44_cp_values(fit, "xor"), .wp44_frame_values(fit, "xor", c("x0", "x1")))
  pr <- prune(fit, cp = 0.05)
  v <- c(v, .wp44_cp_values(pr, "xor_pruned_005"), .wp44_frame_values(pr, "xor_pruned_005", c("x0", "x1")))
  v$xor_pruned_005_pred <- vec(predict(pr, d))
  v$xor_pruned_005_leaves <- sum(pr$frame$var == "<leaf>")
  pr2 <- prune(fit, cp = 0.5)
  v <- c(v, .wp44_cp_values(pr2, "xor_pruned_05"))
  v$xor_pruned_05_leaves <- sum(pr2$frame$var == "<leaf>")
  v$xor_pruned_05_pred <- vec(predict(pr2, d))
  stopifnot(v$xor_pruned_005_leaves == 4)
  list(values = v, reference = "rpart::rpart(method = 'anova'); rpart::prune (weakest-link complexity, frame$complexity <= cp)",
       call = "prune(rpart(y ~ x0 + x1, data = cart_xor, control = rpart.control(cp = 0.001, minsplit = 20, minbucket = 7, xval = 0)), cp = 0.05)", packages = "rpart")
})
