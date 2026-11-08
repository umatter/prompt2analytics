# validation/r/cases/ml_tree_importance.R -- impurity-decrease variable importance of the tree
# ensembles (T3-WP44: WT-5, ML-105). One dataset: a step in x0 that explains ~95% of the variance,
# smaller linear effects of x1 and x2 and noise.
#  * tree_importance_rpart: rpart's variable.importance of a primary-split tree (maxcompete = 0,
#    maxsurrogate = 0, cp = 0) is the summed SSE decrease of its splits, the quantity randomForest's
#    IncNodePurity and gbm's relative.influence sum per tree. Two tree settings: the random-forest
#    builder's (minsplit 2, minbucket 1, depth 3) and the quantile forest's (min_node_size 5:
#    minsplit 11, minbucket 5, depth 4).
#  * gbm_stump_influence: gbm(distribution = "gaussian", interaction.depth = 1, n.minobsinnode = 1,
#    bag.fraction = 1, shrinkage = 0.1): relative.influence (unscaled), initF and the fitted values.
#  * rf_importance_purity: randomForest IncNodePurity (mtry = 2) over five seeds and nodesize 5 and
#    10: a magnitude and ranking reference only (the bootstrap draws cannot be matched).
suppressPackageStartupMessages({library(rpart); library(gbm); library(randomForest)})

.wp44_tree_importance_data <- function(ctx) {
  set.seed(20261109)
  n <- 200
  x0 <- runif(n); x1 <- runif(n); x2 <- rnorm(n)
  y <- 10 * (x0 > 0.5) + 0.8 * x1 + 0.3 * x2 + rnorm(n, 0, 0.5)
  ctx$gen_dataset("tree_importance", data.frame(y = round(y, 10), x0 = round(x0, 10), x1 = round(x1, 10), x2 = round(x2, 10)))
}
.wp44_imp3 <- function(fit) { v <- setNames(numeric(3), c("x0", "x1", "x2")); vi <- fit$variable.importance; v[names(vi)] <- vi; v }

case("tree_importance_rpart", "ml/trees", "summed SSE decrease of the primary splits (rpart variable.importance)", function(ctx) {
  d <- .wp44_tree_importance_data(ctx)
  v <- list()
  for (s in list(list(tag = "rf_d3", ms = 2, mb = 1, md = 3), list(tag = "qrf_d4", ms = 11, mb = 5, md = 4))) {
    ctl <- rpart.control(cp = 0, minsplit = s$ms, minbucket = s$mb, maxdepth = s$md, xval = 0, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0)
    fit <- rpart(y ~ x0 + x1 + x2, data = d, method = "anova", control = ctl)
    imp <- .wp44_imp3(fit)
    v[[paste0(s$tag, "_importance")]] <- named(imp)
    v[[paste0(s$tag, "_share")]] <- named(imp / sum(imp))
    v[[paste0(s$tag, "_frame_n")]] <- vec(fit$frame$n)
    v[[paste0(s$tag, "_n_splits")]] <- sum(fit$frame$var != "<leaf>")
  }
  v$tss <- sum((d$y - mean(d$y))^2)
  list(values = v, reference = "rpart::rpart(cp = 0, maxcompete = 0, maxsurrogate = 0)$variable.importance",
       call = "rpart(y ~ x0 + x1 + x2, data = tree_importance, control = rpart.control(cp = 0, minsplit = 2, minbucket = 1, maxdepth = 3, xval = 0, maxsurrogate = 0, maxcompete = 0, usesurrogate = 0))", packages = "rpart")
})

case("gbm_stump_influence", "ml/trees", "gbm Gaussian stumps: relative.influence (summed SSE decrease), initF, fitted values", function(ctx) {
  d <- .wp44_tree_importance_data(ctx)
  fit <- gbm(y ~ x0 + x1 + x2, data = d, distribution = "gaussian", n.trees = 50, interaction.depth = 1, n.minobsinnode = 1,
             shrinkage = 0.1, bag.fraction = 1, train.fraction = 1, keep.data = TRUE, verbose = FALSE)
  ri <- relative.influence(fit, n.trees = 50)
  v <- list(influence = named(ri), influence_share = named(ri / sum(ri)), init_f = fit$initF, fitted = vec(fit$fit),
            split_var = vec(sapply(fit$trees, function(t) t[[1]][1])))   # 0-based split variable of each stump
  list(values = v, reference = "gbm::gbm(distribution = 'gaussian', interaction.depth = 1, n.minobsinnode = 1, bag.fraction = 1); gbm::relative.influence",
       call = "gbm(y ~ x0 + x1 + x2, data = tree_importance, distribution = 'gaussian', n.trees = 50, interaction.depth = 1, n.minobsinnode = 1, shrinkage = 0.1, bag.fraction = 1, train.fraction = 1)", packages = c("gbm"))
})

case("rf_importance_purity", "ml/trees", "randomForest IncNodePurity (mtry = 2): magnitude and ranking reference over five seeds", function(ctx) {
  d <- .wp44_tree_importance_data(ctx)
  X <- d[, c("x0", "x1", "x2")]
  v <- list()
  for (ns in c(5, 10)) {
    shares <- matrix(NA, 5, 3); raw <- matrix(NA, 5, 3)
    for (s in 1:5) {
      set.seed(s)
      rf <- randomForest(X, d$y, ntree = 500, mtry = 2, nodesize = ns)
      imp <- importance(rf)[, "IncNodePurity"]
      raw[s, ] <- imp; shares[s, ] <- imp / sum(imp)
    }
    v[[paste0("ns", ns, "_share_min")]] <- vec(apply(shares, 2, min))
    v[[paste0("ns", ns, "_share_max")]] <- vec(apply(shares, 2, max))
    v[[paste0("ns", ns, "_share_mean")]] <- vec(colMeans(shares))
    v[[paste0("ns", ns, "_purity_min")]] <- vec(apply(raw, 2, min))
    v[[paste0("ns", ns, "_purity_max")]] <- vec(apply(raw, 2, max))
    stopifnot(all(shares[, 1] > shares[, 3]), all(shares[, 3] > shares[, 2]))   # x0 > x2 > x1 for every seed
  }
  list(values = v, reference = "randomForest::randomForest(ntree = 500, mtry = 2, nodesize = 5 | 10); importance(type = 2) (IncNodePurity), seeds 1-5",
       call = "randomForest(tree_importance[, c('x0', 'x1', 'x2')], tree_importance$y, ntree = 500, mtry = 2, nodesize = 5)", packages = c("randomForest"))
})
