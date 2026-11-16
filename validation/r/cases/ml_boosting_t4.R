# validation/r/cases/ml_boosting_t4.R -- terminal-node estimates of gradient boosting (T4-WP20: ML-94, WT-1).
# One dataset: a large-scale linear target with Student-t(3) noise (WT-1's scale case) and a binary target.
#  * gbm_laplace_stumps: gbm(distribution = "laplace", interaction.depth = 1, n.minobsinnode = 1,
#    bag.fraction = 1, shrinkage = 0.1, n.trees = 50): each leaf is the median of y - F in the leaf.
#  * gbm_bernoulli_stumps: the same with distribution = "bernoulli": each leaf is the Newton step
#    sum(y - p) / sum(p (1 - p)) (Friedman 2001, Algorithm 5).
suppressPackageStartupMessages(library(gbm))

.wp20_gbm_leaf_data <- function(ctx) {
  set.seed(20261115)
  n <- 200
  x0 <- runif(n, 0, 10); x1 <- runif(n); x2 <- rnorm(n)
  y_lad <- 100 * x0 + 400 * (x1 > 0.5) + 5 * rt(n, df = 3)
  y_bin <- rbinom(n, 1, plogis(-1 + 0.4 * x0 - x1))
  ctx$gen_dataset("gbm_leaf_t4", data.frame(y_lad = round(y_lad, 10), y_bin = y_bin,
                                            x0 = round(x0, 10), x1 = round(x1, 10), x2 = round(x2, 10)))
}
.wp20_gbm_stumps <- function(d, y, dist) {
  gbm(as.formula(paste(y, "~ x0 + x1 + x2")), data = d, distribution = dist, n.trees = 50, interaction.depth = 1,
      n.minobsinnode = 1, shrinkage = 0.1, bag.fraction = 1, train.fraction = 1, keep.data = TRUE, verbose = FALSE)
}

case("gbm_laplace_stumps", "ml/gbm", "gbm Laplace stumps: median terminal nodes, initF, fitted values, split variables, relative influence", function(ctx) {
  d <- .wp20_gbm_leaf_data(ctx)
  fit <- .wp20_gbm_stumps(d, "y_lad", "laplace")
  ri <- relative.influence(fit, n.trees = 50)
  v <- list(init_f = fit$initF, fitted = vec(fit$fit), split_var = vec(sapply(fit$trees, function(t) t[[1]][1])),
            influence = named(ri), mae = mean(abs(d$y_lad - fit$fit)))
  list(values = v, reference = "gbm::gbm(distribution = 'laplace', interaction.depth = 1, n.minobsinnode = 1, bag.fraction = 1); gbm::relative.influence",
       call = "gbm(y_lad ~ x0 + x1 + x2, data = gbm_leaf_t4, distribution = 'laplace', n.trees = 50, interaction.depth = 1, n.minobsinnode = 1, shrinkage = 0.1, bag.fraction = 1, train.fraction = 1)", packages = c("gbm"))
})

case("gbm_bernoulli_stumps", "ml/gbm", "gbm Bernoulli stumps: Newton-step terminal nodes, initF, fitted probabilities, split variables, relative influence", function(ctx) {
  d <- .wp20_gbm_leaf_data(ctx)
  fit <- .wp20_gbm_stumps(d, "y_bin", "bernoulli")
  ri <- relative.influence(fit, n.trees = 50)
  v <- list(init_f = fit$initF, fitted_prob = vec(plogis(fit$fit)),
            split_var = vec(sapply(fit$trees, function(t) t[[1]][1])), influence = named(ri))
  list(values = v, reference = "gbm::gbm(distribution = 'bernoulli', interaction.depth = 1, n.minobsinnode = 1, bag.fraction = 1); gbm::relative.influence",
       call = "gbm(y_bin ~ x0 + x1 + x2, data = gbm_leaf_t4, distribution = 'bernoulli', n.trees = 50, interaction.depth = 1, n.minobsinnode = 1, shrinkage = 0.1, bag.fraction = 1, train.fraction = 1)", packages = c("gbm"))
})
