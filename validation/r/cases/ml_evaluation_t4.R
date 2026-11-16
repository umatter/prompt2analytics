# validation/r/cases/ml_evaluation_t4.R -- T4-WP23 ROC references (ML-91, ML-92, ML-93).
#  * ml_roc_small -- four score vectors from the findings (dataset roc_small: set 1-4 = ml91, ml91u, ml92, ml93): ties at the top score of 1.0 (and the same
#    ranking on an unbounded scale), a perfect separation, and a tie group holding both classes. pROC::roc with
#    levels = c(0, 1), direction = "<"; the curve in decreasing threshold order (pROC's thresholds reversed: Inf, the
#    midpoints, -Inf), which is p2a's order (+inf, then each distinct score with the rule score >= threshold).
#  * ml_roc_ties -- 240 simulated rows, classes coded 3 and 7, two scores with heavy ties: probabilities rounded to
#    0.1 (ties at 1.0, as trees and forests give) and integer scores on an unbounded scale.
# Pinned: the AUC (pROC::auc, equal to wilcox.test's W / (n1 n0) by stopifnot), tpr and fpr at every curve point,
# and Youden's optimum: sensitivity, specificity and the observed score that starts the optimal rule (the smallest
# score above pROC's first "best" midpoint; pROC lists tied optima in increasing threshold order and p2a reports the
# first of them).
suppressPackageStartupMessages(library(pROC))

.wp23_roc_values <- function(pred, lab, levels, prefix) {
  r <- pROC::roc(lab, pred, levels = levels, direction = "<", quiet = TRUE)
  w <- suppressWarnings(stats::wilcox.test(pred[lab == levels[2]], pred[lab == levels[1]],
                                           exact = FALSE, correct = FALSE))$statistic
  n1 <- sum(lab == levels[2]); n0 <- sum(lab == levels[1])
  a <- as.numeric(pROC::auc(r))
  stopifnot(isTRUE(all.equal(a, unname(w) / (n1 * n0), tolerance = 1e-14)))
  o <- order(r$thresholds, decreasing = TRUE)
  b <- pROC::coords(r, "best", best.method = "youden", ret = c("threshold", "sensitivity", "specificity"),
                    transpose = FALSE)
  out <- list()
  out[[paste0(prefix, "_auc")]] <- a
  out[[paste0(prefix, "_tpr")]] <- vec(r$sensitivities[o])
  out[[paste0(prefix, "_fpr")]] <- vec(1 - r$specificities[o])
  out[[paste0(prefix, "_best_sensitivity")]] <- b$sensitivity[1]
  out[[paste0(prefix, "_best_specificity")]] <- b$specificity[1]
  out[[paste0(prefix, "_best_threshold")]] <- min(pred[pred > b$threshold[1]])
  out[[paste0(prefix, "_n_best")]] <- nrow(b)
  out
}

case("ml_roc_small", "ml/evaluation", "pROC ROC curve, AUC and Youden optimum on tied scores from the findings", function(ctx) {
  s <- list(
    ml91 = list(p = c(1, 1, 0.6, 0.2, 0.1, 0.9), y = c(1, 0, 1, 0, 0, 1)),
    ml91u = list(p = c(5, 5, 3, -1, -2, 4), y = c(1, 0, 1, 0, 0, 1)),
    ml92 = list(p = c(0.1, 0.2, 0.3, 0.7, 0.8, 0.9), y = c(0, 0, 0, 1, 1, 1)),
    ml93 = list(p = c(0.5, 0.5, 0.5, 0.5, 0.2, 0.9), y = c(1, 1, 0, 0, 0, 1))
  )
  long <- do.call(rbind, lapply(seq_along(s), function(i) data.frame(set = i, pred = s[[i]]$p, label = s[[i]]$y)))
  d <- ctx$gen_dataset("roc_small", cbind(id = seq_len(nrow(long)), long))
  v <- list()
  for (i in seq_along(s)) {
    k <- names(s)[i]; di <- d[d$set == i, ]
    v <- c(v, .wp23_roc_values(di$pred, di$label, c(0, 1), k))
  }
  list(values = v, reference = "pROC::roc(levels = c(0, 1), direction = '<'), pROC::auc, pROC::coords('best', best.method = 'youden')",
       call = "roc(label, pred, levels = c(0, 1), direction = '<'); coords(r, 'best', best.method = 'youden')", packages = "pROC")
})

case("ml_roc_ties", "ml/evaluation", "pROC ROC curve, AUC and Youden optimum, 240 rows, tied probabilities and integer scores, classes 3/7", function(ctx) {
  set.seed(20261115)
  n <- 240
  x <- rnorm(n)
  cls <- ifelse(runif(n) < plogis(-0.3 + 1.4 * x), 7L, 3L)
  s_prob <- pmin(1, round(plogis(2.5 * x + rnorm(n, 0, 0.8) + 1.0), 1))
  s_lin <- round(2 * x + rnorm(n), 0)
  d <- ctx$gen_dataset("roc_ties", data.frame(id = seq_len(n), cls = cls, s_prob = s_prob, s_lin = s_lin))
  stopifnot(is.integer(d$cls))
  v <- c(.wp23_roc_values(d$s_prob, d$cls, c(3, 7), "prob"), .wp23_roc_values(d$s_lin, d$cls, c(3, 7), "lin"))
  list(values = v, reference = "pROC::roc(levels = c(3, 7), direction = '<'), pROC::auc, pROC::coords('best', best.method = 'youden')",
       call = "roc(cls, score, levels = c(3, 7), direction = '<') for score in s_prob, s_lin", packages = "pROC")
})
