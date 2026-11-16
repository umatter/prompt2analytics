# validation/r/cases/ml_trees_t4.R -- T4-WP18: partykit::ctree (identity transforms, conditional
# covariance, Bonferroni as partykit computes it: 1 - (1 - p)^m over the m tested inputs).
# Tree outputs are written in preorder (partykit's node numbering); internal nodes only for splits.
suppressPackageStartupMessages(library(partykit))


ct_tree_values <- function(ct, d, xs, classification) {
  ids <- nodeids(ct)
  split_var <- c(); split_break <- c(); split_stat <- c(); split_padj <- c(); node_n <- c()
  term_pred <- c(); term_prob <- c()
  for (id in ids) {
    nd <- node_party(ct[[id]])
    node_n <- c(node_n, nrow(data_party(ct, id)))
    sp <- split_node(nd)
    if (!is.null(sp)) {
      v <- varid_split(sp) - 1L                    # response is variable 1 in the model frame
      split_var <- c(split_var, v - 1L)            # 0-based feature index
      split_break <- c(split_break, breaks_split(sp))
      cr <- info_node(nd)$criterion
      split_stat <- c(split_stat, cr["statistic", xs[v]])
      split_padj <- c(split_padj, cr["p.value", xs[v]])
    } else {
      yy <- data_party(ct, id)[["(response)"]]
      if (classification) {
        pr <- as.numeric(table(yy)) / length(yy)
        term_prob <- c(term_prob, pr); term_pred <- c(term_pred, which.max(pr) - 1L)
      } else term_pred <- c(term_pred, mean(yy))
    }
  }
  root <- info_node(node_party(ct))$criterion
  stat <- root["statistic", xs]
  m <- sum(!is.na(stat))
  raw <- pchisq(stat, df = if (classification) nlevels(d[[ncol(d)]]) - 1 else 1, lower.tail = FALSE)
  out <- list(n_nodes = length(ids), n_terminal = length(nodeids(ct, terminal = TRUE)), depth = depth(ct),
              node_n = vec(node_n), split_var = vec(split_var), split_break = vec(split_break),
              split_statistic = vec(split_stat), split_p_adjusted = vec(split_padj),
              terminal_prediction = vec(term_pred),
              root_statistic = vec(stat), root_p = vec(raw), root_p_adjusted = vec(-expm1(m * log1p(-raw))))
  if (classification) out$terminal_class_prob = vec(term_prob)
  out
}

case("ctree_reg_t4", "ml/ctree", "partykit::ctree regression tree, default control", function(ctx) {
  set.seed(20261115)
  n <- 200
  x1 <- round(rnorm(n), 6); x2 <- round(runif(n, 0, 4), 6); x3 <- round(rnorm(n), 1)  # x3: many ties
  y <- round(1.5 * (x1 > 0.3) + 0.8 * x2 + rnorm(n, sd = 0.9), 6)
  d <- ctx$gen_dataset("ctree_reg_t4", data.frame(x1 = x1, x2 = x2, x3 = x3, y = y))
  ct <- partykit::ctree(y ~ x1 + x2 + x3, data = d)
  v <- ct_tree_values(ct, d, c("x1", "x2", "x3"), FALSE)
  v$fitted <- vec(unname(predict(ct)))
  list(values = v, reference = "partykit::ctree(y ~ x1 + x2 + x3), ctree_control() defaults",
       call = "ctree(y ~ x1 + x2 + x3, data = d)", packages = c("partykit", "libcoin"))
})

case("ctree_cls_t4", "ml/ctree", "partykit::ctree classification tree (3 classes), default control", function(ctx) {
  set.seed(20261116)
  n <- 240
  x1 <- round(rnorm(n), 6); x2 <- round(runif(n), 6); x3 <- round(rnorm(n), 6)
  lin <- 1.2 * x1 + 2.5 * (x2 > 0.6) + rnorm(n, sd = 0.8)
  cls <- as.character(cut(lin, c(-Inf, -0.3, 1.4, Inf), labels = c("a", "b", "c")))
  code <- match(cls, c("a", "b", "c")) - 1L
  d <- ctx$gen_dataset("ctree_cls_t4", data.frame(x1 = x1, x2 = x2, x3 = x3, code = code, cls = cls))
  d$cls <- factor(d$cls)
  ct <- partykit::ctree(cls ~ x1 + x2 + x3, data = d)
  v <- ct_tree_values(ct, d[, c("x1", "x2", "x3", "cls")], c("x1", "x2", "x3"), TRUE)
  v$fitted_class <- vec(as.integer(predict(ct)) - 1L)
  list(values = v, reference = "partykit::ctree(factor(cls) ~ x1 + x2 + x3), ctree_control() defaults",
       call = "ctree(cls ~ x1 + x2 + x3, data = d)", packages = c("partykit", "libcoin"))
})

case("ctree_null_root_t4", "ml/ctree", "partykit::ctree root statistics when the root is not split", function(ctx) {
  set.seed(20261117)
  n <- 150
  d <- ctx$gen_dataset("ctree_null_root_t4",
                       data.frame(x1 = round(rnorm(n), 6), x2 = round(rexp(n), 6), y = round(rnorm(n), 6)))
  ct <- partykit::ctree(y ~ x1 + x2, data = d)
  stopifnot(length(nodeids(ct)) == 1L)
  cr <- info_node(node_party(ct))$criterion
  raw <- pchisq(cr["statistic", c("x1", "x2")], 1, lower.tail = FALSE)
  list(values = list(n_nodes = 1, root_statistic = vec(cr["statistic", c("x1", "x2")]), root_p = vec(raw),
                     root_p_adjusted = vec(cr["p.value", c("x1", "x2")])),
       reference = "partykit::ctree(y ~ x1 + x2): info_node(node_party(ct))$criterion of the unsplit root",
       call = "ctree(y ~ x1 + x2, data = d)", packages = c("partykit", "libcoin"))
})

case("ctree_montecarlo_t4", "ml/ctree", "partykit::ctree root p-values, testtype = 'MonteCarlo' (nresample 9999)", function(ctx) {
  set.seed(20261118)
  n <- 120
  x1 <- round(rnorm(n), 6); x2 <- round(rnorm(n), 6)
  y <- round(0.22 * x1 + rnorm(n), 6)
  d <- ctx$gen_dataset("ctree_montecarlo_t4", data.frame(x1 = x1, x2 = x2, y = y))
  set.seed(1)
  ct <- partykit::ctree(y ~ x1 + x2, data = d,
                        control = partykit::ctree_control(testtype = "MonteCarlo", nresample = 9999L, stump = TRUE))
  cr <- info_node(node_party(ct))$criterion
  st <- cr["statistic", c("x1", "x2")]
  list(values = list(root_statistic = vec(st), root_p_asymptotic = vec(pchisq(st, 1, lower.tail = FALSE)),
                     root_p_mc = vec(cr["p.value", c("x1", "x2")]), nresample = 9999),
       reference = "partykit::ctree(testtype = 'MonteCarlo', nresample = 9999, stump = TRUE), set.seed(1); Monte Carlo p-values, unadjusted as partykit leaves them for this testtype (R RNG: compare within Monte Carlo error)",
       call = "ctree(y ~ x1 + x2, control = ctree_control(testtype = 'MonteCarlo', nresample = 9999, stump = TRUE))",
       packages = c("partykit", "libcoin"))
})
